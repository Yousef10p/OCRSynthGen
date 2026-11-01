"""text_pool/: real Arabic and English text for the template slots, from Wikipedia.

Wikipedia is used as the trusted source: written and checked by editors, cited, openly licensed
(CC BY-SA 4.0), and broad in topic, so pages get a real vocabulary instead of a few demo sentences. The
articles come from the wikimedia/wikipedia dataset (dump of 2023-11-01) through Hugging Face's dataset
server, a few thousand random articles per language; nothing else is downloaded.

Each article is split into the kinds of text a document has, and each kind fills the matching slots:
    titles      article titles                   -> title slots
    headings    section headings (short lines)    -> headings, table headers, labels
    items       list lines                        -> list items
    paragraphs  running text, cut at sentences    -> paragraphs, quotes, captions, footnotes, sidebars
Reference sections are skipped, and lines with much text in another script are dropped (an Arabic line full
of Latin words, say), so a page stays in one language. Every piece keeps its article, so each sample's
labels can say which articles its text came from (CC BY-SA asks for attribution).

    .venv-pipeline\\Scripts\\python -m pipeline.text_pool --articles 5000
"""
import argparse
import json
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "text_pool"
DATASET, DUMP = "wikimedia/wikipedia", "20231101"
ROWS = "https://datasets-server.huggingface.co/rows?dataset={d}&config={c}&split=train&offset={o}&length=100"
SIZE = "https://datasets-server.huggingface.co/size?dataset={d}&config={c}"
LICENSE = "CC BY-SA 4.0 (Wikipedia)"
# Sections after which an article is only references and links
STOP = {"ar": ("المراجع", "مراجع", "المصادر", "مصادر", "وصلات خارجية", "انظر أيضا", "انظر أيضًا", "روابط خارجية",
               "هوامش", "ملاحظات", "قراءات إضافية", "المراجع والمصادر"),
        "en": ("References", "See also", "External links", "Notes", "Further reading", "Bibliography",
               "Sources", "Citations", "Footnotes")}
ARABIC = re.compile(r"[؀-ۿ]")
LATIN = re.compile(r"[A-Za-z]")
SENTENCE_END = {"ar": re.compile(r"(?<=[.!؟])\s+"), "en": re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")}


def get(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "OCRSynthGen/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def one_script(text, lang):
    """True if the letters are (almost) all in the page's script."""
    ar, la = len(ARABIC.findall(text)), len(LATIN.findall(text))
    return (ar >= 0.95 * (ar + la) and ar > 0) if lang == "ar" else (la >= 0.98 * (ar + la) and la > 0)


def clean(line):
    return re.sub(r"\s+", " ", line.replace("‏", "").replace("‎", "")).strip()


def split_article(text, lang):
    """(headings, items, paragraphs) of one article, stopping at its reference sections."""
    headings, items, paragraphs = [], [], []
    for raw in text.split("\n"):
        line = clean(raw)
        if not line:
            continue
        if line.rstrip(" :") in STOP[lang]:
            break
        if not one_script(line, lang) or any(c in line for c in "{}[]|=<>"):
            continue
        words = len(line.split())
        if raw.startswith(" ") and 2 <= words <= 25:
            items.append(line)
        elif words >= 15:
            paragraphs.append(line)
        elif 1 <= words <= 8 and not line.endswith((".", "!", "?", "؟", ":", "،", ",")):
            headings.append(line)
    return headings, items, paragraphs


def fetch(lang, offset):
    try:
        return get(ROWS.format(d=DATASET, c=f"{DUMP}.{lang}", o=offset))["rows"]
    except Exception as e:
        print(f"  {lang} offset {offset}: {type(e).__name__}", flush=True)
        return []


def build(lang, articles, seed=0):
    total = get(SIZE.format(d=DATASET, c=f"{DUMP}.{lang}"))["size"]["config"]["num_rows"]
    rng = random.Random(seed)
    offsets = [rng.randrange(total - 100) for _ in range(articles // 100)]
    pool = {"source": f"{DATASET} {DUMP}.{lang}", "license": LICENSE, "articles": [],
            "titles": [], "headings": [], "items": [], "paragraphs": []}
    with ThreadPoolExecutor(4) as ex:
        for rows in ex.map(lambda o: fetch(lang, o), offsets):
            for row in rows:
                r = row["row"]
                headings, items, paragraphs = split_article(r["text"], lang)
                if not paragraphs:
                    continue
                a = len(pool["articles"])
                pool["articles"].append({"title": r["title"], "url": r["url"]})
                title = clean(r["title"])
                if one_script(title, lang) and len(title.split()) <= 10:
                    pool["titles"].append([title, a])
                pool["headings"] += [[h, a] for h in headings]
                pool["items"] += [[i, a] for i in items]
                pool["paragraphs"] += [[p, a] for p in paragraphs]
    POOL.mkdir(exist_ok=True)
    (POOL / f"{lang}.json").write_text(json.dumps(pool, ensure_ascii=False), encoding="utf-8")
    print(f"{lang}: {len(pool['articles'])} articles -> {len(pool['titles'])} titles, {len(pool['headings'])} "
          f"headings, {len(pool['items'])} list items, {len(pool['paragraphs'])} paragraphs", flush=True)


class TextPool:
    """Wikipedia text for one language, cut to the length a slot wants. Records the articles it used."""

    def __init__(self, lang, folder=POOL):
        data = json.loads((Path(folder) / f"{lang}.json").read_text(encoding="utf-8"))
        self.lang, self.articles, self.license = lang, data["articles"], data["license"]
        self.titles, self.headings, self.items = data["titles"], data["headings"], data["items"]
        self.paragraphs = data["paragraphs"]
        self.used = set()
        self._by_length = {}
        for h in self.headings + self.titles:
            self._by_length.setdefault(len(h[0].split()), []).append(h)

    @staticmethod
    def exists(lang, folder=POOL):
        return (Path(folder) / f"{lang}.json").exists()

    def _take(self, entry):
        self.used.add(entry[1])
        return entry[0]

    def sources(self):
        return [self.articles[a] for a in sorted(self.used)]

    def short(self, rng, n):
        """A heading-like line of about n words (a heading or title of that length, else words of a sentence)."""
        near = [k for k in range(max(1, n - 1), n + 2) if k in self._by_length]
        if near:
            return self._take(rng.choice(self._by_length[rng.choice(near)]))
        return self.words(rng, n)

    def words(self, rng, n):
        """n consecutive words from a paragraph, without its closing full stop."""
        entry = rng.choice(self.paragraphs)
        words = self._take(entry).split()
        start = rng.randrange(max(1, len(words) - n + 1))
        return " ".join(words[start:start + n]).rstrip(".،,؛;:")

    def prose(self, rng, n):
        """n words of running text: consecutive sentences from a paragraph (and the next ones from another
        paragraph if it runs out), ending on a whole sentence when one ends close to n, else cut at n words."""
        words = []
        while len(words) < n:
            sentences = SENTENCE_END[self.lang].split(self._take(rng.choice(self.paragraphs)))
            start = rng.randrange(len(sentences)) if not words else 0
            for s in sentences[start:]:
                if len(words) >= n:
                    break
                if words and len(words) >= 0.8 * n and len(words) + len(s.split()) > n:
                    return " ".join(words)  # a whole sentence ends near n: stop there
                words += s.split()
        words = words[:n]
        words[-1] = words[-1].rstrip(".،,؛;:") + "."
        return " ".join(words)

    def item(self, rng, n):
        fits = [i for i in rng.sample(self.items, min(40, len(self.items))) if len(i[0].split()) <= max(n, 3) + 3]
        return self._take(rng.choice(fits)) if fits else self.prose(rng, n)

    def text(self, role, n, rng):
        """Text for a slot of this role and about n words."""
        if role in ("title", "subtitle", "heading", "table_header", "header", "footer"):
            return self.short(rng, n)
        if role == "label":
            return self.short(rng, min(n, 3)) + ":"
        if role == "list_item":
            return self.item(rng, n)
        if role in ("paragraph", "quote", "sidebar", "footnote", "caption"):
            return self.prose(rng, n)
        return self.words(rng, n)  # cells, field values and anything else: a few words


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", type=int, default=5000, help="random articles per language")
    ap.add_argument("--lang", nargs="+", default=["ar", "en"])
    a = ap.parse_args()
    for lang in a.lang:
        build(lang, a.articles)
