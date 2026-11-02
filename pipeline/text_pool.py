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
# words a short text (a table cell, a field value) should not end on: it would read as cut off
FUNCTION_WORDS = dict.fromkeys(
    "a an the of in on at to for by with from into and or but as that which who whose than is are was were be been "
    "being has have had it its this these those his her their our we they he she there not also such only most more very "
    "في من على إلى الى عن مع أن ان إن التي الذي الذين و أو ثم بين عند حتى منذ هو هي هذا هذه ذلك تلك كان كانت قد "
    "لم لن ما لا كما بعد قبل حيث لكن ولكن".split(), True)
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

    # -- document mode: one page follows one article ----------------------------------------------------------
    # begin(rng) picks an article for the page; then the title is its title, headings its section headings in
    # order, running text continues its paragraphs sentence by sentence, list items come from its lists. Only
    # when it runs out does the page move on to another article, so a page reads as one document on one
    # subject instead of sentences from many.

    def _index(self):
        if not hasattr(self, "_by_article"):
            self._by_article = {}
            for kind in ("paragraphs", "headings", "items", "titles"):
                for entry in getattr(self, kind):
                    self._by_article.setdefault(entry[1], {}).setdefault(kind, []).append(entry[0])
            self._rich = [a for a, d in self._by_article.items() if len(d.get("paragraphs", [])) >= 3]
        return self._by_article

    def begin(self, rng):
        """Start a page: pick its article."""
        self._index()
        self.used.clear()
        self._doc = True
        self._title_used = False
        self._switch(rng)

    def _switch(self, rng):
        self._cur = rng.choice(self._rich)
        d = self._by_article[self._cur]
        self._sentences = [s for p in d.get("paragraphs", []) for s in SENTENCE_END[self.lang].split(p) if s.strip()]
        self._s = 0
        self._headings = list(d.get("headings", []))
        self._items = list(d.get("items", []))
        self.used.add(self._cur)

    def _next_sentence(self, rng):
        if self._s >= len(self._sentences):
            self._switch(rng)
        self._s += 1
        return self._sentences[self._s - 1]

    def doc_prose(self, rng, n):
        """n words continuing the page's article; ends on a whole sentence when one ends at 60-100% of n."""
        words = []
        while len(words) < n:
            s = self._next_sentence(rng).split()
            if words and len(words) >= 0.6 * n and len(words) + len(s) > n:
                self._s -= 1  # this sentence starts the next slot instead
                return " ".join(words)
            words += s
        words = words[:n]
        words[-1] = words[-1].rstrip(".،,؛;:") + "."
        return " ".join(words)

    def doc_short(self, rng, n, kind="heading"):
        """A heading of about n words from the page's article (its title first, for the page title)."""
        title = self.articles[self._cur]["title"]
        if kind == "title" and not self._title_used and len(title.split()) <= n + 4:
            self._title_used = True
            return title
        for i, h in enumerate(self._headings):
            if len(h.split()) <= n + 2:
                return self._headings.pop(i)
        # This article has no more headings that fit: the page moves on to another article (a new section on
        # a new subject), headed by that article's title, so heading and text stay on the same subject
        for _ in range(5):
            self._switch(rng)
            title = self.articles[self._cur]["title"]
            if len(title.split()) <= n + 2:
                return title
        return " ".join(title.split()[:n + 2])

    def doc_item(self, rng, n):
        for i, it in enumerate(self._items):
            if len(it.split()) <= max(n, 3) + 3:
                return self._items.pop(i)
        limit = max(n, 3) + 3
        for _ in range(4):  # a whole sentence that fits, if one of the next few does
            sentence = self._next_sentence(rng).split()
            if len(sentence) <= limit:
                return " ".join(sentence)
        return " ".join(sentence[:limit]).rstrip(".،,؛;:") + "."

    def doc_words(self, rng, n):
        """The start of the next sentence, about n words, not ending on a word like "of" or "is" ("Zeus is an"
        becomes "Zeus"); a start made only of such words ("It is") is skipped for the next sentence."""
        for _ in range(4):
            words = [w.rstrip(".،,؛;:") for w in self._next_sentence(rng).split()[:n]]
            while words and FUNCTION_WORDS.get(words[-1].lower()):
                words.pop()
            if words and not all(FUNCTION_WORDS.get(w.lower()) for w in words):
                return " ".join(words)
        return " ".join(words) if words else self.doc_short(rng, n)

    def text(self, role, n, rng):
        """Text for a slot of this role and about n words (from the page's article after begin())."""
        if getattr(self, "_doc", False):
            if role in ("title", "subtitle", "heading", "table_header", "header", "footer"):
                return self.doc_short(rng, n, role)
            if role == "label":
                return " ".join(self.doc_short(rng, min(n, 3)).split()[:3]).rstrip(".،,؛;:") + ":"
            if role == "list_item":
                return self.doc_item(rng, n)
            if role in ("paragraph", "quote", "sidebar", "footnote", "caption"):
                return self.doc_prose(rng, n)
            return self.doc_words(rng, n)
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
