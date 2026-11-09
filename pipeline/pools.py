"""The text and photo pools the generator fills pages from (made once, not in git).

    python -m pipeline.pools text --articles 5000     -> text_pool/ (Wikipedia, Arabic and English)
    python -m pipeline.pools images                   -> img_pool/ (captioned pictures, see below)
    python -m pipeline.pools fonts                    -> font_pool/ (100 font families, see below)
    python -m pipeline.pools graphics                 -> img_pool/graphics/ (charts, flowcharts, molecules, equations)
"""
import argparse
import hashlib
import io
import json
import random
import re
import shutil
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


# ====================================================================================================================
# text_pool/: Wikipedia text
# ====================================================================================================================
# text_pool/: real Arabic and English text for the template slots, from Wikipedia.
#
# Wikipedia is used as the trusted source: written and checked by editors, cited, openly licensed
# (CC BY-SA 4.0), and broad in topic, so pages get a real vocabulary instead of a few demo sentences. The
# articles come from the wikimedia/wikipedia dataset (dump of 2023-11-01) through Hugging Face's dataset
# server, a few thousand random articles per language; nothing else is downloaded.
#
# Each article is split into the kinds of text a document has, and each kind fills the matching slots:
#     titles      article titles                   -> title slots
#     headings    section headings (short lines)    -> headings, table headers, labels
#     items       list lines                        -> list items
#     paragraphs  running text, cut at sentences    -> paragraphs, quotes, captions, footnotes, sidebars
# Reference sections are skipped, and lines with much text in another script are dropped (an Arabic line full
# of Latin words, say), so a page stays in one language. Every piece keeps its article, so each sample's
# labels can say which articles its text came from (CC BY-SA asks for attribution).
#
#     .venv-pipeline\\Scripts\\python -m pipeline.pools text --articles 5000

TEXT_POOL = ROOT / "text_pool"
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


def get_json(url, tries=4):
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
        return get_json(ROWS.format(d=DATASET, c=f"{DUMP}.{lang}", o=offset))["rows"]
    except Exception as e:
        print(f"  {lang} offset {offset}: {type(e).__name__}", flush=True)
        return []


def build_text_pool(lang, articles, seed=0):
    total = get_json(SIZE.format(d=DATASET, c=f"{DUMP}.{lang}"))["size"]["config"]["num_rows"]
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
    TEXT_POOL.mkdir(exist_ok=True)
    (TEXT_POOL / f"{lang}.json").write_text(json.dumps(pool, ensure_ascii=False), encoding="utf-8")
    print(f"{lang}: {len(pool['articles'])} articles -> {len(pool['titles'])} titles, {len(pool['headings'])} "
          f"headings, {len(pool['items'])} list items, {len(pool['paragraphs'])} paragraphs", flush=True)


class TextPool:
    """Wikipedia text for one language, cut to the length a slot wants. Records the articles it used."""

    def __init__(self, lang, folder=TEXT_POOL):
        data = json.loads((Path(folder) / f"{lang}.json").read_text(encoding="utf-8"))
        self.lang, self.articles, self.license = lang, data["articles"], data["license"]
        self.titles, self.headings, self.items = data["titles"], data["headings"], data["items"]
        self.paragraphs = data["paragraphs"]
        self.used = set()
        self._by_length = {}
        for h in self.headings + self.titles:
            self._by_length.setdefault(len(h[0].split()), []).append(h)

    @staticmethod
    def exists(lang, folder=TEXT_POOL):
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




# ====================================================================================================================
# img_pool/: captioned pictures
# ====================================================================================================================
# img_pool/: pictures for the templates' picture slots (<img> tags), each with captions written by people or a
# captioning model. Surya describes every picture it sees (<img alt="Photo: ..."/><p class="img-desc">...</p>),
# so a picture's label is made from its own captions (generator.picture_html). Three datasets:
#
#   * output_dataset/k*/  (uploaded; Flickr8k): 8,000 photos in Images/, 5 captions each in captions.txt
#     ("<file> <caption>" per line). Flickr8k is for research use.
#   * Polaris (huggingface.co/datasets/yuwd/Polaris, BSD-3-Clause): photos (COCO / nocaps) with 5-10 human
#     reference captions ("refs"); the rated candidate captions ("cand") are not used. -> polaris/
#   * PD12M (huggingface.co/datasets/Spawning/PD12M, public domain / CC0): photos and museum pieces with one
#     long generated caption ("The image shows ..."). -> pd12m/
#
# Every picture is saved again with Pillow (plain JPEG, at most MAX_SIDE px on its longer side). Pictures whose
# captions suggest text in them (signs, posters, maps, coins, letters...) are skipped: text in a picture would
# put words on the page that the labels don't know about. The index, img_pool/captions.jsonl, has one line per
# picture: {"file": path in img_pool, "source", "license", "captions": [...], "category"}.
#
# A template's picture slot carries a hint (from the original page: an image link's keyword such as "?food",
# the alt text or class names such as "team-photo"). ImagePool.pick(rng, hint) maps the hint to a category and
# picks from it, so a food page gets food photos and a team section gets people. A picture's category comes from
# its captions (the same words).
#
#     .venv-pipeline\\Scripts\\python -m pipeline.pools images --polaris-shards 30 --pd12m 20000

IMG_POOL = ROOT / "img_pool"
INDEX = IMG_POOL / "captions.jsonl"
UPLOADED = IMG_POOL / "output_dataset"
MAX_SIDE = 1024
TEXTY = re.compile(r"\b(sign|signs|signage|poster|label|logo|text|texts|writing|written|words?|letters?|map|chart|"
                   r"diagram|inscription|plaque|menu|book|books|newspaper|magazine|banner|graffiti|billboard|"
                   r"scoreboard|advert\w*|ticket|stamp|coins?|medal|banknote|document|manuscript|page|calligraphy|"
                   r"typography|font|numbers?|license plate|screen|monitor|brand|packag\w*|title|caption|"
                   r"certificate|card|cards|seal|emblem|badge)\b", re.I)
# Words of a picture's hint (or captions) -> category (the first that matches)
HINTS = [("food", r"food|restaurant|cafe|coffee|recipe|menu|bakery|dish|meal|kitchen|chef|drink|pizza|fruit|"
                  r"wine|tea\b|cake|grocery|catering"),
         ("people", r"team|person|people|portrait|avatar|profile|user|author|testimonial|doctor|teacher|staff|"
                    r"member|founder|ceo|client|customer|student|speaker|instructor|trainer|face|headshot|man|woman|"
                    r"fashion|clothing|wear|style|beauty|salon|spa|model|wedding|family|kid|child|patient|"
                    r"attorney|lawyer|employee|volunteer|artist|musician|girl|boy|men\b|women"),
         ("animals", r"animal|pet|dog|cat|zoo|wildlife|bird|horse|vet|cow|sheep|insect|butterfl|fish"),
         ("sports", r"sport|fitness|gym|football|soccer|basketball|tennis|run|yoga|athlet|game|match|stadium|"
                    r"skate|surf|ski\b|player"),
         ("transport", r"car|transport|vehicle|auto|bike|train|plane|flight|airline|bus|truck|ship|logistic|"
                       r"shipping|delivery|taxi|motor|boat"),
         ("architecture", r"city|building|architecture|office|hotel|real.?estate|house|home|interior|property|"
                          r"apartment|construction|urban|church|museum|bank|company|business|corporate|"
                          r"technolog|tech|software|startup|finance|consult|agency|service|law|legal|school|"
                          r"universit|hospital|clinic|event|conference|bridge|tower|castle"),
         ("landscapes", r"travel|landscape|mountain|beach|hero|banner|background|tour|destination|vacation|"
                        r"holiday|resort|sea|ocean|lake|desert|scenery|river|sky"),
         ("nature", r"nature|garden|forest|park|flower|plant|tree|green|eco|environment|farm|agricultur|leaf|grass"),
         ("patterns", r"abstract|pattern|texture|art|design|gallery|portfolio|creative|wallpaper|decor")]


def clean_caption(text):
    """One caption tidied: spaces, no space before punctuation, a capital first letter, a full stop; a caption
    cut off mid-sentence (PD12M's are cut at a length) keeps only its complete sentences ("" when none is)."""
    text = re.sub(r"\s+", " ", str(text)).strip().strip('"')
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    if not text:
        return ""
    if text[-1] not in ".!?":
        cut = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
        text = text[:cut + 1] if cut > 0 else (text + "." if len(text.split()) <= 25 else "")
    return text[:1].upper() + text[1:] if text else ""


def save(data, path):
    """Open, shrink to MAX_SIDE and save again as JPEG; returns (width, height)."""
    image = Image.open(io.BytesIO(data))
    image.draft("RGB", (MAX_SIDE, MAX_SIDE))  # JPEG: decode at a smaller size, much faster for big photos
    image = image.convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=85)
    return image.size


def record(file, source, license_, captions):
    """An index line, or None when the picture has no usable caption or its captions suggest text in it."""
    captions = [c for c in (clean_caption(c) for c in captions) if c]
    if not captions or any(TEXTY.search(c) for c in captions):
        return None
    return {"file": file, "source": source, "license": license_, "captions": captions,
            "category": hint_category(" ".join(captions)) or "general"}


def uploaded_records():
    """The uploaded Flickr8k folders: output_dataset/k*/Images/<file> with output_dataset/k*/captions.txt."""
    out = []
    for folder in sorted(UPLOADED.glob("k*")):
        captions = {}
        for line in (folder / "captions.txt").read_text(encoding="utf-8").splitlines():
            name, _, caption = line.strip().partition(" ")
            if name and caption:
                captions.setdefault(name, []).append(caption)
        for name, caps in sorted(captions.items()):
            if (folder / "Images" / name).exists():
                r = record(f"output_dataset/{folder.name}/Images/{name}", "Flickr8k", "research use", caps)
                if r:
                    out.append(r)
    return out


def polaris_records(shards):
    """Polaris: the first `shards` train shards, each picture once with its human reference captions."""
    from huggingface_hub import HfFileSystem, hf_hub_download
    import pyarrow.parquet as pq
    files = sorted(f.split("Polaris/", 1)[1] for f in HfFileSystem().glob("datasets/yuwd/Polaris/data/train-*.parquet"))
    out, seen = [], set()
    for k, name in enumerate(files[:shards]):
        local = Path(hf_hub_download("yuwd/Polaris", name, repo_type="dataset", local_dir=IMG_POOL / "_download"))
        for batch in pq.ParquetFile(local).iter_batches(batch_size=100, columns=["refs", "img"]):  # small batches:
            for refs, img in zip(batch.column("refs").to_pylist(), batch.column("img").to_pylist()):  # little memory
                key = hashlib.md5(img["bytes"]).hexdigest()[:16]
                if key in seen:
                    continue
                seen.add(key)
                r = record(f"polaris/{key}.jpg", "Polaris", "BSD-3-Clause (images: COCO / nocaps)", refs)
                if r:
                    if not (IMG_POOL / r["file"]).exists():
                        try:
                            save(img["bytes"], IMG_POOL / r["file"])
                        except Exception:  # noqa: BLE001  (a picture Pillow can't open)
                            continue
                    out.append(r)
        local.unlink()
        print(f"  polaris shard {k + 1}/{min(shards, len(files))}: {len(out)} pictures", flush=True)
    return out


def fetch_pd12m(row):
    r = record(f"pd12m/{row['id']}.jpg", f"PD12M ({row['source']})", row["license"], [row["caption"]])
    if not r:
        return None
    if not (IMG_POOL / r["file"]).exists():
        for k in range(3):
            try:
                with urllib.request.urlopen(row["url"], timeout=60) as f:
                    save(f.read(), IMG_POOL / r["file"])
                break
            except Exception:  # noqa: BLE001  (network, or not an image Pillow can open)
                if k == 2:
                    return None
                time.sleep(2 * (k + 1))
    return r


def pd12m_records(n, threads=16, seed=0):
    """PD12M: n pictures drawn at random from its metadata shards (so many collections are mixed), downloaded."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    rng = random.Random(seed)
    rows = []
    for s in rng.sample(range(125), 12):  # 12 of the 125 metadata shards, 100,000 rows each
        t = pq.read_table(hf_hub_download("Spawning/PD12M", f"metadata/pd12m.{s:03d}.parquet", repo_type="dataset",
                                          local_dir=IMG_POOL / "_download"),
                          columns=["id", "url", "caption", "width", "height", "license", "source"])
        rows += [r for r in t.to_pylist() if min(int(r["width"]), int(r["height"])) >= 300
                 and 0.4 <= int(r["width"]) / max(1, int(r["height"])) <= 2.5]
    rng.shuffle(rows)
    out, k = [], 0
    with ThreadPoolExecutor(threads) as pool:
        while len(out) < n and k < len(rows):
            batch = rows[k:k + 4 * threads]
            k += len(batch)
            before = len(out)
            out += [r for r in pool.map(fetch_pd12m, batch) if r]
            if len(out) // 1000 != before // 1000:
                print(f"  pd12m: {len(out)} pictures", flush=True)
    return out[:n]


def build_img_pool(polaris_shards=30, pd12m=20000):
    """img_pool/captions.jsonl from the uploaded Flickr8k folders, Polaris and PD12M (files already there are
    kept, so a stopped run can be started again)."""
    IMG_POOL.mkdir(exist_ok=True)
    records = uploaded_records()
    print(f"Flickr8k (uploaded): {len(records)} pictures", flush=True)
    records += polaris_records(polaris_shards)
    records += pd12m_records(pd12m)
    INDEX.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    shutil.rmtree(IMG_POOL / "_download", ignore_errors=True)
    counts = {}
    for r in records:
        counts[r["source"].split(" ")[0]] = counts.get(r["source"].split(" ")[0], 0) + 1
    print(f"{len(records)} pictures in {INDEX}: {counts}")


def hint_category(hint):
    """The category a picture hint asks for, or None (any picture will do). Words match from their start
    ("team" is people, not "tea")."""
    hint = (hint or "").lower()
    return next((category for category, words in HINTS if re.search(rf"\b(?:{words})", hint)), None)


# ====================================================================================================================
# font_pool/: fonts
# ====================================================================================================================
# font_pool/: font files for the pages, from Google Fonts (SIL Open Font License or Apache 2.0; made once, not in
# git). Families are chosen by Google Fonts' own popularity rank (its catalog, fonts.google.com/metadata/fonts):
#
#   top-arabic   the 10 most used families with Arabic letters (every one has Latin letters too)
#   top-latin    the 10 most used families for English (serif and sans serif)
#   extra        80 more: every other family with Arabic letters, then English families of every kind (serif,
#                sans serif, display, handwriting, monospace for code), most used first
#
# Left out: brand fonts, colour and icon fonts, fonts made for another script (Japanese, say) and Nastaliq
# (calligraphy for Urdu, not used in Arabic print). Each family's regular (400) and bold (700) TrueType files are
# kept only if they have every character their pages need (checked in the file's character map with fontTools):
# Arabic letters, the tashkeel marks and Arabic-Indic digits for an Arabic family; A-Z, a-z, 0-9 and punctuation for
# a Latin one. The index, font_pool/fonts.json: [{"family", "group", "category", "scripts", "files"}].
#
#     .venv-pipeline\\Scripts\\python -m pipeline.pools fonts

FONT_POOL = ROOT / "font_pool"
FONT_INDEX = FONT_POOL / "fonts.json"
GOOGLE_FONTS = "https://fonts.google.com/metadata/fonts"
GOOGLE_CSS = "https://fonts.googleapis.com/css2?family={family}:wght@{weights}"
ARABIC_NEEDS = [chr(c) for c in [*range(0x0621, 0x063B), *range(0x0641, 0x0653), *range(0x0660, 0x066A)]]
LATIN_NEEDS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()%-'\"")
LATIN_KINDS = ["Serif", "Sans Serif", "Serif", "Monospace", "Sans Serif", "Display", "Serif", "Handwriting"]


def download(url, headers=None, timeout=60, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "OCRSynthGen/1.0", **(headers or {})})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def font_catalog():
    """Google Fonts' catalog: usable families, most used first."""
    text = download(GOOGLE_FONTS).decode("utf-8")
    families = json.loads(text[text.index("{"):])["familyMetadataList"]
    usable = [f for f in families if f.get("isOpenSource", True) and not f.get("isBrandFont")
              and not f.get("colorCapabilities") and "400" in f["fonts"] and "latin" in f["subsets"]
              and f.get("primaryScript", "") in ("", "Arab", "Latn")
              and not re.search(r"Nastaliq|Icons|Symbols|Emoji", f["family"])]
    return sorted(usable, key=lambda f: f["popularity"])


def font_order(catalog):
    """[(family record, group)]: the top 10 Arabic, the top 10 English, then the extras in their order."""
    arabic = [f for f in catalog if "arabic" in f["subsets"]]
    latin = [f for f in catalog if "arabic" not in f["subsets"]]
    top_latin = [f for f in latin if f["category"] in ("Serif", "Sans Serif")]
    out = [(f, "top-arabic") for f in arabic[:10]] + [(f, "top-latin") for f in top_latin[:10]]
    out += [(f, "extra") for f in arabic[10:]]
    taken = {f["family"] for f, _ in out}
    by_kind = {k: [f for f in latin if f["category"] == k and f["family"] not in taken] for k in set(LATIN_KINDS)}
    while any(by_kind.values()):  # the other English families, kind by kind in LATIN_KINDS' proportions
        for kind in LATIN_KINDS:
            if by_kind[kind]:
                out.append((by_kind[kind].pop(0), "extra"))
    return out


def font_covers(path, needs):
    from fontTools.ttLib import TTFont
    cmap = TTFont(path, lazy=True).getBestCmap() or {}
    return all(ord(c) in cmap for c in needs)


def fetch_family(record):
    """{weight: path} of a family's regular and bold TrueType files (bold left out if the family has none)."""
    family = record["family"]
    weights = [w for w in ("400", "700") if w in record["fonts"]]
    css = download(GOOGLE_CSS.format(family=urllib.parse.quote(family), weights=";".join(weights)),
                   headers={"User-Agent": "curl/8.0"}).decode("utf-8")  # a plain client gets whole .ttf files
    folder = FONT_POOL / re.sub(r"[^A-Za-z0-9]+", "_", family)
    files = {}
    for block in re.findall(r"@font-face \{(.*?)\}", css, re.S):  # weight and file, whatever else the block says
        weight, url = re.search(r"font-weight: (\d+);", block), re.search(r"src: url\((\S+?\.ttf)\)", block)
        if not (weight and url):
            continue
        weight, url = weight.group(1), url.group(1)
        path = folder / f"{weight}.ttf"
        if not path.exists():
            folder.mkdir(parents=True, exist_ok=True)
            path.write_bytes(download(url))
        files[weight] = path
    return files


def build_font_pool(top=10, extra=80):
    FONT_POOL.mkdir(exist_ok=True)
    quota = {"top-arabic": top, "top-latin": top, "extra": extra}
    kept, done = [], {k: 0 for k in quota}
    for record, group in font_order(font_catalog()):
        if done[group] >= quota[group]:
            continue
        scripts = ["arabic", "latin"] if "arabic" in record["subsets"] else ["latin"]
        try:
            files = fetch_family(record)
        except Exception as e:  # noqa: BLE001  (a family the API won't give as .ttf: the next one takes its place)
            print(f"  {record['family']}: skipped ({type(e).__name__})", flush=True)
            continue
        needs = (ARABIC_NEEDS if "arabic" in scripts else []) + LATIN_NEEDS
        if "400" not in files:
            print(f"  {record['family']}: skipped (no regular TrueType file given)", flush=True)
            continue
        if not all(font_covers(p, needs) for p in files.values()):
            print(f"  {record['family']}: skipped (missing characters)", flush=True)
            continue
        kept.append({"family": record["family"], "group": group, "category": record["category"], "scripts": scripts,
                     "popularity": record["popularity"], "files": {w: p.relative_to(ROOT).as_posix() for w, p in files.items()}})
        done[group] += 1
        print(f"  {group:10s} {record['family']} ({record['category']}, {'+'.join(scripts)})", flush=True)
        if done == quota:
            break
    FONT_INDEX.write_text(json.dumps(kept, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(kept)} families in {FONT_INDEX}: {done}")


def font_pool():
    """{family: {"scripts", "category", "files": {weight: Path}}} from font_pool/fonts.json ({} if not made)."""
    if not FONT_INDEX.exists():
        return {}
    return {f["family"]: {"scripts": f["scripts"], "category": f["category"],
                          "files": {int(w): ROOT / p for w, p in f["files"].items()}}
            for f in json.loads(FONT_INDEX.read_text(encoding="utf-8"))}


# ====================================================================================================================
# img_pool/graphics/: charts, flowcharts, molecules, equations
# ====================================================================================================================
# Pictures Surya labels other than a photo, each drawn once with the exact answer Surya gives for it (its format
# taken from Surya's own answers on real pages, surya_inference/debug_docling_outputs):
#
#   Figure          a bar, line, pie or scatter chart (matplotlib, random data): <img alt="Bar chart: ..."/>, the
#                   legend in <p class="img-text">, the axes in <p class="chart-axes">, the data drawn in
#                   <table class="chart-data">, a description in <p class="img-desc">
#   Diagram         a flowchart (matplotlib boxes and arrows): <img alt="Flowchart: ..."/>, its words in
#                   <p class="img-text">, the same chart as Mermaid in <pre class="mermaid">, <p class="img-desc">
#   Chemical-Block  a molecule (2D layout by RDKit, drawn with matplotlib; molecules from ZINC-250k):
#                   <img alt="Chemical structure of the molecule SMILES."/>
#   Equation-Block  a displayed equation (matplotlib's mathtext): <math display="block">LaTeX</math>
#
# The words in charts and flowcharts are Arabic on Arabic ones (matplotlib 3.11 joins and orders Arabic itself),
# English on English ones; descriptions are English, as Surya writes them.
# Every index line: {"file", "label", "lang" ("ar", "en" or "any"), "html"} (+ "height_em" for equations).
#
#     .venv-pipeline\\Scripts\\python -m pipeline.pools graphics

GRAPHICS = IMG_POOL / "graphics"
GRAPHICS_INDEX = GRAPHICS / "graphics.jsonl"
ZINC = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
ARABIC_FONT = ROOT / "originals" / "ocrsmith_assets" / "fonts" / "IBMPlexSansArabic-Regular.ttf"
TOPICS = [("Sales", "المبيعات"), ("Revenue", "الإيرادات"), ("Population", "عدد السكان"), ("Students", "عدد الطلاب"),
          ("Exports", "الصادرات"), ("Average temperature", "متوسط درجة الحرارة"), ("Production", "الإنتاج"),
          ("Visitors", "عدد الزوار"), ("Operating costs", "تكاليف التشغيل"), ("Rainfall", "كمية الأمطار"),
          ("Net profit", "صافي الربح"), ("Orders", "عدد الطلبات"), ("Energy use", "استهلاك الطاقة"),
          ("Graduates", "عدد الخريجين"), ("Patients", "عدد المرضى"), ("Market share", "الحصة السوقية")]
AXES = {"year": ("Year", "السنة", [str(y) for y in range(2012, 2025)], [str(y) for y in range(2012, 2025)]),
        "month": ("Month", "الشهر", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                  ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]),
        "quarter": ("Quarter", "الربع", ["Q1", "Q2", "Q3", "Q4"], ["الأول", "الثاني", "الثالث", "الرابع"]),
        "city": ("City", "المدينة", ["Riyadh", "Jeddah", "Dammam", "Cairo", "Dubai", "Amman", "Doha", "Kuwait"],
                 ["الرياض", "جدة", "الدمام", "القاهرة", "دبي", "عمّان", "الدوحة", "الكويت"]),
        "sector": ("Sector", "القطاع", ["Health", "Education", "Industry", "Trade", "Transport", "Tourism"],
                   ["الصحة", "التعليم", "الصناعة", "التجارة", "النقل", "السياحة"])}
SERIES = [(["2023", "2024"], ["2023", "2024"]), (["North", "South"], ["الشمال", "الجنوب"]),
          (["Men", "Women"], ["الرجال", "النساء"]), (["Plan", "Actual"], ["المخطط", "الفعلي"]),
          (["Product A", "Product B", "Product C"], ["المنتج أ", "المنتج ب", "المنتج ج"])]
STEPS = [("Receive request", "استلام الطلب"), ("Review documents", "مراجعة المستندات"), ("Collect data", "جمع البيانات"),
         ("Clean the data", "تنظيف البيانات"), ("Train the model", "تدريب النموذج"), ("Evaluate results", "تقييم النتائج"),
         ("Approve budget", "اعتماد الميزانية"), ("Prepare report", "إعداد التقرير"), ("Send to manager", "الإرسال إلى المدير"),
         ("Update records", "تحديث السجلات"), ("Notify customer", "إشعار العميل"), ("Archive the file", "أرشفة الملف"),
         ("Schedule meeting", "جدولة الاجتماع"), ("Test the system", "اختبار النظام"), ("Deliver the order", "تسليم الطلب"),
         ("Register the student", "تسجيل الطالب"), ("Pay the fees", "دفع الرسوم"), ("Issue the certificate", "إصدار الشهادة")]
QUESTIONS = [("Is it complete?", "هل هو مكتمل؟"), ("Approved?", "هل تمت الموافقة؟"), ("Is the data valid?", "هل البيانات صحيحة؟"),
             ("Paid?", "هل تم الدفع؟")]
FIXED = {"start": ("Start", "البداية"), "end": ("End", "النهاية"), "yes": ("Yes", "نعم"), "no": ("No", "لا")}
EQUATIONS = [r"E = m c^{2}", r"F = G \frac{m_1 m_2}{r^{2}}", r"x = \frac{-b \pm \sqrt{b^{2} - 4ac}}{2a}",
             r"\int_{{a}}^{{b}} f(x)\,dx = F(b) - F(a)", r"\sum_{{i=1}}^{{{n}}} i = \frac{{{n}({n}+1)}}{{2}}",
             r"\bar{{x}} = \frac{{1}}{{{n}}} \sum_{{i=1}}^{{{n}}} x_i", r"\sigma^{{2}} = \frac{{1}}{{N}} \sum_{{i=1}}^{{N}} (x_i - \mu)^{{2}}",
             r"a^{{2}} + b^{{2}} = c^{{2}}", r"\frac{{d}}{{dx}} \left( x^{{{k}}} \right) = {k} x^{{{k1}}}", r"e^{{i \pi}} + 1 = 0",
             r"P(A \mid B) = \frac{{P(B \mid A)\, P(A)}}{{P(B)}}", r"\lim_{{x \to 0}} \frac{{\sin x}}{{x}} = 1",
             r"f(x) = {a} x^{{2}} + {b} x + {c}", r"y = {a} \sin({k} x) + {c}", r"\nabla \cdot \mathbf{{E}} = \frac{{\rho}}{{\varepsilon_0}}",
             r"\hat{{y}} = \mathrm{{softmax}}(W x + b)", r"L = -\sum_{{i=1}}^{{{n}}} y_i \log \hat{{y}}_i",
             r"\theta_{{t+1}} = \theta_t - \eta \nabla_{{\theta}} J(\theta_t)", r"A = \pi r^{{2}}", r"V = \frac{{4}}{{3}} \pi r^{{3}}",
             r"s = v_0 t + \frac{{1}}{{2}} a t^{{2}}", r"\binom{{{n}}}{{{k}}} = \frac{{{n}!}}{{{k}!\,({n}-{k})!}}",
             r"\int_{{0}}^{{\infty}} e^{{-x^{{2}}}}\,dx = \frac{{\sqrt{{\pi}}}}{{2}}", r"\frac{{\partial u}}{{\partial t}} = \alpha \frac{{\partial^{{2}} u}}{{\partial x^{{2}}}}",
             r"\cos^{{2}}\theta + \sin^{{2}}\theta = 1", r"PV = nRT", r"\Delta G = \Delta H - T \Delta S", r"pH = -\log_{{10}} [H^{{+}}]",
             r"R = \frac{{V}}{{I}} = {a}\,\Omega", r"\left| x - {a} \right| < {b}"]


def chart(rng, lang, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname=str(ARABIC_FONT)) if lang == "ar" else FontProperties(family="DejaVu Sans")
    k = 1 if lang == "ar" else 0
    kind = rng.choice(["bar", "bar", "hbar", "line", "line", "pie", "scatter"])
    topic = rng.choice(TOPICS)
    axis = rng.choice(["year", "month", "quarter", "city", "sector"] if kind != "line" else ["year", "month", "quarter"])
    names_en, names_ar, cats_en, cats_ar = AXES[axis]
    n = rng.randint(3, min(8, len(cats_en)))
    start = rng.randrange(len(cats_en) - n + 1)
    cats = (cats_ar if lang == "ar" else cats_en)[start:start + n]
    series = rng.choice(SERIES) if kind in ("bar", "line") and rng.random() < 0.5 else ([topic[0]], [topic[1]])
    snames = series[k]
    scale = rng.choice([1, 10, 100, 1000])
    data = [[round(rng.uniform(1, 9) * scale * (1 if scale > 1 else 1), 0 if scale > 1 else 1) for _ in cats] for _ in snames]
    if kind == "scatter":
        cats = [str(round(rng.uniform(0, 100), 1)) for _ in range(n + 4)]
        data = [[round(rng.uniform(1, 9) * scale, 0 if scale > 1 else 1) for _ in cats]]
        names_en, names_ar = "Measured value", "القيمة المقاسة"
    num = lambda v: f"{v:g}" if isinstance(v, float) else str(v)  # noqa: E731
    xname, yname = (names_ar if lang == "ar" else names_en), topic[k]
    plt.style.use(rng.choice(["default", "ggplot", "bmh", "seaborn-v0_8-whitegrid", "fast"]))
    fig, ax = plt.subplots(figsize=(rng.uniform(4.2, 6.2), rng.uniform(2.8, 4.2)), dpi=150)
    labels = [c for c in cats]
    if kind == "pie":
        total = sum(data[0])
        share = [round(100 * v / total, 1) for v in data[0]]
        ax.pie(data[0], labels=labels, autopct="%1.1f%%", textprops={"fontproperties": font})
        rows = [[c, f"{s:.1f}%"] for c, s in zip(cats, share)]
        head = [xname, yname]
    else:
        idx = list(range(n))
        if kind == "bar":
            w = 0.8 / len(snames)
            for j, values in enumerate(data):
                ax.bar([i + j * w for i in idx], values, w, label=snames[j])
            ax.set_xticks([i + w * (len(snames) - 1) / 2 for i in idx], labels, fontproperties=font)
        elif kind == "hbar":
            ax.barh(idx, data[0])
            ax.set_yticks(idx, labels, fontproperties=font)
        elif kind == "line":
            for j, values in enumerate(data):
                ax.plot(idx, values, marker=rng.choice(["o", "s", "^", ""]), label=snames[j])
            ax.set_xticks(idx, labels, fontproperties=font)
        else:
            ax.scatter([float(c) for c in cats], data[0], s=18)
        value_axis = ax.xaxis if kind == "hbar" else ax.yaxis
        (ax.set_ylabel if kind == "hbar" else ax.set_xlabel)(xname, fontproperties=font)
        (ax.set_xlabel if kind == "hbar" else ax.set_ylabel)(yname, fontproperties=font)
        if len(snames) > 1:
            ax.legend(prop=font)
        rows = [[c] + [num(d[i]) for d in data] for i, c in enumerate(cats)]
        head = [xname] + list(snames) if len(snames) > 1 else [xname, yname]
    ax.set_title(f"{topic[k]}", fontproperties=font)
    fig.tight_layout()
    fig.savefig(path)
    if kind != "pie":
        fig.canvas.draw()
        lo, hi = (ax.get_xlim() if kind == "hbar" else ax.get_ylim())
        ticks = [t for t in (value_axis.get_majorticklocs()) if lo - 1e-9 <= t <= hi + 1e-9]
        axes_text = (f"x: {xname}" + (f", {cats[0]} to {cats[-1]}" if axis == "year" and kind != "scatter" else "")
                     + f"; y: {yname}, {ticks[0]:g} to {ticks[-1]:g}")
        if kind == "scatter":
            xlo, xhi = ax.get_xlim()
            xt = [t for t in ax.xaxis.get_majorticklocs() if xlo - 1e-9 <= t <= xhi + 1e-9]
            axes_text = f"x: {xname}, {xt[0]:g} to {xt[-1]:g}; y: {yname}, {ticks[0]:g} to {ticks[-1]:g}"
        if kind == "hbar":
            axes_text = f"x: {yname}, {ticks[0]:g} to {ticks[-1]:g}; y: {xname}"
    plt.close(fig)
    title_en = f"{topic[0]} by {names_en.lower()}" if kind != "scatter" else f"{topic[0]} against a measured value"
    what = {"bar": "Bar chart", "hbar": "Bar chart", "line": "Line chart", "pie": "Pie chart", "scatter": "Scatter plot"}[kind]
    html = f'<img alt="{what}: {html_escape(title_en)}."/>'
    if len(snames) > 1:
        html += '<p class="img-text">' + "<br/>".join(text_escape(s) for s in snames) + "</p>"
    if kind != "pie":
        html += f'<p class="chart-axes">{text_escape(axes_text)}</p>'
    html += ('<table class="chart-data"><tr>' + "".join(f"<th>{text_escape(h)}</th>" for h in head) + "</tr>"
             + "".join("<tr>" + "".join(f"<td>{text_escape(v)}</td>" for v in r) + "</tr>" for r in rows) + "</table>")
    shown = {"bar": "a bar chart", "hbar": "a horizontal bar chart", "line": "a line chart", "pie": "a pie chart",
             "scatter": "a scatter plot"}[kind]
    desc = f"{shown[0].upper()}{shown[1:]} showing {topic[0].lower()} by {names_en.lower()}" if kind != "scatter" else \
        f"A scatter plot of {topic[0].lower()} against a measured value"
    if len(snames) > 1:
        desc += " for " + " and ".join(series[0])
    html += f'<p class="img-desc">{text_escape(desc)}.</p>'
    return {"label": "Figure", "lang": lang, "html": html}


def diagram(rng, lang, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import FancyBboxPatch, Polygon
    font = FontProperties(fname=str(ARABIC_FONT), size=10) if lang == "ar" else FontProperties(family="DejaVu Sans", size=10)
    k = 1 if lang == "ar" else 0
    steps = rng.sample(STEPS, rng.randint(2, 5))
    nodes = [("start", FIXED["start"])] + [("step", s) for s in steps] + [("end", FIXED["end"])]
    ask = rng.random() < 0.5
    if ask:  # a question after the first step: yes goes on, no goes to a side step and back
        q = rng.choice(QUESTIONS)
        side = rng.choice([s for s in STEPS if s not in steps])
        nodes.insert(2, ("ask", q))
    ids = [chr(65 + i) for i in range(len(nodes))]
    h = 0.9 * len(nodes) + 0.4
    fig, ax = plt.subplots(figsize=(rng.uniform(3.4, 4.6) + (1.6 if ask else 0), h), dpi=150)
    ax.set_xlim(-2.2, 2.2 + (2.6 if ask else 0))
    ax.set_ylim(-h + 0.3, 0.5)
    ax.axis("off")
    face = rng.choice(["#e8f0fe", "#fef3e2", "#e9f7ef", "#f3e8fd", "#ffffff"])
    for i, (kind, words) in enumerate(nodes):
        y = -0.9 * i
        text = words[k]
        if kind == "ask":
            ax.add_patch(Polygon([(0, y + 0.38), (1.5, y), (0, y - 0.38), (-1.5, y)], closed=True, fc=face, ec="#333"))
        else:
            ax.add_patch(FancyBboxPatch((-1.4, y - 0.27), 2.8, 0.54, boxstyle="round,pad=0.02,rounding_size=0.25"
                                        if kind in ("start", "end") else "square,pad=0.02", fc=face, ec="#333"))
        ax.text(0, y, text, ha="center", va="center", fontproperties=font)
        if i:
            ax.annotate("", (0, y + 0.3), (0, y + 0.9 - 0.3), arrowprops={"arrowstyle": "->", "color": "#333"})
    if ask:
        y = -0.9 * 2
        ax.add_patch(FancyBboxPatch((1.9, y - 0.27), 2.6, 0.54, boxstyle="square,pad=0.02", fc=face, ec="#333"))
        ax.text(3.2, y, side[k], ha="center", va="center", fontproperties=font)
        ax.annotate("", (1.9, y), (1.5, y), arrowprops={"arrowstyle": "->", "color": "#333"})
        ax.annotate("", (1.4, -0.9), (3.2, y + 0.3), arrowprops={"arrowstyle": "->", "color": "#333",
                                                                   "connectionstyle": "angle,angleA=90,angleB=0"})
        ax.text(1.65, y + 0.12, FIXED["no"][k], fontproperties=font, fontsize=8)
        ax.text(0.08, y - 0.62, FIXED["yes"][k], fontproperties=font, fontsize=8)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    quote = lambda w: '"' + w.replace('"', "'") + '"'  # noqa: E731
    lines = ["graph TD"]
    for i, (kind, words) in enumerate(nodes):
        shape = ("{%s}" if kind == "ask" else "([%s])" if kind in ("start", "end") else "[%s]") % quote(words[k])
        if i == 0:
            lines.append(f"    {ids[0]}{shape}")
        else:
            edge = f"-->|{FIXED['yes'][k]}|" if nodes[i - 1][0] == "ask" else "-->"
            lines.append(f"    {ids[i - 1]} {edge} {ids[i]}{shape}")
    words_drawn = [w[k] for _, w in nodes]
    if ask:
        lines += [f"    C -->|{FIXED['no'][k]}| S[{quote(side[k])}]", "    S --> B"]
        words_drawn += [side[k], FIXED["no"][k], FIXED["yes"][k]]
    title = " then ".join(s[0].lower() for s in steps[:2])
    html = (f'<img alt="Flowchart: steps to {html_escape(title)}."/>'
            f'<p class="img-text">{"<br/>".join(text_escape(w) for w in words_drawn)}</p>'
            f'<pre class="mermaid">{text_escape(chr(10).join(lines))}</pre>'
            f'<p class="img-desc">A flowchart of {len(nodes)} steps from start to end'
            f'{", with a decision that loops back when the answer is no" if ask else ""}.</p>')
    return {"label": "Diagram", "lang": lang, "html": html}


def molecule(smiles, path):
    """A molecule drawn from RDKit's 2D layout (bonds as lines, double bonds doubled, letters for atoms other than
    carbon with their hydrogens); RDKit's own drawing library isn't used (blocked on some Windows machines)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() > 40 or any(a.GetFormalCharge() for a in mol.GetAtoms()):
        return None  # charges aren't drawn: such a label would say more than the picture shows
    smiles = Chem.MolToSmiles(mol, isomericSmiles=False)  # canonical, without stereo marks (no wedges are drawn)
    mol = Chem.MolFromSmiles(smiles)
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:  # noqa: BLE001
        return None
    rdDepictor.Compute2DCoords(mol)
    pos = mol.GetConformer().GetPositions()[:, :2]
    span = pos.max(0) - pos.min(0) + 1.2
    fig, ax = plt.subplots(figsize=(max(1.6, span[0] * 0.42), max(1.4, span[1] * 0.42)), dpi=150)
    ax.set_aspect("equal")
    ax.axis("off")
    for b in mol.GetBonds():
        (x0, y0), (x1, y1) = pos[b.GetBeginAtomIdx()], pos[b.GetEndAtomIdx()]
        order = int(b.GetBondTypeAsDouble())
        dx, dy = y1 - y0, x0 - x1
        norm = max(1e-6, (dx * dx + dy * dy) ** 0.5)
        offs = [0] if order == 1 else ([-0.07, 0.07] if order == 2 else [-0.11, 0, 0.11])
        for o in offs:
            ax.plot([x0 + dx / norm * o, x1 + dx / norm * o], [y0 + dy / norm * o, y1 + dy / norm * o], color="black", lw=1.2)
    for a in mol.GetAtoms():
        if a.GetSymbol() == "C":
            continue
        hs = a.GetTotalNumHs()
        text = a.GetSymbol() + ("H" if hs else "") + (str(hs) if hs > 1 else "")
        x, y = pos[a.GetIdx()]
        ax.text(x, y, text, ha="center", va="center", fontsize=9, color={"O": "#c00", "N": "#00c"}.get(a.GetSymbol(), "black"),
                bbox={"fc": "white", "ec": "none", "pad": 0.5})
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return {"label": "Chemical-Block", "lang": "any", "html": f'<img alt="Chemical structure of the molecule {html_escape(smiles)}."/>'}


def equation(rng, path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import mathtext
    from matplotlib.font_manager import FontProperties
    t = rng.choice(EQUATIONS)
    n, k = rng.randint(2, 9), rng.randint(2, 6)
    latex = t.format(n=n, k=k, k1=k - 1, a=rng.randint(2, 9), b=rng.randint(1, 9), c=rng.randint(1, 9)) if "{{" in t else t
    if rng.random() < 0.4:
        latex += rf" \quad ({rng.randint(1, 30)})"
    try:
        mathtext.math_to_image(f"${latex}$", str(path), prop=FontProperties(size=12), dpi=200, format="png")
    except Exception:  # noqa: BLE001  (a formula mathtext can't draw)
        return None
    height = Image.open(path).size[1]
    # drawn at 12 pt and 200 dpi: on the page it is sized in em, so it follows the page's own text size
    return {"label": "Equation-Block", "lang": "any", "html": f'<math display="block">{text_escape(latex)}</math>',
            "height_em": round(height / 200 * 72 / 12, 3)}


def html_escape(text):
    """Text for HTML: & < > escaped, and " too (it is also used inside alt="...")."""
    import html
    return html.escape(str(text), quote=True)


def text_escape(text):
    """Text between tags: & < > escaped, quotes kept as Surya writes them (a Mermaid label is ["Start"])."""
    import html
    return html.escape(str(text), quote=False)


def build_graphics(charts=2000, diagrams=1000, molecules=1500, equations=1500, seed=0):
    rng = random.Random(seed)
    records = []
    for name, n, make in (("charts", charts, lambda i, p: chart(rng, ("ar", "en")[i % 2], p)),
                          ("diagrams", diagrams, lambda i, p: diagram(rng, ("ar", "en")[i % 2], p)),
                          ("equations", equations, lambda i, p: equation(rng, p))):
        (GRAPHICS / name).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            path = GRAPHICS / name / f"{name[:-1]}_{i:05d}.png"
            r = make(i, path)
            if r:
                records.append({"file": path.relative_to(IMG_POOL).as_posix(), **r})
            if (i + 1) % 250 == 0:
                print(f"  {name}: {i + 1}/{n}", flush=True)
    smiles = [line.split(",")[0].strip().strip('"') for line in download(ZINC).decode("utf-8").replace('\n",', '",').splitlines()[1:]]
    rng.shuffle(smiles)
    (GRAPHICS / "molecules").mkdir(parents=True, exist_ok=True)
    made = 0
    for s in smiles:
        if made >= molecules:
            break
        path = GRAPHICS / "molecules" / f"molecule_{made:05d}.png"
        r = molecule(s, path)
        if r:
            records.append({"file": path.relative_to(IMG_POOL).as_posix(), **r})
            made += 1
            if made % 250 == 0:
                print(f"  molecules: {made}/{molecules}", flush=True)
    GRAPHICS_INDEX.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    counts = {}
    for r in records:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    print(f"{len(records)} graphics in {GRAPHICS_INDEX}: {counts}")


class ImagePool:
    """The pictures in img_pool/ (captions.jsonl), by category, with their captions."""

    def __init__(self, folder=IMG_POOL):
        folder = Path(folder)
        index, sources = folder / "captions.jsonl", folder / "sources.json"
        if index.exists():
            records = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif sources.exists():  # an older pool without captions
            records = json.loads(sources.read_text(encoding="utf-8"))
        else:
            records = [{"file": p.name, "category": "general"} for p in folder.glob("*.jpg")]
        self.files, self.by_category, self.captions = [], {}, {}
        for r in records:
            path = folder / r["file"]
            if path.exists():
                self.files.append(path)
                self.by_category.setdefault(r.get("category", "general"), []).append(path)
                self.captions[path.name] = r.get("captions") or []
        # charts, flowcharts, molecules and equations (img_pool/graphics/graphics.jsonl): {label: {lang: [(path, record)]}}
        self.graphics = {}
        index = folder / "graphics" / "graphics.jsonl"
        for line in index.read_text(encoding="utf-8").splitlines() if index.exists() else []:
            r = json.loads(line)
            path = folder / r["file"]
            if path.exists():
                self.graphics.setdefault(r["label"], {}).setdefault(r["lang"], []).append((path, r))

    def __bool__(self):
        return bool(self.files)

    def pick(self, rng, hint=None):
        """A random picture; from the hint's category when the hint names one and the pool has it."""
        category = hint_category(hint)
        return rng.choice(self.by_category.get(category) or self.files)

    def graphic(self, rng, label, lang):
        """(path, record) of a random drawn picture of this label in this page's language, or None."""
        found = [g for key in (lang, "any") for g in self.graphics.get(label, {}).get(key, [])]
        return rng.choice(found) if found else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool", choices=["text", "images", "fonts", "graphics"])
    ap.add_argument("--articles", type=int, default=5000, help="text: random articles per language")
    ap.add_argument("--lang", nargs="+", default=["ar", "en"], help="text: languages")
    ap.add_argument("--polaris-shards", type=int, default=30, help="images: Polaris train shards (about 2,200 pictures each)")
    ap.add_argument("--pd12m", type=int, default=20000, help="images: PD12M pictures to download")
    a = ap.parse_args()
    if a.pool == "text":
        for lang in a.lang:
            build_text_pool(lang, a.articles)
    elif a.pool == "images":
        build_img_pool(a.polaris_shards, a.pd12m)
    elif a.pool == "fonts":
        build_font_pool()
    else:
        build_graphics()
