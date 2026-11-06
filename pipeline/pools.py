"""The text and photo pools the generator fills pages from (made once, not in git).

    python -m pipeline.pools text --articles 5000     -> text_pool/ (Wikipedia, Arabic and English)
    python -m pipeline.pools images                   -> img_pool/ (photos, about 500 MB)
"""
import argparse
import io
import json
import random
import re
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
# img_pool/: photos
# ====================================================================================================================
# img_pool/: pictures for the templates' picture slots (<img> tags) and grey picture boxes.
#
# Two sources, each picture saved again with Pillow (only plain JPEGs land in the folder), at most 1000 px on
# its longer side, with its category, size, licence and author in img_pool/sources.json:
#
#   * Lorem Picsum (picsum.photos): all of its ~1,000 Unsplash photos (Unsplash License), category "general";
#   * Wikimedia Commons "Quality images" by subject (animals, landscapes, nature, architecture, people, food,
#     sports, transport, patterns and textures): photos reviewed by Commons for quality, each under its own
#     free licence (mostly CC BY-SA / CC BY / public domain; the author and licence are kept per file).
#
# Photos rather than documents or posters: a picture with text in it would put words on the page that the
# labels don't know about, so Commons files whose titles suggest text (signs, posters, maps, logos, books,
# plaques…) are skipped.
#
# A template's picture slot carries a hint (from the original page: an image link's keyword such as
# "?food", the alt text or class names such as "team-photo"). ImagePool.pick(rng, hint) maps the hint to a
# category and picks from it, so a food page gets food photos and a team section gets people.
#
#     .venv-pipeline\\Scripts\\python -m pipeline.pools images                   (download / complete the pool)

IMG_POOL = ROOT / "img_pool"
SOURCES = IMG_POOL / "sources.json"
MAX_SIDE = 1000
HEADERS = {"User-Agent": "OCRSynthGen/1.0 (synthetic OCR research; https://github.com/Yousef10p/OCRSynthGen)"}
PICSUM_LIST = "https://picsum.photos/v2/list?page={page}&limit=100"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
# Commons category -> (our category, how many)
COMMONS = {"Quality images of animals": ("animals", 300), "Quality images of landscapes": ("landscapes", 300),
           "Quality images of nature": ("nature", 300), "Quality images of architecture": ("architecture", 300),
           "Quality images of people": ("people", 250), "Quality images of food": ("food", 250),
           "Quality images of sports": ("sports", 150), "Quality images of transport": ("transport", 150),
           "Quality images of patterns and textures": ("patterns", 200)}
TEXTY = re.compile(r"\b(sign|signs|signage|poster|label|logo|text|map|chart|diagram|inscription|plaque|menu|book|"
                   r"newspaper|banner|graffiti|board|screen|scoreboard|advert|billboard|ticket|stamp|coin|"
                   r"document|letter|manuscript|page|calligraphy|typography|font|monument|memorial|grave|brand|"
                   r"package|packaging|bottle|can|box|product|shop|store|market|stall|street)\b", re.I)
# Words of a picture's hint -> category (the first that matches)
HINTS = [("food", r"food|restaurant|cafe|coffee|recipe|menu|bakery|dish|meal|kitchen|chef|drink|pizza|fruit|"
                  r"wine|tea\b|cake|grocery|catering"),
         ("people", r"team|person|people|portrait|avatar|profile|user|author|testimonial|doctor|teacher|staff|"
                    r"member|founder|ceo|client|customer|student|speaker|instructor|trainer|face|headshot|man|woman|"
                    r"fashion|clothing|wear|style|beauty|salon|spa|model|wedding|family|kid|child|patient|"
                    r"attorney|lawyer|employee|volunteer|artist|musician"),
         ("animals", r"animal|pet|dog|cat|zoo|wildlife|bird|horse|vet"),
         ("sports", r"sport|fitness|gym|football|soccer|basketball|tennis|run|yoga|athlet|game|match|stadium"),
         ("transport", r"car|transport|vehicle|auto|bike|train|plane|flight|airline|bus|truck|ship|logistic|"
                       r"shipping|delivery|taxi|motor"),
         ("architecture", r"city|building|architecture|office|hotel|real.?estate|house|home|interior|property|"
                          r"apartment|construction|urban|church|museum|bank|company|business|corporate|"
                          r"technolog|tech|software|startup|finance|consult|agency|service|law|legal|school|"
                          r"universit|hospital|clinic|event|conference"),
         ("landscapes", r"travel|landscape|mountain|beach|hero|banner|background|tour|destination|vacation|"
                        r"holiday|resort|sea|ocean|lake|desert|scenery"),
         ("nature", r"nature|garden|forest|park|flower|plant|tree|green|eco|environment|farm|agricultur"),
         ("patterns", r"abstract|pattern|texture|art|design|gallery|portfolio|creative|wallpaper|decor")]


def download(url, timeout=60, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def save(data, name):
    """Open, shrink to MAX_SIDE and save again as JPEG; returns (width, height)."""
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    image.save(IMG_POOL / name, quality=85)
    return image.size


# -- Picsum -------------------------------------------------------------------------------------------------

def picsum_records():
    photos, page = [], 1
    while True:
        batch = json.loads(download(PICSUM_LIST.format(page=page)))
        if not batch:
            return photos
        photos += batch
        page += 1


def fetch_picsum(photo):
    name = f"picsum_{photo['id']}.jpg"
    w, h = int(photo["width"]), int(photo["height"])
    scale = MAX_SIDE / max(w, h)
    try:
        if not (IMG_POOL / name).exists():
            save(download(f"https://picsum.photos/id/{photo['id']}/{round(w * scale)}/{round(h * scale)}.jpg"), name)
        size = Image.open(IMG_POOL / name).size
    except Exception as e:
        print(f"  picsum {photo['id']} skipped: {type(e).__name__}", flush=True)
        return None
    return {"file": name, "source": "picsum.photos", "category": "general", "width": size[0], "height": size[1],
            "author": photo["author"], "url": photo["url"], "license": "Unsplash License"}


# -- Wikimedia Commons --------------------------------------------------------------------------------------

def commons(params):
    url = COMMONS_API + "?" + urllib.parse.urlencode({**params, "format": "json", "maxlag": 5})
    time.sleep(0.5)  # one API request at a time, gently
    return json.loads(download(url))


def commons_files(category, quota):
    """File records (title, thumb url, licence, author) from a category and its subcategories (2 levels)."""
    found, queue, seen = [], [(category, 0)], set()
    while queue and len(found) < quota:
        cat, depth = queue.pop(0)
        if cat in seen:
            continue
        seen.add(cat)
        cont = {}
        while len(found) < quota:
            d = commons({"action": "query", "generator": "categorymembers", "gcmtitle": f"Category:{cat}",
                         "gcmtype": "file", "gcmlimit": 50, "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata",
                         "iiurlwidth": MAX_SIDE, **cont})
            for page in d.get("query", {}).get("pages", {}).values():
                ii = (page.get("imageinfo") or [{}])[0]
                title = page["title"]
                if ii.get("mime") not in ("image/jpeg", "image/png") or TEXTY.search(title):
                    continue
                w, h = ii.get("width", 0), ii.get("height", 0)
                if min(w, h) < 500 or not 0.4 <= w / max(h, 1) <= 2.5:
                    continue
                md = ii.get("extmetadata", {})
                author = re.sub(r"<[^>]+>", "", md.get("Artist", {}).get("value", "")).strip()
                found.append({"title": title, "thumb": ii.get("thumburl") or ii["url"], "page": ii.get("descriptionurl"),
                              "license": md.get("LicenseShortName", {}).get("value", "see page"), "author": author[:120]})
            if "continue" not in d:
                break
            cont = {k: v for k, v in d["continue"].items()}
        if depth < 2 and len(found) < quota:
            d = commons({"action": "query", "list": "categorymembers", "cmtitle": f"Category:{cat}", "cmtype": "subcat",
                         "cmlimit": 50})
            subs = [c["title"].split(":", 1)[1] for c in d.get("query", {}).get("categorymembers", [])]
            random.Random(cat).shuffle(subs)
            queue += [(s, depth + 1) for s in subs]
    return found[:quota]


def fetch_commons(job):
    record, category, n = job
    name = f"commons_{category}_{n:04d}.jpg"
    try:
        if not (IMG_POOL / name).exists():
            save(download(record["thumb"]), name)
        size = Image.open(IMG_POOL / name).size
    except Exception as e:
        print(f"  commons {record['title'][:50]} skipped: {type(e).__name__}", flush=True)
        return None
    return {"file": name, "source": "commons.wikimedia.org", "category": category, "width": size[0],
            "height": size[1], "author": record["author"], "url": record["page"], "title": record["title"],
            "license": record["license"]}


def build_img_pool():
    IMG_POOL.mkdir(exist_ok=True)
    records = [r for r in ThreadPoolExecutor(8).map(fetch_picsum, picsum_records()) if r]
    print(f"picsum: {len(records)} photos", flush=True)
    for cat, (category, quota) in COMMONS.items():
        files = commons_files(cat, quota)
        got = [r for r in ThreadPoolExecutor(4).map(fetch_commons, [(f, category, i) for i, f in enumerate(files)]) if r]
        records += got
        print(f"commons {category}: {len(got)} photos", flush=True)
        SOURCES.write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")
    keep = {r["file"] for r in records}
    for stale in IMG_POOL.glob("*.jpg"):  # files of an older pool that are no longer listed
        if stale.name not in keep:
            stale.unlink()
    SOURCES.write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")
    counts = {}
    for r in records:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    print(f"{len(records)} pictures in {IMG_POOL}: {counts}")


def hint_category(hint):
    """The category a picture hint asks for, or None (any picture will do). Words match from their start
    ("team" is people, not "tea")."""
    hint = (hint or "").lower()
    return next((category for category, words in HINTS if re.search(rf"\b(?:{words})", hint)), None)


class ImagePool:
    """The pictures in img_pool/, by category."""

    def __init__(self, folder=IMG_POOL):
        folder = Path(folder)
        sources = folder / "sources.json"
        records = json.loads(sources.read_text(encoding="utf-8")) if sources.exists() else \
            [{"file": p.name, "category": "general"} for p in folder.glob("*.jpg")]
        self.files, self.by_category = [], {}
        for r in records:
            if (folder / r["file"]).exists():
                path = folder / r["file"]
                self.files.append(path)
                self.by_category.setdefault(r["category"], []).append(path)

    def __bool__(self):
        return bool(self.files)

    def pick(self, rng, hint=None):
        """A random picture; from the hint's category when the hint names one and the pool has it."""
        category = hint_category(hint)
        return rng.choice(self.by_category.get(category) or self.files)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool", choices=["text", "images"])
    ap.add_argument("--articles", type=int, default=5000, help="text: random articles per language")
    ap.add_argument("--lang", nargs="+", default=["ar", "en"], help="text: languages")
    a = ap.parse_args()
    if a.pool == "text":
        for lang in a.lang:
            build_text_pool(lang, a.articles)
    else:
        build_img_pool()
