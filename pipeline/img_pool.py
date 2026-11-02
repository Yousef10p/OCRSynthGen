"""img_pool/: pictures for the templates' picture slots (<img> tags) and grey picture boxes.

Two sources, each picture saved again with Pillow (only plain JPEGs land in the folder), at most 1000 px on
its longer side, with its category, size, licence and author in img_pool/sources.json:

  * Lorem Picsum (picsum.photos): all of its ~1,000 Unsplash photos (Unsplash License), category "general";
  * Wikimedia Commons "Quality images" by subject (animals, landscapes, nature, architecture, people, food,
    sports, transport, patterns and textures): photos reviewed by Commons for quality, each under its own
    free licence (mostly CC BY-SA / CC BY / public domain; the author and licence are kept per file).

Photos rather than documents or posters: a picture with text in it would put words on the page that the
labels don't know about, so Commons files whose titles suggest text (signs, posters, maps, logos, books,
plaques…) are skipped.

A template's picture slot carries a hint (from the original page: an image link's keyword such as
"?food", the alt text or class names such as "team-photo"). ImagePool.pick(rng, hint) maps the hint to a
category and picks from it, so a food page gets food photos and a team section gets people.

    .venv-pipeline\\Scripts\\python -m pipeline.img_pool                 (download / complete the pool)
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
POOL = ROOT / "img_pool"
SOURCES = POOL / "sources.json"
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


def get(url, timeout=60, tries=3):
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
    image.save(POOL / name, quality=85)
    return image.size


# -- Picsum -------------------------------------------------------------------------------------------------

def picsum_records():
    photos, page = [], 1
    while True:
        batch = json.loads(get(PICSUM_LIST.format(page=page)))
        if not batch:
            return photos
        photos += batch
        page += 1


def fetch_picsum(photo):
    name = f"picsum_{photo['id']}.jpg"
    w, h = int(photo["width"]), int(photo["height"])
    scale = MAX_SIDE / max(w, h)
    try:
        if not (POOL / name).exists():
            save(get(f"https://picsum.photos/id/{photo['id']}/{round(w * scale)}/{round(h * scale)}.jpg"), name)
        size = Image.open(POOL / name).size
    except Exception as e:
        print(f"  picsum {photo['id']} skipped: {type(e).__name__}", flush=True)
        return None
    return {"file": name, "source": "picsum.photos", "category": "general", "width": size[0], "height": size[1],
            "author": photo["author"], "url": photo["url"], "license": "Unsplash License"}


# -- Wikimedia Commons --------------------------------------------------------------------------------------

def commons(params):
    url = COMMONS_API + "?" + urllib.parse.urlencode({**params, "format": "json", "maxlag": 5})
    time.sleep(0.5)  # one API request at a time, gently
    return json.loads(get(url))


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
        if not (POOL / name).exists():
            save(get(record["thumb"]), name)
        size = Image.open(POOL / name).size
    except Exception as e:
        print(f"  commons {record['title'][:50]} skipped: {type(e).__name__}", flush=True)
        return None
    return {"file": name, "source": "commons.wikimedia.org", "category": category, "width": size[0],
            "height": size[1], "author": record["author"], "url": record["page"], "title": record["title"],
            "license": record["license"]}


def build():
    POOL.mkdir(exist_ok=True)
    records = [r for r in ThreadPoolExecutor(8).map(fetch_picsum, picsum_records()) if r]
    print(f"picsum: {len(records)} photos", flush=True)
    for cat, (category, quota) in COMMONS.items():
        files = commons_files(cat, quota)
        got = [r for r in ThreadPoolExecutor(4).map(fetch_commons, [(f, category, i) for i, f in enumerate(files)]) if r]
        records += got
        print(f"commons {category}: {len(got)} photos", flush=True)
        SOURCES.write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")
    keep = {r["file"] for r in records}
    for stale in POOL.glob("*.jpg"):  # files of an older pool that are no longer listed
        if stale.name not in keep:
            stale.unlink()
    SOURCES.write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")
    counts = {}
    for r in records:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    print(f"{len(records)} pictures in {POOL}: {counts}")


def hint_category(hint):
    """The category a picture hint asks for, or None (any picture will do). Words match from their start
    ("team" is people, not "tea")."""
    hint = (hint or "").lower()
    return next((category for category, words in HINTS if re.search(rf"\b(?:{words})", hint)), None)


class ImagePool:
    """The pictures in img_pool/, by category."""

    def __init__(self, folder=POOL):
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
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()
