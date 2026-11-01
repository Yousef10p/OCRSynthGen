"""img_pool/: pictures for the grey picture boxes (class "ph") of the templates.

Photos from Lorem Picsum (picsum.photos): Unsplash photos, free to use under the Unsplash licence; each one's
author and Unsplash page are kept in img_pool/sources.json. Photos rather than document figures, because a
picture with text in it would put words on the page that the labels don't know about. (The bank's DocLayNet
picture crops are left out for that reason: their PDFs had no text layer, but some show logos, screenshots or
labelled diagrams.)

Every download is opened and saved again with Pillow, so only plain JPEGs land in the folder.

    .venv-pipeline\\Scripts\\python -m pipeline.img_pool --n 500
"""
import argparse
import io
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "img_pool"
LIST_URL = "https://picsum.photos/v2/list?page={page}&limit=100"
WIDTH = 800  # pixels; enough for a picture box on a 96 dpi page
HEADERS = {"User-Agent": "arabic-template-bank/1.0"}


def get(url, timeout=30):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout) as r:
        return r.read()


def picsum_list(n):
    """Up to n photo records (id, author, size, Unsplash page) from Picsum's list."""
    photos, page = [], 1
    while len(photos) < n:
        batch = json.loads(get(LIST_URL.format(page=page)))
        if not batch:
            break
        photos += batch
        page += 1
    return photos[:n]


def fetch(photo):
    """Download one photo at WIDTH px wide, keeping its shape; returns its sources.json record or None."""
    w = WIDTH
    h = max(1, round(WIDTH * int(photo["height"]) / int(photo["width"])))
    name = f"picsum_{photo['id']}.jpg"
    try:
        image = Image.open(io.BytesIO(get(f"https://picsum.photos/id/{photo['id']}/{w}/{h}.jpg")))
        image.convert("RGB").save(POOL / name, quality=85)
    except Exception as e:
        print(f"  skipped {photo['id']}: {type(e).__name__}")
        return None
    return {"file": name, "source": "picsum.photos", "author": photo["author"], "url": photo["url"],
            "license": "Unsplash License"}


def build(n):
    POOL.mkdir(exist_ok=True)
    records = [r for r in ThreadPoolExecutor(8).map(fetch, picsum_list(n)) if r]
    (POOL / "sources.json").write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(records)} pictures in {POOL}")


class ImagePool:
    """The pictures in img_pool/, picked at random."""

    def __init__(self, folder=POOL):
        self.files = sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in (".jpg", ".png"))

    def __bool__(self):
        return bool(self.files)

    def pick(self, rng):
        return rng.choice(self.files)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="photos to download from Picsum")
    build(ap.parse_args().n)
