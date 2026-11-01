"""Attack SynthDoc's clean renders with OCRSmith's paper and capture effects, keeping every label aligned.

The same idea as attack.py, with OCRSmith (github.com/atlasia-ma/OCRSmith) in place of Augraphy and
scan-simulator:

1. Paper: the page is laid on one of OCRSmith's papers (tinted paper with grain, plain, gradient or a
   photo texture), multiplied in so the ink stays dark and only the white turns to paper. Nothing moves.
2. Capture: one of OCRSmith's presets (scan, photo, fax, archive). OCRSmith's effects take the page's
   annotation with the image and move every box through the same rotation, perspective, curl or wrinkle,
   so SynthDoc's word boxes are handed to it as an OCRSmith Page (one region per line, each word named by
   its index) and read back afterwards. Lines and slots become the union of their words.

The legibility guard is attack.py's: a page whose text ends up too faint is attacked again with a new seed,
then the most readable try is kept and marked low_contrast.

Writes synthdoc_ocrsmith/: images/, labels/, overlays/, summary.json (same label format as synthdoc_attacked/).
Runs with OCRSmith's venv (no other install needed):

    ..\\ocrsmith\\.venv\\Scripts\\python synthdoc_adapter\\attack_ocrsmith.py --workers 6
"""
import argparse
import json
import random
import shutil
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "demo"))
from labels import MIN_CONTRAST, build_label, contrast, load_page, summary  # noqa: E402
from render_demo import background  # noqa: E402  (OCRSmith papers, as in our own renders)

from ocrsmith.core.degradations import build_preset  # noqa: E402
from ocrsmith.domain.annotations import Line, Page, Region, RegionType, Word  # noqa: E402
from ocrsmith.domain.geometry import BBox  # noqa: E402
from ocrsmith.text.script import Direction  # noqa: E402

SRC, OUT = ROOT / "synthdoc_out", ROOT / "synthdoc_ocrsmith"
# OCRSmith's capture presets, weighted as in demo/render_demo.py ("clean" left out: this is the attacked set)
PRESETS = {"scan": 4.0, "photo": 3.0, "fax": 0.7, "archive": 1.5}


def to_page(words, width, height):
    """SynthDoc words as an OCRSmith Page: one region per line, each word's text set to its index."""
    by_line = {}
    for i, w in enumerate(words):
        by_line.setdefault(w["line_id"], []).append(
            Word(str(i), BBox.from_xywh(w["x"], w["y"], w["width"], w["height"])))
    regions = []
    for ws in by_line.values():
        box = BBox(min(w.bbox.x0 for w in ws), min(w.bbox.y0 for w in ws),
                   max(w.bbox.x1 for w in ws), max(w.bbox.y1 for w in ws))
        regions.append(Region(RegionType.PARAGRAPH, box, (Line("", box, tuple(ws), Direction.RTL),)))
    return Page(width, height, tuple(regions), Direction.RTL)


def from_page(page, n):
    """Word boxes [x, y, w, h] by index; None for a word pushed off the page."""
    boxes = [None] * n
    for region in page.regions:
        for line in region.lines:
            for w in line.words:
                b = w.bbox
                if b.x1 - b.x0 >= 1 and b.y1 - b.y0 >= 1:
                    x, y = int(b.x0), int(b.y0)
                    boxes[int(w.text)] = [x, y, int(np.ceil(b.x1)) - x, int(np.ceil(b.y1)) - y]
    return boxes


def degrade(image, words, seed):
    """OCRSmith paper, then one capture preset. Returns (image, word boxes, attacks)."""
    rng = random.Random(seed)
    paper, kind = background(rng, *image.size)
    # Multiply: white page -> paper, ink stays ink (the page's own tints are kept too)
    out = Image.fromarray((np.asarray(image, np.float32) * np.asarray(paper.convert("RGB"), np.float32) / 255)
                          .round().astype(np.uint8))
    preset = rng.choices(list(PRESETS), weights=list(PRESETS.values()))[0]
    out, page, records = build_preset(preset).apply(out, to_page(words, *image.size), rng)
    return (out.convert("RGB"), from_page(page, len(words)),
            [f"ocrsmith-paper:{kind}", f"ocrsmith:{preset}"] + [r.name for r in records])


def draw(image, boxes, color, path):
    out = image.copy()
    d = ImageDraw.Draw(out)
    for x, y, w, h in boxes:
        d.rectangle([x, y, x + w, y + h], outline=color, width=1)
    out.save(path, quality=85)


def attack(page):
    seed = zlib.crc32(page.encode()) % 2**31
    image = Image.open(SRC / "images" / f"{page}.jpg").convert("RGB")
    ocr, slots, words, cut = load_page(SRC, page)

    best = None
    for s in (seed + k * 1000 for k in range(5)):
        out, boxes, attacks = degrade(image, words, s)
        c = contrast(np.asarray(out.convert("L")), [b for b in boxes if b])
        if best is None or c > best[0]:
            best = (c, out, boxes, attacks, s)
        if c >= MIN_CONTRAST:
            break
    page_contrast, out, word_boxes, attacks, used_seed = best
    label = build_label(page, out.size, used_seed, attacks, page_contrast, ocr, slots, words, word_boxes, cut)
    out.save(OUT / "images" / f"{page}.jpg", quality=88)
    (OUT / "labels" / f"{page}.json").write_text(json.dumps(label, ensure_ascii=False), encoding="utf-8")
    draw(out, [w["box"] for w in label["words"]], (60, 160, 60), OUT / "overlays" / f"{page}_words.jpg")
    draw(out, [s["box"] for s in label["slots"]], (200, 40, 40), OUT / "overlays" / f"{page}_slots.jpg")
    return summary(label, len(words))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="attack only the first N pages (0 = all)")
    a = ap.parse_args()
    shutil.rmtree(OUT, ignore_errors=True)
    for d in ("images", "labels", "overlays"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    pages = [p for doc in json.loads((SRC / "manifest.json").read_text(encoding="utf-8"))["documents"] for p in doc["pages"]]
    pages = pages[: a.limit] if a.limit else pages
    with Pool(a.workers) as pool:
        results = pool.map(attack, pages)
    (OUT / "summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    lost, total = sum(r["lost_words"] for r in results), sum(r["words"] for r in results)
    print(f"{sum(r['low_contrast'] for r in results)} page(s) still below contrast {MIN_CONTRAST} (marked low_contrast); "
          f"{sum(r['no_text'] for r in results)} with no text")
    print(f"attacked {len(results)} pages; {lost} of {total} words lost (pushed off the page)")


if __name__ == "__main__":
    main()
