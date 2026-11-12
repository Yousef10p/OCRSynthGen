"""Do the labels' boxes fit what the image draws? Compares every block's box with the ink on the page image.

    python surya_probe/check_boxes.py <dataset batch folder> [--examples 3] [--effect clean]

For every sample of a batch (labels/<name>.json, the page image in images/ or pages/<name>_page.jpg for a crop; or
a folder of DocPipeline.sample() files, <name>.json with <name>.jpg / <name>_page.jpg):
  page      every box lies on the page (0 <= x0 < x1 <= width, same for y)
  overlap   no two blocks share more than a fifth of the smaller one (a box holding another block's content)
and for every block but pictures (a photo or drawing fills its box), with ink = pixels far from the block's
background (its median grey):
  empty     the box holds ink (it covers something drawn)
  cut       (text blocks: Text, Section-Header, List-Group, Caption, Footnote, Page-Header/Footer, Table-Of-Contents,
            Bibliography) no glyph crosses the box's edge: strokes of ink on both sides of an edge along more than
            2% of it (at least 3 px) mean letters are cut off (box too small or shifted); a rule or coloured band
            along the edge (ink over most of it, or runs longer than 5% of it) is no letter
  loose     (text blocks) no empty margin wider than 2.5% of the page (width for left/right, height for top/bottom) between the
            box and its ink (box too large or shifted)
Prints, per check, the samples that fail and examples; exit code 1 when anything fails."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

TEXT = {"Text", "Section-Header", "List-Group", "Caption", "Footnote", "Page-Header", "Page-Footer",
        "Table-Of-Contents", "Bibliography"}
PICTURES = {"Image", "Figure", "Diagram", "Chemical-Block"}


def page_image(folder, name, label):
    """The page as a grey array in the label's pixels (the label keeps the 2x render's pixels)."""
    found = [q for q in (folder / "pages" / f"{name}_page.jpg", folder / f"{name}_page.jpg",
                         folder / "images" / f"{name}.jpg", folder / f"{name}.jpg") if q.exists()]
    if not found or (label["surya"].get("crop") and not found[0].stem.endswith("_page")):
        return None  # a crop without its whole page
    with Image.open(found[0]) as im:
        return gray(im, label)


def gray(im, label):
    """A page image (PIL) as the grey array check() reads: in the label's pixels, noise smoothed on effect pages."""
    im = im.convert("L")
    if im.size != (label["width"], label["height"]):
        im = im.resize((label["width"], label["height"]))
    if label.get("effect") != "clean":
        im = im.filter(ImageFilter.MedianFilter(3))  # scanner and paper noise
    return np.asarray(im, dtype=np.int16)


def ink_of(g, box, pad):
    """(ink mask of the box grown by pad, the box's place in it): ink = far from the box's median grey."""
    h, w = g.shape
    x0, y0, x1, y1 = (round(v) for v in box)
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    region = g[Y0:Y1, X0:X1]
    inside = g[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if region.size == 0 or inside.size == 0:
        return None, None
    bg = np.median(inside)
    return np.abs(region - bg) > 70, (x0 - X0, y0 - Y0, x1 - X0, y1 - Y0)


def check(label, g, problems, stem):
    H, W = label["height"], label["width"]
    blocks = label["surya"]["blocks"]
    for b in blocks:
        x0, y0, x1, y1 = b["box"]
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H or x1 <= x0 or y1 <= y0:
            problems["page"].append(f"{stem}: {b['label']} box {b['box']} not on the {W}x{H} page")
    for i, a in enumerate(blocks):
        for c in blocks[i + 1:]:
            ix = min(a["box"][2], c["box"][2]) - max(a["box"][0], c["box"][0])
            iy = min(a["box"][3], c["box"][3]) - max(a["box"][1], c["box"][1])
            area = lambda b: max(1, (b[2] - b[0]) * (b[3] - b[1]))  # noqa: E731
            if ix > 0 and iy > 0 and ix * iy > 0.2 * min(area(a["box"]), area(c["box"])):
                problems["overlap"].append(f"{stem}: {a['label']} {a['html'][:30]!r} and {c['label']} {c['html'][:30]!r} overlap")
    if g is None:
        return
    for b in blocks:
        if b["label"] in PICTURES:  # a photo or drawing fills its box with its own picture
            continue
        ink, at = ink_of(g, b["box"], 6)
        if ink is None:
            continue
        x0, y0, x1, y1 = at
        x0, y0 = max(0, x0), max(0, y0)
        box_ink = ink[y0:y1, x0:x1]
        rows, cols = np.where(box_ink.sum(1) >= 2)[0], np.where(box_ink.sum(0) >= 2)[0]
        what = f"{b['label']} {b['html'][:40]!r}"
        if not len(rows) or not len(cols):
            problems["empty"].append(f"{stem}: {what} box {b['box']} holds no ink")
            continue
        # cut: ink right inside and right outside an edge at the same place
        edges = {"top": (ink[y0, x0:x1], ink[max(0, y0 - 3):y0, x0:x1].any(0) if y0 >= 1 else None),
                 "bottom": (ink[y1 - 1, x0:x1], ink[y1:y1 + 3, x0:x1].any(0) if y1 < ink.shape[0] else None),
                 "left": (ink[y0:y1, x0], ink[y0:y1, max(0, x0 - 3):x0].any(1) if x0 >= 1 else None),
                 "right": (ink[y0:y1, x1 - 1], ink[y0:y1, x1:x1 + 3].any(1) if x1 < ink.shape[1] else None)}
        for side, (inner, outer) in edges.items():
            if outer is None or not len(inner) or b["label"] not in TEXT or outer.mean() > 0.6:
                continue  # (a rule or a coloured band along the edge is no letter)
            both = np.concatenate([[0], (inner & outer).astype(int), [0]])
            starts, ends = np.flatnonzero(np.diff(both) == 1), np.flatnonzero(np.diff(both) == -1)
            cross = int(sum(e - s for s, e in zip(starts, ends) if e - s < 0.05 * len(inner) + 3))  # strokes, not lines
            if cross >= 3 and cross > 0.02 * len(inner):
                problems["cut"].append(f"{stem}: {what} cut at the {side} ({cross} of {len(inner)} px)")
        # loose: empty margin between the box and its ink
        gaps = {"top": rows[0] / H, "bottom": (y1 - y0 - 1 - rows[-1]) / H,
                "left": cols[0] / W, "right": (x1 - x0 - 1 - cols[-1]) / W}
        for side, gap in gaps.items():
            if gap > 0.025 and b["label"] in TEXT:  # (a table's or panel's own fill and faint rules are drawn too)
                problems["loose"].append(f"{stem}: {what} has {gap:.1%} of the page empty at the {side}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--effect", help="only samples with this effect (clean, ocrsmith, augraphy, scan-simulator)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    folder, problems, n = Path(a.folder), defaultdict(list), 0
    for jf in sorted((folder / "labels" if (folder / "labels").is_dir() else folder).glob("*.json")):
        label = json.loads(jf.read_text(encoding="utf-8"))
        if "surya" not in label or (a.effect and label.get("effect") != a.effect):
            continue
        n += 1
        check(label, page_image(folder, jf.stem, label), problems, jf.stem)
    print(f"{n} samples checked")
    for name in ("page", "overlap", "empty", "cut", "loose"):
        found = problems.get(name, [])
        stems = {x.split(":")[0] for x in found}
        print(f"  {name:8} {'ok' if not found else f'{len(found)} problems in {len(stems)} samples'}")
        for x in found[:a.examples]:
            print(f"      {x}")
    sys.exit(1 if any(problems.values()) else 0)


if __name__ == "__main__":
    main()
