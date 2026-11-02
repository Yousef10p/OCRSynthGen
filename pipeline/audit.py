"""Audit finished samples, independently of the checks that ran while they were made.

For every <name>.jpg in a folder (the generator's run folders, pipeline_out/qa, or a dataset's images/):
  * the ground truth parses (one <text>... <bbox>x1, y1, x2, y2</bbox></text> per line) and says exactly
    what the label's slots say (same texts, same boxes, same order);
  * every box lies on the image and has a size; every word of the label lies inside a region's box;
  * the text of every region is the text the filled HTML page has in that slot (only whole words at the end
    may be missing: those the generator trimmed so the slot fits), so no text was invented or changed;
  * every region has ink in the clean render (the text is drawn there) and still stands out after the effect;
  * no words of two different regions overlap (more than 20% of the smaller word);
then draws the ground-truth boxes on each page (<out>/<name>_gt.jpg) and a contact sheet (<out>/sheet_*.jpg).

    .venv-pipeline\\Scripts\\python -m pipeline.audit pipeline_out\\qa
"""
import argparse
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from lxml import html as lhtml
from PIL import Image, ImageDraw

LINE = re.compile(r"<text>(.*) <bbox>(-?\d+), (-?\d+), (-?\d+), (-?\d+)</bbox></text>")
OVERLAP = 0.20


def parse_gt(text):
    out, bad = [], 0
    for line in text.splitlines():
        m = LINE.fullmatch(line)
        if not m:
            bad += 1
            continue
        out.append((html.unescape(m.group(1)), [int(v) for v in m.groups()[1:]]))
    return out, bad


def squash(s):
    return re.sub(r"\s+", " ", s).strip()


def ink(gray, box):
    """How much of a box is drawn on: the share of pixels far from the box's own background (its median)."""
    x1, y1, x2, y2 = box
    crop = gray[max(0, y1):y2, max(0, x1):x2].astype(np.float32)
    if crop.size == 0:
        return 0.0, 0.0
    bg = np.median(crop)
    far = np.abs(crop - bg) > 40
    return float(far.mean()), float(np.abs(crop - bg).max())


def audit_one(jpg):
    stem = jpg.with_suffix("")
    label = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    gt, bad = parse_gt(Path(f"{stem}.md").read_text(encoding="utf-8"))
    problems = Counter()
    if bad:
        problems["gt lines that don't parse"] += bad
    slots = label["slots"]
    if len(gt) != len(slots):
        problems["gt regions != label slots"] += 1
    for (text, box), s in zip(gt, slots):
        if text != s["text"] or box != [int(round(v)) for v in s["bbox"]]:
            problems["gt differs from label"] += 1
    w, h = label["width"], label["height"]
    for _, (x1, y1, x2, y2) in gt:
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            problems["box off the image"] += 1
        if x2 <= x1 or y2 <= y1:
            problems["empty box"] += 1
    # every word of the label belongs to some region's box (the regions are the union of their words)
    for wd in label["words"]:
        bx, by, bw, bh = wd["box"]
        if not any(x1 - 1 <= bx and y1 - 1 <= by and bx + bw <= x2 + 1 and by + bh <= y2 + 1
                   for _, (x1, y1, x2, y2) in gt):
            problems["word outside every region"] += 1
    # the filled page's text
    page = Path(f"{stem}.html")
    if page.exists():
        root = lhtml.fromstring(page.read_text(encoding="utf-8"))
        dom = {el.get("data-slot"): squash(el.text_content()) for el in root.iter() if el.get("data-slot")}
        for s in slots:
            want, got = squash(s["text"]), dom.get(s["id"])
            if got is None:
                problems["slot not in the page"] += 1
            elif not same_words(want, got):
                problems["text differs from the page"] += 1
    # ink in the clean render and after the effect
    clean_path = Path(f"{stem}_clean.jpg")
    final = np.asarray(Image.open(jpg).convert("L"))
    clean = np.asarray(Image.open(clean_path).convert("L")) if clean_path.exists() else None
    if clean is not None and (clean.shape != final.shape or label.get("effect") not in ("clean", "augraphy")):
        # OCRSmith and scan-simulator move the page: the boxes are for the final image only
        clean = None
    for text, box in gt:
        if not text.strip():
            continue
        if clean is not None and ink(clean, box)[0] < 0.01:
            problems["no ink in the clean render"] += 1
        share, peak = ink(final, box)
        if peak < 60 or share < 0.005:
            problems["text faded after the effect"] += 1
    # text of two regions on top of each other: compared word by word, since a turned page makes the upright
    # boxes of neighbouring paragraphs overlap a little while their words don't
    def area(b):
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    owned = []
    for wd in label["words"]:
        bx, by, bw, bh = wd["box"]
        box = [bx, by, bx + bw, by + bh]
        inside = [k for k, (_, (x1, y1, x2, y2)) in enumerate(gt)
                  if x1 - 1 <= box[0] and y1 - 1 <= box[1] and box[2] <= x2 + 1 and box[3] <= y2 + 1]
        if inside:
            owned.append((min(inside, key=lambda k: area(gt[k][1])), box))
    for i in range(len(owned)):
        ri, a = owned[i]
        for j in range(i + 1, len(owned)):
            rj, b = owned[j]
            if ri == rj:
                continue
            iw = min(a[2], b[2]) - max(a[0], b[0])
            ih = min(a[3], b[3]) - max(a[1], b[1])
            if iw > 0 and ih > 0 and iw * ih > OVERLAP * max(1, min(area(a), area(b))):
                problems["text of two regions overlaps"] += 1
    return label, gt, problems


def same_words(want, got):
    """The rendered words are the page's words, in order, as typed (the end may be trimmed to fit)."""
    a, b = want.replace(" ", ""), got.replace(" ", "")
    if a == b or b.startswith(a):
        return True
    # a word the generator shortened at the end (trimmed "..." or a hyphen it added to break a long word)
    a = a.replace("…", "").replace("-", "").replace("­", "")
    return b.replace("-", "").replace("­", "").startswith(a)


def draw(jpg, gt, out):
    im = Image.open(jpg).convert("RGB")
    d = ImageDraw.Draw(im)
    for k, (_, (x1, y1, x2, y2)) in enumerate(gt):
        d.rectangle([x1, y1, x2, y2], outline=(230, 30, 30) if k % 2 == 0 else (30, 90, 230), width=3)
    im.thumbnail((900, 1200))
    im.save(out / f"{jpg.stem}_gt.jpg", quality=85)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", help="where to draw (default: <folder>/audit)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    folder = Path(a.folder)
    images = folder / "images" if (folder / "images").is_dir() else folder
    out = Path(a.out) if a.out else folder / "audit"
    out.mkdir(parents=True, exist_ok=True)
    jpgs = sorted(p for p in images.glob("*.jpg") if not re.search(r"_(clean|words|slots|gt)$", p.stem))
    if images != folder:  # a dataset: labels/ and gt/ live beside images/
        raise SystemExit("point the audit at a generator or QA folder (with .json, .md and .html beside the .jpg)")
    total, thumbs, rows = Counter(), [], []
    for jpg in jpgs:
        label, gt, problems = audit_one(jpg)
        total.update(problems)
        rows.append({"name": jpg.stem, "template": label.get("template"), "source": label.get("source"),
                     "lang": label.get("lang"), "effect": label.get("effect"), "regions": len(gt),
                     "problems": dict(problems)})
        thumbs.append((draw(jpg, gt, out), jpg.stem))
        print(jpg.stem, label.get("source"), label.get("lang"), label.get("effect"), len(gt), "regions",
              dict(problems) or "ok", flush=True)
    for k in range(0, len(thumbs), 12):  # contact sheets, 12 pages each
        part = thumbs[k:k + 12]
        tw, th = 300, 420
        sheet = Image.new("RGB", (4 * tw, 3 * th), "white")
        for i, (im, name) in enumerate(part):
            t = im.copy()
            t.thumbnail((tw - 10, th - 30))
            sheet.paste(t, ((i % 4) * tw + 5, (i // 4) * th + 25))
            ImageDraw.Draw(sheet).text(((i % 4) * tw + 5, (i // 4) * th + 5), name, fill="black")
        sheet.save(out / f"sheet_{k // 12 + 1:02d}.jpg", quality=85)
    report = {"samples": len(rows), "with_problems": sum(1 for r in rows if r["problems"]),
              "problems": dict(total), "by_source": dict(Counter(r["source"] for r in rows)),
              "by_lang": dict(Counter(r["lang"] for r in rows)), "by_effect": dict(Counter(r["effect"] for r in rows)),
              "rows": rows}
    (out / "audit.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
