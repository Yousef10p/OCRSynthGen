"""Quality checks on a rendered page, before it can become a training sample.

render_issues() looks at what the browser measured after SynthDoc laid the page out (pipeline/doc_pipeline.py
adds the measurements to the Arabic extension's word script) and counts, per kind of problem:

  hidden      words that are in a slot's text but were not measured on the page (fully clipped, outside every
              visible area, display:none): text the ground truth would claim but the image doesn't show
  cut         words partly hidden: cut on the left or right by a box edge, or more than 3% of their letters'
              height hidden above or below
  mismeasured words whose box is much wider than their letters (punctuation at both ends of a word in
              right-to-left text splits it into direction runs, and its box spans the line)
  overlap     letters of two different slots on top of each other (more than 20% of the smaller glyph box:
              the word's width by its letters' measured top and bottom)
  picture     a word more than 15% under a picture (an <img>, a picture box or a filled grey box)
  offpage     letters outside the page
  generated   text drawn by CSS (numbered list markers, ::before/::after content): on the image but in no slot
  contrast    a slot whose text has less than 3:1 contrast with what is behind it
  empty       no text on the page at all

DocPipeline makes the page again (new style and text, then another template) until every count is 0, so a
sample's ground truth always matches what is visible. run() is the test harness: python -m pipeline.qa.
"""
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

BULLETS = {"disc", "circle", "square", "none", "disclosure-closed", "disclosure-open"}


def _area(b):
    return max(0.0, b[2]) * max(0.0, b[3])


def _inter(a, b):
    ix = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    iy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0.0, ix) * max(0.0, iy)


INK_MIN = 0.01  # share of a word's letter box that must differ from its background


def ink_issues(image, words):
    """{"invisible": n}: words measured as shown whose letter box has (almost) no ink in the rendered image.
    Whatever hides them (Paged.js moving a line to the next page, a clip the measuring can't see), the
    pixels say whether they are there. Words of punctuation only are skipped (too little ink to judge)."""
    gray = np.asarray(image.convert("L"), np.int16)
    n = 0
    for w in words:
        if not re.search(r"[^\W_]", w["text"]):
            continue
        x, y, bw, bh = (int(round(v)) for v in (w.get("glyph_box") or w["ink_box"]))
        crop = gray[max(0, y):max(0, y + bh), max(0, x):max(0, x + bw)]
        if crop.size < 20:
            continue
        if (np.abs(crop - np.median(crop)) > 40).mean() < INK_MIN:
            n += 1
    return {"invisible": n} if n else {}


def render_issues(ocr, qa):
    """{issue: count} for a rendered page (empty dict: the page is fine)."""
    words = ocr["words"]
    issues = Counter()
    if not words:
        issues["empty"] += 1
        return dict(issues)
    measured = Counter(w.get("slot") for w in words)
    for slot, text in (qa or {}).get("slots", {}).items():
        missing = len(text.split()) - measured.get(slot, 0)
        if missing > 0:
            issues["hidden"] += missing
    pw, ph = ocr["width"], ocr["height"]
    for w in words:
        if w.get("cut_x") or w.get("cut_y", 0) > 0.03:
            issues["cut"] += 1
        if w["width"] > max(3, len(w["text"])) * w["font_size_px"] * 1.0:
            issues["mismeasured"] += 1
        x, y, bw, bh = w.get("glyph_box") or w["ink_box"]
        if x < -1 or y < -1 or x + bw > pw + 1 or y + bh > ph + 1:
            issues["offpage"] += 1
    # letters of two slots on top of each other (glyph boxes: the letters' own top and bottom, not the font's
    # taller line slot; sorted by top edge, so only nearby words are compared)
    ink = sorted(((w.get("glyph_box") or w["ink_box"], w.get("slot")) for w in words), key=lambda t: t[0][1])
    for i, (a, sa) in enumerate(ink):
        for b, sb in ink[i + 1:]:
            if b[1] > a[1] + a[3]:
                break
            if sa != sb and _inter(a, b) > 0.2 * min(_area(a), _area(b)):
                issues["overlap"] += 1
    for p in (qa or {}).get("pictures", []):
        rect = (p["x"], p["y"], p["width"], p["height"])
        issues["picture"] += sum(_inter((w["x"], w["y"], w["width"], w["height"]), rect)
                                 > 0.15 * _area((w["x"], w["y"], w["width"], w["height"])) for w in words)
    for g in ocr.get("untracked_generated_text", []):
        if g["kind"] == "list-marker":
            if g["content"] not in BULLETS:
                issues["generated"] += 1
        elif re.search(r"[^\W_]", g["content"].strip("\"'")):
            issues["generated"] += 1
    issues["contrast"] += sum(1 for c in (qa or {}).get("contrast", []) if c["ratio"] < 3.0)
    issues["covered"] += sum(c["points"] for c in (qa or {}).get("covered", []))
    return {k: v for k, v in issues.items() if v}


# -- test harness -------------------------------------------------------------------------------------------

def run():
    """Make many samples across languages, sources and effects; report what the checks caught and how often
    a page had to be made again. Writes pipeline_out/qa/ (samples) and pipeline_out/qa/report.json."""
    import argparse
    from pathlib import Path

    from pipeline.doc_pipeline import ROOT, DocPipeline
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "pipeline_out" / "qa"))
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    pipe = DocPipeline(a.out)
    rows, start = [], time.time()
    for i in range(a.n):
        s = pipe.sample(a.seed * 10_000 + i, f"q{i:03d}")
        rows.append(s)
        print(json.dumps({k: s[k] for k in ("name", "template", "source", "lang", "effect", "words", "attempts",
                                            "rejected", "legibility")}, ensure_ascii=False), flush=True)
    by = defaultdict(Counter)
    for r in rows:
        by["source"][r["source"]] += 1
        by["lang"][r["lang"]] += 1
        by["effect"][r["effect"]] += 1
        for issue, n in r["rejected"].items():
            by["rejections"][issue] += n
    report = {"samples": len(rows), "seconds": round(time.time() - start), **{k: dict(v) for k, v in by.items()},
              "mean_attempts": round(sum(r["attempts"] for r in rows) / len(rows), 2)}
    Path(a.out, "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    run()
