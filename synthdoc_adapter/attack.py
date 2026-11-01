"""Attack SynthDoc's clean renders with Augraphy, then scan-simulator, keeping every label aligned.

1. Augraphy (github.com/sparkfish/augraphy): ink, paper and lighting effects only (ink bleed, low ink,
   paper texture, dirty drum, shadows, JPEG…). None of them moves a pixel, so the boxes stay as they are.
2. scan-simulator (github.com/s1mb1o/scan-simulator): one of its presets (scan-clean, scan-heavy,
   photo-indoor, photo-outdoor, photocopy, archive), which rotate, warp, fold and curl the page.
   Boxes are carried through as a 16-bit mask with each word painted in its own id; scan-simulator warps
   masks with nearest-neighbour sampling, so after the attack each word's box is the extent of its id.
   Lines and slots become the union of their words.

Two guards keep labels honest: words whose letters are cut by a box edge are dropped first, and a page whose
text ends up too faint (median paper-minus-ink contrast below MIN_CONTRAST) is attacked again with another
seed, then with scan-simulator alone; if every try is faint the most readable one is kept, marked low_contrast.

Writes synthdoc_attacked/: images/, labels/ (words, lines, slots with new boxes), overlays/, summary.json.
Runs in the attacks venv; scan-simulator is used from ../scan-simulator (its pyproject does not install):

    .venv-attacks\\Scripts\\python synthdoc_adapter\\attack.py --workers 6
"""
import argparse
import json
import shutil
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import find_objects

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "scan-simulator"))
from scan_simulator import ScanSimulator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import MIN_CONTRAST, build_label, contrast, load_page, summary  # noqa: E402

SRC, OUT = ROOT / "synthdoc_out", ROOT / "synthdoc_attacked"
PRESETS = {"scan-clean": 2, "scan-heavy": 2, "photo-indoor": 2, "photo-outdoor": 2, "photocopy": 1.5, "archive": 1.5}


def augraphy_pipeline(seed):
    """Pixel-level Augraphy effects only: nothing here moves text, so labels need no change."""
    from augraphy import (AugraphyPipeline, BadPhotoCopy, Brightness, BrightnessTexturize, ColorPaper, DirtyDrum,
                          DirtyScreen, InkBleed, InkMottling, Jpeg, Letterpress, LightingGradient, LowInkPeriodicLines,
                          LowInkRandomLines, NoiseTexturize, OneOf, ShadowCast, SubtleNoise)
    return AugraphyPipeline(
        ink_phase=[OneOf([InkBleed(), LowInkRandomLines(), LowInkPeriodicLines(), InkMottling()], p=0.6),
                   Letterpress(p=0.25)],
        paper_phase=[OneOf([NoiseTexturize(), BrightnessTexturize()], p=0.6), ColorPaper(p=0.3)],
        post_phase=[OneOf([DirtyDrum(), DirtyScreen(), SubtleNoise()], p=0.6),
                    OneOf([LightingGradient(), ShadowCast(), Brightness()], p=0.5), Jpeg(p=0.5), BadPhotoCopy(p=0.15)],
        random_seed=seed)


def run_augraphy(image, seed):
    """Augraphy's noise generator sometimes fails to compile under numba for one random draw: try the next
    seed, and after three failures leave the page to scan-simulator alone. Returns (image, seed or None)."""
    for s in range(seed, seed + 3):
        try:
            return augraphy_pipeline(s).augment(image)["output"], s
        except Exception:
            continue
    return image, None




def draw(image, boxes, color, path):
    out = image.copy()
    for b in boxes:
        x, y, w, h = map(int, b)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 1)
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 85])






def degrade(image, mask, seed, use_augraphy=True):
    """Augraphy (optional) then one scan-simulator preset. Returns (image, mask, attacks)."""
    out, augraphy_seed = run_augraphy(image, seed) if use_augraphy else (image, None)
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    assert out.shape[:2] == image.shape[:2], "Augraphy changed the page size"
    rng = np.random.default_rng(seed)
    preset = str(rng.choice(list(PRESETS), p=np.array(list(PRESETS.values())) / sum(PRESETS.values())))
    out, mask = ScanSimulator.from_preset(preset, seed=seed)(out, mask)
    return out, mask, (["augraphy"] if augraphy_seed is not None else []) + [f"scan-simulator:{preset}"]


def word_mask(shape, words):
    """Word i painted with id i + 1 (0 = no word)."""
    mask = np.zeros(shape, np.uint16)
    for i, w in enumerate(words):
        x, y = max(0, round(w["x"])), max(0, round(w["y"]))
        mask[y:round(w["y"] + w["height"]) + 1, x:round(w["x"] + w["width"]) + 1] = i + 1
    return mask


def mask_boxes(mask, n):
    """Each word's box [x, y, w, h] after the attack: the extent of its id (None if it is gone)."""
    found = find_objects(mask.astype(np.int32), max_label=n)
    return [[f[1].start, f[0].start, f[1].stop - f[1].start, f[0].stop - f[0].start] if f else None for f in found]


def attack(page):
    seed = zlib.crc32(page.encode()) % 2**31
    image = cv2.imread(str(SRC / "images" / f"{page}.jpg"))
    ocr, slots, words, cut = load_page(SRC, page)
    mask = word_mask(image.shape[:2], words)

    # Text must stay readable: up to 4 seeds with both libraries, then scan-simulator alone; else the best try
    best = None
    tries = [(seed + k * 1000, True) for k in range(4)] + [(seed, False)]
    for s, use_augraphy in tries:
        out, out_mask, attacks = degrade(image, mask, s, use_augraphy)
        boxes = mask_boxes(out_mask, len(words))
        c = contrast(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), [b for b in boxes if b])
        if best is None or c > best[0]:
            best = (c, out, boxes, attacks, s)
        if c >= MIN_CONTRAST:
            break
    page_contrast, out, word_boxes, attacks, used_seed = best
    label = build_label(page, (out.shape[1], out.shape[0]), used_seed, attacks, page_contrast, ocr, slots, words,
                        word_boxes, cut)
    cv2.imwrite(str(OUT / "images" / f"{page}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    (OUT / "labels" / f"{page}.json").write_text(json.dumps(label, ensure_ascii=False), encoding="utf-8")
    draw(out, [w["box"] for w in label["words"]], (60, 160, 60), OUT / "overlays" / f"{page}_words.jpg")
    draw(out, [s["box"] for s in label["slots"]], (40, 40, 200), OUT / "overlays" / f"{page}_slots.jpg")
    return summary(label, len(words))




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="attack only the first N pages (0 = all)")
    ap.add_argument("--resume", action="store_true", help="keep pages already attacked, do the rest")
    a = ap.parse_args()
    if not a.resume:
        shutil.rmtree(OUT, ignore_errors=True)
    for d in ("images", "labels", "overlays"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    pages = [p for doc in json.loads((SRC / "manifest.json").read_text(encoding="utf-8"))["documents"] for p in doc["pages"]]
    pages = pages[: a.limit] if a.limit else pages
    done = {p.stem for p in (OUT / "labels").glob("*.json")} if a.resume else set()
    with Pool(a.workers) as pool:
        results = pool.map(attack, [p for p in pages if p not in done])
    for p in sorted(done & set(pages)):
        label = json.loads((OUT / "labels" / f"{p}.json").read_text(encoding="utf-8"))
        results.append(summary(label, len(label["words"]) + label["lost_words"]))
    results.sort(key=lambda r: r["page"])
    (OUT / "summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    lost, total = sum(r["lost_words"] for r in results), sum(r["words"] for r in results)
    skipped = sum("augraphy" not in r["attacks"] for r in results)
    faint = sum(r["low_contrast"] for r in results)
    print(f"{faint} page(s) still below contrast {MIN_CONTRAST} after all tries (marked low_contrast)")
    print(f"attacked {len(results)} pages; {lost} of {total} words lost (holes or off-page); "
          f"Augraphy skipped on {skipped} page(s)")


if __name__ == "__main__":
    main()
