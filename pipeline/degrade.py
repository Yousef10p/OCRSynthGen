"""Step 4: degrade the rendered page with OCRSmith, Augraphy or scan-simulator, at a chosen strength.

Every tool takes a strength from 0 to 1 (the generator page has a slider per tool, the dataset command
--strength-<tool>). 0.5 is the tool's own default settings; 0 is the lightest, 1 roughly twice as strong:

  * OCRSmith: every degradation of the chosen capture preset (scan, photo, fax, archive) has its numbers
    scaled by f = 2 * strength around their neutral value (rotation angle, noise sigma, blur radius, warp
    magnitude, stain strength... x f; JPEG quality and downscale toward "none"; contrast/brightness factors
    toward 1) and its probability by (0.5 + strength);
  * Augraphy: the ink, paper and post-processing effects get their intensity ranges scaled the same way and
    their probabilities by (0.5 + strength); only effects that don't move pixels are used;
  * scan-simulator: the chosen preset's transforms (rotation, perspective, folds, noise, blur, stains...) have
    their ranges scaled by f around neutral and their probabilities by (0.5 + strength).

Boxes move with the page: OCRSmith moves its own annotation (Page/Region/Line/Word); scan-simulator warps a
16-bit word-id mask with the image; Augraphy moves nothing.

legibility() checks the result word by word, so a strong setting can't silently destroy text: no word may be
lost (pushed off the page, under a hole), at most 1% of words may be faint (ink vs paper < 25 of 255) or
drowned (letters that barely stand out from the noise around them: separation()), and the words' sharpness
may not drop below 10% of the clean page's (blur, downscaling). DocPipeline retries
with new seeds, then with less strength, and keeps only a page that passes.
"""
import copy
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

for _p in (Path(__file__).resolve().parents[1] / "synthdoc_adapter", Path(__file__).resolve().parents[1] / "demo"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import attack  # noqa: E402  synthdoc_adapter/attack.py (Augraphy helpers, word mask)
import attack_ocrsmith  # noqa: E402  synthdoc_adapter/attack_ocrsmith.py (OCRSmith paper, Page conversion)
from ocrsmith.core.degradations import build_preset  # noqa: E402
from scan_simulator import ScanSimulator  # noqa: E402
from scan_simulator.config import PipelineConfig  # noqa: E402
from scan_simulator.presets import PRESETS as SCAN_PRESETS  # noqa: E402

TOOLS = ("ocrsmith", "augraphy", "scan-simulator")
DEFAULT_STRENGTH = 0.5
OCRSMITH_PRESETS = attack_ocrsmith.PRESETS  # scan 4, photo 3, fax 0.7, archive 1.5
SCAN_PRESET_WEIGHTS = attack.PRESETS        # scan-clean, scan-heavy, photo-indoor, photo-outdoor, photocopy, archive


def numeric(v):
    """A number or a range of numbers (the settings a strength can scale)."""
    if isinstance(v, (tuple, list)):
        return bool(v) and all(numeric(x) for x in v)
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def toward(value, neutral, f, lo=None, hi=None):
    """value moved from neutral by f (f = 1: unchanged); works on numbers and (low, high) ranges."""
    if isinstance(value, (tuple, list)):
        out = [toward(v, neutral, f, lo, hi) for v in value]
        return type(value)(sorted(out)) if isinstance(value, tuple) else sorted(out)
    v = neutral + (value - neutral) * f
    v = max(lo, v) if lo is not None else v
    v = min(hi, v) if hi is not None else v
    return round(v) if isinstance(value, int) and not isinstance(value, bool) else v


# -- OCRSmith -----------------------------------------------------------------------------------------------

def ocrsmith_pipeline(preset, strength):
    pipeline = build_preset(preset)
    f, pm = 2 * strength, 0.5 + strength
    for d in pipeline.degradations:
        name = type(d).__name__
        d.probability = min(1.0, d.probability * pm)
        for k, v in list(vars(d).items()):
            if k in ("probability", "fill", "edge", "angle") or not numeric(v):
                continue
            if k == "scale" and name != "Downscale" or k == "size":
                continue  # a spatial scale or a kernel size, not a strength
            if k == "quality":
                setattr(d, k, toward(v, 100, f, 5, 100))
            elif k == "scale":  # Downscale: 1 = no loss
                setattr(d, k, toward(v, 1.0, f, 0.25, 1.0))
            elif k == "factor":  # Contrast, Brightness: 1 = unchanged
                setattr(d, k, toward(v, 1.0, f, 0.3, 3.0))
            elif k == "count":
                setattr(d, k, toward(v, 0, f, 0))
            else:
                setattr(d, k, toward(v, 0, f, 0))
    return pipeline


def degrade_ocrsmith(image, words, seed, strength):
    """OCRSmith paper, then one capture preset at this strength; returns (image, word boxes, attacks)."""
    rng = random.Random(seed)
    paper, kind = attack_ocrsmith.background(rng, *image.size)
    out = Image.fromarray((np.asarray(image, np.float32) * np.asarray(paper.convert("RGB"), np.float32) / 255)
                          .round().astype(np.uint8))
    preset = rng.choices(list(OCRSMITH_PRESETS), weights=list(OCRSMITH_PRESETS.values()))[0]
    out, page, records = ocrsmith_pipeline(preset, strength).apply(out, attack_ocrsmith.to_page(words, *image.size), rng)
    return (out.convert("RGB"), attack_ocrsmith.from_page(page, len(words)),
            [f"ocrsmith-paper:{kind}", f"ocrsmith:{preset}"] + [r.name for r in records])


# -- Augraphy -----------------------------------------------------------------------------------------------

def augraphy_pipeline(seed, strength):
    from augraphy import (AugraphyPipeline, BadPhotoCopy, Brightness, BrightnessTexturize, ColorPaper, DirtyDrum,
                          DirtyScreen, InkBleed, InkMottling, Jpeg, Letterpress, LightingGradient, LowInkPeriodicLines,
                          LowInkRandomLines, NoiseTexturize, OneOf, ShadowCast, SubtleNoise)
    f, pm = 2 * strength, 0.5 + strength

    def p(x):
        return min(1.0, x * pm)
    return AugraphyPipeline(
        ink_phase=[OneOf([InkBleed(intensity_range=toward((0.4, 0.7), 0, f, 0.05, 0.95)),
                          LowInkRandomLines(count_range=toward((5, 10), 0, f, 1)),
                          LowInkPeriodicLines(count_range=toward((2, 5), 0, f, 1)),
                          InkMottling(ink_mottling_alpha_range=toward((0.2, 0.3), 0, f, 0.02, 0.6))], p=p(0.6)),
                   Letterpress(p=p(0.25))],
        paper_phase=[OneOf([NoiseTexturize(sigma_range=toward((3, 10), 0, f, 1)),
                            BrightnessTexturize(texturize_range=toward((0.8, 0.99), 1.0, f, 0.5, 0.999))], p=p(0.6)),
                     ColorPaper(saturation_range=toward((10, 40), 0, f, 1), p=p(0.3))],
        post_phase=[OneOf([DirtyDrum(noise_intensity=toward(0.5, 0, f, 0.05, 0.6)),  # above 0.6: black bands over lines
                           DirtyScreen(), SubtleNoise(subtle_range=max(1, round(10 * f)))], p=p(0.6)),
                    OneOf([LightingGradient(),
                           ShadowCast(shadow_opacity_range=toward((0.2, 0.9), 0, f, 0.05, 0.95)),
                           Brightness(brightness_range=toward((0.8, 1.4), 1.0, f, 0.4, 1.8))], p=p(0.5)),
                    Jpeg(quality_range=toward((25, 95), 100, f, 5, 100), p=p(0.5)),
                    BadPhotoCopy(p=p(0.15))],
        random_seed=seed)


def degrade_augraphy(image, words, seed, strength):
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    out, used = None, None
    for s in range(seed, seed + 3):  # Augraphy's noise code sometimes fails for one random draw
        try:
            out, used = augraphy_pipeline(s, strength).augment(bgr)["output"], s
            break
        except Exception:
            continue
    if out is None:
        out = bgr
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR) if out.ndim == 2 else out
    boxes = [[round(w["x"]), round(w["y"]), round(w["width"]), round(w["height"])] for w in words]
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)), boxes, (["augraphy"] if used is not None else [])


# -- scan-simulator -----------------------------------------------------------------------------------------

NEUTRAL = {"quality": 100, "darkness": 0}
INTEGER_SIZES = ("size", "kernel", "line_width", "radius", "shift")


def scan_config(preset, strength):
    cfg = copy.deepcopy(SCAN_PRESETS[preset])
    f, pm = 2 * strength, 0.5 + strength
    for name, params in cfg["transforms"].items():
        params["p"] = min(1.0, params.get("p", 1.0) * pm)
        for k, v in list(params.items()):
            if k in ("p", "color", "type", "direction") or not numeric(v):
                continue
            if k == "quality":
                params[k] = toward(v, 100, f, 5, 100)
            elif k in INTEGER_SIZES:
                params[k] = toward(v, 0, f, 1)
            elif k == "count":
                params[k] = toward(v, 0, f, 1)
            else:
                params[k] = toward(v, 0, f)
    return PipelineConfig.from_dict(cfg)


SCAN_PAD = 0.04  # paper added on each side before scan-simulator, times the page's longer side (x 1 at strength 0.5)


def degrade_scan(image, words, seed, strength, use_augraphy=False):
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    attacks = []
    if use_augraphy:
        img, _, a = degrade_augraphy(image, words, seed, strength)
        bgr, attacks = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR), a
    rng = np.random.default_rng(seed)
    w = SCAN_PRESET_WEIGHTS
    preset = str(rng.choice(list(w), p=np.array(list(w.values())) / sum(w.values())))
    # more paper around the page first (its own edge colour): rotation and perspective then turn the page
    # inside the picture instead of pushing its corners, and the words there, out of the frame
    pad = int(round(SCAN_PAD * max(bgr.shape[:2]) * (0.5 + strength)))
    bgr = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    words = [{**wd, "x": wd["x"] + pad, "y": wd["y"] + pad} for wd in words]
    out, mask = ScanSimulator(scan_config(preset, strength), seed=seed)(bgr, attack.word_mask(bgr.shape[:2], words))
    return (Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)), attack.mask_boxes(mask, len(words)),
            attacks + [f"scan-simulator:{preset}"])


def degrade(image, words, seed, effect, strength):
    """(image, word boxes, attacks) for one effect at one strength."""
    if effect == "ocrsmith":
        return degrade_ocrsmith(image, words, seed, strength)
    if effect == "augraphy":
        return degrade_augraphy(image, words, seed, strength)
    return degrade_scan(image, words, seed, strength, use_augraphy=effect.startswith("augraphy"))


# -- is the text still readable? ----------------------------------------------------------------------------

def _crop(gray, box):
    x, y, w, h = (int(round(v)) for v in box)
    return gray[max(0, y):max(0, y) + h, max(0, x):max(0, x) + w]


def separation(c0, c1):
    """How clearly a degraded word's letters stand out from what is around them, or None (no letters).

    The clean crop says which share of the word's pixels are letters and whether they are darker or lighter
    than the background (Otsu threshold). In the degraded crop that share of its darkest (or lightest)
    pixels counts as letters, the rest as background; the score is the distance between their means over
    the sum of their spreads. Noise that keeps contrast high but drowns the letters makes it small. It needs
    no alignment, so it works after rotation and warps too."""
    t, _ = cv2.threshold(c0.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = float((c0 < t).mean())
    letters_dark = dark <= 0.5
    share = dark if letters_dark else 1 - dark
    if share < 0.02:
        return None
    v = np.sort(c1.ravel())
    k = max(1, int(share * v.size))
    letters, ground = (v[:k], v[k:]) if letters_dark else (v[-k:], v[:-k])
    if not ground.size:
        return None
    return abs(float(ground.mean()) - float(letters.mean())) / (float(letters.std()) + float(ground.std()) + 1.0)


FAINT_GAP = 40  # grey levels between a word's letters and its background below which it reads as washed out


def gap(c):
    """Grey levels between the two sides of a crop split by Otsu: its letters and its background. Noise keeps
    the crop's range wide when lighting washes text out, but not this gap (washed-out words ~25, others 100+)."""
    t, _ = cv2.threshold(c.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low, high = c[c < t], c[c >= t]
    return abs(float(high.mean()) - float(low.mean())) if low.size and high.size else 0.0


def dark_share(c):
    """Share of a crop darker than halfway between its paper (90th percentile) and its ink (2nd)."""
    paper, ink = np.percentile(c, 90), np.percentile(c, 2)
    return float((c < (paper + ink) / 2).mean()) if paper - ink >= 25 else 0.0


def covered(c0, c1):
    """How much more of a word's box is dark after the effect than before: a streak or blot over the word
    darkens far more of the box than its letters (blur and ink bleed only spread them a little)."""
    return dark_share(c1) - dark_share(c0)


def legibility(clean, out, words, boxes, sample=250):
    """Per-word checks after an effect: {"lost": n, "faint": share, "sharpness": median ratio,
    "separation": median, "drowned": share of words whose letters barely stand out, "edge": words cut by the
    frame (a photo preset that crops the page leaves part of a word outside the picture)}."""
    lost = sum(b is None for b in boxes)
    width, height = out.size
    edge = sum(1 for b in boxes if b and (b[0] <= 1 or b[1] <= 1 or b[0] + b[2] >= width - 1 or b[1] + b[3] >= height - 1))
    g0 = np.asarray(clean.convert("L"), np.float32)
    g1 = np.asarray(out.convert("L"), np.float32)
    idx = [i for i, b in enumerate(boxes) if b]
    if len(idx) > sample:
        idx = random.Random(0).sample(idx, sample)
    faint, ratios, seps, blots = 0, [], [], []
    for i in idx:
        c1 = _crop(g1, boxes[i])
        c0 = _crop(g0, (words[i]["x"], words[i]["y"], words[i]["width"], words[i]["height"]))
        if c1.size < 20 or c0.size < 20:
            continue
        if np.percentile(c1, 90) - np.percentile(c1, 5) < 25 or gap(c1) < FAINT_GAP:
            faint += 1
        s0 = cv2.Laplacian(c0, cv2.CV_32F).var()
        if s0 > 1:
            ratios.append(cv2.Laplacian(c1, cv2.CV_32F).var() / s0)
        s = separation(c0, c1)
        if s is not None:
            seps.append(s)
        blots.append(covered(c0, c1))
    n = max(1, len(idx))
    return {"lost": lost, "edge": edge, "faint": round(faint / n, 3),
            "sharpness": round(float(np.median(ratios)), 3) if ratios else 1.0,
            "separation": round(float(np.median(seps)), 2) if seps else 9.0,
            "drowned": round(sum(s < SEPARATION_MIN for s in seps) / max(1, len(seps)), 3),
            "blotted": sum(b > BLOT_MAX for b in blots), "blot_max": round(max(blots, default=0.0), 3)}


# a word with this much more of its box dark than before is under a streak or blot (set by looking at the
# highest-scoring words of 120 degraded pages: blotted-out words scored 0.33-0.51, still readable ones <= 0.27)
BLOT_MAX = 0.30
SEPARATION_MIN = 1.2  # below this a word's letters barely stand out from the noise around them


def readable(check):
    """No word lost, cut by the frame or blotted out; at most 1% of words faint or drowned (measurement
    noise: at default strength there are none, and a streak over a few words must not pass); sharpness at
    least 10% of the clean page's (a strong blur still readable passes, a page blurred away does not)."""
    return (check["lost"] == 0 and check.get("edge", 0) == 0 and check.get("blotted", 0) == 0
            and check["faint"] <= 0.01 and check["sharpness"] >= 0.10 and check.get("drowned", 0) <= 0.01)
