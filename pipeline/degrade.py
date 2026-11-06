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

Boxes move with the page: OCRSmith moves its own annotation (Page/Region/Line/Word: each word's four corners,
through the same rotation, perspective, curl or wrinkles as the pixels); scan-simulator warps a word-id mask
with the image (word_mask: words that overlap go in separate layers); Augraphy moves nothing.

legibility() checks the result word by word, so a strong setting can't silently destroy text: no word may be
lost (pushed off the page, under a hole), at most 1% of words may be faint (ink vs paper < 25 of 255) or
drowned (letters that barely stand out from the noise around them: separation()), and the words' sharpness
may not drop below 10% of the clean page's (blur, downscaling). DocPipeline retries
with new seeds, then with less strength, and keeps only a page that passes.
"""
import copy
import random

import cv2
import numpy as np
from ocrsmith.core.backgrounds import gradient_background, image_background, paper_background, solid_background
from ocrsmith.core.degradations import build_preset
from ocrsmith.core.degradations import physical as _ocr_physical
from ocrsmith.core.degradations.base import map_page as _ocr_map_page
from ocrsmith.domain.annotations import Line, Page, Region, RegionType, Word
from ocrsmith.domain.geometry import BBox
from ocrsmith.text.script import Direction
from PIL import Image
from scan_simulator import ScanSimulator
from scan_simulator.config import PipelineConfig
from scan_simulator.presets import PRESETS as SCAN_PRESETS
from scan_simulator.transforms import damage as _scan_damage, paper as _scan_paper, scanner as _scan_scanner
from scipy import fft as _fft
from scipy.ndimage import find_objects, gaussian_filter as _gaussian_filter

from pipeline.style import OCRSMITH


# scan-simulator blurs whole pages with scipy's gaussian_filter at radii up to a fifth of the page (paper colour,
# ink fading, wear patches). On our 2x pages that costs 3-30 s per transform. The same blur, faster and with
# the same result (within 0.05 of a grey level): large radii through an FFT (its cost doesn't grow with the
# radius), and a mostly empty mask (a scuff, a stain) blurred only around what is in it. Replaced in memory, for
# scan-simulator's transforms only; its folder is not changed.
def _fft_blur(padded, sigma, pad):
    """Gaussian blur of an array already padded by pad on every side; the unpadded middle."""
    h, w = padded.shape
    fy, fx = _fft.fftfreq(h)[:, None], _fft.rfftfreq(w)[None, :]
    kernel = np.exp(-2 * (np.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))
    out = _fft.irfft2(_fft.rfft2(padded, workers=-1) * kernel, s=padded.shape, workers=-1)
    return out[pad:h - pad, pad:w - pad]


def fast_gaussian_filter(input, sigma, order=0, output=None, mode="reflect", cval=0.0, truncate=4.0, **kwargs):
    a = np.asarray(input)
    if (order or output is not None or mode != "reflect" or truncate != 4.0 or kwargs or a.ndim != 2
            or np.ndim(sigma) or not np.issubdtype(a.dtype, np.floating) or sigma < 16):
        return _gaussian_filter(input, sigma, order=order, output=output, mode=mode, cval=cval, truncate=truncate,
                                **kwargs)
    r = int(truncate * sigma + 0.5)  # scipy's kernel radius
    rows, cols = np.flatnonzero(a.any(axis=1)), np.flatnonzero(a.any(axis=0))
    if not rows.size:
        return np.zeros_like(a)
    h, w = a.shape
    y0, y1, x0, x1 = max(0, rows[0] - r), min(h, rows[-1] + 1 + r), max(0, cols[0] - r), min(w, cols[-1] + 1 + r)
    # beyond the window the input is zero up to the kernel's reach: zeros there, scipy's mirror at the image's edges
    win = a[y0:y1, x0:x1]
    win = np.pad(win, ((r if y0 == 0 else 0, r if y1 == h else 0), (r if x0 == 0 else 0, r if x1 == w else 0)), mode="symmetric")
    win = np.pad(win, ((0 if y0 == 0 else r, 0 if y1 == h else r), (0 if x0 == 0 else r, 0 if x1 == w else r)))
    out = np.zeros_like(a)
    out[y0:y1, x0:x1] = _fft_blur(win, sigma, r)
    return out


for _module in (_scan_damage, _scan_paper, _scan_scanner):
    _module.gaussian_filter = fast_gaussian_filter


# OCRSmith's page curl and wrinkles move the pixels by a field d: the new pixel q shows the old pixel q + d(q).
# Its boxes move each corner p to p - d(p), the field read at the old place, but the corner's new place q is
# where q + d(q) = p. Where the field changes fast (near a curled edge) the two differ by up to 40 px: boxes
# slid off their words. Here the corner's new place is found by repeating q = p - d(q) (the field changes by
# well under a pixel per pixel, so this converges in a few rounds). The pixels are unchanged. Replaced in
# memory; OCRSmith's folder is not changed.
def _displace(self, image, page, rng):
    width, height = image.size
    dx, dy = self._field(width, height, rng)[:2]
    dx, dy = np.broadcast_to(dx, (height, width)), np.broadcast_to(dy, (height, width))

    def mapper(x, y):
        qx, qy = x, y
        for _ in range(10):
            col, row = min(max(int(round(qx)), 0), width - 1), min(max(int(round(qy)), 0), height - 1)
            qx, qy = x - float(dx[row, col]), y - float(dy[row, col])
        return qx, qy
    return _ocr_physical._remap(image, dx, dy, self.fill), _ocr_map_page(page, mapper, width, height), self._field_params


_ocr_physical._DisplacementDegradation.apply = _displace

TOOLS = ("ocrsmith", "augraphy", "scan-simulator")
DEFAULT_STRENGTH = 0.5
# how often each preset is picked
OCRSMITH_PRESETS = {"scan": 4.0, "photo": 3.0, "fax": 0.7, "archive": 1.5}
SCAN_PRESET_WEIGHTS = {"scan-clean": 2, "scan-heavy": 2, "photo-indoor": 2, "photo-outdoor": 2, "photocopy": 1.5,
                       "archive": 1.5}


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


def ocrsmith_paper(rng, w, h):
    """An OCRSmith paper for the page to be printed on: (RGBA image, its kind)."""
    kind = rng.choices(["paper", "solid", "gradient", "photo texture"], weights=[5, 2, 1.5, 0.8])[0]

    def tint():
        # One factor for all channels, from cream to white: real paper never turns pink or green.
        f = rng.random()
        return tuple(int(a + f * (b - a)) for a, b in zip((226, 218, 198), (255, 255, 252)))

    if kind == "paper":
        bg = paper_background(tint(), rng.uniform(2.0, 9.0), rng)
    elif kind == "solid":
        bg = solid_background(tint())
    elif kind == "gradient":
        bg = gradient_background(tint(), tint(), rng.choice(["horizontal", "vertical", "diagonal"]))
    else:
        # OCRSmith's other photos (bricks, containers, city...) hide the text; a document sits on paper.
        kind = "photo texture (white paper)"
        bg = image_background(OCRSMITH / "backgrounds" / "white_paper.jpg")
    return bg(w, h).convert("RGBA"), kind


def to_page(words, width, height):
    """Words as an OCRSmith Page (which its degradations move): one region per line, each word's text set to
    its index."""
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
    """Word boxes [x, y, w, h] by index from an OCRSmith Page; None for a word pushed off the page."""
    boxes = [None] * n
    for region in page.regions:
        for line in region.lines:
            for w in line.words:
                b = w.bbox
                if b.x1 - b.x0 >= 1 and b.y1 - b.y0 >= 1:
                    x, y = int(b.x0), int(b.y0)
                    boxes[int(w.text)] = [x, y, int(np.ceil(b.x1)) - x, int(np.ceil(b.y1)) - y]
    return boxes


def degrade_ocrsmith(image, words, seed, strength):
    """OCRSmith paper, then one capture preset at this strength; returns (image, word boxes, attacks)."""
    rng = random.Random(seed)
    paper, kind = ocrsmith_paper(rng, *image.size)
    out = Image.fromarray((np.asarray(image, np.float32) * np.asarray(paper.convert("RGB"), np.float32) / 255)
                          .round().astype(np.uint8))
    preset = rng.choices(list(OCRSMITH_PRESETS), weights=list(OCRSMITH_PRESETS.values()))[0]
    out, page, records = ocrsmith_pipeline(preset, strength).apply(out, to_page(words, *image.size), rng)
    return (out.convert("RGB"), from_page(page, len(words)),
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


def word_mask(shape, words, layers=3):
    """Word i painted with id i + 1 (0 = no word), in the first of `layers` channels where it overlaps no word
    already painted: a word painted over another (two lines' boxes touching) would cut the other's box.
    scan-simulator moves this mask with the page."""
    mask = np.zeros((*shape, layers), np.uint16)
    for i, w in enumerate(words):
        x, y = max(0, round(w["x"])), max(0, round(w["y"]))
        area = mask[y:round(w["y"] + w["height"]) + 1, x:round(w["x"] + w["width"]) + 1]
        k = next((k for k in range(layers) if not area[..., k].any()), layers - 1)
        area[..., k] = i + 1
    return mask


def mask_boxes(mask, n):
    """Each word's box [x, y, w, h] after the effect: the extent of its id (None if it is gone)."""
    boxes = [None] * n
    for k in range(mask.shape[2]):
        for i, f in enumerate(find_objects(mask[..., k].astype(np.int32), max_label=n)):
            if f:
                boxes[i] = [f[1].start, f[0].start, f[1].stop - f[1].start, f[0].stop - f[0].start]
    return boxes


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
    out, mask = ScanSimulator(scan_config(preset, strength), seed=seed)(bgr, word_mask(bgr.shape[:2], words))
    return (Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)), mask_boxes(mask, len(words)),
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
    scale = width * height / (clean.width * clean.height)

    def at_border(b, w, h):
        return b[0] <= 1 or b[1] <= 1 or b[0] + b[2] >= w - 1 or b[1] + b[3] >= h - 1
    # cut by the frame: at the border after the effect, and either not at the border before it (text running
    # to the page's edge is whole) or its box shrank (part of it went out of the picture)
    edge = 0
    for w, b in zip(words, boxes):
        if b and at_border(b, width, height):
            c = (w["x"], w["y"], w["width"], w["height"])
            if not at_border(c, clean.width, clean.height) or b[2] * b[3] < 0.8 * c[2] * c[3] * scale:
                edge += 1
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


def warm_up():
    """Augraphy compiles its noise code (numba) on first use, about 15 s: do it before the first real page."""
    degrade_augraphy(Image.new("RGB", (200, 200), "white"), [], 0, DEFAULT_STRENGTH)
