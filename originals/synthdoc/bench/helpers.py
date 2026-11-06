"""
Data loading, entity matching, scoring, and shared constants.
"""

import json
import re
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ── Constants ────────────────────────────────────────────────────────────

NER_ENTITY_TYPES = {"PERSON", "ORG", "DATE", "ADDRESS", "AMOUNT", "PHONE", "ID"}

# Full document entity types (standard NER + structural/document types)
FULL_ENTITY_TYPES = NER_ENTITY_TYPES | {
    "FIELD_LABEL", "TABLE_CELL", "TABLE_HEADER",
    "TITLE", "SECTION_TITLE", "TEXT", "CAPTION",
}

PRED_TYPE_MAP = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "Person_Name": "PERSON",
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
    "Organization_Name": "ORG",
    "LOC": "ADDRESS",
    "LOCATION": "ADDRESS",
    "location": "ADDRESS",
    "GPE": "ADDRESS",
    "MONEY": "AMOUNT",
    "CARDINAL": "AMOUNT",
    "DATE": "DATE",
    "MISC": "MISC",
    "ADDRESS": "ADDRESS",
    "AMOUNT": "AMOUNT",
    "PHONE": "PHONE",
    "ID": "ID",
}

STRUCTURAL_TYPES = {
    "FIELD_LABEL", "TABLE_CELL", "TABLE_HEADER",
    "TITLE", "SECTION_TITLE", "TEXT", "CAPTION",
}

COCO_IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


# ── Data loading ─────────────────────────────────────────────────────────

def load_ner_pages(data_dir: Path, entity_types: Optional[set] = None) -> List[Dict]:
    """Load label JSONs and extract ground truth entities + full GT text.

    Args:
        data_dir: Path to data/ directory containing images/ and labels/
        entity_types: Set of entity types to include in gt_entities.
            None = NER_ENTITY_TYPES (standard 7 types).
            Use FULL_ENTITY_TYPES for all 14 types.
    """
    if entity_types is None:
        entity_types = NER_ENTITY_TYPES

    labels_dir = data_dir / "labels"
    images_dir = data_dir / "images"

    pages = []
    for label_path in sorted(labels_dir.glob("*.json")):
        stem = label_path.stem
        image_path = images_dir / f"{stem}.jpg"
        if not image_path.exists():
            continue

        with open(label_path) as f:
            label = json.load(f)

        gt_entities = []
        all_texts = []
        for ann in label.get("annotations", []):
            etype = ann.get("entity_type", "")
            text = ann.get("text_content", "").strip()
            if text:
                all_texts.append(text)
            if etype in entity_types and text:
                gt_entities.append({"text": text, "type": etype})

        doc_id = label.get("document_id", stem.rsplit("_page_", 1)[0] if "_page_" in stem else stem)
        template_type = label.get("template_type", doc_id[:3])

        pages.append({
            "image_path": str(image_path),
            "stem": stem,
            "doc_id": doc_id,
            "template_type": template_type,
            "gt_entities": gt_entities,
            "gt_text": " ".join(all_texts),
        })

    return pages


def load_seg_pages(data_dir: Path) -> List[Dict]:
    """Load label JSONs and extract structural annotations with bounding boxes."""
    labels_dir = data_dir / "labels"
    images_dir = data_dir / "images"

    pages = []
    for label_path in sorted(labels_dir.glob("*.json")):
        stem = label_path.stem
        image_path = images_dir / f"{stem}.jpg"
        if not image_path.exists():
            continue

        with open(label_path) as f:
            label = json.load(f)

        gt_regions = []
        for ann in label.get("annotations", []):
            etype = ann.get("entity_type", "")
            box = ann.get("box_2d")
            if etype in STRUCTURAL_TYPES and box and len(box) == 4:
                gt_regions.append({"type": etype, "box_2d": box})

        doc_id = label.get("document_id", stem.rsplit("_page_", 1)[0] if "_page_" in stem else stem)
        template_type = label.get("template_type", doc_id[:3])

        pages.append({
            "image_path": str(image_path),
            "stem": stem,
            "template_type": template_type,
            "gt_regions": gt_regions,
            "width": label.get("width", 816),
            "height": label.get("height", 1056),
        })

    return pages


# ── Text normalization & matching ────────────────────────────────────────

def normalize(text: str) -> str:
    return text.lower().strip().replace(",", "").replace(".", "")


def match_entities(
    gt_entities: List[Dict],
    pred_entities: List[Dict],
    strict_type: bool,
) -> Tuple[int, int, int]:
    """
    Match predicted entities to ground truth by normalized text.
    Returns (tp, fp, fn).
    """
    gt_matched = [False] * len(gt_entities)
    pred_matched = [False] * len(pred_entities)
    gt_norm = [normalize(e["text"]) for e in gt_entities]
    pred_norm = [normalize(e["text"]) for e in pred_entities]
    pred_types = [PRED_TYPE_MAP.get(e["type"], e["type"]) for e in pred_entities]

    for gi, gt in enumerate(gt_entities):
        for pi, pred in enumerate(pred_entities):
            if pred_matched[pi]:
                continue
            text_match = (
                gt_norm[gi] == pred_norm[pi]
                or gt_norm[gi] in pred_norm[pi]
                or pred_norm[pi] in gt_norm[gi]
            )
            if not text_match:
                continue
            if strict_type and pred_types[pi] != gt["type"]:
                continue
            gt_matched[gi] = True
            pred_matched[pi] = True
            break

    tp = sum(gt_matched)
    fn = len(gt_entities) - tp
    fp = sum(1 for m in pred_matched if not m)
    return tp, fp, fn


def detailed_match(
    gt_entities: List[Dict],
    pred_entities: List[Dict],
) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
    """
    Relaxed text match returning matched pairs with types + unmatched.
    Returns (matched_pairs, unmatched_gt, unmatched_pred).
    Each matched pair: (gt_type, pred_type, gt_text, pred_text).
    """
    gt_matched = [False] * len(gt_entities)
    pred_matched = [False] * len(pred_entities)
    gt_norm = [normalize(e["text"]) for e in gt_entities]
    pred_norm = [normalize(e["text"]) for e in pred_entities]
    pred_types = [PRED_TYPE_MAP.get(e["type"], e["type"]) for e in pred_entities]

    matched_pairs = []
    for gi, gt in enumerate(gt_entities):
        for pi in range(len(pred_entities)):
            if pred_matched[pi]:
                continue
            text_match = (
                gt_norm[gi] == pred_norm[pi]
                or gt_norm[gi] in pred_norm[pi]
                or pred_norm[pi] in gt_norm[gi]
            )
            if not text_match:
                continue
            gt_matched[gi] = True
            pred_matched[pi] = True
            matched_pairs.append((gt["type"], pred_types[pi], gt["text"], pred_entities[pi]["text"]))
            break

    unmatched_gt = [
        (gt_entities[i]["type"], gt_entities[i]["text"])
        for i in range(len(gt_entities)) if not gt_matched[i]
    ]
    unmatched_pred = [
        (pred_types[i], pred_entities[i]["text"])
        for i in range(len(pred_entities)) if not pred_matched[i]
    ]
    return matched_pairs, unmatched_gt, unmatched_pred


# ── Scoring ──────────────────────────────────────────────────────────────

def calc_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Return (precision, recall, f1) rounded to 4 decimals."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


# ── OCR quality metrics ──────────────────────────────────────────────────

def _levenshtein(a: List, b: List) -> int:
    """Single-row Levenshtein distance."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(
                prev[j + 1] + 1,
                curr[j] + 1,
                prev[j] + (0 if ca == cb else 1),
            ))
        prev = curr
    return prev[-1]


def compute_wer(ref: str, hyp: str) -> float:
    """Word Error Rate between reference and hypothesis text."""
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _levenshtein(ref_words, hyp_words) / len(ref_words)


def compute_cer(ref: str, hyp: str) -> float:
    """Character Error Rate between reference and hypothesis text."""
    ref = ref.strip()
    hyp = hyp.strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(list(ref), list(hyp)) / len(ref)


# ── Segmentation helpers ─────────────────────────────────────────────────

def compute_iou(a: List[float], b: List[float]) -> float:
    """IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match_boxes(
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int]:
    """Greedy IoU matching. Returns (tp, fp, fn)."""
    if not gt_boxes and not pred_boxes:
        return 0, 0, 0
    if not gt_boxes:
        return 0, len(pred_boxes), 0
    if not pred_boxes:
        return 0, 0, len(gt_boxes)

    pairs = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            iou = compute_iou(gt, pred)
            if iou >= iou_threshold:
                pairs.append((iou, gi, pi))
    pairs.sort(key=lambda x: -x[0])

    gt_matched, pred_matched = set(), set()
    for iou, gi, pi in pairs:
        if gi in gt_matched or pi in pred_matched:
            continue
        gt_matched.add(gi)
        pred_matched.add(pi)

    tp = len(gt_matched)
    return tp, len(pred_boxes) - len(pred_matched), len(gt_boxes) - tp


def compute_coco_map(
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    thresholds: Optional[List[float]] = None,
) -> Tuple[float, Dict[float, Tuple[float, float, float]]]:
    """
    COCO-style mAP: F1 at IoU thresholds [0.5:0.05:0.95], averaged.
    Returns (mAP, {threshold: (precision, recall, f1)}).
    """
    if thresholds is None:
        thresholds = COCO_IOU_THRESHOLDS
    per_thresh = {}
    for t in thresholds:
        tp, fp, fn = greedy_match_boxes(gt_boxes, pred_boxes, iou_threshold=t)
        p, r, f1 = calc_prf(tp, fp, fn)
        per_thresh[t] = (p, r, f1)
    mean_f1 = float(np.mean([v[2] for v in per_thresh.values()]))
    return round(mean_f1, 4), per_thresh


# ── Statistical helpers ──────────────────────────────────────────────────

def bootstrap_ci(
    scores: List[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for a list of scores.
    Returns (mean, ci_lower, ci_upper).
    """
    scores = np.array(scores)
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(seed)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(scores, size=len(scores), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(np.mean(scores)), lo, hi


def paired_bootstrap_test(
    scores_a: List[float],
    scores_b: List[float],
    n_boot: int = 5000,
    seed: int = 42,
) -> float:
    """
    Paired bootstrap test: proportion of resamples where A > B.
    Returns p-value (two-sided). p < 0.05 = significant difference.
    """
    a = np.array(scores_a)
    b = np.array(scores_b)
    assert len(a) == len(b), "Score lists must be same length"
    if len(a) == 0:
        return 1.0
    observed_diff = np.mean(a) - np.mean(b)
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_boot):
        idx = rng.choice(len(a), size=len(a), replace=True)
        boot_diff = np.mean(a[idx]) - np.mean(b[idx])
        if boot_diff * observed_diff <= 0:
            count += 1
    return count / n_boot


# ── Response parsing ─────────────────────────────────────────────────────

def _strip_response(text: str) -> str:
    """Strip think tags and code blocks from model response."""
    txt = text.strip()
    if "</think>" in txt:
        txt = txt[txt.find("</think>") + 8:].strip()
    if "```json" in txt:
        txt = txt[txt.find("```json") + 7:txt.find("```", txt.find("```json") + 7)].strip()
    elif "```" in txt:
        txt = txt[txt.find("```") + 3:txt.find("```", txt.find("```") + 3)].strip()
    if not txt.startswith("["):
        m = re.search(r'\[.*\]', txt, re.DOTALL)
        if m:
            txt = m.group(0)
    return txt


def parse_ner_response(text: str) -> List[Dict]:
    """Parse Qwen3 NER JSON response into [{text, type, confidence}, ...]."""
    txt = _strip_response(text)
    try:
        return [
            {
                "text": e["text"],
                "type": e.get("type", "MISC"),
                "confidence": float(e.get("confidence", 1.0)),
            }
            for e in json.loads(txt)
            if isinstance(e, dict) and e.get("text", "").strip()
        ]
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        return []


def parse_seg_response(text: str) -> List[Dict]:
    """Parse VLM segmentation JSON response into [{type, box_2d}, ...]."""
    txt = _strip_response(text)
    try:
        data = json.loads(txt)
        regions = []
        for item in data:
            if not isinstance(item, dict):
                continue
            rtype = item.get("type", "")
            box = item.get("box") or item.get("box_2d") or item.get("bbox")
            if rtype in STRUCTURAL_TYPES and box and len(box) == 4:
                regions.append({"type": rtype, "box_2d": [float(c) for c in box]})
        return regions
    except (json.JSONDecodeError, TypeError):
        return []


# ── Image degradation ───────────────────────────────────────────────────

DEGRADATION_LEVELS = {
    "noise":   [5, 15, 30, 50, 80],
    "blur":    [1, 3, 5, 9, 15],
    "jpeg":    [80, 50, 30, 15, 5],
    "rotation": [1, 3, 5, 10, 15],
}


def degrade_image(
    image_path: str,
    mode: str,
    level: int,
    out_path: Optional[str] = None,
) -> str:
    """
    Apply degradation to an image.

    Args:
        mode: "noise", "blur", "jpeg", "rotation"
        level: 0-4 index into DEGRADATION_LEVELS (0=lightest, 4=heaviest)
        out_path: optional output path. If None, writes to temp file.

    Returns path to degraded image.
    """
    from PIL import Image, ImageFilter
    import tempfile

    img = Image.open(image_path).convert("RGB")
    param = DEGRADATION_LEVELS[mode][level]

    if mode == "noise":
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, param, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    elif mode == "blur":
        img = img.filter(ImageFilter.GaussianBlur(radius=param))

    elif mode == "jpeg":
        buf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(buf, format="JPEG", quality=param)
        buf.close()
        img = Image.open(buf.name)
        if out_path is None:
            return buf.name

    elif mode == "rotation":
        img = img.rotate(param, fillcolor=(255, 255, 255), expand=False)

    if out_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        out_path = tmp.name
        tmp.close()

    img.save(out_path, format="JPEG", quality=95)
    return out_path


# ── Reading order metrics ───────────────────────────────────────────────

def kendall_tau(order_a: List[int], order_b: List[int]) -> float:
    """
    Kendall's tau rank correlation coefficient.
    Returns value in [-1, 1]. 1 = perfect agreement.
    """
    n = len(order_a)
    if n < 2:
        return 1.0
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff_a = order_a[i] - order_a[j]
            diff_b = order_b[i] - order_b[j]
            if diff_a * diff_b > 0:
                concordant += 1
            elif diff_a * diff_b < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 1.0
    return (concordant - discordant) / total


def reading_order_score(
    gt_entities: List[Dict],
    ocr_text: str,
) -> Tuple[float, float]:
    """
    Measure how well OCR preserves the reading order of GT entities.

    Returns (tau, maintained_fraction):
        tau: Kendall's tau on entity positions found in OCR text
        maintained_fraction: fraction of consecutive GT pairs in correct order
    """
    if not gt_entities or not ocr_text.strip():
        return 0.0, 0.0

    ocr_lower = ocr_text.lower()
    positions = []
    for ent in gt_entities:
        norm = normalize(ent["text"])
        pos = ocr_lower.find(norm)
        positions.append(pos)

    # Filter to entities actually found in OCR
    found = [(i, pos) for i, pos in enumerate(positions) if pos >= 0]
    if len(found) < 2:
        return 1.0 if len(found) == 1 else 0.0, 1.0 if len(found) == 1 else 0.0

    gt_order = list(range(len(found)))
    ocr_order = [0] * len(found)
    sorted_by_pos = sorted(range(len(found)), key=lambda k: found[k][1])
    for rank, idx in enumerate(sorted_by_pos):
        ocr_order[idx] = rank

    tau = kendall_tau(gt_order, ocr_order)

    # Maintained fraction: consecutive pairs in correct relative order
    maintained = 0
    for k in range(len(found) - 1):
        if found[k][1] < found[k + 1][1]:
            maintained += 1
    frac = maintained / (len(found) - 1)

    return round(tau, 4), round(frac, 4)


# ── Table structure metrics ─────────────────────────────────────────────

def _cluster_1d(values: List[float], tolerance: float) -> List[List[int]]:
    """Cluster 1D values into groups within tolerance. Returns list of index groups."""
    if not values:
        return []
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    clusters = [[indexed[0][0]]]
    for i in range(1, len(indexed)):
        if indexed[i][1] - indexed[i - 1][1] <= tolerance:
            clusters[-1].append(indexed[i][0])
        else:
            clusters.append([indexed[i][0]])
    return clusters


def table_structure_score(
    gt_cells: List[List[float]],
    pred_cells: List[List[float]],
    row_tolerance: float = 15.0,
) -> Dict[str, float]:
    """
    Compare table grid structure from bounding boxes.

    Groups cells into rows by y-center, then evaluates:
    - row_count_acc: 1.0 if row counts match, else ratio
    - avg_col_acc: average column-count accuracy per matched row
    - cell_match_f1: IoU-based cell matching F1 at 0.5

    Returns dict with structure metrics.
    """
    if not gt_cells and not pred_cells:
        return {"row_count_acc": 1.0, "avg_col_acc": 1.0, "cell_match_f1": 1.0}
    if not gt_cells or not pred_cells:
        return {"row_count_acc": 0.0, "avg_col_acc": 0.0, "cell_match_f1": 0.0}

    # Cluster into rows by y-center
    gt_y_centers = [(b[1] + b[3]) / 2 for b in gt_cells]
    pred_y_centers = [(b[1] + b[3]) / 2 for b in pred_cells]

    gt_rows = _cluster_1d(gt_y_centers, row_tolerance)
    pred_rows = _cluster_1d(pred_y_centers, row_tolerance)

    n_gt_rows = len(gt_rows)
    n_pred_rows = len(pred_rows)

    # Row count accuracy
    row_acc = min(n_gt_rows, n_pred_rows) / max(n_gt_rows, n_pred_rows) if max(n_gt_rows, n_pred_rows) > 0 else 1.0

    # Match rows by y-center proximity
    gt_row_y = [np.mean([gt_y_centers[i] for i in row]) for row in gt_rows]
    pred_row_y = [np.mean([pred_y_centers[i] for i in row]) for row in pred_rows]

    col_accs = []
    pred_row_used = [False] * len(pred_rows)
    for gi, gy in enumerate(gt_row_y):
        best_pi, best_dist = -1, float("inf")
        for pi, py in enumerate(pred_row_y):
            if pred_row_used[pi]:
                continue
            dist = abs(gy - py)
            if dist < best_dist:
                best_dist = dist
                best_pi = pi
        if best_pi >= 0 and best_dist < row_tolerance * 2:
            pred_row_used[best_pi] = True
            n_gt_cols = len(gt_rows[gi])
            n_pred_cols = len(pred_rows[best_pi])
            col_accs.append(min(n_gt_cols, n_pred_cols) / max(n_gt_cols, n_pred_cols))
        else:
            col_accs.append(0.0)

    avg_col_acc = float(np.mean(col_accs)) if col_accs else 0.0

    # Cell-level F1
    tp, fp, fn = greedy_match_boxes(gt_cells, pred_cells, iou_threshold=0.5)
    _, _, f1 = calc_prf(tp, fp, fn)

    return {
        "row_count_acc": round(row_acc, 4),
        "avg_col_acc": round(avg_col_acc, 4),
        "cell_match_f1": f1,
    }


# ── Inference timing ────────────────────────────────────────────────────

class InferenceTimer:
    """
    Context manager for measuring inference time and GPU memory.

    Usage:
        with InferenceTimer("model_name") as timer:
            run_model(pages)
        print(timer.result)
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.result = {}

    def __enter__(self):
        import time
        self._start = time.perf_counter()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                self._has_cuda = True
            else:
                self._has_cuda = False
        except ImportError:
            self._has_cuda = False
        return self

    def __exit__(self, *exc):
        import time
        elapsed = time.perf_counter() - self._start
        self.result = {
            "name": self.name,
            "time_s": round(elapsed, 2),
        }
        if self._has_cuda:
            import torch
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            self.result["gpu_peak_mb"] = round(peak_mb, 1)
        return False
