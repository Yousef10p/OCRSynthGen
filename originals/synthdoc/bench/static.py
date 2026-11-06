"""
Static evaluation — no GPU needed.
Dataset stats, OCR quality, NER scoring, error attribution,
per-type analysis, confusion matrix, segmentation scoring,
bootstrap CI, per-document aggregation.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .helpers import (
    NER_ENTITY_TYPES, FULL_ENTITY_TYPES, PRED_TYPE_MAP, STRUCTURAL_TYPES,
    normalize, match_entities, detailed_match, calc_prf,
    greedy_match_boxes, compute_coco_map,
    compute_wer, compute_cer,
    bootstrap_ci, paired_bootstrap_test,
)


# ── Dataset statistics ───────────────────────────────────────────────────

def dataset_stats(pages: List[Dict]) -> Dict:
    """Compute summary statistics from loaded NER pages."""
    doc_ids = set(p["doc_id"] for p in pages)
    templates = set(p["template_type"] for p in pages)
    entities_per_page = [len(p["gt_entities"]) for p in pages]

    pages_per_doc = defaultdict(int)
    for p in pages:
        pages_per_doc[p["doc_id"]] += 1

    type_counts = defaultdict(int)
    for p in pages:
        for e in p["gt_entities"]:
            type_counts[e["type"]] += 1

    tmpl_type_counts = defaultdict(lambda: defaultdict(int))
    for p in pages:
        for e in p["gt_entities"]:
            tmpl_type_counts[p["template_type"]][e["type"]] += 1

    return {
        "n_docs": len(doc_ids),
        "n_templates": len(templates),
        "n_pages": len(pages),
        "pages_per_doc": dict(pages_per_doc),
        "entities_per_page": entities_per_page,
        "type_counts": dict(type_counts),
        "tmpl_type_counts": {k: dict(v) for k, v in tmpl_type_counts.items()},
    }


# ── OCR quality ──────────────────────────────────────────────────────────

def evaluate_ocr_quality(
    pages: List[Dict],
    ocr_keys: List[Tuple[str, str]],
) -> pd.DataFrame:
    """
    Compare OCR output to GT text using WER and CER.

    ocr_keys: list of (display_name, ocr_key) tuples,
        e.g. [("qwen-vl", "qwenvl_ocr"), ("deepseek", "deepseek_ocr")]

    Returns DataFrame with one row per (page, ocr_engine).
    """
    rows = []
    for page in pages:
        gt_text = page.get("gt_text", "")
        if not gt_text.strip():
            continue
        for ocr_name, ocr_key in ocr_keys:
            ocr_text = page.get(ocr_key, "")
            if not ocr_text.strip():
                wer, cer = 1.0, 1.0
            else:
                wer = compute_wer(gt_text, ocr_text)
                cer = compute_cer(gt_text, ocr_text)
            rows.append({
                "page": page["stem"],
                "template_type": page["template_type"],
                "ocr": ocr_name,
                "wer": round(wer, 4),
                "cer": round(cer, 4),
                "gt_words": len(gt_text.split()),
                "ocr_words": len(ocr_text.split()),
            })
    return pd.DataFrame(rows)


# ── NER evaluation ──────────────────────────────────────────────────────

def evaluate_ner(pages: List[Dict], ocr_configs: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Score NER predictions against ground truth.

    ocr_configs: list of (display_name, ner_key) tuples,
        e.g. [("qwen-vl", "ner_qwenvl"), ("deepseek", "ner_deepseek")]

    Returns a DataFrame with one row per (page, ocr).
    """
    rows = []
    for page in pages:
        for ocr_name, ner_key in ocr_configs:
            pred = page.get(ner_key, [])
            s_tp, s_fp, s_fn = match_entities(page["gt_entities"], pred, True)
            r_tp, r_fp, r_fn = match_entities(page["gt_entities"], pred, False)
            s_p, s_r, s_f1 = calc_prf(s_tp, s_fp, s_fn)
            r_p, r_r, r_f1 = calc_prf(r_tp, r_fp, r_fn)
            rows.append({
                "page": page["stem"],
                "doc_id": page["doc_id"],
                "template_type": page["template_type"],
                "ocr": ocr_name,
                "gt_count": len(page["gt_entities"]),
                "pred_count": len(pred),
                "strict_precision": s_p, "strict_recall": s_r, "strict_f1": s_f1,
                "relaxed_precision": r_p, "relaxed_recall": r_r, "relaxed_f1": r_f1,
            })
    return pd.DataFrame(rows)


# ── Per-entity match results ─────────────────────────────────────────────

def per_entity_matches(
    pages: List[Dict],
    ocr_configs: List[Tuple[str, str]],
) -> pd.DataFrame:
    """
    Per-entity match results: for each GT entity, what the model returned.

    Returns a DataFrame with one row per GT entity per model, columns:
        model, page, template, entity_type, gt_entity, model_result, pred_text, pred_type
    """
    rows = []
    for page in pages:
        for ocr_name, ner_key in ocr_configs:
            pred = page.get(ner_key, [])
            matched_pairs, unmatched_gt, unmatched_pred = detailed_match(
                page["gt_entities"], pred
            )

            # Matched GT entities
            for gt_type, pred_type, gt_text, pred_text in matched_pairs:
                if gt_type == pred_type:
                    result = "hit"
                else:
                    result = "type_mismatch"
                rows.append({
                    "model": ocr_name,
                    "page": page["stem"],
                    "template": page["template_type"],
                    "entity_type": gt_type,
                    "gt_entity": gt_text,
                    "model_result": result,
                    "pred_text": pred_text,
                    "pred_type": pred_type,
                })

            # Unmatched GT entities (misses)
            for gt_type, gt_text in unmatched_gt:
                rows.append({
                    "model": ocr_name,
                    "page": page["stem"],
                    "template": page["template_type"],
                    "entity_type": gt_type,
                    "gt_entity": gt_text,
                    "model_result": "miss",
                    "pred_text": "",
                    "pred_type": "",
                })

    return pd.DataFrame(rows)


# ── OCR → NER error attribution ──────────────────────────────────────────

def ocr_ner_error_attribution(
    pages: List[Dict],
    ocr_configs: List[Tuple[str, str, str]],
) -> pd.DataFrame:
    """
    For each missed GT entity, determine if OCR or NER is at fault.

    ocr_configs: list of (display_name, ocr_text_key, ner_key) tuples,
        e.g. [("qwen-vl", "qwenvl_ocr", "ner_qwenvl")]

    Logic: if GT entity text is absent from OCR output → "OCR miss".
    If present in OCR but NER didn't extract it → "NER miss".

    Returns DataFrame with columns: ocr, entity_type, total_gt, found,
        ocr_miss, ner_miss, ocr_miss_pct, ner_miss_pct.
    """
    agg = defaultdict(lambda: {"ocr_miss": 0, "ner_miss": 0, "found": 0})

    for page in pages:
        for ocr_name, ocr_key, ner_key in ocr_configs:
            ocr_text = normalize(page.get(ocr_key, ""))
            pred = page.get(ner_key, [])
            pred_norm = {normalize(e["text"]) for e in pred}

            for gt_ent in page["gt_entities"]:
                gt_norm = normalize(gt_ent["text"])
                key = (ocr_name, gt_ent["type"])

                in_ocr = gt_norm in ocr_text

                if not in_ocr:
                    agg[key]["ocr_miss"] += 1
                elif gt_norm not in pred_norm:
                    agg[key]["ner_miss"] += 1
                else:
                    agg[key]["found"] += 1

    rows = []
    for (ocr_name, etype), counts in sorted(agg.items()):
        total = counts["ocr_miss"] + counts["ner_miss"] + counts["found"]
        rows.append({
            "ocr": ocr_name,
            "entity_type": etype,
            "total_gt": total,
            "found": counts["found"],
            "ocr_miss": counts["ocr_miss"],
            "ner_miss": counts["ner_miss"],
            "ocr_miss_pct": round(counts["ocr_miss"] / total, 4) if total else 0,
            "ner_miss_pct": round(counts["ner_miss"] / total, 4) if total else 0,
        })
    return pd.DataFrame(rows)


# ── Per-entity-type analysis + confusion ─────────────────────────────────

def per_type_analysis(
    pages: List[Dict],
    ocr_configs: List[Tuple[str, str]],
    entity_types: Optional[set] = None,
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Compute per-entity-type P/R/F1 and confusion matrices.

    Args:
        entity_types: Set of entity types to evaluate. Default: NER_ENTITY_TYPES (7).
            Use FULL_ENTITY_TYPES for all 14 types.

    Returns:
        type_df:  DataFrame with columns [ocr, entity_type, TP, FP, FN, P, R, F1]
        confusion: {ocr_name: {(gt_type, pred_type): count}}
        mismatches: {ocr_name: [(gt_text, pred_text, gt_type, pred_type), ...]}
    """
    if entity_types is None:
        entity_types = NER_ENTITY_TYPES

    per_type = {}
    confusion = {}
    mismatches = {}

    for ocr_name, ner_key in ocr_configs:
        confusion[ocr_name] = defaultdict(int)
        mismatches[ocr_name] = []
        for etype in entity_types:
            per_type[(ocr_name, etype)] = {"tp": 0, "fp": 0, "fn": 0}

        for page in pages:
            pred = page.get(ner_key, [])
            matched_pairs, unmatched_gt, unmatched_pred = detailed_match(
                page["gt_entities"], pred
            )

            for gt_type, pred_type, gt_text, pred_text in matched_pairs:
                confusion[ocr_name][(gt_type, pred_type)] += 1
                if gt_type == pred_type:
                    per_type[(ocr_name, gt_type)]["tp"] += 1
                else:
                    per_type[(ocr_name, gt_type)]["fn"] += 1
                    if pred_type in entity_types:
                        per_type[(ocr_name, pred_type)]["fp"] += 1
                    mismatches[ocr_name].append((gt_text, pred_text, gt_type, pred_type))

            for gt_type, gt_text in unmatched_gt:
                per_type[(ocr_name, gt_type)]["fn"] += 1

            for pred_type, pred_text in unmatched_pred:
                if pred_type in entity_types:
                    per_type[(ocr_name, pred_type)]["fp"] += 1

    rows = []
    for ocr_name, _ in ocr_configs:
        for etype in sorted(entity_types):
            s = per_type[(ocr_name, etype)]
            p, r, f1 = calc_prf(s["tp"], s["fp"], s["fn"])
            rows.append({
                "ocr": ocr_name, "entity_type": etype,
                "TP": s["tp"], "FP": s["fp"], "FN": s["fn"],
                "P": p, "R": r, "F1": f1,
            })

    return pd.DataFrame(rows), dict(confusion), dict(mismatches)


# ── NER confidence & calibration ─────────────────────────────────────────

def calibration_analysis(
    pages: List[Dict],
    ocr_configs: List[Tuple[str, str]],
    n_bins: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    Compute calibration bins: group predictions by confidence,
    measure accuracy (correct entity?) per bin.

    Returns {ocr_name: DataFrame with columns [bin_center, count, accuracy, avg_confidence]}.
    """
    result = {}
    for ocr_name, ner_key in ocr_configs:
        confidences = []
        correct = []

        for page in pages:
            pred = page.get(ner_key, [])
            gt_norm = {normalize(e["text"]): e["type"] for e in page["gt_entities"]}

            for ent in pred:
                pred_norm = normalize(ent["text"])
                pred_type = PRED_TYPE_MAP.get(ent["type"], ent["type"])
                conf = ent.get("confidence", 1.0)

                is_correct = (pred_norm in gt_norm and gt_norm[pred_norm] == pred_type)
                confidences.append(conf)
                correct.append(int(is_correct))

        if not confidences:
            result[ocr_name] = pd.DataFrame()
            continue

        confidences = np.array(confidences)
        correct = np.array(correct)

        bins = np.linspace(0, 1, n_bins + 1)
        rows = []
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (confidences >= lo) & (confidences < hi + (1e-9 if i == n_bins - 1 else 0))
            if mask.sum() == 0:
                continue
            rows.append({
                "bin_center": round((lo + hi) / 2, 2),
                "count": int(mask.sum()),
                "accuracy": round(float(correct[mask].mean()), 4),
                "avg_confidence": round(float(confidences[mask].mean()), 4),
            })
        result[ocr_name] = pd.DataFrame(rows)

    return result


# ── Statistical significance ─────────────────────────────────────────────

def compute_significance(
    df: pd.DataFrame,
    ocr_a: str = "qwen-vl",
    ocr_b: str = "deepseek",
    metric: str = "strict_f1",
) -> Dict:
    """
    Bootstrap CI for each OCR engine + paired significance test.
    df must have columns: page, ocr, <metric>.
    """
    pages_a = df[df["ocr"] == ocr_a].sort_values("page")
    pages_b = df[df["ocr"] == ocr_b].sort_values("page")

    scores_a = pages_a[metric].values.tolist()
    scores_b = pages_b[metric].values.tolist()

    mean_a, lo_a, hi_a = bootstrap_ci(scores_a)
    mean_b, lo_b, hi_b = bootstrap_ci(scores_b)

    common_pages = set(pages_a["page"]) & set(pages_b["page"])
    paired_a = pages_a[pages_a["page"].isin(common_pages)].sort_values("page")[metric].values
    paired_b = pages_b[pages_b["page"].isin(common_pages)].sort_values("page")[metric].values
    p_val = paired_bootstrap_test(paired_a.tolist(), paired_b.tolist())

    return {
        ocr_a: {"mean": mean_a, "ci_lo": lo_a, "ci_hi": hi_a, "n": len(scores_a)},
        ocr_b: {"mean": mean_b, "ci_lo": lo_b, "ci_hi": hi_b, "n": len(scores_b)},
        "p_value": p_val,
        "significant": p_val < 0.05,
    }


# ── Per-document aggregation ─────────────────────────────────────────────

def evaluate_ner_per_doc(
    pages: List[Dict],
    ocr_configs: List[Tuple[str, str]],
) -> pd.DataFrame:
    """
    Aggregate NER evaluation at document level (merge all pages of a doc).
    Returns DataFrame with one row per (doc_id, ocr).
    """
    doc_pages = defaultdict(list)
    for p in pages:
        doc_pages[p["doc_id"]].append(p)

    rows = []
    for doc_id, doc_page_list in sorted(doc_pages.items()):
        template_type = doc_page_list[0]["template_type"]
        n_pages = len(doc_page_list)

        for ocr_name, ner_key in ocr_configs:
            all_gt = []
            all_pred = []
            for page in doc_page_list:
                all_gt.extend(page["gt_entities"])
                all_pred.extend(page.get(ner_key, []))

            s_tp, s_fp, s_fn = match_entities(all_gt, all_pred, True)
            r_tp, r_fp, r_fn = match_entities(all_gt, all_pred, False)
            s_p, s_r, s_f1 = calc_prf(s_tp, s_fp, s_fn)
            r_p, r_r, r_f1 = calc_prf(r_tp, r_fp, r_fn)

            rows.append({
                "doc_id": doc_id,
                "template_type": template_type,
                "n_pages": n_pages,
                "ocr": ocr_name,
                "gt_count": len(all_gt),
                "pred_count": len(all_pred),
                "strict_precision": s_p, "strict_recall": s_r, "strict_f1": s_f1,
                "relaxed_precision": r_p, "relaxed_recall": r_r, "relaxed_f1": r_f1,
            })

    return pd.DataFrame(rows)


# ── Segmentation evaluation ─────────────────────────────────────────────

def evaluate_seg(
    seg_pages: List[Dict],
    model_configs: List[Tuple[str, str]],
) -> pd.DataFrame:
    """
    Score segmentation predictions against ground truth.
    Includes F1@0.5 and COCO mAP@[0.5:0.95].

    Returns DataFrame with one row per (page, model).
    """
    rows = []
    for page in seg_pages:
        for model_name, key in model_configs:
            pred = page.get(key, [])
            total_tp, total_fp, total_fn = 0, 0, 0

            all_gt_boxes = [r["box_2d"] for r in page["gt_regions"]]
            all_pred_boxes = [r["box_2d"] for r in pred]

            for stype in STRUCTURAL_TYPES:
                gt_boxes = [r["box_2d"] for r in page["gt_regions"] if r["type"] == stype]
                pred_boxes = [r["box_2d"] for r in pred if r["type"] == stype]
                tp, fp, fn = greedy_match_boxes(gt_boxes, pred_boxes)
                total_tp += tp
                total_fp += fp
                total_fn += fn

            p, r, f1 = calc_prf(total_tp, total_fp, total_fn)
            map_score, _ = compute_coco_map(all_gt_boxes, all_pred_boxes)

            rows.append({
                "page": page["stem"],
                "template_type": page["template_type"],
                "model": model_name,
                "gt_count": len(page["gt_regions"]),
                "pred_count": len(pred),
                "precision": p, "recall": r, "f1": f1,
                "mAP": map_score,
            })
    return pd.DataFrame(rows)


# ── Robustness / degradation evaluation ─────────────────────────────────

def evaluate_robustness(
    pages: List[Dict],
    ocr_fn,
    ocr_key: str,
    ocr_name: str = "model",
    modes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Run OCR on progressively degraded images, measure WER/CER at each level.

    Args:
        pages: list of page dicts (must have gt_text and image_path)
        ocr_fn: callable(pages) that runs OCR and writes ocr_key in-place
        ocr_key: key where OCR output is stored (e.g. "qwenvl_ocr")
        modes: degradation modes to test (default: all)

    Returns DataFrame with columns: mode, level, param, mean_wer, mean_cer.
    """
    from .helpers import DEGRADATION_LEVELS, degrade_image, compute_wer, compute_cer
    import copy
    from pathlib import Path

    if modes is None:
        modes = list(DEGRADATION_LEVELS.keys())

    rows = []

    for mode in modes:
        for level_idx in range(5):
            param = DEGRADATION_LEVELS[mode][level_idx]

            # Create degraded copies
            degraded_pages = []
            temp_paths = []
            for page in pages:
                deg_path = degrade_image(page["image_path"], mode, level_idx)
                temp_paths.append(deg_path)
                p_copy = copy.copy(page)
                p_copy["image_path"] = deg_path
                p_copy.pop(ocr_key, None)
                degraded_pages.append(p_copy)

            # Run OCR
            ocr_fn(degraded_pages)

            # Measure quality
            wers, cers = [], []
            for p_orig, p_deg in zip(pages, degraded_pages):
                gt_text = p_orig.get("gt_text", "")
                ocr_text = p_deg.get(ocr_key, "")
                if gt_text.strip():
                    wers.append(compute_wer(gt_text, ocr_text))
                    cers.append(compute_cer(gt_text, ocr_text))

            # Cleanup temp files
            for tp in temp_paths:
                Path(tp).unlink(missing_ok=True)

            rows.append({
                "ocr": ocr_name,
                "mode": mode,
                "level": level_idx,
                "param": param,
                "mean_wer": round(float(np.mean(wers)), 4) if wers else 0.0,
                "mean_cer": round(float(np.mean(cers)), 4) if cers else 0.0,
                "n_pages": len(wers),
            })
            print(f"  Robustness {ocr_name} {mode} L{level_idx} (param={param}): "
                  f"WER={rows[-1]['mean_wer']:.3f} CER={rows[-1]['mean_cer']:.3f}")

    return pd.DataFrame(rows)


# ── Reading order evaluation ────────────────────────────────────────────

def evaluate_reading_order(
    pages: List[Dict],
    ocr_keys: List[Tuple[str, str]],
) -> pd.DataFrame:
    """
    Evaluate how well OCR output preserves entity reading order.

    ocr_keys: list of (display_name, ocr_key) tuples.

    Returns DataFrame with one row per (page, ocr) containing
    tau (Kendall's tau) and maintained_frac.
    """
    from .helpers import reading_order_score

    rows = []
    for page in pages:
        for ocr_name, ocr_key in ocr_keys:
            ocr_text = page.get(ocr_key, "")
            tau, frac = reading_order_score(page["gt_entities"], ocr_text)
            rows.append({
                "page": page["stem"],
                "template_type": page["template_type"],
                "ocr": ocr_name,
                "tau": tau,
                "maintained_frac": frac,
                "n_entities": len(page["gt_entities"]),
            })
    return pd.DataFrame(rows)


# ── Cross-template generalization ───────────────────────────────────────

def cross_template_split(
    pages: List[Dict],
    holdout_templates: Optional[List[str]] = None,
    holdout_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[str]]:
    """
    Split pages into train/test by template type for cross-template evaluation.

    If holdout_templates is given, those exact templates become the test set.
    Otherwise, randomly select holdout_ratio of templates.

    Returns (train_pages, test_pages, holdout_template_names).
    """
    templates = sorted(set(p["template_type"] for p in pages))

    if holdout_templates is None:
        rng = np.random.RandomState(seed)
        n_holdout = max(1, int(len(templates) * holdout_ratio))
        holdout_templates = sorted(rng.choice(templates, size=n_holdout, replace=False).tolist())

    holdout_set = set(holdout_templates)
    train = [p for p in pages if p["template_type"] not in holdout_set]
    test = [p for p in pages if p["template_type"] in holdout_set]

    return train, test, list(holdout_templates)


def evaluate_cross_template(
    pages: List[Dict],
    ocr_configs: List[Tuple[str, str]],
    holdout_templates: Optional[List[str]] = None,
    holdout_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Evaluate NER on held-out templates vs seen templates.

    Returns (seen_df, holdout_df, holdout_template_names).
    Both DataFrames have same schema as evaluate_ner().
    """
    train_pages, test_pages, holdout = cross_template_split(
        pages, holdout_templates, holdout_ratio
    )

    seen_df = evaluate_ner(train_pages, ocr_configs) if train_pages else pd.DataFrame()
    holdout_df = evaluate_ner(test_pages, ocr_configs) if test_pages else pd.DataFrame()

    if not seen_df.empty:
        seen_df["split"] = "seen"
    if not holdout_df.empty:
        holdout_df["split"] = "holdout"

    return seen_df, holdout_df, holdout


# ── Table structure evaluation ──────────────────────────────────────────

def evaluate_table_structure(
    seg_pages: List[Dict],
    model_configs: List[Tuple[str, str]],
) -> pd.DataFrame:
    """
    Evaluate table structure detection (row/column accuracy) per page.

    Uses TABLE_CELL regions from GT and predictions.
    Returns DataFrame with structure metrics per (page, model).
    """
    from .helpers import table_structure_score

    rows = []
    for page in seg_pages:
        gt_cells = [r["box_2d"] for r in page["gt_regions"] if r["type"] == "TABLE_CELL"]
        if not gt_cells:
            continue

        for model_name, pred_key in model_configs:
            pred = page.get(pred_key, [])
            pred_cells = [r["box_2d"] for r in pred if r["type"] == "TABLE_CELL"]

            scores = table_structure_score(gt_cells, pred_cells)
            rows.append({
                "page": page["stem"],
                "template_type": page["template_type"],
                "model": model_name,
                "gt_cells": len(gt_cells),
                "pred_cells": len(pred_cells),
                **scores,
            })

    return pd.DataFrame(rows)
