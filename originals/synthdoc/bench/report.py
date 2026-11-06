"""
All visualization / charting functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple

from .helpers import NER_ENTITY_TYPES, COCO_IOU_THRESHOLDS, calc_prf

COLORS = {"qwen-vl": "#2196F3", "deepseek": "#FF9800"}


# ── Dataset stats charts ─────────────────────────────────────────────────

def plot_entity_distribution(type_counts: Dict[str, int]) -> None:
    """Horizontal bar chart of GT entity counts by type."""
    order = sorted(type_counts.keys(), key=lambda t: -type_counts[t])
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(range(len(order)), [type_counts[t] for t in order], color="#3498db")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    for i, t in enumerate(order):
        ax.text(type_counts[t] + 5, i, str(type_counts[t]), va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Ground Truth Entity Distribution by Type")
    plt.tight_layout()
    plt.show()


def plot_template_type_heatmap(tmpl_type_counts: Dict[str, Dict[str, int]]) -> None:
    """Heatmap of entity counts by template x type."""
    tmpl_df = pd.DataFrame(tmpl_type_counts).T.fillna(0).astype(int)
    tmpl_df = tmpl_df[sorted(tmpl_df.columns)]
    tmpl_df["total"] = tmpl_df.sum(axis=1)
    tmpl_df = tmpl_df.sort_values("total", ascending=False)
    plot_df = tmpl_df.drop(columns="total")

    fig, ax = plt.subplots(figsize=(8, max(6, len(plot_df) * 0.3)))
    im = ax.imshow(plot_df.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(plot_df.columns)))
    ax.set_xticklabels(plot_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index, fontsize=7)
    for i in range(len(plot_df.index)):
        for j in range(len(plot_df.columns)):
            v = int(plot_df.iloc[i, j])
            if v > 0:
                color = "white" if v > plot_df.values.max() * 0.6 else "black"
                ax.text(j, i, str(v), ha="center", va="center", fontsize=5, color=color)
    fig.colorbar(im, ax=ax, shrink=0.3)
    ax.set_title("Entity Counts by Template x Type")
    plt.tight_layout()
    plt.show()


def plot_pages_per_doc(pages_per_doc: Dict[str, int]) -> None:
    """Histogram of pages per document."""
    ppd = list(pages_per_doc.values())
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(ppd, bins=range(1, max(ppd) + 2), color="#2ecc71", edgecolor="white", align="left")
    ax.set_xlabel("Pages per Document")
    ax.set_ylabel("Documents")
    ax.set_title("Document Length Distribution")
    plt.tight_layout()
    plt.show()


# ── OCR quality charts ──────────────────────────────────────────────────

def plot_ocr_quality(ocr_df: pd.DataFrame) -> None:
    """
    WER and CER comparison across OCR engines.
    Two subplots: (1) bar chart of mean WER/CER, (2) box plot per template.
    """
    engines = ocr_df["ocr"].unique()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Bar chart of means
    means = ocr_df.groupby("ocr")[["wer", "cer"]].mean()
    x = np.arange(len(means))
    w = 0.3
    for i, (ocr_name, row) in enumerate(means.iterrows()):
        c = COLORS.get(ocr_name, "#999")
        ax1.bar(i - w / 2, row["wer"], w, color=c, label="WER" if i == 0 else "")
        ax1.bar(i + w / 2, row["cer"], w, color=c, alpha=0.5, label="CER" if i == 0 else "")
        ax1.text(i - w / 2, row["wer"] + 0.01, f"{row['wer']:.3f}", ha="center", fontsize=9)
        ax1.text(i + w / 2, row["cer"] + 0.01, f"{row['cer']:.3f}", ha="center", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([o.upper() for o in means.index])
    ax1.set_ylabel("Error Rate")
    ax1.set_title("Mean OCR Error Rate")
    ax1.legend()

    # Box plot of WER by engine
    data = [ocr_df[ocr_df["ocr"] == e]["wer"].values for e in engines]
    bp = ax2.boxplot(data, labels=[e.upper() for e in engines], patch_artist=True)
    for patch, eng in zip(bp["boxes"], engines):
        patch.set_facecolor(COLORS.get(eng, "#ccc"))
        patch.set_alpha(0.6)
    ax2.set_ylabel("WER")
    ax2.set_title("WER Distribution by OCR Engine")

    plt.tight_layout()
    plt.show()


def plot_ocr_quality_by_template(ocr_df: pd.DataFrame) -> None:
    """Heatmap of mean WER per template x OCR engine."""
    pivot = ocr_df.pivot_table(index="template_type", columns="ocr", values="wer")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(6, max(6, len(pivot) * 0.25)))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.upper() for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.3)
    ax.set_title("WER by Template and OCR Engine (lower = better)")
    plt.tight_layout()
    plt.show()


# ── Error attribution chart ─────────────────────────────────────────────

def plot_error_attribution(attr_df: pd.DataFrame) -> None:
    """
    Stacked bar chart: for each entity type, show OCR miss vs NER miss proportions.
    One group of bars per OCR engine.
    """
    engines = attr_df["ocr"].unique()
    etypes = sorted(attr_df["entity_type"].unique())

    fig, axes = plt.subplots(1, len(engines), figsize=(7 * len(engines), 5), squeeze=False)
    for ax, eng in zip(axes[0], engines):
        sub = attr_df[attr_df["ocr"] == eng].set_index("entity_type")
        sub = sub.reindex(etypes).fillna(0)

        x = np.arange(len(etypes))
        w = 0.5
        ax.bar(x, sub["found"], w, label="Found", color="#2ecc71")
        ax.bar(x, sub["ner_miss"], w, bottom=sub["found"], label="NER Miss", color="#e74c3c")
        ax.bar(x, sub["ocr_miss"], w,
               bottom=sub["found"] + sub["ner_miss"], label="OCR Miss", color="#3498db")

        ax.set_xticks(x)
        ax.set_xticklabels(etypes, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Entity Count")
        ax.set_title(f"Error Attribution: {eng.upper()}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_error_attribution_pct(attr_df: pd.DataFrame) -> None:
    """Grouped bar: OCR miss % vs NER miss % per entity type, side by side per engine."""
    engines = attr_df["ocr"].unique()
    etypes = sorted(attr_df["entity_type"].unique())
    n_eng = len(engines)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(etypes))
    total_w = 0.7
    w = total_w / n_eng

    for i, eng in enumerate(engines):
        sub = attr_df[attr_df["ocr"] == eng].set_index("entity_type").reindex(etypes).fillna(0)
        offset = (i - (n_eng - 1) / 2) * w
        c = COLORS.get(eng, "#999")
        ax.bar(x + offset, sub["ocr_miss_pct"] * 100, w * 0.45, color=c,
               label=f"{eng.upper()} OCR miss")
        ax.bar(x + offset + w * 0.45, sub["ner_miss_pct"] * 100, w * 0.45, color=c,
               alpha=0.5, label=f"{eng.upper()} NER miss")

    ax.set_xticks(x)
    ax.set_xticklabels(etypes, rotation=45, ha="right")
    ax.set_ylabel("Miss Rate (%)")
    ax.set_title("Error Attribution: OCR vs NER Miss Rate by Entity Type")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.show()


# ── Calibration / reliability diagram ───────────────────────────────────

def plot_calibration_diagram(calibration_data: Dict[str, pd.DataFrame]) -> None:
    """
    Reliability diagram: accuracy vs confidence for each OCR pipeline.
    calibration_data: {ocr_name: DataFrame with bin_center, accuracy, count, avg_confidence}.
    """
    n = len(calibration_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)

    for ax, (ocr_name, cal_df) in zip(axes[0], calibration_data.items()):
        if cal_df.empty:
            ax.set_title(f"{ocr_name.upper()} — no confidence data")
            continue

        bins = cal_df["bin_center"].values
        acc = cal_df["accuracy"].values
        counts = cal_df["count"].values

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")

        # Bars for accuracy
        bar_width = 0.08
        bars = ax.bar(bins, acc, bar_width, color=COLORS.get(ocr_name, "#999"),
                       alpha=0.7, label="Accuracy")

        # Annotate counts
        for b, a, c in zip(bins, acc, counts):
            if c > 0:
                ax.text(b, a + 0.02, str(c), ha="center", fontsize=7, color="gray")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Mean Confidence")
        ax.set_ylabel("Fraction Correct")
        ax.set_title(f"Calibration: Qwen3 + {ocr_name.upper()}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# ── Bootstrap CI chart ──────────────────────────────────────────────────

def plot_bootstrap_ci(sig_result: Dict, metric: str = "strict_f1") -> None:
    """
    Bar chart with error bars showing F1 and 95% CI for each OCR engine.
    sig_result: output of compute_significance().
    """
    engines = [k for k in sig_result if k not in ("p_value", "significant")]
    means = [sig_result[e]["mean"] for e in engines]
    ci_lo = [sig_result[e]["ci_lo"] for e in engines]
    ci_hi = [sig_result[e]["ci_hi"] for e in engines]
    errors = [[m - lo for m, lo in zip(means, ci_lo)],
              [hi - m for m, hi in zip(means, ci_hi)]]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(engines))
    colors = [COLORS.get(e, "#999") for e in engines]
    ax.bar(x, means, color=colors, yerr=errors, capsize=8, error_kw={"lw": 2})

    for i, e in enumerate(engines):
        n = sig_result[e]["n"]
        ax.text(i, means[i] + errors[1][i] + 0.01,
                f"{means[i]:.3f}\n[{ci_lo[i]:.3f}, {ci_hi[i]:.3f}]",
                ha="center", fontsize=8)

    p_val = sig_result.get("p_value", None)
    sig = sig_result.get("significant", False)
    if p_val is not None:
        label = f"p = {p_val:.4f} ({'significant' if sig else 'not significant'})"
        ax.text(0.5, 0.02, label, transform=ax.transAxes, ha="center", fontsize=9,
                style="italic", color="green" if sig else "gray")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Qwen3 + {e.upper()}" for e in engines])
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} with 95% Bootstrap CI")
    plt.tight_layout()
    plt.show()


# ── Per-document results ────────────────────────────────────────────────

def plot_per_doc_f1_distribution(doc_df: pd.DataFrame) -> None:
    """Overlapping histograms of document-level strict F1 per OCR engine."""
    engines = doc_df["ocr"].unique()
    fig, ax = plt.subplots(figsize=(8, 4))
    for eng in engines:
        scores = doc_df[doc_df["ocr"] == eng]["strict_f1"].values
        ax.hist(scores, bins=20, alpha=0.5, color=COLORS.get(eng, "#999"),
                label=f"{eng.upper()} (mean={np.mean(scores):.3f})", edgecolor="white")
    ax.set_xlabel("Strict F1")
    ax.set_ylabel("Documents")
    ax.set_title("Document-Level F1 Distribution")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_per_doc_pages_vs_f1(doc_df: pd.DataFrame) -> None:
    """Scatter: number of pages vs document F1 with trendline."""
    engines = doc_df["ocr"].unique()
    fig, axes = plt.subplots(1, len(engines), figsize=(6 * len(engines), 4), squeeze=False)

    for ax, eng in zip(axes[0], engines):
        sub = doc_df[doc_df["ocr"] == eng]
        ax.scatter(sub["n_pages"], sub["strict_f1"], alpha=0.5, s=30,
                   c=COLORS.get(eng, "#999"))

        # Trendline
        if len(sub) > 2:
            z = np.polyfit(sub["n_pages"], sub["strict_f1"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(sub["n_pages"].min(), sub["n_pages"].max(), 50)
            ax.plot(x_line, p(x_line), "--", color="red", alpha=0.5)

        ax.set_xlabel("Pages per Document")
        ax.set_ylabel("Strict F1")
        ax.set_title(f"Qwen3 + {eng.upper()}")

    plt.suptitle("Document Length vs F1", y=1.02)
    plt.tight_layout()
    plt.show()


# ── NER result charts ────────────────────────────────────────────────────

def plot_ner_leaderboard(df: pd.DataFrame) -> None:
    """Side-by-side strict vs relaxed F1 bar chart."""
    lb = df.groupby("ocr")[["strict_f1", "relaxed_f1"]].mean().sort_values("strict_f1", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(lb))
    for i, (ocr, row) in enumerate(lb.iterrows()):
        c = COLORS.get(ocr, "#999")
        ax.bar(i - 0.15, row["strict_f1"], 0.3, color=c, label="Strict" if i == 0 else "")
        ax.bar(i + 0.15, row["relaxed_f1"], 0.3, color=c, alpha=0.4, label="Relaxed" if i == 0 else "")
        ax.text(i - 0.15, row["strict_f1"] + 0.01, f"{row['strict_f1']:.3f}", ha="center", fontsize=9)
        ax.text(i + 0.15, row["relaxed_f1"] + 0.01, f"{row['relaxed_f1']:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Qwen3 + {o.upper()}" for o in lb.index])
    ax.set_ylabel("F1")
    ax.set_title("Qwen3 NER: Qwen-VL vs DeepSeek OCR")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_ner_heatmap(df: pd.DataFrame) -> None:
    """Template x OCR heatmap of strict F1."""
    pivot = df.pivot_table(index="template_type", columns="ocr", values="strict_f1")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(6, max(8, len(pivot) * 0.25)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=0.7)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.upper() for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if v < 0.2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.3)
    ax.set_title("Strict F1 by Template and OCR Engine")
    plt.tight_layout()
    plt.show()


def plot_type_mismatch_gap(df: pd.DataFrame) -> None:
    """Top 20 templates with largest relaxed-strict gap."""
    df = df.copy()
    df["gap"] = df["relaxed_f1"] - df["strict_f1"]
    gap = df.groupby("template_type")["gap"].mean().sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(gap)), gap.values, color="#e74c3c")
    ax.set_yticks(range(len(gap)))
    ax.set_yticklabels(gap.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Relaxed F1 - Strict F1")
    ax.set_title("Biggest Type Mismatch Gaps")
    plt.tight_layout()
    plt.show()


def plot_precision_recall_scatter(df: pd.DataFrame) -> None:
    """Precision vs Recall scatter, one panel per OCR engine."""
    engines = df["ocr"].unique()
    fig, axes = plt.subplots(1, len(engines), figsize=(10, 5), squeeze=False)
    for ax, ocr in zip(axes[0], engines):
        sub = df[df["ocr"] == ocr]
        ax.scatter(sub["strict_recall"], sub["strict_precision"], alpha=0.5, s=30,
                   c=COLORS.get(ocr, "#999"))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.2)
        ax.set_title(f"Qwen3 + {ocr.upper()}")
        ax.set_xlabel("Recall")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    axes[0][0].set_ylabel("Precision")
    plt.tight_layout()
    plt.show()


def styled_full_table(df: pd.DataFrame):
    """Return a styled DataFrame of strict F1 by template x OCR."""
    wide = df.pivot_table(index="template_type", columns="ocr", values="strict_f1").round(3)
    wide["avg"] = wide.mean(axis=1)
    wide = wide.sort_values("avg", ascending=False)
    return wide.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=0.7).format("{:.3f}")


# ── Per-entity-type charts ───────────────────────────────────────────────

def plot_per_type_f1(type_df: pd.DataFrame, title: str = "Per-Entity-Type F1") -> None:
    """Grouped bar chart: F1 by entity type, side-by-side per OCR.

    Entity types are derived from the data (works for both 7-type and 14-type sets).
    """
    etypes = sorted(type_df["entity_type"].unique())
    ocr_names = type_df["ocr"].unique()
    n_ocr = len(ocr_names)
    w = min(0.35, 0.8 / max(n_ocr, 1))
    x = np.arange(len(etypes))

    fig, ax = plt.subplots(figsize=(max(10, len(etypes) * 0.8), 5))
    for i, ocr_name in enumerate(ocr_names):
        f1s = []
        for t in etypes:
            match = type_df[(type_df["ocr"] == ocr_name) & (type_df["entity_type"] == t)]["F1"]
            f1s.append(match.values[0] if len(match) > 0 else 0.0)
        bars = ax.bar(x + i * w, f1s, w, label=ocr_name,
                      color=COLORS.get(ocr_name, f"C{i}"))
        for bar, val in zip(bars, f1s):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", fontsize=6)
    ax.set_xticks(x + w * (n_ocr - 1) / 2)
    ax.set_xticklabels(etypes, rotation=45 if len(etypes) > 8 else 0, ha="right" if len(etypes) > 8 else "center")
    ax.set_ylabel("Strict F1")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(confusion: Dict, ocr_name: str) -> None:
    """Confusion matrix heatmap for one OCR engine."""
    all_types = sorted(NER_ENTITY_TYPES)
    matrix = np.zeros((len(all_types), len(all_types)), dtype=int)
    for (gt_t, pred_t), count in confusion.items():
        if gt_t in all_types and pred_t in all_types:
            gi = all_types.index(gt_t)
            pi = all_types.index(pred_t)
            matrix[gi][pi] += count

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(all_types)))
    ax.set_xticklabels(all_types, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(all_types)))
    ax.set_yticklabels(all_types, fontsize=8)
    ax.set_xlabel("Predicted Type")
    ax.set_ylabel("Ground Truth Type")
    for i in range(len(all_types)):
        for j in range(len(all_types)):
            v = matrix[i][j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                        color="white" if v > matrix.max() * 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.6)
    ax.set_title(f"Type Confusion: Qwen3 + {ocr_name.upper()}")
    plt.tight_layout()
    plt.show()


def print_top_mismatches(mismatches: Dict[str, List]) -> None:
    """Print top type mismatches with examples."""
    print("\n=== Top Type Mismatches (text matched, type wrong) ===")
    for ocr_name, items in mismatches.items():
        print(f"\n--- {ocr_name.upper()} ---")
        mismatch_counts = defaultdict(int)
        mismatch_examples = defaultdict(list)
        for gt_text, pred_text, gt_type, pred_type in items:
            key = f"{gt_type} -> {pred_type}"
            mismatch_counts[key] += 1
            if len(mismatch_examples[key]) < 3:
                mismatch_examples[key].append(gt_text[:50])
        for key, count in sorted(mismatch_counts.items(), key=lambda x: -x[1])[:10]:
            examples = ", ".join(mismatch_examples[key])
            print(f"  {key}: {count}x  (e.g. {examples})")


# ── Segmentation charts ──────────────────────────────────────────────────

def plot_seg_leaderboard(seg_df: pd.DataFrame) -> None:
    """Segmentation P/R/F1/mAP bar chart."""
    has_map = "mAP" in seg_df.columns
    metrics = ["precision", "recall", "f1"] + (["mAP"] if has_map else [])
    lb = seg_df.groupby("model")[metrics].mean().sort_values("f1", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(lb))
    n_metrics = len(metrics)
    w = 0.18
    alphas = [1.0, 0.7, 0.5, 0.35]
    labels = ["Precision", "Recall", "F1@0.5", "mAP@[.5:.95]"]

    for i, (model, row) in enumerate(lb.iterrows()):
        c = COLORS.get(model, "#999")
        for mi, metric in enumerate(metrics):
            offset = (mi - (n_metrics - 1) / 2) * w
            bar = ax.bar(i + offset, row[metric], w, color=c, alpha=alphas[mi],
                         label=labels[mi] if i == 0 else "")
            ax.text(i + offset, row[metric] + 0.01, f"{row[metric]:.3f}",
                    ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in lb.index])
    ax.set_ylabel("Score")
    ax.set_title("Segmentation: Qwen-VL vs DeepSeek")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_map_thresholds(
    seg_pages: List[Dict],
    model_configs: List[Tuple[str, str]],
) -> None:
    """
    Line plot of F1 across IoU thresholds for each model.
    Computed from raw seg_pages data.
    """
    from .helpers import compute_iou, greedy_match_boxes

    thresholds = COCO_IOU_THRESHOLDS

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, key in model_configs:
        f1_at_thresh = []
        for t in thresholds:
            all_tp, all_fp, all_fn = 0, 0, 0
            for page in seg_pages:
                gt_boxes = [r["box_2d"] for r in page["gt_regions"]]
                pred_boxes = [r["box_2d"] for r in page.get(key, [])]
                tp, fp, fn = greedy_match_boxes(gt_boxes, pred_boxes, iou_threshold=t)
                all_tp += tp
                all_fp += fp
                all_fn += fn
            _, _, f1 = calc_prf(all_tp, all_fp, all_fn)
            f1_at_thresh.append(f1)

        c = COLORS.get(model_name, "#999")
        ax.plot(thresholds, f1_at_thresh, "o-", color=c, label=model_name.upper(), linewidth=2)
        mean_f1 = np.mean(f1_at_thresh)
        ax.axhline(mean_f1, color=c, linestyle="--", alpha=0.3)
        ax.text(0.95, mean_f1 + 0.01, f"mAP={mean_f1:.3f}", color=c, fontsize=8)

    ax.set_xlabel("IoU Threshold")
    ax.set_ylabel("F1")
    ax.set_title("Segmentation F1 across IoU Thresholds")
    ax.set_xticks(thresholds)
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds], rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ── Robustness / degradation curves ─────────────────────────────────────

def plot_robustness_curves(rob_df: pd.DataFrame) -> None:
    """
    Line plots of WER/CER vs degradation level, one subplot per degradation mode.
    """
    modes = rob_df["mode"].unique()
    engines = rob_df["ocr"].unique()

    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 4), squeeze=False)
    for ax, mode in zip(axes[0], modes):
        for eng in engines:
            sub = rob_df[(rob_df["mode"] == mode) & (rob_df["ocr"] == eng)]
            sub = sub.sort_values("level")
            c = COLORS.get(eng, "#999")
            ax.plot(sub["level"], sub["mean_wer"], "o-", color=c, label=f"{eng.upper()} WER", linewidth=2)
            ax.plot(sub["level"], sub["mean_cer"], "s--", color=c, alpha=0.5, label=f"{eng.upper()} CER")

        ax.set_xlabel("Degradation Level")
        ax.set_ylabel("Error Rate")
        ax.set_title(f"Robustness: {mode}")
        ax.set_xticks(range(5))
        ax.legend(fontsize=7)

    plt.suptitle("OCR Robustness to Image Degradation", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_robustness_heatmap(rob_df: pd.DataFrame) -> None:
    """Heatmap: degradation mode x level, showing WER for each OCR engine."""
    engines = rob_df["ocr"].unique()

    fig, axes = plt.subplots(1, len(engines), figsize=(6 * len(engines), 3), squeeze=False)
    for ax, eng in zip(axes[0], engines):
        sub = rob_df[rob_df["ocr"] == eng]
        pivot = sub.pivot_table(index="mode", columns="level", values="mean_wer")

        im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"L{c}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.iloc[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if v > 0.5 else "black")
        ax.set_title(f"{eng.upper()} — WER by Degradation")

    plt.tight_layout()
    plt.show()


# ── Reading order charts ────────────────────────────────────────────────

def plot_reading_order(order_df: pd.DataFrame) -> None:
    """
    Reading order evaluation: Kendall's tau and maintained fraction per OCR engine.
    """
    engines = order_df["ocr"].unique()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Bar chart of mean tau
    means = order_df.groupby("ocr")[["tau", "maintained_frac"]].mean()
    x = np.arange(len(means))
    w = 0.3
    for i, (eng, row) in enumerate(means.iterrows()):
        c = COLORS.get(eng, "#999")
        ax1.bar(i - w / 2, row["tau"], w, color=c, label="Kendall's tau" if i == 0 else "")
        ax1.bar(i + w / 2, row["maintained_frac"], w, color=c, alpha=0.5,
                label="Maintained order" if i == 0 else "")
        ax1.text(i - w / 2, row["tau"] + 0.01, f"{row['tau']:.3f}", ha="center", fontsize=9)
        ax1.text(i + w / 2, row["maintained_frac"] + 0.01, f"{row['maintained_frac']:.3f}",
                 ha="center", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([e.upper() for e in means.index])
    ax1.set_ylabel("Score")
    ax1.set_title("Reading Order Preservation")
    ax1.legend()

    # Histogram of tau values
    for eng in engines:
        taus = order_df[order_df["ocr"] == eng]["tau"].values
        ax2.hist(taus, bins=20, alpha=0.5, color=COLORS.get(eng, "#999"),
                 label=eng.upper(), edgecolor="white")
    ax2.set_xlabel("Kendall's tau")
    ax2.set_ylabel("Pages")
    ax2.set_title("Tau Distribution")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_reading_order_by_template(order_df: pd.DataFrame) -> None:
    """Heatmap: Kendall's tau per template x OCR engine."""
    pivot = order_df.pivot_table(index="template_type", columns="ocr", values="tau")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(6, max(6, len(pivot) * 0.25)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.upper() for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v < 0 else "black")
    fig.colorbar(im, ax=ax, shrink=0.3)
    ax.set_title("Reading Order (tau) by Template and OCR Engine")
    plt.tight_layout()
    plt.show()


# ── Table structure charts ──────────────────────────────────────────────

def plot_table_structure(tbl_df: pd.DataFrame) -> None:
    """Bar chart of table structure metrics per model."""
    metrics = ["row_count_acc", "avg_col_acc", "cell_match_f1"]
    labels = ["Row Count Acc", "Column Acc", "Cell F1@0.5"]
    means = tbl_df.groupby("model")[metrics].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(means))
    w = 0.22

    for mi, (metric, label) in enumerate(zip(metrics, labels)):
        offset = (mi - 1) * w
        for i, (model, row) in enumerate(means.iterrows()):
            c = COLORS.get(model, "#999")
            bar = ax.bar(i + offset, row[metric], w, color=c,
                         alpha=1.0 - mi * 0.2, label=label if i == 0 else "")
            ax.text(i + offset, row[metric] + 0.01, f"{row[metric]:.2f}",
                    ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in means.index])
    ax.set_ylabel("Score")
    ax.set_title("Table Structure Detection")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# ── Inference cost chart ────────────────────────────────────────────────

def plot_inference_cost(cost_data: List[Dict]) -> None:
    """
    Bar chart of inference time and GPU memory per model.
    cost_data: list of dicts from InferenceTimer.result, plus "n_pages" key.
    """
    names = [d["name"] for d in cost_data]
    times = [d["time_s"] for d in cost_data]
    n_pages = [d.get("n_pages", 1) for d in cost_data]
    per_page = [t / n if n > 0 else 0 for t, n in zip(times, n_pages)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Time per page
    colors = [COLORS.get(n, "#999") for n in names]
    axes[0].barh(range(len(names)), per_page, color=colors)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels([n.upper() for n in names])
    for i, v in enumerate(per_page):
        axes[0].text(v + 0.01, i, f"{v:.2f}s", va="center", fontsize=9)
    axes[0].set_xlabel("Seconds per Page")
    axes[0].set_title("Inference Speed")
    axes[0].invert_yaxis()

    # GPU memory
    has_gpu = any("gpu_peak_mb" in d for d in cost_data)
    if has_gpu:
        mem = [d.get("gpu_peak_mb", 0) for d in cost_data]
        axes[1].barh(range(len(names)), mem, color=colors)
        axes[1].set_yticks(range(len(names)))
        axes[1].set_yticklabels([n.upper() for n in names])
        for i, v in enumerate(mem):
            if v > 0:
                axes[1].text(v + 10, i, f"{v:.0f} MB", va="center", fontsize=9)
        axes[1].set_xlabel("GPU Peak Memory (MB)")
        axes[1].set_title("GPU Memory Usage")
        axes[1].invert_yaxis()
    else:
        axes[1].text(0.5, 0.5, "No GPU data", transform=axes[1].transAxes,
                     ha="center", va="center", fontsize=12, color="gray")
        axes[1].set_title("GPU Memory (N/A)")

    plt.tight_layout()
    plt.show()


# ── Cross-template generalization chart ─────────────────────────────────

def plot_cross_template(seen_df: pd.DataFrame, holdout_df: pd.DataFrame, holdout_names: List[str]) -> None:
    """
    Side-by-side comparison: seen templates vs held-out templates.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Grouped bar: seen vs holdout mean F1
    if not seen_df.empty and not holdout_df.empty:
        engines = seen_df["ocr"].unique()
        x = np.arange(len(engines))
        w = 0.3

        seen_means = seen_df.groupby("ocr")["strict_f1"].mean()
        hold_means = holdout_df.groupby("ocr")["strict_f1"].mean()

        for i, eng in enumerate(engines):
            c = COLORS.get(eng, "#999")
            s_val = seen_means.get(eng, 0)
            h_val = hold_means.get(eng, 0)
            ax1.bar(i - w / 2, s_val, w, color=c, label="Seen" if i == 0 else "")
            ax1.bar(i + w / 2, h_val, w, color=c, alpha=0.4, label="Held-out" if i == 0 else "")
            ax1.text(i - w / 2, s_val + 0.01, f"{s_val:.3f}", ha="center", fontsize=8)
            ax1.text(i + w / 2, h_val + 0.01, f"{h_val:.3f}", ha="center", fontsize=8)

        ax1.set_xticks(x)
        ax1.set_xticklabels([e.upper() for e in engines])
        ax1.set_ylabel("Strict F1")
        ax1.set_title("Seen vs Held-Out Templates")
        ax1.legend()

    # Generalization gap
    if not seen_df.empty and not holdout_df.empty:
        engines = seen_df["ocr"].unique()
        gaps = []
        for eng in engines:
            s = seen_df[seen_df["ocr"] == eng]["strict_f1"].mean()
            h = holdout_df[holdout_df["ocr"] == eng]["strict_f1"].mean()
            gaps.append(s - h)

        colors_bar = ["red" if g > 0.05 else "orange" if g > 0 else "green" for g in gaps]
        ax2.barh(range(len(engines)), gaps, color=colors_bar)
        ax2.set_yticks(range(len(engines)))
        ax2.set_yticklabels([e.upper() for e in engines])
        ax2.axvline(0, color="black", linewidth=0.5)
        for i, g in enumerate(gaps):
            ax2.text(g + 0.005 if g >= 0 else g - 0.005, i,
                     f"{g:+.3f}", va="center", ha="left" if g >= 0 else "right", fontsize=9)
        ax2.set_xlabel("F1 Gap (seen - held-out)")
        ax2.set_title(f"Generalization Gap\n(held-out: {', '.join(holdout_names[:5])})")

    plt.tight_layout()
    plt.show()


# ── Qualitative failure visualization ───────────────────────────────────

def plot_qualitative_overlay(
    page: Dict,
    pred_key: str,
    model_name: str = "",
    max_boxes: int = 50,
) -> None:
    """
    Overlay GT (green) and predicted (red) bounding boxes on a document image.
    Works for both NER (text entities) and segmentation (regions).
    """
    from PIL import Image

    img = Image.open(page["image_path"]).convert("RGB")
    fig, ax = plt.subplots(1, 1, figsize=(8, 11))
    ax.imshow(img)

    # Draw GT boxes (green)
    gt_regions = page.get("gt_regions", [])
    for i, r in enumerate(gt_regions[:max_boxes]):
        box = r["box_2d"]
        rect = plt.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=1.5, edgecolor="green", facecolor="none", linestyle="-"
        )
        ax.add_patch(rect)
        ax.text(box[0], box[1] - 2, r["type"], fontsize=5, color="green",
                va="bottom", weight="bold")

    # Draw predicted boxes (red)
    pred_regions = page.get(pred_key, [])
    for i, r in enumerate(pred_regions[:max_boxes]):
        box = r["box_2d"]
        rect = plt.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=1.5, edgecolor="red", facecolor="none", linestyle="--"
        )
        ax.add_patch(rect)
        ax.text(box[2], box[1] - 2, r["type"], fontsize=5, color="red",
                va="bottom", ha="right")

    ax.set_title(
        f"{model_name} — {page.get('stem', '')}\n"
        f"GT: {len(gt_regions)} regions (green)  |  "
        f"Pred: {len(pred_regions)} regions (red)",
        fontsize=10,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_qualitative_grid(
    pages: List[Dict],
    pred_key: str,
    model_name: str = "",
    n_samples: int = 4,
    seed: int = 42,
) -> None:
    """
    Grid of qualitative overlay examples. Picks worst, best, and random samples.
    """
    from PIL import Image
    from .helpers import greedy_match_boxes, calc_prf

    # Score each page
    scored = []
    for page in pages:
        gt_boxes = [r["box_2d"] for r in page.get("gt_regions", [])]
        pred_boxes = [r["box_2d"] for r in page.get(pred_key, [])]
        tp, fp, fn = greedy_match_boxes(gt_boxes, pred_boxes)
        _, _, f1 = calc_prf(tp, fp, fn)
        scored.append((f1, page))

    scored.sort(key=lambda x: x[0])

    # Pick samples: worst, best, 2 random from middle
    picks = []
    if len(scored) >= 4:
        picks.append(scored[0])                           # worst
        picks.append(scored[-1])                          # best
        mid = scored[len(scored) // 4: 3 * len(scored) // 4]
        rng = np.random.RandomState(seed)
        mid_idx = rng.choice(len(mid), size=min(n_samples - 2, len(mid)), replace=False)
        for idx in mid_idx:
            picks.append(mid[idx])
    else:
        picks = scored[:n_samples]

    n = len(picks)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 8))
    if n == 1:
        axes = [axes]

    for ax, (f1, page) in zip(axes, picks):
        img = Image.open(page["image_path"]).convert("RGB")
        ax.imshow(img)

        for r in page.get("gt_regions", []):
            box = r["box_2d"]
            rect = plt.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=1, edgecolor="green", facecolor="none"
            )
            ax.add_patch(rect)

        for r in page.get(pred_key, []):
            box = r["box_2d"]
            rect = plt.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=1, edgecolor="red", facecolor="none", linestyle="--"
            )
            ax.add_patch(rect)

        ax.set_title(f"F1={f1:.2f}\n{page.get('template_type', '')}", fontsize=9)
        ax.axis("off")

    fig.suptitle(f"{model_name} — GT (green) vs Pred (red)", fontsize=12)
    plt.tight_layout()
    plt.show()
