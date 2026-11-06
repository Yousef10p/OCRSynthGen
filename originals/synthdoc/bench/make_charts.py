"""
CLI entry point for generating benchmark charts from results CSVs.

Usage:
    python -m synthdoc.bench.make_charts --results results/benchmark_results.csv --output charts/
    python -m synthdoc.bench.make_charts --results results/ --output charts/
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import report


def save_fig(output_dir, name):
    """Save current matplotlib figure and close."""
    path = output_dir / f"{name}.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  {name}.png")


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark charts from results CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m synthdoc.bench.make_charts --results results/benchmark_results.csv --output charts/
  python -m synthdoc.bench.make_charts --results results/ --output charts/
        """,
    )
    parser.add_argument("--results", type=str, required=True,
                        help="Path to benchmark_results.csv or results directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for chart images")
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find result files
    if results_path.is_dir():
        results_dir = results_path
        ner_path = results_dir / "benchmark_results.csv"
        ocr_path = results_dir / "ocr_results.csv"
        type_path = results_dir / "per_type_results.csv"
        stats_path = results_dir / "dataset_stats.json"
        timings_path = results_dir / "timings.json"
        attr_path = results_dir / "error_attribution.csv"
        cal_path = results_dir / "calibration.json"
        sig_path = results_dir / "significance.json"
        confusion_path = results_dir / "confusion.json"
    else:
        results_dir = results_path.parent
        ner_path = results_path
        ocr_path = results_dir / "ocr_results.csv"
        type_path = results_dir / "per_type_results.csv"
        stats_path = results_dir / "dataset_stats.json"
        timings_path = results_dir / "timings.json"
        attr_path = results_dir / "error_attribution.csv"
        cal_path = results_dir / "calibration.json"
        sig_path = results_dir / "significance.json"
        confusion_path = results_dir / "confusion.json"

    print(f"Generating charts in {output_dir}/\n")

    # ── Dataset stats charts ──────────────────────────────────────────
    if stats_path.exists():
        print("Dataset stats:")
        with open(stats_path) as f:
            stats = json.load(f)

        report.plot_entity_distribution(stats["type_counts"])
        save_fig(output_dir, "entity_distribution")

        if stats.get("tmpl_type_counts"):
            report.plot_template_type_heatmap(stats["tmpl_type_counts"])
            save_fig(output_dir, "template_type_heatmap")

        if stats.get("pages_per_doc"):
            report.plot_pages_per_doc(stats["pages_per_doc"])
            save_fig(output_dir, "pages_per_doc")

    # ── NER charts (split by entity_set) ─────────────────────────────
    if ner_path.exists():
        ner_df = pd.read_csv(ner_path)
        has_entity_set = "entity_set" in ner_df.columns

        # Generate charts for each entity set
        entity_sets = ner_df["entity_set"].unique() if has_entity_set else ["standard"]

        for eset in entity_sets:
            subset = ner_df[ner_df["entity_set"] == eset] if has_entity_set else ner_df
            n_types = 7 if eset == "standard" else 14
            suffix = f"_{eset}" if has_entity_set else ""
            label = f" ({eset}, {n_types} types)" if has_entity_set else ""

            print(f"NER results{label}:")

            report.plot_ner_leaderboard(subset)
            save_fig(output_dir, f"ner_leaderboard{suffix}")

            report.plot_ner_heatmap(subset)
            save_fig(output_dir, f"ner_heatmap{suffix}")

            report.plot_type_mismatch_gap(subset)
            save_fig(output_dir, f"strict_vs_relaxed_gap{suffix}")

            report.plot_precision_recall_scatter(subset)
            save_fig(output_dir, f"precision_recall_scatter{suffix}")

            # Per-document charts
            if "doc_id" in subset.columns:
                report.plot_per_doc_f1_distribution(subset)
                save_fig(output_dir, f"per_doc_f1_distribution{suffix}")

                if "n_pages" in subset.columns:
                    report.plot_per_doc_pages_vs_f1(subset)
                    save_fig(output_dir, f"pages_vs_f1{suffix}")

    # ── OCR charts ────────────────────────────────────────────────────
    if ocr_path.exists():
        print("OCR results:")
        ocr_df = pd.read_csv(ocr_path)

        report.plot_ocr_quality(ocr_df)
        save_fig(output_dir, "ocr_quality")

        report.plot_ocr_quality_by_template(ocr_df)
        save_fig(output_dir, "ocr_quality_by_template")

    # ── Per-type charts (split by entity_set) ─────────────────────────
    if type_path.exists():
        type_df = pd.read_csv(type_path)
        has_entity_set = "entity_set" in type_df.columns

        entity_sets = type_df["entity_set"].unique() if has_entity_set else ["standard"]

        for eset in entity_sets:
            subset = type_df[type_df["entity_set"] == eset] if has_entity_set else type_df
            n_types = 7 if eset == "standard" else 14
            suffix = f"_{eset}" if has_entity_set else ""
            label = f" ({eset}, {n_types} types)" if has_entity_set else ""

            print(f"Per-type results{label}:")
            report.plot_per_type_f1(subset, title=f"Per-Entity-Type F1 — {eset} ({n_types} types)")
            save_fig(output_dir, f"per_type_f1{suffix}")

    # ── Error attribution charts ─────────────────────────────────────
    if attr_path.exists():
        print("Error attribution:")
        attr_df = pd.read_csv(attr_path)

        report.plot_error_attribution(attr_df)
        save_fig(output_dir, "error_attribution")

        report.plot_error_attribution_pct(attr_df)
        save_fig(output_dir, "error_attribution_pct")

    # ── Calibration charts ───────────────────────────────────────────
    if cal_path.exists():
        print("Calibration:")
        with open(cal_path) as f:
            cal_json = json.load(f)

        cal_data = {k: pd.DataFrame(v) for k, v in cal_json.items()}
        if cal_data:
            report.plot_calibration_diagram(cal_data)
            save_fig(output_dir, "calibration")

    # ── Confusion matrix charts ──────────────────────────────────────
    if confusion_path.exists():
        print("Confusion matrices:")
        with open(confusion_path) as f:
            confusion_json = json.load(f)

        # Plot confusion for standard entity set
        std_conf = confusion_json.get("standard", {})
        for ocr_name, pairs in std_conf.items():
            # Reconstruct tuple-keyed dict from "GT->PRED" string keys
            conf_dict = {}
            for key_str, count in pairs.items():
                gt, pred = key_str.split("->")
                conf_dict[(gt, pred)] = count
            report.plot_confusion_matrix(conf_dict, ocr_name)
            save_fig(output_dir, f"confusion_{ocr_name}")

    # ── Significance charts ──────────────────────────────────────────
    if sig_path.exists():
        print("Significance tests:")
        with open(sig_path) as f:
            sig_results = json.load(f)

        for pair_name, sig_data in sig_results.items():
            report.plot_bootstrap_ci(sig_data)
            save_fig(output_dir, f"significance_{pair_name}")

    # ── Timing charts ─────────────────────────────────────────────────
    if timings_path.exists():
        print("Timing results:")
        with open(timings_path) as f:
            timings = json.load(f)

        if timings:
            report.plot_inference_cost(timings)
            save_fig(output_dir, "inference_cost")

    n_charts = len(list(output_dir.glob("*.png")))
    print(f"\nDone. {n_charts} charts saved to {output_dir}/")


if __name__ == "__main__":
    main()
