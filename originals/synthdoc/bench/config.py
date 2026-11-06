"""
Config-driven benchmark runner.

Usage:
    python -m synthdoc.bench run benchmarks/ner_comparison.yaml
    python -m synthdoc.bench run benchmarks/ner_comparison.yaml --limit 50
    python -m synthdoc.bench run benchmarks/ner_comparison.yaml --force-ocr
    python -m synthdoc.bench list
"""

import json
import sys
import yaml
import pandas as pd
from pathlib import Path
from collections import Counter

from .helpers import load_ner_pages, FULL_ENTITY_TYPES, NER_ENTITY_TYPES, InferenceTimer
from .run_benchmark import _load_ocr_cache, _save_ocr_cache, evaluate_model

BENCHMARKS_DIR = Path(__file__).parent / "configs"


def load_engines(path=None):
    if path is None:
        path = BENCHMARKS_DIR / "engines.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_run_config(path):
    p = Path(path).resolve()
    with open(p) as f:
        config = yaml.safe_load(f)
    base = p.parent
    for key in ("data", "output"):
        val = config.get(key)
        if val and not Path(val).is_absolute():
            config[key] = str(base / val)
    return config


def list_engines():
    engines = load_engines()
    for section in ("ocr", "ner", "seg"):
        items = engines.get(section, {})
        if not items:
            continue
        print(f"\n{section.upper()} engines:")
        for name, cfg in items.items():
            gpu = "GPU" if cfg.get("gpu") else "CPU"
            mid = cfg.get("model_id", "built-in")
            runner = cfg.get("runner", name)
            alias = f" (runner: {runner})" if runner != name else ""
            print(f"  {name:20} {gpu:4}  {mid}{alias}")


def _resolve(engine_name, engine_type, engines_config):
    section = engines_config.get(engine_type, {})
    engine = section.get(engine_name, {})
    if not engine:
        available = ", ".join(section.keys()) or "none"
        raise ValueError(f"Unknown {engine_type} engine '{engine_name}'. Available: {available}")
    runner = engine.get("runner", engine_name)
    model_id = engine.get("model_id")
    return runner, model_id


def _dispatch_ocr(runner, pages, model_id=None, on_progress=None):
    if runner == "tesseract":
        from .run_benchmark import run_tesseract_ocr
        run_tesseract_ocr(pages)
    elif runner == "qwenvl":
        from .live import run_qwenvl_ocr
        run_qwenvl_ocr(pages, model_id=model_id)
    elif runner == "deepseek":
        from .live import run_deepseek_ocr
        run_deepseek_ocr(pages, on_progress=on_progress, model_id=model_id)
    elif runner == "glm":
        from .live import run_glm_ocr
        run_glm_ocr(pages, model_id=model_id)
    else:
        raise ValueError(
            f"Unknown OCR runner '{runner}'. "
            f"Add a dispatch entry in synthdoc.bench/config.py::_dispatch_ocr"
        )


def _dispatch_ner(runner, pages, ocr_key, model_id=None, ner_dir=None, ner_engine=None):
    if runner == "spacy":
        from .run_benchmark import run_spacy_ner
        return run_spacy_ner(pages, ocr_key)
    elif runner == "deberta":
        from .run_benchmark import run_deberta_ner
        return run_deberta_ner(pages, ocr_key, model_id=model_id)
    elif runner == "qwen3":
        from .live import run_qwen3_ner
        run_qwen3_ner(pages, [ocr_key], model_id=model_id, ner_dir=ner_dir, ner_engine=ner_engine or "qwen3")
        return f"ner_{ocr_key.replace('_ocr', '')}"
    elif runner == "finetuned":
        # model_id is a checkpoint path, relative to synthetic-document-generator/ (BENCHMARKS_DIR.parent)
        from pathlib import Path
        ck = Path(model_id) if Path(model_id).is_absolute() else (BENCHMARKS_DIR.parent / model_id)
        from .live import run_finetuned_ner
        run_finetuned_ner(pages, ocr_key, ck, ner_engine=ner_engine or ck.parent.name, ner_dir=ner_dir)
        return f"ner_{ocr_key.replace('_ocr', '')}"
    else:
        raise ValueError(
            f"Unknown NER runner '{runner}'. "
            f"Add a dispatch entry in synthdoc.bench/config.py::_dispatch_ner"
        )


def run_from_config(config_path, limit=None, force_ocr=False):
    config = load_run_config(config_path)
    engines = load_engines()

    data_dir = Path(config["data"])
    output_dir = Path(config.get("output", data_dir / "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run: {config.get('name', config_path)}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}\n")

    pages = load_ner_pages(data_dir, entity_types=FULL_ENTITY_TYPES)
    if not pages:
        print(f"Error: no pages found in {data_dir}")
        sys.exit(1)

    limit = limit or config.get("limit")
    if limit:
        pages = pages[:limit]
        print(f"Limited to {len(pages)} pages")

    print(f"Loaded {len(pages)} pages")

    from .static import dataset_stats
    stats = dataset_stats(pages)
    print(
        f"  Documents: {stats['n_docs']}, "
        f"Templates: {stats['n_templates']}, "
        f"Entities: {sum(stats['type_counts'].values())}"
    )
    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    ocr_engines = config.get("ocr", [])
    if isinstance(ocr_engines, str):
        ocr_engines = [ocr_engines]

    for ocr_name in ocr_engines:
        _run_ocr_step(ocr_name, pages, engines, data_dir, force_ocr)

    ner_engines = config.get("ner", [])
    if not ocr_engines or not ner_engines:
        print("\nNo OCR/NER engines configured. Done.")
        return

    all_results, all_timings, model_configs, failed = _run_ner_steps(
        ocr_engines, ner_engines, pages, engines, data_dir
    )

    _save_results(pages, all_results, all_timings, model_configs, output_dir)

    if all_results:
        _print_summary(all_results)

    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        sys.exit(1)

    print()


def _run_ocr_step(ocr_name, pages, engines, data_dir, force_ocr):
    ocr_key = f"{ocr_name}_ocr"
    runner, model_id = _resolve(ocr_name, "ocr", engines)

    if not force_ocr:
        if ocr_key in pages[0]:
            print(f"  Reusing in-memory {ocr_key}")
            return
        _load_ocr_cache(pages, ocr_key, data_dir)

    remaining = [p for p in pages if ocr_key not in p]
    if not remaining:
        print(f"  {ocr_name} OCR: all {len(pages)} pages cached")
        return

    print(f"  Running {ocr_name} OCR on {len(remaining)}/{len(pages)} pages")
    save_fn = lambda: _save_ocr_cache(pages, ocr_key, data_dir)
    with InferenceTimer(f"{ocr_name}_ocr") as timer:
        _dispatch_ocr(
            runner, remaining, model_id=model_id,
            on_progress=save_fn if runner == "deepseek" else None,
        )
    _save_ocr_cache(pages, ocr_key, data_dir)
    print(f"  {timer.result['time_s']}s")


def _run_ner_steps(ocr_engines, ner_engines, pages, engines, data_dir):
    all_results = []
    all_timings = []
    model_configs = []
    failed = []

    for ocr_name in ocr_engines:
        ocr_key = f"{ocr_name}_ocr"
        if ocr_key not in pages[0]:
            _load_ocr_cache(pages, ocr_key, data_dir)

        for ner_name in ner_engines:
            model_name = f"{ocr_name}_{ner_name}"
            print(f"\n{'=' * 60}")
            print(f"Running: {model_name}")
            print(f"{'=' * 60}")

            try:
                runner, model_id = _resolve(ner_name, "ner", engines)
                with InferenceTimer(f"{model_name}_ner") as timer:
                    ner_key = _dispatch_ner(runner, pages, ocr_key, model_id=model_id)
                all_timings.append(timer.result)

                results = evaluate_model(pages, model_name, ner_key, ocr_key)
                all_results.append(results)
                model_configs.append((model_name, ocr_key, ner_key))

                for eset in ["standard", "full"]:
                    subset = results["ner_df"][results["ner_df"]["entity_set"] == eset]
                    n = 7 if eset == "standard" else 14
                    print(f"\n  {model_name} [{eset} - {n} types]:")
                    print(f"    Strict F1:  {subset['strict_f1'].mean():.4f}")
                    print(f"    Relaxed F1: {subset['relaxed_f1'].mean():.4f}")

            except Exception as e:
                import traceback
                print(f"\n  FAILED: {model_name}: {e}")
                traceback.print_exc()
                failed.append(model_name)
                if len(ocr_engines) == 1 and len(ner_engines) == 1:
                    sys.exit(1)

    return all_results, all_timings, model_configs, failed


def _save_results(pages, all_results, all_timings, model_configs, output_dir):
    if not all_results:
        return

    current = set(m for m, _, _ in model_configs)

    def merge(path, new_df, key_col):
        if path.exists():
            old = pd.read_csv(path)
            if key_col not in old.columns:
                return new_df
            return pd.concat([old[~old[key_col].isin(current)], new_df], ignore_index=True)
        return new_df

    def merge_json(path, new_data):
        if path.exists():
            with open(path) as f:
                old = json.load(f)
            for key in list(old.keys()):
                if any(m in key for m in current):
                    del old[key]
            old.update(new_data)
            return old
        return new_data

    ner_df = pd.concat([r["ner_df"] for r in all_results], ignore_index=True)
    ner_df = merge(output_dir / "benchmark_results.csv", ner_df, "ocr")
    ner_df.to_csv(output_dir / "benchmark_results.csv", index=False)
    print(f"Saved: benchmark_results.csv")

    ocr_dfs = [r["ocr_df"] for r in all_results if not r["ocr_df"].empty]
    if ocr_dfs:
        ocr_df = pd.concat(ocr_dfs, ignore_index=True)
        ocr_df = merge(output_dir / "ocr_results.csv", ocr_df, "ocr")
        ocr_df.to_csv(output_dir / "ocr_results.csv", index=False)
        print(f"Saved: ocr_results.csv")

    type_df = pd.concat([r["type_df"] for r in all_results], ignore_index=True)
    type_df = merge(output_dir / "per_type_results.csv", type_df, "ocr")
    type_df.to_csv(output_dir / "per_type_results.csv", index=False)
    print(f"Saved: per_type_results.csv")

    ent_dfs = [r["entity_matches"] for r in all_results if not r["entity_matches"].empty]
    if ent_dfs:
        ent_df = pd.concat(ent_dfs, ignore_index=True)
        ent_df = merge(output_dir / "entity_matches.csv", ent_df, "model")
        ent_df.to_csv(output_dir / "entity_matches.csv", index=False)
        print(f"Saved: entity_matches.csv")

    if model_configs:
        ocr_keys = sorted(set(o for _, o, _ in model_configs))
        ocr_path = output_dir / "ocr_outputs.csv"
        if ocr_path.exists():
            ocr_text = pd.read_csv(ocr_path)
        else:
            ocr_text = pd.DataFrame({
                "stem": [p["stem"] for p in pages],
                "template_type": [p["template_type"] for p in pages],
                "gt_text": [p.get("gt_text", "") for p in pages],
            })
        stem_idx = {row["stem"]: i for i, row in ocr_text.iterrows()}
        for okey in ocr_keys:
            if okey not in ocr_text.columns:
                ocr_text[okey] = ""
            else:
                ocr_text[okey] = ocr_text[okey].fillna("").astype(str)
            for page in pages:
                if page["stem"] in stem_idx:
                    ocr_text.at[stem_idx[page["stem"]], okey] = str(page.get(okey, ""))
        ocr_text.to_csv(ocr_path, index=False)
        print(f"Saved: ocr_outputs.csv")

    if all_timings:
        t_path = output_dir / "timings.json"
        if t_path.exists():
            with open(t_path) as f:
                old = json.load(f)
            old = [t for t in old if not any(m in t.get("name", "") for m in current)]
            all_timings = old + all_timings
        with open(t_path, "w") as f:
            json.dump(all_timings, f, indent=2)
        print(f"Saved: timings.json")

    all_conf = {"standard": {}, "full": {}}
    all_mm = {"standard": {}, "full": {}}
    for r in all_results:
        for eset in ["standard", "full"]:
            all_conf[eset].update(r["confusion"][eset])
            all_mm[eset].update(r["mismatches"][eset])

    if any(all_conf.values()):
        conf_json = {}
        for eset, conf in all_conf.items():
            conf_json[eset] = {
                name: {f"{gt}->{pred}": c for (gt, pred), c in pairs.items()}
                for name, pairs in conf.items()
            }
        conf_path = output_dir / "confusion.json"
        if conf_path.exists():
            with open(conf_path) as f:
                old = json.load(f)
            for eset in ["standard", "full"]:
                old_eset = old.get(eset, {})
                for key in list(old_eset.keys()):
                    if key in current:
                        del old_eset[key]
                old_eset.update(conf_json.get(eset, {}))
                conf_json[eset] = old_eset
        with open(conf_path, "w") as f:
            json.dump(conf_json, f, indent=2)
        print(f"Saved: confusion.json")

    if any(all_mm.values()):
        mm_json = {}
        for eset, mm in all_mm.items():
            mm_json[eset] = {}
            for name, items in mm.items():
                patterns = Counter(f"{gt_t}->{pred_t}" for _, _, gt_t, pred_t in items)
                mm_json[eset][name] = {
                    "total": len(items),
                    "top_patterns": dict(patterns.most_common(20)),
                    "examples": [
                        {"gt_text": gt, "pred_text": pt, "gt_type": gt_t, "pred_type": pt_t}
                        for gt, pt, gt_t, pt_t in items[:50]
                    ],
                }
        mm_path = output_dir / "mismatches.json"
        if mm_path.exists():
            with open(mm_path) as f:
                old = json.load(f)
            for eset in ["standard", "full"]:
                old_eset = old.get(eset, {})
                for key in list(old_eset.keys()):
                    if key in current:
                        del old_eset[key]
                old_eset.update(mm_json.get(eset, {}))
                mm_json[eset] = old_eset
        with open(mm_path, "w") as f:
            json.dump(mm_json, f, indent=2)
        print(f"Saved: mismatches.json")

    if model_configs:
        from .static import ocr_ner_error_attribution, calibration_analysis, compute_significance

        standard_pages = []
        for page in pages:
            p = dict(page)
            p["gt_entities"] = [e for e in page["gt_entities"] if e["type"] in NER_ENTITY_TYPES]
            standard_pages.append(p)

        attr_configs = [(n, o, k) for n, o, k in model_configs]
        attr_df = ocr_ner_error_attribution(standard_pages, attr_configs)
        if not attr_df.empty:
            attr_df = merge(output_dir / "error_attribution.csv", attr_df, "model")
            attr_df.to_csv(output_dir / "error_attribution.csv", index=False)
            print(f"Saved: error_attribution.csv")

        cal_configs = [(n, k) for n, _, k in model_configs]
        cal = calibration_analysis(standard_pages, cal_configs)
        cal_json = {k: v.to_dict(orient="records") for k, v in cal.items() if not v.empty}
        if cal_json:
            cal_json = merge_json(output_dir / "calibration.json", cal_json)
            with open(output_dir / "calibration.json", "w") as f:
                json.dump(cal_json, f, indent=2)
            print(f"Saved: calibration.json")

        if len(model_configs) >= 2:
            combined_ner = pd.concat([r["ner_df"] for r in all_results], ignore_index=True)
            std_ner = combined_ner[combined_ner["entity_set"] == "standard"]
            sig = {}
            for i in range(len(model_configs)):
                for j in range(i + 1, len(model_configs)):
                    a = model_configs[i][0]
                    b = model_configs[j][0]
                    try:
                        sig[f"{a}_vs_{b}"] = compute_significance(std_ner, a, b)
                    except Exception:
                        pass
            if sig:
                sig = merge_json(output_dir / "significance.json", sig)
                with open(output_dir / "significance.json", "w") as f:
                    json.dump(sig, f, indent=2)
                print(f"Saved: significance.json")


def _print_summary(all_results):
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    combined = pd.concat([r["ner_df"] for r in all_results], ignore_index=True)
    for eset in ["standard", "full"]:
        subset = combined[combined["entity_set"] == eset]
        if subset.empty:
            continue
        n = 7 if eset == "standard" else 14
        print(f"\n  {eset.upper()} ({n} types):")
        summary = subset.groupby("ocr").agg({
            "strict_f1": "mean",
            "relaxed_f1": "mean",
            "strict_precision": "mean",
            "strict_recall": "mean",
        }).round(4)
        print(summary.to_string())
