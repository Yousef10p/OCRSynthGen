"""
CLI entry point for running the SDS benchmark.

Usage:
    python -m synthdoc.bench.run_benchmark --data ./data --output results/
    python -m synthdoc.bench.run_benchmark --data ./data --output results/ --model tesseract_spacy
    python -m synthdoc.bench.run_benchmark --data ./data --output results/ --model qwenvl_qwen3 --model deepseek_qwen3
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from .helpers import load_ner_pages, load_seg_pages, InferenceTimer, NER_ENTITY_TYPES, FULL_ENTITY_TYPES


# ── Model definitions ────────────────────────────────────────────────────

MODELS = {
    # Tier 1: CPU baselines
    "tesseract_spacy": {
        "tier": 1,
        "description": "Tesseract 5 + spaCy en_core_web_trf",
        "ocr": "tesseract",
        "ner": "spacy",
        "gpu": False,
    },
    # Tier 2: Transformer baselines
    "tesseract_deberta": {
        "tier": 2,
        "description": "Tesseract 5 + DeBERTa-v3 NER",
        "ocr": "tesseract",
        "ner": "deberta",
        "gpu": True,
    },
    # Tier 3: VLM baselines
    "qwenvl_qwen3": {
        "tier": 3,
        "description": "Qwen3-VL-8B OCR + Qwen3-8B NER",
        "ocr": "qwenvl",
        "ner": "qwen3",
        "gpu": True,
    },
    "deepseek_qwen3": {
        "tier": 3,
        "description": "DeepSeek-OCR-2 + Qwen3-8B NER",
        "ocr": "deepseek",
        "ner": "qwen3",
        "gpu": True,
    },
}


# ── Dependency checks ────────────────────────────────────────────────────

def check_dependencies(model_names):
    """Check all required dependencies before running. Fail fast with clear errors."""
    errors = []

    # Check data dependencies per model
    for model_name in model_names:
        config = MODELS[model_name]

        # OCR dependencies
        if config["ocr"] == "tesseract":
            try:
                import pytesseract
                # Verify tesseract binary is available
                pytesseract.get_tesseract_version()
            except ImportError:
                errors.append(f"{model_name}: pytesseract not installed. Run: pip install pytesseract")
            except Exception:
                errors.append(f"{model_name}: tesseract binary not found. Install Tesseract 5: https://github.com/tesseract-ocr/tesseract")

        if config["ocr"] in ("qwenvl", "deepseek"):
            try:
                import torch
                if not torch.cuda.is_available():
                    errors.append(f"{model_name}: requires GPU but torch.cuda.is_available() is False")
            except ImportError:
                errors.append(f"{model_name}: requires torch with CUDA. Run: pip install torch")
            try:
                import transformers
            except ImportError:
                errors.append(f"{model_name}: requires transformers. Run: pip install transformers")

        # NER dependencies
        if config["ner"] == "spacy":
            try:
                import spacy
                spacy.load("en_core_web_trf")
            except ImportError:
                errors.append(f"{model_name}: spacy not installed. Run: pip install spacy")
            except OSError:
                errors.append(f"{model_name}: spacy model not found. Run: python -m spacy download en_core_web_trf")

        if config["ner"] == "deberta":
            try:
                import torch
                if not torch.cuda.is_available():
                    errors.append(f"{model_name}: DeBERTa NER requires GPU but torch.cuda.is_available() is False")
            except ImportError:
                errors.append(f"{model_name}: requires torch with CUDA. Run: pip install torch")

        if config["ner"] == "qwen3":
            try:
                import torch
                if not torch.cuda.is_available():
                    errors.append(f"{model_name}: Qwen3 NER requires GPU but torch.cuda.is_available() is False")
            except ImportError:
                errors.append(f"{model_name}: requires torch with CUDA. Run: pip install torch")

    return errors


# ── OCR runners ──────────────────────────────────────────────────────────

def run_tesseract_ocr(pages):
    """Run Tesseract OCR on all pages. Writes 'tesseract_ocr' key in-place."""
    import pytesseract
    from PIL import Image

    for i, page in enumerate(pages):
        img = Image.open(page["image_path"])
        page["tesseract_ocr"] = pytesseract.image_to_string(img)
        if (i + 1) % 50 == 0:
            print(f"  Tesseract OCR: {i + 1}/{len(pages)}")
    print(f"Tesseract OCR done — {len(pages)} pages")


# ── NER runners ──────────────────────────────────────────────────────────

def run_spacy_ner(pages, ocr_key):
    """Run spaCy NER on OCR text. Writes 'ner_{ocr_key_prefix}' key in-place."""
    import spacy
    nlp = spacy.load("en_core_web_trf")

    SPACY_TYPE_MAP = {
        "PERSON": "PERSON",
        "ORG": "ORG",
        "GPE": "ADDRESS",
        "LOC": "ADDRESS",
        "DATE": "DATE",
        "MONEY": "AMOUNT",
        "CARDINAL": "AMOUNT",
    }

    ner_key = f"ner_{ocr_key.split('_')[0]}"

    for i, page in enumerate(pages):
        text = page.get(ocr_key, "")
        if not text.strip():
            page[ner_key] = []
            continue

        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            mapped_type = SPACY_TYPE_MAP.get(ent.label_)
            if mapped_type:
                entities.append({
                    "text": ent.text,
                    "type": mapped_type,
                    "confidence": 1.0,
                })
        page[ner_key] = entities

        if (i + 1) % 50 == 0:
            print(f"  spaCy NER: {i + 1}/{len(pages)}")

    print(f"spaCy NER done — {len(pages)} pages")
    return ner_key


def run_deberta_ner(pages, ocr_key, model_id=None):
    """Run DeBERTa NER on OCR text. Writes 'ner_{ocr_key_prefix}' key in-place."""
    from transformers import pipeline

    mid = model_id or "dslim/deberta-v3-large-ner"
    ner_pipeline = pipeline(
        "ner",
        model=mid,
        aggregation_strategy="simple",
        device=0,
    )

    DEBERTA_TYPE_MAP = {
        "PER": "PERSON",
        "ORG": "ORG",
        "LOC": "ADDRESS",
        "MISC": "MISC",
    }

    ner_key = f"ner_{ocr_key.split('_')[0]}"

    for i, page in enumerate(pages):
        text = page.get(ocr_key, "")
        if not text.strip():
            page[ner_key] = []
            continue

        if len(text) > 4096:
            print(f"  Warning: page {page['stem']} has {len(text)} chars, truncating to 4096 for DeBERTa")

        results = ner_pipeline(text[:4096])
        entities = []
        for r in results:
            mapped_type = DEBERTA_TYPE_MAP.get(r["entity_group"])
            if mapped_type:
                entities.append({
                    "text": r["word"],
                    "type": mapped_type,
                    "confidence": float(r["score"]),
                })
        page[ner_key] = entities

        if (i + 1) % 50 == 0:
            print(f"  DeBERTa NER: {i + 1}/{len(pages)}")

    print(f"DeBERTa NER done — {len(pages)} pages")
    return ner_key


# ── OCR result caching ───────────────────────────────────────────────────

def _ocr_cache_path(output_dir: Path, ocr_key: str) -> Path:
    return output_dir / f"_cache_{ocr_key}.json"


def _save_ocr_cache(pages, ocr_key: str, output_dir: Path) -> None:
    """Save OCR text results to disk so they survive crashes."""
    cache = {page["stem"]: page[ocr_key] for page in pages if ocr_key in page}
    path = _ocr_cache_path(output_dir, ocr_key)
    with open(path, "w") as f:
        json.dump(cache, f)
    print(f"  OCR cache saved: {path.name} ({len(cache)} pages)")


def _load_ocr_cache(pages, ocr_key: str, output_dir: Path) -> bool:
    """Load cached OCR results from disk. Returns True if cache was loaded.

    Validates cached values — rejects entries that are "None" (from broken
    infer() calls) or empty strings.  If every cached value is bad the
    cache is treated as missing so OCR re-runs automatically.

    Also detects stale caches — if < 10% of current page stems match the
    cache keys, the cache is from a different dataset and is discarded.
    """
    path = _ocr_cache_path(output_dir, ocr_key)
    if not path.exists():
        return False
    with open(path) as f:
        cache = json.load(f)

    # Detect stale cache: if overlap is very low, this cache is from a
    # different dataset (e.g. old 53-page run vs new 528-page dataset).
    page_stems = {p["stem"] for p in pages}
    overlap = len(page_stems & set(cache.keys()))
    overlap_pct = overlap / len(pages) if pages else 0
    if len(cache) > 0 and overlap_pct < 0.1:
        archive = path.with_suffix(f".{len(cache)}pages.bak.json")
        path.rename(archive)
        print(
            f"  ⚠ STALE CACHE DETECTED: {path.name} has {len(cache)} entries "
            f"but only {overlap}/{len(pages)} match current dataset "
            f"({overlap_pct:.0%} overlap). Archived to {archive.name}"
        )
        return False

    valid, skipped = 0, 0
    for page in pages:
        val = cache.get(page["stem"])
        if val is None:
            continue
        # Reject obvious bad values
        if val in ("None", "null", ""):
            skipped += 1
            continue
        page[ocr_key] = val
        valid += 1

    if skipped:
        print(f"  Cache {path.name}: skipped {skipped} bad entries (\"None\" / empty)")
    if valid > 0:
        print(f"  Loaded OCR cache: {path.name} ({valid}/{len(pages)} pages)")
        return True
    print(f"  Cache {path.name} has no valid entries — will re-run OCR")
    return False


# ── Pipeline orchestration ───────────────────────────────────────────────

def run_model_pipeline(model_name, pages, output_dir=None):
    """Run a complete OCR → NER pipeline for a given model config."""
    config = MODELS[model_name]
    timings = []

    # OCR step
    ocr_key = f"{config['ocr']}_ocr"

    # Try loading from in-memory, then from disk cache
    need_ocr = True
    if ocr_key in pages[0]:
        print(f"  Reusing existing {ocr_key} (already in memory)")
        need_ocr = False
    elif output_dir:
        _load_ocr_cache(pages, ocr_key, output_dir)

    # Find pages still missing OCR (partial cache or first run)
    remaining = [p for p in pages if ocr_key not in p] if need_ocr else []

    if remaining:
        print(f"  Running OCR on {len(remaining)}/{len(pages)} pages")
        if config["ocr"] == "tesseract":
            with InferenceTimer(f"{model_name}_ocr") as timer:
                run_tesseract_ocr(remaining)
            timings.append(timer.result)
        elif config["ocr"] == "qwenvl":
            from .live import run_qwenvl_ocr
            with InferenceTimer(f"{model_name}_ocr") as timer:
                run_qwenvl_ocr(remaining)
            timings.append(timer.result)
        elif config["ocr"] == "deepseek":
            from .live import run_deepseek_ocr
            save_fn = (lambda: _save_ocr_cache(pages, ocr_key, output_dir)) if output_dir else None
            with InferenceTimer(f"{model_name}_ocr") as timer:
                run_deepseek_ocr(remaining, on_progress=save_fn)
            timings.append(timer.result)

        # Final cache save (catches last batch)
        if output_dir:
            _save_ocr_cache(pages, ocr_key, output_dir)

    # NER step
    if config["ner"] == "spacy":
        with InferenceTimer(f"{model_name}_ner") as timer:
            ner_key = run_spacy_ner(pages, ocr_key)
        timings.append(timer.result)
    elif config["ner"] == "deberta":
        with InferenceTimer(f"{model_name}_ner") as timer:
            ner_key = run_deberta_ner(pages, ocr_key)
        timings.append(timer.result)
    elif config["ner"] == "qwen3":
        from .live import run_qwen3_ner
        with InferenceTimer(f"{model_name}_ner") as timer:
            run_qwen3_ner(pages, [ocr_key])
        timings.append(timer.result)
        ner_key = f"ner_{config['ocr']}"

    return ner_key, ocr_key, timings


# ── Evaluation ───────────────────────────────────────────────────────────

def evaluate_model(pages, model_name, ner_key, ocr_key):
    """Run all evaluations for a single model.

    Produces two evaluation lines per model:
      - "standard" (7 NER types): apples-to-apples comparison of off-the-shelf models
      - "full" (14 types): measures what fine-tuning unlocks (structural + document types)

    Pages must have been loaded with FULL_ENTITY_TYPES so gt_entities contains all 14 types.
    """
    from .static import evaluate_ner, evaluate_ocr_quality, per_type_analysis, per_entity_matches

    display_name = model_name

    # Build standard-filtered pages (only 7 NER types in gt_entities)
    standard_pages = []
    for page in pages:
        p = dict(page)
        p["gt_entities"] = [e for e in page["gt_entities"] if e["type"] in NER_ENTITY_TYPES]
        standard_pages.append(p)

    # ── Standard NER evaluation (7 types) ─────────────────────────
    ner_std = evaluate_ner(standard_pages, [(display_name, ner_key)])
    ner_std["entity_set"] = "standard"

    type_std, conf_std, mm_std = per_type_analysis(
        standard_pages, [(display_name, ner_key)], entity_types=NER_ENTITY_TYPES)
    type_std["entity_set"] = "standard"

    # ── Full document evaluation (14 types) ───────────────────────
    ner_full = evaluate_ner(pages, [(display_name, ner_key)])
    ner_full["entity_set"] = "full"

    type_full, conf_full, mm_full = per_type_analysis(
        pages, [(display_name, ner_key)], entity_types=FULL_ENTITY_TYPES)
    type_full["entity_set"] = "full"

    # ── OCR evaluation (same regardless of entity set) ────────────
    ocr_df = evaluate_ocr_quality(pages, [(display_name, ocr_key)])

    # ── Per-entity match results (standard types) ────────────────
    entity_df = per_entity_matches(standard_pages, [(display_name, ner_key)])

    return {
        "model": model_name,
        "ner_df": pd.concat([ner_std, ner_full], ignore_index=True),
        "ocr_df": ocr_df,
        "type_df": pd.concat([type_std, type_full], ignore_index=True),
        "confusion": {"standard": conf_std, "full": conf_full},
        "mismatches": {"standard": mm_std, "full": mm_full},
        "entity_matches": entity_df,
    }


# ── OCR-only command ─────────────────────────────────────────────────────

def _preflight(work_dir: Path, need_cache: str = None):
    """Check required directory structure. Exit with clear message if missing."""
    errors = []
    if not work_dir.exists():
        errors.append(f"Directory does not exist: {work_dir}")
    else:
        images_dir = work_dir / "images"
        labels_dir = work_dir / "labels"
        if not images_dir.exists():
            errors.append(f"Missing: {images_dir}")
        elif not any(images_dir.glob("*.jpg")):
            errors.append(f"No .jpg files in {images_dir}")
        if not labels_dir.exists():
            errors.append(f"Missing: {labels_dir}")
        elif not any(labels_dir.glob("*.json")):
            errors.append(f"No .json files in {labels_dir}")
        if need_cache:
            cache_path = _ocr_cache_path(work_dir, need_cache)
            if not cache_path.exists():
                errors.append(f"Missing OCR cache: {cache_path.name}")
                errors.append(f"  Run: python -m synthdoc.bench ocr --dir {work_dir} --engine {need_cache.replace('_ocr', '')}")
    if errors:
        print("Preflight check failed:")
        for e in errors:
            print(f"  ✗ {e}")
        print(f"\nExpected layout:")
        print(f"  {work_dir}/")
        print(f"  ├── images/*.jpg")
        print(f"  ├── labels/*.json")
        if need_cache:
            print(f"  └── _cache_{need_cache}.json")
        sys.exit(1)


def ocr_main(args):
    """Run OCR engine only and save cache. No NER, no evaluation."""
    work_dir = Path(args.dir)
    engine = args.engine
    ocr_key = f"{engine}_ocr"

    _preflight(work_dir)

    print(f"Loading data from {work_dir}...")
    pages = load_ner_pages(work_dir, entity_types=FULL_ENTITY_TYPES)
    if args.limit:
        pages = pages[:args.limit]
    print(f"  {len(pages)} pages loaded")

    # Load existing cache
    _load_ocr_cache(pages, ocr_key, work_dir)
    remaining = [p for p in pages if ocr_key not in p]

    if not remaining:
        print(f"All {len(pages)} pages already cached. Nothing to do.")
        return

    print(f"Running {engine} OCR on {len(remaining)}/{len(pages)} pages...")

    if engine == "tesseract":
        with InferenceTimer(f"{engine}_ocr") as timer:
            run_tesseract_ocr(remaining)
    elif engine == "qwenvl":
        from .live import run_qwenvl_ocr
        with InferenceTimer(f"{engine}_ocr") as timer:
            run_qwenvl_ocr(remaining)
    elif engine == "deepseek":
        from .live import run_deepseek_ocr
        save_fn = lambda: _save_ocr_cache(pages, ocr_key, work_dir)
        with InferenceTimer(f"{engine}_ocr") as timer:
            run_deepseek_ocr(remaining, on_progress=save_fn)

    _save_ocr_cache(pages, ocr_key, work_dir)
    valid = sum(1 for p in pages if ocr_key in p)
    print(f"Done. {valid}/{len(pages)} pages cached.")


# ── NER-only command ─────────────────────────────────────────────────────

def ner_main(args):
    """Run NER on cached OCR output, then evaluate and save results."""
    work_dir = Path(args.dir)
    ner_engine = args.engine
    ocr_engine = args.ocr
    ocr_key = f"{ocr_engine}_ocr"
    model_name = f"{ocr_engine}_{ner_engine}"

    _preflight(work_dir, need_cache=ocr_key)

    print(f"Loading data from {work_dir}...")
    pages = load_ner_pages(work_dir, entity_types=FULL_ENTITY_TYPES)
    if args.limit:
        pages = pages[:args.limit]
    print(f"  {len(pages)} pages loaded")

    # Load OCR cache
    _load_ocr_cache(pages, ocr_key, work_dir)
    cached = sum(1 for p in pages if ocr_key in p)
    if cached < len(pages):
        print(f"Warning: only {cached}/{len(pages)} pages have OCR cache. Missing pages will be skipped.")
        pages = [p for p in pages if ocr_key in p]

    # Run NER
    print(f"Running {ner_engine} NER on {len(pages)} pages...")
    if ner_engine == "spacy":
        with InferenceTimer(f"{model_name}_ner") as timer:
            ner_key = run_spacy_ner(pages, ocr_key)
    elif ner_engine == "deberta":
        with InferenceTimer(f"{model_name}_ner") as timer:
            ner_key = run_deberta_ner(pages, ocr_key)
    elif ner_engine == "qwen3":
        from .live import run_qwen3_ner
        with InferenceTimer(f"{model_name}_ner") as timer:
            run_qwen3_ner(pages, [ocr_key])
        ner_key = f"ner_{ocr_engine}"

    # Evaluate
    print(f"\nEvaluating {model_name}...")
    results = evaluate_model(pages, model_name, ner_key, ocr_key)

    # Save results (reuse merge logic from main)
    current_model_names = {model_name}

    def _merge(path, new_df, key_col):
        if path.exists():
            old = pd.read_csv(path)
            if key_col not in old.columns:
                return new_df
            old = old[~old[key_col].isin(current_model_names)]
            return pd.concat([old, new_df], ignore_index=True)
        return new_df

    ner_df = results["ner_df"]
    ner_df = _merge(work_dir / "benchmark_results.csv", ner_df, "ocr")
    ner_df.to_csv(work_dir / "benchmark_results.csv", index=False)
    print(f"NER results saved to {work_dir / 'benchmark_results.csv'}")

    if not results["ocr_df"].empty:
        ocr_df = _merge(work_dir / "ocr_results.csv", results["ocr_df"], "ocr")
        ocr_df.to_csv(work_dir / "ocr_results.csv", index=False)
        print(f"OCR results saved to {work_dir / 'ocr_results.csv'}")

    type_df = _merge(work_dir / "per_type_results.csv", results["type_df"], "ocr")
    type_df.to_csv(work_dir / "per_type_results.csv", index=False)
    print(f"Per-type results saved to {work_dir / 'per_type_results.csv'}")

    entity_df = results["entity_matches"]
    if not entity_df.empty:
        entity_df = _merge(work_dir / "entity_matches.csv", entity_df, "model")
        entity_df.to_csv(work_dir / "entity_matches.csv", index=False)
        print(f"Per-entity matches saved to {work_dir / 'entity_matches.csv'}")

    # Print summary
    for eset in ["standard", "full"]:
        subset = results["ner_df"][results["ner_df"]["entity_set"] == eset]
        if subset.empty:
            continue
        n_types = 7 if eset == "standard" else 14
        print(f"\n  {model_name} [{eset} — {n_types} types]:")
        print(f"    Strict F1:  {subset['strict_f1'].mean():.4f}")
        print(f"    Relaxed F1: {subset['relaxed_f1'].mean():.4f}")

    print(f"\nDone.")


# ── Main (legacy full pipeline) ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run SDS benchmark on generated dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m synthdoc.bench.run_benchmark --data ./data --output results/
  python -m synthdoc.bench.run_benchmark --data ./data --output results/ --model tesseract_spacy
  python -m synthdoc.bench.run_benchmark --data ./data --output results/ --model qwenvl_qwen3 --model deepseek_qwen3
  python -m synthdoc.bench.run_benchmark --data ./data --output results/ --limit 50
        """,
    )
    parser.add_argument("--data", type=str, required=True, help="Path to data/ directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory for results")
    parser.add_argument("--model", type=str, action="append", dest="models",
                        choices=list(MODELS.keys()),
                        help="Model(s) to run (default: all). Can specify multiple times.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of pages to process (for testing)")
    args = parser.parse_args()

    data_dir = Path(args.data)
    output_dir = Path(args.output)

    # ── Validate inputs ──────────────────────────────────────────────
    if not data_dir.exists():
        print(f"Error: data directory does not exist: {data_dir}")
        sys.exit(1)

    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        print(f"Error: expected data/images/ and data/labels/ in {data_dir}")
        print(f"  images/: {'found' if images_dir.exists() else 'MISSING'}")
        print(f"  labels/: {'found' if labels_dir.exists() else 'MISSING'}")
        sys.exit(1)

    # ── Select models ────────────────────────────────────────────────
    model_names = args.models if args.models else list(MODELS.keys())

    # ── Check all dependencies upfront ───────────────────────────────
    print("Checking dependencies...")
    dep_errors = check_dependencies(model_names)
    if dep_errors:
        print(f"\nDependency errors ({len(dep_errors)}):")
        for err in dep_errors:
            print(f"  ✗ {err}")
        print("\nFix the errors above or select different models with --model.")
        print(f"Available models: {', '.join(MODELS.keys())}")
        sys.exit(1)
    print(f"All dependencies OK for: {', '.join(model_names)}\n")

    # ── Load data ────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_dir}...")
    pages = load_ner_pages(data_dir, entity_types=FULL_ENTITY_TYPES)

    if not pages:
        print(f"Error: no pages found in {data_dir}. Check that images/ and labels/ have matching files.")
        sys.exit(1)

    if args.limit:
        pages = pages[:args.limit]
        print(f"Limited to {len(pages)} pages (--limit {args.limit})")

    print(f"Loaded {len(pages)} pages")

    # ── Dataset stats ────────────────────────────────────────────────
    from .static import dataset_stats
    stats = dataset_stats(pages)
    print(f"  Documents: {stats['n_docs']}")
    print(f"  Templates: {stats['n_templates']}")
    print(f"  Entity types: {len(stats['type_counts'])}")
    print(f"  Total entities: {sum(stats['type_counts'].values())}")
    print()

    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ── Run models ───────────────────────────────────────────────────
    all_ner_dfs = []
    all_ocr_dfs = []
    all_type_dfs = []
    all_entity_dfs = []
    all_timings = []
    all_confusion = {"standard": {}, "full": {}}
    all_mismatches = {"standard": {}, "full": {}}
    model_configs = []  # (model_name, ocr_key, ner_key) for cross-model analyses
    failed_models = []

    for model_name in model_names:
        print(f"{'=' * 60}")
        print(f"Running: {model_name} ({MODELS[model_name]['description']})")
        print(f"{'=' * 60}")

        try:
            ner_key, ocr_key, timings = run_model_pipeline(model_name, pages, output_dir)
            all_timings.extend(timings)

            results = evaluate_model(pages, model_name, ner_key, ocr_key)
            all_ner_dfs.append(results["ner_df"])
            if not results["ocr_df"].empty:
                all_ocr_dfs.append(results["ocr_df"])
            all_type_dfs.append(results["type_df"])
            all_entity_dfs.append(results["entity_matches"])
            model_configs.append((model_name, ocr_key, ner_key))
            for eset in ["standard", "full"]:
                all_confusion[eset].update(results["confusion"][eset])
                all_mismatches[eset].update(results["mismatches"][eset])

            # Print summary — two lines per model
            ner_df = results["ner_df"]
            for eset in ["standard", "full"]:
                subset = ner_df[ner_df["entity_set"] == eset]
                n_types = 7 if eset == "standard" else 14
                print(f"\n  {model_name} [{eset} — {n_types} types]:")
                print(f"    Strict F1:  {subset['strict_f1'].mean():.4f}")
                print(f"    Relaxed F1: {subset['relaxed_f1'].mean():.4f}")
            if not results["ocr_df"].empty:
                ocr_df = results["ocr_df"]
                print(f"  {model_name} [OCR]:")
                print(f"    WER: {ocr_df['wer'].mean():.4f}")
                print(f"    CER: {ocr_df['cer'].mean():.4f}")
            print()

        except Exception as e:
            import traceback
            print(f"\n  FAILED: {model_name}")
            print(f"  Error: {e}")
            traceback.print_exc()
            failed_models.append(model_name)
            # Don't continue silently — re-raise if this was the only model
            if len(model_names) == 1:
                sys.exit(1)
            print(f"  Continuing with remaining models...\n")

    # ── Helper: merge new results with existing CSV ─────────────────
    # Models run in this session replace their old rows; other models' rows are kept.
    current_model_names = set(m for m, _, _ in model_configs)

    def _merge_csv(path: Path, new_df: pd.DataFrame, key_col: str) -> pd.DataFrame:
        """Merge new_df into existing CSV, replacing rows for current models."""
        if path.exists():
            old = pd.read_csv(path)
            if key_col not in old.columns:
                return new_df  # incompatible old format, replace entirely
            old = old[~old[key_col].isin(current_model_names)]
            return pd.concat([old, new_df], ignore_index=True)
        return new_df

    def _merge_json_dict(path: Path, new_data: dict) -> dict:
        """Merge new JSON dict into existing, replacing keys for current models."""
        if path.exists():
            with open(path) as f:
                old = json.load(f)
            # Remove entries for models we just ran
            for key in list(old.keys()):
                if any(m in key for m in current_model_names):
                    del old[key]
            old.update(new_data)
            return old
        return new_data

    # ── Save OCR text outputs (side-by-side comparison) ─────────────
    if model_configs:
        ocr_keys_used = sorted(set(okey for _, okey, _ in model_configs))
        # Load existing OCR outputs and merge new columns
        ocr_out_path = output_dir / "ocr_outputs.csv"
        if ocr_out_path.exists():
            ocr_text_df = pd.read_csv(ocr_out_path)
        else:
            ocr_text_df = pd.DataFrame({
                "stem": [p["stem"] for p in pages],
                "template_type": [p["template_type"] for p in pages],
                "gt_text": [p.get("gt_text", "") for p in pages],
            })
        # Add/overwrite columns for each OCR engine run this session
        stem_to_idx = {row["stem"]: i for i, row in ocr_text_df.iterrows()}
        for okey in ocr_keys_used:
            if okey not in ocr_text_df.columns:
                ocr_text_df[okey] = ""
            else:
                # Ensure string dtype (CSV reload may infer float64 from NaN)
                ocr_text_df[okey] = ocr_text_df[okey].fillna("").astype(str)
            for page in pages:
                if page["stem"] in stem_to_idx:
                    ocr_text_df.at[stem_to_idx[page["stem"]], okey] = str(page.get(okey, ""))
        ocr_text_df.to_csv(ocr_out_path, index=False)
        print(f"OCR text outputs saved to {ocr_out_path}")

    # ── Save results (merge with existing) ────────────────────────────
    if all_ner_dfs:
        combined_ner = pd.concat(all_ner_dfs, ignore_index=True)
        combined_ner = _merge_csv(output_dir / "benchmark_results.csv", combined_ner, "ocr")
        combined_ner.to_csv(output_dir / "benchmark_results.csv", index=False)
        print(f"NER results saved to {output_dir / 'benchmark_results.csv'}")

    if all_ocr_dfs:
        combined_ocr = pd.concat(all_ocr_dfs, ignore_index=True)
        combined_ocr = _merge_csv(output_dir / "ocr_results.csv", combined_ocr, "ocr")
        combined_ocr.to_csv(output_dir / "ocr_results.csv", index=False)
        print(f"OCR results saved to {output_dir / 'ocr_results.csv'}")

    if all_type_dfs:
        combined_types = pd.concat(all_type_dfs, ignore_index=True)
        combined_types = _merge_csv(output_dir / "per_type_results.csv", combined_types, "ocr")
        combined_types.to_csv(output_dir / "per_type_results.csv", index=False)
        print(f"Per-type results saved to {output_dir / 'per_type_results.csv'}")

    if all_entity_dfs:
        combined_entities = pd.concat(all_entity_dfs, ignore_index=True)
        combined_entities = _merge_csv(output_dir / "entity_matches.csv", combined_entities, "model")
        combined_entities.to_csv(output_dir / "entity_matches.csv", index=False)
        print(f"Per-entity matches saved to {output_dir / 'entity_matches.csv'}")

    if all_timings:
        timings_path = output_dir / "timings.json"
        if timings_path.exists():
            with open(timings_path) as f:
                old_timings = json.load(f)
            # Keep timings from other models
            old_timings = [t for t in old_timings if not any(m in t.get("name", "") for m in current_model_names)]
            all_timings = old_timings + all_timings
        with open(timings_path, "w") as f:
            json.dump(all_timings, f, indent=2)
        print(f"Timings saved to {timings_path}")

    # ── Save confusion matrices (merge with existing) ───────────────
    if all_confusion["standard"] or all_confusion["full"]:
        confusion_json = {}
        for eset, conf in all_confusion.items():
            confusion_json[eset] = {}
            for ocr_name, pairs in conf.items():
                confusion_json[eset][ocr_name] = {
                    f"{gt}->{pred}": count
                    for (gt, pred), count in pairs.items()
                }
        # Merge with existing
        conf_path = output_dir / "confusion.json"
        if conf_path.exists():
            with open(conf_path) as f:
                old_conf = json.load(f)
            for eset in ["standard", "full"]:
                old_eset = old_conf.get(eset, {})
                # Remove entries for models we just ran
                for key in list(old_eset.keys()):
                    if key in current_model_names:
                        del old_eset[key]
                old_eset.update(confusion_json.get(eset, {}))
                confusion_json[eset] = old_eset
        with open(conf_path, "w") as f:
            json.dump(confusion_json, f, indent=2)
        print(f"Confusion matrices saved to {conf_path}")

    # ── Save top mismatches (merge with existing) ─────────────────
    if all_mismatches["standard"] or all_mismatches["full"]:
        from collections import Counter
        mm_json = {}
        for eset, mm_dict in all_mismatches.items():
            mm_json[eset] = {}
            for ocr_name, items in mm_dict.items():
                patterns = Counter(
                    f"{gt_t}->{pred_t}" for _, _, gt_t, pred_t in items
                )
                mm_json[eset][ocr_name] = {
                    "total": len(items),
                    "top_patterns": dict(patterns.most_common(20)),
                    "examples": [
                        {"gt_text": gt, "pred_text": pt, "gt_type": gt_t, "pred_type": pt_t}
                        for gt, pt, gt_t, pt_t in items[:50]
                    ],
                }
        # Merge with existing
        mm_path = output_dir / "mismatches.json"
        if mm_path.exists():
            with open(mm_path) as f:
                old_mm = json.load(f)
            for eset in ["standard", "full"]:
                old_eset = old_mm.get(eset, {})
                for key in list(old_eset.keys()):
                    if key in current_model_names:
                        del old_eset[key]
                old_eset.update(mm_json.get(eset, {}))
                mm_json[eset] = old_eset
        with open(mm_path, "w") as f:
            json.dump(mm_json, f, indent=2)
        print(f"Mismatches saved to {mm_path}")

    # ── Cross-model analyses ─────────────────────────────────────────
    if model_configs:
        from .static import (
            ocr_ner_error_attribution, calibration_analysis, compute_significance,
        )

        # Build standard-filtered pages for error attribution
        standard_pages = []
        for page in pages:
            p = dict(page)
            p["gt_entities"] = [
                e for e in page["gt_entities"] if e["type"] in NER_ENTITY_TYPES
            ]
            standard_pages.append(p)

        # Error attribution: was it OCR or NER that missed the entity?
        print("\nCross-model analyses:")
        attr_configs = [(name, okey, nkey) for name, okey, nkey in model_configs]
        attr_df = ocr_ner_error_attribution(standard_pages, attr_configs)
        if not attr_df.empty:
            attr_df = _merge_csv(output_dir / "error_attribution.csv", attr_df, "model")
            attr_df.to_csv(output_dir / "error_attribution.csv", index=False)
            print(f"  Error attribution saved to {output_dir / 'error_attribution.csv'}")

        # Calibration: confidence vs accuracy
        cal_configs = [(name, nkey) for name, _, nkey in model_configs]
        cal_data = calibration_analysis(standard_pages, cal_configs)
        cal_json = {
            k: v.to_dict(orient="records") for k, v in cal_data.items() if not v.empty
        }
        if cal_json:
            cal_json = _merge_json_dict(output_dir / "calibration.json", cal_json)
            with open(output_dir / "calibration.json", "w") as f:
                json.dump(cal_json, f, indent=2)
            print(f"  Calibration saved to {output_dir / 'calibration.json'}")

        # Significance: pairwise bootstrap test (standard entity set)
        if len(model_configs) >= 2 and all_ner_dfs:
            std_ner = combined_ner[combined_ner["entity_set"] == "standard"]
            sig_results = {}
            for i in range(len(model_configs)):
                for j in range(i + 1, len(model_configs)):
                    name_a = model_configs[i][0]
                    name_b = model_configs[j][0]
                    try:
                        sig = compute_significance(std_ner, name_a, name_b)
                        sig_results[f"{name_a}_vs_{name_b}"] = sig
                    except Exception:
                        pass
            if sig_results:
                sig_results = _merge_json_dict(output_dir / "significance.json", sig_results)
                with open(output_dir / "significance.json", "w") as f:
                    json.dump(sig_results, f, indent=2)
                print(f"  Significance tests saved to {output_dir / 'significance.json'}")

    # ── Summary ──────────────────────────────────────────────────────
    if all_ner_dfs:
        print(f"\n{'=' * 60}")
        print("BENCHMARK SUMMARY")
        print(f"{'=' * 60}")
        for eset in ["standard", "full"]:
            subset = combined_ner[combined_ner["entity_set"] == eset]
            if subset.empty:
                continue
            n_types = 7 if eset == "standard" else 14
            print(f"\n  {eset.upper()} ({n_types} entity types):")
            summary = subset.groupby("ocr").agg({
                "strict_f1": "mean",
                "relaxed_f1": "mean",
                "strict_precision": "mean",
                "strict_recall": "mean",
            }).round(4)
            print(summary.to_string(index=True))

    if failed_models:
        print(f"\nFAILED MODELS: {', '.join(failed_models)}")
        sys.exit(1)

    if not all_ner_dfs:
        print("\nError: no models produced results.")
        sys.exit(1)

    print()


if __name__ == "__main__":
    main()
