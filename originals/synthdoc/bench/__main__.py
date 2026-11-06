"""
CLI entry point: python -m synthdoc.bench <command>

Commands:
    run    Run benchmark from a YAML config file
    list   Show available engines from engines.yaml
    ocr    Run OCR only (saves cache for later NER)
    ner    Run NER + evaluation using cached OCR output
    bench  Run full OCR→NER→evaluation pipeline (legacy)

Examples:
    python -m synthdoc.bench run  benchmarks/ner_comparison.yaml
    python -m synthdoc.bench run  benchmarks/ner_comparison.yaml --limit 50
    python -m synthdoc.bench list
    python -m synthdoc.bench ocr  --dir run01/ --engine deepseek
    python -m synthdoc.bench ner  --dir run01/ --engine qwen3 --ocr deepseek
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="synthdoc-bench",
        description="synthdoc benchmark: OCR and NER as independent steps",
    )
    sub = parser.add_subparsers(dest="command")

    # ── run (config-driven) ──────────────────────────────────────────
    run_p = sub.add_parser("run", help="Run benchmark from YAML config")
    run_p.add_argument("config", help="Path to run config YAML")
    run_p.add_argument("--limit", type=int, default=None,
                        help="Limit number of pages")
    run_p.add_argument("--force-ocr", action="store_true",
                        help="Re-run OCR even if cached")

    # ── list ─────────────────────────────────────────────────────────
    sub.add_parser("list", help="Show available engines from engines.yaml")

    # ── ocr ──────────────────────────────────────────────────────────
    ocr_p = sub.add_parser("ocr", help="Run OCR engine and save cache")
    ocr_p.add_argument("--dir", required=True,
                        help="Working directory (images/, labels/, caches, results)")
    ocr_p.add_argument("--engine", required=True,
                        choices=["tesseract", "qwenvl", "deepseek"],
                        help="OCR engine to run")
    ocr_p.add_argument("--limit", type=int, default=None,
                        help="Limit number of pages (for testing)")

    # ── ner ──────────────────────────────────────────────────────────
    ner_p = sub.add_parser("ner", help="Run NER on cached OCR output + evaluate")
    ner_p.add_argument("--dir", required=True,
                        help="Working directory (images/, labels/, caches, results)")
    ner_p.add_argument("--engine", required=True,
                        choices=["spacy", "deberta", "qwen3"],
                        help="NER engine to run")
    ner_p.add_argument("--ocr", required=True,
                        choices=["tesseract", "qwenvl", "deepseek"],
                        help="Which OCR cache to load")
    ner_p.add_argument("--limit", type=int, default=None,
                        help="Limit number of pages (for testing)")

    # ── bench (legacy) ───────────────────────────────────────────────
    bench_p = sub.add_parser("bench", help="Full OCR -> NER -> eval pipeline (original)")
    bench_p.add_argument("--dir", required=True,
                          help="Working directory (images/, labels/, caches, results)")
    bench_p.add_argument("--model", action="append", dest="models",
                          choices=["tesseract_spacy", "tesseract_deberta",
                                   "qwenvl_qwen3", "deepseek_qwen3"])
    bench_p.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        from .config import run_from_config
        run_from_config(args.config, limit=args.limit, force_ocr=args.force_ocr)
    elif args.command == "list":
        from .config import list_engines
        list_engines()
    elif args.command == "ocr":
        from .run_benchmark import ocr_main
        ocr_main(args)
    elif args.command == "ner":
        from .run_benchmark import ner_main
        ner_main(args)
    elif args.command == "bench":
        from .run_benchmark import main as bench_main
        # Patch sys.argv for legacy argparse
        argv = ["run_benchmark", "--data", args.dir, "--output", args.dir]
        if args.models:
            for m in args.models:
                argv += ["--model", m]
        if args.limit:
            argv += ["--limit", str(args.limit)]
        sys.argv = argv
        bench_main()


if __name__ == "__main__":
    main()
