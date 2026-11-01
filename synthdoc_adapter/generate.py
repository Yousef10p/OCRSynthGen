"""Render exported bank templates with SynthDoc, plus word/line OCR labels: writes synthdoc_out/.

Uses the SynthDoc install and Arabic extension in ../synthdoc-arabic-test (unchanged; run with its venv):

    ..\\synthdoc-arabic-test\\.venv\\Scripts\\python synthdoc_adapter\\generate.py --count 1

Output (SynthDoc's own format): images/, pdfs/, html/, labels/ (entity boxes per slot), overlays/,
ocr/ (word and line boxes), ocr_overlays/, manifest.json.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "synthdoc_out"
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=1, help="documents per template")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4, help="templates rendered in parallel (threads)")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.environ["BANK_SYNTHDOC_SEED"] = str(a.seed)
    shutil.rmtree(OUT, ignore_errors=True)

    import synthdoc
    from synthdoc.cli import generate
    from synthdoc_arabic import pipeline

    templates = HERE / "templates"
    names = sorted(d.name[: -len("_template")] for d in templates.iterdir() if d.name.endswith("_template"))
    synthdoc.add_template_dir(templates)
    pipeline.install()
    try:
        generate(templates=names, count=a.count, out_dir=OUT, keep_pdfs=True, parallel_workers=a.workers)
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
