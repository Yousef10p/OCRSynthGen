"""python -m pipeline --n 16 [--out pipeline_out/cli] [--seed 1] [--templates bank|llm]"""
import argparse
import json
import sys

from pipeline.doc_pipeline import ROOT, DocPipeline
from pipeline.llm_templates import LLMTemplates

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--out", default=str(ROOT / "pipeline_out" / "cli"))
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--templates", choices=["bank", "llm"], default="bank")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
pipe = DocPipeline(a.out, templates=a.templates, llm=LLMTemplates.from_env() if a.templates == "llm" else None)
pipe.run(a.n, a.seed, on_sample=lambda s: print(json.dumps(s, ensure_ascii=False)))
