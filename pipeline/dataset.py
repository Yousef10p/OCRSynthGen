"""Make a training set with DocPipeline: many samples in parallel, an index, a train/val split.

    .venv-pipeline\\Scripts\\python -m pipeline.dataset --n 10000 --out datasets/ocr_v1 --workers 6

Writes to --out:
    images/<id>.jpg      what the model is shown: the page (after effects, or clean), or a block or table cut out of it
    prompts/<id>.prompt.txt   the Surya OCR 2.5 prompt the sample is made for
    targets/<id>.target.html|.json   Surya's answer to it (see pipeline/surya.py)
    pages/<id>_page.jpg  the whole page, for samples whose image is a cut-out
    labels/<id>.json     word, line and slot boxes [x, y, w, h] in image pixels, text, and how the page was made
    markdown/<id>.md     the ground truth: one <text>... <bbox>x1, y1, x2, y2</bbox></text> line per text block
    html/<id>.html       the filled page
    manifest.jsonl       one line per sample: id, split, files, language, template, effects, word count, ...
    dataset.json         how the set was made (options, counts, sources and licences)

Interrupted? Run the same command again: samples already in manifest.jsonl are kept and the rest are made.
Every sample has its own seed (--seed * 1,000,000 + its number), so a set can be made again exactly.
About 2-4 s per sample per worker; each worker needs about 1.5 GB of memory.
"""
import argparse
import json
import os
import shutil
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FOLDERS = {".jpg": "images", ".prompt.txt": "prompts", ".target.html": "targets", ".target.json": "targets",
           "_page.jpg": "pages", ".json": "labels", ".md": "markdown", ".html": "html"}
_pipe = None


def init(out, options):
    global _pipe
    from pipeline.doc_pipeline import DocPipeline, warm_up
    warm_up()
    _pipe = DocPipeline(Path(out) / "_work" / str(os.getpid()), extras=False, **options)


def make(seed, name, out, val):
    """One sample, moved from the worker's folder into images/, labels/, markdown/ and html/."""
    summary = _pipe.sample(seed, name)
    files = {}
    for ext, folder in FOLDERS.items():
        if not (_pipe.out / f"{name}{ext}").exists():  # the page image only for cut-out tasks, one target kind
            continue
        target = Path(out) / folder / f"{name}{ext}"
        shutil.move(str(_pipe.out / f"{name}{ext}"), target)
        files[folder] = f"{folder}/{name}{ext}"
    # Split by a hash of the name: the same sample is always in the same split
    split = "val" if zlib.crc32(name.encode()) % 10_000 < val * 10_000 else "train"
    return {"id": name, "split": split, **files, **summary}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True, help="samples in the set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val", type=float, default=0.05, help="share of samples in the val split")
    ap.add_argument("--language", choices=["ar", "en"], help="only templates of this language (default: both)")
    ap.add_argument("--source", choices=["grammar", "websight", "websight2", "doclaynet"],
                    help="only templates from this source (default: all four equally)")
    for tool in ("ocrsmith", "augraphy", "scan-simulator"):
        ap.add_argument(f"--strength-{tool}", type=float, default=0.5,
                        help=f"{tool} strength, 0 (lightest) to 1 (strongest); 0.5 = the tool's own settings")
    ap.add_argument("--effect", choices=["clean", "ocrsmith", "augraphy", "scan-simulator",
                                         "augraphy+scan-simulator"], help="only this effect (default: the mix)")
    ap.add_argument("--text", choices=["wikipedia", "demo"], default=None)
    ap.add_argument("--task", choices=["full_ocr", "layout", "block_ocr", "table"],
                    help="the Surya OCR 2.5 prompt every sample is made for (default: 40%% full_ocr, 20%% layout, "
                         "25%% block_ocr, 15%% table)")
    ap.add_argument("--tashkeel", choices=["none", "partial", "full"],
                    help="Arabic diacritics on every Arabic page at this level (default: 40%% none, 30%% partial, 30%% full)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(a.out).resolve()
    for folder in FOLDERS.values():
        (out / folder).mkdir(parents=True, exist_ok=True)
    options = {"languages": {a.language: 1} if a.language else None, "sources": {a.source: 1} if a.source else None,
               "strengths": {t: getattr(a, f"strength_{t.replace('-', '_')}") for t in ("ocrsmith", "augraphy", "scan-simulator")},
               "effects": {a.effect: 1} if a.effect else None, "text": a.text,
               "tashkeel": {a.tashkeel: 1} if a.tashkeel else None, "tasks": {a.task: 1} if a.task else None}

    manifest = out / "manifest.jsonl"
    done = set()
    if manifest.exists():
        done = {json.loads(line)["id"] for line in manifest.read_text(encoding="utf-8").splitlines() if line}
    todo = [(a.seed * 1_000_000 + i, f"{i:07d}") for i in range(a.n) if f"{i:07d}" not in done]
    print(f"{len(done)} samples already made, {len(todo)} to make, {a.workers} workers -> {out}", flush=True)
    start, made, failed = time.time(), 0, 0
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(str(out), options)) as pool, \
            manifest.open("a", encoding="utf-8") as index:
        futures = {pool.submit(make, seed, name, str(out), a.val): name for seed, name in todo}
        for future in as_completed(futures):
            try:
                index.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
                index.flush()
                made += 1
            except Exception as e:  # reported and left for the next run, which makes it again
                failed += 1
                print(f"  {futures[future]} failed: {type(e).__name__}: {e}", flush=True)
            if (made + failed) % 50 == 0 or made + failed == len(todo):
                rate = (made + failed) / (time.time() - start)
                print(f"  {made + failed}/{len(todo)} ({failed} failed), {rate:.1f} samples/s", flush=True)
    shutil.rmtree(out / "_work", ignore_errors=True)

    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    count = lambda key: {v: sum(r[key] == v for r in rows) for v in sorted({r[key] for r in rows})}
    (out / "dataset.json").write_text(json.dumps({
        "samples": len(rows), "splits": count("split"), "languages": count("lang"), "effects": count("effect"),
        "template_sources": count("source"), "options": vars(a),
        "text": "Wikipedia (ar, en), CC BY-SA 4.0: each label lists its articles; a shared set must keep "
                "CC BY-SA and the attribution",
        "pictures": "img_pool/: Unsplash photos via picsum.photos (Unsplash License) and Wikimedia Commons quality "
                    "images (each file's licence and author in img_pool/sources.json)",
        "templates": "bank2000/: Grammar (own), WebSight v0.1 and v0.2 (CC BY 4.0), DocLayNet v1.1 (CDLA-Permissive-1.0)",
    }, indent=1), encoding="utf-8")
    print(f"done: {len(rows)} samples in {out} ({failed} failed this run; run again to make them)")


if __name__ == "__main__":
    main()
