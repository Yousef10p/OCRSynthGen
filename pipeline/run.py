"""Make a training set with the generator (many samples in parallel, an index, a train/val split).

    python -m pipeline.run dataset --n 1000 --out datasets\\v1 --workers 4 --seed 1

(`python -m pipeline.run website` opens the local gallery, pipeline/website.py, where that file exists.)
"""
import argparse
import importlib.util
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


# ====================================================================================================================
# dataset: many pages in parallel, an index, a train/val split
# ====================================================================================================================
# Make a training set with DocPipeline: many samples in parallel, an index, a train/val split.
#
#     .venv-pipeline\\Scripts\\python -m pipeline.run dataset --n 10000 --out datasets/ocr_v1 --workers 6
#
# Writes to --out:
#     images/<id>.jpg      what the model is shown: the page (after effects, or clean), or a block or table cut out of it
#     prompts/<id>.prompt.txt   the Surya OCR 2.5 prompt the sample is made for
#     targets/<id>.target.txt|.json   Surya's answer to it (STEP 5 in pipeline/generator.py)
#     pages/<id>_page.jpg  the whole page, for samples whose image is a cut-out
#     labels/<id>.json     word, line and slot boxes [x, y, w, h] in image pixels, text, and how the page was made
#     markdown/<id>.md     the ground truth: one <text>... <bbox>x1, y1, x2, y2</bbox></text> line per text block
#     html/<id>.html       the filled page
#     manifest.jsonl       one line per sample: id, split, files, language, template, effects, word count, ...
#     dataset.json         how the set was made (options, counts, sources and licences)
#
# Interrupted? Run the same command again: samples already in manifest.jsonl are kept and the rest are made.
# Every sample has its own seed (--seed * 1,000,000 + its number), so a set can be made again exactly.
# About 2-4 s per sample per worker; each worker needs about 1.5 GB of memory.

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FOLDERS = {".jpg": "images", ".prompt.txt": "prompts", ".target.txt": "targets", ".target.json": "targets",
           "_page.jpg": "pages", ".json": "labels", ".md": "markdown", ".html": "html"}
_pipe = None


def init(out, options):
    global _pipe
    from pipeline.generator import DocPipeline, warm_up
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


def dataset(argv):
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
    ap.add_argument("--text", choices=["pools", "wikipedia", "demo"], default=None)
    ap.add_argument("--task", choices=["full_ocr", "layout", "block_ocr", "table", "full_ocr+block_ocr"],
                    help="the Surya OCR 2.5 prompt every sample is made for (default: 40%% full_ocr, 25%% layout, "
                         "35%% block_ocr; table only when asked); full_ocr+block_ocr: only those two, 40:35")
    ap.add_argument("--tashkeel", choices=["none", "partial", "full"],
                    help="Arabic diacritics on every Arabic page at this level (default: 40%% none, 30%% partial, 30%% full)")
    ap.add_argument("--colour-share", type=float, default=0.5,
                    help="share of pages in colour, 0 to 1 (the rest light: white, off-white or a light tint)")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(a.out).resolve()
    for folder in FOLDERS.values():
        (out / folder).mkdir(parents=True, exist_ok=True)
    options = {"languages": {a.language: 1} if a.language else None, "sources": {a.source: 1} if a.source else None,
               "strengths": {t: getattr(a, f"strength_{t.replace('-', '_')}") for t in ("ocrsmith", "augraphy", "scan-simulator")},
               "colour_share": a.colour_share,
               "effects": {a.effect: 1} if a.effect else None, "text": a.text,
               "tashkeel": {a.tashkeel: 1} if a.tashkeel else None, "tasks": ({"full_ocr": 0.4, "block_ocr": 0.35} if a.task == "full_ocr+block_ocr"
                         else {a.task: 1} if a.task else None)}

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
        "pictures": "img_pool/: captioned pictures from Flickr8k (research use), Polaris (BSD-3-Clause; COCO / nocaps "
                    "images) and PD12M (public domain / CC0); each picture's source and licence in "
                    "img_pool/captions.jsonl",
        "templates": "bank2000/: Grammar (own), WebSight v0.1 and v0.2 (CC BY 4.0), DocLayNet v1.1 (CDLA-Permissive-1.0)",
    }, indent=1), encoding="utf-8")
    print(f"done: {len(rows)} samples in {out} ({failed} failed this run; run again to make them)")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "website":
        if importlib.util.find_spec("pipeline.website") is None:
            raise SystemExit("the website (pipeline/website.py) is not part of this copy; use generate.py or the dataset command")
        from pipeline.website import website
        website(sys.argv[2:])
    elif len(sys.argv) >= 2 and sys.argv[1] == "dataset":
        dataset(sys.argv[2:])
    else:
        raise SystemExit(__doc__)
