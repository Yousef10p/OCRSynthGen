"""Run the generator: make a training set, or open the website to look at pages one by one.

    python -m pipeline.run dataset --n 1000 --out datasets\\v1 --workers 4 --seed 1
    python -m pipeline.run website                     (then open http://localhost:8765)
"""
import argparse
import json
import os
import random
import shutil
import sys
import threading
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

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
    ap.add_argument("--text", choices=["wikipedia", "demo"], default=None)
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




# ====================================================================================================================
# website: Generate 16 samples, preview each one
# ====================================================================================================================
# Generator gallery: press a button, get 16 fresh samples from DocPipeline, preview each one.
#
#     .venv-pipeline\\Scripts\\python pipeline\\server.py      then open http://localhost:8765
#
# Samples are made in parallel worker processes (each keeps its own SynthDoc browser and warmed-up Augraphy)
# and appear on the page one by one as they finish. Every run is kept in pipeline_out/run_<time>_<port>/.

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "pipeline_out"
STATIC = Path(__file__).resolve().parent / "static"
PROBE = ROOT / "surya_probe"  # the Surya OCR 2.5 tests (surya_probe/probe.py), shown on probe.html
PORT, WORKERS = 8765, 4
CHOICES = {
    "language": {"mix": None, "ar": {"ar": 1}, "en": {"en": 1}},
    "effect": {"mix": None, "clean": {"clean": 1}, "ocrsmith": {"ocrsmith": 1}, "augraphy": {"augraphy": 1},
               "scan-simulator": {"scan-simulator": 1}, "augraphy+scan-simulator": {"augraphy+scan-simulator": 1}},
    "source": {"mix": None, "grammar": {"grammar": 1}, "websight": {"websight": 1}, "websight2": {"websight2": 1},
               "doclaynet": {"doclaynet": 1}},
    "text": {"wikipedia": "wikipedia", "demo": "demo"},
    "tashkeel": {"mix": None, "none": {"none": 1}, "partial": {"partial": 1}, "full": {"full": 1}},
    "task": {"mix": None, "full_ocr": {"full_ocr": 1}, "layout": {"layout": 1}, "block_ocr": {"block_ocr": 1},
             "table": {"table": 1}},
}
JOBS, LOCK = {}, threading.Lock()


def worker_init():
    from pipeline.generator import warm_up
    warm_up()


def make_sample(out_dir, seed, name, options):
    """Runs in a worker process: one DocPipeline sample."""
    from pipeline.generator import DocPipeline
    from pipeline.llm_templates import LLMTemplates
    pipe = DocPipeline(out_dir, languages=CHOICES["language"][options["language"]],
                       effects=CHOICES["effect"][options["effect"]], sources=CHOICES["source"][options["source"]],
                       templates=options["templates"], llm=LLMTemplates.from_env() if options["templates"] == "llm" else None,
                       text=CHOICES["text"][options["text"]], strengths=options["strengths"],
                       tashkeel=CHOICES["tashkeel"][options["tashkeel"]], tasks=CHOICES["task"][options["task"]],
                       colour_share=options["colour_share"])
    return pipe.sample(seed, name)


def start_job(options):
    n = max(1, min(int(options.get("n", 16)), 64))
    strengths = {}
    for tool in ("ocrsmith", "augraphy", "scan-simulator"):  # 0-100 on the page, 0-1 for the pipeline
        try:
            strengths[tool] = min(1.0, max(0.0, float((options.get("strengths") or {}).get(tool, 50)) / 100))
        except (TypeError, ValueError):
            strengths[tool] = 0.5
    try:  # 0-100 on the page: the share of pages in colour
        colour_share = min(1.0, max(0.0, float(options.get("colour", 50)) / 100))
    except (TypeError, ValueError):
        colour_share = 0.5
    options = {k: options.get(k) if options.get(k) in CHOICES[k] else next(iter(CHOICES[k])) for k in CHOICES} | \
              {"templates": "llm" if options.get("templates") == "llm" else "bank", "strengths": strengths,
               "colour_share": colour_share}
    run = time.strftime("run_%Y%m%d_%H%M%S") + f"_{PORT}"  # two generators started in the same second don't mix
    out_dir, seed = OUT / run, random.randrange(10**6)
    job = {"id": run, "n": n, "options": options, "started": time.time(), "samples": [None] * n, "errors": []}
    with LOCK:
        JOBS[run] = job
    for i in range(n):
        future = POOL.submit(make_sample, str(out_dir), seed * 1000 + i, f"s{i + 1:02d}", options)
        future.add_done_callback(lambda f, i=i: finish(job, i, f))
    return job


def finish(job, i, future):
    with LOCK:
        try:
            job["samples"][i] = future.result()
        except Exception as e:  # shown on the card instead of a page
            job["samples"][i] = {"name": f"s{i + 1:02d}", "error": f"{type(e).__name__}: {e}"}
            job["errors"].append(i)


def probe_answers():
    """[{image, kind, label, lang, effect, size, ours, answers: {prompt: {status, seconds, finish_reason, tokens,
    content}}}] from the Surya probe: per image and prompt the run with the most tokens allowed ("_schema": asked
    with the JSON schema). ours: what our pipeline wrote for the same image (surya_probe/inputs/index.json)."""
    path = PROBE / "results.jsonl"
    if not path.exists():
        return []
    index = {}
    if (PROBE / "inputs" / "index.json").exists():
        index = {d["image"]: d for d in json.loads((PROBE / "inputs" / "index.json").read_text(encoding="utf-8"))}
    images = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        name = Path(r["image"]).name
        task = r["task"] + ("_schema" if r["schema"] else "")
        entry = images.setdefault(name, {"image": name, "size": r["size"], "answers": {},
                                         "kind": index.get(name, {}).get("kind", "reference"),
                                         "label": index.get(name, {}).get("label"),
                                         "lang": index.get(name, {}).get("lang"), "effect": index.get(name, {}).get("effect"),
                                         "ours": {k[5:]: v for k, v in index.get(name, {}).items() if k.startswith("ours_")}})
        old = entry["answers"].get(task)
        if old is None or r["max_tokens"] >= old["max_tokens"]:
            entry["answers"][task] = {"status": r["status"], "seconds": r["seconds"], "max_tokens": r["max_tokens"],
                                      "finish_reason": r.get("finish_reason"),
                                      "tokens": (r.get("usage") or {}).get("completion_tokens"),
                                      "content": r.get("content") if r["status"] == 200 else r.get("error")}
    return sorted(images.values(), key=lambda e: (e["kind"] != "reference", e["image"])) + batch_answers()


def batch_answers():
    """The same entries for every sample Surya answered in surya_probe/<batch>/<prompt>/surya.jsonl (batch100,
    clean100): its image, Surya's answer and our label (samples/<id>/our_label.* when there, else the target)."""
    entries = []
    for path in sorted(PROBE.glob("*/*/surya.jsonl")):
        folder = path.parent
        rows = {}
        if (folder / "manifest.jsonl").exists():
            rows = {r["id"]: r for r in map(json.loads, (folder / "manifest.jsonl").read_text(encoding="utf-8").splitlines()) if r}
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            ours = next((folder / "samples" / r["id"]).glob("our_label.*"), None) if (folder / "samples" / r["id"]).is_dir() else None
            ours = ours or folder / r["target"]
            entries.append({"image": f"{folder.parent.name}/{folder.name}/{r['image']}", "title": f"{folder.name}/{r['id']}",
                            "kind": f"{folder.parent.name}/{folder.name}", "lang": r.get("lang"),
                            "label": rows.get(r["id"], {}).get("source"), "effect": rows.get(r["id"], {}).get("effect"),
                            "ours": {r["task"]: ours.read_text(encoding="utf-8") if ours.is_file() else None},
                            "answers": {r["task"]: {"status": r["status"], "seconds": r.get("seconds"), "prompt": r.get("prompt"),
                                                    "finish_reason": r.get("finish_reason"),
                                                    "tokens": (r.get("usage") or {}).get("completion_tokens"),
                                                    "content": r.get("content") if r["status"] == 200 else r.get("error")}}})
    return entries


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, *args):
        pass

    def end_headers(self):
        if not self.path.startswith("/api/"):  # the page itself: always the current version after an update
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def send_json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/info":
            from pipeline.llm_templates import LLMTemplates
            llm = LLMTemplates.from_env()
            return self.send_json({"workers": WORKERS, "llm": llm.available(), "llm_name": llm.name,
                                   "llm_problem": llm.problem})
        if url.path == "/api/job":
            job = JOBS.get(parse_qs(url.query).get("id", [""])[0])
            if job is None:
                return self.send_json({"error": "no such job"}, 404)
            with LOCK:
                return self.send_json({**job, "elapsed": round(time.time() - job["started"], 1)})
        if url.path == "/api/probe":  # the Surya probe: every image and Surya's answers (surya_probe/results.jsonl)
            return self.send_json(probe_answers())
        if url.path.startswith("/probe/"):  # the probe's input images, and the batches' (surya_probe/<batch>/...)
            name = unquote(url.path[len("/probe/"):])
            path = (PROBE / ("" if "/" in name else "inputs") / name).resolve()
            if PROBE.resolve() not in path.parents or not path.is_file():
                return self.send_error(404)
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png" if path.suffix == ".png" else "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return self.wfile.write(data)
        if url.path.startswith("/out/"):  # generated files
            path = (OUT / url.path[len("/out/"):]).resolve()
            if OUT.resolve() not in path.parents or not path.is_file():
                return self.send_error(404)
            data = path.read_bytes()
            kind = {".jpg": "image/jpeg", ".json": "application/json", ".md": "text/plain", ".txt": "text/plain",
                    ".html": "text/plain"}.get(path.suffix, "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", kind + ("; charset=utf-8" if kind.startswith(("text", "application/json")) else ""))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return self.wfile.write(data)
        return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/generate":
            return self.send_error(404)
        options = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        job = start_job(options)
        self.send_json({"id": job["id"], "n": job["n"]})


class Server(ThreadingHTTPServer):
    # Python sets SO_REUSEADDR, which on Windows lets a second server take a port already in use (requests
    # then reach either one). Refuse instead.
    allow_reuse_address = False


def website(argv):
    global PORT, POOL
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", PORT)))
    PORT = ap.parse_args(argv).port
    try:
        httpd = Server(("127.0.0.1", PORT), Handler)
    except OSError:
        raise SystemExit(f"port {PORT} is in use (is the generator already running?); try --port {PORT + 1}")
    POOL = ProcessPoolExecutor(WORKERS, initializer=worker_init)
    for _ in range(WORKERS):  # start the workers (and Augraphy's warm-up) before the first click
        POOL.submit(time.sleep, 0)
    print(f"generator website: http://localhost:{PORT}  (workers: {WORKERS}, output: {OUT})", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("dataset", "website"):
        raise SystemExit(__doc__)
    (dataset if sys.argv[1] == "dataset" else website)(sys.argv[2:])
