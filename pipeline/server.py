"""Generator gallery: press a button, get 16 fresh samples from DocPipeline, preview each one.

    .venv-pipeline\\Scripts\\python pipeline\\server.py      then open http://localhost:8765

Samples are made in parallel worker processes (each keeps its own SynthDoc browser and warmed-up Augraphy)
and appear on the page one by one as they finish. Every run is kept in pipeline_out/run_<time>_<port>/.
"""
import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
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
    from pipeline.doc_pipeline import warm_up
    warm_up()


def make_sample(out_dir, seed, name, options):
    """Runs in a worker process: one DocPipeline sample."""
    from pipeline.doc_pipeline import DocPipeline
    from pipeline.llm_templates import LLMTemplates
    pipe = DocPipeline(out_dir, languages=CHOICES["language"][options["language"]],
                       effects=CHOICES["effect"][options["effect"]], sources=CHOICES["source"][options["source"]],
                       templates=options["templates"], llm=LLMTemplates.from_env() if options["templates"] == "llm" else None,
                       text=CHOICES["text"][options["text"]], strengths=options["strengths"],
                       tashkeel=CHOICES["tashkeel"][options["tashkeel"]], tasks=CHOICES["task"][options["task"]])
    return pipe.sample(seed, name)


def start_job(options):
    n = max(1, min(int(options.get("n", 16)), 64))
    strengths = {}
    for tool in ("ocrsmith", "augraphy", "scan-simulator"):  # 0-100 on the page, 0-1 for the pipeline
        try:
            strengths[tool] = min(1.0, max(0.0, float((options.get("strengths") or {}).get(tool, 50)) / 100))
        except (TypeError, ValueError):
            strengths[tool] = 0.5
    options = {k: options.get(k) if options.get(k) in CHOICES[k] else next(iter(CHOICES[k])) for k in CHOICES} | \
              {"templates": "llm" if options.get("templates") == "llm" else "bank", "strengths": strengths}
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
    return sorted(images.values(), key=lambda e: (e["kind"] != "reference", e["image"]))


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
        if url.path.startswith("/probe/"):  # the probe's input images
            path = (PROBE / "inputs" / url.path[len("/probe/"):]).resolve()
            if (PROBE / "inputs").resolve() not in path.parents or not path.is_file():
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", PORT)))
    PORT = ap.parse_args().port
    try:
        httpd = Server(("127.0.0.1", PORT), Handler)
    except OSError:
        raise SystemExit(f"port {PORT} is in use (is the generator already running?); try --port {PORT + 1}")
    POOL = ProcessPoolExecutor(WORKERS, initializer=worker_init)
    for _ in range(WORKERS):  # start the workers (and Augraphy's warm-up) before the first click
        POOL.submit(time.sleep, 0)
    print(f"generator gallery: http://localhost:{PORT}  (workers: {WORKERS}, output: {OUT})", flush=True)
    httpd.serve_forever()
