"""Training pages on the fly: fresh, checked samples from the generator, in memory, for a training loop.

    from train import DataGenerator, generate_data

    if __name__ == "__main__":                       # (needed on Windows: the workers are new processes)
        batch = generate_data(mode="batch", num_samples=8)          # one batch, then the workers stop
        data = generate_data(mode="epoch", num_samples=1000, epoch=3)

        with DataGenerator(workers=4, language="ar") as gen:        # keep the workers for a whole training run
            for epoch in range(5):
                pages = gen.epoch(1000, epoch=epoch)                # an epoch's pages, the same for the same seed
                rows = [p.swift_row() for p in pages]                # ms-swift rows: messages + images
            batch = gen.batch(8)                                     # or: the next 8 pages of an endless stream

Every sample is made by DocPipeline.sample() (pipeline/generator.py: template, fill, render, effect, labels), with
its own render checks, then checked again as a finished training example (validate=True):
  format   pipeline.checking.problems(): the answer in Surya 2.5's compact / JSON format
  dataset  surya_probe/check_dataset.py sample_problems(): Surya's prompt, the encoder round trip, tags and
           attributes, block forms, picture blocks the pools make, no tashkeel in the answer, no lost words
  labels   surya_probe/check_labels.py check() + structure(): everything drawn is labelled and nothing else,
           inline tags, headings, nesting, lists, edges, footnotes, captions, sections, reading order
  boxes    surya_probe/check_boxes.py check(): every box on the page; on clean pages also no overlap, no empty
           box, no cut letters, no loose margin (on effect pages noise and tilt make those unreliable)
  length   image + prompt + answer tokens within max_length (Surya's tokenizer; skipped if it can't be loaded)
  surya    Surya's own code parses the answer as the probe page's Inference tab does (surya_probe/
           surya_pipeline.py, in a process of its own; needs the Surya clone in surya/)
A sample failing any check is made again from the next seed of its slot (same balanced block, seed % 9), up to
max_retries times; then GenerationError. Same seed, same options and pools: the same samples.
"""
from __future__ import annotations

import atexit
import io
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image

ROOT = Path(__file__).resolve().parent
CHECKS = ("format", "dataset", "labels", "boxes", "length", "surya")
CHOICES = {  # the website's choices (pipeline/run.py CHOICES); a dict of weights is taken as it is
    "language": ("ar", "en"),
    "effect": ("clean", "ocrsmith", "augraphy", "scan-simulator", "augraphy+scan-simulator"),
    "source": ("grammar", "websight", "websight2", "doclaynet"),
    "task": ("full_ocr", "layout", "block_ocr", "table"),
    "tashkeel": ("none", "partial", "full"),
}
REPLACE = 9 * 10**18  # a replacement's seed: the slot's seed + k * REPLACE (keeps seed % 9, the balanced block)


class GenerationError(RuntimeError):
    """A slot had no sample passing the checks after max_retries replacements (the reasons are in the message);
    .samples holds the request's samples that did pass."""

    def __init__(self, message: str, samples: list | None = None):
        super().__init__(message)
        self.samples = samples or []


@dataclass
class Sample:
    """One training example: the image the model is shown, Surya's prompt and our answer, and how it was made."""
    id: str
    seed: int
    task: str                  # full_ocr | layout | block_ocr | table
    prompt: str                # Surya's own prompt for the task
    target: str                # the answer (compact HTML for full_ocr / block_ocr, JSON for layout / table)
    image_bytes: bytes         # JPEG: the page after effects, or the block / table cut out of it
    label: dict                # the labels JSON: words, lines, slots, Surya blocks, page size, style...
    info: dict                 # DocPipeline.sample()'s summary: template, source, lang, effect, blocks...
    page_bytes: bytes | None = None   # JPEG of the whole page when the image is a cut-out
    html: str = ""             # the filled page
    markdown: str = ""         # one <text>... <bbox> line per text block
    checks: dict = field(default_factory=dict)  # checks run, rejected seeds and why

    @property
    def image(self) -> Image.Image:
        """The image as RGB (decoded on each call: keep it if you need it twice)."""
        with Image.open(io.BytesIO(self.image_bytes)) as im:
            return im.convert("RGB")

    def swift_row(self, image=None) -> dict:
        """The sample as an ms-swift row (as notebooks/Surya_finetune_swift_Colab.ipynb writes them): the user
        turn is <image> + Surya's prompt, the assistant turn our answer. image: a path to use instead of the
        image itself (a PIL image by default)."""
        return {"messages": [{"role": "user", "content": "<image>" + self.prompt},
                             {"role": "assistant", "content": self.target}],
                "images": [image if image is not None else self.image]}


# -- options -----------------------------------------------------------------------------------------------------

def _weights(value, name: str) -> dict | None:
    """"mix" (None: the pipeline's own mix), one choice, a list of choices (even), or {choice: weight}."""
    if value in (None, "mix", "mixed"):
        return None
    if isinstance(value, str):
        value = {value: 1.0}
    elif not isinstance(value, dict):
        value = {v: 1.0 for v in value}
    bad = [k for k in value if k not in CHOICES[name]]
    if bad or not value or any(w < 0 for w in value.values()) or not sum(value.values()):
        raise ValueError(f"{name}: {value!r} - choose from {list(CHOICES[name])}, 'mix', or {{choice: weight}}")
    return {k: float(w) for k, w in value.items()}


# -- the worker processes (each keeps its DocPipeline: one browser, warmed-up effects) -----------------------------

_W = {}


def _load(name: str):
    """A surya_probe script as a module (by path: the folder has scripts named like other modules)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"surya_probe_{name}", ROOT / "surya_probe" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _init_worker(work: str, pipe_options: dict, settings: dict) -> None:
    checks = settings["checks"]
    if "surya" in checks:  # Surya's own modules clash with the generator's copy: a process of their own, started
        # (and Surya loaded in it) before the browser: a process started while it runs never got its first task
        _W["surya"] = ProcessPoolExecutor(1, mp_context=get_context("spawn"))
        _W["surya"].submit(_surya_parse, "block_ocr", "<p>.</p>", _blank_jpeg()).result()
        from multiprocessing import util
        util.Finalize(None, _stop_helper, exitpriority=100)  # stopped with the worker (else its exit waits minutes)
    sys.path.insert(0, str(ROOT))
    import pipeline.generator as G
    G.ATTEMPTS, G.ORDER_TRIES = settings["render_attempts"], settings["order_tries"]
    if settings["low_dpi"] is not None:
        G.LOW_DPI = dict(settings["low_dpi"])
    G.warm_up()
    llm = None
    if pipe_options.get("templates") == "llm":
        from pipeline.llm_templates import LLMTemplates
        llm = LLMTemplates.from_env()
    _W["pipe"] = G.DocPipeline(Path(work) / str(os.getpid()), extras=False, llm=llm, **pipe_options)
    _W["settings"] = settings
    from pipeline.checking import problems
    _W["format"] = problems
    if "dataset" in checks:
        _W["dataset"] = _load("check_dataset")
        _W["pictures"] = _W["dataset"].known_pictures()
    if "length" in checks:
        _W["tokenizer"] = (_W.get("dataset") or _load("check_dataset")).load_tokenizer()
    if "labels" in checks:
        _W["labels"] = _load("check_labels")
    if "boxes" in checks:
        _W["boxes"] = _load("check_boxes")


def _stop_helper() -> None:
    helper = _W.pop("surya", None)
    if helper is not None:
        processes = list((helper._processes or {}).values())
        helper.shutdown(wait=False, cancel_futures=True)
        for p in processes:
            p.terminate()


def _blank_jpeg() -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(b, "JPEG")
    return b.getvalue()


def _surya_parse(task: str, target: str, jpeg: bytes) -> dict:
    """Runs in the Surya helper process: surya_probe/surya_pipeline.run() on the answer, as the Inference tab."""
    sp = _W.get("surya_pipeline")
    if sp is None:
        sp = _W["surya_pipeline"] = _load("surya_pipeline")
        sp.load_surya()
        sp.load_surya = lambda: None  # once: it adds Surya to sys.path and sys.modules
    with Image.open(io.BytesIO(jpeg)) as im:
        return sp.run(task, target, im.convert("RGB"), "stop")


def _check(name: str, files: dict) -> list[str]:
    """What is wrong with one finished sample (empty: nothing), by the checks asked for."""
    checks, s = _W["settings"]["checks"], files
    label, task, target, prompt = s["label"], s["task"], s["target"], s["prompt"]
    asked = _W["settings"]["tasks"]
    out = [] if asked is None or task in asked else [f"task: {task}, not one asked for {sorted(asked)} (the "
                                                     f"pipeline's fallback: {s['info'].get('note', '')})"[:200]]
    if "format" in checks:
        out += [f"format: {p}" for p in _W["format"](task, target)]
    with Image.open(io.BytesIO(s["image"])) as im:
        size, mode = im.size, im.mode
    if "dataset" in checks and task != "table":  # (check_dataset reads full_ocr, block_ocr and layout)
        out += [f"dataset: {k} {d}".strip() for k, d in
                _W["dataset"].sample_problems(task, prompt, target, label, size, mode, _W["pictures"])]
    blank = s["info"].get("template") == "blank"
    if "labels" in checks and not blank:
        found = defaultdict(list)
        _W["labels"].check(label, s["html"], found, name)
        _W["labels"].structure(label, s["html"], found, name)
        out += [f"labels {k}: {x}" for k, xs in found.items() for x in xs]
    if "boxes" in checks and not blank:
        found = defaultdict(list)
        with Image.open(io.BytesIO(s["page"] or s["image"])) as im:
            _W["boxes"].check(label, _W["boxes"].gray(im, label), found, name)
        strict = label.get("effect") == "clean"  # on effect pages noise and tilt fool the ink test: boxes on the page only
        out += [f"boxes {k}: {x}" for k, xs in found.items() if strict or k == "page" for x in xs]
    if "length" in checks and _W.get("tokenizer") is not None:
        n = _W["dataset"].sequence_length(_W["tokenizer"], *size, prompt, target) if "dataset" in _W else \
            _load("check_dataset").sequence_length(_W["tokenizer"], *size, prompt, target)
        if n > _W["settings"]["max_length"]:
            out.append(f"length: {n} tokens > {_W['settings']['max_length']}")
    if "surya" in checks and task in ("full_ocr", "layout", "block_ocr"):
        result = _W["surya"].submit(_surya_parse, task, target, s["image"]).result()
        if result.get("status") != "ok" or (task == "block_ocr" and not str(result.get("html", "")).strip()):
            out.append(f"surya: {result.get('status')} {result.get('note', '')}".strip())
    return out


def _read(pipe, name: str, info: dict) -> dict:
    """The sample's files, read into memory and removed (unless kept: the generator's own folder)."""
    out = pipe.out
    files = {"image": (out / f"{name}.jpg").read_bytes(), "task": info["task"],
             "prompt": (out / f"{name}.prompt.txt").read_text(encoding="utf-8"),
             "target": (out / info["target_file"]).read_text(encoding="utf-8"),
             "label": json.loads((out / f"{name}.json").read_text(encoding="utf-8")),
             "page": (out / f"{name}_page.jpg").read_bytes() if (out / f"{name}_page.jpg").exists() else None,
             "html": (out / f"{name}.html").read_text(encoding="utf-8") if (out / f"{name}.html").exists() else "",
             "markdown": (out / f"{name}.md").read_text(encoding="utf-8") if (out / f"{name}.md").exists() else "",
             "info": info}
    for f in out.glob(f"{name}[._]*"):
        f.unlink(missing_ok=True)
    return files


def _make(seed: int, name: str) -> dict:
    """One slot in a worker: its seed, else replacements, until a sample passes every check."""
    pipe, settings = _W["pipe"], _W["settings"]
    rejected = []
    for k in range(settings["max_retries"] + 1):
        s = seed + k * REPLACE
        try:
            info = pipe.sample(s, name)
        except Exception as e:  # the pipeline's own checks failed on every render (or an error): next seed
            rejected.append({"seed": s, "why": [f"pipeline: {type(e).__name__}: {e}"[:300]]})
            continue
        files = _read(pipe, name, info)
        why = _check(name, files)
        if not why:
            files["seed"], files["checks"] = s, {"ran": list(settings["checks"]), "rejected": rejected,
                                                  "attempts": files["label"].get("attempts")}
            return files
        rejected.append({"seed": s, "why": why[:10]})
    raise GenerationError(f"{name}: no sample passed the checks in {settings['max_retries'] + 1} seeds: "
                          + json.dumps(rejected, ensure_ascii=False)[:2000])


# -- the public API ------------------------------------------------------------------------------------------------

class DataGenerator:
    """Worker processes making checked samples on request. Start it once per training run (starting the workers
    and their browsers takes a while), ask for batch() or epoch() as often as needed, then close() (or use it
    in a with block)."""

    def __init__(self, *, seed: int = 0, workers: int | None = None,
                 language="mix", effect="mix", source="mix", task="mix", tashkeel="mix",
                 templates: str = "bank", text: str | None = None,
                 strengths: dict | None = None, colour_share: float = 0.5,
                 graphic_share: float = 0.35, special_share: float = 0.5, blank_share: float = 0.01,
                 balance: bool = True, render_checks: bool = True, render_attempts: int = 8, order_tries: int = 6,
                 low_dpi: dict | None = None, validate: bool | Iterable[str] = True, surya_check: str | bool = "auto",
                 max_length: int = 6144, max_retries: int = 5, out_dir: str | Path | None = None):
        """
        seed            the run's seed: same seed, options and pools -> the same samples
        workers         processes making pages (default: min(4, CPUs)); each needs about 1.5 GB
        language        "mix" (ar 50%, en 50%), "ar", "en", or {"ar": 0.7, "en": 0.3}
        effect          "mix" (clean 15%, ocrsmith / augraphy / scan-simulator the rest evenly), one of
                        clean, ocrsmith, augraphy, scan-simulator, augraphy+scan-simulator, or weights
        source          template source: "mix" (all four evenly), grammar, websight, websight2, doclaynet
        task            Surya prompt: "mix" (full_ocr 40%, layout 25%, block_ocr 35%), full_ocr, layout,
                        block_ocr, table, or weights such as {"full_ocr": 0.4, "block_ocr": 0.35}
        tashkeel        Arabic marks: "mix" (none 40%, partial 30%, full 30%), none, partial, full
        templates       "bank" (bank2000) or "llm" (templates written by an LLM, needs its key in .env)
        text            "pools" (dataset_pool's documents; default when built), "wikipedia" (the small pool) or "demo"
        strengths       effect strengths 0-1 (0.5 = the tool's own settings), e.g. {"ocrsmith": 0.7}
        colour_share    share of pages in colour (the rest white, off-white or a light tint)
        graphic_share   share of picture slots given a drawn chart, flowchart or molecule (not balanced pages)
        special_share   share of pages with special blocks (equations, code, forms...; not balanced pages)
        blank_share     share of blank pages (Blank-Page)
        balance         each sample one of the nine structured blocks by its seed (even per batch)
        render_checks   the pipeline's own checks on each render (hidden, cut, overlapping text...)
        render_attempts renders per seed until one passes them (a new template every second one)
        order_tries     draws per seed until its reading order is unambiguous
        low_dpi         share of samples per task shown at 96 DPI, e.g. {"layout": 0.5, "full_ocr": 0.2} (default)
        validate        True (all of CHECKS), False (the pipeline's own checks only), or some of CHECKS
        surya_check     "auto" (Surya's parser when the clone surya/ is there), True (required), False
        max_length      most tokens of image + prompt + answer (longer samples are made again)
        max_retries     replacement seeds per slot before GenerationError
        out_dir         if set, every batch / epoch is also saved there (save())
        """
        pipe_options = {"languages": _weights(language, "language"), "effects": _weights(effect, "effect"),
                        "sources": _weights(source, "source"), "tasks": _weights(task, "task"),
                        "tashkeel": _weights(tashkeel, "tashkeel"), "templates": templates, "text": text,
                        "strengths": {k.replace("_", "-"): float(v) for k, v in (strengths or {}).items()},
                        "colour_share": colour_share, "graphic_share": graphic_share, "special_share": special_share,
                        "blank_share": blank_share, "balance": balance, "check": render_checks}
        if templates not in ("bank", "llm") or text not in (None, "pools", "wikipedia", "demo"):
            raise ValueError("templates: 'bank' or 'llm'; text: 'wikipedia', 'demo' or None")
        bad = [k for k in pipe_options["strengths"] if k not in ("ocrsmith", "augraphy", "scan-simulator")]
        if bad or not all(0 <= v <= 1 for v in pipe_options["strengths"].values()):
            raise ValueError(f"strengths: {{ocrsmith, augraphy, scan-simulator: 0-1}}, got {strengths!r}")
        checks = set(CHECKS if validate is True else () if validate is False else validate)
        if checks - set(CHECKS):
            raise ValueError(f"validate: True, False or some of {CHECKS}")
        has_surya = (ROOT / "surya" / "surya" / "recognition").is_dir()
        if surya_check is True and not has_surya:
            raise ValueError("surya_check=True needs Surya's code in surya/ (git clone https://github.com/datalab-to/surya)")
        if surya_check is False or not has_surya:
            checks.discard("surya")
        self.checks = tuple(c for c in CHECKS if c in checks)
        self.seed, self.out_dir = seed, Path(out_dir) if out_dir else None
        self.stats = Counter()  # made, rejected (by check), failed
        self.rejections = []  # the last 200 rejected seeds: {"seed", "why"} (what to look at in the pipeline)
        self._next = 0  # the next sample of the batch stream
        self._work = Path(tempfile.mkdtemp(prefix="ocrsynth_"))
        asked = pipe_options["tasks"]
        settings = {"tasks": None if asked is None else {k for k, w in asked.items() if w > 0},
                    "render_attempts": render_attempts, "order_tries": order_tries, "low_dpi": low_dpi,
                    "checks": self.checks, "max_length": max_length, "max_retries": max_retries}
        self.workers = workers or min(4, os.cpu_count() or 1)
        self._pool = ProcessPoolExecutor(self.workers, mp_context=get_context("spawn"), initializer=_init_worker,
                                         initargs=(str(self._work), pipe_options, settings))
        atexit.register(self.close)

    def _run(self, seeds: list[int], prefix: str) -> list[Sample]:
        futures = [self._pool.submit(_make, s, f"{prefix}{k:06d}") for k, s in enumerate(seeds)]
        out, errors = [], []
        for k, f in enumerate(futures):
            try:
                r = f.result()
            except GenerationError as e:
                errors.append(str(e))
                self.stats["failed"] += 1
                continue
            for rej in r["checks"]["rejected"]:
                self.stats.update(w.split(":")[0].split(" ")[0] for w in rej["why"][:1])
                self.rejections = (self.rejections + [rej])[-200:]
            self.stats["made"] += 1
            out.append(Sample(id=f"{prefix}{k:06d}", seed=r["seed"], task=r["task"], prompt=r["prompt"],
                              target=r["target"], image_bytes=r["image"], label=r["label"], info=r["info"],
                              page_bytes=r["page"], html=r["html"], markdown=r["markdown"], checks=r["checks"]))
        if errors:
            raise GenerationError(f"{len(errors)} of {len(seeds)} samples failed: " + " | ".join(errors)[:3000], out)
        if self.out_dir:
            save(out, self.out_dir)
        return out

    def batch(self, n: int = 8) -> list[Sample]:
        """The next n samples of this generator's endless stream (never the same twice in a run)."""
        start, self._next = self._next, self._next + n
        return self._run([self.seed * 10**12 + 5 * 10**11 + start + k for k in range(n)], f"b{start:09d}_")

    def epoch(self, n: int, epoch: int = 0) -> list[Sample]:
        """All n samples of an epoch: the same for the same seed and epoch, new ones for another epoch."""
        if not 0 <= epoch < 500_000 or not 0 < n <= 10**6:
            raise ValueError("epoch: 0 to 499,999; n: 1 to 1,000,000")
        return self._run([self.seed * 10**12 + epoch * 10**6 + k for k in range(n)], f"e{epoch:05d}_")

    def stream(self, batch_size: int = 8) -> Iterator[Sample]:
        """Endless samples, made batch_size at a time (for an iterable dataset)."""
        while True:
            yield from self.batch(batch_size)

    def close(self) -> None:
        """Stops the workers (and their browsers) and removes their folder."""
        if getattr(self, "_pool", None) is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
            for _ in range(5):  # a browser may hold its folder for a moment after it stops
                shutil.rmtree(self._work, ignore_errors=True)
                if not self._work.exists():
                    break
                time.sleep(1)

    def __enter__(self) -> "DataGenerator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def generate_data(mode: str = "batch", num_samples: int = 8, epoch: int = 0, **options) -> list[Sample]:
    """Samples in one call: mode "batch" (num_samples fresh samples) or "epoch" (the epoch's num_samples, the
    same for the same seed and epoch). options: DataGenerator's. Starts and stops the workers each call; in a
    training loop keep one DataGenerator instead."""
    if mode not in ("batch", "epoch"):
        raise ValueError("mode: 'batch' or 'epoch'")
    with DataGenerator(**options) as gen:
        return gen.batch(num_samples) if mode == "batch" else gen.epoch(num_samples, epoch)


def save(samples: Iterable[Sample], out_dir: str | Path) -> Path:
    """Samples written as `python -m pipeline.run dataset` writes them (images/, prompts/, targets/, pages/,
    labels/, markdown/, html/, manifest.jsonl, so surya_probe's checks read them) plus train.jsonl, the
    ms-swift rows with image paths. Appends to what is there."""
    out = Path(out_dir)
    for folder in ("images", "prompts", "targets", "pages", "labels", "markdown", "html"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    with (out / "manifest.jsonl").open("a", encoding="utf-8") as manifest, \
            (out / "train.jsonl").open("a", encoding="utf-8") as swift:
        for s in samples:
            suffix = Path(s.info["target_file"]).name[len(s.info["name"]):]
            files = {"images": f"images/{s.id}.jpg", "prompts": f"prompts/{s.id}.prompt.txt",
                     "targets": f"targets/{s.id}{suffix}", "labels": f"labels/{s.id}.json",
                     "markdown": f"markdown/{s.id}.md", "html": f"html/{s.id}.html"}
            (out / files["images"]).write_bytes(s.image_bytes)
            (out / files["prompts"]).write_text(s.prompt, encoding="utf-8", newline="\n")
            (out / files["targets"]).write_text(s.target, encoding="utf-8", newline="\n")
            (out / files["labels"]).write_text(json.dumps(s.label, ensure_ascii=False), encoding="utf-8")
            (out / files["markdown"]).write_text(s.markdown, encoding="utf-8", newline="\n")
            (out / files["html"]).write_text(s.html, encoding="utf-8")
            if s.page_bytes:
                files["pages"] = f"pages/{s.id}_page.jpg"
                (out / files["pages"]).write_bytes(s.page_bytes)
            manifest.write(json.dumps({"id": s.id, "split": "train", **files, **s.info, "name": s.id},
                                      ensure_ascii=False) + "\n")
            swift.write(json.dumps(s.swift_row(str((out / files["images"]).resolve())), ensure_ascii=False) + "\n")
    return out


def _pairs(text: str, name: str) -> dict:
    """"full_ocr=0.4,block_ocr=0.35" -> {"full_ocr": 0.4, "block_ocr": 0.35}."""
    try:
        return {k.strip(): float(v) for k, v in (part.split("=") for part in text.split(",") if part.strip())}
    except ValueError:
        raise SystemExit(f"--{name}: write it as key=number,key=number (got {text!r})")


def _choice(name: str):
    """A --language / --effect / --source / --task / --tashkeel value: mix, one choice, or weights."""
    def parse(text: str):
        return _pairs(text, name) if "=" in text else text
    return parse


def main(argv: list[str] | None = None) -> None:
    """Make samples from the command line with every DataGenerator setting as a flag, save them, print a summary.

        python train.py --n 8
        python train.py --n 200 --epoch 3 --language ar --task full_ocr=0.6,block_ocr=0.4 --tashkeel none
    """
    import argparse
    ap = argparse.ArgumentParser(
        description="Make checked training samples (train.DataGenerator) and save them. Every setting has the "
                    "pipeline's default; give only those you want to change.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    run = ap.add_argument_group("what to make")
    run.add_argument("--n", type=int, default=8, help="samples to make")
    run.add_argument("--epoch", type=int, default=None,
                     help="make epoch N's samples (the same every time for the same seed); without it: a fresh batch")
    run.add_argument("--seed", type=int, default=0, help="same seed and settings -> the same samples")
    run.add_argument("--workers", type=int, default=None, help="processes (default: min(4, CPUs)); ~1.5 GB each")
    run.add_argument("--out", default="datasets/train_try", help="folder the samples are saved in")
    run.add_argument("--no-save", action="store_true", help="make and check them, write nothing")
    page = ap.add_argument_group("pages (mix, one choice, or weights such as ar=0.7,en=0.3)")
    for name, help_ in (("language", "ar, en (mix: 50/50)"),
                        ("effect", "clean, ocrsmith, augraphy, scan-simulator, augraphy+scan-simulator "
                                   "(mix: clean 15%%, the rest even)"),
                        ("source", "grammar, websight, websight2, doclaynet (mix: even)"),
                        ("task", "full_ocr, layout, block_ocr, table (mix: full_ocr 40%%, layout 25%%, block_ocr 35%%)"),
                        ("tashkeel", "none, partial, full (mix: 40/30/30)")):
        page.add_argument(f"--{name}", type=_choice(name), default="mix", help=help_)
    page.add_argument("--templates", choices=["bank", "llm"], default="bank", help="llm needs its key in .env")
    page.add_argument("--text", choices=["pools", "wikipedia", "demo"], default=None,
                      help="default: pools (dataset_pool's documents) when built, else wikipedia")
    for tool in ("ocrsmith", "augraphy", "scan-simulator"):
        page.add_argument(f"--strength-{tool}", type=float, default=0.5,
                          help="0 lightest, 0.5 the tool's own settings, 1 strongest")
    page.add_argument("--colour-share", type=float, default=0.5, help="share of pages in colour")
    page.add_argument("--graphic-share", type=float, default=0.35,
                      help="share of picture slots given a chart, flowchart or molecule")
    page.add_argument("--special-share", type=float, default=0.5,
                      help="share of pages with special blocks (unbalanced pages)")
    page.add_argument("--blank-share", type=float, default=0.01, help="share of blank pages")
    page.add_argument("--no-balance", action="store_true",
                      help="don't give each sample one of the nine structured blocks in turn")
    page.add_argument("--low-dpi", default=None,
                      help="share shown at 96 DPI per task, e.g. layout=0.5,full_ocr=0.2 (default: those)")
    check = ap.add_argument_group("checks")
    check.add_argument("--checks", default="all",
                       help=f"all, none (the pipeline's own only), or some of {','.join(CHECKS)}")
    check.add_argument("--no-validate", action="store_true", help="same as --checks none")
    check.add_argument("--surya-check", choices=["auto", "yes", "no"], default="auto",
                       help="Surya's own parser (auto: when the clone surya/ is there)")
    check.add_argument("--no-render-checks", action="store_true", help="skip the pipeline's own page checks")
    check.add_argument("--render-attempts", type=int, default=8, help="renders per seed until one passes them")
    check.add_argument("--order-tries", type=int, default=6, help="draws per seed until the reading order is clear")
    check.add_argument("--max-length", type=int, default=6144, help="most tokens of image + prompt + answer")
    check.add_argument("--max-retries", type=int, default=5, help="replacement seeds per sample before giving up")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    checks = "none" if a.no_validate else a.checks
    options = dict(
        seed=a.seed, workers=a.workers, **{name: getattr(a, name) for name in CHOICES},
        templates=a.templates, text=a.text,
        strengths={t: getattr(a, f"strength_{t.replace('-', '_')}") for t in ("ocrsmith", "augraphy", "scan-simulator")},
        colour_share=a.colour_share, graphic_share=a.graphic_share, special_share=a.special_share,
        blank_share=a.blank_share, balance=not a.no_balance, render_checks=not a.no_render_checks,
        render_attempts=a.render_attempts, order_tries=a.order_tries,
        low_dpi=_pairs(a.low_dpi, "low-dpi") if a.low_dpi else None,
        validate=True if checks == "all" else False if checks == "none" else [c.strip() for c in checks.split(",")],
        surya_check={"auto": "auto", "yes": True, "no": False}[a.surya_check],
        max_length=a.max_length, max_retries=a.max_retries)
    mode = "batch" if a.epoch is None else f"epoch {a.epoch}"
    print(f"{a.n} samples, {mode}; settings: " + ", ".join(f"{k}={v}" for k, v in options.items()), flush=True)
    start = time.time()
    try:
        with DataGenerator(**options) as gen:
            print(f"{gen.workers} workers, checks: {', '.join(gen.checks) or 'the pipeline only'}", flush=True)
            try:
                samples = gen.batch(a.n) if a.epoch is None else gen.epoch(a.n, a.epoch)
            except GenerationError as e:
                print(f"GenerationError: {str(e)[:600]}", flush=True)
                samples = e.samples
            stats, rejections = dict(gen.stats), gen.rejections
    except ValueError as e:
        raise SystemExit(f"bad setting: {e}")
    out = None if a.no_save else save(samples, a.out)
    for s in samples:
        print(f"  {s.id}  {s.task:9} {s.info['lang']}  {s.info['effect']:15} {s.info['template']}  "
              f"{s.target[:70]!r}")
    print(f"{len(samples)} samples in {time.time() - start:.0f}s" + (f" -> {out.resolve()}" if out else " (not saved)")
          + f"  {stats}")
    for r in rejections:
        print(f"  rejected seed {r['seed']}: {r['why'][0][:150]}")


if __name__ == "__main__":
    main()
