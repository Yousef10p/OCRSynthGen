"""The 2,000-template bank (bank2000/): diverse, checked, Arabic and English, ready for the generator.

Candidates come from four sources:
  grammar    3,000 new templates from the rich grammar (src/grammar.py, rich=True): random combinations of
             blocks (paragraphs, lists, tables, forms, card grids, key/value lists, stat boxes, timelines,
             picture-beside-text rows, callouts...) in random page layouts (single, columns, grid, sidebar,
             letter, report)
  websight   WebSight v0.1 templates of the full bank (templates/w_*)
  websight2  WebSight v0.2 (Tailwind) pages (src/from_websight2.py -> data/raw/candidates_websight2.jsonl)
  doclaynet  DocLayNet v1.1 layouts of the full bank (templates/d_*)

Each candidate is adapted:
  * language: half Arabic (right to left), half English (left to right). Grammar and DocLayNet templates use
    logical CSS (start/end), so only <html lang dir> changes; WebSight v0.1 CSS was flipped to right to left
    by rtlcss, so an English template has it flipped back; WebSight v0.2 pages got their language when
    converted;
  * pictures: every grey picture box becomes a real picture slot, <div class="pic-frame"><img data-pic="pic_N"
    src="{{ pic_N }}"></div>, with a hint of what it shows (from the original page: "?food", alt text, class
    names), listed in template.json "pictures"; the generator fills src with a photo of that kind.

Then, per source:
  1. diversity: candidates are taken in farthest-first order on their layout features (src/near_dedup.py:
     page skeleton and layout CSS for flowing pages, box positions for DocLayNet): each next template is the
     one least like everything taken so far, and anything 75% or more like a taken template (Jaccard) is
     never taken. No two templates have the same structure (structure hash);
  2. screening: each is rendered once by the generator (random style, theme, text, pictures) and kept only if
     the page passes every check of pipeline/qa.py (no hidden, cut, overlapping, covered, off-page or
     low-contrast text);
  3. the first that pass, up to the source's quota (grammar 500, websight 450, websight2 450, doclaynet 600).

If fewer than 2,000 pass, the bank is topped up (also on its own: --top-up): more candidates of the four
sources, each unlike every template already in the bank, are screened the same way; only if those still fall
short are LLM templates used (pipeline/llm_templates.py: the ones saved earlier, then new ones from Groq).

Writes bank2000/templates/b_NNNN/{template.html.j2, template.json[, img_*.png]}, bank2000/index.jsonl and
bank2000/report.json. Runs in .venv-pipeline (the screening renders with SynthDoc):

    .venv-pipeline\\Scripts\\python -m src.bank2000 --workers 6
"""
import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from lxml import html as lhtml

from src.bank import RAW, ROOT, TEMPLATES
from src.dedup import structure_hash

BANK = ROOT / "bank2000"
STAGING = BANK / "_staging"
QUOTAS = {"grammar": 500, "websight": 450, "websight2": 450, "doclaynet": 600}
TOTAL = 2000
TOP_UP = ["grammar", "websight2", "websight", "doclaynet", "llm"]  # the order the top-up tries sources in
SCREEN_FACTOR = 1.7      # how many diverse candidates per quota place are screened, at most
SIMILAR = 0.75           # never two templates sharing this much of their layout features
PIC_CSS = (".pic-frame { overflow: hidden; min-block-size: 2em; }\n"
           ".pic-frame > img { display: block; inline-size: 100%; block-size: 100%; object-fit: cover; }\n")
PH = re.compile(r'<div\b([^>]*?)\bclass="([^"]*?)(?<![\w-])ph(?![\w-])([^"]*)"([^>]*)>\s*</div>')


# -- candidates ---------------------------------------------------------------------------------------------

def candidates(source, n, seed=0):
    """[(meta, text, folder or None)] for one source."""
    rng = random.Random(f"{seed}-{source}")
    if source == "grammar":
        from src import grammar
        out = []
        for i in range(n):
            meta, text = grammar.generate(7_000_000 + seed * 100_000 + i, f"gr_{i:05d}", rich=True)
            out.append((meta, text, None))
        return out
    if source == "llm":  # templates the LLM wrote earlier, then new ones from Groq for the rest
        from pipeline.llm_templates import SAVED, LLMTemplates
        out = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(SAVED.glob("*.json"))] if SAVED.exists() else []
        out = [(d["meta"], d["text"], None) for d in out]
        llm = LLMTemplates.from_env()
        tries = 0
        while len(out) < n and tries < 3 * n and llm.available():
            tries += 1
            try:
                made = llm.template(rng, f"llm_bank_{seed}_{tries}")
            except Exception as e:
                print(f"  LLM call failed: {type(e).__name__}: {e}"[:200], flush=True)
                continue
            if made:
                out.append((*made, None))
        return out[:n]
    if source == "grammar" and seed:  # top-up rounds: seeds the first build did not use
        from src import grammar
        return [(*grammar.generate(8_000_000 + seed * 100_000 + i, f"gr_t{seed}_{i:05d}", rich=True), None)
                for i in range(n)]
    if source == "websight2":
        rows = [json.loads(line) for line in (RAW / "candidates_websight2.jsonl").read_text(encoding="utf-8").splitlines()]
        rng.shuffle(rows)
        return [(r["meta"], r["text"], None) for r in rows[:n]]
    prefix = {"websight": "w_", "doclaynet": "d_"}[source]
    ids = [line.split('"id": "', 1)[1].split('"', 1)[0] for line in (ROOT / "bank_index.jsonl").read_text(encoding="utf-8").splitlines()
           if f'"id": "{prefix}' in line]
    out = []
    for tid in rng.sample(ids, min(n, len(ids))):
        folder = TEMPLATES / tid
        out.append((json.loads((folder / "template.json").read_text(encoding="utf-8")),
                    (folder / "template.html.j2").read_text(encoding="utf-8"), folder))
    return out


# -- adapting a candidate -----------------------------------------------------------------------------------

def unflip(text):
    """A WebSight v0.1 template's CSS flipped back to left to right (rtlcss flips are their own inverse)."""
    from src.from_websight import flip
    first = re.search(r"<style>(.*?)</style>", text, re.S)
    inline = re.findall(r'style="([^"]*)"', text)
    flipped = flip([first.group(1)] + [f"x {{ {s} }}" for s in inline])
    if None in flipped:
        return None
    sheet, styles = flipped[0], [re.fullmatch(r"\s*x\s*\{(.*)\}\s*", c, re.S).group(1).strip() for c in flipped[1:]]
    text = text[:first.start(1)] + sheet + text[first.end(1):]
    it = iter(styles)
    return re.sub(r'style="([^"]*)"', lambda m: f'style="{next(it)}"', text)


def hint_of(attrs, classes):
    m = re.search(r'data-hint="([^"]*)"', attrs)
    words = (m.group(1) if m else "") + " " + classes
    return " ".join(w for w in re.findall(r"[a-zA-Z]{3,}", words) if w not in ("box", "pic", "frame"))[:80]


def adapt(meta, text, lang):
    """(meta, text) in the bank2000 form (language, picture slots), or None if it can't be adapted."""
    meta = json.loads(json.dumps(meta))
    if "http" in text.lower():  # a web address (a background photo from the internet): the page isn't self-contained
        return None
    if meta.get("lang") and meta["lang"] != lang:
        lang = meta["lang"]  # WebSight v0.2 pages got their language when converted
    if lang == "en" and meta["source"] in ("websight", "llm"):  # CSS flipped to right to left when converted
        text = unflip(text)
        if text is None:
            return None
    text = re.sub(r'<html lang="\w+" dir="\w+">', f'<html lang="{lang}" dir="{"rtl" if lang == "ar" else "ltr"}">', text, 1)
    pictures = []

    def frame(m):
        pre, cls_a, cls_b, post = m.groups()
        attrs = pre + post
        hint = hint_of(attrs, cls_a + cls_b)
        pid = f"pic_{len(pictures)}"
        pictures.append({"id": pid, "hint": hint})
        attrs = re.sub(r'\s*data-hint="[^"]*"', "", attrs)
        classes = " ".join((cls_a + " " + cls_b).split() + ["pic-frame"])
        return f'<div{attrs.rstrip()} class="{classes}"><img data-pic="{pid}" src="{{{{ {pid} }}}}" alt=""></div>'
    text = PH.sub(frame, text)
    if re.search(r'class="[^"]*(?<![\w-])ph(?![\w-])', text):
        return None  # a picture box that isn't an empty div: leave this candidate out
    text = text.replace("</style>", PIC_CSS + "</style>", 1)
    meta.update(lang=lang, pictures=pictures)
    for s in meta["slots"]:
        s["dir"] = "rtl" if lang == "ar" else "ltr"
    meta["source_template"] = meta["id"]
    meta["structure_hash"] = structure_hash(text, meta)
    return meta, text


# -- diversity ----------------------------------------------------------------------------------------------

def features(meta, text):
    from src.near_dedup import box_features, flow_features
    root = lhtml.fromstring(text)
    roles = {s["id"]: s["role"] for s in meta["slots"]}
    feats = box_features(root, roles, meta) if meta["positioning"] == "absolute" else flow_features(root, roles)
    return feats | {f"role:{s['role']}" for s in meta["slots"]} | {f"pictures:{min(3, len(meta['pictures']))}"}


def jac(a, b):
    return len(a & b) / max(1, len(a | b))


def farthest_first(items, rng, limit=None, taken=()):
    """Indexes of up to limit items in farthest-first order (each next one least like those before it and the
    feature sets in taken); items too like one already taken (Jaccard >= SIMILAR) are left out."""
    if not items:
        return []
    feats = [f for _, f in items]
    near = [max((jac(f, t) for t in taken), default=0.0) for f in feats]  # highest similarity to anything taken
    alive = {i for i in range(len(items)) if near[i] < SIMILAR}
    order = []
    if not taken and alive:
        order.append(rng.choice(sorted(alive)))
        alive.discard(order[0])
    while alive and (limit is None or len(order) < limit):
        if order:
            last = feats[order[-1]]
            for i in list(alive):
                near[i] = max(near[i], jac(feats[i], last))
                if near[i] >= SIMILAR:
                    alive.discard(i)
            if not alive:
                break
        nxt = min(alive, key=lambda i: (near[i], i))
        order.append(nxt)
        alive.discard(nxt)
    return order


# -- screening ----------------------------------------------------------------------------------------------

_pipe = None


def screen_init(work):
    global _pipe
    sys.stdout = open(os.devnull, "w")  # SynthDoc prints a lot per page
    from pipeline.doc_pipeline import DocPipeline
    _pipe = DocPipeline(Path(work) / str(os.getpid()), extras=False, bank=STAGING)


def screen(job):
    """Render a staged candidate once with a random style; returns (cid, issues or an error)."""
    cid, seed = job
    from pipeline.qa import render_issues
    folder = STAGING / "templates" / cid
    meta = json.loads((folder / "template.json").read_text(encoding="utf-8"))
    try:
        rng = random.Random(seed)
        page, values, _ = _pipe.fill(meta, None, folder, rng, meta["lang"])
        _, ocr, _, pages, qa = _pipe.render(page, values, cid)
        issues = render_issues(ocr, qa)
        return cid, issues, len(ocr["words"])
    except Exception as e:
        return cid, {"error": f"{type(e).__name__}: {e}"[:200]}, 0


def stage(cid, meta, text, folder):
    out = STAGING / "templates" / cid
    out.mkdir(parents=True, exist_ok=True)
    (out / "template.html.j2").write_text(text, encoding="utf-8")
    (out / "template.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    for name in meta.get("images", []):
        shutil.copyfile(folder / name, out / name)


# -- validation ---------------------------------------------------------------------------------------------

def validate(meta, text, folder):
    """Problems with one bank2000 template (empty when it is fine)."""
    from jinja2 import Environment, meta as jmeta
    errors = []
    lang = meta.get("lang")
    if lang not in ("ar", "en"):
        errors.append("lang")
    if f'<html lang="{lang}" dir="{"rtl" if lang == "ar" else "ltr"}">' not in text:
        errors.append("html lang/dir")
    slots = [s["id"] for s in meta["slots"]]
    pics = [p["id"] for p in meta.get("pictures", [])]
    if slots != [f"slot_{i}" for i in range(len(slots))] or pics != [f"pic_{i}" for i in range(len(pics))]:
        errors.append("slot/picture ids")
    env = Environment()
    if jmeta.find_undeclared_variables(env.parse(text)) != set(slots) | set(pics) | {"style"}:
        errors.append("template variables")
    lower = text.lower()
    errors += [f"contains {bad}" for bad in ("http", "<script") if bad in lower]
    for src in re.findall(r'<img[^>]*\ssrc="([^"]*)"', text):
        if not (re.fullmatch(r"\{\{ pic_\d+ \}\}", src) or (src in meta.get("images", []) and (folder / src).is_file())):
            errors.append(f"img src {src[:40]}")
    if not slots:
        errors.append("no slots")
    try:
        env.from_string(text).render(style="", **{k: "x" for k in slots + pics})
    except Exception as e:
        errors.append(f"render: {e}")
    return errors


# -- the build ----------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", type=int, default=4000, help="candidates read per source before the diversity pick")
    ap.add_argument("--top-up", action="store_true", help="only add templates to the existing bank, up to 2,000")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.top_up:
        return top_up(a)
    start = time.time()
    shutil.rmtree(STAGING, ignore_errors=True)
    rng = random.Random(a.seed)
    report = {"sources": {}}
    ordered = {}
    for source, quota in QUOTAS.items():
        raw = candidates(source, a.pool if source != "grammar" else 3000, a.seed)
        seen, items = set(), []
        for k, (meta, text, folder) in enumerate(raw):
            made = adapt(meta, text, "ar" if k % 2 == 0 else "en")
            if not made or made[0]["structure_hash"] in seen:
                continue
            seen.add(made[0]["structure_hash"])
            items.append(((made[0], made[1], folder), features(made[0], made[1])))
        take = farthest_first(items, rng, int(quota * SCREEN_FACTOR))
        ordered[source] = [items[i][0] for i in take]
        report["sources"][source] = {"candidates": len(raw), "adapted": len(items), "screened": len(take)}
        print(f"{source}: {len(raw)} candidates, {len(items)} adapted, screening the {len(take)} most diverse"
              f" [{time.time() - start:.0f}s]", flush=True)
    jobs, staged = [], {}
    for source, rows in ordered.items():
        for k, (meta, text, folder) in enumerate(rows):
            cid = f"{source}_{k:04d}"
            stage(cid, meta, text, folder)
            staged[cid] = (source, k, meta, text, folder)
            jobs.append((cid, rng.randrange(2**31)))
    results = screen_all(jobs, staged, a.workers, start)
    # keep the first passing templates of each source, in diversity order, up to the quota
    shutil.rmtree(BANK / "templates", ignore_errors=True)
    (BANK / "templates").mkdir(parents=True)
    kept = []
    for source, quota in QUOTAS.items():
        rows = [cid for cid in staged if staged[cid][0] == source]
        rows.sort(key=lambda c: staged[c][1])
        passed = [c for c in rows if not results.get(c, ({"missing": 1}, 0))[0]]
        rejected = Counter(k for c in rows for k in results.get(c, ({"missing": 1}, 0))[0])
        for cid in passed[:quota]:
            kept.append(write_template(len(kept) + 1, *staged[cid][2:]))
        report["sources"][source].update(passed=len(passed), kept=min(quota, len(passed)),
                                         rejected_for=dict(rejected))
        print(f"{source}: {len(passed)} of {len(rows)} passed the checks, {min(quota, len(passed))} kept", flush=True)
    report["seconds"] = round(time.time() - start)
    finish(kept, report)
    if len(kept) < TOTAL:
        top_up(a)


def screen_all(jobs, staged, workers, start):
    """Write the staging index (the generator reads it), then screen every job; {cid: (issues, words)}."""
    index = [{"id": cid, "source": staged[cid][0], "lang": staged[cid][2]["lang"]} for cid, _ in jobs]
    (STAGING / "index.jsonl").write_text("\n".join(json.dumps(e) for e in index), encoding="utf-8")
    results = {}
    with Pool(workers, initializer=screen_init, initargs=(str(BANK / "_work"),)) as pool:
        for n, (cid, issues, words) in enumerate(pool.imap_unordered(screen, jobs), 1):
            results[cid] = (issues, words)
            if n % 100 == 0 or n == len(jobs):
                ok = sum(1 for i, w in results.values() if not i)
                print(f"  screened {n}/{len(jobs)}, {ok} passed [{time.time() - start:.0f}s]", flush=True)
    shutil.rmtree(BANK / "_work", ignore_errors=True)
    return results


def write_template(n, meta, text, folder):
    tid = f"b_{n:04d}"
    meta = {**meta, "id": tid}
    out = BANK / "templates" / tid
    out.mkdir()
    (out / "template.html.j2").write_text(text, encoding="utf-8")
    (out / "template.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    for name in meta.get("images", []):
        shutil.copyfile(folder / name, out / name)
    return meta


def top_up(a):
    """Add screened templates, each unlike everything already in the bank, until it holds TOTAL."""
    start = time.time()
    report = json.loads((BANK / "report.json").read_text(encoding="utf-8"))
    ids = [json.loads(line)["id"] for line in (BANK / "index.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [json.loads((BANK / "templates" / t / "template.json").read_text(encoding="utf-8")) for t in ids]
    taken = [features(m, (BANK / "templates" / m["id"] / "template.html.j2").read_text(encoding="utf-8")) for m in kept]
    hashes = {m["structure_hash"] for m in kept}
    used = {m["source_template"] for m in kept}
    rng = random.Random(a.seed + 1)
    for round_, source in enumerate(TOP_UP, 1):
        need = TOTAL - len(kept)
        if need <= 0:
            break
        raw = candidates(source, 3 * need if source == "llm" else a.pool, a.seed + round_)
        items = []
        for k, (meta, text, folder) in enumerate(raw):
            made = None if meta["id"] in used else adapt(meta, text, "ar" if (len(kept) + k) % 2 == 0 else "en")
            if not made or made[0]["structure_hash"] in hashes:
                continue
            hashes.add(made[0]["structure_hash"])
            items.append(((made[0], made[1], folder), features(made[0], made[1])))
        take = farthest_first(items, rng, int(need * SCREEN_FACTOR) + 10, taken)
        print(f"top-up {source}: {len(raw)} candidates, {len(items)} new, screening {len(take)} unlike the bank"
              f" [{time.time() - start:.0f}s]", flush=True)
        if not take:
            continue
        shutil.rmtree(STAGING, ignore_errors=True)
        staged, jobs = {}, []
        for k, i in enumerate(take):
            cid = f"top_{source}_{k:04d}"
            stage(cid, *items[i][0])
            staged[cid] = (source, k, *items[i][0])
            jobs.append((cid, rng.randrange(2**31)))
        results = screen_all(jobs, staged, a.workers, start)
        passed = sorted((c for c in staged if not results.get(c, ({"missing": 1}, 0))[0]), key=lambda c: staged[c][1])
        added = 0
        for cid in passed:
            f = items[take[staged[cid][1]]][1]
            if len(kept) >= TOTAL:
                break
            if added and any(jac(f, t) >= SIMILAR for t in taken[-added:]):  # too like one this round added
                continue
            kept.append(write_template(len(kept) + 1, *staged[cid][2:]))
            taken.append(f)
            added += 1
        report.setdefault("top_up", []).append({"source": source, "screened": len(take), "passed": len(passed),
                                                "added": added})
        print(f"top-up {source}: {len(passed)} passed, {added} added, bank now {len(kept)}", flush=True)
    report["seconds"] = report.get("seconds", 0) + round(time.time() - start)
    finish(kept, report)


def finish(kept, report):
    """Validate the bank, write index.jsonl and report.json."""
    problems = {m["id"]: validate(m, (BANK / "templates" / m["id"] / "template.html.j2").read_text(encoding="utf-8"),
                                  BANK / "templates" / m["id"]) for m in kept}
    problems = {k: v for k, v in problems.items() if v}
    hashes = Counter(m["structure_hash"] for m in kept)
    (BANK / "index.jsonl").write_text("\n".join(json.dumps(
        {"id": m["id"], "source": m["source"], "lang": m["lang"], "source_template": m["source_template"],
         "positioning": m["positioning"], "slots": len(m["slots"]), "pictures": len(m["pictures"]),
         "features": m["features"], "structure_hash": m["structure_hash"]}, ensure_ascii=False) for m in kept) + "\n",
        encoding="utf-8")
    report.update(templates=len(kept), languages=dict(Counter(m["lang"] for m in kept)),
                  by_source=dict(Counter(m["source"] for m in kept)),
                  with_pictures=sum(1 for m in kept if m["pictures"]), invalid=problems,
                  duplicate_structures=sum(v - 1 for v in hashes.values() if v > 1))
    (BANK / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("sources", "invalid")} |
                     {"invalid": len(problems)}, indent=1), flush=True)


if __name__ == "__main__":
    main()
