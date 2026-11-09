"""Are the bank's templates usable, and is their reading order clear? Renders every template twice (clean page,
full_ocr; other text, fonts and colours each time) and records the order the pipeline gives its blocks.

    python surya_probe/check_templates.py OUT.jsonl [--workers 3] [--ids b_0001 ...] [--bank DIR]

Reported per template:
  failed     a render failed every attempt (the template can't make a page)
  unstable   the two renders put the same elements in a different order
  against    (flow pages) the order goes against the template's own document order on more than 15% of block pairs
             (DocLayNet pages list their elements top to bottom across the page, not in reading order: not checked)
A template is ambiguous when it is unstable or against. A run already in OUT.jsonl is kept (started again, it goes
on where it stopped); only the summary is printed when every template is done.
"""
import argparse
import json
import sys
import tempfile
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import generator as G  # noqa: E402

SEEDS = (11, 23)
pipe, captured = None, []


def init(bank):
    global pipe
    real = G.reading_order

    def recorded(out, lang, width, height):
        res = real(out, lang, width, height)
        captured.append({"order": [b.get("dom", 0) for b in res]})
        return res
    G.reading_order = recorded
    pipe = G.DocPipeline(Path(tempfile.mkdtemp(prefix="check_templates_")), extras=False, bank=bank)


def check(entry):
    out = {"id": entry["id"], "source": entry["source"], "lang": entry["lang"], "runs": []}
    G._cache[("index", str(pipe.bank))] = [entry]  # this template only
    for seed in SEEDS:
        captured.clear()
        try:
            pipe.sample(seed, f"{entry['id']}_{seed}", language=entry["lang"], effect="clean", task="full_ocr")
            out["runs"].append({"order": captured[-1]["order"] if captured else []})
        except Exception as e:  # noqa: BLE001
            out["runs"].append({"error": f"{type(e).__name__}: {e}"[:300]})
        for f in pipe.out.glob(f"{entry['id']}_{seed}*"):
            f.unlink(missing_ok=True)
    return out


def discordant(seq):
    n = len(seq)
    return sum(seq[a] > seq[b] for a in range(n) for b in range(a + 1, n)) / max(1, n * (n - 1) // 2)


def verdict(t):
    """(failed, unstable, against) for one template's two renders."""
    seqs = [r["order"] for r in t["runs"] if "order" in r]
    failed = any("error" in r for r in t["runs"])
    against = t["source"] != "doclaynet" and max((discordant(s) for s in seqs), default=0) > 0.15
    unstable = False
    if len(seqs) == 2:
        a, b = seqs
        common = [d for d in a if a.count(d) == 1 and b.count(d) == 1]
        pos = {d: k for k, d in enumerate(x for x in b if x in set(common))}
        unstable = discordant([pos[d] for d in common]) > 0
    return failed, unstable, against


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--bank", default=str(G.BANK2000))
    a = ap.parse_args()
    bank = Path(a.bank).resolve()
    entries = [json.loads(l) for l in (bank / "index.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out = Path(a.out)
    done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()} if out.exists() else set()
    todo = [e for e in entries if (not a.ids or e["id"] in a.ids) and e["id"] not in done]
    print(f"{len(todo)} templates to check, {a.workers} workers", flush=True)
    start = time.time()
    with Pool(a.workers, initializer=init, initargs=(bank,), maxtasksperchild=200) as pool, \
            out.open("a", encoding="utf-8") as f:
        for k, res in enumerate(pool.imap_unordered(check, todo), 1):
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            if k % 100 == 0:
                print(f"  {k}/{len(todo)}  {k / (time.time() - start):.2f} templates/s", flush=True)
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    tot, by = Counter(), Counter()
    for t in rows:
        failed, unstable, against = verdict(t)
        tot.update(failed=failed, unstable=unstable, against=against, ambiguous=unstable or against)
        by[(t["source"], t["lang"])] += unstable or against
    print(f"{len(rows)} templates: {dict(tot)}")
    print("ambiguous by source:", {f"{s} {l}": n for (s, l), n in sorted(by.items())})


if __name__ == "__main__":
    main()
