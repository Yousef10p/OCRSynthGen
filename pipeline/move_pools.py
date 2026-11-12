"""Move every downloaded pool into dataset_pool/ (one rename each: nothing is copied, nothing is changed inside).

    .venv-pipeline\\Scripts\\python -m pipeline.move_pools          what would move (nothing moves)
    .venv-pipeline\\Scripts\\python -m pipeline.move_pools --go     move them

    text_pool/pools/<name>     -> dataset_pool/<name>       text, parallel, layout, tables, captions, equations, code,
                                                            molecules, references, chartdata, cosyn, diagrams, vocab,
                                                            _lid, _logs, _download, _downloads, README.md
    text_pool/ar.json, en.json -> dataset_pool/wikipedia/   the small Wikipedia pool
    img_pool/                  -> dataset_pool/images/       as a whole: captions.jsonl and photos.jsonl share pd12m/
    font_pool/                 -> dataset_pool/fonts/        as a whole: fonts.json and fonts-more.json share files

Readers find each pool in either place (pools.home(), pools.pool_file()), and every index keeps working: picture
paths are relative to images/, font and flowchart paths written from the repository (font_pool/..., img_pool/...)
are found under fonts/ and images/. Run it when no pool job is writing (content_pools, flowchart_pool): it stops
if one is running. Moving back: rename the folders the other way (the list above).
"""
import argparse
import os
import subprocess
import sys

from pipeline.pools import DATASET_POOL, ROOT


def moves():
    """[(from, to)] for everything still in a first home."""
    out = []
    old = ROOT / "text_pool" / "pools"
    if old.exists():
        out += [(p, DATASET_POOL / p.name) for p in sorted(old.iterdir())]
    out += [(p, DATASET_POOL / "wikipedia" / p.name) for p in sorted((ROOT / "text_pool").glob("*")) if p.is_file()]
    for first, name in (("img_pool", "images"), ("font_pool", "fonts")):
        if (ROOT / first).exists():
            out.append((ROOT / first, DATASET_POOL / name))
    return out


def running_jobs():
    """Command lines of pool jobs running now (Windows: CIM; elsewhere: ps)."""
    if os.name == "nt":
        cmd = ["powershell", "-NoProfile", "-Command",
               "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | % CommandLine"]
    else:
        cmd = ["ps", "-eo", "args"]
    try:
        lines = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.splitlines()
    except Exception:  # noqa: BLE001  (can't tell: say so)
        return ["(could not list processes)"]
    return [x.strip() for x in lines if "pipeline.content_pools" in x or "pipeline.flowchart_pool" in x]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true", help="move (without it: only show the moves)")
    ap.add_argument("--force", action="store_true", help="move even if a pool job seems to be running")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    todo = moves()
    if not todo:
        print(f"nothing to move: the pools are in {DATASET_POOL}" if DATASET_POOL.exists() else
              "nothing to move: no pools here (text_pool/, img_pool/, font_pool/); the generator uses dataset_pool_sample/")
        return
    clash = [t for _, t in todo if t.exists()]
    for f, t in todo:
        print(f"  {f.relative_to(ROOT)}  ->  {t.relative_to(ROOT)}" + ("   (ALREADY THERE: not moved)" if t.exists() else ""))
    if not a.go:
        print(f"{len(todo)} moves; nothing moved (add --go)")
        return
    if clash:
        sys.exit(f"stopped: {len(clash)} targets exist already; move or remove them first")
    jobs = running_jobs()
    if jobs and not a.force:
        sys.exit("stopped: pool jobs are running (they write into these folders):\n  " + "\n  ".join(jobs))
    left = []
    for f, t in todo:
        t.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(f, t)  # the same drive: a rename, nothing copied
        except PermissionError:
            if not f.is_dir():
                raise
            # a folder some program holds open (Explorer, a shell inside it) can't be renamed, but its files can:
            # moved one by one, the empty folder left behind
            for p in sorted(f.rglob("*"), key=lambda p: -len(p.parts)):
                if p.is_file():
                    (t / p.relative_to(f)).parent.mkdir(parents=True, exist_ok=True)
                    os.rename(p, t / p.relative_to(f))
            left.append(f)
        print(f"  moved {f.relative_to(ROOT)}")
    for empty in (ROOT / "text_pool" / "pools", ROOT / "text_pool"):
        if empty.exists() and not any(empty.iterdir()):
            empty.rmdir()
    print(f"done: the pools are in {DATASET_POOL}")
    for f in left:
        print(f"  {f.relative_to(ROOT)} was held open by a program: its files moved, the empty folder is left")


if __name__ == "__main__":
    main()
