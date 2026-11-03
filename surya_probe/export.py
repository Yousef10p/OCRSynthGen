"""Lay the probe's answers out like suryan_outputs/: one folder per image, the image and one file per prompt.

    python surya_probe/export.py            -> surya_probe/outputs/<image>/{<image>, full_ocr.txt, block_ocr.txt,
                                               layout.txt, table.txt, layout_schema.txt, table_schema.txt, info.json}

Each file holds the model's answer exactly as it came (the longest run when an image was asked twice), or the
error the server gave. info.json lists, per prompt: status, seconds, finish_reason, tokens, max_tokens.
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rows = [json.loads(line) for line in (HERE / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    best = {}
    for r in rows:  # per image and prompt: the run with the most tokens allowed
        key = (Path(r["image"]).name, r["task"] + ("_schema" if r["schema"] else ""))
        if key not in best or r["max_tokens"] >= best[key]["max_tokens"]:
            best[key] = r
    out = HERE / "outputs"
    images = sorted({name for name, _ in best})
    for name in images:
        folder = out / Path(name).stem
        folder.mkdir(parents=True, exist_ok=True)
        if (HERE / "inputs" / name).exists():
            shutil.copyfile(HERE / "inputs" / name, folder / name)
        info = {}
        for (image, task), r in sorted(best.items()):
            if image != name:
                continue
            text = r.get("content") if r["status"] == 200 else f"HTTP {r['status']}: {r.get('error', '')}"
            (folder / f"{task}.txt").write_text(text or "", encoding="utf-8", newline="\n")
            info[task] = {k: r.get(k) for k in ("status", "seconds", "finish_reason", "usage", "max_tokens")}
        (folder / "info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(len(images), "images ->", out)


if __name__ == "__main__":
    main()
