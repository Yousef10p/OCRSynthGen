"""One folder per answered page: the image, the prompt, Surya's answer and our label side by side.

    python surya_probe/export_batch.py surya_probe/batch100/full_ocr
        -> surya_probe/batch100/full_ocr/samples/<id>/{image.jpg, prompt.txt, surya_output.txt, our_label.txt|json, info.json}
"""
import json
import shutil
import sys
from pathlib import Path

folder = Path(sys.argv[1])
count = 0
for line in (folder / "surya.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    out = folder / "samples" / r["id"]
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(folder / r["image"], out / "image.jpg")
    (out / "prompt.txt").write_text(r["prompt"], encoding="utf-8")
    (out / "surya_output.txt").write_text(r["content"], encoding="utf-8")  # exactly as the API answered
    target = folder / r["target"]
    shutil.copyfile(target, out / ("our_label" + target.suffix))
    info = {k: r.get(k) for k in ("id", "task", "lang", "status", "seconds", "finish_reason", "usage")}
    (out / "info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    count += 1
print(count, "pages ->", folder / "samples")
