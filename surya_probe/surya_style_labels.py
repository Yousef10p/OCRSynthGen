"""Our pages' labels made the way Surya's own code makes them (surya_repo_findings.md):

  full_ocr   the page as <div data-bbox data-label> HTML, encoded with Surya's rules (pipeline/surya_encode.py)
  block_ocr  the cut-out block's HTML, encoded the same way
  layout     [{"label", "bbox", "count"}], count = the encoded block's length in Surya's tokens, rounded to 50
  table      unchanged (rows then columns)

    python surya_probe/surya_style_labels.py surya_probe/batch100/full_ocr   -> <folder>/surya_style/<id>.txt|.json

"""
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import surya_encode as compact  # noqa: E402  our implementation of Surya's encode rules
tokenizer = Tokenizer.from_file(hf_hub_download("datalab-to/surya-ocr-2.5", "tokenizer.json"))


def count(html):
    """Surya's layout count: the block's OCR answer length in tokens, a multiple of 10 (its real answers)."""
    n = len(tokenizer.encode(compact.encode(html), add_special_tokens=False).ids)
    return max(10, round(n / 10) * 10)


def cut_block(surya):
    """The block a block_ocr cut-out shows: the one inside the crop."""
    x0, y0, x1, y1 = surya["crop"]
    inside = [b for b in surya["blocks"] if b["box"][0] >= x0 - 1 and b["box"][1] >= y0 - 1
              and b["box"][2] <= x1 + 1 and b["box"][3] <= y1 + 1]
    return inside[0] if len(inside) == 1 else None


folder = Path(sys.argv[1])
out = folder / "surya_style"
out.mkdir(exist_ok=True)
made = {}
for line in (folder / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    surya = json.loads((folder / row["labels"]).read_text(encoding="utf-8"))["surya"]
    task = surya["task"]
    if task == "full_ocr":
        label, suffix = compact.encode(surya["divs"]), ".txt"
    elif task == "block_ocr":
        block = cut_block(surya)
        if block is None:
            continue
        label, suffix = compact.encode(block["html"]), ".txt"
    elif task == "layout":
        width, height = surya["page_size"]
        boxes = [" ".join(str(max(0, min(1000, round(v * 1000 / (width if i % 2 == 0 else height)))))
                          for i, v in enumerate(b["box"])) for b in surya["blocks"]]
        label = json.dumps([{"label": b["label"], "bbox": box, "count": count(b["html"])}
                            for b, box in zip(surya["blocks"], boxes)], ensure_ascii=False)
        suffix = ".json"
    else:
        label, suffix = surya["target"], ".json"
    (out / (row["id"] + suffix)).write_text(label, encoding="utf-8")
    made[task] = made.get(task, 0) + 1
print(folder.name, made)
