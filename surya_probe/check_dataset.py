"""Is a batch folder ready for fine-tuning Surya OCR 2.5? Checks every full_ocr / block_ocr sample as training reads it.

    python surya_probe/check_dataset.py <batch folder> [max tokens, default 6144]

  pairing    image, prompt and target files exist and belong to the same sample; the prompt is Surya's own prompt
             for the task (pipeline/surya.py PROMPTS, copied from Surya's prompts.py)
  format     a target is one line, not empty, without </div>; full_ocr starts with a block header
             <|Label|><|x0|><|y0|><|x1|><|y1|>, block_ocr has none; boxes are 0-1000 with x0 < x1, y0 < y1
  encoder    Surya's own decode() then encode() (originals/surya/inference/compact.py) gives the target back unchanged
  vocabulary only Surya's labels (compact.py BLOCK_LABELS), tags (prompts.py ALLOWED_TAGS) and attributes
  length     image tokens (Surya's resize limits) + prompt + answer + end token, against the training limit
  layout     a layout target is a JSON list of {label, bbox, count}: Surya's labels, boxes 0-1000, count a
             whole ten (10 or more)
  pictures   every picture block (Image, Figure, Diagram, Chemical-Block) is one the pools make: a photo's captions
             (generator.picture_html) or a drawn chart, flowchart or molecule (img_pool/graphics)
  kinds      every other new block in Surya's form: Equation-Block <math display="block">, Code-Block <pre>,
             Bibliography <ul style="list-style-type: none; padding-left: 0;">, Table-Of-Contents and Form <table>,
             Complex-Block <img alt=...>, Blank-Page empty and over the whole page
  tashkeel   no Arabic mark in any answer (marks are drawn, never labelled)
  words      the page's label has words (a blank page has none), and none were lost to the effect
  labels     how many samples have each of Surya's block labels, and which never came up
"""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline  # noqa: E402,F401  (puts originals/ on the path)
from surya.inference import compact  # noqa: E402  Surya's own compact.py
from pipeline.generator import ALL_MARKS, PROMPTS, no_marks, picture_html  # noqa: E402
from pipeline.pools import ImagePool  # noqa: E402

# Surya's prompts.py ALLOWED_TAGS / ALLOWED_ATTRIBUTES, and compact.py's attribute order (merged table cells)
TAGS = {"math", "br", "i", "b", "u", "del", "sup", "sub", "table", "tr", "td", "p", "th", "div", "pre", "h1", "h2",
        "h3", "h4", "h5", "ul", "ol", "li", "input", "a", "span", "img", "hr", "tbody", "small", "caption", "strong",
        "thead", "big", "code", "chem"}
ATTRS = {"class", "colspan", "rowspan", "display", "checked", "type", "border", "value", "style", "href", "alt",
         "align", "data-bbox", "data-label", "data-cs", "data-rs", "data-kind", "text"}
HEADER = re.compile(r"<\|([A-Za-z-]+)\|><\|(\d+)\|><\|(\d+)\|><\|(\d+)\|><\|(\d+)\|>")


def image_tokens(width, height, factor=32, low=1792 * 28, high=3072 * 2048):
    """Surya's image processor: resized to multiples of 32 within its pixel limits, one token per 32x32."""
    h, w = max(factor, round(height / factor) * factor), max(factor, round(width / factor) * factor)
    if h * w > high:
        beta = math.sqrt(height * width / high)
        h, w = max(factor, math.floor(height / beta / factor) * factor), max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < low:
        beta = math.sqrt(low / (height * width))
        h, w = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return (h // factor) * (w // factor)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    folder, limit = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 6144
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(hf_hub_download("datalab-to/surya-ocr-2.5", "tokenizer.json"))
    except Exception as e:  # no tokenizer: the length check is skipped
        tokenizer = None
        print("length check skipped:", e)
    rows = [json.loads(line) for line in (folder / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    pool = ImagePool()  # every picture block the pool's captions can make
    known_pictures = {picture_html(c) for c in pool.captions.values() if c}
    known_pictures |= {r["html"] for by_lang in pool.graphics.values() for items in by_lang.values() for _, r in items}
    known_pictures = {no_marks(h) for h in known_pictures}  # as the labels have them: a mark (عمّان) is drawn only
    forms = {"Equation-Block": r'<math display="block">[^<]+</math>', "Code-Block": r"<pre>.*</pre>",
             "Bibliography": r'<ul style="list-style-type: none; padding-left: 0;">.*</ul>',
             "Table-Of-Contents": r"<table>.*</table>", "Form": r"<table>.*</table>", "Complex-Block": r'<img alt="[^"]*"/>.*',
             "Blank-Page": r""}
    labels_seen = Counter()
    problems, lengths, tasks, ids = Counter(), [], Counter(), Counter(r["id"] for r in rows)
    examples = {}

    def bad(kind, sample, detail=""):
        problems[kind] += 1
        examples.setdefault(kind, f"{sample}: {detail}"[:200])

    for r in rows:
        task = r.get("task")
        if task not in ("full_ocr", "block_ocr", "layout"):
            continue
        tasks[task] += 1
        sid = r["id"]
        if ids[sid] > 1:
            bad("id used twice", sid)
        paths = {k: folder / r[k] for k in ("images", "prompts", "targets", "labels")}
        missing = [k for k, p in paths.items() if not p.exists()]
        if missing:
            bad("missing file", sid, ", ".join(missing))
            continue
        if not all(sid in p.name for p in paths.values()):
            bad("files of another sample", sid, str([p.name for p in paths.values()]))
        prompt = paths["prompts"].read_text(encoding="utf-8")
        target = paths["targets"].read_text(encoding="utf-8")
        label = json.loads(paths["labels"].read_text(encoding="utf-8")).get("surya", {})
        if prompt != PROMPTS[task]:
            bad("prompt is not Surya's", sid, repr(prompt[:80]))
        if label.get("task") != task or label.get("target") != target:
            bad("label JSON disagrees with the target file", sid)
        page = json.loads(paths["labels"].read_text(encoding="utf-8"))
        if not page.get("words") and page.get("template") != "blank":
            bad("no words on the page", sid)
        blocks = (page.get("surya") or {}).get("blocks", [])
        labels_seen.update({b["label"] for b in blocks})
        for b in blocks:
            if b["label"] in ("Image", "Figure", "Diagram", "Chemical-Block") and known_pictures and b["html"] not in known_pictures:
                bad(f"{b['label']} block not one the pools make", sid, b["html"][:120])
            if b["label"] in forms and not re.fullmatch(forms[b["label"]], b["html"], re.S):
                bad(f"{b['label']} block not in Surya's form", sid, b["html"][:120])
            if b["label"] == "Blank-Page" and b["box"] != [0, 0, page["width"], page["height"]]:
                bad("Blank-Page box not the whole page", sid, str(b["box"]))
        if ALL_MARKS.search(target):
            bad("tashkeel in the answer", sid, ALL_MARKS.search(target).group(0))
        if page.get("lost_words"):
            bad("words lost to the effect", sid, str(page.get("lost_words")))
        if task == "layout":
            try:
                entries = json.loads(target)
            except ValueError:
                bad("layout target is not JSON", sid, target[:60])
                continue
            for e in entries:
                box = [int(v) for v in str(e.get("bbox", "")).split()] if isinstance(e, dict) else []
                if not isinstance(e, dict) or e.get("label") not in compact.BLOCK_LABELS:
                    bad("layout entry label not Surya's", sid, str(e)[:80])
                elif len(box) != 4 or not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000):
                    bad("layout box out of 0-1000 or empty", sid, str(e)[:80])
                elif not (isinstance(e.get("count"), int) and e["count"] >= 10 and e["count"] % 10 == 0):
                    bad("layout count not a whole ten", sid, str(e)[:80])
            continue
        if not target.strip():
            bad("empty target", sid)
        outside_pre = re.sub(r"<pre[^>]*>.*?</pre>", "", target, flags=re.S)  # code keeps its line breaks, as Surya's
        if "\n" in outside_pre or "</div>" in target:
            bad("new line (outside <pre>) or </div> in the target", sid)
        heads = list(HEADER.finditer(target))
        if task == "full_ocr" and not target.startswith("<|"):
            bad("full_ocr target does not start with a block header", sid, target[:60])
        if task == "block_ocr" and heads:
            bad("block_ocr target has a block header", sid, heads[0].group(0))
        for m in heads:
            x0, y0, x1, y1 = map(int, m.groups()[1:])
            if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
                bad("box out of 0-1000 or empty", sid, m.group(0))
            if m.group(1) not in compact.BLOCK_LABELS:
                bad("label not Surya's", sid, m.group(1))
        if compact.encode(compact.decode(target)) != target:
            bad("Surya's decode -> encode changes the target", sid)
        for m in re.finditer(r"<([a-zA-Z][a-zA-Z0-9]*)([^<>]*)>", target):
            if m.group(1).lower() not in TAGS:
                bad("tag not in Surya's list", sid, m.group(1))
            for a in re.findall(r"([a-zA-Z-]+)=", re.sub(r'"[^"]*"', '""', m.group(2))):  # not inside a value (C=C)
                if a.lower() not in ATTRS:
                    bad("attribute not in Surya's list", sid, a)
        with Image.open(paths["images"]) as im:
            width, height = im.size
            if im.mode not in ("RGB", "L"):
                bad("image mode", sid, im.mode)
        if label.get("crop") is not None and task == "full_ocr":
            bad("full_ocr sample has a crop", sid)
        if tokenizer:
            chat = f"<|im_start|>user\n<|vision_start|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"
            n = (image_tokens(width, height) + len(tokenizer.encode(chat, add_special_tokens=False).ids)
                 + len(tokenizer.encode(target, add_special_tokens=False).ids) + 1)
            lengths.append((n, sid, task))

    print(f"{folder}: {sum(tasks.values())} samples {dict(tasks)}")
    print("problems:", dict(problems) or "none")
    print("samples with each label:", dict(sorted(labels_seen.items(), key=lambda kv: -kv[1])))
    print("Surya labels never seen:", sorted(set(compact.BLOCK_LABELS) - set(labels_seen) - {"Picture"}) or "none",
          "(Picture is not generated: Surya's model has no such label)")
    for k, v in examples.items():
        print(f"  e.g. {k}: {v}")
    if lengths:
        ns = sorted(n for n, _, _ in lengths)
        over = [x for x in lengths if x[0] > limit]
        print(f"sequence length (image + prompt + answer + end): median {ns[len(ns) // 2]}, max {ns[-1]}; "
              f"{len(over)} over {limit} (left out in training){': ' + str(over[:5]) if over else ''}")


if __name__ == "__main__":
    main()
