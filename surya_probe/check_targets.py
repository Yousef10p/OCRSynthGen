"""Our full_ocr and block_ocr targets next to Surya OCR 2.5's answer for the same image.

    python surya_probe/check_targets.py make 12 16     -> surya_probe/_check/: 12 full_ocr and 16 block_ocr samples
                                                          from the current pipeline, and full.txt / block.txt (images)
    python surya_probe/probe.py @full  --tasks full_ocr  --schema off --max-tokens 4096 --out surya_probe/check.jsonl
    python surya_probe/probe.py @block --tasks block_ocr --schema off --max-tokens 4096 --out surya_probe/check.jsonl
    python surya_probe/check_targets.py compare        -> per sample and in total: blocks matched by box, labels,
                                                          text, tags, reading order (surya_probe/check_report.json)
    python surya_probe/check_targets.py compare-probe  -> the same for the probe's 12 pages and 28 cut-outs
                                                          (results.jsonl), without new requests

The text comparison is the character error rate (edit distance / our length) of the text without tags, once as
written and once without diacritics (Surya often reads tashkeel differently from what the page shows).
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE / "_check"
HEADER = re.compile(r"<\|(?!box\|)([A-Za-z][A-Za-z-]*)\|><\|(\d{1,4})\|><\|(\d{1,4})\|><\|(\d{1,4})\|><\|(\d{1,4})\|>")
BOX = re.compile(r"<\|box\|>(?:<\|\d{1,4}\|>){4}")
TAG = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)([^>]*)>")
END = "<|im_end!>"
MARKS = re.compile(r"[ً-ْٰـ]")  # tashkeel, superscript alef, tatweel


def make(n_full, n_block):
    sys.path.insert(0, str(HERE.parent))
    from pipeline.generator import DocPipeline
    WORK.mkdir(exist_ok=True)
    lists = {"full_ocr": [], "block_ocr": []}
    k = 0
    for task, n in (("full_ocr", n_full), ("block_ocr", n_block)):
        for i in range(n):
            lang = "ar" if i % 2 == 0 else "en"
            pipe = DocPipeline(WORK, languages={lang: 1}, tasks={task: 1}, extras=False)
            name = f"{'f' if task == 'full_ocr' else 'b'}{i:02d}"
            s = pipe.sample(70000 + k, name)
            k += 1
            label = json.loads((WORK / f"{name}.json").read_text(encoding="utf-8"))
            if label["surya"]["task"] != task:  # block_ocr falls back to full_ocr when no block can be cut out
                print("skip", name, "became", label["surya"]["task"], flush=True)
                continue
            lists[task].append(str(WORK / f"{name}.jpg"))
            print(name, task, lang, s.get("effect"), flush=True)
    (WORK / "full.txt").write_text("\n".join(lists["full_ocr"]), encoding="utf-8")
    (WORK / "block.txt").write_text("\n".join(lists["block_ocr"]), encoding="utf-8")


def blocks(text):
    """[(label, box, content)] of a compact answer; content-only answers give one block (None, None, text)."""
    text = text.replace(END, "")
    marks = list(HEADER.finditer(text))
    if not marks:
        return [(None, None, text)]
    out = []
    lead = text[:marks[0].start()]
    if lead.strip():
        out.append((None, None, lead))
    for k, m in enumerate(marks):
        end = marks[k + 1].start() if k + 1 < len(marks) else len(text)
        out.append((m.group(1), [int(v) for v in m.groups()[1:]], text[m.end():end]))
    return out


def plain(content):
    return re.sub(r"\s+", " ", TAG.sub(" ", BOX.sub("", content))).strip()


def tags(content):
    """The tag names in order (opening and closing), without attributes."""
    return [("/" if m.group(1) else "") + m.group(2).lower() for m in TAG.finditer(BOX.sub("", content))]


def distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ours, theirs):
    return distance(ours, theirs) / max(1, len(ours))


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def inversions(seq):
    return sum(1 for i in range(len(seq)) for j in range(i + 1, len(seq)) if seq[i] > seq[j])


def compare_full(ours, theirs):
    a, b = blocks(ours), blocks(theirs)
    pairs = sorted(((iou(x[1], y[1]), i, j) for i, x in enumerate(a) for j, y in enumerate(b) if x[1] and y[1]),
                   reverse=True)
    used_a, used_b, match = set(), set(), []
    for v, i, j in pairs:
        if v >= 0.5 and i not in used_a and j not in used_b:
            used_a.add(i), used_b.add(j), match.append((i, j, v))
    match.sort()
    rows = []
    for i, j, v in match:
        x, y = a[i], b[j]
        rows.append({"ours": x[0], "surya": y[0], "iou": round(v, 3),
                     "cer": round(cer(plain(x[2]), plain(y[2])), 3),
                     "cer_no_marks": round(cer(MARKS.sub("", plain(x[2])), MARKS.sub("", plain(y[2]))), 3),
                     "same_tags": tags(x[2]) == tags(y[2]), "ours_tags": " ".join(tags(x[2])[:12]),
                     "surya_tags": " ".join(tags(y[2])[:12])})
    return {"ours_blocks": len(a), "surya_blocks": len(b), "matched": len(match),
            "only_ours": [(a[i][0], plain(a[i][2])[:60]) for i in range(len(a)) if i not in used_a],
            "only_surya": [(b[j][0], plain(b[j][2])[:60]) for j in range(len(b)) if j not in used_b],
            "order_inversions": inversions([j for _, j, _ in match]), "pairs": rows,
            "surya_has_newline": "\n" in theirs, "surya_closes_cells": bool(re.search(r"</t[dhr]>", theirs))}


def compare_block(ours, theirs):
    b = blocks(theirs)
    content = "".join(c for _, _, c in b)
    return {"surya_has_header": b[0][0] is not None, "surya_label": b[0][0], "surya_blocks": len(b),
            "cer": round(cer(plain(ours), plain(content)), 3),
            "cer_no_marks": round(cer(MARKS.sub("", plain(ours)), MARKS.sub("", plain(content))), 3),
            "same_tags": tags(ours) == tags(content), "ours_tags": " ".join(tags(ours)[:14]),
            "surya_tags": " ".join(tags(content)[:14])}


def cut_label(surya):
    """The label of the block a block_ocr cut-out shows (the one block inside the crop)."""
    c = surya.get("crop")
    if not c:
        return None
    inside = [b["label"] for b in surya["blocks"]
              if b["box"][0] >= c[0] - 1 and b["box"][1] >= c[1] - 1 and b["box"][2] <= c[2] + 1 and b["box"][3] <= c[3] + 1]
    return inside[0] if len(inside) == 1 else None


def compare(results=HERE / "check.jsonl"):
    sys.stdout.reconfigure(encoding="utf-8")
    report = {}
    if results == "probe":
        rows = compare_probe()
    else:
        rows = []
        for line in results.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            path = Path(r["image"])
            label = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            rows.append((path.stem, None, label, label["surya"]["target"], r))
    for name, d, label, ours, r in rows:
        if r["status"] != 200:
            report[name] = {"task": r["task"], "error": r["status"]}
            continue
        result = (compare_full if r["task"] == "full_ocr" else compare_block)(ours, r["content"])
        block = d["label"] if d and d["kind"] == "cut" else cut_label(label["surya"]) if "surya" in label else None
        report[name] = {"task": r["task"], "lang": label.get("lang"), "effect": label.get("effect"),
                        "finish": r.get("finish_reason"), "block": block, **result}
    out = "check_report_probe.json" if results == "probe" else "check_report.json"
    (HERE / out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    full = [v for v in report.values() if v.get("task") == "full_ocr" and "error" not in v]
    block = [v for v in report.values() if v.get("task") == "block_ocr" and "error" not in v]
    pairs = [p for v in full for p in v["pairs"]]
    if full:
        print(f"full_ocr: {len(full)} pages, our blocks {sum(v['ours_blocks'] for v in full)}, Surya's "
              f"{sum(v['surya_blocks'] for v in full)}, matched (IoU >= 0.5) {len(pairs)}")
        print("  same label:", sum(p["ours"] == p["surya"] for p in pairs), "/", len(pairs),
              "| label pairs that differ:", dict(Counter((p["ours"], p["surya"]) for p in pairs if p["ours"] != p["surya"])))
        print("  median IoU:", sorted(p["iou"] for p in pairs)[len(pairs) // 2] if pairs else None,
              "| median CER:", sorted(p["cer"] for p in pairs)[len(pairs) // 2] if pairs else None,
              "| without diacritics:", sorted(p["cer_no_marks"] for p in pairs)[len(pairs) // 2] if pairs else None)
        print("  same tags:", sum(p["same_tags"] for p in pairs), "/", len(pairs),
              "| reading-order inversions:", [v["order_inversions"] for v in full],
              "| newline in Surya:", sum(v["surya_has_newline"] for v in full),
              "| Surya closes cells:", sum(v["surya_closes_cells"] for v in full))
    if block:
        print(f"block_ocr: {len(block)} cut-outs, Surya added a header: {sum(v['surya_has_header'] for v in block)}, "
              f"same tags: {sum(v['same_tags'] for v in block)}, median CER: "
              f"{sorted(v['cer'] for v in block)[len(block) // 2]}, without diacritics: "
              f"{sorted(v['cer_no_marks'] for v in block)[len(block) // 2]}")


def compare_probe():
    """The same comparison on the probe's images (surya_probe/inputs): our targets as the current surya.py writes
    them from the blocks kept at probe time, next to Surya's answers in results.jsonl (4,096-token runs)."""
    sys.path.insert(0, str(HERE.parent))
    from pipeline import generator as SURYA
    from surya.inference.compact import encode
    index = {d["image"]: d for d in json.loads((HERE / "inputs" / "index.json").read_text(encoding="utf-8"))}
    lines = []
    for line in (HERE / "results.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        name = Path(r["image"]).name
        if r["max_tokens"] != 4096 or r["schema"] or name not in index:
            continue
        d = index[name]
        task = "full_ocr" if d["kind"] == "page" else "block_ocr"
        if r["task"] != task:
            continue
        if d["kind"] == "page":
            label = json.loads((HERE / "_work" / name.replace("_page.jpg", ".json")).read_text(encoding="utf-8"))
            ours = encode(SURYA.divs(label["surya"]["blocks"], *d["size"]))
        else:
            label = {"lang": d["lang"], "effect": d["effect"]}
            ours = encode(d["ours_html"])
        lines.append((name, d, label, ours, r))
    return lines


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(int(sys.argv[2]), int(sys.argv[3]))
    else:
        compare("probe" if sys.argv[1] == "compare-probe" else HERE / "check.jsonl")
