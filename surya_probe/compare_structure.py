"""Surya's answers next to our Surya-style labels, structure only (labels, boxes, order, tags; not the text).

    python surya_probe/compare_structure.py surya_probe/batch100   -> prints a summary, writes batch100/comparison.json
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

HEADER = re.compile(r"<\|(?!box\|)([A-Za-z][A-Za-z-]*)\|><\|(\d+)\|><\|(\d+)\|><\|(\d+)\|><\|(\d+)\|>")
TAG = re.compile(r"<(/?[a-zA-Z][a-zA-Z0-9]*)")
END = re.compile(r"<\|im_end[!|]>\s*$")


def blocks(text):
    """[(label, box, content, count)] from a compact answer or a layout JSON answer."""
    text = END.sub("", text.strip())
    if text.startswith("["):
        text = re.sub(r"<\|(\d+)\|>", r" \1", text)  # Surya sometimes writes a coordinate as a token in its JSON
        try:
            return [(e.get("label"), [int(v) for v in str(e.get("bbox", "")).split()], "", e.get("count"))
                    for e in json.loads(text)]
        except (ValueError, AttributeError):
            pass
    marks = list(HEADER.finditer(text))
    return [(m.group(1), [int(v) for v in m.groups()[1:]], text[m.end():marks[k + 1].start() if k + 1 < len(marks) else len(text)],
             None) for k, m in enumerate(marks)]


def skeleton(content):
    """The tags in order, without attributes or text: the structure of a block."""
    return " ".join(TAG.findall(content))


def iou(a, b):
    if len(a) != 4 or len(b) != 4:
        return 0.0
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def compare_page(ours, theirs):
    a, b = blocks(ours), blocks(theirs)
    pairs = sorted(((iou(x[1], y[1]), i, j) for i, x in enumerate(a) for j, y in enumerate(b)), reverse=True)
    used_a, used_b, match = set(), set(), []
    for v, i, j in pairs:
        if v >= 0.5 and i not in used_a and j not in used_b:
            used_a.add(i), used_b.add(j), match.append((i, j, v))
    match.sort()
    order = [j for _, j, _ in match]
    return {"ours": len(a), "surya": len(b), "matched": len(match),
            "same_label": sum(a[i][0] == b[j][0] for i, j, _ in match),
            "label_pairs": [(a[i][0], b[j][0]) for i, j, _ in match if a[i][0] != b[j][0]],
            "iou": [round(v, 3) for _, _, v in match],
            "same_tags": sum(skeleton(a[i][2]) == skeleton(b[j][2]) for i, j, _ in match),
            "tag_pairs": [(a[i][0], skeleton(a[i][2])[:80], skeleton(b[j][2])[:80]) for i, j, _ in match
                          if skeleton(a[i][2]) != skeleton(b[j][2])],
            "counts": [(a[i][3], b[j][3]) for i, j, _ in match if a[i][3] is not None and b[j][3] is not None],
            "order_inversions": sum(1 for x in range(len(order)) for y in range(x + 1, len(order)) if order[x] > order[y]),
            "only_ours": [a[i][0] for i in range(len(a)) if i not in used_a],
            "only_surya": [b[j][0] for j in range(len(b)) if j not in used_b],
            "surya_is_json": theirs.strip().startswith("[")}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(sys.argv[1])
    report = {}
    for folder in sorted(p for p in root.iterdir() if (p / "surya.jsonl").exists()):
        for line in (folder / "surya.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            style = next((folder / "surya_style").glob(r["id"] + ".*"), None)
            if style is None:
                continue
            ours, theirs = style.read_text(encoding="utf-8"), r["content"]
            if r["task"] == "block_ocr":
                b = blocks(theirs)
                content = END.sub("", theirs.strip()) if not b else "".join(x[2] for x in b)
                result = {"same_tags": skeleton(ours) == skeleton(content), "surya_header": [x[0] for x in b],
                          "ours_tags": skeleton(ours)[:100], "surya_tags": skeleton(content)[:100]}
            elif r["task"] == "table":
                result = {"surya_form": "rows/cols JSON" if '"Row"' in theirs or '"Col"' in theirs
                          else "JSON" if theirs.strip().startswith("[") else "HTML blocks" if blocks(theirs) else "HTML"}
            else:
                result = compare_page(ours, theirs)
            report.setdefault(r["task"], {})[f"{folder.name}/{r['id']}"] = {"lang": r.get("lang"),
                                                                          "finish": r.get("finish_reason"), **result}
    (root / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for task, pages in report.items():
        v = list(pages.values())
        print(f"\n== {task}: {len(v)} pages answered")
        if task == "block_ocr":
            print(f"  same tag structure: {sum(p['same_tags'] for p in v)}/{len(v)}; Surya added a block header on "
                  f"{sum(bool(p['surya_header']) for p in v)}")
            print("  most common differences:", Counter((p["ours_tags"][:50], p["surya_tags"][:50]) for p in v
                                                      if not p["same_tags"]).most_common(4))
        elif task == "table":
            print("  Surya answered as:", dict(Counter(p["surya_form"] for p in v)))
        else:
            n = lambda k: sum(p[k] for p in v)  # noqa: E731
            ious = sorted(x for p in v for x in p["iou"])
            print(f"  blocks: ours {n('ours')}, Surya {n('surya')}, matched {n('matched')} (box overlap >= 0.5); "
                  f"median overlap {ious[len(ious) // 2] if ious else '-'}")
            print(f"  same label: {n('same_label')}/{n('matched')}; same tag structure: {n('same_tags')}/{n('matched')}; "
                  f"pages in the same reading order: {sum(p['order_inversions'] == 0 for p in v)}/{len(v)}")
            print("  label differences (ours, Surya):", Counter(x for p in v for x in p["label_pairs"]).most_common(6))
            print("  tag differences (label, ours, Surya):", Counter(x for p in v for x in p["tag_pairs"]).most_common(4))
            if task == "layout":
                c = [(o, s) for p in v for o, s in p["counts"]]
                print(f"  Surya answered JSON on {sum(p['surya_is_json'] for p in v)}/{len(v)} pages; count pairs "
                      f"(ours, Surya): {c[:12]}")


if __name__ == "__main__":
    main()
