"""What Surya OCR 2.5's answers look like: blocks, tags, differences between prompts (surya_probe/results.jsonl)."""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

BLOCK = re.compile(r"<\|([A-Za-z][A-Za-z-]*)\|><\|(\d+)\|><\|(\d+)\|><\|(\d+)\|><\|(\d+)\|>")
TAG = re.compile(r"</?([a-zA-Z][a-zA-Z0-9]*)[^>]*>")


def parse(content):
    """[(label, [x0, y0, x1, y1], html)] and whatever comes before the first block."""
    marks = list(BLOCK.finditer(content))
    head = content[:marks[0].start()] if marks else content
    out = []
    for k, m in enumerate(marks):
        end = marks[k + 1].start() if k + 1 < len(marks) else len(content)
        out.append((m.group(1), [int(v) for v in m.groups()[1:]], content[m.end():end]))
    return head, out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "surya_probe/results.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    ok = [r for r in rows if r["status"] == 200]
    print("answers:", len(rows), "| ok:", len(ok), "| by status:", dict(Counter(r["status"] for r in rows)),
          "| schema on -> status:", dict(Counter(r["status"] for r in rows if r["schema"])))
    print("finish reasons:", dict(Counter((r["max_tokens"], r.get("finish_reason")) for r in ok)))
    labels, tags, closing, starts = Counter(), defaultdict(Counter), Counter(), Counter()
    by_image = defaultdict(dict)
    for r in ok:
        if r["max_tokens"] != 4096 and any(x["image"] == r["image"] and x["task"] == r["task"] and x["max_tokens"] == 4096
                                          and x["status"] == 200 for x in ok):
            continue  # keep the longer answer
        head, blocks = parse(r["content"])
        by_image[r["image"]][r["task"]] = (head, blocks, r)
        starts[head[:12] if head else "<block>"] += 1
        for label, box, html in blocks:
            labels[label] += 1
            for t in TAG.findall(html):
                tags[label][t.lower()] += 1
            for t in ("tr", "td", "th", "li", "p", "ul", "table", "h1", "h2", "h3"):
                closing[(t, f"</{t}>" in html, f"<{t}" in html)] += 1
    print("\nwhat answers start with:", dict(starts))
    print("\nblock labels:", dict(labels))
    print("\ntags inside blocks, by label:")
    for label, c in sorted(tags.items()):
        print(f"  {label:15s} {dict(c.most_common(12))}")
    print("\nclosing tags (tag, closed?, opened?) -> blocks:",
          {k: v for k, v in sorted(closing.items()) if k[2]})
    print("\nprompts compared on the same image (same blocks? same labels?):")
    same = Counter()
    for image, answers in sorted(by_image.items()):
        if "full_ocr" not in answers:
            continue
        base = answers["full_ocr"][1]
        row = []
        for task in ("block_ocr", "layout", "table"):
            if task in answers:
                b = answers[task][1]
                s = ("identical" if answers[task][2]["content"] == answers["full_ocr"][2]["content"]
                     else "same labels" if [x[0] for x in b] == [x[0] for x in base] else f"different ({len(b)} vs {len(base)} blocks)")
                same[(task, s.split(" (")[0])] += 1
                row.append(f"{task}: {s}")
        print(f"  {Path(image).name:34s} full_ocr {len(base):2d} blocks | " + " | ".join(row))
    print("\nsummary:", dict(same))


if __name__ == "__main__":
    main()
