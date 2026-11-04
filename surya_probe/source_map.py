"""Where each piece of text on our page comes from in the page's HTML, and what Surya made of it.

For every labelled text (a data-slot in html/<id>.html): the element it sits in (h1-h6, p, li, td, th, dt, dd,
caption, figcaption, label, a, ... with <b>/<strong> if bold markup is around it) -> the Surya block whose box
holds it: Surya's label and the block's first tag. Prints the table, writes <folder>/source_map.json.

    python surya_probe/source_map.py surya_probe/clean100/full_ocr
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_structure import TAG, blocks  # noqa: E402

BLOCK = ["h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th", "dt", "dd", "caption", "figcaption", "label",
         "p", "blockquote", "header", "footer", "nav", "aside"]
folder = Path(sys.argv[1])
rows = []
for line in (folder / "surya.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    lab = json.loads((folder / "labels" / f"{r['id']}.json").read_text(encoding="utf-8"))
    width, height = lab["surya"]["page_size"]
    theirs = [b for b in blocks(r["content"]) if len(b[1]) == 4]
    soup = BeautifulSoup((folder / "html" / f"{r['id']}.html").read_text(encoding="utf-8"), "html.parser")
    for slot in lab["slots"]:
        el = soup.find(attrs={"data-slot": slot["id"]})
        if el is None:
            continue
        home = next((p.name for p in [el, *el.parents] if p.name in BLOCK), "div")
        bold = any(p.name in ("b", "strong") for p in [el, *el.parents]) or el.find(["b", "strong"]) is not None
        x0, y0, x1, y1 = slot["bbox"]
        box = [x0 * 1000 / width, y0 * 1000 / height, x1 * 1000 / width, y1 * 1000 / height]
        area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))

        def inside(t):
            w, h = min(box[2], t[2]) - max(box[0], t[0]), min(box[3], t[3]) - max(box[1], t[1])
            return w * h / area if w > 0 and h > 0 else 0
        share, j = max(((inside(t[1]), j) for j, t in enumerate(theirs)), default=(0, -1))
        tags = TAG.findall(theirs[j][2]) if share >= 0.6 else []
        rows.append({"page": r["id"], "lang": r["lang"], "role": slot["role"], "element": home + ("+b" if bold else ""),
                     "surya_label": theirs[j][0] if share >= 0.6 else "(none)", "surya_tag": tags[0] if tags else "",
                     "surya_bold": "b" in tags, "text": slot["text"][:40]})
(folder / "source_map.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
table = defaultdict(Counter)
for x in rows:
    table[(x["element"], x["role"])][f"{x['surya_label']} <{x['surya_tag']}>" + (" +b" if x["surya_bold"] else "")] += 1
sys.stdout.reconfigure(encoding="utf-8")
print(f"{len(rows)} texts on {len({x['page'] for x in rows})} pages\n")
for key, c in sorted(table.items(), key=lambda kv: -sum(kv[1].values())):
    n = sum(c.values())
    print(f"{key[0]:12s} {key[1]:14s} {n:4d}  " + ", ".join(f"{k} {v}" for k, v in c.most_common(4)))
