"""Structure hash, exact-duplicate removal and bank_index.jsonl."""
import hashlib
import json
import re
import shutil

from lxml import html

from src.bank import REPORTS, ROOT, SLOT_RE, TEMPLATES, page_dims

BOX_RE = re.compile(r"(inset-inline-start|inset-block-start|inline-size|block-size):\s*(-?[\d.]+)mm")
SOURCE_ORDER = {"grammar": 0, "websight": 1, "doclaynet": 2}


def boxes(root, meta, grid=20):
    """Absolutely positioned boxes (class "box"), quantised to a grid x grid page."""
    w, h = page_dims(meta["page"]["size"], meta["page"]["orientation"])
    out = []
    for el in root.find_class("box"):
        v = dict(BOX_RE.findall(el.get("style", "")))
        out.append([round(float(v[k]) / d * grid) for k, d in
                    (("inset-inline-start", w), ("inset-block-start", h), ("inline-size", w), ("block-size", h))])
    return out


def structure_hash(text, meta):
    """sha256 of [positioning, (tag, role) per slot in DOM order, column count, quantised boxes if absolute]."""
    root = html.fromstring(text)
    roles = {s["id"]: s["role"] for s in meta["slots"]}
    seq = []
    for t in root.xpath("//text()"):  # document order; a tail text belongs to the parent of its element
        if m := SLOT_RE.fullmatch(t.strip()):
            el = t.getparent().getparent() if t.is_tail else t.getparent()
            seq.append([el.tag, roles[f"slot_{m.group(1)}"]])
    key = [meta["positioning"], seq, meta["features"]["columns"]]
    if meta["positioning"] == "absolute":
        key.append(boxes(root, meta))
    return hashlib.sha256(json.dumps(key).encode()).hexdigest()


def load_bank():
    metas = [json.loads(p.read_text(encoding="utf-8")) for p in TEMPLATES.glob("*/template.json")]
    return sorted(metas, key=lambda m: (SOURCE_ORDER[m["source"]], m["id"]))


def run():
    seen, removed = set(), []
    for meta in load_bank():
        if meta["structure_hash"] in seen:
            removed.append(meta["id"])
            shutil.rmtree(TEMPLATES / meta["id"])
        seen.add(meta["structure_hash"])
    with open(ROOT / "bank_index.jsonl", "w", encoding="utf-8") as f:
        for m in load_bank():
            f.write(json.dumps({k: m[k] for k in ("id", "source", "features", "structure_hash")}) + "\n")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "dedup.json").write_text(json.dumps({"removed": len(removed), "removed_ids": removed}, indent=2))
    print(f"dedup: removed {len(removed)} exact duplicates")
    return removed


if __name__ == "__main__":
    run()
