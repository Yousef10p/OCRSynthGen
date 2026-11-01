"""Export bank templates as SynthDoc templates: synthdoc_adapter/templates/bank_<id>_template/.

Each folder holds what SynthDoc expects (template.html, template.json, a Python class) plus slots.json.
Every placeholder is wrapped in a span whose `property` SynthDoc boxes, so each slot becomes a typed
entity box in SynthDoc's labels.

    .venv/Scripts/python -m synthdoc_adapter.export --n 20
"""
import argparse
import json
import random
import re
import shutil
from pathlib import Path

from src.bank import SLOT_RE, TEMPLATES, page_dims
from src.dedup import load_bank

OUT = Path(__file__).resolve().parent / "templates"
# Slot role -> a `property` that SynthDoc's renderer collects (doco:* where one fits, else schema:*).
ROLE_PROPERTY = {"title": "doco:title", "subtitle": "doco:sectionTitle", "heading": "doco:sectionTitle",
                 "table_header": "doco:columnHeader", "cell": "doco:tableCell", "label": "doco-fd:fieldLabel",
                 "field_value": "doco-fd:fieldValue", "caption": "doco:caption", "list_item": "schema:itemListElement",
                 "quote": "schema:quotation", "footnote": "schema:comment", "header": "schema:headline",
                 "footer": "schema:creditText", "page_number": "schema:position", "sidebar": "schema:abstract"}
# Paged.js (vendored by SynthDoc) only looks for overflow to the right: with an RTL root, overflowing content
# is laid out off-page at negative x (and still labelled) instead of on a new page. As in
# ../synthdoc-arabic-test, Paged.js's own boxes stay LTR and only the content inside a page is RTL.
# Selectors match by substring because SynthDoc skips Paged.js for HTML containing its class prefix.
# A bank page is also capped at one sheet, like our own renders: Paged.js splits CSS multi-column layouts
# into thin scattered columns over many pages, so overflow is cut at the page edge instead (cut words are
# flagged by the OCR labels and dropped by attack.py).
PAGEDJS_RTL_FIX = """<style>
html, body, [class*="djs_page"], [class*="djs_sheet"], [class*="djs_area"] { direction: ltr; }
body > *, [class*="djs_page_content"] > div { direction: rtl; }
.page { max-block-size: var(--page-height); overflow: hidden; }
*, *::before, *::after { letter-spacing: normal !important; text-transform: none !important; }
* { text-overflow: clip !important; }
</style>
"""
INIT = '''from bank_template import BankTemplateBase


class BankTemplate(BankTemplateBase):
    """Bank template {id} ({source}), rendered by SynthDoc."""
'''


def tag_slots(text, slots):
    roles = {s["id"]: s["role"] for s in slots}

    def wrap(m):
        role = roles[f"slot_{m.group(1)}"]
        return (f'<span property="{ROLE_PROPERTY.get(role, "schema:text")}" data-slot="slot_{m.group(1)}"'
                f' data-role="{role}">{m.group(0)}</span>')
    return SLOT_RE.sub(wrap, text)


def synthdoc_html(meta, text=None, lang="ar", folder=None):
    """The template as SynthDoc renders it: RTL fix, page size in mm, every slot tagged with its role.

    text defaults to the bank's template file; folder is where its pictures are (default: the bank).
    lang="en" turns the page left to right for English text."""
    folder = folder or TEMPLATES / meta["id"]
    text = text or (folder / "template.html.j2").read_text(encoding="utf-8")
    # SynthDoc renders from a temporary file, so pictures are linked by absolute path
    text = text.replace('src="img_', f'src="{folder.as_uri()}/img_')
    fix = PAGEDJS_RTL_FIX if lang == "ar" else PAGEDJS_RTL_FIX.replace("direction: rtl", "direction: ltr")
    text = text.replace("</head>", fix + "</head>", 1)
    if lang != "ar":
        text = text.replace('<html lang="ar" dir="rtl">', f'<html lang="{lang}" dir="ltr">', 1)
    # Paged.js knows "A4" but not "Legal"/"Letter" written this way and falls back to Letter portrait:
    # give the page size in millimetres instead.
    w, h = page_dims(meta["page"]["size"], meta["page"]["orientation"])
    text = re.sub(r"@page \{ size: [^;]+;", f"@page {{ size: {w}mm {h}mm;", text, count=1)
    return tag_slots(text, meta["slots"])


def export(meta):
    folder = OUT / f"bank_{meta['id']}_template"
    folder.mkdir(parents=True)
    (folder / "template.html").write_text(synthdoc_html(meta), encoding="utf-8")
    (folder / "template.json").write_text(json.dumps(
        {"name": f"bank_{meta['id']}", "category": f"bank_{meta['source']}", "class": "BankTemplate"}), encoding="utf-8")
    (folder / "slots.json").write_text(json.dumps(
        {"id": meta["id"], "source": meta["source"], "slots": meta["slots"]}, ensure_ascii=False), encoding="utf-8")
    (folder / "__init__.py").write_text(INIT.format(**meta), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="templates to export, spread over the three sources (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    shutil.rmtree(OUT, ignore_errors=True)
    rng, bank = random.Random(a.seed), load_bank()
    sources = ["grammar", "websight", "doclaynet"]
    picked = bank if a.n <= 0 else [m for i, src in enumerate(sources)
                                    for m in rng.sample([m for m in bank if m["source"] == src], a.n // 3 + (i < a.n % 3))]
    for meta in picked:
        export(meta)
    print(f"exported {len(picked)} templates -> {OUT}")


if __name__ == "__main__":
    main()
