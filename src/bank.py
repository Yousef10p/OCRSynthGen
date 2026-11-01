"""Shared helpers for all sources: paths, roles, page sizes, base CSS, slots, writing a template folder."""
import json
import re
import shutil
from pathlib import Path

from lxml import html
from lxml.html import builder as E

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "templates"
REPORTS = ROOT / "reports"
RAW = ROOT / "data" / "raw"
ROLES = json.loads((ROOT / "config" / "roles.json").read_text(encoding="utf-8"))
PAGE_MM = {"A4": (210, 297), "A5": (148, 210), "Letter": (215.9, 279.4), "Legal": (215.9, 355.6)}
FONT_STACK = '"Noto Naskh Arabic", "Amiri", serif'
SLOT_RE = re.compile(r"\{\{ slot_(\d+) \}\}")


def page_dims(size, orientation):
    """Page (width, height) in mm."""
    w, h = PAGE_MM[size]
    return (h, w) if orientation == "landscape" else (w, h)


def root_css(size, orientation, base_size="11pt", line_height=1.6, align="justify", margin="18mm"):
    """CSS variables with defaults plus the base rules every template shares."""
    w, h = page_dims(size, orientation)
    return (
        f"@page {{ size: {size} {orientation}; margin: 0; }}\n"
        f":root {{ --font-body: {FONT_STACK}; --font-heading: {FONT_STACK}; --base-size: {base_size};"
        f" --line-height: {line_height}; --align: {align}; --page-width: {w}mm; --page-height: {h}mm;"
        f" --page-margin: {margin}; }}\n"
        "html { font-family: var(--font-body); font-size: var(--base-size); line-height: var(--line-height); }\n"
        "body { margin: 0; }\n"
        "h1, h2, h3, h4 { font-family: var(--font-heading); margin-block: 0 0.4em; }\n"
        ".page { box-sizing: border-box; inline-size: var(--page-width); min-block-size: var(--page-height);"
        " padding: var(--page-margin); text-align: var(--align); }\n"
        ".ph { background: #d0d0d0; }\n"
    )


def new_document(css):
    """Empty RTL document whose head holds the template's own CSS, then the {{ style }} hook. Returns (root, body)."""
    root = E.HTML(E.HEAD(E.META(charset="utf-8"), E.STYLE(css), E.STYLE("{{ style }}")), E.BODY())
    root.set("lang", "ar")
    root.set("dir", "rtl")
    return root, root.find("body")


def add_slot(el, slots, role, words=None):
    """Put a placeholder in `el` and record its slot. Numbers are fixed to DOM order by renumber()."""
    el.text = "{{ slot_%d }}" % len(slots)
    slots.append({"role": role, "words": list(words or ROLES[role]), "dir": "rtl"})
    return el


def renumber(root, slots):
    """Renumber placeholders 0..k-1 in DOM order; return the slot list in that order with ids."""
    ordered = []
    for el in root.iter():
        m = SLOT_RE.fullmatch(el.text or "")
        if m:
            el.text = "{{ slot_%d }}" % len(ordered)
            ordered.append({"id": f"slot_{len(ordered)}", **slots[int(m.group(1))]})
    return ordered


def to_html(root):
    return html.tostring(root, encoding="unicode", doctype="<!DOCTYPE html>", pretty_print=True)


def clear_source(prefix):
    """Remove previously generated templates of one source (prefix g_, w_ or d_)."""
    for d in TEMPLATES.glob(f"{prefix}*"):
        shutil.rmtree(d)


def write_template(meta, text, files=None):
    """Write template.html.j2, template.json and any extra files (e.g. picture crops) to templates/{id}/."""
    out = TEMPLATES / meta["id"]
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (files or {}).items():
        (out / name).write_bytes(data)
    (out / "template.html.j2").write_text(text, encoding="utf-8")
    (out / "template.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
