"""
Run SynthDoc's own ``generate()`` on the Arabic templates and add OCR ground truth.

Nothing in the SynthDoc checkout is edited. Three functions are wrapped at
runtime instead:

* ``PagedJSRenderer.extract_page_data`` also measures every visible word (see
  :mod:`synthdoc_arabic.ocr_gt`) in the same page, and repairs table field ids
  for Arabic column headers (see :func:`fix_arabic_field_ids`);
* ``PagedJSRenderer.render_and_extract`` saves those measurements under the
  document's temporary name;
* ``synthdoc.cli._assign_bates_and_rename`` maps temporary names to the final
  Bates ids and writes ``ocr/<id>.json`` and ``ocr_overlays/<id>_ocr.jpg``.
"""
import json
import re
import shutil
import tempfile
import threading
from pathlib import Path

import synthdoc.cli as cli
from synthdoc.pagedjs_renderer import PagedJSRenderer

from .ocr_gt import WORDS_JS, build_page_gt, draw_overlay, write_json

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"

_state = threading.local()
_raw_dir = None


def fix_arabic_field_ids(page_data) -> int:
    """Give Arabic table columns distinct field ids.

    SynthDoc names a table cell's field ``<table>_<slug(column header)>`` with an
    ASCII-only slug (``/[^a-z0-9]+/g -> '_'``). An Arabic header slugs to nothing,
    so every column of a table shares the id ``<table>_``. Re-slug those with a
    Unicode-aware pattern.
    """
    fixed = 0
    for page in page_data:
        for ann in page.get("annotations", []):
            fid, header = ann.get("field_id"), ann.get("column_header")
            if fid and header and fid.endswith("_") and not re.search(r"[a-z0-9]", header.lower()):
                slug = re.sub(r"\W+", "_", header).strip("_")
                if slug:
                    ann["field_id"] = fid + slug
                    fixed += 1
    return fixed


def _extract_page_data(orig):
    def extract_page_data(self, page):
        data = orig(self, page)
        fix_arabic_field_ids(data)
        _state.ocr = page.evaluate(WORDS_JS)
        return data
    return extract_page_data


def _render_and_extract(orig):
    def render_and_extract(self, *args, **kwargs):
        _state.ocr = None
        result = orig(self, *args, **kwargs)
        base_name, ocr = kwargs.get("base_name"), _state.ocr
        if ocr and base_name:
            pages = ocr["pages"]
            for page in pages:
                stem = f"{base_name}_page_{page['page_number']}" if len(pages) > 1 else base_name
                write_json(_raw_dir / f"{stem}.json",
                           {"lang": ocr["lang"], "dir": ocr["dir"], "variant": ocr["variant"], "page": page})
        return result
    return render_and_extract


def _finalize(all_results, manifest, data_dir: Path) -> None:
    docs = [doc for _, doc_list in all_results for doc in doc_list]
    for doc, entry in zip(docs, manifest["documents"]):
        for page_info, bates_id in zip(doc["pages"], entry["pages"]):
            raw_path = _raw_dir / f"{Path(page_info['temp_label']).stem}.json"
            if not raw_path.exists():
                print(f"  [ocr] no measurements for {bates_id}")
                continue
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            gt = build_page_gt(raw["page"], {
                "document_id": bates_id,
                "image": f"images/{bates_id}.jpg",
                "template": entry["template"],
                "document_bates_range": [entry["bates_start"], entry["bates_end"]],
                "total_pages": len(entry["pages"]),
                "language": raw["lang"],
                "direction": raw["dir"],
                "variant": raw["variant"],
            })
            write_json(data_dir / "ocr" / f"{bates_id}.json", gt)
            draw_overlay(data_dir / "images" / f"{bates_id}.jpg", gt,
                         data_dir / "ocr_overlays" / f"{bates_id}_ocr.jpg")


def _assign_bates_and_rename(orig):
    def assign(all_results, data_dir, keep_pdfs):
        manifest, total_docs, total_pages = orig(all_results, data_dir, keep_pdfs)
        print("Writing OCR ground truth (ocr/, ocr_overlays/)...")
        _finalize(all_results, manifest, Path(data_dir))
        return manifest, total_docs, total_pages
    return assign


def install() -> None:
    """Wrap the three SynthDoc functions (idempotent)."""
    global _raw_dir
    if getattr(PagedJSRenderer, "_synthdoc_arabic", False):
        return
    _raw_dir = Path(tempfile.mkdtemp(prefix="synthdoc_ar_ocr_"))
    PagedJSRenderer.extract_page_data = _extract_page_data(PagedJSRenderer.extract_page_data)
    PagedJSRenderer.render_and_extract = _render_and_extract(PagedJSRenderer.render_and_extract)
    cli._assign_bates_and_rename = _assign_bates_and_rename(cli._assign_bates_and_rename)
    PagedJSRenderer._synthdoc_arabic = True


def cleanup() -> None:
    if _raw_dir is not None:
        shutil.rmtree(_raw_dir, ignore_errors=True)
