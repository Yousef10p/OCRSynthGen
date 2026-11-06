import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

STANDARD_TYPES = {"PERSON", "ORG", "DATE", "ADDRESS", "AMOUNT", "PHONE", "ID"}


def _human_label(template_type: str) -> str:
    """Convert template_type to human-readable label."""
    special = {
        "aia_g702_g703": "AIA G702/G703",
        "cms1500": "CMS-1500",
        "ub04_hospital_claims": "UB-04 Hospital Claims",
        "irs_form_1040": "IRS Form 1040",
        "w2_form": "W-2 Form",
        "w4_form": "W-4 Form",
        "w9_form": "W-9 Form",
        "i9_form": "I-9 Form",
        "nda": "NDA",
        "msds": "MSDS",
        "cms485_home_health": "CMS-485 Home Health",
        "adp_report": "ADP Report",
        "irs_form": "IRS Form",
        "irs_tax_transcript": "IRS Tax Transcript",
        "irs_transcript": "IRS Transcript",
        "aia_construction_pay_app": "AIA Construction Pay App",
        "schedule_k1": "Schedule K-1",
        "schedule_k1_1065": "Schedule K-1 (1065)",
        "tax_1099": "Tax 1099",
        "uspto_document": "USPTO Document",
    }
    return special.get(template_type, template_type.replace("_", " ").title())


def get_template_category(template_type: str) -> str:
    """Determine category for a template based on name patterns."""
    t = template_type.lower()

    if any(x in t for x in ["medical", "cms", "prescription", "patient", "lab_report",
                             "hospital", "health", "eob", "explanation_of_benefits", "after_visit"]):
        return "Healthcare"
    if any(x in t for x in ["legal", "pleading", "affidavit", "court", "deed", "agreement",
                             "contract", "nda", "power_of_attorney", "will", "patent", "trademark"]):
        return "Legal"
    if any(x in t for x in ["bank", "invoice", "statement", "loan", "insurance", "claim",
                             "purchase_order", "credit", "payroll", "pay_stub", "remittance"]):
        return "Financial"
    if any(x in t for x in ["tax", "irs", "w2", "w4", "w9", "i9", "1040", "1099",
                             "schedule_k1", "transcript"]):
        return "Tax & Government"
    if any(x in t for x in ["passport", "driver_license", "birth_certificate", "visa",
                             "social_security", "voter", "diploma", "certificate"]):
        return "Identity Documents"
    if any(x in t for x in ["flightary", "ticket", "itiner", "event", "restaurant", "hotel"]):
        return "Travel & Hospitality"
    if any(x in t for x in ["blueprint", "building", "construction", "permit", "site_daily",
                             "change_order", "aia"]):
        return "Construction"
    if any(x in t for x in ["work_order", "timesheet", "meeting_minutes", "order_confirmation",
                             "shipping", "delivery", "fax", "business_card"]):
        return "Business Operations"
    if any(x in t for x in ["lease", "rental", "closing", "warranty_deed", "rent_roll"]):
        return "Real Estate"
    return "General"


def make_thumbnail(img: Image.Image, max_w: int = 300) -> Image.Image:
    """Resize image to fit within max width, preserving aspect ratio."""
    if img.width <= max_w:
        return img.copy()
    ratio = max_w / img.width
    new_h = int(img.height * ratio)
    return img.resize((max_w, new_h), Image.Resampling.LANCZOS)


def scan_data_dir(data_dir: Path, site_dir: Path, run_id: str = None):
    """Scan labels and build template data in the format the viewer expects.

    Returns (templates_data, categories) where templates_data is a list of dicts
    matching the schema expected by viewer.js buildTemplateHTML().

    Supports two formats:
    - Manifest-based (new): reads manifest.json for document grouping, flat Bates filenames
    - Legacy: parses _page_ suffixes from filenames to group pages into documents

    Args:
        data_dir: Path to the data directory containing labels/ and images/
        site_dir: Path to the static site directory (None for live API mode)
        run_id: Optional run ID to include in API image URLs for correct routing
    """
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        return _scan_with_manifest(data_dir, site_dir, run_id, manifest_path)
    return _scan_legacy(data_dir, site_dir, run_id)


def _build_template_data(template_type, doc_id, pages, site_dir, run_id, data_dir):
    """Build a single template data dict from a list of (page_num, label_file, label_data, stem) tuples.

    Shared helper used by both manifest and legacy scanning.
    """
    first_data = pages[0][2]
    display_name = _human_label(template_type)
    category = get_template_category(template_type)
    total_pages = len(pages)
    orig_width = first_data.get("image_width", first_data.get("width", 816))
    orig_height = first_data.get("image_height", first_data.get("height", 1056))

    all_annotations = []
    for page_num, lf, data, stem in pages:
        for ann in data.get("annotations", []):
            if len(ann.get("box_2d", [])) != 4:
                continue
            all_annotations.append({
                "_idx": len(all_annotations),
                "label": ann.get("label", ""),
                "entity_type": ann["entity_type"],
                "text_content": (ann.get("text_content") or "")[:120],
                "box_2d": ann["box_2d"],
                "page": page_num,
                "section_id": ann.get("section_id"),
                "field_id": ann.get("field_id"),
                "table_id": ann.get("table_id"),
                "row_id": ann.get("row_id"),
                "column_header": ann.get("column_header"),
                "rdfa_typeof": ann.get("rdfa_typeof"),
                "rdfa_property": ann.get("rdfa_property"),
            })

    all_sections = []
    for page_num, lf, data, stem in pages:
        for sec in data.get("sections", []):
            if len(sec.get("box_2d", [])) != 4:
                continue
            all_sections.append({
                "_idx": len(all_sections),
                "section_id": sec.get("section_id", ""),
                "section_type": sec.get("section_type", "FormSection"),
                "box_2d": sec["box_2d"],
                "page": page_num,
                "rdfa_typeof": sec.get("rdfa_typeof"),
            })

    all_fields = []
    for page_num, lf, data, stem in pages:
        for fld in data.get("fields", []):
            if len(fld.get("box_2d", [])) != 4:
                continue
            all_fields.append({
                "_idx": len(all_fields),
                "field_id": fld.get("field_id", ""),
                "field_type": fld.get("field_type", "FormField"),
                "section_id": fld.get("section_id"),
                "box_2d": fld["box_2d"],
                "page": page_num,
                "rdfa_typeof": fld.get("rdfa_typeof"),
            })

    unique_pages = sorted(set(a["page"] for a in all_annotations)) or list(range(1, total_pages + 1))
    unique_labels = sorted(set(a["label"] for a in all_annotations if a["label"]))
    unique_types = sorted(set(a["entity_type"] for a in all_annotations if a["entity_type"]))

    page_buttons_data = []
    page_containers_data = []
    for page_num, lf, data, stem in pages:
        if site_dir:
            thumb_src = f"assets/inspector/thumbs/{stem}.jpg"
            img_src = f"assets/inspector/images/{stem}.jpg"
        else:
            qs = f"?run_id={run_id}" if run_id else ""
            thumb_src = f"/api/inspector/image/{stem}.jpg{qs}"
            img_src = f"/api/inspector/image/{stem}.jpg{qs}"

        page_buttons_data.append({
            "page_num": page_num,
            "thumb_src": thumb_src,
        })
        page_containers_data.append({
            "page_num": page_num,
            "img_src": img_src,
            "annotations": [a for a in all_annotations if a["page"] == page_num],
            "fields": [f for f in all_fields if f["page"] == page_num],
        })

    return {
        "template_name": template_type,
        "display_name": display_name,
        "doc_id": doc_id,
        "category": category,
        "template_type": template_type,
        "total_pages": total_pages,
        "annotation_count": len(all_annotations),
        "orig_width": orig_width,
        "orig_height": orig_height,
        "annotations": all_annotations,
        "sections": all_sections,
        "fields": all_fields,
        "unique_pages": unique_pages,
        "unique_labels": unique_labels,
        "unique_types": unique_types,
        "page_buttons_data": page_buttons_data,
        "page_containers_data": page_containers_data,
    }


def _scan_with_manifest(data_dir: Path, site_dir: Path, run_id: str, manifest_path: Path):
    """Scan using manifest.json for document grouping (per-page Bates format)."""
    labels_dir = data_dir / "labels"
    images_dir = data_dir / "images"

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # Group manifest entries by template (pick first document per template)
    template_doc_map = {}
    for doc_entry in manifest.get("documents", []):
        template = doc_entry["template"]
        if template not in template_doc_map:
            template_doc_map[template] = doc_entry

    templates_data = []
    categories = defaultdict(list)

    for template_type in sorted(template_doc_map.keys()):
        doc_entry = template_doc_map[template_type]
        bates_pages = doc_entry["pages"]

        # Build Bates range for display
        if doc_entry["bates_start"] == doc_entry["bates_end"]:
            doc_id = doc_entry["bates_start"]
        else:
            doc_id = f"{doc_entry['bates_start']}-{doc_entry['bates_end']}"

        # Load each page's label data
        pages = []
        for bates_id in bates_pages:
            lf = labels_dir / f"{bates_id}.json"
            img_path = images_dir / f"{bates_id}.jpg"
            if not lf.exists() or not img_path.exists():
                continue
            with open(lf, encoding="utf-8") as f:
                data = json.load(f)
            page_num = data.get("page_number", 1)
            pages.append((page_num, lf, data, bates_id))

        if not pages:
            continue

        pages.sort(key=lambda x: x[0])

        template_data = _build_template_data(template_type, doc_id, pages, site_dir, run_id, data_dir)
        templates_data.append(template_data)
        categories[template_data["category"]].append(template_data)

    return templates_data, categories


def _scan_legacy(data_dir: Path, site_dir: Path, run_id: str):
    """Scan using legacy _page_ suffix parsing for document grouping."""
    labels_dir = data_dir / "labels"
    images_dir = data_dir / "images"

    if not labels_dir.exists():
        return [], defaultdict(list)

    doc_pages = defaultdict(list)
    for lf in sorted(labels_dir.glob("*.json")):
        with open(lf, encoding="utf-8") as f:
            data = json.load(f)

        stem = lf.stem
        img_path = images_dir / f"{stem}.jpg"
        if not img_path.exists():
            continue

        if "_page_" in stem:
            doc_id = stem.rsplit("_page_", 1)[0]
        else:
            doc_id = stem

        page_num = data.get("page_number", 1)
        doc_pages[doc_id].append((page_num, lf, data, stem))

    template_code_map = {}
    for doc_id in sorted(doc_pages.keys()):
        first_data = doc_pages[doc_id][0][2]
        template_type = first_data.get("template_type", doc_id)
        if template_type not in template_code_map:
            template_code_map[template_type] = doc_id

    templates_data = []
    categories = defaultdict(list)

    for template_type in sorted(template_code_map.keys()):
        doc_id = template_code_map[template_type]
        pages = sorted(doc_pages[doc_id], key=lambda x: x[0])

        template_data = _build_template_data(template_type, doc_id, pages, site_dir, run_id, data_dir)
        templates_data.append(template_data)
        categories[template_data["category"]].append(template_data)

    return templates_data, categories


def build_thumbnails(templates_data: list, images_dir: Path, thumbs_dir: Path):
    """Generate thumbnails for all pages."""
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for tmpl in templates_data:
        for btn in tmpl["page_buttons_data"]:
            stem = btn["thumb_src"].rsplit("/", 1)[-1].replace(".jpg", "")
            thumb_path = thumbs_dir / f"{stem}.jpg"
            if thumb_path.exists():
                count += 1
                continue
            img_path = images_dir / f"{stem}.jpg"
            if not img_path.exists():
                continue
            img = Image.open(img_path)
            thumb = make_thumbnail(img, 300)
            thumb.save(thumb_path, quality=80)
            count += 1
    return count
