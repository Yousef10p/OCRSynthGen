"""
Ontology Adapter — transforms flat NER label data into 7-layer ontology JSON.

Produces documents that validate against schemas/v1.0/document.schema.json
without changing the existing label format used by sdd-finetune and benchmarks.
"""
from typing import Dict, Any, Optional


# Map generic NER entity types to ontology content block types
_ENTITY_TO_BLOCK_TYPE = {
    "FIELD_LABEL": "text",
    "TABLE_CELL": "text",
    "TABLE_HEADER": "text",
    "TITLE": "heading",
    "SECTION_TITLE": "heading",
    "TEXT": "text",
    "CAPTION": "text",
    "PERSON": "text",
    "PERSON_NAME": "text",
    "ORG": "text",
    "ORGANIZATION_NAME": "text",
    "DATE": "text",
    "ADDRESS": "text",
    "AMOUNT": "text",
    "MONEY": "text",
    "PHONE": "text",
    "PHONE_NUMBER": "text",
    "ID": "text",
    "IDENTIFIER": "text",
    "EMAIL": "text",
    "SIGNATURE": "signature",
    "BARCODE": "barcode",
    "QR_CODE": "qr_code",
    "CHECKBOX": "checkbox",
    "TABLE": "table",
}


def _box2d_to_bbox(box_2d: list) -> dict:
    """Convert [x_min, y_min, x_max, y_max] normalized 0-1000 to schema BoundingBox."""
    return {
        "x_min": box_2d[0],
        "y_min": box_2d[1],
        "x_max": box_2d[2],
        "y_max": box_2d[3],
        "coordinate_system": "normalized",
    }


def _infer_block_type(annotation: dict) -> str:
    """Infer ontology block_type from NER annotation entity_type/generic_type."""
    for key in ("generic_type", "entity_type"):
        val = annotation.get(key, "")
        if val in _ENTITY_TO_BLOCK_TYPE:
            return _ENTITY_TO_BLOCK_TYPE[val]
    return "text"


def build_ontology_document(
    label_data: Dict[str, Any],
    template_instance=None,
    page_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a 7-layer ontology document from flat NER label data.

    Args:
        label_data: The existing JSON label dict (document_id, annotations, etc.)
        template_instance: A BaseTemplate instance (for archetype/category metadata).
                          If None, uses safe defaults.
        page_info: Optional raw page_info dict from PagedJS renderer.

    Returns:
        Dict conforming to schemas/v1.0/document.schema.json
    """
    doc_id = label_data.get("document_id", "unknown")
    template_type = label_data.get("template_type", "")

    # --- Layer 1: Metadata ---
    archetype = "form"
    category = "business"
    document_type = template_type or doc_id

    if template_instance is not None:
        archetype = getattr(template_instance, "get_archetype", lambda: "form")()
        category = getattr(template_instance, "get_category", lambda: "business")()
        document_type = getattr(template_instance, "get_document_type", lambda: document_type)()

    metadata = {
        "document_id": doc_id,
        "document_type": document_type,
        "archetype": archetype,
        "language": "en-US",
        "page_count": label_data.get("total_pages", 1),
        "category": category,
    }

    # --- Layer 3: Content (blocks from annotations) ---
    blocks = []
    reading_order = []

    for idx, ann in enumerate(label_data.get("annotations", [])):
        block_id = ann.get("label", f"block_{idx}")
        block_type = _infer_block_type(ann)

        block = {
            "block_id": block_id,
            "block_type": block_type,
        }

        text = ann.get("text_content") or ann.get("text", "")
        if text:
            block["text_content"] = text

        if "box_2d" in ann:
            block["bbox"] = _box2d_to_bbox(ann["box_2d"])

        entity_type = ann.get("entity_type", "")
        if entity_type == "FIELD_LABEL":
            block["label"] = text

        blocks.append(block)
        reading_order.append(block_id)

    # Add section containers as blocks
    for sec in label_data.get("sections", []):
        sec_id = sec.get("section_id", "")
        if not sec_id:
            continue
        sec_block = {
            "block_id": sec_id,
            "block_type": "container",
        }
        if "box_2d" in sec:
            sec_block["bbox"] = _box2d_to_bbox(sec["box_2d"])
        blocks.append(sec_block)

    content = {
        "blocks": blocks,
        "reading_order": reading_order,
    }

    # --- Layer 4: Layout ---
    width = label_data.get("width", 816)
    height = label_data.get("height", 1056)

    # Infer page size from pixel dimensions (at 96 DPI: letter = 816x1056)
    if width > height:
        orientation = "landscape"
    else:
        orientation = "portrait"

    layout = {
        "page": {
            "size": "letter",
            "width": width,
            "height": height,
            "orientation": orientation,
        },
        "flow": {
            "type": "sequential",
            "alignment": "left",
        },
    }

    # --- Assemble document ---
    document = {
        "metadata": metadata,
        "content": content,
        "layout": layout,
    }

    return document
