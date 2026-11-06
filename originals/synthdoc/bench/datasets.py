"""
Dataset loaders — DocLayNet, D4LA, and SDD in unified page format.

Every loader returns List[Dict] where each dict has:
    image_path: str
    stem: str
    template_type: str          (dataset name for external sets)
    gt_regions: List[Dict]      each {"type": str, "box_2d": [x1, y1, x2, y2]}
    width: float
    height: float
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set

# ── Category mappings ───────────────────────────────────────────────────
# Map external dataset labels → our canonical types.
# Types not in this map are kept as-is (pass through).

DOCLAYNET_TO_CANONICAL = {
    "Title": "TITLE",
    "Section-header": "SECTION_TITLE",
    "Text": "TEXT",
    "Caption": "CAPTION",
    "Table": "TABLE",
    "List-item": "LIST_ITEM",
    "Picture": "FIGURE",
    "Formula": "FORMULA",
    "Page-header": "PAGE_HEADER",
    "Page-footer": "PAGE_FOOTER",
    "Footnote": "FOOTNOTE",
}

D4LA_TO_CANONICAL = {
    "DocTitle": "TITLE",
    "ParaTitle": "SECTION_TITLE",
    "RegionTitle": "SECTION_TITLE",
    "ParaText": "TEXT",
    "OtherText": "TEXT",
    "ListText": "TEXT",
    "RegionList": "TEXT",
    "RegionKV": "FIELD_LABEL",
    "FigureName": "CAPTION",
    "TableName": "CAPTION",
    "Table": "TABLE",
    "Figure": "FIGURE",
    "Equation": "FORMULA",
    "Header": "PAGE_HEADER",
    "PageHeader": "PAGE_HEADER",
    "PageFooter": "PAGE_FOOTER",
    "PageNumber": "PAGE_FOOTER",
    "Footer": "PAGE_FOOTER",
    "Abstract": "TEXT",
    "Author": "TEXT",
    "Date": "TEXT",
    "LetterHead": "TEXT",
    "LetterDear": "TEXT",
    "LetterSign": "TEXT",
    "Question": "TEXT",
    "Number": "TEXT",
    "Reference": "TEXT",
    "Catalog": "TEXT",
}

# Overlapping types for fair cross-dataset comparison
SHARED_TYPES = {"TITLE", "SECTION_TITLE", "TEXT", "CAPTION", "TABLE"}


def _filter_types(regions: List[Dict], keep: Optional[Set[str]]) -> List[Dict]:
    """Filter regions to only include types in keep set."""
    if keep is None:
        return regions
    return [r for r in regions if r["type"] in keep]


# ── DocLayNet loader ────────────────────────────────────────────────────

def load_doclaynet(
    root: Path,
    split: str = "test",
    limit: Optional[int] = None,
    type_filter: Optional[Set[str]] = None,
) -> List[Dict]:
    """
    Load DocLayNet v1.1 (COCO format).

    Expected structure:
        root/
            COCO/
                test.json  (or val.json, train.json)
            PNG/
                test/
                    <hash>.png

    Download: https://huggingface.co/datasets/ds4sd/DocLayNet-v1.1
    """
    root = Path(root)
    coco_path = root / "COCO" / f"{split}.json"
    images_dir = root / "PNG" / split

    with open(coco_path) as f:
        coco = json.load(f)

    # Build lookups
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    img_map = {img["id"]: img for img in coco["images"]}

    # Group annotations by image
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    pages = []
    for img_id, img_info in sorted(img_map.items()):
        fname = img_info["file_name"]
        image_path = images_dir / fname
        if not image_path.exists():
            continue

        regions = []
        for ann in anns_by_img.get(img_id, []):
            cat_name = cat_map[ann["category_id"]]
            canonical = DOCLAYNET_TO_CANONICAL.get(cat_name, cat_name)
            # COCO bbox format: [x, y, width, height] → [x1, y1, x2, y2]
            bx, by, bw, bh = ann["bbox"]
            regions.append({
                "type": canonical,
                "box_2d": [bx, by, bx + bw, by + bh],
            })

        regions = _filter_types(regions, type_filter)

        pages.append({
            "image_path": str(image_path),
            "stem": Path(fname).stem,
            "template_type": "doclaynet",
            "gt_regions": regions,
            "width": float(img_info["width"]),
            "height": float(img_info["height"]),
        })

        if limit and len(pages) >= limit:
            break

    return pages


# ── D4LA loader ─────────────────────────────────────────────────────────

def load_d4la(
    root: Path,
    split: str = "val",
    limit: Optional[int] = None,
    type_filter: Optional[Set[str]] = None,
) -> List[Dict]:
    """
    Load D4LA dataset (COCO format).

    Expected structure:
        root/
            annotations/
                val.json  (or train.json)
            images/
                val/
                    <name>.jpg

    Download: https://huggingface.co/datasets/juliozhao/doclayout-yolo-D4LA
    """
    root = Path(root)
    coco_path = root / "annotations" / f"{split}.json"
    images_dir = root / "images" / split

    with open(coco_path) as f:
        coco = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    img_map = {img["id"]: img for img in coco["images"]}

    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    pages = []
    for img_id, img_info in sorted(img_map.items()):
        fname = img_info["file_name"]
        image_path = images_dir / fname
        if not image_path.exists():
            continue

        regions = []
        for ann in anns_by_img.get(img_id, []):
            cat_name = cat_map[ann["category_id"]]
            canonical = D4LA_TO_CANONICAL.get(cat_name, cat_name)
            bx, by, bw, bh = ann["bbox"]
            regions.append({
                "type": canonical,
                "box_2d": [bx, by, bx + bw, by + bh],
            })

        regions = _filter_types(regions, type_filter)

        pages.append({
            "image_path": str(image_path),
            "stem": Path(fname).stem,
            "template_type": "d4la",
            "gt_regions": regions,
            "width": float(img_info["width"]),
            "height": float(img_info["height"]),
        })

        if limit and len(pages) >= limit:
            break

    return pages


# ── SDD loader (re-export for convenience) ──────────────────────────────

def load_sdd(
    data_dir: Path,
    limit: Optional[int] = None,
    type_filter: Optional[Set[str]] = None,
) -> List[Dict]:
    """Load SDD segmentation pages (same as load_seg_pages but with optional filter)."""
    from .helpers import load_seg_pages

    pages = load_seg_pages(data_dir)
    if type_filter:
        for page in pages:
            page["gt_regions"] = _filter_types(page["gt_regions"], type_filter)
    if limit:
        pages = pages[:limit]
    return pages


# ── Utility ─────────────────────────────────────────────────────────────

def dataset_summary(pages: List[Dict]) -> Dict:
    """Quick summary of a loaded dataset."""
    from collections import Counter
    type_counts = Counter()
    for p in pages:
        for r in p["gt_regions"]:
            type_counts[r["type"]] += 1
    return {
        "n_pages": len(pages),
        "n_regions": sum(type_counts.values()),
        "type_counts": dict(type_counts.most_common()),
    }
