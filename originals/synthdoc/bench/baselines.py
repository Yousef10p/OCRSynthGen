"""
Baseline model wrappers for document layout segmentation.

Every function takes a list of page dicts and writes a predictions key in-place.
Predictions are List[Dict] with {"type": str, "box_2d": [x1, y1, x2, y2]}.

Models:
    - DETR (DocLayNet fine-tuned)
    - RT-DETR / Heron (Docling)
    - DocLayout-YOLO (ultralytics)
    - Florence-2 (VLM, DocLayNet fine-tuned)
    - DiT + Cascade R-CNN (optional, needs detectron2)

All functions follow the same pattern:
    1. Load model
    2. Run inference on each page, write predictions key
    3. Unload model
    4. Return None (pages modified in-place)

To swap pretrained → fine-tuned: pass checkpoint= with path to fine-tuned weights.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

log = logging.getLogger(__name__)


# ── Category reverse-maps ───────────────────────────────────────────────
# Baseline models output their native labels.  We map them to our canonical
# types so evaluate_seg() works across all models.

from .datasets import DOCLAYNET_TO_CANONICAL

# DocLayNet COCO category id → name (standard ordering)
_DOCLAYNET_ID_TO_NAME = {
    0: "Caption", 1: "Footnote", 2: "Formula", 3: "List-item",
    4: "Page-footer", 5: "Page-header", 6: "Picture", 7: "Section-header",
    8: "Table", 9: "Text", 10: "Title",
}


def _map_label(label: str, label_map: Dict[str, str]) -> str:
    """Map a predicted label to canonical type."""
    return label_map.get(label, label)


# ── DETR (DocLayNet) ────────────────────────────────────────────────────

def run_detr_doclaynet(
    pages: List[Dict],
    pred_key: str = "detr_regions",
    checkpoint: str = "cmarkea/detr-layout-detection",
    score_threshold: float = 0.5,
) -> None:
    """
    Run DETR fine-tuned on DocLayNet (11 classes).
    HuggingFace Transformers pipeline.

    pip install transformers torch
    """
    import torch
    from transformers import DetrForObjectDetection, DetrImageProcessor
    from PIL import Image

    processor = DetrImageProcessor.from_pretrained(checkpoint)
    model = DetrForObjectDetection.from_pretrained(checkpoint)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    id2label = model.config.id2label

    for i, page in enumerate(pages):
        img = Image.open(page["image_path"]).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[img.height, img.width]])
        if torch.cuda.is_available():
            target_sizes = target_sizes.cuda()
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=score_threshold
        )[0]

        regions = []
        for score, label_id, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            native_label = id2label[label_id.item()]
            canonical = _map_label(native_label, DOCLAYNET_TO_CANONICAL)
            x1, y1, x2, y2 = box.tolist()
            regions.append({
                "type": canonical,
                "box_2d": [x1, y1, x2, y2],
                "score": round(score.item(), 4),
            })

        page[pred_key] = regions
        if (i + 1) % 50 == 0:
            print(f"  DETR: {i + 1}/{len(pages)}")

    del model, processor
    torch.cuda.empty_cache()
    print(f"DETR done — {len(pages)} pages")


# ── RT-DETR / Heron (Docling) ──────────────────────────────────────────

def run_heron(
    pages: List[Dict],
    pred_key: str = "heron_regions",
    checkpoint: str = "docling-project/docling-layout-heron-101",
    score_threshold: float = 0.3,
) -> None:
    """
    Run Docling Heron (RT-DETRv2) layout model.
    Uses HuggingFace Transformers RTDetr classes.

    pip install transformers torch
    """
    import torch
    from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
    from PIL import Image

    processor = RTDetrImageProcessor.from_pretrained(checkpoint)
    model = RTDetrV2ForObjectDetection.from_pretrained(checkpoint)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    id2label = model.config.id2label

    for i, page in enumerate(pages):
        img = Image.open(page["image_path"]).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[img.height, img.width]])
        if torch.cuda.is_available():
            target_sizes = target_sizes.cuda()
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=score_threshold
        )[0]

        regions = []
        for score, label_id, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            native_label = id2label[label_id.item()]
            canonical = _map_label(native_label, DOCLAYNET_TO_CANONICAL)
            x1, y1, x2, y2 = box.tolist()
            regions.append({
                "type": canonical,
                "box_2d": [x1, y1, x2, y2],
                "score": round(score.item(), 4),
            })

        page[pred_key] = regions
        if (i + 1) % 50 == 0:
            print(f"  Heron: {i + 1}/{len(pages)}")

    del model, processor
    torch.cuda.empty_cache()
    print(f"Heron done — {len(pages)} pages")


# ── DocLayout-YOLO ──────────────────────────────────────────────────────

def run_doclayout_yolo(
    pages: List[Dict],
    pred_key: str = "doclayout_yolo_regions",
    checkpoint: str = "juliozhao/DocLayout-YOLO-DocStructBench",
    score_threshold: float = 0.25,
) -> None:
    """
    Run DocLayout-YOLO (YOLOv10 variant, document-pretrained).

    pip install ultralytics huggingface-hub
    """
    import torch
    from huggingface_hub import hf_hub_download

    # DocLayout-YOLO ships as a .pt file
    if Path(checkpoint).exists():
        model_path = checkpoint
    else:
        model_path = hf_hub_download(
            repo_id=checkpoint,
            filename="doclayout_yolo_docstructbench_imgsz1024.pt",
        )

    from ultralytics import YOLO
    model = YOLO(model_path)

    for i, page in enumerate(pages):
        results = model.predict(
            page["image_path"],
            conf=score_threshold,
            imgsz=1024,
            verbose=False,
        )[0]

        regions = []
        for box in results.boxes:
            cls_id = int(box.cls.item())
            native_label = model.names[cls_id]
            canonical = _map_label(native_label, DOCLAYNET_TO_CANONICAL)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            regions.append({
                "type": canonical,
                "box_2d": [x1, y1, x2, y2],
                "score": round(box.conf.item(), 4),
            })

        page[pred_key] = regions
        if (i + 1) % 50 == 0:
            print(f"  DocLayout-YOLO: {i + 1}/{len(pages)}")

    del model
    torch.cuda.empty_cache()
    print(f"DocLayout-YOLO done — {len(pages)} pages")


# ── Florence-2 (VLM) ───────────────────────────────────────────────────

# Florence-2 label map (DocLayNet fine-tune uses these)
_FLORENCE_LABEL_MAP = {
    "title": "TITLE",
    "section-header": "SECTION_TITLE",
    "section header": "SECTION_TITLE",
    "text": "TEXT",
    "caption": "CAPTION",
    "table": "TABLE",
    "list-item": "TEXT",
    "list item": "TEXT",
    "picture": "FIGURE",
    "figure": "FIGURE",
    "formula": "FORMULA",
    "page-header": "PAGE_HEADER",
    "page header": "PAGE_HEADER",
    "page-footer": "PAGE_FOOTER",
    "page footer": "PAGE_FOOTER",
    "footnote": "FOOTNOTE",
}


def run_florence2(
    pages: List[Dict],
    pred_key: str = "florence2_regions",
    checkpoint: str = "yifeihu/Florence-2-DocLayNet-Fixed",
    score_threshold: float = 0.0,
) -> None:
    """
    Run Florence-2 fine-tuned on DocLayNet.
    Uses <OD> task prompt for object detection.

    pip install transformers torch
    """
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM
    from PIL import Image

    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.float16, trust_remote_code=True,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    task_prompt = "<OD>"

    for i, page in enumerate(pages):
        img = Image.open(page["image_path"]).convert("RGB")
        w, h = img.size

        inputs = processor(text=task_prompt, images=img, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=1024, do_sample=False,
            )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        result = processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(w, h)
        )

        regions = []
        od_result = result.get(task_prompt, result.get("<OD>", {}))
        bboxes = od_result.get("bboxes", [])
        labels = od_result.get("labels", [])

        for bbox, label in zip(bboxes, labels):
            canonical = _FLORENCE_LABEL_MAP.get(label.lower().strip(), label)
            x1, y1, x2, y2 = bbox
            regions.append({
                "type": canonical,
                "box_2d": [float(x1), float(y1), float(x2), float(y2)],
            })

        page[pred_key] = regions
        if (i + 1) % 50 == 0:
            print(f"  Florence-2: {i + 1}/{len(pages)}")

    del model, processor
    torch.cuda.empty_cache()
    print(f"Florence-2 done — {len(pages)} pages")


# ── DiT + Cascade R-CNN (optional) ─────────────────────────────────────

def run_dit_cascade(
    pages: List[Dict],
    pred_key: str = "dit_regions",
    checkpoint: str = "microsoft/dit-large",
    config_url: str = "https://raw.githubusercontent.com/microsoft/unilm/master/dit/object_detection/publaynet_configs/cascade/cascade_dit_large.yaml",
    weights_url: Optional[str] = None,
    score_threshold: float = 0.5,
) -> None:
    """
    Run DiT + Cascade R-CNN via detectron2.
    Requires: pip install detectron2
    Optional — skip if detectron2 not installed.
    """
    try:
        import detectron2
    except ImportError:
        print("DiT requires detectron2 — skipping. Install: pip install detectron2")
        for page in pages:
            page[pred_key] = []
        return

    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.data import MetadataCatalog
    from PIL import Image
    import numpy as np
    import torch

    cfg = get_cfg()
    cfg.merge_from_file(config_url)
    if weights_url:
        cfg.MODEL.WEIGHTS = weights_url
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = DefaultPredictor(cfg)

    # PubLayNet class names (DiT default)
    dit_labels = {0: "TEXT", 1: "TITLE", 2: "TEXT", 3: "TABLE", 4: "FIGURE"}

    for i, page in enumerate(pages):
        img = np.array(Image.open(page["image_path"]).convert("RGB"))
        outputs = predictor(img)

        instances = outputs["instances"].to("cpu")
        regions = []
        for j in range(len(instances)):
            cls_id = instances.pred_classes[j].item()
            canonical = dit_labels.get(cls_id, "TEXT")
            box = instances.pred_boxes[j].tensor[0].tolist()
            score = instances.scores[j].item()
            regions.append({
                "type": canonical,
                "box_2d": box,
                "score": round(score, 4),
            })

        page[pred_key] = regions
        if (i + 1) % 50 == 0:
            print(f"  DiT: {i + 1}/{len(pages)}")

    del predictor
    torch.cuda.empty_cache()
    print(f"DiT done — {len(pages)} pages")


# ── VLM wrappers (re-export from live.py for convenience) ──────────────

def run_qwenvl_seg(pages, pred_key="qwenvl_regions"):
    """Qwen3-VL segmentation — delegates to live.py."""
    from .live import run_qwenvl_seg as _run
    _run(pages)


def run_deepseek_seg(pages, pred_key="deepseek_regions"):
    """DeepSeek-OCR-2 segmentation — delegates to live.py."""
    from .live import run_deepseek_seg as _run
    _run(pages)


# ── Registry ────────────────────────────────────────────────────────────

BASELINE_MODELS = {
    "detr": {
        "fn": run_detr_doclaynet,
        "pred_key": "detr_regions",
        "checkpoint": "cmarkea/detr-layout-detection",
        "type": "detector",
    },
    "heron": {
        "fn": run_heron,
        "pred_key": "heron_regions",
        "checkpoint": "docling-project/docling-layout-heron-101",
        "type": "detector",
    },
    "doclayout-yolo": {
        "fn": run_doclayout_yolo,
        "pred_key": "doclayout_yolo_regions",
        "checkpoint": "juliozhao/DocLayout-YOLO-DocStructBench",
        "type": "detector",
    },
    "florence2": {
        "fn": run_florence2,
        "pred_key": "florence2_regions",
        "checkpoint": "yifeihu/Florence-2-DocLayNet-Fixed",
        "type": "vlm",
    },
    "dit": {
        "fn": run_dit_cascade,
        "pred_key": "dit_regions",
        "checkpoint": "microsoft/dit-large",
        "type": "detector",
    },
    "qwen-vl": {
        "fn": run_qwenvl_seg,
        "pred_key": "qwenvl_regions",
        "checkpoint": "Qwen/Qwen3-VL-8B-Instruct",
        "type": "vlm",
    },
    "deepseek": {
        "fn": run_deepseek_seg,
        "pred_key": "deepseek_regions",
        "checkpoint": "deepseek-ai/DeepSeek-OCR-2",
        "type": "vlm",
    },
}
