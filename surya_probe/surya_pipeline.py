"""Surya's own pipeline on one saved answer: what its library would return for it. Nothing is saved.

    python surya_probe/surya_pipeline.py < {"task", "content", "image", "finish_reason"}   -> JSON on stdout

Surya's real code from the clone (surya/) runs on the answer, with a stand-in for the model that hands back the
saved answer instead of asking again:
  full_ocr   RecognitionPredictor._full_page_ocr (surya_ocr): labels renamed to Surya's public names
             (LAYOUT_PRED_RELABEL), picture blocks emptied (SKIP_OCR_LABELS: html "" and skipped), text blocks over
             a blank area dropped, boxes in pixels. An answer Surya rejects (cut by the token cap, a repetition
             loop, nothing parsed, last block cut) is redone with layout + block OCR, more model calls: reported only.
  layout     LayoutPredictor (surya_layout): the JSON parsed, labels renamed, text blocks over a blank area
             dropped, boxes in pixels. An answer that is not the JSON list (asked without the schema) fails there.
  block_ocr  clean_block_html (block mode): the compact form decoded, merged cells as colspan / rowspan.
torch and the inference servers are not needed: the package start-up (surya/__init__.py) is skipped.
"""
import json
import logging
import sys
import types
from pathlib import Path

from PIL import Image

SURYA = Path(__file__).resolve().parents[1] / "surya"
END_MARKERS = ("<|im_end!>", "<|im_end|>", "<|endoftext|>")  # openai_client._strip_end_marker


class NeedsModel(Exception):
    """Surya would ask the model again (the block-mode fallback)."""


def load_surya():
    """Surya's modules from the clone, without its start-up (torch, servers); settings as surya/settings.py has
    them by default (the values this code reads)."""
    sys.path.insert(0, str(SURYA))
    for name in ("surya", "surya.common", "surya.inference"):
        module = types.ModuleType(name)
        module.__path__ = [str(SURYA / name.replace(".", "/"))]
        sys.modules[name] = module
    sys.modules["surya.inference"].SuryaInferenceManager = object
    sys.modules["surya.inference"].get_default_manager = lambda: None
    log = types.ModuleType("surya.logging")
    log.get_logger = lambda: logging.getLogger("surya")
    sys.modules["surya.logging"] = log
    settings = types.ModuleType("surya.settings")
    settings.settings = types.SimpleNamespace(BBOX_SCALE=1000, SURYA_FULLPAGE_REGEN=False, DISABLE_TQDM=True,
                                              SURYA_MAX_TOKENS_FULL_PAGE=12288, SURYA_MAX_TOKENS_LAYOUT=3072,
                                              SURYA_MAX_TOKENS_BLOCK_CEILING=8192, SURYA_GUIDED_LAYOUT=True)
    sys.modules["surya.settings"] = settings


class SavedAnswer:
    """Stands in for SuryaInferenceManager: the first request gets the saved answer; any other is a fallback."""

    def __init__(self, raw, finish_reason):
        self.raw, self.finish_reason, self.used = raw, finish_reason, False

    def generate(self, batch):
        from surya.inference.schema import BatchOutputItem
        if self.used or len(batch) != 1:
            raise NeedsModel(batch[0].prompt_type if batch else "")
        self.used = True
        return [BatchOutputItem(raw=self.raw, token_count=0, error=False, metadata=batch[0].metadata,
                                finish_reason=self.finish_reason)]


def run(task, content, image, finish_reason):
    load_surya()
    from surya.inference.parsers import clean_block_html, html_looks_truncated, parse_full_page_html
    from surya.recognition import RecognitionPredictor, _detect_repeat_loop
    from surya.layout import LayoutPredictor
    from surya.layout.label import LAYOUT_PRED_RELABEL
    w, h = image.size
    norm = lambda bbox: [round(v * 1000 / s) for v, s in zip(bbox, (w, h, w, h))]  # noqa: E731  pixels -> 0-1000
    raw = content.rstrip()
    for marker in END_MARKERS:
        raw = raw[:-len(marker)].rstrip() if raw.endswith(marker) else raw
    manager = SavedAnswer(raw, finish_reason)
    if task == "block_ocr":
        return {"call": "block mode: clean_block_html(answer)", "status": "ok",
                "error": finish_reason == "length", "html": clean_block_html(raw)}
    if task == "layout":
        result = LayoutPredictor(manager)([image])[0]
        if result.error:
            return {"call": "LayoutPredictor (surya_layout)", "status": "error",
                    "note": "Surya's layout parser finds no JSON list in this answer (asked without the JSON schema, "
                            "the model answers in its full-page form), so surya_layout returns no blocks."}
        return {"call": "LayoutPredictor (surya_layout)", "status": "ok", "size": [w, h], "relabel": LAYOUT_PRED_RELABEL,
                "blocks": [{"order": b.position, "label": b.label, "raw_label": b.raw_label, "box": norm(b.bbox),
                            "bbox": [round(v) for v in b.bbox], "count": b.count} for b in result.bboxes]}
    try:  # full_ocr
        page = RecognitionPredictor(manager)._full_page_ocr([image])[0]
    except NeedsModel:
        why = [w for w, bad in (("cut by the token cap (finish_reason length)", finish_reason == "length"),
                                ("ends in a repetition loop", _detect_repeat_loop(raw))) if bad]
        try:
            parsed = parse_full_page_html(raw)
            if not parsed:
                why.append("nothing parsed")
            elif html_looks_truncated(raw) and finish_reason != "stop":
                why.append("its last block looks cut off")
        except Exception as e:
            why.append(f"not parsable ({type(e).__name__})")
        return {"call": "RecognitionPredictor full-page OCR (surya_ocr)", "status": "rejected",
                "note": "Surya rejects this answer (" + ("; ".join(why) or "see recognition._build_page")
                        + ") and would redo the page with layout + block OCR: more model calls, not run here."}
    parsed = parse_full_page_html(raw)
    return {"call": "RecognitionPredictor full-page OCR (surya_ocr)", "status": "ok", "size": [w, h],
            "dropped": len(parsed) - len(page.blocks), "relabel": LAYOUT_PRED_RELABEL,  # dropped: text over a blank area
            "blocks": [{"order": b.reading_order, "label": b.label, "raw_label": b.raw_label, "box": norm(b.bbox),
                        "bbox": [round(v) for v in b.bbox], "skipped": b.skipped, "html": b.html} for b in page.blocks]}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    with Image.open(request["image"]) as im:
        print(json.dumps(run(request["task"], request["content"], im.convert("RGB"), request.get("finish_reason")),
                         ensure_ascii=False))
