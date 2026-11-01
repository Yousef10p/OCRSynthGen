"""Label helpers shared by the two attack scripts (attack.py and attack_ocrsmith.py).

Needs only NumPy, since the two scripts run in different venvs (the attacks venv and OCRSmith's).
"""
import json

import numpy as np

# Paper minus ink brightness inside word boxes (0-255). Below this the text is too faint to be a fair label.
MIN_CONTRAST = 35


def union(boxes):
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    return [x0, y0, max(b[0] + b[2] for b in boxes) - x0, max(b[1] + b[3] for b in boxes) - y0]


def readable(w):
    """Keep a word unless its letters are cut. The OCR labels flag a word as clipped when its line box pokes
    out of an overflow:hidden box by half a pixel, which is every word in a tight DocLayNet box."""
    return not w["clipped"] or w["height"] >= 0.8 * w["font_size_px"]


def contrast(gray, boxes):
    """Median, over words, of paper brightness minus ink darkness inside the word box (0-255)."""
    values = [np.percentile(c, 90) - np.percentile(c, 5)
              for x, y, w, h in boxes[:200] if (c := gray[y:y + h, x:x + w]).size >= 20]
    return float(np.median(values)) if values else 0.0


def load_page(src, page):
    """SynthDoc's word/line OCR labels and slot boxes for one page, with cut words split off."""
    ocr = json.loads((src / "ocr" / f"{page}.json").read_text(encoding="utf-8"))
    slots = json.loads((src / "labels" / f"{page}.json").read_text(encoding="utf-8"))["annotations"]
    return (ocr, slots, *split_words(ocr))


def split_words(ocr):
    """(words whose letters are whole, number of cut words dropped).

    Besides words cut by a box edge, this drops words cut by the page edge: a table too wide for its column
    (English words are longer than the Arabic ones a template was laid out for) runs off the page. Their boxes
    are already trimmed to what is visible, so a clipped word that ends at the page edge counts as cut."""
    pw, ph = ocr["width"] - 0.5, ocr["height"] - 0.5

    def at_edge(w):
        return w["x"] <= 0.5 or w["y"] <= 0.5 or w["x"] + w["width"] >= pw or w["y"] + w["height"] >= ph
    # cut_x (set by the pipeline's word measuring) marks a word cut on its left or right by any box
    words = [w for w in ocr["words"] if readable(w) and not w.get("cut_x") and not (w["clipped"] and at_edge(w))]
    return words, len(ocr["words"]) - len(words)


def build_label(page, size, seed, attacks, page_contrast, ocr, slots, words, word_boxes, cut):
    """The attacked page's labels: word boxes as given, lines and slots rebuilt as the union of their words."""
    lines = [{"id": ln["id"], "text": ln["text"],
              "box": union([word_boxes[i] for i, w in enumerate(words) if w["line_id"] == ln["id"]])}
             for ln in ocr["lines"]]

    def inside(w, a):  # word centre inside the slot's original box
        cx, cy = w["x"] + w["width"] / 2, w["y"] + w["height"] / 2
        return a["x"] <= cx <= a["x"] + a["width"] and a["y"] <= cy <= a["y"] + a["height"]
    slot_boxes = [{"entity_type": a["entity_type"], "property": a.get("rdfa_property"), "text": a.get("text_content"),
                   "box": union([word_boxes[i] for i, w in enumerate(words) if inside(w, a)])} for a in slots]
    return {"page": page, "width": size[0], "height": size[1], "seed": seed,
            "attacks": attacks, "contrast": round(page_contrast, 1), "no_text": not any(word_boxes),
            "low_contrast": any(word_boxes) and page_contrast < MIN_CONTRAST,
            "words": [{"id": w["id"], "text": w["text"], "line_id": w["line_id"], "box": b}
                      for w, b in zip(words, word_boxes) if b],
            "lines": [ln for ln in lines if ln["box"]], "slots": [s for s in slot_boxes if s["box"]],
            "lost_words": sum(b is None for b in word_boxes), "cut_words_dropped": cut}


def summary(label, words):
    return {"page": label["page"], "attacks": label["attacks"], "contrast": label["contrast"],
            "low_contrast": label["low_contrast"], "no_text": label["no_text"],
            "lost_words": label["lost_words"], "words": words}
