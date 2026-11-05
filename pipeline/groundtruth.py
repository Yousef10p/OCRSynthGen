"""Slot-level ground truth: one box and one text per filled slot, in reading order.

regions() groups a page's labelled words by the template slot they are in (the word measuring records each
word's data-slot): a region's text is its words in reading (logical) order, its box the box around their boxes
after the effects. Reading order: the page's document order for flowing layouts (what a reader follows down
the columns); for DocLayNet's positioned boxes, top to bottom, then right to left (Arabic) or left to right.

to_text() writes the format the OCR model learns to produce, one tag per region, the box at the end, as
x1, y1, x2, y2 (top-left and bottom-right corners) in the image's pixels:

    <text>المملكة العربية السعودية <bbox>50, 20, 450, 65</bbox></text>
    <text>وزارة الداخلية <bbox>50, 100, 300, 140</bbox></text>
"""


def joined(pieces, text):
    """The slot's words as written in its text. The browser may break a long word across two lines (CSS
    overflow-wrap: anywhere in a narrow box), and each part is then measured as a word: "historica" "lly".
    When the measured parts spell exactly the slot's text, the text's own spacing is used ("historically");
    otherwise the parts are joined with spaces."""
    tokens = (text or "").split()
    if not tokens or "".join(pieces) != "".join(tokens):
        return " ".join(pieces)
    out, i = [], 0
    for token in tokens:
        part = ""
        while i < len(pieces) and len(part) < len(token):
            part += pieces[i]
            i += 1
        if part != token:
            return " ".join(pieces)
        out.append(token)
    return " ".join(out)


def regions(meta_slots, words, boxes, lang, positioning, texts=None):
    """texts: each slot's text as the page holds it (the QA script's slots), to keep a word whole when the
    browser broke it across lines."""
    roles = {s["id"]: s["role"] for s in meta_slots}
    texts = texts or {}
    groups, order = {}, []
    for w, box in zip(words, boxes):
        slot = w.get("slot")
        if not box or slot not in roles:
            continue
        if slot not in groups:
            groups[slot] = []
            order.append(slot)
        groups[slot].append((w["text"], box))
    out = []
    for slot in order:
        items = groups[slot]
        x1 = min(b[0] for _, b in items)
        y1 = min(b[1] for _, b in items)
        x2 = max(b[0] + b[2] for _, b in items)
        y2 = max(b[1] + b[3] for _, b in items)
        out.append({"id": slot, "role": roles[slot], "text": joined([t for t, _ in items], texts.get(slot)),
                    "box": [x1, y1, x2 - x1, y2 - y1], "bbox": [x1, y1, x2, y2]})
    if positioning == "absolute" and out:
        band = max(8, sorted(r["box"][3] for r in out)[len(out) // 2] // 2)
        side = -1 if lang == "ar" else 1
        out.sort(key=lambda r: (r["bbox"][1] // band, side * r["bbox"][0]))
    return out


def to_text(regs):
    """The tagged ground truth: one <text> line per region, text first, its box at the end."""
    lines = []
    for r in regs:
        x1, y1, x2, y2 = (int(round(v)) for v in r["bbox"])
        text = r["text"].replace("<", "&lt;").replace(">", "&gt;")  # only what would break a tag
        lines.append(f"<text>{text} <bbox>{x1}, {y1}, {x2}, {y2}</bbox></text>")
    return "\n".join(lines) + "\n"
