"""Labels in the shape of Surya OCR 2.5's outputs (experimental, branch surya-format).

Each sample is made for one of Surya's four prompts, drawn per page (TASKS), and its label is what Surya
answers to that prompt. The shapes follow Surya's own outputs on our pages and on reference pages
(suryan_outputs/ in the main checkout), keeping only the answers Surya got right:

  full_ocr   the page -> one <div data-bbox="x0 y0 x1 y1" data-label="Label">content</div> per block, in reading
             order. Content is HTML: <h1>-<h3> for headers, <p> for text (with <b>/<i> when the whole block is
             bold or italic), <ul><li>• item</li></ul> for lists (the marker as drawn, inside the item),
             <table><tr><th>..</th></tr><tr><td>..</td></tr></table> for tables, and
             <img alt="Photo: ..."/><p class="img-desc">...</p> for pictures.
  layout     the page -> a JSON list of {"label", "bbox", "count"} per block, same blocks and order; count is
             Surya's estimate of the block's output length, which on the reference outputs is close to
             0.345 per Latin character + 1.05 per Arabic character of the block's HTML + 3, rounded to tens.
  block_ocr  the page -> every block, as for full_ocr: on the reference pages Surya answered this prompt with
             the same blocks, labels and order as full_ocr, header and footer text always in <p> (which ours
             always is).
  table      a table cut out of the page, with a margin -> a JSON list with one entry {"label": "Table", "bbox",
             "count"}, the box written as tokens <|x0|><|y0|><|x1|><|y1|> on the cut-out: Surya's only valid
             answer to this prompt (on a table-only image; on whole pages it never gave valid JSON). The rows and
             columns the prompt asks for are kept in the labels JSON (surya.table_structure).

Boxes are "x0 y0 x1 y1", integers normalized to 0-1000 of the image's width and height. Labels used: Page-Header,
Page-Footer, Section-Header, Text, List-Group, Table, Caption, Footnote, Image (Surya's other labels, such as
Equation-Block or Code-Block, don't occur in our templates).

full_ocr also gets Surya's raw token sequence, as its table prompt revealed it on whole pages:
<|Label|><|x0|><|y0|><|x1|><|y1|>content..., tables with their closing </tr>, </th>, </td> left out.
"""
import html as _html
import json
import re

PROMPTS = {
    "full_ocr": 'OCR this image to HTML. Each block is a div with data-label and data-bbox "(x0 y0 x1 y1, normalized 0-1000).',
    "layout": 'Output the layout of this image as JSON. Each entry is a dict with "label", "bbox", and "count" fields. Bbox is x0 y0 x1 y1, normalized 0-1000.',
    "table": 'Output the table rows then columns as JSON. Each entry is a dict with "label" ("Row" or "Col") and "bbox" (x0 y0 x1 y1, normalized 0-1000).',
    "block_ocr": "OCR this block image to HTML.",
}
TASKS = {"full_ocr": 0.4, "layout": 0.2, "block_ocr": 0.25, "table": 0.15}
SUFFIX = {"full_ocr": ".target.html", "block_ocr": ".target.html", "layout": ".target.json", "table": ".target.json"}

ROLE_LABEL = {"title": "Section-Header", "subtitle": "Section-Header", "heading": "Section-Header",
              "table_header": "Section-Header", "header": "Page-Header", "footer": "Page-Footer",
              "page_number": "Page-Footer", "caption": "Caption", "footnote": "Footnote", "list_item": "List-Group"}
ROLE_TAG = {"title": "h1", "heading": "h2", "subtitle": "h3", "table_header": "h3"}
MARKERS = {"disc": "•", "circle": "◦", "square": "▪"}
ARABIC_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
# Picture descriptions by img_pool category (Surya describes each picture; our photos have no captions yet)
PHOTOS = {
    "animals": ("An animal in its natural surroundings.", "A photograph of an animal in its natural surroundings."),
    "landscapes": ("A wide landscape view.", "A wide landscape photograph showing terrain, sky and distant scenery."),
    "nature": ("A close view of plants and natural scenery.", "A photograph of plants and natural scenery."),
    "architecture": ("A building and its surroundings.", "A photograph of a building or structure and its surroundings."),
    "people": ("People in an everyday setting.", "A photograph of people in an everyday setting."),
    "food": ("A dish of food.", "A close-up photograph of a dish of food."),
    "sports": ("A sports scene.", "A photograph of a sports activity in progress."),
    "transport": ("A vehicle.", "A photograph of a vehicle or means of transport."),
    "patterns": ("A close-up of a texture.", "A close-up photograph of a surface texture or pattern."),
    "general": ("A general scene.", "A photograph of a general scene."),
}
ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")


def esc(text):
    return _html.escape(text, quote=False)


def count(content):
    """Surya's "count" for a block: about how long its HTML output is, in tens."""
    arabic = len(ARABIC.findall(content))
    return max(10, int(round((0.345 * (len(content) - arabic) + 1.05 * arabic + 3) / 10)) * 10)


def norm(box, width, height):
    """[x0, y0, x1, y1] in pixels -> "x0 y0 x1 y1" in 0-1000."""
    x0, y0, x1, y1 = box
    v = [round(x0 / width * 1000), round(y0 / height * 1000), round(x1 / width * 1000), round(y1 / height * 1000)]
    v = [min(1000, max(0, a)) for a in v]
    return f"{v[0]} {v[1]} {max(v[0], v[2])} {max(v[1], v[3])}"


def union(boxes):
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


# -- carrying structure boxes through the effects -----------------------------------------------------------

def carry(struct):
    """[(key, rect)] of the boxes to move with the page: tables, their rows and columns, lists, pictures (page
    pixels, {x, y, width, height}). Columns are the union of the cells that span one column."""
    out = []
    for i, t in enumerate(struct.get("tables", [])):
        out.append((("table", i), t["rect"]))
        cols = {}
        for j, row in enumerate(t["rows"]):
            out.append((("row", i, j), row["rect"]))
            k = 0
            for cell in row["cells"]:
                if cell["colspan"] == 1:
                    cols.setdefault(k, []).append(cell["rect"])
                k += cell["colspan"]
        for k, rects in sorted(cols.items()):
            x0 = min(r["x"] for r in rects)
            y0 = min(r["y"] for r in rects)
            out.append((("col", i, k), {"x": x0, "y": y0, "width": max(r["x"] + r["width"] for r in rects) - x0,
                                        "height": max(r["y"] + r["height"] for r in rects) - y0}))
    for i, lst in enumerate(struct.get("lists", [])):
        out.append((("list", i), lst["rect"]))
    for i, im in enumerate(struct.get("images", [])):
        out.append((("image", i), im["rect"]))
    return out


def corners(plan):
    return [pt for _, r in plan for pt in ((r["x"], r["y"]), (r["x"] + r["width"], r["y"]),
                                           (r["x"] + r["width"], r["y"] + r["height"]), (r["x"], r["y"] + r["height"]))]


def rebuild(plan, moved, width, height):
    """{key: [x0, y0, x1, y1]} after the effect (the box of its moved corners, kept to the image); a box with
    fewer than two corners left, or nothing on the image, is left out."""
    out = {}
    for n, (key, _) in enumerate(plan):
        pts = [p for p in moved[4 * n:4 * n + 4] if p]
        if len(pts) < 2:
            continue
        x0, y0 = max(0, min(p[0] for p in pts)), max(0, min(p[1] for p in pts))
        x1, y1 = min(width, max(p[0] for p in pts)), min(height, max(p[1] for p in pts))
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            out[key] = [x0, y0, x1, y1]
    return out


# -- blocks -------------------------------------------------------------------------------------------------

def blocks(regions, roles, struct, rects, lang, positioning, categories, width, height):
    """The page's blocks in reading order: [{label, box [x0, y0, x1, y1] (image pixels), html}]."""
    by_slot = {r["id"]: r for r in regions}
    styles, order = struct.get("styles", {}), struct.get("order", {})
    used, out = set(), []
    for i, t in enumerate(struct.get("tables", [])):
        rows, boxes, members = [], [], []
        for j, row in enumerate(t["rows"]):
            cells = []
            for cell in row["cells"]:
                text = " ".join(by_slot[s]["text"] for s in cell["slots"] if s in by_slot)
                members += cell["slots"]
                inner = f"<b>{esc(text)}</b>" if text and cell["bold"] else esc(text)
                span = f' colspan="{cell["colspan"]}"' if cell["colspan"] > 1 else ""
                cells.append((cell["tag"], span, inner, bool(text)))
            if any(c[3] for c in cells):  # a row cut off the page (no labelled text) isn't in the image
                rows.append(cells)
                boxes.append(rects.get(("row", i, j)))
        used.update(members)
        if not rows:
            continue
        box = union(boxes) or rects.get(("table", i)) or union([by_slot[s]["bbox"] for s in members if s in by_slot])
        body = "".join("<tr>" + "".join(f"<{tag}{span}>{inner}</{tag}>" for tag, span, inner, _ in cells) + "</tr>"
                       for cells in rows)
        out.append({"label": "Table", "box": box, "html": f"<table>{body}</table>", "table": i,
                    "dom": t.get("order", 0)})
    for i, lst in enumerate(struct.get("lists", [])):
        items, members = [], []
        for k, slots in enumerate(lst["items"]):
            text = " ".join(by_slot[s]["text"] for s in slots if s in by_slot)
            members += slots
            if text:
                items.append((marker(lst["marker"], len(items) + 1, lang), text))
        used.update(members)
        if not items:
            continue
        box = rects.get(("list", i)) or union([by_slot[s]["bbox"] for s in members if s in by_slot])
        body = "".join(f"<li>{esc(m + ' ' + t if m else t)}</li>" for m, t in items)
        out.append({"label": "List-Group", "box": box, "html": f"<ul>{body}</ul>", "dom": lst.get("order", 0)})
    for i, im in enumerate(struct.get("images", [])):
        box = rects.get(("image", i))
        if not box:
            continue
        short, long = PHOTOS.get(categories.get(im.get("src"), "general"), PHOTOS["general"])
        out.append({"label": "Image", "box": box, "html": f'<img alt="Photo: {esc(short)}"/><p class="img-desc">{esc(long)}</p>',
                    "dom": im.get("order", 0)})
    # every other labelled slot is a block of its own; a label and the value beside it are one Text block, and
    # list items outside any list markup make one List-Group
    rest = [r for r in regions if r["id"] not in used]
    k = 0
    while k < len(rest):
        r, role = rest[k], roles.get(rest[k]["id"], "paragraph")
        nxt = rest[k + 1] if k + 1 < len(rest) else None
        if role == "label" and nxt and roles.get(nxt["id"]) == "field_value" and same_line(r["bbox"], nxt["bbox"]):
            out.append(text_block("Text", f"{r['text']} {nxt['text']}", union([r["bbox"], nxt["bbox"]]),
                                  styles.get(r["id"], {}), order.get(r["id"], 0)))
            k += 2
            continue
        if role == "list_item":
            run = [r]
            while k + len(run) < len(rest) and roles.get(rest[k + len(run)]["id"]) == "list_item":
                run.append(rest[k + len(run)])
            body = "".join(f"<li>{esc(x['text'])}</li>" for x in run)
            out.append({"label": "List-Group", "box": union([x["bbox"] for x in run]), "html": f"<ul>{body}</ul>",
                        "dom": order.get(r["id"], 0)})
            k += len(run)
            continue
        label = ROLE_LABEL.get(role, "Text")
        if label == "Section-Header":
            tag = (styles.get(r["id"]) or {}).get("h") or ROLE_TAG.get(role, "h2")
            tag = tag if tag in ("h1", "h2", "h3") else "h3"
            out.append({"label": label, "box": r["bbox"], "html": f"<{tag}>{esc(r['text'])}</{tag}>",
                        "dom": order.get(r["id"], 0)})
        else:
            out.append(text_block(label, r["text"], r["bbox"], styles.get(r["id"], {}), order.get(r["id"], 0)))
        k += 1
    out = reading_order([b for b in out if b["box"]], positioning, lang)
    return edges(out, width, height)


def edges(out, width, height, band=0.12):
    """Page-Header and Page-Footer only where Surya uses them, at the top or bottom of the page (a web page's
    links "footer" halfway down is Text). Items that touch are one block, joined by a space on one line and by
    <br/> across lines; items far apart stay separate blocks (as Surya writes them: a row of links is one
    header, three titles spread over the top are three)."""
    merged = []
    for b in out:
        if (b["label"] == "Page-Header" and b["box"][1] > band * height
                or b["label"] == "Page-Footer" and b["box"][3] < (1 - band) * height):
            b = {**b, "label": "Text"}
        prev = merged[-1] if merged else None
        if (prev and b["label"] == prev["label"] and b["label"] in ("Page-Header", "Page-Footer")
                and touching(prev["box"], b["box"], width)):
            sep = " " if same_line(prev["box"], b["box"]) else "<br/>"
            prev["html"] = f"<p>{inner(prev['html'])}{sep}{inner(b['html'])}</p>"
            prev["box"] = union([prev["box"], b["box"]])
            continue
        merged.append(dict(b))
    return merged


def inner(content):
    return content[3:-4] if content.startswith("<p>") and content.endswith("</p>") else content


def text_block(label, text, box, style, dom):
    inner = esc(text)
    if style.get("italic"):
        inner = f"<i>{inner}</i>"
    if style.get("bold") and label in ("Text", "Caption", "Footnote"):
        inner = f"<b>{inner}</b>"
    return {"label": label, "box": box, "html": f"<p>{inner}</p>", "dom": dom}


def marker(style, n, lang):
    if style in MARKERS:
        return MARKERS[style]
    if style == "none":
        return ""
    if style in ("decimal", "decimal-leading-zero"):
        return f"{n}."
    if style == "arabic-indic":
        return f"{n}.".translate(ARABIC_DIGITS)
    return "•"


def same_line(a, b):
    return min(a[3], b[3]) - max(a[1], b[1]) > 0.5 * min(a[3] - a[1], b[3] - b[1])


def touching(a, b, width):
    """Next to each other: on one line with a gap under 3% of the page width, or one under the other (sharing
    some width) with a gap under a line's height."""
    if same_line(a, b):
        return max(a[0], b[0]) - min(a[2], b[2]) < 0.03 * width
    line = min(a[3] - a[1], b[3] - b[1])
    return min(a[2], b[2]) > max(a[0], b[0]) and max(a[1], b[1]) - min(a[3], b[3]) < line


def reading_order(out, positioning, lang):
    """Flowing pages: document order (a block's place is that of its first labelled text, or of its element for
    pictures). Absolutely placed pages (DocLayNet): top to bottom in bands, then right to left for Arabic."""
    if positioning == "absolute":
        heights = sorted(b["box"][3] - b["box"][1] for b in out)
        band = max(8, heights[len(heights) // 2] // 2) if heights else 8
        side = -1 if lang == "ar" else 1
        return sorted(out, key=lambda b: (b["box"][1] // band, side * b["box"][0]))
    return sorted(out, key=lambda b: b["dom"])


# -- the targets --------------------------------------------------------------------------------------------

def full_ocr(blks, width, height):
    return "\n".join(f'<div data-bbox="{norm(b["box"], width, height)}" data-label="{b["label"]}">{b["html"]}</div>'
                     for b in blks)


def layout(blks, width, height):
    return json.dumps([{"label": b["label"], "bbox": norm(b["box"], width, height), "count": count(b["html"])}
                       for b in blks], ensure_ascii=False, indent=2)


def raw(blks, width, height):
    """Surya's token sequence for the page (closing row and cell tags dropped, as it writes tables)."""
    parts = []
    for b in blks:
        box = "".join(f"<|{v}|>" for v in norm(b["box"], width, height).split())
        content = re.sub(r"</(tr|th|td)>", "", b["html"]) if b["label"] == "Table" else b["html"]
        parts.append(f"<|{b['label']}|>{box}{content}")
    return "".join(parts)


def on_crop(box, crop):
    """A box on the page -> "x0 y0 x1 y1" in 0-1000 of the cut-out [x0, y0, x1, y1]."""
    return norm([box[0] - crop[0], box[1] - crop[1], box[2] - crop[0], box[3] - crop[1]],
                crop[2] - crop[0], crop[3] - crop[1])


def table(block, crop):
    """A table cut out of the page: one entry, as Surya answered (box as tokens, its own count)."""
    tokens = "".join(f"<|{v}|>" for v in on_crop(block["box"], crop).split())
    return json.dumps([{"label": "Table", "bbox": tokens, "count": count(block["html"])}], ensure_ascii=False, indent=2)


def table_structure(rects, i, crop):
    """Rows then columns of table i on the cut-out, as the table prompt describes them."""
    rows = [rects[k] for k in sorted(k for k in rects if k[0] == "row" and k[1] == i)]
    cols = [rects[k] for k in sorted(k for k in rects if k[0] == "col" and k[1] == i)]
    return ([{"label": "Row", "bbox": on_crop(b, crop)} for b in rows]
            + [{"label": "Col", "bbox": on_crop(b, crop)} for b in cols])


LAYOUT_LABELS = {"Caption", "Footnote", "Equation-Block", "List-Group", "Page-Header", "Page-Footer", "Image",
                 "Section-Header", "Table", "Text", "Complex-Block", "Code-Block", "Form", "Table-Of-Contents", "Figure",
                 "Chemical-Block", "Diagram", "Bibliography", "Blank-Page"}  # Surya's layout JSON schema
BBOX = re.compile(r"^\d{1,4} \d{1,4} \d{1,4} \d{1,4}$")
TOKEN_BBOX = re.compile(r"^<\|(\d{1,4})\|><\|(\d{1,4})\|><\|(\d{1,4})\|><\|(\d{1,4})\|>$")
DIV = re.compile(r'<div data-bbox="([^"]*)" data-label="([^"]*)">(.*)</div>')


def problems(task, target):
    """What is wrong with a target for Surya's format (empty list: nothing): the div lines of full_ocr and
    block_ocr, the layout JSON schema of its guided decoding, the one-entry table answer."""
    out = []

    def box_ok(b, where):
        m = TOKEN_BBOX.match(b or "") if task == "table" else BBOX.match(b or "")
        if not m:
            out.append(f"{where}: bbox {b!r}")
        else:
            x0, y0, x1, y1 = map(int, m.groups() if task == "table" else b.split())
            if x1 < x0 or y1 < y0 or max(x0, y0, x1, y1) > 1000:
                out.append(f"{where}: bbox order/range {b}")
    if task in ("full_ocr", "block_ocr"):
        for k, line in enumerate(target.split("\n")):
            m = DIV.fullmatch(line)
            if not m:
                out.append(f"line {k}: not a block div")
                continue
            box_ok(m.group(1), f"line {k}")
            if m.group(2) not in LAYOUT_LABELS:
                out.append(f"line {k}: label {m.group(2)}")
    elif task in ("layout", "table"):
        try:
            entries = json.loads(target)
        except ValueError as e:
            return [f"JSON: {e}"]
        if not isinstance(entries, list) or len(entries) > 200:
            out.append("not a list of at most 200 entries")
        labels = LAYOUT_LABELS if task == "layout" else {"Table"}
        for k, e in enumerate(entries):
            if set(e) != {"label", "bbox", "count"}:
                out.append(f"entry {k}: keys {sorted(e)}")
            if e.get("label") not in labels:
                out.append(f"entry {k}: label {e.get('label')}")
            box_ok(e.get("bbox"), f"entry {k}")
            if not (isinstance(e.get("count"), int) and e["count"] % 10 == 0):
                out.append(f"entry {k}: count {e.get('count')}")
        if task == "table" and len(entries) != 1:
            out.append("table: not one entry")
    return out


def clear(crop, others, slack=2):
    """No other block reaches into the cut-out (more than slack pixels)."""
    return not any(min(crop[2], o[2]) - max(crop[0], o[0]) > slack and min(crop[3], o[3]) - max(crop[1], o[1]) > slack
                   for o in others)


def crop_box(box, width, height, rng, others=(), margin=(6, 24), share=None):
    """A cut-out around a block: its box plus a random margin in pixels or (share) a share of the box's size on
    each side, kept to the image, and on each side short of the page's other blocks (others), so no part of a
    neighbour's text, unlabelled in the target, is in the picture."""
    if share:
        mx = max(margin[0], round(rng.uniform(*share) * (box[2] - box[0])))
        my = max(margin[0], round(rng.uniform(*share) * (box[3] - box[1])))
    else:
        mx = my = rng.randint(*margin)
    x0, y0, x1, y1 = box
    left, top, right, bottom = x0 - mx, y0 - my, x1 + mx, y1 + my
    for o in others:
        if o[2] <= left or o[0] >= right or o[3] <= top or o[1] >= bottom:
            continue  # not in the cut-out
        if o[3] <= y0:
            top = max(top, o[3] + 1)
        elif o[1] >= y1:
            bottom = min(bottom, o[1] - 1)
        elif o[2] <= x0:
            left = max(left, o[2] + 1)
        elif o[0] >= x1:
            right = min(right, o[0] - 1)
        # a block overlapping this one itself (a box drawn around it) can't be kept out
    return [max(0, int(left)), max(0, int(top)), min(width, int(right + 0.999)), min(height, int(bottom + 0.999))]
