"""Labels in Surya OCR 2.5's own output format.

Each sample is made for one of Surya's prompts, drawn per page (TASKS), and its label is the answer Surya 2.5
gives to that prompt, written the way the model writes it. Found by asking the model itself, one image at a
time (surya_probe/, surya_inference_findings.md), and by Surya's own code (datalab-to/surya: inference/prompts.py,
compact.py, parsers.py), whose prompt texts are "the model's training-time contract":

  full_ocr   the page -> compact blocks, one after another with nothing between them:
             <|Label|><|x0|><|y0|><|x1|><|y1|>content<|Label|>... The header is one label token and four coordinate
             tokens (0-1000); a block has no </div>; </td>, </th> and </tr> are left out (</table>, </p>, </li>,
             </ul>, </h1> are kept); merged cells are a data-kind="parent" cell with data-cs / data-rs followed by
             data-kind="ghost" cells. Content: <h1>-<h3>, <p> (with <b>/<i>), <ul><li>• item</li></ul> (the
             marker as drawn, inside the item), <table>, <img alt="Photo: ..."/><p class="img-desc">...</p>.
  layout     the page -> the JSON array on one line: [{"label": "Text", "bbox": "x0 y0 x1 y1", "count": 120}, ...]
             (Surya's parse_layout reads this; the model writes it under JSON-schema guided decoding). count is
             the model's estimate of the tokens it needs to OCR the block: the block's answer in Surya's own
             tokens, rounded up to tens.
  block_ocr  one block cut out of the page -> only that block's content, compact: <p>...</p>, <h2>...</h2>,
             <table>..., <ul>... (no label, no box). Pictures are not OCR'd block by block (Surya skips Image).
  table      a table cut out of the page -> rows then columns as a JSON array on one line, [{"label": "Row",
             "bbox": "x0 y0 x1 y1"}, ..., {"label": "Col", ...}] (parse_table_rec). This is Surya 2's format: the
             2.5 model card says row and column geometry needs a Surya 2 checkpoint, and 2.5 never answered it
             (its tables come from block_ocr). So it is 0% of the default mix; ask for it explicitly.

Boxes are integers normalized to 0-1000 of the image's width and height, (x0, y0) top left. Labels used:
Page-Header, Page-Footer, Section-Header, Text, List-Group, Table, Caption, Footnote, Image.
The model ends every answer with its end token (<|im_end|>, shown by this server as <|im_end!>); a target
leaves it out, as the tokenizer adds it.

The compact text itself is written by Surya's own encode() (pipeline/surya_compact.py, its compact.py copied
unchanged): this module decides the blocks, labels, tags, boxes and order as <div data-bbox data-label> HTML.
"""
import html as _html
import json
import math
import re

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from pipeline import surya_compact  # Surya's own compact.py: encode() / decode()

TOKENIZER = None  # Surya 2.5's tokenizer, loaded on first use

PROMPTS = {
    # exactly as Surya's prompts.py (HIGH_ACCURACY_BBOX_PROMPT): no quote before "(x0"
    "full_ocr": "OCR this image to HTML. Each block is a div with data-label and data-bbox (x0 y0 x1 y1, normalized 0-1000).",
    "layout": 'Output the layout of this image as JSON. Each entry is a dict with "label", "bbox", and "count" fields. Bbox is x0 y0 x1 y1, normalized 0-1000.',
    "table": 'Output the table rows then columns as JSON. Each entry is a dict with "label" ("Row" or "Col") and "bbox" (x0 y0 x1 y1, normalized 0-1000).',
    "block_ocr": "OCR this block image to HTML.",
}
TASKS = {"full_ocr": 0.4, "layout": 0.25, "block_ocr": 0.35, "table": 0.0}
SUFFIX = {"full_ocr": ".target.txt", "block_ocr": ".target.txt", "layout": ".target.json", "table": ".target.json"}

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


def esc(text):
    return _html.escape(text, quote=False)


def count(content):
    """Surya's "count" for a block: its OCR answer's length in Surya's own tokens, rounded up to tens (as Surya
    answers on 443 of 698 blocks in surya_probe/batch100; a character estimate: 411)."""
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = Tokenizer.from_file(hf_hub_download("datalab-to/surya-ocr-2.5", "tokenizer.json"))
    return max(10, math.ceil(len(TOKENIZER.encode(compact(content), add_special_tokens=False).ids) / 10) * 10)


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
        grid, boxes, members = merged_grid(t["rows"]), [], []
        rows = []
        for j, row in enumerate(grid):
            cells = []
            for cell in row:
                if cell.get("ghost"):
                    cells.append((cell["tag"], ' data-kind="ghost"', "", False))
                    continue
                text = " ".join(by_slot[s]["text"] for s in cell["slots"] if s in by_slot)
                members += cell["slots"]
                inner = f"<b>{esc(text)}</b>" if text and cell["bold"] else esc(text)
                span = "".join(f' {k}="{v}"' for k, v in (("data-cs", cell["colspan"]), ("data-rs", cell.get("rowspan", 1)))
                               if v > 1)
                cells.append((cell["tag"], span + (' data-kind="parent"' if span else ""), inner, bool(text)))
            if any(c[3] for c in cells):  # a row cut off the page (no labelled text) isn't in the image
                rows.append(cells)
                boxes.append(rects.get(("row", i, j)))
        used.update(members)
        if not rows:
            continue
        box = union(boxes) or rects.get(("table", i)) or union([by_slot[s]["bbox"] for s in members if s in by_slot])
        body = "".join("<tr>" + "".join(f"<{tag}{attrs}>{inner}</{tag}>" for tag, attrs, inner, _ in cells) + "</tr>"
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
            bold = (styles.get(r["id"]) or {}).get("bold"), (styles.get(nxt["id"]) or {}).get("bold")
            if bold[0] and not bold[1]:  # only the label bold: Surya writes <p><b>label:</b> value</p>
                out.append({"label": "Text", "box": union([r["bbox"], nxt["bbox"]]), "dom": order.get(r["id"], 0),
                            "html": f"<p><b>{esc(r['text'])}</b> {esc(nxt['text'])}</p>"})
            else:
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
        if role == "label":  # a label with no value beside it (surya_probe/batch100): a plain one is Text to Surya
            # (74 of 83), a bold one a Section-Header, most often <h3>
            if (styles.get(r["id"]) or {}).get("bold"):
                out.append({"label": "Section-Header", "box": r["bbox"], "html": f"<h3>{esc(r['text'])}</h3>",
                            "dom": order.get(r["id"], 0)})
            else:
                out.append(text_block("Text", r["text"], r["bbox"], styles.get(r["id"], {}), order.get(r["id"], 0)))
        elif label == "Section-Header":
            tag = (styles.get(r["id"]) or {}).get("h") or ROLE_TAG.get(role, "h2")
            tag = tag if tag in ("h1", "h2", "h3") else "h3"
            out.append({"label": label, "box": r["bbox"], "html": f"<{tag}>{esc(r['text'])}</{tag}>",
                        "dom": order.get(r["id"], 0)})
        else:
            out.append(text_block(label, r["text"], r["bbox"], styles.get(r["id"], {}), order.get(r["id"], 0)))
        k += 1
    out = reading_order([b for b in out if b["box"]], positioning, lang, width, height)
    heights = sorted(r["bbox"][3] - r["bbox"][1] for r in regions)
    line = heights[len(heights) // 4] if heights else 0.02 * height  # most short slots are one line
    return joined(edges(out, width, height), width, line, height)


def joined(out, width, line, height=None):
    """Neighbours Surya writes as one block (surya_probe/batch100): a list right under another list (less than
    0.8 of a line apart) is one List-Group (45 of 67), text on one line almost touching another (less than 1%
    of the page width apart) is one Text (20 of 23), and short one-line texts stacked with almost no gap
    (each at most 6 words, less than 0.2 of a line apart, in the same column: label lines, short facts) are one
    Text, its lines split by <br/> (30 of 35; 3 of 31 when further apart)."""
    shift = 0.0035 * height if height else 0  # Surya draws a block's top edge this much higher (as_surya)
    merged = []
    for b in out:
        prev = merged[-1] if merged else None
        a, c = (prev or {}).get("box"), b["box"]
        last = (prev or {}).get("last_line", a)  # a stack's lowest line
        if (prev and prev["label"] == b["label"] == "List-Group" and min(a[2], c[2]) > max(a[0], c[0])
                and c[1] - a[3] < 0.8 * line and not same_line(a, c)):
            prev["html"] = prev["html"][:-len("</ul>")] + b["html"][len("<ul>"):]
        elif (prev and prev["label"] == b["label"] == "Text" and same_line(a, c)
              and max(a[0], c[0]) - min(a[2], c[2]) < 0.01 * width):
            prev["html"] = f"<p>{inner(prev['html'])} {inner(b['html'])}</p>"
        elif (prev and (prev.get("last_line") or short_line(prev, line)) and short_line(b, line)
              and min(last[2], c[2]) - max(last[0], c[0]) >= 0.5 * min(last[2] - last[0], c[2] - c[0])
              and c[1] > last[1] + 0.5 * line and c[1] - shift - last[3] < 0.2 * line):
            prev["html"] = f"<p>{inner(prev['html'])}<br/>{inner(b['html'])}</p>"
            prev["last_line"] = c
        else:
            merged.append(dict(b))
            continue
        prev["box"] = union([a, c])
    for b in merged:
        b.pop("last_line", None)
    return merged


def short_line(b, line):
    """A Text block of one line and at most 6 words (a single <p>)."""
    return (b["label"] == "Text" and b["html"].startswith("<p>") and b["html"].count("<p>") == 1
            and b["box"][3] - b["box"][1] < 1.6 * line
            and len(_html.unescape(re.sub(r"<[^>]+>", " ", b["html"])).split()) <= 6)


def merged_grid(rows):
    """The table's rows with every merged cell written Surya 2.5's way: the cell itself (the parent, with its
    colspan / rowspan), then a ghost cell for each other grid place it covers, to its right in its row and at
    its column in the rows below."""
    out, below = [], {}  # below: {(row, column): parent} still to be filled by rowspans
    for r, row in enumerate(rows):
        line, c = [], 0
        cells = list(row["cells"])
        while cells or (r, c) in below:
            if (r, c) in below:
                line.append({"ghost": True, "tag": below.pop((r, c))["tag"]})
                c += 1
                continue
            cell = cells.pop(0)
            line.append(cell)
            for dr in range(cell.get("rowspan", 1)):
                for dc in range(cell["colspan"]):
                    if dr or dc:
                        if dr == 0:
                            line.append({"ghost": True, "tag": cell["tag"]})
                        else:
                            below[(r + dr, c + dc)] = cell
            c += cell["colspan"]
        out.append(line)
    return out


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


def reading_order(out, positioning, lang, width, height):
    """Flowing pages: document order (a block's place is that of its first labelled text, or of its element for
    pictures). Absolutely placed pages (DocLayNet): English column by column, as Surya reads them (half the
    order differences of top-to-bottom bands in surya_probe/batch100); Arabic top to bottom in bands, right to
    left (Surya doesn't read those column by column)."""
    if positioning == "absolute" and lang != "ar":
        return columns_first(out, 0.004 * width, 0.008 * height)
    if positioning == "absolute":
        heights = sorted(b["box"][3] - b["box"][1] for b in out)
        band = max(8, heights[len(heights) // 2] // 2) if heights else 8
        side = -1 if lang == "ar" else 1
        return sorted(out, key=lambda b: (b["box"][1] // band, side * b["box"][0]))
    return sorted(out, key=lambda b: b["dom"])


def columns_first(out, gap_x, gap_y):
    """Left to right in columns (an empty strip from top to bottom between them), else top to bottom in bands
    (an empty strip across), each part ordered the same way; blocks with neither left top to bottom."""
    if len(out) <= 1:
        return out
    for axis, gap in ((0, gap_x), (1, gap_y)):
        spans = sorted((b["box"][axis], b["box"][axis + 2]) for b in out)
        cuts, end = [], spans[0][1]
        for start, stop in spans[1:]:
            if start - end >= gap:
                cuts.append((end + start) / 2)
            end = max(end, stop)
        if cuts:
            parts = [[] for _ in range(len(cuts) + 1)]
            for b in out:
                parts[sum((b["box"][axis] + b["box"][axis + 2]) / 2 > x for x in cuts)].append(b)
            return [b for part in parts for b in columns_first(part, gap_x, gap_y)]
    return sorted(out, key=lambda b: (b["box"][1], b["box"][0]))


# -- the targets --------------------------------------------------------------------------------------------

def compact(content):
    """A block's HTML in Surya 2.5's spelling, by Surya's own encoder: canonical tags, </td>, </th> and </tr>
    left out where implied, formatting whitespace dropped."""
    return surya_compact.encode(content)


def on_page(b):
    """A block's HTML as Surya writes it on a whole page: a page header's text bare, without <p> (7 of the 10
    probe pages that have one). Cut out alone (block_ocr) the model can't tell it is a header and writes <p>."""
    return inner(b["html"]) if b["label"] == "Page-Header" else b["html"]


def as_surya(b, width, height):
    """A block's box as Surya draws it: its left and top edges a little further out than the text (on 1851
    blocks in surya_probe/batch100, clean pages too, the median: 6.5 and 3.5 thousandths of the page, pictures
    3.3 and 1.8; right and bottom edges the same)."""
    dx, dy = (3.3, 1.8) if b["label"] == "Image" else (6.5, 3.5)
    x0, y0, x1, y1 = b["box"]
    return [max(0, x0 - dx * width / 1000), max(0, y0 - dy * height / 1000), x1, y1]


def full_ocr(blks, width, height):
    """The full_ocr answer: the page's <div data-bbox data-label> HTML encoded by Surya's own encode() (each
    block's header tokens and compact content, nothing between blocks)."""
    return surya_compact.encode(divs(blks, width, height))


def divs(blks, width, height):
    """The same blocks as the div HTML Surya's decoder gives back (readable; kept in the labels JSON)."""
    return "\n".join(f'<div data-bbox="{norm(as_surya(b, width, height), width, height)}" data-label="{b["label"]}">{on_page(b)}</div>'
                     for b in blks)


def layout(blks, width, height):
    """The layout answer: the JSON array on one line."""
    return json.dumps([{"label": b["label"], "bbox": norm(as_surya(b, width, height), width, height), "count": count(b["html"])}
                       for b in blks], ensure_ascii=False)


def block_ocr(block):
    """The block_ocr answer for one block cut out of the page: its compact content only."""
    return compact(block["html"])


def on_crop(box, crop):
    """A box on the page -> "x0 y0 x1 y1" in 0-1000 of the cut-out [x0, y0, x1, y1]."""
    return norm([box[0] - crop[0], box[1] - crop[1], box[2] - crop[0], box[3] - crop[1]],
                crop[2] - crop[0], crop[3] - crop[1])


def table(rects, i, crop):
    """The table answer for table i cut out of the page: rows then columns, a JSON array on one line."""
    return json.dumps(table_structure(rects, i, crop), ensure_ascii=False)


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
HEADER = re.compile(r"<\|([A-Za-z][A-Za-z-]*)\|><\|(\d{1,4})\|><\|(\d{1,4})\|><\|(\d{1,4})\|><\|(\d{1,4})\|>")
CELL_END = re.compile(r"</(td|th|tr)>")


def problems(task, target):
    """What is wrong with a target for Surya 2.5's format (empty list: nothing): compact blocks for full_ocr,
    compact content without headers for block_ocr, a one-line JSON array for layout (its guided-decoding
    schema) and table (Row / Col)."""
    out = []

    def box_ok(values, where):
        x0, y0, x1, y1 = map(int, values)
        if x1 < x0 or y1 < y0 or max(x0, y0, x1, y1) > 1000:
            out.append(f"{where}: bbox order/range {' '.join(values)}")
    if task in ("full_ocr", "block_ocr"):
        if "\n" in target or "</div>" in target or "<div" in target:
            out.append("newline or div in a compact answer")
        if CELL_END.search(target):
            out.append("</td>, </th> or </tr> written out")
        heads = list(HEADER.finditer(target))
        if task == "block_ocr":
            if heads:
                out.append("block_ocr: a block header in the answer")
            if not target.strip():
                out.append("block_ocr: empty")
        else:
            if not heads or heads[0].start() != 0:
                out.append("full_ocr: does not start with a block header")
            for k, m in enumerate(heads):
                if m.group(1) not in LAYOUT_LABELS:
                    out.append(f"block {k}: label {m.group(1)}")
                box_ok(m.groups()[1:], f"block {k}")
                end = heads[k + 1].start() if k + 1 < len(heads) else len(target)
                if m.end() == end:
                    out.append(f"block {k}: empty")
    elif task in ("layout", "table"):
        if "\n" in target:
            out.append("JSON not on one line")
        try:
            entries = json.loads(target)
        except ValueError as e:
            return out + [f"JSON: {e}"]
        if not isinstance(entries, list) or len(entries) > 200:
            out.append("not a list of at most 200 entries")
        keys, labels = ({"label", "bbox", "count"}, LAYOUT_LABELS) if task == "layout" else ({"label", "bbox"}, {"Row", "Col"})
        for k, e in enumerate(entries):
            if set(e) != keys:
                out.append(f"entry {k}: keys {sorted(e)}")
            if e.get("label") not in labels:
                out.append(f"entry {k}: label {e.get('label')}")
            if not BBOX.match(e.get("bbox") or ""):
                out.append(f"entry {k}: bbox {e.get('bbox')!r}")
            else:
                box_ok(e["bbox"].split(), f"entry {k}")
            if task == "layout" and not (isinstance(e.get("count"), int) and e["count"] % 10 == 0):
                out.append(f"entry {k}: count {e.get('count')}")
        if task == "table":
            kinds = [e.get("label") for e in entries]
            if kinds != sorted(kinds, key=lambda x: x != "Row") or "Row" not in kinds or "Col" not in kinds:
                out.append("table: rows then columns, at least one of each")
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
