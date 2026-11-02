"""Source 1: random RTL document templates built from a small block grammar with lxml.

generate(seed, tid) is the first build's grammar (300 templates, unchanged so they stay reproducible).
generate(seed, tid, rich=True) adds more kinds of block (card grids with pictures, key/value lists, stat boxes,
timelines, picture-beside-text rows, callouts, signature lines) and more page layouts (letter, report with a
title band, two-band header), for the 2,000-template bank.

Run on its own:  python -m src.grammar --seed 0 --n 300
"""
import argparse
import random

from lxml.html import builder as E

from src.bank import add_slot, clear_source, new_document, renumber, root_css, to_html, write_template
from src.dedup import structure_hash

PARAGRAPH_WORDS = {"S": [8, 25], "M": [25, 60], "L": [60, 120]}
BLOCK_WEIGHTS = {"heading": 14, "paragraph": 30, "list": 12, "table": 10, "form": 6, "aside": 5, "quote": 4, "figure": 6}
BORDERS = ["1px solid #000", "1px solid #888", "1px dashed #555", "3px double #000", "rows", "none"]
RICH_WEIGHTS = {**BLOCK_WEIGHTS, "paragraph": 24, "cards": 6, "keyvalue": 6, "stats": 4, "timeline": 4, "media": 6,
                "callout": 5}
# Picture hints: what a grammar picture shows (img_pool picks a photo of that kind)
PICTURE_HINTS = ["team", "office", "city", "nature", "food", "travel", "product", "people", "architecture",
                 "landscape", "animal", "sport", "car", "pattern", "garden", "event"]
# Abjad letters (alif, ba, jim, dal, ha, waw, zay, ha, ta, ya) as CSS escapes, so the file has no literal text.
ARABIC_LETTERS = ('@counter-style arabic-letters { system: alphabetic; symbols: "\\0623" "\\0628" "\\062C"'
                  ' "\\062F" "\\0647" "\\0648" "\\0632" "\\062D" "\\0637" "\\064A"; suffix: ") "; }\n')


def css(**props):
    """css(margin_block="1em", text_align="start") -> 'margin-block: 1em; text-align: start'."""
    return "; ".join(f"{k.replace('_', '-')}: {v}" for k, v in props.items())


def heading(rng, slots, f, ctx):
    el = add_slot(rng.choice([E.H2, E.H2, E.H3])(), slots, "heading")
    style = css(font_size=f"calc(var(--base-size) * {rng.choice([1.15, 1.3, 1.5])})",
                text_align=rng.choice(["start", "start", "center", "end"]),
                margin_block=f"{rng.choice([0.6, 1, 1.4])}em 0.4em")
    if rng.random() < 0.25:
        style += "; " + css(border_block_end="1px solid #000", padding_block_end="0.15em")
    el.set("style", style)
    return el


def paragraph(rng, slots, f, ctx):
    return add_slot(E.P(style=css(margin_block=f"0 {rng.choice([0.4, 0.7, 1])}em",
                                  text_indent=rng.choice(["0", "0", "1.5em"]),
                                  text_align=rng.choice(["var(--align)", "var(--align)", "justify", "start"]),
                                  font_size=rng.choice(["1em", "0.95em", "1.05em"]))),
                    slots, "paragraph", PARAGRAPH_WORDS[rng.choice("SML")])


def make_list(rng, slots, depth, max_depth):
    ordered = rng.random() < 0.5
    kind = rng.choice(["decimal", "arabic-indic", "arabic-letters"]) if ordered else "disc"
    lst = (E.OL if ordered else E.UL)(style=css(list_style_type=kind, margin_block="0.3em",
                                                padding_inline_start=f"{rng.choice([1.2, 1.8, 2.4])}em"))
    n = rng.randint(2, 5)
    nest_at = rng.randrange(n) if depth < max_depth else -1
    for i in range(n):
        li = add_slot(E.LI(), slots, "list_item")
        if i == nest_at:
            li.append(make_list(rng, slots, depth + 1, max_depth))
        lst.append(li)
    return lst


def list_block(rng, slots, f, ctx):
    depth = rng.choices([1, 2, 3], weights=[5, 3, 1])[0]
    f["lists"] += 1
    f["list_depth"] = max(f["list_depth"], depth)
    return make_list(rng, slots, 1, depth)


def table(rng, slots, f, ctx):
    f["tables"] += 1
    cols, rows = rng.randint(2, ctx["max_cols"]), rng.randint(2, 15)
    border = rng.choice(BORDERS)
    cell_border = {"rows": css(border_block_end="1px solid #666"), "none": ""}.get(border, css(border=border))
    cell_style = "; ".join(filter(None, [cell_border, css(padding=f"0.2em {rng.choice([0.3, 0.5, 0.8])}em",
                                                          text_align=rng.choice(["start", "center"]))]))
    tbl = E.TABLE(style=css(border_collapse="collapse", inline_size="100%", margin_block="0.6em",
                            font_size=rng.choice(["0.85em", "0.9em", "1em"]), break_inside="avoid"))
    if rng.random() < 0.8:
        head_style = cell_style + "; " + css(background=rng.choice(["#e8e8e8", "transparent"]), font_weight="bold")
        tbl.append(E.THEAD(E.TR(*[add_slot(E.TH(style=head_style), slots, "table_header") for _ in range(cols)])))
    span = None
    if rng.random() < 0.4:
        span = (rng.randrange(rows - 1), rng.randrange(cols - 1), rng.choice([(2, 1), (1, 2), (2, 2)]))
    covered, body = set(), E.TBODY()
    for r in range(rows):
        tr = E.TR()
        for c in range(cols):
            if (r, c) in covered:
                continue
            td = add_slot(E.TD(style=cell_style), slots, "cell")
            if span and (r, c) == span[:2]:
                rs, cs = span[2]
                if rs > 1:
                    td.set("rowspan", str(rs))
                if cs > 1:
                    td.set("colspan", str(cs))
                covered |= {(r + i, c + j) for i in range(rs) for j in range(cs)}
            tr.append(td)
        body.append(tr)
    tbl.append(body)
    return tbl


def form(rng, slots, f, ctx):
    f["forms"] += 1
    per_row = rng.choice([1, 1, 2])
    field = rng.choice([css(border_block_end=f"1px {rng.choice(['solid', 'dotted'])} #000"),
                        css(border="1px solid #555", padding_inline="0.4em")])
    box = E.DIV(style=css(margin_block="0.6em"))
    for _ in range(rng.randint(2, 6)):
        row = E.DIV(style=css(display="flex", gap="0.8em", align_items="baseline", margin_block="0.35em"))
        for _ in range(per_row):
            row.append(add_slot(E.SPAN(style=css(font_weight="bold", white_space="nowrap")), slots, "label"))
            row.append(add_slot(E.SPAN(style=field + "; " + css(flex="1", min_inline_size="5em")), slots, "field_value"))
        box.append(row)
    return box


def aside(rng, slots, f, ctx):
    f["asides"] += 1
    el = E.ASIDE(style=css(border=f"{rng.choice([1, 2])}px {rng.choice(['solid', 'dashed'])} #444",
                           background=rng.choice(["#f4f4f4", "transparent"]), padding="0.6em 0.9em",
                           margin_block="0.8em", border_radius=rng.choice(["0", "6px"]), break_inside="avoid"))
    if rng.random() < 0.6:
        el.append(add_slot(E.H4(), slots, "heading"))
    for _ in range(rng.randint(1, 2)):
        el.append(add_slot(E.P(style=css(margin_block="0.2em")), slots, "sidebar"))
    return el


def quote(rng, slots, f, ctx):
    return add_slot(E.BLOCKQUOTE(style=css(border_inline_start=f"{rng.choice([2, 4])}px solid #888",
                                           padding_inline_start="0.9em", margin_inline="1.5em 0.5em",
                                           margin_block="0.8em", font_size="1.05em")), slots, "quote")


def figure(rng, slots, f, ctx):
    return E.FIGURE(
        E.DIV({"class": "ph", "style": css(block_size=f"{rng.randint(25, 70)}mm", margin_inline="auto",
                                           inline_size=rng.choice(["100%", "75%", "50%"]))}),
        add_slot(E.FIGCAPTION(style=css(font_size="0.85em", text_align="center", margin_block_start="0.3em")),
                 slots, "caption"),
        style=css(margin_block="0.8em", margin_inline="0", break_inside="avoid"))


def footnotes(rng, slots):
    notes = E.DIV(style=css(border_block_start="1px solid #000", margin_block_start="1.2em",
                            padding_block_start="0.4em", font_size="0.8em"))
    for _ in range(rng.randint(1, 3)):
        notes.append(add_slot(E.P(style=css(margin_block="0.15em")), slots, "footnote"))
    return notes


def band(rng, slots, tag, role, page_number):
    """Header or footer band; justify-content spreads its parts from inline-start to inline-end."""
    edge = "border_block_end" if role == "header" else "border_block_start"
    el = tag(style=css(display="flex", justify_content="space-between", gap="1em", font_size="0.85em",
                       padding_block="0.3em", margin_block="0.8em", **{edge: "1px solid #444"}))
    for _ in range(rng.choice([1, 1, 2])):
        el.append(add_slot(E.SPAN(), slots, role))
    if page_number:
        el.insert(rng.choice([0, len(el)]), add_slot(E.SPAN(), slots, "page_number"))
    return el


def picture(rng, style):
    return E.DIV({"class": "ph", "data-hint": rng.choice(PICTURE_HINTS), "style": style})


def cards(rng, slots, f, ctx):
    """A grid of 2-3 cards: optional picture, heading, short text."""
    n = rng.choice([2, 3]) if ctx["max_cols"] >= 3 else 2
    with_pic = rng.random() < 0.6
    grid = E.DIV(style=css(display="grid", grid_template_columns=f"repeat({n}, 1fr)", gap=f"{rng.choice([4, 6, 8])}mm",
                           margin_block="0.8em", break_inside="avoid"))
    f["columns"] = max(f["columns"], n)
    for _ in range(n):
        card = E.DIV(style=css(border=rng.choice(["1px solid #999", "none"]), padding=rng.choice(["0", "0.6em"]),
                               border_radius=rng.choice(["0", "6px"])))
        if with_pic:
            card.append(picture(rng, css(block_size=f"{rng.randint(20, 35)}mm", margin_block_end="0.4em")))
        card.append(add_slot(E.H4(style=css(margin_block="0.2em")), slots, "heading"))
        card.append(add_slot(E.P(style=css(margin_block="0.2em", font_size="0.92em")), slots, "paragraph", [8, 25]))
        grid.append(card)
    return grid


def keyvalue(rng, slots, f, ctx):
    """Key/value pairs: a definition list, or a two-column grid of label + value."""
    f["forms"] += 1
    if rng.random() < 0.5:
        dl = E.DL(style=css(margin_block="0.6em"))
        for _ in range(rng.randint(3, 6)):
            dl.append(add_slot(E.DT(style=css(font_weight="bold", margin_block_start="0.4em")), slots, "label"))
            dl.append(add_slot(E.DD(style=css(margin_inline_start="1.5em")), slots, "field_value"))
        return dl
    grid = E.DIV(style=css(display="grid", grid_template_columns=rng.choice(["1fr 2fr", "auto 1fr", "1fr 1fr"]),
                           gap="0.3em 1em", margin_block="0.6em", break_inside="avoid"))
    for _ in range(rng.randint(3, 7)):
        grid.append(add_slot(E.DIV(style=css(font_weight="bold")), slots, "label"))
        grid.append(add_slot(E.DIV(), slots, "field_value"))
    return grid


def stats(rng, slots, f, ctx):
    """A row of 2-4 stat boxes: a big number and a short label."""
    n = rng.randint(2, min(4, ctx["max_cols"] + 1))
    row = E.DIV(style=css(display="grid", grid_template_columns=f"repeat({n}, 1fr)", gap="4mm", margin_block="0.8em",
                          text_align="center", break_inside="avoid"))
    for _ in range(n):
        box = E.DIV(style=css(border=rng.choice(["1px solid #999", "2px solid #444", "none"]), padding="0.5em"))
        box.append(add_slot(E.DIV(style=css(font_size="1.8em", font_weight="bold")), slots, "cell", [1, 1]))
        box.append(add_slot(E.DIV(style=css(font_size="0.85em")), slots, "label", [1, 3]))
        row.append(box)
    return row


def timeline(rng, slots, f, ctx):
    """Dated entries: a date/label column and a text column."""
    f["lists"] += 1
    box = E.DIV(style=css(margin_block="0.8em", border_inline_start="2px solid #666", padding_inline_start="0.9em"))
    for _ in range(rng.randint(3, 5)):
        entry = E.DIV(style=css(margin_block="0.5em"))
        entry.append(add_slot(E.DIV(style=css(font_weight="bold", font_size="0.9em")), slots, "label", [1, 3]))
        entry.append(add_slot(E.DIV(), slots, "paragraph", [8, 30]))
        box.append(entry)
    return box


def media(rng, slots, f, ctx):
    """A picture beside text (picture first or second in reading order)."""
    row = E.DIV(style=css(display="flex", gap="5mm", align_items="flex-start", margin_block="0.8em",
                          break_inside="avoid"))
    pic = picture(rng, css(flex=f"0 0 {rng.choice([30, 35, 40])}%", block_size=f"{rng.randint(25, 45)}mm"))
    text = E.DIV(style=css(flex="1"))
    if rng.random() < 0.6:
        text.append(add_slot(E.H3(style=css(margin_block="0 0.3em")), slots, "heading"))
    text.append(add_slot(E.P(style=css(margin_block="0")), slots, "paragraph", [20, 60]))
    for el in ([pic, text] if rng.random() < 0.5 else [text, pic]):
        row.append(el)
    return row


def callout(rng, slots, f, ctx):
    """A highlighted note: a box with a side bar or a border, a short heading and text."""
    f["asides"] += 1
    style = rng.choice([css(border_inline_start="4px solid #333", background="#f3f3f3", padding="0.6em 0.9em"),
                        css(border="1px solid #333", padding="0.6em 0.9em", border_radius="4px"),
                        css(border_block="2px solid #333", padding="0.5em 0")])
    box = E.DIV(style=style + "; " + css(margin_block="0.8em", break_inside="avoid"))
    box.append(add_slot(E.STRONG(style=css(display="block", margin_block_end="0.2em")), slots, "heading"))
    box.append(add_slot(E.P(style=css(margin_block="0")), slots, "paragraph", [12, 40]))
    return box


def signature(rng, slots):
    """Signature lines at the end of a letter or form: label, a line, a name."""
    row = E.DIV(style=css(display="flex", justify_content="space-between", gap="10mm", margin_block_start="2em"))
    for _ in range(rng.choice([1, 2])):
        col = E.DIV(style=css(flex="1", max_inline_size="45%"))
        col.append(add_slot(E.DIV(style=css(font_weight="bold")), slots, "label"))
        col.append(E.DIV(style=css(border_block_end="1px solid #000", block_size="2.2em")))
        col.append(add_slot(E.DIV(style=css(font_size="0.9em")), slots, "field_value", [2, 4]))
        row.append(col)
    return row


def title_band(rng, slots):
    """A report title in a bordered or tinted band, with a subtitle line."""
    band = E.DIV(style=css(border_block=f"{rng.choice([2, 3])}px solid #222", padding="0.6em 0", margin_block_end="1em",
                           text_align=rng.choice(["start", "center"])))
    band.append(add_slot(E.H1(style=css(margin="0", font_size=f"calc(var(--base-size) * {rng.choice([1.8, 2.1, 2.4])})")),
                         slots, "title"))
    band.append(add_slot(E.DIV(style=css(font_size="1.05em", margin_block_start="0.3em")), slots, "subtitle"))
    return band


def letterhead(rng, slots):
    """A letter's top: sender block (with an optional logo picture), date and reference, addressee."""
    top = E.DIV(style=css(display="flex", justify_content="space-between", align_items="flex-start", gap="8mm",
                          border_block_end="1px solid #444", padding_block_end="0.6em", margin_block_end="1em"))
    sender = E.DIV()
    sender.append(add_slot(E.DIV(style=css(font_weight="bold", font_size="1.3em")), slots, "title"))
    for _ in range(rng.randint(1, 3)):
        sender.append(add_slot(E.DIV(style=css(font_size="0.85em")), slots, "header"))
    top.append(sender)
    if rng.random() < 0.6:
        top.append(picture(rng, css(inline_size="22mm", block_size="22mm", flex="0 0 22mm")))
    meta = E.DIV(style=css(margin_block_end="1em"))
    for _ in range(2):
        line = E.DIV()
        line.append(add_slot(E.SPAN(style=css(font_weight="bold")), slots, "label"))
        line.append(E.SPAN(" "))
        line.append(add_slot(E.SPAN(), slots, "field_value"))
        meta.append(line)
    return [top, meta, add_slot(E.P(style=css(font_weight="bold", margin_block="0.5em")), slots, "heading")]


BLOCKS = {"heading": heading, "paragraph": paragraph, "list": list_block, "table": table, "form": form,
          "aside": aside, "quote": quote, "figure": figure, "cards": cards, "keyvalue": keyvalue, "stats": stats,
          "timeline": timeline, "media": media, "callout": callout}


def blocks(rng, slots, f, ctx, n):
    weights = RICH_WEIGHTS if ctx.get("rich") else BLOCK_WEIGHTS
    names = rng.choices(list(weights), weights=list(weights.values()), k=n)
    return [BLOCKS[name](rng, slots, f, ctx) for name in names]


def sidebar(rng, slots, f):
    f["asides"] += 1
    el = E.ASIDE(style=css(background=rng.choice(["#f2f2f2", "transparent"]), padding="0.6em",
                           border_inline_start=rng.choice(["1px solid #999", "none"])))
    el.append(add_slot(E.H3(), slots, "heading"))
    for _ in range(rng.randint(2, 4)):
        el.append(add_slot(E.P(style=css(font_size="0.9em", margin_block="0 0.6em")), slots, "sidebar"))
    if rng.random() < 0.3:
        el.append(list_block(rng, slots, f, {}))
    return el


def generate(seed, tid, rich=False):
    """One template from one seed: same seed, same template. rich=True: more blocks and layouts (see top)."""
    rng = random.Random(seed)
    size = rng.choices(["A4", "A5", "Letter", "Legal"], weights=[5, 2, 2, 1])[0]
    orientation = "landscape" if rng.random() < 0.15 else "portrait"
    if rich:
        layout = rng.choices(["single", "multicol", "grid", "sidebar", "letter", "report"],
                             weights=[3, 2, 2, 2, 2, 2])[0]
    else:
        layout = rng.choices(["single", "multicol", "grid", "sidebar"], weights=[4, 2, 2, 2])[0]
    ncols = rng.choice([2, 3]) if layout in ("multicol", "grid") else 1
    slots = []
    f = {"columns": 2 if layout == "sidebar" else ncols, "tables": 0, "lists": 0, "list_depth": 0, "forms": 0,
         "asides": 0, "has_header": rng.random() < 0.5, "has_footer": rng.random() < 0.5, "slot_count": 0}
    ctx = {"max_cols": {1: 8, 2: 5, 3: 3}[ncols] if layout != "sidebar" else 6, "rich": rich}
    page_css = root_css(size, orientation, base_size=rng.choice(["10pt", "11pt", "12pt"]),
                        line_height=rng.choice([1.4, 1.6, 1.8]), align=rng.choice(["justify", "start"]),
                        margin=f"{rng.randint(12, 25)}mm")
    root, body = new_document(page_css + ARABIC_LETTERS)
    page = E.DIV({"class": "page"})
    body.append(page)
    numbered = rng.choice(["header", "footer"]) if rng.random() < 0.7 else None
    if f["has_header"]:
        page.append(band(rng, slots, E.HEADER, "header", numbered == "header"))
    if layout == "letter":
        page.extend(letterhead(rng, slots))
    elif layout == "report":
        page.append(title_band(rng, slots))
    elif rng.random() < 0.85:
        page.append(add_slot(E.H1(style=css(font_size=f"calc(var(--base-size) * {rng.choice([1.8, 2.2, 2.6])})",
                                            text_align=rng.choice(["center", "start"]), margin_block="0.3em 0.5em")),
                             slots, "title"))
        if rng.random() < 0.5:
            page.append(add_slot(E.P(style=css(font_size="1.2em", text_align="center", margin_block="0 1em")),
                                 slots, "subtitle"))
    n = rng.randint(3, 6) if size == "A5" else rng.randint(4, 11)
    if layout in ("single", "letter", "report"):
        main = E.MAIN(*blocks(rng, slots, f, ctx, n if layout != "letter" else max(3, n // 2)))
        if layout == "letter":
            main.append(signature(rng, slots))
    elif layout == "multicol":
        main = E.MAIN(*blocks(rng, slots, f, ctx, n + 2),
                      style=css(column_count=ncols, column_gap="8mm",
                                column_rule=rng.choice(["none", "1px solid #bbb"])))
    elif layout == "grid":
        cols = [E.DIV() for _ in range(ncols)]
        for i, b in enumerate(blocks(rng, slots, f, ctx, n + 2)):
            cols[i * ncols // (n + 2)].append(b)
        main = E.MAIN(*cols, style=css(display="grid", grid_template_columns=f"repeat({ncols}, 1fr)",
                                       column_gap="8mm"))
    else:
        start = rng.random() < 0.5  # sidebar at inline-start (first grid track under rtl) or inline-end
        side, content = sidebar(rng, slots, f), E.DIV(*blocks(rng, slots, f, ctx, n))
        main = E.MAIN(*([side, content] if start else [content, side]),
                      style=css(display="grid", column_gap="7mm",
                                grid_template_columns="1fr 2.4fr" if start else "2.4fr 1fr"))
    if rng.random() < 0.3:
        main.append(footnotes(rng, slots))
    page.append(main)
    if f["has_footer"]:
        page.append(band(rng, slots, E.FOOTER, "footer", numbered == "footer"))
    slots = renumber(root, slots)
    f["slot_count"] = len(slots)
    meta = {"id": tid, "source": "grammar", "source_ref": f"seed={seed}", "license": "own-generated",
            "attribution": "arabic-template-bank grammar generator (src/grammar.py)", "positioning": "flow",
            "page": {"size": size, "orientation": orientation}, "slots": slots, "features": f}
    text = to_html(root)
    meta["structure_hash"] = structure_hash(text, meta)
    return meta, text


def run(seed=0, n=300):
    clear_source("g_")
    for i in range(n):
        write_template(*generate(seed * 1_000_000 + i, f"g_{i:04d}"))
    print(f"grammar: wrote {n} templates")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    run(a.seed, a.n)
