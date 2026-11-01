"""Source 1: random RTL document templates built from a small block grammar with lxml.

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


BLOCKS = {"heading": heading, "paragraph": paragraph, "list": list_block, "table": table, "form": form,
          "aside": aside, "quote": quote, "figure": figure}


def blocks(rng, slots, f, ctx, n):
    names = rng.choices(list(BLOCK_WEIGHTS), weights=list(BLOCK_WEIGHTS.values()), k=n)
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


def generate(seed, tid):
    """One template from one seed: same seed, same template."""
    rng = random.Random(seed)
    size = rng.choices(["A4", "A5", "Letter", "Legal"], weights=[5, 2, 2, 1])[0]
    orientation = "landscape" if rng.random() < 0.15 else "portrait"
    layout = rng.choices(["single", "multicol", "grid", "sidebar"], weights=[4, 2, 2, 2])[0]
    ncols = rng.choice([2, 3]) if layout in ("multicol", "grid") else 1
    slots = []
    f = {"columns": 2 if layout == "sidebar" else ncols, "tables": 0, "lists": 0, "list_depth": 0, "forms": 0,
         "asides": 0, "has_header": rng.random() < 0.5, "has_footer": rng.random() < 0.5, "slot_count": 0}
    ctx = {"max_cols": {1: 8, 2: 5, 3: 3}[ncols] if layout != "sidebar" else 6}
    page_css = root_css(size, orientation, base_size=rng.choice(["10pt", "11pt", "12pt"]),
                        line_height=rng.choice([1.4, 1.6, 1.8]), align=rng.choice(["justify", "start"]),
                        margin=f"{rng.randint(12, 25)}mm")
    root, body = new_document(page_css + ARABIC_LETTERS)
    page = E.DIV({"class": "page"})
    body.append(page)
    numbered = rng.choice(["header", "footer"]) if rng.random() < 0.7 else None
    if f["has_header"]:
        page.append(band(rng, slots, E.HEADER, "header", numbered == "header"))
    if rng.random() < 0.85:
        page.append(add_slot(E.H1(style=css(font_size=f"calc(var(--base-size) * {rng.choice([1.8, 2.2, 2.6])})",
                                            text_align=rng.choice(["center", "start"]), margin_block="0.3em 0.5em")),
                             slots, "title"))
        if rng.random() < 0.5:
            page.append(add_slot(E.P(style=css(font_size="1.2em", text_align="center", margin_block="0 1em")),
                                 slots, "subtitle"))
    n = rng.randint(3, 6) if size == "A5" else rng.randint(4, 11)
    if layout == "single":
        main = E.MAIN(*blocks(rng, slots, f, ctx, n))
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
