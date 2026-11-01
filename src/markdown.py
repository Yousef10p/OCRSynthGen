"""Convert a filled template page (HTML) to Markdown, in DOM order, which is the reading order.

Headings -> #, lists -> - / 1. (nested by indent), tables -> pipe tables, quotes -> >, pictures -> ![](...),
grey placeholder boxes -> ![figure](), everything else -> one paragraph per text block.

    python -m src.markdown renders/g_0000.html
"""
import sys

from lxml import html

BLOCK = {"address", "article", "aside", "blockquote", "div", "dl", "fieldset", "figcaption", "figure", "footer",
         "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre",
         "section", "table", "ul", "img"}
SKIP = {"head", "style", "script", "title"}


def text(el):
    """All text inside el; separate text nodes (a label span and its value span) get a space between them."""
    return " ".join(" ".join(el.itertext()).split())


def has_blocks(el):
    return any(isinstance(d.tag, str) and d.tag in BLOCK for d in el.iterdescendants())


def md_list(lst, depth=0):
    lines = []
    for n, li in enumerate((c for c in lst if c.tag == "li"), 1):
        own = " ".join(" ".join(p.split()) for p in [li.text or ""] + [
            text(c) + " " + (c.tail or "") for c in li if c.tag not in ("ul", "ol")] if p.strip())
        bullet = f"{n}." if lst.tag == "ol" else "-"
        lines.append(f"{'  ' * depth}{bullet} {own}".rstrip())
        lines += [line for sub in li if sub.tag in ("ul", "ol") for line in md_list(sub, depth + 1)]
    return lines


def md_table(table):
    rows = []
    for tr in table.iter("tr"):
        cells = []
        for cell in tr:
            if cell.tag in ("td", "th"):
                cells += [text(cell).replace("|", "\\|")] + [""] * (int(cell.get("colspan", 1)) - 1)
        rows.append(cells)
    if not rows:
        return ""
    width = max(map(len, rows))
    rows = [r + [""] * (width - len(r)) for r in rows]
    line = lambda r: "| " + " | ".join(r) + " |"
    return "\n".join([line(rows[0]), line(["---"] * width)] + [line(r) for r in rows[1:]])


def blocks(el):
    """Markdown blocks for the children of el, in order."""
    if (el.text or "").strip():
        yield " ".join(el.text.split())
    for child in el:
        if not isinstance(child.tag, str) or child.tag in SKIP:
            continue
        tag, cls = child.tag, child.get("class", "").split()
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            yield "#" * int(tag[1]) + " " + text(child)
        elif tag in ("ul", "ol"):
            yield "\n".join(md_list(child))
        elif tag == "table":
            yield md_table(child)
        elif tag == "img":
            yield f"![]({child.get('src', '')})"
        elif "ph" in cls:  # a grey picture box, or one the pipeline filled with a photo (data-img)
            yield f"![figure]({child.get('data-img', '')})"
        elif tag == "hr":
            yield "---"
        elif tag == "blockquote" and not has_blocks(child):
            yield "> " + text(child)
        elif has_blocks(child):
            yield from blocks(child)
        elif text(child):
            yield text(child)
        if (child.tail or "").strip():
            yield " ".join(child.tail.split())


def to_markdown(page_html):
    body = html.document_fromstring(page_html).find("body")
    return "\n\n".join(b for b in blocks(body) if b.strip()) + "\n"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(to_markdown(open(sys.argv[1], encoding="utf-8").read()))
