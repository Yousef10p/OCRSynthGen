"""Surya 2.5's training-target spelling, written from reading Surya's compact.py (surya_repo_findings.md).

A page label is HTML, one <div data-bbox="x0 y0 x1 y1" data-label="Label"> per block; encode() turns it into
the text the model is trained to write:

  - a block div becomes <|Label|><|x0|><|y0|><|x1|><|y1|>; its </div> is left out when another block or the
    end follows, and whitespace right after the header is dropped;
  - </td>, </th>, </tr> are left out where the next tag implies them;
  - a data-bbox on any other tag becomes <|box|> and four coordinates just before that tag;
  - whitespace between two block-level tags, outside any element, or a newline next to a block-level tag, is
    dropped; text inside <pre>, <code>, <math> is kept as it is;
  - tags have one spelling: lower case, attributes in a fixed order, double quotes, `checked` without a value,
    <br/> <hr/> <img/> <input/>.
"""
import re

LABELS = {"Caption", "Footnote", "Equation-Block", "List-Group", "Page-Header", "Page-Footer", "Image",
          "Section-Header", "Table", "Text", "Complex-Block", "Code-Block", "Form", "Table-Of-Contents", "Figure",
          "Chemical-Block", "Diagram", "Bibliography", "Blank-Page", "Picture"}
ATTR_ORDER = ["type", "checked", "value", "colspan", "rowspan", "data-cs", "data-rs", "data-kind", "display",
              "class", "align", "style", "border", "href", "alt", "text"]
VOID = {"br", "hr", "img", "input"}
BLOCK_TAGS = {"div", "p", "table", "thead", "tbody", "tr", "td", "th", "ul", "ol", "li", "h1", "h2", "h3", "h4",
              "h5", "h6", "hr", "br", "caption", "comment"}
VERBATIM = {"pre", "code", "math"}
IMPLIED_BY = {"td": {"td", "th", "tr", "/tr", "/table", "/thead", "/tbody", "thead", "tbody"},
              "th": {"td", "th", "tr", "/tr", "/table", "/thead", "/tbody", "thead", "tbody"},
              "tr": {"tr", "/table", "/thead", "/tbody", "thead", "tbody"}}
TAG = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9-]*)((?:\s+[^<>]*?)?)\s*(/?)>")
ATTR = re.compile(r"""([a-zA-Z_:][-a-zA-Z0-9_:.]*)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?""")
BOX_VALUE = re.compile(r"(0|[1-9]\d{0,3})( (0|[1-9]\d{0,3})){3}")
SPACE = " \t\n\r\f"


def attributes(text):
    return [(m.group(1).lower(), next((g for g in m.group(2, 3, 4) if g is not None), None)) for m in ATTR.finditer(text)]


def tag(name, attrs, self_closing):
    parts = []
    for k, v in sorted(attrs, key=lambda kv: (ATTR_ORDER.index(kv[0]), "") if kv[0] in ATTR_ORDER else (len(ATTR_ORDER), kv[0])):
        parts.append(k if k == "checked" else f'{k}=""' if v is None else f'{k}="{v.replace(chr(34), "&quot;")}"')
    inner = " ".join([name] + parts)
    return f"<{inner}/>" if name in VOID or self_closing else f"<{inner}>"


def block_header(attrs):
    a = dict(attrs)
    if set(a) != {"data-bbox", "data-label"} or a["data-label"] not in LABELS:
        return None
    try:
        box = [int(v) for v in (a["data-bbox"] or "").split()]
    except ValueError:
        return None
    return (a["data-label"], box) if len(box) == 4 and all(0 <= v <= 1000 for v in box) else None


def encode(html):
    items = []  # ("text", s) or ("tag", raw, "/" or "", name, attrs, self_closing)
    pos = 0
    for m in TAG.finditer(html):
        if m.start() > pos:
            items.append(("text", html[pos:m.start()]))
        items.append(("tag", m.group(0), m.group(1), m.group(2).lower(), attributes(m.group(3)), bool(m.group(4))))
        pos = m.end()
    if pos < len(html):
        items.append(("text", html[pos:]))

    def next_tag(j):  # the next item that is not whitespace text
        while j < len(items) and items[j][0] == "text" and items[j][1].strip(SPACE) == "":
            j += 1
        return items[j] if j < len(items) else None

    def is_block_tag(j):
        return 0 <= j < len(items) and items[j][0] == "tag" and items[j][3] in BLOCK_TAGS

    out, open_tags, block_level, verbatim, after_header = [], [], None, 0, False
    for i, it in enumerate(items):
        if it[0] == "text":
            text = it[1]
            if after_header and not verbatim:
                text = text.lstrip(SPACE)
                if not text:
                    continue
            after_header = False
            if not verbatim and text.strip(SPACE) == "":
                before, after = is_block_tag(i - 1), is_block_tag(i + 1)
                if (before and after) or not open_tags or ("\n" in text and (before or after)):
                    continue
            out.append(text)
            continue
        _, raw, close, name, attrs, self_closing = it
        after_header = False
        if verbatim and name not in VERBATIM:
            out.append(raw)
            continue
        if not close:
            header = block_header(attrs) if name == "div" and not open_tags else None
            if header:
                out.append(f"<|{header[0]}|>" + "".join(f"<|{v}|>" for v in header[1]))
                open_tags.append("div")
                block_level, after_header = len(open_tags), True
                continue
            box = dict(attrs).get("data-bbox")
            if box is not None and BOX_VALUE.fullmatch(box) and all(int(v) <= 1000 for v in box.split()):
                out.append("<|box|>" + "".join(f"<|{v}|>" for v in box.split()))
                attrs = [(k, v) for k, v in attrs if k != "data-bbox"]
            if name in VERBATIM:
                verbatim += 1
            out.append(tag(name, attrs, self_closing))
            if name not in VOID and not self_closing:
                open_tags.append(name)
            continue
        if name in VERBATIM and verbatim:
            verbatim -= 1
        if name in open_tags:
            while open_tags and open_tags[-1] != name:
                open_tags.pop()
            open_tags.pop()
        nxt = next_tag(i + 1)
        if name == "div" and block_level is not None and len(open_tags) == block_level - 1:
            block_level = None
            if out and not out[-1].startswith("<"):
                out[-1] = out[-1].rstrip("\n")
            if nxt is None or (nxt[0] == "tag" and not nxt[2] and nxt[3] == "div" and block_header(nxt[4])):
                continue
        elif name in IMPLIED_BY:
            if nxt is None or (nxt[0] == "tag" and ((nxt[3] == "div" and (nxt[2] == "/" or block_header(nxt[4])))
                                                     or (nxt[3] != "div" and nxt[2] + nxt[3] in IMPLIED_BY[name]))):
                continue
        out.append(f"</{name}>")
    return "".join(out).strip("\n")
