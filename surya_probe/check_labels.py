"""Does each label keep everything its page draws? Compares every sample's label with the page it was made from.

    python surya_probe/check_labels.py <folder> [--examples 3]

<folder> is a dataset folder (labels/<name>.json and html/<name>.html, as generate.py writes it) or one holding
each sample's <name>.json (the label, with "slots" and "surya") next to its <name>.html (the filled page). A slot is drawn when the label lists it with text. For
the drawn slots (the page's own HTML, so CSS-only styling is out of reach here):
  text        every drawn slot's text is in the label's blocks
  extra       a text block (Text, Section-Header, List-Group, Caption, Footnote, Page-Header/Footer) has no word
              that no drawn slot has (bullets and numbers aside): nothing written that the page does not show
  inline      a slot inside the page's <u>/<ins>, <sub>, <sup>, <del>/<s>, <i>/<em>, <b>/<strong> keeps that tag
              around its text (<b> not in a heading or header cell, whose own tag says it; <i>/<b> not in a
              Page-Header, which Surya writes bare)
  heading     a heading slot (<h1>..<h6>) in a Section-Header keeps its level (h6 -> h5: Surya's tags stop at h5)
  nesting     a list inside a drawn list item, with drawn items, is nested in the label
  break       two drawn slots of one element with a <br> between them are on two lines in the label: two Text
              blocks (each item its own block), or a <br/> in any other block
and the blocks' structure and place on the page:
  table       no Table row without text (a table is drawn with its rows filled, not with empty rows)
  list        an <ol> item starts with its number, a <ul> item with a bullet, never two marks
  looks       a List-Group looks like a list and a Text does not: no list without marks (list-style-type: none),
              its items one under the other (not side by side), its marks on the line's start side (the block
              reaches past its items on the right on Arabic pages, on the left otherwise); no Text with two lines
              opening with a bullet or number, nor with list items one under the other; no Text with <br/>
  edges       nothing of the page's content above a Page-Header or below a Page-Footer, a Footnote in the lower
              half with no content below it
  header      a header or page-number slot is never labelled Text; a body slot (paragraph, heading, title, list
              item, cell...) is never only in a Page-Header
  notes       every Footnote opens with its number (<p><sup>1</sup> ...) and that number is a mark in the text
              (word<sup>1</sup>), every mark has its Footnote, no Page-Footer above or level with a Footnote, no
              footnote text in a Page-Footer
  caption     a Caption next to a picture, figure, diagram, molecule, panel or table (within 10% of the page)
  role        a slot in a page heading (<h1>..<h6>) is in a Section-Header or Page-Header, not Text or a list
  section     every Section-Header heads a section: the next block is not a heading of the same or a higher
              level, and there is one
  place       a Bibliography ends its column, a Table-Of-Contents opens it (only headings above it)
  order       the page's reading order is not marked ambiguous (a grid readable by rows or by columns, blocks
              side by side that can't be cut apart)
  edges also wants Page-Header / Page-Footer in <p>, with nothing of the content beside them in their band
Prints, per check, the samples that fail and examples; exit code 1 when anything fails."""
import argparse
import html as htmllib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from bs4 import BeautifulSoup

TEXT_LABELS = {"Text", "Section-Header", "List-Group", "Caption", "Footnote", "Page-Header", "Page-Footer"}
INLINE = {"u": "u", "ins": "u", "sub": "sub", "sup": "sup", "del": "del", "s": "del", "strike": "del",
          "i": "i", "em": "i", "b": "b", "strong": "b"}
MARKS = set("•○■◦▪-–—*·")


def norm(t):
    return " ".join(htmllib.unescape(t).split())


def words(t):
    return [w for w in re.findall(r"\w+", norm(t)) if not w.isdigit()]


HEADS = ["h1", "h2", "h3", "h4", "h5", "h6"]
BLOCKS = ["p", "div", "ul", "ol", "li", "table", "td", "th", "figure", "blockquote", "pre"]


def heading_of(el):
    """The heading a slot is the text of, or None: a block element between them (a converter's paragraph nested in
    a heading) is moved out of the heading by the page script, so it is not the heading's text."""
    h = el.find_parent(HEADS)
    if h is None:
        return None
    b = el.find_parent(BLOCKS + HEADS)
    return h if b is h else None


def find_block(blocks, text):
    """The label block whose text holds this slot's text (the shortest one: a slot also in a long block is in it)."""
    hits = [b for b in blocks if text and text in b["plain"]]
    return min(hits, key=lambda b: len(b["plain"])) if hits else None


def tags_around(block_html, text):
    """The tags around the first run of this text in the block's HTML (each text node holding part of it)."""
    soup = BeautifulSoup(block_html, "html.parser")
    words = text.split()
    # the text node that starts this slot: the longest opening of the slot's text that a node holds (its first word
    # alone can also be the label before it: "Bioburden: <i>Bioburden is ...</i>")
    for n in range(min(len(words), 6), 0, -1):
        start = " ".join(words[:n])
        for node in soup.find_all(string=True):
            if norm(str(node)).startswith(start) or (n == len(words) and start in norm(str(node))):
                return {p.name for p in node.parents if p.name}
    for node in soup.find_all(string=True):
        if words and words[0] in norm(str(node)):
            return {p.name for p in node.parents if p.name}
    return set()


def check(label, page, problems, stem):
    slots = {s["id"]: norm(s["text"]) for s in label.get("slots", []) if norm(s.get("text", ""))}
    blocks = [{"label": b["label"], "html": b["html"], "plain": norm(re.sub(r"<[^>]+>", " ", b["html"]))}
              for b in label["surya"]["blocks"]]
    for b in blocks:  # the label writes a block's text as one run; its plain text has the same words
        b["plain"] = " ".join(b["plain"].split())
    soup = BeautifulSoup(page, "html5lib")
    els = {s["data-slot"]: s for s in soup.find_all(attrs={"data-slot": True})}
    # text
    for sid, text in slots.items():
        if not find_block(blocks, text):
            problems["text"].append(f"{stem}: {sid} {text[:60]!r} is in no block")
    # extra
    drawn_words = set(w for t in slots.values() for w in words(t))
    for b in blocks:
        if b["label"] in TEXT_LABELS:
            extra = [w for w in words(b["plain"]) if w not in drawn_words and not set(w) <= MARKS]
            if extra:
                problems["extra"].append(f"{stem}: {b['label']} has {extra[:5]} that no drawn slot has")
    for sid, text in slots.items():
        el = els.get(sid)
        b = find_block(blocks, text)
        if el is None or b is None:
            continue
        # the same text can be in two blocks (a page with two "Biography" headings): any of them may be this slot's
        cands = [x for x in blocks if text in x["plain"]]
        around = set().union(*(tags_around(x["html"], text) for x in cands))
        heading = heading_of(el)
        header_cell = el.find_parent("th")
        # inline
        for p in el.parents:
            if p.name in ("p", "li", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6", "div", "body", "figcaption",
                          "blockquote", "dd", "dt", "label", "section", "article", "header", "footer", "main"):
                break
            want = INLINE.get(p.name)
            if not want or want in around:
                continue
            if want == "b" and (heading or header_cell or b["label"] in ("Section-Header", "Page-Header")):
                continue
            if want == "i" and b["label"] == "Page-Header":
                continue
            problems["inline"].append(f"{stem}: {sid} {text[:40]!r} is in <{p.name}> on the page, not <{want}> in its {b['label']}")
        # heading
        if heading is not None and b["label"] == "Section-Header":
            want = heading.name if heading.name != "h6" else "h5"
            if not any(f"<{want}>" in x["html"] for x in cands if x["label"] == "Section-Header"):
                problems["heading"].append(f"{stem}: {sid} {text[:40]!r} is <{heading.name}> on the page: {b['html'][:60]}")
    # nesting
    for li in soup.find_all("li"):
        inner = li.find(["ul", "ol"])
        if inner is None:
            continue
        own = [s["data-slot"] for s in li.find_all(attrs={"data-slot": True}) if s.find_parent("li") is li]
        kids = [s["data-slot"] for s in inner.find_all(attrs={"data-slot": True})]
        if any(s in slots for s in own) and any(s in slots for s in kids):
            b = find_block(blocks, slots[next(s for s in kids if s in slots)])
            if b and not re.search(r"<li>(?:(?!</li>).)*?<(?:ul|ol)", b["html"], re.S):
                problems["nesting"].append(f"{stem}: a nested list under {slots[next(s for s in own if s in slots)][:40]!r} is flat")
    # break
    for br in soup.find_all("br"):
        el = br.find_parent(["p", "li", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6", "figcaption", "blockquote", "dd", "dt", "label"])
        if el is None:
            continue
        before = after = None
        for node in el.descendants:
            if node is br:
                before, after = before, "next"
                continue
            if getattr(node, "get", None) and node.get("data-slot") in slots:
                if after == "next":
                    after = node["data-slot"]
                    break
                before = node["data-slot"]
        if before and after not in (None, "next"):
            b = find_block(blocks, slots[before])
            if b and b["label"] == "Text":  # lines set apart by <br> are separate items: a Text block each
                if slots[after] in b["plain"] and slots[before] + " " + slots[after] in b["plain"]:
                    problems["break"].append(f"{stem}: {slots[before][-30:]!r} and {slots[after][:30]!r} (a <br> apart) in one Text")
            elif b and not re.search(re.escape(htmllib.escape(slots[before].split()[-1], quote=False)) + r"(</\w+>)*<br/>", b["html"]):
                problems["break"].append(f"{stem}: no <br/> between {slots[before][-30:]!r} and {slots[after][:30]!r}")


NUMBER = re.compile(r"^\s*(\[?[0-9٠-٩]+[.)\]]|\[?[0-9٠-٩]+\])")
BULLET = re.compile(r"^\s*[•○■◦▪]\s")
FIGURES = {"Image", "Figure", "Diagram", "Chemical-Block", "Complex-Block", "Table", "Form", "Code-Block", "Equation-Block"}


def structure(label, page, problems, stem):
    blocks, height = label["surya"]["blocks"], label.get("height") or 1
    for b in blocks:
        if b["label"] == "Table":
            for row in re.findall(r"<tr>(.*?)</tr>", b["html"], re.S):
                cells = re.findall(r"<t[dh]([^>]*)>(.*?)</t[dh]>", row, re.S)
                if cells and not any(t.strip() for _, t in cells):
                    problems["table"].append(f"{stem}: a Table row with no text: {row[:80]}")
                    break
        if b["label"] == "List-Group":
            for m in re.finditer(r"<(ul|ol)([^>]*)>(.*?)(?=<(?:ul|ol)[ >]|</(?:ul|ol)>)", b["html"], re.S):
                tag, attrs, body = m.groups()
                for item in re.findall(r"<li>([^<]*)", body):
                    if not item.strip():
                        continue
                    if tag == "ol" and not NUMBER.match(item):
                        problems["list"].append(f"{stem}: an <ol> item without its number: {item[:40]!r}")
                    if tag == "ul" and "none" not in attrs and not BULLET.match(item):
                        problems["list"].append(f"{stem}: a <ul> item without its bullet: {item[:40]!r}")
                    if re.match(r"^\s*[•○■◦▪]\s*[•○■◦▪–*-]\s", item):
                        problems["list"].append(f"{stem}: two marks: {item[:40]!r}")
    body = [b["box"] for b in blocks if b["label"] not in ("Page-Header", "Page-Footer", "Footnote")]
    tol = 0.005 * height
    for b in blocks:
        x0, y0, x1, y1 = b["box"]
        beside = any(min(o[3], y1) - max(o[1], y0) > tol for o in body)
        if b["label"] == "Page-Header" and (any(o[3] <= y0 + tol for o in body) or beside):
            problems["edges"].append(f"{stem}: content above or beside the Page-Header at y={y0 / height:.2f}")
        if b["label"] == "Page-Footer" and (any(o[1] >= y1 - tol for o in body) or beside):
            problems["edges"].append(f"{stem}: content below or beside the Page-Footer at y={y1 / height:.2f}")
        if b["label"] in ("Page-Header", "Page-Footer") and not re.fullmatch(r"<p>.*</p>", b["html"], re.S):
            problems["edges"].append(f"{stem}: a {b['label']} not in <p>: {b['html'][:50]!r}")
        if b["label"] == "Footnote" and (y1 < height / 2 or any(o[1] >= y1 - tol for o in body)):
            problems["edges"].append(f"{stem}: a Footnote at y={y0 / height:.2f} with content below it or in the upper half")
        if b["label"] == "Caption":
            near = [o for o in blocks if o["label"] in FIGURES and min(abs(o["box"][1] - y1), abs(y0 - o["box"][3])) < 0.1 * height]
            if not near:
                problems["caption"].append(f"{stem}: a Caption at y={y0 / height:.2f} with no picture or table next to it")
    # every Section-Header heads a section: the next block in reading order (page footers and footnotes aside)
    # is not a heading of the same or a higher level, and there is one
    content = [b for b in blocks if b["label"] not in ("Page-Footer", "Footnote")]
    lvl = lambda b: int(m.group(1)) if (m := re.match(r"<h([1-6])>", b["html"])) else 6  # noqa: E731
    for k, b in enumerate(content):
        if b["label"] != "Section-Header":
            continue
        nxt = content[k + 1] if k + 1 < len(content) else None
        if nxt is None or (nxt["label"] == "Section-Header" and lvl(nxt) <= lvl(b)):
            problems["section"].append(f"{stem}: {b['html'][:50]!r} heads no section"
                                       + (f" (next: {nxt['html'][:40]!r})" if nxt else " (last block)"))
    # a Bibliography ends its column (nothing of the content below it), a Table-Of-Contents opens it (only headings
    # above it)
    same_col = lambda a, b: min(a[2], b[2]) - max(a[0], b[0]) > 0.3 * min(a[2] - a[0], b[2] - b[0])  # noqa: E731
    for b in blocks:
        if b["label"] == "Bibliography" and any(same_col(o["box"], b["box"]) and o["box"][1] >= b["box"][3] - tol
                                                 for o in content if o is not b):
            problems["place"].append(f"{stem}: content below the Bibliography")
        if b["label"] == "Table-Of-Contents" and any(
                same_col(o["box"], b["box"]) and o["box"][3] <= b["box"][1] + tol
                and o["label"] not in ("Section-Header", "Page-Header", "Image", "Figure") for o in content if o is not b):
            problems["place"].append(f"{stem}: body text above the Table-Of-Contents")
    if label["surya"].get("order_ambiguous"):
        problems["order"].append(f"{stem}: reading order ambiguous ({', '.join(label['surya']['order_ambiguous'])})")
    soup = BeautifulSoup(page, "html5lib")
    slots = {s["id"]: norm(s["text"]) for s in label.get("slots", []) if norm(s.get("text", ""))}
    plain = [(b["label"], norm(re.sub(r"<[^>]+>", " ", b["html"]))) for b in blocks]
    for el in soup.find_all(attrs={"data-slot": True}):
        t = slots.get(el["data-slot"])
        if not t or heading_of(el) is None:
            continue
        hits = [lab for lab, txt in plain if t in txt]
        if hits and all(lab not in ("Section-Header", "Page-Header") for lab in hits):
            problems["role"].append(f"{stem}: heading {t[:40]!r} is labelled {hits[0]}")
    looks(label, blocks, plain, problems, stem)
    notes(label, blocks, problems, stem)


def notes(label, blocks, problems, stem):
    """A Footnote is the note to a mark in the text and stands apart from the page footer."""
    roles = {norm(s["text"]): s["role"] for s in label.get("slots", []) if norm(s.get("text", ""))}
    marks = [m for b in blocks if b["label"] not in ("Footnote", "Page-Footer", "Page-Header")
             for m in re.findall(r"\S<sup>([^<]+)</sup>", b["html"])]
    feet = [b for b in blocks if b["label"] == "Page-Footer"]
    for b in blocks:
        if b["label"] == "Footnote":
            m = re.match(r"<p>(?:<[bi]>)*<sup>([^<]+)</sup> ", b["html"])
            if not m:
                problems["notes"].append(f"{stem}: a Footnote without its number: {b['html'][:50]!r}")
            elif m.group(1) not in marks:
                problems["notes"].append(f"{stem}: Footnote {m.group(1)} has no mark in the text: {b['html'][:50]!r}")
            if any(f["box"][1] < b["box"][3] and min(f["box"][2], b["box"][2]) > max(f["box"][0], b["box"][0]) for f in feet):
                problems["notes"].append(f"{stem}: a Page-Footer above or level with a Footnote: {b['html'][:40]!r}")
        if b["label"] == "Page-Footer":
            txt = norm(re.sub(r"<[^>]+>", " ", b["html"]))
            if any(r in ("footnote", "footnote_mark") for t, r in roles.items() if t and t in txt and len(t) > 3):
                problems["notes"].append(f"{stem}: footnote text in a Page-Footer: {b['html'][:50]!r}")
    for m in marks:
        if not any(re.match(rf"<p>(?:<[bi]>)*<sup>{re.escape(m)}</sup> ", b["html"]) for b in blocks if b["label"] == "Footnote"):
            problems["notes"].append(f"{stem}: mark {m} in the text with no Footnote")


BODY_ROLES = {"paragraph", "list_item", "heading", "title", "subtitle", "cell", "label", "field_value", "sidebar",
              "quote", "caption", "footnote", "table_header"}


def looks(label, blocks, plain, problems, stem):
    """A List-Group is drawn as a list and a Text is not; page-header text is a Page-Header and nothing else is."""
    slots = [s for s in label.get("slots", []) if norm(s.get("text", ""))]
    labels_of = lambda t: [lab for lab, txt in plain if norm(t) in txt]  # noqa: E731
    rtl = label.get("lang") == "ar"
    for b in blocks:
        if b["label"] == "List-Group":
            # 1. marks: no list drawn without them (the bullet or number check above covers each item)
            if "list-style-type: none" in b["html"]:
                problems["looks"].append(f"{stem}: a List-Group drawn without marks: {b['html'][:60]!r}")
            inside = lambda s: (b["box"][0] - 2 <= (s["bbox"][0] + s["bbox"][2]) / 2 <= b["box"][2] + 2  # noqa: E731
                                and b["box"][1] - 2 <= (s["bbox"][1] + s["bbox"][3]) / 2 <= b["box"][3] + 2)
            items = [s for s in slots if s["role"] == "list_item" and inside(s)  # (the same words may stand elsewhere)
                     and norm(s["text"]) in norm(re.sub(r"<[^>]+>", " ", b["html"]))]
            # 2. separated: items one under the other, never two on one line (an item's own slots, e.g. a bold term
            # and its text, may share its line)
            spans, free = [], list(items)  # each item's slots, in order (two items may have the same words)
            for li in re.findall(r"<li>(.*?)(?=<li>|</li>|<ul|<ol)", b["html"], re.S):
                own = [s for s in free if norm(s["text"]) in norm(re.sub(r"<[^>]+>", " ", li))]
                own = [s for k, s in enumerate(own) if norm(s["text"]) not in {norm(o["text"]) for o in own[:k]}]
                free = [s for s in free if s not in own]
                if own:
                    spans.append([min(s["bbox"][0] for s in own), min(s["bbox"][1] for s in own),
                                  max(s["bbox"][2] for s in own), max(s["bbox"][3] for s in own)])
            # side by side: on one line (more than half their height shared) and apart across it
            if any(min(a[3], c[3]) - max(a[1], c[1]) > 0.5 * min(a[3] - a[1], c[3] - c[1])
                   and min(a[2], c[2]) <= max(a[0], c[0]) for k, a in enumerate(spans) for c in spans[k + 1:]):
                problems["looks"].append(f"{stem}: List-Group items side by side: {b['html'][:60]!r}")
            # 3. direction: the marks stand on the line's start side, so the block reaches past its items there
            if items:
                x0, x1 = min(s["box"][0] for s in items), max(s["box"][0] + s["box"][2] for s in items)
                if (rtl and b["box"][2] < x1 + 2) or (not rtl and b["box"][0] > x0 - 2):
                    problems["looks"].append(f"{stem}: List-Group marks not on the {'right' if rtl else 'left'}: {b['html'][:50]!r}")
        if b["label"] == "Text" and "<br" in b["html"]:  # one item per Text: lines a page breaks are blocks apart
            problems["looks"].append(f"{stem}: a Text with <br/>: {b['html'][:60]!r}")
        if b["label"] == "Text":
            # 4. a Text that looks like a list: lines opening with marks, or list items one under the other
            lines = re.split(r"<br/>", re.sub(r"^<p>|</p>$", "", b["html"]))
            if sum(bool(BULLET.match(re.sub(r"<[^>]+>", "", x)) or NUMBER.match(re.sub(r"<[^>]+>", "", x))) for x in lines) >= 2:
                problems["looks"].append(f"{stem}: a Text drawn as a list: {b['html'][:60]!r}")
            txt = norm(re.sub(r"<[^>]+>", " ", b["html"]))
            ys = sorted(s["box"][1] for s in slots if s["role"] == "list_item" and norm(s["text"]) in txt)
            if len(ys) >= 2 and len({round(y / 8) for y in ys}) == len(ys):
                problems["looks"].append(f"{stem}: list items one under the other labelled Text: {b['html'][:60]!r}")
    for s in slots:
        hits = labels_of(s["text"])
        # 5. page-header text is a Page-Header (a page number at the bottom a Page-Footer), never Text
        # (the same words may also stand in the body: "2" in a paragraph, a title repeated in the header)
        if s["role"] in ("header", "page_number") and "Text" in hits and not {"Page-Header", "Page-Footer"} & set(hits):
            problems["header"].append(f"{stem}: {s['role']} {s['text'][:40]!r} is labelled Text")
        # 6. no body text in a Page-Header
        if s["role"] in BODY_ROLES and hits and all(lab == "Page-Header" for lab in hits):
            problems["header"].append(f"{stem}: {s['role']} {s['text'][:40]!r} is labelled Page-Header")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--examples", type=int, default=3)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    problems, n = defaultdict(list), 0
    folder = Path(a.folder)
    dataset = (folder / "labels").is_dir()  # a dataset folder (labels/, html/), or one with both side by side
    for jf in sorted((folder / "labels" if dataset else folder).glob("*.json")):
        page = folder / "html" / f"{jf.stem}.html" if dataset else jf.with_suffix(".html")
        if not page.exists():
            continue
        label = json.loads(jf.read_text(encoding="utf-8"))
        if "surya" not in label:
            continue
        n += 1
        check(label, page.read_text(encoding="utf-8"), problems, jf.stem)
        structure(label, page.read_text(encoding="utf-8"), problems, jf.stem)
    print(f"{n} samples checked")
    for name in ("text", "extra", "inline", "heading", "nesting", "break", "table", "list", "looks", "edges", "header", "notes",
                 "caption", "role",
                 "section", "place", "order"):
        found = problems.get(name, [])
        stems = {x.split(":")[0] for x in found}
        print(f"  {name:8} {'ok' if not found else f'{len(found)} problems in {len(stems)} samples'}")
        for x in found[:a.examples]:
            print(f"      {x}")
    sys.exit(1 if any(problems.values()) else 0)


if __name__ == "__main__":
    main()
