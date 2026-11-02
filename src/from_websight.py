"""Source 2: WebSight v0.1 pages stripped to RTL templates (text -> slots, media -> grey boxes, CSS flipped).

Run on its own:  python -m src.from_websight --seed 0 --n 150
"""
import argparse
import itertools
import json
import math
import random
import re
import shutil
import subprocess

from bs4 import BeautifulSoup, Comment, NavigableString

from src.bank import RAW, ROLES, ROOT, clear_source, root_css, write_template
from src.dedup import structure_hash

CACHE = RAW / "websight_v0.1.jsonl"
MAX_ROWS = 3000
DROP = ["script", "noscript", "link", "meta", "title", "base", "template", "object", "embed", "audio", "option"]
MEDIA = ["img", "svg", "video", "canvas", "iframe", "picture"]
PH_HINT = re.compile(r"image|img|photo|picture|video|placeholder|thumbnail|banner|logo|avatar|map", re.I)
KEEP_ATTRS = ("class", "id", "style", "colspan", "rowspan")
TAG_ROLE = {"h1": "title", "h2": "heading", "h3": "heading", "h4": "heading", "h5": "heading", "h6": "heading",
            "p": "paragraph", "li": "list_item", "th": "table_header", "td": "cell", "figcaption": "caption",
            "blockquote": "quote"}
FIELD_CSS = "display: inline-block; min-inline-size: 10em; border: 1px solid #999; padding: 0.2em 0.4em"
EXTRA_CSS = ".ph { background: #d0d0d0 !important; background-image: none !important; min-block-size: 2em; }\n"


def rows(limit):
    """First `limit` rows of the WebSight v0.1 train stream (HTML only), cached under data/raw/."""
    if not CACHE.exists() or sum(1 for _ in CACHE.open(encoding="utf-8")) < limit:
        from datasets import load_dataset
        # columns=["text"] streams only the HTML column of the parquet shards, never the screenshots.
        ds = load_dataset("HuggingFaceM4/WebSight", "v0.1", split="train", streaming=True, columns=["text"])
        first = next(iter(ds))
        print("websight: row keys", list(first), "features", ds.features)
        if not isinstance(first.get("text"), str):
            raise SystemExit("websight: no string 'text' field in v0.1 rows; stopping (schema differs from spec)")
        RAW.mkdir(parents=True, exist_ok=True)
        with CACHE.open("w", encoding="utf-8") as f:
            for i, row in enumerate(itertools.islice(ds, limit)):
                f.write(json.dumps({"row": i, "text": row["text"]}) + "\n")
    with CACHE.open(encoding="utf-8") as f:
        yield from (json.loads(line) for line in itertools.islice(f, limit))


def clean_css(css):
    """Drop comments, imports, web fonts, url() references and generated text."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    css = re.sub(r"@import[^;]*;", "", css)
    css = re.sub(r"@font-face\s*\{[^}]*\}", "", css)
    css = re.sub(r"url\([^)]*\)", "none", css)
    return re.sub(r"(?<![-\w])content\s*:\s*(['\"]).*?\1", 'content: ""', css)


def css_columns(css):
    """Largest column count declared by grid-template-columns or column-count (1 if none)."""
    n = 1
    for v in re.findall(r"grid-template-columns\s*:\s*([^;}]+)", css):
        r = re.search(r"repeat\(\s*(\d+)", v)
        tracks = re.findall(r"minmax\([^)]*\)|[\d.]+(?:fr|px|%|em|rem)|\bauto\b", v)
        n = max(n, int(r.group(1)) if r else 2 if "repeat(" in v else len(tracks))
    return max([n] + [int(c) for c in re.findall(r"column-count\s*:\s*(\d+)", css)])


def picture_hint(el):
    """Words that say what a picture shows: its link's keywords (WebSight v0.2 links such as
    https://source.unsplash.com/random/300x200/?food), alt text, class names and id."""
    src = str(el.get("src", ""))
    words = re.findall(r"[a-zA-Z]{3,}", src.split("?", 1)[1]) if "?" in src else []
    words += re.findall(r"[a-zA-Z]{3,}", " ".join([str(el.get("alt", "")), " ".join(el.get("class", [])),
                                                    str(el.get("id", ""))]))
    return " ".join(w.lower() for w in words if w.lower() not in ("random", "http", "https", "source", "unsplash",
                                                                   "img", "image", "rounded", "full", "object",
                                                                   "cover", "auto", "mx", "my"))[:80]


def to_placeholder(soup, el):
    ph = soup.new_tag("div", attrs={"class": " ".join(el.get("class", []) + ["ph"])})
    if hint := picture_hint(el):
        ph["data-hint"] = hint
    size = [f"inline-size: {el['width']}px" if str(el.get("width", "")).isdigit() else "",
            f"block-size: {el['height']}px" if str(el.get("height", "")).isdigit() else ""]
    if el.name != "div" and not any(size):
        size = ["inline-size: 100%", "block-size: 150px"]
    if any(size):
        ph["style"] = "; ".join(filter(None, size))
    el.replace_with(ph)


def convert_field(soup, el):
    """Form inputs become spans: text-like fields -> field_value slot, buttons -> label slot."""
    kind = el.get("type", "text").lower() if el.name == "input" else el.name
    if kind == "hidden":
        return el.decompose()
    span = soup.new_tag("span", attrs={"class": " ".join(el.get("class", []))})
    if kind in ("checkbox", "radio"):
        span["style"] = "display: inline-block; inline-size: 0.9em; block-size: 0.9em; border: 1px solid #666"
    elif kind in ("submit", "button", "reset", "image"):
        span["data-slot-role"], span["data-n"] = "label", len((el.get("value") or "").split())
    else:
        span["data-slot-role"], span["style"] = "field_value", FIELD_CSS
        if kind == "textarea":
            span["style"] += "; display: block; min-block-size: 4em"
    el.replace_with(span)


def role_of(node, n):
    """Role from the nearest semantic ancestor; p/fallback text inside header/footer takes that role."""
    names = [p.name for p in node.parents]
    section = next((x for x in names if x in ("header", "footer")), None)
    for name in names:
        if name in ("button", "label") or (name == "a" and "nav" in names):
            return "label"
        if name in TAG_ROLE and not (name == "p" and section):
            return TAG_ROLE[name]
    return section or ("paragraph" if n >= 8 else "label")


def word_range(n):
    return [max(1, math.floor(0.7 * n)), math.ceil(1.3 * n) + 1]


def add_slots(body):
    """Replace every non-empty text node (and converted field) with {{ slot_N }} in DOM order."""
    slots = []
    for node in list(body.descendants):
        if isinstance(node, NavigableString) and not isinstance(node, Comment) and node.strip():
            n = len(node.split())
            role, words = role_of(node, n), word_range(n)
        elif getattr(node, "attrs", None) and "data-slot-role" in node.attrs:
            role, n = node.attrs.pop("data-slot-role"), int(node.attrs.pop("data-n", 0))
            words = word_range(n) if n else ROLES[role]
        else:
            continue
        ph = "{{ slot_%d }}" % len(slots)
        slots.append({"id": f"slot_{len(slots)}", "role": role, "words": words, "dir": "rtl"})
        if isinstance(node, NavigableString):
            lead, trail = node[: len(node) - len(node.lstrip())], node[len(node.rstrip()):]
            node.replace_with(lead + ph + trail)
        else:
            node.string = ph
    return slots


def strip_page(text):
    """Parse and strip one page: (soup, sheet css, slots, features) before CSS flipping, or None."""
    soup = BeautifulSoup(text, "lxml")
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()
    sheet = clean_css("\n".join(s.get_text() for s in soup.find_all("style")))
    for t in soup.find_all(DROP + ["style", "head"]):
        t.decompose()
    body = soup.body
    if body is None:  # a few rows are a bare <style> with no page
        return None
    for el in body.find_all(MEDIA):
        to_placeholder(soup, el)
    for el in body.find_all(True):
        hint = " ".join(el.get("class", [])) + " " + str(el.get("id", ""))
        if el.parent and not el.get_text(strip=True) and PH_HINT.search(hint) and "ph" not in el.get("class", []):
            el.clear()
            to_placeholder(soup, el)
    had_fields = bool(body.find_all(["input", "select", "textarea"]))
    for el in body.find_all(["input", "select", "textarea"]):
        convert_field(soup, el)
    for el in body.find_all(True):
        el.attrs = {k: el.attrs[k] for k in KEEP_ATTRS + ("data-slot-role", "data-n", "data-hint") if k in el.attrs}
        if "style" in el.attrs:
            el["style"] = clean_css(el["style"])
    slots = add_slots(body)
    lists = body.find_all(["ul", "ol"])
    inline = " ".join(el["style"] for el in body.find_all(style=True))
    f = {"columns": css_columns(sheet + inline), "tables": len(body.find_all("table")), "lists": len(lists),
         "list_depth": max((1 + len(l.find_parents(["ul", "ol"])) for l in lists), default=0),
         "forms": len(body.find_all("form")) or int(had_fields), "asides": len(body.find_all("aside")),
         "has_header": body.find("header") is not None, "has_footer": body.find("footer") is not None,
         "slot_count": len(slots)}
    return soup, sheet, slots, f


def flip(css_list):
    """Flip many CSS strings to RTL in one rtlcss (node) call."""
    node = shutil.which("node") or r"C:\Program Files\nodejs\node.exe"
    out = subprocess.run([node, str(ROOT / "src" / "rtlcss_flip.js")], input=json.dumps(css_list),
                         capture_output=True, text=True, encoding="utf-8", check=True, cwd=ROOT)
    return json.loads(out.stdout)


def fonts(css, heading=False):
    """font-family -> var(--font-heading) in heading rules, var(--font-body) elsewhere."""
    def rule(m):
        var = "var(--font-heading)" if heading or re.search(r"\bh[1-6]\b", m.group(1)) else "var(--font-body)"
        return m.group(1) + "{" + re.sub(r"font-family\s*:[^;}]*", f"font-family: {var}", m.group(2)) + "}"
    return re.sub(r"([^{}]*)\{([^{}]*)\}", rule, css)


def finish(soup, sheet, flipped, size):
    """Put flipped CSS back, rebuild <head>, set lang/dir; returns the template text."""
    sheet, inline = flipped[0], flipped[1:]
    for el, css in zip(soup.body.find_all(style=True), inline):
        heading = el.name in TAG_ROLE and TAG_ROLE[el.name] in ("title", "heading")
        el["style"] = re.fullmatch(r"\s*x\s*\{(.*)\}\s*", fonts(css, heading), re.S).group(1).strip()
    own = root_css(size, "portrait") + re.sub(r"\{([{%#])", r"{ \1", fonts(sheet)) + "\n" + EXTRA_CSS
    head = soup.new_tag("head")
    head.append(soup.new_tag("meta", attrs={"charset": "utf-8"}))
    for css in (own, "{{ style }}"):
        style = soup.new_tag("style")
        style.string = css
        head.append(style)
    soup.html.attrs = {}
    soup.html.insert(0, head)
    # BeautifulSoup sorts attributes on output, so the <html> tag is written by hand to keep lang before dir.
    return "<!DOCTYPE html>\n" + soup.html.decode(formatter="html5").replace("<html>", '<html lang="ar" dir="rtl">', 1)


def run(seed=0, n=150):
    clear_source("w_")
    if n <= 0:
        return
    kept, scanned = [], 0
    for row in rows(MAX_ROWS):
        scanned += 1
        page = strip_page(row["text"])
        if page is None:
            continue
        soup, sheet, slots, f = page
        if len(slots) >= 15 and (f["tables"] or f["lists"] or f["columns"] > 1):
            kept.append((row["row"], soup, sheet, slots, f))
            if len(kept) == n:
                break
    batches = [[sheet] + [f"x {{ {el['style']} }}" for el in soup.body.find_all(style=True)]
               for _, soup, sheet, _, _ in kept]
    flat, written = flip([c for b in batches for c in b]), 0
    rng = random.Random(seed)
    for (row, soup, sheet, slots, f), batch in zip(kept, batches):
        flipped, flat = flat[: len(batch)], flat[len(batch):]
        size = rng.choice(["A4", "A4", "Letter"])
        if None in flipped:
            print(f"websight: rtlcss could not parse row {row}, skipped")
            continue
        text = finish(soup, sheet, flipped, size)
        meta = {"id": f"w_{row:05d}", "source": "websight", "source_ref": str(row), "license": "CC-BY-4.0",
                "attribution": "WebSight (HuggingFaceM4)", "positioning": "flow",
                "page": {"size": size, "orientation": "portrait"}, "slots": slots, "features": f}
        meta["structure_hash"] = structure_hash(text, meta)
        write_template(meta, text)
        written += 1
    print(f"websight: scanned {scanned} rows, kept {len(kept)}, wrote {written} templates")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    run(a.seed, a.n)
