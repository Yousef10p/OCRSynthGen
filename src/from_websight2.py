"""Source 5: WebSight v0.2 pages (Tailwind CSS) -> template candidates for the 2,000-template bank.

WebSight v0.2 pages are styled only by Tailwind CSS 2.2.19 classes, loaded from a CDN link. Templates must not
load anything from the web, so the stylesheet is kept in data/raw/ and each page gets only the Tailwind rules
its own classes use (plus Tailwind's base rules): a few KB instead of 2.9 MB.

Per page:
  1. classes that tie the page to a browser window are removed: h-screen / min-h-screen (100vh), fixed and
     sticky (a fixed bar would sit on top of the text), animate-*, and state variants (hover:, focus:...);
  2. the purged Tailwind CSS goes into a <style>, then the page is stripped exactly as a v0.1 page
     (src/from_websight.py): text -> slots, pictures -> picture boxes with a hint ("?food" -> food);
  3. it is kept only if document-like: 12+ slots and a table, list, form or columns;
  4. half the pages (by a hash of their position) become Arabic templates (CSS flipped to right-to-left by
     rtlcss), half English (CSS as it is).

Writes data/raw/candidates_websight2.jsonl (one {"meta", "text"} per line).

    .venv\\Scripts\\python -m src.from_websight2 --shards 60 --rows 500 --workers 4
"""
import argparse
import json
import random
import re
import sys
import time
from multiprocessing import Pool

from src.bank import RAW
from src.dedup import structure_hash
from src.from_websight import finish, flip, strip_page

TAILWIND = RAW / "tailwind-2.2.19.min.css"
TAILWIND_URL = "https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"
OUT = RAW / "candidates_websight2.jsonl"
DROP_CLASS = re.compile(r"^(?:h-screen|min-h-screen|max-h-screen|fixed|sticky|animate-.*|transition.*|duration-.*|"
                        r"ease-.*|transform|cursor-.*|.*(?:hover|focus|active|group-hover|focus-within|disabled|"
                        r"visited|placeholder):.*)$")
STATE = re.compile(r":(?:hover|focus|active|visited|disabled|focus-within|focus-visible|checked)\b")
CLASS = re.compile(r"\.((?:\\[0-9a-fA-F]{1,6} ?|\\.|[\w-])+)")
_purger = None


def unescape(name):
    name = re.sub(r"\\([0-9a-fA-F]{1,6}) ?", lambda m: chr(int(m.group(1), 16)), name)
    return re.sub(r"\\(.)", r"\1", name)


def parse_css(text):
    """[(media or None, selector, body)] of a minified stylesheet (comments and @keyframes left out)."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    rules, i, n = [], 0, len(text)

    def block(start):  # index of the brace closing the block that opens at start
        depth = 0
        for j in range(start, n):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            if depth == 0:
                return j
        return n

    while i < n:
        open_ = text.find("{", i)
        if open_ < 0:
            break
        head = text[i:open_].strip()
        close = block(open_)
        if head.startswith("@media"):
            for _, sel, body in parse_css(text[open_ + 1:close]):
                rules.append((head, sel, body))
        elif not head.startswith("@"):
            rules.append((None, head, text[open_ + 1:close]))
        i = close + 1
    return rules


class Purger:
    """Tailwind rules by class, to give a page only the rules its classes use."""

    def __init__(self):
        if not TAILWIND.exists():
            import urllib.request
            TAILWIND.write_bytes(urllib.request.urlopen(TAILWIND_URL, timeout=120).read())
        self.rules = parse_css(TAILWIND.read_text(encoding="utf-8"))

    def css(self, used):
        out, last_media, parts = [], None, []
        for media, selector, body in self.rules:
            keep = []
            for sel in selector.split(","):
                if STATE.search(sel):
                    continue
                classes = {unescape(c) for c in CLASS.findall(sel)}
                if classes <= used:  # base rules (no class) and rules whose classes the page all has
                    keep.append(sel)
            if not keep:
                continue
            rule = f"{','.join(keep)}{{{body}}}"
            if media:
                out.append(f"{media}{{{rule}}}")
            else:
                out.append(rule)
        return "\n".join(out)


def convert(html, ref, size, lang):
    """One v0.2 page -> (meta, text), or None if it isn't document-like."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    used = set()
    for el in soup.find_all(class_=True):
        el["class"] = [c for c in el.get("class", []) if not DROP_CLASS.match(c)]
        used |= set(el["class"])
    head = soup.head or soup.new_tag("head")
    if soup.head is None:
        soup.html.insert(0, head)
    style = soup.new_tag("style")
    style.string = _purger.css(used)
    head.append(style)
    page = strip_page(str(soup))
    if page is None:
        return None
    soup, sheet, slots, f = page
    if len(slots) < 12 or not (f["tables"] or f["lists"] or f["forms"] or f["columns"] > 1):
        return None
    inline = [f"x {{ {el['style']} }}" for el in soup.body.find_all(style=True)]
    css = flip([sheet] + inline) if lang == "ar" else [sheet] + inline
    if None in css:
        return None
    text = finish(soup, sheet, css, size)
    if lang == "en":
        text = text.replace('<html lang="ar" dir="rtl">', '<html lang="en" dir="ltr">', 1)
    for s in slots:
        s["dir"] = "rtl" if lang == "ar" else "ltr"
    meta = {"id": f"w2_{ref}", "source": "websight2", "source_ref": f"v0.2:{ref}", "license": "CC-BY-4.0",
            "attribution": "WebSight v0.2 (HuggingFaceM4)", "positioning": "flow", "lang": lang,
            "page": {"size": size, "orientation": "portrait"}, "slots": slots, "features": f}
    meta["structure_hash"] = structure_hash(text, meta)
    return meta, text


def init():
    global _purger
    _purger = Purger()


def shard_task(args):
    shard, n_shards, rows = args
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceM4/WebSight", "v0.2", split="train", streaming=True, columns=["text"])
    out, stats = [], {"shard": shard, "pages": 0, "kept": 0, "errors": 0}
    for i, row in enumerate(ds.shard(n_shards, shard)):
        if i >= rows:
            break
        stats["pages"] += 1
        ref = f"s{shard:03d}_r{i:05d}"
        rng = random.Random(ref)
        try:
            made = convert(row["text"], ref, rng.choice(["A4", "A4", "Letter"]), rng.choice(["ar", "en"]))
        except Exception:
            stats["errors"] += 1
            continue
        if made:
            out.append({"meta": made[0], "text": made[1]})
            stats["kept"] += 1
    return stats, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=60, help="how many of the 738 shards to read (spread evenly)")
    ap.add_argument("--rows", type=int, default=500, help="rows per shard")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    total = 738
    shards = sorted({round(k * total / a.shards) % total for k in range(a.shards)})
    start, kept = time.time(), 0
    with Pool(a.workers, initializer=init) as pool, OUT.open("w", encoding="utf-8") as f:
        for stats, out in pool.imap_unordered(shard_task, [(s, total, a.rows) for s in shards]):
            for c in out:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
            kept += len(out)
            print(f"[{time.time() - start:5.0f}s] shard {stats['shard']}: {stats['pages']} pages, {stats['kept']} kept,"
                  f" {stats['errors']} errors (total kept {kept})", flush=True)
    print(f"websight v0.2: {kept} candidates -> {OUT}")


if __name__ == "__main__":
    main()
