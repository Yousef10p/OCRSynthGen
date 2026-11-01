"""Source 3: DocLayNet layout boxes mirrored to RTL as absolutely positioned templates.

Only geometry is kept, never the original text. Pictures with no PDF text inside are kept as PNG crops;
pictures with text and formulas become grey boxes.

Run on its own:  python -m src.from_doclaynet --seed 0 --n 100
"""
import argparse
import io
import itertools
import json
import math
import random
import statistics

from lxml.html import builder as E
from PIL import Image

from src.bank import RAW, add_slot, clear_source, new_document, page_dims, renumber, root_css, to_html, write_template
from src.dedup import structure_hash

NAMES = ["docling-project/DocLayNet-v1.1", "ds4sd/DocLayNet-v1.1", "pierreguillou/DocLayNet-small"]
COLUMNS = ["bboxes", "category_id", "pdf_cells", "metadata"]
CACHE = RAW / "doclaynet_candidates.jsonl"
PAGES = RAW / "doclaynet_pages"
TRAIN_FILES = "datasets/docling-project/DocLayNet-v1.1/data/train-*.parquet"
ROWS_PER_SHARD, KEEP_EVERY = 300, 15  # the stream is ordered by category, so sample every shard
CATEGORIES = {1: "Caption", 2: "Footnote", 3: "Formula", 4: "List-item", 5: "Page-footer", 6: "Page-header",
              7: "Picture", 8: "Section-header", 9: "Table", 10: "Text", 11: "Title"}
ROLE = {"Title": "title", "Section-header": "heading", "Text": "paragraph", "List-item": "list_item",
        "Caption": "caption", "Footnote": "footnote", "Page-header": "header", "Page-footer": "footer"}
TAG = {"title": E.H1, "heading": E.H2, "paragraph": E.P, "caption": E.P, "footnote": E.P,
       "header": E.HEADER, "footer": E.FOOTER}
PT_PER_MM = 72 / 25.4
BOX_CSS = (".page { position: relative; block-size: var(--page-height); padding: 0; overflow: hidden; }\n"
           ".box { position: absolute; margin: 0; padding: 0; overflow: hidden; box-sizing: border-box; }\n"
           "h1.box, h2.box { font-weight: bold; }\n"
           "ul.box { list-style-position: inside; }\n"
           ".box table { inline-size: 100%; block-size: 100%; border-collapse: collapse; table-layout: fixed; }\n"
           ".box th, .box td { border: 1px solid #888; padding: 0 0.2em; overflow: hidden; }\n")


def open_stream():
    """First dataset name that streams and has the fields the spec lists; stops if none does."""
    from datasets import load_dataset
    for name in NAMES:
        try:
            ds = load_dataset(name, split="train", streaming=True, columns=COLUMNS)
            row = next(iter(ds))
        except Exception as e:
            print(f"doclaynet: {name} failed: {e}")
            continue
        meta_keys = {"coco_width", "coco_height", "original_width", "original_height", "doc_category"}
        if set(COLUMNS) - set(row) or meta_keys - set(row["metadata"]):
            raise SystemExit(f"doclaynet: {name} schema differs from spec: {list(row)} / {list(row['metadata'])}")
        print(f"doclaynet: using {name}, row keys {list(row)}, metadata keys {list(row['metadata'])}")
        return name, ds
    raise SystemExit("doclaynet: no dataset name worked")


def candidates():
    """Every KEEP_EVERY-th page of the first ROWS_PER_SHARD rows of each train shard; cached without any text."""
    if not CACHE.exists():
        name, ds = open_stream()
        RAW.mkdir(parents=True, exist_ok=True)
        with CACHE.open("w", encoding="utf-8") as f:
            for shard in range(ds.num_shards):
                rows = itertools.islice(ds.shard(ds.num_shards, shard), 0, ROWS_PER_SHARD, KEEP_EVERY)
                for i, r in enumerate(rows):
                    cells = [[{"bbox": c["bbox"], "size": c["font"]["size"]} for c in box] for box in r["pdf_cells"]]
                    f.write(json.dumps({"ref": f"{name}:train:shard{shard}:row{i * KEEP_EVERY}",
                                        "bboxes": r["bboxes"], "category_id": r["category_id"], "cells": cells,
                                        "metadata": r["metadata"]}) + "\n")
                print(f"doclaynet: cached shard {shard + 1}/{ds.num_shards}")
    return [json.loads(line) for line in CACHE.open(encoding="utf-8")]


def balanced(cands, n, rng):
    """n pages spread evenly over the doc categories (round robin over shuffled per-category lists)."""
    by_cat = {}
    for c in cands:
        by_cat.setdefault(c["metadata"]["doc_category"], []).append(c)
    pools = [rng.sample(v, len(v)) for _, v in sorted(by_cat.items())]
    picked = [p.pop() for p in itertools.chain.from_iterable(itertools.repeat(pools, n)) if p]
    return picked[:n]


def shard_row(page):
    shard, row = page["ref"].split(":")[-2:]
    return int(shard[len("shard"):]), int(row[len("row"):])


def keeps_picture(cat_id, cells):
    """A picture is kept only if no PDF text cell lies inside it."""
    return CATEGORIES[cat_id] == "Picture" and not cells


def fetch_pages(pages):
    """Page images of the given pages, cached as data/raw/doclaynet_pages/{shard}_{row}.png.

    Read straight from the train parquet shards with a small buffered reader, because `datasets` streaming
    would pull a whole 330 MB row group per shard to reach a few images.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    todo = {}
    for p in pages:
        s, r = shard_row(p)
        if not (PAGES / f"{s}_{r}.png").exists():
            todo.setdefault(s, {})[r] = p["metadata"]["page_hash"]
    if not todo:
        return
    PAGES.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    files = sorted(fs.glob(TRAIN_FILES))
    for s, rows in sorted(todo.items()):
        pf = pq.ParquetFile(fs.open(files[s], block_size=4 * 2**20), buffer_size=4 * 2**20, pre_buffer=False)
        i = 0
        for batch in pf.iter_batches(batch_size=8, columns=["image", "metadata"]):
            for rec in batch.to_pylist():
                if i in rows:
                    if rec["metadata"]["page_hash"] != rows[i]:
                        raise SystemExit(f"doclaynet: shard {s} row {i} is not the cached page; stopping")
                    (PAGES / f"{s}_{i}.png").write_bytes(rec["image"]["bytes"])
                i += 1
            if i > max(rows):
                break
        print(f"doclaynet: fetched {len(rows)} page images from shard {s}")


def crop(image, bbox, m, w_mm, h_mm):
    """Picture region of the square-resized page image, resized back to its real aspect ratio at 96 dpi."""
    sx, sy = image.width / m["coco_width"], image.height / m["coco_height"]
    x, y, w, h = bbox
    box = (max(0, round(x * sx)), max(0, round(y * sy)), min(image.width, round((x + w) * sx)),
           min(image.height, round((y + h) * sy)))
    size = (max(1, round(w_mm * 96 / 25.4)), max(1, round(h_mm * 96 / 25.4)))
    buf = io.BytesIO()
    image.convert("RGB").crop(box).resize(size, Image.LANCZOS).save(buf, "PNG", optimize=True)
    return buf.getvalue()


def choose_page(w_pt, h_pt):
    """Page size whose aspect ratio is closest to the original page."""
    orientation = "landscape" if w_pt > h_pt else "portrait"
    sizes = ["A5", "A4"] if max(w_pt, h_pt) < 650 else ["A4", "Letter", "Legal"]
    size = min(sizes, key=lambda s: abs(math.log(page_dims(s, orientation)[0] / page_dims(s, orientation)[1]
                                                 / (w_pt / h_pt))))
    return size, orientation


def bands(values, tol):
    """Number of clusters in sorted 1-D values with gaps larger than tol."""
    values = sorted(values)
    return sum(1 for a, b in zip([None] + values, values) if a is None or b - a > tol)


def font_size(cells, h, scale):
    """Font size in target pt: median pdf_cells size if usable, else box height / estimated lines."""
    sizes = [c["size"] * scale for c in cells if c["size"] > 1]
    if sizes:
        return statistics.median(sizes)
    lines = bands([c["bbox"][1] + c["bbox"][3] / 2 for c in cells],
                  0.5 * statistics.median([c["bbox"][3] for c in cells])) if cells else None
    return h / (lines or max(1, round(h / 15)))


def word_range(w, h, fs):
    capacity = (w / (0.5 * fs)) * (h / (1.5 * fs)) / 6
    return [max(1, math.floor(0.6 * capacity)), max(1, math.floor(0.95 * capacity))]


def table(rng, slots, cells, w, h, fs):
    """Table filling its box; rows x cols from pdf_cells positions when there are enough, else random."""
    if len(cells) >= 4:
        med_h = statistics.median([c["bbox"][3] for c in cells])
        rows = bands([c["bbox"][1] + c["bbox"][3] / 2 for c in cells], 0.5 * med_h)
        cols = bands([c["bbox"][0] for c in cells], 2 * med_h)
        rows, cols = min(max(rows, 2), 30), min(max(cols, 2), 10)
    else:
        rows, cols = rng.randint(3, 8), rng.randint(2, 5)
    words = word_range(w / cols, h / rows, fs)
    tbl = E.TABLE()
    for r in range(rows):
        role, tag = ("table_header", E.TH) if r == 0 else ("cell", E.TD)
        tbl.append(E.TR(*[add_slot(tag(), slots, role, words) for _ in range(cols)]))
    return tbl


def columns(text_boxes):
    """Most Text boxes side by side in one horizontal band (1 = single column)."""
    best = 1
    for x, y, w, h in text_boxes:
        side = [b for b in text_boxes if b[1] < y + h and y < b[1] + b[3] and (b[0] >= x + w or b[0] + b[2] <= x)]
        best = max(best, 1 + bands([b[0] for b in side], 1) if side else 1)
    return min(best, 4)


def generate(page, tid, rng, image=None):
    m = page["metadata"]
    ow, oh = m["original_width"], m["original_height"]
    size, orientation = choose_page(ow, oh)
    pw, ph = page_dims(size, orientation)
    mm = min(pw / ow, ph / oh)  # target mm per original pt, aspect ratio kept
    ox, oy = (pw - ow * mm) / 2, (ph - oh * mm) / 2
    to_mm = lambda b: [ox + b[0] / m["coco_width"] * ow * mm, oy + b[1] / m["coco_height"] * oh * mm,
                       b[2] / m["coco_width"] * ow * mm, b[3] / m["coco_height"] * oh * mm]
    root, body = new_document(root_css(size, orientation, margin="0mm") + BOX_CSS)
    pg = E.DIV({"class": "page"})
    body.append(pg)
    slots, f = [], {"columns": 1, "tables": 0, "lists": 0, "list_depth": 0, "forms": 0, "asides": 0,
                    "has_header": False, "has_footer": False, "slot_count": 0}
    boxes = sorted(zip(page["bboxes"], page["category_id"], page["cells"]), key=lambda t: (round(t[0][1] / 10), t[0][0]))
    text_boxes, files = [], {}
    for bbox, cat_id, cells in boxes:
        cat = CATEGORIES[cat_id]
        x, y, w, h = to_mm(bbox)
        x, y = max(x, 0), max(y, 0)  # a few annotations start just outside the page
        if w < 1 or h < 1:
            continue
        w_pt, h_pt = w * PT_PER_MM, h * PT_PER_MM
        fs = min(max(font_size(cells, h_pt, mm * PT_PER_MM), 5), 40)
        style = (f"inset-inline-start: {x:.1f}mm; inset-block-start: {y:.1f}mm; inline-size: {w:.1f}mm;"
                 f" block-size: {h:.1f}mm; font-size: {fs:.1f}pt")
        if image is not None and keeps_picture(cat_id, cells):
            name = f"img_{len(files)}.png"
            files[name] = crop(image, bbox, m, w, h)
            pg.append(E.IMG({"class": "box", "src": name, "alt": "", "style": style}))
        elif cat in ("Picture", "Formula"):
            pg.append(E.DIV({"class": "box ph", "style": style}))
        elif cat == "Table":
            f["tables"] += 1
            pg.append(E.DIV(table(rng, slots, cells, w_pt, h_pt, fs), {"class": "box", "style": style}))
        elif cat == "List-item":
            f["lists"], f["list_depth"] = f["lists"] + 1, 1
            pg.append(E.UL(add_slot(E.LI(), slots, "list_item", word_range(w_pt, h_pt, fs)),
                           {"class": "box", "style": style}))
        else:
            role = ROLE[cat]
            f["has_header"] |= role == "header"
            f["has_footer"] |= role == "footer"
            if role == "paragraph":
                text_boxes.append((x, y, w, h))
            pg.append(add_slot(TAG[role]({"class": "box", "style": style}), slots, role, word_range(w_pt, h_pt, fs)))
    if not slots:
        return None
    slots = renumber(root, slots)
    f["columns"], f["slot_count"] = columns(text_boxes), len(slots)
    meta = {"id": tid, "source": "doclaynet", "source_ref": page["ref"], "license": "CDLA-Permissive-1.0",
            "attribution": f"DocLayNet v1.1 (IBM Research), {m['doc_category']}, "
                           + ("layout geometry and text-free picture crops" if files else "layout geometry only"),
            "positioning": "absolute", "page": {"size": size, "orientation": orientation}, "slots": slots,
            "features": f}
    if files:
        meta["images"] = sorted(files)
    text = to_html(root)
    meta["structure_hash"] = structure_hash(text, meta)
    return meta, text, files


def run(seed=0, n=100):
    clear_source("d_")
    if n <= 0:
        return
    rng = random.Random(seed)
    cands = [c for c in candidates() if any(cid not in (3, 7) for cid in c["category_id"])]
    picked = balanced(cands, n, rng)
    with_pictures = [p for p in picked if any(map(keeps_picture, p["category_id"], p["cells"]))]
    fetch_pages(with_pictures)
    written = 0
    for i, page in enumerate(picked):
        image = Image.open(PAGES / "{}_{}.png".format(*shard_row(page))) if page in with_pictures else None
        out = generate(page, f"d_{i:04d}", rng, image)
        if out:
            write_template(*out)
            written += 1
    print(f"doclaynet: {len(cands)} candidate pages, wrote {written} templates")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    run(a.seed, a.n)
