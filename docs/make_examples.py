"""Make the before/after example images used in README.md (docs/examples/*.png).

Run after a build:  python -m docs.make_examples
Windows only: screenshots are taken with headless Microsoft Edge.
"""
import difflib
import io
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from jinja2 import Environment
from PIL import Image

from src.bank import RAW, ROOT, TEMPLATES, page_dims
from src.from_doclaynet import PAGES, candidates, fetch_pages, shard_row

OUT = ROOT / "docs" / "examples"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
ROWS_API = "https://datasets-server.huggingface.co/rows?dataset=HuggingFaceM4/WebSight&config=v0.1&split=train"
WEBSIGHT = ["w_00011", "w_00089"]
GRAMMAR = ["g_0007"]
WIDTH = 420  # width of every example image in the README
SAMPLE = "هذا نص عربي تجريبي يوضح شكل القالب بعد ملئه بالكلمات".split()


def shot(url, size, out):
    """Screenshot `url` at window `size`. Edge writes the file a few seconds after the command returns."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        png = Path(profile) / "shot.png"
        subprocess.run([EDGE, "--headless=new", "--no-first-run", "--disable-gpu", "--hide-scrollbars",
                        f"--user-data-dir={profile}", f"--window-size={size[0]},{size[1]}",
                        f"--screenshot={png}", url], capture_output=True)
        for _ in range(60):
            if png.exists() and png.stat().st_size:
                time.sleep(0.5)
                save(Image.open(png), out)
                return
            time.sleep(0.5)
    raise SystemExit(f"no screenshot for {url}")


def save(image, out):
    image = image.convert("RGB")
    image.resize((WIDTH, round(image.height * WIDTH / image.width)), Image.LANCZOS).save(out, optimize=True)
    print("wrote", out.relative_to(ROOT))


def page_px(meta):
    w, h = page_dims(meta["page"]["size"], meta["page"]["orientation"])
    return round(w * 96 / 25.4), round(h * 96 / 25.4)


def meta_of(tid):
    return json.loads((TEMPLATES / tid / "template.json").read_text(encoding="utf-8"))


def websight(tid):
    """Original = the dataset's own screenshot of that row; ours = the gallery preview at the same width."""
    row = int(tid[2:])
    got = json.load(urllib.request.urlopen(f"{ROWS_API}&offset={row}&length=1", timeout=120))["rows"][0]["row"]
    cached = json.loads(next(l for i, l in enumerate((RAW / "websight_v0.1.jsonl").open(encoding="utf-8")) if i == row))
    if difflib.SequenceMatcher(None, got["text"], cached["text"]).ratio() < 0.99:
        raise SystemExit(f"{tid}: API row differs from the cached row")
    original = Image.open(io.BytesIO(urllib.request.urlopen(got["image"]["src"], timeout=120).read()))
    save(original, OUT / f"{tid}_original.png")
    shot((ROOT / "previews" / f"{tid}.html").as_uri(), original.size, OUT / f"{tid}_template.png")


def doclaynet(tid):
    """Original = the DocLayNet page image (square-resized in the dataset, so stretched back); ours = preview."""
    meta = meta_of(tid)
    page = next(c for c in candidates() if c["ref"] == meta["source_ref"])
    fetch_pages([page])  # downloads the page image only if it is not cached yet
    size = page_px(meta)
    save(Image.open(PAGES / "{}_{}.png".format(*shard_row(page))).resize(size), OUT / f"{tid}_original.png")
    shot((ROOT / "previews" / f"{tid}.html").as_uri(), size, OUT / f"{tid}_template.png")


def grammar(tid):
    """No original exists; show the empty template and the same template filled with sample Arabic words."""
    meta = meta_of(tid)
    shot((ROOT / "previews" / f"{tid}.html").as_uri(), page_px(meta), OUT / f"{tid}_template.png")
    words = lambda n: " ".join(SAMPLE[i % len(SAMPLE)] for i in range(n))
    values = {s["id"]: words(sum(s["words"]) // 2) for s in meta["slots"]}
    text = (TEMPLATES / tid / "template.html.j2").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        filled = Path(d) / "filled.html"
        filled.write_text(Environment().from_string(text).render(style="", **values), encoding="utf-8")
        shot(filled.as_uri(), page_px(meta), OUT / f"{tid}_filled.png")


def pick_doclaynet():
    """One DocLayNet page with a kept picture and one with a table."""
    metas = [meta_of(p.name) for p in sorted(TEMPLATES.glob("d_*"))]
    with_picture = next(m["id"] for m in metas if m.get("images") and m["features"]["slot_count"] >= 8)
    with_table = next(m["id"] for m in metas if m["features"]["tables"] and m["id"] != with_picture)
    return [with_picture, with_table]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for tid in WEBSIGHT:
        websight(tid)
    for tid in pick_doclaynet():
        doclaynet(tid)
    for tid in GRAMMAR:
        grammar(tid)


if __name__ == "__main__":
    main()
