"""POC demo of the next stage: turn empty templates into varied, scan-like Arabic page images.

For every chosen template and variant:
  1. fill each slot with Arabic sample text (our own sentences, numbers, dates; repeats are fine);
  2. pick a random style through the template's {{ style }} hook: font family and weights (13 Arabic
     families), body size, line spacing, alignment, ink colour, page margins, digit style;
  3. render the page with headless Edge on a transparent background;
  4. lay it on a random paper (tinted paper with grain, plain, gradient, or a photo texture);
  5. degrade it like a real capture (clean, flatbed scan, phone photo, fax, archive).

Fonts, backgrounds and degradations are OCRSmith's (MIT), so this runs with OCRSmith's venv:

    cd D:\\code\\OCR\\arabic-template-bank
    ..\\ocrsmith\\.venv\\Scripts\\python demo\\render_demo.py --per-source 8 --variants 3

Output: demo/out/*.jpg and demo/index.html. No OCR labels yet: word boxes come with the real renderer.

With --gallery it renders every template in the bank once (renders/{id}.jpg, about 30 minutes) and rebuilds
gallery/index.html, whose cards then show each empty template next to its rendered page.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from jinja2 import Environment
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bank import TEMPLATES, page_dims  # noqa: E402
from src.dedup import load_bank  # noqa: E402

from fill_style import OCRSMITH, fill, font_families, style  # noqa: E402

from ocrsmith.core.backgrounds import gradient_background, image_background, paper_background, solid_background  # noqa: E402,E501
from ocrsmith.core.degradations import build_preset  # noqa: E402
from ocrsmith.domain.annotations import Page  # noqa: E402

DEMO = ROOT / "demo"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
SCALE = 2  # device pixels per CSS pixel: about 190 dpi
PRESETS = {"clean": 1.0, "scan": 4.0, "photo": 3.0, "fax": 0.7, "archive": 1.5}
# -- 3-5. render, paper, capture -------------------------------------------------------------------

def background(rng, w, h):
    kind = rng.choices(["paper", "solid", "gradient", "photo texture"], weights=[5, 2, 1.5, 0.8])[0]

    def tint():
        # One factor for all channels, from cream to white: real paper never turns pink or green.
        f = rng.random()
        return tuple(int(a + f * (b - a)) for a, b in zip((226, 218, 198), (255, 255, 252)))

    if kind == "paper":
        bg = paper_background(tint(), rng.uniform(2.0, 9.0), rng)
    elif kind == "solid":
        bg = solid_background(tint())
    elif kind == "gradient":
        bg = gradient_background(tint(), tint(), rng.choice(["horizontal", "vertical", "diagonal"]))
    else:
        # OCRSmith's other photos (bricks, containers, city…) hide the text; a document sits on paper.
        path = OCRSMITH / "backgrounds" / "white_paper.jpg"
        kind = "photo texture (white paper)"
        bg = image_background(path)
    return bg(w, h).convert("RGBA"), kind


def screenshot(html_path, size, work, scale):
    """Edge writes the screenshot a moment after the command returns, so wait for the file."""
    png = work / (html_path.stem + ".png")
    profile = tempfile.mkdtemp(dir=work)
    subprocess.run([EDGE, "--headless=new", "--no-first-run", "--disable-gpu", "--hide-scrollbars",
                    f"--user-data-dir={profile}", f"--window-size={size[0]},{size[1]}",
                    f"--force-device-scale-factor={scale}", "--default-background-color=00000000",
                    "--virtual-time-budget=5000", f"--screenshot={png}", html_path.as_uri()], capture_output=True)
    for _ in range(120):
        if png.exists() and png.stat().st_size:
            time.sleep(0.5)
            image = Image.open(png).convert("RGBA")
            shutil.rmtree(profile, ignore_errors=True)  # a browser profile per page adds up over 500 pages
            return image
        time.sleep(0.5)
    raise RuntimeError(f"no screenshot for {html_path}")


def filled_page(meta, rng, families):
    """Steps 1-2: the template filled with text and styled. Returns (style choices, html with file:// links)."""
    s, css = style(rng, families)
    text = (TEMPLATES / meta["id"] / "template.html.j2").read_text(encoding="utf-8")
    values = {slot["id"]: fill(slot, rng, s["digits"]) for slot in meta["slots"]}
    page = Environment().from_string(text).render(style=css, **values)
    return s, page.replace('src="img_', f'src="{(TEMPLATES / meta["id"]).as_uri()}/img_')


def save_html(page, out_dir, name):
    """Keep the filled page as {out_dir}/{name}.html, with fonts and pictures linked relative to it."""
    for folder in (OCRSMITH / "fonts", TEMPLATES):
        if folder.drive.lower() == Path(out_dir).resolve().drive.lower():  # no relative path across drives
            page = page.replace(folder.as_uri(), Path(os.path.relpath(folder, out_dir)).as_posix())
    (out_dir / f"{name}.html").write_text(page, encoding="utf-8")


def make_variant(job):
    """Fill, style, render, paper and capture one template; writes {out_dir}/{name}.jpg and .html."""
    meta, name, seed, families, work, out_dir, scale = job
    rng = random.Random(seed)
    s, page = filled_page(meta, rng, families)
    save_html(page, out_dir, name)
    html_path = work / f"{name}.html"
    html_path.write_text(page, encoding="utf-8")
    w_mm, h_mm = page_dims(meta["page"]["size"], meta["page"]["orientation"])
    size = (round(w_mm * 96 / 25.4), round(h_mm * 96 / 25.4))
    ink = screenshot(html_path, size, work, scale)
    paper, s["background"] = background(rng, *ink.size)
    image = Image.alpha_composite(paper, ink).convert("RGB")
    s["capture"] = rng.choices(list(PRESETS), weights=list(PRESETS.values()))[0]
    image, _, records = build_preset(s["capture"]).apply(image, Page(image.width, image.height), rng)
    s["effects"] = [r.to_dict()["name"] for r in records]
    image.convert("RGB").save(out_dir / f"{name}.jpg", quality=85)
    return {"id": meta["id"], "source": meta["source"], "name": name, **s}


# -- gallery ----------------------------------------------------------------------------------------

GALLERY = Environment(autoescape=True).from_string("""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Rendered variants demo</title>
<style>
body { font: 14px sans-serif; margin: 16px; background: #eceff1; color: #222; }
h1 { font-size: 20px; margin: 0 0 4px; } p.lead { margin: 0 0 14px; color: #555; max-inline-size: 70em; }
.row { background: #fff; border: 1px solid #ccc; padding: 10px; margin-block-end: 14px; }
.row h2 { font-size: 14px; margin: 0 0 8px; } .row h2 a { color: #1565c0; font-weight: normal; margin-inline-start: 8px; }
.variants { display: flex; gap: 12px; flex-wrap: wrap; }
figure { margin: 0; inline-size: 300px; } figure img { inline-size: 300px; display: block; border: 1px solid #bbb; }
figcaption { font-size: 11.5px; color: #444; margin-block-start: 4px; line-height: 1.45; }
.tag { display: inline-block; background: #eef; border-radius: 3px; padding: 0 4px; margin: 1px 2px 1px 0; }
.cap-scan { background: #e8f5e9; } .cap-photo { background: #fff3e0; } .cap-fax { background: #fce4ec; }
.cap-archive { background: #efebe9; } .cap-clean { background: #f5f5f5; }
#bar { margin-block-end: 12px; }
</style></head><body>
<h1>From empty template to varied, scan-like pages (POC demo)</h1>
<p class="lead">Each row is one template from the bank; each image is the same template rendered with a random
font family and weight, size, line spacing, alignment, ink, digits, paper and capture style. The text is
sample Arabic (repeats are expected). Click an image to open it full size.</p>
<div id="bar"><label>Source <select id="source"><option value="">all</option><option>grammar</option>
<option>websight</option><option>doclaynet</option></select></label> · {{ rows|length }} templates ·
{{ count }} images</div>
{% for row in rows %}
<div class="row" data-source="{{ row.source }}">
  <h2>{{ row.id }} ({{ row.source }})<a href="../previews/{{ row.id }}.html" target="_blank">empty template</a></h2>
  <div class="variants">
  {% for v in row.variants %}
    <figure><a href="{{ v.file }}" target="_blank"><img loading="lazy" src="{{ v.file }}" alt="variant {{ v.variant }}"></a>
      <figcaption><span class="tag">{{ v.font }} {{ v.body_weight }}</span>{% if v.heading_font != v.font %}<span class="tag">headings: {{ v.heading_font }}</span>{% endif %}
        <span class="tag">{{ v.size_pt }}pt · line {{ v.line_height }}</span><span class="tag">{{ v.align }}</span>
        <span class="tag">digits: {{ v.digits }}</span><span class="tag">{{ v.background }}</span>
        <span class="tag cap-{{ v.capture }}">{{ v.capture }}</span>{% if v.effects %} {{ v.effects|join(", ") }}{% endif %}</figcaption></figure>
  {% endfor %}
  </div>
</div>
{% endfor %}
<script>
document.getElementById("source").onchange = (e) => {
  for (const row of document.querySelectorAll(".row"))
    row.style.display = !e.target.value || row.dataset.source === e.target.value ? "" : "none";
};
</script></body></html>
""")


def run_jobs(jobs, workers):
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(make_variant, jobs))


def gallery_seed(seed, tid):
    # Seeded per template, so one template's render does not depend on which others are in the bank.
    return random.Random(f"{seed}-{tid}").getrandbits(32)


def gallery_html_only(seed):
    """Rewrite renders/{id}.html without rendering again: the text and style depend only on the seed."""
    out, families = ROOT / "renders", font_families()
    done = json.loads((out / "renders.json").read_text(encoding="utf-8"))
    for m in load_bank():
        s, page = filled_page(m, random.Random(gallery_seed(seed, m["id"])), families)
        if (s["font"], s["size_pt"]) != (done[m["id"]]["font"], done[m["id"]]["size_pt"]):
            raise SystemExit(f"{m['id']}: style differs from its image; render again with --gallery")
        save_html(page, out, m["id"])
    print(f"renders: wrote {len(done)} filled html pages -> renders/")
    from src import gallery
    gallery.run()


def render_gallery(seed, workers):
    """One render per template for the bank gallery: renders/{id}.jpg and .html, plus renders/renders.json."""
    out = ROOT / "renders"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    families = font_families()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        jobs = [(m, m["id"], gallery_seed(seed, m["id"]), families, Path(tmp), out, 1.25) for m in load_bank()]
        results = run_jobs(jobs, workers)
    (out / "renders.json").write_text(json.dumps({r["id"]: r for r in results}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"renders: {len(results)} templates -> renders/")
    from src import gallery
    gallery.run()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-source", type=int, default=8, help="templates picked from each source")
    ap.add_argument("--variants", type=int, default=3, help="renders per template")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gallery", action="store_true",
                    help="render every template once for the bank gallery instead of the demo set")
    ap.add_argument("--html-only", action="store_true",
                    help="with --gallery: only rewrite the filled html pages of existing renders")
    a = ap.parse_args()
    if a.gallery:
        return gallery_html_only(a.seed) if a.html_only else render_gallery(a.seed, a.workers)
    rng = random.Random(a.seed)
    bank = load_bank()
    picked = [m for src in ("grammar", "websight", "doclaynet")
              for m in rng.sample([m for m in bank if m["source"] == src], a.per_source)]
    shutil.rmtree(DEMO / "out", ignore_errors=True)
    (DEMO / "out").mkdir(parents=True)
    families = font_families()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        jobs = [(m, f"{m['id']}__v{k}", rng.getrandbits(32), families, Path(tmp), DEMO / "out", SCALE)
                for m in picked for k in range(a.variants)]
        results = run_jobs(jobs, a.workers)
    for k, r in enumerate(results):
        r["variant"], r["file"] = k % a.variants, f"out/{r['name']}.jpg"
    rows = [{"id": m["id"], "source": m["source"], "variants": [r for r in results if r["id"] == m["id"]]}
            for m in picked]
    (DEMO / "variants.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    (DEMO / "index.html").write_text(GALLERY.render(rows=rows, count=len(results)), encoding="utf-8")
    print(f"demo: {len(results)} images from {len(picked)} templates, {len(families)} font families -> demo/index.html")


if __name__ == "__main__":
    main()
