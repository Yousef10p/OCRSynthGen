"""Demo gallery: two previews per template and gallery/index.html (static, opens from disk).

previews/{id}.html shows each slot as a role marker; previews/{id}.text.html fills the slots with sample
Arabic words, so the template's own typography (fonts, sizes, line height, alignment) becomes visible.
The gallery shows both side by side. The text previews load the default fonts from assets/fonts/.
"""
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict

from jinja2 import Environment
from markupsafe import Markup, escape
from PIL import ImageFont

from src.bank import ROOT, TEMPLATES, page_dims
from src.dedup import load_bank
from src.markdown import to_markdown

PREVIEWS, GALLERY, FONTS, RENDERS = ROOT / "previews", ROOT / "gallery", ROOT / "assets" / "fonts", ROOT / "renders"
SYNTHDOC = ROOT / "synthdoc_out"
SOURCES = [("grammar", "Grammar"), ("websight", "WebSight"), ("doclaynet", "DocLayNet")]
# The gallery previews every template of the bank's first build and PER_SOURCE others per source
PER_SOURCE = 400
FIRST_BUILD = re.compile(r"^(g_\d{4}|w_\d{5}|d_\d{4})$")
# Versions of each bank template in the Show menu: (value, label, which items have it)
VERSIONS = [("empty", "empty", lambda it: True), ("ocrsmith", "OCRSmith render", lambda it: it["render"]),
            ("synthdoc", "SynthDoc-augraphy-scan-simulator", lambda it: it["synthdoc"]),
            ("synthdoc_ocrsmith", "SynthDoc-OCRSmith", lambda it: it["synthdoc"] and it["synthdoc"]["ocrsmith"])]
# SynthDoc pages attacked by synthdoc_adapter/attack.py and attack_ocrsmith.py
ATTACKED, ATTACKED_OCRSMITH = ROOT / "synthdoc_attacked", ROOT / "synthdoc_ocrsmith"
# The OCRSmith proof of concept (../ocrsmith/poc_templates): its templates, previews and OCRSmith's own pages.
OCRSMITH_POC = ROOT.parent / "ocrsmith" / "poc_templates"
MM_TO_PX = 96 / 25.4
ROLE_COLORS = {
    "title": "#d62728", "subtitle": "#ff7f0e", "heading": "#e377c2", "paragraph": "#1f77b4",
    "list_item": "#2ca02c", "table_header": "#9467bd", "cell": "#8c564b", "caption": "#17becf",
    "label": "#bcbd22", "field_value": "#7f7f7f", "quote": "#393b79", "footnote": "#637939",
    "header": "#8c6d31", "footer": "#843c39", "page_number": "#000000", "sidebar": "#7b4173",
}
PREVIEW_CSS = (
    "body { background: #fff; }\n.page { outline: 1px solid #ccc; }\n.box { outline: 1px dashed #9ab; }\n"
    "mark.slot { background: #e9e9e9; color: #333; border: 1.5px solid var(--c); border-radius: 3px;"
    " padding: 0 0.3em; font: 9pt/1.5 sans-serif; unicode-bidi: isolate; white-space: nowrap; }\n"
    + "".join(f'mark.slot[data-role="{r}"] {{ --c: {c}; }}\n' for r, c in ROLE_COLORS.items())
)
TEXT_CSS = "body { background: #fff; }\n.page { outline: 1px solid #ccc; }\n"
# Our own neutral sample sentence ("this page shows what the template looks like filled with real Arabic text").
SAMPLE = ("تعرض هذه الصفحة مثالا لشكل القالب بعد ملئه بنص عربي حقيقي يوضح الخطوط والأحجام"
          " والمسافات بين الأسطر والفقرات والجداول والقوائم في الوثيقة").split()
WEIGHTS = {"regular": 400, "medium": 500, "semibold": 600, "bold": 700}
ENV = Environment(autoescape=False)

INDEX = ENV.from_string("""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Arabic template bank</title>
<style>
body { font: 14px sans-serif; margin: 16px; background: #f3f3f3; }
#bar { display: flex; gap: 12px; align-items: center; margin-block-end: 12px; flex-wrap: wrap; }
#grid { display: flex; flex-wrap: wrap; gap: 14px; }
.card { background: #fff; border: 1px solid #ccc; padding: 6px; }
.pair { display: flex; gap: 8px; }
.pair figure { margin: 0; inline-size: 180px; position: relative; }
.pair figcaption { font-size: 11px; color: #666; margin-block-start: 2px; }
.card .frame { overflow: hidden; border: 1px solid #ddd; }
.card iframe { border: 0; transform-origin: 0 0; pointer-events: none; display: block; }
.card .frame img { inline-size: 178px; display: block; }
.card figure a { position: absolute; inset: 0; }
.card a.html { display: inline-block; margin-inline-start: 8px; padding: 1px 8px; border: 1px solid #1565c0;
               border-radius: 3px; color: #1565c0; text-decoration: none; font-size: 12px; }
.card a.html:hover { background: #e3f2fd; }
.card .cap { font-size: 12px; color: #444; margin-block-start: 4px; display: flex; flex-wrap: wrap; gap: 4px;
             align-items: center; max-inline-size: 368px; }
.card .cap a.html { margin-inline-start: 0; }
#empty { color: #555; font-size: 16px; margin: 48px 0; text-align: center; }
.legend span { display: inline-block; padding: 0 6px; margin: 2px; border: 2px solid; border-radius: 3px; font-size: 12px; }
</style></head><body>
<div id="bar">
  <label><b>Show</b> <select id="show"><option value="">choose…</option>
    {% for group, options in menu %}<optgroup label="{{ group }}">
      {% for key, label, n in options %}<option value="{{ key }}">{{ label }} ({{ n }})</option>{% endfor %}
    </optgroup>{% endfor %}</select></label>
  <label>Feature <select id="feature"><option value="">any</option>
    <option value="tables">tables</option><option value="lists">lists</option>
    <option value="multi">multi-column</option><option value="forms">forms</option>
    <option value="asides">asides</option><option value="band">header/footer</option></select></label>
  <span id="shown"></span>
</div>
<div class="legend">{% for r, c in colors.items() %}<span style="border-color: {{ c }}">{{ r }}</span>{% endfor %}</div>
<div id="grid"></div>
<p><button id="more" style="display:none" onclick="more()">Show 60 more</button></p>
<p id="empty">Nothing shown yet: pick a set from the <b>Show</b> menu above.</p>
<script>
const ITEMS = {{ items_json }};
const TESTS = {tables: f => f.tables > 0, lists: f => f.lists > 0, multi: f => f.columns > 1,
               forms: f => f.forms > 0, asides: f => f.asides > 0, band: f => f.has_header || f.has_footer};
const grid = document.getElementById("grid");
// Menu values are "<source>:<version>". Versions of a bank template: empty (the template alone), ocrsmith
// (our render pipeline: OCRSmith fonts, paper and capture effects), synthdoc (SynthDoc's render + labels,
// attacked with Augraphy + scan-simulator) and synthdoc_ocrsmith (the same render attacked with OCRSmith's effects).
function frame(it, view, label) {
  const scale = 178 / it.w, url = `${it.base || "../previews/"}${it.id}${view}.html`;
  return `<figure><div class="frame" style="height:${Math.round(it.h * scale)}px">
      <iframe loading="lazy" scrolling="no" src="${url}" width="${it.w}" height="${it.h}"
              style="transform: scale(${scale})"></iframe></div>
    <figcaption>${label}</figcaption><a href="${url}" target="_blank" title="open ${label}"></a></figure>`;
}
const button = (href, label) => `<a class="html" href="${href}" target="_blank">${label} ↗</a>`;
const cap = (it, extra = "") => `<div class="cap">${it.id} · ${it.page} · ${it.features.slot_count} slots ${extra}</div>`;
// empty: the template alone
function emptyCard(it) {
  return `<div class="pair">${frame(it, "", "empty template")}</div>${cap(it)}`;
}
// ocrsmith: the template, and our render of it with OCRSmith's fonts, paper and capture effects
function ocrsmithRenderCard(it) {
  const r = it.render, url = `../renders/${it.id}.jpg`;
  return `<div class="pair">${frame(it, "", "empty template")}
    <figure><div class="frame"><img loading="lazy" src="${url}" alt="rendered ${it.id}"></div>
      <figcaption>${r.font} ${r.size_pt}pt · ${r.background} · ${r.capture}</figcaption>
      <a href="${url}" target="_blank" title="open the rendered page"></a></figure></div>
    ${cap(it, button(`../previews/${it.id}.code.html`, "HTML code") + button(`../previews/${it.id}.md.html`, "Markdown"))}`;
}
// OCRSmith documents: OCRSmith's own page, and the empty template exported from it (ocrsmith/poc_templates)
function ocrsmithDocCard(it) {
  const o = it.ocrsmith;
  return `<div class="pair">
    <figure><div class="frame"><img loading="lazy" src="${o.page}" alt="OCRSmith page"></div>
      <figcaption>OCRSmith page (${o.genre})</figcaption><a href="${o.page}" target="_blank" title="open OCRSmith's page"></a></figure>
    ${frame(it, "", "empty template · " + o.route)}</div>${cap(it)}`;
}
// synthdoc*: the template, and SynthDoc's render attacked (pages and labels in ../<dir>/), boxes moved to match
const synthdocCard = (dir, effects) => (it) => {
  const d = it.synthdoc, p = d.pages[0], clean = `../synthdoc_out/images/${p}.jpg`, done = d[effects];
  const attacked = done ? `../${dir}/images/${p}.jpg` : clean;
  const boxes = done
    ? button(`../${dir}/overlays/${p}_words.jpg`, "Word boxes") + button(`../${dir}/overlays/${p}_slots.jpg`, "Slot boxes")
      + button(`../${dir}/labels/${p}.json`, "Labels") + button(clean, "Clean SynthDoc page")
    : button(`../synthdoc_out/ocr_overlays/${p}_ocr.jpg`, "Word boxes") + button(`../synthdoc_out/overlays/${p}_overlay.jpg`, "Slot boxes");
  return `<div class="pair">${frame(it, "", "empty template")}
    <figure><div class="frame"><img loading="lazy" src="${attacked}" alt="SynthDoc page ${p}"></div>
      <figcaption>SynthDoc: ${d.font} ${d.size_pt}pt${done ? " · " + done : ""}</figcaption>
      <a href="${attacked}" target="_blank" title="open the page"></a></figure></div>
    ${cap(it, button(`../previews/${p}.code.html`, "HTML code") + button(`../previews/${p}.md.html`, "Markdown") + boxes)}`;
};
const VERSIONS = {empty: [() => true, emptyCard], ocrsmith: [(it) => it.render, ocrsmithRenderCard],
                  synthdoc: [(it) => it.synthdoc, synthdocCard("synthdoc_attacked", "preset")],
                  synthdoc_ocrsmith: [(it) => it.synthdoc && it.synthdoc.ocrsmith, synthdocCard("synthdoc_ocrsmith", "ocrsmith")],
                  documents: [() => true, ocrsmithDocCard]};
// Nothing is built until a set is picked: the page opens empty.
// Cards are added PAGE at a time ("Show more"), so a set of thousands opens at once.
const PAGE = 60;
let matching = [], shown = 0;
function apply() {
  const value = document.getElementById("show").value, feat = document.getElementById("feature").value;
  const [source, version] = value.split(":");
  grid.innerHTML = "";
  document.getElementById("empty").style.display = value ? "none" : "";
  matching = value ? ITEMS.filter((it) => it.source === source && VERSIONS[version][0](it)
                                  && (!feat || TESTS[feat](it.features))) : [];
  shown = 0;
  more();
}
function more() {
  const version = document.getElementById("show").value.split(":")[1];
  for (const it of matching.slice(shown, shown + PAGE)) {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = VERSIONS[version][1](it);
    grid.appendChild(card);
  }
  shown = Math.min(shown + PAGE, matching.length);
  document.getElementById("shown").textContent = matching.length ? `${shown} of ${matching.length} shown` : "";
  document.getElementById("more").style.display = shown < matching.length ? "" : "none";
}
document.getElementById("show").onchange = apply;
document.getElementById("feature").onchange = apply;
const preset = location.hash.slice(1);  // gallery/index.html#grammar:synthdoc opens with that set picked
if ([...document.getElementById("show").options].some((o) => o.value === preset && preset))
  document.getElementById("show").value = preset;
apply();
</script></body></html>
""")


def marker(slot):
    lo, hi = slot["words"]
    return f'<mark class="slot" data-role="{slot["role"]}">{slot["role"]} · {lo}–{hi}</mark>'


def sample_text(slot, k):
    """Sample Arabic words for slot k: the middle of its word range, starting at a different word per slot."""
    if slot["role"] == "page_number":
        return "٣"  # Arabic-Indic three
    n = sum(slot["words"]) // 2
    return " ".join(SAMPLE[(k + i) % len(SAMPLE)] for i in range(n))


def font_css():
    """@font-face rules for the fonts in assets/fonts, as seen from previews/."""
    rules = []
    for path in sorted(FONTS.glob("*.ttf")):
        family, style = ImageFont.truetype(str(path), 12).getname()
        key = style.lower().replace(" ", "").replace("italic", "") or "regular"
        rules.append(f'@font-face {{ font-family: "{family}"; src: url("../assets/fonts/{path.name}");'
                     f' font-weight: {WEIGHTS.get(key, 400)}; font-style: {"italic" if "Italic" in style else "normal"}; }}')
    return "\n".join(rules) + "\n"


CODE = Environment(autoescape=True).from_string("""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{{ id }}: {{ kind }}</title>
<style>
body { margin: 0; font: 14px sans-serif; background: #1e1e1e; color: #ddd; }
header { position: sticky; top: 0; display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
         padding: 10px 16px; background: #2d2d2d; border-block-end: 1px solid #444; }
header b { color: #fff; } header span { color: #aaa; font-size: 12px; }
header a, header button { color: #9cdcfe; background: none; border: 1px solid #555; border-radius: 3px;
                          padding: 3px 10px; font: inherit; font-size: 13px; cursor: pointer; text-decoration: none; }
pre { margin: 0; padding: 16px; white-space: pre-wrap; word-break: break-word;
      font: 13px/1.55 Consolas, "Courier New", monospace; }
pre .ar { unicode-bidi: isolate; color: #ce9178; }
pre.md { unicode-bidi: plaintext; font-size: 14px; }  /* each Markdown line takes the direction of its text */
textarea { position: absolute; inset-inline-start: -9999px; }
</style></head><body>
<header><b>{{ id }}</b><span>{{ kind }} of the filled page: {{ desc }}</span>
  <button id="copy">Copy</button>
  <a href="{{ page_href }}" target="_blank">Open as page ↗</a>
  <a href="{{ image_href }}" target="_blank">Image ↗</a></header>
<pre dir="ltr" class="{{ pre_class }}">{{ shown }}</pre>
<textarea id="raw" readonly>{{ code }}</textarea>
<script>
document.getElementById("copy").onclick = (e) => {  // copies the untouched source, not the display markup
  const raw = document.getElementById("raw"); raw.select();
  e.target.textContent = document.execCommand("copy") ? "Copied" : "Press Ctrl+C";
};
</script></body></html>
""")
ARABIC_RUN = re.compile(r"[^<>\n]*[؀-ۿ][^<>\n]*")


def code_markup(code):
    """Escaped source with every Arabic text run isolated, so mixed Arabic/tag lines display in order."""
    parts, last = [], 0
    for m in ARABIC_RUN.finditer(code):
        parts += [escape(code[last:m.start()]), Markup('<span class="ar" dir="rtl">'), escape(m.group()),
                  Markup("</span>")]
        last = m.end()
    return Markup("").join(parts + [escape(code[last:])])


def render_preview(text, meta, values, style):
    page = ENV.from_string(text).render(style=style, **values)
    return page.replace('src="img_', f'src="../templates/{meta["id"]}/img_')  # previews live in another folder


def run():
    shutil.rmtree(PREVIEWS, ignore_errors=True)
    PREVIEWS.mkdir()
    GALLERY.mkdir(exist_ok=True)
    items, text_css = [], font_css() + TEXT_CSS
    # Rendered pages from demo/render_demo.py --gallery, if they were made.
    renders = RENDERS / "renders.json"
    renders = json.loads(renders.read_text(encoding="utf-8")) if renders.exists() else {}
    bank = load_bank()
    totals = Counter(m["source"] for m in bank)
    for m in gallery_sample(bank):
        text = (TEMPLATES / m["id"] / "template.html.j2").read_text(encoding="utf-8")
        markers = {s["id"]: marker(s) for s in m["slots"]}
        words = {s["id"]: sample_text(s, k) for k, s in enumerate(m["slots"])}
        (PREVIEWS / f'{m["id"]}.html').write_text(render_preview(text, m, markers, PREVIEW_CSS), encoding="utf-8")
        (PREVIEWS / f'{m["id"]}.text.html').write_text(render_preview(text, m, words, text_css), encoding="utf-8")
        filled = RENDERS / f'{m["id"]}.html'
        if filled.exists():  # the filled page as HTML source and as Markdown, each on its own page
            code_pages(m["id"], filled.read_text(encoding="utf-8"), f'../renders/{m["id"]}.html',
                       f'../renders/{m["id"]}.jpg', "before paper and scan effects")
        w, h = page_dims(m["page"]["size"], m["page"]["orientation"])
        items.append({"id": m["id"], "source": m["source"], "features": m["features"],
                      "page": f'{m["page"]["size"]} {m["page"]["orientation"]}',
                      "w": round(w * MM_TO_PX), "h": round(h * MM_TO_PX),
                      "render": renders.get(m["id"]) if (RENDERS / f'{m["id"]}.jpg').exists() else None,
                      "synthdoc": None})
    attach_synthdoc({it["id"]: it for it in items})
    items += ocrsmith_items()
    count = lambda source, has: sum(it["source"] == source and bool(has(it)) for it in items)
    menu = [(label, [(f"{key}:{version}", f"{label} · {name}" + (f" (of {totals[key]:,} in the bank)"
                                                                 if version == "empty" and totals[key] > n else ""), n)
                     for version, name, has in VERSIONS if (n := count(key, has))]) for key, label in SOURCES]
    menu.append(("OCRSmith documents", [("ocrsmith-poc:documents", "OCRSmith pages → templates",
                                         count("ocrsmith-poc", bool))]))
    (GALLERY / "index.html").write_text(INDEX.render(menu=[g for g in menu if g[1] and g[1][0][2]],
                                                     items_json=json.dumps(items), colors=ROLE_COLORS), encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # menu labels contain "→"
    print("gallery: " + "; ".join(f"{label} {n}" for _, opts in menu for _, label, n in opts) + " -> gallery/index.html")


def gallery_sample(bank, per_source=PER_SOURCE):
    """Every template of the bank's first build, plus per_source others from each source (a fixed random
    sample): previews of tens of thousands of templates would take gigabytes."""
    first = [m for m in bank if FIRST_BUILD.match(m["id"])]
    rest = defaultdict(list)
    for m in bank:
        if not FIRST_BUILD.match(m["id"]):
            rest[m["source"]].append(m)
    picked = [m for ms in rest.values() for m in random.Random(0).sample(ms, min(per_source, len(ms)))]
    return sorted(first + picked, key=lambda m: m["id"])


def code_pages(name, code, page_href, image_href, when):
    """previews/{name}.code.html (HTML source), previews/{name}.md (+ .md.html viewer)."""
    md = to_markdown(code)
    (PREVIEWS / f"{name}.md").write_text(md, encoding="utf-8")
    (PREVIEWS / f"{name}.code.html").write_text(CODE.render(
        id=name, kind="HTML code", desc=f"the template with its text and style, {when}", code=code,
        shown=code_markup(code), pre_class="", page_href=page_href, image_href=image_href), encoding="utf-8")
    (PREVIEWS / f"{name}.md.html").write_text(CODE.render(
        id=name, kind="Markdown", desc="its text in reading order, with headings, lists, tables and figures",
        code=md, shown=md, pre_class="md", page_href=page_href, image_href=image_href), encoding="utf-8")


def ocrsmith_items():
    """Cards for the OCRSmith templates (export_templates.py + make_gallery.py in ../ocrsmith/poc_templates)."""
    if not (OCRSMITH_POC / "previews").exists():
        return []
    items = []
    for path in sorted((OCRSMITH_POC / "out").glob("*/template.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        flow = m["positioning"] == "flow"
        # A flow template is a whole document: show the document's first page
        page = f"p_0000{m['id'][2:]}_01.png" if flow else f"{m['id']}.png"
        w, h = page_dims(m["page"]["size"], m["page"]["orientation"])
        items.append({"id": m["id"], "source": "ocrsmith-poc", "features": m["features"], "render": None,
                      "page": f'{m["page"]["size"]} {m["page"]["orientation"]}',
                      "w": round(w * MM_TO_PX), "h": round(h * MM_TO_PX),
                      "base": "../../ocrsmith/poc_templates/previews/",
                      "ocrsmith": {"page": f"../../ocrsmith/poc_templates/renders/{page}", "genre": m["source_ref"],
                                   "route": "flow" if flow else "layout"}})
    return items


def attach_synthdoc(bank):
    """Give each bank item its SynthDoc render (synthdoc_adapter/generate.py), if one was made."""
    manifest = SYNTHDOC / "manifest.json"
    if not manifest.exists():
        return
    for doc in json.loads(manifest.read_text(encoding="utf-8"))["documents"]:
        tid, first = doc["template"].removeprefix("bank_"), doc["pages"][0]
        if tid not in bank:
            continue
        html_file, ocr = SYNTHDOC / "html" / f"{first}.html", SYNTHDOC / "ocr" / f"{first}.json"
        variant = (json.loads(ocr.read_text(encoding="utf-8")).get("variant") or {}) if ocr.exists() else {}
        if html_file.exists():
            code_pages(first, html_file.read_text(encoding="utf-8"), f"../synthdoc_out/html/{first}.html",
                       f"../synthdoc_out/images/{first}.jpg", "as SynthDoc rendered it")
        bank[tid]["synthdoc"] = {"doc": first, "pages": doc["pages"], "font": variant.get("font", ""),
                                 "size_pt": variant.get("size_pt", ""), "preset": None, "ocrsmith": None}
        # Captions for the attacked versions, from the attacks named in their labels
        attacks = attacks_of(ATTACKED, first)
        if attacks:  # ["augraphy", "scan-simulator:<preset>"]
            augraphy = "Augraphy · " if "augraphy" in attacks else ""
            bank[tid]["synthdoc"]["preset"] = f'{augraphy}scan-simulator {attacks[-1].split(":")[-1]}'
        attacks = attacks_of(ATTACKED_OCRSMITH, first)
        if attacks:  # ["ocrsmith-paper:<kind>", "ocrsmith:<preset>", <effects>...]
            bank[tid]["synthdoc"]["ocrsmith"] = f'OCRSmith · {attacks[0].split(":", 1)[1]} · {attacks[1].split(":")[1]}'


def attacks_of(folder, page):
    labels = folder / "labels" / f"{page}.json"
    return json.loads(labels.read_text(encoding="utf-8"))["attacks"] if labels.exists() else None


if __name__ == "__main__":
    run()
