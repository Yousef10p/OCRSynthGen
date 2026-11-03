"""DocPipeline: one class for the whole path from an empty template to a labelled, scan-like page.

    1. template  a template from the 2,000-template bank (bank2000/, each Arabic or English), or one written
                 by an LLM (Groq) when asked
    2. fill      a random typography and visual theme (pipeline/themes.py), text from Wikipedia that follows
                 one article per page (pipeline/text_pool.py), a photo from img_pool/ for every picture slot
                 (by the slot's hint: food, people, city...)
    3. render    SynthDoc lays the page out in Chromium (at 2x, about 192 dpi) and measures every word; a
                 script in the page fits text to its boxes and fixes contrast first; the page is checked
                 (pipeline/qa.py) and made again until no text is hidden, cut, overlapping or covered
    4. effects   clean (15%), or OCRSmith, Augraphy or scan-simulator at a chosen strength
                 (pipeline/degrade.py); every box moves with the page; checked word by word for lost,
                 faint or blurred-away text, retried with new seeds and then less strength
    5. outputs   the image; the slot-level ground truth (<text>... <bbox>x1, y1, x2, y2</bbox></text>, one
                 line per filled slot, pipeline/groundtruth.py) as the Markdown; labels JSON (words, lines,
                 slots); the filled HTML

Each sample writes, in out_dir:
    <name>.jpg        final page                     <name>.md       the ground truth: one line per text block,
    <name>.json       labels: words, lines, slots                      <text>... <bbox>x1, y1, x2, y2</bbox></text>
    <name>.html       filled page                    <name>_clean.jpg, _words.jpg, _slots.jpg (extras)

Runs in .venv-pipeline (SynthDoc, OCRSmith, Augraphy and scan-simulator in one environment):

    .venv-pipeline\\Scripts\\python -m pipeline --n 16
"""
import json
import random
import re
import sys
import time
from pathlib import Path

from jinja2 import Template
from markupsafe import Markup
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "demo", ROOT / "synthdoc_adapter"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import attack  # noqa: E402  (Augraphy warm-up, word mask; synthdoc_adapter/attack.py)
import numpy as np  # noqa: E402
from export import synthdoc_html  # noqa: E402
from fill_style import fill, font_families, style, style_latin  # noqa: E402
from labels import build_label, contrast, split_words  # noqa: E402
from playwright.sync_api import Browser  # noqa: E402
from synthdoc.ner_data_generator import NERDataGenerator  # noqa: E402
from synthdoc_arabic import pipeline as synthdoc_arabic  # noqa: E402
from synthdoc_arabic.ocr_gt import build_page_gt  # noqa: E402

# -- rendering at a higher resolution -----------------------------------------------------------------------
# SynthDoc's browser page has no resolution setting: a page is screenshot at 1 pixel per CSS pixel (96 dpi;
# 9 pt text is 12 px tall), too small for OCR training. Playwright pages are opened here with a device scale
# factor of RENDER_SCALE (2 = about 192 dpi); the measured boxes are in CSS pixels and are scaled to match.
RENDER_SCALE = 2
_new_page = Browser.new_page


def _scaled_new_page(self, *args, **kwargs):
    kwargs.setdefault("device_scale_factor", RENDER_SCALE)
    return _new_page(self, *args, **kwargs)


Browser.new_page = _scaled_new_page

# -- what the Arabic extension's word script measures, extended (in memory; the SynthDoc folder is unchanged)
# Per word: cut_x, cut on its left or right; cut_y, the share of its letters' height hidden above or below
# (measured on the letters with canvas metrics, against the clipping edge, not on the font's taller line slot);
# slot, the template slot it is in. Per page (qa): every slot's full text (to find words that are hidden), the
# rectangles of pictures, and every slot's text contrast against what is behind it.
CUT_Y_JS = ("(() => { const m = ctx.measureText(shown); const base = b.t + m.fontBoundingBoxAscent;"
            " const top = base - m.actualBoundingBoxAscent, bottom = base + m.actualBoundingBoxDescent;"
            " return (Math.max(0, clip.t - top) + Math.max(0, bottom - clip.b)) / Math.max(1, bottom - top); })()")
CONTRAST_JS = """
  const rgb = (c) => { const m = (c || '').match(/[\\d.]+/g); return m ? m.map(Number) : [0, 0, 0, 0]; };
  const lum = ([r, g, b]) => { const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
                               return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b); };
  // an element's own paint: null for a picture, its colour if opaque, undefined if see-through
  const paint = (e) => { const s = getComputedStyle(e);
    if (s.backgroundImage && s.backgroundImage !== 'none') return null;
    const c = rgb(s.backgroundColor); return c.length < 4 || c[3] > 0.5 ? c.slice(0, 3) : undefined; };
  // what is painted under a point of the text: the elements stacked there, from the text down, so another
  // element's dark band under the text counts (not only the text's own boxes); undefined if not visible there
  const under = (el, x, y) => { if (x < 0 || y < 0 || x >= innerWidth || y >= innerHeight) return undefined;
    const stack = document.elementsFromPoint(x, y), i = stack.findIndex((e) => e === el || el.contains(e));
    if (i < 0) return undefined;
    for (const e of stack.slice(i)) { const p = paint(e); if (p !== undefined) return p; }
    return [255, 255, 255]; };
  // otherwise: the nearest ancestor whose box covers the text (text overflowing a dark band stands on the
  // page, not on the band)
  const behind = (el) => { const r = el.getBoundingClientRect();
    for (let e = el; e; e = e.parentElement) { const b = e.getBoundingClientRect();
      if (e !== el && e !== document.documentElement && e !== document.body && (r.top < b.top - 1 || r.bottom > b.bottom + 1
          || r.left < b.left - 1 || r.right > b.right + 1)) continue;
      const p = paint(e); if (p !== undefined) return p; }
    return [255, 255, 255]; };
  const pair = (fg, bg) => { const a = lum(fg), b = lum(bg); return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05); };
  // the worst contrast over the text's lines; the colour behind it is that of its worst line (null: a picture)
  const look = (el) => { const fg = rgb(getComputedStyle(el).color).slice(0, 3);
    let worst = 21, bg = null;
    const rects = [...el.getClientRects()].filter((r) => r.width > 0 && r.height > 0).slice(0, 8);
    for (const r of rects.length ? rects : [el.getBoundingClientRect()]) {
      let c = under(el, r.left + r.width / 2, r.top + r.height / 2);
      if (c === undefined) c = behind(el);
      if (c && pair(fg, c) < worst) { worst = pair(fg, c); bg = c; }
    }
    return {ratio: worst, bg}; };
  const ratio = (el) => look(el).ratio;
"""
QA_JS = """(() => {""" + CONTRAST_JS + """
  const slots = {}, pictures = [], contrast = [], covered = [];
  for (const s of content.querySelectorAll('[data-slot]')) {
    slots[s.dataset.slot] = ((slots[s.dataset.slot] || '') + ' ' + s.textContent).trim();
    if (s.textContent.trim()) contrast.push({slot: s.dataset.slot, ratio: Math.round(ratio(s) * 100) / 100});
    // anything opaque painted over the text (a band, a box, a picture positioned on top of it): three points
    // across the middle of each of its lines
    let hits = 0;
    for (const r of [...s.getClientRects()].filter((r) => r.width > 0 && r.height > 0)) {
      for (const f of [0.2, 0.5, 0.8]) {
        const x = r.left + r.width * f, y = r.top + r.height / 2;
        if (x < 0 || y < 0 || x >= innerWidth || y >= innerHeight) continue;
        const stack = document.elementsFromPoint(x, y);
        const i = stack.findIndex((e) => e === s || s.contains(e));
        const above = i < 0 ? stack.slice(0, 1) : stack.slice(0, i);
        if (above.some((e) => !e.contains(s) && paint(e) !== undefined)) hits++;
      }
    }
    if (hits) covered.push({slot: s.dataset.slot, points: hits});
  }
  for (const im of content.querySelectorAll('img, .ph, .pic-frame, [data-img]')) {
    const r = im.getBoundingClientRect();
    if (r.width >= 2 && r.height >= 2) pictures.push({x: r.left - pr.left, y: r.top - pr.top, width: r.width, height: r.height});
  }
  // the page's structure, for block-level labels (pipeline/surya.py): tables with their rows and cells, lists
  // with their items and marker, pictures, each slot's heading tag and weight, all in document order
  const rel = (r) => ({x: r.left - pr.left, y: r.top - pr.top, width: r.width, height: r.height});
  const own = (el, scope) => [...el.querySelectorAll('[data-slot]')].filter((s) => s.closest(scope) === el)
    .map((s) => s.dataset.slot);
  const struct = {tables: [], lists: [], images: [], styles: {}, order: {}};
  let order = 0;
  for (const el of content.querySelectorAll('[data-slot], table, ul, ol, .pic-frame, [data-img], img')) {
    el.dataset.order = order++;
  }
  for (const t of content.querySelectorAll('table')) {
    if (t.parentElement.closest('table')) continue;  // a table inside a cell belongs to the outer table
    const rows = [];
    for (const tr of t.querySelectorAll('tr')) {
      if (tr.closest('table') !== t) continue;
      const cells = [...tr.children].filter((c) => c.tagName === 'TD' || c.tagName === 'TH').map((c) => ({
        tag: c.tagName.toLowerCase(), colspan: c.colSpan || 1, rowspan: c.rowSpan || 1, rect: rel(c.getBoundingClientRect()),
        slots: [...c.querySelectorAll('[data-slot]')].map((s) => s.dataset.slot),
        bold: +getComputedStyle(c).fontWeight >= 600}));
      if (cells.length) rows.push({rect: rel(tr.getBoundingClientRect()), cells});
    }
    if (rows.length) struct.tables.push({rect: rel(t.getBoundingClientRect()), rows, order: +t.dataset.order});
  }
  for (const l of content.querySelectorAll('ul, ol')) {
    if (l.parentElement.closest('ul, ol, table')) continue;  // nested lists are items of the outer one
    const items = [...l.querySelectorAll('li')].map((li) => own(li, 'li')).filter((s) => s.length);
    const first = l.querySelector('li');
    const marker = first && getComputedStyle(first).display === 'list-item' ? getComputedStyle(first).listStyleType : 'none';
    if (items.length) struct.lists.push({rect: rel(l.getBoundingClientRect()), items, marker, order: +l.dataset.order});
  }
  for (const im of content.querySelectorAll('.pic-frame, [data-img], img')) {
    if (im.tagName === 'IMG' && im.closest('.pic-frame, [data-img]')) continue;
    const r = im.getBoundingClientRect();
    if (r.width < 8 || r.height < 8) continue;
    const img = im.tagName === 'IMG' ? im : im.querySelector('img');
    const src = img ? img.getAttribute('src') || '' : (getComputedStyle(im).backgroundImage.match(/url\\("?([^")]*)"?\\)/) || [])[1] || '';
    struct.images.push({rect: rel(r), src: src.split('/').pop(), order: +im.dataset.order});
  }
  for (const s of content.querySelectorAll('[data-slot]')) {
    const cs = getComputedStyle(s), h = s.closest('h1, h2, h3, h4, h5, h6');
    struct.styles[s.dataset.slot] = {h: h ? h.tagName.toLowerCase() : null, bold: +cs.fontWeight >= 600,
                                     italic: cs.fontStyle === 'italic'};
    if (!(s.dataset.slot in struct.order)) struct.order[s.dataset.slot] = +s.dataset.order;
  }
  return {slots, pictures, contrast, covered, struct};
})()"""
# glyph_y: the top and bottom of the word's letters on the page (canvas metrics around the baseline). Word boxes
# are the font's whole line slot, taller than the letters: on closely set lines the slots of two lines overlap
# while the letters don't, so overlap is checked on the letters.
GLYPH_JS = ("(() => { const m = ctx.measureText(shown); const base = b.t + m.fontBoundingBoxAscent;"
            " return [base - m.actualBoundingBoxAscent - pr.top, base + m.actualBoundingBoxDescent - pr.top]; })()")
if "cut_x:" not in synthdoc_arabic.WORDS_JS:
    synthdoc_arabic.WORDS_JS = synthdoc_arabic.WORDS_JS.replace(
        "split_word:", "cut_x: (l > b.l + 1 || r < b.r - 1), cut_y: " + CUT_Y_JS + ", glyph_y: " + GLYPH_JS
                       + ", slot: (el.closest('[data-slot]') || {dataset: {}}).dataset.slot || null, split_word:", 1)
    synthdoc_arabic.WORDS_JS = synthdoc_arabic.WORDS_JS.replace(
        "generated_text: generated};", "generated_text: generated, qa: " + QA_JS + "};", 1)

from pipeline import degrade as D  # noqa: E402
from pipeline import groundtruth as GT  # noqa: E402
from pipeline import surya as SURYA  # noqa: E402
from pipeline import tashkeel as TASHKEEL  # noqa: E402
from pipeline.img_pool import ImagePool  # noqa: E402
from pipeline.qa import ink_issues, render_issues  # noqa: E402
from pipeline.text_pool import TextPool  # noqa: E402
from pipeline.themes import theme  # noqa: E402

BANK2000 = ROOT / "bank2000"
LANGUAGES = {"ar": 0.5, "en": 0.5}
# "clean": some pages stay exactly as SynthDoc rendered them (born-digital documents); the rest are split evenly
# between the three tools ("augraphy+scan-simulator" is there when asked for)
EFFECTS = {"clean": 0.15, "ocrsmith": 0.85 / 3, "augraphy": 0.85 / 3, "scan-simulator": 0.85 / 3}
STRENGTHS = {tool: D.DEFAULT_STRENGTH for tool in D.TOOLS}
ATTEMPTS = 8         # renders per sample until one passes the checks (a new template every second one)
EFFECT_TRIES = 3     # seeds per strength level before the strength is lowered

# Runs in Paged.js's "before" hook, once the page's fonts and pictures have loaded and before Paged.js lays the
# page out (window.PagedConfig must exist before Paged.js loads: this goes at the end of <head>, before
# SynthDoc's Paged.js script):
#  1. colours: a web design's colours are turned around the colour wheel (themes.py: hue only, so contrast
#     stays);
#  2. text that doesn't fit its box (a template box sized for its source's text is often too small for the
#     words a slot gets in our fonts): line-height 1.2, then a smaller font down to 80%, then words removed
#     from the end of its last slot until it fits;
#  3. a row wider than its column (a form row: a no-wrap label and a 5em value in a narrow column) may wrap;
#  4. the page itself is one sheet with overflow:hidden: a slot running into the bottom margin or off a side
#     loses words from its end; a slot left empty hides its paragraph;
#  5. contrast: a slot whose text has less than 4.5:1 contrast with what is behind it gets near-black or white
#     text, whichever reads better;
#  6. empty boxes of a solid colour, at least 30 x 30 px (a web page's image placeholder the converter couldn't
#     name: a grey or coloured square or banner with nothing in it) get a photo.
PAGE_JS = """window.PagedConfig = {auto: true, before: async () => {
  const cfg = SYNTH_CONFIG;
  await document.fonts.ready;
  await Promise.all([...document.images].map((i) => i.complete ? null : new Promise((r) => { i.onload = i.onerror = r; })));""" \
  + CONTRAST_JS + """
  if (cfg.hue || cfg.saturate !== 1) {
    const turn = (c) => { const m = (c || '').match(/[\\d.]+/g); if (!m || m.length < 3) return null;
      let [r, g, b] = m.slice(0, 3).map((v) => v / 255); const a = m[3] === undefined ? 1 : +m[3];
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b), l = (mx + mn) / 2; let h = 0, s = 0;
      if (mx !== mn) { const d = mx - mn; s = l > 0.5 ? d / (2 - mx - mn) : d / (mx + mn);
        h = mx === r ? (g - b) / d + (g < b ? 6 : 0) : mx === g ? (b - r) / d + 2 : (r - g) / d + 4; h *= 60; }
      if (s < 0.08) return null;
      h = (h + cfg.hue) % 360; s = Math.min(1, s * cfg.saturate);
      return `hsla(${h.toFixed(1)}, ${(s * 100).toFixed(1)}%, ${(l * 100).toFixed(1)}%, ${a})`; };
    for (const el of document.body.querySelectorAll('*')) {
      if (el.tagName === 'IMG') continue;
      const cs = getComputedStyle(el);
      for (const [prop, val] of [['color', cs.color], ['background-color', cs.backgroundColor],
          ['border-top-color', cs.borderTopColor], ['border-bottom-color', cs.borderBottomColor],
          ['border-left-color', cs.borderLeftColor], ['border-right-color', cs.borderRightColor]]) {
        const t = turn(val); if (t) el.style.setProperty(prop, t);
      }
    }
  }
  for (const el of document.body.querySelectorAll("*")) {
    if (el.classList.contains("page") || !el.textContent.trim()) continue;
    const cs = getComputedStyle(el);
    if (!/hidden|clip/.test(cs.overflowX + " " + cs.overflowY)) continue;
    const fits = () => el.scrollHeight <= el.clientHeight + 1 && el.scrollWidth <= el.clientWidth + 1;
    if (fits()) continue;
    el.style.setProperty("line-height", "1.2", "important");
    let size = parseFloat(cs.fontSize);
    for (const min = size * 0.8; !fits() && size > min;) {
      size *= 0.95;
      el.style.setProperty("font-size", size + "px", "important");
    }
    const spans = [...el.querySelectorAll("[data-slot]")];
    for (let i = spans.length - 1; i >= 0 && !fits(); i--) {
      const words = spans[i].textContent.split(/\\s+/).filter(Boolean);
      let lo = 0, hi = words.length;  // the most words that still fit
      while (lo < hi) {
        const mid = Math.ceil((lo + hi) / 2);
        spans[i].textContent = words.slice(0, mid).join(" ");
        if (fits()) lo = mid; else hi = mid - 1;
      }
      spans[i].textContent = words.slice(0, lo).join(" ");
    }
  }
  // a box whose text fits can still clip the tops and tails of letters (hamza, accents and descenders reach
  // past the line box, and DocLayNet boxes are tight): it lets their ink show; a box framing a picture keeps
  // clipping it
  for (const el of document.body.querySelectorAll("*")) {
    if (el.classList.contains("page") || el.classList.contains("pic-frame") || !el.textContent.trim()) continue;
    if (el.querySelector("img, .pic-frame, [data-img]")) continue;
    const cs = getComputedStyle(el);
    if (!/hidden|clip/.test(cs.overflowX + " " + cs.overflowY)) continue;
    if (el.scrollHeight <= el.clientHeight + 1 && el.scrollWidth <= el.clientWidth + 1)
      el.style.setProperty("overflow", "visible", "important");
  }
  for (const el of document.body.querySelectorAll("*")) {
    if (el.classList.contains("page") || !el.textContent.trim() || el.scrollWidth <= el.clientWidth + 1) continue;
    if (/hidden|clip/.test(getComputedStyle(el).overflowX) || !el.clientWidth) continue;
    el.style.setProperty("flex-wrap", "wrap", "important");
    for (const c of el.children) {
      c.style.setProperty("white-space", "normal", "important");
      c.style.setProperty("min-inline-size", "0", "important");
      c.style.setProperty("overflow-wrap", "anywhere", "important");
    }
  }
  for (const page of document.querySelectorAll(".page")) {
    const pr = page.getBoundingClientRect(), bottom = pr.bottom - parseFloat(getComputedStyle(page).paddingBottom);
    const out = (el) => [...el.getClientRects()].some((r) => r.bottom > bottom + 1 || r.right > pr.right + 1
                                                             || r.left < pr.left - 1);
    const spans = [...page.querySelectorAll("[data-slot]")];
    for (let i = spans.length - 1; i >= 0; i--) {
      const s = spans[i];
      if (!out(s)) continue;
      const words = s.textContent.split(/\\s+/).filter(Boolean);
      let lo = 0, hi = words.length;  // the most words that stay on the page
      while (lo < hi) {
        const mid = Math.ceil((lo + hi) / 2);
        s.textContent = words.slice(0, mid).join(" ");
        if (out(s)) hi = mid - 1; else lo = mid;
      }
      s.textContent = words.slice(0, lo).join(" ");
      const block = s.closest("p, li, h1, h2, h3, h4, h5, h6, blockquote, figcaption, dt, dd, caption, label");
      if (!lo && block && !block.textContent.trim()) block.style.display = "none";
    }
  }
  for (const s of document.querySelectorAll("[data-slot]")) {
    if (!s.textContent.trim()) continue;
    const {ratio: r, bg} = look(s);
    if (r >= 4.5) continue;
    s.style.setProperty("color", lum(bg || [255, 255, 255]) > 0.18 ? "#111111" : "#ffffff", "important");
  }
  const photos = cfg.photos; let k = 0;
  for (const el of document.body.querySelectorAll("*")) {
    if (k >= photos.length) break;
    if (/^(TD|TH|TR|TABLE|INPUT|SELECT|TEXTAREA|BUTTON|HR|BR|IMG|SVG|SCRIPT|STYLE)$/.test(el.tagName)) continue;
    if (el.textContent.trim() || el.querySelector("img") || el.dataset.img) continue;
    const r = el.getBoundingClientRect();
    if (r.width < 30 || r.height < 30) continue;
    const m = getComputedStyle(el).backgroundColor.match(/\\d+(\\.\\d+)?/g);
    if (!m || (m[3] !== undefined && +m[3] === 0)) continue;
    const [R, G, B] = m.map(Number), mean = (R + G + B) / 3;
    if (mean > 238) continue;  // a white or near-white box is page, not a picture placeholder
    el.style.setProperty("background", `url("${photos[k++]}") center / cover no-repeat`, "important");
    el.dataset.img = "grey-box";
  }
}};"""
# The opening tag of a grey picture box (old bank): any element whose class list has "ph"
PICTURE_BOX = re.compile(r'<(\w+)\b([^>]*\bclass="(?:[^"]*\s)?ph(?:\s[^"]*)?"[^>]*)>')
_cache = {}


def bank_entries(bank_dir):
    """The bank's index: [{"id", "source", "lang", ...}]."""
    key = ("index", str(bank_dir))
    if key not in _cache:
        path = bank_dir / "index.jsonl"
        entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not entries:
            raise RuntimeError(f"{path} is empty")
        _cache[key] = entries
    return _cache[key]


def arabic_fonts():
    if "fonts" not in _cache:
        _cache["fonts"] = font_families()
    return _cache["fonts"]


def text_pool(lang):
    if ("text", lang) not in _cache:
        _cache[("text", lang)] = TextPool(lang)
    return _cache[("text", lang)]


def pick(rng, weights):
    return rng.choices(list(weights), weights=list(weights.values()))[0]


class DocPipeline:
    """Template -> fill (style, theme, text, pictures) -> SynthDoc render + checks -> effects -> ground truth.

    bank: the template bank folder (default bank2000/).
    languages, effects, sources: {choice: weight}. In bank2000 every template is Arabic or English, so the
    language picks the template; sources are grammar, websight, websight2, doclaynet (equal by default).
    strengths: {"ocrsmith"|"augraphy"|"scan-simulator": 0..1} (0.5 = the tool's own settings).
    templates "bank" or "llm" (an LLMTemplates, passed as llm, writes them).
    pictures: an ImagePool (default img_pool/; False leaves picture boxes grey).
    text: "wikipedia" (text_pool/) or "demo". extras: also write the clean render and the drawn boxes.
    tasks: {"full_ocr"|"layout"|"block_ocr"|"table": weight}, the Surya OCR 2.5 prompt each sample is made for
    (pipeline/surya.py; default SURYA.TASKS)."""

    def __init__(self, out_dir, languages=None, effects=None, sources=None, templates="bank", llm=None,
                 pictures=None, text=None, extras=True, bank=None, strengths=None, check=True, tashkeel=None,
                 tasks=None):
        self.out = Path(out_dir).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.bank = Path(bank) if bank else BANK2000
        self.languages, self.effects = languages or LANGUAGES, effects or EFFECTS
        self.sources, self.templates, self.llm = sources, templates, llm
        self.strengths = {**STRENGTHS, **(strengths or {})}
        self.tashkeel = tashkeel or TASHKEEL.LEVELS  # Arabic pages: none / partial / full marks
        self.tasks = tasks or SURYA.TASKS
        self.pictures = ImagePool() if pictures is None else pictures
        self.categories = {f.name: c for c, files in getattr(self.pictures, "by_category", {}).items() for f in files}
        self.text = text or ("wikipedia" if TextPool.exists("ar") and TextPool.exists("en") else "demo")
        self.extras, self.check = extras, check
        synthdoc_arabic.install()  # SynthDoc then also measures every word (the Arabic extension)
        self._synthdoc = None

    # -- 1. template --------------------------------------------------------------------------------------

    def template(self, rng, name, lang, need_table=False):
        """(meta, template text or None, folder with its files, note). need_table: only bank templates with a
        table (the Surya "table" task)."""
        note = ""
        if self.templates == "llm":
            if self.llm is not None and self.llm.available():
                from src.bank2000 import adapt  # same form as the bank: language, <img> picture slots
                try:
                    made = self.llm.template(rng, f"llm_{name}")
                except Exception as e:  # e.g. still rate-limited after waiting: reuse an earlier LLM template
                    made, note = None, f"LLM call failed ({type(e).__name__}: {e})"
                    made = self.llm.saved(rng)
                    if made and (made := adapt(*made, lang)):
                        return (*made, self.out, note + "; an earlier LLM template reused")
                if made and (made := adapt(*made, lang)):
                    return (*made, self.out, "")
                note = (note or "LLM page had too little text") + "; bank template used"
            else:
                note = f"{self.llm.problem if self.llm else 'no LLM'}; bank template used"
        entries = bank_entries(self.bank)
        if need_table:
            entries = [e for e in entries if (e.get("features") or {}).get("tables")] or entries
        entries = [e for e in entries if e.get("lang", lang) == lang] or entries
        present = sorted({e["source"] for e in entries})
        weights = {s: w for s, w in (self.sources or {s: 1 for s in present}).items() if s in present}
        source = pick(rng, weights or {s: 1 for s in present})
        tid = rng.choice([e["id"] for e in entries if e["source"] == source])
        folder = self.bank / "templates" / tid
        meta = json.loads((folder / "template.json").read_text(encoding="utf-8"))
        return meta, None, folder, note

    # -- 2. fill --------------------------------------------------------------------------------------------

    def fill(self, meta, text, folder, rng, lang):
        """(page for SynthDoc, values for its slots, pictures and style, record of everything picked)."""
        if lang == "ar":
            s, css = style(rng, arabic_fonts(), transparent=False)
        else:
            s, css = style_latin(rng, transparent=False)
        look, theme_css, theme_js = theme(rng, meta["source"])
        words = text_pool(lang) if self.text == "wikipedia" else None
        if words:
            words.begin(rng)  # one article per page
        values = {slot["id"]: Markup(fill(slot, rng, s["digits"], lang, words)) for slot in meta["slots"]}
        level = pick(rng, self.tashkeel) if lang == "ar" else "none"  # one level per page
        if level != "none":
            values = {k: Markup(TASHKEEL.apply(str(v), level, rng)) for k, v in values.items()}
            # marks above and below the letters need room between lines, as in printed marked text
            s = {**s, "line_height": max(s["line_height"], TASHKEEL.LINE_HEIGHT[level])}
            css += f"\n:root {{ --line-height: {s['line_height']}; }}"
        used = []
        for pic in meta.get("pictures", []):  # <img src="{{ pic_N }}"> slots: a photo of the hinted kind
            photo = self.pictures.pick(rng, pic.get("hint")) if self.pictures else None
            values[pic["id"]] = Markup(photo.as_uri() if photo else "")
            if photo:
                used.append(photo.name)
        page, more = self.fill_pictures(synthdoc_html(meta, text, lang, folder), rng, theme_js)
        s = {**s, "theme": look, "tashkeel": level, "pictures": used + more, "text": self.text,
             "text_sources": words.sources() if words else []}
        # The style travels with the page, so SynthDoc's word labels record it (the Arabic extension reads this)
        variant = {"template": meta["id"], "source": meta["source"], "lang": lang, **s}
        tag = Markup.escape(json.dumps(variant, ensure_ascii=False))
        page = page.replace("</head>", f'<meta name="synthdoc-ar-variant" content="{tag}">\n</head>', 1)
        return page, {**values, "style": Markup(css + theme_css)}, s

    def fill_pictures(self, page, rng, theme_js=None):
        """Grey picture boxes of the old bank get a photo as background; the page script is added."""
        used = []
        if self.pictures:
            def put(m):
                photo = self.pictures.pick(rng, re.search(r'data-hint="([^"]*)"', m.group(2)) and
                                        re.search(r'data-hint="([^"]*)"', m.group(2)).group(1))
                used.append(photo.name)
                bg = f"background: url({photo.as_uri()}) center / cover no-repeat !important;"
                attrs = m.group(2)
                attrs = attrs.replace(' style="', f' style="{bg} ', 1) if ' style="' in attrs else f'{attrs} style="{bg}"'
                return f'<{m.group(1)}{attrs} data-img="img_pool/{photo.name}">'
            page = PICTURE_BOX.sub(put, page)
        config = {"photos": [self.pictures.pick(rng).as_uri() for _ in range(12)] if self.pictures else [],
                  "hue": 0, "saturate": 1.0, **(theme_js or {})}
        script = PAGE_JS.replace("SYNTH_CONFIG", json.dumps(config))
        return page.replace("</head>", f"<script>{script}</script>\n</head>", 1), used

    # -- 3. render --------------------------------------------------------------------------------------------

    def render(self, page, values, name):
        """SynthDoc renders page 1 and measures it: (image, words and lines, slot boxes, pages, qa)."""
        if self._synthdoc is None:
            self._synthdoc = NERDataGenerator(output_dir=str(self.out / "_synthdoc"), create_overlays=False,
                                              output_formats=["json"])
        result = self._synthdoc.generate_dataset_sample(template_html=page, template_data=values, output_id=name)
        first = result["pages"][0] if "pages" in result else result
        raw_name = f"{name}_page_1" if "pages" in result else name
        raw = json.loads((synthdoc_arabic._raw_dir / f"{raw_name}.json").read_text(encoding="utf-8"))
        ocr = build_page_gt(raw["page"], {"language": raw["lang"], "direction": raw["dir"], "variant": raw["variant"]})
        for word, measured in zip(ocr["words"], raw["page"]["words"]):  # same order: lines are runs of words
            word["cut_x"], word["cut_y"] = measured.get("cut_x", False), measured.get("cut_y", 0)
            word["slot"] = measured.get("slot")
            top, bottom = measured.get("glyph_y") or (word["y"], word["y"] + word["height"])
            word["glyph_box"] = [word["x"], top, word["width"], max(1.0, bottom - top)]
        slots = json.loads(Path(first["label_path"]).read_text(encoding="utf-8"))["annotations"]
        qa = raw["page"].get("qa") or {}
        image = Image.open(first["image_path"]).convert("RGB")
        for page_result in result.get("pages", [result]):  # SynthDoc's own copies are no longer needed
            for key in ("image_path", "label_path"):
                Path(page_result[key]).unlink(missing_ok=True)
        (synthdoc_arabic._raw_dir / f"{raw_name}.json").unlink(missing_ok=True)
        scale_measurements(ocr, slots, qa, image.width / max(1, ocr["width"]))
        return image, ocr, slots, result.get("total_pages", 1), qa

    # -- 4. effects -------------------------------------------------------------------------------------------

    def degrade(self, clean, words, rng, effect, points=()):
        """The effect at the configured strength, checked word by word: up to EFFECT_TRIES seeds, then 60% and
        30% of the strength; a page that never reads well stays clean.
        points: (x, y) on the clean page (corners of tables, lists, pictures) carried through the effect as tiny
        boxes after the words (painted last, so they never split a word's box); not part of the checks.
        Returns (image, boxes, attacks, strength used, legibility check, the points moved, None if lost)."""
        exact = [[round(w["x"]), round(w["y"]), round(w["width"]), round(w["height"])] for w in words]
        same = [tuple(pt) for pt in points]
        if effect == "clean":
            return clean, exact, ["clean"], 0.0, {"lost": 0, "faint": 0.0, "sharpness": 1.0}, same
        # each distinct point once (a row's bottom corners are the next row's top ones: in a mask the second dot
        # would paint over the first)
        unique = list(dict.fromkeys((round(x), round(y)) for x, y in same))
        where = {pt: i for i, pt in enumerate(unique)}
        dots = [{"x": x - 1, "y": y - 1, "width": 2, "height": 2, "line_id": f"pt{i}", "text": ""}
                for i, (x, y) in enumerate(unique)]
        tool = "augraphy" if effect == "augraphy" else ("ocrsmith" if effect == "ocrsmith" else "scan-simulator")
        base = self.strengths[tool]
        n = len(words)
        for level in (1.0, 0.6, 0.3):
            for _ in range(EFFECT_TRIES):
                strength = round(base * level, 3)
                out, boxes, attacks = D.degrade(clean, words + dots, rng.randrange(2**31), effect, strength)
                boxes, dot = boxes[:n], [(b[0] + b[2] / 2, b[1] + b[3] / 2) if b else None for b in boxes[n:]]
                moved = [dot[where[(round(x), round(y))]] for x, y in same]
                check = D.legibility(clean, out, words, boxes)
                if D.readable(check):
                    return out, boxes, attacks, strength, check, moved
        return (clean, exact, ["clean (effect left text unreadable)"], 0.0, {"lost": 0, "faint": 0.0, "sharpness": 1.0},
                same)

    # -- 5. outputs -------------------------------------------------------------------------------------------

    # -- one sample, many samples ---------------------------------------------------------------------------

    def sample(self, seed, name=None, language=None, effect=None, task=None):
        """Run the steps once; returns a summary dict (files are written to out_dir)."""
        name = name or f"s{seed}"
        rng = random.Random(seed)
        lang, effect = language or pick(rng, self.languages), effect or pick(rng, self.effects)
        task = task or pick(rng, self.tasks)  # the Surya prompt this sample is made for
        t = {"template": 0, "fill": 0, "render": 0}
        rejected, attempts = {}, 0
        for attempts in range(1, ATTEMPTS + 1):  # until the rendered page passes every check
            if attempts % 2 == 1:  # a new template every second attempt
                mark = time.perf_counter()
                meta, text, folder, note = self.template(rng, name, lang, need_table=task == "table")
                lang = meta.get("lang", lang)
                t["template"] += time.perf_counter() - mark
            mark = time.perf_counter()
            page, values, s = self.fill(meta, text, folder, rng, lang)
            t["fill"] += time.perf_counter() - mark
            mark = time.perf_counter()
            clean, ocr, slots, pages, qa = self.render(page, values, name)
            t["render"] += time.perf_counter() - mark
            issues = ({**render_issues(ocr, qa), **ink_issues(clean, ocr["words"])} if self.check
                      else ({} if ocr["words"] else {"empty": 1}))
            if task == "table":  # the table task needs a table with text on the page (a long page may lose it)
                shown = {w.get("slot") for w in ocr["words"]}
                if not any(sl in shown for tb in (qa.get("struct") or {}).get("tables", [])
                           for row in tb["rows"] for cell in row["cells"] for sl in cell["slots"]):
                    issues["no table"] = 1
            if not issues:
                break
            for k, v in issues.items():
                rejected[k] = rejected.get(k, 0) + v
        else:
            raise RuntimeError(f"no page without hidden, cut or overlapping text after {ATTEMPTS} attempts: {rejected}")
        words, cut = split_words(ocr)  # (the checks passed, so nothing is dropped here)
        for w in words:  # boxes hug the letters, not the line's leading: big text doesn't reach into its neighbours
            top, h = w["glyph_box"][1], w["glyph_box"][3]
            pad = max(2.0, 0.12 * h)
            y0, y1 = max(w["y"], top - pad), min(w["y"] + w["height"], top + h + pad)
            if y1 - y0 >= 0.5 * h:
                w["y"], w["height"] = y0, y1 - y0
        mark = time.perf_counter()
        plan = SURYA.carry(qa.get("struct") or {})  # tables, rows, columns, lists, pictures: moved with the page
        image, boxes, attacks, strength, check, moved = self.degrade(clean, words, rng, effect, SURYA.corners(plan))
        label = build_label(name, image.size, None, attacks, contrast(np.asarray(image.convert("L")), [b for b in boxes if b]),
                            ocr, slots, words, boxes, cut)
        regions = GT.regions(meta["slots"], words, boxes, lang, meta.get("positioning", "flow"))
        label.update({"slots": regions, "template": meta["id"], "source": meta["source"], "lang": lang,
                      "effect": effect, "strength": strength, "legibility": check, "style": s,
                      "pages_rendered": pages, "attempts": attempts, "rejected": rejected,
                      "render_scale": RENDER_SCALE})
        t["effects"] = time.perf_counter() - mark
        mark = time.perf_counter()
        # The HTML says exactly what is on the page: each slot holds its labelled words
        shown = {sl["id"]: "" for sl in meta["slots"]}
        shown.update({r["id"]: r["text"] for r in regions})
        filled = Template(page).render(**{k: Markup.escape(v) for k, v in shown.items()},
                                       **{p["id"]: values[p["id"]] for p in meta.get("pictures", [])},
                                       style=values["style"])
        md = GT.to_text(regions)  # the Markdown is the ground truth: one <text>... <bbox>...</bbox></text> per block
        # Surya: the page's blocks, then the input image and target of this sample's prompt
        width, height = image.size
        rects = SURYA.rebuild(plan, moved, width, height)
        blks = SURYA.blocks(regions, {sl["id"]: sl["role"] for sl in meta["slots"]}, qa.get("struct") or {}, rects,
                            lang, meta.get("positioning", "flow"), self.categories, width, height)
        # a cut-out (block_ocr: one block, not a picture; table: a table) shows its block and nothing of another:
        # blocks whose cut-out would take in part of a neighbour (a turned page's upright boxes overlap) are not
        # used; none left: full_ocr instead
        cuts = []
        if task in ("block_ocr", "table"):
            for b in blks:
                if b["label"] == "Image" or (task == "table" and b["label"] != "Table"):
                    continue
                others = [x["box"] for x in blks if x is not b]
                c = SURYA.crop_box(b["box"], width, height, rng, others,
                                   share=(0.03, 0.12) if task == "table" else None)
                if SURYA.clear(c, others):
                    cuts.append((b, c))
            if not cuts:
                note, task = (note + "; " if note else "") + f"no {task} block that cuts out clean: full_ocr instead", "full_ocr"
        crop, target, structure = None, None, None
        if task == "full_ocr":
            target = SURYA.full_ocr(blks, width, height)
        elif task == "layout":
            target = SURYA.layout(blks, width, height)
        elif task == "block_ocr":  # one block cut out (6-24 px margin), its compact content as the answer
            b, crop = rng.choice(cuts)
            target = SURYA.block_ocr(b)
        else:  # table: a table cut out with a 3-12% margin, its rows then columns
            b, crop = rng.choice(cuts)
            target = SURYA.table(rects, b["table"], crop)
            structure = SURYA.table_structure(rects, b["table"], crop)
        label["surya"] = {"task": task, "prompt": SURYA.PROMPTS[task], "target": target, "crop": crop,
                          "table_structure": structure,
                          "page_size": [width, height], "divs": SURYA.divs(blks, width, height),
                          "blocks": [{"label": b["label"], "box": [round(v) for v in b["box"]], "html": b["html"]}
                                     for b in blks]}
        t["markdown"] = time.perf_counter() - mark

        out = self.out
        (image.crop(crop) if crop else image).save(out / f"{name}.jpg", quality=90)  # what the model is shown
        if crop:
            image.save(out / f"{name}_page.jpg", quality=90)  # the whole page (the .md boxes are on it)
        (out / f"{name}.prompt.txt").write_text(SURYA.PROMPTS[task], encoding="utf-8", newline="\n")
        (out / f"{name}{SURYA.SUFFIX[task]}").write_text(target, encoding="utf-8", newline="\n")
        (out / f"{name}.html").write_text(filled, encoding="utf-8")
        (out / f"{name}.md").write_text(md, encoding="utf-8", newline="\n")  # \n line ends on every system
        (out / f"{name}.json").write_text(json.dumps(label, ensure_ascii=False), encoding="utf-8")
        if self.extras:
            clean.save(out / f"{name}_clean.jpg", quality=88)
            draw(image, [w["box"] for w in label["words"]], (40, 170, 60), out / f"{name}_words.jpg")
            draw(image, [r["box"] for r in regions], (210, 50, 50), out / f"{name}_slots.jpg")
            draw(image, [[b["box"][0], b["box"][1], b["box"][2] - b["box"][0], b["box"][3] - b["box"][1]] for b in blks],
                 (40, 90, 220), out / f"{name}_blocks.jpg")
        times = {k: round(t[k], 2) for k in ("template", "fill", "render", "effects", "markdown")}
        return {"name": name, "seed": seed, "template": meta["id"], "source": meta["source"], "lang": lang,
                "task": task, "target_file": f"{name}{SURYA.SUFFIX[task]}", "blocks": len(blks), "crop": crop,
                "effect": effect, "strength": strength, "attacks": attacks, "font": s["font"], "size_pt": s["size_pt"],
                "theme": s["theme"], "tashkeel": s["tashkeel"], "contrast": label["contrast"], "legibility": check, "words": len(label["words"]),
                "regions": len(regions), "lost_words": label["lost_words"], "pages_rendered": pages, "note": note,
                "attempts": attempts, "rejected": rejected, "pictures": len(s["pictures"]), "text": s["text"],
                "text_sources": len(s["text_sources"]), "seconds": {**times, "total": round(sum(times.values()), 2)}}

    def run(self, n=16, seed=None, on_sample=None):
        """n samples one after another; on_sample(summary) is called as each one is done."""
        seed = random.randrange(10**6) if seed is None else seed
        done = []
        for i in range(n):
            done.append(self.sample(seed * 1000 + i, f"s{i + 1:02d}"))
            if on_sample:
                on_sample(done[-1])
        return done


def scale_measurements(ocr, slots, qa, k):
    """Measurements are in CSS pixels; the image has k image pixels per CSS pixel (RENDER_SCALE)."""
    if abs(k - 1) < 0.01:
        return
    for item in ocr["words"] + ocr["lines"]:
        for key in ("x", "y", "width", "height"):
            item[key] *= k
        for key in ("ink_box", "glyph_box"):
            if key in item:
                item[key] = [v * k for v in item[key]]
        if "font_size_px" in item:
            item["font_size_px"] *= k
    ocr["width"], ocr["height"] = round(ocr["width"] * k), round(ocr["height"] * k)
    for a in slots:
        for key in ("x", "y", "width", "height"):
            a[key] *= k
    for p in qa.get("pictures", []):
        for key in ("x", "y", "width", "height"):
            p[key] *= k
    st = qa.get("struct") or {}
    rects = ([t["rect"] for t in st.get("tables", [])] + [r["rect"] for t in st.get("tables", []) for r in t["rows"]]
             + [c["rect"] for t in st.get("tables", []) for r in t["rows"] for c in r["cells"]]
             + [x["rect"] for x in st.get("lists", []) + st.get("images", [])])
    for rect in rects:
        for key in ("x", "y", "width", "height"):
            rect[key] *= k
    for g in ocr.get("untracked_generated_text", []):
        for key in ("x", "y", "width", "height"):
            if key in g:
                g[key] *= k


def warm_up():
    """Augraphy compiles its noise code (numba) on first use, about 15 s: do it before the first real page."""
    attack.run_augraphy(np.full((200, 200, 3), 255, np.uint8), 0)


def draw(image, boxes, color, path):
    out = image.copy()
    d = ImageDraw.Draw(out)
    for x, y, w, h in boxes:
        d.rectangle([x, y, x + w, y + h], outline=color, width=2)
    out.save(path, quality=82)
