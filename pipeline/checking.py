"""Checks and fixes: everything around the steps in generator.py that makes the labels match the image.

  * fixes to the originals (originals/ stays unchanged; these are applied in memory when this file is
    imported): SynthDoc renders at 2x, keeps one browser open, and takes the screenshot as soon as
    Paged.js is done; scan-simulator's big blurs run faster; OCRSmith's page curl moves boxes right;
  * what is measured on the rendered page (cut letters, each word's slot, tables, lists, pictures);
  * PAGE_JS, run in the page before layout: text fitted to its box, contrast fixed, photos in empty boxes;
  * checks: render_issues() and ink_issues() on the rendered page (hidden, cut, overlapping, covered,
    invisible text: the page is made again), legibility() after the effect (lost, faint, blurred or
    blotted words: the effect is tried again, lighter), split_words() (words with cut letters left out),
    to_ink() (a list's or table's box is what is drawn in it), problems() (is an answer in Surya's format).
"""
import atexit
import json
import random
import re
from collections import Counter

import cv2
import numpy as np
from ocrsmith.core.degradations import physical as _ocr_physical
from ocrsmith.core.degradations.base import map_page as _ocr_map_page
from playwright.sync_api import Browser, sync_playwright
from scan_simulator.transforms import damage as _scan_damage, paper as _scan_paper, scanner as _scan_scanner
from scipy import fft as _fft
from scipy.ndimage import gaussian_filter as _gaussian_filter
from surya.inference.prompts import LAYOUT_LABEL_SET
from synthdoc import pagedjs_renderer
from synthdoc_arabic import pipeline as synthdoc_arabic


# == SynthDoc: render at 2x, one browser, screenshot when Paged.js is done; what the page measures ===================

# -- rendering at a higher resolution -----------------------------------------------------------------------
# SynthDoc's browser page has no resolution setting: a page is screenshot at 1 pixel per CSS pixel (96 dpi;
# 9 pt text is 12 px tall), too small for OCR training. Pages are opened here with a device scale factor of
# RENDER_SCALE; the measured boxes are in CSS pixels and are scaled to match (scale_measurements).
RENDER_SCALE = 2
_new_page = Browser.new_page


def _scaled_new_page(self, *args, **kwargs):
    kwargs.setdefault("device_scale_factor", RENDER_SCALE)
    return _new_page(self, *args, **kwargs)


Browser.new_page = _scaled_new_page

# -- one browser for every page -----------------------------------------------------------------------------
_kept = {}  # "playwright", "browser": started on the first page, stopped when the process ends


class _OneBrowser:
    """Stands in for sync_playwright() in SynthDoc's renderer: `with ... as p: p.chromium.launch()` gives the
    process's one browser, and closing it closes only the pages opened since (with their contexts)."""

    def __enter__(self):
        if "playwright" not in _kept:
            _kept["playwright"] = sync_playwright().start()
            atexit.register(_stop)
        self.pages = []
        return self

    @property
    def chromium(self):
        return self

    def launch(self, **kwargs):
        if "browser" not in _kept or not _kept["browser"].is_connected():
            _kept["browser"] = _kept["playwright"].chromium.launch(**kwargs)
        return self

    def new_page(self, **kwargs):
        page = _kept["browser"].new_page(**kwargs)
        self.pages.append(page)
        return page

    def close(self):
        for page in self.pages:
            if not page.is_closed():
                page.context.close()
        self.pages = []

    def __exit__(self, *error):
        self.close()
        return False


def _stop():
    for key in ("browser", "playwright"):
        try:
            thing = _kept.pop(key, None)
            if thing:
                thing.close() if key == "browser" else thing.stop()
        except Exception:
            pass


pagedjs_renderer.sync_playwright = _OneBrowser

# -- the screenshot as soon as Paged.js is done -------------------------------------------------------------
# Paged.js calls PagedConfig.after once every page is laid out; a line added before its script (only in the
# copy SynthDoc sends to the browser) records that.
PAGED_DONE = "<script>if (window.PagedConfig) window.PagedConfig.after = () => { window.__pagedDone = true; };</script>\n"
_inject = pagedjs_renderer.PagedJSRenderer.inject_pagedjs_script


def _inject_with_done(self, html_content):
    html_content = _inject(self, html_content)
    return re.sub(r'<script src="[^"]*paged[^"]*"></script>', lambda m: PAGED_DONE + m.group(0), html_content, count=1)


def _wait_for_pagination(self, page):
    """The number of pages, once Paged.js is done (at most 500 ms after its pages appear, as SynthDoc waited)."""
    page.wait_for_selector(".pagedjs_pages", timeout=self.timeout)
    try:
        page.wait_for_function("window.__pagedDone === true", timeout=500)
    except Exception:  # not done after 500 ms: what SynthDoc would have taken
        pass
    return page.evaluate('document.querySelectorAll(".pagedjs_page").length')


pagedjs_renderer.PagedJSRenderer.inject_pagedjs_script = _inject_with_done
pagedjs_renderer.PagedJSRenderer.wait_for_pagination = _wait_for_pagination

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
  // the page's structure, for block-level labels (STEP 5): tables with their rows and cells, lists
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


# == PAGE_JS: fixes inside the page before Paged.js lays it out ======================================================

# Runs in Paged.js's "before" hook, once the page's fonts and pictures have loaded and before Paged.js lays the
# page out (window.PagedConfig must exist before Paged.js loads: this goes at the end of <head>, before
# SynthDoc's Paged.js script):
#  1. colours: a web design's colours are turned around the colour wheel (STEP 2, themes: hue only, so contrast
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


# == checks of the rendered page =====================================================================================

# Quality checks on a rendered page, before it can become a training sample.
# 
# render_issues() looks at what the browser measured after SynthDoc laid the page out (STEP 3
# adds the measurements to the Arabic extension's word script) and counts, per kind of problem:
# 
#   hidden      words that are in a slot's text but were not measured on the page (fully clipped, outside every
#               visible area, display:none): text the ground truth would claim but the image doesn't show
#   cut         words partly hidden: cut on the left or right by a box edge, or more than 3% of their letters'
#               height hidden above or below
#   mismeasured words whose box is much wider than their letters (punctuation at both ends of a word in
#               right-to-left text splits it into direction runs, and its box spans the line)
#   overlap     letters of two different slots on top of each other (more than 20% of the smaller glyph box:
#               the word's width by its letters' measured top and bottom)
#   picture     a word more than 15% under a picture (an <img>, a picture box or a filled grey box)
#   offpage     letters outside the page
#   generated   text drawn by CSS (numbered list markers, ::before/::after content): on the image but in no slot
#   contrast    a slot whose text has less than 3:1 contrast with what is behind it
#   empty       no text on the page at all
# 
# DocPipeline makes the page again (new style and text, then another template) until every count is 0, so a
# sample's ground truth always matches what is visible.

BULLETS = {"disc", "circle", "square", "none", "disclosure-closed", "disclosure-open"}


def _area(b):
    return max(0.0, b[2]) * max(0.0, b[3])


def _inter(a, b):
    ix = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    iy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0.0, ix) * max(0.0, iy)


INK_MIN = 0.01  # share of a word's letter box that must differ from its background


def ink_issues(image, words):
    """{"invisible": n}: words measured as shown whose letter box has (almost) no ink in the rendered image.
    Whatever hides them (Paged.js moving a line to the next page, a clip the measuring can't see), the
    pixels say whether they are there. Words of punctuation only are skipped (too little ink to judge)."""
    gray = np.asarray(image.convert("L"), np.int16)
    n = 0
    for w in words:
        if not re.search(r"[^\W_]", w["text"]):
            continue
        x, y, bw, bh = (int(round(v)) for v in (w.get("glyph_box") or w["ink_box"]))
        crop = gray[max(0, y):max(0, y + bh), max(0, x):max(0, x + bw)]
        if crop.size < 20:
            continue
        if (np.abs(crop - np.median(crop)) > 40).mean() < INK_MIN:
            n += 1
    return {"invisible": n} if n else {}


def render_issues(ocr, qa):
    """{issue: count} for a rendered page (empty dict: the page is fine)."""
    words = ocr["words"]
    issues = Counter()
    if not words:
        issues["empty"] += 1
        return dict(issues)
    measured = Counter(w.get("slot") for w in words)
    for slot, text in (qa or {}).get("slots", {}).items():
        missing = len(text.split()) - measured.get(slot, 0)
        if missing > 0:
            issues["hidden"] += missing
    pw, ph = ocr["width"], ocr["height"]
    for w in words:
        if w.get("cut_x") or w.get("cut_y", 0) > 0.03:
            issues["cut"] += 1
        if w["width"] > max(3, len(w["text"])) * w["font_size_px"] * 1.0:
            issues["mismeasured"] += 1
        x, y, bw, bh = w.get("glyph_box") or w["ink_box"]
        if x < -1 or y < -1 or x + bw > pw + 1 or y + bh > ph + 1:
            issues["offpage"] += 1
    # letters of two slots on top of each other (glyph boxes: the letters' own top and bottom, not the font's
    # taller line slot; sorted by top edge, so only nearby words are compared)
    ink = sorted(((w.get("glyph_box") or w["ink_box"], w.get("slot")) for w in words), key=lambda t: t[0][1])
    for i, (a, sa) in enumerate(ink):
        for b, sb in ink[i + 1:]:
            if b[1] > a[1] + a[3]:
                break
            if sa != sb and _inter(a, b) > 0.2 * min(_area(a), _area(b)):
                issues["overlap"] += 1
    for p in (qa or {}).get("pictures", []):
        rect = (p["x"], p["y"], p["width"], p["height"])
        issues["picture"] += sum(_inter((w["x"], w["y"], w["width"], w["height"]), rect)
                                 > 0.15 * _area((w["x"], w["y"], w["width"], w["height"])) for w in words)
    for g in ocr.get("untracked_generated_text", []):
        if g["kind"] == "list-marker":
            if g["content"] not in BULLETS:
                issues["generated"] += 1
        elif re.search(r"[^\W_]", g["content"].strip("\"'")):
            issues["generated"] += 1
    issues["contrast"] += sum(1 for c in (qa or {}).get("contrast", []) if c["ratio"] < 3.0)
    issues["covered"] += sum(c["points"] for c in (qa or {}).get("covered", []))
    return {k: v for k, v in issues.items() if v}




def whole_letters(w):
    """Keep a word unless its letters are cut. The OCR labels flag a word as clipped when its line box pokes
    out of an overflow:hidden box by half a pixel, which is every word in a tight DocLayNet box."""
    return not w["clipped"] or w["height"] >= 0.8 * w["font_size_px"]


def split_words(ocr):
    """(words whose letters are whole, number of cut words dropped).

    Besides words cut by a box edge, this drops words cut by the page edge: a table too wide for its column
    (English words are longer than the Arabic ones a template was laid out for) runs off the page. Their boxes
    are already trimmed to what is visible, so a clipped word that ends at the page edge counts as cut."""
    pw, ph = ocr["width"] - 0.5, ocr["height"] - 0.5

    def at_edge(w):
        return w["x"] <= 0.5 or w["y"] <= 0.5 or w["x"] + w["width"] >= pw or w["y"] + w["height"] >= ph

    def measured_wide(w):
        """A box far wider than the word's letters: punctuation at both ends of a word in right-to-left text
        puts it in several direction runs, and its box then spans the whole line, over its neighbours."""
        return w["width"] > max(3, len(w["text"])) * w["font_size_px"] * 1.0
    # cut_x and cut_y (set by the pipeline's word measuring): a word cut on its left or right by any box, or
    # with more than a tenth of its height hidden above or below
    words = [w for w in ocr["words"] if whole_letters(w) and not w.get("cut_x") and w.get("cut_y", 0) <= 0.1
             and not (w["clipped"] and at_edge(w)) and not measured_wide(w)]
    return words, len(ocr["words"]) - len(words)


# == effects: faster blurs (scan-simulator), boxes on curled pages (OCRSmith) ========================================

# scan-simulator blurs whole pages with scipy's gaussian_filter at radii up to a fifth of the page (paper colour,
# ink fading, wear patches). On our 2x pages that costs 3-30 s per transform. The same blur, faster and with
# the same result (within 0.05 of a grey level): large radii through an FFT (its cost doesn't grow with the
# radius), and a mostly empty mask (a scuff, a stain) blurred only around what is in it. Replaced in memory, for
# scan-simulator's transforms only; its folder is not changed.
def _fft_blur(padded, sigma, pad):
    """Gaussian blur of an array already padded by pad on every side; the unpadded middle."""
    h, w = padded.shape
    fy, fx = _fft.fftfreq(h)[:, None], _fft.rfftfreq(w)[None, :]
    kernel = np.exp(-2 * (np.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))
    out = _fft.irfft2(_fft.rfft2(padded, workers=-1) * kernel, s=padded.shape, workers=-1)
    return out[pad:h - pad, pad:w - pad]


def fast_gaussian_filter(input, sigma, order=0, output=None, mode="reflect", cval=0.0, truncate=4.0, **kwargs):
    a = np.asarray(input)
    if (order or output is not None or mode != "reflect" or truncate != 4.0 or kwargs or a.ndim != 2
            or np.ndim(sigma) or not np.issubdtype(a.dtype, np.floating) or sigma < 16):
        return _gaussian_filter(input, sigma, order=order, output=output, mode=mode, cval=cval, truncate=truncate,
                                **kwargs)
    r = int(truncate * sigma + 0.5)  # scipy's kernel radius
    rows, cols = np.flatnonzero(a.any(axis=1)), np.flatnonzero(a.any(axis=0))
    if not rows.size:
        return np.zeros_like(a)
    h, w = a.shape
    y0, y1, x0, x1 = max(0, rows[0] - r), min(h, rows[-1] + 1 + r), max(0, cols[0] - r), min(w, cols[-1] + 1 + r)
    # beyond the window the input is zero up to the kernel's reach: zeros there, scipy's mirror at the image's edges
    win = a[y0:y1, x0:x1]
    win = np.pad(win, ((r if y0 == 0 else 0, r if y1 == h else 0), (r if x0 == 0 else 0, r if x1 == w else 0)), mode="symmetric")
    win = np.pad(win, ((0 if y0 == 0 else r, 0 if y1 == h else r), (0 if x0 == 0 else r, 0 if x1 == w else r)))
    out = np.zeros_like(a)
    out[y0:y1, x0:x1] = _fft_blur(win, sigma, r)
    return out


for _module in (_scan_damage, _scan_paper, _scan_scanner):
    _module.gaussian_filter = fast_gaussian_filter


# OCRSmith's page curl and wrinkles move the pixels by a field d: the new pixel q shows the old pixel q + d(q).
# Its boxes move each corner p to p - d(p), the field read at the old place, but the corner's new place q is
# where q + d(q) = p. Where the field changes fast (near a curled edge) the two differ by up to 40 px: boxes
# slid off their words. Here the corner's new place is found by repeating q = p - d(q) (the field changes by
# well under a pixel per pixel, so this converges in a few rounds). The pixels are unchanged. Replaced in
# memory; OCRSmith's folder is not changed.
def _displace(self, image, page, rng):
    width, height = image.size
    dx, dy = self._field(width, height, rng)[:2]
    dx, dy = np.broadcast_to(dx, (height, width)), np.broadcast_to(dy, (height, width))

    def mapper(x, y):
        qx, qy = x, y
        for _ in range(10):
            col, row = min(max(int(round(qx)), 0), width - 1), min(max(int(round(qy)), 0), height - 1)
            qx, qy = x - float(dx[row, col]), y - float(dy[row, col])
        return qx, qy
    return _ocr_physical._remap(image, dx, dy, self.fill), _ocr_map_page(page, mapper, width, height), self._field_params


_ocr_physical._DisplacementDegradation.apply = _displace


# == is the text still readable after the effect? ====================================================================

def _crop(gray, box):
    x, y, w, h = (int(round(v)) for v in box)
    return gray[max(0, y):max(0, y) + h, max(0, x):max(0, x) + w]


def separation(c0, c1):
    """How clearly a degraded word's letters stand out from what is around them, or None (no letters).

    The clean crop says which share of the word's pixels are letters and whether they are darker or lighter
    than the background (Otsu threshold). In the degraded crop that share of its darkest (or lightest)
    pixels counts as letters, the rest as background; the score is the distance between their means over
    the sum of their spreads. Noise that keeps contrast high but drowns the letters makes it small. It needs
    no alignment, so it works after rotation and warps too."""
    t, _ = cv2.threshold(c0.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = float((c0 < t).mean())
    letters_dark = dark <= 0.5
    share = dark if letters_dark else 1 - dark
    if share < 0.02:
        return None
    v = np.sort(c1.ravel())
    k = max(1, int(share * v.size))
    letters, ground = (v[:k], v[k:]) if letters_dark else (v[-k:], v[:-k])
    if not ground.size:
        return None
    return abs(float(ground.mean()) - float(letters.mean())) / (float(letters.std()) + float(ground.std()) + 1.0)


FAINT_GAP = 40  # grey levels between a word's letters and its background below which it reads as washed out


def gap(c):
    """Grey levels between the two sides of a crop split by Otsu: its letters and its background. Noise keeps
    the crop's range wide when lighting washes text out, but not this gap (washed-out words ~25, others 100+)."""
    t, _ = cv2.threshold(c.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low, high = c[c < t], c[c >= t]
    return abs(float(high.mean()) - float(low.mean())) if low.size and high.size else 0.0


def dark_share(c):
    """Share of a crop darker than halfway between its paper (90th percentile) and its ink (2nd)."""
    paper, ink = np.percentile(c, 90), np.percentile(c, 2)
    return float((c < (paper + ink) / 2).mean()) if paper - ink >= 25 else 0.0


def covered(c0, c1):
    """How much more of a word's box is dark after the effect than before: a streak or blot over the word
    darkens far more of the box than its letters (blur and ink bleed only spread them a little)."""
    return dark_share(c1) - dark_share(c0)


def legibility(clean, out, words, boxes, sample=250):
    """Per-word checks after an effect: {"lost": n, "faint": share, "sharpness": median ratio,
    "separation": median, "drowned": share of words whose letters barely stand out, "edge": words cut by the
    frame (a photo preset that crops the page leaves part of a word outside the picture)}."""
    lost = sum(b is None for b in boxes)
    width, height = out.size
    scale = width * height / (clean.width * clean.height)

    def at_border(b, w, h):
        return b[0] <= 1 or b[1] <= 1 or b[0] + b[2] >= w - 1 or b[1] + b[3] >= h - 1
    # cut by the frame: at the border after the effect, and either not at the border before it (text running
    # to the page's edge is whole) or its box shrank (part of it went out of the picture)
    edge = 0
    for w, b in zip(words, boxes):
        if b and at_border(b, width, height):
            c = (w["x"], w["y"], w["width"], w["height"])
            if not at_border(c, clean.width, clean.height) or b[2] * b[3] < 0.8 * c[2] * c[3] * scale:
                edge += 1
    g0 = np.asarray(clean.convert("L"), np.float32)
    g1 = np.asarray(out.convert("L"), np.float32)
    idx = [i for i, b in enumerate(boxes) if b]
    if len(idx) > sample:
        idx = random.Random(0).sample(idx, sample)
    faint, ratios, seps, blots = 0, [], [], []
    for i in idx:
        c1 = _crop(g1, boxes[i])
        c0 = _crop(g0, (words[i]["x"], words[i]["y"], words[i]["width"], words[i]["height"]))
        if c1.size < 20 or c0.size < 20:
            continue
        if np.percentile(c1, 90) - np.percentile(c1, 5) < 25 or gap(c1) < FAINT_GAP:
            faint += 1
        s0 = cv2.Laplacian(c0, cv2.CV_32F).var()
        if s0 > 1:
            ratios.append(cv2.Laplacian(c1, cv2.CV_32F).var() / s0)
        s = separation(c0, c1)
        if s is not None:
            seps.append(s)
        blots.append(covered(c0, c1))
    n = max(1, len(idx))
    return {"lost": lost, "edge": edge, "faint": round(faint / n, 3),
            "sharpness": round(float(np.median(ratios)), 3) if ratios else 1.0,
            "separation": round(float(np.median(seps)), 2) if seps else 9.0,
            "drowned": round(sum(s < SEPARATION_MIN for s in seps) / max(1, len(seps)), 3),
            "blotted": sum(b > BLOT_MAX for b in blots), "blot_max": round(max(blots, default=0.0), 3)}


# a word with this much more of its box dark than before is under a streak or blot (set by looking at the
# highest-scoring words of 120 degraded pages: blotted-out words scored 0.33-0.51, still readable ones <= 0.27)
BLOT_MAX = 0.30
SEPARATION_MIN = 1.2  # below this a word's letters barely stand out from the noise around them


def readable(check):
    """No word lost, cut by the frame or blotted out; at most 1% of words faint or drowned (measurement
    noise: at default strength there are none, and a streak over a few words must not pass); sharpness at
    least 10% of the clean page's (a strong blur still readable passes, a page blurred away does not)."""
    return (check["lost"] == 0 and check.get("edge", 0) == 0 and check.get("blotted", 0) == 0
            and check["faint"] <= 0.01 and check["sharpness"] >= 0.10 and check.get("drowned", 0) <= 0.01)


# == lists and tables: the box of what is drawn ======================================================================

def overlap(a, b):
    """The part two boxes [x0, y0, x1, y1] share (either one if the other is None or they don't meet)."""
    if not a or not b:
        return a or b
    c = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    return c if c[2] > c[0] and c[3] > c[1] else a


def to_ink(plan, gray):
    """Lists and tables: also the box of what is drawn in them, as ("list_ink", i) / ("table_ink", i). Their HTML
    element is often the page's whole width while the items or cells are much narrower (centred text, bullets
    at the far side), and the label's box must hold what the image shows. On the clean page (gray), the box
    around every pixel that differs from the element's most common shade (its background): text, bullets,
    table lines. The element's own box stays for the rules that group and order blocks (tuned on it)."""
    out = list(plan)
    for key, r in plan:
        if key[0] in ("list", "table"):
            x0, y0 = max(0, int(r["x"])), max(0, int(r["y"]))
            c = gray[y0:int(r["y"] + r["height"]) + 1, x0:int(r["x"] + r["width"]) + 1]
            if c.size:
                ink = abs(c.astype(int) - int(np.bincount(c.ravel().astype(np.uint8)).argmax())) > 40
                rows, cols = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
                if rows.size:
                    out.append(((key[0] + "_ink",) + key[1:], {"x": x0 + cols[0], "y": y0 + rows[0],
                                                              "width": cols[-1] - cols[0] + 1, "height": rows[-1] - rows[0] + 1}))
    return out


# == is an answer in Surya's format? =================================================================================

LAYOUT_LABELS = set(LAYOUT_LABEL_SET)  # Surya's own layout labels (prompts.py)
BBOX = re.compile(r"^\d{1,4} \d{1,4} \d{1,4} \d{1,4}$")
HEADER = re.compile(r"<\|([A-Za-z][A-Za-z-]*)\|><\|(\d{1,4})\|><\|(\d{1,4})\|><\|(\d{1,4})\|><\|(\d{1,4})\|>")
CELL_END = re.compile(r"</(td|th|tr)>")


def problems(task, target):
    """What is wrong with a target for Surya 2.5's format (empty list: nothing): compact blocks for full_ocr,
    compact content without headers for block_ocr, a one-line JSON array for layout (its guided-decoding
    schema) and table (Row / Col)."""
    out = []

    def box_ok(values, where):
        x0, y0, x1, y1 = map(int, values)
        if x1 < x0 or y1 < y0 or max(x0, y0, x1, y1) > 1000:
            out.append(f"{where}: bbox order/range {' '.join(values)}")
    if task in ("full_ocr", "block_ocr"):
        if "\n" in target or "</div>" in target or "<div" in target:
            out.append("newline or div in a compact answer")
        if CELL_END.search(target):
            out.append("</td>, </th> or </tr> written out")
        heads = list(HEADER.finditer(target))
        if task == "block_ocr":
            if heads:
                out.append("block_ocr: a block header in the answer")
            if not target.strip():
                out.append("block_ocr: empty")
        else:
            if not heads or heads[0].start() != 0:
                out.append("full_ocr: does not start with a block header")
            for k, m in enumerate(heads):
                if m.group(1) not in LAYOUT_LABELS:
                    out.append(f"block {k}: label {m.group(1)}")
                box_ok(m.groups()[1:], f"block {k}")
                end = heads[k + 1].start() if k + 1 < len(heads) else len(target)
                if m.end() == end:
                    out.append(f"block {k}: empty")
    elif task in ("layout", "table"):
        if "\n" in target:
            out.append("JSON not on one line")
        try:
            entries = json.loads(target)
        except ValueError as e:
            return out + [f"JSON: {e}"]
        if not isinstance(entries, list) or len(entries) > 200:
            out.append("not a list of at most 200 entries")
        keys, labels = ({"label", "bbox", "count"}, LAYOUT_LABELS) if task == "layout" else ({"label", "bbox"}, {"Row", "Col"})
        for k, e in enumerate(entries):
            if set(e) != keys:
                out.append(f"entry {k}: keys {sorted(e)}")
            if e.get("label") not in labels:
                out.append(f"entry {k}: label {e.get('label')}")
            if not BBOX.match(e.get("bbox") or ""):
                out.append(f"entry {k}: bbox {e.get('bbox')!r}")
            else:
                box_ok(e["bbox"].split(), f"entry {k}")
            if task == "layout" and not (isinstance(e.get("count"), int) and e["count"] % 10 == 0):
                out.append(f"entry {k}: count {e.get('count')}")
        if task == "table":
            kinds = [e.get("label") for e in entries]
            if kinds != sorted(kinds, key=lambda x: x != "Row") or "Row" not in kinds or "Col" not in kinds:
                out.append("table: rows then columns, at least one of each")
    return out
