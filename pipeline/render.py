"""Rendering a filled template with SynthDoc in Chromium, and what is measured on the rendered page.

SynthDoc lays the page out with Paged.js in Chromium, takes a screenshot and boxes every slot; its Arabic
extension (synthdoc_arabic) also measures every word. This module changes how that runs, in memory (the
SynthDoc folder is not changed):

  * pages are rendered at RENDER_SCALE device pixels per CSS pixel (2: about 192 dpi);
  * one Chromium serves every page of a process (SynthDoc starts and closes one per page, about 1.6 s each);
    every page still gets a fresh browser context, so nothing carries over from one page to the next;
  * after Paged.js has laid the page out, the screenshot is taken as soon as it says it is done (SynthDoc
    always waited 500 ms; it still waits at most that long);
  * the word script also measures cut letters, each word's letters and slot, and the page checks (qa).

synthdoc_html() turns a bank template into the page SynthDoc renders; PAGE_JS runs in that page first.
"""
import atexit
import re

from playwright.sync_api import Browser, sync_playwright
from synthdoc import pagedjs_renderer
from synthdoc_arabic import pipeline as synthdoc_arabic

# -- rendering at a higher resolution -----------------------------------------------------------------------
# SynthDoc's browser page has no resolution setting: a page is screenshot at 1 pixel per CSS pixel (96 dpi;
# 9 pt text is 12 px tall), too small for OCR training. Pages are opened here with a device scale factor of
# RENDER_SCALE; the measured boxes are in CSS pixels and are scaled to match (doc_pipeline.scale_measurements).
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


# -- the template as SynthDoc renders it --------------------------------------------------------------------
SLOT_RE = re.compile(r"\{\{ slot_(\d+) \}\}")
PAGE_MM = {"A4": (210, 297), "A5": (148, 210), "Letter": (215.9, 279.4), "Legal": (215.9, 355.6)}
ROLE_PROPERTY = {"title": "doco:title", "subtitle": "doco:sectionTitle", "heading": "doco:sectionTitle",
                 "table_header": "doco:columnHeader", "cell": "doco:tableCell", "label": "doco-fd:fieldLabel",
                 "field_value": "doco-fd:fieldValue", "caption": "doco:caption", "list_item": "schema:itemListElement",
                 "quote": "schema:quotation", "footnote": "schema:comment", "header": "schema:headline",
                 "footer": "schema:creditText", "page_number": "schema:position", "sidebar": "schema:abstract"}
# Paged.js (vendored by SynthDoc) only looks for overflow to the right: with an RTL root, overflowing content
# is laid out off-page at negative x (and still labelled) instead of on a new page. As in
# ../synthdoc-arabic-test, Paged.js's own boxes stay LTR and only the content inside a page is RTL.
# Selectors match by substring because SynthDoc skips Paged.js for HTML containing its class prefix.
# A bank page is also capped at one sheet, like our own renders: Paged.js splits CSS multi-column layouts
# into thin scattered columns over many pages, so overflow is cut at the page edge instead (cut words are
# flagged by the word measuring and the page is made again: qa.py, labels.split_words).
PAGEDJS_RTL_FIX = """<style>
html, body, [class*="djs_page"], [class*="djs_sheet"], [class*="djs_area"] { direction: ltr; }
body > *, [class*="djs_page_content"] > div { direction: rtl; }
.page { max-block-size: var(--page-height); overflow: hidden; }
*, *::before, *::after { letter-spacing: normal !important; text-transform: none !important; }
* { text-overflow: clip !important; }
</style>
"""


def page_dims(size, orientation):
    """Page (width, height) in mm."""
    w, h = PAGE_MM[size]
    return (h, w) if orientation == "landscape" else (w, h)


def tag_slots(text, slots):
    """Every {{ slot_N }} wrapped in a span whose `property` SynthDoc boxes (each slot becomes a typed box)."""
    roles = {s["id"]: s["role"] for s in slots}

    def wrap(m):
        role = roles[f"slot_{m.group(1)}"]
        return (f'<span property="{ROLE_PROPERTY.get(role, "schema:text")}" data-slot="slot_{m.group(1)}"'
                f' data-role="{role}">{m.group(0)}</span>')
    return SLOT_RE.sub(wrap, text)


def synthdoc_html(meta, text, lang, folder):
    """The template as SynthDoc renders it: RTL fix, page size in mm, every slot tagged with its role.

    text: the template text, or None for the template file in folder (where its pictures are too).
    lang="en" turns the page left to right for English text."""
    text = text or (folder / "template.html.j2").read_text(encoding="utf-8")
    # SynthDoc renders from a temporary file, so pictures are linked by absolute path
    text = text.replace('src="img_', f'src="{folder.as_uri()}/img_')
    fix = PAGEDJS_RTL_FIX if lang == "ar" else PAGEDJS_RTL_FIX.replace("direction: rtl", "direction: ltr")
    text = text.replace("</head>", fix + "</head>", 1)
    if lang != "ar":
        text = text.replace('<html lang="ar" dir="rtl">', f'<html lang="{lang}" dir="ltr">', 1)
    # Paged.js knows "A4" but not "Legal"/"Letter" written this way and falls back to Letter portrait:
    # give the page size in millimetres instead.
    w, h = page_dims(meta["page"]["size"], meta["page"]["orientation"])
    text = re.sub(r"@page \{ size: [^;]+;", f"@page {{ size: {w}mm {h}mm;", text, count=1)
    return tag_slots(text, meta["slots"])


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
