"""
Word- and line-level OCR ground truth for SynthDoc pages.

SynthDoc labels entities with the bounding box of the RDFa element that holds
them. That is the right unit for NER, but block elements span their whole
container (a name in a card gets the card's width), and only marked-up text is
labelled. OCR needs every visible word with a tight box. This module measures
both in the same Playwright page SynthDoc screenshots, after Paged.js has
paginated it, so the boxes and the images come from one layout:

* words: every whitespace-separated run of visible text, boxed with
  ``Range.getClientRects()``, clipped by overflow-hidden ancestors;
* lines: consecutive words of one layout block that share a visual line.

Text is kept in logical (reading) order, the convention for Arabic OCR ground
truth; ``direction`` records the visual direction.
"""
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List

WORDS_JS = r"""() => {
  const INLINE = new Set(['inline', 'contents', 'ruby']);
  const blockIds = new Map();
  const blockOf = (el, pageEl) => {
    for (let e = el; e && e !== pageEl; e = e.parentElement) {
      if (!INLINE.has(getComputedStyle(e).display)) return e;
    }
    return pageEl;
  };
  const blockId = b => { if (!blockIds.has(b)) blockIds.set(b, blockIds.size); return blockIds.get(b); };
  const clipBoxOf = (el, pageEl, pr) => {
    let box = {l: pr.left, t: pr.top, r: pr.right, b: pr.bottom};
    for (let e = el; e && e !== pageEl; e = e.parentElement) {
      const cs = getComputedStyle(e);
      if (cs.overflowX !== 'visible' || cs.overflowY !== 'visible') {
        const r = e.getBoundingClientRect();
        box = {l: Math.max(box.l, r.left), t: Math.max(box.t, r.top),
               r: Math.min(box.r, r.right), b: Math.min(box.b, r.bottom)};
      }
    }
    return box;
  };
  const union = rects => ({
    l: Math.min(...rects.map(r => r.left)), t: Math.min(...rects.map(r => r.top)),
    r: Math.max(...rects.map(r => r.right)), b: Math.max(...rects.map(r => r.bottom))
  });
  const sameLine = (a, b) => Math.min(a.b, b.bottom) - Math.max(a.t, b.top) > 0.5 * Math.min(a.b - a.t, b.height);
  const rectsOf = (node, start, end) => {
    const range = document.createRange();
    range.setStart(node, start); range.setEnd(node, end);
    return Array.from(range.getClientRects()).filter(r => r.width > 0 && r.height > 0);
  };
  // A word broken over two lines becomes one piece per line, split by characters.
  const pieces = (node, start, end) => {
    const rects = rectsOf(node, start, end);
    if (!rects.length) return [];
    const groups = [];
    for (const r of rects) {
      const g = groups.find(g => sameLine(g, r));
      if (g) Object.assign(g, union([{left: g.l, top: g.t, right: g.r, bottom: g.b}, r])); else groups.push(union([r]));
    }
    if (groups.length === 1) return [{start, end, box: groups[0]}];
    const out = [];
    for (let i = start; i < end; i++) {
      const cr = rectsOf(node, i, i + 1);
      if (!cr.length) continue;
      const last = out[out.length - 1];
      if (last && sameLine(last.box, cr[0])) {
        last.end = i + 1;
        last.box = union([{left: last.box.l, top: last.box.t, right: last.box.r, bottom: last.box.b}, ...cr]);
      } else {
        out.push({start: i, end: i + 1, box: union(cr)});
      }
    }
    return out;
  };

  // Glyph ink vs. layout box: Range rects are the font's content area, but some
  // glyphs (Arabic marks, swashes) can draw outside it. Canvas metrics give both.
  const ctx = document.createElement('canvas').getContext('2d');
  const inkOverflow = (shown, cs, height) => {
    ctx.font = `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
    ctx.direction = cs.direction; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    const m = ctx.measureText(shown);
    const r1 = v => Math.round(Math.max(0, v) * 10) / 10;
    return {
      overflow: [r1(m.actualBoundingBoxAscent - m.fontBoundingBoxAscent), r1(m.actualBoundingBoxRight - m.width),
                 r1(m.actualBoundingBoxDescent - m.fontBoundingBoxDescent), r1(m.actualBoundingBoxLeft)],
      metrics_delta: Math.round((height - (m.fontBoundingBoxAscent + m.fontBoundingBoxDescent)) * 10) / 10
    };
  };

  const variantMeta = document.querySelector('meta[name="synthdoc-ar-variant"]');
  return {
    variant: variantMeta ? JSON.parse(variantMeta.content) : null,
    lang: document.documentElement.lang || null,
    dir: document.documentElement.dir || getComputedStyle(document.documentElement).direction,
    pages: Array.from(document.querySelectorAll('.pagedjs_page')).map((pageEl, pageIdx) => {
      const pr = pageEl.getBoundingClientRect();
      const words = [];
      const walker = document.createTreeWalker(pageEl, NodeFilter.SHOW_TEXT);
      for (let node = walker.nextNode(); node; node = walker.nextNode()) {
        const text = node.nodeValue;
        if (!text || !text.trim()) continue;
        const el = node.parentElement;
        if (!el || el.closest('script, style, noscript, template')) continue;
        if (!el.checkVisibility({opacityProperty: true, visibilityProperty: true})) continue;
        const cs = getComputedStyle(el);
        if (cs.color === 'rgba(0, 0, 0, 0)') continue;
        const block = blockId(blockOf(el, pageEl));
        const clip = clipBoxOf(el, pageEl, pr);
        const re = /\S+/gu;
        for (let m = re.exec(text); m; m = re.exec(text)) {
          for (const p of pieces(node, m.index, m.index + m[0].length)) {
            const b = p.box;
            const l = Math.max(b.l, clip.l), t = Math.max(b.t, clip.t);
            const r = Math.min(b.r, clip.r), btm = Math.min(b.b, clip.b);
            if (r - l < 1 || btm - t < 1) continue;          // fully clipped: not on the image
            // The image shows the transformed text, not the DOM text
            let shown = text.slice(p.start, p.end);
            if (cs.textTransform === 'uppercase') shown = shown.toUpperCase();
            else if (cs.textTransform === 'lowercase') shown = shown.toLowerCase();
            else if (cs.textTransform === 'capitalize') shown = shown.charAt(0).toUpperCase() + shown.slice(1);
            const ink = inkOverflow(shown, cs, b.b - b.t);
            words.push({
              text: shown,
              x: l - pr.left, y: t - pr.top, width: r - l, height: btm - t,
              ink_overflow_px: ink.overflow, metrics_delta_px: ink.metrics_delta,
              clipped: (l > b.l + 0.5 || t > b.t + 0.5 || r < b.r - 0.5 || btm < b.b - 0.5),
              split_word: p.start !== m.index || p.end !== m.index + m[0].length,
              block: block,
              direction: cs.direction,
              font_family: cs.fontFamily.split(',')[0].replace(/['"]/g, '').trim(),
              font_size_px: parseFloat(cs.fontSize),
              font_weight: cs.fontWeight
            });
          }
        }
      }
      // Text drawn by CSS (pseudo-element content, list markers) is on the image
      // but not in the DOM, so no ground truth can include it. Report it.
      const generated = [];
      const content = pageEl.querySelector('[class*="djs_page_content"]') || pageEl;
      for (const el of content.querySelectorAll('*')) {
        if (!el.checkVisibility({opacityProperty: true, visibilityProperty: true})) continue;
        const r = el.getBoundingClientRect();
        const where = {x: r.left - pr.left, y: r.top - pr.top, width: r.width, height: r.height};
        for (const pseudo of ['::before', '::after']) {
          const c = getComputedStyle(el, pseudo).content;
          if (c && c !== 'none' && c !== 'normal' && !c.startsWith('url(') && /[^\s"']/.test(c)) {
            generated.push({kind: pseudo, content: c, element: el.tagName.toLowerCase() + (el.className ? '.' + el.className : ''), ...where});
          }
        }
        const cs = getComputedStyle(el);
        if (cs.display === 'list-item' && cs.listStyleType !== 'none') {
          generated.push({kind: 'list-marker', content: cs.listStyleType, element: el.tagName.toLowerCase(), ...where});
        }
      }
      return {page_number: pageIdx + 1, width: pr.width, height: pr.height, words, generated_text: generated};
    })
  };
}"""

_ARABIC = re.compile(r"[؀-ٟ٪-ۯۺ-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_LATIN = re.compile(r"[A-Za-zÀ-ɏ]")
_DIGIT = re.compile(r"[0-9٠-٩۰-۹]")


def script_of(text: str) -> str:
    """arabic | latin | digits | mixed | symbol, from the letters a word contains."""
    has = [name for name, rx in (("arabic", _ARABIC), ("latin", _LATIN)) if rx.search(text)]
    if len(has) == 2:
        return "mixed"
    if has:
        return has[0]
    if _DIGIT.search(text):
        return "digits"
    return "symbol"


def _box_2d(x, y, w, h, pw, ph) -> List[int]:
    return [int(x / pw * 1000), int(y / ph * 1000), int((x + w) / pw * 1000), int((y + h) / ph * 1000)]


def _ink_box(w: Dict, pw: float, ph: float) -> List[float]:
    """Layout box grown by the glyph ink that reaches outside it, clamped to the page."""
    top, right, bottom, left = w["ink_overflow_px"]
    x1, y1 = max(0.0, w["x"] - left), max(0.0, w["y"] - top)
    x2, y2 = min(pw, w["x"] + w["width"] + right), min(ph, w["y"] + w["height"] + bottom)
    return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]


def build_page_gt(page: Dict, meta: Dict) -> Dict:
    """Group a page's measured words into lines and assemble its ground-truth record."""
    pw, ph = page["width"], page["height"]
    words = page["words"]
    lines: List[Dict] = []
    current = None
    for idx, w in enumerate(words):
        if current is not None and current["block"] == w["block"]:
            top = max(current["_t"], w["y"])
            bottom = min(current["_b"], w["y"] + w["height"])
            if bottom - top > 0.5 * min(current["_b"] - current["_t"], w["height"]):
                current["word_ids"].append(idx)
                current["_t"], current["_b"] = min(current["_t"], w["y"]), max(current["_b"], w["y"] + w["height"])
                continue
        current = {"block": w["block"], "word_ids": [idx], "_t": w["y"], "_b": w["y"] + w["height"]}
        lines.append(current)

    out_words, out_lines = [], []
    for line_id, line in enumerate(lines):
        ws = [words[i] for i in line["word_ids"]]
        x1 = min(w["x"] for w in ws); y1 = min(w["y"] for w in ws)
        x2 = max(w["x"] + w["width"] for w in ws); y2 = max(w["y"] + w["height"] for w in ws)
        text = " ".join(unicodedata.normalize("NFC", w["text"]) for w in ws)
        inks = [_ink_box(w, pw, ph) for w in ws]
        ix1, iy1 = min(b[0] for b in inks), min(b[1] for b in inks)
        ix2, iy2 = max(b[0] + b[2] for b in inks), max(b[1] + b[3] for b in inks)
        out_lines.append({
            "id": line_id, "text": text,
            "x": round(x1, 2), "y": round(y1, 2), "width": round(x2 - x1, 2), "height": round(y2 - y1, 2),
            "box_2d": _box_2d(x1, y1, x2 - x1, y2 - y1, pw, ph),
            "ink_box": [round(ix1, 2), round(iy1, 2), round(ix2 - ix1, 2), round(iy2 - iy1, 2)],
            "direction": ws[0]["direction"], "script": script_of(text),
            "block_id": line["block"], "word_ids": [],
        })
        for w in ws:
            wid = len(out_words)
            out_lines[-1]["word_ids"].append(wid)
            text_w = unicodedata.normalize("NFC", w["text"])
            out_words.append({
                "id": wid, "line_id": line_id, "text": text_w,
                "x": round(w["x"], 2), "y": round(w["y"], 2),
                "width": round(w["width"], 2), "height": round(w["height"], 2),
                "box_2d": _box_2d(w["x"], w["y"], w["width"], w["height"], pw, ph),
                "script": script_of(text_w), "direction": w["direction"],
                "font_family": w["font_family"], "font_size_px": w["font_size_px"],
                "font_weight": w["font_weight"], "clipped": w["clipped"], "split_word": w["split_word"],
                # glyph ink beyond the box edges, [top, right, bottom, left] in px,
                # and the box grown to include it, [x, y, width, height]
                "ink_overflow_px": w["ink_overflow_px"],
                "ink_box": _ink_box(w, pw, ph),
            })
            if abs(w["metrics_delta_px"]) > 1:
                out_words[-1]["metrics_delta_px"] = w["metrics_delta_px"]

    return {
        **meta,
        "page_number": page["page_number"],
        "width": int(round(pw)), "height": int(round(ph)),
        "text": "\n".join(l["text"] for l in out_lines),
        "lines": out_lines,
        "words": out_words,
        "untracked_generated_text": page.get("generated_text", []),
    }


def draw_overlay(image_path: Path, gt: Dict, out_path: Path) -> None:
    """Lines in blue, words in green, on a copy of the page image."""
    from PIL import Image, ImageDraw
    img = Image.open(image_path).convert("RGB")
    sx, sy = img.width / gt["width"], img.height / gt["height"]
    draw = ImageDraw.Draw(img, "RGBA")
    for l in gt["lines"]:
        draw.rectangle([l["x"] * sx - 2, l["y"] * sy - 2, (l["x"] + l["width"]) * sx + 2,
                        (l["y"] + l["height"]) * sy + 2], outline=(30, 90, 220, 255), width=1)
    for w in gt["words"]:
        draw.rectangle([w["x"] * sx, w["y"] * sy, (w["x"] + w["width"]) * sx, (w["y"] + w["height"]) * sy],
                       outline=(0, 160, 60, 255), fill=(0, 200, 80, 40), width=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=92)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
