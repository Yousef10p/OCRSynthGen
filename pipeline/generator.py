"""The generator: one class, DocPipeline, makes a training page for Surya OCR 2.5 from an empty template.

    from pipeline.generator import DocPipeline
    DocPipeline("pipeline_out/mine").sample(seed=7)

This file is the whole page-making path, in the order it runs (search for "STEP"):

    STEP 1  template   a template from bank2000/ (or one an LLM wrote: llm_templates.py)
    STEP 2  fill       slot text (Wikipedia), typography, colour theme, Arabic diacritics, photos
    STEP 3  render     SynthDoc lays the page out in Chromium and measures every word; the page is checked
                       and made again until no text is hidden, cut, overlapping or covered
    STEP 4  effect     clean, OCRSmith, Augraphy or scan-simulator; every box moves with the page
    STEP 5  Surya      the page's blocks (label, tags, box, order) as <div data-bbox data-label> HTML, and the
                       answer to the sample's prompt, written by Surya's own encode() (compact_original.py)
    STEP 6  files      image, prompt, answer, labels JSON, .md, .html

If something fails, the error says in which step (DocPipeline.sample). The tools come from their own folders
next to this repository, unchanged: SynthDoc (+ its Arabic extension), OCRSmith, Augraphy, scan-simulator.
"""
import atexit
import copy
import html
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from jinja2 import Template
from markupsafe import Markup
from ocrsmith.core.backgrounds import gradient_background, image_background, paper_background, solid_background
from ocrsmith.core.degradations import build_preset
from ocrsmith.core.degradations import physical as _ocr_physical
from ocrsmith.core.degradations.base import map_page as _ocr_map_page
from ocrsmith.domain.annotations import Line, Page, Region, RegionType, Word
from ocrsmith.domain.geometry import BBox
from ocrsmith.text.script import Direction
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import Browser, sync_playwright
from scan_simulator import ScanSimulator
from scan_simulator.config import PipelineConfig
from scan_simulator.presets import PRESETS as SCAN_PRESETS
from scan_simulator.transforms import damage as _scan_damage, paper as _scan_paper, scanner as _scan_scanner
from scipy import fft as _fft
from scipy.ndimage import find_objects, gaussian_filter as _gaussian_filter
from synthdoc import pagedjs_renderer
from synthdoc.ner_data_generator import NERDataGenerator
from synthdoc_arabic import pipeline as synthdoc_arabic
from synthdoc_arabic.ocr_gt import build_page_gt
from tokenizers import Tokenizer

from pipeline.compact_original import encode  # Surya's own encoder (its compact.py): writes every answer
from pipeline.pools import ImagePool, TextPool

ROOT = Path(__file__).resolve().parents[1]


# ====================================================================================================================
# STEP 2 - fill: slot text, typography, colour themes, Arabic diacritics
# ====================================================================================================================
# Step 2, fill: what goes into a template before it is rendered.
#
#   * text: each slot's text (from the Wikipedia text pool, pipeline/pools.py, or sample sentences);
#   * typography: fonts, size, weights, line height, alignment, margins, ink, digits;
#   * theme: colours and heading / table / box styles, so one template looks different on every page;
#   * tashkeel: Arabic diacritics at one level per page (none, partial, full).

# == text and typography =====================================================================================

# Text and style for filling a template: the text of each slot and a random typography per page.
# 
# Slots get text from the Wikipedia text pool (pipeline/pools.py) or, without it, from the sample
# sentences below; numbers, dates and phone numbers are random. Arabic fonts are read from OCRSmith's font
# folder, English pages use installed Latin fonts.

OCRSMITH = ROOT.parent / "ocrsmith" / "assets"
INKS = [(0, 0, 0), (20, 20, 20), (30, 30, 45), (10, 25, 60), (45, 35, 30), (60, 10, 10)]
WEIGHTS = {"thin": 100, "extralight": 200, "light": 300, "regular": 400, "medium": 500, "semibold": 600,
           "bold": 700, "extrabold": 800, "black": 900}

# Our own neutral sentences; repetition across pages is fine for a demo.
SENTENCES = [
    "تعمل المؤسسة على تطوير خدماتها الرقمية لتسهيل وصول المواطنين إلى المعلومات.",
    "شهد العام الماضي نموا ملحوظا في عدد المشاريع الصغيرة والمتوسطة.",
    "يهدف هذا التقرير إلى عرض أهم النتائج التي توصل إليها فريق العمل.",
    "تقع المدينة على ضفاف النهر وتشتهر بأسواقها القديمة وحدائقها الواسعة.",
    "يرجى تعبئة النموذج بخط واضح وإرفاق المستندات المطلوبة قبل الموعد المحدد.",
    "ارتفعت نسبة المشاركة في البرنامج التدريبي إلى أكثر من سبعين في المئة.",
    "تتضمن الخطة الجديدة تحسين البنية التحتية وتوسيع شبكة النقل العام.",
    "أكدت اللجنة أهمية التعاون بين الجامعات والقطاع الخاص في مجال البحث العلمي.",
    "يقدم المركز دورات مجانية في اللغة العربية والحاسوب والمهارات المهنية.",
    "تم تحديث قاعدة البيانات لتشمل جميع السجلات الصادرة خلال الربع الأخير.",
    "تعتمد الدراسة على بيانات ميدانية جمعت من عدة مناطق مختلفة.",
    "يسر الإدارة أن تعلن عن افتتاح الفرع الجديد في الأسبوع القادم.",
    "يجب مراجعة الشروط والأحكام بعناية قبل التوقيع على العقد.",
    "حققت الشركة أرباحا صافية تجاوزت توقعات المحللين لهذا العام.",
    "تساهم القراءة اليومية في تنمية مهارات التفكير والتعبير لدى الطلاب.",
    "أوصى الاجتماع بإعداد دليل شامل للإجراءات الإدارية المعتمدة.",
    "يمكن التواصل مع خدمة العملاء عبر الهاتف أو البريد الإلكتروني.",
    "تشير المؤشرات الأولية إلى تحسن الأداء في معظم القطاعات الاقتصادية.",
]
WORDS = [w.strip(".،") for s in SENTENCES for w in s.split()]
# English pages (the pipeline fills a share of templates in English, left to right)
SENTENCES_EN = [
    "The institution is developing its digital services to make information easier to reach.",
    "Last year saw clear growth in the number of small and medium-sized businesses.",
    "This report presents the main findings reached by the working group.",
    "The city lies on the bank of the river and is known for its old markets and wide gardens.",
    "Please fill in the form clearly and attach the required documents before the deadline.",
    "Participation in the training programme rose to more than seventy percent.",
    "The new plan includes better infrastructure and a wider public transport network.",
    "The committee stressed the value of cooperation between universities and industry.",
    "The centre offers free courses in languages, computing and professional skills.",
    "The database has been updated to include all records issued in the last quarter.",
    "The study is based on field data collected from several different regions.",
    "The management is pleased to announce the opening of a new branch next week.",
    "The terms and conditions should be reviewed carefully before signing the contract.",
    "The company reported a net profit that exceeded analyst expectations this year.",
    "Daily reading helps students develop their thinking and writing skills.",
    "The meeting recommended preparing a complete guide to the approved procedures.",
    "Customer service can be reached by telephone or by electronic mail.",
    "Early indicators point to better performance in most sectors of the economy.",
]
WORDS_EN = [w.strip(".,") for s in SENTENCES_EN for w in s.split()]
# Windows fonts with Latin glyphs, used by name (no @font-face needed): Chromium finds them installed
LATIN_FONTS = ["Georgia", "Times New Roman", "Cambria", "Constantia", "Book Antiqua", "Palatino Linotype",
               "Arial", "Calibri", "Segoe UI", "Verdana", "Tahoma", "Candara", "Corbel", "Trebuchet MS"]
ARABIC_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


# -- text ------------------------------------------------------------------------------------------

def prose(rng, n, sentences=SENTENCES):
    """n words of running text, with sentence ends and the odd comma."""
    out = []
    while len(out) < n:
        out += rng.choice(sentences).split()
    words = out[:n]
    if not words[-1].endswith("."):
        words[-1] = words[-1].rstrip("،") + "."
    return " ".join(words)


def phrase(rng, n, words=WORDS):
    start = rng.randrange(len(words))
    return " ".join(words[(start + i) % len(words)] for i in range(n))


def number(rng):
    return rng.choice([str(rng.randint(1, 999)), f"{rng.randint(1, 99)}.{rng.randint(0, 99):02d}",
                       f"{rng.randint(1, 99)}%", f"{rng.randint(1000, 99999):,}"])


def field(rng):
    return rng.choice([f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1990, 2025)}",
                       f"05{rng.randint(0, 9)} {rng.randint(100, 999)} {rng.randint(1000, 9999)}",
                       f"{rng.randint(100, 99999):,}.{rng.randint(0, 99):02d}", str(rng.randint(10000, 999999))])


def fill(slot, rng, digits, lang="ar", source=None):
    """HTML-escaped text for one slot: right kind of text for its role, random length in its word range.

    lang="en" fills it with English instead of Arabic. source (a pipeline TextPool) supplies the words
    from real text instead of the demo sentences below; numbers, dates and phone numbers stay random."""
    sentences, words = (SENTENCES_EN, WORDS_EN) if lang == "en" else (SENTENCES, WORDS)
    lo, hi = slot["words"]
    n = rng.randint(lo, max(lo, hi))
    role = slot["role"]
    if role == "page_number":
        text = str(rng.randint(1, 120))
    elif role == "cell" and rng.random() < 0.55:
        text = number(rng)
    elif role == "field_value" and rng.random() < 0.6:
        text = field(rng)
    elif source is not None:
        text = source.text(role, n, rng)
    elif role == "label":
        text = phrase(rng, n, words) + ":"
    elif role in ("paragraph", "quote", "footnote", "sidebar", "list_item", "caption"):
        text = prose(rng, n, sentences)
    else:
        text = phrase(rng, n, words)
    return html.escape(text.translate(ARABIC_DIGITS) if digits == "arabic-indic" else text)


# -- style -----------------------------------------------------------------------------------------

def font_families():
    """Arabic families in OCRSmith's font folder: {family: {(weight, italic): file}}."""
    families = {}
    for path in sorted((OCRSMITH / "fonts").glob("*.[ot]tf")):
        family, style = ImageFont.truetype(str(path), 12).getname()
        if "Mono" in family:  # Latin only
            continue
        key = style.lower().replace(" ", "")
        weight = next((w for k, w in sorted(WEIGHTS.items(), key=lambda kv: -len(kv[0])) if k in key), 400)
        families.setdefault(family, {})[(weight, "italic" in key)] = path
    return families


def pick_weight(rng, faces, wanted):
    weights = sorted({w for w, italic in faces if not italic})
    return min(weights, key=lambda w: abs(w - rng.choice(wanted)))


def style(rng, families):
    """Random typography for one page: (its settings, CSS for the template's {{ style }} hook)."""
    body_family = rng.choice(sorted(families))
    head_family = body_family if rng.random() < 0.7 else rng.choice(sorted(families))
    body_w = pick_weight(rng, families[body_family], [300, 400, 400, 500])
    head_w = pick_weight(rng, families[head_family], [600, 700, 700, 800, 900])
    s = page_style(rng, body_family, head_family, body_w, head_w)
    faces = "".join(
        f'@font-face {{ font-family: "{fam}"; src: url("{path.as_uri()}"); font-weight: {w};'
        f' font-style: {"italic" if it else "normal"}; }}' + "\n"
        for fam in sorted({body_family, head_family}) for (w, it), path in families[fam].items())
    return s, faces + style_css(s)


def style_latin(rng):
    """The same random typography for an English page, from installed Latin fonts."""
    body_family = rng.choice(LATIN_FONTS)
    head_family = body_family if rng.random() < 0.7 else rng.choice(LATIN_FONTS)
    s = page_style(rng, body_family, head_family, 400, 700, "western")
    # Arabic letter numbering in lists means nothing on an English page
    return s, style_css(s) + "ol { list-style-type: decimal !important; }\n"


def page_style(rng, body_family, head_family, body_w, head_w, digits=None):
    """Size, spacing, alignment, margins, ink and digits (picked at random unless given), in a fixed draw order."""
    ink = rng.choice(INKS)
    return {"font": body_family, "heading_font": head_family, "body_weight": body_w, "heading_weight": head_w,
            "size_pt": round(rng.uniform(9, 14), 1), "line_height": round(rng.uniform(1.35, 2.0), 2),
            "align": rng.choice(["start", "justify", "justify"]), "margin_mm": rng.randint(10, 28),
            "ink": f"rgb{ink}", "digits": digits or rng.choice(["western", "western", "arabic-indic"])}


def style_css(s):
    return (f':root {{ --font-body: "{s["font"]}", serif; --font-heading: "{s["heading_font"]}", serif;'
            f' --base-size: {s["size_pt"]}pt; --line-height: {s["line_height"]}; --align: {s["align"]};'
            f' --page-margin: {s["margin_mm"]}mm; }}\n'
            f".page {{ color: {s['ink']}; font-weight: {s['body_weight']}; }}\n"
            f"h1, h2, h3, h4, th {{ font-weight: {s['heading_weight']}; }}\n")


# -- themes --------------------------------------------------------------------------------------------------

# Random visual themes: the same template looks different on every page, and always readable.
# 
# theme(rng, source) -> (record, css, js). The CSS goes into the template's {{ style }} hook after the
# typography (STEP 2); the JS settings go to the page script (PAGE_JS, STEP 3).
# 
#   * Grammar and DocLayNet templates are plain black-on-white layouts, so a theme styles them fully: a colour
#     palette (13 professional accents, each dark enough to read on white) with headings in one of 6
#     treatments, tables in one of 5, side boxes and quotes in one of 4, bullets, rules, a page tint, spacing
#     and corners. Grammar pages also get spacing changes; DocLayNet pages only colours (their boxes have
#     fixed sizes, and padding would push text out of them).
#   * WebSight pages come with their own design. They keep it, but in the browser every colour on the page is
#     turned around the colour wheel by the same angle (hue only: lightness stays, so contrast stays), and
#     more or less saturated, so one web design comes in many colour schemes.
# 
# All themes use only marks that are not text: bullets are disc, circle or square (ordered lists lose their
# numbers, which would be text on the image but not in the ground truth). Whatever a theme does, the page
# script then checks every slot's contrast against what is behind it and darkens (or lightens) text below
# 4.5:1 (PAGE_JS, STEP 3).

ACCENTS = ["#1f4e79", "#0b6e4f", "#7b1e3a", "#5b2a86", "#8a4b08", "#0e5e6f", "#2f3e46", "#a4161a", "#374785",
           "#2d6a4f", "#9a3412", "#4a4e69", "#1d3557"]
THEME_INKS = ["#111111", "#1a1a1a", "#222222", "#262626", "#1c2833", "#2b2118"]


def mix(color, white):
    """color mixed with white (0 = color, 1 = white)."""
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(c + (255 - c) * white):02x}" for c in (r, g, b))


def theme(rng, source):
    accent = rng.choice(ACCENTS)
    tint, soft = mix(accent, rng.uniform(0.88, 0.94)), mix(accent, rng.uniform(0.95, 0.975))
    rec = {"accent": accent, "heading": rng.choice(["plain", "accent", "underline", "bar", "band", "rule-above"]),
           "table": rng.choice(["grid", "striped", "header-tint", "rules", "boxed"]),
           "box": rng.choice(["tint", "bar", "outline", "plain"]),
           "page": rng.choices(["white", "off-white", "tint"], weights=[5, 3, 2])[0],
           "bullet": rng.choice(["disc", "square", "circle"]), "radius": rng.choice([0, 0, 3, 6]),
           "paragraph_gap": round(rng.uniform(0.3, 1.0), 2), "heading_gap": round(rng.uniform(0.5, 1.4), 2),
           "ink": rng.choice(THEME_INKS)}
    page_bg = {"white": "#ffffff", "off-white": rng.choice(["#fbfaf7", "#fafafa", "#f9f8f4"]), "tint": soft}[rec["page"]]
    imp = " !important"
    css = [f"ol, ul {{ list-style-type: {rec['bullet']}{imp}; }}",
           f"li::marker {{ color: {accent}; }}",
           f"html, body, .page {{ background-color: {page_bg}{imp}; }}"]
    js = {"hue": 0, "saturate": 1.0}
    if source in ("websight", "websight2"):
        # Their own design stays; its colours are turned around the colour wheel (in the page script)
        js = {"hue": rng.choice([0, 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]),
              "saturate": round(rng.uniform(0.7, 1.25), 2)}
        rec["hue"], rec["saturate"] = js["hue"], js["saturate"]
        return rec, "\n".join(css) + "\n", js
    heading = {"plain": f"color: {rec['ink']}{imp};",
               "accent": f"color: {accent}{imp};",
               "underline": f"color: {accent}{imp}; border-block-end: 2px solid {accent}{imp};",
               "bar": f"color: {accent}{imp}; border-inline-start: 4px solid {accent}{imp}; padding-inline-start: 0.4em;",
               "band": f"color: {accent}{imp}; background-color: {tint}{imp}; padding-inline: 0.3em;",
               "rule-above": f"color: {rec['ink']}{imp}; border-block-start: 2px solid {accent}{imp};"}[rec["heading"]]
    if source == "doclaynet":  # fixed-size boxes: colours only, nothing that takes room
        heading = heading.replace("padding-inline-start: 0.4em;", "").replace("padding-inline: 0.3em;", "")
        heading = heading.replace("border-block-end: 2px", "border-block-end: 1px").replace(
            "border-block-start: 2px", "border-block-start: 1px").replace("border-inline-start: 4px", "border-inline-start: 3px")
    css += [f"h1, h2, h3, h4, h5, h6, dt, strong {{ {heading} }}",
            f"body, .page, p, li, td, dd, span, div {{ color: {rec['ink']}; }}",
            f"th {{ color: {accent}{imp}; }}"]
    table = {"grid": f"th, td {{ border-color: {mix(accent, 0.55)}{imp}; }}",
             "striped": f"tbody tr:nth-child(even) td {{ background-color: {soft}{imp}; }} "
                        f"th {{ border-block-end: 2px solid {accent}{imp}; }}",
             "header-tint": f"th {{ background-color: {tint}{imp}; }} th, td {{ border-color: {mix(accent, 0.6)}{imp}; }}",
             "rules": f"th, td {{ border-inline: none{imp}; border-block-start: none{imp}; "
                      f"border-block-end: 1px solid {mix(accent, 0.6)}{imp}; }}",
             "boxed": f"table {{ outline: 2px solid {accent}; }} th {{ background-color: {soft}{imp}; }}"}[rec["table"]]
    box = {"tint": f"aside, blockquote {{ background-color: {soft}{imp}; border-color: {mix(accent, 0.5)}{imp}; }}",
           "bar": f"aside, blockquote {{ border-inline-start: 4px solid {accent}{imp}; }}",
           "outline": f"aside, blockquote {{ border-color: {accent}{imp}; }}",
           "plain": ""}[rec["box"]]
    css += [table, box, f"hr {{ border-color: {mix(accent, 0.5)}; }}"]
    if source == "grammar":
        css += [f"p {{ margin-block-end: {rec['paragraph_gap']}em; }}",
                f"h2, h3 {{ margin-block-start: {rec['heading_gap']}em; }}",
                f"aside, blockquote, table, .ph {{ border-radius: {rec['radius']}px; }}"]
    return rec, "\n".join(c for c in css if c) + "\n", js


# -- tashkeel ------------------------------------------------------------------------------------------------

# Tashkeel: Arabic diacritics (harakat) on the slot texts, so the OCR model also learns marked text.
# 
# Wikipedia text has almost no marks, so they are added by a small diacritization model: text2tashkeel's
# default "rawi-ensemble" (BiLSTM models in ONNX, run by onnxruntime, offline, about 2% diacritic error rate
# as its authors report; https://github.com/TigreGotico/text2tashkeel, Apache 2.0). The text itself never
# changes: a word takes the model's marks only if its letters are the same as before (the model also "restores"
# hamza, sometimes wrongly: لامائي -> لَأَمَائِيٍّ), and a word that already had marks in Wikipedia keeps them.
# Latin letters, digits and punctuation pass through unchanged.
# 
# Per page, one level:
#     none      the text as it is (most printed Arabic)
#     partial   shadda kept everywhere and all marks on about a third of the words (how newspapers, school
#               books and signs often mark only some words)
#     full      every letter marked (religious, classical and teaching texts)
# 
# The labels are always what is drawn: the ground truth comes from the rendered page's words, marks included.

TASHKEEL = {"none": 0.4, "partial": 0.3, "full": 0.3}
ARABIC = re.compile(r"[ء-ي]")
MARKS = re.compile(r"[ً-ْٰ]")
VOWELS = re.compile(r"[ً-ِْٰ]")  # every mark but shadda (U+0651)
PARTIAL_WORDS = 0.35
TASHKEEL_LINE_HEIGHT = {"partial": 1.6, "full": 1.75}  # the least line height a page with marks gets
_model = None


def tashkeel_model():
    global _model
    if _model is None:
        from text2tashkeel import Diacritizer
        _model = Diacritizer()  # one onnxruntime session, reused
    return _model


def add_tashkeel(text, level, rng):
    """The text with marks at this level ("none", "partial" or "full")."""
    if level == "none" or not ARABIC.search(text):
        return text
    words, marked = text.split(" "), tashkeel_model().diacritize(text).split(" ")
    if len(marked) != len(words):  # can't pair the words up: leave the text as it is
        return text
    words = [m if MARKS.sub("", m) == w else w for w, m in zip(words, marked)]  # same letters only
    if level == "full":
        return " ".join(words)
    return " ".join(w if rng.random() < PARTIAL_WORDS else VOWELS.sub("", w) for w in words)


# ====================================================================================================================
# STEP 3 - render: SynthDoc in Chromium, word measuring, page checks
# ====================================================================================================================
# Rendering a filled template with SynthDoc in Chromium, and what is measured on the rendered page.
#
# SynthDoc lays the page out with Paged.js in Chromium, takes a screenshot and boxes every slot; its Arabic
# extension (synthdoc_arabic) also measures every word. This module changes how that runs, in memory (the
# SynthDoc folder is not changed):
#
#   * pages are rendered at RENDER_SCALE device pixels per CSS pixel (2: about 192 dpi);
#   * one Chromium serves every page of a process (SynthDoc starts and closes one per page, about 1.6 s each);
#     every page still gets a fresh browser context, so nothing carries over from one page to the next;
#   * after Paged.js has laid the page out, the screenshot is taken as soon as it says it is done (SynthDoc
#     always waited 500 ms; it still waits at most that long);
#   * the word script also measures cut letters, each word's letters and slot, and the page checks (qa).
#
# synthdoc_html() turns a bank template into the page SynthDoc renders; PAGE_JS runs in that page first.

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
# flagged by the word measuring and the page is made again: render_issues, split_words).
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


# -- checks of the rendered page -----------------------------------------------------------------------------

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


# -- test harness -------------------------------------------------------------------------------------------


# ====================================================================================================================
# STEP 4 - effect: OCRSmith, Augraphy, scan-simulator (boxes move with the page)
# ====================================================================================================================
# Step 4: degrade the rendered page with OCRSmith, Augraphy or scan-simulator, at a chosen strength.
#
# Every tool takes a strength from 0 to 1 (the generator page has a slider per tool, the dataset command
# --strength-<tool>). 0.5 is the tool's own default settings; 0 is the lightest, 1 roughly twice as strong:
#
#   * OCRSmith: every degradation of the chosen capture preset (scan, photo, fax, archive) has its numbers
#     scaled by f = 2 * strength around their neutral value (rotation angle, noise sigma, blur radius, warp
#     magnitude, stain strength... x f; JPEG quality and downscale toward "none"; contrast/brightness factors
#     toward 1) and its probability by (0.5 + strength);
#   * Augraphy: the ink, paper and post-processing effects get their intensity ranges scaled the same way and
#     their probabilities by (0.5 + strength); only effects that don't move pixels are used;
#   * scan-simulator: the chosen preset's transforms (rotation, perspective, folds, noise, blur, stains...) have
#     their ranges scaled by f around neutral and their probabilities by (0.5 + strength).
#
# Boxes move with the page: OCRSmith moves its own annotation (Page/Region/Line/Word: each word's four corners,
# through the same rotation, perspective, curl or wrinkles as the pixels); scan-simulator warps a word-id mask
# with the image (word_mask: words that overlap go in separate layers); Augraphy moves nothing.
#
# legibility() checks the result word by word, so a strong setting can't silently destroy text: no word may be
# lost (pushed off the page, under a hole), at most 1% of words may be faint (ink vs paper < 25 of 255) or
# drowned (letters that barely stand out from the noise around them: separation()), and the words' sharpness
# may not drop below 10% of the clean page's (blur, downscaling). DocPipeline retries
# with new seeds, then with less strength, and keeps only a page that passes.

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

TOOLS = ("ocrsmith", "augraphy", "scan-simulator")
DEFAULT_STRENGTH = 0.5
# how often each preset is picked
OCRSMITH_PRESETS = {"scan": 4.0, "photo": 3.0, "fax": 0.7, "archive": 1.5}
SCAN_PRESET_WEIGHTS = {"scan-clean": 2, "scan-heavy": 2, "photo-indoor": 2, "photo-outdoor": 2, "photocopy": 1.5,
                       "archive": 1.5}


def numeric(v):
    """A number or a range of numbers (the settings a strength can scale)."""
    if isinstance(v, (tuple, list)):
        return bool(v) and all(numeric(x) for x in v)
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def toward(value, neutral, f, lo=None, hi=None):
    """value moved from neutral by f (f = 1: unchanged); works on numbers and (low, high) ranges."""
    if isinstance(value, (tuple, list)):
        out = [toward(v, neutral, f, lo, hi) for v in value]
        return type(value)(sorted(out)) if isinstance(value, tuple) else sorted(out)
    v = neutral + (value - neutral) * f
    v = max(lo, v) if lo is not None else v
    v = min(hi, v) if hi is not None else v
    return round(v) if isinstance(value, int) and not isinstance(value, bool) else v


# -- OCRSmith -----------------------------------------------------------------------------------------------

def ocrsmith_pipeline(preset, strength):
    pipeline = build_preset(preset)
    f, pm = 2 * strength, 0.5 + strength
    for d in pipeline.degradations:
        name = type(d).__name__
        d.probability = min(1.0, d.probability * pm)
        for k, v in list(vars(d).items()):
            if k in ("probability", "fill", "edge", "angle") or not numeric(v):
                continue
            if k == "scale" and name != "Downscale" or k == "size":
                continue  # a spatial scale or a kernel size, not a strength
            if k == "quality":
                setattr(d, k, toward(v, 100, f, 5, 100))
            elif k == "scale":  # Downscale: 1 = no loss
                setattr(d, k, toward(v, 1.0, f, 0.25, 1.0))
            elif k == "factor":  # Contrast, Brightness: 1 = unchanged
                setattr(d, k, toward(v, 1.0, f, 0.3, 3.0))
            elif k == "count":
                setattr(d, k, toward(v, 0, f, 0))
            else:
                setattr(d, k, toward(v, 0, f, 0))
    return pipeline


def ocrsmith_paper(rng, w, h):
    """An OCRSmith paper for the page to be printed on: (RGBA image, its kind)."""
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
        # OCRSmith's other photos (bricks, containers, city...) hide the text; a document sits on paper.
        kind = "photo texture (white paper)"
        bg = image_background(OCRSMITH / "backgrounds" / "white_paper.jpg")
    return bg(w, h).convert("RGBA"), kind


def to_page(words, width, height):
    """Words as an OCRSmith Page (which its degradations move): one region per line, each word's text set to
    its index."""
    by_line = {}
    for i, w in enumerate(words):
        by_line.setdefault(w["line_id"], []).append(
            Word(str(i), BBox.from_xywh(w["x"], w["y"], w["width"], w["height"])))
    regions = []
    for ws in by_line.values():
        box = BBox(min(w.bbox.x0 for w in ws), min(w.bbox.y0 for w in ws),
                   max(w.bbox.x1 for w in ws), max(w.bbox.y1 for w in ws))
        regions.append(Region(RegionType.PARAGRAPH, box, (Line("", box, tuple(ws), Direction.RTL),)))
    return Page(width, height, tuple(regions), Direction.RTL)


def from_page(page, n):
    """Word boxes [x, y, w, h] by index from an OCRSmith Page; None for a word pushed off the page."""
    boxes = [None] * n
    for region in page.regions:
        for line in region.lines:
            for w in line.words:
                b = w.bbox
                if b.x1 - b.x0 >= 1 and b.y1 - b.y0 >= 1:
                    x, y = int(b.x0), int(b.y0)
                    boxes[int(w.text)] = [x, y, int(np.ceil(b.x1)) - x, int(np.ceil(b.y1)) - y]
    return boxes


def degrade_ocrsmith(image, words, seed, strength):
    """OCRSmith paper, then one capture preset at this strength; returns (image, word boxes, attacks)."""
    rng = random.Random(seed)
    paper, kind = ocrsmith_paper(rng, *image.size)
    out = Image.fromarray((np.asarray(image, np.float32) * np.asarray(paper.convert("RGB"), np.float32) / 255)
                          .round().astype(np.uint8))
    preset = rng.choices(list(OCRSMITH_PRESETS), weights=list(OCRSMITH_PRESETS.values()))[0]
    out, page, records = ocrsmith_pipeline(preset, strength).apply(out, to_page(words, *image.size), rng)
    return (out.convert("RGB"), from_page(page, len(words)),
            [f"ocrsmith-paper:{kind}", f"ocrsmith:{preset}"] + [r.name for r in records])


# -- Augraphy -----------------------------------------------------------------------------------------------

def augraphy_pipeline(seed, strength):
    from augraphy import (AugraphyPipeline, BadPhotoCopy, Brightness, BrightnessTexturize, ColorPaper, DirtyDrum,
                          DirtyScreen, InkBleed, InkMottling, Jpeg, Letterpress, LightingGradient, LowInkPeriodicLines,
                          LowInkRandomLines, NoiseTexturize, OneOf, ShadowCast, SubtleNoise)
    f, pm = 2 * strength, 0.5 + strength

    def p(x):
        return min(1.0, x * pm)
    return AugraphyPipeline(
        ink_phase=[OneOf([InkBleed(intensity_range=toward((0.4, 0.7), 0, f, 0.05, 0.95)),
                          LowInkRandomLines(count_range=toward((5, 10), 0, f, 1)),
                          LowInkPeriodicLines(count_range=toward((2, 5), 0, f, 1)),
                          InkMottling(ink_mottling_alpha_range=toward((0.2, 0.3), 0, f, 0.02, 0.6))], p=p(0.6)),
                   Letterpress(p=p(0.25))],
        paper_phase=[OneOf([NoiseTexturize(sigma_range=toward((3, 10), 0, f, 1)),
                            BrightnessTexturize(texturize_range=toward((0.8, 0.99), 1.0, f, 0.5, 0.999))], p=p(0.6)),
                     ColorPaper(saturation_range=toward((10, 40), 0, f, 1), p=p(0.3))],
        post_phase=[OneOf([DirtyDrum(noise_intensity=toward(0.5, 0, f, 0.05, 0.6)),  # above 0.6: black bands over lines
                           DirtyScreen(), SubtleNoise(subtle_range=max(1, round(10 * f)))], p=p(0.6)),
                    OneOf([LightingGradient(),
                           ShadowCast(shadow_opacity_range=toward((0.2, 0.9), 0, f, 0.05, 0.95)),
                           Brightness(brightness_range=toward((0.8, 1.4), 1.0, f, 0.4, 1.8))], p=p(0.5)),
                    Jpeg(quality_range=toward((25, 95), 100, f, 5, 100), p=p(0.5)),
                    BadPhotoCopy(p=p(0.15))],
        random_seed=seed)


def degrade_augraphy(image, words, seed, strength):
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    out, used = None, None
    for s in range(seed, seed + 3):  # Augraphy's noise code sometimes fails for one random draw
        try:
            out, used = augraphy_pipeline(s, strength).augment(bgr)["output"], s
            break
        except Exception:
            continue
    if out is None:
        out = bgr
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR) if out.ndim == 2 else out
    boxes = [[round(w["x"]), round(w["y"]), round(w["width"]), round(w["height"])] for w in words]
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)), boxes, (["augraphy"] if used is not None else [])


# -- scan-simulator -----------------------------------------------------------------------------------------

INTEGER_SIZES = ("size", "kernel", "line_width", "radius", "shift")


def scan_config(preset, strength):
    cfg = copy.deepcopy(SCAN_PRESETS[preset])
    f, pm = 2 * strength, 0.5 + strength
    for name, params in cfg["transforms"].items():
        params["p"] = min(1.0, params.get("p", 1.0) * pm)
        for k, v in list(params.items()):
            if k in ("p", "color", "type", "direction") or not numeric(v):
                continue
            if k == "quality":
                params[k] = toward(v, 100, f, 5, 100)
            elif k in INTEGER_SIZES:
                params[k] = toward(v, 0, f, 1)
            elif k == "count":
                params[k] = toward(v, 0, f, 1)
            else:
                params[k] = toward(v, 0, f)
    return PipelineConfig.from_dict(cfg)


def word_mask(shape, words, layers=3):
    """Word i painted with id i + 1 (0 = no word), in the first of `layers` channels where it overlaps no word
    already painted: a word painted over another (two lines' boxes touching) would cut the other's box.
    scan-simulator moves this mask with the page."""
    mask = np.zeros((*shape, layers), np.uint16)
    for i, w in enumerate(words):
        x, y = max(0, round(w["x"])), max(0, round(w["y"]))
        area = mask[y:round(w["y"] + w["height"]) + 1, x:round(w["x"] + w["width"]) + 1]
        k = next((k for k in range(layers) if not area[..., k].any()), layers - 1)
        area[..., k] = i + 1
    return mask


def mask_boxes(mask, n):
    """Each word's box [x, y, w, h] after the effect: the extent of its id (None if it is gone)."""
    boxes = [None] * n
    for k in range(mask.shape[2]):
        for i, f in enumerate(find_objects(mask[..., k].astype(np.int32), max_label=n)):
            if f:
                boxes[i] = [f[1].start, f[0].start, f[1].stop - f[1].start, f[0].stop - f[0].start]
    return boxes


SCAN_PAD = 0.04  # paper added on each side before scan-simulator, times the page's longer side (x 1 at strength 0.5)


def degrade_scan(image, words, seed, strength, use_augraphy=False):
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    attacks = []
    if use_augraphy:
        img, _, a = degrade_augraphy(image, words, seed, strength)
        bgr, attacks = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR), a
    rng = np.random.default_rng(seed)
    w = SCAN_PRESET_WEIGHTS
    preset = str(rng.choice(list(w), p=np.array(list(w.values())) / sum(w.values())))
    # more paper around the page first (its own edge colour): rotation and perspective then turn the page
    # inside the picture instead of pushing its corners, and the words there, out of the frame
    pad = int(round(SCAN_PAD * max(bgr.shape[:2]) * (0.5 + strength)))
    bgr = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    words = [{**wd, "x": wd["x"] + pad, "y": wd["y"] + pad} for wd in words]
    out, mask = ScanSimulator(scan_config(preset, strength), seed=seed)(bgr, word_mask(bgr.shape[:2], words))
    return (Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)), mask_boxes(mask, len(words)),
            attacks + [f"scan-simulator:{preset}"])


def degrade(image, words, seed, effect, strength):
    """(image, word boxes, attacks) for one effect at one strength."""
    if effect == "ocrsmith":
        return degrade_ocrsmith(image, words, seed, strength)
    if effect == "augraphy":
        return degrade_augraphy(image, words, seed, strength)
    return degrade_scan(image, words, seed, strength, use_augraphy=effect.startswith("augraphy"))


# -- is the text still readable? ----------------------------------------------------------------------------

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


def warm_up():
    """Augraphy compiles its noise code (numba) on first use, about 15 s: do it before the first real page."""
    degrade_augraphy(Image.new("RGB", (200, 200), "white"), [], 0, DEFAULT_STRENGTH)


# ====================================================================================================================
# STEP 5 - Surya: the page's blocks and the answers
# ====================================================================================================================
# Labels in Surya OCR 2.5's own output format.
#
# Each sample is made for one of Surya's prompts, drawn per page (TASKS), and its label is the answer Surya 2.5
# gives to that prompt, written the way the model writes it. Found by asking the model itself, one image at a
# time (surya_probe/, README.md), and by Surya's own code (datalab-to/surya: inference/prompts.py,
# compact.py, parsers.py), whose prompt texts are "the model's training-time contract":
#
#   full_ocr   the page -> compact blocks, one after another with nothing between them:
#              <|Label|><|x0|><|y0|><|x1|><|y1|>content<|Label|>... The header is one label token and four coordinate
#              tokens (0-1000); a block has no </div>; </td>, </th> and </tr> are left out (</table>, </p>, </li>,
#              </ul>, </h1> are kept); merged cells are a data-kind="parent" cell with data-cs / data-rs followed by
#              data-kind="ghost" cells. Content: <h1>-<h3>, <p> (with <b>/<i>), <ul><li>• item</li></ul> (the
#              marker as drawn, inside the item), <table>, <img alt="Photo: ..."/><p class="img-desc">...</p>.
#   layout     the page -> the JSON array on one line: [{"label": "Text", "bbox": "x0 y0 x1 y1", "count": 120}, ...]
#              (Surya's parse_layout reads this; the model writes it under JSON-schema guided decoding). count is
#              the model's estimate of the tokens it needs to OCR the block: the block's answer in Surya's own
#              tokens, rounded up to tens.
#   block_ocr  one block cut out of the page -> only that block's content, compact: <p>...</p>, <h2>...</h2>,
#              <table>..., <ul>... (no label, no box). Pictures are not OCR'd block by block (Surya skips Image).
#   table      a table cut out of the page -> rows then columns as a JSON array on one line, [{"label": "Row",
#              "bbox": "x0 y0 x1 y1"}, ..., {"label": "Col", ...}] (parse_table_rec). This is Surya 2's format: the
#              2.5 model card says row and column geometry needs a Surya 2 checkpoint, and 2.5 never answered it
#              (its tables come from block_ocr). So it is 0% of the default mix; ask for it explicitly.
#
# Boxes are integers normalized to 0-1000 of the image's width and height, (x0, y0) top left. Labels used:
# Page-Header, Page-Footer, Section-Header, Text, List-Group, Table, Caption, Footnote, Image.
# The model ends every answer with its end token (<|im_end|>, shown by this server as <|im_end!>); a target
# leaves it out, as the tokenizer adds it.
#
# This module decides the blocks: their labels, tags, boxes and order, as <div data-bbox data-label> HTML (divs()).
# Surya's own encode() (pipeline/compact_original.py, its compact.py unchanged) then writes the answer:
# full_ocr = encode(divs(...)), block_ocr = encode(block html); see DocPipeline.sample().

TOKENIZER = None  # Surya 2.5's tokenizer, loaded on first use

PROMPTS = {
    # exactly as Surya's prompts.py (HIGH_ACCURACY_BBOX_PROMPT): no quote before "(x0"
    "full_ocr": "OCR this image to HTML. Each block is a div with data-label and data-bbox (x0 y0 x1 y1, normalized 0-1000).",
    "layout": 'Output the layout of this image as JSON. Each entry is a dict with "label", "bbox", and "count" fields. Bbox is x0 y0 x1 y1, normalized 0-1000.',
    "table": 'Output the table rows then columns as JSON. Each entry is a dict with "label" ("Row" or "Col") and "bbox" (x0 y0 x1 y1, normalized 0-1000).',
    "block_ocr": "OCR this block image to HTML.",
}
TASKS = {"full_ocr": 0.4, "layout": 0.25, "block_ocr": 0.35, "table": 0.0}
SUFFIX = {"full_ocr": ".target.txt", "block_ocr": ".target.txt", "layout": ".target.json", "table": ".target.json"}

ROLE_LABEL = {"title": "Section-Header", "subtitle": "Section-Header", "heading": "Section-Header",
              "table_header": "Section-Header", "header": "Page-Header", "footer": "Page-Footer",
              "page_number": "Page-Footer", "caption": "Caption", "footnote": "Footnote", "list_item": "List-Group"}
ROLE_TAG = {"title": "h1", "heading": "h2", "subtitle": "h3", "table_header": "h3"}
MARKERS = {"disc": "•", "circle": "◦", "square": "▪"}
# Picture descriptions by img_pool category (Surya describes each picture; our photos have no captions yet)
PHOTOS = {
    "animals": ("An animal in its natural surroundings.", "A photograph of an animal in its natural surroundings."),
    "landscapes": ("A wide landscape view.", "A wide landscape photograph showing terrain, sky and distant scenery."),
    "nature": ("A close view of plants and natural scenery.", "A photograph of plants and natural scenery."),
    "architecture": ("A building and its surroundings.", "A photograph of a building or structure and its surroundings."),
    "people": ("People in an everyday setting.", "A photograph of people in an everyday setting."),
    "food": ("A dish of food.", "A close-up photograph of a dish of food."),
    "sports": ("A sports scene.", "A photograph of a sports activity in progress."),
    "transport": ("A vehicle.", "A photograph of a vehicle or means of transport."),
    "patterns": ("A close-up of a texture.", "A close-up photograph of a surface texture or pattern."),
    "general": ("A general scene.", "A photograph of a general scene."),
}


def esc(text):
    return html.escape(text, quote=False)


def count(content):
    """Surya's "count" for a block: its OCR answer's length in Surya's own tokens, rounded up to tens (as Surya
    answers on 443 of 698 blocks in surya_probe/batch100; a character estimate: 411)."""
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = Tokenizer.from_file(hf_hub_download("datalab-to/surya-ocr-2.5", "tokenizer.json"))
    return max(10, math.ceil(len(TOKENIZER.encode(encode(content), add_special_tokens=False).ids) / 10) * 10)


def norm(box, width, height):
    """[x0, y0, x1, y1] in pixels -> "x0 y0 x1 y1" in 0-1000."""
    x0, y0, x1, y1 = box
    v = [round(x0 / width * 1000), round(y0 / height * 1000), round(x1 / width * 1000), round(y1 / height * 1000)]
    v = [min(1000, max(0, a)) for a in v]
    return f"{v[0]} {v[1]} {max(v[0], v[2])} {max(v[1], v[3])}"


def union(boxes):
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


# -- carrying structure boxes through the effects -----------------------------------------------------------

def carry(struct):
    """[(key, rect)] of the boxes to move with the page: tables, their rows and columns, lists, pictures (page
    pixels, {x, y, width, height}). Columns are the union of the cells that span one column."""
    out = []
    for i, t in enumerate(struct.get("tables", [])):
        out.append((("table", i), t["rect"]))
        cols = {}
        for j, row in enumerate(t["rows"]):
            out.append((("row", i, j), row["rect"]))
            k = 0
            for cell in row["cells"]:
                if cell["colspan"] == 1:
                    cols.setdefault(k, []).append(cell["rect"])
                k += cell["colspan"]
        for k, rects in sorted(cols.items()):
            x0 = min(r["x"] for r in rects)
            y0 = min(r["y"] for r in rects)
            out.append((("col", i, k), {"x": x0, "y": y0, "width": max(r["x"] + r["width"] for r in rects) - x0,
                                        "height": max(r["y"] + r["height"] for r in rects) - y0}))
    for i, lst in enumerate(struct.get("lists", [])):
        out.append((("list", i), lst["rect"]))
    for i, im in enumerate(struct.get("images", [])):
        out.append((("image", i), im["rect"]))
    return out


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


def corners(plan):
    return [pt for _, r in plan for pt in ((r["x"], r["y"]), (r["x"] + r["width"], r["y"]),
                                           (r["x"] + r["width"], r["y"] + r["height"]), (r["x"], r["y"] + r["height"]))]


def rebuild(plan, moved, width, height):
    """{key: [x0, y0, x1, y1]} after the effect (the box of its moved corners, kept to the image); a box with
    fewer than two corners left, or nothing on the image, is left out."""
    out = {}
    for n, (key, _) in enumerate(plan):
        pts = [p for p in moved[4 * n:4 * n + 4] if p]
        if len(pts) < 2:
            continue
        x0, y0 = max(0, min(p[0] for p in pts)), max(0, min(p[1] for p in pts))
        x1, y1 = min(width, max(p[0] for p in pts)), min(height, max(p[1] for p in pts))
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            out[key] = [x0, y0, x1, y1]
    return out


# -- blocks -------------------------------------------------------------------------------------------------

def blocks(regions, roles, struct, rects, lang, positioning, categories, width, height):
    """The page's blocks in reading order: [{label, box [x0, y0, x1, y1] (image pixels), html}]."""
    by_slot = {r["id"]: r for r in regions}
    styles, order = struct.get("styles", {}), struct.get("order", {})
    used, out = set(), []
    for i, t in enumerate(struct.get("tables", [])):
        grid, boxes, members = merged_grid(t["rows"]), [], []
        rows = []
        for j, row in enumerate(grid):
            cells = []
            for cell in row:
                if cell.get("ghost"):
                    cells.append((cell["tag"], ' data-kind="ghost"', "", False))
                    continue
                text = " ".join(by_slot[s]["text"] for s in cell["slots"] if s in by_slot)
                members += cell["slots"]
                inner = f"<b>{esc(text)}</b>" if text and cell["bold"] else esc(text)
                span = "".join(f' {k}="{v}"' for k, v in (("data-cs", cell["colspan"]), ("data-rs", cell.get("rowspan", 1)))
                               if v > 1)
                cells.append((cell["tag"], span + (' data-kind="parent"' if span else ""), inner, bool(text)))
            if any(c[3] for c in cells):  # a row cut off the page (no labelled text) isn't in the image
                rows.append(cells)
                boxes.append(rects.get(("row", i, j)))
        used.update(members)
        if not rows:
            continue
        box = union(boxes) or rects.get(("table", i)) or union([by_slot[s]["bbox"] for s in members if s in by_slot])
        ink = overlap(union(boxes), rects.get(("table_ink", i)))  # the rows on the image, kept to what is drawn
        body = "".join("<tr>" + "".join(f"<{tag}{attrs}>{inner}</{tag}>" for tag, attrs, inner, _ in cells) + "</tr>"
                       for cells in rows)
        out.append({"label": "Table", "box": box, "ink": ink, "html": f"<table>{body}</table>", "table": i,
                    "dom": t.get("order", 0)})
    for i, lst in enumerate(struct.get("lists", [])):
        items, members = [], []
        for k, slots in enumerate(lst["items"]):
            text = " ".join(by_slot[s]["text"] for s in slots if s in by_slot)
            members += slots
            if text:
                items.append((marker(lst["marker"], len(items) + 1, lang), text))
        used.update(members)
        if not items:
            continue
        box = rects.get(("list", i)) or union([by_slot[s]["bbox"] for s in members if s in by_slot])
        body = "".join(f"<li>{esc(m + ' ' + t if m else t)}</li>" for m, t in items)
        out.append({"label": "List-Group", "box": box, "ink": rects.get(("list_ink", i)), "html": f"<ul>{body}</ul>",
                    "dom": lst.get("order", 0)})
    for i, im in enumerate(struct.get("images", [])):
        box = rects.get(("image", i))
        if not box:
            continue
        short, long = PHOTOS.get(categories.get(im.get("src"), "general"), PHOTOS["general"])
        out.append({"label": "Image", "box": box, "html": f'<img alt="Photo: {esc(short)}"/><p class="img-desc">{esc(long)}</p>',
                    "dom": im.get("order", 0)})
    # every other labelled slot is a block of its own; a label and the value beside it are one Text block, and
    # list items outside any list markup make one List-Group
    rest = [r for r in regions if r["id"] not in used]
    k = 0
    while k < len(rest):
        r, role = rest[k], roles.get(rest[k]["id"], "paragraph")
        nxt = rest[k + 1] if k + 1 < len(rest) else None
        if role == "label" and nxt and roles.get(nxt["id"]) == "field_value" and same_line(r["bbox"], nxt["bbox"]):
            bold = (styles.get(r["id"]) or {}).get("bold"), (styles.get(nxt["id"]) or {}).get("bold")
            if bold[0] and not bold[1]:  # only the label bold: Surya writes <p><b>label:</b> value</p>
                out.append({"label": "Text", "box": union([r["bbox"], nxt["bbox"]]), "dom": order.get(r["id"], 0),
                            "html": f"<p><b>{esc(r['text'])}</b> {esc(nxt['text'])}</p>"})
            else:
                out.append(text_block("Text", f"{r['text']} {nxt['text']}", union([r["bbox"], nxt["bbox"]]),
                                      styles.get(r["id"], {}), order.get(r["id"], 0)))
            k += 2
            continue
        if role == "list_item":
            run = [r]
            while k + len(run) < len(rest) and roles.get(rest[k + len(run)]["id"]) == "list_item":
                run.append(rest[k + len(run)])
            body = "".join(f"<li>{esc(x['text'])}</li>" for x in run)
            out.append({"label": "List-Group", "box": union([x["bbox"] for x in run]), "html": f"<ul>{body}</ul>",
                        "dom": order.get(r["id"], 0)})
            k += len(run)
            continue
        label = ROLE_LABEL.get(role, "Text")
        if role == "label":  # a label with no value beside it (surya_probe/batch100): a plain one is Text to Surya
            # (74 of 83), a bold one a Section-Header, most often <h3>
            if (styles.get(r["id"]) or {}).get("bold"):
                out.append({"label": "Section-Header", "box": r["bbox"], "html": f"<h3>{esc(r['text'])}</h3>",
                            "dom": order.get(r["id"], 0)})
            else:
                out.append(text_block("Text", r["text"], r["bbox"], styles.get(r["id"], {}), order.get(r["id"], 0)))
        elif label == "Section-Header":
            tag = (styles.get(r["id"]) or {}).get("h") or ROLE_TAG.get(role, "h2")
            tag = tag if tag in ("h1", "h2", "h3") else "h3"
            out.append({"label": label, "box": r["bbox"], "html": f"<{tag}>{esc(r['text'])}</{tag}>",
                        "dom": order.get(r["id"], 0)})
        else:
            out.append(text_block(label, r["text"], r["bbox"], styles.get(r["id"], {}), order.get(r["id"], 0)))
        k += 1
    out = reading_order([b for b in out if b["box"]], positioning, lang, width, height)
    heights = sorted(r["bbox"][3] - r["bbox"][1] for r in regions)
    line = heights[len(heights) // 4] if heights else 0.02 * height  # most short slots are one line
    out = joined(edges(out, width, height), width, line, height)
    for b in out:  # grouped and ordered: now each list and table gets the box of what is drawn in it
        ink = b.pop("ink", None)
        if ink:
            b["box"] = ink
    return out


def joined(out, width, line, height=None):
    """Neighbours Surya writes as one block (surya_probe/batch100): a list right under another list (less than
    0.8 of a line apart) is one List-Group (45 of 67), text on one line almost touching another (less than 1%
    of the page width apart) is one Text (20 of 23), and short one-line texts stacked with almost no gap
    (each at most 6 words, less than 0.2 of a line apart, in the same column: label lines, short facts) are one
    Text, its lines split by <br/> (30 of 35; 3 of 31 when further apart)."""
    shift = 0.0035 * height if height else 0  # how much higher Surya's own box tops were where this was measured
    merged = []
    for b in out:
        prev = merged[-1] if merged else None
        a, c = (prev or {}).get("box"), b["box"]
        last = (prev or {}).get("last_line", a)  # a stack's lowest line
        if (prev and prev["label"] == b["label"] == "List-Group" and min(a[2], c[2]) > max(a[0], c[0])
                and c[1] - a[3] < 0.8 * line and not same_line(a, c)):
            prev["html"] = prev["html"][:-len("</ul>")] + b["html"][len("<ul>"):]
        elif (prev and prev["label"] == b["label"] == "Text" and same_line(a, c)
              and max(a[0], c[0]) - min(a[2], c[2]) < 0.01 * width):
            prev["html"] = f"<p>{inner(prev['html'])} {inner(b['html'])}</p>"
        elif (prev and (prev.get("last_line") or short_line(prev, line)) and short_line(b, line)
              and min(last[2], c[2]) - max(last[0], c[0]) >= 0.5 * min(last[2] - last[0], c[2] - c[0])
              and c[1] > last[1] + 0.5 * line and c[1] - shift - last[3] < 0.2 * line):
            prev["html"] = f"<p>{inner(prev['html'])}<br/>{inner(b['html'])}</p>"
            prev["last_line"] = c
        else:
            merged.append(dict(b))
            continue
        prev["box"] = union([a, c])
        if prev.get("ink") or b.get("ink"):
            prev["ink"] = union([prev.get("ink") or a, b.get("ink") or c])
    for b in merged:
        b.pop("last_line", None)
    return merged


def short_line(b, line):
    """A Text block of one line and at most 6 words (a single <p>)."""
    return (b["label"] == "Text" and b["html"].startswith("<p>") and b["html"].count("<p>") == 1
            and b["box"][3] - b["box"][1] < 1.6 * line
            and len(html.unescape(re.sub(r"<[^>]+>", " ", b["html"])).split()) <= 6)


def merged_grid(rows):
    """The table's rows with every merged cell written Surya 2.5's way: the cell itself (the parent, with its
    colspan / rowspan), then a ghost cell for each other grid place it covers, to its right in its row and at
    its column in the rows below."""
    out, below = [], {}  # below: {(row, column): parent} still to be filled by rowspans
    for r, row in enumerate(rows):
        line, c = [], 0
        cells = list(row["cells"])
        while cells or (r, c) in below:
            if (r, c) in below:
                line.append({"ghost": True, "tag": below.pop((r, c))["tag"]})
                c += 1
                continue
            cell = cells.pop(0)
            line.append(cell)
            for dr in range(cell.get("rowspan", 1)):
                for dc in range(cell["colspan"]):
                    if dr or dc:
                        if dr == 0:
                            line.append({"ghost": True, "tag": cell["tag"]})
                        else:
                            below[(r + dr, c + dc)] = cell
            c += cell["colspan"]
        out.append(line)
    return out


def edges(out, width, height, band=0.12):
    """Page-Header and Page-Footer only where Surya uses them, at the top or bottom of the page (a web page's
    links "footer" halfway down is Text). Items that touch are one block, joined by a space on one line and by
    <br/> across lines; items far apart stay separate blocks (as Surya writes them: a row of links is one
    header, three titles spread over the top are three)."""
    merged = []
    for b in out:
        if (b["label"] == "Page-Header" and b["box"][1] > band * height
                or b["label"] == "Page-Footer" and b["box"][3] < (1 - band) * height):
            b = {**b, "label": "Text"}
        prev = merged[-1] if merged else None
        if (prev and b["label"] == prev["label"] and b["label"] in ("Page-Header", "Page-Footer")
                and touching(prev["box"], b["box"], width)):
            sep = " " if same_line(prev["box"], b["box"]) else "<br/>"
            prev["html"] = f"<p>{inner(prev['html'])}{sep}{inner(b['html'])}</p>"
            prev["box"] = union([prev["box"], b["box"]])
            continue
        merged.append(dict(b))
    return merged


def inner(content):
    return content[3:-4] if content.startswith("<p>") and content.endswith("</p>") else content


def text_block(label, text, box, style, dom):
    inner = esc(text)
    if style.get("italic"):
        inner = f"<i>{inner}</i>"
    if style.get("bold") and label in ("Text", "Caption", "Footnote"):
        inner = f"<b>{inner}</b>"
    return {"label": label, "box": box, "html": f"<p>{inner}</p>", "dom": dom}


def marker(style, n, lang):
    if style in MARKERS:
        return MARKERS[style]
    if style == "none":
        return ""
    if style in ("decimal", "decimal-leading-zero"):
        return f"{n}."
    if style == "arabic-indic":
        return f"{n}.".translate(ARABIC_DIGITS)
    return "•"


def same_line(a, b):
    return min(a[3], b[3]) - max(a[1], b[1]) > 0.5 * min(a[3] - a[1], b[3] - b[1])


def touching(a, b, width):
    """Next to each other: on one line with a gap under 3% of the page width, or one under the other (sharing
    some width) with a gap under a line's height."""
    if same_line(a, b):
        return max(a[0], b[0]) - min(a[2], b[2]) < 0.03 * width
    line = min(a[3] - a[1], b[3] - b[1])
    return min(a[2], b[2]) > max(a[0], b[0]) and max(a[1], b[1]) - min(a[3], b[3]) < line


def reading_order(out, positioning, lang, width, height):
    """Flowing pages: document order (a block's place is that of its first labelled text, or of its element for
    pictures). Absolutely placed pages (DocLayNet): English column by column, as Surya reads them (half the
    order differences of top-to-bottom bands in surya_probe/batch100); Arabic top to bottom in bands, right to
    left (Surya doesn't read those column by column)."""
    if positioning == "absolute" and lang != "ar":
        return columns_first(out, 0.004 * width, 0.008 * height)
    if positioning == "absolute":
        heights = sorted(b["box"][3] - b["box"][1] for b in out)
        band = max(8, heights[len(heights) // 2] // 2) if heights else 8
        side = -1 if lang == "ar" else 1
        return sorted(out, key=lambda b: (b["box"][1] // band, side * b["box"][0]))
    return sorted(out, key=lambda b: b["dom"])


def columns_first(out, gap_x, gap_y):
    """Left to right in columns (an empty strip from top to bottom between them), else top to bottom in bands
    (an empty strip across), each part ordered the same way; blocks with neither left top to bottom."""
    if len(out) <= 1:
        return out
    for axis, gap in ((0, gap_x), (1, gap_y)):
        spans = sorted((b["box"][axis], b["box"][axis + 2]) for b in out)
        cuts, end = [], spans[0][1]
        for start, stop in spans[1:]:
            if start - end >= gap:
                cuts.append((end + start) / 2)
            end = max(end, stop)
        if cuts:
            parts = [[] for _ in range(len(cuts) + 1)]
            for b in out:
                parts[sum((b["box"][axis] + b["box"][axis + 2]) / 2 > x for x in cuts)].append(b)
            return [b for part in parts for b in columns_first(part, gap_x, gap_y)]
    return sorted(out, key=lambda b: (b["box"][1], b["box"][0]))


# -- the targets --------------------------------------------------------------------------------------------

def on_page(b):
    """A block's HTML as Surya writes it on a whole page: a page header's text bare, without <p> (7 of the 10
    probe pages that have one). Cut out alone (block_ocr) the model can't tell it is a header and writes <p>."""
    return inner(b["html"]) if b["label"] == "Page-Header" else b["html"]


def divs(blks, width, height):
    """The page's blocks as Surya's <div data-bbox data-label> HTML: encode() turns it into the full_ocr answer."""
    return "\n".join(f'<div data-bbox="{norm(b["box"], width, height)}" data-label="{b["label"]}">{on_page(b)}</div>'
                     for b in blks)


def layout(blks, width, height):
    """The layout answer: the JSON array on one line."""
    return json.dumps([{"label": b["label"], "bbox": norm(b["box"], width, height), "count": count(b["html"])}
                       for b in blks], ensure_ascii=False)


def on_crop(box, crop):
    """A box on the page -> "x0 y0 x1 y1" in 0-1000 of the cut-out [x0, y0, x1, y1]."""
    return norm([box[0] - crop[0], box[1] - crop[1], box[2] - crop[0], box[3] - crop[1]],
                crop[2] - crop[0], crop[3] - crop[1])


def table(rects, i, crop):
    """The table answer for table i cut out of the page: rows then columns, a JSON array on one line."""
    return json.dumps(table_structure(rects, i, crop), ensure_ascii=False)


def table_structure(rects, i, crop):
    """Rows then columns of table i on the cut-out, as the table prompt describes them."""
    rows = [rects[k] for k in sorted(k for k in rects if k[0] == "row" and k[1] == i)]
    cols = [rects[k] for k in sorted(k for k in rects if k[0] == "col" and k[1] == i)]
    return ([{"label": "Row", "bbox": on_crop(b, crop)} for b in rows]
            + [{"label": "Col", "bbox": on_crop(b, crop)} for b in cols])


LAYOUT_LABELS = {"Caption", "Footnote", "Equation-Block", "List-Group", "Page-Header", "Page-Footer", "Image",
                 "Section-Header", "Table", "Text", "Complex-Block", "Code-Block", "Form", "Table-Of-Contents", "Figure",
                 "Chemical-Block", "Diagram", "Bibliography", "Blank-Page"}  # Surya's layout JSON schema
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


def clear(crop, others, slack=2):
    """No other block reaches into the cut-out (more than slack pixels)."""
    return not any(min(crop[2], o[2]) - max(crop[0], o[0]) > slack and min(crop[3], o[3]) - max(crop[1], o[1]) > slack
                   for o in others)


def crop_box(box, width, height, rng, others=(), margin=(6, 24), share=None):
    """A cut-out around a block: its box plus a random margin in pixels or (share) a share of the box's size on
    each side, kept to the image, and on each side short of the page's other blocks (others), so no part of a
    neighbour's text, unlabelled in the target, is in the picture."""
    if share:
        mx = max(margin[0], round(rng.uniform(*share) * (box[2] - box[0])))
        my = max(margin[0], round(rng.uniform(*share) * (box[3] - box[1])))
    else:
        mx = my = rng.randint(*margin)
    x0, y0, x1, y1 = box
    left, top, right, bottom = x0 - mx, y0 - my, x1 + mx, y1 + my
    for o in others:
        if o[2] <= left or o[0] >= right or o[3] <= top or o[1] >= bottom:
            continue  # not in the cut-out
        if o[3] <= y0:
            top = max(top, o[3] + 1)
        elif o[1] >= y1:
            bottom = min(bottom, o[1] - 1)
        elif o[2] <= x0:
            left = max(left, o[2] + 1)
        elif o[0] >= x1:
            right = min(right, o[0] - 1)
        # a block overlapping this one itself (a box drawn around it) can't be kept out
    return [max(0, int(left)), max(0, int(top)), min(width, int(right + 0.999)), min(height, int(bottom + 0.999))]


# ====================================================================================================================
# STEP 6 - labels: words, lines, slots, the .md
# ====================================================================================================================

# == labels: words, lines and slots ==========================================================================

# The labels JSON of a sample: words, lines and slots with their boxes after the effects.

# Paper minus ink brightness inside word boxes (0-255). Below this the text is too faint to be a fair label.
MIN_CONTRAST = 35


def union_xywh(boxes):
    """The box [x, y, w, h] around boxes [x, y, w, h] (None if there are none)."""
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    return [x0, y0, max(b[0] + b[2] for b in boxes) - x0, max(b[1] + b[3] for b in boxes) - y0]


def whole_letters(w):
    """Keep a word unless its letters are cut. The OCR labels flag a word as clipped when its line box pokes
    out of an overflow:hidden box by half a pixel, which is every word in a tight DocLayNet box."""
    return not w["clipped"] or w["height"] >= 0.8 * w["font_size_px"]


def contrast(gray, boxes):
    """Median, over words, of paper brightness minus ink darkness inside the word box (0-255)."""
    values = [np.percentile(c, 90) - np.percentile(c, 5)
              for x, y, w, h in boxes[:200] if (c := gray[y:y + h, x:x + w]).size >= 20]
    return float(np.median(values)) if values else 0.0


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


def build_label(page, size, seed, attacks, page_contrast, ocr, slots, words, word_boxes, cut):
    """The page's labels: word boxes as given, lines and slots rebuilt from their words. A line's or a slot's
    text is its labelled words only, so a word that is cut, hidden or lost is in no text either."""
    def group(keep):
        kept = [(w, word_boxes[i]) for i, w in enumerate(words) if word_boxes[i] and keep(w)]
        return " ".join(w["text"] for w, _ in kept), union_xywh([b for _, b in kept])

    def inside(w, a):  # word centre inside the slot's original box
        cx, cy = w["x"] + w["width"] / 2, w["y"] + w["height"] / 2
        return a["x"] <= cx <= a["x"] + a["width"] and a["y"] <= cy <= a["y"] + a["height"]
    lines = [dict(zip(("id", "text", "box"), (ln["id"], *group(lambda w, ln=ln: w["line_id"] == ln["id"]))))
             for ln in ocr["lines"]]
    slot_boxes = [dict(zip(("entity_type", "property", "text", "box"),
                           (a["entity_type"], a.get("rdfa_property"), *group(lambda w, a=a: inside(w, a)))))
                  for a in slots]
    return {"page": page, "width": size[0], "height": size[1], "seed": seed,
            "attacks": attacks, "contrast": round(page_contrast, 1), "no_text": not any(word_boxes),
            "low_contrast": any(word_boxes) and page_contrast < MIN_CONTRAST,
            "words": [{"id": w["id"], "text": w["text"], "line_id": w["line_id"], "box": b}
                      for w, b in zip(words, word_boxes) if b],
            "lines": [ln for ln in lines if ln["box"]], "slots": [s for s in slot_boxes if s["box"]],
            "lost_words": sum(b is None for b in word_boxes), "cut_words_dropped": cut}


# Slot-level ground truth: one box and one text per filled slot, in reading order.
# 
# regions() groups a page's labelled words by the template slot they are in (the word measuring records each
# word's data-slot): a region's text is its words in reading (logical) order, its box the box around their boxes
# after the effects. Reading order: the page's document order for flowing layouts (what a reader follows down
# the columns); for DocLayNet's positioned boxes, top to bottom, then right to left (Arabic) or left to right.
# 
# to_text() writes the format the OCR model learns to produce, one tag per region, the box at the end, as
# x1, y1, x2, y2 (top-left and bottom-right corners) in the image's pixels:
# 
#     <text>المملكة العربية السعودية <bbox>50, 20, 450, 65</bbox></text>
#     <text>وزارة الداخلية <bbox>50, 100, 300, 140</bbox></text>

def join_word_parts(pieces, text):
    """The slot's words as written in its text. The browser may break a long word across two lines (CSS
    overflow-wrap: anywhere in a narrow box), and each part is then measured as a word: "historica" "lly".
    When the measured parts spell exactly the slot's text, the text's own spacing is used ("historically");
    otherwise the parts are join_word_parts with spaces."""
    tokens = (text or "").split()
    if not tokens or "".join(pieces) != "".join(tokens):
        return " ".join(pieces)
    out, i = [], 0
    for token in tokens:
        part = ""
        while i < len(pieces) and len(part) < len(token):
            part += pieces[i]
            i += 1
        if part != token:
            return " ".join(pieces)
        out.append(token)
    return " ".join(out)


def slot_regions(meta_slots, words, boxes, lang, positioning, texts=None):
    """texts: each slot's text as the page holds it (the QA script's slots), to keep a word whole when the
    browser broke it across lines."""
    roles = {s["id"]: s["role"] for s in meta_slots}
    texts = texts or {}
    groups, order = {}, []
    for w, box in zip(words, boxes):
        slot = w.get("slot")
        if not box or slot not in roles:
            continue
        if slot not in groups:
            groups[slot] = []
            order.append(slot)
        groups[slot].append((w["text"], box))
    out = []
    for slot in order:
        items = groups[slot]
        x1 = min(b[0] for _, b in items)
        y1 = min(b[1] for _, b in items)
        x2 = max(b[0] + b[2] for _, b in items)
        y2 = max(b[1] + b[3] for _, b in items)
        out.append({"id": slot, "role": roles[slot], "text": join_word_parts([t for t, _ in items], texts.get(slot)),
                    "box": [x1, y1, x2 - x1, y2 - y1], "bbox": [x1, y1, x2, y2]})
    if positioning == "absolute" and out:
        band = max(8, sorted(r["box"][3] for r in out)[len(out) // 2] // 2)
        side = -1 if lang == "ar" else 1
        out.sort(key=lambda r: (r["bbox"][1] // band, side * r["bbox"][0]))
    return out


def to_text(regs):
    """The tagged ground truth: one <text> line per region, text first, its box at the end."""
    lines = []
    for r in regs:
        x1, y1, x2, y2 = (int(round(v)) for v in r["bbox"])
        text = r["text"].replace("<", "&lt;").replace(">", "&gt;")  # only what would break a tag
        lines.append(f"<text>{text} <bbox>{x1}, {y1}, {x2}, {y2}</bbox></text>")
    return "\n".join(lines) + "\n"


# ====================================================================================================================
# DocPipeline: STEP 1 to STEP 6 for one page
# ====================================================================================================================
# The class runs the steps above for one page: sample(). Search for STEP 1 ... STEP 6 below.

BANK2000 = ROOT / "bank2000"
LANGUAGES = {"ar": 0.5, "en": 0.5}
# "clean": some pages stay exactly as SynthDoc rendered them (born-digital documents); the rest are split evenly
# between the three tools ("augraphy+scan-simulator" is there when asked for)
EFFECTS = {"clean": 0.15, "ocrsmith": 0.85 / 3, "augraphy": 0.85 / 3, "scan-simulator": 0.85 / 3}
STRENGTHS = {tool: DEFAULT_STRENGTH for tool in TOOLS}
ATTEMPTS = 8         # renders per sample until one passes the checks (a new template every second one)
EFFECT_TRIES = 3     # seeds per strength level before the strength is lowered
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
    (STEP 5; default TASKS)."""

    def __init__(self, out_dir, languages=None, effects=None, sources=None, templates="bank", llm=None,
                 pictures=None, text=None, extras=True, bank=None, strengths=None, check=True, tashkeel=None,
                 tasks=None):
        self.out = Path(out_dir).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.bank = Path(bank) if bank else BANK2000
        self.languages, self.effects = languages or LANGUAGES, effects or EFFECTS
        self.sources, self.templates, self.llm = sources, templates, llm
        self.strengths = {**STRENGTHS, **(strengths or {})}
        self.tashkeel = tashkeel or TASHKEEL  # Arabic pages: none / partial / full marks
        self.tasks = tasks or TASKS
        self.pictures = ImagePool() if pictures is None else pictures
        self.categories = {f.name: c for c, files in getattr(self.pictures, "by_category", {}).items() for f in files}
        self.text = text or ("wikipedia" if TextPool.exists("ar") and TextPool.exists("en") else "demo")
        self.extras, self.check = extras, check
        synthdoc_arabic.install()  # SynthDoc then also measures every word (the Arabic extension)
        self._synthdoc = None

    # -- STEP 1: template --------------------------------------------------------------------------------------

    def template(self, rng, name, lang, need_table=False):
        """(meta, template text or None, folder with its files, note). need_table: only bank templates with a
        table (the Surya "table" task)."""
        note = ""
        if self.templates == "llm":
            if self.llm is not None and self.llm.available():
                from pipeline.llm_templates import adapt  # same form as the bank: language, <img> picture slots
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

    # -- STEP 2: fill --------------------------------------------------------------------------------------------

    def fill(self, meta, text, folder, rng, lang):
        """(page for SynthDoc, values for its slots, pictures and style, record of everything picked)."""
        if lang == "ar":
            s, css = style(rng, arabic_fonts())
        else:
            s, css = style_latin(rng)
        look, theme_css, theme_js = theme(rng, meta["source"])
        words = text_pool(lang) if self.text == "wikipedia" else None
        if words:
            words.begin(rng)  # one article per page
        values = {slot["id"]: Markup(fill(slot, rng, s["digits"], lang, words)) for slot in meta["slots"]}
        level = pick(rng, self.tashkeel) if lang == "ar" else "none"  # one level per page
        if level != "none":
            values = {k: Markup(add_tashkeel(str(v), level, rng)) for k, v in values.items()}
            # marks above and below the letters need room between lines, as in printed marked text
            s = {**s, "line_height": max(s["line_height"], TASHKEEL_LINE_HEIGHT[level])}
            css += f"\n:root {{ --line-height: {s['line_height']}; }}"
        used = []
        for pic in meta.get("pictures", []):  # <img src="{{ pic_N }}"> slots: a photo of the hinted kind
            photo = self.pictures.pick(rng, pic.get("hint")) if self.pictures else None
            values[pic["id"]] = Markup(photo.as_uri() if photo else "")
            if photo:
                used.append(photo.name)
        page = self.page_script(synthdoc_html(meta, text, lang, folder), rng, theme_js)
        s = {**s, "theme": look, "tashkeel": level, "pictures": used, "text": self.text,
             "text_sources": words.sources() if words else []}
        # The style travels with the page, so SynthDoc's word labels record it (the Arabic extension reads this)
        variant = {"template": meta["id"], "source": meta["source"], "lang": lang, **s}
        tag = Markup.escape(json.dumps(variant, ensure_ascii=False))
        page = page.replace("</head>", f'<meta name="synthdoc-ar-variant" content="{tag}">\n</head>', 1)
        return page, {**values, "style": Markup(css + theme_css)}, s

    def page_script(self, page, rng, theme_js):
        """The page with PAGE_JS (STEP 3) added: the theme's colours and 12 photos for empty boxes."""
        config = {"photos": [self.pictures.pick(rng).as_uri() for _ in range(12)] if self.pictures else [],
                  "hue": 0, "saturate": 1.0, **(theme_js or {})}
        script = PAGE_JS.replace("SYNTH_CONFIG", json.dumps(config))
        return page.replace("</head>", f"<script>{script}</script>\n</head>", 1)

    # -- STEP 3: render --------------------------------------------------------------------------------------------

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

    # -- STEP 4: effect -------------------------------------------------------------------------------------------

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
                out, boxes, attacks = degrade(clean, words + dots, rng.randrange(2**31), effect, strength)
                boxes, dot = boxes[:n], [(b[0] + b[2] / 2, b[1] + b[3] / 2) if b else None for b in boxes[n:]]
                moved = [dot[where[(round(x), round(y))]] for x, y in same]
                check = legibility(clean, out, words, boxes)
                if readable(check):
                    return out, boxes, attacks, strength, check, moved
        return (clean, exact, ["clean (effect left text unreadable)"], 0.0, {"lost": 0, "faint": 0.0, "sharpness": 1.0},
                same)

    # -- one sample, many samples ---------------------------------------------------------------------------

    def sample(self, seed, name=None, language=None, effect=None, task=None):
        """Make one page: STEP 1 to STEP 6. Returns a summary dict (the files are written to out_dir). An error
        says which step it happened in: "sample s07 failed in STEP 4 effect: ..." (the original error follows)."""
        name = name or f"s{seed}"
        at = ["STEP 1 template"]
        try:
            return self._make(seed, name, language, effect, task, at)
        except Exception as e:
            raise RuntimeError(f"sample {name} failed in {at[0]}: {type(e).__name__}: {e}") from e

    def _make(self, seed, name, language, effect, task, at):
        """The steps of sample(); at[0] is kept set to the step running."""
        rng = random.Random(seed)
        lang, effect = language or pick(rng, self.languages), effect or pick(rng, self.effects)
        task = task or pick(rng, self.tasks)  # the Surya prompt this sample is made for
        t = {"template": 0, "fill": 0, "render": 0}
        rejected, attempts = {}, 0
        for attempts in range(1, ATTEMPTS + 1):  # until the rendered page passes every check
            if attempts % 2 == 1:  # a new template every second attempt
                at[0] = "STEP 1 template"
                mark = time.perf_counter()
                meta, text, folder, note = self.template(rng, name, lang, need_table=task == "table")
                lang = meta.get("lang", lang)
                t["template"] += time.perf_counter() - mark
            at[0] = "STEP 2 fill"
            mark = time.perf_counter()
            page, values, s = self.fill(meta, text, folder, rng, lang)
            t["fill"] += time.perf_counter() - mark
            at[0] = "STEP 3 render"
            mark = time.perf_counter()
            clean, ocr, slots, pages, qa = self.render(page, values, name)
            t["render"] += time.perf_counter() - mark
            at[0] = "STEP 3 page checks"
            issues = ({**render_issues(ocr, qa), **ink_issues(clean, ocr["words"])} if self.check
                      else ({} if ocr["words"] else {"empty": 1}))
            if self.check and ocr["words"]:  # a word the ground truth would leave out (split_words: clipped by
                dropped = split_words(ocr)[1]  # its box or the page edge) is shown on the image: make the page again
                if dropped:
                    issues["cut"] = issues.get("cut", 0) + dropped
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
        at[0] = "STEP 4 effect"
        mark = time.perf_counter()
        plan = carry(qa.get("struct") or {})  # tables, rows, columns, lists, pictures: moved with the page
        plan = to_ink(plan, np.asarray(clean.convert("L")))  # lists and tables: only what is drawn
        image, boxes, attacks, strength, check, moved = self.degrade(clean, words, rng, effect, corners(plan))
        at[0] = "STEP 6 labels"
        label = build_label(name, image.size, None, attacks, contrast(np.asarray(image.convert("L")), [b for b in boxes if b]),
                            ocr, slots, words, boxes, cut)
        regions = slot_regions(meta["slots"], words, boxes, lang, meta.get("positioning", "flow"), qa.get("slots"))
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
        md = to_text(regions)  # the Markdown is the ground truth: one <text>... <bbox>...</bbox></text> per block
        # Surya: the page's blocks, then the input image and target of this sample's prompt
        at[0] = "STEP 5 Surya blocks and answer"
        width, height = image.size
        rects = rebuild(plan, moved, width, height)
        blks = blocks(regions, {sl["id"]: sl["role"] for sl in meta["slots"]}, qa.get("struct") or {}, rects,
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
                c = crop_box(b["box"], width, height, rng, others,
                                   share=(0.03, 0.12) if task == "table" else None)
                if clear(c, others):
                    cuts.append((b, c))
            if not cuts:
                note, task = (note + "; " if note else "") + f"no {task} block that cuts out clean: full_ocr instead", "full_ocr"
        crop, target, structure = None, None, None
        if task == "full_ocr":
            target = encode(divs(blks, width, height))
        elif task == "layout":
            target = layout(blks, width, height)
        elif task == "block_ocr":  # one block cut out (6-24 px margin), its compact content as the answer
            b, crop = rng.choice(cuts)
            target = encode(b["html"])  # the block's content only
        else:  # table: a table cut out with a 3-12% margin, its rows then columns
            b, crop = rng.choice(cuts)
            target = table(rects, b["table"], crop)
            structure = table_structure(rects, b["table"], crop)
        label["surya"] = {"task": task, "prompt": PROMPTS[task], "target": target, "crop": crop,
                          "table_structure": structure,
                          "page_size": [width, height], "divs": divs(blks, width, height),
                          "blocks": [{"label": b["label"], "box": [round(v) for v in b["box"]], "html": b["html"]}
                                     for b in blks]}
        t["markdown"] = time.perf_counter() - mark

        at[0] = "STEP 6 writing the files"
        out = self.out
        (image.crop(crop) if crop else image).save(out / f"{name}.jpg", quality=90)  # what the model is shown
        if crop:
            image.save(out / f"{name}_page.jpg", quality=90)  # the whole page (the .md boxes are on it)
        (out / f"{name}.prompt.txt").write_text(PROMPTS[task], encoding="utf-8", newline="\n")
        (out / f"{name}{SUFFIX[task]}").write_text(target, encoding="utf-8", newline="\n")
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
                "task": task, "target_file": f"{name}{SUFFIX[task]}", "blocks": len(blks), "crop": crop,
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


def draw(image, boxes, color, path):
    out = image.copy()
    d = ImageDraw.Draw(out)
    for x, y, w, h in boxes:
        d.rectangle([x, y, x + w, y + h], outline=color, width=2)
    out.save(path, quality=82)


