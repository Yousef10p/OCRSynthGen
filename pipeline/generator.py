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
                       answer to the sample's prompt, written by Surya's own encode() (originals/surya)
    STEP 6  files      image, prompt, answer, labels JSON, .md, .html

If something fails, the error says in which step (DocPipeline.sample). The checks and fixes around these steps
(page checks, readability after the effect, box fixes, render speed) are in checking.py. SynthDoc (+ its
Arabic extension), OCRSmith, Augraphy, scan-simulator and Surya's encoder and prompts are the unchanged copies
in originals/.
"""
import copy
import html
import json
import math
import os
import random
import re
import time
import urllib.parse
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from jinja2 import Template
from markupsafe import Markup
from ocrsmith.core.backgrounds import gradient_background, image_background, paper_background, solid_background
from ocrsmith.core.degradations import build_preset
from ocrsmith.domain.annotations import Line, Page, Region, RegionType, Word
from ocrsmith.domain.geometry import BBox
from ocrsmith.text.script import Direction
from PIL import Image, ImageDraw, ImageFont
from scan_simulator import ScanSimulator
from scan_simulator.config import PipelineConfig
from scan_simulator.presets import PRESETS as SCAN_PRESETS
from scipy.ndimage import find_objects
from synthdoc.ner_data_generator import NERDataGenerator
from synthdoc_arabic import pipeline as synthdoc_arabic
from synthdoc_arabic.ocr_gt import build_page_gt
from surya.inference.compact import encode  # Surya's own encoder: writes every answer
from surya.inference.prompts import BLOCK_PROMPT, HIGH_ACCURACY_BBOX_PROMPT, LAYOUT_PROMPT, TABLE_REC_PROMPT
from tokenizers import Tokenizer

from pipeline.checking import (PAGE_JS, RENDER_SCALE, ink_issues, legibility, overlap, readable, render_issues,
                               split_words, to_ink)
from pipeline.pools import ImagePool, TextPool, font_pool

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

OCRSMITH = ROOT / "originals" / "ocrsmith_assets"  # OCRSmith's fonts and paper backgrounds
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


def font_faces(families, names):
    """@font-face rules for the families that come as files ({} faces: installed, used by name)."""
    return "".join(
        f'@font-face {{ font-family: "{fam}"; src: url("{path.as_uri()}"); font-weight: {w};'
        f' font-style: {"italic" if it else "normal"}; }}' + "\n"
        for fam in sorted(names) for (w, it), path in families.get(fam, {}).items())


def style(rng, families):
    """Random typography for one page: (its settings, CSS for the template's {{ style }} hook)."""
    body_family = rng.choice(sorted(families))
    head_family = body_family if rng.random() < 0.7 else rng.choice(sorted(families))
    body_w = pick_weight(rng, families[body_family], [300, 400, 400, 500])
    head_w = pick_weight(rng, families[head_family], [600, 700, 700, 800, 900])
    s = page_style(rng, body_family, head_family, body_w, head_w)
    return s, font_faces(families, {body_family, head_family}) + style_css(s)


def style_latin(rng):
    """The same random typography for an English page: installed Latin fonts and the font pool's (every family
    there has Latin letters, the Arabic ones too)."""
    families = latin_fonts()
    body_family = rng.choice(sorted(families))
    head_family = body_family if rng.random() < 0.7 else rng.choice(sorted(families))
    s = page_style(rng, body_family, head_family, 400, 700, "western")
    # Arabic letter numbering in lists means nothing on an English page
    return s, font_faces(families, {body_family, head_family}) + style_css(s) + "ol { list-style-type: decimal !important; }\n"


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


def hex_rgb(color):
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def blend(a, b, t):
    """Colour a moved toward colour b by t (0 = a, 1 = b)."""
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(hex_rgb(a), hex_rgb(b)))


def hsl(h, s, l):
    """A colour from hue (degrees), saturation and lightness (0-1)."""
    c = (1 - abs(2 * l - 1)) * s
    x, m = c * (1 - abs((h / 60) % 2 - 1)), l - c / 2
    r, g, b = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x)][int(h // 60) % 6]
    return "#" + "".join(f"{round((v + m) * 255):02x}" for v in (r, g, b))


def luminance(color):
    def f(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (f(v) for v in hex_rgb(color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a, b):
    """WCAG contrast between two colours: 1 (same) to 21 (black on white)."""
    la, lb = luminance(a), luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def readable_on(color, background, least):
    """color, or black or white (whichever reads better) if it has less than `least` contrast."""
    if contrast_ratio(color, background) >= least:
        return color
    return max(("#111111", "#ffffff"), key=lambda c: contrast_ratio(c, background))


def colour_page(rng):
    """A coloured page: (background, text ink, accent, light tint, stronger tint, border mix). Dark (40%) or
    coloured light / mid (60%) background of any hue; text at least 7:1 against it, the accent at least 4.5:1,
    and the tints that boxes and table rows get keep the text at least 4.5:1 too."""
    hue = rng.uniform(0, 360)
    dark = rng.random() < 0.4
    sat = rng.uniform(0.15, 0.6) if dark else rng.uniform(0.25, 0.75)
    light = rng.uniform(0.10, 0.30) if dark else rng.uniform(0.62, 0.88)
    page = hsl(hue, sat, light)
    # a mid colour no text reaches 7:1 on: made darker (dark page) or lighter until black or white text does
    while max(contrast_ratio("#ffffff", page), contrast_ratio("#111111", page)) < 7.0:
        light = light - 0.02 if dark else light + 0.02
        page = hsl(hue, sat, light)
    if dark:
        ink = rng.choice(["#ffffff", "#f5f5f5", "#fdf6e3", hsl(hue, rng.uniform(0.2, 0.5), rng.uniform(0.92, 0.97))])
    else:
        ink = rng.choice(THEME_INKS + [hsl(hue, rng.uniform(0.3, 0.7), rng.uniform(0.08, 0.15))])
    ink = readable_on(ink, page, 7.0)
    accent_hue = (hue + rng.choice([0, 30, 150, 180, 210, 330])) % 360
    accent = hsl(accent_hue, rng.uniform(0.5, 0.85), rng.uniform(0.70, 0.85) if dark else rng.uniform(0.18, 0.32))
    accent = readable_on(accent, page, 4.5)
    soft, tint = blend(page, ink, rng.uniform(0.05, 0.08)), blend(page, ink, rng.uniform(0.10, 0.15))
    if contrast_ratio(ink, tint) < 4.5:
        soft, tint = blend(page, ink, 0.04), blend(page, ink, 0.07)
    return page, ink, accent, soft, tint, lambda color, t: blend(color, page, t)


COLOUR_SHARE = 0.5  # share of pages in colour (the rest light); DocPipeline(colour_share=...), the website's slider


def theme(rng, source, colour_share=COLOUR_SHARE):
    """(record, CSS, settings for the page script). A share of the pages (1 - colour_share) is light: white,
    off-white or a light tint, with near-black text. The rest are in colour (colour_page): any hue, dark or
    bright, with text and accents picked to stay readable on it. After that the page script still checks every
    text's contrast and fixes anything under 4.5:1, and the page checks reject anything under 3:1."""
    if rng.random() >= colour_share:  # light page, near-black text
        accent = rng.choice(ACCENTS)
        tint, soft = mix(accent, rng.uniform(0.88, 0.94)), mix(accent, rng.uniform(0.95, 0.975))
        page_kind = rng.choices(["white", "off-white", "tint"], weights=[5, 3, 2])[0]
        page_bg = {"white": "#ffffff", "off-white": rng.choice(["#fbfaf7", "#fafafa", "#f9f8f4"]), "tint": soft}[page_kind]
        ink, faded = rng.choice(THEME_INKS), mix
    else:  # coloured page, readable text
        page_bg, ink, accent, soft, tint, faded = colour_page(rng)
        page_kind = "colour"
    rec = {"accent": accent, "heading": rng.choice(["plain", "accent", "underline", "bar", "band", "rule-above"]),
           "table": rng.choice(["grid", "striped", "header-tint", "rules", "boxed"]),
           "box": rng.choice(["tint", "bar", "outline", "plain"]), "page": page_kind, "page_color": page_bg,
           "bullet": rng.choice(["disc", "square", "circle"]), "radius": rng.choice([0, 0, 3, 6]),
           "paragraph_gap": round(rng.uniform(0.3, 1.0), 2), "heading_gap": round(rng.uniform(0.5, 1.4), 2),
           "ink": ink}
    imp = " !important"
    css = [f"ol, ul {{ list-style-type: {rec['bullet']}{imp}; }}",
           # a list laid out in a row (a menu: flex, grid or inline items) has no room for a marker outside its
           # items: it would be drawn into the previous item's words. There the marker goes inside its own item.
           '[class~="flex"] > li, [class~="inline-flex"] > li, [class~="grid"] > li, [class*=":flex"] > li, '
           '[class*=":grid"] > li, [style*="flex"] > li, [style*="grid"] > li, li[class~="inline-block"], '
           f'li[class~="inline"], li[style*="inline"] {{ list-style-position: inside{imp}; }}',
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
               "band": f"color: {readable_on(accent, tint, 4.5)}{imp}; background-color: {tint}{imp}; padding-inline: 0.3em;",
               "rule-above": f"color: {rec['ink']}{imp}; border-block-start: 2px solid {accent}{imp};"}[rec["heading"]]
    if source == "doclaynet":  # fixed-size boxes: colours only, nothing that takes room
        heading = heading.replace("padding-inline-start: 0.4em;", "").replace("padding-inline: 0.3em;", "")
        heading = heading.replace("border-block-end: 2px", "border-block-end: 1px").replace(
            "border-block-start: 2px", "border-block-start: 1px").replace("border-inline-start: 4px", "border-inline-start: 3px")
    th = readable_on(accent, tint if rec["table"] == "header-tint" else soft if rec["table"] == "boxed" else page_bg, 4.5)
    css += [f"h1, h2, h3, h4, h5, h6, dt, strong {{ {heading} }}",
            f"body, .page, p, li, td, dd, span, div {{ color: {rec['ink']}; }}",
            f"th {{ color: {th}{imp}; }}"]
    table = {"grid": f"th, td {{ border-color: {faded(accent, 0.55)}{imp}; }}",
             "striped": f"tbody tr:nth-child(even) td {{ background-color: {soft}{imp}; }} "
                        f"th {{ border-block-end: 2px solid {accent}{imp}; }}",
             "header-tint": f"th {{ background-color: {tint}{imp}; }} th, td {{ border-color: {faded(accent, 0.6)}{imp}; }}",
             "rules": f"th, td {{ border-inline: none{imp}; border-block-start: none{imp}; "
                      f"border-block-end: 1px solid {faded(accent, 0.6)}{imp}; }}",
             "boxed": f"table {{ outline: 2px solid {accent}; }} th {{ background-color: {soft}{imp}; }}"}[rec["table"]]
    box = {"tint": f"aside, blockquote {{ background-color: {soft}{imp}; border-color: {faded(accent, 0.5)}{imp}; }}",
           "bar": f"aside, blockquote {{ border-inline-start: 4px solid {accent}{imp}; }}",
           "outline": f"aside, blockquote {{ border-color: {accent}{imp}; }}",
           "plain": ""}[rec["box"]]
    css += [table, box, f"hr {{ border-color: {faded(accent, 0.5)}; }}"]
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
# The marks are drawn but never in the labels: the words are measured with their marks (the boxes hold them),
# then the marks are taken out of every text the labels are made from (no_marks, in DocPipeline._make). The model
# learns to read the letters whether a page is marked or not; Surya's answers have the letters only.

TASHKEEL = {"none": 0.4, "partial": 0.3, "full": 0.3}
ARABIC = re.compile(r"[ء-ي]")
MARKS = re.compile(r"[ً-ْٰ]")
VOWELS = re.compile(r"[ً-ِْٰ]")  # every mark but shadda (U+0651)
PARTIAL_WORDS = 0.35
ALL_MARKS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")  # tashkeel and Quranic marks


def no_marks(text):
    return ALL_MARKS.sub("", text)
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
# SynthDoc (originals/synthdoc) lays the page out with Paged.js in Chromium, takes a screenshot and boxes
# every slot; its Arabic extension (originals/synthdoc_arabic) measures every word. synthdoc_html() turns a
# bank template into the page SynthDoc renders; DocPipeline.render() runs it. How it runs faster and at 2x,
# what else the page measures, and the checks of the result are in checking.py.

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
# After the effect checking.legibility() checks every word is still readable; if not, DocPipeline tries again
# with new seeds, then with less strength.

TOOLS = ("ocrsmith", "augraphy", "scan-simulator")
DEFAULT_STRENGTH = 0.5
MAX_ANGLE = 2.0  # a page is never turned more than this many degrees (OCRSmith and scan-simulator Rotation)
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
        if name == "Rotation":
            d.max_angle = min(d.max_angle, MAX_ANGLE)
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
        if name == "Rotation" and "angle" in params:
            a = params["angle"]
            params["angle"] = ([max(-MAX_ANGLE, min(MAX_ANGLE, x)) for x in a] if isinstance(a, (list, tuple))
                               else max(-MAX_ANGLE, min(MAX_ANGLE, a)))
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
# This step decides the blocks: their labels, tags, boxes and order, as <div data-bbox data-label> HTML (divs()).
# Surya's own encode() (originals/surya/inference/compact.py, unchanged) then writes the answer:
# full_ocr = encode(divs(...)), block_ocr = encode(block html); see DocPipeline.sample().

TOKENIZER = None  # Surya 2.5's tokenizer, loaded on first use

# Surya's own prompts (originals/surya/inference/prompts.py)
PROMPTS = {"full_ocr": HIGH_ACCURACY_BBOX_PROMPT, "layout": LAYOUT_PROMPT, "table": TABLE_REC_PROMPT, "block_ocr": BLOCK_PROMPT}
TASKS = {"full_ocr": 0.4, "layout": 0.25, "block_ocr": 0.35, "table": 0.0}
SUFFIX = {"full_ocr": ".target.txt", "block_ocr": ".target.txt", "layout": ".target.json", "table": ".target.json"}

ROLE_LABEL = {"title": "Section-Header", "subtitle": "Section-Header", "heading": "Section-Header",
              "table_header": "Section-Header", "header": "Page-Header", "footer": "Page-Footer",
              "page_number": "Page-Footer", "caption": "Caption", "footnote": "Footnote", "list_item": "List-Group"}
ROLE_TAG = {"title": "h1", "heading": "h2", "subtitle": "h3", "table_header": "h3"}
MARKERS = {"disc": "•", "circle": "◦", "square": "▪"}
# Picture descriptions by img_pool category, for a picture without captions (an older pool)
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


# What kind of picture the captions describe -> Surya's alt prefix and how its description starts (Surya writes
# "Photo:" for 129 of 133 pictures in surya_probe/batch100, "Artwork:" and "Illustration:" for the rest). The kind
# is what the picture is, so only the start of a caption counts ("a painting of a man", not "girls pass a painting").
PICTURE_KINDS = [("Artwork", "An artwork showing", re.compile(
                     r"^(?:an? |the )?(?:\w+ ){0,2}(?:painting|artwork|sculpture|statue|engraving|etching|lithograph|"
                     r"fresco|mosaic)s?\b", re.I)),
                 ("Illustration", "An illustration showing", re.compile(
                     r"^(?:an? |the )?(?:\w+ ){0,2}(?:illustration|drawing|sketch|cartoon|watercolou?r)s?\b", re.I))]
CAPTION_LEAD = re.compile(r"^(the|this|an?) (image|picture|photo|photograph|scene) (shows|depicts|is of|features) ", re.I)


def picture_html(captions, category="general"):
    """A picture's block as Surya writes it, from the picture's own captions: <img alt="Photo: <the shortest
    caption, one sentence>"/><p class="img-desc"><the longest caption></p>. A caption written as "The image shows
    X" (PD12M) becomes "A photograph showing X", Surya's most common start; a person's caption, a sentence of its
    own, is kept as written. Artwork / Illustration when the caption starts with one. A picture without captions
    gets its category's description."""
    if not captions:
        short, long = PHOTOS.get(category, PHOTOS["general"])
        return f'<img alt="Photo: {esc(short)}"/><p class="img-desc">{esc(long)}</p>'
    short, long = min(captions, key=len), max(captions, key=len)
    led = bool(CAPTION_LEAD.match(long))
    short, long = CAPTION_LEAD.sub("", short), CAPTION_LEAD.sub("", long)
    kind, lead = next(((k, l) for k, l, start in PICTURE_KINDS if start.match(long) or start.match(short)),
                      ("Photo", "A photograph showing"))
    short = re.split(r"(?<=[.!?]) ", short)[0]  # the alt text: one sentence
    short = short[:1].upper() + short[1:]
    if led:  # "The image shows a man ..." -> "A photograph showing a man ..."
        long = f"{lead} {long[:1].lower() + long[1:] if not long[1:2].isupper() else long}"
    else:
        long = long[:1].upper() + long[1:]
    return f'<img alt="{kind}: {esc(short)}"/><p class="img-desc">{esc(long)}</p>'

# -- special blocks: every other label Surya writes ----------------------------------------------------------
#
# Surya 2.5 writes 19 block labels; the templates' own slots give Text, Section-Header, List-Group, Table, Image,
# Caption, Footnote, Page-Header and Page-Footer. The rest are added here, each in the form Surya's own answers give
# it (surya_inference/debug_docling_outputs: real pages Surya read):
#
#   picture boxes (GRAPHIC_SHARE of them) get a drawn picture from img_pool/graphics/ instead of a photo:
#     Figure          a chart: <img alt="Bar chart: ..."/>, legend, axes, the data as <table class="chart-data">
#     Diagram         a flowchart: <img alt="Flowchart: ..."/>, its words, its Mermaid source, a description
#     Chemical-Block  a molecule: <img alt="Chemical structure of the molecule SMILES."/>
#   a paragraph's box (on SPECIAL_SHARE of pages, one or two) gets a block instead of its text:
#     Equation-Block  an equation drawn from LaTeX: <math display="block">LaTeX</math>
#     Code-Block      Python code (from Python's own library): <pre>code</pre>
#     Bibliography    numbered references: <ul style="list-style-type: none; padding-left: 0;"><li>[1] ...</li></ul>
#     Table-Of-Contents  entries and page numbers: <table><tr><td>1<td>Title<td>5 ...
#     Form            labels, blank lines and check boxes: <table><tr><td><b>1</b> Name: ____ ...
#     Complex-Block   an infographic panel (a photo, a figure, a sentence): <img alt="Photo: ..."/><p><b>42%</b></p>...
#   BLANK_SHARE of samples are a blank page (paper and the effect only): one block, 0 0 1000 1000, no content.
#
# Surya never wrote Code-Block or Complex-Block on a whole page in those answers (code came back as Text with
# <pre>; Code-Block with <pre> came in its layout answer), so their content follows that layout answer and Surya's
# general picture form. "Picture" is not generated: Surya's model has no such label (its code renames the model's
# Image to Picture), and teaching both for one thing would make them a coin toss.
#
# What the label says is what the block shows: the block is drawn from the same HTML (a code line, a table cell)
# its label is written from; a block that doesn't fit its box (a DocLayNet box has a fixed size) or is split by a
# page break makes the page fail its checks, and it is made again.

GRAPHIC_SHARE = 0.35
GRAPHIC_KINDS = {"Figure": 0.45, "Diagram": 0.3, "Chemical-Block": 0.25}
SPECIAL_SHARE = 0.5
SPECIAL_KINDS = {"Equation-Block": 1.0, "Code-Block": 1.0, "Bibliography": 0.8, "Table-Of-Contents": 0.7,
                 "Form": 0.8, "Complex-Block": 0.6}
BLANK_SHARE = 0.01
NO_CUT = {"Image", "Figure", "Diagram", "Chemical-Block", "Blank-Page"}  # pictures: never a block_ocr cut-out
SPECIAL_SLOT = r'<(p|div|blockquote|section|article|aside)(\s[^>]*)?>\s*<span[^>]*data-slot="{slot}"[^>]*>\{{\{{ {slot} \}}\}}</span>\s*</\1>'
SPECIAL_CSS = (  # a special block's own look, whatever the theme does to tables and lists
    "[data-special] { break-inside: avoid; }\n"
    ".sp-toc td, .sp-toc th { border: none !important; background: transparent !important; padding: 0.12em 0.4em !important; }\n"
    ".sp-toc td:last-child { text-align: end; }\n"
    ".sp-form { border-collapse: collapse; inline-size: 100%; }\n"
    ".sp-form td { border: 1px solid currentColor !important; padding: 0.3em 0.5em !important; vertical-align: top; }\n"
    'img[src*="/graphics/"] { object-fit: contain !important; background: #fff; }\n')
NAMES_EN = (["A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.", "J.", "K.", "L.", "M.", "N.", "P.", "R.", "S.", "T.", "W."],
            ["Smith", "Johnson", "Chen", "Garcia", "Kumar", "Müller", "Rossi", "Tanaka", "Brown", "Martin", "Lee", "Novak",
             "Haddad", "Silva", "Kowalski", "Nguyen", "Dubois", "Ahmed"])
NAMES_AR = (["أحمد", "محمد", "سارة", "خالد", "فاطمة", "عبدالله", "نورة", "يوسف", "مريم", "عمر", "ليلى", "حسن"],
            ["العتيبي", "الزهراني", "المصري", "الحربي", "القحطاني", "الشمري", "السيد", "الكعبي", "النجار", "الخطيب"])
VENUES = {"en": ["Journal of Applied Physics", "Proceedings of the National Academy of Sciences", "IEEE Transactions on Pattern Analysis",
                 "Nature Communications", "Journal of Economic Studies", "Computational Linguistics", "The Lancet", "Water Research",
                 "Annual Review of Sociology", "Journal of Modern History"],
          "ar": ["مجلة العلوم الإنسانية", "المجلة العربية للعلوم", "مجلة جامعة الملك سعود", "مجلة الدراسات الاقتصادية",
                 "المجلة العلمية للتربية", "مجلة البحوث الهندسية", "مجلة اللغة العربية وآدابها", "مجلة العلوم الصحية"]}
FORM_WORDS = {"en": {"yes": "Yes", "no": "No", "options": [["Single", "Married"], ["Full time", "Part time", "Contract"],
                                                         ["Email", "Phone", "Post"], ["Male", "Female"]],
                     "sign": "Signature", "date": "Date"},
              "ar": {"yes": "نعم", "no": "لا", "options": [["أعزب", "متزوج"], ["دوام كامل", "دوام جزئي", "عقد"],
                                                         ["البريد الإلكتروني", "الهاتف", "البريد"], ["ذكر", "أنثى"]],
                     "sign": "التوقيع", "date": "التاريخ"}}
STATS = ["42%", "3.5x", "1,250", "87%", "12 million", "2 of 3", "65%", "4,800", "9.2", "300+"]
_code = []


def code_snippets():
    """Short Python functions from Python's own library (PSF licence): 4 to 14 lines, none longer than 64
    characters, so a block never runs off its box."""
    if not _code:
        lib = Path(os.__file__).parent
        for path in sorted(lib.glob("*.py"))[:200]:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            for i, line in enumerate(lines):
                if not line.startswith("def ") and not line.startswith("    def "):
                    continue
                indent = len(line) - len(line.lstrip())
                body = [line]
                for nxt in lines[i + 1:i + 15]:
                    if nxt.strip() and len(nxt) - len(nxt.lstrip()) <= indent:
                        break
                    body.append(nxt)
                body = [b[indent:] for b in body]
                while body and not body[-1].strip():
                    body.pop()
                if 4 <= len(body) <= 14 and all(len(b) <= 64 and "\t" not in b and '"""' not in b for b in body):
                    _code.append("\n".join(body))
    return _code


def toc(rng, lang, words, digits):
    """(render, label) of a table of contents: chapters in bold, sections under them, page numbers going up."""
    n, page, rows, chapter = rng.randint(5, 12), rng.randint(1, 9), [], 0
    three = rng.random() < 0.5  # number | title | page, or "number title . . . . ." | page
    dots = not three and rng.random() < 0.7
    for i in range(n):
        top = i == 0 or rng.random() < 0.3
        if top:
            chapter, section = chapter + 1, 0
            num = str(chapter)
        else:
            section += 1
            num = f"{chapter}.{section}"
        title = html.escape(words.text("heading", rng.randint(2, 6), rng) if words else phrase(rng, rng.randint(2, 5), WORDS_EN if lang == "en" else WORDS))
        page += rng.randint(1, 6)
        num_s, page_s = (num.translate(ARABIC_DIGITS), str(page).translate(ARABIC_DIGITS)) if digits == "arabic-indic" else (num, str(page))
        b = (lambda t: f"<b>{t}</b>") if top else (lambda t: t)
        cells = [b(num_s), b(title), b(page_s)] if three else [b(f"{num_s} {title}" + (" . . . . ." if dots else "")), b(page_s)]
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    body = "".join(rows)
    return f'<table class="sp-toc" style="inline-size: 100%">{body}</table>', f"<table>{body}</table>"


def bibliography(rng, lang, words, digits):
    first, last = NAMES_AR if lang == "ar" else NAMES_EN
    items = []
    for i in range(rng.randint(3, 7)):
        people = [f"{rng.choice(first)} {rng.choice(last)}" for _ in range(rng.randint(1, 3))]
        joined = "، ".join(people) if lang == "ar" else (", ".join(people[:-1]) + (", and " if len(people) > 2 else " and ") + people[-1]
                                                          if len(people) > 1 else people[0])
        title = (words.text("heading", rng.randint(4, 9), rng) if words else phrase(rng, rng.randint(4, 8), WORDS_EN if lang == "en" else WORDS)).rstrip(".")
        year = str(rng.randint(1985, 2024))
        n = str(i + 1)
        if digits == "arabic-indic":
            year, n = year.translate(ARABIC_DIGITS), n.translate(ARABIC_DIGITS)
        sep = "، " if lang == "ar" else ", "
        items.append(f"<li>[{n}] {html.escape(joined)}. {html.escape(title)}. <i>{html.escape(rng.choice(VENUES[lang]))}</i>{sep}{year}.</li>")
    body = "".join(items)
    return (f'<ul style="list-style-type: none !important; padding: 0 !important; margin: 0.4em 0">{body}</ul>',
            f'<ul style="list-style-type: none; padding-left: 0;">{body}</ul>')


def form(rng, lang, words, digits):
    """(render, label) of a form: numbered fields with a blank line, check boxes, and a signature and date row."""
    w = FORM_WORDS[lang]
    rows = []
    for i in range(rng.randint(2, 5)):
        num = str(i + 1).translate(ARABIC_DIGITS) if digits == "arabic-indic" else str(i + 1)
        name = html.escape((words.text("label", 2, rng) if words else phrase(rng, 2, WORDS_EN if lang == "en" else WORDS) + ":"))
        if rng.random() < 0.4:
            options = rng.choice(w["options"] + [[w["yes"], w["no"]]])
            boxes = "   ".join(f'<input type="checkbox"{" checked" if rng.random() < 0.3 else ""}/> {html.escape(o)}' for o in options)
            rows.append(f"<tr><td><b>{num}</b> {name}<br/>{boxes}</td></tr>")
        else:
            rows.append(f"<tr><td><b>{num}</b> {name} {'_' * rng.randint(12, 24)}</td></tr>")
    rows.append(f"<tr><td><b>{w['sign']}</b> {'_' * 14}</td><td><b>{w['date']}</b> {'_' * 10}</td></tr>")
    body = "".join(rows)
    return f'<table class="sp-form">{body}</table>', f"<table>{body}</table>"


def complex_panel(rng, lang, words, pictures, info):
    """(render, label) of an infographic panel: a photo, a big figure and a sentence side by side."""
    photo = pictures.pick(rng) if pictures else None
    if photo is None:
        return None
    stat = rng.choice(STATS)
    text = html.escape(words.text("caption", rng.randint(8, 16), rng) if words else prose(rng, rng.randint(8, 14), SENTENCES_EN if lang == "en" else SENTENCES))
    alt = re.search(r'<img alt="([^"]*)"/>', info.get(photo.name, ("Image", '<img alt="Photo: A general scene."/>'))[1]).group(1)
    tint = rng.choice(["#eef4fb", "#fdf3e7", "#eef8f0", "#f6eefb", "#f3f3f3"])
    render = (f'<div style="display: flex; gap: 0.8em; align-items: center; padding: 0.6em; background: {tint};'
              f' border-radius: 6px"><img src="{photo.as_uri()}" alt="" style="inline-size: 32%; block-size: auto;'
              f' border-radius: 4px"><div><p style="font-size: 2.2em; font-weight: 700; margin: 0">{stat}</p>'
              f'<p style="margin: 0.2em 0 0">{text}</p></div></div>')
    return render, f'<img alt="{alt}"/><p><b>{stat}</b></p><p>{text}</p>'


def special_block(kind, rng, lang, words, digits, pictures, info, mono):
    """(render HTML, label HTML) of one special block, or None if it can't be made."""
    if kind == "Equation-Block":
        g = pictures.graphic(rng, "Equation-Block", lang) if pictures else None
        if not g:
            return None
        path, r = g
        return (f'<div style="text-align: center; margin: 0.5em 0"><img src="{path.as_uri()}" alt=""'
                f' style="block-size: {r["height_em"]}em; inline-size: auto"></div>', r["html"])
    if kind == "Code-Block":
        code = html.escape(rng.choice(code_snippets()), quote=False)
        bg = rng.choice(["#f6f8fa", "#f4f4f4", "#ffffff", "#fbf7ee"])
        # no ligatures: a code font would draw -> as an arrow and <= as one sign, and the label has the characters
        return (f'<pre dir="ltr" style="font-family: \'{mono}\', monospace; font-size: 0.8em; line-height: 1.35;'
                f' font-variant-ligatures: none; font-feature-settings: \'liga\' 0, \'calt\' 0, \'dlig\' 0;'
                f' white-space: pre; text-align: left; margin: 0.4em 0; padding: 0.5em 0.7em; background: {bg};'
                f' border: 1px solid #ccc">{code}</pre>', f"<pre>{code}</pre>")
    if kind == "Table-Of-Contents":
        return toc(rng, lang, words, digits)
    if kind == "Bibliography":
        return bibliography(rng, lang, words, digits)
    if kind == "Form":
        return form(rng, lang, words, digits)
    return complex_panel(rng, lang, words, pictures, info)


def add_specials(page, meta, rng, lang, words, digits, pictures, info, mono):
    """The page with one or two paragraph boxes holding a special block instead: (page, [{label, html}])."""
    slots = [s["id"] for s in meta["slots"] if s["role"] in ("paragraph", "quote", "sidebar")]
    rng.shuffle(slots)
    specials = []
    for slot in slots:
        if len(specials) >= (2 if rng.random() < 0.3 else 1):
            break
        m = re.search(SPECIAL_SLOT.format(slot=slot), page)
        if not m:
            continue
        kind = pick(rng, SPECIAL_KINDS)
        made = special_block(kind, rng, lang, words, digits, pictures, info, mono)
        if not made:
            continue
        render, label = made
        k = len(specials)
        specials.append({"label": kind, "html": label})
        div = f'<div{m.group(2) or ""} data-special="{k}">{{% raw %}}{render}{{% endraw %}}</div>'
        page = page[:m.start()] + div + page[m.end():]
    return page, specials



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
    pixels, {x, y, width, height}). Columns are the union of the cells that span one column; a place taken by a
    cell merged down from a row above (rowspan) is skipped, so the next cells keep their own column."""
    out = []
    for i, t in enumerate(struct.get("tables", [])):
        out.append((("table", i), t["rect"]))
        cols, taken = {}, set()  # taken: (row, column) covered by a rowspan from a row above
        for j, row in enumerate(t["rows"]):
            out.append((("row", i, j), row["rect"]))
            k = 0
            for cell in row["cells"]:
                while (j, k) in taken:
                    k += 1
                if cell["colspan"] == 1:
                    cols.setdefault(k, []).append(cell["rect"])
                for dr in range(1, cell.get("rowspan", 1)):
                    taken.update((j + dr, k + dc) for dc in range(cell["colspan"]))
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
    for i, sp in enumerate(struct.get("specials", [])):
        out.append((("special", i), sp["rect"]))
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

def blocks(regions, roles, struct, rects, lang, pictures, width, height, specials=()):
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
        label, content = pictures.get(urllib.parse.unquote(im.get("src") or ""), ("Image", picture_html([])))
        out.append({"label": label, "box": box, "html": content, "dom": im.get("order", 0)})
    for i, sp in enumerate(struct.get("specials", [])):  # equations, code, contents, references, forms, panels
        box = rects.get(("special_ink", i)) or rects.get(("special", i))
        if box and sp["k"] < len(specials):
            out.append({"label": specials[sp["k"]]["label"], "box": box, "html": specials[sp["k"]]["html"],
                        "dom": sp.get("order", 0)})
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
    paths = struct.get("paths", {})  # the containers each block sits in (keyed by its document order)
    frames, size = struct.get("frames", {}), struct.get("size") or [0, 0]  # its element's box on the clean page
    sx, sy = (width / size[0], height / size[1]) if size[0] and size[1] else (0, 0)
    scaled = lambda f: [f["x"] * sx, f["y"] * sy, (f["x"] + f["width"]) * sx, (f["y"] + f["height"]) * sy] if f and sx else None
    containers = {int(k): scaled(v) for k, v in struct.get("boxes", {}).items()}
    for b in out:
        b["path"] = paths.get(str(b.get("dom")), [])
        b["frame"] = scaled(frames.get(str(b.get("dom"))))
        b["containers"] = containers
    out = reading_order([b for b in out if b["box"]], lang, width, height)
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


def reading_order(out, lang, width, height):
    """One rule for every page: the page's visual groups first, then what is inside each group, each time cut along
    empty strips, rows first (a strip across: the part above is read before the part below), then columns, Arabic
    right to left and English left to right (xy_cut, mirrored for Arabic).

    The groups are the template's containers (b["path"]: the containers a block sits in, from the page script): a
    column, a card or a sidebar is read whole before the next. Cutting the bare blocks instead lets an empty strip
    that happens to line up across a row of cards (under their pictures, say) read all the pictures, then all the
    headings; with other text the strip is gone and the cards are read one by one. On 730 bank templates rendered
    twice, that made the order change with the text on 303. Inside a container whose text flows through columns
    (column-count: a negative id in the path) the order is the flow: down one column, then the next. Blocks without
    a path are cut as one group.

    Boxes for the order are the containers' own boxes and the blocks' frames (the element's box on the clean
    page), not the ink: the ink of a short text ends higher, and the effects move it, so cuts made on ink come and
    go with the text. A DocLayNet region has a fixed size (its container), a card is measured as the whole card.
    A block without a frame uses its box."""
    gx, gy = 0.004 * width, 0.008 * height
    where = (lambda b: b.get("frame") or b["box"]) if all(b.get("frame") for b in out) else (lambda b: b["box"])

    def box_of(key, members):  # a group is measured by its container's own box (fixed by the layout) when known
        box = isinstance(key, int) and (members[0].get("containers") or {}).get(key)
        return box or union([where(b) for b in members])
    flip = (lambda b: [width - b[2], b[1], width - b[0], b[3]]) if lang == "ar" else (lambda b: b)

    def order(blocks, depth):
        groups = {}
        for b in blocks:  # the container at this depth; a block with no deeper container stands alone
            path = b.get("path") or []
            groups.setdefault(path[depth] if len(path) > depth else ("block", id(b)), []).append(b)
        if len(groups) == 1 and len(blocks) > 1:  # one wrapper around everything: look inside it
            return order(blocks, depth + 1)
        units = [{"box": flip(box_of(key, g)), "members": g} for key, g in groups.items()]
        parent = (blocks[0].get("path") or [0])[depth - 1] if depth else 0
        if parent < 0:  # text flowing through columns: in its flow (document) order
            units.sort(key=lambda u: min(b.get("dom", 0) for b in u["members"]))
        else:
            units = xy_cut(units, gx, gy, rows_first=True)
        return [b for u in units for b in (order(u["members"], depth + 1) if len(u["members"]) > 1 else u["members"])]

    return order(out, 0)


def xy_cut(out, gap_x, gap_y, rows_first=False):
    """Blocks cut into parts along empty strips: columns (an empty strip from top to bottom, parts left to right)
    and rows (an empty strip across, parts top to bottom), columns tried first or rows first, each part ordered
    the same way; blocks that can't be cut, top to bottom then left to right."""
    if len(out) <= 1:
        return out
    for axis, gap in (((1, gap_y), (0, gap_x)) if rows_first else ((0, gap_x), (1, gap_y))):
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
            return [b for part in parts for b in xy_cut(part, gap_x, gap_y, rows_first)]
    return sorted(out, key=lambda b: (b["box"][1], b["box"][0]))


# -- the targets --------------------------------------------------------------------------------------------

def on_page(b):
    """A block's HTML as Surya writes it on a whole page: a page header's text bare, without <p> (7 of the 10
    probe pages that have one). Cut out alone (block_ocr) the model can't tell it is a header and writes <p>."""
    return inner(b["html"]) if b["label"] == "Page-Header" else b["html"]


# How Surya 2.5 draws a block's box compared with the box on the text: its mean edge shift (x0, y0, x1, y1) in
# thousandths of the page, measured on 1,219 blocks of surya_probe/batch100 (full_ocr + layout, Surya's box paired
# with ours at IoU >= 0.7). The left and top edges sit further out; it does not grow with the letters' size.
SURYA_SHIFT = {"Image": (-4.1, -1.7, -1.4, -1.1), "other": (-6.9, -3.0, 2.0, -0.8)}
TEXT_MARGIN = 3.0  # then a text block's box gets this much room on every side (thousandths of the page)


def as_surya(b, width, height):
    """A block's box as Surya writes it (SURYA_SHIFT) with some room around text (TEXT_MARGIN), for the answers'
    block headers only."""
    image = b["label"] == "Image"
    shift = SURYA_SHIFT["Image" if image else "other"]
    room = 0 if image else TEXT_MARGIN
    return [v + (s + (room if k >= 2 else -room)) * (width if k % 2 == 0 else height) / 1000
            for k, (v, s) in enumerate(zip(b["box"], shift))]


def divs(blks, width, height):
    """The page's blocks as Surya's <div data-bbox data-label> HTML: encode() turns it into the full_ocr answer."""
    return "\n".join(f'<div data-bbox="{norm(as_surya(b, width, height), width, height)}" data-label="{b["label"]}">'
                     f'{on_page(b)}</div>' for b in blks)


def layout(blks, width, height):
    """The layout answer: the JSON array on one line."""
    return json.dumps([{"label": b["label"], "bbox": norm(as_surya(b, width, height), width, height),
                        "count": count(b["html"])} for b in blks], ensure_ascii=False)


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


def whole_table(rects, i, n_rows, crop, slack=2):
    """Every row and column of table i (n_rows rows) is on the image and in the cut-out: a table running off the
    page (or past its block's box) would give rows that are not in the picture."""
    parts = [rects[k] for k in rects if k[0] in ("row", "col") and k[1] == i]
    rows = sum(k[0] == "row" and k[1] == i for k in rects)
    return bool(parts) and rows == n_rows and all(r[0] >= crop[0] - slack and r[1] >= crop[1] - slack and r[2] <= crop[2] + slack
                               and r[3] <= crop[3] + slack and r[2] - r[0] > 2 and r[3] - r[1] > 2 for r in parts)


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


def contrast(gray, boxes):
    """Median, over words, of paper brightness minus ink darkness inside the word box (0-255)."""
    values = [np.percentile(c, 90) - np.percentile(c, 5)
              for x, y, w, h in boxes[:200] if (c := gray[y:y + h, x:x + w]).size >= 20]
    return float(np.median(values)) if values else 0.0


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
    """Families with Arabic letters: OCRSmith's and the font pool's (pools.py, font_pool/)."""
    if "fonts" not in _cache:
        families = font_families()
        for fam, f in font_pool().items():
            if "arabic" in f["scripts"]:
                families.setdefault(fam, {}).update({(w, False): path for w, path in f["files"].items()})
        _cache["fonts"] = families
    return _cache["fonts"]


def latin_fonts():
    """Families for English pages: {name: {}} for the installed ones, files for the font pool's (no monospace)."""
    if "latin" not in _cache:
        families = {name: {} for name in LATIN_FONTS}
        for fam, f in font_pool().items():
            if f["category"] != "Monospace":
                families[fam] = {(w, False): path for w, path in f["files"].items()}
        _cache["latin"] = families
    return _cache["latin"]


def mono_fonts():
    """Monospace families for code blocks: the font pool's, else the installed Consolas."""
    if "mono" not in _cache:
        families = {fam: {(w, False): path for w, path in f["files"].items()}
                    for fam, f in font_pool().items() if f["category"] == "Monospace"}
        _cache["mono"] = families or {"Consolas": {}}
    return _cache["mono"]


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
                 tasks=None, colour_share=COLOUR_SHARE, graphic_share=GRAPHIC_SHARE, special_share=SPECIAL_SHARE,
                 blank_share=BLANK_SHARE):
        self.out = Path(out_dir).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.bank = Path(bank).resolve() if bank else BANK2000
        self.languages, self.effects = languages or LANGUAGES, effects or EFFECTS
        self.sources, self.templates, self.llm = sources, templates, llm
        self.strengths = {**STRENGTHS, **(strengths or {})}
        self.tashkeel = tashkeel or TASHKEEL  # Arabic pages: none / partial / full marks
        self.tasks = tasks or TASKS
        self.colour_share = colour_share  # share of pages in colour, the rest light (theme)
        self.pictures = ImagePool() if pictures is None else pictures
        captions = getattr(self.pictures, "captions", {})  # file name -> (label, its HTML in the answers)
        self.picture_info = {f.name: ("Image", picture_html(captions.get(f.name, []), c))
                             for c, files in getattr(self.pictures, "by_category", {}).items() for f in files}
        for label, by_lang in getattr(self.pictures, "graphics", {}).items():
            for items in by_lang.values():
                self.picture_info.update({path.name: (label, r["html"]) for path, r in items})
        self.graphic_share, self.special_share, self.blank_share = graphic_share, special_share, blank_share
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

    def fill(self, meta, text, folder, rng, lang, specials=True):
        """(page for SynthDoc, values for its slots, pictures and style, record of everything picked)."""
        if lang == "ar":
            s, css = style(rng, arabic_fonts())
        else:
            s, css = style_latin(rng)
        look, theme_css, theme_js = theme(rng, meta["source"], self.colour_share)
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
        for pic in meta.get("pictures", []):  # <img src="{{ pic_N }}"> slots: a photo of the hinted kind, or a
            photo = None  # chart, flowchart or molecule (special blocks above)
            if self.pictures and rng.random() < self.graphic_share:
                drawn = self.pictures.graphic(rng, pick(rng, GRAPHIC_KINDS), lang)
                photo = drawn[0] if drawn else None
            if photo is None and self.pictures:
                photo = self.pictures.pick(rng, pic.get("hint"))
            values[pic["id"]] = Markup(photo.as_uri() if photo else "")
            if photo:
                used.append(photo.name)
        page, specials = synthdoc_html(meta, text, lang, folder), []
        if rng.random() < self.special_share and specials:
            mono = rng.choice(sorted(mono_fonts()))
            page, specials = add_specials(page, meta, rng, lang, words, s["digits"], self.pictures, self.picture_info, mono)
            css += font_faces(mono_fonts(), {mono})
        css += SPECIAL_CSS
        page = self.page_script(page, rng, theme_js)
        s = {**s, "theme": look, "tashkeel": level, "pictures": used, "text": self.text,
             "specials": specials, "text_sources": words.sources() if words else []}
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
        if rng.random() < self.blank_share:
            return self.blank(rng, seed, name, lang, effect, task)
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
            # special blocks only in the first half of the attempts: a page that keeps failing with them (a block
            # too big for its box, say) is then made without them, so every sample can be made
            page, values, s = self.fill(meta, text, folder, rng, lang, specials=attempts <= ATTEMPTS // 2)
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
        for w in words:  # tashkeel is drawn, never labelled: every text below is made from these words
            w["text"] = no_marks(w["text"])
        if qa.get("slots"):
            qa["slots"] = {k: no_marks(v) for k, v in qa["slots"].items()}
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
                      lang, self.picture_info, width, height, s.get("specials", []))
        for b in blks:  # a special block's text (references, a form) may have marks of its own
            b["html"] = no_marks(b["html"])
        # a cut-out (block_ocr: one block, not a picture; table: a table) shows its block and nothing of another:
        # blocks whose cut-out would take in part of a neighbour (a turned page's upright boxes overlap) are not
        # used; none left: full_ocr instead
        cuts = []
        if task in ("block_ocr", "table"):
            for b in blks:
                if b["label"] in NO_CUT or (task == "table" and b["label"] != "Table"):
                    continue
                others = [x["box"] for x in blks if x is not b]
                c = crop_box(b["box"], width, height, rng, others,
                                   share=(0.03, 0.12) if task == "table" else None)
                if clear(c, others) and (task != "table" or whole_table(
                        rects, b["table"], len((qa.get("struct") or {}).get("tables", [])[b["table"]]["rows"]), c)):
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

    def blank(self, rng, seed, name, lang, effect, task):
        """A blank page (blank_share of the samples): paper in a page colour, the effect, and one Blank-Page block
        over the whole page with nothing in it, as Surya answers a blank page."""
        mark = time.perf_counter()
        w_mm, h_mm = page_dims(rng.choice(list(PAGE_MM)), rng.choice(["portrait", "portrait", "landscape"]))
        size = (round(w_mm / 25.4 * 96 * RENDER_SCALE), round(h_mm / 25.4 * 96 * RENDER_SCALE))
        look = theme(rng, "grammar", self.colour_share)[0]
        image, attacks, strength = Image.new("RGB", size, hex_rgb(look["page_color"])), ["clean"], 0.0
        if effect != "clean":
            tool = "augraphy" if effect == "augraphy" else ("ocrsmith" if effect == "ocrsmith" else "scan-simulator")
            strength = self.strengths[tool]
            image, _, attacks = degrade(image, [], rng.randrange(2**31), effect, strength)
        task = task if task in ("full_ocr", "layout") else "full_ocr"  # nothing on it to cut out
        width, height = image.size
        blks = [{"label": "Blank-Page", "box": [0, 0, width, height], "html": ""}]
        target = encode(divs(blks, width, height)) if task == "full_ocr" else layout(blks, width, height)
        label = {"page": name, "width": width, "height": height, "seed": seed, "attacks": attacks, "words": [],
                 "lines": [], "slots": [], "lost_words": 0, "cut_words_dropped": 0, "template": "blank",
                 "source": "blank", "lang": lang, "effect": effect, "strength": strength, "style": {"theme": look},
                 "surya": {"task": task, "prompt": PROMPTS[task], "target": target, "crop": None, "table_structure": None,
                           "page_size": [width, height], "divs": divs(blks, width, height),
                           "blocks": [{"label": "Blank-Page", "box": [0, 0, width, height], "html": ""}]}}
        image.save(self.out / f"{name}.jpg", quality=90)
        (self.out / f"{name}.prompt.txt").write_text(PROMPTS[task], encoding="utf-8", newline="\n")
        (self.out / f"{name}{SUFFIX[task]}").write_text(target, encoding="utf-8", newline="\n")
        (self.out / f"{name}.md").write_text("", encoding="utf-8")
        (self.out / f"{name}.json").write_text(json.dumps(label, ensure_ascii=False), encoding="utf-8")
        seconds = round(time.perf_counter() - mark, 2)
        return {"name": name, "seed": seed, "template": "blank", "source": "blank", "lang": lang, "task": task,
                "target_file": f"{name}{SUFFIX[task]}", "blocks": 1, "crop": None, "effect": effect, "strength": strength,
                "attacks": attacks, "font": None, "size_pt": None, "theme": look, "tashkeel": "none", "contrast": None,
                "legibility": None, "words": 0, "regions": 0, "lost_words": 0, "pages_rendered": 0, "note": "blank page",
                "attempts": 1, "rejected": {}, "pictures": 0, "text": None, "text_sources": 0,
                "seconds": {"template": 0, "fill": 0, "render": 0, "effects": seconds, "markdown": 0, "total": seconds}}

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
             + [x["rect"] for x in st.get("lists", []) + st.get("images", []) + st.get("specials", [])])
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


