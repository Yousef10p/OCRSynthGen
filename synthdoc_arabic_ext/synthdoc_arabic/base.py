"""
Shared plumbing for the Arabic templates.

``ArabicTemplateMixin`` goes in front of a SynthDoc ``BaseTemplate`` subclass. It
seeds an :class:`ArabicData` per document, picks a visual variant (Arabic font,
accent colour, page size) and builds the template string from:

1. the original SynthDoc template's ``styles.css`` (when ``BASE_TEMPLATE`` is set),
   so ported templates keep their layout;
2. ``assets/rtl_common.css`` - direction, Arabic-safe typography;
3. the template folder's own ``styles.css`` - RTL overrides or a full stylesheet;
4. the variant's ``@font-face`` rules and CSS variables.
"""
import inspect
import json
import os
import random
import zlib
from pathlib import Path
from typing import Optional

from .data import ArabicData

ASSETS = Path(__file__).parent / "assets"
FONT_DIR = ASSETS / "fonts"

# family name -> (CSS family, [(file, weight)], generic fallback)
FONTS = {
    "amiri": ("SD Amiri", [("Amiri-Regular.ttf", "400"), ("Amiri-Bold.ttf", "700")], "serif"),
    "noto_naskh": ("SD Noto Naskh Arabic", [("NotoNaskhArabic-VF.ttf", "400 700")], "serif"),
    "cairo": ("SD Cairo", [("Cairo-VF.ttf", "200 1000")], "sans-serif"),
    "tajawal": ("SD Tajawal", [("Tajawal-Regular.ttf", "400"), ("Tajawal-Bold.ttf", "700")], "sans-serif"),
    "ibm_plex": ("SD IBM Plex Sans Arabic", [("IBMPlexSansArabic-Regular.ttf", "400"),
                                             ("IBMPlexSansArabic-Bold.ttf", "700")], "sans-serif"),
    "noto_kufi": ("SD Noto Kufi Arabic", [("NotoKufiArabic-VF.ttf", "100 900")], "sans-serif"),
}

ACCENTS = ["#1f4e79", "#0b6e4f", "#7a1f2b", "#2c3e50", "#5b2c83", "#8a5a00"]


def font_face_css(key: str) -> str:
    family, files, _ = FONTS[key]
    rules = []
    for filename, weight in files:
        url = (FONT_DIR / filename).resolve().as_uri()
        rules.append(
            f"@font-face {{ font-family: '{family}'; src: url('{url}') format('truetype'); "
            f"font-weight: {weight}; font-style: normal; }}"
        )
    return "\n".join(rules)


def font_stack(key: str) -> str:
    family, _, generic = FONTS[key]
    # Noto Naskh is always loaded as the fallback for any glyph the main font lacks
    return f"'{family}', 'SD Noto Naskh Arabic', {generic}"


class ArabicTemplateMixin:
    """Arabic data, fonts and RTL styling for a SynthDoc template."""

    BASE_TEMPLATE = None              # original SynthDoc template class whose CSS is reused
    FONT_CHOICES = ["noto_naskh"]
    PAGE_SIZES = ["A4"]

    _variant = None

    # -- per-document setup ---------------------------------------------------
    def _next_seed(self) -> int:
        """Deterministic per-document seed: SYNTHDOC_AR_SEED + template + document index."""
        cls = type(self)
        index = cls.__dict__.get("_doc_index", 0)
        cls._doc_index = index + 1
        base = int(os.environ.get("SYNTHDOC_AR_SEED", "2026"))
        return base * 100_000 + (zlib.crc32(cls.__name__.encode()) % 1000) * 100 + index

    def arabic_setup(self, seed: Optional[int] = None) -> ArabicData:
        if seed is None:
            seed = self._next_seed()
        random.seed(seed)
        self.fake.seed_instance(seed)
        ar = ArabicData(seed)
        rng = random.Random(seed ^ 0x5EED)
        self._variant = {
            "seed": seed,
            "font": rng.choice(self.FONT_CHOICES),
            "accent": rng.choice(ACCENTS),
            "page_size": rng.choice(self.PAGE_SIZES),
            "eastern_digits": ar.eastern,
            "country": ar.country_code,
        }
        return ar

    @property
    def variant(self) -> dict:
        if self._variant is None:
            self._variant = {"seed": None, "font": self.FONT_CHOICES[0], "accent": ACCENTS[0],
                             "page_size": self.PAGE_SIZES[0]}
        return self._variant

    # -- template string ------------------------------------------------------
    def _base_css(self) -> str:
        if self.BASE_TEMPLATE is None:
            return ""
        path = Path(inspect.getfile(self.BASE_TEMPLATE)).parent / "styles.css"
        return path.read_text(encoding="utf-8")

    def _variant_css(self) -> str:
        v = self.variant
        fonts = {v["font"], "noto_naskh"}
        faces = "\n".join(font_face_css(k) for k in sorted(fonts))
        return (
            f"{faces}\n"
            f":root {{ --ar-font: {font_stack(v['font'])}; --ar-accent: {v['accent']}; }}\n"
            f"@page {{ size: {v['page_size']}; }}\n"
        )

    def get_template_string(self) -> str:
        html = self._load_file("template.html", required=True)
        css = "\n".join([
            "/* ---- original SynthDoc template CSS ---- */",
            self._base_css(),
            "/* ---- Arabic: common RTL rules ---- */",
            (ASSETS / "rtl_common.css").read_text(encoding="utf-8"),
            "/* ---- Arabic: template styles ---- */",
            self._load_file("styles.css"),
            "/* ---- Arabic: document variant ---- */",
            self._variant_css(),
        ])
        # The variant travels with the page so the OCR ground truth can record it
        meta = json.dumps({"template": type(self).__name__, **self.variant})
        head = f"<meta name=\"synthdoc-ar-variant\" content='{meta}'>\n<style>\n{css}\n</style>\n</head>"
        return html.replace("</head>", head, 1)
