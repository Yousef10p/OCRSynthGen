"""SynthDoc template class for exported bank templates (imported by each bank_<id>_template/__init__.py).

generate_sample_data() fills every slot with our sample Arabic text and picks a random style, exactly as
demo/render_demo.py does, but on a white page: SynthDoc renders and labels it, there is no paper or
scan effect here. Values are passed as Markup because SynthDoc's Jinja autoescapes, which would break
the quotes in the style CSS.
"""
import html
import json
import os
import random
import sys
import zlib
from pathlib import Path

from markupsafe import Markup
from synthdoc.templates.base import BaseTemplate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
from fill_style import fill, font_families, style  # noqa: E402

FAMILIES = font_families()


class BankTemplateBase(BaseTemplate):
    _variant = None

    def get_archetype(self) -> str:
        return "report"

    def get_category(self) -> str:
        return "business"

    def _slots(self):
        return json.loads(self._load_file("slots.json", required=True))

    def _next_seed(self) -> int:
        """Deterministic per document: BANK_SYNTHDOC_SEED + template + document index."""
        cls = type(self)
        index = cls.__dict__.get("_doc_index", 0)
        cls._doc_index = index + 1
        base = int(os.environ.get("BANK_SYNTHDOC_SEED", "0"))
        return base * 100_000_000 + (zlib.crc32(self._slots()["id"].encode()) % 100_000) * 100 + index

    def generate_sample_data(self, seed=None):
        info = self._slots()
        seed = self._next_seed() if seed is None else seed
        rng = random.Random(seed)
        s, css = style(rng, FAMILIES, transparent=False)
        values = {slot["id"]: Markup(fill(slot, rng, s["digits"])) for slot in info["slots"]}
        self._variant = {"template": info["id"], "source": info["source"], "seed": seed, **s}
        return {**values, "style": Markup(css)}

    def get_template_string(self) -> str:
        page = self._load_file("template.html", required=True)
        # The style choices travel with the page, so the OCR labels record them (Arabic extension reads this)
        meta = html.escape(json.dumps(self._variant or {}, ensure_ascii=False), quote=True)
        return page.replace("</head>", f'<meta name="synthdoc-ar-variant" content="{meta}">\n</head>', 1)
