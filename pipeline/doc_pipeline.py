"""DocPipeline: one class for the whole path from an empty template to a labelled, scan-like page.

    1. template  an empty template from the bank (or, optionally, one written by a small local LLM)
    2. text      filled with Arabic or English (picked at random, mostly Arabic) and a random typography
    3. render    SynthDoc renders the page and measures every word, line and slot (the labels)
    4. effects   none (a clean page, 15%), or OCRSmith, Augraphy or scan-simulator (equally often);
                 every box is moved with the page
    5. markdown  the filled page as Markdown, with the bounding boxes at the very top

Grey picture boxes in a template get a random photo from img_pool/ (see img_pool.py) in step 2, and the
words come from Wikipedia (text_pool/, see text_pool.py) unless text="demo".

Each sample writes, in out_dir:
    <name>.jpg            final page                   <name>_clean.jpg   SynthDoc's page before effects
    <name>.html           filled page (HTML)           <name>.md          Markdown with the boxes on top
    <name>.json           labels: words, lines, slots  <name>_words.jpg / <name>_slots.jpg   boxes drawn

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

import attack  # noqa: E402  (Augraphy + scan-simulator, synthdoc_adapter/attack.py)
import attack_ocrsmith  # noqa: E402  (OCRSmith paper + capture presets)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from export import ROLE_PROPERTY, synthdoc_html  # noqa: E402
from fill_style import fill, font_families, style, style_latin  # noqa: E402
from labels import MIN_CONTRAST, build_label, contrast, split_words  # noqa: E402
from synthdoc.ner_data_generator import NERDataGenerator  # noqa: E402
from synthdoc_arabic import pipeline as synthdoc_arabic  # noqa: E402
from synthdoc_arabic.ocr_gt import build_page_gt  # noqa: E402

# The Arabic extension's word measuring script only says whether a word is clipped at all. A word whose
# side is cut (a long English word in a narrow table cell) must be dropped from the labels, so the script is
# extended here, in memory, to say that too (nothing in the SynthDoc folder is changed).
if "cut_x:" not in synthdoc_arabic.WORDS_JS:
    synthdoc_arabic.WORDS_JS = synthdoc_arabic.WORDS_JS.replace(
        "split_word:", "cut_x: (l > b.l + 1 || r < b.r - 1), split_word:", 1)

from pipeline.img_pool import ImagePool  # noqa: E402
from pipeline.text_pool import TextPool  # noqa: E402
from src.bank import TEMPLATES  # noqa: E402
from src.dedup import load_bank  # noqa: E402
from src.markdown import to_markdown  # noqa: E402

LANGUAGES = {"ar": 0.8, "en": 0.2}
# "clean": some pages stay exactly as SynthDoc rendered them, sharp and noise-free (born-digital documents)
# The rest are split evenly between the three effect libraries (Augraphy followed by scan-simulator, the
# "augraphy+scan-simulator" effect, is still there when asked for)
EFFECTS = {"clean": 0.15, "ocrsmith": 0.85 / 3, "augraphy": 0.85 / 3, "scan-simulator": 0.85 / 3}
SOURCES = {"grammar": 1, "websight": 1, "doclaynet": 1}
ROLE = {prop: role for role, prop in ROLE_PROPERTY.items()}  # doco:sectionTitle -> heading
TRIES = 4  # attempts with new seeds when an effect leaves the text too faint
# Runs at the end of the page body, so before Paged.js copies the page (Paged.js starts as soon as parsing
# ends, before DOMContentLoaded): an element with no text in it, at least 30 x 30 px, with a plain grey
# background (the look of an image placeholder) gets one of PHOTOS, covering it. Text on grey (a grey section
# or banner) is design and stays; table cells and form fields are left alone.
GREY_BOXES_JS = """(() => {
  const photos = PHOTOS; let k = 0;
  for (const el of document.body.querySelectorAll("*")) {
    if (k >= photos.length) break;
    if (/^(TD|TH|TR|TABLE|INPUT|SELECT|TEXTAREA|BUTTON|HR|BR|IMG|SVG|SCRIPT|STYLE)$/.test(el.tagName)) continue;
    if (el.textContent.trim() || el.querySelector("img") || el.dataset.img) continue;
    const r = el.getBoundingClientRect();
    if (r.width < 30 || r.height < 30) continue;
    const m = getComputedStyle(el).backgroundColor.match(/\\d+(\\.\\d+)?/g);
    if (!m || (m[3] !== undefined && +m[3] === 0)) continue;
    const [R, G, B] = m.map(Number), mean = (R + G + B) / 3;
    if (Math.max(R, G, B) - Math.min(R, G, B) > 16 || mean < 110 || mean > 238) continue;
    el.style.setProperty("background", `url("${photos[k++]}") center / cover no-repeat`, "important");
    el.dataset.img = "grey-box";
  }
})();"""
# The opening tag of a grey picture box: any element whose class list has "ph"
PICTURE_BOX = re.compile(r'<(\w+)\b([^>]*\bclass="(?:[^"]*\s)?ph(?:\s[^"]*)?"[^>]*)>')
_cache = {}


def bank():
    if "bank" not in _cache:
        _cache["bank"] = load_bank()
    return _cache["bank"]


def bank_ids():
    """{source: [template ids]} from bank_index.jsonl: a sample loads only its own template.json, so a bank
    of 100,000 templates doesn't slow every worker's start."""
    if "ids" not in _cache:
        ids = {}
        for line in (ROOT / "bank_index.jsonl").read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            ids.setdefault(entry["source"], []).append(entry["id"])
        if not ids:
            raise RuntimeError("bank_index.jsonl is empty: run python -m src.dedup")
        _cache["ids"] = ids
    return _cache["ids"]


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
    """Template -> Arabic/English text -> SynthDoc render + labels -> effects -> Markdown with boxes.

    languages, effects and sources are {choice: weight}; templates is "bank" or "llm" (an LLMTemplates,
    passed as llm, writes them; if it can't, the sample falls back to the bank and says so).
    pictures is the ImagePool for grey picture boxes (default img_pool/; False keeps them grey).
    text is "wikipedia" (text_pool/, the default once it is built) or "demo" (a few built-in sentences)."""

    def __init__(self, out_dir, languages=None, effects=None, sources=None, templates="bank", llm=None,
                 pictures=None, text=None, extras=True):
        self.out = Path(out_dir).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.languages, self.effects = languages or LANGUAGES, effects or EFFECTS
        self.sources, self.templates, self.llm = sources or SOURCES, templates, llm
        self.pictures = ImagePool() if pictures is None else pictures
        self.text = text or ("wikipedia" if TextPool.exists("ar") and TextPool.exists("en") else "demo")
        # extras: the clean render and the drawn word and slot boxes, for looking at a sample (the generator
        # page); a training set (pipeline/dataset.py) leaves them out
        self.extras = extras
        synthdoc_arabic.install()  # SynthDoc then also measures every word (the Arabic extension)
        self._synthdoc = None

    # -- the five steps -------------------------------------------------------------------------------

    def template(self, rng, name):
        """1. An empty template: (meta, template text, folder with its pictures, note)."""
        note = ""
        if self.templates == "llm":
            if self.llm is not None and self.llm.available():
                try:
                    made = self.llm.template(rng, f"llm_{name}")
                except Exception as e:  # e.g. still rate-limited after waiting: reuse an earlier LLM template
                    made, note = None, f"LLM call failed ({type(e).__name__}: {e})"
                    made = self.llm.saved(rng)
                    if made:
                        return (*made, self.out, note + "; an earlier LLM template reused")
                if made:
                    return (*made, self.out, "")
                note = (note or "LLM page had too little text") + "; bank template used"
            else:
                note = f"{self.llm.problem if self.llm else 'no LLM'}; bank template used"
        source = pick(rng, self.sources)
        tid = rng.choice(bank_ids()[source])
        meta = json.loads((TEMPLATES / tid / "template.json").read_text(encoding="utf-8"))
        return meta, None, None, note

    def fill(self, meta, text, folder, rng, lang):
        """2. SynthDoc's template HTML plus the values for its slots and style."""
        if lang == "ar":
            s, css = style(rng, arabic_fonts(), transparent=False)
        else:
            s, css = style_latin(rng, transparent=False)
        words = text_pool(lang) if self.text == "wikipedia" else None
        if words:
            words.used.clear()
        values = {slot["id"]: Markup(fill(slot, rng, s["digits"], lang, words)) for slot in meta["slots"]}
        page, used = self.fill_pictures(synthdoc_html(meta, text, lang, folder), rng)
        # Where the words came from: the Wikipedia articles used (CC BY-SA asks for attribution)
        s = {**s, "pictures": used, "text": self.text, "text_sources": words.sources() if words else []}
        # The style travels with the page, so SynthDoc's word labels record it (the Arabic extension reads this)
        variant = {"template": meta["id"], "source": meta["source"], "lang": lang, **s}
        tag = Markup.escape(json.dumps(variant, ensure_ascii=False))
        page = page.replace("</head>", f'<meta name="synthdoc-ar-variant" content="{tag}">\n</head>', 1)
        return page, {**values, "style": Markup(css)}, s

    def fill_pictures(self, page, rng):
        """Each grey picture box gets a random photo from the pool, covering the box (its size is unchanged)."""
        used = []
        if not self.pictures:
            return page, used

        def put(m):
            photo = self.pictures.pick(rng)
            used.append(photo.name)
            # !important: WebSight templates force their boxes grey with !important rules
            bg = f"background: url({photo.as_uri()}) center / cover no-repeat !important;"
            attrs = m.group(2)
            attrs = attrs.replace(' style="', f' style="{bg} ', 1) if ' style="' in attrs else f'{attrs} style="{bg}"'
            return f'<{m.group(1)}{attrs} data-img="img_pool/{photo.name}">'
        page = PICTURE_BOX.sub(put, page)
        # Web pages also have grey picture areas the converter could not name as such (a ".logo" square, a
        # hero image area): find them in the browser, before SynthDoc lays the page out, and fill them too
        spare = [self.pictures.pick(rng).as_uri() for _ in range(12)]
        script = GREY_BOXES_JS.replace("PHOTOS", json.dumps(spare))
        end = page.rfind("</body>")
        return page[:end] + f"<script>{script}</script>\n" + page[end:], used

    def render(self, page, values, name):
        """3. SynthDoc renders page 1 and labels it: (image, word/line labels, slot boxes)."""
        if self._synthdoc is None:
            self._synthdoc = NERDataGenerator(output_dir=str(self.out / "_synthdoc"), create_overlays=False,
                                              output_formats=["json"])
        result = self._synthdoc.generate_dataset_sample(template_html=page, template_data=values, output_id=name)
        first = result["pages"][0] if "pages" in result else result
        raw_name = f"{name}_page_1" if "pages" in result else name
        raw = json.loads((synthdoc_arabic._raw_dir / f"{raw_name}.json").read_text(encoding="utf-8"))
        ocr = build_page_gt(raw["page"], {"language": raw["lang"], "direction": raw["dir"], "variant": raw["variant"]})
        for word, measured in zip(ocr["words"], raw["page"]["words"]):  # same order: lines are runs of words
            word["cut_x"] = measured.get("cut_x", False)
        slots = json.loads(Path(first["label_path"]).read_text(encoding="utf-8"))["annotations"]
        image = Image.open(first["image_path"]).convert("RGB")
        # SynthDoc's own copies are no longer needed (they would double the size of a large set)
        for page in result.get("pages", [result]):
            for key in ("image_path", "label_path"):
                Path(page[key]).unlink(missing_ok=True)
        (synthdoc_arabic._raw_dir / f"{raw_name}.json").unlink(missing_ok=True)
        return image, ocr, slots, result.get("total_pages", 1)

    def degrade(self, image, words, rng, effect):
        """4. One effect library, retried with new seeds while the text is too faint.

        Returns (image, word boxes, attacks, contrast, seed); word boxes are [x, y, w, h] or None if lost.
        effect "clean" keeps SynthDoc's page exactly as rendered: no paper, blur, noise or warp."""
        if effect == "clean":
            boxes = [[round(w["x"]), round(w["y"]), round(w["width"]), round(w["height"])] for w in words]
            return image, boxes, ["clean"], contrast(np.asarray(image.convert("L")), boxes), None
        best = None
        for _ in range(TRIES):
            seed = rng.randrange(2**31)
            if effect == "ocrsmith":
                out, boxes, attacks = attack_ocrsmith.degrade(image, words, seed)
            else:
                bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                if effect == "augraphy":  # ink and paper only: nothing moves
                    out, used = attack.run_augraphy(bgr, seed)
                    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR) if out.ndim == 2 else out
                    boxes = [[round(w["x"]), round(w["y"]), round(w["width"]), round(w["height"])] for w in words]
                    attacks = ["augraphy"] if used is not None else []
                else:  # scan-simulator, with or without Augraphy first: boxes ride along in a word-id mask
                    out, mask, attacks = attack.degrade(bgr, attack.word_mask(bgr.shape[:2], words), seed,
                                                        use_augraphy=effect.startswith("augraphy"))
                    boxes = attack.mask_boxes(mask, len(words))
                out = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            c = contrast(np.asarray(out.convert("L")), [b for b in boxes if b])
            if best is None or c > best[3]:
                best = (out, boxes, attacks, c, seed)
            if c >= MIN_CONTRAST or not any(boxes):
                break
        return best

    def markdown(self, filled, label, name):
        """5. Markdown of the filled page, with every box (after effects) in a header at the very top."""
        def row(item):
            return json.dumps(item, ensure_ascii=False)
        side = -1 if label["lang"] == "ar" else 1  # reading order: top to bottom, then right to left for Arabic
        regions = [{"type": ROLE.get(s["property"], "text"), "box": s["box"], "text": s["text"]}
                   for s in sorted(label["slots"], key=lambda s: (s["box"][1] // 12, side * s["box"][0]))]
        lines = [{"box": ln["box"], "text": ln["text"]} for ln in label["lines"]]
        head = ["---", f"image: {name}.jpg", f"width: {label['width']}", f"height: {label['height']}",
                f"language: {label['lang']}", f"template: {label['template']}",
                f"effects: {row(label['attacks'])}",
                f"text: {'Wikipedia, CC BY-SA 4.0 (articles listed in ' + name + '.json)' if label['style']['text'] == 'wikipedia' else 'demo sentences'}",
                "boxes: [x, y, width, height] in image pixels, after the effects",
                f"regions:  # {len(regions)} text blocks (template slots)"]
        head += [f"  - {row(r)}" for r in regions]
        head += [f"lines:  # {len(lines)} lines; word boxes are in {name}.json"]
        head += [f"  - {row(ln)}" for ln in lines]
        return "\n".join(head + ["---", "", to_markdown(filled)])

    # -- one sample, many samples ---------------------------------------------------------------------

    def sample(self, seed, name=None, language=None, effect=None):
        """Run the five steps once; returns a summary dict (files are written to out_dir)."""
        name = name or f"s{seed}"
        rng = random.Random(seed)
        lang, effect = language or pick(rng, self.languages), effect or pick(rng, self.effects)
        t = {}

        t.update(template=0, fill=0, render=0)
        for _ in range(3):  # a template that ends up with no readable text (all pictures) is swapped for another
            mark = time.perf_counter()
            meta, text, folder, note = self.template(rng, name)
            t["template"] += time.perf_counter() - mark
            mark = time.perf_counter()
            page, values, s = self.fill(meta, text, folder, rng, lang)
            filled = Template(page).render(**values)
            t["fill"] += time.perf_counter() - mark
            mark = time.perf_counter()
            clean, ocr, slots, pages = self.render(page, values, name)
            t["render"] += time.perf_counter() - mark
            words, cut = split_words(ocr)
            if words:
                break
        mark = time.perf_counter()
        image, boxes, attacks, c, used_seed = self.degrade(clean, words, rng, effect)
        label = build_label(name, image.size, used_seed, attacks, c, ocr, slots, words, boxes, cut)
        label.update({"template": meta["id"], "source": meta["source"], "lang": lang, "effect": effect,
                      "style": s, "pages_rendered": pages})
        t["effects"] = time.perf_counter() - mark
        mark = time.perf_counter()
        md = self.markdown(filled, label, name)
        t["markdown"] = time.perf_counter() - mark

        out = self.out
        image.save(out / f"{name}.jpg", quality=88)
        (out / f"{name}.html").write_text(filled, encoding="utf-8")
        (out / f"{name}.md").write_text(md, encoding="utf-8")
        (out / f"{name}.json").write_text(json.dumps(label, ensure_ascii=False), encoding="utf-8")
        if self.extras:
            clean.save(out / f"{name}_clean.jpg", quality=88)
            draw(image, [w["box"] for w in label["words"]], (40, 170, 60), out / f"{name}_words.jpg")
            draw(image, [sl["box"] for sl in label["slots"]], (210, 50, 50), out / f"{name}_slots.jpg")
        times = {k: round(t[k], 2) for k in ("template", "fill", "render", "effects", "markdown")}
        return {"name": name, "seed": seed, "template": meta["id"], "source": meta["source"], "lang": lang,
                "effect": effect, "attacks": attacks, "font": s["font"], "size_pt": s["size_pt"],
                "contrast": label["contrast"], "low_contrast": label["low_contrast"], "words": len(label["words"]),
                "lost_words": label["lost_words"], "pages_rendered": pages, "note": note,
                "pictures": len(s["pictures"]), "text": s["text"], "text_sources": len(s["text_sources"]),
                "seconds": {**times, "total": round(sum(times.values()), 2)}}

    def run(self, n=16, seed=None, on_sample=None):
        """n samples one after another; on_sample(summary) is called as each one is done."""
        seed = random.randrange(10**6) if seed is None else seed
        done = []
        for i in range(n):
            done.append(self.sample(seed * 1000 + i, f"s{i + 1:02d}"))
            if on_sample:
                on_sample(done[-1])
        return done


def warm_up():
    """Augraphy compiles its noise code (numba) on first use, about 15 s: do it before the first real page."""
    attack.run_augraphy(np.full((200, 200, 3), 255, np.uint8), 0)


def draw(image, boxes, color, path):
    out = image.copy()
    d = ImageDraw.Draw(out)
    for x, y, w, h in boxes:
        d.rectangle([x, y, x + w, y + h], outline=color, width=2)
    out.save(path, quality=82)
