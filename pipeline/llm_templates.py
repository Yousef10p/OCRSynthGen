"""Optional template source: an LLM writes a document page, which becomes an empty template.

The model writes an ordinary HTML page with English text (small models write English HTML far more reliably
than Arabic). The page then goes through the same steps a WebSight page went through when the bank was built:
text -> slots, pictures -> grey boxes, CSS flipped to right-to-left. So an LLM page and a WebSight page end up in
the same template format.

Any OpenAI-compatible chat endpoint works. LLMTemplates.from_env() picks:
  * Groq, when GROQ_API_KEY is set (in arabic-template-bank/.env or the environment); GROQ_MODEL chooses the
    model (default openai/gpt-oss-20b, the fastest text model Groq offers as of November 2026);
  * otherwise Ollama on this machine (for example: ollama pull qwen2.5-coder:1.5b).

Nothing here is needed when templates come from the bank.

How a page becomes a template (as WebSight pages became bank templates): scripts, links and media are dropped,
pictures become grey boxes (class "ph"), form fields become field slots, every piece of text becomes a slot
{{ slot_N }} with a role from its tag, and the CSS is flipped to right to left with rtlcss (node; npm install
in the repository folder). adapt() then gives it the bank's form: its language, and each grey box a picture
slot.
"""
import json
import math
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup, Comment, NavigableString

from pipeline.generator import page_dims

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
SAVED = Path(__file__).resolve().parents[1] / "pipeline_out" / "llm_templates"  # every LLM template made
GROQ_URL, GROQ_MODEL = "https://api.groq.com/openai/v1", "openai/gpt-oss-20b"
KINDS = ["invoice", "monthly report with a statistics table", "official letter", "application form",
         "company newsletter with two columns", "meeting minutes", "price list", "product data sheet",
         "school timetable", "event programme", "bank statement", "employment contract",
         "restaurant menu", "travel itinerary", "lab results report", "annual report summary page"]
PROMPT = ("Write one complete HTML page for a printed A4 {kind}. Put all CSS in one <style> element in <head>. "
          "Use headings, paragraphs and {extra} with realistic English text. No scripts, no images, no links "
          "to other files. Reply with the HTML only.")
_checked = {}  # (url, model, key) -> (time, available, problem)
EXTRAS = ["a table", "a bulleted list", "a table and a list", "two columns", "labelled fields"]


def read_env(path=ENV_FILE):
    """KEY=value lines of a .env file (no extra package needed); the real environment wins."""
    values = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key and not key.startswith("#"):
                values[key.strip()] = value.strip().strip('"').strip("'")
    return {**values, **{k: v for k, v in os.environ.items() if k.startswith(("GROQ_", "LLM_"))}}


class LLMTemplates:
    """Templates written by an LLM behind an OpenAI-compatible endpoint (Groq or a local Ollama)."""

    def __init__(self, url="http://localhost:11434/v1", model="qwen2.5-coder:1.5b", api_key=None, timeout=120):
        self.url, self.model, self.api_key, self.timeout = url.rstrip("/"), model, api_key, timeout
        self.name = f"Groq · {model}" if "groq.com" in url else f"local · {model}"
        self.problem = None  # why available() said no, for the page

    @classmethod
    def from_env(cls):
        env = read_env()
        if env.get("GROQ_API_KEY"):
            return cls(GROQ_URL, env.get("GROQ_MODEL") or GROQ_MODEL, env["GROQ_API_KEY"])
        return cls()

    def _request(self, path, body=None, timeout=None):
        headers = {"Content-Type": "application/json", "User-Agent": "arabic-template-bank/1.0"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.url}/{path}", json.dumps(body).encode() if body else None, headers)
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            return json.load(r)

    def available(self):
        """True if the endpoint answers and has the model (else self.problem says why); cached for a minute."""
        key = (self.url, self.model, self.api_key)
        if key in _checked and time.time() - _checked[key][0] < 60:
            self.problem = _checked[key][2]
            return _checked[key][1]
        ok = self._check()
        _checked[key] = (time.time(), ok, self.problem)
        return ok

    def _check(self):
        try:
            models = [m["id"] for m in self._request("models", timeout=8).get("data", [])]
        except urllib.error.HTTPError as e:
            self.problem = f"{self.name}: HTTP {e.code} (check the API key)" if e.code in (401, 403) else                 f"{self.name}: HTTP {e.code}"
            return False
        except Exception as e:
            self.problem = f"{self.name}: not reachable ({type(e).__name__})"
            return False
        if any(m == self.model or m.split(":")[0] == self.model.split(":")[0] for m in models):
            self.problem = None
            return True
        self.problem = f"{self.name}: model not offered; some that are: {', '.join(sorted(models)[:8])}"
        return False

    def ask(self, prompt, seed):
        body = {"model": self.model, "seed": seed, "temperature": 0.9, "max_tokens": 8192,
                "messages": [{"role": "user", "content": prompt}]}
        if "gpt-oss" in self.model:  # a reasoning model: keep the thinking short, the page is what we want
            body["reasoning_effort"] = "low"
        for attempt in range(4):  # Groq answers 429 when its per-minute limit is used up: wait as it says
            try:
                return self._request("chat/completions", body)["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code != 429 or attempt == 3:
                    raise
                time.sleep(min(float(e.headers.get("retry-after") or 2 ** (attempt + 1)), 20))

    def template(self, rng, name):
        """(meta, template text) for one new template, or None if the page has too little to fill.

        Every template made is kept in SAVED, so later samples can reuse it (see saved())."""
        kind = rng.choice(KINDS)
        page = self.ask(PROMPT.format(kind=kind, extra=rng.choice(EXTRAS)), rng.randrange(2**31))
        made = to_template(page, name, kind)
        if made:
            SAVED.mkdir(parents=True, exist_ok=True)
            path = SAVED / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}.json"
            path.write_text(json.dumps({"meta": made[0], "text": made[1]}, ensure_ascii=False), encoding="utf-8")
        return made

    @staticmethod
    def saved(rng):
        """A template the LLM wrote earlier, (meta, text), or None if there is none yet."""
        files = sorted(SAVED.glob("*.json")) if SAVED.exists() else []
        if not files:
            return None
        data = json.loads(rng.choice(files).read_text(encoding="utf-8"))
        return data["meta"], data["text"]


def to_template(page, name, kind=""):
    """An LLM's HTML page -> (meta, template text) in the bank's format, as for WebSight pages; None if unusable."""
    fenced = re.search(r"```(?:html)?\s*(.*?)```", page, re.S)
    stripped = strip_page(fenced.group(1) if fenced else page)
    if stripped is None:
        return None
    soup, sheet, slots, features = stripped
    if len(slots) < 5:
        return None
    flipped = flip([sheet] + [f"x {{ {el['style']} }}" for el in soup.body.find_all(style=True)])
    if None in flipped:
        return None
    meta = {"id": name, "source": "llm", "source_ref": kind, "positioning": "flow",
            "page": {"size": "A4", "orientation": "portrait"}, "slots": slots, "features": features}
    return meta, finish(soup, sheet, flipped, "A4")


# -- an HTML page -> a template ------------------------------------------------------------------------------

DROP = ["script", "noscript", "link", "meta", "title", "base", "template", "object", "embed", "audio", "option"]
MEDIA = ["img", "svg", "video", "canvas", "iframe", "picture"]
PH_HINT = re.compile(r"image|img|photo|picture|video|placeholder|thumbnail|banner|logo|avatar|map", re.I)
KEEP_ATTRS = ("class", "id", "style", "colspan", "rowspan")
TAG_ROLE = {"h1": "title", "h2": "heading", "h3": "heading", "h4": "heading", "h5": "heading", "h6": "heading",
            "p": "paragraph", "li": "list_item", "th": "table_header", "td": "cell", "figcaption": "caption",
            "blockquote": "quote"}
FIELD_CSS = "display: inline-block; min-inline-size: 10em; border: 1px solid #999; padding: 0.2em 0.4em"
EXTRA_CSS = ".ph { background: #d0d0d0 !important; background-image: none !important; min-block-size: 2em; }\n"
FONT_STACK = '"Noto Naskh Arabic", "Amiri", serif'
# word range of each slot role (a converted form field gets its role's)
ROLES = {"title": [3, 10], "subtitle": [4, 12], "heading": [2, 8], "paragraph": [25, 120], "list_item": [3, 20],
         "table_header": [1, 3], "cell": [1, 4], "caption": [5, 15], "label": [1, 3], "field_value": [1, 6],
         "quote": [10, 40], "footnote": [8, 30], "header": [2, 8], "footer": [2, 8], "page_number": [1, 1],
         "sidebar": [10, 50]}
PIC_CSS = (".pic-frame { overflow: hidden; min-block-size: 2em; }\n"
           ".pic-frame > img { display: block; inline-size: 100%; block-size: 100%; object-fit: cover; }\n")
PH = re.compile(r'<div\b([^>]*?)\bclass="([^"]*?)(?<![\w-])ph(?![\w-])([^"]*)"([^>]*)>\s*</div>')


def root_css(size, orientation, base_size="11pt", line_height=1.6, align="justify", margin="18mm"):
    """CSS variables with defaults plus the base rules every template shares."""
    w, h = page_dims(size, orientation)
    return (
        f"@page {{ size: {size} {orientation}; margin: 0; }}\n"
        f":root {{ --font-body: {FONT_STACK}; --font-heading: {FONT_STACK}; --base-size: {base_size};"
        f" --line-height: {line_height}; --align: {align}; --page-width: {w}mm; --page-height: {h}mm;"
        f" --page-margin: {margin}; }}\n"
        "html { font-family: var(--font-body); font-size: var(--base-size); line-height: var(--line-height); }\n"
        "body { margin: 0; }\n"
        "h1, h2, h3, h4 { font-family: var(--font-heading); margin-block: 0 0.4em; }\n"
        ".page { box-sizing: border-box; inline-size: var(--page-width); min-block-size: var(--page-height);"
        " padding: var(--page-margin); text-align: var(--align); }\n"
        ".ph { background: #d0d0d0; }\n"
    )


def clean_css(css):
    """Drop comments, imports, web fonts, url() references and generated text."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    css = re.sub(r"@import[^;]*;", "", css)
    css = re.sub(r"@font-face\s*\{[^}]*\}", "", css)
    css = re.sub(r"url\([^)]*\)", "none", css)
    return re.sub(r"(?<![-\w])content\s*:\s*(['\"]).*?\1", 'content: ""', css)


def css_columns(css):
    """Largest column count declared by grid-template-columns or column-count (1 if none)."""
    n = 1
    for v in re.findall(r"grid-template-columns\s*:\s*([^;}]+)", css):
        r = re.search(r"repeat\(\s*(\d+)", v)
        tracks = re.findall(r"minmax\([^)]*\)|[\d.]+(?:fr|px|%|em|rem)|\bauto\b", v)
        n = max(n, int(r.group(1)) if r else 2 if "repeat(" in v else len(tracks))
    return max([n] + [int(c) for c in re.findall(r"column-count\s*:\s*(\d+)", css)])


def picture_hint(el):
    """Words that say what a picture shows: its link's keywords (WebSight v0.2 links such as
    https://source.unsplash.com/random/300x200/?food), alt text, class names and id."""
    src = str(el.get("src", ""))
    words = re.findall(r"[a-zA-Z]{3,}", src.split("?", 1)[1]) if "?" in src else []
    words += re.findall(r"[a-zA-Z]{3,}", " ".join([str(el.get("alt", "")), " ".join(el.get("class", [])),
                                                    str(el.get("id", ""))]))
    return " ".join(w.lower() for w in words if w.lower() not in ("random", "http", "https", "source", "unsplash",
                                                                   "img", "image", "rounded", "full", "object",
                                                                   "cover", "auto", "mx", "my"))[:80]


def to_placeholder(soup, el):
    ph = soup.new_tag("div", attrs={"class": " ".join(el.get("class", []) + ["ph"])})
    if hint := picture_hint(el):
        ph["data-hint"] = hint
    size = [f"inline-size: {el['width']}px" if str(el.get("width", "")).isdigit() else "",
            f"block-size: {el['height']}px" if str(el.get("height", "")).isdigit() else ""]
    if el.name != "div" and not any(size):
        size = ["inline-size: 100%", "block-size: 150px"]
    if any(size):
        ph["style"] = "; ".join(filter(None, size))
    el.replace_with(ph)


def convert_field(soup, el):
    """Form inputs become spans: text-like fields -> field_value slot, buttons -> label slot."""
    kind = el.get("type", "text").lower() if el.name == "input" else el.name
    if kind == "hidden":
        return el.decompose()
    span = soup.new_tag("span", attrs={"class": " ".join(el.get("class", []))})
    if kind in ("checkbox", "radio"):
        span["style"] = "display: inline-block; inline-size: 0.9em; block-size: 0.9em; border: 1px solid #666"
    elif kind in ("submit", "button", "reset", "image"):
        span["data-slot-role"], span["data-n"] = "label", len((el.get("value") or "").split())
    else:
        span["data-slot-role"], span["style"] = "field_value", FIELD_CSS
        if kind == "textarea":
            span["style"] += "; display: block; min-block-size: 4em"
    el.replace_with(span)


def role_of(node, n):
    """Role from the nearest semantic ancestor; p/fallback text inside header/footer takes that role."""
    names = [p.name for p in node.parents]
    section = next((x for x in names if x in ("header", "footer")), None)
    for name in names:
        if name in ("button", "label") or (name == "a" and "nav" in names):
            return "label"
        if name in TAG_ROLE and not (name == "p" and section):
            return TAG_ROLE[name]
    return section or ("paragraph" if n >= 8 else "label")


def word_range(n):
    return [max(1, math.floor(0.7 * n)), math.ceil(1.3 * n) + 1]


def add_slots(body):
    """Replace every non-empty text node (and converted field) with {{ slot_N }} in DOM order."""
    slots = []
    for node in list(body.descendants):
        if isinstance(node, NavigableString) and not isinstance(node, Comment) and node.strip():
            n = len(node.split())
            role, words = role_of(node, n), word_range(n)
        elif getattr(node, "attrs", None) and "data-slot-role" in node.attrs:
            role, n = node.attrs.pop("data-slot-role"), int(node.attrs.pop("data-n", 0))
            words = word_range(n) if n else ROLES[role]
        else:
            continue
        ph = "{{ slot_%d }}" % len(slots)
        slots.append({"id": f"slot_{len(slots)}", "role": role, "words": words, "dir": "rtl"})
        if isinstance(node, NavigableString):
            lead, trail = node[: len(node) - len(node.lstrip())], node[len(node.rstrip()):]
            node.replace_with(lead + ph + trail)
        else:
            node.string = ph
    return slots


def strip_page(text):
    """Parse and strip one page: (soup, sheet css, slots, features) before CSS flipping, or None."""
    soup = BeautifulSoup(text, "lxml")
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()
    sheet = clean_css("\n".join(s.get_text() for s in soup.find_all("style")))
    for t in soup.find_all(DROP + ["style", "head"]):
        t.decompose()
    body = soup.body
    if body is None:  # a few rows are a bare <style> with no page
        return None
    for el in body.find_all(MEDIA):
        to_placeholder(soup, el)
    for el in body.find_all(True):
        hint = " ".join(el.get("class", [])) + " " + str(el.get("id", ""))
        if el.parent and not el.get_text(strip=True) and PH_HINT.search(hint) and "ph" not in el.get("class", []):
            el.clear()
            to_placeholder(soup, el)
    had_fields = bool(body.find_all(["input", "select", "textarea"]))
    for el in body.find_all(["input", "select", "textarea"]):
        convert_field(soup, el)
    for el in body.find_all(True):
        el.attrs = {k: el.attrs[k] for k in KEEP_ATTRS + ("data-slot-role", "data-n", "data-hint") if k in el.attrs}
        if "style" in el.attrs:
            el["style"] = clean_css(el["style"])
    slots = add_slots(body)
    lists = body.find_all(["ul", "ol"])
    inline = " ".join(el["style"] for el in body.find_all(style=True))
    f = {"columns": css_columns(sheet + inline), "tables": len(body.find_all("table")), "lists": len(lists),
         "list_depth": max((1 + len(l.find_parents(["ul", "ol"])) for l in lists), default=0),
         "forms": len(body.find_all("form")) or int(had_fields), "asides": len(body.find_all("aside")),
         "has_header": body.find("header") is not None, "has_footer": body.find("footer") is not None,
         "slot_count": len(slots)}
    return soup, sheet, slots, f


# Node script: a JSON array of CSS strings on stdin -> each flipped to RTL by rtlcss (null if it can't parse)
RTLCSS_JS = ('const rtlcss = require("rtlcss"); let input = ""; process.stdin.setEncoding("utf8");'
             ' process.stdin.on("data", (d) => (input += d)); process.stdin.on("end", () => {'
             ' process.stdout.write(JSON.stringify(JSON.parse(input).map((css) => {'
             ' try { return rtlcss.process(css); } catch (e) { return null; } }))); });')


def flip(css_list):
    """Flip many CSS strings to RTL in one rtlcss (node) call."""
    node = shutil.which("node") or r"C:\Program Files\nodejs\node.exe"
    out = subprocess.run([node, "-e", RTLCSS_JS], input=json.dumps(css_list),
                         capture_output=True, text=True, encoding="utf-8", check=True, cwd=ROOT)
    return json.loads(out.stdout)


def fonts(css, heading=False):
    """font-family -> var(--font-heading) in heading rules, var(--font-body) elsewhere."""
    def rule(m):
        var = "var(--font-heading)" if heading or re.search(r"\bh[1-6]\b", m.group(1)) else "var(--font-body)"
        return m.group(1) + "{" + re.sub(r"font-family\s*:[^;}]*", f"font-family: {var}", m.group(2)) + "}"
    return re.sub(r"([^{}]*)\{([^{}]*)\}", rule, css)


def finish(soup, sheet, flipped, size):
    """Put flipped CSS back, rebuild <head>, set lang/dir; returns the template text."""
    sheet, inline = flipped[0], flipped[1:]
    for el, css in zip(soup.body.find_all(style=True), inline):
        heading = el.name in TAG_ROLE and TAG_ROLE[el.name] in ("title", "heading")
        el["style"] = re.fullmatch(r"\s*x\s*\{(.*)\}\s*", fonts(css, heading), re.S).group(1).strip()
    own = root_css(size, "portrait") + re.sub(r"\{([{%#])", r"{ \1", fonts(sheet)) + "\n" + EXTRA_CSS
    head = soup.new_tag("head")
    head.append(soup.new_tag("meta", attrs={"charset": "utf-8"}))
    for css in (own, "{{ style }}"):
        style = soup.new_tag("style")
        style.string = css
        head.append(style)
    soup.html.attrs = {}
    soup.html.insert(0, head)
    # BeautifulSoup sorts attributes on output, so the <html> tag is written by hand to keep lang before dir.
    return "<!DOCTYPE html>\n" + soup.html.decode(formatter="html5").replace("<html>", '<html lang="ar" dir="rtl">', 1)


def unflip(text):
    """A WebSight v0.1 template's CSS flipped back to left to right (rtlcss flips are their own inverse)."""
    first = re.search(r"<style>(.*?)</style>", text, re.S)
    inline = re.findall(r'style="([^"]*)"', text)
    flipped = flip([first.group(1)] + [f"x {{ {s} }}" for s in inline])
    if None in flipped:
        return None
    sheet, styles = flipped[0], [re.fullmatch(r"\s*x\s*\{(.*)\}\s*", c, re.S).group(1).strip() for c in flipped[1:]]
    text = text[:first.start(1)] + sheet + text[first.end(1):]
    it = iter(styles)
    return re.sub(r'style="([^"]*)"', lambda m: f'style="{next(it)}"', text)


def hint_of(attrs, classes):
    m = re.search(r'data-hint="([^"]*)"', attrs)
    words = (m.group(1) if m else "") + " " + classes
    return " ".join(w for w in re.findall(r"[a-zA-Z]{3,}", words) if w not in ("box", "pic", "frame"))[:80]


def adapt(meta, text, lang):
    """(meta, text) in the bank2000 form (language, picture slots), or None if it can't be adapted."""
    meta = json.loads(json.dumps(meta))
    if "http" in text.lower():  # a web address (a background photo from the internet): the page isn't self-contained
        return None
    if meta.get("lang") and meta["lang"] != lang:
        lang = meta["lang"]  # WebSight v0.2 pages got their language when converted
    if lang == "en" and meta["source"] in ("websight", "llm"):  # CSS flipped to right to left when converted
        text = unflip(text)
        if text is None:
            return None
    text = re.sub(r'<html lang="\w+" dir="\w+">', f'<html lang="{lang}" dir="{"rtl" if lang == "ar" else "ltr"}">', text, 1)
    pictures = []

    def frame(m):
        pre, cls_a, cls_b, post = m.groups()
        attrs = pre + post
        hint = hint_of(attrs, cls_a + cls_b)
        pid = f"pic_{len(pictures)}"
        pictures.append({"id": pid, "hint": hint})
        attrs = re.sub(r'\s*data-hint="[^"]*"', "", attrs)
        classes = " ".join((cls_a + " " + cls_b).split() + ["pic-frame"])
        return f'<div{attrs.rstrip()} class="{classes}"><img data-pic="{pid}" src="{{{{ {pid} }}}}" alt=""></div>'
    text = PH.sub(frame, text)
    if re.search(r'class="[^"]*(?<![\w-])ph(?![\w-])', text):
        return None  # a picture box that isn't an empty div: leave this candidate out
    text = text.replace("</style>", PIC_CSS + "</style>", 1)
    meta.update(lang=lang, pictures=pictures)
    for s in meta["slots"]:
        s["dir"] = "rtl" if lang == "ar" else "ltr"
    meta["source_template"] = meta["id"]
    return meta, text


if __name__ == "__main__":
    # Fill the saved set slowly, within the per-minute limit:  python -m pipeline.llm_templates --n 30
    import argparse
    import random
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    llm, rng = LLMTemplates.from_env(), random.Random()
    if not llm.available():
        raise SystemExit(llm.problem)
    for i in range(a.n):
        made = llm.template(rng, f"llm_{time.strftime('%H%M%S')}_{i}")
        print(f"{i + 1}/{a.n}", "kept" if made else "too little text, skipped",
              made[0]["source_ref"] if made else "", flush=True)
    print(f"{len(list(SAVED.glob('*.json')))} LLM templates saved in {SAVED}")
