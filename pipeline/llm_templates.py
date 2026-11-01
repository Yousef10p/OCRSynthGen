"""Optional template source: an LLM writes a document page, which becomes an empty template.

The model writes an ordinary HTML page with English text (small models write English HTML far more reliably
than Arabic). The page then goes through exactly the same steps as a WebSight page (src/from_websight.py):
text -> slots, pictures -> grey boxes, CSS flipped to right-to-left. So an LLM page and a WebSight page end up
in the same template format.

Any OpenAI-compatible chat endpoint works. LLMTemplates.from_env() picks:
  * Groq, when GROQ_API_KEY is set (in arabic-template-bank/.env or the environment); GROQ_MODEL chooses the
    model (default openai/gpt-oss-20b, the fastest text model Groq offers as of November 2026);
  * otherwise Ollama on this machine (for example: ollama pull qwen2.5-coder:1.5b).

Nothing here is needed when templates come from the bank.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from src.from_websight import finish, flip, strip_page

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
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
