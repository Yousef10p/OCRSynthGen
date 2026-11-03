"""Ask Surya OCR 2.5 (the API in surya_inference/) about images, one request at a time, and keep every answer.

The address, model name and key are read from surya_inference/suraya.py at run time, so the key is never
copied into this repository. Each request waits for the previous answer. Every answer is appended to a JSON
Lines file with what was asked: image, prompt, whether the guided-decoding JSON schema was sent, max_tokens,
status, seconds, finish_reason, token usage and the raw content.

    python surya_probe/probe.py --out surya_probe/results.jsonl --tasks full_ocr layout table block_ocr \
        --schema both img1.jpg img2.png ...
"""
import argparse
import base64
import io
import json
import re
import sys
import time
from pathlib import Path

import requests
from PIL import Image

CLIENT = Path(r"D:\code\OCR\arabic-template-bank-surya\surya_inference\suraya.py")


def settings(client=CLIENT):
    """(url, key, model, prompts, layout schema, table schema) from the user's client, without importing it."""
    src = client.read_text(encoding="utf-8")
    url = re.search(r'API_URL\s*=\s*"([^"]+)"', src).group(1)
    key = re.search(r'API_KEY\s*=\s*"([^"]+)"', src).group(1)
    model = re.search(r'MODEL_NAME\s*=\s*"([^"]+)"', src).group(1)
    scope = {}
    start = src.index("PROMPTS = {")
    end = src.index("def analyze_document")
    exec(src[start:end], scope)  # the prompt and schema constants only
    return url, key, model, scope["PROMPTS"], scope["LAYOUT_JSON_SCHEMA"], scope["TABLE_REC_JSON_SCHEMA"]


def ask(image_path, task, schema, max_tokens, conf):
    url, key, model, prompts, layout_schema, table_schema = conf
    data = Path(image_path).read_bytes()
    kind = "jpeg" if str(image_path).lower().endswith((".jpg", ".jpeg")) else "png"
    payload = {"model": model, "temperature": 0.0, "max_tokens": max_tokens,
               "messages": [{"role": "user", "content": [
                   {"type": "text", "text": prompts[task]},
                   {"type": "image_url", "image_url": {"url": f"data:image/{kind};base64,{base64.b64encode(data).decode()}"}}]}]}
    if schema and task in ("layout", "table"):
        payload["response_format"] = {"type": "json_schema", "json_schema": {
            "name": f"{task}_schema", "schema": layout_schema if task == "layout" else table_schema, "strict": True}}
    t = time.time()
    try:
        r = requests.post(url, headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
                          json=payload, timeout=600)
        seconds = round(time.time() - t, 1)
        if r.status_code != 200:
            return {"status": r.status_code, "seconds": seconds, "error": r.text[:2000]}
        j = r.json()
        choice = j["choices"][0]
        return {"status": 200, "seconds": seconds, "finish_reason": choice.get("finish_reason"),
                "usage": j.get("usage"), "content": choice["message"]["content"]}
    except Exception as e:  # network trouble: record it and go on with the next request
        return {"status": None, "seconds": round(time.time() - t, 1), "error": f"{type(e).__name__}: {e}"[:2000]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="+", default=["full_ocr", "layout", "table", "block_ocr"])
    ap.add_argument("--schema", choices=["on", "off", "both"], default="on",
                    help="send the guided-decoding JSON schema for layout and table (as surya_inference/suraya.py does)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    conf = settings()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():  # continue an interrupted run
        for line in out.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            done.add((d["image"], d["task"], d["schema"], d["max_tokens"]))
    for image in a.images:
        w, h = Image.open(image).size
        for task in a.tasks:
            for schema in ([True, False] if a.schema == "both" and task in ("layout", "table")
                           else [a.schema != "off" and task in ("layout", "table")]):
                key = (str(image), task, schema, a.max_tokens)
                if key in done:
                    continue
                result = ask(image, task, schema, a.max_tokens, conf)  # waits for the answer
                row = {"image": str(image), "size": [w, h], "task": task, "schema": schema,
                       "max_tokens": a.max_tokens, **result}
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"{Path(image).name} {task} schema={schema} -> {result['status']} {result['seconds']}s "
                      f"{result.get('finish_reason', '')} {len(result.get('content') or '')} chars", flush=True)


if __name__ == "__main__":
    main()
