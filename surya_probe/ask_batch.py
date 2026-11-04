"""Ask the Surya 2.5 API about every page of a generator batch, one request at a time, with the page's own prompt.

    python surya_probe/ask_batch.py surya_probe/batch100/full_ocr     -> surya_probe/batch100/full_ocr/surya.jsonl

One line per page: id, task, image, prompt, our target file, and Surya's answer (status, seconds, finish_reason,
usage, content). An interrupted run continues with the pages not yet answered. The API address and key are read
from surya_inference/suraya.py at run time (probe.settings), never written anywhere.
"""
import base64
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe import settings  # noqa: E402

# Surya's own answer limits per prompt (surya/settings.py: SURYA_MAX_TOKENS_*)
MAX_TOKENS = {"full_ocr": 12288, "block_ocr": 8192, "layout": 3072, "table": 3072}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    folder = Path(sys.argv[1])
    url, key, model = settings()[:3]
    out = folder / "surya.jsonl"
    done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()} if out.exists() else set()
    rows = [json.loads(line) for line in (folder / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    for n, row in enumerate(rows, 1):
        if row["id"] in done:
            continue
        image, prompt = folder / row["images"], (folder / row["prompts"]).read_text(encoding="utf-8")
        data = base64.b64encode(image.read_bytes()).decode()
        payload = {"model": model, "temperature": 0.0, "max_tokens": MAX_TOKENS[row["task"]],
                   "messages": [{"role": "user", "content": [
                       {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}},
                       {"type": "text", "text": prompt}]}]}
        start = time.time()
        try:  # waits for the answer before the next page
            r = requests.post(url, json=payload, timeout=900,
                              headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            answer = {"status": r.status_code}
            if r.status_code == 200:
                choice = r.json()["choices"][0]
                answer.update(finish_reason=choice.get("finish_reason"), usage=r.json().get("usage"),
                              content=choice["message"]["content"])
            else:
                answer["error"] = r.text[:2000]
        except Exception as e:  # network trouble: recorded, the next run asks again
            answer = {"status": None, "error": f"{type(e).__name__}: {e}"[:2000]}
        answer["seconds"] = round(time.time() - start, 1)
        line = {"id": row["id"], "task": row["task"], "lang": row.get("lang"), "image": row["images"],
                "prompt": prompt, "target": row["targets"], **answer}
        if answer["status"] == 200:
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"{folder.name} {n}/{len(rows)} {row['id']} {row['task']}: {answer['status']} {answer['seconds']}s "
              f"{answer.get('finish_reason', '')} {len(answer.get('content') or '')} chars"
              + ("" if answer["status"] == 200 else f" | {answer.get('error', '')[:200]}"), flush=True)


if __name__ == "__main__":
    main()
