"""Inputs for the Surya probe: generated pages (Arabic and English, every source, clean and degraded), and
single blocks and tables cut out of them, each with what our pipeline knows about it (inputs/index.json)."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import generator as SURYA  # noqa: E402
from pipeline.compact_original import encode  # noqa: E402
from pipeline.generator import DocPipeline  # noqa: E402

OUT = Path(__file__).resolve().parent / "inputs"
PLAN = [  # (language, source, effect)
    ("en", "grammar", "clean"), ("ar", "grammar", "clean"), ("en", "doclaynet", "clean"), ("ar", "doclaynet", "clean"),
    ("en", "websight", "clean"), ("ar", "websight2", "clean"), ("en", "websight2", "ocrsmith"),
    ("ar", "websight", "augraphy"), ("en", "doclaynet", "scan-simulator"), ("ar", "grammar", "ocrsmith"),
    ("en", "grammar", "augraphy"), ("ar", "doclaynet", "scan-simulator"),
]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    work = OUT.parent / "_work"
    index, rng = [], random.Random(5)
    for k, (lang, source, effect) in enumerate(PLAN):
        pipe = DocPipeline(work, languages={lang: 1}, sources={source: 1}, effects={effect: 1},
                           tasks={"table" if k % 3 == 0 else "full_ocr": 1}, extras=False)
        s = pipe.sample(40000 + k, f"g{k:02d}")
        label = json.loads((work / f"g{k:02d}.json").read_text(encoding="utf-8"))
        page = work / (f"g{k:02d}_page.jpg" if label["surya"]["crop"] else f"g{k:02d}.jpg")
        from PIL import Image
        im = Image.open(page)
        im.save(OUT / f"g{k:02d}_page.jpg", quality=92)
        width, height = im.size
        blks = label["surya"]["blocks"]
        index.append({"image": f"g{k:02d}_page.jpg", "kind": "page", "lang": lang, "source": source, "effect": effect,
                      "size": [width, height], "ours_full_ocr": encode(SURYA.divs(blks, width, height)),
                      "ours_layout": SURYA.layout(blks, width, height)})
        print("page", k, lang, source, effect, len(blks), "blocks", flush=True)
        # cut-outs: one table (if any) and two other blocks of different kinds, clean of neighbours
        others_of = lambda b: [x["box"] for x in blks if x is not b]  # noqa: E731
        picked = [b for b in blks if b["label"] == "Table"][:1]
        rest = [b for b in blks if b["label"] not in ("Table",)]
        rng.shuffle(rest)
        seen = set()
        for b in rest:
            if b["label"] not in seen and len(picked) < 3:
                picked.append(b)
                seen.add(b["label"])
        for j, b in enumerate(picked):
            crop = SURYA.crop_box(b["box"], width, height, rng, others_of(b), share=(0.03, 0.12))
            if not SURYA.clear(crop, others_of(b)) or crop[2] - crop[0] < 40 or crop[3] - crop[1] < 20:
                continue
            name = f"g{k:02d}_cut{j}_{b['label']}.jpg"
            im.crop(crop).save(OUT / name, quality=92)
            index.append({"image": name, "kind": "cut", "label": b["label"], "lang": lang, "effect": effect,
                          "size": [crop[2] - crop[0], crop[3] - crop[1]], "ours_html": b["html"],
                          "ours_box_on_cut": SURYA.on_crop(b["box"], crop)})
            print("  cut", name, flush=True)
    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(index), "inputs")


if __name__ == "__main__":
    main()
