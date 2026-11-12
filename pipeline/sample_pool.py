"""Make dataset_pool_sample/: about 20 items per label from the full pools, so a fresh copy of the repository (without
the 80 GB of dataset_pool/) generates real pages. Same layout as dataset_pool/, so the generator reads it the same
way (pools.home(): dataset_pool/, then the first homes, then this). Only items the generator's readers accept are
taken; every item keeps its source and licence.

    .venv-pipeline\\Scripts\\python -m pipeline.sample_pool          (needs the full pools; rewrites the sample)

    text/{ar,en}          20 documents each            Text, Section-Header, List-Group, Table-Of-Contents
    captions/{ar,en}      20 each                      Caption
    layout/               20 page headers, 20 footnotes   Page-Header, Footnote (English)
    tables/{ar,en}        20 tables each               Table
    references/{ar,en}    20 works each                Bibliography
    vocab/                20 forms, 20 processes       Form, Diagram
    code/                 20 functions                 Code-Block
    molecules/            20 molecules                 Chemical-Block
    equations/            20 formulas                  Equation-Block
    chartdata/            20 each: SBS Figures, UniChart, World Bank   Figure
    images/               20 photos (photos.jsonl), 20 with human captions (captions.jsonl), 20 flowcharts,
                          20 drawings per label in graphics/   Image, Complex-Block, Diagram, Figure, ...
    fonts/                20 families (10 with Arabic letters, 8 Latin, 2 monospace)
    _lid/                 fastText's language identifier (checks English reference titles)
"""
import json
import random
import shutil
import sys

from pipeline import content_pools as C
from pipeline.pools import ROOT, SAMPLE_POOL, ImagePool, chart_data, home, pool_file

N = 20
FONTS = ["Amiri", "Noto Naskh Arabic", "Noto Kufi Arabic", "Noto Sans Arabic", "IBM Plex Sans Arabic", "Markazi Text",
         "Scheherazade New", "Reem Kufi", "El Messiri", "Readex Pro",  # with Arabic letters
         "Open Sans", "Roboto Condensed", "DM Sans", "Lora", "Libre Baskerville", "Playfair Display", "PT Serif",
         "Work Sans",  # Latin
         "Source Code Pro", "JetBrains Mono"]  # monospace (code blocks)


def take(name, prefix, ok, rng, n=N, tries=4000):
    """n items of a pool that ok(item) accepts (different ones)."""
    reader, out, seen = C.pool(name, prefix), [], set()
    for _ in range(tries):
        if len(out) == n or not len(reader):
            break
        item = reader.pick(rng)
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)[:2000]
        if key not in seen and ok(item):
            seen.add(key)
            out.append(item)
    return out


def write(folder, file, items):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / file).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items), encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if C.POOLS == SAMPLE_POOL:
        sys.exit("the full pools are not here (dataset_pool/): the sample is made from them")
    rng = random.Random(20)
    if SAMPLE_POOL.exists():
        shutil.rmtree(SAMPLE_POOL)
    out = SAMPLE_POOL
    readable = lambda d, lang: len(C.split_markdown(d["text"], lang)[2]) >= 3  # noqa: E731  (big_text_pool's test)
    for lang in ("ar", "en"):
        write(out / "text" / lang, "text-00000.jsonl", take(f"text/{lang}", "", lambda d: readable(d, lang), rng))
        write(out / "captions" / lang, "captions-00000.jsonl",
              take(f"captions/{lang}", "", lambda c: len(str(c.get("text", "")).split()) >= 3, rng))
        write(out / "tables" / lang, "tables-00000.jsonl",
              take(f"tables/{lang}", "", lambda t: len(C._cells(t)) >= 3 and all(len(c) <= 60 for r in C._cells(t) for c in r), rng))
        write(out / "references" / lang, "references-00000.jsonl",
              take(f"references/{lang}", "", lambda w: C.work_entry(w, lang) is not None, rng))
    for kind in ("page-header", "footnote"):
        write(out / "layout", f"{kind}-00000.jsonl", take("layout", kind, lambda x: len(x["text"].split()) >= 2, rng))
    write(out / "vocab", "forms-00000.jsonl", take("vocab", "forms", lambda f: bool(f.get("sections")), rng))
    write(out / "vocab", "flowcharts-00000.jsonl", take("vocab", "flowcharts", lambda f: len(f.get("steps", [])) >= 6, rng))
    write(out / "code", "code-00000.jsonl", take("code", "", lambda c: C.code_of(c) is not None, rng))
    write(out / "molecules", "molecules-00000.jsonl", take("molecules", "", lambda m: m.get("heavy_atoms", 99) <= 30, rng))
    write(out / "equations", "equations-00000.jsonl", take("equations", "", lambda e: len(e["latex"]) <= 120, rng))
    for source in ("sbsfigures", "unichart", "worldbank"):
        write(out / "chartdata", f"{source}-00000.jsonl",
              take("chartdata", source, lambda c: chart_data(c, "en", random.Random(0)) is not None, rng))
    C.index_pools(out)

    # pictures: their index lines and files, at the same paths under images/
    images, pool = home("images"), ImagePool()
    def copy(rel):
        (out / "images" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(images / rel, out / "images" / rel)
    for index in ("photos.jsonl", "captions.jsonl"):
        lines = [json.loads(x) for x in open(images / index, encoding="utf-8")]
        picked = [r for r in rng.sample(lines, 400) if (images / r["file"]).exists() and r.get("captions")][:N]
        write(out / "images", index, picked)
        for r in picked:
            copy(r["file"])
    flows = [pool.flowcharts.pick(rng) for _ in range(N)]
    for r in flows:
        rel = pool_file(r["file"], "images").relative_to(images)
        copy(rel)
        r["file"] = rel.as_posix()  # under images/
    write(out / "images" / "flowcharts", "flowcharts.jsonl", flows)
    drawn = []
    for label, by_lang in pool.graphics.items():
        found = [r for items in by_lang.values() for _, r in items]
        drawn += rng.sample(found, min(N, len(found)))
    for r in drawn:
        copy(r["file"])
    write(out / "images" / "graphics", "graphics.jsonl", drawn)

    # fonts and the language identifier
    fonts = [f for f in json.loads((home("fonts") / "fonts-more.json").read_text(encoding="utf-8")) if f["family"] in FONTS]
    for f in fonts:
        for p in f["files"].values():
            src = pool_file(p, "fonts")
            dst = out / "fonts" / src.relative_to(home("fonts"))
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    (out / "fonts" / "fonts-more.json").write_text(json.dumps(fonts, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "_lid").mkdir(parents=True, exist_ok=True)
    shutil.copy2(C.LID_MODEL, out / "_lid" / C.LID_MODEL.name)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"{out.relative_to(ROOT)}: {sum(1 for p in out.rglob('*') if p.is_file())} files, {size / 2**20:.1f} MB; "
          f"{len(fonts)} font families")


if __name__ == "__main__":
    main()
