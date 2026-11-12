"""Large, varied content pools for every label (made once, not in git), next to the small pools of pools.py.

    python -m pipeline.content_pools photos      -> img_pool/pd12m/ + img_pool/photos.jsonl   (PD12M, many collections)
    python -m pipeline.content_pools text        -> text_pool/pools/text/<lang>/   (+ tables/, captions/ found in it)
    python -m pipeline.content_pools tables      -> text_pool/pools/tables/en/     (PubTabNet, FinTabNet, SynthTabNet)
    python -m pipeline.content_pools equations   -> text_pool/pools/equations/     (arXiv, Wikipedia, UniMER LaTeX)
    python -m pipeline.content_pools code        -> text_pool/pools/code/          (CodeSearchNet functions)
    python -m pipeline.content_pools molecules   -> text_pool/pools/molecules/     (PubChem SMILES)
    python -m pipeline.content_pools references  -> text_pool/pools/references/ar/  (OpenAlex works; a key in .env for more)
    python -m pipeline.content_pools references-en -> text_pool/pools/references/en/  (Crossref works)
    python -m pipeline.content_pools chartdata   -> text_pool/pools/chartdata/     (World Bank series, Arabic + English names)
    python -m pipeline.content_pools text-more   -> text_pool/pools/text/ar/   (HPLT, 101 Billion Arabic Words; no copies)
    python -m pipeline.content_pools cosyn       -> text_pool/pools/cosyn/         (CoSyn-400K rendering code + data)
    python -m pipeline.content_pools layout      -> text_pool/pools/layout/        (DocLayNet headers, footers, footnotes...)
    python -m pipeline.content_pools parallel    -> text_pool/pools/parallel/      (UN Arabic-English sentences)
    python -m pipeline.content_pools arxivcap    -> text_pool/pools/captions/en/   (arXiv figure captions)
    python -m pipeline.content_pools owid        -> text_pool/pools/chartdata/owid-*  (Our World in Data charts, English)
    python -m pipeline.content_pools index       -> <file>.idx next to every pool file (run after any build: the
                                                    generator reads the pools through these)
    python -m pipeline.content_pools vocab       -> text_pool/pools/vocab/         (flowchart steps, form fields: LLM, ar + en)
    python -m pipeline.content_pools vocab-more  -> text_pool/pools/vocab/*-h5-*   (15 more kinds of process per sector)
    python -m pipeline.content_pools charts-more -> text_pool/pools/chartdata/     (SBS Figures specs, UniChart tables)
    python -m pipeline.content_pools chartgen    -> text_pool/pools/chartdata/     (ChartGen-200K code + CSV + summary)
    python -m pipeline.content_pools plotqa      -> text_pool/pools/chartdata/     (PlotQA series)
    python -m pipeline.content_pools kaggle-charts -> text_pool/pools/chartdata/   (VisText, Benetech; Kaggle CLI first)
    python -m pipeline.content_pools diagrams    -> text_pool/pools/diagrams/      (TikZ, PlantUML, Mermaid code)
    python -m pipeline.content_pools fonts-more  -> font_pool/fonts-more.json      (100 readable fonts, top 10s included)
    python -m pipeline.content_pools lang        -> drops what is not English or Arabic from every pool (run after a build)
    python -m pipeline.flowchart_pool --n 50000  -> img_pool/flowcharts/           (drawn flowcharts + Surya's answer)

Every pool is JSON lines in files of at most 50,000 lines (<source>-00000.jsonl, ...), one item per line, each
with its source and licence. Duplicates are dropped inside and across sources: every item (for text: every
paragraph) is reduced to a key (Arabic letters folded, tashkeel and digits dropped, punctuation and case ignored)
and a Bloom filter remembers the keys already taken. Datasets are read in pieces (random parquet row groups over
all of a dataset's files), so a pool mixes the whole dataset, and nothing is kept on disk but the pools.
"""
import argparse
import csv
import html
import io
import json
import os
import random
import re
import shutil
import subprocess
import time
import unicodedata
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import xxhash

from pipeline.pools import DATASET_POOL, IMG_POOL, ROOT, SAMPLE_POOL, fetch_pd12m

# the content pools: dataset_pool/<name> once moved there (pipeline/move_pools.py), else text_pool/pools/<name>,
# else (a fresh copy of the repository) the sample pool, dataset_pool_sample/
POOLS = next((p for p in (DATASET_POOL, ROOT / "text_pool" / "pools", SAMPLE_POOL) if (p / "text").exists()),
             ROOT / "text_pool" / "pools")
DOWNLOAD = POOLS / "_download"
SHARD = 50_000


# -- shared: writing, keys, reading datasets in pieces ---------------------------------------------------------------

class Shards:
    """Writes items to folder/<name>-00000.jsonl, a new file every SHARD items."""

    def __init__(self, folder, name):
        self.folder, self.name, self.n, self.file = Path(folder), name, 0, None
        self.folder.mkdir(parents=True, exist_ok=True)

    def add(self, item):
        if self.n % SHARD == 0:
            self.close()
            self.file = open(self.folder / f"{self.name}-{self.n // SHARD:05d}.jsonl", "w", encoding="utf-8")
        self.file.write(json.dumps(item, ensure_ascii=False) + "\n")
        self.n += 1

    def close(self):
        if self.file:
            self.file.close()


class Seen:
    """A Bloom filter (2**bits bits, 3 probes): new(key) says whether a key is new, and remembers it."""

    def __init__(self, bits=31):
        self.mask, self.bits = (1 << bits) - 1, bytearray(1 << (bits - 3))

    def has(self, key):
        return all(self.bits[i >> 3] >> (i & 7) & 1 for i in self._probes(key))

    def add(self, key):
        for i in self._probes(key):
            self.bits[i >> 3] |= 1 << (i & 7)

    def new(self, key):
        if self.has(key):
            return False
        self.add(key)
        return True

    def _probes(self, key):
        return [(key >> (40 * k)) & self.mask for k in range(3)]


FOLD = {**dict.fromkeys(map(ord, "أإآٱ"), "ا"), ord("ى"): "ي", ord("ة"): "ه", ord("ؤ"): "و", ord("ئ"): "ي"}
MARKS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")  # tashkeel and tatweel


def key(text, digits=False):
    """A 128-bit key for duplicates: the same for texts that differ only in tashkeel, alef/ya/ta-marbuta forms,
    case, spacing and punctuation (and digits, unless digits=True)."""
    text = MARKS.sub("", unicodedata.normalize("NFKC", text)).translate(FOLD).lower()
    text = re.sub(r"[\W_]+" if digits else r"[\W_\d]+", " ", text).strip()
    return xxhash.xxh3_128_intdigest(text.encode("utf-8"))


_meta = {}


def parquet_files(repo, pattern, max_files=None, rng=None):
    from huggingface_hub import HfFileSystem
    files = sorted(HfFileSystem().glob(f"datasets/{repo}/{pattern}"))
    if max_files and len(files) > max_files:
        files = sorted((rng or random).sample(files, max_files))
    return files


def row_group(path, rg, columns, tries=5):
    """One row group of a parquet file on Hugging Face, only the columns asked for (ranged reads, so a column
    of images is never downloaded)."""
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq
    for k in range(tries):
        try:
            with HfFileSystem().open(path, block_size=4 << 20) as f:
                pf = pq.ParquetFile(f, metadata=_meta.get(path))
                _meta[path] = pf.metadata
                return pf.read_row_group(rg, columns=columns).to_pylist()
        except Exception as e:  # noqa: BLE001  (network)
            print(f"  retry {k + 1} {path} rg {rg}: {str(e)[:120]}", flush=True)
            time.sleep(10 * (k + 1))
    return []


def sampled_rows(files, columns, rng, take=0.35, threads=8):
    """Rows of random row groups over all the files (a random `take` of each group), lazily."""
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq
    jobs = []
    for path in files:
        with HfFileSystem().open(path) as f:
            _meta[path] = pq.ParquetFile(f).metadata
        jobs += [(path, rg) for rg in range(_meta[path].num_row_groups)]
    rng.shuffle(jobs)
    with ThreadPoolExecutor(threads) as pool:
        for i in range(0, len(jobs), threads * 2):
            for rows in pool.map(lambda j: row_group(j[0], j[1], columns), jobs[i:i + threads * 2]):
                yield from rng.sample(rows, int(len(rows) * take))


def download(url, path, tries=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "OCRSynthGen pool builder"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            return path
        except Exception as e:  # noqa: BLE001
            print(f"  retry {k + 1} {url}: {e}", flush=True)
            time.sleep(10 * (k + 1))
    raise RuntimeError(f"could not download {url}")


def get_json(url, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "OCRSynthGen pool builder"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            print(f"  retry {k + 1} {url[:100]}: {e}", flush=True)
            if 400 <= getattr(e, "code", 0) < 500 and e.code != 429:
                return None  # refused: asking again won't help
            time.sleep(5 * (k + 1))
    return None


# -- photos: PD12M, spread over its collections -------------------------------------------------------------------

def build_photos(n=300_000, shards=40, share=0.06, threads=32, seed=1, max_side=1600):
    """n PD12M photos (public domain / CC0) into img_pool/pd12m/, indexed in img_pool/photos.jsonl (the same lines
    as captions.jsonl). Candidates come from `shards` of the 125 metadata files; no collection gives more than
    `share` of the photos until the rest run out, so museums, archives and photo libraries are all in. Photos
    over `max_side` px are skipped: they are shrunk to 1,024 px anyway, and big ones are slow to download.
    Starts again where it stopped."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    index = IMG_POOL / "photos.jsonl"
    done = {json.loads(line)["file"] for line in open(index, encoding="utf-8")} if index.exists() else set()
    rng = random.Random(seed)
    cand = []
    for s in rng.sample(range(125), shards):
        path = hf_hub_download("Spawning/PD12M", f"metadata/pd12m.{s:03d}.parquet", repo_type="dataset",
                               local_dir=DOWNLOAD)
        rows = [r for r in pq.read_table(path, columns=["id", "url", "caption", "width", "height", "license", "source"]).to_pylist()
                if 300 <= min(int(r["width"]), int(r["height"])) and max(int(r["width"]), int(r["height"])) <= max_side
                and 0.4 <= int(r["width"]) / max(1, int(r["height"])) <= 2.5]
        cand += rng.sample(rows, min(len(rows), 3 * n // shards))
        Path(path).unlink()
    rng.shuffle(cand)
    per, first, rest = Counter(), [], []
    for r in cand:
        (first if per[r["source"]] < share * n else rest).append(r)
        per[r["source"]] += 1
    todo = [r for r in first + rest if f"pd12m/{r['id']}.jpg" not in done]
    print(f"photos: {len(done)} done, {len(todo)} candidates from {len(per)} collections", flush=True)
    with open(index, "a", encoding="utf-8") as out, ThreadPoolExecutor(threads) as pool:
        for i in range(0, len(todo), threads * 8):
            if len(done) >= n:
                break
            for r in pool.map(fetch_pd12m, todo[i:i + threads * 8]):
                if r and r["file"] not in done and len(done) < n:
                    done.add(r["file"])
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
            out.flush()
            print(f"  photos: {len(done)}", flush=True)


# -- text: Wikipedia, web and PDF documents (and the tables and captions in them) -----------------------------------

TEXT = {"ar": [("finewiki", "HuggingFaceFW/finewiki", "data/arwiki/*.parquet", 400_000, None, "CC BY-SA 4.0 (Wikipedia)"),
               ("fineweb2", "HuggingFaceFW/fineweb-2", "data/arb_Arab/train/*.parquet", 500_000, None,
                "ODC-By 1.0 (FineWeb-2; Common Crawl terms of use)"),
               ("finepdfs", "HuggingFaceFW/finepdfs", "data/arb_Arab/train/*.parquet", 150_000, None,
                "ODC-By 1.0 (FinePDFs; Common Crawl terms of use)")],
        "en": [("finewiki", "HuggingFaceFW/finewiki", "data/enwiki/*.parquet", 400_000, 8, "CC BY-SA 4.0 (Wikipedia)"),
               ("fineweb-edu", "HuggingFaceFW/fineweb-edu", "sample/10BT/*.parquet", 500_000, None,
                "ODC-By 1.0 (FineWeb-Edu; Common Crawl terms of use)"),
               ("finepdfs", "HuggingFaceFW/finepdfs", "data/eng_Latn/train/*.parquet", 150_000, 10,
                "ODC-By 1.0 (FinePDFs; Common Crawl terms of use)")]}
# more Arabic web text, added later (python -m pipeline.content_pools text-more): copies of what the pool has are left out
TEXT_MORE = {"ar": [("hplt", "HPLT/HPLT2.0_cleaned", "ara_Arab_*/*.parquet", 300_000, 24,
                     "CC0-1.0 tag (HPLT 2.0; the text's rights stay with its sites)"),
                    ("arabic101b", "ClusterlabAi/101_billion_arabic_words_dataset", "data/*.parquet", 300_000, 30,
                     "Apache-2.0 (101 Billion Arabic Words)")]}
MAX_CHARS = 20_000
ARABIC = re.compile(r"[\u0621-\u064a]")
LATIN = re.compile(r"[A-Za-z]")
CAPTION = re.compile(r"^\W{0,3}(Figure|Fig\.|Table|Chart|Map|Plate|Diagram|شكل|الشكل|جدول|الجدول|صورة|الصورة|خريطة|رسم بياني)"
                     r"\s*\(?[0-9٠-٩]+(?:[.\-–][0-9٠-٩]+)*\)?\s*[:.\-–]\s*\S.{6,240}$", re.M | re.I)
MD_TABLE = re.compile(r"(?:^\|.*\|[ \t]*\n?){3,}", re.M)


SHAPED = re.compile(r"[ﭐ-﷿ﹰ-﻿]")  # Arabic letters in their drawn shapes: text cut from a PDF


def broken_arabic(text):
    """Arabic from a PDF that lost its order: letters left in their drawn shapes (U+FExx), or words written
    backwards."""
    return len(SHAPED.findall(text)) > 0.02 * max(1, len(ARABIC.findall(text))) or reversed_arabic(text)


def reversed_arabic(text):
    """Arabic from a PDF written backwards ("لدعملا" for "المعدل"): words end in "لا" more than they start with "ال"."""
    words = re.findall(r"[\u0621-\u064a]{3,}", text)
    return sum(w.endswith("لا") for w in words) >= max(3, sum(w.startswith("ال") for w in words))


def good_text(text, lang):
    """The document tidied and cut to MAX_CHARS at a paragraph end, or None if it is not clean text in `lang`."""
    text = re.sub(r"\n{3,}", "\n\n", text.replace("\r", "")).strip()
    if len(text) > MAX_CHARS:
        text = text[:text.rfind("\n\n", 0, MAX_CHARS)].strip()
    ar, la = len(ARABIC.findall(text)), len(LATIN.findall(text))
    if len(text) < 400 or ar + la < 300:
        return None
    if lang == "ar" and (ar < 0.75 * (ar + la) or broken_arabic(text)):
        return None
    if lang == "en" and (la < 0.95 * (ar + la)):
        return None
    return text


def md_tables(text):
    """The Markdown tables of a document as lists of rows of cells (2-40 rows, 2-12 columns)."""
    for block in MD_TABLE.findall(text):
        rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in block.strip().splitlines()]
        rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) or not c for c in r)]
        if 2 <= len(rows) <= 40 and 2 <= max(map(len, rows)) <= 12 and sum(bool(c) for r in rows for c in r) >= 6:
            yield rows


def remember(seen, folder, keys_of):
    """Put the keys of every item already in a pool folder into a Bloom filter (so new sources skip copies)."""
    for path in sorted(Path(folder).glob("*.jsonl")):
        for line in open(path, encoding="utf-8"):
            for k in keys_of(json.loads(line)):
                seen.add(k)
        print(f"  remembered {path.name}", flush=True)


def build_text(lang, seed=0, only=None):
    """The text pool of one language; with `only` (names in TEXT_MORE), just those sources, added to the pool."""
    rng = random.Random(seed)
    seen, seen_tables, seen_caps = Seen(31), Seen(27), Seen(26)
    sources = TEXT[lang] if not only else [s for s in TEXT[lang] + TEXT_MORE.get(lang, []) if s[0] in only]
    if only:
        para_keys = lambda r: [key(p) for p in re.split(r"\n+", r["text"]) if len(p) >= 40 and not p.startswith("|")]  # noqa: E731
        remember(seen, POOLS / "text" / lang, para_keys)
        remember(seen_tables, POOLS / "tables" / lang, lambda r: [key(json.dumps(r["rows"], ensure_ascii=False), digits=True)])
        remember(seen_caps, POOLS / "captions" / lang, lambda r: [key(r["text"], digits=True)])
    tables = Shards(POOLS / "tables" / lang, "-".join(only) if only else "text")
    captions = Shards(POOLS / "captions" / lang, "-".join(only) if only else "text")
    for name, repo, pattern, want, max_files, license_ in sources:
        out = Shards(POOLS / "text" / lang, name)
        url = "u" if name == "hplt" else "url"
        columns = ["text", url] + (["title"] if name == "finewiki" else [])
        read = dropped = 0
        take = 1.0 if name == "finewiki" else 0.35  # every Wikipedia article with sections; a spread of the rest
        for row in sampled_rows(parquet_files(repo, pattern, max_files, rng), columns, rng, take):
            read += 1
            text = good_text(row["text"] or "", lang)
            paras = [key(p) for p in re.split(r"\n+", text or "") if len(p) >= 40 and not p.startswith("|")]
            if not paras or (name == "finewiki" and text.count("\n## ") < 2):
                dropped += 1
                continue
            if 2 * sum(seen.has(k) for k in paras) >= len(paras):  # half its paragraphs are taken already
                dropped += 1
                continue
            for k in paras:
                seen.add(k)
            out.add({"id": f"{name}/{out.n}", "lang": lang, "source": name, "license": license_, "url": row[url],
                     "title": row.get("title"), "text": text})
            for rows in md_tables(text):
                if seen_tables.new(key(json.dumps(rows, ensure_ascii=False), digits=True)):
                    tables.add({"lang": lang, "source": name, "license": license_, "url": row[url], "rows": rows})
            for m in CAPTION.finditer(text):
                line = m.group(0).strip(" *#>")
                if seen_caps.new(key(line, digits=True)):
                    captions.add({"lang": lang, "source": name, "license": license_, "kind": m.group(1), "text": line})
            if out.n % 10_000 == 0:
                print(f"  text {lang} {name}: {out.n} kept of {read} read; tables {tables.n}, captions {captions.n}",
                      flush=True)
            if out.n >= want:
                break
        out.close()
        print(f"text {lang} {name}: {out.n} documents ({dropped} of {read} dropped)", flush=True)
    tables.close()
    captions.close()


# -- tables: real table structures with merged cells ----------------------------------------------------------------

TABLES = [("pubtabnet", "docling-project/PubTabNet_OTSL", "CDLA-Permissive-1.0 (PubTabNet annotations)"),
          ("fintabnet", "docling-project/FinTabNet_OTSL", "CDLA-Permissive-1.0 (FinTabNet, IBM)"),
          ("synthtabnet", "docling-project/SynthTabNet_OTSL", "CDLA-Permissive-1.0 (SynthTabNet)")]


def otsl_html(structure, cells):
    """PubTabNet-style structure tokens ('<td', ' colspan="2"', '>', '</td>') with the cells' text put in."""
    out, k = [], 0
    for tok in structure:
        out.append(tok)
        if tok in ("<td>", ">"):
            if k < len(cells):  # a token is a tag (<b>, <sup>) or one character, which may be < > &
                out.append("".join(t if len(t) > 1 else html.escape(t, quote=False) for t in cells[k].get("tokens") or []))
            k += 1
    return "".join(out)


def build_tables(seed=0):
    rng = random.Random(seed)
    seen = Seen(28)
    for name, repo, license_ in TABLES:
        out = Shards(POOLS / "tables" / "en", name)
        for row in sampled_rows(parquet_files(repo, "data/*.parquet"), ["html", "cells", "rows", "cols", "filename"],
                                rng, take=1.0):
            cells = (row["cells"] or [[]])[0]
            if not (2 <= (row["rows"] or 0) <= 40 and 2 <= (row["cols"] or 0) <= 12) or sum(bool(c.get("tokens")) for c in cells) < 4:
                continue
            table = "<table>" + otsl_html(row["html"], cells) + "</table>"
            if seen.new(key(table, digits=True)):
                out.add({"lang": "en", "source": name, "license": license_, "file": row["filename"], "rows": row["rows"],
                         "cols": row["cols"], "spans": "span=" in table, "html": table})
                if out.n % 50_000 == 0:
                    print(f"  tables {name}: {out.n}", flush=True)
        out.close()
        print(f"tables {name}: {out.n}", flush=True)


# -- equations: LaTeX from arXiv, Wikipedia and UniMER --------------------------------------------------------------

EQUATIONS = [("arxiv", "OleehyO/latex-formulas", "raw_formulas/*.parquet", "latex_formula", "OpenRAIL (latex-formulas)"),
             ("arxiv", "OleehyO/latex-formulas", "cleaned_formulas/*.parquet", "latex_formula", "OpenRAIL (latex-formulas)"),
             ("wikipedia", "piushorn/wikipedia-latex-formulas-319k", "data.parquet", "formula", "CC BY-SA 4.0 (Wikipedia)")]
CSV_EQUATIONS = [("fusion", "https://huggingface.co/datasets/hoang-quoc-trung/fusion-image-to-latex-datasets/resolve/main/"
                  "train/printed_mathematical_expressions_train.csv", "Apache-2.0 tag (merged printed sets)")]


def tidy_latex(s):
    s = re.sub(r"\s+", " ", (s or "").strip())
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = re.sub(r"\\(label|tag|nonumber|notag)\b(\{[^{}]*\})?", "", s).strip()
    s = re.sub(r"[,.;]\s*$", "", s).strip()
    return s


def skeleton(s):
    """The shape of a formula: numbers and single letters made the same, so near-copies can be capped."""
    return re.sub(r"(?<![\\a-zA-Z])[a-zA-Z](?![a-zA-Z])", "x", re.sub(r"\d+(\.\d+)?", "0", s))


def build_equations(seed=0, per_shape=3):
    """LaTeX formulas, tidied (no label, tag or $), 4-400 characters, balanced braces, no macros of their own;
    exact copies dropped and at most `per_shape` formulas of one shape (same symbols, other numbers or letters)."""
    rng = random.Random(seed)
    seen, shapes = Seen(28), Counter()
    out = Shards(POOLS / "equations" / "raw", "latex")

    def add(raw, source, license_):
        s = tidy_latex(raw)
        if not 4 <= len(s) <= 400 or s.count("{") != s.count("}") or "\\newcommand" in s or "\\def" in s:
            return
        shape = xxhash.xxh64_intdigest(skeleton(s).encode("utf-8"))
        if shapes[shape] < per_shape and seen.new(key(re.sub(r"\\[,;:!]|\\(left|right)\b|\s", "", s), digits=True)):
            shapes[shape] += 1
            out.add({"source": source, "license": license_, "latex": s})

    for name, repo, pattern, col, license_ in EQUATIONS:
        before = out.n
        for row in sampled_rows(parquet_files(repo, pattern), [col], rng, take=1.0):
            add(row[col], name, license_)
        print(f"equations {name} ({repo}/{pattern}): {out.n - before}", flush=True)
    for name, url, license_ in CSV_EQUATIONS:
        before, path = out.n, download(url, DOWNLOAD / f"{name}.csv")
        with open(path, encoding="utf-8", newline="") as f:
            rows = csv.DictReader(f)
            col = next(c for c in rows.fieldnames if "latex" in c.lower() or "formula" in c.lower() or "label" in c.lower())
            for row in rows:
                add(row[col], name, license_)
        path.unlink()
        print(f"equations {name}: {out.n - before}", flush=True)
    path = download("https://huggingface.co/datasets/wanderkid/UniMER_Dataset/resolve/main/UniMER-1M.zip",
                    DOWNLOAD / "UniMER-1M.zip")
    with zipfile.ZipFile(path) as z:
        before = out.n
        for name in [n for n in z.namelist() if n.endswith(".txt") and "train" in n.lower()]:
            for line in io.TextIOWrapper(z.open(name), encoding="utf-8"):
                add(line, "unimer", "Apache-2.0 tag (UniMER-1M)")
        print(f"equations unimer: {out.n - before}", flush=True)
    path.unlink()
    out.close()
    print(f"equations: {out.n}", flush=True)


# -- code: functions in six languages -------------------------------------------------------------------------------

def tidy_code(text):
    """A function with its first line at the margin and its body one level in (the dataset strips the first
    line's indent but keeps the body's), or None when its indentation is odd."""
    lines = (text or "").expandtabs(4).rstrip().splitlines()
    body = lines[1:]
    if not body or not any(x.strip() for x in body):
        return None
    last = body[-1]
    if last.strip() in ("}", "};", "})", "});", "end"):  # the closing line is where the first line was
        cut = len(last) - len(last.lstrip())
    else:  # Python: the body is one level (4) in
        cut = max(0, min(len(x) - len(x.lstrip()) for x in body if x.strip()) - 4)
    if any(x.strip() and len(x) - len(x.lstrip()) < cut for x in body):
        return None
    return "\n".join([lines[0].strip()] + [x[cut:] for x in body])


def build_code(seed=0, per_repo=15, per_language=100_000):
    """Functions of 4-30 lines no wider than 90 characters, at most `per_repo` from one repository and
    `per_language` per language (Go, Java, JavaScript, PHP, Python, Ruby)."""
    rng = random.Random(seed)
    seen, repos, langs = Seen(27), Counter(), Counter()
    out = Shards(POOLS / "code", "codesearchnet")
    files = parquet_files("code-search-net/code_search_net", "all/*.parquet")
    for row in sampled_rows(files, ["func_code_string", "language", "repository_name", "func_code_url"], rng, take=0.6):
        code = tidy_code(row["func_code_string"])
        lines = code.splitlines() if code else []
        if not 4 <= len(lines) <= 30 or max(map(len, lines)) > 90 or langs[row["language"]] >= per_language:
            continue
        if repos[row["repository_name"]] >= per_repo or not seen.new(key(code, digits=True)):
            continue
        repos[row["repository_name"]] += 1
        langs[row["language"]] += 1
        out.add({"source": "codesearchnet", "license": f"licence of {row['repository_name']} (GitHub)",
                 "language": row["language"], "url": row["func_code_url"], "code": code})
        if out.n % 50_000 == 0:
            print(f"  code: {out.n} {dict(langs)}", flush=True)
    out.close()
    print(f"code: {out.n} {dict(langs)}", flush=True)


# -- molecules: PubChem SMILES, many ring systems -------------------------------------------------------------------

def build_molecules(n=500_000, per_scaffold=2, seed=0):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    import numpy as np
    import pyarrow as pa
    parts = []
    for name in [f.split("canonicalized/", 1)[1] for f in parquet_files("sagawa/pubchem-10m-canonicalized", "**/*.parquet")]:
        path = Path(hf_hub_download("sagawa/pubchem-10m-canonicalized", name, repo_type="dataset", local_dir=DOWNLOAD))
        parts.append(pq.read_table(path, columns=["smiles"]).column("smiles"))
        path.unlink()
    column = pa.chunked_array([c for p in parts for c in p.chunks]).combine_chunks()  # kept as Arrow: little memory
    smiles = (column[int(i)].as_py() for i in np.random.default_rng(seed).permutation(len(column)))
    scaffolds, seen = Counter(), set()
    out = Shards(POOLS / "molecules", "pubchem")
    for s in smiles:
        mol = Chem.MolFromSmiles(s or "")
        if mol is None or "." in s or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
            continue
        if not 8 <= mol.GetNumHeavyAtoms() <= 45 or mol.GetRingInfo().NumRings() > 7 or not mol.GetRingInfo().NumRings():
            continue
        flat = Chem.MolToSmiles(mol, isomericSmiles=False)
        scaffold = Chem.MolToSmiles(Chem.MurckoDecompose(mol))  # (MurckoScaffold's module is blocked on Windows)
        if flat in seen or scaffolds[scaffold] >= per_scaffold:
            continue
        seen.add(flat)
        scaffolds[scaffold] += 1
        out.add({"source": "pubchem", "license": "public domain (PubChem; via sagawa/pubchem-10m-canonicalized, Apache-2.0)",
                 "smiles": flat, "heavy_atoms": mol.GetNumHeavyAtoms()})
        if out.n % 50_000 == 0:
            print(f"  molecules: {out.n} ({len(scaffolds)} ring systems)", flush=True)
        if out.n >= n:
            break
    out.close()
    print(f"molecules: {out.n} from {len(scaffolds)} ring systems", flush=True)


# -- references: OpenAlex works, for bibliographies and footnotes ----------------------------------------------------

OPENALEX = "https://api.openalex.org/works?per-page=200&select=id,doi,display_name,publication_year,type,authorships,primary_location,biblio,language"


def work(w, lang):
    loc = w.get("primary_location") or {}
    src = loc.get("source") or {}
    b = w.get("biblio") or {}
    authors = [a["author"]["display_name"] for a in w.get("authorships") or [] if (a.get("author") or {}).get("display_name")]
    title = re.sub(r"<[^>]+>|\s+", " ", w.get("display_name") or "").strip()
    if not title or not authors or not w.get("publication_year"):
        return None
    if lang == "ar" and not ARABIC.search(title):
        return None
    return {"id": w["id"].rsplit("/", 1)[-1], "lang": lang, "type": w.get("type"), "title": title, "authors": authors[:8],
            "year": w["publication_year"], "venue": src.get("display_name"), "publisher": src.get("host_organization_name"),
            "volume": b.get("volume"), "issue": b.get("issue"), "first_page": b.get("first_page"),
            "last_page": b.get("last_page"), "doi": w.get("doi"), "source": "openalex", "license": "CC0 (OpenAlex)"}


def build_references(n_ar=900_000):
    """Arabic works from OpenAlex's API (English ones: build_references_en, from Crossref). Works are read in
    slices of one quarter of a year (OpenAlex gives at most 10,000 per query without cursor paging). Without a key
    (OPENALEX_API_KEY in .env) OpenAlex allows one small free budget a day per network (about 1,000 pages of
    200 works), so a run stops when it is used up and the next run picks up where it stopped
    (references/ar/slices_done.json lists the slices read whole)."""
    from pipeline.llm_templates import read_env
    api_key = read_env().get("OPENALEX_API_KEY")
    url = OPENALEX + (f"&api_key={api_key}" if api_key else "")
    folder = POOLS / "references" / "ar"
    folder.mkdir(parents=True, exist_ok=True)
    state = folder / "slices_done.json"
    done = set(json.loads(state.read_text())) if state.exists() else set()
    seen = {json.loads(line)["id"] for path in folder.glob("*.jsonl") for line in open(path, encoding="utf-8")}
    out = Shards(folder, f"openalex-{time.strftime('%Y%m%d-%H%M')}")
    quarters = [("01-01", "03-31"), ("04-01", "06-30"), ("07-01", "09-30"), ("10-01", "12-31")]
    try:
        for year in range(2026, 1949, -1):
            for q, (start, end) in enumerate(quarters, 1):
                slice_ = f"{year}Q{q}"
                if slice_ in done:
                    continue
                for p in range(1, 51):
                    page = get_json(f"{url}&filter=language:ar,from_publication_date:{year}-{start},"
                                    f"to_publication_date:{year}-{end}&page={p}", tries=3)
                    if page is None:
                        raise RuntimeError("OpenAlex refused (the free daily budget is used up): run again tomorrow")
                    for w in page["results"]:
                        r = work(w, "ar")
                        if r and r["id"] not in seen:
                            seen.add(r["id"])
                            out.add(r)
                    if len(page["results"]) < 200:
                        break
                    time.sleep(0.12)
                done.add(slice_)
                state.write_text(json.dumps(sorted(done)))
                print(f"  references ar: {len(seen)} ({slice_})", flush=True)
                if len(seen) >= n_ar:
                    raise RuntimeError("enough")
    except RuntimeError as e:
        print(f"  references ar: stopped: {e}", flush=True)
    out.close()
    print(f"references ar: {len(seen)} ({out.n} new)", flush=True)


CROSSREF = ("https://api.crossref.org/works?sample=100&select=DOI,title,author,container-title,published,volume,issue,"
            "page,type,publisher&filter=type:{type}")


def build_references_en(n=300_000):
    """English works from Crossref (no key needed; metadata facts, mostly CC0): random samples of 100 per kind of
    work, Latin-script titles only."""
    types = ["journal-article", "book-chapter", "proceedings-article", "book", "report", "dissertation",
             "monograph", "posted-content"]
    out, seen, k = Shards(POOLS / "references" / "en", "crossref"), set(), 0
    while out.n < n and k < 20 * n // 100:
        k += 1
        page = get_json(CROSSREF.format(type=types[k % len(types)]), tries=4)
        for w in (page or {}).get("message", {}).get("items", []):
            title = re.sub(r"<[^>]+>|\s+", " ", " ".join(w.get("title") or [])).strip()
            authors = [" ".join(x for x in (a.get("given"), a.get("family")) if x) or a.get("name", "")
                       for a in w.get("author") or []]
            authors = [a for a in authors if a]
            year = ((w.get("published") or {}).get("date-parts") or [[None]])[0][0]
            if not title or ARABIC.search(title) or len(LATIN.findall(title)) < 0.8 * len(title.replace(" ", ""))\
                    or not authors or not year or w["DOI"] in seen:
                continue
            seen.add(w["DOI"])
            pages = (w.get("page") or "").split("-")
            out.add({"id": w["DOI"], "lang": "en", "type": w.get("type"), "title": title, "authors": authors[:8],
                     "year": year, "venue": (w.get("container-title") or [None])[0], "publisher": w.get("publisher"),
                     "volume": w.get("volume"), "issue": w.get("issue"), "first_page": pages[0] or None,
                     "last_page": pages[-1] if len(pages) > 1 else None, "doi": f"https://doi.org/{w['DOI']}",
                     "source": "crossref", "license": "Crossref metadata (facts; mostly CC0)"})
        if k % 100 == 0:
            print(f"  references en: {out.n}", flush=True)
    out.close()
    print(f"references en: {out.n}", flush=True)


# -- chart data: World Bank indicators, names in Arabic and English ----------------------------------------------

def build_chartdata(min_points=8):
    names = {}
    for lang in ("en", "ar"):
        for kind in ("indicator", "country"):
            page, pages = 1, 1
            while page <= pages:
                r = get_json(f"https://api.worldbank.org/v2/{lang}/{kind}?format=json&per_page=1000&page={page}"
                             + ("&source=2" if kind == "indicator" else ""))
                pages = r[0]["pages"]
                for x in r[1]:
                    names.setdefault((kind, x["id"]), {})[lang] = x["name"].strip()
                page += 1
    zipped = download("https://databank.worldbank.org/data/download/WDI_CSV.zip", DOWNLOAD / "WDI_CSV.zip")
    out = Shards(POOLS / "chartdata", "worldbank")
    with zipfile.ZipFile(zipped) as z:
        data = next(n for n in z.namelist() if re.search(r"WDICSV\.csv$|WDIData\.csv$", n))
        member = z.open(data)  # closed below: an open member keeps the zip file locked on Windows
        rows = csv.reader(io.TextIOWrapper(member, encoding="utf-8-sig"))
        head = next(rows)
        years = [i for i, h in enumerate(head) if h.strip().isdigit()]
        for r in rows:
            points = [(int(head[i]), float(r[i])) for i in years if r[i].strip()]
            ind, country = names.get(("indicator", r[3]), {}), names.get(("country", r[1]), {})
            if len(points) < min_points or not ARABIC.search(ind.get("ar", "")) or not ARABIC.search(country.get("ar", "")):
                continue
            out.add({"source": "worldbank", "license": "CC BY 4.0 (World Bank WDI)", "indicator": r[3],
                     "name_en": ind.get("en", r[2]), "name_ar": ind["ar"], "country": r[1],
                     "country_en": country.get("en", r[0]), "country_ar": country["ar"],
                     "years": [y for y, _ in points], "values": [v for _, v in points]})
        member.close()
    out.close()
    zipped.unlink()
    print(f"chartdata: {out.n} series", flush=True)


# -- Our World in Data: 4,000+ real charts' data and titles (English) ----------------------------------------------

def build_owid(per_chart=30, seed=0):
    """Every grapher chart of Our World in Data (its sitemap): title, subtitle, source, its columns (titles and
    units) and the yearly values of up to `per_chart` of its countries or regions: chartdata/owid-*.jsonl.
    CC BY 4.0 (Our World in Data; some data is third-party, named in "citation"). One chart a second."""
    rng = random.Random(seed)
    sitemap = urllib.request.urlopen(urllib.request.Request("https://ourworldindata.org/sitemap.xml",
                                                            headers={"User-Agent": "OCRSynthGen pool builder"}), timeout=60)
    slugs = sorted(set(re.findall(r"https://ourworldindata.org/grapher/([a-z0-9-]+)", sitemap.read().decode())))
    done = {json.loads(line)["slug"] for p in (POOLS / "chartdata").glob("owid*.jsonl") for line in open(p, encoding="utf-8")}
    slugs = [s for s in slugs if s not in done]  # a run again fetches only the charts still missing
    out = Shards(POOLS / "chartdata", f"owid-{time.strftime('%Y%m%d-%H%M')}" if done else "owid")
    for i, slug in enumerate(slugs):
        meta = get_json(f"https://ourworldindata.org/grapher/{slug}.metadata.json", tries=3)
        try:
            req = urllib.request.Request(f"https://ourworldindata.org/grapher/{slug}.csv?useColumnShortNames=true",
                                         headers={"User-Agent": "OCRSynthGen pool builder"})
            rows = list(csv.DictReader(io.StringIO(urllib.request.urlopen(req, timeout=90).read().decode("utf-8"))))
        except Exception as e:  # noqa: BLE001
            print(f"  owid {slug}: {e}", flush=True)
            continue
        if not meta or not rows or "year" not in rows[0] and "day" not in rows[0]:
            continue
        time_col = "year" if "year" in rows[0] else "day"
        cols = [c for c in rows[0] if c not in ("entity", "code", time_col)]
        series = {}
        for r in rows:
            series.setdefault(r["entity"], []).append(r)
        entities = rng.sample(sorted(series), min(per_chart, len(series)))
        info = meta.get("columns") or {}
        out.add({"source": "owid", "license": "CC BY 4.0 (Our World in Data)", "slug": slug,
                 "title": (meta.get("chart") or {}).get("title"), "subtitle": (meta.get("chart") or {}).get("subtitle"),
                 "citation": (meta.get("chart") or {}).get("citation"), "time": time_col,
                 "columns": [{"name": c, "title": (info.get(c) or {}).get("titleShort") or (info.get(c) or {}).get("titleLong"),
                              "unit": (info.get(c) or {}).get("unit")} for c in cols],
                 "series": {e: {"times": [r[time_col] for r in series[e]],
                                **{c: [r[c] for r in series[e]] for c in cols}} for e in entities}})
        if (i + 1) % 250 == 0:
            print(f"  owid: {out.n} of {i + 1} charts", flush=True)
        time.sleep(1)
    out.close()
    print(f"chartdata owid: {out.n} charts", flush=True)


# -- CoSyn-400K: the code that drew 408k charts, diagrams, tables, documents, ... (for more picture styles) ----------

def build_cosyn(seed=0):
    """CoSyn-400K's rendering code and data, without its images: cosyn/<kind>-*.jsonl, kinds chart, chemical,
    circuit, diagram, document, graphic, math, music, nutrition, table. ODC-By (Ai2: research and education)."""
    rng = random.Random(seed)
    for kind in ["chart", "diagram", "table", "document", "math", "graphic", "chemical", "circuit", "nutrition", "music"]:
        out = Shards(POOLS / "cosyn", kind)
        for row in sampled_rows(parquet_files("allenai/CoSyn-400K", f"{kind}/*.parquet"), ["id", "metadata", "data", "code"],
                                rng, take=1.0):
            out.add({"source": "cosyn-400k", "license": "ODC-By 1.0 (CoSyn-400K; Ai2 research and education use)",
                     "kind": kind, "id": row["id"], "metadata": row["metadata"], "data": row["data"], "code": row["code"]})
        out.close()
        print(f"cosyn {kind}: {out.n}", flush=True)


# -- layout text: what real pages put in headers, footers, footnotes, captions and headings (DocLayNet) ------------

DOCLAYNET = {1: "caption", 2: "footnote", 4: "list-item", 5: "page-footer", 6: "page-header", 8: "section-header",
             11: "title"}


def build_layout(seed=0):
    """The text of every caption, footnote, list item, page header and footer, section header and title on the
    80,863 DocLayNet pages (reports, manuals, laws, patents, tenders, papers): layout/<kind>-*.jsonl. The words of
    an element are the PDF's text cells whose middle is inside it, top to bottom. A text seen before (numbers aside,
    so "Page 3" and "Page 4" are one) is left out. CDLA-Permissive-1.0."""
    rng = random.Random(seed)
    outs = {k: Shards(POOLS / "layout", k) for k in DOCLAYNET.values()}
    seen = Seen(26)
    for row in sampled_rows(parquet_files("docling-project/DocLayNet-v1.2", "data/*.parquet"),
                            ["bboxes", "category_id", "pdf_cells", "metadata"], rng, take=1.0):
        cells = [c for page in row["pdf_cells"] or [] for c in (page if isinstance(page, list) else [page])]
        meta = row["metadata"] or {}
        for (x, y, w, h), cat in zip(row["bboxes"] or [], row["category_id"] or []):
            kind = DOCLAYNET.get(cat)
            if not kind:
                continue
            inside = [c for c in cells if x <= c["bbox"][0] + c["bbox"][2] / 2 <= x + w and y <= c["bbox"][1] + c["bbox"][3] / 2 <= y + h]
            inside.sort(key=lambda c: (round(c["bbox"][1] / 4), c["bbox"][0]))
            text = re.sub(r"\s+", " ", " ".join(c["text"] for c in inside)).strip()
            letters = sum(ch.isalpha() for ch in text)
            if letters and len(LATIN.findall(text)) < 0.85 * letters:  # a Chinese or Japanese page: not ours
                continue
            if 2 <= len(text) <= 600 and seen.new(key(text)):
                outs[kind].add({"source": "doclaynet", "license": "CDLA-Permissive-1.0 (DocLayNet)", "kind": kind,
                                "doc_category": meta.get("doc_category"), "text": text})
    for k, out in outs.items():
        out.close()
        print(f"layout {k}: {out.n}", flush=True)


# -- parallel: official Arabic and English sentences (UN Parallel Corpus) -------------------------------------------

def build_parallel(n=1_000_000, seed=0):
    """Aligned Arabic-English sentences from United Nations documents (resolutions, reports, letters):
    parallel/un-*.jsonl, both sides 6-80 words, each Arabic sentence once. UN terms: credit the UN as the source."""
    rng = random.Random(seed)
    seen, out = Seen(28), Shards(POOLS / "parallel", "un")
    for row in sampled_rows(parquet_files("Helsinki-NLP/un_pc", "ar-en/*.parquet"), ["translation"], rng, take=0.12):
        t = row["translation"] or {}
        ar, en = re.sub(r"\s+", " ", t.get("ar") or "").strip(), re.sub(r"\s+", " ", t.get("en") or "").strip()
        if not (6 <= len(ar.split()) <= 80 and 6 <= len(en.split()) <= 80) or len(ARABIC.findall(ar)) < 0.7 * len(ar.replace(" ", "")):
            continue
        if seen.new(key(ar)):
            out.add({"source": "un-pc", "license": "UN Parallel Corpus (credit the United Nations)", "ar": ar, "en": en})
            if out.n % 100_000 == 0:
                print(f"  parallel: {out.n}", flush=True)
            if out.n >= n:
                break
    out.close()
    print(f"parallel: {out.n}", flush=True)


# -- captions of arXiv figures (ArxivCap) ------------------------------------------------------------------------

def build_arxivcap(n=400_000, seed=0, max_files=700, name="arxivcap"):
    """Real figure captions of arXiv papers, read without the figures: captions/en/arxivcap-*.jsonl. A caption
    with math ($...$) is left out; citation and reference marks are taken out or numbered. CC BY 4.0 tag."""
    rng = random.Random(seed)
    seen = Seen(26)
    remember(seen, POOLS / "captions" / "en", lambda r: [key(r["text"], digits=True)])
    out = Shards(POOLS / "captions" / "en", name)  # a second run: another name, other files (its seed)
    files = parquet_files("MMInstruction/ArxivCap", "data/*.parquet", max_files, rng)
    for row in sampled_rows(files, ["caption_images.list.item.caption", "arxiv_id"], rng, take=1.0):
        for c in row.get("caption_images") or []:
            text = re.sub(r"\s*<cit\.>", "", c.get("caption") or "")
            text = re.sub(r"<ref>", lambda m: str(rng.randint(1, 12)), text)
            text = re.sub(r"\s+", " ", text).strip()
            if 20 <= len(text) <= 300 and "$" not in text and "<" not in text and seen.new(key(text, digits=True)):
                out.add({"lang": "en", "source": "arxivcap", "license": "CC BY 4.0 tag (ArxivCap; per-paper arXiv licence)",
                         "kind": "Figure", "arxiv_id": row["arxiv_id"], "text": text})
        if out.n >= n:
            break
    out.close()
    print(f"captions arxivcap: {out.n}", flush=True)


# ==================================================================================================================
# Using the pools: the generator reads them through what follows
# ==================================================================================================================
# A random item is one seek away: index_pools() writes <file>.idx next to every pool file, the byte offset of each
# of its lines (8 bytes each). Pool(name).pick(rng) draws an item uniformly over all of a folder's files, so the
# same seed gives the same item whatever was read before.

def index_pools(folder=POOLS):
    """<file>.idx for every pool file that has none or changed since."""
    from array import array
    for path in sorted(Path(folder).rglob("*.jsonl")):
        idx = path.with_suffix(".idx")
        if {"_logs", "_download"} & set(path.parts) or (idx.exists() and idx.stat().st_mtime >= path.stat().st_mtime):
            continue
        offsets, pos = array("Q"), 0
        with open(path, "rb") as f:
            for line in f:
                if line.strip():
                    offsets.append(pos)
                pos += len(line)
        idx.write_bytes(offsets.tobytes())
        print(f"  {path.relative_to(folder)}: {len(offsets)}", flush=True)


class Pool:
    """The items of one pool folder (text/ar, equations, vocab with prefix "forms", ...); empty when missing."""

    def __init__(self, name, prefix=""):
        from itertools import accumulate
        self.files = sorted(p for p in (POOLS / name).glob(f"{prefix}*.jsonl") if p.with_suffix(".idx").exists())
        self.ends = list(accumulate(p.with_suffix(".idx").stat().st_size // 8 for p in self.files))

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def pick(self, rng):
        from bisect import bisect_right
        i = rng.randrange(len(self))
        f = bisect_right(self.ends, i)
        j = i - (self.ends[f - 1] if f else 0)
        with open(self.files[f].with_suffix(".idx"), "rb") as x:
            x.seek(8 * j)
            offset = int.from_bytes(x.read(8), "little")
        with open(self.files[f], "rb") as x:
            x.seek(offset)
            return json.loads(x.readline())


_pools = {}


def pool(name, prefix=""):
    if (name, prefix) not in _pools:
        _pools[(name, prefix)] = Pool(name, prefix)
    return _pools[(name, prefix)]


# -- text: TextPool's document mode over the pool's documents -------------------------------------------------------

def split_markdown(text, lang):
    """(headings, items, paragraphs) of a pool document: '#' lines are headings, '-', '*' and '1.' lines list
    items, long lines paragraphs, and short lines without a full stop headings too (web text has no '#').
    Tables, reference sections and lines in another script are left out."""
    from pipeline.pools import STOP, clean, one_script
    headings, items, paragraphs = [], [], []
    for raw in text.split("\n"):
        line = clean(raw)
        if not line or line.startswith("|"):
            continue
        head, item = re.match(r"#{1,6}\s+(.*)", line), re.match(r"(?:[-*+•]|\d{1,2}[.)])\s+(.*)", line)
        body = (head or item).group(1) if (head or item) else line
        body = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", body)  # a Markdown link: its words
        body = re.sub(r"\*\*|__|`", "", body).strip()
        if body.rstrip(" :") in STOP[lang]:
            break
        if not body or not one_script(body, lang) or any(c in body for c in "{}[]|=<>"):
            continue
        words = len(body.split())
        if head and words <= 12:
            headings.append(body)
        elif item and 2 <= words <= 25:
            items.append(body)
        elif words >= 15:
            paragraphs.append(body)
        elif not item and words <= 8 and not body.endswith((".", "!", "?", "؟", ":", "،", ",")):
            headings.append(body)
    return headings, items, paragraphs


# a web page's navigation and widgets, not its text: never a heading of ours
BOILERPLATE = re.compile(r"^\W*(?:tags?|share|related( posts| articles)?|comments?|leave a (reply|comment)|read more|"
                         r"you may also like|categories|archives|subscribe|follow us|الوسوم|وسوم|شارك|مشاركة|تعليقات?|"
                         r"اقرأ أيضا|اقرأ ايضا|مواضيع ذات صلة|مقالات ذات صلة|تابعنا|اشترك|التصنيفات?)\b", re.I)


def readable(text):
    """Text a page would print as words: half its characters letters ("8 34 8 34." is no list item)."""
    chars = text.replace(" ", "")
    return bool(chars) and sum(c.isalpha() for c in chars) >= 0.5 * len(chars)


def has_text(lang):
    return len(pool(f"text/{lang}")) > 0


def big_text_pool(lang):
    """A TextPool whose page follows one random pool document (Wikipedia, web or PDF), split into its title,
    headings, list items and paragraphs, and switches to another random one when it runs out."""
    from pipeline.pools import SENTENCE_END, TextPool

    from pipeline.pools import tidy

    class BigTextPool(TextPool):
        big = True  # fill() also takes real captions, page headers and footnotes from the pools (pool_text)

        def __init__(self):
            self.lang, self.license, self.reader = lang, "per document (see its source)", pool(f"text/{lang}")
            self.articles, self.titles, self.headings, self.items, self.paragraphs = {}, [], [], [], []
            self.used, self._by_length, self._by_article, self._doc = set(), {}, {}, False

        def _index(self):
            return self._by_article

        def begin(self, rng):
            self.articles, self._by_article = {}, {}
            self.used.clear()
            self._doc, self._title_used = True, False
            self._switch(rng)

        def _switch(self, rng):
            for _ in range(20):
                d = self.reader.pick(rng)
                headings, items, paragraphs = ([t for t in map(tidy, part) if readable(t)]
                                               for part in split_markdown(d["text"], lang))
                headings = [h for h in headings if not BOILERPLATE.match(h)]
                if len(paragraphs) >= 3:
                    break
            title = d.get("title") or (headings[0] if headings else " ".join(paragraphs[0].split()[:6]))
            self.articles[d["id"]] = {"title": title, "url": d["url"], "source": d["source"], "license": d["license"]}
            self._by_article[d["id"]] = {"paragraphs": paragraphs, "headings": headings, "items": items}
            self._cur = d["id"]
            self._sentences = [s for p in paragraphs for s in SENTENCE_END[lang].split(p) if s.strip()]
            self._s, self._headings, self._items = 0, list(headings), list(items)
            self.used.add(d["id"])

    return BigTextPool()


# -- lines, tables, references, forms and code for the page ---------------------------------------------------------

LINE_SHARE = 0.5  # share of captions (both languages), page headers and footnotes (English: DocLayNet) from the pools


def pool_text(role, n, rng, lang):
    """A real line for a slot of this role, or None (then the page's document gives it): a figure caption from
    documents and arXiv, an English page header or footnote from DocLayNet's pages. At most about n words (a
    caption cut at the end of its first sentence, or at n words); a footnote without its number (the page gives
    every note its own)."""
    kind = {"caption": (f"captions/{lang}", ""), "header": ("layout", "page-header"),
            "footnote": ("layout", "footnote")}.get(role)
    if not kind or (role != "caption" and lang != "en") or rng.random() >= LINE_SHARE or not len(pool(*kind)):
        return None
    from pipeline.pools import tidy
    for _ in range(6):
        text = tidy(" ".join(str(pool(*kind).pick(rng).get("text", "")).split()))
        if role == "footnote":
            text = re.sub(r"^(?:\(\s*[a-z]\s*\)|\(?\d{1,3}\)?|[*†‡§])[\s.)]*", "", text).strip()
        words = text.split()
        if len(words) > max(n, 6):
            first = re.split(r"(?<=[.!?؟])\s", text, 1)[0].split()
            words = first if len(first) <= max(n, 6) else words[:max(n, 6)]
        text = " ".join(words).rstrip(",;:،")
        if len(words) >= 2:
            return text
    return None


def _cells(item):
    """A pool table's rows as lists of cell texts (HTML tables: each cell in order, spans ignored)."""
    rows = item.get("rows")
    if isinstance(rows, list):
        return [[" ".join(str(c).split()) for c in r] for r in rows]
    found = []
    for tr in re.findall(r"<tr>(.*?)</tr>", item.get("html", ""), re.S):
        found.append([" ".join(html.unescape(re.sub(r"<[^>]+>", " ", c)).split())
                      for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)])
    return found


def table_rows(rng, lang, n_rows, n_cols):
    """Rows of a real table (dataset_pool/tables) to fill a template's table of n_rows x n_cols cells: the first
    rows, cut to n_cols columns, every cell filled; None when the pool has none that big after a few draws."""
    reader = pool(f"tables/{lang}")
    if not len(reader):
        return None
    for _ in range(8):  # (an empty cell keeps the template's own text: table_contexts skips it)
        rows = [r[:n_cols] for r in _cells(reader.pick(rng)) if len(r) >= n_cols][:n_rows]
        cells = [c for r in rows for c in r]
        if len(rows) >= min(n_rows, 2) and all(len(c) <= 60 for c in cells) and sum(map(bool, cells)) >= 0.7 * len(cells):
            return rows
    return None


def reference(rng, lang):
    """A real work (dataset_pool/references: OpenAlex, Crossref) as {authors, title, venue, year}, or None."""
    reader = pool(f"references/{lang}")
    for _ in range(6 if len(reader) else 0):
        work = work_entry(reader.pick(rng), lang)
        if work:
            return work
    return None


def work_entry(w, lang):
    """A pool work as {authors, title, venue, year} when it makes a reference in this language, else None."""
    from pipeline.pools import tidy
    authors = [" ".join(a.split()) for a in w.get("authors") or [] if a and len(a) <= 40]
    title = html.unescape(html.unescape(str(w.get("title") or "")))  # Crossref's titles carry escaped markup
    title = tidy(" ".join(re.sub(r"</?[a-zA-Z][^>]*>", " ", title).split())).rstrip(".")  # (<i>, <title>): words
    # the title in the page's language: Arabic letters, or English (plain Latin letters and fastText not sure of
    # another language: Portuguese, German and Spanish titles slipped through the pool's language check)
    lang_ok = is_arabic(title) if lang == "ar" else (all(c.isascii() for c in title if c.isalpha())
                                                     and not foreign_sentence(title, "en", sure=0.4))
    if authors and 3 <= len(title.split()) <= 25 and w.get("year") and lang_ok:
        return {"authors": authors, "title": title, "venue": w.get("venue") or w.get("publisher"), "year": w["year"]}
    return None


def form_fields(rng, lang):
    """The fields of a real form (the vocab pool: sections of fields, Arabic and English) as [(label, type,
    options)], or None."""
    reader = pool("vocab", "forms")
    if not len(reader):
        return None
    f = reader.pick(rng)
    fields = [fd for sec in f.get("sections", []) for fd in sec.get("fields", [])]
    out = []
    for fd in fields:
        label = (fd.get("label") or {}).get(lang)
        options = [o.get(lang) if isinstance(o, dict) else o for o in fd.get("options") or []]
        if label and len(label) <= 40:
            out.append((label, fd.get("type", "text"), [o for o in options if o and len(o) <= 24][:4]))
    return out or None


def code_block(rng):
    """A real function (dataset_pool/code: CodeSearchNet, six languages) that fits a block without wrapping:
    4-20 lines of at most 72 characters, tabs as 4 spaces, ASCII only (comments in other scripts are dropped
    with the item), Python ones valid (ast.parse); None after a few draws."""
    reader = pool("code")
    for _ in range(12 if len(reader) else 0):
        code = code_of(reader.pick(rng))
        if code:
            return code
    return None


def code_of(item):
    """A pool function as its block's text when it fits one (code_block), else None."""
    import ast
    lines = [line.rstrip() for line in item["code"].expandtabs(4).split("\n")]
    while lines and not lines[-1].strip():
        lines.pop()
    code = "\n".join(lines)
    if not lines or not 4 <= len(lines) <= 20 or max(map(len, lines)) > 72 or not code.isascii():
        return None
    if item.get("language") == "python":
        try:
            ast.parse(code)
        except SyntaxError:
            return None
    return code


# -- pictures drawn for the page: charts, flowcharts, molecules, equations --------------------------------------

FRESH = IMG_POOL / "graphics" / "fresh"  # one folder per process; only its last 40 pictures are kept
_drawn = Counter()
_mathjax = None


def draw_graphic(rng, label, lang):
    """(path, record) of a new picture of this label from the pools, or None when its pool is missing: a chart of
    real data (SBS Figures, UniChart or World Bank), a flowchart of a real process (the vocab pool), a PubChem
    molecule, an equation drawn by MathJax. Charts and flowcharts are English on every page (the user's choice:
    English is enough for them, its readers are many more)."""
    from pipeline import pools as P
    if label in ("Figure", "Diagram"):
        lang = "en"
    if label == "Figure":
        name = ("chartdata", rng.choice(["sbsfigures", "unichart", "worldbank"]))
    else:
        name = {"Diagram": ("vocab", "flowcharts"), "Chemical-Block": ("molecules", ""),
                "Equation-Block": ("equations", "")}.get(label)
    if not name or not len(pool(*name)):
        return None
    folder = FRESH / str(os.getpid())
    folder.mkdir(parents=True, exist_ok=True)
    for _ in range(24 if label == "Figure" else 8):  # (a chart whose data doesn't fit is skipped, nothing drawn)
        _drawn[label] += 1
        n = _drawn[label]
        path = folder / f"{label}-{n:07d}.{'svg' if label == 'Equation-Block' else 'png'}"
        (folder / f"{label}-{n - 40:07d}{path.suffix}").unlink(missing_ok=True)  # a new name each time: never cached
        item = pool(*name).pick(rng)
        try:
            if label == "Figure":
                if not P.chart_data(item, lang, random.Random(0)):
                    _drawn[label] -= 1
                    continue
                r = P.chart(rng, lang, path, real=item)
            elif label == "Diagram":
                r = P.diagram(rng, lang, path, process=item)
            elif label == "Chemical-Block":
                r = P.molecule(item["smiles"], path)
            else:  # a few formulas came wrapped in quotes
                r = equation_svg(item["latex"].strip().strip('"').strip(), path)
        except Exception as e:  # noqa: BLE001  (a picture that can't be drawn: take another)
            print(f"  {label} not drawn: {type(e).__name__}: {str(e)[:100]}", flush=True)
            r = None
        if r:
            return path, r
    return None


def equation_svg(latex, path):
    """The formula drawn by MathJax (pipeline/mathjax_svg.js, kept running) into an SVG file, or None when MathJax
    can't draw it or Node.js isn't there. Its height on the page is in em, from MathJax's ex (1 ex is about
    0.45 em), so it follows the page's own text size."""
    global _mathjax
    from pipeline.pools import text_escape
    if _mathjax is None:
        try:
            _mathjax = subprocess.Popen(["node", str(ROOT / "pipeline" / "mathjax_svg.js")], cwd=ROOT, text=True,
                                        encoding="utf-8", stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        except OSError:
            _mathjax = False
    if not _mathjax:
        return None
    _mathjax.stdin.write(json.dumps({"latex": latex}) + "\n")
    _mathjax.stdin.flush()
    r = json.loads(_mathjax.stdout.readline() or "{}")
    if r.get("error", True) or not r.get("height_ex") or r["width_ex"] > 55 or r["height_ex"] > 12:
        return None
    path.write_text(r["svg"], encoding="utf-8")
    return {"label": "Equation-Block", "html": f'<math display="block">{text_escape(latex)}</math>',
            "height_em": round(r["height_ex"] * 0.45, 3)}


# -- vocabularies written by an LLM: flowchart steps and form fields, Arabic and English ------------------------

SECTORS = ["government services", "banking", "insurance", "hospitals and clinics", "pharmacy", "schools",
           "universities", "human resources", "manufacturing", "logistics and shipping", "customs and immigration",
           "retail and e-commerce", "restaurants and hotels", "tourism and travel", "real estate", "construction",
           "electricity and water utilities", "telecommunications", "software development", "cybersecurity",
           "agriculture", "courts and legal services", "municipal services", "traffic and transport", "aviation",
           "research laboratories", "environment", "sports clubs", "charities", "media and publishing",
           "car service centres", "accounting and tax", "procurement", "quality control", "Hajj and Umrah services",
           "social welfare"]
FLOW_PROMPT = """For the sector "{sector}", write 5 different real processes that a flowchart in a document would show
(round {round}: {hint}).
For each: a title, 22 short action steps (2-6 words, imperative, in a plausible order) and 8 yes/no questions (2-6 words,
ending with "?"), every one in English and in Modern Standard Arabic. Answer with JSON only:
[{{"title": {{"en": "...", "ar": "..."}}, "steps": [{{"en": "...", "ar": "..."}}], "questions": [{{"en": "...", "ar": "..."}}]}}]"""
FORM_PROMPT = """For the sector "{sector}", write 3 different real paper forms (round {round}: {hint}).
For each: a title and 3-5 sections, each with 4-8 fields. A field has a label (1-5 words), a type (text, date, number,
phone, email, id, choice, multi, yes-no, signature) and, for choice and multi, 2-5 options. Everything in English and in
Modern Standard Arabic. Answer with JSON only:
[{{"title": {{"en": "...", "ar": "..."}}, "sections": [{{"title": {{"en": "...", "ar": "..."}}, "fields": [{{"label": {{"en": "...", "ar": "..."}}, "type": "...", "options": [{{"en": "...", "ar": "..."}}]}}]}}]}}]"""


def ask_json(llm, prompt, seed):
    for k in range(3):
        try:
            text = llm.ask(prompt, seed + k) or ""
            return json.loads(text[text.find("["):text.rfind("]") + 1])
        except Exception as e:  # noqa: BLE001  (bad JSON, network)
            print(f"  retry {k + 1}: {str(e)[:120]}", flush=True)
            time.sleep(5)
    return []


VOCAB_HINTS = ["the most common ones", "less common ones that specialists handle", "internal back-office ones",
               "ones a small business or a citizen meets", "rare or seasonal ones",
               "ones about complaints and appeals", "ones about registration and renewal", "ones about inspections and audits",
               "ones about payments, refunds and fees", "ones about safety and emergencies", "ones about hiring and training",
               "ones about requests for documents or certificates", "ones about maintenance and repairs",
               "ones about approvals that pass several managers", "ones about data, records and archives",
               "ones about suppliers and contracts", "ones about customers' feedback and surveys",
               "ones about cancelling, transferring or closing something", "ones done online through a portal",
               "ones for people with special needs or the elderly"]


def build_vocab(first=0):
    """Flowchart processes (steps, questions) and form layouts (sections, fields), per sector, from the LLM of
    llm_templates (Groq when GROQ_API_KEY is in .env). Only items with both an English and an Arabic text are kept.
    `first` skips the hints already asked (the first run asked 5); a later run adds new files and no repeated titles."""
    from pipeline.llm_templates import LLMTemplates
    llm = LLMTemplates.from_env()
    pair = lambda x: isinstance(x, dict) and x.get("en") and ARABIC.search(x.get("ar") or "")  # noqa: E731
    tag = f"-h{first}" if first else ""
    flows, forms = Shards(POOLS / "vocab", "flowcharts" + tag), Shards(POOLS / "vocab", "forms" + tag)
    titles = {key(json.loads(line)["title"]["en"]) for p in (POOLS / "vocab").glob("*.jsonl") for line in open(p, encoding="utf-8")}
    for i, (sector, (r, hint)) in enumerate((s, h) for s in SECTORS for h in enumerate(VOCAB_HINTS, 1)):
        if r <= first:
            continue
        for p in ask_json(llm, FLOW_PROMPT.format(sector=sector, round=r, hint=hint), 100 * i):
            steps, questions = [s for s in p.get("steps", []) if pair(s)], [q for q in p.get("questions", []) if pair(q)]
            if pair(p.get("title")) and len(steps) >= 10 and len(questions) >= 3 and key(p["title"]["en"]) not in titles:
                titles.add(key(p["title"]["en"]))
                flows.add({"sector": sector, "title": p["title"], "steps": steps, "questions": questions,
                           "source": f"LLM ({llm.name})", "license": "generated"})
        for part in (1, 2):
            for f in ask_json(llm, FORM_PROMPT.format(sector=sector, round=r, hint=hint), 100 * i + 10 * part):
                sections = [{"title": s["title"], "fields": [x for x in s.get("fields", []) if pair(x.get("label"))]}
                            for s in f.get("sections", []) if pair(s.get("title"))]
                if pair(f.get("title")) and sum(len(s["fields"]) for s in sections) >= 8 and key(f["title"]["en"]) not in titles:
                    titles.add(key(f["title"]["en"]))
                    forms.add({"sector": sector, "title": f["title"], "sections": sections,
                               "source": f"LLM ({llm.name})", "license": "generated"})
        print(f"  vocab {sector}: {flows.n} flowcharts, {forms.n} forms", flush=True)
    flows.close()
    forms.close()
    print(f"vocab: {flows.n} flowchart processes, {forms.n} forms", flush=True)


# -- language check: only English and Arabic stay ----------------------------------------------------------------------

LID_MODEL = POOLS / "_lid" / "lid.176.ftz"  # fastText's language identifier (176 languages), from the fast-langdetect wheel
WANT = {"en": {"en"}, "ar": {"ar", "arz"}, "any": {"en", "ar", "arz"}, None: None}  # None: the script check alone
LANG_POOLS = [("text/ar", "ar"), ("text/en", "en"), ("parallel", "any"), ("tables/ar", "any"), ("tables/en", None),
              ("captions/ar", "ar"), ("captions/en", "en"), ("layout", "en"), ("references/ar", "ar"),
              ("references/en", "en"), ("code", "en"), ("equations", None), ("cosyn", "en"), ("chartdata", "any"),
              ("diagrams", "en"), ("vocab", "any"), ("../../img_pool/photos.jsonl", "en"),
              ("../../img_pool/flowcharts/flowcharts.jsonl", "en")]
SKIP_KEYS = {"source", "license", "url", "uri", "id", "doi", "repo", "path", "colors", "color", "style", "smiles",
             "language", "kind", "doc_category", "lang", "spans", "authors", "venue", "publisher", "citation", "file",
             "label"}
CODE_TEXT = re.compile(r'"([^"\n]{2,})"|\'([^\'\n]{2,})\'|(?:#|//|%)\s?([^\n]{3,})|/\*(.*?)\*/|\[([^\[\]\n]{2,})\]'
                       r'|\{([^{}\n\\]{2,})\}|:\s*([^\n:;{}]{3,})$', re.S | re.M)
PERSIAN_URDU = set("پچژگٹڈڑںےۀ")
LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S")
_lid = []


def foreign_letters(text, arabic=False):
    """How many letters are neither Latin (with IPA, modifier letters, the fi/fl ligatures), Greek, letter-like or
    math symbols (ℓ, 𝑥) nor Arabic; with `arabic`, Persian and Urdu letters count too."""
    return sum(1 for c in text if c.isalpha() and (
        not (ord(c) < 0x300 or 0xFB00 <= ord(c) < 0xFB07 or 0x370 <= ord(c) < 0x400 or 0x1E00 <= ord(c) < 0x1F00 or 0x2100 <= ord(c) < 0x2150
             or 0x600 <= ord(c) < 0x700 or 0x750 <= ord(c) < 0x780 or 0x8A0 <= ord(c) < 0x900
             or 0xFB50 <= ord(c) < 0xFE00 or 0xFE70 <= ord(c) < 0xFF00 or 0x1D400 <= ord(c) < 0x1D800)
        or (arabic and c in PERSIAN_URDU)))


def is_arabic(text):
    letters = [c for c in text if c.isalpha()]
    return sum(1 for c in letters if 0x600 <= ord(c) < 0x900 or 0xFB50 <= ord(c) < 0xFF00) >= len(letters) / 2


def item_text(item):
    """(the pieces of text a reader sees, every string) of a pool item, ids, links and licences left out; `code`
    gives its strings, comments and bracketed labels."""
    seen, every = [], []

    def walk(x, k=None):
        if isinstance(x, dict):
            for kk, v in x.items():
                if kk not in SKIP_KEYS:
                    walk(v, kk)
        elif isinstance(x, list):
            for v in x:
                walk(v, k)
        elif isinstance(x, str):
            every.append(x)
            if k == "mermaid":  # a drawn flowchart: the words of its boxes and arrows, as one text
                seen.append(" ".join(m for g in re.findall(r"\[([^\]]+)\]|\{([^}]+)\}|-- (\w+) -->", x) for m in g if m))
            else:
                seen.extend([m for g in CODE_TEXT.findall(x) for m in g if m] if k == "code"
                            else re.sub(r"<[^>]+>", "\n", x).split("\n"))
    if "mermaid" in item:  # its html repeats the Mermaid
        item = {k: v for k, v in item.items() if k != "html"}
    walk(item)
    return [p for p in seen if p.strip()], " ".join(every)


def foreign_sentence(text, want, sure=0.6):
    """True when fastText is sure (top guess `sure`+, wanted languages under 5%) a text is in another language.
    Text lowercased, as capitals confuse it."""
    if not _lid:
        import fasttext
        _lid.append(fasttext.load_model(str(LID_MODEL)))
    labels, probs = _lid[0].predict(re.sub(r"\s+", " ", text.lower())[:1000], k=3)
    wanted = sum(p for lb, p in zip(labels, probs) if lb[9:] in WANT[want])
    return labels[0][9:] not in WANT[want] and probs[0] >= sure and wanted < 0.05


def sentence(p):
    return len(p.split()) >= 4 and sum(c.isalpha() for c in p) >= 20


def lang_ok(item, want):
    """True when every letter is Latin, Greek or Arabic and no sentence (4+ words) is surely another language;
    an item without sentences is judged as a whole when it has 12+ letters, and dropped only when fastText is very
    sure (90%) it is another language (short labels and headers are often misread)."""
    pieces, every = item_text(item)
    if foreign_letters(every, arabic=want == "ar"):
        return False
    if want is None:
        return True
    sentences = [p for p in pieces if sentence(p)]
    if sentences:
        return not any(foreign_sentence(s, want) for s in sentences)
    short = " ".join(pieces)
    return sum(c.isalpha() for c in short) < 12 or not foreign_sentence(short, want, sure=0.9)


def clean_doc(item, want):
    """A text document with every line in another language or script taken out (a Markdown table whole if any
    row goes); None when its title goes or over 40% of its letters went."""
    title = item.get("title") or ""
    if foreign_letters(title, want == "ar") or (sentence(title) and foreign_sentence(title, want)):
        return None
    lines = item["text"].split("\n")
    bad = [bool(foreign_letters(ln, want == "ar")) or (sentence(ln) and foreign_sentence(ln, want)) for ln in lines]
    i = 0
    while i < len(lines):  # a table goes whole
        if lines[i].lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            if any(bad[i:j]):
                bad[i:j] = [True] * (j - i)
            i = j
        else:
            i += 1
    if not any(bad):
        return item
    total = sum(c.isalpha() for c in item["text"])
    kept = [ln for ln, b in zip(lines, bad) if not b]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    if sum(c.isalpha() for c in text) < 0.6 * total:
        return None
    return {**item, "text": text}


def _check_file(job):
    """One pool file, rewritten in place: (path, kept, dropped, lines cleaned, language counts, examples)."""
    path, want, docs = job
    counts = {"en": 0, "ar": 0, "headings_en": 0, "headings_ar": 0, "lists_en": 0, "lists_ar": 0}
    kept = dropped = cleaned = 0
    examples = []
    tmp = path.with_suffix(".tmp")
    with open(path, encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as dst:
        for line in src:
            item = json.loads(line)
            if docs:
                new = clean_doc(item, want)
                cleaned += new is not None and new is not item
            else:
                new = item if lang_ok(item, want) else None
            if new is None:
                dropped += 1
                if len(examples) < 3:
                    examples.append(" | ".join(item_text(item)[0])[:120])
                continue
            dst.write(line if new is item else json.dumps(new, ensure_ascii=False) + "\n")
            kept += 1
            lang = "ar" if is_arabic(item_text(new)[1][:3000]) else "en"
            counts[lang] += 1
            if docs:
                for ln in new["text"].split("\n"):
                    if ln.startswith("#"):
                        counts["headings_" + lang] += 1
                    elif LIST_ITEM.match(ln):
                        counts["lists_" + lang] += 1
    if dropped or cleaned:
        os.replace(tmp, path)
        path.with_suffix(".idx").unlink(missing_ok=True)
    else:
        tmp.unlink()
    return path, kept, dropped, cleaned, counts, examples


def check_languages(only=None, workers=8):
    """Keep only English and Arabic in every pool (LANG_POOLS says what a folder or file wants), files rewritten in
    place (a changed file loses its .idx: run `index` after); text documents lose their foreign lines, other items
    go whole. Prints per folder what was kept and how much of it is English / Arabic; _logs/lang-stats.json."""
    jobs = []
    for folder, want in LANG_POOLS:
        if only and folder not in only:
            continue
        target = (POOLS / folder).resolve()
        for path in ([target] if target.suffix == ".jsonl" else sorted(target.glob("*.jsonl"))):
            jobs.append((path, want, folder.startswith("text/")))
    stats = {}
    with ProcessPoolExecutor(workers) as pool:
        for path, kept, dropped, cleaned, counts, examples in pool.map(_check_file, jobs):
            folder = next(f for f, _ in LANG_POOLS if (POOLS / f).resolve() in (path, path.parent))
            s = stats.setdefault(folder, {"kept": 0, "dropped": 0, "cleaned": 0, **{k: 0 for k in counts}})
            s["kept"], s["dropped"], s["cleaned"] = s["kept"] + kept, s["dropped"] + dropped, s["cleaned"] + cleaned
            for k, v in counts.items():
                s[k] += v
            for e in examples[:1]:
                print(f"    dropped from {path.name}: {e!r}", flush=True)
            print(f"  {path.name}: kept {kept:,}, dropped {dropped:,}, cleaned {cleaned:,}", flush=True)
    for folder, s in stats.items():
        print(f"{folder}: kept {s['kept']:,} (English {s['en']:,}, Arabic {s['ar']:,}), dropped {s['dropped']:,}, "
              f"cleaned {s['cleaned']:,}", flush=True)
    (POOLS / "_logs" / "lang-stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")


# -- more charts and diagrams ----------------------------------------------------------------------------------------

def build_charts_more(seed=0):
    """Chart data to draw again: chartdata/sbsfigures-*.jsonl (SBS Figures' chart specs: title, axis names, series
    with x, y and colours; synthetic, 4.2M rows, several questions per chart) and chartdata/unichart-*.jsonl
    (UniChart's tables read off real charts: Statista, Pew, OWID, OECD, PlotQA; a Markdown table each)."""
    rng = random.Random(seed)
    seen = Seen(27)
    out = Shards(POOLS / "chartdata", "sbsfigures")
    for row in sampled_rows(parquet_files("omron-sinicx/sbsfigures", "data/*.parquet"), ["json"], rng, take=0.4):
        try:
            spec = json.loads(row["json"])
        except (TypeError, ValueError):
            continue
        if isinstance(spec, dict) and spec.get("models") and seen.new(key(row["json"], digits=True)):
            out.add({"source": "sbsfigures", "license": "see omron-sinicx/sbsfigures (synthetic charts)", **spec})
        if out.n >= 600_000:
            break
    out.close()
    print(f"chartdata sbsfigures: {out.n}", flush=True)
    out = Shards(POOLS / "chartdata", "unichart")
    for row in sampled_rows(parquet_files("ahmed-masry/unichart-table-data", "data/*.parquet"), ["table"], rng, take=1.0):
        table = row["table"] or ""
        if table.startswith('"'):
            try:
                table = json.loads(table)
            except ValueError:
                continue
        lines = [ln for ln in table.splitlines() if ln.strip().startswith("|")]
        if 3 <= len(lines) <= 60 and seen.new(key(table, digits=True)):
            out.add({"source": "unichart", "license": "research (UniChart; tables of Statista, Pew, OWID, OECD, PlotQA charts)",
                     "table": "\n".join(lines)})
    out.close()
    print(f"chartdata unichart: {out.n}", flush=True)


def build_chartgen(seed=0):
    """chartdata/chartgen-*.jsonl: ChartGen-200K (CC BY 4.0), 221k charts as the Python code that drew them, their
    data as CSV and a summary of what they show; no images."""
    rng, seen = random.Random(seed), Seen(25)
    out = Shards(POOLS / "chartdata", "chartgen")
    for row in sampled_rows(parquet_files("SD122025/ChartGen-200K", "data/*.parquet"), ["id", "code", "summary", "csv"], rng, take=1.0):
        if row["code"] and seen.new(key((row["csv"] or "") + row["code"], digits=True)):
            out.add({"source": "chartgen-200k", "license": "CC BY 4.0", "id": row["id"], "summary": row["summary"],
                     "csv": row["csv"], "code": row["code"]})
    out.close()
    print(f"chartdata chartgen: {out.n}", flush=True)


def build_plotqa(seed=0):
    """chartdata/plotqa-*.jsonl: PlotQA's 224k plots (achang/plot_qa), each the series it draws: name, colour, and
    y by x position (World Bank and similar indicators). This copy has no titles or tick labels; no images."""
    rng, seen = random.Random(seed), Seen(25)
    out = Shards(POOLS / "chartdata", "plotqa")
    tag = lambda t, s: (re.search(f"<{t}>(.*?)</{t}>", s, re.S) or [None, ""])[1]  # noqa: E731
    for row in sampled_rows(parquet_files("achang/plot_qa", "data/*.parquet"), ["text"], rng, take=1.0):
        text = re.sub(r"<s_bboxes>.*?</s_bboxes>", "", row["text"] or "", flags=re.S)
        series = [{"name": tag("s_name", s), "color": tag("s_color", s), "x": tag("s_x", s).split("<sep/>"),
                   "y": tag("s_y", s).split("<sep/>")} for s in re.split(r"(?<=</s_color>)<sep/>", text) if "<s_y>" in s]
        if series and seen.new(key(text, digits=True)):
            out.add({"source": "plotqa", "license": "CC BY 4.0 (PlotQA)", "series": series})
    out.close()
    print(f"chartdata plotqa: {out.n}", flush=True)


KAGGLE = POOLS / "_downloads"  # Kaggle needs a login: `kaggle` (token in ~/.kaggle/access_token) fetched these


def build_kaggle_charts():
    """chartdata/vistext-*.jsonl from VITIS / VisText (8,822 Statista charts: data table, scene graph with title,
    axes and ticks, and captions) and chartdata/benetech-*.jsonl from Benetech's competition (60k charts: type, every
    text with its role, the plotted series); both downloaded to _downloads/ by the Kaggle CLI, no images kept."""
    import zipfile
    charts = {}
    for path in sorted((KAGGLE / "vitis" / "tabular").glob("data_*.json")):
        for row in json.load(open(path, encoding="utf-8")):
            c = charts.setdefault(row["img_id"], {"source": "vistext (Kaggle: vitis-chart-to-text)", "license": "see VisText (MIT code; Statista charts)",
                                                  "datatable": row["datatable"], "scenegraph": row["scenegraph"], "captions": []})
            for cap in (row["caption_L1"], row["caption_L2L3"]):
                if cap and cap not in c["captions"]:
                    c["captions"].append(cap)
    out = Shards(POOLS / "chartdata", "vistext")
    for c in charts.values():
        out.add(c)
    out.close()
    print(f"chartdata vistext: {out.n}", flush=True)
    out = Shards(POOLS / "chartdata", "benetech")
    with zipfile.ZipFile(KAGGLE / "benetech" / "benetech-making-graphs-accessible.zip") as z:
        for name in sorted(n for n in z.namelist() if n.startswith("train/annotations/") and n.endswith(".json")):
            a = json.loads(z.read(name))
            out.add({"source": f"benetech ({a.get('source')})", "license": "Kaggle competition data (Benetech: Making Graphs Accessible)",
                     "chart_type": a.get("chart-type"), "texts": [{"text": t["text"], "role": t.get("role")} for t in a.get("text", [])],
                     "series": a.get("data-series"), "x_type": (a.get("axes", {}).get("x-axis") or {}).get("values-type"),
                     "y_type": (a.get("axes", {}).get("y-axis") or {}).get("values-type")})
    out.close()
    print(f"chartdata benetech: {out.n}", flush=True)


def build_diagrams(seed=0):
    """Diagram source code to draw: diagrams/tikz-* (DaTikZ v3: TikZ from arXiv, TeX StackExchange, ...; code and
    caption), plantuml-* (PlantUML files from GitHub, The Stack v2), mermaid-* (13k Mermaid samples) and
    flowcharts-* (QZhou: flowchart nodes and edges as JSON, with their drawing style)."""
    rng = random.Random(seed)
    seen = Seen(26)
    out = Shards(POOLS / "diagrams", "tikz")
    for row in sampled_rows(parquet_files("nllg/datikz-v3", "data/train-*.parquet"), ["caption", "code", "uri", "origin"], rng, take=1.0):
        code = row["code"] or ""
        if "tikzpicture" in code and len(code) <= 12_000 and seen.new(key(code, digits=True)):
            out.add({"source": f"datikz-v3 ({row['origin']})", "license": "the origin's (arXiv paper / CC BY-SA for StackExchange)",
                     "uri": row["uri"], "caption": (row["caption"] or "")[:1500], "code": code})
    out.close()
    print(f"diagrams tikz: {out.n}", flush=True)
    out = Shards(POOLS / "diagrams", "plantuml")
    for row in sampled_rows(parquet_files("devgpt-aimotion/the-stack-v2_PlantUML_full", "data/*.parquet"),
                            ["code", "repo_name", "path", "detected_licenses"], rng, take=1.0):
        code = row["code"] or ""
        if "@start" in code and 60 <= len(code) <= 8_000 and seen.new(key(code, digits=True)):
            out.add({"source": "the-stack-v2 PlantUML", "license": ", ".join(row["detected_licenses"] or []) or "unknown",
                     "repo": row["repo_name"], "path": row["path"], "code": code})
    out.close()
    print(f"diagrams plantuml: {out.n}", flush=True)
    out = Shards(POOLS / "diagrams", "mermaid")
    for row in sampled_rows(parquet_files("injaeryou/mermaid_samples_13k", "data/unwrapped-*.parquet"), ["text"], rng, take=1.0):
        if seen.new(key(row["text"] or "", digits=True)):
            out.add({"source": "mermaid_samples_13k", "license": "MIT", "code": row["text"]})
    out.close()
    print(f"diagrams mermaid: {out.n}", flush=True)
    out = Shards(POOLS / "diagrams", "flowcharts")
    for row in sampled_rows(parquet_files("Kingsoft-LLM/QZhou-Flowchart-QA", "*/*.parquet"), ["data_json", "style_json"], rng, take=1.0):
        if seen.new(key(row["data_json"] or "", digits=True)):
            out.add({"source": "QZhou-Flowchart-QA", "license": "Apache-2.0", "data": row["data_json"], "style": row["style_json"]})
    out.close()
    print(f"diagrams flowcharts: {out.n}", flush=True)


# -- more fonts: every readable Google Fonts family, and open Arabic fonts from GitHub --------------------------------

EXTRA_ARABIC_FONTS = {  # family: (repo, regular, bold or None); all SIL OFL 1.1
    "Sahel": ("rastikerdar/sahel-font", "dist/Sahel.ttf", "dist/Sahel-Bold.ttf"),
    "Samim": ("rastikerdar/samim-font", "dist/Samim.ttf", "dist/Samim-Bold.ttf"),
    "Shabnam": ("rastikerdar/shabnam-font", "dist/Shabnam.ttf", "dist/Shabnam-Bold.ttf"),
    "Tanha": ("rastikerdar/tanha-font", "dist/Tanha.ttf", None),
    "Parastoo": ("rastikerdar/parastoo-font", "dist/Parastoo.ttf", "dist/Parastoo-Bold.ttf"),
    "Gandom": ("rastikerdar/gandom-font", "dist/Gandom.ttf", None),
    "Nahid": ("rastikerdar/nahid-font", "dist/Nahid.ttf", None),
    "Estedad": ("aminabedi68/Estedad", "fonts/ttf/Estedad-Regular.ttf", "fonts/ttf/Estedad-Bold.ttf"),
}
NOT_READABLE = re.compile(r"Stencil|Outline|Inline|Shadow|Sketch|Hairline|Glitch|Pixel|Dot|Barcode|Wavy|Zebra|Ink\b|Fun\b|"
                          r"Decor|3D|Mosaic|Nastaliq|Gulzar|Quran|Qahiri|Icons|Symbols|Emoji|Blackletter|Fraktur|Rough|Brush|Marker|Graffiti")


def build_fonts_more():
    """font_pool/fonts-more.json: every readable family (Serif, Sans Serif, Monospace; no decorative names; regular
    weight; all the letters, digits and punctuation of its scripts) in Google Fonts, Arabic-only ones included, plus
    EXTRA_ARABIC_FONTS. Files go to font_pool/<family>/<weight>.ttf next to the generator's 100; its fonts.json is
    left alone."""
    from pipeline import pools
    text = pools.download(pools.GOOGLE_FONTS).decode("utf-8")
    families = json.loads(text[text.index("{"):])["familyMetadataList"]
    readable = [f for f in families if f.get("isOpenSource", True) and not f.get("colorCapabilities")  # Noto is a "brand" font
                and "400" in f["fonts"] and f["category"] in ("Serif", "Sans Serif", "Monospace")
                and f.get("primaryScript", "") in ("", "Arab", "Latn")  # a CJK or Indic family's Latin is an add-on
                and ("arabic" in f["subsets"] or "latin" in f["subsets"]) and not NOT_READABLE.search(f["family"])]
    kept = []
    for record in sorted(readable, key=lambda f: f["popularity"]):
        scripts = [s for s in ("arabic", "latin") if s in record["subsets"]]
        try:
            files = pools.fetch_family(record)
        except Exception as e:  # noqa: BLE001
            print(f"  {record['family']}: skipped ({type(e).__name__})", flush=True)
            continue
        needs = (pools.ARABIC_NEEDS if "arabic" in scripts else []) + (pools.LATIN_NEEDS if "latin" in scripts else list("0123456789"))
        if "400" not in files or not all(pools.font_covers(p, needs) for p in files.values()):
            print(f"  {record['family']}: skipped (no regular file or missing characters)", flush=True)
            continue
        kept.append({"family": record["family"], "source": "Google Fonts", "license": "SIL OFL 1.1 / Apache-2.0 / UFL",
                     "category": record["category"], "scripts": scripts, "popularity": record["popularity"],
                     "files": {w: p.relative_to(pools.ROOT).as_posix() for w, p in files.items()}})
        print(f"  {len(kept):4d} {record['family']} ({record['category']}, {'+'.join(scripts)})", flush=True)
    for family, (repo, regular, bold) in EXTRA_ARABIC_FONTS.items():
        if family in {f["family"] for f in kept}:  # Google Fonts has it too now
            continue
        files = {}
        for weight, path in (("400", regular), ("700", bold)):
            if path:
                dest = pools.FONT_POOL / family / f"{weight}.ttf"
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(pools.download(f"https://raw.githubusercontent.com/{repo}/HEAD/{path}"))
                files[weight] = dest
        scripts = ["arabic", "latin"] if pools.font_covers(files["400"], pools.LATIN_NEEDS) else ["arabic"]
        if all(pools.font_covers(p, pools.ARABIC_NEEDS) for p in files.values()):
            kept.append({"family": family, "source": f"github.com/{repo}", "license": "SIL OFL 1.1", "category": "Sans Serif",
                         "scripts": scripts, "files": {w: p.relative_to(pools.ROOT).as_posix() for w, p in files.items()}})
            print(f"  {len(kept):4d} {family} (GitHub, {'+'.join(scripts)})", flush=True)
    chosen = pick_fonts(kept)
    (pools.FONT_POOL / "fonts-more.json").write_text(json.dumps(chosen, ensure_ascii=False, indent=1), encoding="utf-8")
    keep = {Path(p).parent.name for f in chosen + json.loads(pools.FONT_INDEX.read_text(encoding="utf-8")) for p in f["files"].values()}
    for f in kept:  # the families not chosen leave the disk (the generator's fonts.json ones stay)
        folder = (pools.ROOT / next(iter(f["files"].values()))).parent
        if folder.name not in keep:
            shutil.rmtree(folder, ignore_errors=True)
    arabic = sum("arabic" in f["scripts"] for f in chosen)
    print(f"fonts-more: {len(chosen)} of {len(kept)} readable families ({arabic} with Arabic)", flush=True)


OFFICE_LOOKALIKES = ["Tinos", "Arimo", "Carlito", "Caladea", "Cousine", "Gelasio", "EB Garamond"]  # Times New Roman,
# Arial, Calibri, Cambria, Courier New, Georgia, Garamond: the fonts real documents use most, as open metric twins


def pick_fonts(readable, n=100):
    """At most n families, in this order: the 10 most used Arabic and the 10 most used Latin text families (Google
    Fonts popularity), the open twins of the Office fonts, every other Arabic one, the readable ones the generator
    already has (font_pool/fonts.json), then the most used Latin ones. Each is marked with why it is in."""
    from pipeline import pools
    have = {f["family"] for f in json.loads(pools.FONT_INDEX.read_text(encoding="utf-8"))}
    by_pop = sorted(readable, key=lambda f: f.get("popularity", 10**6))
    arabic = [f for f in by_pop if "arabic" in f["scripts"]]
    latin = [f for f in by_pop if f["scripts"] == ["latin"]]
    order = ([(f, "top-arabic") for f in arabic[:10]]
             + [(f, "top-latin") for f in [f for f in latin if f["category"] != "Monospace"][:10]]
             + [(f, "office") for f in latin if f["family"] in OFFICE_LOOKALIKES]
             + [(f, "arabic") for f in arabic[10:]]
             + [(f, "have") for f in latin if f["family"] in have]
             + [(f, "popular") for f in latin])
    chosen, names = [], set()
    for f, why in order:
        if f["family"] not in names and len(chosen) < n:
            chosen.append({**f, "group": why, "in_generator": f["family"] in have})
            names.add(f["family"])
    return chosen


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool", choices=["photos", "text", "tables", "equations", "code", "molecules", "references", "chartdata",
                                    "vocab", "references-en", "index", "text-more", "cosyn", "layout", "parallel",
                                    "arxivcap", "owid", "vocab-more", "charts-more", "diagrams", "fonts-more", "lang", "chartgen", "plotqa", "kaggle-charts"])
    ap.add_argument("--lang", choices=["ar", "en"], help="text: one language (default both)")
    ap.add_argument("--n", type=int, help="photos: how many (default 300,000)")
    a = ap.parse_args()
    if a.pool == "photos":
        build_photos(a.n or 300_000)
    elif a.pool == "text":
        for lang in [a.lang] if a.lang else ["ar", "en"]:
            build_text(lang)
    elif a.pool == "vocab-more":
        build_vocab(first=5)
    elif a.pool == "text-more":
        build_text("ar", only=[name for name, *_ in TEXT_MORE["ar"]])
    else:
        {"tables": build_tables, "equations": build_equations, "code": build_code, "molecules": build_molecules,
         "references": build_references, "references-en": build_references_en, "chartdata": build_chartdata,
         "vocab": build_vocab, "index": index_pools, "cosyn": build_cosyn, "layout": build_layout,
         "parallel": build_parallel, "arxivcap": build_arxivcap, "owid": build_owid, "charts-more": build_charts_more,
         "diagrams": build_diagrams, "fonts-more": build_fonts_more, "lang": check_languages, "chartgen": build_chartgen,
         "plotqa": build_plotqa, "kaggle-charts": build_kaggle_charts}[a.pool]()
