# OCRSynthGen: Arabic and English training pages for Surya OCR 2.5

OCRSynthGen makes synthetic document pages in Arabic and English, each labelled exactly as
[Surya OCR 2.5](https://huggingface.co/datalab-to/surya-ocr-2.5) answers, to fine-tune Surya. Every sample is a page
image (clean, scanned, photocopied or photographed), Surya's own prompt for one of its tasks, and the answer Surya
should give: the page's blocks with their labels, boxes, reading order and HTML.

- 2,000 page templates (books, reports, websites, scanned documents), filled with real content: 2.6 million
  documents, 1.6 million tables, 3.5 million formulas, 300,000 photos, 100,000 flowcharts, chart data, molecules,
  references, forms and code, in Arabic and English.
- All 19 block labels Surya writes, the nine structured ones (equations, figures, code, tables, forms, diagrams,
  molecules, contents, panels) in equal numbers.
- Effects that make the page look scanned or photographed: OCRSmith, Augraphy, scan-simulator, each with a strength.
- Every sample is checked before it is kept: its format, Surya's own parser, every text drawn is labelled and
  nothing else, boxes against the ink on the image, its length in tokens. A sample that fails is made again.

---

## 1. Setup

Windows or Linux, Python 3.12.

```bash
py -3.12 -m venv .venv-pipeline
```
```bash
.venv-pipeline\Scripts\pip install -r requirements.txt
```
```bash
.venv-pipeline\Scripts\python -m playwright install chromium
```

(On Linux: `python3.12 -m venv .venv-pipeline` and `.venv-pipeline/bin/...`; add `--with-deps` to the Playwright
install.)

Optional:

- **Node.js** and `npm install` (installs `mathjax-full` from `package.json`): equations are then drawn by MathJax
  from the 3.5 million formulas; without it the generator uses ready-made equation pictures.
- **Surya's code** for one more check (Surya's own parser reads every answer, `--surya-check auto`):
  `git clone https://github.com/datalab-to/surya surya` in this folder.

That is all a first run needs: the repository holds a sample of every content pool (`dataset_pool_sample/`, below).

## 2. Generate a dataset

```bash
.venv-pipeline\Scripts\python generate.py --n 100 --epoch 2 --strength-ocrsmith 0.3 --strength-augraphy 0.3 --strength-scan-simulator 0.3 --workers 4 --out datasets\OCR-Test
```

| Option | Default | Meaning |
|---|---|---|
| `--n` | 8 | samples to make |
| `--epoch N` | none | make epoch N's samples, with seed N: the same pages every time for the same N, other pages for another N. Without it: a fresh batch with a new seed each run (printed, to make it again) |
| `--strength-ocrsmith` | 0.5 | OCRSmith's scan / photo / fax / archive look: 0 lightest, 0.5 the tool's own settings, 1 strongest |
| `--strength-augraphy` | 0.5 | Augraphy's ink and paper effects, same scale |
| `--strength-scan-simulator` | 0.5 | scan-simulator's scans, photocopies and photos, same scale |
| `--workers` | min(4, CPUs) | processes making pages in parallel; each needs about 1.5 GB of memory |
| `--out` | `datasets/train_try` | folder the samples are written to (added to what is there) |

Every other setting keeps its default: both languages, all effects (15% clean pages), all template sources, the
task mix (40% `full_ocr`, 25% `layout`, 35% `block_ocr`), Arabic diacritics on some Arabic pages. Rotation and
perspective each tilt a page at most 1 degree.

**All the settings** are flags of `train.py` (the same generator, every option):

```bash
.venv-pipeline\Scripts\python train.py --n 200 --epoch 3 --language ar --task full_ocr=0.6,block_ocr=0.4 --tashkeel none --out datasets\epoch3
```

| Option | Default | Meaning |
|---|---|---|
| `--n`, `--epoch`, `--workers`, `--out` | as above | |
| `--seed` | 0 | the run's seed (`--epoch N` makes epoch N of this seed's run) |
| `--no-save` | off | make and check the samples, write nothing |
| `--language` | `mix` | `ar`, `en`, or weights such as `ar=0.7,en=0.3` (mix: 50/50) |
| `--effect` | `mix` | `clean`, `ocrsmith`, `augraphy`, `scan-simulator`, `augraphy+scan-simulator`, or weights (mix: clean 15%, the rest even) |
| `--source` | `mix` | template source: `grammar` (own), `websight`, `websight2` (web pages), `doclaynet` (scanned documents' layouts), or weights |
| `--task` | `mix` | Surya prompt: `full_ocr`, `layout`, `block_ocr`, `table`, or weights (mix: 40/25/35, `table` only when asked) |
| `--tashkeel` | `mix` | Arabic diacritics: `none`, `partial`, `full` (mix: 40/30/30). Drawn, never in the answer |
| `--templates` | `bank` | `bank` (the 2,000 templates) or `llm` (pages written by an LLM; needs `GROQ_API_KEY` in `.env`, see `.env.example`) |
| `--text` | `pools` | `pools` (the documents of the content pools), `wikipedia` (a small Wikipedia pool, if built), `demo` |
| `--strength-ocrsmith`, `--strength-augraphy`, `--strength-scan-simulator` | 0.5 | as above |
| `--colour-share` | 0.5 | share of pages in colour (the rest white, off-white or a light tint) |
| `--graphic-share` | 0.35 | share of picture boxes given a chart, flowchart or molecule instead of a photo |
| `--special-share` | 0.5 | share of pages given equations, code, forms, contents or references (when not balanced) |
| `--blank-share` | 0.01 | share of blank pages (`Blank-Page`) |
| `--no-balance` | off | stop giving each sample one of the nine structured blocks in turn |
| `--low-dpi` | `layout=0.5,full_ocr=0.2` | share of samples per task shown at 96 DPI instead of 192 |
| `--checks` | `all` | `all`, `none`, or some of `format,dataset,labels,boxes,length,surya` (section 5) |
| `--no-validate` | off | same as `--checks none` (the generator's own page checks still run) |
| `--surya-check` | `auto` | Surya's own parser on every answer: `auto` (when `surya/` is there), `yes`, `no` |
| `--no-render-checks` | off | skip the generator's own page checks (hidden, cut or overlapping text...) |
| `--render-attempts` | 8 | renders per seed until one passes those checks |
| `--order-tries` | 6 | draws per seed until the reading order is unambiguous |
| `--max-length` | 6144 | most tokens of image + prompt + answer (Surya's tokenizer); longer samples are made again |
| `--max-retries` | 5 | new seeds per sample before giving up on it |

## 3. What a dataset folder holds

```
datasets/OCR-Test/
  images/<id>.jpg            what the model is shown: the page, or the block / table cut out of it
  prompts/<id>.prompt.txt    Surya's prompt for the sample's task
  targets/<id>.target.txt    Surya's answer (full_ocr, block_ocr), or .target.json (layout, table)
  pages/<id>_page.jpg        the whole page, for samples whose image is a cut-out
  labels/<id>.json           everything about the page: words, lines and slots with boxes, the Surya blocks
                             (label, box in pixels, HTML), the template, style, effect, text sources and licences
  markdown/<id>.md           one <text>... <bbox>x0, y0, x1, y1</bbox></text> line per block
  html/<id>.html             the filled page the image was rendered from
  manifest.jsonl             one line per sample: files, language, template, task, effect...
  train.jsonl                one ms-swift row per sample: {"messages": [user "<image>" + prompt, assistant answer],
                             "images": [image path]}
```

Sample ids are `e<epoch>_<n>` (an epoch) or `b<start>_<n>` (a batch). Running the same epoch into the same folder
again writes the same files again and adds its rows to `manifest.jsonl` and `train.jsonl` a second time: use
another epoch number or another folder.

## 4. Tasks and labels

The four tasks are Surya's own prompts:

| Task | Prompt (as Surya has it) | Answer |
|---|---|---|
| `full_ocr` | `OCR this image to HTML. Each block is a div with data-label and data-bbox (x0 y0 x1 y1, normalized 0-1000).` | every block in reading order, in Surya's compact form: `<\|Label\|><\|x0\|><\|y0\|><\|x1\|><\|y1\|>` then its HTML |
| `layout` | `Output the layout of this image as JSON. ...` | `[{"label", "bbox": "x0 y0 x1 y1", "count"}, ...]` |
| `block_ocr` | `OCR this block image to HTML.` | one block cut out of the page, its HTML |
| `table` | `Output the table rows then columns as JSON. ...` | `[{"label": "Row" or "Col", "bbox"}, ...]` |

Boxes are whole numbers 0-1000 of the image's width and height. The 19 labels Surya writes, and what feeds each:

| Label | Content | HTML in the answer |
|---|---|---|
| Text | paragraphs of the page's document (one document per page) | `<p>...</p>` (bold, italic, sub/superscript as drawn) |
| Section-Header | the document's headings | `<h1>`...`<h5>` |
| List-Group | the document's list items, drawn with bullets or numbers | `<ul>` / `<ol>` with the mark in each item |
| Table-Of-Contents | the document's headings with page numbers | `<table>` |
| Caption | real figure captions (Arabic and English documents, arXiv) or the document | `<p>` |
| Page-Header | real page headers (DocLayNet, English) or the document's title | `<p>` |
| Page-Footer | page facts: page numbers, copyright, document numbers, dates, the running title | `<p>` |
| Footnote | numbered notes to a mark in the text (DocLayNet's real footnotes in English) | `<p><sup>1</sup> ...</p>` |
| Table | real tables (Wikipedia, documents, PubTabNet, FinTabNet, SynthTabNet) in the templates' tables | `<table>` with merged cells |
| Bibliography | real works (OpenAlex, Crossref) | `<ul>` of references |
| Form | real form fields (Arabic and English) with lines and check boxes | `<table>` |
| Code-Block | real functions in six languages (CodeSearchNet) | `<pre>` |
| Equation-Block | real formulas (arXiv, Wikipedia, UniMER and others), drawn by MathJax | `<math display="block">LaTeX</math>` |
| Chemical-Block | PubChem molecules drawn with RDKit's layout | `<img alt="Chemical structure of ..."/>` |
| Figure | charts of real data (SBS Figures, UniChart, World Bank), English | `<img alt="Bar chart: ..."/>`, title, axes, data table, description |
| Diagram | flowcharts of real processes, English | `<img alt="Flowchart: ..."/>`, Mermaid, description |
| Image | photos with their captions (PD12M, Flickr8k, Polaris) | `<img alt="Photo: ..."/><p class="img-desc">...</p>` |
| Complex-Block | panels: a photo, a figure and a sentence | `<img alt="..."/>` + text |
| Blank-Page | an empty page (1%) | empty |

(Surya's twentieth label, `Picture`, is not made: Surya's model never writes it.)

## 5. The checks

Every sample `generate.py` / `train.py` keeps has passed:

| Check | What it checks |
|---|---|
| the generator's own | no hidden, cut, clipped or overlapping text, every picture drawn whole, every structured block on the page, an unambiguous reading order (pages failing are drawn again) |
| `format` | the answer is in Surya's format for its task (`pipeline/checking.py` `problems()`) |
| `dataset` | Surya's prompt, Surya's encoder gives the answer back unchanged, only Surya's tags and labels, picture blocks the pools make, no diacritics in answers (`surya_probe/check_dataset.py`) |
| `labels` | every text the page draws is in the answer and nothing else, with its bold / italic / sub / sup, headings at their level, nested lists nested, page headers and footers in their place, footnotes with their marks (`surya_probe/check_labels.py`) |
| `boxes` | every box on the page; on clean pages also every box holds its ink, cuts no letters and has no empty margin (`surya_probe/check_boxes.py`) |
| `length` | image + prompt + answer within `--max-length` tokens |
| `surya` | Surya's own parser accepts the answer (needs `surya/`, section 1) |

The same checks run on a whole folder:

```bash
.venv-pipeline\Scripts\python surya_probe\check_dataset.py datasets\OCR-Test
```
```bash
.venv-pipeline\Scripts\python surya_probe\check_labels.py datasets\OCR-Test
```
```bash
.venv-pipeline\Scripts\python surya_probe\check_boxes.py datasets\OCR-Test --effect clean
```

## 6. Training with it

**From a folder:** `train.jsonl` is ready for ms-swift (`swift sft --dataset datasets/OCR-Test/train.jsonl ...`).
Make a new folder per epoch (`--epoch 1`, `--epoch 2`, ...) so every epoch has new pages.

**In a training loop**, with no dataset made beforehand (`train.py`):

```python
from train import DataGenerator

if __name__ == "__main__":                    # needed on Windows: the workers are new processes
    with DataGenerator(workers=4, task={"full_ocr": 0.4, "block_ocr": 0.35}, tashkeel="none") as gen:
        for epoch in range(5):
            pages = gen.epoch(1000, epoch=epoch)            # the same pages for the same seed and epoch
            rows = [p.swift_row() for p in pages]            # {"messages": [...], "images": [PIL image]}
            ...                                              # train on rows
        batch = gen.batch(8)                                 # or the next 8 pages of an endless stream
```

| `DataGenerator` | |
|---|---|
| `DataGenerator(**options)` | starts the workers once (about 10 s); the options are `train.py`'s, as Python values: `language="ar"`, `task={"full_ocr": 0.6, "block_ocr": 0.4}`, `strengths={"ocrsmith": 0.3}`, `validate=["format", "labels"]`, ... |
| `.batch(n)` | the next n samples (never the same twice in a run) |
| `.epoch(n, epoch=k)` | the n samples of epoch k |
| `.stream(batch_size)` | an endless iterator of samples |
| `.stats`, `.rejections` | how many samples each check sent back, and why |
| `generate_data(mode="batch" or "epoch", num_samples, epoch, **options)` | one call (starts and stops the workers) |
| `save(samples, folder)` | writes them as section 3 shows |

A `Sample` has `image` (PIL), `image_bytes` (JPEG), `prompt`, `target`, `task`, `label` (the labels JSON), `info`
(template, language, effect...), `page_bytes` (the whole page for cut-outs), `checks` and `swift_row()`. A sample
that can't be made after `--max-retries` new seeds raises `GenerationError` (its `.samples` are the ones that
were made).

Three settings Surya needs in ms-swift (Surya is a Qwen3.5 model with its own vocabulary): load Surya's own
tokenizer (`TokenizersBackend.from_pretrained("datalab-to/surya-ocr-2.5")`: a Qwen tokenizer breaks Arabic into
bytes), set the image token ids to 11 and 12, and add the end token by its id (2) so it is never merged with the
answer's last characters.

**Speed**: about 2-3 s per sample per worker, plus about 10 s to start the workers (4 workers: 3,000-5,000
samples an hour). An epoch of 1,000 samples is
about 200-300 MB of images in memory.

## 7. Content pools

`dataset_pool_sample/` (in git, 16 MB) holds about 20 items of every pool: 20 documents per language, 20 tables,
captions and references per language, 20 page headers, footnotes, forms, processes, functions, molecules, formulas,
20 charts' data per source, 20 photos of each photo pool, 20 flowcharts, 20 ready-made drawings per picture label
and 20 font families. It is enough to generate every label; its pages repeat its content.

The full pools (about 80 GB) go in `dataset_pool/` with the same layout; the generator reads `dataset_pool/` when it
is there and the sample otherwise:

```
dataset_pool/
  text/{ar,en}  parallel  layout  tables/{ar,en}  captions/{ar,en}  equations  code  molecules
  references/{ar,en}  vocab  chartdata  cosyn  diagrams
  images/   photos.jsonl + captions.jsonl (photos in pd12m/, polaris/, output_dataset/), flowcharts/, graphics/
  fonts/    fonts-more.json and the font files (100 families)
  _lid/     fastText's language identifier
```

Items are JSON lines (at most 50,000 per file), each with its `source` and `license`, an `.idx` next to each file
(`Pool(name).pick(rng)` reads one random line). `pipeline/content_pools.py` builds them from the public datasets
(its docstring lists every command; some sources need a Kaggle token or an OpenAlex key), `python -m
pipeline.content_pools index` indexes them, `python -m pipeline.sample_pool` makes the sample from them.

## 8. The code

| File | What it does |
|---|---|
| `generate.py` | the command line above (the everyday settings) |
| `train.py` | `DataGenerator`: samples on request for a training loop, every setting, the checks; also a command line |
| `pipeline/generator.py` | one sample: template, fill, render, effect, Surya answer (`DocPipeline.sample(seed)`, steps `STEP 1`..`STEP 6`) |
| `pipeline/checking.py` | the scripts run in the page (fit text, lists, footnotes, headers...), the page checks, the answer format check |
| `pipeline/pools.py` | the small pools and readers (text, pictures, fonts), charts, flowcharts and molecules drawn |
| `pipeline/content_pools.py` | the large pools: builders and readers |
| `pipeline/flowchart_pool.py` | draws the flowchart pool |
| `pipeline/sample_pool.py`, `pipeline/move_pools.py` | make `dataset_pool_sample/`; move the pools into `dataset_pool/` |
| `pipeline/run.py` | `python -m pipeline.run dataset ...`: an older dataset command (the generator's own checks only) |
| `pipeline/llm_templates.py` | optional templates written by an LLM |
| `surya_probe/check_*.py`, `surya_probe/surya_pipeline.py` | the checks of section 5, Surya's parser on an answer |
| `bank2000/` | the 2,000 templates (`template.html.j2` with `{{ slot_N }}`, `template.json` with each slot's role) |
| `originals/` | SynthDoc and its Arabic extension (rendering), OCRSmith, scan-simulator, Augraphy, Surya's `compact.py` and prompts, copied unchanged with their licences |

## 9. Sources and licences

Templates: own grammar, WebSight v0.1 / v0.2 (CC BY 4.0), DocLayNet (CDLA-Permissive-1.0). Text: FineWiki
(CC BY-SA 4.0), FineWeb-2 / FineWeb-Edu / FinePDFs (ODC-By 1.0), HPLT, 101 Billion Arabic Words. Photos: PD12M (public
domain / CC0), Flickr8k (research use), Polaris (BSD-3-Clause). Each pool item and each sample's labels name their
sources; a shared dataset must keep CC BY-SA's attribution. The tools in `originals/` keep their own licences.
