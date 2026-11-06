# SDS Bench — Synthetic Document Dataset Benchmark

A reproducible benchmark for evaluating OCR, NER, and document layout models on synthetic documents with pixel-perfect ground truth.

## Architecture

```
generate  →  benchmark  →  charts
Playwright    torch        matplotlib
```

No external services required.

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd synthetic-document-dataset
pip install faker jinja2 playwright PyMuPDF rdflib pyyaml Pillow
playwright install chromium

# 2. Generate dataset (~30 min, CPU only)
synthdoc --config examples/config_standard.yaml --count 10 --templates \
    restaurant_receipt invoice business_card resume \
    bank_statement credit_card_statement irs_form_1040 w2_form \
    passport driver_license \
    work_order aia_g702_g703 bill_of_materials purchase_order \
    blueprint_technical_drawing \
    cms1500 ub04_hospital_claims explanation_of_benefits \
    medical_prescription lab_report \
    closing_disclosure insurance_policy commercial_rent_roll \
    pay_stub remittance_advice \
    legal_pleading commercial_invoice certificate_of_analysis \
    home_inspection_report academic_transcript

# 3. Run benchmark (GPU)
pip install torch transformers pytesseract spacy pandas numpy matplotlib pyyaml
python -m spacy download en_core_web_trf
python -m synthdoc.bench run benchmarks/ner_comparison.yaml

# 4. Generate charts
python -m synthdoc.bench.make_charts --results results/ner_comparison/ --output charts/
```### Verify data quality

Overlays (images with bounding boxes) are generated automatically during step 2. Browse `data/overlays/` to spot-check annotations.

Generate an HTML inspection report to browse all documents in one page:
```bash
synthdoc --inspection-only                      # From existing data
```

### Generation output structure

```
data/
├── images/          # JPG page images (816x1056 px)
├── labels/          # JSON annotations with bounding boxes
├── html/            # Source HTML with RDFa markup (ground truth source)
├── overlays/        # Debug images with bbox overlays
├── pdfs/            # Multi-page PDFs
└── logs/            # Generation logs
```

### Running on a remote GPU server

If you generate locally but need to run inference on a separate GPU machine:

**Option A — rsync the repo** (recommended during development)
```bash
rsync -avz --exclude __pycache__ --exclude .git \
    ./synthetic-document-dataset/ user@gpu-server:~/sds-bench/
```

**Option B — zip just the data**
```bash
# Local: zip and send
cd data && zip -r ../sdd-nano.zip images/ labels/ html/ && cd ..
scp sdd-nano.zip user@gpu-server:~/

# Server: unzip and clone repo for benchmark code
ssh user@gpu-server
mkdir -p ~/sds-bench && cd ~/sds-bench
unzip ~/sdd-nano.zip -d data/
git clone <repo-url> repo && cp -r repo/synthetic-document-dataset/synthdoc.bench .
cp -r repo/synthetic-document-dataset/benchmarks .
```

**After deploying (either option), run on the server:**
```bash
ssh user@gpu-server
cd ~/sds-bench

# Install benchmark dependencies
pip install torch transformers pytesseract spacy pandas numpy matplotlib pyyaml Pillow
python -m spacy download en_core_web_trf

# See available engines
python -m synthdoc.bench list

# Run from a config file
python -m synthdoc.bench run benchmarks/ner_comparison.yaml

# Quick test on 50 pages
python -m synthdoc.bench run benchmarks/ner_comparison.yaml --limit 50

# Generate charts
python -m synthdoc.bench.make_charts --results results/ --output charts/
```

## Template Categories

### Baseline (10) — Public Domain
Documents commonly found in existing benchmarks and public datasets.

| Template | Category |
|----------|----------|
| `restaurant_receipt` | Retail (SROIE standard) |
| `invoice` | Financial (RVL-CDIP standard) |
| `business_card` | Business |
| `resume` | HR |
| `bank_statement` | Financial |
| `credit_card_statement` | Financial |
| `irs_form_1040` | Tax/Government |
| `w2_form` | Tax/Government |
| `passport` | Identity |
| `driver_license` | Identity |

### High Value Niche (20) — Rare/Specialized
Documents rarely found in public training sets. Maximum commercial value.

| Template | Category | Complexity |
|----------|----------|------------|
| `work_order` | Construction | Parts vs labor tables |
| `aia_g702_g703` | Construction | Complex payment grids |
| `bill_of_materials` | Manufacturing | Recursive hierarchies |
| `purchase_order` | Procurement | B2B tables |
| `blueprint_technical_drawing` | Construction | Spatial/non-linear text |
| `cms1500` | Healthcare | Red dropout ink form |
| `ub04_hospital_claims` | Healthcare | Dense billing grids |
| `explanation_of_benefits` | Healthcare | Insurance breakdowns |
| `medical_prescription` | Healthcare | Script variance |
| `lab_report` | Healthcare | Results with ranges |
| `closing_disclosure` | Real Estate | Dense financial figures |
| `insurance_policy` | Insurance | Declaration page |
| `commercial_rent_roll` | Real Estate | Spreadsheet-style |
| `pay_stub` | HR | Earnings/deductions |
| `remittance_advice` | Financial | B2B payment details |
| `legal_pleading` | Legal | Caption box + line numbers |
| `commercial_invoice` | Trade | Customs/HS codes |
| `certificate_of_analysis` | Industrial | Quality specs |
| `home_inspection_report` | Real Estate | Checklist + narrative |
| `academic_transcript` | Education | Grade tables |

## Pipeline

### Generation (ground truth)

```
HTML + RDFa  →  Paged.js  →  PDF  →  JPG images  +  JSON labels
  Faker data     Chromium     PyMuPDF   (input)        (ground truth)
```

Templates contain RDFa markup (`typeof="schema:Person"`, `property="schema:legalName"`) that encodes entity types directly in the HTML. During rendering, Paged.js extracts bounding boxes for every annotated element. The result is pixel-perfect alignment between the image and its labels — no manual annotation needed.

### Benchmark (four evaluation tracks)

Evaluate models across four tracks: can they read the text (OCR), can they find the entities (NER), can they find the regions (segmentation), and how much does fine-tuning on synthetic data help?

#### Track 1 — OCR (image → text)

Run each OCR engine on every document image, compare output to ground truth text.

```
JPG image  →  OCR engine  →  predicted text  →  compare to GT text  →  WER, CER
```

| OCR engine | Model ID | Type | GPU? |
|------------|----------|------|------|
| Tesseract 5 | (system binary) | Traditional | No |
| Qwen3-VL | `Qwen/Qwen3-VL-8B-Instruct` | VLM | Yes |
| DeepSeek-OCR | `deepseek-ai/DeepSeek-OCR-2` | VLM | Yes |

**Evaluation**: Word Error Rate (WER) and Character Error Rate (CER) per page. Breakdown by template type to find which document layouts each engine struggles with.

#### Track 2 — NER (text → entities)

For each OCR output, run every NER model. All models map to a consistent type set (PERSON, ORG, DATE, ADDRESS, AMOUNT, PHONE, ID).

```
OCR text  →  NER model  →  predicted entities  →  compare to GT entities  →  P/R/F1
```

| NER model | Model ID | Type | GPU? |
|-----------|----------|------|------|
| spaCy | `en_core_web_trf` | Pipeline | No |
| DeBERTa | `dslim/deberta-v3-large-ner` | Transformer | Yes |
| Qwen3 | `Qwen/Qwen3-8B` | LLM (vLLM) | Yes |

**Evaluation**: Strict F1 (text + type match), Relaxed F1 (text only), per-entity-type P/R/F1, confusion matrix. Error attribution: when NER misses an entity, is it because OCR dropped the text or because NER failed to extract it?

#### Track 3 — Document segmentation (image → regions)

Run layout detection models directly on images, compare predicted bounding boxes to ground truth regions. No OCR needed — this evaluates whether models can locate and classify document regions (tables, headers, fields, etc.).

```
JPG image  →  layout model  →  predicted boxes + types  →  compare to GT boxes  →  mAP, F1@0.5
```

| Layout model | Model ID | Type | GPU? |
|--------------|----------|------|------|
| DETR | `cmarkea/detr-layout-detection` | Transformer | Yes |
| Heron | `docling-project/docling-layout-heron-101` | RT-DETR | Yes |
| DocLayout-YOLO | `juliozhao/DocLayout-YOLO-DocStructBench` | YOLO | Yes |
| Florence-2 | `yifeihu/Florence-2-DocLayNet-Fixed` | VLM | Yes |
| DiT | `microsoft/dit-large` | Transformer | Yes |

**Evaluation**: F1@IoU=0.5, COCO mAP@[0.5:0.95], per-type breakdown (TABLE_CELL, FIELD_LABEL, TITLE, etc.), table structure accuracy (row/column matching).

#### Entity type sets

The ground truth contains 14 entity types. Off-the-shelf NER models only recognize 7. Every evaluation reports **two lines**:

| Evaluation | Entity types | Purpose |
|------------|-------------|---------|
| **Standard NER** (7 types) | PERSON, ORG, DATE, ADDRESS, AMOUNT, PHONE, ID | Apples-to-apples comparison of off-the-shelf models |
| **Full document** (14 types) | + FIELD_LABEL, TABLE_CELL, TABLE_HEADER, TITLE, SECTION_TITLE, TEXT, CAPTION | Measures what fine-tuning unlocks |

Off-the-shelf models score 0 on the full set for types they don't know. After fine-tuning, those types should light up — that delta is the value of the synthetic training data.

#### Track 4 — Fine-tuning (baseline → fine-tuned)

After baselining, fine-tune the same models on the synthetic training data and re-benchmark on both entity type sets. Target: ~7-8B parameter models (Qwen3-8B, DeBERTa-v3-large) that can be fine-tuned on a single GPU.

```
Baseline (7 types)   →  fine-tune on synthetic data  →  re-benchmark (7 types)   →  measure NER improvement
Baseline (14 types)  →  fine-tune on synthetic data  →  re-benchmark (14 types)  →  measure document understanding improvement
```

## Running the Benchmark

There are two ways to run the benchmark: **config-driven** (recommended) and **manual step-by-step** (legacy).

### Config-driven (`run` command)

Define what to run in a YAML file, then execute it with a single command.

```bash
# List available engines
python -m synthdoc.bench list

# Run from a config file
python -m synthdoc.bench run benchmarks/ner_comparison.yaml

# Quick test on 50 pages
python -m synthdoc.bench run benchmarks/ner_comparison.yaml --limit 50

# Re-run OCR even if cached
python -m synthdoc.bench run benchmarks/ner_comparison.yaml --force-ocr
```

**Run config format** (e.g. `benchmarks/ner_comparison.yaml`):

```yaml
name: NER Comparison

data: ../data          # relative to this file, or absolute path
output: ../results/ner_comparison

ocr: qwenvl           # single engine, or a list: [qwenvl, deepseek]

ner:                   # list of NER engines to run against each OCR output
  - spacy
  - deberta
  - qwen3
```

The runner does OCR first (using cache if available), then runs each NER engine against each OCR output, evaluates, and saves all results. Output files are identical to the manual workflow — CSVs, JSONs, and the same summary table.

See `benchmarks/` for example configs.

**Adding a new model** — edit `benchmarks/engines.yaml`:

```yaml
ner:
  my-new-model:
    description: "My custom model"
    model_id: org/my-model-name
    runner: qwen3        # reuse an existing runner, just swap the model weights
    gpu: true
```

The `runner` field maps to dispatch logic in `synthdoc.bench/config.py`. If the new model uses the same inference code as an existing engine (e.g. another vLLM-hosted LLM), set `runner` to that engine name. If it needs completely new code, add a new dispatch entry in `config.py`.

Available runners and their model IDs are listed by `python -m synthdoc.bench list`.

### Manual step-by-step (legacy)

Run OCR and NER as separate commands. Useful for debugging a single step.

```bash
# Step 1: OCR (caches results to _cache_<engine>_ocr.json)
python -m synthdoc.bench ocr --dir ./data --engine tesseract
python -m synthdoc.bench ocr --dir ./data --engine qwenvl
python -m synthdoc.bench ocr --dir ./data --engine deepseek

# Step 2: NER (reads OCR cache, runs NER, evaluates, saves results)
python -m synthdoc.bench ner --dir ./data --engine spacy   --ocr tesseract
python -m synthdoc.bench ner --dir ./data --engine deberta  --ocr qwenvl
python -m synthdoc.bench ner --dir ./data --engine qwen3    --ocr deepseek

# Full pipeline (original, runs OCR + NER together)
python -m synthdoc.bench bench --dir ./data --model tesseract_spacy
python -m synthdoc.bench bench --dir ./data --model deepseek_qwen3 --limit 20
```

### Legacy full pipeline (`run_benchmark.py`)

```bash
# CPU baseline
python -m synthdoc.bench.run_benchmark --data ./data --output results/ --model tesseract_spacy

# All models
python -m synthdoc.bench.run_benchmark --data ./data --output results/

# Quick test
python -m synthdoc.bench.run_benchmark --data ./data --output results/ --limit 20
```

## Metrics

### NER Evaluation
- **Strict F1**: Text match + type match
- **Relaxed F1**: Text match only (ignores type)
- **Per-entity-type P/R/F1**: Breakdown by PERSON, ORG, DATE, ADDRESS, AMOUNT, PHONE, ID
- **Confusion matrix**: Type misclassification patterns

### OCR Evaluation
- **WER**: Word Error Rate
- **CER**: Character Error Rate

### Segmentation Evaluation
- **F1@0.5**: IoU threshold 0.5
- **COCO mAP@[.5:.95]**: Standard COCO metric

### Statistical Rigor
- **Bootstrap CI**: 95% confidence intervals (2000 resamples)
- **Paired significance test**: p-values for model comparisons

## Label Format

Each JSON label file contains:

```json
{
  "document_id": "INV_000001",
  "template_type": "invoice",
  "page_number": 1,
  "total_pages": 2,
  "width": 816,
  "height": 1056,
  "annotations": [
    {
      "text_content": "Acme Corporation",
      "entity_type": "ORG",
      "box_2d": [45, 102, 312, 128],
      "property": "schema:legalName",
      "typeof": "schema:Organization"
    }
  ]
}
```

**Bounding box format**: `[x1, y1, x2, y2]` normalized to 0-1000 scale.

**Entity types**: PERSON, ORG, DATE, ADDRESS, AMOUNT, PHONE, ID, FIELD_LABEL, TABLE_CELL, TABLE_HEADER, TITLE, SECTION_TITLE, TEXT, CAPTION

## Module Reference

| File | Purpose | Dependencies |
|------|---------|-------------|
| `config.py` | Config-driven runner, engine registry, orchestration | yaml |
| `__main__.py` | CLI entry point (`python -m synthdoc.bench run/list/ocr/ner/bench`) | argparse |
| `run_benchmark.py` | Legacy pipeline, OCR/NER runners (tesseract, spacy, deberta) | pytesseract, spacy, transformers |
| `live.py` | GPU inference runners (Qwen3-VL, DeepSeek, Qwen3 via vLLM) | torch, transformers, vllm |
| `helpers.py` | Metrics, matching, scoring, degradation, response parsing | numpy, PIL |
| `static.py` | Offline evaluation suite (NER, OCR, segmentation, significance) | pandas, numpy |
| `datasets.py` | Data loaders (SDD, DocLayNet, D4LA) | json, PIL |
| `baselines.py` | 6 segmentation model wrappers | torch, transformers |
| `report.py` | 60+ chart/visualization functions | matplotlib |
| `make_charts.py` | Chart generation CLI | matplotlib, pandas |

## Config Files

```
benchmarks/
├── engines.yaml              # Engine registry — all available OCR/NER/seg models
├── ner_comparison.yaml       # Example: compare NER engines on QwenVL OCR
└── (add your own configs)
```

### engines.yaml

Defines every engine the benchmark knows about. Each engine has:

| Field | Required | Description |
|-------|----------|-------------|
| `description` | yes | Human-readable name |
| `model_id` | no | HuggingFace model ID (for transformer/vllm models) |
| `runner` | no | Name of an existing runner to reuse (defaults to engine name) |
| `gpu` | yes | Whether the engine requires a GPU |

When `runner` is set, the engine reuses an existing runner's dispatch code but swaps the `model_id`. This lets you add model variants without writing Python:

```yaml
ner:
  deberta:
    description: "DeBERTa-v3-large NER"
    model_id: dslim/deberta-v3-large-ner
    gpu: true

  deberta-base:                    # same runner, smaller model
    description: "DeBERTa-v3-base NER"
    model_id: dslim/deberta-v3-base-ner
    runner: deberta                 # reuses deberta dispatch code
    gpu: true
```

To add an entirely new engine type (not just a new model), add a dispatch entry in `synthdoc.bench/config.py` (`_dispatch_ocr`, `_dispatch_ner`, or `_dispatch_seg`).

### Run config

A run config specifies which engines to use for a benchmark run:

```yaml
name: My Benchmark              # optional display name
data: ../data                   # directory with images/ and labels/
output: ../results/my_bench    # where to write results
limit: 100                      # optional: cap pages

ocr:                            # single engine or list
  - qwenvl

ner:                            # each runs against each OCR output
  - spacy
  - deberta
  - qwen3
  - qwen3-small
```

Paths are resolved relative to the config file's location.

## Input

The benchmark expects this directory layout:

```
data/
├── images/              # JPG page images (816x1056 px)
│   ├── doc_001_page_1.jpg
│   ├── doc_001_page_2.jpg
│   └── ...
└── labels/              # JSON ground truth annotations
    ├── doc_001_page_1.json
    ├── doc_001_page_2.json
    └── ...
```

OCR results are cached automatically as `_cache_<engine>_ocr.json` in the data directory. Subsequent runs reuse the cache unless you pass `--force-ocr`.

## Output

All results are written to the output directory:

```
results/
├── benchmark_results.csv       # Per-page NER scores (strict/relaxed P/R/F1)
├── ocr_results.csv             # Per-page OCR quality (WER, CER)
├── per_type_results.csv        # Per-entity-type P/R/F1 breakdown
├── entity_matches.csv          # Per-entity match details (GT vs predicted)
├── ocr_outputs.csv             # Side-by-side OCR text (GT + all engines)
├── dataset_stats.json          # Document/template/entity counts
├── timings.json                # Inference time + GPU memory per step
├── confusion.json              # Type misclassification matrices
├── mismatches.json             # Top mismatch patterns + examples
├── error_attribution.csv       # OCR vs NER error breakdown
├── calibration.json            # Confidence vs accuracy
└── significance.json           # Pairwise bootstrap p-values
```

Results from multiple runs are **merged** — re-running a model replaces its rows in the CSVs but preserves data from other models.

## Reproducing Results

```bash
# Generate dataset
synthdoc --config examples/config_standard.yaml --count 10 --templates <all 30>

# Run benchmark
python -m synthdoc.bench run benchmarks/ner_comparison.yaml

# Charts
python -m synthdoc.bench.make_charts --results results/ner_comparison/ --output charts/
```
