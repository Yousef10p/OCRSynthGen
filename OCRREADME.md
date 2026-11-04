# OCRREADME: our training data for Surya OCR 2.5

How Surya OCR 2.5 labels a page, how we know, what our generator writes, and how to fine-tune Surya on it.

The earlier version of this file (a step-by-step walkthrough of how the generator makes a page) is in the git
history, at commit `e3ad56fc4f`.

**Contents**

1. [In one minute](#1-in-one-minute)
2. [Surya 2.5 and its four prompts](#2-surya-25-and-its-four-prompts)
3. [How Surya writes a label (from its code)](#3-how-surya-writes-a-label-from-its-code)
4. [What we asked Surya, and where it is saved](#4-what-we-asked-surya-and-where-it-is-saved)
5. [What Surya's answers showed, and what we changed](#5-what-suryas-answers-showed-and-what-we-changed)
6. [What we left as it is](#6-what-we-left-as-it-is)
7. [One page, start to end](#7-one-page-start-to-end)
8. [How close we are now](#8-how-close-we-are-now)
9. [Fine-tuning: the notebook](#9-fine-tuning-the-notebook)
10. [How to run it](#10-how-to-run-it)
11. [Not finished yet](#11-not-finished-yet)
12. [Files](#12-files)

---

## 1. In one minute

- **The goal:** fine-tune Surya OCR 2.5 on our generated pages so it reads better, without changing the
  *shape* of its answers.
- **How Surya labels a page:** one HTML `<div>` per block (label + box + content), turned into a short token
  form by its own encoder. We found this in Surya's code (section 3).
- **How Surya really behaves:** we sent it 100 pages per prompt and kept every image, prompt and answer
  (section 4). Where its answers are consistent, our labels now do the same (section 5).
- **Where it is not consistent** (heading levels, bold in headings, joining paragraphs), no rule can match it,
  so we keep one fixed choice (section 6).
- **Result:** the same answer format exactly, the same block label 92% of the time, the boxes overlap 0.88
  on average (section 8).
- **Training:** only `full_ocr` and `block_ocr`, in a Colab notebook (sections 9 and 10).

---

## 2. Surya 2.5 and its four prompts

`datalab-to/surya-ocr-2.5`: one vision-language model (689M weights) that answers four fixed prompts. Surya's
code calls the prompt texts *the model's training-time contract*, so we use them word for word.

| Task | Prompt | The answer |
|---|---|---|
| `full_ocr` | `OCR this image to HTML. Each block is a div with data-label and data-bbox (x0 y0 x1 y1, normalized 0-1000).` | every block of the page: label, box, content |
| `block_ocr` | `OCR this block image to HTML.` | the content of one cut-out block, nothing else |
| `layout` | `Output the layout of this image as JSON. Each entry is a dict with "label", "bbox", and "count" fields. Bbox is x0 y0 x1 y1, normalized 0-1000.` | one JSON line: label, box and length of every block |
| `table` | `Output the table rows then columns as JSON. ...` | rows then columns (Surya 2's format; not seen from 2.5 yet) |

Boxes are always `x0 y0 x1 y1`, whole numbers from 0 to 1000 (a share of the image's width and height,
top left is 0 0).

---

## 3. How Surya writes a label (from its code)

Surya's training code is not public (`surya/common/order/order_ar.py` says it was copied from
`training/...` "training code stripped"), but its inference code shows the label format exactly.
We cloned it into `surya/` (commit 9d5aadf; that folder is not in our repository).

**Step 1: a page label is HTML, one `div` per block, in reading order**

```html
<div data-bbox="425 337 567 353" data-label="Text"><p>MAD, MAD/I, and:</p></div>
<div data-bbox="0 745 300 821" data-label="List-Group"><ul><li>▪ PRINT FORMAT HELLOW</li>...</ul></div>
```

The label is one of 19 (`surya/layout/schema.py`): Text, Section-Header, List-Group, Table, Image, Caption,
Footnote, Page-Header, Page-Footer, ... We use 9 of them.

**Step 2: `encode()` turns it into the text the model writes** (`surya/inference/compact.py`)

| Rule | Example |
|---|---|
| a block `div` becomes a label token and four number tokens | `<div data-bbox="425 337 567 353" data-label="Text">` -> `<\|Text\|><\|425\|><\|337\|><\|567\|><\|353\|>` |
| no `</div>`: the next block (or the end) closes a block | |
| nothing between blocks: no spaces, no new lines | |
| in tables `</td>`, `</th>`, `</tr>` are left out | `<tr><td>a<td>b</table>` |
| every tag has one spelling: lower case, fixed attribute order, `<br/>` | |

**Step 3: the answer ends with the end token** `<|im_end|>` (token id 2). The model writes it to stop.

Other facts from the code:

- `block_ocr` cut-outs are the block's box plus 4 pixels (`_crop_block`); pictures are not read block by block.
- Images are made smaller to fit 3072 x 2048 pixels at most.
- Longest answers: full page 12,288 tokens, one block 8,192, layout 3,072.
- The layout `count` is the model's guess of how many tokens the block's text needs.
- The reading order comes from a trained model (`order_ar.py`), not from a rule.

Our copy of the encoder rules is `pipeline/surya_encode.py`; our labels equal its output on 100 of 100 pages.
More detail: [surya_repo_findings.md](surya_repo_findings.md).

---

## 4. What we asked Surya, and where it is saved

We sent generated pages to a Surya 2.5 server, one request at a time, temperature 0.

| Folder | Pages | Answered |
|---|---|---|
| `surya_probe/batch100/full_ocr/` | 100 pages with effects (scan, noise...) | 100 |
| `surya_probe/batch100/block_ocr/` | 100 cut-out blocks | 100 |
| `surya_probe/batch100/layout/` | 100 pages | 75 (the server went down) |
| `surya_probe/batch100/table/` | 109 tables | 0 (the server went down) |
| `surya_probe/clean100/full_ocr/`, `block_ocr/` | 100 + 100 clean pages, no effects | waiting for the server |

Every answered page has its own folder, so you can look at it directly:

```
surya_probe/batch100/<task>/samples/<id>/
    image.jpg          what Surya was shown
    prompt.txt         the prompt
    surya_output.txt   Surya's answer, exactly as it came back
    our_label.txt      our label for the same image (.json for layout)
    info.json          time, tokens, how the answer ended
```

`our_label` there was made before the changes in section 5; new pages get the new rules.

---

## 5. What Surya's answers showed, and what we changed

We matched every block of ours to Surya's block for the same image and counted what Surya did.
Where Surya does one thing most of the time, our labels now do the same (`pipeline/surya.py`).

| What | What Surya does | Our label now |
|---|---|---|
| A short label line with no value, not bold ("History:") | `Text` `<p>` (74 of 83) | `Text` `<p>` (was `Section-Header` `<h4>`) |
| The same, bold | `Section-Header`, most often `<h3>` | `Section-Header` `<h3>` |
| "Label: value" with only the label bold | `<p><b>Label:</b> value</p>` | the same (was all bold) |
| A list right under another list | one `List-Group` (45 of 67) | joined into one |
| Text on one line almost touching other text | one `Text` (20 of 23) | joined into one |
| Box edges | left edge about 0.65% and top edge about 0.35% of the page further out than the text, on clean pages too | the same |
| Layout `count` | the block's length in Surya's tokens, rounded **up** to 10 (not to 50) | the same |
| Two-column English pages | read column by column | the same |
| A page header's text on a full page | bare, without `<p>` | the same |
| A cut-out block | content only, never a label or box (100 of 100) | the same |

---

## 6. What we left as it is

Here Surya's answers do not follow a pattern; any fixed rule matches it about half the time, so we keep a
simple one.

| What | What Surya does | Our label |
|---|---|---|
| Heading level `h1` / `h2` / `h3` | differs for the same kind of heading; no rule above 59% | the page's own heading level |
| `<b>` inside a heading | 78 of 285 headings, no pattern | none |
| Two paragraphs one under the other | joined about half the time (with `<br/>`) | kept apart |
| Groups of numbers on Arabic pages (`٠٥١ ٥٦٢ ٨٢٤٧`) | order reversed 8 times, kept 14 | kept |
| A grid of "label: value" pairs | sometimes one `<table>`, sometimes `Text` lines | `Text` lines |
| Reading order on other pages | a trained model; once even left to right on an Arabic page | the page's own order |
| Text characters | identical on 844 of 1013 blocks; the rest mostly one or two characters | what the page shows |

---

## 7. One page, start to end

Page `full_ocr/0000004` (English, 6 blocks).

**The prompt:** the `full_ocr` prompt from section 2.

**Our label, as HTML (step 1):**

```html
<div data-bbox="425 337 567 353" data-label="Text"><p>MAD, MAD/I, and:</p></div>
<div data-bbox="65 709 229 730" data-label="Text"><p>MAD (Michigan.</p></div>
<div data-bbox="0 745 300 821" data-label="List-Group"><ul><li>▪ PRINT FORMAT HELLOW</li><li>▪ END OF PROGRAM</li><li>▪ P'T HELLOW</li></ul></div>
<div data-bbox="572 742 721 763" data-label="Text"><p>Developed in.</p></div>
<div data-bbox="602 782 690 803" data-label="Text"><p>History:</p></div>
<div data-bbox="557 891 736 907" data-label="Text"><p>"Hello, world" example:</p></div>
```

**Our training answer, encoded (step 2):**

```
<|Text|><|425|><|337|><|567|><|353|><p>MAD, MAD/I, and:</p><|Text|><|65|><|709|><|229|><|730|><p>MAD (Michigan.</p><|List-Group|><|0|><|745|><|300|><|821|><ul><li>▪ PRINT FORMAT HELLOW</li><li>▪ END OF PROGRAM</li><li>▪ P'T HELLOW</li></ul><|Text|><|572|><|742|><|721|><|763|><p>Developed in.</p><|Text|><|602|><|782|><|690|><|803|><p>History:</p><|Text|><|557|><|891|><|736|><|907|><p>"Hello, world" example:</p>
```

**Surya's real answer for the same image:**

```
<|Text|><|425|><|334|><|571|><|354|><p>MAD, MAD/I, and:</p><|Text|><|64|><|708|><|230|><|731|><p>MAD (Michigan.</p><|List-Group|><|20|><|748|><|300|><|817|><ul><li>■ PRINT FORMAT HELLOW</li><li>■ END OF PROGRAM</li><li>■ P'T HELLOW</li></ul><|Text|><|570|><|739|><|722|><|763|><p>Developed in.</p><|Text|><|600|><|780|><|693|><|804|><p>History:</p><|Text|><|553|><|888|><|740|><|909|><p>"Hello, world" example:</p><|im_end!>
```

(The server shows the end token `<|im_end|>` as `<|im_end!>`.) Same blocks, same order, same labels, same tags; boxes within a few thousandths. (Before section 5's
changes, our three short label lines were `Section-Header` `<h4>`; Surya wrote `Text` `<p>`.)

**block_ocr** gets only the content of the cut-out, for example `<p>يتمثل دورهم أيضًا في.</p>`.

**layout** gets one JSON line, for example
`[{"label": "Page-Header", "bbox": "761 95 841 117", "count": 20}, {"label": "Image", "bbox": "196 310 811 387", "count": 40}, ...]`.

---

## 8. How close we are now

Pages made with the current code, compared with Surya's answers on the same images:

| Check | full_ocr (78 pages) | block_ocr (64 blocks) |
|---|---|---|
| Answer format correct | 100 of 100 | 100 of 100 |
| Equal to Surya's own encoding | 100 of 100 | 100 of 100 |
| Same block label | 852 of 931 (92%) | - |
| Same tags in the block | 606 of 931 (65%; the rest is mostly heading levels) | 38 of 64 |
| Box overlap with Surya's box | 0.88 on average (was 0.79) | - |
| Surya adds a label or box to a cut-out | - | never |

Layout (60 pages with the same image): same label 537 of 582 (92%), same `count` 62%. Surya's own layout answers are not always clean:
18 of 75 start as JSON and switch to the full-page form halfway (that server does not force JSON).

A perfect score is not possible: these are Surya's *answers*, with its own mistakes and changes of mind,
not the labels it was trained on (those are private).

---

## 9. Fine-tuning: the notebook

`notebooks/Surya_finetune_Colab.ipynb`, for a free Colab T4.

- **Trains:** `full_ocr` and `block_ocr` pages only. They give all of Surya 2.5's text (tables come out as
  HTML inside them). The table prompt is left out: we have not seen 2.5 answer it.
- **How:** all weights except the vision encoder (601M of 689M), 8-bit AdamW, learning rate 1e-5,
  2 epochs, one batch folder per epoch. Saves to Google Drive and continues after a disconnect.
- **Checks:** before training and after every epoch it reads 8 validation pages and prints how similar the
  answers are to our labels.

Three traps it avoids (each found by testing):

1. **The tokenizer.** transformers loads Surya's tokenizer as a Qwen tokenizer, which splits Arabic into
   bytes; the model then answers cut-outs with nothing. The notebook loads it the way Surya stores it
   (`TokenizersBackend`).
2. **The end token.** Surya's vocabulary has tokens such as `.<` and `><`. Writing the whole chat as one text
   glues the answer's last character to `<|im_end|>`, which then becomes plain text pieces (on 38 of 66 pages);
   the model would learn to type "im_end" instead of stopping. The notebook adds the answer's own tokens and
   then the end token's id (2).
3. **Picture descriptions.** For a picture Surya writes a description of that picture; ours are stock
   sentences per kind of picture ("A photograph of a general scene."). Their words are left out of the loss,
   so the model does not learn them; the tags around them are still learned.

Tested on CPU with the same library versions as Colab (transformers 5.22.1, bitsandbytes 0.50.2): training
steps, validation, saving and resuming. Not yet run on a T4.

---

## 10. How to run it

1. **Make the training pages on your PC**, one folder per epoch, a new seed each (in the project folder):

   ```bash
   ..\arabic-template-bank\.venv-pipeline\Scripts\python -m pipeline.dataset --n 1000 --out batches\epoch_001 --task full_ocr+block_ocr --tashkeel none --seed 1
   ```
   ```bash
   ..\arabic-template-bank\.venv-pipeline\Scripts\python -m pipeline.dataset --n 1000 --out batches\epoch_002 --task full_ocr+block_ocr --tashkeel none --seed 2
   ```

   About 15-25 minutes each with 4 workers. 5% of each folder are validation pages.
2. **Upload both folders to Google Drive** as `MyDrive/surya/batches/epoch_001` and `epoch_002`, whole.
3. **Open the notebook in Colab:** *File -> Upload notebook*, pick `notebooks/Surya_finetune_Colab.ipynb`.
4. **Choose the GPU:** *Runtime -> Change runtime type -> T4 GPU*.
5. **Run:** *Runtime -> Run all*, allow Drive access. You see `cuda Tesla T4`, `601M of 689M weights train`,
   the score before training, then one line per step (loss, seconds per page, GPU memory).
6. **If Colab disconnects:** *Run all* again; it continues from the last save in `MyDrive/surya/run1/`.
7. **At the end:** the scores before and after each epoch, and the model in `MyDrive/surya/run1/model`.

If you see *CUDA out of memory*, set `MAX_TOKENS = 4096` in the settings cell and *Run all* again.

After training, compare the original and the fine-tuned model on a few real documents and a few `layout`
prompts: the notebook's own check only uses our generated pages.

---

## 11. Not finished yet

- The 100 clean pages, the last 25 layout pages and the 109 table pages wait for the Surya server.
- `block_ocr` cut-outs have 6-24 pixels around the block; Surya's own have 4.
- Whether Surya 2.5 answers the table prompt at all.

---

## 12. Files

| File | What it is |
|---|---|
| `pipeline/surya.py` | our labels in Surya's format: blocks, rules from section 5, the four answers |
| `pipeline/surya_encode.py` | Surya's encoding rules, written from its `compact.py` |
| `pipeline/dataset.py` | makes a batch folder (`--task full_ocr+block_ocr`) |
| `notebooks/Surya_finetune_Colab.ipynb` | the fine-tuning notebook |
| `surya_probe/ask_batch.py` | sends a batch to the Surya server, keeps every answer |
| `surya_probe/export_batch.py` | makes the `samples/<id>/` folders |
| `surya_probe/compare_structure.py` | our labels vs Surya's answers: labels, boxes, order, tags |
| `surya_probe/source_map.py` | each text's HTML element -> what Surya made of it |
| `surya_repo_findings.md` | the details and numbers behind sections 3, 5 and 6 |
