# OCRREADME: our training data for Surya OCR 2.5

This file explains, with an example for every point:

- how Surya OCR 2.5 writes a label (read from Surya's own code),
- how Surya really answers (we asked it 100 pages per prompt),
- what our generator writes, and why,
- how to fine-tune Surya on our pages.

The older version of this file (how the generator builds a page, step by step) is in git history at commit
`e3ad56fc4f`.

**Contents**

1. [The goal in one minute](#1-the-goal-in-one-minute)
2. [Surya 2.5 and its four prompts](#2-surya-25-and-its-four-prompts)
3. [Where we learned it from](#3-where-we-learned-it-from)
4. [A label is HTML: one div per block](#4-a-label-is-html-one-div-per-block)
5. [The HTML is shortened into tokens](#5-the-html-is-shortened-into-tokens)
6. [One page label, four answers](#6-one-page-label-four-answers)
7. [What we asked Surya, and where it is saved](#7-what-we-asked-surya-and-where-it-is-saved)
8. [What Surya does, and what we changed to match](#8-what-surya-does-and-what-we-changed-to-match)
9. [What Surya does not do the same way twice](#9-what-surya-does-not-do-the-same-way-twice)
10. [One page, start to end](#10-one-page-start-to-end)
11. [How close we are now](#11-how-close-we-are-now)
12. [Fine-tuning: three traps the notebook avoids](#12-fine-tuning-three-traps-the-notebook-avoids)
13. [How to run the notebook](#13-how-to-run-the-notebook)
14. [Not finished yet](#14-not-finished-yet)
15. [Files](#15-files)

---

## 1. The goal in one minute

We want Surya to **read better** on our kind of pages, but to **answer in the same shape** as before.

So every training answer we make must look like an answer Surya would give itself: the same format, the same
block labels, the same tags, the same kind of boxes. Only the text should be "more correct".

> **Example.** If Surya writes a short line like `History:` as a normal text block, our label must do the same.
> If our label called it a heading, training would teach Surya to call it a heading, and its answers would
> change shape.

---

## 2. Surya 2.5 and its four prompts

Surya OCR 2.5 (`datalab-to/surya-ocr-2.5`) is one model (689 million weights) that answers four fixed
prompts. Surya's code says the prompt texts are *the model's training-time contract*: change a word and the
model may answer differently. We use them word for word.

| Task | What it is shown | What it answers |
|---|---|---|
| `full_ocr` | a whole page | every block: its label, its box, its content |
| `block_ocr` | one block cut out of a page | only that block's content |
| `layout` | a whole page | a list of blocks: label, box, and how long the block's text is |
| `table` | a table cut out | the table's rows, then its columns |

> **Example: the `full_ocr` prompt, exactly**
> ```
> OCR this image to HTML. Each block is a div with data-label and data-bbox (x0 y0 x1 y1, normalized 0-1000).
> ```

Boxes are always `x0 y0 x1 y1`: left, top, right, bottom, as whole numbers from 0 to 1000 (a share of the
image's width and height; the top left corner is `0 0`).

> **Example.** On a page 1590 pixels wide and 2248 high, a box from pixel (687, 764) to (901, 794) is written
> `432 340 567 353`.

---

## 3. Where we learned it from

**Surya's code.** We cloned [datalab-to/surya](https://github.com/datalab-to/surya) (commit 9d5aadf) into
`surya/` (not part of our repository). It is inference code only: no training scripts, no training data. One
file even says the training code was removed:

> `surya/common/order/order_ar.py`: "Vendored from training/models/rtdetr/order_ar.py (training code stripped)"

So we read the label format from what their code reads and writes (sections 4 to 6).

**Surya's answers.** We sent 100 of our pages per prompt to a Surya 2.5 server and kept every answer
(section 7). That shows what Surya *does*, including what the code doesn't say (sections 8 and 9).

**Where Surya's own labels came from** is not in the repository. A hint, not a proof: the `div` format below
is also the output of Datalab's bigger model, Chandra, and parts of Surya's code say "ported from chandra".
Their labels were probably made by Chandra or the same annotation tools.

---

## 4. A label is HTML: one div per block

A page's label is HTML: one `<div>` per block, in reading order. Each `div` has a box and a label; inside
is the block's content as ordinary HTML.

> **Example**
> ```html
> <div data-bbox="71 33 143 45" data-label="Page-Header">Northwind Logistics</div>
> <div data-bbox="120 87 881 255" data-label="Text"><p>The agent is not trading.</p></div>
> <div data-bbox="120 270 881 400" data-label="List-Group"><ul><li>• first point</li><li>• second point</li></ul></div>
> ```

- `data-bbox`: the box, 0 to 1000.
- `data-label`: one of 19 labels (`surya/layout/schema.py`). We use 9 of them:
  `Text`, `Section-Header`, `List-Group`, `Table`, `Image`, `Caption`, `Footnote`, `Page-Header`, `Page-Footer`.
- Inside: `<p>`, `<h1>`-`<h3>`, `<b>`, `<i>`, `<ul><li>`, `<table>`, `<img/>`, `<br/>`.

**Merged table cells** are written as one "parent" cell with its size, then a "ghost" cell for every other
place it covers.

> **Example: a cell two columns wide**
> ```html
> <tr><td data-cs="2" data-kind="parent">Total</td><td data-kind="ghost"></td></tr>
> ```

**Pictures** get a short title and a description.

> **Example**
> ```html
> <img alt="Photo: People standing on a rocky cliff."/><p class="img-desc">A group of people stands atop a rocky cliff under a blue sky.</p>
> ```

---

## 5. The HTML is shortened into tokens

The model does not write the HTML above directly. Surya's `encode()` (`surya/inference/compact.py`) makes it
shorter, and that shorter form is what the model is trained to write.

**Rule 1: a block's `div` becomes one label token and four number tokens.**

> **Example**
> ```
> <div data-bbox="71 33 143 45" data-label="Page-Header">   ->   <|Page-Header|><|71|><|33|><|143|><|45|>
> ```

**Rule 2: no `</div>`, and nothing between blocks.** The next block (or the end) closes the one before.

> **Example: two blocks**
> ```
> <|Text|><|120|><|87|><|881|><|255|><p>The agent is not trading.</p><|Text|><|120|><|270|><|881|><|300|><p>Next.</p>
> ```

**Rule 3: in tables, `</td>`, `</th>` and `</tr>` are left out** where the next tag makes them clear.

> **Example**
> ```
> <table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>   ->   <table><tr><td>a<td>b<tr><td>c<td>d</table>
> ```

**Rule 4: every tag has one spelling.** Lower case, double quotes, empty tags as `<br/>` and `<img/>`, and
attributes in a fixed order (`type, checked, value, colspan, rowspan, data-cs, data-rs, data-kind, ...`).

> **Example**
> ```
> <TD data-kind='parent' data-cs=2>   ->   <td data-cs="2" data-kind="parent">
> <br>                               ->   <br/>
> ```

**Rule 5: the answer ends with the end token** `<|im_end|>` (token number 2). Writing it is how the model
stops.

We don't rewrite these rules: `pipeline/surya_compact.py` is Surya's own `compact.py`, copied unchanged
(Apache-2.0, commit 9d5aadf), and every target is written by its `encode()`. Checks: Surya's 100 answers turned
back into `<div>` HTML and encoded again come out unchanged (100 of 100), and our earlier targets are identical
to what `encode()` writes (200 of 200).

---

## 6. One page label, four answers

The same page label gives the answer to each prompt.

**`full_ocr`: the whole label, shortened (section 5).**

**`block_ocr`: the content of one block, without its label and box.** Surya cuts the block out with 4 pixels
around it (`_crop_block` in `surya/recognition/`). Pictures are never sent one by one.

> **Example: the cut-out shows one paragraph; the answer is only**
> ```
> <p>يتمثل دورهم أيضاً في.</p>
> ```

**`layout`: one JSON line**, with every block's label, box and `count` (how many tokens the block's text will
take, rounded to tens).

> **Example (a real Surya answer, shortened)**
> ```json
> [{"label": "Page-Header", "bbox": "761 95 841 117", "count": 20}, {"label": "Image", "bbox": "196 310 811 387", "count": 40}]
> ```

**`table`: rows, then columns, as JSON.**

> **Example (made up to show the form)**
> ```json
> [{"label": "Row", "bbox": "0 0 1000 120"}, {"label": "Row", "bbox": "0 120 1000 260"}, {"label": "Col", "bbox": "0 0 480 1000"}]
> ```
> This is Surya 2's table format. Surya 2.5's model card says rows and columns need a Surya 2 model, and we
> have not seen 2.5 answer it yet.

Other limits from the code: images are made smaller to fit 3072 x 2048 pixels; the longest answers are
12,288 tokens (full page), 8,192 (one block) and 3,072 (layout).

---

## 7. What we asked Surya, and where it is saved

We sent pages to a Surya 2.5 server, one at a time, always the same way (temperature 0).

| Folder | What | Answered |
|---|---|---|
| `surya_probe/batch100/full_ocr/` | 100 pages (with scan and paper effects) | 100 |
| `surya_probe/batch100/block_ocr/` | 100 cut-out blocks | 100 |
| `surya_probe/batch100/layout/` | 100 pages | 75 (the server went down) |
| `surya_probe/batch100/table/` | 109 tables | 0 (the server went down) |
| `surya_probe/clean100/` | 100 pages + 100 blocks with no effects at all | waiting for the server |

Every answered page has its own folder:

> **Example: `surya_probe/batch100/full_ocr/samples/0000004/`**
> ```
> image.jpg          what Surya was shown
> prompt.txt         the prompt
> surya_output.txt   Surya's answer, exactly as it came back
> our_label.txt      our label for the same image
> info.json          time, tokens used, how the answer ended
> ```

(`our_label` there was made before the changes in section 8.)

---

## 8. What Surya does, and what we changed to match

We matched every block of ours to Surya's block on the same image and counted what Surya did. Where Surya
does one thing most of the time, our labels now do the same (`pipeline/surya.py`).

**8.1 A short label line with nothing beside it is text, not a heading** (74 of 83).

> **Example**
> ```
> ours before:  <|Section-Header|>...<h4>History:</h4>
> Surya:        <|Text|>...<p>History:</p>
> ours now:     <|Text|>...<p>History:</p>
> ```

If that line is **bold** on the page, Surya calls it a heading, most often `<h3>`. Ours now does too.

> **Example (bold "Career:")**
> ```
> ours now:     <|Section-Header|>...<h3>Career:</h3>
> ```

**8.2 "Label: value" with only the label bold: only the label is in `<b>`.**

> **Example**
> ```
> ours before:  <p><b>Charts: 650542</b></p>
> Surya:        <p><b>Charts:</b> 650542</p>
> ours now:     <p><b>Charts:</b> 650542</p>
> ```

**8.3 A list right under another list is one list** (45 of 67).

> **Example: list items placed one under the other**
> ```
> ours before:  <|List-Group|>...<ul><li>◦ Pamela K.</li></ul><|List-Group|>...<ul><li>◦ When Hamamoto was sworn in by U.S.</li></ul>
> Surya:        <|List-Group|>...<ul style="list-style-type: none;"><li>○ Pamela K.</li><li>○ When Hamamoto was sworn in by U.S.</li>...</ul>
> ours now:     one List-Group with both items
> ```

**8.4 Pieces of text on one line that almost touch are one block** (20 of 23).

> **Example (made up to show the idea)**
> ```
> ours before:  <|Text|>...<p>Population:</p><|Text|>...<p>12,000</p>
> ours now:     <|Text|>...<p>Population: 12,000</p>
> ```

**8.5 Surya's boxes start a little further left and higher** than the text: the left edge by about 0.65% of
the page width, the top by about 0.35% of the height. The right and bottom edges are where ours are. Same on
clean pages, so it is how Surya draws boxes, not noise. Ours now do the same.

> **Example**
> ```
> ours before:  <|Text|><|432|><|340|><|567|><|353|>
> Surya:        <|Text|><|425|><|334|><|571|><|354|>
> ours now:     <|Text|><|425|><|337|><|567|><|353|>
> ```
> Average overlap with Surya's boxes: 0.79 before, 0.88 now.

**8.6 Layout `count` is the block's length in Surya's own tokens, rounded up to tens.** Surya's real counts
are multiples of 10 (784 of 837 are not multiples of 50). Rounding up matches Surya on 63% of blocks, the
best of every rule we tried.

> **Example**
> ```
> <p>MAD (Michigan.</p>  ->  11 tokens  ->  rounded up: count 20
> ```

**8.7 Two-column English pages are read one column at a time.**

> **Example**
> ```
> ours before:  left column line 1, right column line 1, left column line 2, ...
> Surya:        all of the left column, then all of the right column
> ours now:     all of the left column, then all of the right column
> ```

**8.8 A cut-out block gets its content only: never a label or a box** (100 of 100).

> **Example**
> ```
> Surya:  <p>يتمثل دورهم أيضاً في.</p>
> never:  <|Text|><|0|><|0|><|1000|><|1000|><p>...</p>
> ```

---

## 9. What Surya does not do the same way twice

Here Surya's answers follow no pattern. Any fixed rule matches it only about half the time, so we keep one
simple choice. Training will make Surya a little more consistent here; the answer format does not change.

**9.1 Heading levels.** Similar headings get `h2` one time and `h3` the next. No rule matched more than 59%.
We keep the page's own level.

> **Example: three headings written the same way (`<h2>`) on one page (page 0000003)**
> ```
> Surya:  <h2><b>Agriculture</b></h2>   <h3><b>Wilmot baronets, of Witney</b></h3>   <h3><b>Horneytown, North Carolina</b></h3>
> ```

**9.2 Bold inside headings.** Surya adds `<b>` to 78 of 285 headings, without a pattern. We add none.

> **Example**
> ```
> Surya, page 0000003:  <h2><b>Education Institutions</b></h2>
> Surya, page 0000001:  <h2>History</h2>
> ours:                 <h2>Education Institutions</h2>, <h2>History</h2>
> ```

**9.3 Two paragraphs one under the other.** Joined with `<br/>` about half the time. We keep them apart.

> **Example**
> ```
> Surya:  <p>KSRTC bus facility available.<br/>The nearest railway.</p>
> ours:   <p>KSRTC bus facility available.</p>  and  <p>The nearest railway.</p>  (two blocks)
> ```

**9.4 Groups of numbers on Arabic pages.** Reversed 8 times, kept 14 times. We keep the order of the text.

> **Example**
> ```
> ours:   ٠٥١ ٥٦٢ ٨٢٤٧
> Surya:  ٨٢٤٧ ٥٦٢ ٠٥١   (sometimes)
> ```

**9.5 A grid of "label: value" pairs.** Sometimes one `<table>`, sometimes separate text lines. We write text
lines.

> **Example**
> ```
> Surya, sometimes:  <table><tr><td>الوصف:<td>٧١٢٥٥٦<tr><td>الخصائص المدارية:<td>٢٢٤ ٠٥٩ ...</table>
> ours:              <p><b>الوصف:</b> ٧١٢٥٥٦</p> ...
> ```

**9.6 Reading order on other pages.** Surya decides it with a trained model (`common/order/order_ar.py`),
not a rule; once it read three Arabic columns from left to right. We keep the page's own order.

**9.7 Layout answers are not always JSON.** 18 of 75 start as JSON and switch to the full-page form halfway,
because that server does not force JSON. We always write JSON.

> **Example: a broken Surya layout answer**
> ```
> {"label": "Section-Header", "bbox": "788<|267|><|837|><|283|>", "count": 10}, ...
> ```

**9.8 Text characters.** Identical on 844 of 1013 blocks; most others differ by a character or two (for
example Surya writes the Arabic comma `،` where the page shows `,`). We write what the page shows.

---

## 10. One page, start to end

Page `full_ocr/0000004`: an English page with 6 blocks.

**Our label, as HTML:**

```html
<div data-bbox="425 337 567 353" data-label="Text"><p>MAD, MAD/I, and:</p></div>
<div data-bbox="65 709 229 730" data-label="Text"><p>MAD (Michigan.</p></div>
<div data-bbox="0 745 300 821" data-label="List-Group"><ul><li>▪ PRINT FORMAT HELLOW</li><li>▪ END OF PROGRAM</li><li>▪ P'T HELLOW</li></ul></div>
<div data-bbox="572 742 721 763" data-label="Text"><p>Developed in.</p></div>
<div data-bbox="602 782 690 803" data-label="Text"><p>History:</p></div>
<div data-bbox="557 891 736 907" data-label="Text"><p>"Hello, world" example:</p></div>
```

**Our training answer (shortened, section 5):**

```
<|Text|><|425|><|337|><|567|><|353|><p>MAD, MAD/I, and:</p><|Text|><|65|><|709|><|229|><|730|><p>MAD (Michigan.</p><|List-Group|><|0|><|745|><|300|><|821|><ul><li>▪ PRINT FORMAT HELLOW</li><li>▪ END OF PROGRAM</li><li>▪ P'T HELLOW</li></ul><|Text|><|572|><|742|><|721|><|763|><p>Developed in.</p><|Text|><|602|><|782|><|690|><|803|><p>History:</p><|Text|><|557|><|891|><|736|><|907|><p>"Hello, world" example:</p>
```

**Surya's real answer for the same image** (the server shows the end token `<|im_end|>` as `<|im_end!>`):

```
<|Text|><|425|><|334|><|571|><|354|><p>MAD, MAD/I, and:</p><|Text|><|64|><|708|><|230|><|731|><p>MAD (Michigan.</p><|List-Group|><|20|><|748|><|300|><|817|><ul><li>■ PRINT FORMAT HELLOW</li><li>■ END OF PROGRAM</li><li>■ P'T HELLOW</li></ul><|Text|><|570|><|739|><|722|><|763|><p>Developed in.</p><|Text|><|600|><|780|><|693|><|804|><p>History:</p><|Text|><|553|><|888|><|740|><|909|><p>"Hello, world" example:</p><|im_end!>
```

Same 6 blocks, same order, same labels, same tags, boxes a few thousandths apart.

---

## 11. How close we are now

Pages made with the current code, compared with Surya's answers on the same images:

| Check | full_ocr (78 pages) | block_ocr (64 blocks) | layout (60 pages) |
|---|---|---|---|
| Answer format correct | 100 of 100 | 100 of 100 | |
| Same as Surya's own encoding | 100 of 100 | 100 of 100 | |
| Same block label | 852 of 931 (92%) | | 537 of 582 (92%) |
| Same tags in the block | 606 of 931 (65%) | 38 of 64 | |
| Average box overlap | 0.88 (was 0.79) | | |
| Same `count` | | | 62% |
| Label or box added to a cut-out | | never | |

Most of what is left is section 9. A perfect score is not possible: these are Surya's *answers*, with its own
mistakes and changes of mind, not the labels it was trained on (those are private).

---

## 12. Fine-tuning: three traps the notebook avoids

`notebooks/Surya_finetune_Colab.ipynb` trains Surya on `full_ocr` and `block_ocr` pages (they produce all of
Surya 2.5's text; tables come out as HTML inside them). It trains every weight except the vision encoder
(601M of 689M), with 8-bit AdamW, learning rate 1e-5, 2 epochs, and checks 8 validation pages before training
and after every epoch. Each trap below was found by testing.

**Trap 1: the tokenizer.** Loaded the default way, transformers gives Surya a Qwen tokenizer that breaks
Arabic letters into bytes, and the model then answers cut-outs with nothing. The notebook loads it the way
Surya stores it.

> **Example: `<p>مرحبا</p>`**
> ```
> default (Qwen):      <p>  �  �  �  �  �  �  �  �  �  �  </p>     12 tokens
> as Surya stores it:  <p>  م  ر  ح  ب  ا  </p>                     7 tokens
> ```

**Trap 2: the end token.** Surya's vocabulary has about 25,000 extra tokens such as `.<` and `><`, and the
longest match wins. If the whole chat is turned into tokens as one text, the answer's last character takes
the `<` of `<|im_end|>`, and the end token falls apart into plain pieces (on 38 of 66 pages). Surya would
learn to *type* "im_end" instead of stopping. The notebook adds the answer's own tokens, then the end token
by its number (2).

> **Example: `<p>The end.</p><|im_end|>`**
> ```
> as one text:  ...  p  ><  |  im  _  end  |  >        the end token is gone
> notebook:     ...  </p>  <|im_end|>                   the real end token
> ```

**Trap 3: picture descriptions.** Surya describes each picture. Ours are fixed sentences per kind of picture.
If Surya learned them, it would describe every picture the same way. So their words are not trained; the
tags around them are.

> **Example: the parts in [brackets] are not trained**
> ```
> <img alt="Photo: [A general scene.]"/><p class="img-desc">[A photograph of a general scene.]</p>
> ```

Tested on a computer without a GPU, with the same library versions as Colab (transformers 5.22.1,
bitsandbytes 0.50.2): training steps, validation, saving and continuing after a stop. Not yet run on a T4.

---

## 13. How to run the notebook

1. **Make the pages on your PC**, one folder per epoch, a different seed each (in the project folder):
   ```bash
   ..\arabic-template-bank\.venv-pipeline\Scripts\python -m pipeline.dataset --n 1000 --out batches\epoch_001 --task full_ocr+block_ocr --tashkeel none --seed 1
   ```
   ```bash
   ..\arabic-template-bank\.venv-pipeline\Scripts\python -m pipeline.dataset --n 1000 --out batches\epoch_002 --task full_ocr+block_ocr --tashkeel none --seed 2
   ```
   About 15-25 minutes each. 5% of the pages in each folder are kept for validation.

   **Or make them in Colab instead:** `notebooks/Generate_batches_Colab.ipynb` sets up the generator in Colab
   and writes the folders straight to Drive (step 2 is then done for you). It needs a read-only GitHub token
   in Colab's *Secrets* as `GITHUB_TOKEN`; the notebook's first cell says how.
2. **Upload both folders to Google Drive**, whole, as `MyDrive/surya/batches/epoch_001` and `epoch_002`.
3. **Open the notebook in Colab:** *File -> Upload notebook* -> `notebooks/Surya_finetune_Colab.ipynb`.
4. **Choose the GPU:** *Runtime -> Change runtime type -> T4 GPU*.
5. **Run:** *Runtime -> Run all*, and allow Drive access. You will see `cuda Tesla T4`,
   `601M of 689M weights train`, the score before training, then one line per step in this form:
   ```
   epoch 1 step 12/250: loss ..., ... s/page, GPU ... GB
   ```
6. **If Colab disconnects:** *Run all* again. It continues from the last save in `MyDrive/surya/run1/`.
7. **At the end:** the score before training and after each epoch, and the model in `MyDrive/surya/run1/model`.

If you see *CUDA out of memory*, set `MAX_TOKENS = 4096` in the settings cell and *Run all* again.

After training, compare the original and the trained model on a few real documents and a few `layout`
prompts. The notebook's own check only uses our generated pages.

---

## 14. Not finished yet

- The 100 clean pages, the last 25 layout pages and the 109 table pages are waiting for the Surya server.
- Our `block_ocr` cut-outs have 6-24 pixels around the block; Surya's own have 4.
- Whether Surya 2.5 answers the table prompt at all.

---

## 15. Files

| File | What it is |
|---|---|
| `pipeline/surya.py` | our labels in Surya's format, with the rules of section 8 |
| `pipeline/surya_compact.py` | Surya's own `compact.py` (section 5), copied unchanged: `encode()` writes every target |
| `notebooks/Surya_finetune_swift_Colab.ipynb` | the same fine-tuning with ms-swift |
| `pipeline/dataset.py` | makes a batch folder (`--task full_ocr+block_ocr`) |
| `notebooks/Surya_finetune_Colab.ipynb` | the fine-tuning notebook |
| `notebooks/Generate_batches_Colab.ipynb` | makes the batch folders in Colab, into Drive |
| `synthdoc_arabic_ext/` | SynthDoc's Arabic add-on (a copy of `synthdoc-arabic-test/arabic_ext`), so Colab gets it from the repository |
| `surya_probe/ask_batch.py` | sends a batch to the Surya server and keeps every answer |
| `surya_probe/export_batch.py` | makes the `samples/<id>/` folders |
| `surya_probe/compare_structure.py` | our labels vs Surya's answers: labels, boxes, order, tags |
| `surya_probe/source_map.py` | each text's HTML element, and what Surya made of it |
