# How Surya 2.5's training labels are made: what Surya's own code shows

Source: a clone of [datalab-to/surya](https://github.com/datalab-to/surya) (commit 9d5aadf, `surya/` next to this file).
The repository is inference code only: no training scripts, no datasets. One file says where the training code
lives: `surya/common/order/order_ar.py` was "vendored from training/models/rtdetr/order_ar.py (training code
stripped)", a private `training/` tree. So how they labelled is read from the format their code reads and writes.

## 1. A label is HTML: one div per block

Every page's ground truth is HTML, one block after another, each a `div` with a box and a label:

```html
<div data-bbox="71 33 143 45" data-label="Page-Header">Northwind Logistics</div>
<div data-bbox="120 87 881 255" data-label="Text"><p>agent is not trading. If <math>\rho</math> ...</p></div>
```

- `data-bbox`: `x0 y0 x1 y1`, integers 0-1000 of the page (`MAX_COORD = 1000`).
- `data-label`: one of 19 block labels (`compact.BLOCK_LABELS`): Caption, Footnote, Equation-Block, List-Group,
  Page-Header, Page-Footer, Image, Section-Header, Table, Text, Complex-Block, Code-Block, Form,
  Table-Of-Contents, Figure, Chemical-Block, Diagram, Bibliography, Blank-Page (and Picture). We use 9 of them.
- Inside: plain HTML. Math is `<math>...</math>` (LaTeX inside), tables are HTML (`<thead>`/`<tbody>` and
  `border="1"` occur), merged cells are a `data-kind="parent"` cell with `data-cs` / `data-rs` followed by
  `data-kind="ghost"` cells (`merged_cells.py`, `tests/test_compact_decode.py`).

This is the format their parsers read (`parsers.parse_full_page_html`, `test_div_format_unchanged`) and what
`compact.decode()` turns the model's answer back into.

## 2. The training target is that HTML, encoded

`surya/inference/compact.py` says the model's output is the HTML above, mechanically shortened by `encode()`:

| Rule | Example |
|---|---|
| A block `div` becomes one label token + four coordinate tokens, and has no `</div>` | `<|Page-Header|><|71|><|33|><|143|><|45|>` |
| `</td>`, `</th>`, `</tr>` are left out | `<table><tr><td>a<td>b<tr><td>c<td>d</table>` |
| A `data-bbox` on any other tag (row box, nested image) becomes `<|box|>` + four coordinates before that tag | `<table><|box|><|0|><|0|><|100|><|50|><tr>...` |
| Newlines next to block-level tags are dropped; text in `<pre>`, `<code>`, `<math>` is kept byte for byte | |
| Every tag in one spelling (`canonical_tag`): lower case, double quotes, void tags as `<br/>` `<img/>` `<hr/>` `<input/>`, `checked` without a value | |
| Attributes in a fixed order (`_ATTR_ORDER`): type, checked, value, colspan, rowspan, data-cs, data-rs, data-kind, display, class, align, style, border, href, alt, text; any other attribute after these, sorted | `<th data-rs="2" data-kind="parent">` |

So their pipeline was: **HTML label -> `canonicalize` -> `encode` -> training target**. The prompt texts are
"the model's training-time contract - do not paraphrase without retraining" (`prompts.py`).

## 3. One label, four prompts

The same page label serves every prompt (`prompts.py`, `recognition/`, `layout/`):

| Prompt | Target made from the page label |
|---|---|
| full page (`HIGH_ACCURACY_BBOX_PROMPT`) | the whole page, encoded (section 2) |
| layout (`LAYOUT_PROMPT`) | one-line JSON `[{"label", "bbox": "x0 y0 x1 y1", "count"}]` per block; `count` = the model's token estimate of the block's OCR answer, **a multiple of 50** (`layout/schema.py`, `parsers.py`) |
| block (`BLOCK_PROMPT`) | one block cut out of the page with its box padded by **4 px** (`recognition._crop_block`); the answer is the block's inner HTML. Image, Figure, Diagram and Blank-Page blocks are never sent (`SKIP_OCR_LABELS`) |
| table rows/cols (`TABLE_REC_PROMPT`) | JSON rows then columns; needs a Surya 2 checkpoint (README limitations) |

Reading order is the order of the blocks in the label; Surya also has a separate RF-DETR layout detector with an
autoregressive reading-order head over the same 19-class taxonomy (`common/order/`, `fast_layout/`).

## 4. Where the HTML labels came from

Not in the repository. What the code suggests (not confirmed): the `div data-bbox data-label` HTML is the output
format of Datalab's larger model, Chandra, and parts of Surya's recognition code are "chandra-style" or "ported
from chandra". So the page labels were most likely produced by Chandra or the same annotation pipeline, then
checked and encoded. The training data itself was never published.

## 5. What this means for our generator

| | Ours now | Surya's contract | To do |
|---|---|---|---|
| Page label | blocks with label, box, HTML (`surya.blocks`) | the same | none |
| Encoding | our own `compact()` (drops `</td></th></tr>`) | `compact.encode()` with canonical spelling and attribute order | done: our targets equal `encode()` of our div HTML (`pipeline/surya_encode.py`) on 100 of 100 pages |
| Layout `count` | Surya's own tokens, rounded up to 10 | a token estimate; Surya's real answers are multiples of 10 (784 of 837 not multiples of 50) | done |
| Block crop margin | 6-24 px | 4 px | use 4 px |
| Labels | 9 | 19 | fine (a subset) |

## 6. Matching Surya's real answers (surya_probe/batch100)

Surya answered 100 pages per prompt (full_ocr, block_ocr; layout 75 so far). Where Surya is consistent, our
labels now follow it (`pipeline/surya.py`); where it is not, we keep our rule, since no rule matches it better.

| What | Surya's answers | Our rule now |
|---|---|---|
| A label line with no value, not bold | Text `<p>` on 74 of 83 | Text `<p>` (was Section-Header `<h4>`) |
| A label line with no value, bold | Section-Header, most often `<h3>` | Section-Header `<h3>` (was bare `<b>`) |
| A list right under another list | one List-Group on 45 of 67 | joined when less than 0.8 of a line apart |
| Text on one line almost touching other text | one Text on 20 of 23 | joined when less than 1% of the page width apart |
| Layout `count` | equals our tokens rounded up to 10 on 443 of 698 | Surya's tokenizer, rounded up to 10 |
| Two-column English DocLayNet pages | read column by column | column by column (Arabic: bands, right to left) |
| Heading level h1/h2/h3 | inconsistent: no rule above 59% | the page's own heading level |
| `<b>` inside a heading | 78 of 285, no pattern | none |
| Two paragraphs one under the other | joined about half the time, no pattern | kept apart |
| Reading order elsewhere | from a trained order model (`common/order/order_ar.py`), not a rule | document order |
| A label with a value beside it, only the label bold | `<p><b>label:</b> value</p>` | the same (was the whole line in `<b>`) |
| Box edges | left edge 6.5, top edge 3.5 thousandths of the page further out than the text (pictures 3.3 and 1.8), on clean pages too | the same (median overlap with Surya's boxes 0.79 -> 0.90) |
| Picture descriptions | describes each picture | ours are stock sentences, so the fine-tuning notebook leaves their words out of the loss |
| Text characters | identical on 844 of 1013 blocks; number groups on Arabic pages reversed 8 times, kept 14 | unchanged (no consistent rule) |
| Layout answer form | 18 of 75 switch from JSON to full_ocr blocks halfway (the server has no guided decoding) | JSON |

Agreement on the same pages, before -> after: full_ocr same label 1012/1142 -> 1063/1150, same tags 716 -> 771,
reading-order inversions 520 -> 412; layout same label 627/706 -> 657/709, same count 368 -> 442; block_ocr same
tags 51 -> 56 of 100.

## 7. The end token in training

Surya's vocabulary has about 25,000 added tokens such as `.<`, `><` and `;<`, matched longest first. Tokenizing a
whole chat as text therefore joins the answer's last character with the `<` of `<|im_end|>` on most pages (38 of 66
checked), and the end token becomes the plain pieces `| im _ end | >`: a model trained that way learns to write
those instead of stopping. The fine-tuning notebook builds the answer from the target's own tokens followed by the
end token's id (2), the way the model writes it.
