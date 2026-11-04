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
| Encoding | our own `compact()` (drops `</td></th></tr>`) | `compact.encode()` with canonical spelling and attribute order | write our labels as `div` HTML and run Surya's `encode()`, so the spelling is theirs exactly |
| Layout `count` | rounded to 10 | multiple of 50 | round to 50 |
| Block crop margin | 6-24 px | 4 px | use 4 px |
| Labels | 9 | 19 | fine (a subset) |

The 100 Surya answers per prompt in `surya_probe/batch100/` (image, prompt, Surya's answer, our label for each
page) show how close our labels already are.
