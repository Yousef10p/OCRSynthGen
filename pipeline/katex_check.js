// Keeps the formulas KaTeX can draw: reads text_pool/pools/equations/raw/*.jsonl, writes
// text_pool/pools/equations/latex-00000.jsonl, ... (50,000 lines each). The pages draw equations with KaTeX, so a
// formula it can't draw (an unknown command, a macro of the paper's own) would break the page.
//
//     npm install katex@0.16.11   (or set NODE_PATH to a folder that has it)
//     node pipeline/katex_check.js
const fs = require("fs");
const path = require("path");
const readline = require("readline");
const katex = require("katex");

const dir = path.join(__dirname, "..", "text_pool", "pools", "equations");
const files = fs.readdirSync(path.join(dir, "raw")).filter((f) => f.endsWith(".jsonl")).sort();
let kept = 0, read = 0, out = null;

function write(line) {
  if (kept % 50000 === 0) {
    if (out) out.end();
    out = fs.createWriteStream(path.join(dir, `latex-${String(kept / 50000).padStart(5, "0")}.jsonl`));
  }
  out.write(line + "\n");
  kept++;
}

(async () => {
  for (const f of files) {
    const lines = readline.createInterface({ input: fs.createReadStream(path.join(dir, "raw", f)) });
    for await (const line of lines) {
      read++;
      try {
        katex.renderToString(JSON.parse(line).latex, { displayMode: true, throwOnError: true, strict: "ignore" });
        write(line);
      } catch (e) { /* KaTeX can't draw it */ }
    }
    console.log(`${f}: ${kept} of ${read} kept`);
  }
  if (out) out.end();
})();
