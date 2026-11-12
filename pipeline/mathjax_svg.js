// Draws LaTeX as SVG with MathJax, for the Equation-Block pictures (pipeline/content_pools.py runs it once and
// keeps it open). One JSON line in, {"latex": "..."}; one JSON line out, {"svg", "width_ex", "height_ex", "error"}.
//
//     npm install            (mathjax-full is in package.json)
const readline = require("readline");
const { mathjax } = require("mathjax-full/js/mathjax.js");
const { TeX } = require("mathjax-full/js/input/tex.js");
const { SVG } = require("mathjax-full/js/output/svg.js");
const { liteAdaptor } = require("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = require("mathjax-full/js/handlers/html.js");
const { AllPackages } = require("mathjax-full/js/input/tex/AllPackages.js");

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const doc = mathjax.document("", {
  InputJax: new TeX({ packages: AllPackages.filter((p) => p !== "bussproofs") }),
  OutputJax: new SVG({ fontCache: "none" }),  // every glyph drawn in the file: it stands alone
});

readline.createInterface({ input: process.stdin }).on("line", (line) => {
  let out;
  try {
    const node = doc.convert(JSON.parse(line).latex, { display: true });
    const svg = adaptor.innerHTML(node);
    const size = (name) => parseFloat((svg.match(new RegExp(`${name}="([\\d.]+)ex"`)) || [0, 0])[1]);
    out = { svg, width_ex: size("width"), height_ex: size("height"),
            error: /data-mjx-error|merror|fill="red"/.test(svg) };  // an unknown command is drawn in red
  } catch (e) {
    out = { error: true, message: String(e).slice(0, 200) };
  }
  process.stdout.write(JSON.stringify(out) + "\n");
});
