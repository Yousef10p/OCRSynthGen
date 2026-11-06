// Reads a JSON array of CSS strings on stdin, flips each to RTL with rtlcss, prints a JSON array.
// A string rtlcss cannot parse comes back as null.
const rtlcss = require("rtlcss");

let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (d) => (input += d));
process.stdin.on("end", () => {
  const out = JSON.parse(input).map((css) => {
    try {
      return rtlcss.process(css);
    } catch (e) {
      return null;
    }
  });
  process.stdout.write(JSON.stringify(out));
});
