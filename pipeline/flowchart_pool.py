"""A pool of drawn flowcharts, each with the answer Surya gives for it (not yet used by the generator).

    .venv-pipeline\\Scripts\\python -m pipeline.flowchart_pool --n 50000                       (the first 50,000)
    .venv-pipeline\\Scripts\\python -m pipeline.flowchart_pool --n 150000 --start 50000 --seed 1 (150,000 more, added)

Words: the English steps and yes/no questions of the LLM processes (text_pool/pools/vocab/flowcharts*.jsonl), a few
steps of one process in their order. Shape: a chain from a start to an end, with up to three questions whose "no"
goes to a side step that loops back, a side step that rejoins further on, or a second end. Drawing (matplotlib),
picked per chart (`look()`): direction (top-down, left-right, and rarely bottom-up or right-left), one of the 100
readable fonts (font_pool/fonts-more.json), size, colours, a colour per step, box and terminal shapes (stadium,
ellipse, circle, rounded), numbered steps, capitals, a title above, a hand-drawn look, dashed arrows, shadows,
boxed answer words, grid paper; a box's words are wrapped and the box sized to them.

Answer, as Surya writes a flowchart (pools.diagram / diagram_html, Surya's 27 flowcharts): <img alt="Flowchart:
...">, a title drawn above it in <p class="img-text">, the flow as Mermaid (graph TD/LR/BT/RL; a node's shape and
words where it first appears; a box's wrapped lines joined by spaces; edge words as -- Yes -->), <p class="img-desc">.
Out: img_pool/flowcharts/<n // 1000>/<n>.png and img_pool/flowcharts/flowcharts.jsonl ({file, label, lang, html,
mermaid, ...}); --start adds to them."""
import argparse
import json
import random
import re
import textwrap
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from pipeline.pools import home, pool_file  # noqa: E402  (where the pools are: dataset_pool/ or their first homes)
OUT = home("images") / "flowcharts"
VOCAB = home("vocab")
PALETTES = [  # (step fill, terminal fill, question fill, line colour)
    ("#ffffff", "#ffffff", "#ffffff", "#222222"), ("#e8f0fe", "#c6dafc", "#fde8c8", "#1a3d7c"),
    ("#e9f7ef", "#b9e4c9", "#fff3c4", "#1e5631"), ("#f3e8fd", "#dcc4f7", "#fde2e4", "#4a2475"),
    ("#fef3e2", "#fbd9a5", "#dbeafe", "#7a3e00"), ("#f1f5f9", "#cbd5e1", "#fef9c3", "#334155"),
    ("#ffffff", "#222222", "#ffffff", "#222222"), ("#e0f7fa", "#80deea", "#ffe0b2", "#006064"),
    ("#fce4ec", "#f8bbd0", "#e1f5fe", "#880e4f"), ("#fafafa", "#90caf9", "#ffcc80", "#424242"),
    ("#fff8e1", "#ffd54f", "#c8e6c9", "#5d4037"), ("#ede7f6", "#9575cd", "#ffe082", "#311b92"),
    ("#e3f2fd", "#1565c0", "#fff59d", "#0d47a1"), ("#f9fbe7", "#c0ca33", "#ffccbc", "#33691e"),
    ("#ffffff", "#e0e0e0", "#f5f5f5", "#000000"), ("#eceff1", "#607d8b", "#ffe0b2", "#263238")]
STEP_COLOURS = ["#ffcdd2", "#f8bbd0", "#e1bee7", "#c5cae9", "#bbdefb", "#b2ebf2", "#c8e6c9", "#f0f4c3", "#ffe0b2",
                "#d7ccc8", "#cfd8dc", "#b2dfdb"]
ANSWERS = [("Yes", "No"), ("Yes", "No"), ("Yes", "No"), ("Y", "N"), ("True", "False"), ("YES", "NO"),
           ("Pass", "Fail"), ("Approved", "Rejected"), ("OK", "Not OK"), ("Valid", "Invalid")]
STARTS, ENDS = ["Start", "Start", "Begin", "START", "Start process"], ["End", "End", "Finish", "Stop", "Done", "END", "Complete"]
SECOND_ENDS = ["Reject", "Cancel", "Stop", "Close request", "End", "Abort", "Exit", "Return to applicant"]


def clean(words):
    """LLM words in plain characters (no-break hyphens, curly quotes...), as a font draws them."""
    words = unicodedata.normalize("NFKC", words).translate({0x2010: "-", 0x2011: "-", 0x2013: "-", 0x2014: "-",
                                                             0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"', 0xA0: " "})
    return re.sub(r"\s+", " ", words).strip()


_cmaps = {}


def covers(font_file, text):
    if font_file not in _cmaps:
        from fontTools.ttLib import TTFont
        _cmaps[font_file] = set(TTFont(font_file, lazy=True).getBestCmap() or {})
    return all(ord(c) in _cmaps[font_file] for c in text if not c.isspace())


def line_height(font_file):
    """A text line's height in em, as matplotlib spaces lines (the font's ascent + descent)."""
    from fontTools.ttLib import TTFont
    f = TTFont(font_file, lazy=True)
    return (f["hhea"].ascent - f["hhea"].descent) / f["head"].unitsPerEm


def processes():
    out = []
    for path in sorted(VOCAB.glob("flowcharts*.jsonl")):
        for line in open(path, encoding="utf-8"):
            p = json.loads(line)
            out.append((clean(p["title"]["en"]), [clean(s["en"]) for s in p["steps"]], [clean(q["en"]) for q in p["questions"]]))
    return out


def fonts():
    entries = json.loads((home("fonts") / "fonts-more.json").read_text(encoding="utf-8"))
    return [{w: str(pool_file(p, "fonts")) for w, p in f["files"].items()} for f in entries if "latin" in f["scripts"]]


def look(rng):
    """Everything about how one chart is drawn."""
    across = rng.random() < 0.3
    return {"across": across, "flip": rng.random() < 0.12,  # bottom-up / right-left
            "numbered": rng.random() < 0.15, "upper": rng.random() < 0.08, "title": rng.random() < 0.25,
            "sketch": rng.random() < 0.08, "dashed": rng.random() < 0.1, "shadow": rng.random() < 0.15,
            "rainbow": rng.random() < 0.15, "boxed_answers": rng.random() < 0.2, "grid": rng.random() < 0.06,
            "end_style": rng.choice(["stadium", "stadium", "ellipse", "round", "circle"]),
            "step_style": rng.choice(["square", "round", "round"]), "size": rng.uniform(8, 12),
            "width": rng.randint(10, 14) if across else rng.randint(12, 22), "palette": rng.choice(PALETTES),
            "lw": rng.uniform(0.7, 2.2), "head": rng.choice(["-|>", "->", "-|>", "-|>,head_width=0.25,head_length=0.6", "-|>,head_width=0.5,head_length=0.7"]),
            "gap": rng.uniform(2.0, 3.6), "dpi": rng.choice([150, 180, 200]),
            "paper": rng.choice(["#ffffff", "#ffffff", "#ffffff", "#fcfcf7", "#f7f7f7", "#fffdf5"])}


def plan(rng, process, lk):
    """Nodes [(id, kind, words)] and edges [(a, b, words)]: the main chain, then the side nodes."""
    title, steps, questions = process
    k = rng.randint(3, min(5 if lk["across"] else 11, len(steps)))
    first = rng.randint(0, len(steps) - k)
    chain = [("start", rng.choice(STARTS))] if rng.random() < 0.9 else []
    chain += [("step", s.rstrip(".")) for s in steps[first:first + k]]
    n_q = rng.choice([0, 1, 1, 2] if lk["across"] else [0, 1, 1, 2, 2, 3])
    qs = rng.sample(questions, min(n_q, len(questions)))
    spots = sorted(rng.sample(range(2, len(chain)), min(len(qs), max(0, len(chain) - 2))), reverse=True)
    for spot, q in zip(spots, qs):
        chain.insert(spot, ("ask", q if q.endswith("?") else q + "?"))
    chain.append(("end", rng.choice(ENDS)))
    if lk["numbered"]:
        n = 0
        for j, (kind, words) in enumerate(chain):
            if kind == "step":
                n += 1
                chain[j] = (kind, f"{n}. {words}")
    extra = [s.rstrip(".") for s in steps if s not in steps[first:first + k]]
    nodes = [(chr(65 + i), kind, words) for i, (kind, words) in enumerate(chain)]
    yes, no = rng.choice(ANSWERS)
    edges = [(nodes[i][0], nodes[i + 1][0], yes if nodes[i][1] == "ask" else None) for i in range(len(nodes) - 1)]
    side = []
    for i, (nid, kind, _) in enumerate(nodes):
        if kind != "ask":
            continue
        sid = chr(65 + len(nodes) + len(side))
        how = rng.choice(["loop", "loop", "rejoin", "end"]) if extra else "end"
        if how == "end":
            side.append((sid, "end", rng.choice(SECOND_ENDS), nid, None))
        else:
            words = extra.pop(rng.randrange(len(extra)))
            back = [n[0] for n in nodes[max(0, i - 3):i] if n[1] == "step"]
            target = rng.choice(back) if how == "loop" and back else nodes[min(len(nodes) - 1, i + 2)][0]
            side.append((sid, "step", words, nid, target))
        edges.append((nid, sid, no))
        if side[-1][4]:
            edges.append((sid, side[-1][4], None))
    if lk["upper"]:
        nodes = [(a, b, w.upper()) for a, b, w in nodes]
        side = [(a, b, w.upper(), c, d) for a, b, w, c, d in side]
    return title, nodes, side, edges, (yes, no)


def draw(rng, title, nodes, side, edges, answers, font, path, lk):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon
    from matplotlib.textpath import TextPath
    across, size = lk["across"], lk["size"]
    bold = rng.random() < 0.2 and "700" in font
    prop = FontProperties(fname=font["700" if bold else "400"], size=size)
    small = FontProperties(fname=font["400"], size=size * 0.85)
    big = FontProperties(fname=font.get("700", font["400"]), size=size * 1.35)
    line_h = line_height(font["700" if bold else "400"])
    fill_step, fill_end, fill_ask, line = lk["palette"]
    lw = lk["lw"]
    words_of, box = {}, {}
    for nid, kind, words in [(n[0], n[1], n[2]) for n in nodes] + [(s[0], s[1], s[2]) for s in side]:
        lines = textwrap.wrap(words, lk["width"], break_long_words=False, break_on_hyphens=False) or [words]  # never inside a word
        tw = max(TextPath((0, 0), ln, prop=prop).get_extents().width for ln in lines)
        th = size * line_h * (1.15 * len(lines) - 0.15)
        w, h = tw + size * 1.6, th + size * 1.0
        if kind == "ask":
            w, h = w * 1.55, h * 1.7
        elif kind in ("start", "end") and lk["end_style"] == "circle":
            w = h = max(w, h) * 1.1
        elif kind in ("start", "end"):
            w = w + size * 1.2
        words_of[nid], box[nid] = (lines, kind), [0, 0, w, h]
    gap = size * lk["gap"]
    pos = 0
    for nid, _, _ in nodes:  # the main chain on one line
        w, h = box[nid][2:]
        half = w / 2 if across else h / 2
        pos += half
        box[nid][0:2] = (pos, 0) if across else (0, -pos)
        pos += half + gap
    lane = {1: max(box[n[0]][3 if across else 2] for n in nodes) / 2, -1: 0}
    lane[-1] = lane[1]
    for j, (sid, kind, words, frm, to) in enumerate(side):  # side boxes beside their question, sides alternating
        sign = 1 if j % 2 == 0 else -1
        w, h = box[sid][2:]
        room = TextPath((0, 0), answers[1], prop=small).get_extents().width + size * 1.5
        off = lane[sign] + max(gap, room) + (h if across else w) / 2
        fx, fy = box[frm][0:2]
        box[sid][0:2] = (fx, fy - sign * off) if across else (fx + sign * off, fy)
        lane[sign] = off + (h if across else w) / 2
    if lk["flip"]:  # bottom-up (or right-left): the same chart mirrored along its direction
        for b in box.values():
            b[0 if across else 1] = -b[0 if across else 1]
    xs = [b[0] - b[2] / 2 for b in box.values()] + [b[0] + b[2] / 2 for b in box.values()]
    ys = [b[1] - b[3] / 2 for b in box.values()] + [b[1] + b[3] / 2 for b in box.values()]
    pad = size * 2.5
    top = size * 3.2 if lk["title"] else 0
    style_ctx = plt.xkcd(scale=0.8, length=120, randomness=1.5) if lk["sketch"] else matplotlib.rc_context()
    with style_ctx:
        fig = plt.figure(figsize=((max(xs) - min(xs) + 2 * pad) / 72, (max(ys) - min(ys) + 2 * pad + top) / 72), dpi=lk["dpi"])
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad + top)
        ax.axis("off")
        if lk["grid"]:
            for gx in range(int(min(xs) - pad), int(max(xs) + pad), 18):
                ax.axvline(gx, color="#d6e4f0", lw=0.5, zorder=0)
            for gy in range(int(min(ys) - pad), int(max(ys) + pad + top), 18):
                ax.axhline(gy, color="#d6e4f0", lw=0.5, zorder=0)
        if lk["title"]:
            ax.text((min(xs) + max(xs)) / 2, max(ys) + pad * 0.4 + top / 2, title, ha="center", va="center",
                    fontproperties=big, color=line)
        colours = iter(rng.sample(STEP_COLOURS, len(STEP_COLOURS)) * 3)
        for nid, (lines, kind) in words_of.items():
            x, y, w, h = box[nid]
            fill = {"ask": fill_ask, "start": fill_end, "end": fill_end}.get(kind, fill_step)
            if lk["rainbow"] and kind == "step":
                fill = next(colours)
            color = "#ffffff" if fill in ("#222222", "#1565c0", "#607d8b", "#9575cd") else "#111111"
            def shape(o):  # the box moved by o (its shadow is the same box a little down-right)
                if kind == "ask":
                    return Polygon([(x + o, y - o + h / 2), (x + o + w / 2, y - o), (x + o, y - o - h / 2),
                                    (x + o - w / 2, y - o)], closed=True)
                if kind in ("start", "end") and lk["end_style"] in ("ellipse", "circle"):
                    e = lk["end_style"] == "ellipse"
                    return Ellipse((x + o, y - o), w * (1.1 if e else 1), h * (1.15 if e else 1))
                style = ("round,pad=0,rounding_size=%f" % (h / 2) if kind in ("start", "end") and lk["end_style"] == "stadium"
                         else "round,pad=0,rounding_size=%f" % (size * 0.6) if (kind in ("start", "end") or lk["step_style"] == "round")
                         else "square,pad=0")
                return FancyBboxPatch((x + o - w / 2, y - o - h / 2), w, h, boxstyle=style)
            if lk["shadow"]:
                ax.add_patch(shape(size * 0.35)).set(fc="#00000038", ec="none", zorder=1)
            ax.add_patch(shape(0)).set(fc=fill, ec=line, lw=lw, zorder=2)
            ax.text(x, y, "\n".join(lines), ha="center", va="center", fontproperties=prop, color=color,
                    linespacing=1.15, zorder=3)

        def edge_point(nid, toward):
            x, y, w, h = box[nid]
            dx, dy = toward[0] - x, toward[1] - y
            if abs(dx) * h > abs(dy) * w:
                return (x + w / 2 * (1 if dx > 0 else -1), y)
            return (x, y + h / 2 * (1 if dy > 0 else -1))
        main = [n[0] for n in nodes]
        for a, b, words in edges:
            (ax_, ay), (bx, by) = box[a][0:2], box[b][0:2]
            if a in main:  # along the chain, or a question -> its side box
                p, q, route = edge_point(a, (bx, by)), edge_point(b, (ax_, ay)), "arc3"
            else:  # side box -> back to (or further on) the chain: out of its far end, along, then in
                if across:
                    p = (ax_, ay + (box[a][3] / 2) * (1 if by > ay else -1))
                    q = (bx, by + (box[b][3] / 2) * (1 if ay > by else -1))
                    route = "angle,angleA=0,angleB=90"
                else:
                    p = (ax_ + (box[a][2] / 2) * (1 if bx > ax_ else -1), ay) if abs(by - ay) < 1 else (ax_, ay + box[a][3] / 2 * (1 if by > ay else -1))
                    q = (bx + box[b][2] / 2 * (1 if ax_ > bx else -1), by)
                    route = "angle,angleA=90,angleB=0" if abs(by - ay) >= 1 else "arc3"
            ax.add_patch(FancyArrowPatch(p, q, arrowstyle=lk["head"], mutation_scale=size * 1.1, color=line, lw=lw, connectionstyle=route,
                                         shrinkA=0, shrinkB=0, linestyle="--" if lk["dashed"] else "-", zorder=1.5))
            if words:
                dx, dy = q[0] - p[0], q[1] - p[1]
                boxed = {"bbox": {"boxstyle": "round,pad=0.2", "fc": lk["paper"], "ec": line, "lw": 0.6}} if lk["boxed_answers"] else {}
                if abs(dx) > abs(dy):  # sideways: above the line, just after the start
                    ax.text(p[0] + size * 0.5 * (1 if dx > 0 else -1), p[1] + size * 0.3, words, fontproperties=small,
                            color=line, ha="left" if dx > 0 else "right", va="bottom", zorder=4, **boxed)
                else:  # up or down: beside the line, just after the start
                    ax.text(p[0] + size * 0.5, p[1] + size * 0.4 * (1 if dy > 0 else -1), words, fontproperties=small,
                            color=line, ha="left", va="bottom" if dy > 0 else "top", zorder=4, **boxed)
        fig.savefig(path, facecolor=lk["paper"])
        plt.close(fig)


def mermaid(nodes, side, edges, lk):
    """Surya's Mermaid: graph TD/LR/BT/RL, a node's shape and words where it first appears; words quoted only
    when Mermaid needs it."""
    label = lambda w: '"' + w.replace('"', "'") + '"' if re.search(r'[?()\[\]{}"|<>#:;]', w) else w  # noqa: E731
    end = "((%s))" if lk["end_style"] == "circle" else "([%s])"
    shape = {nid: ("{%s}" if kind == "ask" else end if kind in ("start", "end") else "[%s]") % label(words)
             for nid, kind, words in [(n[0], n[1], n[2]) for n in nodes] + [(s[0], s[1], s[2]) for s in side]}
    seen = set()
    node = lambda n: n + ("" if n in seen or seen.add(n) else shape[n])  # noqa: E731
    lines = ["graph " + {(False, False): "TD", (True, False): "LR", (False, True): "BT", (True, True): "RL"}[(lk["across"], lk["flip"])]]
    for a, b, words in edges:
        lines.append(f"    {node(a)} {'-- ' + words + ' -->' if words else '-->'} {node(b)}")
    return "\n".join(lines)


def make(job):
    """One flowchart: (index, seed) -> its index line, or {"error"} when it can't be drawn."""
    from pipeline.pools import diagram_html
    i, seed, procs, fonts_ = job
    rng = random.Random(seed)
    try:
        lk = look(rng)
        title, nodes, side, edges, answers = plan(rng, rng.choice(procs), lk)
        font = rng.choice(fonts_)
        drawn = "".join(n[2] for n in nodes) + "".join(s[2] for s in side) + "".join(answers) + (title if lk["title"] else "")
        if not all(covers(font[w], drawn) for w in font):
            return {"error": "font lacks a character"}
        path = OUT / f"{i // 1000:03d}" / f"{i:06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        draw(rng, title, nodes, side, edges, answers, font, path, lk)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    mm = mermaid(nodes, side, edges, lk)
    n_ask = sum(1 for n in nodes if n[1] == "ask")
    way = {(False, False): "vertical", (True, False): "horizontal", (False, True): "bottom-up", (True, True): "right-to-left"}[(lk["across"], lk["flip"])]
    desc = (f"A {way} flowchart of {len(nodes) + len(side)} steps from {nodes[0][2]} to {nodes[-1][2]}"
            + (f", with {n_ask} decision{'s' if n_ask > 1 else ''}" if n_ask else "") + ".")
    return {"file": (path.relative_to(ROOT) if path.is_relative_to(ROOT) else path).as_posix(), "label": "Diagram",
            "lang": "en", "title": title, "html": diagram_html(f"Flowchart: {title}.", desc, mermaid=mm,
                                                               texts=[title] if lk["title"] else ()),
            "mermaid": mm, "look": {k: v for k, v in lk.items() if k != "palette"},
            "source": "generated (LLM process words, matplotlib)", "license": "generated"}


def build(n=50_000, seed=0, workers=8, start=0):
    procs, fonts_ = processes(), fonts()
    OUT.mkdir(parents=True, exist_ok=True)
    done = errors = 0
    with open(OUT / "flowcharts.jsonl", "a" if start else "w", encoding="utf-8") as out, ProcessPoolExecutor(workers) as pool:
        for item in pool.map(make, [(i, seed * 10**7 + i, procs, fonts_) for i in range(start, start + n)], chunksize=50):
            if "error" in item:
                errors += 1
                continue
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            done += 1
            if done % 5000 == 0:
                print(f"  {done} flowcharts", flush=True)
    print(f"flowcharts: {done} drawn, {errors} failed, in {OUT}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--start", type=int, default=0, help="first file number; above 0 adds to flowcharts.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sample", help="draw 16 into this folder to look at, instead of the pool")
    a = ap.parse_args()
    if a.sample:
        OUT = Path(a.sample)
        procs, fonts_ = processes(), fonts()
        for i in range(16):
            item = make((i, a.seed * 10**7 + i, procs, fonts_))
            print(i, item.get("error") or item["mermaid"].split("\n")[0], {k: v for k, v in item.get("look", {}).items() if v is True})
    else:
        build(a.n, a.seed, a.workers, a.start)
