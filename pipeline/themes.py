"""Random visual themes: the same template looks different on every page, and always readable.

theme(rng, source) -> (record, css, js). The CSS goes into the template's {{ style }} hook after the
typography (pipeline/style.py); the JS settings go to the page script (pipeline/render.py, PAGE_JS).

  * Grammar and DocLayNet templates are plain black-on-white layouts, so a theme styles them fully: a colour
    palette (13 professional accents, each dark enough to read on white) with headings in one of 6
    treatments, tables in one of 5, side boxes and quotes in one of 4, bullets, rules, a page tint, spacing
    and corners. Grammar pages also get spacing changes; DocLayNet pages only colours (their boxes have
    fixed sizes, and padding would push text out of them).
  * WebSight pages come with their own design. They keep it, but in the browser every colour on the page is
    turned around the colour wheel by the same angle (hue only: lightness stays, so contrast stays), and
    more or less saturated, so one web design comes in many colour schemes.

All themes use only marks that are not text: bullets are disc, circle or square (ordered lists lose their
numbers, which would be text on the image but not in the ground truth). Whatever a theme does, the page
script then checks every slot's contrast against what is behind it and darkens (or lightens) text below
4.5:1 (pipeline/doc_pipeline.py).
"""

ACCENTS = ["#1f4e79", "#0b6e4f", "#7b1e3a", "#5b2a86", "#8a4b08", "#0e5e6f", "#2f3e46", "#a4161a", "#374785",
           "#2d6a4f", "#9a3412", "#4a4e69", "#1d3557"]
INKS = ["#111111", "#1a1a1a", "#222222", "#262626", "#1c2833", "#2b2118"]


def mix(color, white):
    """color mixed with white (0 = color, 1 = white)."""
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(c + (255 - c) * white):02x}" for c in (r, g, b))


def theme(rng, source):
    accent = rng.choice(ACCENTS)
    tint, soft = mix(accent, rng.uniform(0.88, 0.94)), mix(accent, rng.uniform(0.95, 0.975))
    rec = {"accent": accent, "heading": rng.choice(["plain", "accent", "underline", "bar", "band", "rule-above"]),
           "table": rng.choice(["grid", "striped", "header-tint", "rules", "boxed"]),
           "box": rng.choice(["tint", "bar", "outline", "plain"]),
           "page": rng.choices(["white", "off-white", "tint"], weights=[5, 3, 2])[0],
           "bullet": rng.choice(["disc", "square", "circle"]), "radius": rng.choice([0, 0, 3, 6]),
           "paragraph_gap": round(rng.uniform(0.3, 1.0), 2), "heading_gap": round(rng.uniform(0.5, 1.4), 2),
           "ink": rng.choice(INKS)}
    page_bg = {"white": "#ffffff", "off-white": rng.choice(["#fbfaf7", "#fafafa", "#f9f8f4"]), "tint": soft}[rec["page"]]
    imp = " !important"
    css = [f"ol, ul {{ list-style-type: {rec['bullet']}{imp}; }}",
           f"li::marker {{ color: {accent}; }}",
           f"html, body, .page {{ background-color: {page_bg}{imp}; }}"]
    js = {"hue": 0, "saturate": 1.0}
    if source in ("websight", "websight2"):
        # Their own design stays; its colours are turned around the colour wheel (in the page script)
        js = {"hue": rng.choice([0, 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]),
              "saturate": round(rng.uniform(0.7, 1.25), 2)}
        rec["hue"], rec["saturate"] = js["hue"], js["saturate"]
        return rec, "\n".join(css) + "\n", js
    heading = {"plain": f"color: {rec['ink']}{imp};",
               "accent": f"color: {accent}{imp};",
               "underline": f"color: {accent}{imp}; border-block-end: 2px solid {accent}{imp};",
               "bar": f"color: {accent}{imp}; border-inline-start: 4px solid {accent}{imp}; padding-inline-start: 0.4em;",
               "band": f"color: {accent}{imp}; background-color: {tint}{imp}; padding-inline: 0.3em;",
               "rule-above": f"color: {rec['ink']}{imp}; border-block-start: 2px solid {accent}{imp};"}[rec["heading"]]
    if source == "doclaynet":  # fixed-size boxes: colours only, nothing that takes room
        heading = heading.replace("padding-inline-start: 0.4em;", "").replace("padding-inline: 0.3em;", "")
        heading = heading.replace("border-block-end: 2px", "border-block-end: 1px").replace(
            "border-block-start: 2px", "border-block-start: 1px").replace("border-inline-start: 4px", "border-inline-start: 3px")
    css += [f"h1, h2, h3, h4, h5, h6, dt, strong {{ {heading} }}",
            f"body, .page, p, li, td, dd, span, div {{ color: {rec['ink']}; }}",
            f"th {{ color: {accent}{imp}; }}"]
    table = {"grid": f"th, td {{ border-color: {mix(accent, 0.55)}{imp}; }}",
             "striped": f"tbody tr:nth-child(even) td {{ background-color: {soft}{imp}; }} "
                        f"th {{ border-block-end: 2px solid {accent}{imp}; }}",
             "header-tint": f"th {{ background-color: {tint}{imp}; }} th, td {{ border-color: {mix(accent, 0.6)}{imp}; }}",
             "rules": f"th, td {{ border-inline: none{imp}; border-block-start: none{imp}; "
                      f"border-block-end: 1px solid {mix(accent, 0.6)}{imp}; }}",
             "boxed": f"table {{ outline: 2px solid {accent}; }} th {{ background-color: {soft}{imp}; }}"}[rec["table"]]
    box = {"tint": f"aside, blockquote {{ background-color: {soft}{imp}; border-color: {mix(accent, 0.5)}{imp}; }}",
           "bar": f"aside, blockquote {{ border-inline-start: 4px solid {accent}{imp}; }}",
           "outline": f"aside, blockquote {{ border-color: {accent}{imp}; }}",
           "plain": ""}[rec["box"]]
    css += [table, box, f"hr {{ border-color: {mix(accent, 0.5)}; }}"]
    if source == "grammar":
        css += [f"p {{ margin-block-end: {rec['paragraph_gap']}em; }}",
                f"h2, h3 {{ margin-block-start: {rec['heading_gap']}em; }}",
                f"aside, blockquote, table, .ph {{ border-radius: {rec['radius']}px; }}"]
    return rec, "\n".join(c for c in css if c) + "\n", js
