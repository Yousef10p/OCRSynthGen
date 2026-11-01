"""Validate every template and write reports/bank_stats.json."""
import json
import re
import statistics
from collections import Counter

import jsonschema
from jinja2 import Environment, StrictUndefined, meta as jmeta

from src.bank import REPORTS, ROOT, TEMPLATES
from src.dedup import load_bank

PHYSICAL = re.compile(r"\b(left|right)\b", re.I)
ENV = Environment(undefined=StrictUndefined)
VALIDATOR = jsonschema.Draft202012Validator(json.loads((ROOT / "schema" / "template.schema.json").read_text()))


def check(meta):
    """List of problems with one template (empty when it passes)."""
    text = (TEMPLATES / meta["id"] / "template.html.j2").read_text(encoding="utf-8")
    errors = [f"schema: {e.message}" for e in VALIDATOR.iter_errors(meta)]
    expected = [f"slot_{i}" for i in range(len(meta["slots"]))]
    if [s["id"] for s in meta["slots"]] != expected or meta["features"]["slot_count"] != len(expected):
        errors.append("slot ids/slot_count do not match slot_0..slot_{k-1}")
    try:
        found = jmeta.find_undeclared_variables(ENV.parse(text))
    except Exception as e:
        return errors + [f"jinja parse: {e}"]
    if found != set(expected) | {"style"}:
        errors.append(f"variables differ: extra {sorted(found - set(expected) - {'style'})},"
                      f" missing {sorted(set(expected) - found)}")
    lower = text.lower()
    if '<html lang="ar" dir="rtl">' not in text:
        errors.append('missing <html lang="ar" dir="rtl">')
    errors += [f"contains {bad}" for bad in ("http", "<script") if bad in lower]
    # Pictures are allowed only as local crops listed in template.json and stored in the template folder.
    srcs = re.findall(r'<img[^>]*\ssrc="([^"]*)"', text)
    if lower.count("<img") != len(srcs) or sorted(srcs) != meta.get("images", []):
        errors.append("<img> tags do not match template.json images")
    errors += [f"missing image {s}" for s in srcs if not (TEMPLATES / meta["id"] / s).is_file()]
    if meta["source"] in ("grammar", "doclaynet") and PHYSICAL.search(text):
        errors.append("physical left/right in CSS")
    try:
        ENV.from_string(text).render(style="", **{k: "x" for k in expected})
    except Exception as e:
        errors.append(f"render: {e}")
    return errors


def stats(metas):
    pct = lambda pred: round(100 * sum(map(pred, metas)) / max(1, len(metas)), 1)
    counts = [m["features"]["slot_count"] for m in metas]
    dedup = REPORTS / "dedup.json"
    return {
        "templates": len(metas),
        "per_source": dict(Counter(m["source"] for m in metas)),
        "slots_per_template": {"min": min(counts), "median": statistics.median(counts), "max": max(counts)},
        "roles": dict(Counter(s["role"] for m in metas for s in m["slots"]).most_common()),
        "feature_coverage_pct": {
            "tables": pct(lambda m: m["features"]["tables"] > 0),
            "lists": pct(lambda m: m["features"]["lists"] > 0),
            "multi_column": pct(lambda m: m["features"]["columns"] > 1),
            "forms": pct(lambda m: m["features"]["forms"] > 0),
            "header_or_footer": pct(lambda m: m["features"]["has_header"] or m["features"]["has_footer"]),
        },
        "page_sizes": dict(Counter(f'{m["page"]["size"]} {m["page"]["orientation"]}' for m in metas)),
        "duplicates_removed": json.loads(dedup.read_text())["removed"] if dedup.exists() else None,
    }


def run():
    metas = load_bank()
    failures = {m["id"]: errs for m in metas if (errs := check(m))}
    for tid, errs in list(failures.items())[:20]:
        print(f"FAIL {tid}: {'; '.join(errs)}")
    s = stats(metas) | {"failed": len(failures)}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "bank_stats.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))
    print(f"validate: {len(metas) - len(failures)}/{len(metas)} templates pass")
    return not failures


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
