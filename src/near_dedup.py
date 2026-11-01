"""Remove near-duplicate templates: layouts that are almost the same, not only exact copies (src/dedup.py).

Each template becomes a set of layout features, compared only with templates of the same source:
  * flow pages (WebSight, Grammar): runs of 4 consecutive elements of the page skeleton, each written as
    depth:tag:slot-role, plus the stylesheet's layout declarations (display, grid columns, flex direction,
    floats, column count, widths). Colours, fonts and the text itself play no part;
  * positioned pages (DocLayNet): each box as kind + position and size on a grid of 1/20 of the page.
Two templates whose sets are at least THRESHOLD similar (Jaccard: shared features / all features) are
near-duplicates. Candidate pairs are found with MinHash LSH (128 hashes in 32 bands of 4), then checked
exactly. Each group of near-duplicates keeps one template: one from the bank's first build if the group has
one (the gallery and render sets refer to those), else the one with the most slots.

    .venv\\Scripts\\python -m src.near_dedup [--threshold 0.9] [--dry-run]
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict

import numpy as np
from lxml import html

from src.bank import ROOT, SLOT_RE, TEMPLATES, page_dims
from src.dedup import FIRST_BUILD, load_bank, write_index

THRESHOLD = 0.9
N_HASH, BANDS = 128, 32
REPORT = ROOT / "reports" / "near_dedup.json"
LAYOUT_PROPS = re.compile(r"\b(display|grid-template-columns|grid-template-areas|flex-direction|flex-wrap|float|"
                          r"column-count|columns|inline-size|width|max-width|position|justify-content)\s*:\s*([^;}]+)")
_rng = np.random.default_rng(0)
_A = _rng.integers(0, 2**63, N_HASH, dtype=np.uint64) * np.uint64(2) + np.uint64(1)  # odd multipliers
_B = _rng.integers(0, 2**63, N_HASH, dtype=np.uint64)


def flow_features(root, roles):
    skeleton = []

    def walk(el, depth):
        for child in el:
            if not isinstance(child.tag, str) or child.tag in ("style", "script", "head", "meta"):
                continue
            role = ""
            if child.text and (m := SLOT_RE.fullmatch(child.text.strip())):
                role = roles.get(f"slot_{m.group(1)}", "")
            skeleton.append(f"{min(depth, 12)}:{child.tag}:{role}")
            walk(child, depth + 1)
    walk(root.find("body") if root.find("body") is not None else root, 0)
    feats = {"|".join(skeleton[i:i + 4]) for i in range(max(1, len(skeleton) - 3))}
    css = " ".join(s.text or "" for s in root.iter("style")) + " " + " ".join(el.get("style", "") for el in root.iter()
                                                                            if isinstance(el.tag, str))
    feats |= {f"css:{p.lower()}:{re.sub(r'\s+', ' ', v.strip().lower())}" for p, v in LAYOUT_PROPS.findall(css)}
    return feats


MM = re.compile(r"(inset-inline-start|inset-block-start|inline-size|block-size):\s*([\d.]+)mm")


def box_features(root, roles, meta):
    w, h = page_dims(meta["page"]["size"], meta["page"]["orientation"])
    feats = set()
    for el in root.iter():
        if not isinstance(el.tag, str) or "box" not in (el.get("class") or ""):
            continue
        v = dict(MM.findall(el.get("style", "")))
        if len(v) < 4:
            continue
        m = SLOT_RE.search(el.text_content() or "")
        kind = roles.get(f"slot_{m.group(1)}", el.tag) if m else ("picture" if "ph" in el.get("class") or
                                                                  el.tag == "img" else el.tag)
        q = [round(float(v[k]) / d * 20) for k, d in (("inset-inline-start", w), ("inset-block-start", h),
                                                    ("inline-size", w), ("block-size", h))]
        feats.add(f"{kind}:{q[0]}:{q[1]}:{q[2]}:{q[3]}")
    return feats


def features(meta):
    text = (TEMPLATES / meta["id"] / "template.html.j2").read_text(encoding="utf-8")
    root = html.fromstring(text)
    roles = {s["id"]: s["role"] for s in meta["slots"]}
    return box_features(root, roles, meta) if meta["positioning"] == "absolute" else flow_features(root, roles)


def minhash(feats):
    """128 MinHash values: multiply-shift hashing, (a*x + b) mod 2^64 (numpy wraps), upper 32 bits."""
    x = np.array([int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
                  for f in feats] or [0], dtype=np.uint64)
    with np.errstate(over="ignore"):
        h = (_A[:, None] * x[None, :] + _B[:, None]) >> np.uint64(32)
    return h.min(axis=1)


def groups(items, threshold):
    """Union-find over pairs whose exact Jaccard is >= threshold. items: [(id, feature set, signature)]."""
    rows = N_HASH // BANDS
    buckets = defaultdict(list)
    for i, (_, _, sig) in enumerate(items):
        for b in range(BANDS):
            buckets[(b, sig[b * rows:(b + 1) * rows].tobytes())].append(i)
    parent = list(range(len(items)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    checked, pairs = set(), []
    for members in buckets.values():
        for k, a in enumerate(members):
            for b in members[k + 1:]:
                if (a, b) in checked or find(a) == find(b):
                    continue
                checked.add((a, b))
                fa, fb = items[a][1], items[b][1]
                j = len(fa & fb) / len(fa | fb) if fa | fb else 1.0
                if j >= threshold:
                    parent[find(b)] = find(a)
                    pairs.append((items[a][0], items[b][0], round(j, 3)))
    clusters = defaultdict(list)
    for i in range(len(items)):
        clusters[find(i)].append(items[i][0])
    return [c for c in clusters.values() if len(c) > 1], pairs


def run(threshold=THRESHOLD, dry_run=False):
    bank = {m["id"]: m for m in load_bank()}
    by_source = defaultdict(list)
    for tid, meta in bank.items():
        by_source[meta["source"]].append(tid)
    removed, report = [], {"threshold": threshold, "sources": {}}
    for source, ids in sorted(by_source.items()):
        items = []
        for tid in ids:
            feats = features(bank[tid])
            items.append((tid, feats, minhash(feats)))
        clusters, pairs = groups(items, threshold)
        drop = []
        for c in clusters:
            keep = max(c, key=lambda t: (bool(FIRST_BUILD.match(t)), len(bank[t]["slots"]), t))
            drop += [t for t in c if t != keep]
        removed += drop
        report["sources"][source] = {"templates": len(ids), "groups": len(clusters), "removed": len(drop),
                                     "kept": len(ids) - len(drop), "example_pairs": pairs[:20]}
        print(f"{source}: {len(ids)} templates, {len(clusters)} groups of near-duplicates, "
              f"{len(drop)} removed, {len(ids) - len(drop)} kept", flush=True)
    report["removed_ids"] = sorted(removed)
    if not dry_run:
        for tid in removed:
            shutil.rmtree(TEMPLATES / tid)
        REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
        write_index()
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    run(a.threshold, a.dry_run)
