"""Single entry point: all sources -> dedup -> validate -> gallery.

python -m src.build --seed 0 --n-grammar 300 --n-websight 150 --n-doclaynet 100
"""
import argparse

from src import dedup, from_doclaynet, from_websight, gallery, grammar, validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-grammar", type=int, default=300)
    ap.add_argument("--n-websight", type=int, default=150)
    ap.add_argument("--n-doclaynet", type=int, default=100)
    a = ap.parse_args()
    grammar.run(a.seed, a.n_grammar)
    from_websight.run(a.seed, a.n_websight)
    from_doclaynet.run(a.seed, a.n_doclaynet)
    dedup.run()
    ok = validate.run()
    gallery.run()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
