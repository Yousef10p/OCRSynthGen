"""Make checked training samples with only the everyday settings; everything else keeps the pipeline's default.

    .venv-pipeline\\Scripts\\python generate.py --n 8
    .venv-pipeline\\Scripts\\python generate.py --n 1000 --epoch 3 --strength-ocrsmith 0.3 --workers 4 --out datasets\\epoch3

All other settings (language, task, checks...): train.py --help.
"""
import argparse
import sys
import time

from train import DataGenerator, GenerationError, save


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8, help="samples to make (default: 8)")
    ap.add_argument("--epoch", type=int, default=None,
                    help="make epoch N's samples with seed N (the same every time; another epoch, other pages); "
                         "without it: a fresh batch with a new seed each run (printed)")
    for tool in ("ocrsmith", "augraphy", "scan-simulator"):
        ap.add_argument(f"--strength-{tool}", type=float, default=0.5,
                        help="0 lightest, 0.5 the tool's own settings (default), 1 strongest")
    ap.add_argument("--workers", type=int, default=None, help="processes (default: min(4, CPUs)); ~1.5 GB each")
    ap.add_argument("--out", default="datasets/train_try", help="folder the samples are saved in (default: datasets/train_try)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    strengths = {t: getattr(a, f"strength_{t.replace('-', '_')}") for t in ("ocrsmith", "augraphy", "scan-simulator")}
    start = time.time()
    # the seed: the epoch's own number (epoch 99 -> seed 99), or a new one for a batch (seed 0 every run made the
    # same pages each time)
    seed = a.epoch if a.epoch is not None else int(time.time() * 1000) % 10**9
    with DataGenerator(seed=seed, workers=a.workers, strengths=strengths) as gen:
        print(f"{a.n} samples, {'batch' if a.epoch is None else f'epoch {a.epoch}'}, seed {seed}, {gen.workers} workers, "
              f"strengths {strengths}", flush=True)
        try:
            samples = gen.batch(a.n) if a.epoch is None else gen.epoch(a.n, a.epoch)
        except GenerationError as e:
            print(f"GenerationError: {str(e)[:600]}", flush=True)
            samples = e.samples
        stats = dict(gen.stats)
    out = save(samples, a.out)
    print(f"{len(samples)} samples in {time.time() - start:.0f}s -> {out.resolve()}  {stats}")


if __name__ == "__main__":
    main()
