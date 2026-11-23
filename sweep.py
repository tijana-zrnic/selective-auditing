import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from algorithms import additive_thresholding, additive_thresholding_calibrated, optimally_efficient, oracle
from data import load_annotation, load_labeling, load_safety

RESULTS = Path(__file__).resolve().parent / "results"
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
ZETA = 0.1

# name -> (loader, loader args, number of trials)
STREAMS = {
    "sorrybench": (load_safety, ("sorrybench",), 100),
    "strongreject": (load_safety, ("strongreject",), 100),
    "bias": (load_labeling, ("bias",), 20),
    "stance": (load_labeling, ("stance",), 20),
    "imagenet": (load_labeling, ("imagenet",), 6),
    "imagenetv2": (load_labeling, ("imagenetv2",), 20),
}
for _annotator in ("qwen14b", "gpt4omini"):
    for _cutoff in (0.8, 0.7, 0.6, 0.5):
        STREAMS[f"ucsb_{_annotator}_cutoff{_cutoff}"] = (load_annotation, (_annotator, _cutoff), 50)


def _oe(conf, err, alpha, seed, E_max):
    # The dual update only sees the (1 - zeta) primal part of the audit-skip probability,
    # so the target is rescaled to make the realized error rate land near alpha.
    return optimally_efficient(conf, err, alpha / (1.0 - ZETA), zeta=ZETA, E_max=E_max, seed=seed)


METHODS = {
    "at": lambda c, e, a, s: additive_thresholding(c, e, a, seed=s),
    "at_calibrated": lambda c, e, a, s: additive_thresholding_calibrated(c, e, a, seed=s),
    "oe": lambda c, e, a, s: _oe(c, e, a, s, E_max=25.0),
    "oe_uncapped": lambda c, e, a, s: _oe(c, e, a, s, E_max=None),
}


@lru_cache(maxsize=None)
def load(stream):
    loader, args, _ = STREAMS[stream]
    return loader(*args)


def run(stream, method, alpha):
    conf, err = load(stream)
    if method == "oracle":
        runs = [oracle(conf, err, alpha)]
    else:
        runs = [METHODS[method](conf, err, alpha, s) for s in range(STREAMS[stream][2])]
    error = 100 * np.array([r["error_rate"] for r in runs])
    saved = 100 * np.array([r["saved"] for r in runs])
    return dict(stream=stream, algorithm=method, alpha=alpha,
                error=error.mean(), error_sd=error.std(), saved=saved.mean(), saved_sd=saved.std())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("streams", nargs="*", metavar="stream",
                   help=f"subset of streams to run (default: all): {', '.join(STREAMS)}")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args()
    streams = args.streams or list(STREAMS)
    unknown = set(streams) - set(STREAMS)
    if unknown:
        p.error(f"unknown streams: {', '.join(sorted(unknown))}")

    tasks = [(s, m, a) for s in streams for m in [*METHODS, "oracle"] for a in ALPHAS]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run, *zip(*tasks)))

    RESULTS.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    for stream, sub in df.groupby("stream", sort=False):
        sub.drop(columns="stream").to_csv(RESULTS / f"{stream}.csv", index=False)
        print(f"wrote results/{stream}.csv")


if __name__ == "__main__":
    main()
