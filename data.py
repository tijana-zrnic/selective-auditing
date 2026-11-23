import random
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"

UCSB_TRANSCRIPTS = ["UCSB01", "UCSB02", "UCSB03", "UCSB04", "UCSB05", "UCSB06", "UCSB07", "UCSB10"]


def _shuffle(conf, err, seed=42):
    order = list(range(len(conf)))
    random.Random(seed).shuffle(order)
    return conf[order], err[order]


def _jitter(conf, width, seed=0):
    rng = random.Random(seed)
    return conf + np.array([rng.uniform(-width, width) for _ in range(len(conf))])


def load_labeling(name):
    df = pd.read_csv(DATA / "pac" / f"{name}.csv")
    yhat = "Yhat (GPT4o)" if name in ("bias", "stance") else "Yhat"
    conf = df["confidence"].to_numpy(float)
    err = (df["Y"] != df[yhat]).to_numpy(int)
    conf, err = _shuffle(conf, err)
    if name in ("bias", "stance"):
        conf = _jitter(conf, 0.005)
    return conf, err


def load_safety(name):
    df = pd.read_csv(DATA / "safety" / f"{name}.csv")
    conf, err = _shuffle(df["confidence"].to_numpy(float), df["err"].to_numpy(int))
    return _jitter(conf, 0.01), err


def load_annotation(annotator, cutoff=0.8):
    per = [pd.read_csv(DATA / "ucsb" / f"{annotator}_gpt5.2" / f"{t}.csv").sort_values("position")
           for t in UCSB_TRANSCRIPTS]
    df = pd.concat([d[d.round_num == r] for r in (1, 2) for d in per], ignore_index=True)
    conf = df["confidence"].to_numpy(float)
    err = (df["quality_score"] <= cutoff).to_numpy(int)
    return _jitter(conf, 0.005), err
