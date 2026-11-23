import math
import random

import numpy as np


def _result(n_items, n_errors, n_audited, n_missed):
    return {
        "error_rate": n_missed / n_errors if n_errors > 0 else 0.0,
        "saved": 1.0 - n_audited / n_items,
    }


def additive_thresholding(conf, err, alpha, *, gamma=0.1, eta=0.05, tau0=0.5, seed=0):
    rng = random.Random(seed)
    tau = tau0
    n_errors = n_missed = n_audited = 0
    for c, e in zip(conf, err):
        if c <= tau:
            H, pi = 1, 1.0
        else:
            H = 1 if rng.random() < gamma else 0
            pi = gamma
        n_errors += e
        n_missed += e * (1 - H)
        n_audited += H
        tau += eta * ((1.0 - pi - alpha) * e * H / pi)
    return _result(len(conf), n_errors, n_audited, n_missed)


def additive_thresholding_calibrated(conf, err, alpha, *, gamma=0.1, eta=0.02, beta=0.1,
                                     tau0=0.5, seed=0):
    rng = random.Random(seed)
    grid = np.append(np.round(np.arange(0.0, 1.0, 0.01), 6), np.inf)
    w = np.full(len(grid), 1.0 / len(grid))
    tau = tau0
    n_errors = n_missed = n_audited = 0
    for c, e in zip(conf, err):
        above = (c >= grid).astype(float)
        S = float(w @ above)
        skip = 1.0 - gamma if S >= tau else 0.0
        pi = 1.0 - skip
        H = 1 if rng.random() < pi else 0
        n_errors += e
        n_missed += e * (1 - H)
        n_audited += H
        g = H / pi * (S - (1 - e)) * above
        log_w = np.log(w + 1e-300) - beta * g
        w = np.exp(log_w - log_w.max())
        w /= w.sum()
        tau += eta * (skip - alpha) * e * (H / pi)
    return _result(len(conf), n_errors, n_audited, n_missed)


def optimally_efficient(conf, err, alpha, *, zeta=0.1, lam=0.2, C_A=3e-3, E_max=25.0, seed=0):
    rng = random.Random(seed)
    grid = np.append(np.round(np.arange(0.0, 1.005, 0.01), 6), np.inf)
    K = len(grid)
    log_K = math.log(K)
    w = np.full(K, 1.0 / K)
    G = np.zeros(K)
    E = 0.0
    S = 0.0
    gap = 0.0
    eta_prev = math.inf
    n_errors = n_missed = n_audited = 0
    for c, e in zip(conf, err):
        p = (1.0 - zeta) * w
        p[-1] += zeta
        threshold = grid[int(np.searchsorted(np.cumsum(p), rng.random() * p.sum()))]
        H = 1 if c < threshold else 0
        q = float(p[grid > c].sum())
        e_hat = e * H / q if q > 0.0 else 0.0
        n_errors += e
        n_missed += e * (1 - H)
        n_audited += H

        skip = (c >= grid).astype(float)
        r = float(w @ skip)
        mu = max(lam * math.exp(min(lam * E, 30.0)), 1e-12)

        # AdaHedge on the rewards 1{C_t >= theta_i} (1 - mu_t e_hat_t)
        dR = skip * (1.0 - mu * e_hat)
        if math.isfinite(eta_prev):
            a = np.log(np.clip(w, 1e-300, None)) + eta_prev * dR
            a_max = float(a.max())
            mix = -(a_max + math.log(float(np.exp(a - a_max).sum()))) / eta_prev
        else:
            mix = -float(dR.max())
        gap += max(0.0, -mix - float(w @ dR))
        eta = 16.0 * log_K / gap if gap > 0.0 else math.inf
        G += dR
        if math.isfinite(eta):
            z = eta * G
            w = np.exp(z - z.max())
        else:
            w = (G >= G.max() - 1e-12).astype(float)
        w /= w.sum()
        eta_prev = eta

        S_new = S + (mu * e_hat) ** 2
        A = 12.0 * lam * e_hat ** 2 + (math.sqrt(S_new) - math.sqrt(S)) / mu
        E += e_hat * (r - alpha) - C_A * A
        if E_max is not None:
            E = min(E, E_max)
        S = S_new
    return _result(len(conf), n_errors, n_audited, n_missed)


def oracle(conf, err, alpha):
    conf = np.asarray(conf, dtype=float)
    err = np.asarray(err, dtype=bool)
    tau = float(np.quantile(conf[err], 1.0 - alpha))
    audited = conf <= tau
    return _result(len(conf), int(err.sum()), int(audited.sum()), int((err & ~audited).sum()))
