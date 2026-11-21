"""Helpers shared by the adversarial evaluation scripts."""
from __future__ import annotations

import numpy as np

N_BOOT = 1000


def auc_fast(y, s):
    """ROC AUC via ranks (ties get their average rank)."""
    s = np.asarray(s, float)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / counts)[inv]
    n1 = y.sum()
    n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auc_ci(y, s, rng, n_boot=N_BOOT):
    """95% percentile bootstrap interval of the AUC."""
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    vals = [auc_fast(y[i], s[i]) for i in idx if 0 < y[i].sum() < len(i)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def metric_row(model, variant, y, s, threshold, rng):
    attack, strength = ("none", "none") if variant == "none" else variant.rsplit("_", 1)
    lo, hi = auc_ci(y, s, rng)
    pred = s >= threshold
    return {"model": model, "variant": variant, "attack": attack, "strength": strength,
            "auc": auc_fast(y, s), "auc_ci_low": lo, "auc_ci_high": hi,
            "caught": float(pred[y == 1].mean()), "flagged_real": float(pred[y == 0].mean()),
            "threshold": float(threshold), "n": int(len(y)), "n_fake": int(y.sum())}


def sample_accounts(kind, split, per_kind=50, seed=0):
    """Same rule as xai_eval.data.Data.sample_accounts: per_kind test accounts of each kind."""
    rng = np.random.default_rng(seed)
    ok = split == "test"
    return np.concatenate([rng.choice(np.flatnonzero(ok & (kind == k)), size=per_kind, replace=False) for k in ("human", "bot", "troll")])
