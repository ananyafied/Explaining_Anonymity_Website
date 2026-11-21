"""Collect the adversarial results for one model from outputs/step3_adversarial/ (Phase 3: account stats).

`adversarial_facts(key)` returns None for models that have no adversarial results yet.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ADV_OUT = ROOT / "outputs" / "step3_adversarial"
ADV_DATA = ROOT / "adversarial"
STRENGTHS = ("none", "mild", "strong")


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


@lru_cache(maxsize=None)
def attack_catalog() -> dict:
    """Attack names, features and costs, from adversarial/attacks.py (no data files needed)."""
    import sys
    sys.path.insert(0, str(ROOT))
    from adversarial.attacks import ATTACKS, COMBINED, changed_features

    return {
        "combined": COMBINED.key,
        "attacks": {k: {"name": a.name, "features": list(a.features), "changes": list(changed_features(a)), "cost": a.cost, "how": a.how}
                    for k, a in ATTACKS.items()},
    }


@lru_cache(maxsize=None)
def _metrics(family: str):
    p = ADV_OUT / family / "metrics.csv"
    return pd.read_csv(p) if p.exists() else None


def _shares(path: Path) -> dict:
    a = pd.read_csv(path)
    feats = [c for c in a.columns if c not in ("account_id", "kind")]
    v = a[feats].abs().mean()
    return {f: _num(x) for f, x in (v / v.sum()).items()}


def adversarial_facts(key: str, family: str = "tabular"):
    m = _metrics(family)
    if m is None or key not in set(m["model"]):
        return None
    m = m[m["model"] == key]
    cat = attack_catalog()
    none = m[m["variant"] == "none"].iloc[0]
    results = {"none": {"auc": _num(none.auc), "auc_ci": [_num(none.auc_ci_low), _num(none.auc_ci_high)],
                        "caught": _num(none.caught), "flagged_real": _num(none.flagged_real)}}
    for r in m[m["variant"] != "none"].itertuples():
        results.setdefault(r.attack, {})[r.strength] = {
            "auc": _num(r.auc), "auc_ci": [_num(r.auc_ci_low), _num(r.auc_ci_high)],
            "caught": _num(r.caught), "flagged_real": _num(r.flagged_real)}
    importance = {}
    for meth in ("SHAP", "LIME"):
        for s in STRENGTHS:
            p = ADV_OUT / family / f"attributions_{key}_{meth}_{s}.csv"
            if p.exists():
                importance.setdefault(meth, {})[s] = _shares(p)
    return {
        "family": family,
        "threshold": _num(none.threshold),
        "n": int(none.n),
        "n_fake": int(none.n_fake),
        "results": results,
        "importance": importance,
        **cat,
    }


def importance_shift(a: dict, b: dict) -> float:
    """Half the L1 distance between two credit-share vectors: 0 = identical, 1 = no overlap."""
    keys = sorted(set(a) | set(b))
    return float(0.5 * np.abs(np.array([a.get(k, 0) for k in keys]) - np.array([b.get(k, 0) for k in keys])).sum())
