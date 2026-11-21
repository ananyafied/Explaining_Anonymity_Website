"""Collect the adversarial results for one model from outputs/step3_adversarial/<family>/.

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
def attack_catalog(family: str) -> dict:
    """Attack names, what they change and their cost, from adversarial/attacks.py (no data files needed)."""
    import sys
    sys.path.insert(0, str(ROOT))
    from adversarial.attacks import FAMILY_ATTACKS, changed_features

    attacks, combined = FAMILY_ATTACKS[family]
    return {
        "combined": combined,
        "attacks": {k: {"name": a.name, "features": list(a.features), "changes": list(changed_features(a)), "cost": a.cost, "how": a.how}
                    for k, a in attacks.items()},
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
    cat = attack_catalog(family)
    none = m[m["variant"] == "none"].iloc[0]
    results = {"none": {"auc": _num(none.auc), "auc_ci": [_num(none.auc_ci_low), _num(none.auc_ci_high)],
                        "caught": _num(none.caught), "flagged_real": _num(none.flagged_real)}}
    for r in m[m["variant"] != "none"].itertuples():
        results.setdefault(r.attack, {})[r.strength] = {
            "auc": _num(r.auc), "auc_ci": [_num(r.auc_ci_low), _num(r.auc_ci_high)],
            "caught": _num(r.caught), "flagged_real": _num(r.flagged_real)}
    importance, signal, examples = {}, None, None
    if family == "tabular":
        for meth in ("SHAP", "LIME"):
            for s in STRENGTHS:
                p = ADV_OUT / family / f"attributions_{key}_{meth}_{s}.csv"
                if p.exists():
                    importance.setdefault(meth, {})[s] = _shares(p)
    else:
        # share of credit on the planted signal (burst hours / marker words) as the attack moves it
        sc = pd.read_csv(ADV_OUT / family / "signal_credit.csv")
        sc = sc[sc["model"] == key]
        col = "credit_on_planted" if family == "sequences" else "credit_on_markers"
        signal = {meth: {r.strength: _num(getattr(r, col)) for r in g.itertuples()} for meth, g in sc.groupby("method", sort=False)}
        if family == "sequences":
            signal["_planted_hour_share"] = {r.strength: _num(r.planted_hour_share) for r in sc.drop_duplicates("strength").itertuples()}
        ex = json.loads((ADV_OUT / family / "examples.json").read_text())
        if family == "sequences":
            examples = {k: {"account_id": v["account_id"], **{s: {"counts": v[s]["counts"], "planted": v[s]["planted"],
                                                                   "attributions": v[s]["attributions"][key]} for s in STRENGTHS}}
                        for k, v in ex.items()}
        else:
            examples = {s: {kk: vv for kk, vv in v.items() if kk != "attributions"} | {"attributions": v["attributions"][key]} for s, v in ex.items()}
    step1 = pd.read_csv(ROOT / "outputs" / "step1_fit" / "metrics.csv").set_index("model")
    return {
        "family": family,
        "original_test_auc": _num(step1.loc[key, "test_roc_auc"]),
        "threshold": _num(none.threshold),
        "n": int(none.n),
        "n_fake": int(none.n_fake),
        "results": results,
        "importance": importance,
        "signal_credit": signal,
        "examples": examples,
        **cat,
    }


def importance_shift(a: dict, b: dict) -> float:
    """Half the L1 distance between two credit-share vectors: 0 = identical, 1 = no overlap."""
    keys = sorted(set(a) | set(b))
    return float(0.5 * np.abs(np.array([a.get(k, 0) for k in keys]) - np.array([b.get(k, 0) for k in keys])).sum())
