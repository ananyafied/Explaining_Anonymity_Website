"""Collect the numbers the rules need for each model, from outputs/step1_fit and outputs/step2_explain.

`model_facts(key)` returns a plain dict (JSON-safe). Nothing here judges a model; that is rules.py.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from catalog import BY_KEY, MODELS

ROOT = Path(__file__).resolve().parents[2]
STEP1 = ROOT / "outputs" / "step1_fit"
STEP2 = ROOT / "outputs" / "step2_explain"
DATA = ROOT / "data" / "synthetic"

# Family whose models a multimodal model is compared with (it reads all three of these inputs).
MULTIMODAL_INPUT_FAMILIES = ("tabular", "text", "graph")


def _num(x):
    """float, or None for NaN / missing (keeps the JSON clean)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


@lru_cache(maxsize=None)
def metrics_table() -> pd.DataFrame:
    return pd.read_csv(STEP1 / "metrics.csv").set_index("model")


@lru_cache(maxsize=None)
def scorecard(family_dir: str) -> pd.DataFrame:
    return pd.read_csv(STEP2 / family_dir / "scorecard.csv")


@lru_cache(maxsize=None)
def bayes_test_auc() -> float:
    """AUC of the true probability on the test split: the best any account-stats model can do."""
    a = pd.read_csv(DATA / "accounts.csv")
    t = pd.read_csv(DATA / "truth" / "accounts_truth.csv")
    m = a["split"] == "test"
    return float(roc_auc_score(a.loc[m, "label"], t.loc[m, "true_prob"]))


def reference_auc(family: str) -> tuple[float, str]:
    """(reference AUC, how it was obtained) for judging accuracy within a family."""
    if family == "tabular":
        return bayes_test_auc(), "best achievable from the account stats (the true formula)"
    fams = MULTIMODAL_INPUT_FAMILIES if family == "multimodal" else (family,)
    best = max((m for m in MODELS if m.family in fams), key=lambda m: metrics_table().loc[m.key, "test_roc_auc"])
    where = "the models that read the same inputs" if family == "multimodal" else "this family"
    return float(metrics_table().loc[best.key, "test_roc_auc"]), f"best test AUC in {where} ({best.name})"


def _method_criteria(key: str, family_dir: str):
    """{method: {criterion: value}}, {criterion: value} for the comparison of the two methods, model-level facts."""
    sc = scorecard(family_dir)
    sc = sc[sc["model"] == key]
    methods, agreement, model_level = {}, {}, {}
    for r in sc.itertuples():
        v = _num(r.value)
        if r.method == "model (no explainer)":
            model_level[r.criterion] = v
        elif " vs " in r.method:
            agreement[r.criterion] = v
        else:
            methods.setdefault(r.method, {})[r.criterion] = v
    return methods, agreement, model_level


def _tabular_credit(key: str) -> dict:
    df = pd.read_csv(STEP2 / "tabular" / "credit_by_feature_group.csv")
    out = {}
    for r in df[df["model"] == key].itertuples(index=False):
        out[r.method] = {"informative": _num(r[2]), "proxy": _num(r[3]), "decoys": _num(r[4]), "noise": _num(r[5])}
    return out


INTERACTION = ("burstiness", "account_age_days")


def _tabular_interaction(key: str, methods) -> dict:
    """Share of |attribution| on the two interaction features, per method, and the true share."""
    out = {}
    for meth in methods:
        a = pd.read_csv(STEP2 / "tabular" / f"attributions_{key}_{meth}.csv")
        feats = [c for c in a.columns if c not in ("account_id", "kind")]
        v = a[feats].abs().mean()
        out[meth] = _num(v[list(INTERACTION)].sum() / v.sum())
    ids = a["account_id"].to_numpy()
    t = pd.read_csv(DATA / "truth" / "accounts_truth.csv").set_index("account_id").loc[ids]
    terms = ["term_posts", "term_reply", "term_burst_x_young", "term_night_hinge", "term_url"]
    mean_abs = t[terms].abs().mean()
    # The interaction term's own share of the signal. SHAP splits an interaction's credit between its
    # two inputs, so the pair's combined SHAP share is comparable to this (not to the doubled
    # "credited to both inputs" convention used for correctness scoring).
    out["true"] = _num(mean_abs["term_burst_x_young"] / mean_abs.sum())
    return out


def model_facts(key: str) -> dict:
    m = BY_KEY[key]
    row = metrics_table().loc[key]
    ref, ref_how = reference_auc(m.family)
    methods, agreement, model_level = _method_criteria(key, m.family)
    f = {
        "key": key,
        "family": m.family,
        "kind": m.kind,
        "methods": list(m.methods),
        "auc": _num(row["test_roc_auc"]),
        "auc_ci": [_num(row["auc_ci_low"]), _num(row["auc_ci_high"])],
        "val_auc": _num(row["val_roc_auc"]),
        "train_auc": _num(row["train_roc_auc"]),
        "pr_auc": _num(row["test_pr_auc"]),
        "ref_auc": ref,
        "ref_how": ref_how,
        "per_method": methods,
        "agreement": agreement,
        "model_level": model_level,
    }
    # family-specific evidence
    if m.family == "tabular":
        f["credit"] = _tabular_credit(key)
        f["interaction"] = _tabular_interaction(key, m.methods)
    elif m.family == "sequences":
        tp = pd.read_csv(STEP2 / "sequences" / "timing_probe.csv").set_index("model")
        f["timing"] = {"auc_shuffled_hours": _num(tp.loc[key, "auc_shuffled_hours"]), "auc_drop": _num(tp.loc[key, "auc_drop"])}
    elif m.family == "text":
        rp = pd.read_csv(STEP2 / "text" / "reliance_probe.csv").set_index("model")
        f["reliance"] = {c: _num(rp.loc[key, c]) for c in rp.columns}
    elif m.family == "graph":
        rp = pd.read_csv(STEP2 / "graph" / "reliance_probe.csv").set_index("model")
        f["reliance"] = {c: _num(rp.loc[key, c]) for c in rp.columns}
        f["degree_alone_auc"] = _num(rp.loc["(degree alone)", "auc"])
        if key in ("node2vec_logreg", "deepwalk_logreg"):
            ep = pd.read_csv(STEP2 / "graph" / "embedding_probe.csv").set_index("model")
            f["embedding_probe"] = {c: _num(ep.loc[key, c]) for c in ep.columns}
    elif m.family == "multimodal":
        ak = pd.read_csv(STEP2 / "multimodal" / "block_vs_answer_key.csv").set_index("method")
        f["answer_key"] = {meth: {c: _num(ak.loc[meth, c]) for c in ak.columns} for meth in ak.index}
        co = pd.read_csv(STEP2 / "multimodal" / "coalition_auc.csv")
        f["coalition_auc"] = {r.present: _num(r.test_auc) for r in co.itertuples()}
    return f


def all_facts() -> dict:
    return {m.key: model_facts(m.key) for m in MODELS}


if __name__ == "__main__":
    print(json.dumps(model_facts("lightgbm"), indent=1)[:1500])
