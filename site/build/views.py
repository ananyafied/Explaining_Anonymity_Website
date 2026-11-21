"""Family-specific data the site draws: importance from each method, the answer key, and examples.

`model_view(key)` returns a JSON-safe dict. Examples are the first human, bot and troll in each
family's fixed explained sample (the same accounts for every model in the family).
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from catalog import BY_KEY

ROOT = Path(__file__).resolve().parents[2]
STEP1 = ROOT / "outputs" / "step1_fit"
STEP2 = ROOT / "outputs" / "step2_explain"
DATA = ROOT / "data" / "synthetic"
KINDS = ("human", "bot", "troll")

# Which planted logit term each profile feature belongs to (same as xai_eval.data.TERM_TO_FEATURES).
TERM_TO_FEATURES = {
    "term_posts": ["posts_per_day"],
    "term_reply": ["reply_fraction"],
    "term_burst_x_young": ["burstiness", "account_age_days"],
    "term_night_hinge": ["night_activity_frac"],
    "term_url": ["url_fraction"],
}
ROLE_GROUP = {"informative_linear": "informative", "nonlinear_threshold": "informative", "distractor": "decoy", "noise": "noise"}


def _r(x, d=4):
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else round(x, d)


def _rl(a, d=4):
    return [_r(v, d) for v in np.asarray(a, float).ravel()]


@lru_cache(maxsize=None)
def ground_truth():
    return json.loads((DATA / "truth" / "ground_truth.json").read_text())


@lru_cache(maxsize=None)
def accounts():
    return pd.read_csv(DATA / "accounts.csv").set_index("account_id")


@lru_cache(maxsize=None)
def truth():
    return pd.read_csv(DATA / "truth" / "accounts_truth.csv").set_index("account_id")


@lru_cache(maxsize=None)
def scores(key):
    return pd.read_csv(STEP1 / "predictions" / f"{key}.csv").set_index("account_id")["score"]


def _first_per_kind(df):
    return {k: int(df.loc[df["kind"] == k, "account_id"].iloc[0]) for k in KINDS if (df["kind"] == k).any()}


# ------------------------------------------------------------------ account stats
def feature_roles():
    out = []
    for name, role in ground_truth()["tabular"]["feature_roles"].items():
        if role.startswith("interaction"):
            group = "informative"
        elif role.startswith("redundant_proxy"):
            group = "proxy"
        else:
            group = ROLE_GROUP[role]
        out.append({"name": name, "role": role, "group": group})
    return out


def _true_contrib(ids, features):
    """Signed planted logit contribution of each feature for each account (interaction credited to both inputs)."""
    t = truth().loc[ids]
    out = pd.DataFrame(0.0, index=ids, columns=features)
    for term, feats in TERM_TO_FEATURES.items():
        for f in feats:
            out[f] += t[term].to_numpy()
    return out


def tabular_view(key):
    feats = [f["name"] for f in feature_roles()]
    m = BY_KEY[key]
    attr = {meth: pd.read_csv(STEP2 / "tabular" / f"attributions_{key}_{meth}.csv") for meth in m.methods}
    first = attr[m.methods[0]]
    ids = first["account_id"].to_numpy()
    true = _true_contrib(ids, feats)

    def shares(a):
        v = a[feats].abs().mean()
        return {f: _r(x) for f, x in (v / v.sum()).items()}

    importance = {meth: shares(a) for meth, a in attr.items()}
    by_kind = {meth: {k: shares(a[a["kind"] == k]) for k in KINDS} for meth, a in attr.items()}
    tv = true.abs().mean()
    examples = []
    for kind, aid in _first_per_kind(first).items():
        examples.append({
            "account_id": aid,
            "kind": kind,
            "score": _r(scores(key).loc[aid]),
            "values": {f: _r(accounts().loc[aid, f]) for f in feats},
            "attributions": {meth: {f: _r(a.set_index("account_id").loc[aid, f], 5) for f in feats} for meth, a in attr.items()},
            "true_contribution": {f: _r(true.loc[aid, f]) for f in feats},
        })
    return {
        "features": feature_roles(),
        "formula": ground_truth()["tabular"]["formula"],
        "n_explained": int(len(ids)),
        "importance": importance,
        "importance_by_kind": by_kind,
        "true_importance": {f: _r(x) for f, x in (tv / tv.sum()).items()},
        "examples": examples,
    }


# ------------------------------------------------------------------ hourly activity
@lru_cache(maxsize=None)
def _seq_arrays():
    counts = np.load(DATA / "sequences.npz")["counts"]
    masks = np.load(DATA / "truth" / "sequence_masks.npz")["masks"].astype(bool)
    return counts, masks


def sequences_view(key):
    m = BY_KEY[key]
    counts, masks = _seq_arrays()
    hours = [f"h{i}" for i in range(168)]
    attr = {meth: pd.read_csv(STEP2 / "sequences" / f"attributions_{key}_{meth.replace(' ', '_')}.csv") for meth in m.methods}
    first = attr[m.methods[0]]
    examples = []
    for kind, aid in _first_per_kind(first).items():
        ex = {"account_id": aid, "kind": kind, "score": _r(scores(key).loc[aid]),
              "counts": [int(c) for c in counts[aid]], "planted": [bool(b) for b in masks[aid]], "attributions": {}}
        for meth, a in attr.items():
            v = a.set_index("account_id").loc[aid, hours].to_numpy(float)
            peak = np.abs(v).max()
            ex["attributions"][meth] = _rl(v / peak if peak > 0 else v, 3)  # scaled to [-1, 1] for drawing
        examples.append(ex)
    # share of attribution inside the planted hours, vs the share of hours that are planted (bots + trolls)
    inauth = first[first["kind"] != "human"]["account_id"].to_numpy()
    planted_share = float(masks[inauth].mean())
    mass = {}
    for meth, a in attr.items():
        A = np.abs(a.set_index("account_id").loc[inauth, hours].to_numpy(float))
        mass[meth] = _r(float(np.mean((A * masks[inauth]).sum(1) / np.maximum(A.sum(1), 1e-12))))
    gt = ground_truth()["sequences"]
    return {
        "n_hours": 168,
        "planted_pattern": {"bot_periods": gt["bot_periods"], "troll_window_len": gt["window_len"]},
        "planted_hour_share": _r(planted_share),
        "attribution_inside_planted": mass,
        "examples": examples,
    }


# ------------------------------------------------------------------ post text
def text_view(key):
    gt = ground_truth()["text"]
    markers = set(gt["bot_markers"]) | set(gt["troll_markers"])
    trending = gt["spurious_token"]
    a = pd.read_csv(STEP2 / "text" / f"attributions_{key}.csv")
    top_words = {}
    for meth, g in a.groupby("method"):
        s = g.assign(abs=g["attribution"].abs()).groupby("word").agg(mean=("attribution", "mean"), mean_abs=("abs", "mean"), n=("attribution", "size"))
        s = s[s["n"] >= 3].sort_values("mean_abs", ascending=False).head(12)
        top_words[meth] = [{"word": w, "mean": _r(r["mean"]), "mean_abs": _r(r["mean_abs"]), "n": int(r["n"]),
                            "is_marker": w in markers, "is_trending": w == trending} for w, r in s.iterrows()]
    examples = []
    for stratum in ("marker", "spurious_only", "neutral"):
        g = a[a["stratum"] == stratum]
        if g.empty:
            continue
        pid = int(g["post_id"].iloc[0])
        p = a[a["post_id"] == pid]
        words = list(dict.fromkeys(p["word"]))
        examples.append({
            "post_id": pid,
            "stratum": stratum,
            "words": words,
            "attributions": {meth: [_r(q.set_index("word").loc[w, "attribution"], 5) for w in words] for meth, q in p.groupby("method")},
        })
    return {"markers": sorted(markers), "trending_token": trending, "top_words": top_words, "examples": examples}


# ------------------------------------------------------------------ follower graph
def graph_view(key):
    out = {}
    if key in ("gcn", "gat", "graphsage", "rgcn"):
        a = pd.read_csv(STEP2 / "graph" / f"attributions_{key}.csv")
        t = a[a["kind"] == "troll"]
        ring = {}
        for meth, g in t.groupby("method"):
            # normalise per account so accounts with big neighbourhoods do not dominate
            g = g.assign(norm=g["importance"].abs() / g.groupby("account_id")["importance"].transform(lambda x: x.abs().max()).clip(lower=1e-12))
            ring[meth] = {"ring_edges": _r(g.loc[g["is_ring"], "norm"].mean()), "other_edges": _r(g.loc[~g["is_ring"], "norm"].mean()),
                          "n_ring": int(g["is_ring"].sum()), "n_other": int((~g["is_ring"]).sum())}
        out["troll_edge_importance"] = ring
    return out


# ------------------------------------------------------------------ multimodal
def multimodal_view(key):
    bk = pd.read_csv(STEP2 / "multimodal" / "block_values_by_kind.csv")
    shares = {meth: {k: {r.block: _r(r.mean_share) for r in g[g["kind"] == k].itertuples()} for k in KINDS} for meth, g in bk.groupby("method")}
    return {"blocks": {"tab": "Account stats", "text": "Post text", "graph": "Follower graph"}, "block_share_by_kind": shares}


VIEWS = {"tabular": tabular_view, "sequences": sequences_view, "text": text_view, "graph": graph_view, "multimodal": multimodal_view}


def model_view(key):
    return VIEWS[BY_KEY[key].family](key)
