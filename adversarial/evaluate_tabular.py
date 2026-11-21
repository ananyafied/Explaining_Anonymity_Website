"""Score the 11 account-stat models on the adversarial dataset and explain them under attack.

    python -m adversarial.evaluate_tabular            # needs data/synthetic and adversarial/data

Writes outputs/step3_adversarial/tabular/:
  metrics.csv                                   AUC (with 95% bootstrap interval), share of fakes
                                                caught and of real users flagged, per model x variant
  attributions_<model>_<method>_<strength>.csv  SHAP / LIME for the combined attack at none, mild
                                                and strong, on 50 humans, 50 bots and 50 trolls
  meta.json                                     settings, timings and library versions

The models are used exactly as trained: same files, and the ORIGINAL training data's scaling and
background (a deployed model does not re-fit its scaler on attack traffic). SHAP and LIME use the
step-2 settings (notebooks/02_explain_tabular.ipynb), except that LIME is read from the correct
label (step 2 read a sign-flipped copy; see docs/site_rules.md section 6).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import xai_eval  # noqa: E402,F401  (imports xgboost / lightgbm before torch: macOS OpenMP workaround)
import lightgbm  # noqa: E402,F401
import lime.lime_tabular  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import xgboost as xgb  # noqa: E402

from xai_eval import loaders as L  # noqa: E402
from xai_eval.data import load_data  # noqa: E402

from .attacks import ATTACKS, STRENGTHS  # noqa: E402

warnings.filterwarnings("ignore")

ADV = ROOT / "adversarial" / "data"
OUT = ROOT / "outputs" / "step3_adversarial" / "tabular"

# step-2 settings (notebooks/02_explain_tabular.ipynb)
SEED = 0
N_PER_KIND = 50
N_BACKGROUND = 100
SHAP_NSAMPLES = 1000
LIME_NSAMPLES = 1500
N_BOOT = 1000
EXPLAINED_ATTACK = "combined"


def auc_fast(y, s):
    """ROC AUC via ranks (ties get average rank)."""
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    n1 = y.sum()
    n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auc_ci(y, s, rng):
    idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))
    vals = [auc_fast(y[i], s[i]) for i in idx if 0 < y[i].sum() < len(i)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def sample_accounts(kind, split, per_kind=N_PER_KIND, seed=SEED):
    """Same rule as xai_eval.data.Data.sample_accounts: per_kind test accounts of each kind."""
    rng = np.random.default_rng(seed)
    ok = split == "test"
    return np.concatenate([rng.choice(np.flatnonzero(ok & (kind == k)), size=per_kind, replace=False) for k in ("human", "bot", "troll")])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    D = load_data(ROOT)  # original data: models, scaling, SHAP background, LIME training data
    feats = D.FEATURES
    manifest = json.loads((ADV / "manifest.json").read_text())
    base = pd.read_csv(ADV / "base" / "accounts.csv")
    kind = pd.read_csv(ADV / "base" / "truth" / "accounts_truth.csv")["kind"].to_numpy()
    split = base["split"].to_numpy()
    te = split == "test"
    y = base["label"].to_numpy()[te]
    thresholds = pd.read_csv(ROOT / "outputs" / "step1_fit" / "metrics.csv").set_index("model")["threshold"]

    variants = {"none": base}
    for v in manifest["variants"]:
        variants[v] = pd.read_csv(ADV / v / "accounts.csv")
    for v, acc in variants.items():
        assert (acc["account_id"].to_numpy() == base["account_id"].to_numpy()).all(), v
        assert (acc["label"].to_numpy() == base["label"].to_numpy()).all(), v

    models = {n: L.load_model(n, D) for n in L.TABULAR_MODELS if n != "svd_logreg"}
    scorers = {n: L.tabular_scorer(n, m, D) for n, m in models.items()}
    for n, f in scorers.items():  # the same models step 1 evaluated
        chk = L.verify_reload(n, D, f(D.X_tab))
        assert chk["status"] == "OK", n

    # ---- accuracy on every variant
    rows = []
    for n, f in scorers.items():
        thr = float(thresholds[n])
        for v, acc in variants.items():
            s = f(acc.loc[te, feats].to_numpy(float))
            attack, strength = ("none", "none") if v == "none" else v.rsplit("_", 1)
            lo, hi = auc_ci(y, s, np.random.default_rng(SEED))
            pred = s >= thr
            rows.append({"model": n, "variant": v, "attack": attack, "strength": strength,
                         "auc": auc_fast(y, s), "auc_ci_low": lo, "auc_ci_high": hi,
                         "caught": float(pred[y == 1].mean()), "flagged_real": float(pred[y == 0].mean()),
                         "threshold": thr, "n": int(te.sum()), "n_fake": int(y.sum())})
        print(f"  scored {n}")
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)

    # ---- SHAP and LIME under the combined attack
    s_idx = sample_accounts(kind, split)
    bg = shap.kmeans(D.X_tab[L.make_background(D, k=N_BACKGROUND, seed=SEED)], 10)
    lime_ex = lime.lime_tabular.LimeTabularExplainer(D.X_tab[D.is_tr], feature_names=feats, mode="regression", random_state=SEED)
    timings = {}
    for strength in STRENGTHS:
        acc = variants["none"] if strength == "none" else variants[f"{EXPLAINED_ATTACK}_{strength}"]
        X = acc.loc[s_idx, feats].to_numpy(float)
        for n, f in scorers.items():
            t0 = time.time()
            if n == "xgboost":  # XGBoost's own exact TreeSHAP (log-odds), as in step 2
                A = models[n].get_booster().predict(xgb.DMatrix(pd.DataFrame(X, columns=feats)), pred_contribs=True)[:, :-1]
            elif n in L.TREE_MODELS:
                sv = shap.TreeExplainer(models[n]).shap_values(pd.DataFrame(X, columns=feats))
                sv = np.asarray(sv[-1]) if isinstance(sv, list) else np.asarray(sv)
                A = sv[:, :, -1] if sv.ndim == 3 else sv
            else:
                np.random.seed(SEED)
                A = np.asarray(shap.KernelExplainer(f, bg).shap_values(X, nsamples=SHAP_NSAMPLES, silent=True))
            t_shap = time.time() - t0
            t0 = time.time()
            B = np.zeros_like(X)
            for i, x in enumerate(X):
                e = lime_ex.explain_instance(x, f, num_features=len(feats), num_samples=LIME_NSAMPLES)
                for j, w in e.local_exp[1]:  # label 1 = the explanation (label 0 is LIME's negated copy)
                    B[i, j] = w
            t_lime = time.time() - t0
            for meth, M in (("SHAP", A), ("LIME", B)):
                df = pd.DataFrame(M, columns=feats)
                df.insert(0, "kind", kind[s_idx])
                df.insert(0, "account_id", s_idx)
                df.to_csv(OUT / f"attributions_{n}_{meth}_{strength}.csv", index=False)
            timings[f"{n}/{strength}"] = {"SHAP": t_shap / len(X), "LIME": t_lime / len(X)}
            print(f"  explained {n} ({strength}): SHAP {t_shap:.1f}s, LIME {t_lime:.1f}s")

    meta = {"seed": SEED, "n_per_kind": N_PER_KIND, "n_background": N_BACKGROUND, "shap_nsamples": SHAP_NSAMPLES,
            "lime_nsamples": LIME_NSAMPLES, "n_boot": N_BOOT, "explained_attack": EXPLAINED_ATTACK,
            "adversarial_seed": manifest["seed"], "evaluation_split": "test",
            "libraries": {"shap": shap.__version__, "lime": "0.2.0.1", "xgboost": xgb.__version__, "lightgbm": lightgbm.__version__},
            "seconds_per_explanation": timings}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
