"""Checks on the site's chart data (site/build/views.py). Needs outputs/step2_explain only."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "site" / "build"))

import views as V  # noqa: E402


def test_lime_sign_is_corrected_for_supervised_tabular_models():
    """After the fix, LIME and SHAP should agree in sign on most clearly non-zero attributions."""
    for key in ("lightgbm", "xgboost", "rf", "svm_linear"):
        s = V.load_tabular_attributions(key, "SHAP")
        l = V.load_tabular_attributions(key, "LIME")
        feats = [c for c in s.columns if c not in ("account_id", "kind")]
        S, L = s[feats].to_numpy(), l[feats].to_numpy()
        big = (np.abs(S) > 0.01) & (np.abs(L) > 0.01)
        assert (np.sign(S[big]) == np.sign(L[big])).mean() > 0.7, key
