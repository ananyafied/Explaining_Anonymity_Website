"""Reload every model the website shows and check it reproduces its step-1 predictions.

Run from anywhere:  python site/build/verify_models.py
Needs data/synthetic/ (python -m anonsynth --n-accounts 5000 --seed 0 --out data/synthetic).
Writes site/build/reload_checks.csv and exits non-zero if any model does not match.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import warnings  # noqa: E402

# LightGBM must be imported before torch: on macOS they bundle different OpenMP runtimes, and
# unpickling the LightGBM model after torch is loaded segfaults.
import lightgbm  # noqa: E402,F401
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from xai_eval import loaders as L  # noqa: E402
from xai_eval.data import load_data  # noqa: E402

from catalog import MODELS  # noqa: E402

warnings.filterwarnings("ignore")


def scores_for(name: str, D, cache: dict):
    """Recompute the step-1 score for every account, exactly as the step-2 notebooks do."""
    if name in L.TABULAR_MODELS:
        m = L.load_model(name, D)
        return L.tabular_scorer(name, m, D)(D.X_tab), {}
    if name in L.SEQ_MODELS:
        return L.seq_scores(name, L.load_model(name, D), D.X_seq), {}
    if name in L.TEXT_MODELS:
        ts = L.TextScorer(name, L.load_model(name, D), D)
        return ts.account_scores(D), ({"loose": True} if name == "doc2vec_logreg" else {})
    if name in L.GNN_MODELS:
        x = torch.as_tensor(D.X_tab_std, dtype=torch.float32)
        return L.sigmoid(L.gnn_logits(L.load_model(name, D), x, D.EDGE_INDEX)), {}
    if name in L.EMBED_MODELS:
        emb = L.embedding_matrix(L.load_model(name, D), D.n)
        return L.load_extra(name, D)["classifier"].predict_proba(emb)[:, 1], {}
    if name == "label_spreading":
        return L.LabelSpreading(D, L.load_model(name, D)).all_scores(), {"atol": 1e-6}
    if name == "multimodal_fusion":
        transformer = L.load_model("transformer", D)
        fusion = L.load_model(name, D)
        x_tab, x_txt = L.fusion_inputs(D, transformer)
        with torch.no_grad():
            return L.sigmoid(fusion(x_tab, x_txt, D.EDGE_INDEX).numpy()), {}
    raise KeyError(name)


def main() -> int:
    D = load_data(ROOT)
    rows = []
    for m in MODELS:
        s, kw = scores_for(m.key, D, {})
        rows.append({**L.verify_reload(m.key, D, s, **kw), "family": m.family})
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).with_name("reload_checks.csv"), index=False)
    bad = df[df.status != "OK"]
    print(f"\n{len(df) - len(bad)}/{len(df)} models reproduce their step-1 predictions.")
    return 1 if len(bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
