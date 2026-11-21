"""Score the 4 hourly-activity models on the adversarial dataset and explain them under attack.

    python -m adversarial.evaluate_sequences          # needs data/synthetic and adversarial/data

Writes outputs/step3_adversarial/sequences/:
  metrics.csv                                   AUC (95% bootstrap interval), fakes caught, real users flagged
  attributions_<model>_<method>_<strength>.csv  per-hour credit for 50 humans, 50 bots, 50 trolls under
                                                the combined activity attack (none / mild / strong)
  signal_credit.csv                             share of credit on the planted hours (as moved by the attack)
  examples.json                                 one bot's and one troll's week at each strength, for the site
  meta.json

Models, standardisation (log1p, then the ORIGINAL training mean and std) and the explanation
settings are exactly those of step 2 (notebooks/03_explain_sequences.ipynb).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import xai_eval  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from captum.attr import GradientShap, IntegratedGradients, LayerAttribution, LayerGradCam  # noqa: E402

from xai_eval import loaders as L  # noqa: E402
from xai_eval import metrics as xm  # noqa: E402
from xai_eval.data import load_data  # noqa: E402

from .attacks import STRENGTHS  # noqa: E402
from .common import metric_row, sample_accounts  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

ADV = ROOT / "adversarial" / "data"
OUT = ROOT / "outputs" / "step3_adversarial" / "sequences"
SEED = 0
IG_STEPS = 64
GSHAP_SAMPLES = 20
GSHAP_BASELINES = 30
GSHAP_STDEV = 0.05
EXPLAINED_ATTACK = "activity_combined"
METHODS_FOR = {
    "cnn1d": ["Grad-CAM", "Integrated Gradients"],
    "rnn": ["Integrated Gradients", "GradientSHAP"],
    "lstm": ["Integrated Gradients", "GradientSHAP"],
    "autoencoder_lstm": ["Integrated Gradients", "GradientSHAP"],
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    D = load_data(ROOT)
    T = D.X_seq.shape[1]
    z0 = float((0.0 - D.seq_mu) / D.seq_sd)
    rng = np.random.default_rng(SEED)
    bg_series = D.X_seq[rng.choice(np.flatnonzero(D.is_tr), size=GSHAP_BASELINES, replace=False)]
    std = lambda counts: (np.log1p(counts.astype(float)) - D.seq_mu) / D.seq_sd  # noqa: E731

    manifest = json.loads((ADV / "manifest.json").read_text())
    base = pd.read_csv(ADV / "base" / "accounts.csv")
    kind = pd.read_csv(ADV / "base" / "truth" / "accounts_truth.csv")["kind"].to_numpy()
    split = base["split"].to_numpy()
    te = split == "test"
    y = base["label"].to_numpy()[te]
    thresholds = pd.read_csv(ROOT / "outputs" / "step1_fit" / "metrics.csv").set_index("model")["threshold"]

    variants = {"none": (np.load(ADV / "base" / "sequences.npz")["counts"], np.load(ADV / "base" / "truth" / "sequence_masks.npz")["masks"])}
    for v in manifest["activity_variants"]:
        variants[v] = (np.load(ADV / v / "sequences.npz")["counts"], np.load(ADV / v / "truth" / "sequence_masks.npz")["masks"])

    models = {n: L.load_model(n, D) for n in METHODS_FOR}
    for n, m in models.items():
        assert L.verify_reload(n, D, L.seq_scores(n, m, D.X_seq))["status"] == "OK", n

    rows = []
    for n, m in models.items():
        for v, (counts, _) in variants.items():
            s = L.seq_scores(n, m, std(counts[te]))
            rows.append(metric_row(n, v, y, s, thresholds[n], np.random.default_rng(SEED)))
        print(f"  scored {n}")
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)

    s_idx = sample_accounts(kind, split)
    fakes = np.flatnonzero(kind[s_idx] != "human")
    hours = [f"h{i}" for i in range(T)]
    credit_rows, timings, examples = [], {}, {}
    ex_ids = {k: int(s_idx[np.flatnonzero(kind[s_idx] == k)[0]]) for k in ("bot", "troll")}
    for strength in STRENGTHS:
        counts, masks = variants["none"] if strength == "none" else variants[f"{EXPLAINED_ATTACK}_{strength}"]
        X = std(counts[s_idx])
        M = masks[s_idx]
        for k, aid in ex_ids.items():
            examples.setdefault(k, {"account_id": aid})[strength] = {"counts": [int(c) for c in counts[aid]], "planted": [bool(b) for b in masks[aid]]}
        for n, model in models.items():
            fwd = L.seq_output_fn(n, model)
            for meth in METHODS_FOR[n]:
                x = L.seq_tensor(n, X).clone().requires_grad_(True)
                torch.manual_seed(SEED)
                t0 = time.time()
                if meth == "Grad-CAM":
                    cam = LayerGradCam(model, model.features).attribute(x, relu_attributions=True)
                    A = LayerAttribution.interpolate(cam, (T,), interpolate_mode="linear")
                elif meth == "Integrated Gradients":
                    A = IntegratedGradients(fwd).attribute(x, baselines=torch.full_like(x, z0), n_steps=IG_STEPS, internal_batch_size=64)
                else:
                    A = GradientShap(fwd).attribute(x, baselines=L.seq_tensor(n, bg_series), n_samples=GSHAP_SAMPLES, stdevs=GSHAP_STDEV)
                A = A.detach().numpy().reshape(len(X), T)
                timings[f"{n}/{meth}/{strength}"] = (time.time() - t0) / len(X)
                df = pd.DataFrame(A, columns=hours)
                df.insert(0, "kind", kind[s_idx])
                df.insert(0, "account_id", s_idx)
                df.to_csv(OUT / f"attributions_{n}_{meth.replace(' ', '_')}_{strength}.csv", index=False)
                absA = np.abs(A)
                mass = [xm.mass_inside(absA[i], M[i]) for i in fakes if M[i].any()]
                mauc = [xm.mask_auc(absA[i], M[i]) for i in fakes if M[i].any() and not M[i].all()]
                credit_rows.append({"model": n, "method": meth, "strength": strength,
                                    "credit_on_planted": float(np.mean(mass)), "planted_mask_auc": float(np.nanmean(mauc)),
                                    "planted_hour_share": float(M[fakes].mean())})
                for k, aid in ex_ids.items():
                    j = int(np.flatnonzero(s_idx == aid)[0])
                    peak = np.abs(A[j]).max()
                    examples[k][strength].setdefault("attributions", {}).setdefault(n, {})[meth] = [round(float(v / peak), 3) if peak > 0 else 0.0 for v in A[j]]
            print(f"  explained {n} ({strength})")
    pd.DataFrame(credit_rows).to_csv(OUT / "signal_credit.csv", index=False)
    (OUT / "examples.json").write_text(json.dumps(examples) + "\n")
    meta = {"seed": SEED, "n_per_kind": 50, "ig_steps": IG_STEPS, "gshap_samples": GSHAP_SAMPLES, "gshap_baselines": GSHAP_BASELINES,
            "gshap_stdev": GSHAP_STDEV, "z0": z0, "explained_attack": EXPLAINED_ATTACK, "methods": METHODS_FOR,
            "adversarial_seed": manifest["seed"], "evaluation_split": "test", "seconds_per_explanation": timings}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
