"""Write the site's per-model data: site/public/data/models/<key>.json, and refresh catalog.json.

Run from anywhere:  python site/build/build_models.py
Needs data/synthetic/ (python -m anonsynth --n-accounts 5000 --seed 0 --out data/synthetic) and
outputs/step1_fit + outputs/step2_explain. Does not load any model, so it runs in seconds.
"""
from __future__ import annotations

import json
from pathlib import Path

import adversarial_facts as A
import build_catalog
import facts as F
import rules as R
import views as V
from catalog import BY_KEY, FAMILIES, MODELS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "site" / "public" / "data"

# What each scorecard criterion means, for the site's tooltips. Higher is better for all of them.
CRITERIA = {
    "correctness_informative_share": "Share of credit on the 6 features that actually drive the label",
    "correctness_rank": "Rank correlation between the credit and the true per-account contributions",
    "correctness_top5_precision": "Of the 5 top-ranked features, the share that truly matter",
    "correctness_mask_auc": "How well the credit picks out the planted burst hours (0.5 = chance)",
    "correctness_mask_auc_bot": "As above, bots only",
    "correctness_mask_auc_troll": "As above, trolls only",
    "correctness_mass_inside": "Share of credit inside the planted burst hours",
    "correctness_marker_top1": "How often a planted marker word is ranked first",
    "correctness_marker_mass": "Share of credit on planted marker words",
    "correctness_ring_edge_auc": "How well edge credit picks out a troll's ring edges (0.5 = chance)",
    "correctness_own_feature_rank": "Rank correlation between credit on the account's own stats and the true contributions",
    "correctness_same_ring_top3": "Share of the top-3 seed labels that come from the troll's own ring",
    "faithfulness": "Score change from deleting the top-ranked inputs, minus deleting random ones (in score standard deviations)",
    "stability_noise": "Rank agreement of explanations after a tiny change to the input",
    "stability_drop_word": "Rank agreement of explanations after deleting one unimportant word",
    "stability_drop_item": "Rank agreement of explanations after removing one unimportant item",
    "repeatability_seed": "Rank agreement of two runs with different random seeds",
    "shortcut_top1": "How often #trending is ranked first in posts that contain it",
    "shortcut_mass": "Share of credit on #trending in posts that contain it",
    "seconds_per_explanation": "Seconds to explain one input (lower is faster)",
}


def model_payload(key: str) -> dict:
    m = BY_KEY[key]
    f = F.model_facts(key)
    ev = R.evaluate(f)
    return {
        "key": key,
        "name": m.name,
        "family": m.family,
        "family_name": FAMILIES[m.family][0],
        "kind": m.kind,
        "input": None,  # filled from catalog.json below
        "methods": list(m.methods),
        "performance": {
            "test_auc": f["auc"], "test_auc_ci": f["auc_ci"], "val_auc": f["val_auc"], "train_auc": f["train_auc"],
            "test_pr_auc": f["pr_auc"], "reference_auc": f["ref_auc"], "reference_how": f["ref_how"],
        },
        "scorecard": {"per_method": f["per_method"], "agreement": f["agreement"], "model_level": f["model_level"]},
        "evidence": {k: f[k] for k in ("credit", "interaction", "timing", "reliance", "degree_alone_auc", "embedding_probe", "answer_key", "coalition_auc") if k in f},
        **ev,
        "view": V.model_view(key),
        "attack": attack_payload(key, m.family),
    }


def attack_payload(key: str, family: str):
    """Adversarial results, vulnerabilities and attack-specific checklist; None until the family is attacked."""
    af = A.adversarial_facts(key, family)
    if af is None:
        return None
    found = R.attack_findings(af)
    return {
        "attacks": af["attacks"],
        "combined": af["combined"],
        "threshold": af["threshold"],
        "n": af["n"],
        "n_fake": af["n_fake"],
        "results": af["results"],
        "importance": af["importance"],
        "signal_credit": af["signal_credit"],
        "examples": af["examples"],
        "strengths": [x for x in found if x["type"] == "strength"],
        "weaknesses": [x for x in found if x["type"] == "weakness"],
        "checklist": R.attack_checklist(found),
    }


def main() -> None:
    build_catalog.main()
    cat_path = OUT / "catalog.json"
    cat = json.loads(cat_path.read_text())
    inputs = {m["key"]: m["input"] for m in cat["models"]}
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    labels = {}
    for m in MODELS:
        p = model_payload(m.key)
        p["input"] = inputs[m.key]
        labels[m.key] = p["verdict"]["label"]
        (OUT / "models" / f"{m.key}.json").write_text(json.dumps(p, separators=(",", ":")) + "\n")
    # add the verdict to the catalog so the model picker can show it
    for m in cat["models"]:
        m["verdict"] = labels[m["key"]]
    cat["criteria"] = CRITERIA
    cat["thresholds"] = R.THRESHOLDS
    cat_path.write_text(json.dumps(cat, indent=1) + "\n")
    good = [k for k, v in labels.items() if v == "good"]
    print(f"wrote {len(labels)} model files; good: {', '.join(good)}")


if __name__ == "__main__":
    main()
