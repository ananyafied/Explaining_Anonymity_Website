"""Tests for the website's labelling rules (site/build/rules.py). Synthetic facts only, no data files."""
import copy
import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "site" / "build"))

import rules as R  # noqa: E402

T = R.THRESHOLDS


def tabular(**over):
    f = {
        "key": "toy", "family": "tabular", "kind": "classifier", "methods": ["SHAP", "LIME"],
        "auc": 0.85, "val_auc": 0.86, "train_auc": 0.90, "ref_auc": 0.873, "ref_how": "best achievable",
        "per_method": {
            "SHAP": {"faithfulness": 0.6, "repeatability_seed": 1.0, "stability_noise": 0.95},
            "LIME": {"faithfulness": 0.6, "repeatability_seed": 0.9, "stability_noise": 0.90},
        },
        "agreement": {"agreement_spearman": 0.8},
        "model_level": {},
        "credit": {"SHAP": {"informative": 0.85, "proxy": 0.05, "decoys": 0.05, "noise": 0.05}},
        "interaction": {"SHAP": 0.18, "LIME": 0.17, "true": 0.19},
    }
    f.update(over)
    return f


def ids(found, kind=None):
    return [x["id"] for x in found if kind is None or x["type"] == kind]


def test_clean_model_is_good():
    ev = R.evaluate(tabular())
    assert ev["verdict"]["label"] == "good"
    assert all(c["passed"] for c in ev["verdict"]["checks"])
    assert set(ids(ev["strengths"])) >= {"accurate", "right_features", "methods_agree", "faithful"}
    assert ev["weaknesses"] == []


def test_each_check_can_fail_the_verdict():
    floor = 0.873 - T["auc_margin"]
    assert R.evaluate(tabular(auc=floor - 0.001))["verdict"]["label"] == "weak"
    assert R.evaluate(tabular(auc=floor + 0.001, val_auc=floor + 0.001))["verdict"]["label"] == "good"
    assert R.evaluate(tabular(val_auc=0.85 + T["val_test_gap"] + 0.01))["verdict"]["label"] == "weak"
    f = tabular()
    f["credit"]["SHAP"]["informative"] = T["informative_share"] - 0.01
    assert R.evaluate(f)["verdict"]["label"] == "weak"


def test_accuracy_wording_tiers():
    assert "near_chance" in ids(R.findings(tabular(auc=0.55, val_auc=0.55)))
    assert "inaccurate" in ids(R.findings(tabular(auc=0.80, val_auc=0.80)))
    assert "accurate" in ids(R.findings(tabular()))


def test_shortcut_gap_replaces_overfit():
    f = tabular(val_auc=0.99, train_auc=0.99)
    assert "shortcut_gap" in ids(R.findings(f)) and "overfit" not in ids(R.findings(f))
    assert "overfit" in ids(R.findings(tabular(train_auc=0.85 + T["train_test_gap"] + 0.01)))


def test_propagation_models_are_not_called_overfit():
    f = tabular(kind="propagation", train_auc=0.99)
    assert "overfit" not in ids(R.findings(f))


def test_feature_group_callouts():
    f = tabular()
    f["credit"]["SHAP"].update(proxy=T["group_share"], decoys=T["group_share"] - 0.001, noise=0.2)
    got = ids(R.findings(f), "weakness")
    assert "proxy_credit" in got and "noise_credit" in got and "decoy_credit" not in got


def test_interaction_rule():
    f = tabular(interaction={"SHAP": 0.05, "LIME": 0.05, "true": 0.19})
    assert "misses_interaction" in ids(R.findings(f))
    f = tabular(interaction={"SHAP": 0.10, "LIME": 0.05, "true": 0.19})
    assert "misses_interaction" not in ids(R.findings(f))


def test_anomaly_models_get_the_caveat():
    assert "anomaly_score" in ids(R.findings(tabular(kind="anomaly")), "weakness")


def test_per_method_failures_are_merged_into_one_finding():
    f = tabular()
    for m in ("SHAP", "LIME"):
        f["per_method"][m].update(repeatability_seed=0.5, stability_noise=0.5, faithfulness=0.1)
    found = R.findings(f)
    for fid in ("not_repeatable", "unstable", "unfaithful"):
        hits = [x for x in found if x["id"] == fid]
        assert len(hits) == 1, fid
        assert "SHAP and LIME" in hits[0]["text"]


def test_methods_disagree_and_trivial_agreement():
    assert "methods_disagree" in ids(R.findings(tabular(agreement={"agreement_spearman": 0.1})))
    f = tabular(key="label_spreading", agreement={"agreement_spearman": 0.1})
    assert "methods_disagree" not in ids(R.findings(f))


def test_sequence_rules():
    f = tabular(family="sequences", methods=["Integrated Gradients", "GradientSHAP"],
                per_method={"Integrated Gradients": {"correctness_mask_auc": 0.95}, "GradientSHAP": {"correctness_mask_auc": 0.6}},
                timing={"auc_drop": 0.001}, ref_auc=0.86)
    got = ids(R.findings(f))
    assert {"finds_bursts", "method_misses", "ignores_timing"} <= set(got)
    assert R.check_right_reasons(f)["passed"]


def test_text_shortcut_rule():
    f = tabular(family="text", per_method={"SHAP": {"correctness_marker_mass": 0.8}},
                model_level={"shortcut_model_effect": T["shortcut_effect"] + 0.1}, ref_auc=0.86)
    assert "trending_shortcut" in ids(R.findings(f))


def test_graph_rules():
    f = tabular(family="graph", key="gcn", per_method={"GNNExplainer": {"correctness_ring_edge_auc": 0.6}},
                reliance={"spearman_with_degree": 0.9}, degree_alone_auc=0.91, ref_auc=0.86)
    got = ids(R.findings(f))
    assert "misses_ring" in got and "degree_counter" in got
    f = tabular(family="graph", key="node2vec_logreg", per_method={}, reliance={},
                embedding_probe={"r2_predicting_log_degree": 0.4, "auc_predicting_troll_ring_membership": 0.8}, ref_auc=0.86)
    assert not R.check_right_reasons(f)["passed"]


def test_checklist_has_generic_items_then_triggered_ones():
    f = tabular(auc=0.55, val_auc=0.55)
    cl = R.checklist(R.findings(f))
    assert [c["id"] for c in cl[: len(R.GENERIC_CHECKLIST)]] == [i for i, _ in R.GENERIC_CHECKLIST]
    trig = [c for c in cl if c["id"].startswith("w_")]
    assert trig and all(c["because"] for c in trig)
    assert "w_near_chance" in [c["id"] for c in trig]


def test_every_checklist_trigger_matches_a_finding_rules_can_emit():
    src = inspect.getsource(R)
    # weaknesses, plus findings whose type is chosen at run time (`_finding(kind, ...)`)
    emitted = set(re.findall(r'_finding\((?:W|kind), "([a-z_]+)"', src))
    assert set(R.TRIGGERED_CHECKLIST) <= emitted, set(R.TRIGGERED_CHECKLIST) - emitted


def test_docs_list_every_threshold_with_its_value():
    doc = (ROOT / "docs" / "site_rules.md").read_text()
    for name, value in T.items():
        assert f"`{name}`" in doc, f"{name} missing from docs/site_rules.md"
        row = next(line for line in doc.splitlines() if line.startswith(f"| `{name}` |"))
        assert f"{value:g}" in row, f"{name}: docs do not show the current value {value}"


# ------------------------------------------------------------------ under attack
def attack_facts(none=0.85, comb_strong=0.55, comb_mild=0.70, single=None, shap_none=None, shap_strong=None):
    atk = {k: {"name": k.title(), "cost": c, "changes": [f]} for k, c, f in [
        ("posting_rate", "cheap", "posts_per_day"), ("replies", "cheap", "reply_fraction"),
        ("account_age", "expensive", "account_age_days"), ("combined", "moderate", "posts_per_day")]}
    single = single or {"posting_rate": 0.80, "replies": 0.82, "account_age": 0.84}
    res = {"none": {"auc": none, "caught": 0.7, "flagged_real": 0.2}, "combined": {"mild": {"auc": comb_mild, "caught": 0.5}, "strong": {"auc": comb_strong, "caught": 0.2}}}
    for k, v in single.items():
        res[k] = {"mild": {"auc": (none + v) / 2, "caught": 0.6}, "strong": {"auc": v, "caught": 0.5}}
    shap_none = shap_none or {"posts_per_day": 0.4, "reply_fraction": 0.3, "account_age_days": 0.2, "noise_1": 0.1}
    shap_strong = shap_strong or {"posts_per_day": 0.1, "reply_fraction": 0.1, "account_age_days": 0.5, "noise_1": 0.3}
    return {"results": res, "attacks": atk, "combined": "combined", "importance": {"SHAP": {"none": shap_none, "strong": shap_strong}}}


def test_attack_collapse_holds_and_hurts():
    assert "attack_collapses" in ids(R.attack_findings(attack_facts(comb_strong=0.85 - T["attack_collapse"])))
    assert "attack_holds" in ids(R.attack_findings(attack_facts(comb_strong=0.85 - T["attack_holds"] + 0.001)))
    assert "attack_hurts" in ids(R.attack_findings(attack_facts(comb_strong=0.75)))


def test_single_attack_rules():
    got = ids(R.attack_findings(attack_facts(single={"posting_rate": 0.70, "replies": 0.84, "account_age": 0.84})))
    assert "one_cheap_change" in got
    got = ids(R.attack_findings(attack_facts(single={"posting_rate": 0.82, "replies": 0.82, "account_age": 0.60})))
    assert "only_expensive_hurts" in got and "one_cheap_change" not in got
    got = ids(R.attack_findings(attack_facts(single={"posting_rate": 0.82, "replies": 0.82, "account_age": 0.82})))
    assert "no_single_change" in got


def test_cheap_top_features_and_explanation_shift():
    got = ids(R.attack_findings(attack_facts()))
    assert "cheap_top_features" in got and "explanations_reveal_attack" in got
    same = {"posts_per_day": 0.4, "reply_fraction": 0.3, "account_age_days": 0.2, "noise_1": 0.1}
    got = ids(R.attack_findings(attack_facts(shap_strong=same)))
    assert "explanations_hide_attack" in got


def test_attack_checklist_items_explain_themselves():
    found = R.attack_findings(attack_facts(single={"posting_rate": 0.70, "replies": 0.84, "account_age": 0.84}))
    cl = R.attack_checklist(found)
    assert cl and all(c["because"] and c["id"].startswith("a_") for c in cl)
    src = inspect.getsource(R)
    emitted = set(re.findall(r'_finding\((?:W|kind), "([a-z_]+)"', src))
    assert set(R.ATTACK_CHECKLIST) <= emitted


def test_anomaly_style_attack_rules():
    f = attack_facts(none=0.58, comb_mild=0.42, comb_strong=0.52)
    f["results"]["none"]["flagged_real"] = 0.9
    got = ids(R.attack_findings(f))
    assert {"flags_everyone", "too_typical", "nothing_to_lose"} <= set(got)
    assert not {"attack_holds", "attack_hurts", "attack_collapses", "no_single_change", "only_expensive_hurts"} & set(got)
