"""Turn a model's facts (facts.py) into a verdict, strengths, weaknesses and a deployment checklist.

Pure functions, no file access: easy to unit-test (tests/test_site_rules.py) and documented in
docs/site_rules.md. Every threshold lives in THRESHOLDS; change it there and the docs table.
"""
from __future__ import annotations

THRESHOLDS = {
    "auc_margin": 0.06,  # accuracy is "near the best" within this much of the reference AUC
    "near_chance_auc": 0.60,  # below this, the model barely beats guessing
    "val_test_gap": 0.05,  # validation AUC - test AUC above this = learned something that did not carry over
    "train_test_gap": 0.10,  # train AUC - test AUC above this = memorises training data
    "informative_share": 0.75,  # tabular: SHAP credit on the 6 informative features
    "group_share": 0.10,
    "interaction_fraction": 0.50,  # tabular: credit on burstiness + age below half their true share = interaction missed  # tabular: credit on the proxy / decoys / noise worth calling out
    "localisation_auc": 0.70,  # sequences + graph: AUC of attribution for finding planted hours / ring edges (0.5 = chance)
    "marker_mass": 0.50,  # text: share of credit on marker words in posts that contain one
    "shortcut_effect": 0.50,  # text: deleting #trending moves the score by this many standard deviations
    "timing_drop": 0.02,  # sequences: AUC lost when hours are shuffled; below this the model ignores timing
    "degree_corr": 0.70,  # graph: rank correlation of score with degree above this = degree counter
    "seed_ring_top3": 0.75,  # label spreading: top-3 seeds come from the troll's own ring
    "block_follow": 0.50,  # multimodal: a block's value tracks its planted signal (rank correlation)
    "block_ring_auc": 0.70,  # multimodal: graph block separates ring members
    "agree_high": 0.70,  # rank agreement between the two methods
    "agree_low": 0.40,
    "faithful_high": 0.50,  # deletion gain (score standard deviations): top inputs matter more than random ones
    "faithful_low": 0.20,
    "repeat_low": 0.80,  # same input, different random seed: rank correlation of explanations
    "stable_low": 0.70,  # small input noise: rank correlation of explanations
    # ---- under attack (adversarial dataset)
    "attack_collapse": 0.15,  # AUC lost under the strong combined attack: at least this = collapses
    "attack_holds": 0.05,  # ... at most this = holds up
    "single_attack_drop": 0.10,  # AUC lost to one cheap change (strong): at least this = one change is enough
    "importance_shift": 0.25,  # SHAP credit moved between no attack and strong (0 = same, 1 = none in common)
    "flagged_real_high": 0.50,  # share of real users above the model's own threshold that makes "fakes caught" meaningless
}
T = THRESHOLDS

# Agreement between the two methods is near 1 by construction for these (see notebook 05), so it says nothing.
TRIVIAL_AGREEMENT = {"label_spreading"}


def _pct(x):
    return f"{100 * x:.0f}%"


def _f(x, d=2):
    return f"{x:.{d}f}"


def _best(facts, criterion):
    """(best value, method) of a criterion over the model's methods, ignoring missing values."""
    vals = [(v.get(criterion), m) for m, v in facts["per_method"].items() if v.get(criterion) is not None]
    return max(vals) if vals else (None, None)


def _finding(kind, fid, text, check=None, **evidence):
    return {"type": kind, "id": fid, "text": text, "check": check, "evidence": evidence}


# ------------------------------------------------------------------ the three verdict checks
def check_accuracy(f):
    floor = f["ref_auc"] - T["auc_margin"]
    passed = f["auc"] >= floor
    return {
        "name": "accuracy",
        "passed": passed,
        "detail": f"Test AUC {_f(f['auc'], 3)} vs reference {_f(f['ref_auc'], 3)}, the {f['ref_how']}; needs at least {_f(floor, 3)}.",
    }


def check_generalises(f):
    gap = f["val_auc"] - f["auc"]
    return {
        "name": "generalises",
        "passed": gap <= T["val_test_gap"],
        "detail": f"Validation AUC {_f(f['val_auc'], 3)}, test AUC {_f(f['auc'], 3)} (drop {_f(max(gap, 0), 3)}; at most {_f(T['val_test_gap'])} allowed).",
    }


def check_right_reasons(f):
    """Family-specific: do the explanations find the planted signal?"""
    fam, key = f["family"], f["key"]
    if fam == "tabular":
        v = f["credit"]["SHAP"]["informative"]
        return {"name": "right_reasons", "passed": v >= T["informative_share"],
                "detail": f"SHAP gives {_pct(v)} of credit to the 6 informative features; needs {_pct(T['informative_share'])}."}
    if fam == "sequences":
        v, m = _best(f, "correctness_mask_auc")
        return {"name": "right_reasons", "passed": v >= T["localisation_auc"],
                "detail": f"{m} picks out the planted burst hours with AUC {_f(v)} (0.5 = chance); needs {_f(T['localisation_auc'])}."}
    if fam == "text":
        v, m = _best(f, "correctness_marker_mass")
        return {"name": "right_reasons", "passed": v >= T["marker_mass"],
                "detail": f"{m} puts {_pct(v)} of credit on the planted marker words; needs {_pct(T['marker_mass'])}."}
    if fam == "graph":
        if key in ("node2vec_logreg", "deepwalk_logreg"):
            return {"name": "right_reasons", "passed": False,
                    "detail": "Cannot be checked: explanations are in terms of embedding dimensions, which have no meaning of their own."}
        if key == "label_spreading":
            v, m = _best(f, "correctness_same_ring_top3")
            return {"name": "right_reasons", "passed": v >= T["seed_ring_top3"],
                    "detail": f"For trolls, {_pct(v)} of the top-3 seeds come from the troll's own ring; needs {_pct(T['seed_ring_top3'])}."}
        v, m = _best(f, "correctness_ring_edge_auc")
        return {"name": "right_reasons", "passed": v >= T["localisation_auc"],
                "detail": f"{m} picks out a troll's ring edges with AUC {_f(v)} (0.5 = chance); needs {_f(T['localisation_auc'])}."}
    if fam == "multimodal":
        ak = f["answer_key"]["Block Shapley"]
        ok = ak["text_vs_marker_count"] >= T["block_follow"] and ak["tab_vs_planted_logit"] >= T["block_follow"] and ak["graph_auc_for_troll"] >= T["block_ring_auc"]
        return {"name": "right_reasons", "passed": ok,
                "detail": (f"Block Shapley values follow the planted signal for text ({_f(ak['text_vs_marker_count'])}) and account stats "
                           f"({_f(ak['tab_vs_planted_logit'])}); graph block separates ring members with AUC {_f(ak['graph_auc_for_troll'])} "
                           f"(needs {_f(T['block_follow'])}, {_f(T['block_follow'])} and {_f(T['block_ring_auc'])}).")}
    raise ValueError(fam)


def verdict(f):
    checks = [check_accuracy(f), check_generalises(f), check_right_reasons(f)]
    return {"label": "good" if all(c["passed"] for c in checks) else "weak", "checks": checks}


# ------------------------------------------------------------------ strengths and weaknesses
def findings(f):
    S, W = "strength", "weakness"
    out = []
    add = out.append
    auc, ref = f["auc"], f["ref_auc"]

    # accuracy
    if auc < T["near_chance_auc"]:
        add(_finding(W, "near_chance", f"Barely better than guessing: test AUC {_f(auc, 3)} (0.5 is a coin flip).", "accuracy", auc=auc))
    elif auc >= ref - T["auc_margin"]:
        add(_finding(S, "accurate", f"Accurate: test AUC {_f(auc, 3)}, within {_f(max(ref - auc, 0), 3)} of the reference {_f(ref, 3)}.", "accuracy", auc=auc, ref=ref))
    else:
        add(_finding(W, "inaccurate", (f"Less accurate than it could be: test AUC {_f(auc, 3)}, {_f(ref - auc, 3)} below the reference {_f(ref, 3)}." if ref - auc < 0.10 else f"Misses many fakes: test AUC {_f(auc, 3)}, {_f(ref - auc, 3)} below the reference {_f(ref, 3)}."), "accuracy", auc=auc, ref=ref))

    gap = f["val_auc"] - auc
    if gap > T["val_test_gap"]:
        add(_finding(W, "shortcut_gap", f"Learned something that did not carry over: AUC falls from {_f(f['val_auc'], 3)} on validation to {_f(auc, 3)} on test.", "generalises", gap=gap))
    elif f["kind"] != "propagation" and f["train_auc"] - auc > T["train_test_gap"]:
        # (propagation models are seeded with training labels, so a high train AUC is by construction)
        add(_finding(W, "overfit", f"Memorises its training data: train AUC {_f(f['train_auc'], 3)} vs test {_f(auc, 3)}.", None, gap=f["train_auc"] - auc))

    if f["kind"] == "anomaly":
        add(_finding(W, "anomaly_score", "Scores how unusual an account is, not how fake it is. Its explanations describe unusualness.", None))

    # family-specific evidence about the reasons
    fam, key = f["family"], f["key"]
    if fam == "tabular":
        c = f["credit"]["SHAP"]
        if c["informative"] >= T["informative_share"]:
            add(_finding(S, "right_features", f"Relies on the right features: {_pct(c['informative'])} of SHAP credit goes to the 6 that drive the label.", "right_reasons", share=c["informative"]))
        else:
            add(_finding(W, "wrong_features", f"Only {_pct(c['informative'])} of SHAP credit goes to the 6 features that drive the label.", "right_reasons", share=c["informative"]))
        if c["proxy"] >= T["group_share"]:
            add(_finding(W, "proxy_credit", f"Gives {_pct(c['proxy'])} of credit to posts_per_week, a near-copy of posts_per_day with no effect of its own.", None, share=c["proxy"]))
        if c["decoys"] >= T["group_share"]:
            add(_finding(W, "decoy_credit", f"Gives {_pct(c['decoys'])} of credit to decoy features (follower ratio, hashtag rate) that have no effect.", None, share=c["decoys"]))
        it = f.get("interaction")
        if it and it["SHAP"] < T["interaction_fraction"] * it["true"]:
            add(_finding(W, "misses_interaction", f"Underplays the interaction between burstiness and account age: SHAP gives the pair {_pct(it['SHAP'])} of credit, but they carry {_pct(it['true'])} of the true signal.", None, shap=it["SHAP"], true=it["true"]))
        if c["noise"] >= T["group_share"]:
            add(_finding(W, "noise_credit", f"Gives {_pct(c['noise'])} of credit to pure-noise features.", None, share=c["noise"]))

    elif fam == "sequences":
        v, m = _best(f, "correctness_mask_auc")
        if v >= T["localisation_auc"]:
            add(_finding(S, "finds_bursts", f"{m} points at the planted burst hours (AUC {_f(v)}).", "right_reasons", value=v))
        else:
            add(_finding(W, "misses_bursts", f"Explanations do not point at the planted burst hours (best AUC {_f(v)}).", "right_reasons", value=v))
        weak = [(mm, vv["correctness_mask_auc"]) for mm, vv in f["per_method"].items() if vv.get("correctness_mask_auc") is not None and vv["correctness_mask_auc"] < T["localisation_auc"]]
        if v >= T["localisation_auc"] and weak:
            mm, vv = weak[0]
            add(_finding(W, "method_misses", f"{mm} misses the bursts (AUC {_f(vv)}) even though {m} finds them: which method you use changes the story.", None, value=vv))
        drop = f["timing"]["auc_drop"]
        if drop < T["timing_drop"]:
            add(_finding(W, "ignores_timing", f"Ignores timing: shuffling the 168 hours changes AUC by only {_f(drop, 3)}. It counts busy hours rather than reading the rhythm, so matching human volume could fool it.", None, drop=drop))

    elif fam == "text":
        v, m = _best(f, "correctness_marker_mass")
        if v >= T["marker_mass"]:
            add(_finding(S, "finds_markers", f"{m} puts {_pct(v)} of credit on the planted marker words.", "right_reasons", value=v))
        else:
            add(_finding(W, "misses_markers", f"Only {_pct(v)} of credit goes to the planted marker words.", "right_reasons", value=v))
        eff = f["model_level"].get("shortcut_model_effect")
        if eff is not None and eff >= T["shortcut_effect"]:
            add(_finding(W, "trending_shortcut", f"Leans on #trending: deleting that one hashtag moves the score by {_f(eff, 1)} standard deviations, though it only signalled fakes in the training data.", None, effect=eff))

    elif fam == "graph":
        rel = f["reliance"]
        if key in ("node2vec_logreg", "deepwalk_logreg"):
            ep = f["embedding_probe"]
            add(_finding(W, "opaque_embedding", f"Explanations are about 32 embedding dimensions with no meaning of their own. A probe shows they partly encode degree (R² {_f(ep['r2_predicting_log_degree'])}) and ring membership (AUC {_f(ep['auc_predicting_troll_ring_membership'])}).", "right_reasons", **ep))
        elif key == "label_spreading":
            v, m = _best(f, "correctness_same_ring_top3")
            kind = S if v >= T["seed_ring_top3"] else W
            add(_finding(kind, "seed_ring", f"For trolls, {_pct(v)} of the top-3 seed labels behind the score come from the troll's own ring.", "right_reasons", value=v))
            add(_finding(S, "exact_explanation", "Its explanation is exact: the score is a sum over seed labels, so nothing is approximated.", None))
        else:
            v, m = _best(f, "correctness_ring_edge_auc")
            if v >= T["localisation_auc"]:
                add(_finding(S, "finds_ring", f"{m} points at a troll's ring edges (AUC {_f(v)}).", "right_reasons", value=v))
            else:
                add(_finding(W, "misses_ring", f"Explanations barely find a troll's ring edges (best AUC {_f(v)}, 0.50 = chance).", "right_reasons", value=v))
        sd = rel.get("spearman_with_degree")
        if sd is not None and sd >= T["degree_corr"]:
            add(_finding(W, "degree_counter", f"Acts largely as a degree counter: its score has rank correlation {_f(sd)} with how many connections an account has (degree alone scores AUC {_f(f['degree_alone_auc'])}).", None, corr=sd))

    elif fam == "multimodal":
        ak = f["answer_key"]["Block Shapley"]
        if ak["text_vs_marker_count"] >= T["block_follow"]:
            add(_finding(S, "text_block_right", f"Its text evidence tracks the planted marker words (rank correlation {_f(ak['text_vs_marker_count'])}).", "right_reasons"))
        if ak["tab_vs_planted_logit"] >= T["block_follow"]:
            add(_finding(S, "tab_block_right", f"Its account-stat evidence tracks the true formula (rank correlation {_f(ak['tab_vs_planted_logit'])}).", "right_reasons"))
        if ak["graph_auc_for_troll"] < T["block_ring_auc"]:
            add(_finding(W, "graph_block_blind", f"Its graph evidence does not single out troll rings (AUC {_f(ak['graph_auc_for_troll'])}, 0.50 = chance).", "right_reasons"))
        co = f["coalition_auc"]
        if co.get("text") is not None and co.get("tab+text+graph") is not None and co["tab+text+graph"] - co["text"] < 0.10:
            add(_finding(W, "mostly_text", f"Mostly a text model: text alone reaches AUC {_f(co['text'])} of its full {_f(co['tab+text+graph'])}.", None))

    # explanation quality (every family)
    if key not in TRIVIAL_AGREEMENT and f["agreement"].get("agreement_spearman") is not None:
        a = f["agreement"]["agreement_spearman"]
        m1, m2 = f["methods"]
        if a >= T["agree_high"]:
            add(_finding(S, "methods_agree", f"{m1} and {m2} broadly agree on what matters (rank agreement {_f(a)}).", None, value=a))
        elif a < T["agree_low"]:
            add(_finding(W, "methods_disagree", f"{m1} and {m2} disagree on what matters (rank agreement {_f(a)}): at least one of them is misleading.", None, value=a))
    # per-method checks: one finding per check, naming every method that fails it
    def failing(criterion, below):
        return [(m, v[criterion]) for m, v in f["per_method"].items() if v.get(criterion) is not None and v[criterion] < below]

    def names(ms):
        return " and ".join(m for m, _ in ms)

    def vals(ms):
        return ", ".join(f"{m} {_f(x)}" for m, x in ms)

    def verb(ms, one, many):
        return one if len(ms) == 1 else many

    ms = failing("faithfulness", T["faithful_low"])
    if ms:
        add(_finding(W, "unfaithful", f"{names(ms)} {verb(ms, 'is', 'are')} not faithful: removing what {verb(ms, 'it calls', 'they call')} important changes the score barely more than removing random inputs.", None, values=dict(ms)))
    ms = failing("repeatability_seed", T["repeat_low"])
    if ms:
        add(_finding(W, "not_repeatable", f"{names(ms)} {verb(ms, 'gives', 'give')} a different answer when re-run on the same account (rank agreement between runs: {vals(ms)}).", None, values=dict(ms)))
    ms = failing("stability_noise", T["stable_low"])
    if ms:
        add(_finding(W, "unstable", f"{names(ms)} {verb(ms, 'is', 'are')} unstable: a tiny change to the input reshuffles the explanation (rank agreement: {vals(ms)}).", None, values=dict(ms)))
    faithful = [m for m, v in f["per_method"].items() if (v.get("faithfulness") or 0) >= T["faithful_high"]]
    if faithful:
        add(_finding(S, "faithful", f"{' and '.join(faithful)} {'is' if len(faithful) == 1 else 'are'} faithful: removing the inputs flagged as important really does move the score.", None))
    return out


# ------------------------------------------------------------------ checklist
GENERIC_CHECKLIST = [
    ("later_data", "Test on data collected after your training data. Behaviour drifts, and a validation split from the same period hides that."),
    ("threshold", "Choose the decision threshold on held-out data for the precision you need; a wrongly banned real user costs more than a missed bot."),
    ("audit_sample", "Keep a fixed audit sample of labelled accounts and re-run the explanations on it after every retrain."),
    ("cheap_features", "List which inputs an attacker can change cheaply (posting rate, hashtags, timing) and test the model with those changed."),
]

TRIGGERED_CHECKLIST = {
    "near_chance": "Do not deploy as a detector. At this accuracy it flags real users about as often as fakes.",
    "inaccurate": "Compare with a simple baseline on your own data; a better-ranked model may cost nothing extra.",
    "shortcut_gap": "Find the feature that stopped working between validation and test (here, likely #trending) and remove it or retrain without it.",
    "overfit": "Check accuracy on a fresh sample; regularise or limit depth if it drops.",
    "anomaly_score": "Validate that 'unusual' means 'fake' on your platform: new, niche or very active real users are unusual too.",
    "wrong_features": "Inspect the top features for a few flagged accounts by hand before trusting any decision.",
    "proxy_credit": "Before acting on importance scores, check for near-duplicate features; credit gets split between them.",
    "decoy_credit": "Drop features with no causal link to the label and confirm accuracy holds.",
    "misses_interaction": "Check accounts where two signals only matter together (new accounts that post in bursts); per-feature importance hides such combinations.",
    "noise_credit": "Treat this model's importance ranking as unreliable; it rewards noise.",
    "misses_bursts": "Do not use these explanations to justify flags; they do not show the pattern the model reacts to.",
    "method_misses": "Use the explanation method that passes the correctness test, and say which one you used.",
    "ignores_timing": "Test with fake accounts that post at human volume but on a schedule; this model may miss them.",
    "misses_markers": "Check the words the model reacts to against a list of known spam and abuse terms.",
    "trending_shortcut": "Test with trending hashtags removed or reshuffled; retrain if accuracy depends on them.",
    "opaque_embedding": "If you must explain decisions to users or moderators, prefer a model with interpretable inputs.",
    "seed_ring": "Check that the known labels you seed it with cover every community you care about.",
    "misses_ring": "Do not rely on edge explanations to identify coordinated groups; verify rings with a separate method.",
    "degree_counter": "Test on popular real accounts: a model that counts connections will flag them.",
    "graph_block_blind": "Check coordinated groups separately; this model's graph evidence does not reveal them.",
    "mostly_text": "Test with the text signal weakened (paraphrased or disguised words); the other inputs may not catch up.",
    "methods_disagree": "Before showing an explanation to a moderator, check that both methods agree for that account.",
    "unfaithful": "Run a deletion test (remove the top-ranked inputs, watch the score) before trusting this method.",
    "not_repeatable": "Fix the random seed and average several runs before reporting an explanation.",
    "unstable": "Explain a few slightly perturbed copies of each account; trust only what stays the same.",
}


def checklist(found):
    items = [{"id": i, "text": t, "because": None} for i, t in GENERIC_CHECKLIST]
    seen = set()
    for x in found:
        if x["type"] == "weakness" and x["id"] in TRIGGERED_CHECKLIST and x["id"] not in seen:
            seen.add(x["id"])
            items.append({"id": "w_" + x["id"], "text": TRIGGERED_CHECKLIST[x["id"]], "because": x["text"]})
    return items


# ------------------------------------------------------------------ under attack
def _imp_shift(a, b):
    keys = sorted(set(a) | set(b))
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)


def attack_findings(af):
    """Vulnerabilities and strengths under the adversarial dataset. `af` comes from adversarial_facts.py."""
    S, W = "strength", "weakness"
    out = []
    add = out.append
    res, atk = af["results"], af["attacks"]
    base = res["none"]
    comb = res[af["combined"]]["strong"]
    drop = base["auc"] - comb["auc"]
    caught = f"it catches {_pct(comb['caught'])} of fakes instead of {_pct(base['caught'])}"
    if base["flagged_real"] >= T["flagged_real_high"]:
        add(_finding(W, "flags_everyone", f"Its decision threshold flags {_pct(base['flagged_real'])} of real users, so the share of fakes it 'catches' ({_pct(base['caught'])}) means little.", None, flagged=base["flagged_real"]))
    mild = res[af["combined"]]["mild"]
    if mild["auc"] < 0.5 and mild["auc"] < comb["auc"]:
        add(_finding(W, "too_typical", f"A half-hearted attack fools it more than a full one: AUC falls to {_f(mild['auc'], 3)} at mild strength, worse than a coin flip, but only {_f(comb['auc'], 3)} at strong. Fakes moved halfway toward real users look more typical than real users do, and it flags the unusual.", None, auc=mild["auc"]))
    if base["auc"] < T["near_chance_auc"]:
        add(_finding(W, "nothing_to_lose", f"Barely better than guessing even before the attack (AUC {_f(base['auc'], 3)} on the fresh population), so an attack has little to take away.", None, auc=base["auc"]))
    elif drop >= T["attack_collapse"]:
        add(_finding(W, "attack_collapses", f"Collapses under the combined attack: AUC falls from {_f(base['auc'], 3)} to {_f(comb['auc'], 3)} at strong strength ({_f(mild['auc'], 3)} at mild), and {caught}.", None, drop=drop))
    elif drop <= T["attack_holds"]:
        add(_finding(S, "attack_holds", f"Holds up under the combined attack: AUC {_f(base['auc'], 3)} → {_f(comb['auc'], 3)} at strong strength.", None, drop=drop))
    else:
        add(_finding(W, "attack_hurts", f"Weakened by the combined attack: AUC falls from {_f(base['auc'], 3)} to {_f(comb['auc'], 3)} at strong strength, and {caught}.", None, drop=drop))

    singles = {k: v for k, v in res.items() if k not in ("none", af["combined"])}
    worst = max(singles, key=lambda k: base["auc"] - singles[k]["strong"]["auc"])
    wdrop = base["auc"] - singles[worst]["strong"]["auc"]
    cheap = [k for k in singles if atk[k]["cost"] == "cheap"]
    worst_cheap = max(cheap, key=lambda k: base["auc"] - singles[k]["strong"]["auc"])
    cdrop = base["auc"] - singles[worst_cheap]["strong"]["auc"]
    if cdrop >= T["single_attack_drop"]:
        a = atk[worst_cheap]
        add(_finding(W, "one_cheap_change", f"One cheap change is enough to hurt it: \"{a['name']}\" alone drops AUC from {_f(base['auc'], 3)} to {_f(singles[worst_cheap]['strong']['auc'], 3)}.", None, attack=worst_cheap, drop=cdrop))
    elif base["auc"] < T["near_chance_auc"]:
        pass  # nothing to lose: "no single change hurts it" would be praise for having no signal
    elif atk[worst]["cost"] == "expensive" and wdrop >= T["single_attack_drop"]:
        add(_finding(S, "only_expensive_hurts", f"No cheap single change hurts it much; the most damaging is \"{atk[worst]['name']}\", which costs an attacker money or time.", None, attack=worst, drop=wdrop))
    elif wdrop < T["single_attack_drop"]:
        add(_finding(S, "no_single_change", f"No single change hurts it much (the worst, \"{atk[worst]['name']}\", costs {_f(wdrop, 3)} AUC); an attacker has to change several things at once.", None, attack=worst, drop=wdrop))

    imp = af["importance"].get("SHAP", {})
    if "none" in imp:
        cheap_feats = {f for k in cheap for f in atk[k]["changes"]}
        top3 = sorted(imp["none"], key=lambda k: -imp["none"][k])[:3]
        cheap_top = [x for x in top3 if x in cheap_feats]
        if len(cheap_top) >= 2:
            add(_finding(W, "cheap_top_features", f"Its most important features are cheap to fake: {', '.join(cheap_top)} {'are' if len(cheap_top) > 1 else 'is'} among SHAP's top three.", None, features=cheap_top))
    if "none" in imp and "strong" in imp:
        shift = _imp_shift(imp["none"], imp["strong"])
        t0 = max(imp["none"], key=imp["none"].get)
        t1 = max(imp["strong"], key=imp["strong"].get)
        moved = f"SHAP's top feature goes from {t0} to {t1}" if t0 != t1 else f"SHAP's top feature stays {t0}"
        if shift >= T["importance_shift"]:
            add(_finding(S, "explanations_reveal_attack", f"Its explanations change visibly under attack ({_pct(shift)} of the credit moves; {moved}), so monitoring explanations could flag the attack.", None, shift=shift))
        elif drop >= T["attack_collapse"]:
            add(_finding(W, "explanations_hide_attack", f"Its explanations barely change while it fails ({_pct(shift)} of the credit moves; {moved}), so watching explanations would not reveal the attack.", None, shift=shift))
    return out


ATTACK_CHECKLIST = {
    "attack_collapses": "Before launch, red-team with mimicry: give test fakes real users' stats and measure how many you still catch.",
    "attack_hurts": "Track the share of fakes caught over time; a steady fall is the signature of operators adapting.",
    "one_cheap_change": "Do not let one easily changed behaviour carry the decision; add signals that are costly to fake (account age, network position, device or payment history).",
    "cheap_top_features": "Add inputs that are expensive to fake, and check how much accuracy is left when the cheap ones are removed.",
    "flags_everyone": "Set the threshold from the cost of wrongly flagging a real user, then re-measure how many fakes it catches at that threshold.",
    "too_typical": "Test with fakes that are only partly disguised, not just fully copied ones; for this model the partial disguise is the more dangerous one.",
    "nothing_to_lose": "Do not deploy this model as a detector; compare it with a supervised model trained on labelled accounts.",
    "explanations_hide_attack": "Monitor detection rates and score distributions directly; this model's explanations stay the same while it is being fooled.",
}


def attack_checklist(found):
    return [{"id": "a_" + x["id"], "text": ATTACK_CHECKLIST[x["id"]], "because": x["text"]}
            for x in found if x["type"] == "weakness" and x["id"] in ATTACK_CHECKLIST]


def evaluate(f):
    """Everything the site shows about one model's normal-data results."""
    fs = findings(f)
    return {
        "verdict": verdict(f),
        "strengths": [x for x in fs if x["type"] == "strength"],
        "weaknesses": [x for x in fs if x["type"] == "weakness"],
        "checklist": checklist(fs),
    }
