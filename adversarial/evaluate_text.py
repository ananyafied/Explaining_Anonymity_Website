"""Score the 3 text models on the adversarial dataset and explain them under attack.

    python -m adversarial.evaluate_text               # needs data/synthetic and adversarial/data

Writes outputs/step3_adversarial/text/:
  metrics.csv            account-level AUC (95% bootstrap interval), fakes caught, real users flagged
  attributions.csv       word-level SHAP / LIME for 60 test posts by fake accounts that originally
                         contained a marker word, under the combined text attack (none / mild / strong)
  signal_credit.csv      share of credit on the marker word (or its disguise / synonym)
  examples.json          one post at each strength, explained, for the site
  meta.json

Account scores follow step 1: the transformer averages its six post probabilities; word2vec averages
the word vectors of all the account's posts; doc2vec infers one vector for all of them (every account
here is new, so all are inferred). Posts longer than the transformer's limit (14 words) are cut to 14,
as a deployed model would. SHAP / LIME settings are those of notebooks/04_explain_text.ipynb.
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
import shap  # noqa: E402
import torch  # noqa: E402
from lime.lime_text import LimeTextExplainer  # noqa: E402

from xai_eval import loaders as L  # noqa: E402
from xai_eval.data import load_data  # noqa: E402

from .attacks import STRENGTHS  # noqa: E402
from .common import metric_row  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

ADV = ROOT / "adversarial" / "data"
OUT = ROOT / "outputs" / "step3_adversarial" / "text"
SEED = 0
N_POSTS = 60
SHAP_NSAMPLES = 1000
LIME_NSAMPLES = 1000
EXPLAINED_ATTACK = "text_combined"
MODELS = ("transformer", "doc2vec_logreg", "word2vec_logreg")


class Scorer:
    """Post-level and account-level scores for one text model, on any posts (truncating for the transformer)."""

    def __init__(self, name, D):
        self.name = name
        self.ts = L.TextScorer(name, L.load_model(name, D), D)

    def posts(self, token_lists):
        token_lists = [list(t) for t in token_lists]
        if self.name == "transformer":
            keep = self.ts.max_len - 1  # one slot is the [CLS] token
            out = []
            with torch.no_grad():
                for i in range(0, len(token_lists), 512):
                    ids = torch.as_tensor([self.ts.encode(t[:keep]) for t in token_lists[i : i + 512]], dtype=torch.long)
                    out.append(torch.sigmoid(self.ts.model(ids)).numpy())
            return np.concatenate(out) if out else np.zeros(0)
        return self.ts.clf.predict_proba(self.ts.vectors(token_lists))[:, 1]

    def accounts(self, post_tokens, post_account, account_ids):
        if self.name == "transformer":
            p = self.posts(post_tokens)
            return pd.Series(p).groupby(post_account).mean().reindex(account_ids).to_numpy()
        toks = pd.Series(post_tokens).groupby(post_account).agg(lambda s: [w for t in s for w in t]).reindex(account_ids)
        return self.ts.clf.predict_proba(self.ts.vectors(list(toks)))[:, 1]


def players(tokens):
    return list(dict.fromkeys(tokens))


def keep_words(tokens, words, z):
    kept = {w for w, flag in zip(words, z) if flag}
    return [t for t in tokens if t in kept]


def explain_shap(f, tokens):
    words = players(tokens)
    g = lambda Z: f([keep_words(tokens, words, z) for z in np.atleast_2d(Z)])  # noqa: E731
    np.random.seed(SEED)
    ex = shap.KernelExplainer(g, np.zeros((1, len(words))))
    return words, np.asarray(ex.shap_values(np.ones((1, len(words))), nsamples=SHAP_NSAMPLES, silent=True)).reshape(-1)


def explain_lime(f, tokens):
    words = players(tokens)
    ex = LimeTextExplainer(split_expression=r"\s+", bow=True, random_state=SEED)
    e = ex.explain_instance(" ".join(tokens), lambda texts: np.c_[1 - f([t.split() for t in texts]), f([t.split() for t in texts])],
                            labels=(1,), num_features=len(words), num_samples=LIME_NSAMPLES)
    w = dict(e.as_list(label=1))
    return words, np.array([w.get(x, 0.0) for x in words])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    D = load_data(ROOT)
    markers = D.marker_set
    manifest = json.loads((ADV / "manifest.json").read_text())
    base = pd.read_csv(ADV / "base" / "accounts.csv")
    te_ids = base.loc[base["split"] == "test", "account_id"].to_numpy()
    y = base.set_index("account_id").loc[te_ids, "label"].to_numpy()
    thresholds = pd.read_csv(ROOT / "outputs" / "step1_fit" / "metrics.csv").set_index("model")["threshold"]

    variants = {"none": pd.read_csv(ADV / "base" / "posts.csv")}
    truths = {"none": None}
    for v in manifest["text_variants"]:
        variants[v] = pd.read_csv(ADV / v / "posts.csv")
        truths[v] = pd.read_csv(ADV / v / "truth" / "text_attack_truth.csv").fillna({"replacements": ""})

    scorers = {n: Scorer(n, D) for n in MODELS}
    for n, sc in scorers.items():  # the same models step 1 evaluated (doc2vec by rank, as in step 2)
        chk = L.verify_reload(n, D, sc.ts.account_scores(D), loose=(n == "doc2vec_logreg"))
        assert chk["status"] == "OK", n

    rows = []
    for n, sc in scorers.items():
        for v, posts in variants.items():
            p = posts[posts["account_id"].isin(te_ids)]
            s = sc.accounts([t.split() for t in p["text"]], p["account_id"].to_numpy(), te_ids)
            rows.append(metric_row(n, v, y, s, thresholds[n], np.random.default_rng(SEED)))
        print(f"  scored {n}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)

    # posts to explain: test posts by fake accounts whose original text has a marker word
    fake_te = set(base.loc[(base["split"] == "test") & (base["label"] == 1), "account_id"])
    p0 = variants["none"]
    cand = p0[p0["account_id"].isin(fake_te) & p0["text"].map(lambda t: bool(markers & set(t.split())))]
    post_ids = np.random.default_rng(SEED).choice(cand["post_id"].to_numpy(), size=N_POSTS, replace=False)

    attr_rows, credit_rows, examples, timings = [], [], {}, {}
    for strength in STRENGTHS:
        v = "none" if strength == "none" else f"{EXPLAINED_ATTACK}_{strength}"
        posts = variants[v].set_index("post_id")
        repl = {} if truths[v] is None else dict(zip(truths[v]["post_id"], truths[v]["replacements"]))
        for n, sc in scorers.items():
            for meth, fn in (("SHAP", explain_shap), ("LIME", explain_lime)):
                t0 = time.time()
                shares, top1 = [], []
                for k, pid in enumerate(post_ids):
                    toks = posts.loc[pid, "text"].split()
                    slot = {w for w in toks if w in markers} | {pair.split(">")[1] for pair in repl.get(pid, "").split("|") if pair}
                    words, a = fn(sc.posts, toks)
                    is_slot = np.array([w in slot for w in words])
                    shares.append(np.abs(a)[is_slot].sum() / max(np.abs(a).sum(), 1e-12))
                    top1.append(float(is_slot[int(np.argmax(a))]))
                    attr_rows += [{"model": n, "method": meth, "strength": strength, "post_id": int(pid), "word": w,
                                   "attribution": float(x), "is_marker_slot": bool(m)} for w, x, m in zip(words, a, is_slot)]
                    if k == 0:
                        ex = examples.setdefault(strength, {"post_id": int(pid), "words": words, "is_marker_slot": [bool(m) for m in is_slot],
                                                            "original_text": p0.set_index("post_id").loc[pid, "text"], "attributions": {}})
                        ex["attributions"].setdefault(n, {})[meth] = [round(float(x), 5) for x in a]
                timings[f"{n}/{meth}/{strength}"] = (time.time() - t0) / len(post_ids)
                credit_rows.append({"model": n, "method": meth, "strength": strength,
                                    "credit_on_markers": float(np.mean(shares)), "marker_top1": float(np.mean(top1))})
            print(f"  explained {n} ({strength})", flush=True)
    pd.DataFrame(attr_rows).to_csv(OUT / "attributions.csv", index=False)
    pd.DataFrame(credit_rows).to_csv(OUT / "signal_credit.csv", index=False)
    (OUT / "examples.json").write_text(json.dumps(examples) + "\n")
    meta = {"seed": SEED, "n_posts": N_POSTS, "shap_nsamples": SHAP_NSAMPLES, "lime_nsamples": LIME_NSAMPLES,
            "explained_attack": EXPLAINED_ATTACK, "transformer_max_words": scorers["transformer"].ts.max_len - 1,
            "adversarial_seed": manifest["seed"], "evaluation_split": "test", "seconds_per_explanation": timings}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
