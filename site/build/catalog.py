"""The models the website shows, and how users can refer to them.

Every model here was fitted in step 1 (`outputs/step1_fit/`), explained in step 2
(`outputs/step2_explain/<family>/`) and attacked in step 3 (`outputs/step3_adversarial/<family>/`).
Left out: belief propagation (inverted scores), SVD + logreg (identical to PCA + logreg), and the
graph and multimodal models, which were never attacked.
"""
from __future__ import annotations

from dataclasses import dataclass

# family key -> (display name, folder under outputs/step2_explain/)
FAMILIES = {
    "tabular": ("Account stats", "tabular"),
    "sequences": ("Hourly activity", "sequences"),
    "text": ("Post text", "text"),
}


@dataclass(frozen=True)
class ModelEntry:
    key: str  # step-1 model name, used for file names
    name: str  # display name
    family: str  # key into FAMILIES
    kind: str  # "classifier" (P(inauthentic)) or "anomaly" (anomaly score)
    methods: tuple  # the two explanation methods step 2 ran, as named in its scorecard
    aliases: tuple = ()  # extra names the text box accepts (lower case)


SHAP_LIME = ("SHAP", "LIME")

MODELS: tuple[ModelEntry, ...] = (
    # ---- account stats (12 profile features)
    ModelEntry("lightgbm", "LightGBM", "tabular", "classifier", SHAP_LIME, ("lgbm", "light gbm", "gradient boosting")),
    ModelEntry("xgboost", "XGBoost", "tabular", "classifier", SHAP_LIME, ("xgb", "boosted trees")),
    ModelEntry("rf", "Random forest", "tabular", "classifier", SHAP_LIME, ("rf", "random forest", "forest")),
    ModelEntry("svm_linear", "Linear SVM", "tabular", "classifier", SHAP_LIME, ("linear svm", "linear svc")),
    ModelEntry("svm_rbf", "RBF SVM", "tabular", "classifier", SHAP_LIME, ("rbf svm", "kernel svm", "rbf")),
    ModelEntry("nmf_logreg", "NMF + logistic regression", "tabular", "classifier", SHAP_LIME, ("nmf", "nmf logreg")),
    ModelEntry("pca_logreg", "PCA + logistic regression", "tabular", "classifier", SHAP_LIME, ("pca", "pca logreg")),
    ModelEntry("isolation_forest", "Isolation forest", "tabular", "anomaly", SHAP_LIME, ("iforest", "isolation forest")),
    ModelEntry("lof", "Local outlier factor", "tabular", "anomaly", SHAP_LIME, ("lof", "local outlier factor")),
    ModelEntry("one_class_svm", "One-class SVM", "tabular", "anomaly", SHAP_LIME, ("ocsvm", "one class svm", "one-class svm")),
    ModelEntry("autoencoder_dense", "Dense autoencoder", "tabular", "anomaly", SHAP_LIME, ("autoencoder", "dense autoencoder", "ae")),
    # ---- hourly activity (168 hourly post counts)
    ModelEntry("cnn1d", "1D CNN", "sequences", "classifier", ("Grad-CAM", "Integrated Gradients"), ("cnn", "1d cnn", "convolutional")),
    ModelEntry("lstm", "LSTM", "sequences", "classifier", ("Integrated Gradients", "GradientSHAP"), ("lstm",)),
    ModelEntry("rnn", "RNN", "sequences", "classifier", ("Integrated Gradients", "GradientSHAP"), ("rnn", "recurrent")),
    ModelEntry("autoencoder_lstm", "LSTM autoencoder", "sequences", "anomaly", ("Integrated Gradients", "GradientSHAP"), ("lstm autoencoder", "lstm ae")),
    # ---- post text
    ModelEntry("transformer", "Transformer", "text", "classifier", SHAP_LIME, ("transformer",)),
    ModelEntry("doc2vec_logreg", "Doc2vec + logistic regression", "text", "classifier", SHAP_LIME, ("doc2vec",)),
    ModelEntry("word2vec_logreg", "Word2vec + logistic regression", "text", "classifier", SHAP_LIME, ("word2vec", "w2v")),
)

EXCLUDED = {
    "belief_propagation": "Scores are inverted (test AUC 0.10).",
    "svd_logreg": "Identical to PCA + logistic regression.",
    "baseline_all_authentic": "Reference baseline, not a model.",
    # Dropped on 2026-11-21: the follower graph was never attacked (Phase 5 skipped), so these models
    # have no adversarial results. Their step-1 and step-2 results remain in outputs/.
    **{k: "Not attacked: the site only shows models with adversarial results." for k in (
        "gcn", "gat", "graphsage", "rgcn", "node2vec_logreg", "deepwalk_logreg", "label_spreading", "multimodal_fusion")},
}

BY_KEY = {m.key: m for m in MODELS}


def lookup(text: str) -> ModelEntry | None:
    """Resolve what a user typed to a model (exact key, display name or alias, case-insensitive)."""
    t = " ".join(text.strip().lower().replace("_", " ").split())
    for m in MODELS:
        if t in {m.key.replace("_", " "), m.name.lower(), *m.aliases}:
            return m
    return None


assert len(MODELS) == 18 and len(BY_KEY) == 18
assert not set(BY_KEY) & set(EXCLUDED)
_names = [n for m in MODELS for n in {m.key.replace("_", " "), m.name.lower(), *m.aliases}]
assert len(_names) == len(set(_names)), "an alias resolves to two models"


# Plain-language descriptions shown on every model page (each model keeps its own two methods).
METHOD_INFO = {
    "SHAP": "Shares the model's score out among its inputs using Shapley values from game theory: each input gets its average contribution over every order in which inputs could be added. Exact for tree models (TreeSHAP); sampled for the others (KernelSHAP), which can make re-runs differ slightly.",
    "LIME": "Makes many slightly altered copies of the input, watches how the score changes, and fits a simple straight-line model to those changes. The line's weights are the explanation. Its random sampling means re-runs can disagree.",
    "Integrated Gradients": "Slides the input from a blank starting point (a week with no posts) up to the real input, and adds up how much each part moved the model's output along the way. No randomness: re-runs give the same answer.",
    "GradientSHAP": "SHAP's method for neural networks: repeats the Integrated Gradients idea from many random starting points and averages them. It estimates Shapley values, so it is a form of SHAP, but re-runs can differ.",
    "Grad-CAM": "Looks inside the network's last convolution layer and highlights the stretches of the week that pushed the output up. Works at half the time resolution, so it lights up regions rather than single hours.",
}

# What the importance chart on each family's page is about.
FAMILY_INFO = {
    "tabular": "These models read 12 numbers about each account. Credit is shared out over those 12, and we know which ones truly drive the label.",
    "sequences": "These models read a week of hourly post counts (168 numbers). Credit is shared out over the hours, and we know which hours carry the planted burst pattern.",
    "text": "These models read short posts. Credit is shared out over the words, and we know which words are planted markers and which hashtag is a trap.",
}

assert set(METHOD_INFO) >= {meth for m in MODELS for meth in m.methods}
