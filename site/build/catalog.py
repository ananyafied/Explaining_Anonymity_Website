"""The models the website shows, and how users can refer to them.

Every model here was fitted in step 1 (`outputs/step1_fit/`) and explained in step 2
(`outputs/step2_explain/<family>/`). Belief propagation (inverted scores) and SVD + logreg
(identical to PCA + logreg) are deliberately left out.
"""
from __future__ import annotations

from dataclasses import dataclass

# family key -> (display name, folder under outputs/step2_explain/)
FAMILIES = {
    "tabular": ("Account stats", "tabular"),
    "sequences": ("Hourly activity", "sequences"),
    "text": ("Post text", "text"),
    "graph": ("Follower graph", "graph"),
    "multimodal": ("Multimodal", "multimodal"),
}


@dataclass(frozen=True)
class ModelEntry:
    key: str  # step-1 model name, used for file names
    name: str  # display name
    family: str  # key into FAMILIES
    kind: str  # "classifier" (P(inauthentic)), "anomaly" (anomaly score) or "propagation" (score from known labels)
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
    # ---- follower graph
    ModelEntry("gcn", "GCN", "graph", "classifier", ("GNNExplainer", "Integrated Gradients"), ("gcn", "graph convolutional network")),
    ModelEntry("gat", "GAT", "graph", "classifier", ("GNNExplainer", "Integrated Gradients"), ("gat", "graph attention")),
    ModelEntry("graphsage", "GraphSAGE", "graph", "classifier", ("GNNExplainer", "Integrated Gradients"), ("sage", "graph sage")),
    ModelEntry("rgcn", "R-GCN", "graph", "classifier", ("GNNExplainer", "Integrated Gradients"), ("rgcn", "relational gcn")),
    ModelEntry("node2vec_logreg", "Node2vec + logistic regression", "graph", "classifier", SHAP_LIME, ("node2vec",)),
    ModelEntry("deepwalk_logreg", "DeepWalk + logistic regression", "graph", "classifier", SHAP_LIME, ("deepwalk", "deep walk")),
    ModelEntry("label_spreading", "Label spreading", "graph", "propagation", ("Seed contribution", "Leave-one-seed-out"), ("label propagation",)),
    # ---- multimodal
    ModelEntry("multimodal_fusion", "Multimodal fusion", "multimodal", "classifier", ("Block Shapley", "Drop-one ablation"), ("fusion", "multimodal")),
)

EXCLUDED = {
    "belief_propagation": "Scores are inverted (test AUC 0.10).",
    "svd_logreg": "Identical to PCA + logistic regression.",
    "baseline_all_authentic": "Reference baseline, not a model.",
}

BY_KEY = {m.key: m for m in MODELS}


def lookup(text: str) -> ModelEntry | None:
    """Resolve what a user typed to a model (exact key, display name or alias, case-insensitive)."""
    t = " ".join(text.strip().lower().replace("_", " ").split())
    for m in MODELS:
        if t in {m.key.replace("_", " "), m.name.lower(), *m.aliases}:
            return m
    return None


assert len(MODELS) == 26 and len(BY_KEY) == 26
assert not set(BY_KEY) & set(EXCLUDED)
_names = [n for m in MODELS for n in {m.key.replace("_", " "), m.name.lower(), *m.aliases}]
assert len(_names) == len(set(_names)), "an alias resolves to two models"


# Plain-language descriptions shown on every model page (option A: each model keeps its own two methods).
METHOD_INFO = {
    "SHAP": "Shares the model's score out among its inputs using Shapley values from game theory: each input gets its average contribution over every order in which inputs could be added. Exact for tree models (TreeSHAP); sampled for the others (KernelSHAP), which can make re-runs differ slightly.",
    "LIME": "Makes many slightly altered copies of the input, watches how the score changes, and fits a simple straight-line model to those changes. The line's weights are the explanation. Its random sampling means re-runs can disagree.",
    "Integrated Gradients": "Slides the input from a blank starting point (no posts, or no edges and features) up to the real input, and adds up how much each part moved the model's output along the way. No randomness: re-runs give the same answer.",
    "GradientSHAP": "SHAP's method for neural networks: repeats the Integrated Gradients idea from many random starting points and averages them. It estimates Shapley values, so it is a form of SHAP, but re-runs can differ.",
    "Grad-CAM": "Looks inside the network's last convolution layer and highlights the stretches of the week that pushed the output up. Works at half the time resolution, so it lights up regions rather than single hours.",
    "GNNExplainer": "Learns the smallest set of edges and features that still produces the same prediction for an account, by training a soft on/off mask. Starts from a random mask, so re-runs can differ.",
    "Seed contribution": "Label spreading's score is literally a sum over the accounts whose labels it was given (the seeds). This lists each seed's share of that sum. Exact, with nothing approximated; it is what SHAP would give for this model.",
    "Leave-one-seed-out": "Removes one seed label at a time and recomputes the score; the change is that seed's importance. Exact.",
    "Block Shapley": "Exact Shapley values (the idea behind SHAP) over three blocks of input: account stats, post text and the follower graph. The model is run with every combination of blocks switched on or off.",
    "Drop-one ablation": "Removes one block of input at a time (account stats, text or graph) and measures how far the score falls.",
}

# What the importance chart on each family's page is about.
FAMILY_INFO = {
    "tabular": "These models read 12 numbers about each account. Credit is shared out over those 12, and we know which ones truly drive the label.",
    "sequences": "These models read a week of hourly post counts (168 numbers). Credit is shared out over the hours, and we know which hours carry the planted burst pattern.",
    "text": "These models read short posts. Credit is shared out over the words, and we know which words are planted markers and which hashtag is a trap.",
    "graph": "These models read who interacts with whom. Explanations point at edges (or, for the embedding models, at abstract dimensions), and we know which accounts form coordinated troll rings.",
    "multimodal": "This model reads three kinds of input at once. Credit is shared out over the three blocks: account stats, post text and the follower graph.",
}

assert set(METHOD_INFO) >= {meth for m in MODELS for meth in m.methods}
