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
