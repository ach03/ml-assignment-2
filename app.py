"""Streamlit front-end for Machine Learning Assignment 2 (M.Tech AIML, BITS Pilani WILP).

The app NEVER retrains anything: it loads the pre-trained pipelines from ./model,
scores the uploaded test CSV and shows the evaluation results.
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_curve,
)

from metrics_utils import METRIC_ORDER, compute_metrics, encode_target

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

st.set_page_config(page_title="Classifier Evaluation Lab", page_icon="🧪", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem;}
      div[data-testid="stMetric"] {
          background: rgba(120,120,160,0.10);
          border: 1px solid rgba(120,120,160,0.30);
          border-radius: 12px; padding: 12px 16px;
      }
      .hero {padding: 4px 0 10px 0;}
      .hero h1 {margin-bottom: 0;}
      .hero p {opacity: .75; margin-top: 4px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_meta():
    return json.loads((MODEL_DIR / "meta.json").read_text())


@st.cache_resource
def load_model(file_name):
    return joblib.load(MODEL_DIR / file_name)


@st.cache_data
def load_bundled_test_data():
    return pd.read_csv(ROOT / "test_data.csv")


def evaluate(model_name, X, y, meta):
    """Run a saved pipeline on (X, y) and return metrics + raw predictions."""
    model = load_model(meta["models"][model_name]["file"])
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return compute_metrics(y, y_pred, y_proba, meta["n_classes"]), y_pred, y_proba


try:
    meta = load_meta()
except FileNotFoundError:
    st.error("model/meta.json not found. Run the notebook first and commit the model/ folder.")
    st.stop()

classes = meta["classes"]
features = meta["features"]
target_col = meta["target_col"]
model_names = list(meta["models"].keys())

st.markdown(
    f"""<div class="hero"><h1>🧪 Classifier Evaluation Lab</h1>
    <p>Dataset: <b>{meta['dataset_name']}</b> &nbsp;|&nbsp; {len(features)} features
    &nbsp;|&nbsp; classes: {', '.join(classes)} &nbsp;|&nbsp; 5 pre-trained models</p></div>""",
    unsafe_allow_html=True,
)

trained_ver = meta.get("trained_with", {}).get("scikit-learn")
if trained_ver and trained_ver != sklearn.__version__:
    st.warning(
        f"Models were trained with scikit-learn {trained_ver} but this server runs "
        f"{sklearn.__version__}. Pin scikit-learn=={trained_ver} in requirements.txt if loading fails."
    )

with st.sidebar:
    st.header("1 · Upload test data")
    uploaded = st.file_uploader("CSV with all feature columns + target column", type=["csv"])
    use_sample = st.checkbox("Use the bundled test_data.csv instead", value=False)
    st.download_button(
        "⬇ Download sample test_data.csv",
        data=load_bundled_test_data().to_csv(index=False).encode("utf-8"),
        file_name="test_data.csv",
        mime="text/csv",
    )

    st.header("2 · Choose a model")
    selected = st.selectbox("Classification model", model_names)
    params = meta["models"][selected]["params"]
    st.caption("Tuned hyper-parameters: " + (", ".join(f"{k}={v}" for k, v in params.items()) or "defaults"))

if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the uploaded file as CSV: {exc}")
        st.stop()
    source = f"uploaded file `{uploaded.name}`"
elif use_sample:
    raw = load_bundled_test_data()
    source = "bundled `test_data.csv`"
else:
    st.info("👈 Upload a **test-data CSV** in the sidebar (or tick the bundled sample) to begin.")
    with st.expander("What columns does the CSV need?"):
        st.write(f"**Target column:** `{target_col}` with values from {classes}")
        st.write("**Feature columns:**")
        st.code(", ".join(features))
    st.stop()

missing = [c for c in features + [target_col] if c not in raw.columns]
if missing:
    st.error(f"The CSV is missing required column(s): {missing}")
    st.stop()

raw = raw.dropna(subset=[target_col]).reset_index(drop=True)
y, err = encode_target(raw[target_col], classes)
if err:
    st.error(err)
    st.stop()
X = raw[features]
if len(raw) < 2:
    st.error("Need at least 2 labelled rows to evaluate.")
    st.stop()

st.success(f"Loaded {len(raw)} rows from {source}. Evaluating **{selected}** (no retraining).")

try:
    metrics, y_pred, y_proba = evaluate(selected, X, y, meta)
except Exception as exc:
    st.error(f"Prediction failed - check that feature columns hold the expected data types.\n\n{exc}")
    st.stop()

tab_metrics, tab_cm, tab_roc, tab_all, tab_pred = st.tabs(
    ["📊 Metrics", "🧩 Confusion matrix & report", "📈 ROC curve", "🏁 Compare all models", "🔎 Predictions"]
)

with tab_metrics:
    st.subheader(f"{selected} - evaluation metrics")
    cols = st.columns(len(METRIC_ORDER))
    for col, name in zip(cols, METRIC_ORDER):
        val = metrics[name]
        col.metric(name, "n/a" if np.isnan(val) else f"{val:.4f}")
    if meta["n_classes"] > 2:
        st.caption("Multi-class data: precision / recall / F1 are macro-averaged, AUC is one-vs-rest macro.")
    else:
        st.caption(f"Binary data: precision / recall / F1 are computed for the positive class **{classes[1]}**.")

with tab_cm:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Confusion matrix")
        fig, ax = plt.subplots(figsize=(4.6, 4))
        ConfusionMatrixDisplay(
            confusion_matrix(y, y_pred, labels=list(range(len(classes)))), display_labels=classes
        ).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(selected)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Classification report")
        report = classification_report(
            y, y_pred, labels=list(range(len(classes))), target_names=classes,
            output_dict=True, zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).T.round(4))

with tab_roc:
    if meta["n_classes"] == 2 and len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, y_proba[:, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, lw=2, label=f"{selected} (AUC = {metrics['AUC']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("ROC curve is shown for binary problems when both classes are present in the uploaded data.")

with tab_all:
    st.subheader("All five models on this same data")
    rows = []
    for name in model_names:
        m, _, _ = evaluate(name, X, y, meta)
        rows.append({"ML Model Name": name, **m})
    comp = pd.DataFrame(rows).set_index("ML Model Name")[METRIC_ORDER]
    try:
        st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#2e7d3266"))
    except Exception:
        st.dataframe(comp.round(4))
    best = comp["MCC"].idxmax()
    st.caption(f"Best model on this upload by MCC: **{best}** (MCC = {comp.loc[best, 'MCC']:.4f}). Green = best per metric.")
    st.bar_chart(comp[["Accuracy", "F1", "MCC"]])

with tab_pred:
    st.subheader("Row-level predictions")
    out = raw.copy()
    out["predicted_" + target_col] = [classes[i] for i in y_pred]
    for i, c in enumerate(classes):
        out[f"proba_{c}"] = y_proba[:, i].round(4)
    out["correct"] = y_pred == y
    st.dataframe(out)
    st.download_button(
        "⬇ Download predictions CSV",
        out.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{selected.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
