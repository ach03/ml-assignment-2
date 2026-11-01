"""Shared evaluation helpers used by both this notebook and app.py."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def compute_metrics(y_true, y_pred, y_proba, n_classes):
    """Return the six required metrics as a dict."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    average = "binary" if n_classes == 2 else "macro"

    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc = roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="macro",
                labels=list(range(n_classes)),
            )
    except ValueError:
        auc = float("nan")

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def encode_target(series, classes):
    """Map the target column of an uploaded CSV to integer class ids."""
    as_str = series.astype(str).str.strip()
    lookup = {str(c): i for i, c in enumerate(classes)}
    mapped = as_str.map(lookup)
    if mapped.notna().all():
        return mapped.astype(int).to_numpy(), None

    try:
        ids = as_str.astype(float).astype(int)
        if ids.between(0, len(classes) - 1).all():
            return ids.to_numpy(), None
    except ValueError:
        pass

    unknown = sorted(set(as_str[mapped.isna()]))[:5]
    return None, (
        f"Target column contains unknown labels {unknown}. "
        f"Expected one of: {list(classes)}"
    )
