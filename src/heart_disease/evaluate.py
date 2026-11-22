"""Evaluation: accuracy, precision, recall, F1, confusion matrix (workflow step 6)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

from . import config
from .model import candidate_models


def compute_metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    out = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
    }
    if hasattr(model, "predict_proba"):
        out["roc_auc"] = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    return out


def save_confusion_matrix(model, X_test, y_test) -> str:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues")
    ax.set_title("Confusion matrix (test set)")
    out = config.FIGURES_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def save_feature_importance(model, feature_names) -> str:
    imp = (
        pd.Series(model.feature_importances_, index=feature_names)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(6, 4.5))
    imp.plot.barh(ax=ax)
    ax.set_title("Random Forest feature importance")
    ax.set_xlabel("importance")
    out = config.FIGURES_DIR / "feature_importance.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def compare_models(X, y, folds: int = config.CV_FOLDS) -> pd.DataFrame:
    """Cross-validated comparison of all five algorithms (future-scope item)."""
    rows = []
    for name, mdl in candidate_models().items():
        row = {"model": name}
        for metric in ("accuracy", "precision", "recall", "f1"):
            s = cross_val_score(mdl, X, y, cv=folds, scoring=metric)
            row[f"cv_{metric}_mean"] = round(float(s.mean()), 4)
            row[f"cv_{metric}_std"] = round(float(s.std()), 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cv_f1_mean", ascending=False)
