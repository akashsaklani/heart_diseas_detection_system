"""Exploratory data analysis plots (workflow step 3)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config


def _save(fig, name: str) -> str:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FIGURES_DIR / name
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def class_balance(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.countplot(x=config.TARGET, data=df, ax=ax, hue=config.TARGET, legend=False)
    ax.set_title("Target class distribution")
    return _save(fig, "class_balance.png")


def histograms(df: pd.DataFrame) -> str:
    cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, c in zip(axes.ravel(), cols):
        sns.histplot(df[c], kde=True, ax=ax)
        ax.set_title(c)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Numeric feature distributions")
    return _save(fig, "histograms.png")


def correlation_heatmap(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation heatmap")
    return _save(fig, "correlation_heatmap.png")


def categorical_vs_target(df: pd.DataFrame) -> str:
    cols = ["sex", "cp", "fbs", "restecg", "exang", "slope"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, c in zip(axes.ravel(), cols):
        sns.countplot(x=c, hue=config.TARGET, data=df, ax=ax)
        ax.set_title(f"{c} vs target")
    fig.tight_layout()
    return _save(fig, "categorical_vs_target.png")


def run_all(df: pd.DataFrame) -> list[str]:
    return [
        class_balance(df),
        histograms(df),
        correlation_heatmap(df),
        categorical_vs_target(df),
    ]
