"""Visualization helpers for the PredMain AI dashboard."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "PredMainAI" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve


def _metric_frame(results: dict[str, Any]) -> pd.DataFrame:
    """Convert model metrics into a dataframe."""
    rows = []
    for model_name, metrics in results["metrics"].items():
        rows.append(
            {
                "model": model_name,
                "display_name": next(
                    (item["display_name"] for item in results["leaderboard"] if item["model"] == model_name),
                    model_name,
                ),
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "roc_auc": metrics["roc_auc"],
            }
        )
    return pd.DataFrame(rows)


def plot_class_distribution(dataframe: pd.DataFrame):
    """Create a class distribution bar chart."""
    fig, ax = plt.subplots(figsize=(5, 3))
    if "failure_label" in dataframe.columns:
        counts = dataframe["failure_label"].map({0: "Normal", 1: "Failure"}).value_counts().reindex(["Normal", "Failure"], fill_value=0)
    else:
        counts = pd.Series({"Normal": len(dataframe), "Failure": 0})
    colors = ["#3b82f6", "#ef4444"]
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_ylabel("Rows")
    ax.set_title("Class Distribution")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_f1_comparison(results: dict[str, Any]):
    """Create an F1 comparison bar chart."""
    frame = _metric_frame(results).sort_values("f1", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(frame["display_name"], frame["f1"], color="#2563eb")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1-score")
    ax.set_title("Model F1 Comparison")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_roc_curve(results: dict[str, Any]):
    """Create ROC curves for each model when labels support ROC-AUC."""
    per_sample = results["per_sample"]
    y_true = np.array([1 if row["true_label"] == "FAILURE" else 0 for row in per_sample])
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(np.unique(y_true)) < 2:
        ax.text(0.5, 0.5, "ROC unavailable: only one class present", ha="center", va="center")
        ax.axis("off")
        return fig
    for model_name in results["metrics"]:
        scores = np.array([row["scores"][model_name] for row in per_sample])
        fpr, tpr, _ = roc_curve(y_true, scores)
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=results["metrics"][model_name]["roc_auc"], estimator_name=model_name).plot(ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    return fig


def plot_anomaly_score_distribution(results: dict[str, Any]):
    """Create anomaly score histograms for all models."""
    per_sample = results["per_sample"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for model_name in results["metrics"]:
        scores = [row["scores"][model_name] for row in per_sample]
        ax.hist(scores, bins=16, alpha=0.45, label=model_name)
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Samples")
    ax.set_title("Anomaly Score Distribution")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(results: dict[str, Any], model_name: str | None = None):
    """Create a confusion matrix for the selected or best model."""
    selected_model = model_name or results["best_model"]
    per_sample = results["per_sample"]
    y_true = np.array([1 if row["true_label"] == "FAILURE" else 0 for row in per_sample])
    y_pred = np.array([row["prediction_flags"][selected_model] for row in per_sample])
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(matrix, display_labels=["Normal", "Failure"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix: {selected_model}")
    fig.tight_layout()
    return fig
