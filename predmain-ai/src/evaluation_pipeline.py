"""End-to-end evaluation pipeline for PredMainAI."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from config import DB_PATH, RESULTS_PATH, configure_logging, int_to_label
from data_ingest import ingest_csv
from database import DatabaseManager
from preprocess import preprocess_dataset
from training import TrainingArtifacts, model_health_by_name, train_models


MODEL_DISPLAY_NAMES = {
    "rule_based": "Rule-Based",
    "isolation_forest": "Isolation Forest",
    "autoencoder": "Autoencoder",
    "lstm": "LSTM",
}


def safe_float(value: Any) -> float | None:
    """Convert numpy scalar values into plain Python floats."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def verdict_for_f1(f1_value: float | None) -> str:
    """Map F1 performance to a simple MVP verdict."""
    if f1_value is None:
        return "Unavailable"
    if f1_value >= 0.75:
        return "Strong"
    if f1_value >= 0.60:
        return "Good"
    if f1_value >= 0.45:
        return "Decent"
    return "Weak"


def roc_auc_or_none(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Return ROC-AUC when the label distribution allows it."""
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, status: str, notes: str) -> dict[str, Any]:
    """Compute the standard model comparison metrics."""
    f1_value = float(f1_score(y_true, y_pred, zero_division=0))
    return {
        "status": status,
        "notes": notes,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": f1_value,
        "roc_auc": roc_auc_or_none(y_true, y_score),
        "support": int(len(y_true)),
        "flagged_count": int(np.sum(y_pred)),
        "verdict": verdict_for_f1(f1_value),
    }


def build_reference_stats(training: TrainingArtifacts) -> dict[str, dict[str, float]]:
    """Compute normal-operation feature statistics from the training split."""
    reference = training.train_frame.copy()
    if int((reference["failure_label"] == 0).sum()) > 0:
        reference = reference.loc[reference["failure_label"] == 0]
    means = {}
    stds = {}
    for feature in training.feature_columns:
        means[feature] = float(reference[feature].mean())
        stds[feature] = float(max(reference[feature].std(ddof=0), 1e-6))
    return {"mean": means, "std": stds}


def feature_deviations(row: pd.Series, reference_stats: dict[str, dict[str, float]], feature_columns: list[str]) -> list[dict[str, float]]:
    """Return a descending list of z-score-like feature deviations."""
    deviations = []
    for feature in feature_columns:
        mean_value = reference_stats["mean"][feature]
        std_value = reference_stats["std"][feature]
        z_value = abs((float(row[feature]) - mean_value) / std_value)
        deviations.append({"feature": feature, "z_score": round(float(z_value), 3)})
    return sorted(deviations, key=lambda item: item["z_score"], reverse=True)


def risk_band(health_score: float) -> str:
    """Return a readable machine health label."""
    if health_score >= 80:
        return "Healthy"
    if health_score >= 55:
        return "Watch"
    if health_score >= 30:
        return "Risky"
    return "Critical"


def explanation_for_row(row: pd.Series, score: float, prediction: int, top_features: list[dict[str, float]]) -> str:
    """Create a short human-readable explanation."""
    feature_summary = ", ".join(
        f"{item['feature']} ({item['z_score']:.1f}z)" for item in top_features[:3]
    )
    if prediction == 1:
        return f"Risk score {score:.2f}: elevated readings in {feature_summary} pushed this sample into the alert range."
    return f"Risk score {score:.2f}: readings stayed near the training baseline, with the largest movement in {feature_summary}."


def sort_leaderboard(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort models by F1-score, then recall, then accuracy."""
    rows = []
    for model_name, metric_row in metrics.items():
        rows.append(
            {
                "model": model_name,
                "display_name": MODEL_DISPLAY_NAMES[model_name],
                **metric_row,
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            item["f1"] if item["f1"] is not None else -1.0,
            item["recall"] if item["recall"] is not None else -1.0,
            item["accuracy"] if item["accuracy"] is not None else -1.0,
        ),
        reverse=True,
    )


def serialize_model_outputs(
    training: TrainingArtifacts,
    reference_stats: dict[str, dict[str, float]],
    best_model_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, int]], list[dict[str, Any]], Counter[str]]:
    """Build the per-sample payload and rows for database persistence."""
    test_frame = training.test_frame.reset_index(drop=True)
    health_lookup = model_health_by_name(training)
    feature_counter: Counter[str] = Counter()
    database_rows: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []

    model_rankings: dict[str, dict[int, int]] = {}
    for model_name, output in training.model_outputs.items():
        order = np.argsort(np.asarray(output.scores, dtype=float))[::-1]
        model_rankings[model_name] = {int(index): int(rank + 1) for rank, index in enumerate(order)}

    for index, row in test_frame.iterrows():
        predictions = {}
        prediction_flags = {}
        scores = {}
        for model_name, output in training.model_outputs.items():
            prediction_value = int(output.predictions[index])
            predictions[model_name] = int_to_label(prediction_value)
            prediction_flags[model_name] = prediction_value
            scores[model_name] = round(float(output.scores[index]), 3)

        best_score = float(scores[best_model_name])
        best_prediction = int(prediction_flags[best_model_name])
        best_health = float(health_lookup[best_model_name][index])
        deviations = feature_deviations(row, reference_stats, training.feature_columns)
        explanation = explanation_for_row(row, best_score, best_prediction, deviations)
        top_feature_names = [item["feature"] for item in deviations[:3]]
        if best_prediction == 1 or int(row["failure_label"]) == 1 or best_score >= 0.6:
            feature_counter.update(top_feature_names[:2])

        sample_payload = {
            "sample_id": int(row["sample_id"]),
            "machine_id": str(row["machine_id"]),
            "timestamp": str(row["timestamp"]),
            "true_label": int_to_label(int(row["failure_label"])),
            "predictions": predictions,
            "prediction_flags": prediction_flags,
            "scores": scores,
            "health_score": round(best_health, 2),
            "health_label": risk_band(best_health),
            "explanation": explanation,
            "top_features": top_feature_names,
        }
        per_sample.append(sample_payload)

        for model_name, output in training.model_outputs.items():
            database_rows.append(
                {
                    "sample_id": int(row["sample_id"]),
                    "model_name": model_name,
                    "predicted_label": int(output.predictions[index]),
                    "health_score": float(health_lookup[model_name][index]),
                    "explanation": explanation_for_row(
                        row,
                        float(output.scores[index]),
                        int(output.predictions[index]),
                        deviations,
                    ),
                    "anomaly_score": float(output.scores[index]),
                    "threshold": float(output.threshold),
                    "rank": model_rankings[model_name][index],
                }
            )
    ranking = sorted(
        [
            {
                "sample_id": item["sample_id"],
                "machine_id": item["machine_id"],
                "score": item["scores"][best_model_name],
                "predicted_label": item["predictions"][best_model_name],
                "true_label": item["true_label"],
                "health_score": item["health_score"],
                "explanation": item["explanation"],
            }
            for item in per_sample
        ],
        key=lambda item: item["score"],
        reverse=True,
    )
    return per_sample, database_rows, ranking, feature_counter


def error_deep_dive(per_sample: list[dict[str, Any]], best_model_name: str, limit: int = 5) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return false-positive and false-negative records for the best model."""
    false_positives = [
        row
        for row in per_sample
        if row["true_label"] == "NORMAL" and row["prediction_flags"][best_model_name] == 1
    ]
    false_negatives = [
        row
        for row in per_sample
        if row["true_label"] == "FAILURE" and row["prediction_flags"][best_model_name] == 0
    ]
    false_positives = sorted(false_positives, key=lambda item: item["scores"][best_model_name], reverse=True)[:limit]
    false_negatives = sorted(false_negatives, key=lambda item: item["scores"][best_model_name], reverse=True)[:limit]
    return false_positives, false_negatives


def persist_results(database: DatabaseManager, training: TrainingArtifacts, metrics: dict[str, dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    """Write evaluation metrics and model outputs to SQLite."""
    database.insert_model_outputs(run_id=training.run_id, batch_id=training.batch_id, rows=rows, split_name="test")
    database.insert_evaluation_results(run_id=training.run_id, batch_id=training.batch_id, metrics=metrics)


def evaluate_training(training: TrainingArtifacts, source_file: str, database: DatabaseManager) -> dict[str, Any]:
    """Compare models, persist outputs, and build results.json."""
    test_frame = training.test_frame.reset_index(drop=True)
    y_true = test_frame["failure_label"].astype(int).to_numpy()
    metrics = {
        model_name: compute_metrics(
            y_true,
            np.asarray(output.predictions, dtype=int),
            np.asarray(output.scores, dtype=float),
            output.status,
            output.notes,
        )
        for model_name, output in training.model_outputs.items()
    }
    leaderboard = sort_leaderboard(metrics)
    best_model_name = leaderboard[0]["model"]
    reference_stats = build_reference_stats(training)
    per_sample, database_rows, anomaly_ranking, feature_counter = serialize_model_outputs(
        training=training,
        reference_stats=reference_stats,
        best_model_name=best_model_name,
    )
    false_positives, false_negatives = error_deep_dive(per_sample, best_model_name)

    persist_results(database, training, metrics, database_rows)
    feature_importance = dict(feature_counter.most_common())
    executive_summary = (
        f"{MODEL_DISPLAY_NAMES[best_model_name]} delivered the best F1-score "
        f"({metrics[best_model_name]['f1']:.3f}) on {len(test_frame)} test samples, "
        f"with {feature_counter.most_common(1)[0][0] if feature_counter else 'sensor drift'} as the top risk signal."
    )

    results_payload = {
        "run_id": training.run_id,
        "batch_id": training.batch_id,
        "source_file": source_file,
        "database_path": str(database.db_path),
        "feature_columns": training.feature_columns,
        "metrics": metrics,
        "leaderboard": leaderboard,
        "best_model": best_model_name,
        "per_sample": per_sample,
        "feature_importance": feature_importance,
        "anomaly_score_ranking": anomaly_ranking,
        "false_positive_deep_dive": false_positives,
        "false_negative_deep_dive": false_negatives,
        "reference_stats": reference_stats,
        "executive_summary": executive_summary,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    return results_payload


def run_full_evaluation(csv_path: str | Path | None = None, database_path: str | Path | None = None) -> dict[str, Any]:
    """Run the complete pipeline from ingestion to results.json."""
    configure_logging()
    database = DatabaseManager(Path(database_path) if database_path else DB_PATH)
    ingested = ingest_csv(csv_path, database=database)
    prepared = preprocess_dataset(ingested, database=database)
    training = train_models(prepared)
    return evaluate_training(training, source_file=str(ingested.source_path), database=database)
