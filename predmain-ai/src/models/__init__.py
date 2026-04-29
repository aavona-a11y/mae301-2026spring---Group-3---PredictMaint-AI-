"""Model package and public model API for PredMain AI."""

from __future__ import annotations

from typing import Any

import pandas as pd

from preprocess import PreparedDataset
from training import TrainingArtifacts, model_health_by_name, train_models as train_pipeline_models


def _prepared_from_payload(preprocessed: dict[str, Any] | PreparedDataset) -> PreparedDataset:
    """Extract a PreparedDataset from either public payload shape."""
    return preprocessed["prepared"] if isinstance(preprocessed, dict) else preprocessed


def train_models(preprocessed: dict[str, Any] | PreparedDataset) -> TrainingArtifacts:
    """Train all predictive maintenance models on the prepared dataset."""
    return train_pipeline_models(_prepared_from_payload(preprocessed))


def run_inference(training: TrainingArtifacts) -> dict[str, Any]:
    """Return per-model predictions, anomaly scores, and health scores."""
    health_scores = model_health_by_name(training)
    model_payload = {}
    for model_name, output in training.model_outputs.items():
        model_payload[model_name] = {
            "predictions": output.predictions,
            "scores": output.scores,
            "threshold": output.threshold,
            "health_scores": health_scores[model_name],
            "status": output.status,
            "notes": output.notes,
        }
    return {
        "run_id": training.run_id,
        "batch_id": training.batch_id,
        "feature_columns": training.feature_columns,
        "test_frame": training.test_frame.copy(),
        "models": model_payload,
    }


def predictions_to_dataframe(inference: dict[str, Any]) -> pd.DataFrame:
    """Convert inference output into a flat dataframe for UI tables."""
    frame = inference["test_frame"].copy().reset_index(drop=True)
    for model_name, payload in inference["models"].items():
        frame[f"{model_name}_prediction"] = payload["predictions"]
        frame[f"{model_name}_score"] = payload["scores"]
        frame[f"{model_name}_health"] = payload["health_scores"]
    return frame
