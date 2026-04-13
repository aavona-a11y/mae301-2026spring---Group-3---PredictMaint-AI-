"""Training orchestration for the PredMainAI MVP."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import MODELS_DIR, RuntimeConfig
from preprocess import PreparedDataset


LOGGER = logging.getLogger(__name__)
MODELS_SOURCE_DIR = Path(__file__).resolve().parent / "models"
if str(MODELS_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_SOURCE_DIR))

from autoencoder import AutoencoderModel  # noqa: E402
from isolation_forest import IsolationForestModel  # noqa: E402
from lstm_model import LSTMModel  # noqa: E402
from rule_based import RuleBasedModel  # noqa: E402


@dataclass
class ModelOutput:
    """Model predictions and metadata for one split."""

    name: str
    predictions: list[int]
    scores: list[float]
    threshold: float
    model_path: str
    status: str = "ok"
    notes: str = ""


@dataclass
class TrainingArtifacts:
    """All trained models and test outputs needed for evaluation."""

    run_id: str
    batch_id: str
    feature_columns: list[str]
    test_frame: pd.DataFrame
    train_frame: pd.DataFrame
    feature_statistics: pd.DataFrame
    model_outputs: dict[str, ModelOutput]


def _scaled_columns(feature_columns: list[str]) -> list[str]:
    return [f"{feature}__scaled" for feature in feature_columns]


def train_models(prepared: PreparedDataset) -> TrainingArtifacts:
    """Train the four MVP models and persist them to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    feature_columns = prepared.feature_columns
    scaled_columns = _scaled_columns(feature_columns)
    train_raw = prepared.raw_splits["train"]
    test_raw = prepared.raw_splits["test"]
    train_scaled = prepared.processed_splits["train"]
    test_scaled = prepared.processed_splits["test"]
    y_train = train_raw["failure_label"].astype(int)

    outputs: dict[str, ModelOutput] = {}

    rule_model = RuleBasedModel(feature_columns=feature_columns).fit(train_raw, y_train)
    rule_path = MODELS_DIR / "rule_based.json"
    rule_model.save(rule_path)
    rule_scores = rule_model.score_samples(test_raw)
    outputs["rule_based"] = ModelOutput(
        name="rule_based",
        predictions=rule_model.predict(test_raw).astype(int).tolist(),
        scores=np.asarray(rule_scores, dtype=float).round(6).tolist(),
        threshold=rule_model.threshold_,
        model_path=str(rule_path),
    )

    isolation_model = IsolationForestModel(feature_columns=scaled_columns, random_state=RuntimeConfig().random_seed)
    isolation_model.fit(train_scaled, y_train)
    isolation_path = MODELS_DIR / "isolation_forest.joblib"
    isolation_model.save(isolation_path)
    isolation_scores = isolation_model.score_samples(test_scaled)
    outputs["isolation_forest"] = ModelOutput(
        name="isolation_forest",
        predictions=isolation_model.predict(test_scaled).astype(int).tolist(),
        scores=np.asarray(isolation_scores, dtype=float).round(6).tolist(),
        threshold=float(min(max(isolation_model.threshold_, 0.0), 1.0)),
        model_path=str(isolation_path),
    )

    autoencoder_model = AutoencoderModel(feature_columns=scaled_columns, config=RuntimeConfig())
    autoencoder_model.fit(train_scaled, y_train)
    autoencoder_path = MODELS_DIR / "autoencoder.pt"
    autoencoder_model.save(autoencoder_path)
    autoencoder_scores = autoencoder_model.score_samples(test_scaled)
    outputs["autoencoder"] = ModelOutput(
        name="autoencoder",
        predictions=autoencoder_model.predict(test_scaled).astype(int).tolist(),
        scores=np.asarray(autoencoder_scores, dtype=float).round(6).tolist(),
        threshold=float(min(max(autoencoder_model.threshold_, 0.0), 1.0)),
        model_path=str(autoencoder_path),
    )

    try:
        lstm_model = LSTMModel(feature_columns=scaled_columns, config=RuntimeConfig())
        lstm_model.fit(train_scaled, y_train)
        lstm_path = MODELS_DIR / "lstm_model.pt"
        lstm_model.save(lstm_path)
        lstm_scores = lstm_model.score_samples(test_scaled)
        outputs["lstm"] = ModelOutput(
            name="lstm",
            predictions=lstm_model.predict(test_scaled).astype(int).tolist(),
            scores=np.asarray(lstm_scores, dtype=float).round(6).tolist(),
            threshold=lstm_model.threshold_,
            model_path=str(lstm_path),
        )
    except ValueError as exc:
        LOGGER.warning("Skipping LSTM model: %s", exc)
        outputs["lstm"] = ModelOutput(
            name="lstm",
            predictions=[0] * len(test_raw),
            scores=[0.0] * len(test_raw),
            threshold=0.5,
            model_path="",
            status="skipped",
            notes=str(exc),
        )

    return TrainingArtifacts(
        run_id=prepared.run_id,
        batch_id=prepared.batch_id,
        feature_columns=feature_columns,
        test_frame=test_raw.copy(),
        train_frame=train_raw.copy(),
        feature_statistics=prepared.feature_statistics.copy(),
        model_outputs=outputs,
    )


def model_health_by_name(training: TrainingArtifacts) -> dict[str, list[float]]:
    """Convert anomaly scores into 0-100 health scores."""
    return {
        model_name: np.clip(100.0 * (1.0 - np.asarray(output.scores, dtype=float)), 0.0, 100.0).round(3).tolist()
        for model_name, output in training.model_outputs.items()
    }
