"""Rule-based baseline model."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .common import min_max_normalize
except ImportError:  # pragma: no cover - script-friendly fallback
    from common import min_max_normalize


@dataclass
class RuleBasedModel:
    """Baseline threshold model using z-score exceedances."""

    feature_columns: list[str]
    sigma_threshold: float = 2.5
    means_: dict[str, float] = field(default_factory=dict)
    stds_: dict[str, float] = field(default_factory=dict)
    threshold_: float = 0.5

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series | None = None) -> "RuleBasedModel":
        """Fit reference statistics from normal training data when labels exist."""
        train_frame = X_train.copy()
        if y_train is not None and len(y_train) == len(train_frame) and int((y_train == 0).sum()) > 0:
            train_frame = train_frame.loc[y_train == 0]
        reference = train_frame[self.feature_columns]
        self.means_ = {feature: float(reference[feature].mean()) for feature in self.feature_columns}
        self.stds_ = {feature: float(max(reference[feature].std(ddof=0), 1e-6)) for feature in self.feature_columns}
        return self

    def _excess_matrix(self, X: pd.DataFrame) -> np.ndarray:
        frame = X[self.feature_columns]
        values = []
        for feature in self.feature_columns:
            z_scores = np.abs((frame[feature].to_numpy(dtype=float) - self.means_[feature]) / self.stds_[feature])
            values.append(np.maximum(z_scores - self.sigma_threshold, 0.0))
        return np.vstack(values).T

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """Return normalized anomaly scores."""
        excess = self._excess_matrix(X)
        raw_scores = excess.max(axis=1)
        return min_max_normalize(raw_scores, minimum=0.0, maximum=max(float(raw_scores.max()), 1.0))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict anomalies from the normalized score."""
        return (self.score_samples(X) >= self.threshold_).astype(int)

    def save(self, path: str | Path) -> None:
        payload = {
            "feature_columns": self.feature_columns,
            "sigma_threshold": self.sigma_threshold,
            "means": self.means_,
            "stds": self.stds_,
            "threshold": self.threshold_,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RuleBasedModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        instance = cls(feature_columns=payload["feature_columns"], sigma_threshold=payload["sigma_threshold"])
        instance.means_ = payload["means"]
        instance.stds_ = payload["stds"]
        instance.threshold_ = payload["threshold"]
        return instance
