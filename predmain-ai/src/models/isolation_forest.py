"""Isolation Forest anomaly detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

try:
    from .common import min_max_normalize, to_numpy
except ImportError:  # pragma: no cover - script-friendly fallback
    from common import min_max_normalize, to_numpy


@dataclass
class IsolationForestModel:
    """Thin wrapper around scikit-learn's IsolationForest."""

    feature_columns: list[str]
    contamination: float = 0.08
    random_state: int = 42
    model_: IsolationForest | None = None
    threshold_: float = 0.5
    score_min_: float = 0.0
    score_max_: float = 1.0

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series | None = None) -> "IsolationForestModel":
        reference = X_train.copy()
        if y_train is not None and len(y_train) == len(reference) and int((y_train == 0).sum()) > 0:
            reference = reference.loc[y_train == 0]
            contamination = float(np.clip(y_train.mean() * 1.5, 0.02, 0.15))
        else:
            contamination = self.contamination
        self.model_ = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=self.random_state,
        )
        self.model_.fit(to_numpy(reference, self.feature_columns))
        train_scores = -self.model_.decision_function(to_numpy(reference, self.feature_columns))
        self.score_min_ = float(train_scores.min())
        self.score_max_ = float(train_scores.max())
        self.threshold_ = float(np.quantile(train_scores, min(0.99, 1.0 - contamination / 2)))
        return self

    def _raw_scores(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("IsolationForestModel has not been fitted yet.")
        return -self.model_.decision_function(to_numpy(X, self.feature_columns))

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        return min_max_normalize(self._raw_scores(X), self.score_min_, self.score_max_)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw_scores = self._raw_scores(X)
        return (raw_scores >= self.threshold_).astype(int)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "IsolationForestModel":
        return joblib.load(Path(path))
