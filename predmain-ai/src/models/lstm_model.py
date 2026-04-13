"""LSTM-based risk model for machine sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import RuntimeConfig

try:
    from .common import to_numpy
except ImportError:  # pragma: no cover - script-friendly fallback
    from common import to_numpy


class _SequenceNet(nn.Module):
    """Compact sequence classifier."""

    def __init__(self, input_dim: int, hidden_dim: int = 24) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(inputs)
        logits = self.output(output[:, -1, :])
        return logits.squeeze(-1)


@dataclass
class LSTMModel:
    """Sequence-aware supervised risk model."""

    feature_columns: list[str]
    config: RuntimeConfig = field(default_factory=RuntimeConfig)
    model_: _SequenceNet | None = None
    threshold_: float = 0.5

    def _build_windows(self, X: pd.DataFrame, y: pd.Series | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        windows: list[np.ndarray] = []
        labels: list[int] = []
        ordered = X.sort_values(["machine_id", "timestamp", "sample_id"]).reset_index(drop=True)
        label_array = pd.Series(y).reset_index(drop=True).to_numpy(dtype=int) if y is not None else None
        label_cursor = 0

        for _, group in ordered.groupby("machine_id", sort=True):
            group = group.sort_values(["timestamp", "sample_id"]).reset_index(drop=True)
            feature_values = to_numpy(group, self.feature_columns)
            group_length = len(group)
            group_labels = label_array[label_cursor : label_cursor + group_length] if label_array is not None else None
            label_cursor += group_length

            for row_index in range(group_length):
                start_index = max(0, row_index - self.config.lstm_window_size + 1)
                window = feature_values[start_index : row_index + 1]
                if len(window) < self.config.lstm_window_size:
                    pad = np.repeat(window[:1], self.config.lstm_window_size - len(window), axis=0)
                    window = np.vstack([pad, window])
                windows.append(window)
                if group_labels is not None:
                    labels.append(int(group_labels[row_index]))

        return np.asarray(windows, dtype=np.float32), np.asarray(labels, dtype=np.float32) if labels else None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series | None = None) -> "LSTMModel":
        if y_train is None or len(set(pd.Series(y_train).astype(int).tolist())) < 2:
            raise ValueError("LSTMModel requires a failure_label column with both normal and failure examples.")

        torch.manual_seed(self.config.random_seed)
        self.model_ = _SequenceNet(len(self.feature_columns))
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        train_windows, train_labels = self._build_windows(X_train, y_train)
        dataset = TensorDataset(
            torch.tensor(train_windows, dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        self.model_.train()
        for _ in range(self.config.lstm_epochs):
            for batch_inputs, batch_labels in loader:
                optimizer.zero_grad()
                logits = self.model_(batch_inputs)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
        return self

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("LSTMModel has not been fitted yet.")
        windows, _ = self._build_windows(X)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.tensor(windows, dtype=torch.float32))
            probabilities = torch.sigmoid(logits).cpu().numpy()
        return probabilities

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.score_samples(X) >= self.threshold_).astype(int)

    def save(self, path: str | Path) -> None:
        if self.model_ is None:
            raise ValueError("LSTMModel has not been fitted yet.")
        torch.save(
            {
                "state_dict": self.model_.state_dict(),
                "feature_columns": self.feature_columns,
                "threshold": self.threshold_,
                "config": self.config.__dict__,
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "LSTMModel":
        payload = torch.load(Path(path), map_location="cpu")
        instance = cls(feature_columns=payload["feature_columns"], config=RuntimeConfig(**payload["config"]))
        instance.model_ = _SequenceNet(len(instance.feature_columns))
        instance.model_.load_state_dict(payload["state_dict"])
        instance.model_.eval()
        instance.threshold_ = payload["threshold"]
        return instance
