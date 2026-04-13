"""PyTorch autoencoder for anomaly detection."""

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
    from .common import min_max_normalize, to_numpy
except ImportError:  # pragma: no cover - script-friendly fallback
    from common import min_max_normalize, to_numpy


class _AutoencoderNet(nn.Module):
    """Compact feed-forward autoencoder."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        hidden_dim = max(4, input_dim * 2)
        bottleneck = max(2, input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))


@dataclass
class AutoencoderModel:
    """Reconstruction-based anomaly detector."""

    feature_columns: list[str]
    config: RuntimeConfig = field(default_factory=RuntimeConfig)
    model_: _AutoencoderNet | None = None
    threshold_: float = 0.5
    error_min_: float = 0.0
    error_max_: float = 1.0

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series | None = None) -> "AutoencoderModel":
        train_frame = X_train.copy()
        if y_train is not None and len(y_train) == len(train_frame) and int((y_train == 0).sum()) > 0:
            train_frame = train_frame.loc[y_train == 0]
        tensor = torch.tensor(to_numpy(train_frame, self.feature_columns), dtype=torch.float32)
        dataset = TensorDataset(tensor, tensor)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        torch.manual_seed(self.config.random_seed)
        self.model_ = _AutoencoderNet(len(self.feature_columns))
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        self.model_.train()
        for _ in range(self.config.autoencoder_epochs):
            for batch_inputs, batch_targets in loader:
                optimizer.zero_grad()
                reconstruction = self.model_(batch_inputs)
                loss = criterion(reconstruction, batch_targets)
                loss.backward()
                optimizer.step()

        train_errors = self._raw_errors(train_frame)
        self.error_min_ = float(train_errors.min())
        self.error_max_ = float(train_errors.max())
        self.threshold_ = float(np.quantile(train_errors, 0.95))
        return self

    def _raw_errors(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("AutoencoderModel has not been fitted yet.")
        self.model_.eval()
        with torch.no_grad():
            tensor = torch.tensor(to_numpy(X, self.feature_columns), dtype=torch.float32)
            reconstruction = self.model_(tensor)
            errors = ((tensor - reconstruction) ** 2).mean(dim=1).cpu().numpy()
        return errors

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        return min_max_normalize(self._raw_errors(X), self.error_min_, self.error_max_)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self._raw_errors(X) >= self.threshold_).astype(int)

    def save(self, path: str | Path) -> None:
        if self.model_ is None:
            raise ValueError("AutoencoderModel has not been fitted yet.")
        torch.save(
            {
                "state_dict": self.model_.state_dict(),
                "feature_columns": self.feature_columns,
                "threshold": self.threshold_,
                "error_min": self.error_min_,
                "error_max": self.error_max_,
                "config": self.config.__dict__,
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "AutoencoderModel":
        payload = torch.load(Path(path), map_location="cpu")
        instance = cls(feature_columns=payload["feature_columns"], config=RuntimeConfig(**payload["config"]))
        instance.model_ = _AutoencoderNet(len(instance.feature_columns))
        instance.model_.load_state_dict(payload["state_dict"])
        instance.model_.eval()
        instance.threshold_ = payload["threshold"]
        instance.error_min_ = payload["error_min"]
        instance.error_max_ = payload["error_max"]
        return instance
