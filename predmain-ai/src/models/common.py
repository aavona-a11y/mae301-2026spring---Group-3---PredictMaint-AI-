"""Shared model utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_numpy(data: pd.DataFrame | pd.Series | np.ndarray, feature_columns: list[str] | None = None) -> np.ndarray:
    """Return a numeric numpy matrix from a dataframe or array."""
    if isinstance(data, pd.DataFrame):
        if feature_columns is None:
            return data.to_numpy(dtype=float)
        return data[feature_columns].to_numpy(dtype=float)
    if isinstance(data, pd.Series):
        return data.to_numpy(dtype=float)
    return np.asarray(data, dtype=float)


def min_max_normalize(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    """Normalize values to the [0, 1] interval."""
    span = max(maximum - minimum, 1e-8)
    normalized = (values - minimum) / span
    return np.clip(normalized, 0.0, 1.0)
