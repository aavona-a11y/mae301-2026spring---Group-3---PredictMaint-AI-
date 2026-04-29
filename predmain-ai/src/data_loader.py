"""Public data-loading API for PredMain AI.

These functions keep the course-facing module name stable while reusing the
database-backed ingestion and preprocessing internals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config import DB_PATH
from data_ingest import ingest_dataframe
from database import DatabaseManager
from preprocess import PreparedDataset, preprocess_dataset


def load_data(csv_path: str | Path | Any) -> pd.DataFrame:
    """Load a CSV file or Streamlit upload object into a dataframe.

    Args:
        csv_path: Filesystem path or file-like object accepted by `pandas.read_csv`.

    Returns:
        Raw dataframe exactly as provided by the user.
    """
    try:
        dataframe = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The uploaded CSV is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError("The uploaded file is not a valid CSV.") from exc

    if dataframe.empty:
        raise ValueError("The uploaded CSV is empty.")
    return dataframe


def preprocess_data(
    dataframe: pd.DataFrame,
    database: DatabaseManager | None = None,
    source_name: str = "uploaded.csv",
) -> dict[str, Any]:
    """Normalize, validate, split, scale, and persist a dataframe.

    Args:
        dataframe: Raw input dataframe.
        database: Optional database manager. A default SQLite database is used
            when no manager is supplied.
        source_name: Name shown in database records and result metadata.

    Returns:
        Dictionary containing the ingested dataset, prepared splits, feature
        columns, and UI-friendly warnings.
    """
    database = database or DatabaseManager(DB_PATH)
    ingested = ingest_dataframe(dataframe, database=database, source_name=source_name)
    prepared = preprocess_dataset(ingested, database=database)
    return {
        "database": database,
        "ingested": ingested,
        "prepared": prepared,
        "feature_columns": prepared.feature_columns,
        "missing_feature_columns": prepared.missing_feature_columns,
        "label_available": prepared.label_available,
        "raw_splits": prepared.raw_splits,
        "processed_splits": prepared.processed_splits,
        "warnings": list(ingested.notes),
    }


def get_training_frame(preprocessed: dict[str, Any] | PreparedDataset) -> pd.DataFrame:
    """Return the processed training split from a preprocessed payload."""
    prepared = preprocessed["prepared"] if isinstance(preprocessed, dict) else preprocessed
    return prepared.processed_splits["train"]


def get_test_frame(preprocessed: dict[str, Any] | PreparedDataset) -> pd.DataFrame:
    """Return the raw test split from a preprocessed payload."""
    prepared = preprocessed["prepared"] if isinstance(preprocessed, dict) else preprocessed
    return prepared.raw_splits["test"]
