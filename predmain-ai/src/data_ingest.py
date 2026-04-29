"""CSV ingestion and normalization for the PredMainAI MVP."""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from config import CANONICAL_SENSOR_COLUMNS, COLUMN_ALIASES, DEFAULT_SAMPLE_CSV, label_to_int
from database import DatabaseManager
from sample_data import generate_synthetic_dataset


LOGGER = logging.getLogger(__name__)


def _slugify_column(value: str) -> str:
    """Normalize free-form column names."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned


def normalize_columns(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename known aliases to the project's canonical schema."""
    rename_map: dict[str, str] = {}
    for original_name in dataframe.columns:
        slug = _slugify_column(original_name)
        canonical_name = None
        for candidate, aliases in COLUMN_ALIASES.items():
            alias_slugs = {_slugify_column(alias) for alias in aliases}
            if slug in alias_slugs:
                canonical_name = candidate
                break
        if canonical_name is not None:
            rename_map[original_name] = canonical_name
    normalized = dataframe.rename(columns=rename_map).copy()
    normalized.columns = [_slugify_column(column) for column in normalized.columns]
    return normalized, rename_map


@dataclass
class IngestedDataset:
    """In-memory representation of a raw ingestion batch."""

    dataframe: pd.DataFrame
    batch_id: str
    source_path: Path
    available_feature_columns: list[str]
    missing_feature_columns: list[str]
    label_available: bool
    notes: list[str] = field(default_factory=list)


def _ensure_machine_and_time_columns(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Fill optional ID/time columns when a CSV omits them."""
    df = dataframe.copy()
    notes: list[str] = []
    if "machine_id" not in df.columns:
        df["machine_id"] = "M-001"
        notes.append("Missing machine_id column; filled with a default machine identifier.")
    if "timestamp" not in df.columns:
        start = pd.Timestamp("2026-01-01 00:00:00")
        df["timestamp"] = [(start + pd.Timedelta(minutes=5 * index)).isoformat() for index in range(len(df))]
        notes.append("Missing timestamp column; generated a simple 5-minute timeline.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("One or more timestamp values could not be parsed. Please provide ISO-like timestamps.")
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return df, notes


def _coerce_numeric_sensor_columns(dataframe: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Convert available sensor columns to numeric types."""
    df = dataframe.copy()
    for feature in feature_columns:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
    return df


def _resolve_input_path(csv_path: str | Path | None) -> Path:
    """Return an existing CSV path, generating the sample dataset if needed."""
    if csv_path is None:
        return generate_synthetic_dataset(DEFAULT_SAMPLE_CSV)
    resolved = Path(csv_path)
    if resolved.exists():
        return resolved
    if resolved == DEFAULT_SAMPLE_CSV or resolved.name == DEFAULT_SAMPLE_CSV.name:
        return generate_synthetic_dataset(DEFAULT_SAMPLE_CSV)
    raise FileNotFoundError(f"CSV file not found: {resolved}")


def _prepare_ingested_dataframe(
    raw_dataframe: pd.DataFrame,
    source_name: str,
    database: DatabaseManager,
) -> IngestedDataset:
    """Normalize, validate, persist, and package a raw dataframe."""
    if raw_dataframe.empty:
        raise ValueError(f"The input CSV is empty: {source_name}")

    normalized, rename_map = normalize_columns(raw_dataframe)
    normalized, notes = _ensure_machine_and_time_columns(normalized)

    available_feature_columns = [feature for feature in CANONICAL_SENSOR_COLUMNS if feature in normalized.columns]
    missing_feature_columns = [feature for feature in CANONICAL_SENSOR_COLUMNS if feature not in normalized.columns]
    if not available_feature_columns:
        raise ValueError(
            "No recognized sensor columns were found. Expected at least one of: "
            + ", ".join(CANONICAL_SENSOR_COLUMNS)
        )

    normalized = _coerce_numeric_sensor_columns(normalized, available_feature_columns)
    normalized["sample_id"] = range(1, len(normalized) + 1)

    label_available = "failure_label" in normalized.columns
    if label_available:
        normalized["failure_label"] = normalized["failure_label"].map(label_to_int)
    else:
        normalized["failure_label"] = 0
        notes.append("Missing failure_label column; evaluation metrics will be less informative for supervised models.")

    ordered_columns = [
        "sample_id",
        "machine_id",
        "timestamp",
        *available_feature_columns,
        "failure_label",
    ]
    normalized = normalized[ordered_columns].sort_values(["machine_id", "timestamp", "sample_id"]).reset_index(drop=True)
    batch_id = uuid.uuid4().hex[:12]
    database.insert_raw_records(normalized, batch_id=batch_id, source_file=source_name)

    if rename_map:
        LOGGER.info("Normalized columns: %s", rename_map)
    for note in notes:
        LOGGER.warning(note)

    return IngestedDataset(
        dataframe=normalized,
        batch_id=batch_id,
        source_path=Path(source_name),
        available_feature_columns=available_feature_columns,
        missing_feature_columns=missing_feature_columns,
        label_available=label_available,
        notes=notes,
    )


def ingest_dataframe(
    dataframe: pd.DataFrame,
    database: DatabaseManager,
    source_name: str = "uploaded.csv",
) -> IngestedDataset:
    """Normalize and persist an in-memory CSV dataframe."""
    return _prepare_ingested_dataframe(dataframe.copy(), source_name=source_name, database=database)


def ingest_csv(csv_path: str | Path | None, database: DatabaseManager) -> IngestedDataset:
    """Load a CSV, normalize its schema, store the raw rows, and return a dataframe."""
    source_path = _resolve_input_path(csv_path)
    raw_dataframe = pd.read_csv(source_path)
    return _prepare_ingested_dataframe(raw_dataframe, source_name=str(source_path), database=database)
