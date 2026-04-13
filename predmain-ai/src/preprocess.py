"""Preprocessing helpers for the PredMainAI MVP."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import MODELS_DIR, TRAIN_RATIO, VALIDATION_RATIO
from data_ingest import IngestedDataset
from database import DatabaseManager, feature_stats_from_dataframe


LOGGER = logging.getLogger(__name__)


@dataclass
class PreparedDataset:
    """All assets needed by the training and evaluation stages."""

    run_id: str
    batch_id: str
    feature_columns: list[str]
    missing_feature_columns: list[str]
    label_available: bool
    raw_splits: dict[str, pd.DataFrame]
    processed_splits: dict[str, pd.DataFrame]
    scaler_path: Path
    metadata_path: Path
    feature_statistics: pd.DataFrame


def _split_group(group: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split one machine timeline into train/validation/test partitions."""
    size = len(group)
    train_end = max(1, int(size * TRAIN_RATIO))
    val_end = max(train_end + 1, int(size * (TRAIN_RATIO + VALIDATION_RATIO)))
    if val_end >= size:
        val_end = size - 1
    if train_end >= val_end:
        train_end = max(1, val_end - 1)
    return {
        "train": group.iloc[:train_end].copy(),
        "validation": group.iloc[train_end:val_end].copy(),
        "test": group.iloc[val_end:].copy(),
    }


def split_dataset(dataframe: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the full dataset by machine while preserving time order."""
    split_parts = {"train": [], "validation": [], "test": []}
    for _, group in dataframe.groupby("machine_id", sort=True):
        group = group.sort_values(["timestamp", "sample_id"]).reset_index(drop=True)
        partitions = _split_group(group)
        for split_name, split_frame in partitions.items():
            split_parts[split_name].append(split_frame)

    combined: dict[str, pd.DataFrame] = {}
    for split_name, frames in split_parts.items():
        combined[split_name] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if combined[split_name].empty:
            raise ValueError(f"The {split_name} split ended up empty. Please provide more data.")
    return combined


def _fill_missing_values(
    splits: dict[str, pd.DataFrame],
    feature_columns: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    """Impute missing numeric values using training medians only."""
    train_frame = splits["train"]
    medians = train_frame[feature_columns].median(numeric_only=True).fillna(0.0).to_dict()
    cleaned_splits: dict[str, pd.DataFrame] = {}
    for split_name, split_frame in splits.items():
        cleaned_frame = split_frame.copy()
        for feature in feature_columns:
            cleaned_frame[feature] = pd.to_numeric(cleaned_frame[feature], errors="coerce")
            cleaned_frame[feature] = cleaned_frame[feature].fillna(medians[feature])
        cleaned_splits[split_name] = cleaned_frame
    return cleaned_splits, {key: float(value) for key, value in medians.items()}


def _scale_features(
    splits: dict[str, pd.DataFrame],
    feature_columns: list[str],
) -> tuple[dict[str, pd.DataFrame], StandardScaler]:
    """Fit a scaler on training rows and append scaled features to each split."""
    scaler = StandardScaler()
    scaler.fit(splits["train"][feature_columns])

    processed_splits: dict[str, pd.DataFrame] = {}
    for split_name, split_frame in splits.items():
        processed_frame = split_frame.copy()
        scaled_values = scaler.transform(processed_frame[feature_columns])
        for index, feature in enumerate(feature_columns):
            processed_frame[f"{feature}__scaled"] = scaled_values[:, index]
        processed_splits[split_name] = processed_frame
    return processed_splits, scaler


def preprocess_dataset(ingested: IngestedDataset, database: DatabaseManager) -> PreparedDataset:
    """Create cleaned/scaled splits, save artifacts, and persist preprocessing outputs."""
    feature_columns = list(ingested.available_feature_columns)
    if len(feature_columns) < 2:
        raise ValueError(
            "PredMainAI needs at least two recognized numeric sensor columns to train the MVP models. "
            f"Found: {feature_columns}"
        )

    raw_splits = split_dataset(ingested.dataframe)
    cleaned_splits, medians = _fill_missing_values(raw_splits, feature_columns)
    processed_splits, scaler = _scale_features(cleaned_splits, feature_columns)
    run_id = uuid.uuid4().hex[:12]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = MODELS_DIR / "preprocessor.joblib"
    metadata_path = MODELS_DIR / "preprocess_metadata.json"
    joblib.dump({"feature_columns": feature_columns, "scaler": scaler, "medians": medians}, scaler_path)
    metadata_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "batch_id": ingested.batch_id,
                "feature_columns": feature_columns,
                "missing_feature_columns": ingested.missing_feature_columns,
                "label_available": ingested.label_available,
                "medians": medians,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for split_name, processed_frame in processed_splits.items():
        database.insert_cleaned_records(
            processed_frame,
            batch_id=ingested.batch_id,
            split_name=split_name,
            missing_columns=ingested.missing_feature_columns,
            scaled_feature_columns=feature_columns,
        )

    feature_statistics = feature_stats_from_dataframe(cleaned_splits["train"], feature_columns)
    database.insert_feature_statistics(
        feature_statistics,
        run_id=run_id,
        batch_id=ingested.batch_id,
        split_name="train",
    )

    LOGGER.info(
        "Prepared dataset with %s features. Split sizes: train=%s, validation=%s, test=%s",
        len(feature_columns),
        len(processed_splits["train"]),
        len(processed_splits["validation"]),
        len(processed_splits["test"]),
    )

    return PreparedDataset(
        run_id=run_id,
        batch_id=ingested.batch_id,
        feature_columns=feature_columns,
        missing_feature_columns=ingested.missing_feature_columns,
        label_available=ingested.label_available,
        raw_splits=cleaned_splits,
        processed_splits=processed_splits,
        scaler_path=scaler_path,
        metadata_path=metadata_path,
        feature_statistics=feature_statistics,
    )
