"""Shared configuration for the PredMainAI MVP."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
DB_PATH = OUTPUTS_DIR / "predmainai.db"
RESULTS_PATH = OUTPUTS_DIR / "results.json"
DEFAULT_SAMPLE_CSV = RAW_DATA_DIR / "sample_sensor_data.csv"

CANONICAL_SENSOR_COLUMNS = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "vibration",
]

COLUMN_ALIASES = {
    "machine_id": {
        "machine_id",
        "machine id",
        "machine",
        "asset_id",
        "asset id",
        "unit_id",
        "product_id",
        "product id",
    },
    "timestamp": {
        "timestamp",
        "time",
        "datetime",
        "date",
        "event_time",
        "event time",
    },
    "air_temperature": {
        "air_temperature",
        "air temperature",
        "air temperature [k]",
        "air_temp",
        "air temp",
    },
    "process_temperature": {
        "process_temperature",
        "process temperature",
        "process temperature [k]",
        "process_temp",
        "process temp",
    },
    "rotational_speed": {
        "rotational_speed",
        "rotational speed",
        "rotational speed [rpm]",
        "rpm",
        "speed",
    },
    "torque": {
        "torque",
        "torque [nm]",
        "load_torque",
        "load torque",
    },
    "tool_wear": {
        "tool_wear",
        "tool wear",
        "tool wear [min]",
        "wear",
    },
    "vibration": {
        "vibration",
        "vibration_level",
        "vibration level",
        "vibe",
    },
    "failure_label": {
        "failure_label",
        "failure",
        "failure label",
        "machine_failure",
        "machine failure",
        "label",
        "target",
    },
}

RANDOM_SEED = 42
TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2


@dataclass(frozen=True)
class RuntimeConfig:
    """Settings that keep the MVP deterministic and lightweight."""

    random_seed: int = RANDOM_SEED
    lstm_window_size: int = 8
    autoencoder_epochs: int = 20
    lstm_epochs: int = 18
    batch_size: int = 32


def ensure_directories() -> None:
    """Create the project directories used by the pipeline."""
    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a consistent logger for CLI usage."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def label_to_int(value: object) -> int:
    """Convert label-like values to 0 or 1."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(float(value) > 0)
    text = str(value).strip().lower()
    return int(text in {"1", "true", "failure", "fail", "yes", "y", "anomaly"})


def int_to_label(value: int | None) -> str:
    """Return a readable label name."""
    if value is None:
        return "UNKNOWN"
    return "FAILURE" if int(value) == 1 else "NORMAL"
