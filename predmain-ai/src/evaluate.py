"""Public evaluation API for PredMain AI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import DB_PATH
from data_loader import load_data, preprocess_data
from database import DatabaseManager
from evaluation_pipeline import evaluate_training, run_full_evaluation
from models import train_models
from training import TrainingArtifacts


def evaluate_models(
    predictions: TrainingArtifacts | dict[str, Any],
    database: DatabaseManager | None = None,
    source_file: str = "uploaded.csv",
) -> dict[str, Any]:
    """Evaluate model outputs and save database/results artifacts.

    Args:
        predictions: Training artifacts returned by `models.train_models`.
        database: Optional SQLite manager.
        source_file: Human-readable source name for results metadata.

    Returns:
        Results dictionary matching `outputs/results.json`.
    """
    training = predictions["training"] if isinstance(predictions, dict) and "training" in predictions else predictions
    if not isinstance(training, TrainingArtifacts):
        raise TypeError("evaluate_models expects TrainingArtifacts from models.train_models().")
    return evaluate_training(training, source_file=source_file, database=database or DatabaseManager(DB_PATH))


def generate_results(
    csv_path: str | Path,
    database: DatabaseManager | None = None,
) -> dict[str, Any]:
    """Run the full analysis pipeline from CSV to results payload."""
    if database is None:
        return run_full_evaluation(csv_path=csv_path)

    dataframe = load_data(csv_path)
    preprocessed = preprocess_data(dataframe, database=database, source_name=str(csv_path))
    training = train_models(preprocessed)
    return evaluate_models(training, database=database, source_file=str(csv_path))


if __name__ == "__main__":
    results = run_full_evaluation()
    print("Evaluation complete. Results saved to outputs/results.json")
    print(results["executive_summary"])
