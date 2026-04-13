"""SQLite helpers for the PredMainAI MVP."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from config import DB_PATH


def utc_now_iso() -> str:
    """Return a stable UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class DatabaseManager:
    """Thin wrapper around SQLite used by the pipeline."""

    db_path: Path = DB_PATH

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.initialize_schema()
        except sqlite3.Error:
            self.db_path = self.db_path.with_name(f"{self.db_path.stem}_runtime.db")
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.initialize_schema()

    def connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with row access by name."""
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=MEMORY")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        return connection

    def initialize_schema(self) -> None:
        """Create the MVP tables if they do not exist yet."""
        statements = [
            """
            CREATE TABLE IF NOT EXISTS raw_sensor_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL,
                sample_id INTEGER NOT NULL,
                source_file TEXT NOT NULL,
                ingested_at TEXT NOT NULL,
                machine_id TEXT,
                timestamp TEXT,
                air_temperature REAL,
                process_temperature REAL,
                rotational_speed REAL,
                torque REAL,
                tool_wear REAL,
                vibration REAL,
                failure_label INTEGER,
                raw_payload TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS cleaned_sensor_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL,
                sample_id INTEGER NOT NULL,
                split_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                machine_id TEXT,
                timestamp TEXT,
                air_temperature REAL,
                process_temperature REAL,
                rotational_speed REAL,
                torque REAL,
                tool_wear REAL,
                vibration REAL,
                failure_label INTEGER,
                scaled_payload TEXT,
                missing_columns TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS model_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                sample_id INTEGER NOT NULL,
                split_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                predicted_label INTEGER,
                health_score REAL,
                explanation TEXT,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS anomaly_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                sample_id INTEGER NOT NULL,
                split_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                anomaly_score REAL,
                threshold REAL,
                rank INTEGER,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS evaluation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1 REAL,
                roc_auc REAL,
                support INTEGER,
                flagged_count INTEGER,
                verdict TEXT,
                summary_json TEXT,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS feature_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                split_name TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                mean REAL,
                std REAL,
                min REAL,
                max REAL,
                median REAL,
                created_at TEXT NOT NULL
            )
            """,
        ]
        with self.connect() as connection:
            for statement in statements:
                connection.execute(statement)
            connection.commit()

    def query_dataframe(self, sql: str, params: Iterable[object] | None = None) -> pd.DataFrame:
        """Return a SQL query as a dataframe."""
        with self.connect() as connection:
            return pd.read_sql_query(sql, connection, params=list(params or []))

    def insert_raw_records(self, dataframe: pd.DataFrame, batch_id: str, source_file: str) -> None:
        """Persist raw ingested rows."""
        inserted_at = utc_now_iso()
        rows = []
        for row in dataframe.to_dict(orient="records"):
            rows.append(
                (
                    batch_id,
                    int(row["sample_id"]),
                    source_file,
                    inserted_at,
                    row.get("machine_id"),
                    row.get("timestamp"),
                    row.get("air_temperature"),
                    row.get("process_temperature"),
                    row.get("rotational_speed"),
                    row.get("torque"),
                    row.get("tool_wear"),
                    row.get("vibration"),
                    row.get("failure_label"),
                    json.dumps(row, default=str),
                )
            )
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO raw_sensor_records (
                    batch_id, sample_id, source_file, ingested_at, machine_id, timestamp,
                    air_temperature, process_temperature, rotational_speed, torque,
                    tool_wear, vibration, failure_label, raw_payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            connection.commit()

    def insert_cleaned_records(
        self,
        dataframe: pd.DataFrame,
        batch_id: str,
        split_name: str,
        missing_columns: list[str],
        scaled_feature_columns: list[str],
    ) -> None:
        """Persist cleaned and scaled rows for one split."""
        created_at = utc_now_iso()
        rows = []
        for row in dataframe.to_dict(orient="records"):
            scaled_payload = {
                feature: row.get(f"{feature}__scaled")
                for feature in scaled_feature_columns
                if f"{feature}__scaled" in row
            }
            rows.append(
                (
                    batch_id,
                    int(row["sample_id"]),
                    split_name,
                    created_at,
                    row.get("machine_id"),
                    row.get("timestamp"),
                    row.get("air_temperature"),
                    row.get("process_temperature"),
                    row.get("rotational_speed"),
                    row.get("torque"),
                    row.get("tool_wear"),
                    row.get("vibration"),
                    row.get("failure_label"),
                    json.dumps(scaled_payload),
                    json.dumps(missing_columns),
                )
            )
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO cleaned_sensor_records (
                    batch_id, sample_id, split_name, created_at, machine_id, timestamp,
                    air_temperature, process_temperature, rotational_speed, torque,
                    tool_wear, vibration, failure_label, scaled_payload, missing_columns
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            connection.commit()

    def insert_feature_statistics(
        self,
        statistics: pd.DataFrame,
        run_id: str,
        batch_id: str,
        split_name: str = "train",
    ) -> None:
        """Persist feature summary statistics."""
        created_at = utc_now_iso()
        rows = []
        for row in statistics.to_dict(orient="records"):
            rows.append(
                (
                    run_id,
                    batch_id,
                    split_name,
                    row["feature_name"],
                    row["mean"],
                    row["std"],
                    row["min"],
                    row["max"],
                    row["median"],
                    created_at,
                )
            )
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO feature_statistics (
                    run_id, batch_id, split_name, feature_name, mean, std, min, max, median, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            connection.commit()

    def insert_model_outputs(
        self,
        run_id: str,
        batch_id: str,
        rows: list[dict[str, object]],
        split_name: str = "test",
    ) -> None:
        """Persist prediction and anomaly-score outputs."""
        created_at = utc_now_iso()
        prediction_rows = []
        score_rows = []
        for row in rows:
            prediction_rows.append(
                (
                    run_id,
                    batch_id,
                    int(row["sample_id"]),
                    split_name,
                    str(row["model_name"]),
                    row.get("predicted_label"),
                    row.get("health_score"),
                    row.get("explanation"),
                    created_at,
                )
            )
            score_rows.append(
                (
                    run_id,
                    batch_id,
                    int(row["sample_id"]),
                    split_name,
                    str(row["model_name"]),
                    row.get("anomaly_score"),
                    row.get("threshold"),
                    row.get("rank"),
                    created_at,
                )
            )
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO model_predictions (
                    run_id, batch_id, sample_id, split_name, model_name, predicted_label,
                    health_score, explanation, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                prediction_rows,
            )
            connection.executemany(
                """
                INSERT INTO anomaly_scores (
                    run_id, batch_id, sample_id, split_name, model_name, anomaly_score,
                    threshold, rank, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                score_rows,
            )
            connection.commit()

    def insert_evaluation_results(self, run_id: str, batch_id: str, metrics: dict[str, dict[str, object]]) -> None:
        """Persist model metrics and summaries."""
        created_at = utc_now_iso()
        rows = []
        for model_name, metric_row in metrics.items():
            rows.append(
                (
                    run_id,
                    batch_id,
                    model_name,
                    metric_row.get("accuracy"),
                    metric_row.get("precision"),
                    metric_row.get("recall"),
                    metric_row.get("f1"),
                    metric_row.get("roc_auc"),
                    metric_row.get("support"),
                    metric_row.get("flagged_count"),
                    metric_row.get("verdict"),
                    json.dumps(metric_row),
                    created_at,
                )
            )
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO evaluation_results (
                    run_id, batch_id, model_name, accuracy, precision, recall, f1, roc_auc,
                    support, flagged_count, verdict, summary_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            connection.commit()

    def latest_counts(self) -> dict[str, int]:
        """Return quick table counts for debugging."""
        counts: dict[str, int] = {}
        with self.connect() as connection:
            for table_name in (
                "raw_sensor_records",
                "cleaned_sensor_records",
                "model_predictions",
                "anomaly_scores",
                "evaluation_results",
                "feature_statistics",
            ):
                counts[table_name] = int(connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
        return counts


def feature_stats_from_dataframe(dataframe: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Build the feature statistics dataframe expected by the database layer."""
    rows = []
    for feature in feature_columns:
        if feature not in dataframe.columns:
            continue
        series = dataframe[feature]
        rows.append(
            {
                "feature_name": feature,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
            }
        )
    return pd.DataFrame(rows, columns=["feature_name", "mean", "std", "min", "max", "median"])
