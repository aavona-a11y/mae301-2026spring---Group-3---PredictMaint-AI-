"""CLI entrypoint for the PredMainAI MVP."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import DB_PATH, DEFAULT_SAMPLE_CSV, RESULTS_PATH, configure_logging, ensure_directories
from data_ingest import ingest_csv
from database import DatabaseManager
from evaluation_pipeline import run_full_evaluation
from preprocess import preprocess_dataset
from results_report import print_results_report
from sample_data import generate_synthetic_dataset
from training import train_models


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description="PredMainAI predictive maintenance MVP")
    parser.add_argument("--db", default=str(DB_PATH), help="Path to the SQLite database")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a CSV into SQLite")
    ingest_parser.add_argument("--csv", default=str(DEFAULT_SAMPLE_CSV), help="Input CSV path")

    train_parser = subparsers.add_parser("train", help="Ingest, preprocess, and train models")
    train_parser.add_argument("--csv", default=str(DEFAULT_SAMPLE_CSV), help="Input CSV path")

    evaluate_parser = subparsers.add_parser("evaluate", help="Run the full pipeline and save results.json")
    evaluate_parser.add_argument("--csv", default=str(DEFAULT_SAMPLE_CSV), help="Input CSV path")

    run_parser = subparsers.add_parser("run", help="Run evaluate and print the report")
    run_parser.add_argument("--csv", default=str(DEFAULT_SAMPLE_CSV), help="Input CSV path")

    report_parser = subparsers.add_parser("report", help="Print the saved report")
    report_parser.add_argument("--results", default=str(RESULTS_PATH), help="Path to results.json")

    sample_parser = subparsers.add_parser("generate-sample", help="Generate a synthetic CSV for demos")
    sample_parser.add_argument("--csv", default=str(DEFAULT_SAMPLE_CSV), help="Where to write the sample CSV")
    return parser


def main() -> None:
    """Execute the requested CLI command."""
    ensure_directories()
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    database = DatabaseManager(Path(args.db))

    if args.command == "generate-sample":
        output_path = generate_synthetic_dataset(Path(args.csv))
        print(f"Synthetic dataset created at {output_path}")
        return

    if args.command == "ingest":
        ingested = ingest_csv(args.csv, database=database)
        print(f"Ingested {len(ingested.dataframe)} rows from {ingested.source_path} into {database.db_path}")
        print(f"Available features: {', '.join(ingested.available_feature_columns)}")
        if ingested.missing_feature_columns:
            print(f"Missing features: {', '.join(ingested.missing_feature_columns)}")
        return

    if args.command == "train":
        ingested = ingest_csv(args.csv, database=database)
        prepared = preprocess_dataset(ingested, database=database)
        training = train_models(prepared)
        print(f"Training complete for run {training.run_id}")
        for model_name, output in training.model_outputs.items():
            print(f"{model_name}: {output.status} -> {output.model_path or 'not saved'}")
        return

    if args.command == "evaluate":
        results = run_full_evaluation(csv_path=args.csv, database_path=args.db)
        print(f"Evaluation complete. Results saved to {RESULTS_PATH}")
        print(results["executive_summary"])
        return

    if args.command == "run":
        run_full_evaluation(csv_path=args.csv, database_path=args.db)
        print_results_report(RESULTS_PATH)
        return

    if args.command == "report":
        print_results_report(args.results)


if __name__ == "__main__":
    main()
