"""Terminal reporting for PredMainAI results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import RESULTS_PATH


DISPLAY_NAMES = {
    "rule_based": "Rule-Based",
    "isolation_forest": "Isolation Forest",
    "autoencoder": "Autoencoder",
    "lstm": "LSTM",
}


def load_results(results_path: str | Path = RESULTS_PATH) -> dict[str, Any]:
    """Load the saved JSON payload."""
    resolved = Path(results_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Results file not found: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def print_section(title: str) -> None:
    """Print a deterministic section header."""
    print(title)
    print("-" * len(title))


def metric_cell(value: float | None) -> str:
    """Format metric values consistently."""
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def print_model_leaderboard(results: dict[str, Any]) -> None:
    print_section("1. Model leaderboard")
    print(f"{'Rank':<5} {'Model':<18} {'F1':<8} {'Recall':<8} {'ROC-AUC':<8} {'Verdict'}")
    for rank, row in enumerate(results["leaderboard"], start=1):
        print(
            f"{rank:<5} {row['display_name']:<18} {metric_cell(row['f1']):<8} "
            f"{metric_cell(row['recall']):<8} {metric_cell(row['roc_auc']):<8} {row['verdict']}"
        )
    print()


def print_fault_detection_summary(results: dict[str, Any]) -> None:
    print_section("2. Fault detection summary")
    model_order = ["rule_based", "isolation_forest", "autoencoder", "lstm"]
    print(f"{'Sample':<7} {'True':<8} {'Rule':<10} {'IF':<10} {'AE':<10} {'LSTM':<10} {'Health':<8}")
    for row in results["per_sample"]:
        print(
            f"{row['sample_id']:<7} {row['true_label']:<8} "
            f"{row['predictions']['rule_based']:<10} "
            f"{row['predictions']['isolation_forest']:<10} "
            f"{row['predictions']['autoencoder']:<10} "
            f"{row['predictions']['lstm']:<10} "
            f"{row['health_score']:<8.1f}"
        )
    print()
    for model_name in model_order:
        predicted_failures = sum(item["prediction_flags"][model_name] for item in results["per_sample"])
        true_failures = sum(1 for item in results["per_sample"] if item["true_label"] == "FAILURE")
        caught = sum(
            1
            for item in results["per_sample"]
            if item["true_label"] == "FAILURE" and item["prediction_flags"][model_name] == 1
        )
        print(f"{DISPLAY_NAMES[model_name]} caught {caught} of {true_failures} true failures and flagged {predicted_failures} rows overall.")
    print()


def print_anomaly_score_ranking(results: dict[str, Any], limit: int = 10) -> None:
    print_section("3. Anomaly score ranking")
    best_model = results["best_model"]
    print(f"Best model: {DISPLAY_NAMES[best_model]}")
    print(f"{'Rank':<5} {'Sample':<7} {'Machine':<8} {'Score':<8} {'Pred':<8} {'Actual':<8} {'Health'}")
    for rank, row in enumerate(results["anomaly_score_ranking"][:limit], start=1):
        print(
            f"{rank:<5} {row['sample_id']:<7} {row['machine_id']:<8} {row['score']:<8.3f} "
            f"{row['predicted_label']:<8} {row['true_label']:<8} {row['health_score']:.1f}"
        )
    print()


def print_feature_fault_report(results: dict[str, Any]) -> None:
    print_section("4. Feature fault report")
    if not results["feature_importance"]:
        print("No dominant risky features were identified in the current run.")
        print()
        return
    for rank, (feature, count) in enumerate(results["feature_importance"].items(), start=1):
        verdict = "Primary driver" if rank == 1 else "Secondary driver" if rank == 2 else "Monitor"
        print(f"{rank:<2}. {feature:<20} {count:<4} mentions   {verdict}")
    print()


def print_error_deep_dive(results: dict[str, Any]) -> None:
    print_section("5. False positive / false negative deep dive")
    print("False positives:")
    if results["false_positive_deep_dive"]:
        for row in results["false_positive_deep_dive"]:
            print(
                f"Sample {row['sample_id']} | score {row['scores'][results['best_model']]:.3f} | "
                f"{', '.join(row['top_features'][:3])} | {row['explanation']}"
            )
    else:
        print("None")
    print()
    print("False negatives:")
    if results["false_negative_deep_dive"]:
        for row in results["false_negative_deep_dive"]:
            print(
                f"Sample {row['sample_id']} | score {row['scores'][results['best_model']]:.3f} | "
                f"{', '.join(row['top_features'][:3])} | {row['explanation']}"
            )
    else:
        print("None")
    print()


def print_executive_summary(results: dict[str, Any]) -> None:
    print_section("6. Executive summary")
    print(results["executive_summary"])


def print_results_report(results_path: str | Path = RESULTS_PATH) -> None:
    """Print the six required report sections in order."""
    results = load_results(results_path)
    print_model_leaderboard(results)
    print_fault_detection_summary(results)
    print_anomaly_score_ranking(results)
    print_feature_fault_report(results)
    print_error_deep_dive(results)
    print_executive_summary(results)
