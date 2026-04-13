"""Compatibility wrapper for the PredMainAI evaluation pipeline."""

from __future__ import annotations

from evaluation_pipeline import run_full_evaluation


if __name__ == "__main__":
    results = run_full_evaluation()
    print(f"Evaluation complete. Results saved to outputs/results.json")
    print(results["executive_summary"])
