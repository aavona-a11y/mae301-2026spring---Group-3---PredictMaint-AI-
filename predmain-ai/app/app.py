"""Streamlit dashboard for PredMain AI."""

from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import data_loader
import evaluate
import models
import visualize
from config import CANONICAL_SENSOR_COLUMNS, DB_PATH, DEFAULT_SAMPLE_CSV
from data_ingest import normalize_columns
from database import DatabaseManager
from sample_data import generate_synthetic_dataset


st.set_page_config(
    page_title="PredMain AI",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)


RISK_COLORS = {
    "HIGH RISK": "#fee2e2",
    "MONITOR": "#fef3c7",
    "NORMAL": "#dcfce7",
}


def apply_page_styles() -> None:
    """Apply compact product-dashboard styling."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
        }
        .risk-high {
            background: #fee2e2;
            border-left: 4px solid #dc2626;
            padding: 0.65rem 0.8rem;
            border-radius: 6px;
        }
        .risk-monitor {
            background: #fef3c7;
            border-left: 4px solid #d97706;
            padding: 0.65rem 0.8rem;
            border-radius: 6px;
        }
        .risk-normal {
            background: #dcfce7;
            border-left: 4px solid #16a34a;
            padding: 0.65rem 0.8rem;
            border-radius: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_uploaded_data(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """Load uploaded CSV bytes into a dataframe."""
    return data_loader.load_data(BytesIO(file_bytes))


@st.cache_resource(show_spinner=False)
def run_cached_analysis(file_bytes: bytes, file_name: str, use_pretrained_models: bool) -> dict[str, Any]:
    """Run the full pipeline with Streamlit caching for fast reruns."""
    dataframe = data_loader.load_data(BytesIO(file_bytes))
    database = DatabaseManager(DB_PATH)
    preprocessed = data_loader.preprocess_data(dataframe, database=database, source_name=file_name)
    training = models.train_models(preprocessed)
    models.run_inference(training)
    results = evaluate.evaluate_models(training, database=database, source_file=file_name)
    results["warnings"] = preprocessed["warnings"]
    results["missing_feature_columns"] = preprocessed["missing_feature_columns"]
    results["processed_dataset"] = training.test_frame.to_csv(index=False)
    results["used_pretrained_models"] = use_pretrained_models
    return results


def sample_file_bytes() -> tuple[bytes, str]:
    """Return bytes for the bundled synthetic sample CSV."""
    sample_path = generate_synthetic_dataset(DEFAULT_SAMPLE_CSV)
    return sample_path.read_bytes(), sample_path.name


def metric_dataframe(results: dict[str, Any]) -> pd.DataFrame:
    """Build a leaderboard dataframe for display."""
    rows = []
    for row in results["leaderboard"]:
        rows.append(
            {
                "Model": row["display_name"],
                "Accuracy": row["accuracy"],
                "Precision": row["precision"],
                "Recall": row["recall"],
                "F1": row["f1"],
                "ROC-AUC": row["roc_auc"],
                "Verdict": row["verdict"],
            }
        )
    return pd.DataFrame(rows)


def sample_dataframe(results: dict[str, Any]) -> pd.DataFrame:
    """Flatten per-sample JSON into a dataframe."""
    rows = []
    for row in results["per_sample"]:
        flat = {
            "sample_id": row["sample_id"],
            "machine_id": row["machine_id"],
            "timestamp": row["timestamp"],
            "true_label": row["true_label"],
            "health_score": row["health_score"],
            "health_label": row["health_label"],
            "explanation": row["explanation"],
        }
        for model_name, prediction in row["predictions"].items():
            flat[f"{model_name}_prediction"] = prediction
            flat[f"{model_name}_score"] = row["scores"][model_name]
            flat[f"{model_name}_flag"] = row["prediction_flags"][model_name]
        rows.append(flat)
    return pd.DataFrame(rows)


def risk_label(score: float) -> str:
    """Convert anomaly score to dashboard risk label."""
    if score >= 0.70:
        return "🔴 HIGH RISK"
    if score >= 0.40:
        return "🟡 MONITOR"
    return "🟢 NORMAL"


def style_risk_rows(dataframe: pd.DataFrame):
    """Color high-risk rows in Streamlit dataframes."""
    def row_style(row: pd.Series) -> list[str]:
        label = row.get("Risk", "")
        if "HIGH" in label:
            return [f"background-color: {RISK_COLORS['HIGH RISK']}"] * len(row)
        if "MONITOR" in label:
            return [f"background-color: {RISK_COLORS['MONITOR']}"] * len(row)
        return [f"background-color: {RISK_COLORS['NORMAL']}"] * len(row)

    return dataframe.style.apply(row_style, axis=1)


def show_dataset_preview(dataframe: pd.DataFrame) -> None:
    """Render dataset preview and basic stats."""
    st.header("1. Data Preview")
    normalized_frame, _ = normalize_columns(dataframe)
    rows, features = dataframe.shape
    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows", f"{rows:,}")
    metric_cols[1].metric("Features", f"{features:,}")
    metric_cols[2].metric("Machines", normalized_frame["machine_id"].nunique() if "machine_id" in normalized_frame.columns else "Unknown")
    metric_cols[3].metric("Failure Labels", "Yes" if "failure_label" in normalized_frame.columns else "No")

    left, right = st.columns([1.4, 1])
    with left:
        st.dataframe(dataframe.head(25), use_container_width=True, hide_index=True)
    with right:
        st.pyplot(visualize.plot_class_distribution(normalized_frame), use_container_width=True)


def show_pipeline_status(results: dict[str, Any]) -> None:
    """Render pipeline completion state."""
    st.header("2. Pipeline Status")
    status_cols = st.columns(4)
    status_cols[0].success("✅ Loading complete")
    status_cols[1].success("✅ Preprocessing complete")
    status_cols[2].success("✅ Models complete")
    status_cols[3].success("✅ Results saved")

    if results.get("missing_feature_columns"):
        missing = ", ".join(results["missing_feature_columns"])
        st.warning(f"⚠️ Missing sensor columns: {missing}")
    for warning in results.get("warnings", []):
        st.warning(f"⚠️ {warning}")


def show_model_leaderboard(results: dict[str, Any]) -> None:
    """Render leaderboard table and top model metrics."""
    st.header("3. Model Leaderboard")
    leaderboard = metric_dataframe(results)
    st.dataframe(
        leaderboard.style.format(
            {
                "Accuracy": "{:.3f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1": "{:.3f}",
                "ROC-AUC": lambda value: "n/a" if pd.isna(value) else f"{value:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def show_risk_dashboard(results: dict[str, Any]) -> None:
    """Render high-risk samples and health summary."""
    st.header("4. Risk Dashboard")
    best_model = results["best_model"]
    ranking = pd.DataFrame(results["anomaly_score_ranking"]).head(10)
    ranking["Risk"] = ranking["score"].map(risk_label)
    ranking = ranking[["sample_id", "machine_id", "score", "Risk", "true_label", "predicted_label", "health_score", "explanation"]]

    high_risk_count = int((ranking["Risk"].str.contains("HIGH")).sum())
    avg_health = float(pd.DataFrame(results["per_sample"])["health_score"].mean())
    cols = st.columns(3)
    cols[0].metric("Best Model", best_model.replace("_", " ").title())
    cols[1].metric("Top-10 High Risk", high_risk_count)
    cols[2].metric("Average Health", f"{avg_health:.1f}/100")
    st.dataframe(style_risk_rows(ranking), use_container_width=True, hide_index=True)


def filter_samples(frame: pd.DataFrame, best_model: str, filter_mode: str) -> pd.DataFrame:
    """Apply the selected sample filter."""
    if filter_mode == "Failures only":
        return frame[frame["true_label"] == "FAILURE"]
    if filter_mode == "False positives":
        return frame[(frame["true_label"] == "NORMAL") & (frame[f"{best_model}_flag"] == 1)]
    if filter_mode == "False negatives":
        return frame[(frame["true_label"] == "FAILURE") & (frame[f"{best_model}_flag"] == 0)]
    if filter_mode == "Anomalies only":
        return frame[frame[f"{best_model}_flag"] == 1]
    return frame


def show_sample_breakdown(results: dict[str, Any], show_explanation: bool) -> None:
    """Render interactive sample-level predictions."""
    st.header("5. Sample Breakdown")
    best_model = results["best_model"]
    frame = sample_dataframe(results)
    filter_mode = st.segmented_control(
        "Filter samples",
        ["All samples", "Failures only", "Anomalies only", "False positives", "False negatives"],
        default="All samples",
    )
    filtered = filter_samples(frame, best_model, filter_mode)
    display_columns = [
        "sample_id",
        "machine_id",
        "true_label",
        "health_score",
        f"{best_model}_prediction",
        f"{best_model}_score",
        "rule_based_prediction",
        "isolation_forest_prediction",
        "autoencoder_prediction",
        "lstm_prediction",
    ]
    if show_explanation:
        display_columns.append("explanation")
    st.dataframe(filtered[display_columns], use_container_width=True, hide_index=True)


def show_visual_analytics(results: dict[str, Any]) -> None:
    """Render model diagnostic plots."""
    st.header("6. Visual Analytics")
    tabs = st.tabs(["ROC Curve", "F1 Comparison", "Score Distributions", "Confusion Matrix"])
    with tabs[0]:
        st.pyplot(visualize.plot_roc_curve(results), use_container_width=True)
    with tabs[1]:
        st.pyplot(visualize.plot_f1_comparison(results), use_container_width=True)
    with tabs[2]:
        st.pyplot(visualize.plot_anomaly_score_distribution(results), use_container_width=True)
    with tabs[3]:
        st.pyplot(visualize.plot_confusion_matrix(results), use_container_width=True)


def show_feature_insights(results: dict[str, Any]) -> None:
    """Render feature importance and plain-English interpretation."""
    st.header("7. Feature Insights")
    importance = pd.DataFrame(
        [{"Feature": feature, "Mentions": count} for feature, count in results["feature_importance"].items()]
    )
    if importance.empty:
        st.info("No dominant feature pattern was detected in this run.")
        return
    st.bar_chart(importance.set_index("Feature"))
    top_features = importance["Feature"].head(2).tolist()
    if len(top_features) >= 2:
        st.info(f"{top_features[0].replace('_', ' ').title()} and {top_features[1].replace('_', ' ').title()} are primary failure drivers.")
    else:
        st.info(f"{top_features[0].replace('_', ' ').title()} is the primary failure driver.")


def show_downloads(results: dict[str, Any]) -> None:
    """Render results download controls."""
    st.header("8. Downloads")
    col1, col2 = st.columns(2)
    col1.download_button(
        "Download results.json",
        data=json.dumps(results, indent=2),
        file_name="predmainai_results.json",
        mime="application/json",
        use_container_width=True,
    )
    col2.download_button(
        "Download processed dataset",
        data=results["processed_dataset"],
        file_name="predmainai_processed_test.csv",
        mime="text/csv",
        use_container_width=True,
    )


def validate_expected_columns(dataframe: pd.DataFrame) -> None:
    """Warn about absent canonical sensors while allowing graceful analysis."""
    normalized_frame, _ = normalize_columns(dataframe)
    normalized_columns = set(normalized_frame.columns)
    missing = [column for column in CANONICAL_SENSOR_COLUMNS if column not in normalized_columns]
    if "torque" in missing:
        st.warning("⚠️ Missing required column: torque")
    elif missing:
        st.warning(f"⚠️ Missing sensor columns: {', '.join(missing)}")


def main() -> None:
    """Run the Streamlit application."""
    apply_page_styles()
    st.title("PredMain AI — Predictive Maintenance Dashboard")

    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        use_pretrained_models = st.checkbox("Use pretrained models", value=True)
        show_explanation = st.checkbox("Show explanation", value=True)
        run_analysis = st.button("Run Analysis", type="primary", use_container_width=True)

    if uploaded_file is None:
        file_bytes, file_name = sample_file_bytes()
        st.info("Using bundled sample data until a CSV is uploaded.")
    else:
        file_bytes, file_name = uploaded_file.getvalue(), uploaded_file.name

    try:
        dataframe = load_uploaded_data(file_bytes, file_name)
    except ValueError as exc:
        st.error(str(exc))
        return

    validate_expected_columns(dataframe)
    show_dataset_preview(dataframe)

    if not run_analysis:
        st.stop()

    try:
        with st.spinner("Loading… Preprocessing… Running models…"):
            results = run_cached_analysis(file_bytes, file_name, use_pretrained_models)
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        return

    show_pipeline_status(results)
    show_model_leaderboard(results)
    show_risk_dashboard(results)
    show_sample_breakdown(results, show_explanation=show_explanation)
    show_visual_analytics(results)
    show_feature_insights(results)

    with st.expander("Detailed JSON output"):
        st.json(results)

    show_downloads(results)


if __name__ == "__main__":
    main()
