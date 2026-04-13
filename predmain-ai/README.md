# PredMainAI

PredMainAI is a database-backed predictive maintenance MVP for class demos. It ingests industrial sensor CSVs, stores raw and cleaned records in SQLite, runs four baseline models, evaluates them, and prints a readable terminal summary.

## What the MVP does

- Ingests a CSV of machine sensor readings.
- Normalizes column names and handles missing optional columns gracefully.
- Stores raw rows, cleaned rows, feature statistics, predictions, anomaly scores, and evaluation metrics in SQLite.
- Trains four models:
  - Rule-based baseline
  - Isolation Forest
  - Autoencoder
  - LSTM risk model
- Writes evaluation output to `outputs/results.json`.
- Prints a polished demo report with leaderboard, risky rows, feature fault summary, and false-positive / false-negative analysis.
- Generates a deterministic synthetic CSV if you do not already have one.

## Project structure

```text
predmain-ai/
|-- data/
|   |-- raw/
|   |-- processed/
|-- outputs/
|   |-- models/
|   |-- results.json
|-- outputs/predmainai.db
|-- requirements.txt
|-- README.md
`-- src/
    |-- config.py
    |-- database.py
    |-- data_ingest.py
    |-- preprocess.py
    |-- training.py
    |-- evaluation_pipeline.py
    |-- results_report.py
    |-- main.py
    |-- evaluate.py
    |-- display_results.py
    `-- models/
        |-- rule_based.py
        |-- isolation_forest.py
        |-- autoencoder.py
        `-- lstm_model.py
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

Generate a demo CSV if you need one:

```bash
python src/main.py generate-sample
```

Ingest only:

```bash
python src/main.py ingest --csv data/raw/sample_sensor_data.csv
```

Train only:

```bash
python src/main.py train --csv data/raw/sample_sensor_data.csv
```

Run the full MVP pipeline:

```bash
python src/main.py evaluate --csv data/raw/sample_sensor_data.csv
```

Print the saved report:

```bash
python src/display_results.py
```

Or run everything and print the report in one step:

```bash
python src/main.py run --csv data/raw/sample_sensor_data.csv
```

## Input schema

PredMainAI looks for these canonical columns when available:

- `machine_id`
- `timestamp`
- `air_temperature`
- `process_temperature`
- `rotational_speed`
- `torque`
- `tool_wear`
- `vibration`
- `failure_label`

Flexible alias handling is built in. For example, columns like `Machine failure`, `Rotational speed [rpm]`, or `Air temperature [K]` are normalized automatically.

Graceful handling rules:

- If `machine_id` is missing, the pipeline fills a default machine id.
- If `timestamp` is missing, the pipeline generates a simple synthetic timeline.
- If some sensor columns are missing, the pipeline uses the available ones and reports what is missing.
- If fewer than two recognized sensor columns exist, the pipeline stops with a clear error.
- If `failure_label` is missing, the pipeline can still ingest the data, but the supervised LSTM path may be skipped or less informative.

## Database schema

The local SQLite database is `outputs/predmainai.db` and includes these tables:

- `raw_sensor_records`
- `cleaned_sensor_records`
- `model_predictions`
- `anomaly_scores`
- `evaluation_results`
- `feature_statistics`

What gets stored:

- Raw ingested rows and source file information
- Cleaned rows plus scaled feature payloads
- Training-set feature statistics
- Per-model prediction rows and anomaly-score rankings
- Final evaluation metrics and verdicts

## Pipeline flow

1. `data_ingest.py` loads and normalizes the CSV, then writes raw rows to SQLite.
2. `preprocess.py` creates train/validation/test splits, imputes missing values, scales features, stores cleaned rows, and saves feature statistics.
3. `training.py` trains the four MVP models and saves artifacts in `outputs/models/`.
4. `evaluation_pipeline.py` scores the test set, writes metrics to SQLite, and saves `outputs/results.json`.
5. `results_report.py` and `display_results.py` print the six-section terminal summary.

## Results file

`outputs/results.json` contains:

- model metrics
- leaderboard ordering
- per-sample predictions and anomaly scores
- feature importance counts
- false positive / false negative deep dive rows
- one-line executive summary

## Notes for the class demo

- The implementation is intentionally compact and readable over highly optimized.
- The synthetic dataset is deterministic, so demo runs are reproducible.
- The current MVP is CPU-friendly and designed to be expanded later.
- A web UI is not included in this pass to keep the MVP focused and easy to run locally.
