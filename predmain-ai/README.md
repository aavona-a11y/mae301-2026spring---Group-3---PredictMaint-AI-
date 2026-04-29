# PredMain AI — Predictive Maintenance System

PredMain AI is a machine learning system that analyzes industrial sensor data to detect anomalies and predict machine failures using Isolation Forest, Autoencoder, and LSTM models.

The project is designed as a Phase 3 MVP for a course demo: a maintenance engineer can upload a CSV dataset, run the predictive maintenance pipeline, review model results, inspect high-risk machine readings, and download the generated analysis.

## Project Overview

PredMain AI supports an end-to-end predictive maintenance workflow:

1. Load industrial sensor data from CSV.
2. Clean and preprocess sensor readings.
3. Store raw records, cleaned records, model predictions, anomaly scores, and evaluation metrics in SQLite.
4. Train and compare multiple predictive maintenance models.
5. Generate a readable dashboard and JSON results summary.

## Models Used

- Rule-Based Baseline: simple threshold-based fault detection.
- Isolation Forest: unsupervised anomaly detection for unusual sensor patterns.
- Autoencoder: neural reconstruction model that flags high reconstruction error.
- LSTM: sequence model for learning temporal machine-risk behavior.

## Main Features

- CSV upload through a Streamlit dashboard.
- Local CLI pipeline for ingestion, training, evaluation, and reporting.
- SQLite-backed storage in `outputs/predmainai.db`.
- Model leaderboard with Accuracy, Precision, Recall, F1, ROC-AUC, and verdict labels.
- Risk dashboard with high-risk samples, anomaly scores, and health scores.
- Interactive sample filtering for failures, anomalies, false positives, and false negatives.
- Visual analytics including ROC curve, F1 comparison, anomaly score distributions, and confusion matrix.
- Downloadable `results.json` and processed dataset outputs.

## Project Structure

```text
predmain-ai/
|-- app/
|   `-- app.py
|-- data/
|   |-- raw/
|   `-- processed/
|-- outputs/
|   |-- models/
|   |-- plots/
|   |-- predmainai.db
|   `-- results.json
|-- src/
|   |-- config.py
|   |-- data_loader.py
|   |-- data_ingest.py
|   |-- database.py
|   |-- display_results.py
|   |-- evaluate.py
|   |-- evaluation_pipeline.py
|   |-- main.py
|   |-- models.py
|   |-- preprocess.py
|   |-- results_report.py
|   |-- sample_data.py
|   |-- training.py
|   |-- visualize.py
|   `-- models/
|       |-- autoencoder.py
|       |-- common.py
|       |-- isolation_forest.py
|       |-- lstm_model.py
|       `-- rule_based.py
|-- README.md
`-- requirements.txt
```

## Local Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/aavona-a11y/mae301-2026spring---Group-3---PredictMaint-AI-.git
cd mae301-2026spring---Group-3---PredictMaint-AI-/predmain-ai
pip install -r requirements.txt
```

Optional virtual environment setup:

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Then install requirements:

```bash
pip install -r requirements.txt
```

## Local Run Instructions

Generate the bundled synthetic demo dataset:

```bash
python src/main.py generate-sample
```

Run the full pipeline and save `outputs/results.json`:

```bash
python src/main.py evaluate --csv data/raw/sample_sensor_data.csv
```

Print the terminal report:

```bash
python src/display_results.py
```

Run the Streamlit web app locally:

```bash
streamlit run app/app.py
```

If `streamlit` is not available on your PATH:

```bash
python -m streamlit run app/app.py
```

Open the local dashboard:

```text
http://localhost:8501
```

## Streamlit Dashboard Demo Flow

1. Open the web app with `streamlit run app/app.py`.
2. Upload a CSV file in the sidebar, or use the bundled sample dataset.
3. Choose whether to use pretrained models.
4. Click `Run Analysis`.
5. Review the data preview, pipeline status, model leaderboard, risk dashboard, sample breakdown, visual analytics, feature insights, and downloads.

## Google Colab Instructions

Google Colab is useful when you want to run the project without installing Python locally.

### 1. Clone the Repository

In a Colab notebook cell:

```python
!git clone https://github.com/aavona-a11y/mae301-2026spring---Group-3---PredictMaint-AI-.git
%cd mae301-2026spring---Group-3---PredictMaint-AI-/predmain-ai
```

### 2. Install Dependencies

```python
!pip install -r requirements.txt
```

### 3. Generate Sample Data

```python
!python src/main.py generate-sample
```

### 4. Run the CLI Pipeline

```python
!python src/main.py evaluate --csv data/raw/sample_sensor_data.csv
```

### 5. Display the Terminal Report

```python
!python src/display_results.py
```

## Streamlit + ngrok Public Deployment

Use ngrok when you want to share the Streamlit dashboard from Colab or from a local machine using a public URL.

### Option A: Run Streamlit + ngrok in Google Colab

Install pyngrok:

```python
!pip install pyngrok
```

Add your ngrok authtoken:

```python
from pyngrok import ngrok

ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN")
```

Start Streamlit in the background:

```python
!streamlit run app/app.py --server.port 8501 --server.headless true &
```

Create a public tunnel:

```python
from pyngrok import ngrok

public_url = ngrok.connect(8501)
print(public_url)
```

Open the printed ngrok URL in a browser to access the dashboard.

### Option B: Run Streamlit + ngrok Locally

Terminal 1:

```bash
streamlit run app/app.py --server.port 8501
```

Terminal 2:

```bash
ngrok http 8501
```

Copy the public forwarding URL printed by ngrok and open it in a browser.

## CSV Input Format

PredMain AI works best with CSV files containing columns similar to:

```text
machine_id
timestamp
air_temperature
process_temperature
rotational_speed
torque
tool_wear
vibration
failure_label
```

The pipeline also supports common alternate column names such as:

- `Machine failure`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`
- `Air temperature [K]`
- `Process temperature [K]`

If optional columns are missing, the app explains what is missing and continues when possible. If the dataset is empty, malformed, or missing too many sensor features, the app shows a clear error message.

## Outputs

PredMain AI generates:

- `outputs/results.json`: model metrics, predictions, anomaly rankings, feature insights, and executive summary.
- `outputs/predmainai.db`: SQLite database containing raw rows, cleaned rows, predictions, anomaly scores, evaluation results, and feature statistics.
- `outputs/models/`: saved model artifacts and preprocessing metadata.

## System Architecture

```text
CSV Upload / CSV File
        |
        v
Data Loading and Validation
        |
        v
Preprocessing and Feature Scaling
        |
        v
Model Training and Inference
        |
        v
Evaluation and SQLite Storage
        |
        v
Streamlit Dashboard + Downloadable Results
```

## Limitations

- The models are MVP baselines intended for demonstration, not production maintenance decisions.
- Performance depends heavily on dataset quality and label availability.
- The LSTM is intentionally compact so it can run on CPU in a classroom or Colab setting.
- Feature explanations are based on model scores and sensor deviations, not guaranteed causal diagnoses.

## References

- Streamlit installation and run command: [Streamlit Docs](https://docs.streamlit.io/get-started/installation)
- ngrok with Google Colab and pyngrok: [ngrok Colab Docs](https://ngrok.com/docs/using-ngrok-with/googleColab/)
