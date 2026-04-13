"""Synthetic data generation for demo-friendly PredMainAI runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_SAMPLE_CSV, RANDOM_SEED, ensure_directories


def generate_synthetic_dataset(output_path: Path = DEFAULT_SAMPLE_CSV, rows_per_machine: int = 160) -> Path:
    """Create a deterministic sensor CSV so the MVP always has runnable input."""
    ensure_directories()
    rng = np.random.default_rng(RANDOM_SEED)
    machine_ids = ["M-100", "M-200", "M-300"]
    start_time = pd.Timestamp("2026-01-01 08:00:00")
    rows: list[dict[str, object]] = []

    for machine_index, machine_id in enumerate(machine_ids):
        base_air = 295.0 + machine_index * 1.2
        base_process = 305.5 + machine_index * 1.1
        base_speed = 1450 + machine_index * 45
        base_torque = 38 + machine_index * 3
        base_wear = 10.0
        base_vibration = 0.30 + machine_index * 0.05

        for step in range(rows_per_machine):
            timestamp = start_time + pd.Timedelta(minutes=5 * (machine_index * rows_per_machine + step))
            wear = base_wear + step * 0.5 + rng.normal(0, 1.0)
            seasonal = np.sin(step / 9.0)
            anomaly_boost = 0.0
            if 55 <= step <= 70 or 120 <= step <= 132:
                anomaly_boost += 1.0 + (step % 4) * 0.2
            if machine_id == "M-300" and 95 <= step <= 108:
                anomaly_boost += 1.5

            air_temperature = base_air + 0.6 * seasonal + rng.normal(0, 0.55) + anomaly_boost * 0.7
            process_temperature = base_process + 0.9 * seasonal + rng.normal(0, 0.6) + anomaly_boost * 1.0
            rotational_speed = base_speed + rng.normal(0, 35) - anomaly_boost * 50 + seasonal * 12
            torque = base_torque + rng.normal(0, 2.2) + anomaly_boost * 6.0 + wear * 0.04
            vibration = base_vibration + rng.normal(0, 0.03) + anomaly_boost * 0.18 + wear * 0.0018

            failure_risk = 0
            if torque > base_torque + 9:
                failure_risk += 1
            if vibration > base_vibration + 0.30:
                failure_risk += 1
            if wear > 70:
                failure_risk += 1
            if rotational_speed < base_speed - 75:
                failure_risk += 1
            failure_label = int(failure_risk >= 2 or anomaly_boost >= 1.6)

            rows.append(
                {
                    "machine_id": machine_id,
                    "timestamp": timestamp.isoformat(),
                    "air_temperature": round(float(air_temperature), 3),
                    "process_temperature": round(float(process_temperature), 3),
                    "rotational_speed": round(float(rotational_speed), 3),
                    "torque": round(float(torque), 3),
                    "tool_wear": round(float(max(wear, 0)), 3),
                    "vibration": round(float(max(vibration, 0.02)), 4),
                    "failure_label": failure_label,
                }
            )

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output_path, index=False)
    return output_path
