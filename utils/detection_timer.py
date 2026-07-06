"""Measure how long the system takes to detect a behavioral mismatch and lock.

This benchmark feeds impostor feature rows one at a time through the trained
inference pipeline with a 2-second interval between checks to simulate the
runtime sliding-window cadence.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Add project root to path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision.inference_engine import InferenceEngine
from decision.threshold_logic import ThresholdLogic


DEFAULT_FEATURES_FILE = Path("data/processed/behavior_features.csv")
SAMPLE_FEATURES_FILE = Path("data/sample/behavior_features_sample.csv")
RESULTS_FILE = Path("results/timing_benchmark.txt")


def _resolve_features_file(features_csv_path: str) -> Path:
    """Resolve the input CSV, falling back to the sample file for testing."""
    path = Path(features_csv_path)
    if path.exists():
        return path

    if path == DEFAULT_FEATURES_FILE and SAMPLE_FEATURES_FILE.exists():
        print(
            "Main features file not found. Using sample data for testing: "
            "data/sample/behavior_features_sample.csv"
        )
        return SAMPLE_FEATURES_FILE

    raise FileNotFoundError(f"Features file not found: {path}")


def _load_impostor_rows(features_csv_path: str, label_to_test: int) -> pd.DataFrame:
    """Load only the rows that match the requested label."""
    resolved_path = _resolve_features_file(features_csv_path)
    data = pd.read_csv(resolved_path)
    label_data = data[data["label"] == label_to_test].copy()
    print(f"Rows found with label={label_to_test}: {len(label_data)}")
    return label_data


def _evaluate_rows_with_timing(
    inference_engine: InferenceEngine,
    threshold_logic: ThresholdLogic,
    rows: pd.DataFrame,
) -> Optional[float]:
    """Feed rows one by one and return the elapsed time until LOCK occurs."""
    if rows.empty:
        return None

    start_time = time.perf_counter()

    # Simulate the initial 2-second sliding-window interval before the first
    # detection decision is available in the real system.
    time.sleep(2.0)

    for index, (_, row) in enumerate(rows.iterrows()):
        feature_frame = pd.DataFrame([row])
        predicted_class, confidence, _ = inference_engine.predict(feature_frame)
        decision = threshold_logic.decide(predicted_class, confidence)

        if decision.get("action") == "lock":
            return time.perf_counter() - start_time

        if index < len(rows) - 1:
            time.sleep(2.0)

    return None


def measure_detection_time(features_csv_path: str, label_to_test: int = 0) -> float:
    """Measure the time taken to detect a mismatch and trigger LOCK.

    Args:
        features_csv_path: Path to the features CSV.
        label_to_test: Label value to evaluate (default: 0 for impostor).

    Returns:
        Detection time in seconds.
    """
    rows = _load_impostor_rows(features_csv_path, label_to_test)
    if rows.empty:
        raise ValueError(
            "No impostor data found - add label=0 rows to your features file"
        )

    inference_engine = InferenceEngine()
    threshold_logic = ThresholdLogic()

    detection_time = _evaluate_rows_with_timing(inference_engine, threshold_logic, rows)
    if detection_time is None:
        raise RuntimeError("No LOCK decision was triggered for the provided rows")

    return float(detection_time)


def run_timing_benchmark(features_csv_path: str) -> None:
    """Run the detection timing benchmark on 10 different impostor samples."""
    resolved_path = _resolve_features_file(features_csv_path)
    data = pd.read_csv(resolved_path)
    impostor_data = data[data["label"] == 0].copy()

    if impostor_data.empty:
        print("No impostor data found - add label=0 rows to your features file")
        return

    sample_count = min(10, len(impostor_data))
    sample_rows = impostor_data.sample(n=sample_count, random_state=42).reset_index(drop=True)

    detection_times: List[float] = []

    for index in range(sample_count):
        row_frame = sample_rows.iloc[[index]].copy()
        temp_path = resolved_path.parent / f"_benchmark_impostor_{index}.csv"
        row_frame.to_csv(temp_path, index=False)
        try:
            detection_time = measure_detection_time(str(temp_path), label_to_test=0)
            detection_times.append(detection_time)
            print(f"Run {index + 1}: {detection_time:.4f} seconds")
        finally:
            if temp_path.exists():
                temp_path.unlink()

    if not detection_times:
        print("No timing results were produced.")
        return

    min_time = min(detection_times)
    max_time = max(detection_times)
    avg_time = sum(detection_times) / len(detection_times)

    summary_line = (
        f"Detection time range: {min_time:.4f} – {max_time:.4f} seconds "
        f"(avg: {avg_time:.4f} seconds)"
    )

    print("\n=== TIMING BENCHMARK SUMMARY ===")
    print(f"Min detection time: {min_time:.4f} seconds")
    print(f"Max detection time: {max_time:.4f} seconds")
    print(f"Average detection time: {avg_time:.4f} seconds")
    print(summary_line)
    print(
        "Note: Detection time includes 2s sliding window interval per sample, "
        "matching real system behavior (check_interval=2.0 in config.yaml)"
    )

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    report_lines = [
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "=== DETECTION TIMING BENCHMARK ===",
    ]
    report_lines.extend(f"Run {index + 1}: {value:.4f} seconds" for index, value in enumerate(detection_times))
    report_lines.extend([
        "",
        f"Min detection time: {min_time:.4f} seconds",
        f"Max detection time: {max_time:.4f} seconds",
        f"Average detection time: {avg_time:.4f} seconds",
        summary_line,
        (
            "Note: Detection time includes 2s sliding window interval per sample, "
            "matching real system behavior (check_interval=2.0 in config.yaml)"
        ),
    ])
    RESULTS_FILE.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Timing benchmark saved to {RESULTS_FILE}")


def main() -> None:
    """CLI entry point for the detection timing benchmark."""
    parser = argparse.ArgumentParser(
        description="Measure how long the system takes to detect a behavioral mismatch and trigger LOCK."
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default=str(SAMPLE_FEATURES_FILE),
        help=(
            "Path to the features CSV. Defaults to the sample behavioral dataset "
            "for quick testing."
        ),
    )

    args = parser.parse_args()
    run_timing_benchmark(args.features_file)


if __name__ == "__main__":
    main()