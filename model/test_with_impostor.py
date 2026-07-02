"""Formal evaluation of the trained behavioral authentication model.

This script evaluates owner recognition and impostor detection separately,
using the trained Isolation Forest model and the saved feature columns.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

import sys

# Add project root to path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.train_model import BehaviorModelTrainer


DEFAULT_FEATURES_FILE = Path("data/processed/behavior_features.csv")
SAMPLE_FEATURES_FILE = Path("data/sample/behavior_features_sample.csv")
MODEL_FILE = "model/behavior_model.pkl"
SCALER_FILE = "model/feature_scaler.pkl"
FEATURE_COLUMNS_FILE = Path("model/behavior_model_features.txt")
REPORT_FILE = Path("results/evaluation_report.txt")


def resolve_features_file(features_file: str) -> Path:
    """Resolve the features file with a sample fallback for quick testing."""
    path = Path(features_file)
    if path.exists():
        return path

    if path == DEFAULT_FEATURES_FILE and SAMPLE_FEATURES_FILE.exists():
        print(
            "Main features file not found. Using sample data for testing: "
            "data/sample/behavior_features_sample.csv"
        )
        return SAMPLE_FEATURES_FILE

    raise FileNotFoundError(f"Features file not found: {path}")


def load_feature_columns() -> list[str]:
    """Load saved feature column order from the trained model artifact."""
    if not FEATURE_COLUMNS_FILE.exists():
        raise FileNotFoundError(
            f"Feature columns file not found: {FEATURE_COLUMNS_FILE}"
        )

    with FEATURE_COLUMNS_FILE.open("r", encoding="utf-8") as file_handle:
        feature_columns = [line.strip() for line in file_handle if line.strip()]

    if not feature_columns:
        raise ValueError(f"No feature columns found in {FEATURE_COLUMNS_FILE}")

    return feature_columns


def load_dataset(features_file: Path) -> pd.DataFrame:
    """Load the evaluation dataset from CSV."""
    return pd.read_csv(features_file)


def prepare_feature_frame(data: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Select, order, and fill feature columns for prediction."""
    missing_columns = [column for column in feature_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required feature columns in data: {missing_columns}"
        )

    return data[feature_columns].fillna(0)


def evaluate_subset(
    trainer: BehaviorModelTrainer,
    subset: pd.DataFrame,
    feature_columns: list[str],
) -> Tuple[np.ndarray, int]:
    """Predict one subset and return predictions plus the total sample count."""
    if subset.empty:
        return np.array([], dtype=int), 0

    feature_frame = prepare_feature_frame(subset, feature_columns)
    scaled_features = trainer.scaler.transform(feature_frame)
    predictions = trainer.model.predict(scaled_features)
    return np.asarray(predictions), len(subset)


def build_report(
    owner_total: int,
    owner_correct: int,
    impostor_total: int,
    impostor_correct: int,
) -> Tuple[str, float, float, float, bool]:
    """Build the formatted report and compute summary rates."""
    owner_recognition_rate = (owner_correct / owner_total * 100.0) if owner_total else 0.0
    frr = 100.0 - owner_recognition_rate if owner_total else 0.0

    impostor_detection_rate = (impostor_correct / impostor_total * 100.0) if impostor_total else 0.0
    far = 100.0 - impostor_detection_rate if impostor_total else 0.0

    combined_total = owner_total + impostor_total
    combined_correct = owner_correct + impostor_correct
    combined_accuracy = (combined_correct / combined_total * 100.0) if combined_total else 0.0
    suitable_for_deployment = far < 10.0 and frr < 15.0

    lines = [
        "=== OWNER RECOGNITION TEST ===",
        f"Total owner samples tested: {owner_total}",
        f"Correctly recognized as owner: {owner_correct}",
        f"Owner Recognition Rate: {owner_recognition_rate:.2f}% (this is 1 - FRR)",
        f"False Rejection Rate (FRR): {frr:.2f}% (owner wrongly locked out)",
        "",
        "=== IMPOSTOR DETECTION TEST ===",
        f"Total impostor samples tested: {impostor_total}",
        f"Correctly detected as impostor (locked): {impostor_correct}",
        f"Impostor Detection Rate: {impostor_detection_rate:.2f}% (this is 1 - FAR)",
        f"False Acceptance Rate (FAR): {far:.2f}% (impostor wrongly allowed through)",
        "",
        "=== OVERALL SUMMARY ===",
        f"Combined Accuracy (weighted): {combined_accuracy:.2f}%",
        f"Model is suitable for deployment: {'YES' if suitable_for_deployment else 'NO'} (YES if FAR < 10% and FRR < 15%)",
    ]

    report_text = "\n".join(lines)
    return report_text, owner_recognition_rate, frr, impostor_detection_rate, suitable_for_deployment


def run_evaluation(features_file: str) -> str:
    """Run the full owner/impostor evaluation and persist the report."""
    resolved_features_file = resolve_features_file(features_file)
    data = load_dataset(resolved_features_file)

    owner_data = data[data["label"] == 1].copy()
    impostor_data = data[data["label"] == 0].copy()

    print(f"Owner rows found: {len(owner_data)}")
    print(f"Impostor rows found: {len(impostor_data)}")

    if impostor_data.empty:
        message = "No impostor data found - add label=0 rows to your features file"
        print(message)
        return message

    trainer = BehaviorModelTrainer(
        features_file=str(resolved_features_file),
        model_file=MODEL_FILE,
        scaler_file=SCALER_FILE,
    )
    trainer.load_model()

    feature_columns = load_feature_columns()
    trainer.feature_columns = feature_columns

    owner_predictions, owner_total = evaluate_subset(trainer, owner_data, feature_columns)
    impostor_predictions, impostor_total = evaluate_subset(trainer, impostor_data, feature_columns)

    owner_correct = int(np.sum(owner_predictions == 1))
    impostor_correct = int(np.sum(impostor_predictions == 0))

    report_body, _, _, _, _ = build_report(
        owner_total=owner_total,
        owner_correct=owner_correct,
        impostor_total=impostor_total,
        impostor_correct=impostor_correct,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    final_report = f"Timestamp: {timestamp}\n\n{report_body}\n"

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(final_report, encoding="utf-8")

    print(final_report)
    print(f"Evaluation report saved to {REPORT_FILE}")

    return final_report


def main() -> None:
    """CLI entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate the trained behavioral authentication model against owner and impostor data."
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default=str(DEFAULT_FEATURES_FILE),
        help="Path to the features CSV (falls back to sample data if the default file is missing).",
    )

    args = parser.parse_args()
    run_evaluation(args.features_file)


if __name__ == "__main__":
    main()