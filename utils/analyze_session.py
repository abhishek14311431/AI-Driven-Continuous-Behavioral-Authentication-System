"""Analyze owner vs impostor behavioral feature distributions.

This script compares the feature means for label==1 (owner) and label==0
(impostor) rows in a features CSV. It no longer relies on timestamp-based
heuristics, which are fragile and hard to interpret.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd


DEFAULT_FEATURES_FILE = Path("data/processed/behavior_features.csv")
SAMPLE_FEATURES_FILE = Path("data/sample/behavior_features_sample.csv")
REPORT_FILE = Path("results/behavior_comparison.txt")
METADATA_COLUMNS = {"timestamp", "label", "window_size"}


def resolve_features_file(features_file: str) -> Path:
    """Resolve the features file, falling back to the sample data when needed."""
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


def load_behavior_data(features_file: str) -> pd.DataFrame:
    """Load the behavioral features dataset."""
    resolved_path = resolve_features_file(features_file)
    return pd.read_csv(resolved_path)


def split_by_label(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset strictly by the label column."""
    if "label" not in df.columns:
        raise ValueError("The features file must contain a 'label' column.")

    owner_data = df[df["label"] == 1].copy()
    impostor_data = df[df["label"] == 0].copy()
    return owner_data, impostor_data


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return all behavioral feature columns excluding metadata columns."""
    return [column for column in df.columns if column not in METADATA_COLUMNS]


def compute_comparison_table(
    owner_data: pd.DataFrame,
    impostor_data: pd.DataFrame,
    feature_columns: List[str],
) -> Tuple[pd.DataFrame, int]:
    """Compute feature-wise mean comparisons and difference percentages."""
    rows = []
    strong_discriminator_count = 0

    for feature in feature_columns:
        owner_avg = owner_data[feature].mean() if not owner_data.empty else 0.0
        impostor_avg = impostor_data[feature].mean() if not impostor_data.empty else float("nan")

        if pd.isna(impostor_avg):
            difference_text = "N/A"
            diff_ratio = None
        elif owner_avg == 0:
            if impostor_avg == 0:
                difference_text = "+0.0%"
                diff_ratio = 0.0
            else:
                difference_text = "+∞%"
                diff_ratio = float("inf")
        else:
            diff_ratio = ((impostor_avg - owner_avg) / abs(owner_avg)) * 100.0
            difference_text = f"{diff_ratio:+.1f}%"

        if diff_ratio is not None and abs(diff_ratio) > 50.0:
            strong_discriminator_count += 1

        rows.append(
            {
                "Feature": feature,
                "Owner Avg": owner_avg,
                "Impostor Avg": impostor_avg,
                "Difference": difference_text,
            }
        )

    comparison_df = pd.DataFrame(rows)
    return comparison_df, strong_discriminator_count


def format_comparison_table(comparison_df: pd.DataFrame) -> str:
    """Format the comparison table for console output and report saving."""
    lines = []
    header = f"{'Feature':<32} | {'Owner Avg':<12} | {'Impostor Avg':<13} | {'Difference':<10}"
    separator = "-" * len(header)
    lines.append(header)
    lines.append(separator)

    for _, row in comparison_df.iterrows():
        owner_avg = f"{row['Owner Avg']:.4f}"
        if pd.isna(row["Impostor Avg"]):
            impostor_avg = "N/A"
        else:
            impostor_avg = f"{row['Impostor Avg']:.4f}"

        lines.append(
            f"{row['Feature']:<32} | {owner_avg:<12} | {impostor_avg:<13} | {row['Difference']:<10}"
        )

    return "\n".join(lines)


def save_report(report_text: str) -> None:
    """Save the comparison report to disk."""
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report_text, encoding="utf-8")


def analyze_behavior(features_file: str) -> None:
    """Analyze owner vs impostor behavior using strict label-based grouping."""
    try:
        df = load_behavior_data(features_file)
        owner_data, impostor_data = split_by_label(df)
        feature_columns = get_feature_columns(df)

        print(f"Owner rows found: {len(owner_data)}")
        print(f"Impostor rows found: {len(impostor_data)}")

        if impostor_data.empty:
            print("No impostor data (label=0) found. Showing owner data statistics only.")

        comparison_df, strong_discriminator_count = compute_comparison_table(
            owner_data, impostor_data, feature_columns
        )

        table_text = format_comparison_table(comparison_df)
        summary_lines = [
            "",
            f"Features with >50% difference between owner and impostor: {strong_discriminator_count} out of {len(feature_columns)}",
            "These features are likely the strongest discriminators for your model.",
        ]

        report_lines = [
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "=== BEHAVIOR COMPARISON ===",
            table_text,
            *summary_lines,
        ]

        report_text = "\n".join(report_lines) + "\n"
        print(report_text)
        save_report(report_text)
        print(f"Comparison report saved to {REPORT_FILE}")

    except Exception as error:
        print(f"Error analyzing data: {error}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare behavioral feature means between owner and impostor rows."
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default=str(DEFAULT_FEATURES_FILE),
        help=(
            "Path to the features CSV. Falls back to the sample dataset if the "
            "default file is missing."
        ),
    )

    args = parser.parse_args()
    analyze_behavior(args.features_file)


if __name__ == "__main__":
    main()
