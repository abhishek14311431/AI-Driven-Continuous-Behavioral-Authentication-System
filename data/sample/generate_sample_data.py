"""Generate synthetic behavioral feature data for the single-user authentication project.

This script creates a reproducible 100-row CSV with 70 owner rows and 30 impostor
rows using a fixed NumPy random seed.
"""

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
OUTPUT_FILE = Path(__file__).with_name("behavior_features_sample.csv")

COLUMNS = [
    "timestamp",
    "cursor_avg_velocity",
    "cursor_max_velocity",
    "cursor_std_velocity",
    "cursor_avg_acceleration",
    "cursor_max_acceleration",
    "cursor_std_acceleration",
    "cursor_avg_direction_change",
    "cursor_movement_smoothness",
    "cursor_total_distance",
    "cursor_time_span",
    "cursor_idle_gap_avg",
    "cursor_idle_gap_max",
    "keyboard_avg_hold_duration",
    "keyboard_std_hold_duration",
    "keyboard_avg_inter_key_delay",
    "keyboard_std_inter_key_delay",
    "keyboard_typing_speed",
    "keyboard_pause_frequency",
    "keyboard_key_hold_variance",
    "keyboard_min_inter_key_delay",
    "keyboard_max_inter_key_delay",
    "window_size",
    "label",
]


def clipped_normal(center: float, scale: float, low: float, high: float, size: int) -> np.ndarray:
    values = np.random.normal(center, scale, size)
    return np.clip(values, low, high)


def build_owner_rows(count: int) -> pd.DataFrame:
    timestamps = 1_700_000_000 + np.arange(count) * 5.0

    cursor_avg_velocity = clipped_normal(300, 35, 200, 400, count)
    cursor_std_velocity = clipped_normal(42, 6, 25, 60, count)
    cursor_max_velocity = cursor_avg_velocity + np.abs(clipped_normal(120, 20, 70, 180, count))

    cursor_avg_acceleration = clipped_normal(55, 8, 30, 80, count)
    cursor_std_acceleration = clipped_normal(10, 2, 5, 16, count)
    cursor_max_acceleration = cursor_avg_acceleration + np.abs(clipped_normal(35, 8, 18, 60, count))

    cursor_avg_direction_change = clipped_normal(0.28, 0.04, 0.15, 0.40, count)
    cursor_movement_smoothness = clipped_normal(0.86, 0.03, 0.78, 0.95, count)
    cursor_time_span = np.full(count, 5.0)
    cursor_total_distance = cursor_avg_velocity * cursor_time_span * clipped_normal(0.88, 0.05, 0.75, 1.0, count)
    cursor_idle_gap_avg = clipped_normal(0.18, 0.03, 0.10, 0.28, count)
    cursor_idle_gap_max = cursor_idle_gap_avg + np.abs(clipped_normal(0.22, 0.05, 0.12, 0.35, count))

    keyboard_avg_hold_duration = clipped_normal(0.11, 0.01, 0.08, 0.15, count)
    keyboard_std_hold_duration = clipped_normal(0.018, 0.004, 0.010, 0.030, count)
    keyboard_avg_inter_key_delay = clipped_normal(0.16, 0.02, 0.10, 0.22, count)
    keyboard_std_inter_key_delay = clipped_normal(0.030, 0.006, 0.015, 0.045, count)
    keyboard_typing_speed = clipped_normal(5.1, 0.35, 4.0, 6.0, count)
    keyboard_pause_frequency = clipped_normal(0.18, 0.03, 0.08, 0.28, count)
    keyboard_key_hold_variance = clipped_normal(0.00024, 0.00005, 0.00010, 0.00040, count)
    keyboard_min_inter_key_delay = np.maximum(
        0.05,
        keyboard_avg_inter_key_delay - np.abs(clipped_normal(0.05, 0.01, 0.02, 0.08, count)),
    )
    keyboard_max_inter_key_delay = keyboard_avg_inter_key_delay + np.abs(clipped_normal(0.11, 0.02, 0.05, 0.16, count))

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "cursor_avg_velocity": cursor_avg_velocity,
            "cursor_max_velocity": cursor_max_velocity,
            "cursor_std_velocity": cursor_std_velocity,
            "cursor_avg_acceleration": cursor_avg_acceleration,
            "cursor_max_acceleration": cursor_max_acceleration,
            "cursor_std_acceleration": cursor_std_acceleration,
            "cursor_avg_direction_change": cursor_avg_direction_change,
            "cursor_movement_smoothness": cursor_movement_smoothness,
            "cursor_total_distance": cursor_total_distance,
            "cursor_time_span": cursor_time_span,
            "cursor_idle_gap_avg": cursor_idle_gap_avg,
            "cursor_idle_gap_max": cursor_idle_gap_max,
            "keyboard_avg_hold_duration": keyboard_avg_hold_duration,
            "keyboard_std_hold_duration": keyboard_std_hold_duration,
            "keyboard_avg_inter_key_delay": keyboard_avg_inter_key_delay,
            "keyboard_std_inter_key_delay": keyboard_std_inter_key_delay,
            "keyboard_typing_speed": keyboard_typing_speed,
            "keyboard_pause_frequency": keyboard_pause_frequency,
            "keyboard_key_hold_variance": keyboard_key_hold_variance,
            "keyboard_min_inter_key_delay": keyboard_min_inter_key_delay,
            "keyboard_max_inter_key_delay": keyboard_max_inter_key_delay,
            "window_size": np.full(count, 5.0),
            "label": np.ones(count, dtype=int),
        }
    )


def build_impostor_rows(count: int, start_timestamp: float) -> pd.DataFrame:
    timestamps = start_timestamp + np.arange(count) * 5.0

    cursor_avg_velocity = clipped_normal(650, 45, 500, 800, count)
    cursor_std_velocity = clipped_normal(78, 10, 50, 110, count)
    cursor_max_velocity = cursor_avg_velocity + np.abs(clipped_normal(180, 25, 110, 260, count))

    cursor_avg_acceleration = clipped_normal(118, 14, 85, 150, count)
    cursor_std_acceleration = clipped_normal(19, 3, 12, 28, count)
    cursor_max_acceleration = cursor_avg_acceleration + np.abs(clipped_normal(55, 10, 30, 80, count))

    cursor_avg_direction_change = clipped_normal(0.52, 0.06, 0.35, 0.70, count)
    cursor_movement_smoothness = clipped_normal(0.63, 0.05, 0.50, 0.76, count)
    cursor_time_span = np.full(count, 5.0)
    cursor_total_distance = cursor_avg_velocity * cursor_time_span * clipped_normal(0.91, 0.04, 0.80, 1.0, count)
    cursor_idle_gap_avg = clipped_normal(0.42, 0.06, 0.25, 0.60, count)
    cursor_idle_gap_max = cursor_idle_gap_avg + np.abs(clipped_normal(0.36, 0.07, 0.18, 0.55, count))

    keyboard_avg_hold_duration = clipped_normal(0.23, 0.02, 0.18, 0.30, count)
    keyboard_std_hold_duration = clipped_normal(0.036, 0.006, 0.020, 0.055, count)
    keyboard_avg_inter_key_delay = clipped_normal(0.31, 0.03, 0.22, 0.40, count)
    keyboard_std_inter_key_delay = clipped_normal(0.055, 0.008, 0.035, 0.075, count)
    keyboard_typing_speed = clipped_normal(2.8, 0.25, 2.0, 3.5, count)
    keyboard_pause_frequency = clipped_normal(0.39, 0.05, 0.25, 0.55, count)
    keyboard_key_hold_variance = clipped_normal(0.00062, 0.00009, 0.00035, 0.00090, count)
    keyboard_min_inter_key_delay = np.maximum(
        0.10,
        keyboard_avg_inter_key_delay - np.abs(clipped_normal(0.08, 0.02, 0.03, 0.12, count)),
    )
    keyboard_max_inter_key_delay = keyboard_avg_inter_key_delay + np.abs(clipped_normal(0.18, 0.03, 0.10, 0.25, count))

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "cursor_avg_velocity": cursor_avg_velocity,
            "cursor_max_velocity": cursor_max_velocity,
            "cursor_std_velocity": cursor_std_velocity,
            "cursor_avg_acceleration": cursor_avg_acceleration,
            "cursor_max_acceleration": cursor_max_acceleration,
            "cursor_std_acceleration": cursor_std_acceleration,
            "cursor_avg_direction_change": cursor_avg_direction_change,
            "cursor_movement_smoothness": cursor_movement_smoothness,
            "cursor_total_distance": cursor_total_distance,
            "cursor_time_span": cursor_time_span,
            "cursor_idle_gap_avg": cursor_idle_gap_avg,
            "cursor_idle_gap_max": cursor_idle_gap_max,
            "keyboard_avg_hold_duration": keyboard_avg_hold_duration,
            "keyboard_std_hold_duration": keyboard_std_hold_duration,
            "keyboard_avg_inter_key_delay": keyboard_avg_inter_key_delay,
            "keyboard_std_inter_key_delay": keyboard_std_inter_key_delay,
            "keyboard_typing_speed": keyboard_typing_speed,
            "keyboard_pause_frequency": keyboard_pause_frequency,
            "keyboard_key_hold_variance": keyboard_key_hold_variance,
            "keyboard_min_inter_key_delay": keyboard_min_inter_key_delay,
            "keyboard_max_inter_key_delay": keyboard_max_inter_key_delay,
            "window_size": np.full(count, 5.0),
            "label": np.zeros(count, dtype=int),
        }
    )


def main() -> None:
    np.random.seed(SEED)

    owner_df = build_owner_rows(70)
    impostor_df = build_impostor_rows(30, start_timestamp=1_700_000_000 + 70 * 5.0)

    df = pd.concat([owner_df, impostor_df], ignore_index=True)
    df = df[COLUMNS]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()