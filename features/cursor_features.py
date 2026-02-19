"""
STEP 2: Cursor Feature Engineering
==================================

Convert raw cursor behavior data into numerical features for ML.
This is feature engineering - NO machine learning yet.

Extracts:
- Average speed
- Acceleration patterns
- Direction variance
- Movement smoothness
"""

import numpy as np
from typing import List, Dict
from dataclasses import dataclass
import pandas as pd


@dataclass
class CursorFeatures:
    """Extracted cursor movement features for ML."""
    avg_velocity: float
    max_velocity: float
    std_velocity: float
    avg_acceleration: float
    max_acceleration: float
    std_acceleration: float
    avg_direction_change: float
    movement_smoothness: float
    total_distance: float
    time_span: float
    idle_gap_avg: float
    idle_gap_max: float


class CursorFeatureExtractor:
    """
    Extract features from raw cursor movement data.
    
    This is STEP 2 - feature engineering from raw behavior data.
    """
    
    @staticmethod
    def extract_from_events(events: List) -> CursorFeatures:
        """
        Extract features from cursor events list.
        
        Args:
            events: List of CursorEvent objects
        
        Returns:
            CursorFeatures object
        """
        if not events or len(events) < 2:
            return CursorFeatureExtractor._empty_features()
        
        velocities = [e.velocity for e in events if hasattr(e, 'velocity')]
        accelerations = [e.acceleration for e in events if hasattr(e, 'acceleration')]
        dx_values = [e.dx for e in events if hasattr(e, 'dx')]
        dy_values = [e.dy for e in events if hasattr(e, 'dy')]
        idle_gaps = [e.idle_gap for e in events if hasattr(e, 'idle_gap')]
        
        if not velocities:
            return CursorFeatureExtractor._empty_features()
        
        # Calculate direction changes (angle variance)
        directions = []
        for i in range(1, len(events)):
            if hasattr(events[i], 'dx') and hasattr(events[i-1], 'dx'):
                dir1 = np.arctan2(events[i-1].dy, events[i-1].dx)
                dir2 = np.arctan2(events[i].dy, events[i].dx)
                angle_diff = abs(dir2 - dir1)
                # Normalize to [0, π]
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                directions.append(angle_diff)
        
        # Calculate movement smoothness (inverse of jerk)
        jerks = []
        for i in range(1, len(accelerations)):
            if hasattr(events[i], 'acceleration') and hasattr(events[i-1], 'acceleration'):
                dt = events[i].timestamp - events[i-1].timestamp
                if dt > 0:
                    jerk = abs(events[i].acceleration - events[i-1].acceleration) / dt
                    jerks.append(jerk)
        
        smoothness = 1.0 / (np.mean(jerks) + 1e-6) if jerks else 0.0
        
        # Calculate total distance
        total_distance = sum(
            (dx**2 + dy**2)**0.5 
            for dx, dy in zip(dx_values, dy_values)
        )
        
        # Time span
        time_span = events[-1].timestamp - events[0].timestamp if len(events) > 1 else 0.0
        
        return CursorFeatures(
            avg_velocity=np.mean(velocities),
            max_velocity=np.max(velocities),
            std_velocity=np.std(velocities),
            avg_acceleration=np.mean(accelerations),
            max_acceleration=np.max(accelerations),
            std_acceleration=np.std(accelerations),
            avg_direction_change=np.mean(directions) if directions else 0.0,
            movement_smoothness=smoothness,
            total_distance=total_distance,
            time_span=time_span,
            idle_gap_avg=np.mean(idle_gaps) if idle_gaps else 0.0,
            idle_gap_max=np.max(idle_gaps) if idle_gaps else 0.0
        )
    
    @staticmethod
    def extract_from_dataframe(df: pd.DataFrame) -> Dict:
        """
        Extract features from pandas DataFrame.
        
        Args:
            df: DataFrame with cursor event columns
        
        Returns:
            Dictionary of feature names and values
        """
        if df.empty or len(df) < 2:
            return CursorFeatureExtractor._empty_features_dict()
        
        velocities = df['velocity'].values
        accelerations = df['acceleration'].values
        dx_values = df['dx'].values
        dy_values = df['dy'].values
        idle_gaps = df.get('idle_gap', pd.Series([0.0] * len(df))).values
        
        # Direction changes
        directions = []
        for i in range(1, len(df)):
            dir1 = np.arctan2(dy_values[i-1], dx_values[i-1])
            dir2 = np.arctan2(dy_values[i], dx_values[i])
            angle_diff = abs(dir2 - dir1)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff
            directions.append(angle_diff)
        
        # Jerk calculation for smoothness
        jerks = []
        timestamps = df['timestamp'].values
        for i in range(1, len(accelerations)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                jerk = abs(accelerations[i] - accelerations[i-1]) / dt
                jerks.append(jerk)
        
        smoothness = 1.0 / (np.mean(jerks) + 1e-6) if jerks else 0.0
        
        # Total distance
        total_distance = np.sum(np.sqrt(dx_values**2 + dy_values**2))
        
        # Time span
        time_span = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
        
        return {
            'cursor_avg_velocity': float(np.mean(velocities)),
            'cursor_max_velocity': float(np.max(velocities)),
            'cursor_std_velocity': float(np.std(velocities)),
            'cursor_avg_acceleration': float(np.mean(accelerations)),
            'cursor_max_acceleration': float(np.max(accelerations)),
            'cursor_std_acceleration': float(np.std(accelerations)),
            'cursor_avg_direction_change': float(np.mean(directions)) if directions else 0.0,
            'cursor_movement_smoothness': float(smoothness),
            'cursor_total_distance': float(total_distance),
            'cursor_time_span': float(time_span),
            'cursor_idle_gap_avg': float(np.mean(idle_gaps)),
            'cursor_idle_gap_max': float(np.max(idle_gaps))
        }
    
    @staticmethod
    def _empty_features() -> CursorFeatures:
        """Return empty features."""
        return CursorFeatures(
            avg_velocity=0.0,
            max_velocity=0.0,
            std_velocity=0.0,
            avg_acceleration=0.0,
            max_acceleration=0.0,
            std_acceleration=0.0,
            avg_direction_change=0.0,
            movement_smoothness=0.0,
            total_distance=0.0,
            time_span=0.0,
            idle_gap_avg=0.0,
            idle_gap_max=0.0
        )
    
    @staticmethod
    def _empty_features_dict() -> Dict:
        """Return empty features as dictionary."""
        return {
            'cursor_avg_velocity': 0.0,
            'cursor_max_velocity': 0.0,
            'cursor_std_velocity': 0.0,
            'cursor_avg_acceleration': 0.0,
            'cursor_max_acceleration': 0.0,
            'cursor_std_acceleration': 0.0,
            'cursor_avg_direction_change': 0.0,
            'cursor_movement_smoothness': 0.0,
            'cursor_total_distance': 0.0,
            'cursor_time_span': 0.0,
            'cursor_idle_gap_avg': 0.0,
            'cursor_idle_gap_max': 0.0
        }
