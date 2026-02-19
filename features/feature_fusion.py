"""
STEP 2: Feature Fusion
=======================

Combine cursor and keyboard features into unified feature vectors.
This prepares data for machine learning (STEP 4).

Creates ML-ready feature vectors from behavioral data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import os

from .cursor_features import CursorFeatureExtractor
from .keyboard_features import KeyboardFeatureExtractor


class FeatureFusion:
    """
    Combine and fuse features from cursor and keyboard behavior.
    
    This is STEP 2 - preparing features for ML training.
    """
    
    def __init__(self, output_file: str = "data/processed/behavior_features.csv"):
        """
        Initialize feature fusion.
        
        Args:
            output_file: Path to save processed features
        """
        self.output_file = output_file
        self.cursor_extractor = CursorFeatureExtractor()
        self.keyboard_extractor = KeyboardFeatureExtractor()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    def fuse_features(self, 
                     cursor_data: Optional[pd.DataFrame] = None,
                     keyboard_data: Optional[pd.DataFrame] = None,
                     window_size: float = 5.0,
                     label: int = 1) -> pd.DataFrame:
        """
        Fuse cursor and keyboard features over a time window.
        
        Args:
            cursor_data: DataFrame with cursor events
            keyboard_data: DataFrame with keyboard events
            window_size: Time window in seconds for feature extraction (2-5s for real-time)
            label: Label for this behavior (1 = owner, 0 = not-owner)
        
        Returns:
            DataFrame with fused features
        """
        features_dict = {}
        
        # Process cursor features
        if cursor_data is not None and not cursor_data.empty:
            cursor_features = self.cursor_extractor.extract_from_dataframe(cursor_data)
            features_dict.update(cursor_features)
        else:
            # Use empty features if no cursor data
            features_dict.update(self.cursor_extractor._empty_features_dict())
        
        # Process keyboard features
        if keyboard_data is not None and not keyboard_data.empty:
            keyboard_features = self.keyboard_extractor.extract_from_dataframe(keyboard_data)
            features_dict.update(keyboard_features)
        else:
            # Use empty features if no keyboard data
            features_dict.update(self.keyboard_extractor._empty_features_dict())
        
        # Add metadata
        features_dict['timestamp'] = datetime.now().timestamp()
        features_dict['window_size'] = window_size
        features_dict['label'] = label  # 1 = owner, 0 = not-owner (for binary classification)
        
        return pd.DataFrame([features_dict])
    
    def fuse_from_files(self,
                       cursor_file: str = "data/raw/cursor_logs.csv",
                       keyboard_file: str = "data/raw/keyboard_logs.csv",
                       window_size: float = 5.0,
                       label: int = 1) -> pd.DataFrame:
        """
        Load data from files and fuse features.
        
        Args:
            cursor_file: Path to cursor log file
            keyboard_file: Path to keyboard log file
            window_size: Time window in seconds
            label: Label for this behavior (1 = owner)
        
        Returns:
            DataFrame with fused features
        """
        cursor_data = None
        keyboard_data = None
        
        if os.path.exists(cursor_file):
            try:
                cursor_data = pd.read_csv(cursor_file)
            except Exception as e:
                print(f"Error loading cursor data: {e}")
        
        if os.path.exists(keyboard_file):
            try:
                keyboard_data = pd.read_csv(keyboard_file)
            except Exception as e:
                print(f"Error loading keyboard data: {e}")
        
        return self.fuse_features(cursor_data, keyboard_data, window_size, label)
    
    def save_features(self, features_df: pd.DataFrame, append: bool = True):
        """
        Save features to CSV file.
        
        Args:
            features_df: DataFrame with features
            append: Whether to append to existing file
        """
        try:
            if append and os.path.exists(self.output_file):
                # Append to existing file
                existing_df = pd.read_csv(self.output_file)
                combined_df = pd.concat([existing_df, features_df], ignore_index=True)
                combined_df.to_csv(self.output_file, index=False)
            else:
                # Create new file
                features_df.to_csv(self.output_file, index=False)
        except Exception as e:
            print(f"Error saving features: {e}")
    
    def process_window(self,
                      cursor_data: Optional[pd.DataFrame],
                      keyboard_data: Optional[pd.DataFrame],
                      window_size: float = 5.0,
                      label: int = 1,
                      save: bool = True) -> pd.DataFrame:
        """
        Process a time window and extract features.
        
        Args:
            cursor_data: Cursor events in window
            keyboard_data: Keyboard events in window
            window_size: Window size in seconds
            label: Label (1 = owner)
            save: Whether to save to file
        
        Returns:
            DataFrame with extracted features
        """
        features_df = self.fuse_features(cursor_data, keyboard_data, window_size, label)
        
        if save:
            self.save_features(features_df, append=True)
        
        return features_df
    
    def get_feature_columns(self) -> List[str]:
        """
        Get list of all feature column names.
        
        Returns:
            List of feature column names
        """
        cursor_cols = [
            'cursor_avg_velocity', 'cursor_max_velocity', 'cursor_std_velocity',
            'cursor_avg_acceleration', 'cursor_max_acceleration', 'cursor_std_acceleration',
            'cursor_avg_direction_change', 'cursor_movement_smoothness',
            'cursor_total_distance', 'cursor_time_span',
            'cursor_idle_gap_avg', 'cursor_idle_gap_max'
        ]
        
        keyboard_cols = [
            'keyboard_avg_hold_duration', 'keyboard_std_hold_duration',
            'keyboard_avg_inter_key_delay', 'keyboard_std_inter_key_delay',
            'keyboard_typing_speed', 'keyboard_pause_frequency',
            'keyboard_key_hold_variance',
            'keyboard_min_inter_key_delay', 'keyboard_max_inter_key_delay'
        ]
        
        return cursor_cols + keyboard_cols
