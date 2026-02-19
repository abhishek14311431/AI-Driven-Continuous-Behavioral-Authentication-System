"""
STEP 5: Real-Time Inference Engine
===================================

Load model & predict confidence scores using 2-5 second sliding windows.
This performs real-time behavioral authentication.

Uses sliding window approach for continuous verification.
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from model.train_model import BehaviorModelTrainer
from model.wrapper import AnomalyDetectionWrapper
from features.feature_fusion import FeatureFusion


class InferenceEngine:
    """
    Perform real-time behavior authentication inference.
    
    This is STEP 5 - real-time prediction using sliding windows.
    Window size: 2-5 seconds for fast detection.
    """
    
    def __init__(self,
                 model_file: str = "model/behavior_model.pkl",
                 scaler_file: str = "model/feature_scaler.pkl",
                 window_size: float = 5.0):
        """
        Initialize inference engine.
        
        Args:
            model_file: Path to trained model
            scaler_file: Path to feature scaler
            window_size: Sliding window size in seconds (2-5s for real-time)
        """
        self.model_file = model_file
        self.scaler_file = scaler_file
        self.window_size = window_size
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.fusion = FeatureFusion()
        self._load_model()
    
    def _load_model(self):
        """Load trained model and scaler."""
        if not os.path.exists(self.model_file):
            raise FileNotFoundError(f"Model file not found: {self.model_file}")
        
        if not os.path.exists(self.scaler_file):
            raise FileNotFoundError(f"Scaler file not found: {self.scaler_file}")
        
        with open(self.model_file, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(self.scaler_file, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load feature columns
        feature_info_file = self.model_file.replace('.pkl', '_features.txt')
        if os.path.exists(feature_info_file):
            with open(feature_info_file, 'r') as f:
                self.feature_columns = [line.strip() for line in f.readlines()]
        else:
            # Fallback to FeatureFusion
            self.feature_columns = self.fusion.get_feature_columns()
    
    def extract_features(self,
                        cursor_data: Optional[pd.DataFrame] = None,
                        keyboard_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Extract features from raw behavior data using sliding window.
        
        Args:
            cursor_data: Cursor events in window
            keyboard_data: Keyboard events in window
        
        Returns:
            DataFrame with extracted features
        """
        return self.fusion.fuse_features(
            cursor_data, 
            keyboard_data, 
            window_size=self.window_size,
            label=1  # Default label (not used in inference)
        )
    
    def predict(self, features_df: pd.DataFrame) -> Tuple[int, float, Dict]:
        """
        Predict user identity and confidence.
        
        Returns:
            - predicted_class: 1 = owner, 0 = not-owner
            - confidence: Confidence score (0-1)
            - probabilities: Dictionary of class probabilities
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not loaded")
        
        # Select and order features
        available_cols = [col for col in self.feature_columns if col in features_df.columns]
        if not available_cols:
            raise ValueError("No matching feature columns found")
        
        # Fill missing features with 0
        feature_vector = features_df[available_cols].fillna(0)
        
        # Ensure all feature columns are present
        for col in self.feature_columns:
            if col not in feature_vector.columns:
                feature_vector[col] = 0.0
        
        # Reorder to match training
        feature_vector = feature_vector[self.feature_columns]
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Predict
        predicted_class = self.model.predict(feature_vector_scaled)[0]
        probabilities = self.model.predict_proba(feature_vector_scaled)[0]
        confidence = np.max(probabilities)
        
        # Create probabilities dictionary
        prob_dict = {}
        if hasattr(self.model, 'classes_'):
            for i, class_label in enumerate(self.model.classes_):
                prob_dict[int(class_label)] = float(probabilities[i])
        
        return int(predicted_class), float(confidence), prob_dict
    
    def predict_from_raw(self,
                        cursor_data: Optional[pd.DataFrame] = None,
                        keyboard_data: Optional[pd.DataFrame] = None) -> Tuple[int, float, Dict]:
        """
        Extract features and predict in one step (sliding window).
        
        Args:
            cursor_data: Cursor events in current window
            keyboard_data: Keyboard events in current window
        
        Returns:
            Tuple of (predicted_class, confidence, probabilities)
        """
        features_df = self.extract_features(cursor_data, keyboard_data)
        return self.predict(features_df)
    
    def get_confidence_score(self, features_df: pd.DataFrame) -> float:
        """
        Get confidence score only (faster for quick checks).
        
        Args:
            features_df: DataFrame with features
        
        Returns:
            Confidence score (0-1)
        """
        _, confidence, _ = self.predict(features_df)
        return confidence
    
    def set_window_size(self, window_size: float):
        """
        Update sliding window size.
        
        Args:
            window_size: New window size in seconds (2-5s recommended)
        """
        if 2.0 <= window_size <= 10.0:
            self.window_size = window_size
        else:
            raise ValueError("Window size should be between 2-10 seconds")