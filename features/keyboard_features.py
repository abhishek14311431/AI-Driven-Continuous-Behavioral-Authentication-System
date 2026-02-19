"""
STEP 2: Keyboard Feature Engineering
======================================

Convert raw keyboard behavior data into numerical features for ML.
This is feature engineering - NO machine learning yet.

Extracts:
- Average key hold time
- Inter-key delay patterns
- Typing rhythm
- Pause frequency
"""

import numpy as np
from typing import List, Dict
from dataclasses import dataclass
import pandas as pd


@dataclass
class KeyboardFeatures:
    """Extracted keyboard typing features for ML."""
    avg_hold_duration: float
    std_hold_duration: float
    avg_inter_key_delay: float
    std_inter_key_delay: float
    typing_speed: float  # keys per second
    pause_frequency: float  # frequency of pauses > threshold
    key_hold_variance: float
    min_inter_key_delay: float
    max_inter_key_delay: float


class KeyboardFeatureExtractor:
    """
    Extract features from raw keyboard typing data.
    
    This is STEP 2 - feature engineering from raw behavior data.
    """
    
    @staticmethod
    def extract_from_keystrokes(keystrokes: List) -> KeyboardFeatures:
        """
        Extract features from keystroke list.
        
        Args:
            keystrokes: List of Keystroke objects
        
        Returns:
            KeyboardFeatures object
        """
        if not keystrokes or len(keystrokes) < 2:
            return KeyboardFeatureExtractor._empty_features()
        
        hold_durations = [ks.hold_duration for ks in keystrokes if hasattr(ks, 'hold_duration')]
        
        # Calculate inter-key delays
        inter_key_delays = []
        for i in range(1, len(keystrokes)):
            if hasattr(keystrokes[i], 'press_time') and hasattr(keystrokes[i-1], 'release_time'):
                delay = keystrokes[i].press_time - keystrokes[i-1].release_time
                if delay >= 0:  # Only valid delays
                    inter_key_delays.append(delay)
        
        # Typing speed (keys per second)
        if len(keystrokes) > 1:
            time_span = keystrokes[-1].release_time - keystrokes[0].press_time
            typing_speed = len(keystrokes) / time_span if time_span > 0 else 0.0
        else:
            typing_speed = 0.0
        
        # Pause frequency (pauses > 0.5 seconds)
        pause_threshold = 0.5
        pauses = [d for d in inter_key_delays if d > pause_threshold]
        pause_frequency = len(pauses) / len(inter_key_delays) if inter_key_delays else 0.0
        
        return KeyboardFeatures(
            avg_hold_duration=np.mean(hold_durations) if hold_durations else 0.0,
            std_hold_duration=np.std(hold_durations) if hold_durations else 0.0,
            avg_inter_key_delay=np.mean(inter_key_delays) if inter_key_delays else 0.0,
            std_inter_key_delay=np.std(inter_key_delays) if inter_key_delays else 0.0,
            typing_speed=typing_speed,
            pause_frequency=pause_frequency,
            key_hold_variance=np.var(hold_durations) if hold_durations else 0.0,
            min_inter_key_delay=np.min(inter_key_delays) if inter_key_delays else 0.0,
            max_inter_key_delay=np.max(inter_key_delays) if inter_key_delays else 0.0
        )
    
    @staticmethod
    def extract_from_dataframe(df: pd.DataFrame) -> Dict:
        """
        Extract features from pandas DataFrame.
        
        Args:
            df: DataFrame with keyboard event columns
        
        Returns:
            Dictionary of feature names and values
        """
        if df.empty or len(df) < 2:
            return KeyboardFeatureExtractor._empty_features_dict()
        
        # Separate press and release events
        press_events = df[df['event_type'] == 'press'].copy()
        release_events = df[df['event_type'] == 'release'].copy()
        
        if len(press_events) < 2 or len(release_events) < 2:
            return KeyboardFeatureExtractor._empty_features_dict()
        
        # Match press and release events to calculate hold durations
        hold_durations = []
        inter_key_delays = []
        
        press_times = press_events['timestamp'].values
        release_times = release_events['timestamp'].values
        
        # Calculate hold durations (simplified - match by order)
        min_len = min(len(press_times), len(release_times))
        for i in range(min_len):
            if release_times[i] > press_times[i]:
                hold_durations.append(release_times[i] - press_times[i])
        
        # Calculate inter-key delays
        for i in range(1, min_len):
            if i < len(release_times) and i < len(press_times):
                delay = press_times[i] - release_times[i-1]
                if delay > 0:
                    inter_key_delays.append(delay)
        
        # Typing speed
        if len(press_times) > 1:
            time_span = release_times[-1] - press_times[0] if len(release_times) > 0 else 0
            typing_speed = len(press_times) / time_span if time_span > 0 else 0.0
        else:
            typing_speed = 0.0
        
        # Pause frequency
        pause_threshold = 0.5
        pauses = [d for d in inter_key_delays if d > pause_threshold]
        pause_frequency = len(pauses) / len(inter_key_delays) if inter_key_delays else 0.0
        
        return {
            'keyboard_avg_hold_duration': float(np.mean(hold_durations)) if hold_durations else 0.0,
            'keyboard_std_hold_duration': float(np.std(hold_durations)) if hold_durations else 0.0,
            'keyboard_avg_inter_key_delay': float(np.mean(inter_key_delays)) if inter_key_delays else 0.0,
            'keyboard_std_inter_key_delay': float(np.std(inter_key_delays)) if inter_key_delays else 0.0,
            'keyboard_typing_speed': float(typing_speed),
            'keyboard_pause_frequency': float(pause_frequency),
            'keyboard_key_hold_variance': float(np.var(hold_durations)) if hold_durations else 0.0,
            'keyboard_min_inter_key_delay': float(np.min(inter_key_delays)) if inter_key_delays else 0.0,
            'keyboard_max_inter_key_delay': float(np.max(inter_key_delays)) if inter_key_delays else 0.0
        }
    
    @staticmethod
    def _empty_features() -> KeyboardFeatures:
        """Return empty features."""
        return KeyboardFeatures(
            avg_hold_duration=0.0,
            std_hold_duration=0.0,
            avg_inter_key_delay=0.0,
            std_inter_key_delay=0.0,
            typing_speed=0.0,
            pause_frequency=0.0,
            key_hold_variance=0.0,
            min_inter_key_delay=0.0,
            max_inter_key_delay=0.0
        )
    
    @staticmethod
    def _empty_features_dict() -> Dict:
        """Return empty features as dictionary."""
        return {
            'keyboard_avg_hold_duration': 0.0,
            'keyboard_std_hold_duration': 0.0,
            'keyboard_avg_inter_key_delay': 0.0,
            'keyboard_std_inter_key_delay': 0.0,
            'keyboard_typing_speed': 0.0,
            'keyboard_pause_frequency': 0.0,
            'keyboard_key_hold_variance': 0.0,
            'keyboard_min_inter_key_delay': 0.0,
            'keyboard_max_inter_key_delay': 0.0
        }
