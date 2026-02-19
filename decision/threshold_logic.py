"""
STEP 6: Decision Logic
======================

Threshold-based decision logic:
- High confidence (≥0.8) → ALLOW (owner detected)
- Medium confidence (0.5-0.8) → MONITOR (suspicious)
- Low confidence (<0.5) → IMMEDIATE LOCK (not-owner detected)

This is a single-user, zero-trust system.
Anyone who is not the owner is treated as an attacker.
"""

import time
from typing import Dict
from enum import Enum

from .inference_engine import InferenceEngine


class SecurityAction(Enum):
    """Security actions based on confidence levels."""
    ALLOW = "allow"      # High confidence - owner verified
    MONITOR = "monitor"  # Medium confidence - suspicious
    LOCK = "lock"        # Low confidence - not-owner detected


class ThresholdLogic:
    """
    Decision logic based on confidence thresholds.
    
    This is STEP 6 - security decision making.
    Single-user system: Only owner (class=1) is authorized.
    """
    
    def __init__(self,
                 allow_threshold: float = 0.8,
                 monitor_threshold: float = 0.5,
                 lock_threshold: float = 0.3):
        """
        Initialize threshold logic.
        
        Args:
            allow_threshold: Confidence ≥ this = ALLOW (default: 0.8)
            monitor_threshold: Confidence between this and allow = MONITOR (default: 0.7)
            lock_threshold: Confidence < this = LOCK (default: 0.3)
        """
        self.allow_threshold = allow_threshold
        self.monitor_threshold = monitor_threshold
        self.lock_threshold = lock_threshold
    
    def decide(self, predicted_class: int, confidence: float) -> Dict:
        """
        Make security decision based on prediction.
        
        CRITICAL: This is a binary classification system.
        - predicted_class = 1 → Owner (authorized)
        - predicted_class = 0 → Not-owner (attacker)
        
        Args:
            predicted_class: Predicted class (1=owner, 0=not-owner)
            confidence: Confidence score (0-1)
        
        Returns:
            Dictionary with decision details
        """
        # Check if predicted as owner
        is_owner = (predicted_class == 1)
        
        # Determine action based on confidence and owner match
        if is_owner:
            if confidence >= self.allow_threshold:
                # High confidence owner detection
                action = SecurityAction.ALLOW
                risk_level = "low"
            elif confidence >= 0.7:
                # Medium/High confidence owner (0.7-0.8)
                action = SecurityAction.MONITOR
                risk_level = "medium"
            else:
                 # Confidence < 0.7 (User Request: Strict Lock)
                 action = SecurityAction.LOCK
                 risk_level = "critical"
        else:
            # Predicted as NOT OWNER (Anomaly)
            # Logically, if it's an anomaly, confidence is usually low (score is negative).
            # But if we use probabilities, high prob of "not owner" means low prob of "owner".
            # The 'confidence' passed here is usually max(prob).
            # If predicted_class is 0, confidence is prob(not_owner).
            # If we are sure it's not the owner -> LOCK.
            action = SecurityAction.LOCK
            risk_level = "high"
        
        decision = {
            'action': action.value,
            'risk_level': risk_level,
            'predicted_class': predicted_class,
            'is_owner': is_owner,
            'confidence': confidence,
            'should_lock': (action == SecurityAction.LOCK),
            'should_monitor': (action == SecurityAction.MONITOR),
            'timestamp': time.time()
        }
        
        return decision
    
    def update_thresholds(self,
                         allow_threshold: float = None,
                         monitor_threshold: float = None,
                         lock_threshold: float = None):
        """Update threshold values."""
        if allow_threshold is not None:
            self.allow_threshold = allow_threshold
        if monitor_threshold is not None:
            self.monitor_threshold = monitor_threshold
        if lock_threshold is not None:
            self.lock_threshold = lock_threshold
    
    def get_thresholds(self) -> Dict:
        """Get current threshold values."""
        return {
            'allow_threshold': self.allow_threshold,
            'monitor_threshold': self.monitor_threshold,
            'lock_threshold': self.lock_threshold
        }


class SecurityDecisionMaker:
    """
    Combines inference and threshold logic for security decisions.
    
    This is the main decision-making component for STEP 6.
    """
    
    def __init__(self,
                 inference_engine: InferenceEngine,
                 threshold_logic: ThresholdLogic):
        """
        Initialize security decision maker.
        
        Args:
            inference_engine: Inference engine for predictions
            threshold_logic: Threshold logic for decisions
        """
        self.inference_engine = inference_engine
        self.threshold_logic = threshold_logic
    
    def make_decision(self,
                     cursor_data=None,
                     keyboard_data=None) -> Dict:
        """
        Make complete security decision from raw behavior data.
        
        Uses sliding window (2-5 seconds) for real-time detection.
        
        Args:
            cursor_data: Cursor events in current window
            keyboard_data: Keyboard events in current window
        
        Returns:
            Dictionary with prediction and decision
        """
        # Predict using sliding window
        predicted_class, confidence, probabilities = self.inference_engine.predict_from_raw(
            cursor_data, keyboard_data
        )
        
        # Make decision
        decision = self.threshold_logic.decide(predicted_class, confidence)
        
        # Combine results
        result = {
            **decision,
            'probabilities': probabilities,
            'window_size': self.inference_engine.window_size
        }
        
        return result
