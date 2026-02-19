"""
STEP 8: Adaptive Learning
==========================

Gradual re-training logic for adaptive model updates.

CRITICAL RULES:
1. Model adapts ONLY to owner behavior changes
2. NEVER retrain after suspicious sessions (attacker behavior)
3. Only retrain after HIGH confidence owner sessions
4. Use sliding window retraining with verified owner data
5. Human behavior changes gradually - adapt slowly

The model must NEVER learn from attacker behavior.
"""

import pandas as pd
import numpy as np
import os
from typing import Optional, Dict
from datetime import datetime, timedelta

from .train_model import BehaviorModelTrainer


class AdaptiveModelUpdater:
    """
    Handle gradual model re-training with new owner data.
    
    CRITICAL: This system NEVER learns from attacker behavior.
    Only high-confidence owner sessions are used for retraining.
    """
    
    def __init__(self,
                 features_file: str = "data/processed/behavior_features.csv",
                 model_file: str = "model/behavior_model.pkl",
                 scaler_file: str = "model/feature_scaler.pkl",
                 update_threshold: int = 100,  # Number of new owner samples before update
                min_update_interval: float = 3600.0,  # Minimum seconds between updates (1 hour)
                min_confidence_for_training: float = 0.85):  # Minimum confidence to use for training
        """
        Initialize adaptive updater.
        
        Args:
            features_file: Path to features file
            model_file: Path to model file
            scaler_file: Path to scaler file
            update_threshold: Number of new owner samples before update
            min_update_interval: Minimum seconds between updates
            min_confidence_for_training: Minimum confidence to consider data for training
        """
        self.trainer = BehaviorModelTrainer(
            features_file=features_file,
            model_file=model_file,
            scaler_file=scaler_file
        )
        self.update_threshold = update_threshold
        self.min_update_interval = min_update_interval
        self.min_confidence_for_training = min_confidence_for_training
        self.last_update_file = "model/last_update.txt"
        self.last_update_time = self._load_last_update_time()
        
        # Track high-confidence sessions (owner verified)
        self.verified_owner_sessions_file = "model/verified_owner_sessions.txt"
    
    def _load_last_update_time(self) -> Optional[datetime]:
        """Load last update timestamp."""
        if os.path.exists(self.last_update_file):
            try:
                with open(self.last_update_file, 'r') as f:
                    timestamp_str = f.read().strip()
                    return datetime.fromtimestamp(float(timestamp_str))
            except:
                pass
        return None
    
    def _save_last_update_time(self):
        """Save current update timestamp."""
        os.makedirs(os.path.dirname(self.last_update_file), exist_ok=True)
        with open(self.last_update_file, 'w') as f:
            f.write(str(datetime.now().timestamp()))
        self.last_update_time = datetime.now()
    
    def mark_verified_owner_session(self, confidence: float, timestamp: float):
        """
        Mark a session as verified owner (high confidence).
        
        CRITICAL: Only high-confidence owner sessions are marked.
        Low confidence or suspicious sessions are IGNORED.
        
        Args:
            confidence: Confidence score from prediction
            timestamp: Session timestamp
        """
        if confidence < self.min_confidence_for_training:
            # Not confident enough - don't mark as verified
            return
        
        try:
            os.makedirs(os.path.dirname(self.verified_owner_sessions_file), exist_ok=True)
            with open(self.verified_owner_sessions_file, 'a') as f:
                f.write(f"{timestamp},{confidence}\n")
        except Exception as e:
            print(f"Error marking verified session: {e}")
    
    def get_verified_owner_sessions(self) -> pd.DataFrame:
        """
        Get verified owner sessions (high confidence).
        
        Returns:
            DataFrame with verified session timestamps and confidences
        """
        if not os.path.exists(self.verified_owner_sessions_file):
            return pd.DataFrame(columns=['timestamp', 'confidence'])
        
        try:
            df = pd.read_csv(
                self.verified_owner_sessions_file,
                names=['timestamp', 'confidence'],
                header=None
            )
            return df
        except:
            return pd.DataFrame(columns=['timestamp', 'confidence'])
    
    def should_update(self) -> bool:
        """
        Check if model should be updated.
        
        CRITICAL: Only updates if:
        1. Enough time has passed since last update
        2. Enough verified owner samples are available
        3. NO suspicious/attacker data is used
        
        Returns:
            True if update conditions are met
        """
        # Check time interval
        if self.last_update_time:
            time_since_update = (datetime.now() - self.last_update_time).total_seconds()
            if time_since_update < self.min_update_interval:
                return False
        
        # Check number of verified owner samples
        if not os.path.exists(self.trainer.features_file):
            return False
        
        try:
            df = pd.read_csv(self.trainer.features_file)
            
            # CRITICAL: Only count owner data (label=1)
            owner_data = df[df['label'] == 1].copy()
            
            if self.last_update_time:
                # Count owner samples since last update
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
                new_owner_samples = owner_data[
                    pd.to_datetime(owner_data['timestamp'], unit='s', errors='coerce') > self.last_update_time
                ]
                return len(new_owner_samples) >= self.update_threshold
            else:
                # First update if we have enough owner samples
                return len(owner_data) >= self.update_threshold
        except Exception as e:
            print(f"Error checking update condition: {e}")
            return False
    
    def incremental_update(self, 
                          use_only_verified: bool = True,
                          min_confidence: float = 0.85) -> Dict:
        """
        Perform incremental model update with new owner data.
        
        CRITICAL: 
        - Only uses owner data (label=1)
        - Optionally filters to high-confidence sessions only
        - NEVER uses attacker data (label=0)
        
        Args:
            use_only_verified: Only use verified high-confidence sessions
            min_confidence: Minimum confidence for verified sessions
        
        Returns:
            Dictionary with update status and metrics
        """
        if not self.should_update():
            return {
                'status': 'skipped',
                'reason': 'Update conditions not met'
            }
        
        try:
            # Load existing model
            self.trainer.load_model()
            
            # Prepare training data
            df = pd.read_csv(self.trainer.features_file)
            
            # CRITICAL: Filter to ONLY owner data (label=1)
            owner_data = df[df['label'] == 1].copy()
            
            if len(owner_data) == 0:
                return {
                    'status': 'error',
                    'error': 'No owner data available for training'
                }
            
            # Optionally filter to verified high-confidence sessions
            if use_only_verified:
                verified_sessions = self.get_verified_owner_sessions()
                if not verified_sessions.empty:
                    # Filter owner_data to only verified sessions
                    owner_data['timestamp_dt'] = pd.to_datetime(
                        owner_data['timestamp'], unit='s', errors='coerce'
                    )
                    verified_timestamps = pd.to_datetime(
                        verified_sessions['timestamp'], unit='s', errors='coerce'
                    )
                    
                    # Match timestamps (within 1 second window)
                    mask = owner_data['timestamp_dt'].apply(
                        lambda x: any(abs((x - ts).total_seconds()) < 1.0 for ts in verified_timestamps)
                    )
                    owner_data = owner_data[mask].copy()
                    owner_data = owner_data.drop('timestamp_dt', axis=1)
            
            if len(owner_data) < self.update_threshold:
                return {
                    'status': 'skipped',
                    'reason': f'Insufficient verified owner samples: {len(owner_data)} < {self.update_threshold}'
                }
            
            # Temporarily update features file with only owner data
            original_file = self.trainer.features_file
            temp_file = original_file + '.temp_owner_only'
            owner_data.to_csv(temp_file, index=False)
            
            # Update trainer to use temp file
            self.trainer.features_file = temp_file
            
            # Retrain with owner data only
            metrics = self.trainer.train()
            
            # Save updated model
            self.trainer.save_model()
            
            # Restore original file path
            self.trainer.features_file = original_file
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            # Update timestamp
            self._save_last_update_time()
            
            # Clear verified sessions (already used)
            if os.path.exists(self.verified_owner_sessions_file):
                os.remove(self.verified_owner_sessions_file)
            
            return {
                'status': 'success',
                'metrics': metrics,
                'update_time': datetime.now().isoformat(),
                'samples_used': len(owner_data)
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def force_update(self) -> Dict:
        """Force immediate model update (use with caution)."""
        self.last_update_time = None  # Reset to allow update
        return self.incremental_update(use_only_verified=True)
    
    def get_update_info(self) -> Dict:
        """Get information about last update and available data."""
        info = {
            'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
            'update_threshold': self.update_threshold,
            'min_update_interval_seconds': self.min_update_interval,
            'min_confidence_for_training': self.min_confidence_for_training,
            'should_update': self.should_update()
        }
        
        if os.path.exists(self.trainer.features_file):
            try:
                df = pd.read_csv(self.trainer.features_file)
                
                # Count owner vs non-owner data
                owner_data = df[df['label'] == 1]
                non_owner_data = df[df['label'] == 0]
                
                info['total_samples'] = len(df)
                info['owner_samples'] = len(owner_data)
                info['non_owner_samples'] = len(non_owner_data)
                
                if self.last_update_time:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
                    new_owner = owner_data[
                        pd.to_datetime(owner_data['timestamp'], unit='s', errors='coerce') > self.last_update_time
                    ]
                    info['new_owner_samples_since_update'] = len(new_owner)
                else:
                    info['new_owner_samples_since_update'] = len(owner_data)
                
                # Count verified sessions
                verified = self.get_verified_owner_sessions()
                info['verified_owner_sessions'] = len(verified)
            except:
                pass
        
        return info


def main():
    """Main adaptive update function."""
    updater = AdaptiveModelUpdater()
    
    print("="*60)
    print("STEP 8: Adaptive Model Update")
    print("="*60)
    print("\nCRITICAL: Model updates ONLY on owner behavior")
    print("Attacker behavior is NEVER used for training.\n")
    
    info = updater.get_update_info()
    
    print("Update Information:")
    print("="*60)
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    if updater.should_update():
        print("\n" + "="*60)
        print("Performing model update...")
        print("="*60)
        result = updater.incremental_update(use_only_verified=True)
        print(f"\nUpdate Result: {result['status']}")
        if result['status'] == 'success':
            print(f"Metrics: {result.get('metrics', {})}")
            print(f"Samples used: {result.get('samples_used', 0)}")
        elif result['status'] == 'error':
            print(f"Error: {result.get('error', 'Unknown error')}")
    else:
        print("\n" + "="*60)
        print("No update needed at this time.")
        print("="*60)


if __name__ == "__main__":
    main()
