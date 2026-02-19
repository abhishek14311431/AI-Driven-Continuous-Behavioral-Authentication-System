
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
import sys
from pathlib import Path
from typing import Tuple, Optional

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from features.feature_fusion import FeatureFusion
from model.wrapper import AnomalyDetectionWrapper


# AnomalyDetectionWrapper moved to model/wrapper.py


class BehaviorModelTrainer:
    """
    Train Isolation Forest model for behavioral authentication.
    
    This is STEP 4 - anomaly detection training.
    Model learns ONLY from owner behavior (label=1).
    Non-owners are detected as ANOMALIES.
    """
    
    def __init__(self, 
                 features_file: str = "data/processed/behavior_features.csv",
                 model_file: str = "model/behavior_model.pkl",
                 scaler_file: str = "model/feature_scaler.pkl"):
        """
        Initialize model trainer.
        
        Args:
            features_file: Path to processed features CSV
            model_file: Path to save trained model
            scaler_file: Path to save feature scaler
        """
        self.features_file = features_file
        self.model_file = model_file
        self.scaler_file = scaler_file
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_file), exist_ok=True)
    
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load training data from features file.
        
        CRITICAL: Only uses data with label=1 (owner behavior).
        Data with label=0 (attacker) is IGNORED during training.
        
        Returns:
            Tuple of (features, labels)
        """
        if not os.path.exists(self.features_file):
            raise FileNotFoundError(f"Features file not found: {self.features_file}")
        
        df = pd.read_csv(self.features_file)
        
        # CRITICAL: Filter to ONLY owner data (label=1)
        # We NEVER train on attacker behavior
        owner_data = df[df['label'] == 1].copy()
        
        if len(owner_data) == 0:
            raise ValueError(
                "No owner data (label=1) found in features file. "
                "Model must be trained ONLY on owner behavior."
            )
        
        print(f"Training on {len(owner_data)} owner behavior samples")
        print(f"Ignoring {len(df) - len(owner_data)} non-owner samples (not used in training)")
        
        # Get feature columns (exclude metadata)
        exclude_cols = ['timestamp', 'window_size', 'label']
        fusion = FeatureFusion()
        self.feature_columns = fusion.get_feature_columns()
        
        # Ensure all feature columns exist
        available_cols = [col for col in self.feature_columns if col in owner_data.columns]
        if len(available_cols) < len(self.feature_columns) * 0.8:  # At least 80% of features
            raise ValueError(f"Missing required feature columns. Expected: {self.feature_columns}")
        
        # Update feature_columns to only available ones
        self.feature_columns = available_cols
        
        # Extract features and labels
        X = owner_data[available_cols].fillna(0)
        y = owner_data['label']  # Should all be 1 (owner)
        
        # Verify all labels are 1
        if not (y == 1).all():
            raise ValueError("CRITICAL ERROR: Non-owner data found in training set!")
        
        return X, y
    
    def train(self, 
              n_estimators: int = 200,
              contamination: float = 0.03,
              random_state: int = 42,
              test_size: float = 0.2) -> dict:
        """
        Train the Isolation Forest model for ANOMALY DETECTION.
        
        Model learns owner behavior patterns and detects anomalies (non-owners).
        This is a ONE-CLASS learning approach - perfect for security!
        
        Args:
            n_estimators: Number of trees in Isolation Forest
            contamination: Expected proportion of anomalies (keep low - we expect few intruders)
            random_state: Random seed for reproducibility
            test_size: Fraction of data for testing
        
        Returns:
            Dictionary with training metrics
        """
        # Load data (only owner data)
        X, y = self.load_training_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Isolation Forest for ANOMALY DETECTION
        # This learns what "normal" (owner) behavior looks like
        # Anything that deviates is flagged as anomaly (not-owner)
        print("Training Isolation Forest for anomaly detection...")
        print(f"  - n_estimators: {n_estimators}")
        print(f"  - contamination: {contamination} (expected anomaly rate)")
        
        isolation_forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
            max_samples='auto',
            bootstrap=True
        )
        
        isolation_forest.fit(X_train_scaled)
        
        # Wrap in our custom class for compatibility
        self.model = AnomalyDetectionWrapper(isolation_forest)
        
        # Evaluate on owner data (should be recognized as owners)
        train_predictions = self.model.predict(X_train_scaled)
        test_predictions = self.model.predict(X_test_scaled)
        
        train_accuracy = np.mean(train_predictions == 1)  # Should all be 1 (owner)
        test_accuracy = np.mean(test_predictions == 1)
        
        # Get anomaly scores for insight
        train_scores = isolation_forest.decision_function(X_train_scaled)
        test_scores = isolation_forest.decision_function(X_test_scaled)
        
        metrics = {
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'n_samples': len(X),
            'n_features': len(self.feature_columns),
            'contamination': contamination,
            'avg_train_score': float(np.mean(train_scores)),
            'avg_test_score': float(np.mean(test_scores)),
            'min_score_threshold': float(np.percentile(train_scores, 5)),
            'model_type': 'IsolationForest'
        }
        
        return metrics
    
    def save_model(self):
        """Save trained model and scaler."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Save model
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler
        with open(self.scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature columns for inference
        feature_info_file = self.model_file.replace('.pkl', '_features.txt')
        with open(feature_info_file, 'w') as f:
            f.write('\n'.join(self.feature_columns))
        
        print(f"[OK] Model saved to {self.model_file}")
        print(f"[OK] Scaler saved to {self.scaler_file}")
        print(f"[OK] Feature columns saved to {feature_info_file}")
    
    def load_model(self):
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
            fusion = FeatureFusion()
            self.feature_columns = fusion.get_feature_columns()
        
        print("[OK] Model and scaler loaded successfully.")


def main():
    """Main training function."""
    trainer = BehaviorModelTrainer()
    
    print("="*60)
    print("STEP 4: Training Behavior Authentication Model")
    print("="*60)
    print("\nCRITICAL: Model trains ONLY on owner behavior (label=1)")
    print("Attacker behavior (label=0) is NEVER used in training.\n")
    
    try:
        metrics = trainer.train()
        
        print("\n" + "="*60)
        print("Training Metrics:")
        print("="*60)
        for key, value in metrics.items():
            if key != 'feature_importance':
                print(f"  {key}: {value}")
        
        print("\nTop 5 Most Important Features:")
        if 'feature_importance' in metrics:
            sorted_features = sorted(
                metrics['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for feature, importance in sorted_features:
                print(f"  {feature}: {importance:.4f}")
        
        trainer.save_model()
        print("\n" + "="*60)
        print("[OK] Training complete!")
        print("="*60)
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
