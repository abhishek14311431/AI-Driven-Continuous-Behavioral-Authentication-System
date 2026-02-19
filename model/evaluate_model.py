"""
STEP 4: Model Evaluation
========================

Evaluate model performance: accuracy, FAR (False Acceptance Rate), FRR (False Rejection Rate).

For binary classification: Owner (1) vs Not-Owner (0)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, Optional
import os

from .train_model import BehaviorModelTrainer


class ModelEvaluator:
    """
    Evaluate model performance metrics.
    
    This is STEP 4 - model evaluation for binary classification.
    """
    
    def __init__(self,
                 model_file: str = "model/behavior_model.pkl",
                 scaler_file: str = "model/feature_scaler.pkl",
                 features_file: str = "data/processed/behavior_features.csv"):
        """
        Initialize model evaluator.
        
        Args:
            model_file: Path to trained model
            scaler_file: Path to feature scaler
            features_file: Path to features file
        """
        self.trainer = BehaviorModelTrainer(
            features_file=features_file,
            model_file=model_file,
            scaler_file=scaler_file
        )
        self.trainer.load_model()
    
    def evaluate(self, test_data: Optional[pd.DataFrame] = None, label_column: str = 'label') -> Dict:
        """
        Evaluate model on test data.
        
        Binary classification:
        - Class 1: Owner (authorized)
        - Class 0: Not-owner (attacker)
        
        Args:
            test_data: Test dataset (if None, uses train_test_split from training data)
            label_column: Name of label column (default: 'label')
        
        Returns:
            Dictionary with evaluation metrics
        """
        if test_data is None:
            # Load and split data (only owner data for training)
            X, y = self.trainer.load_training_data()
            from sklearn.model_selection import train_test_split
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=y if len(y.unique()) > 1 else None
            )
        else:
            # Use provided test data
            exclude_cols = ['timestamp', 'window_size', label_column]
            feature_cols = [col for col in test_data.columns if col not in exclude_cols]
            
            if self.trainer.feature_columns:
                feature_cols = [col for col in self.trainer.feature_columns if col in test_data.columns]
            
            X_test = test_data[feature_cols].fillna(0)
            y_test = test_data[label_column] if label_column in test_data.columns else None
        
        # Scale features
        X_test_scaled = self.trainer.scaler.transform(X_test)
        
        # Predict
        y_pred = self.trainer.model.predict(X_test_scaled)
        y_pred_proba = self.trainer.model.predict_proba(X_test_scaled)
        
        if y_test is None:
            return {'error': 'No labels available for evaluation'}
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # FAR and FRR for binary classification
        # Class 1 = Owner (legitimate), Class 0 = Not-owner (impostor)
        if len(np.unique(y_test)) == 2:
            tn, fp, fn, tp = cm.ravel()
            
            # False Acceptance Rate: Impostor accepted as owner
            far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            # False Rejection Rate: Owner rejected as impostor
            frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            
            # Additional metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            # Single class (only owner data) - can't calculate FAR/FRR
            far = 0.0
            frr = 0.0
            precision = accuracy
            recall = accuracy
            f1_score = accuracy
        
        metrics = {
            'accuracy': float(accuracy),
            'far': float(far),  # False Acceptance Rate (impostor accepted)
            'frr': float(frr),  # False Rejection Rate (owner rejected)
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return metrics
    
    def print_evaluation(self, metrics: Dict):
        """Print evaluation metrics in readable format."""
        print("\n" + "="*60)
        print("MODEL EVALUATION METRICS")
        print("="*60)
        print(f"\nAccuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"False Acceptance Rate (FAR): {metrics['far']:.4f} ({metrics['far']*100:.2f}%)")
        print(f"  → Impostor accepted as owner (SECURITY RISK)")
        print(f"False Rejection Rate (FRR): {metrics['frr']:.4f} ({metrics['frr']*100:.2f}%)")
        print(f"  → Owner rejected as impostor (INCONVENIENCE)")
        print(f"\nPrecision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        
        print("\nConfusion Matrix:")
        print("  (Rows = Actual, Columns = Predicted)")
        print("  [Owner, Not-Owner]")
        cm = np.array(metrics['confusion_matrix'])
        print(f"  Owner:      {cm[0] if len(cm) > 0 else 'N/A'}")
        if len(cm) > 1:
            print(f"  Not-Owner:  {cm[1]}")
        
        print("\nClassification Report:")
        report = metrics['classification_report']
        if isinstance(report, dict):
            print(f"  Precision: {report.get('weighted avg', {}).get('precision', 'N/A')}")
            print(f"  Recall: {report.get('weighted avg', {}).get('recall', 'N/A')}")
            print(f"  F1-Score: {report.get('weighted avg', {}).get('f1-score', 'N/A')}")
        print("="*60 + "\n")


def main():
    """Main evaluation function."""
    evaluator = ModelEvaluator()
    
    print("="*60)
    print("STEP 4: Evaluating Behavior Authentication Model")
    print("="*60)
    print("\nBinary Classification: Owner (1) vs Not-Owner (0)\n")
    
    try:
        metrics = evaluator.evaluate()
        
        if 'error' in metrics:
            print(f"Error: {metrics['error']}")
            return
        
        evaluator.print_evaluation(metrics)
    except Exception as e:
        print(f"Evaluation error: {e}")
        raise


if __name__ == "__main__":
    main()
