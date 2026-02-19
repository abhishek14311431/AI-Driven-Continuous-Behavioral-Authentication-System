import numpy as np

class AnomalyDetectionWrapper:
    """
    Wrapper for IsolationForest to provide a consistent interface for prediction
    and probability scoring. This class must be defined in its own module
    to avoid pickle loading issues across different scripts.
    """
    
    def __init__(self, isolation_forest):
        self.model = isolation_forest
        self.classes_ = np.array([0, 1])  # 0 = not-owner (anomaly), 1 = owner (normal)
    
    def predict(self, X):
        """Predict class: 1 = owner (inlier), 0 = not-owner (outlier)."""
        raw_pred = self.model.predict(X)
        # IsolationForest: 1 = inlier (owner), -1 = outlier (not-owner)
        return np.where(raw_pred == 1, 1, 0)
    
    def predict_proba(self, X):
        """
        Convert anomaly scores to probability-like values.
        Higher score = more likely to be owner.
        """
        # Get anomaly scores (more negative = more anomalous)
        scores = self.model.decision_function(X)
        
        # Normalize scores to [0, 1] range using sigmoid-like transformation
        owner_prob = 1 / (1 + np.exp(-scores * 5))
        
        # Return [not_owner_prob, owner_prob] format
        not_owner_prob = 1 - owner_prob
        return np.column_stack([not_owner_prob, owner_prob])
    
    def score(self, X, y):
        """Calculate accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
