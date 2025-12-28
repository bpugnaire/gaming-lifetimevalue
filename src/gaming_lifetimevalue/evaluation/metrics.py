import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from typing import Dict, Any


def evaluate_classifier(y_true, y_pred, target_names=None) -> Dict[str, Any]:
    """Evaluate classifier performance with accuracy and F1 score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional list of class names for the classification report
        
    Returns:
        Dictionary with accuracy, classification_report, and f1_weighted
    """
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True
    )

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "f1_weighted": report["weighted avg"]["f1-score"],
    }


def evaluate_regressor(y_true, y_pred) -> Dict[str, float]:
    """Evaluate regressor performance with MAE, RMSE, and R² metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with mae, rmse, r2, mean_actual, and mean_predicted
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mean_actual": np.mean(y_true),
        "mean_predicted": np.mean(y_pred),
    }


