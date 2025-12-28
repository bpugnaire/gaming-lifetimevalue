import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
)
from typing import Dict, Any


def evaluate_classifier(y_true, y_pred, target_names=None) -> Dict[str, Any]:
    """Evaluate classification model"""
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
    """Evaluate regression model"""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def compute_cohort_metrics(df_with_predictions, cohort_col="cohort"):
    """Compute metrics per cohort"""
    cohort_metrics = {}

    for cohort_name, group in df_with_predictions.group_by(cohort_col):
        y_true = group["d120_rev"].to_numpy()
        y_pred = group["predicted_d120_rev"].to_numpy()

        cohort_metrics[cohort_name] = evaluate_regressor(y_true, y_pred)

    return cohort_metrics
