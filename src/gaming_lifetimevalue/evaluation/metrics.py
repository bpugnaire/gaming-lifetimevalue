import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix
)
import plotly.graph_objects as go
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

def plot_confusion_matrix(y_test, y_pred, target_map):
    labels = list(target_map.keys())
    cm = confusion_matrix(y_test, y_pred)

    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = go.Figure(data=go.Heatmap(
        z=cm_perc,
        x=labels,
        y=labels,
        text=np.around(cm_perc, 2),
        texttemplate="%{text}",
        colorscale='Viridis',
        hoverinfo='z'
    ))

    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Cohort',
        yaxis_title='Actual Cohort',
        width=600,
        height=600
    )

    return fig