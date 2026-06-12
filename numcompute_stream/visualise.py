"""
visualise.py
------------
Built-in plotting utilities for numcompute_stream.

Provides lightweight, reusable matplotlib functions to visualise streaming 
metrics, compare model performance, and inspect chunk predictions.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Tuple, Optional

__all__ = [
    "plot_metric_over_time",
    "compare_models",
    "plot_predictions_vs_ground_truth"
]

def plot_metric_over_time(
    metric_values: Union[List[float], np.ndarray], 
    title: str = "Metric Over Time", 
    ylabel: str = "Value",
    save_path: Optional[str] = None
) -> None:
    """
    Plots a single metric (e.g., accuracy, error) tracked across data chunks.

    Parameters
    ----------
    metric_values : list or np.ndarray
        A sequence of metric values recorded over time.
    title : str, default="Metric Over Time"
        The title of the plot.
    ylabel : str, default="Value"
        The label for the Y-axis.
    save_path : str, optional
        If provided, saves the plot to this filepath instead of showing it.
    """
    values = np.asarray(metric_values)
    chunks = np.arange(1, len(values) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(chunks, values, marker='o', linestyle='-', color='#2c3e50', linewidth=2, markersize=5)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Chunk Index", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compare_models(
    metric1: Union[List[float], np.ndarray], 
    metric2: Union[List[float], np.ndarray], 
    labels: Tuple[str, str] = ("Model 1", "Model 2"),
    title: str = "Model Comparison Over Time",
    ylabel: str = "Metric Value",
    save_path: Optional[str] = None
) -> None:
    """
    Compares the streaming metric trajectories of two different models.

    Parameters
    ----------
    metric1 : list or np.ndarray
        Sequence of metrics for the first model.
    metric2 : list or np.ndarray
        Sequence of metrics for the second model.
    labels : tuple of str, default=("Model 1", "Model 2")
        Legend labels for the two models.
    title : str, default="Model Comparison Over Time"
        The title of the plot.
    ylabel : str, default="Metric Value"
        The label for the Y-axis.
    save_path : str, optional
        If provided, saves the plot to this filepath instead of showing it.
    """
    m1 = np.asarray(metric1)
    m2 = np.asarray(metric2)
    
    # Pad the shorter array with NaNs if lengths don't match, to prevent plotting errors
    max_len = max(len(m1), len(m2))
    if len(m1) < max_len:
        m1 = np.pad(m1, (0, max_len - len(m1)), constant_values=np.nan)
    if len(m2) < max_len:
        m2 = np.pad(m2, (0, max_len - len(m2)), constant_values=np.nan)

    chunks = np.arange(1, max_len + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(chunks, m1, marker='o', linestyle='-', label=labels[0], color='#2980b9', linewidth=2)
    plt.plot(chunks, m2, marker='s', linestyle='--', label=labels[1], color='#e74c3c', linewidth=2)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Chunk Index", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_predictions_vs_ground_truth(
    y_true: Union[List[float], np.ndarray], 
    y_pred: Union[List[float], np.ndarray],
    title: str = "Predictions vs Ground Truth (Latest Chunk)",
    save_path: Optional[str] = None
) -> None:
    """
    Visualises model predictions against actual labels for a given chunk.
    Uses a scatter comparison, ideal for both classification and regression.

    Parameters
    ----------
    y_true : list or np.ndarray
        Actual ground truth labels.
    y_pred : list or np.ndarray
        Predicted labels from the model.
    title : str, default="Predictions vs Ground Truth (Latest Chunk)"
        The title of the plot.
    save_path : str, optional
        If provided, saves the plot to this filepath instead of showing it.
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    
    if len(yt) != len(yp):
        raise ValueError(f"Shape mismatch: y_true ({len(yt)}) and y_pred ({len(yp)}) must have the same length.")

    indices = np.arange(len(yt))

    plt.figure(figsize=(10, 4))
    
    # Plot true values as a background line/dot
    plt.plot(indices, yt, marker='o', linestyle='-', color='#7f8c8d', alpha=0.6, label="Ground Truth")
    
    # Plot predictions as distinct markers on top
    plt.scatter(indices, yp, color='#e67e22', zorder=5, label="Prediction")
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Sample Index in Chunk", fontsize=12)
    plt.ylabel("Target Value / Class", fontsize=12)
    plt.legend(loc="best", fontsize=11)
    
    # Only show grid on the y-axis for cleaner categorical viewing
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()