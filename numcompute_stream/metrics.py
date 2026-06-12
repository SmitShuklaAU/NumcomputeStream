"""
metrics.py
----------
Streaming classification and regression metrics for numcompute_stream.

Extends the 2.1 static metric functions with chunk-based streaming
evaluators. Each streaming metric supports `.update()`, `.result()`, 
and `.reset()` APIs, alongside optional rolling-window tracking.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, List

# Re-export static functions from Assignment 2.1
from numcompute.metrics import (
    mean_squared_error,
    accuracy_score,
    precision_recall_f1,
    confusion_matrix,
    roc_auc_score,
)

__all__ = [
    "mean_squared_error", "accuracy_score", "precision_recall_f1", 
    "confusion_matrix", "roc_auc_score",
    "StreamAccuracy", "StreamPrecisionRecallF1", 
    "StreamConfusionMatrix", "StreamROCAUC"
]

class _CyclicBuffer:
    """Internal helper to maintain a fast rolling window of recent samples."""
    def __init__(self, window_size: int, dtype: type = float):
        self.window_size = window_size
        self.buffer = np.zeros(window_size, dtype=dtype)
        self.idx = 0
        self.count = 0

    def update(self, data: np.ndarray) -> None:
        data = np.asarray(data).ravel()
        n = len(data)
        if n == 0:
            return
            
        if n >= self.window_size:
            self.buffer[:] = data[-self.window_size:]
            self.count = self.window_size
            self.idx = 0
        else:
            end_idx = self.idx + n
            if end_idx <= self.window_size:
                self.buffer[self.idx:end_idx] = data
                self.idx = end_idx % self.window_size
            else:
                overflow = end_idx - self.window_size
                first_part = n - overflow
                self.buffer[self.idx:] = data[:first_part]
                self.buffer[:overflow] = data[first_part:]
                self.idx = overflow
            self.count = min(self.window_size, self.count + n)

    def get_valid(self) -> np.ndarray:
        """Returns the valid data. (Order is arbitrary but perfectly aligned across buffers)."""
        if self.count < self.window_size:
            return self.buffer[:self.count]
        return self.buffer


class StreamAccuracy:
    """
    Streaming accuracy metric.
    Tracks global accuracy or rolling-window accuracy over recent samples.
    """
    def __init__(self, window_size: Optional[int] = None):
        self.window_size = window_size
        self.reset()

    def reset(self) -> None:
        self.total_correct = 0
        self.total_samples = 0
        if self.window_size is not None:
            self.y_true_buf = _CyclicBuffer(self.window_size, dtype=int)
            self.y_pred_buf = _CyclicBuffer(self.window_size, dtype=int)

    def update(self, y_true_chunk: np.ndarray, y_pred_chunk: np.ndarray) -> "StreamAccuracy":
        yt = np.asarray(y_true_chunk).ravel()
        yp = np.asarray(y_pred_chunk).ravel()

        if self.window_size is None:
            self.total_correct += np.sum(yt == yp)
            self.total_samples += len(yt)
        else:
            self.y_true_buf.update(yt)
            self.y_pred_buf.update(yp)
        return self

    def result(self) -> float:
        if self.window_size is None:
            if self.total_samples == 0:
                return 0.0
            return float(self.total_correct / self.total_samples)
        else:
            yt = self.y_true_buf.get_valid()
            yp = self.y_pred_buf.get_valid()
            if len(yt) == 0:
                return 0.0
            return float(np.mean(yt == yp))


class StreamPrecisionRecallF1:
    """
    Streaming precision, recall, and F1-score for binary classification.
    """
    def __init__(self, pos_label: int = 1, window_size: Optional[int] = None):
        self.pos_label = pos_label
        self.window_size = window_size
        self.reset()

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        if self.window_size is not None:
            self.y_true_buf = _CyclicBuffer(self.window_size, dtype=int)
            self.y_pred_buf = _CyclicBuffer(self.window_size, dtype=int)

    def update(self, y_true_chunk: np.ndarray, y_pred_chunk: np.ndarray) -> "StreamPrecisionRecallF1":
        yt = np.asarray(y_true_chunk).ravel()
        yp = np.asarray(y_pred_chunk).ravel()

        if self.window_size is None:
            self.tp += np.sum((yp == self.pos_label) & (yt == self.pos_label))
            self.fp += np.sum((yp == self.pos_label) & (yt != self.pos_label))
            self.fn += np.sum((yp != self.pos_label) & (yt == self.pos_label))
        else:
            self.y_true_buf.update(yt)
            self.y_pred_buf.update(yp)
        return self

    def result(self) -> Tuple[float, float, float]:
        if self.window_size is None:
            tp, fp, fn = self.tp, self.fp, self.fn
        else:
            yt = self.y_true_buf.get_valid()
            yp = self.y_pred_buf.get_valid()
            if len(yt) == 0:
                return 0.0, 0.0, 0.0
            tp = np.sum((yp == self.pos_label) & (yt == self.pos_label))
            fp = np.sum((yp == self.pos_label) & (yt != self.pos_label))
            fn = np.sum((yp != self.pos_label) & (yt == self.pos_label))

        eps = np.finfo(float).eps
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        return float(precision), float(recall), float(f1)


class StreamConfusionMatrix:
    """
    Streaming confusion matrix aggregator.
    Accumulates counts exactly over time or tracks a rolling subset.
    """
    def __init__(self, num_classes: int, window_size: Optional[int] = None):
        self.num_classes = num_classes
        self.window_size = window_size
        self.reset()

    def reset(self) -> None:
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        if self.window_size is not None:
            self.y_true_buf = _CyclicBuffer(self.window_size, dtype=int)
            self.y_pred_buf = _CyclicBuffer(self.window_size, dtype=int)

    def update(self, y_true_chunk: np.ndarray, y_pred_chunk: np.ndarray) -> "StreamConfusionMatrix":
        yt = np.asarray(y_true_chunk, dtype=int).ravel()
        yp = np.asarray(y_pred_chunk, dtype=int).ravel()

        if self.window_size is None:
            self.matrix += confusion_matrix(yt, yp, self.num_classes)
        else:
            self.y_true_buf.update(yt)
            self.y_pred_buf.update(yp)
        return self

    def result(self) -> np.ndarray:
        if self.window_size is None:
            return self.matrix.copy()
        else:
            yt = self.y_true_buf.get_valid()
            yp = self.y_pred_buf.get_valid()
            if len(yt) == 0:
                return np.zeros((self.num_classes, self.num_classes), dtype=int)
            return confusion_matrix(yt, yp, self.num_classes)


class StreamROCAUC:
    """
    Streaming ROC AUC.
    ROC AUC is global by nature. This metric accumulates all chunk data
    efficiently or uses a rolling window.
    """
    def __init__(self, window_size: Optional[int] = None):
        self.window_size = window_size
        self.reset()

    def reset(self) -> None:
        self.y_true_hist: List[np.ndarray] = []
        self.y_score_hist: List[np.ndarray] = []
        if self.window_size is not None:
            self.y_true_buf = _CyclicBuffer(self.window_size, dtype=int)
            self.y_score_buf = _CyclicBuffer(self.window_size, dtype=float)

    def update(self, y_true_chunk: np.ndarray, y_score_chunk: np.ndarray) -> "StreamROCAUC":
        yt = np.asarray(y_true_chunk).ravel()
        ys = np.asarray(y_score_chunk).ravel()

        if self.window_size is None:
            self.y_true_hist.append(yt)
            self.y_score_hist.append(ys)
        else:
            self.y_true_buf.update(yt)
            self.y_score_buf.update(ys)
        return self

    def result(self) -> float:
        if self.window_size is None:
            if not self.y_true_hist:
                return np.nan
            yt = np.concatenate(self.y_true_hist)
            ys = np.concatenate(self.y_score_hist)
        else:
            yt = self.y_true_buf.get_valid()
            ys = self.y_score_buf.get_valid()

        # AUC is mathematically undefined if only 1 class is present
        if len(yt) == 0 or len(np.unique(yt)) < 2:
            return np.nan
        return roc_auc_score(yt, ys)