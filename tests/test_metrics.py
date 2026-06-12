"""
test_stream_metrics.py
----------------------
Tests for streaming and rolling-window metrics in numcompute_stream.metrics

Run with:
    pytest tests/test_stream_metrics.py -v
"""

import numpy as np
import pytest

from numcompute_stream.metrics import (
    StreamAccuracy,
    StreamPrecisionRecallF1,
    StreamConfusionMatrix,
    StreamROCAUC
)
from numcompute.metrics import (
    accuracy_score,
    precision_recall_f1,
    confusion_matrix,
    roc_auc_score
)

class TestStreamMetrics:
    
    def test_stream_accuracy_global(self):
        y_true = np.array([1, 0, 1, 1, 0, 0, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0, 1])
        
        acc = StreamAccuracy()
        acc.update(y_true[:3], y_pred[:3])
        acc.update(y_true[3:], y_pred[3:])
        
        expected = accuracy_score(y_true, y_pred)
        assert acc.result() == pytest.approx(expected)

    def test_stream_accuracy_rolling(self):
        y_true = np.array([1, 0, 1, 1, 0, 0, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0, 1])
        
        # Window size 4 should only evaluate the last 4 elements: [1, 0, 0, 1] vs [0, 0, 0, 1]
        acc = StreamAccuracy(window_size=4)
        acc.update(y_true[:3], y_pred[:3])
        acc.update(y_true[3:], y_pred[3:])
        
        expected = accuracy_score(y_true[-4:], y_pred[-4:])
        assert acc.result() == pytest.approx(expected)

    def test_stream_precision_recall_f1(self):
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 0, 1, 1, 1, 0, 1])
        
        prf = StreamPrecisionRecallF1()
        prf.update(y_true[:4], y_pred[:4])
        prf.update(y_true[4:], y_pred[4:])
        
        p, r, f = prf.result()
        exp_p, exp_r, exp_f = precision_recall_f1(y_true, y_pred)
        
        assert p == pytest.approx(exp_p)
        assert r == pytest.approx(exp_r)
        assert f == pytest.approx(exp_f)

    def test_stream_confusion_matrix(self):
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 2, 0, 1, 1, 0, 1, 2])
        
        cm = StreamConfusionMatrix(num_classes=3)
        cm.update(y_true[:4], y_pred[:4])
        cm.update(y_true[4:], y_pred[4:])
        
        expected = confusion_matrix(y_true, y_pred, num_classes=3)
        np.testing.assert_array_equal(cm.result(), expected)

    def test_stream_roc_auc_global(self):
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        
        auc = StreamROCAUC()
        auc.update(y_true[:2], y_scores[:2])
        auc.update(y_true[2:], y_scores[2:])
        
        expected = roc_auc_score(y_true, y_scores)
        assert auc.result() == pytest.approx(expected)

    def test_roc_auc_undefined_single_class(self):
        # Ensure AUC safely returns NaN if there's only 1 class in the window
        auc = StreamROCAUC(window_size=2)
        auc.update(np.array([1, 1]), np.array([0.8, 0.9]))
        
        assert np.isnan(auc.result())
        
    def test_reset_behavior(self):
        acc = StreamAccuracy()
        acc.update([1, 0], [1, 0])
        acc.reset()
        assert acc.result() == 0.0