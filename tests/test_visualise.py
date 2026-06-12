"""
test_visualise.py
-----------------
Tests for the visualisation module in numcompute_stream.

Run with:
    pytest tests/test_visualise.py -v
"""

import os
import numpy as np
import pytest

# Ensure matplotlib does not try to open GUI windows during testing
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from numcompute_stream.visualise import (
    plot_metric_over_time,
    compare_models,
    plot_predictions_vs_ground_truth
)

class TestVisualise:

    def test_plot_metric_over_time_saves_file(self, tmp_path):
        metrics = [0.5, 0.6, 0.75, 0.82, 0.88]
        save_file = tmp_path / "metric_plot.png"
        
        plot_metric_over_time(
            metrics, 
            title="Test Accuracy", 
            ylabel="Accuracy", 
            save_path=str(save_file)
        )
        
        # Verify the file was created and is not empty
        assert save_file.exists()
        assert save_file.stat().st_size > 0

    def test_compare_models_saves_file(self, tmp_path):
        m1 = np.array([0.1, 0.2, 0.3])
        m2 = np.array([0.15, 0.25, 0.35, 0.4]) # Deliberately different length
        
        save_file = tmp_path / "comparison_plot.png"
        
        compare_models(
            m1, m2, 
            labels=("Tree", "Forest"), 
            save_path=str(save_file)
        )
        
        assert save_file.exists()
        assert save_file.stat().st_size > 0

    def test_plot_predictions_saves_file(self, tmp_path):
        y_true = [0, 1, 0, 1, 1]
        y_pred = [0, 1, 1, 1, 0]
        
        save_file = tmp_path / "predictions_plot.png"
        
        plot_predictions_vs_ground_truth(
            y_true, y_pred, 
            save_path=str(save_file)
        )
        
        assert save_file.exists()
        assert save_file.stat().st_size > 0

    def test_plot_predictions_shape_mismatch(self):
        y_true = [0, 1, 0]
        y_pred = [0, 1] # Mismatch
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            plot_predictions_vs_ground_truth(y_true, y_pred)