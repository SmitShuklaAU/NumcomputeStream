"""
test_stream_ensemble.py
-----------------------
Tests for the Streaming Random Forest Classifier.

Run with:
    pytest tests/test_stream_ensemble.py -v
"""

import numpy as np
import pytest
from numcompute_stream.ensemble import StreamRandomForestClassifier

class TestStreamRandomForest:

    def test_initialization_and_fit(self):
        """Test that the ensemble initializes the correct number of trees."""
        rf = StreamRandomForestClassifier(n_estimators=5, random_state=42)
        X = np.array([[1.0, 2.0], [2.0, 3.0]])
        y = np.array([0, 1])
        
        rf.partial_fit(X, y)
        
        assert len(rf.estimators_) == 5
        # Ensure all sub-trees are fitted (have roots)
        for tree in rf.estimators_:
            assert tree.root_ is not None

    def test_predict_majority_vote(self):
        """Test pure NumPy majority voting mechanism directly."""
        rf = StreamRandomForestClassifier(n_estimators=3)
        # Mock tree predictions manually
        # 3 estimators, 4 samples
        mock_preds = np.array([
            [0, 1, 1, 0],  # Tree 1
            [0, 0, 1, 0],  # Tree 2
            [1, 1, 1, 0]   # Tree 3
        ])
        
        votes = rf._majority_vote(mock_preds)
        # Expected:
        # Sample 0: [0, 0, 1] -> 0
        # Sample 1: [1, 0, 1] -> 1
        # Sample 2: [1, 1, 1] -> 1
        # Sample 3: [0, 0, 0] -> 0
        np.testing.assert_array_equal(votes, [0, 1, 1, 0])

    def test_incremental_learning_capability(self):
        """Ensure the ensemble can absorb streaming chunks sequentially."""
        rf = StreamRandomForestClassifier(n_estimators=3, min_samples_split=2, random_state=1)
        
        # Chunk 1 (Mostly class 0)
        X1 = np.array([[0.1], [0.2], [0.3]])
        y1 = np.array([0, 0, 0])
        rf.partial_fit(X1, y1)
        
        # Chunk 2 (Mostly class 1)
        X2 = np.array([[9.7], [9.8], [9.9]])
        y2 = np.array([1, 1, 1])
        rf.partial_fit(X2, y2)
        
        # Test extreme points
        X_test = np.array([[0.15], [9.85]])
        preds = rf.predict(X_test)
        
        np.testing.assert_array_equal(preds, [0, 1])

    def test_predict_before_fit_raises(self):
        """Safety catch if model predicts on uninitialized state."""
        rf = StreamRandomForestClassifier()
        with pytest.raises(RuntimeError, match="not fitted"):
            rf.predict(np.array([[1.0, 2.0]]))

    def test_empty_chunk_ignored(self):
        """Ensure empty chunks are bypassed smoothly without throwing dimension errors."""
        rf = StreamRandomForestClassifier(n_estimators=2)
        rf.partial_fit(np.array([[1.0, 2.0]]), np.array([1]))
        rf.partial_fit(np.array([[]]).reshape(0, 2), np.array([]))
        
        assert len(rf.estimators_) == 2