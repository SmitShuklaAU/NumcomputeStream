"""
test_stream_tree.py
-------------------
Tests for the Streaming Decision Tree Classifier.

Run with:
    pytest tests/test_stream_tree.py -v
"""

import numpy as np
import pytest
from numcompute_stream.tree import StreamDecisionTreeClassifier

class TestStreamDecisionTree:

    def test_single_chunk_fit_predict(self):
        """Test standard batch-like usage on a single chunk."""
        # Simple linearly separable data
        X = np.array([[1.0, 1.0], [1.5, 1.5], [8.0, 8.0], [9.0, 9.0]])
        y = np.array([0, 0, 1, 1])
        
        tree = StreamDecisionTreeClassifier(min_samples_split=2, random_state=42)
        tree.fit(X, y)
        
        preds = tree.predict(X)
        np.testing.assert_array_equal(preds, y)

    def test_incremental_growth(self):
        """Test tree growing over multiple chunks (buffer flushes)."""
        tree = StreamDecisionTreeClassifier(min_samples_split=4, random_state=42)
        
        # Chunk 1: Not enough data to split (size 3 < 4)
        X1 = np.array([[1.0], [2.0], [3.0]])
        y1 = np.array([0, 0, 0])
        tree.partial_fit(X1, y1)
        
        assert tree.root_.is_leaf is True
        
        # Chunk 2: Pushes buffer to size 6. Should trigger a split.
        X2 = np.array([[8.0], [9.0], [10.0]])
        y2 = np.array([1, 1, 1])
        tree.partial_fit(X2, y2)
        
        # Tree should have split, making root an internal node
        assert tree.root_.is_leaf is False
        assert tree.root_.left is not None
        assert tree.root_.right is not None
        
        # Verify predictions
        X_test = np.array([[2.5], [8.5]])
        preds = tree.predict(X_test)
        np.testing.assert_array_equal(preds, [0, 1])

    def test_max_depth_limit(self):
        """Ensure the tree respects the maximum depth constraint."""
        # Dataset alternating 0, 1, 0, 1 along a line requires multiple splits
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        
        # Depth 1 should only split once
        tree = StreamDecisionTreeClassifier(max_depth=1, min_samples_split=2)
        tree.fit(X, y)
        
        assert tree.root_.is_leaf is False
        # Children should be leaves because depth hit 1
        assert tree.root_.left.is_leaf is True
        assert tree.root_.right.is_leaf is True

    def test_pure_node_no_split(self):
        """Ensure pure nodes (all same class) don't attempt unnecessary splits."""
        X = np.random.rand(10, 2)
        y = np.zeros(10) # All class 0
        
        tree = StreamDecisionTreeClassifier(min_samples_split=2)
        tree.fit(X, y)
        
        # Despite passing min_samples_split, it shouldn't split a pure node
        assert tree.root_.is_leaf is True
        assert tree.root_.majority_class == 0

    def test_max_features_subset(self):
        """Test the max_features constraint logic."""
        X = np.random.rand(20, 10)
        y = np.random.randint(0, 2, 20)
        
        # Test string param
        tree_sqrt = StreamDecisionTreeClassifier(max_features="sqrt", min_samples_split=5)
        tree_sqrt.fit(X, y)
        assert tree_sqrt.root_ is not None
        
        # Test float param
        tree_float = StreamDecisionTreeClassifier(max_features=0.5, min_samples_split=5)
        tree_float.fit(X, y)
        assert tree_float.root_ is not None

    def test_entropy_criterion(self):
        """Test tree with entropy criterion."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 1, 1])
        
        tree = StreamDecisionTreeClassifier(criterion="entropy", min_samples_split=2)
        tree.fit(X, y)
        
        preds = tree.predict(np.array([[1.5], [3.5]]))
        np.testing.assert_array_equal(preds, [0, 1])