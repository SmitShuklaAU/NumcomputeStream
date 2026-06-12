"""
tree.py
-------
Streaming Decision Tree implementation for numcompute_stream.

Implements an online-growth Decision Tree Classifier. Instead of requiring 
all data in memory, leaves buffer incoming data. Once a leaf reaches 
`min_samples_split`, it computes the best split, converts to an internal 
node, and spawns new leaves dynamically.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Union
from numcompute_stream.pipeline import Estimator

__all__ = ["StreamDecisionTreeClassifier"]

def _gini(y: np.ndarray, classes: np.ndarray) -> float:
    """Computes Gini impurity of a label array."""
    if y.size == 0:
        return 0.0
    # bincount is faster, but we need to map to 0..C first if labels aren't strictly 0..C
    # For safety with subset arrays, we use broadcasting
    counts = np.sum(y[:, None] == classes[None, :], axis=0)
    probs = counts / y.size
    return float(1.0 - np.sum(probs ** 2))

def _entropy(y: np.ndarray, classes: np.ndarray) -> float:
    """Computes Shannon entropy of a label array."""
    if y.size == 0:
        return 0.0
    counts = np.sum(y[:, None] == classes[None, :], axis=0)
    probs = counts / y.size
    probs = probs[probs > 0] # Avoid log(0)
    return float(-np.sum(probs * np.log2(probs)))

class _TreeNode:
    """
    Internal recursive node for the streaming tree.
    """
    def __init__(self, depth: int, max_depth: int, min_samples_split: int, 
                 criterion: str, max_features: Union[str, float, None], 
                 random_state: Optional[int] = None):
        self.depth = depth
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        
        self.rng = np.random.default_rng(random_state)
        
        # Node state
        self.is_leaf = True
        self.feature_idx: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional[_TreeNode] = None
        self.right: Optional[_TreeNode] = None
        
        # Leaf buffers
        self.buffer_X: list[np.ndarray] = []
        self.buffer_y: list[np.ndarray] = []
        self.n_samples_seen: int = 0
        
        # For prediction
        self.class_counts: dict[float, int] = {}
        self.majority_class: float = 0.0

    def _update_class_counts(self, y: np.ndarray) -> None:
        vals, counts = np.unique(y, return_counts=True)
        for v, c in zip(vals, counts):
            self.class_counts[v] = self.class_counts.get(v, 0) + c
            
        # Update majority class
        if self.class_counts:
            self.majority_class = max(self.class_counts.keys(), key=lambda k: self.class_counts[k])

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        """Routes a chunk of data through this node."""
        if len(X) == 0:
            return
            
        self.n_samples_seen += len(X)
        self._update_class_counts(y)
        
        if not self.is_leaf:
            # Route down existing split
            left_mask = X[:, self.feature_idx] <= self.threshold
            right_mask = ~left_mask
            
            if np.any(left_mask):
                self.left.update(X[left_mask], y[left_mask])
            if np.any(right_mask):
                self.right.update(X[right_mask], y[right_mask])
            return

        # It's a leaf. Buffer the data.
        self.buffer_X.append(X)
        self.buffer_y.append(y)
        
        # Check if we should attempt a split
        current_buffer_size = sum(len(chunk) for chunk in self.buffer_y)
        
        if current_buffer_size >= self.min_samples_split and self.depth < self.max_depth:
            self._attempt_split()

    def _attempt_split(self) -> None:
        """Evaluates buffer to find optimal split, clearing buffer if successful."""
        X_buf = np.vstack(self.buffer_X)
        y_buf = np.concatenate(self.buffer_y)
        classes = np.unique(y_buf)
        
        # Pure node, no need to split
        if len(classes) <= 1:
            return
            
        n_samples, n_features = X_buf.shape
        
        # Determine features to evaluate based on max_features
        eval_features = np.arange(n_features)
        if self.max_features == "sqrt":
            n_sel = max(1, int(np.sqrt(n_features)))
            eval_features = self.rng.choice(eval_features, n_sel, replace=False)
        elif self.max_features == "log2":
            n_sel = max(1, int(np.log2(n_features)))
            eval_features = self.rng.choice(eval_features, n_sel, replace=False)
        elif isinstance(self.max_features, float) and 0.0 < self.max_features <= 1.0:
            n_sel = max(1, int(self.max_features * n_features))
            eval_features = self.rng.choice(eval_features, n_sel, replace=False)

        best_impurity = float('inf')
        best_feat = None
        best_thresh = None
        
        impurity_func = _gini if self.criterion == "gini" else _entropy
        parent_impurity = impurity_func(y_buf, classes)

        # O(N log N) split search per feature
        for feat in eval_features:
            col_vals = X_buf[:, feat]
            
            # Find unique thresholds
            thresholds = np.unique(col_vals)
            if len(thresholds) <= 1:
                continue
                
            # Test midpoints
            thresholds = (thresholds[:-1] + thresholds[1:]) / 2.0
            
            for thresh in thresholds:
                left_mask = col_vals <= thresh
                right_mask = ~left_mask
                
                y_left = y_buf[left_mask]
                y_right = y_buf[right_mask]
                
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                    
                w_left = len(y_left) / n_samples
                w_right = len(y_right) / n_samples
                
                impurity = (w_left * impurity_func(y_left, classes)) + \
                           (w_right * impurity_func(y_right, classes))
                           
                if impurity < best_impurity:
                    best_impurity = impurity
                    best_feat = feat
                    best_thresh = thresh

        # If we found a split that improves impurity
        if best_feat is not None and best_impurity < parent_impurity:
            self.feature_idx = best_feat
            self.threshold = best_thresh
            self.is_leaf = False
            
            # Spawn children
            self.left = _TreeNode(self.depth + 1, self.max_depth, self.min_samples_split, 
                                  self.criterion, self.max_features, self.rng.integers(1e9))
            self.right = _TreeNode(self.depth + 1, self.max_depth, self.min_samples_split, 
                                   self.criterion, self.max_features, self.rng.integers(1e9))
            
            # Flush buffer down to new children
            left_mask = X_buf[:, self.feature_idx] <= self.threshold
            right_mask = ~left_mask
            
            self.left.update(X_buf[left_mask], y_buf[left_mask])
            self.right.update(X_buf[right_mask], y_buf[right_mask])
            
            # Clear local buffer to save memory
            self.buffer_X = []
            self.buffer_y = []

    def predict_one(self, x: np.ndarray) -> float:
        """Traverses the tree to predict a single sample."""
        if self.is_leaf:
            return self.majority_class
        if x[self.feature_idx] <= self.threshold:
            return self.left.predict_one(x)
        return self.right.predict_one(x)


class StreamDecisionTreeClassifier(Estimator):
    """
    Streaming Decision Tree Classifier.
    
    Grows incrementally using chunk-wise updates. Buffers incoming samples
    at leaf nodes until `min_samples_split` is met, at which point the leaf
    evaluates and executes the optimal split.
    
    Parameters
    ----------
    max_depth : int, default=5
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    criterion : {'gini', 'entropy'}, default='gini'
        The function to measure the quality of a split.
    max_features : {'sqrt', 'log2', None} or float, default=None
        The number of features to consider when looking for the best split.
    random_state : int, optional
        Controls the randomness of the estimator.
    """
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, 
                 criterion: str = "gini", max_features: Union[str, float, None] = None,
                 random_state: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        
        self.root_: Optional[_TreeNode] = None

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> "StreamDecisionTreeClassifier":
        """
        Incrementally fit the tree on a chunk of data.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        
        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if X.size == 0:
            return self
            
        if self.root_ is None:
            self.root_ = _TreeNode(
                depth=0, 
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=self.max_features,
                random_state=self.random_state
            )
            
        self.root_.update(X, y)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StreamDecisionTreeClassifier":
        """
        Resets the tree and fits entirely on the given data.
        """
        self.root_ = None
        return self.partial_fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the incoming chunk.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        
        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
        """
        if self.root_ is None:
            raise RuntimeError("Estimator not fitted. Call partial_fit() or fit() first.")
            
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        preds = np.zeros(X.shape[0], dtype=np.float64)
        for i in range(X.shape[0]):
            preds[i] = self.root_.predict_one(X[i])
            
        return preds