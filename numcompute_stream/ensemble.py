"""
ensemble.py
-----------
Streaming ensemble methods for numcompute_stream.

Implements a Streaming Random Forest Classifier using mini-batch bagging.
Each tree in the ensemble is updated incrementally using bootstrapped 
samples of the incoming data chunks.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Union, List

from numcompute_stream.pipeline import Estimator
from numcompute_stream.tree import StreamDecisionTreeClassifier

__all__ = ["StreamRandomForestClassifier"]


class StreamRandomForestClassifier(Estimator):
    """
    Streaming Random Forest Classifier.
    
    An ensemble of StreamDecisionTreeClassifiers trained via online bagging.
    As each chunk of data arrives, it is bootstrapped (sampled with replacement)
    independently for each tree. 
    
    Parameters
    ----------
    n_estimators : int, default=10
        The number of trees in the forest.
    max_depth : int, default=5
        The maximum depth of each tree.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    criterion : {'gini', 'entropy'}, default='gini'
        The function to measure the quality of a split.
    max_features : {'sqrt', 'log2', None} or float, default='sqrt'
        The number of features to consider when looking for the best split.
        Default is 'sqrt' to promote tree decorrelation.
    random_state : int, optional
        Controls the randomness of the bootstrapping and feature sampling.
    """
    
    def __init__(self, 
                 n_estimators: int = 10, 
                 max_depth: int = 5, 
                 min_samples_split: int = 2, 
                 criterion: str = "gini", 
                 max_features: Union[str, float, None] = "sqrt",
                 random_state: Optional[int] = None):
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        
        self.estimators_: List[StreamDecisionTreeClassifier] = []
        self._rng = np.random.default_rng(self.random_state)
        
    def _init_estimators(self) -> None:
        """Initializes the ensemble of trees with distinct random seeds."""
        self.estimators_ = []
        # Generate a unique seed for each tree to ensure varied feature sampling
        seeds = self._rng.integers(0, 10**9, size=self.n_estimators)
        
        for seed in seeds:
            tree = StreamDecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=self.max_features,
                random_state=int(seed)
            )
            self.estimators_.append(tree)

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> "StreamRandomForestClassifier":
        """
        Incrementally fit the random forest on a chunk of data.
        Applies mini-batch bagging by bootstrapping the incoming chunk for each tree.
        
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
            
        if not self.estimators_:
            self._init_estimators()
            
        n_samples = X.shape[0]
        
        # Route bootstrapped chunks to each tree
        for tree in self.estimators_:
            # Bootstrap sampling: draw N samples with replacement from the chunk
            indices = self._rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            tree.partial_fit(X_boot, y_boot)
            
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StreamRandomForestClassifier":
        """
        Resets the ensemble and fits entirely on the given data.
        """
        self.estimators_ = []
        return self.partial_fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using majority voting across all trees.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        
        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
        """
        if not self.estimators_:
            raise RuntimeError("Estimator not fitted. Call partial_fit() or fit() first.")
            
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        n_samples = X.shape[0]
        
        # Collect predictions from all trees
        # Shape: (n_estimators, n_samples)
        tree_preds = np.zeros((self.n_estimators, n_samples), dtype=np.float64)
        for i, tree in enumerate(self.estimators_):
            tree_preds[i, :] = tree.predict(X)
            
        # Apply majority vote along the estimators axis
        return self._majority_vote(tree_preds)

    def _majority_vote(self, preds: np.ndarray) -> np.ndarray:
        """
        Computes the mode (majority vote) across estimators for each sample 
        using pure NumPy, handling arbitrary float/int class labels securely.
        
        Parameters
        ----------
        preds : np.ndarray, shape (n_estimators, n_samples)
        
        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        n_estimators, n_samples = preds.shape
        votes = np.empty(n_samples, dtype=preds.dtype)
        
        for i in range(n_samples):
            vals, counts = np.unique(preds[:, i], return_counts=True)
            votes[i] = vals[np.argmax(counts)]
            
        return votes