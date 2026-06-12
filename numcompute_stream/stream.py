"""
stream.py
---------
Streaming Model Trainer for numcompute_stream.

Implements the StreamTrainer class to manage an estimator and an optional
preprocessing pipeline. It orchestrates chunk-wise training, prediction,
and maintains logs of cumulative accuracy and memory footprints over time.
"""

from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Optional
from numcompute_stream.metrics import StreamAccuracy

__all__ = ["StreamTrainer"]

class StreamTrainer:
    """
    Manages the training, scoring, and logging of a streaming machine learning pipeline.
    
    Attributes
    ----------
    model : Estimator
        The core streaming model (e.g., StreamDecisionTreeClassifier).
    preprocessor : Transformer, optional
        An optional streaming preprocessor or pipeline (e.g., StreamStandardScaler).
    history_ : dict
        A dictionary containing lists of logged metrics over time:
        - 'chunk_id': The sequential ID of the chunk processed.
        - 'chunk_size': Number of samples in the chunk.
        - 'memory_mb': Memory footprint of the data chunk in Megabytes.
        - 'chunk_accuracy': Accuracy strictly on the current chunk.
        - 'cumulative_accuracy': Running global accuracy across all scored chunks.
    """
    
    def __init__(self, model: Any, preprocessor: Optional[Any] = None) -> None:
        self.model = model
        self.preprocessor = preprocessor
        
        self.global_accuracy_tracker = StreamAccuracy()
        self.chunk_counter = 0
        
        self.history_: Dict[str, List[float]] = {
            "chunk_id": [],
            "chunk_size": [],
            "memory_mb": [],
            "chunk_accuracy": [],
            "cumulative_accuracy": []
        }

    def fit_chunk(self, X: np.ndarray, y: np.ndarray) -> "StreamTrainer":
        """
        Incrementally trains the preprocessor and model on a new chunk of data.
        
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
        
        if X.size == 0:
            return self
            
        if self.preprocessor is not None:
            # Update the preprocessor state and transform the data for the model
            self.preprocessor.partial_fit(X)
            X_trans = self.preprocessor.transform(X)
        else:
            X_trans = X
            
        self.model.partial_fit(X_trans, y)
        self.chunk_counter += 1
        
        return self

    def score_chunk(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Predicts labels for a chunk, updates global accuracy metrics, and logs the results.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        
        Returns
        -------
        float
            The local accuracy on the provided chunk.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        
        if X.size == 0:
            return 0.0
            
        if self.preprocessor is not None:
            X_trans = self.preprocessor.transform(X)
        else:
            X_trans = X
            
        preds = self.model.predict(X_trans)
        
        # 1. Local Chunk Accuracy
        chunk_acc = float(np.mean(preds == y))
        
        # 2. Cumulative Accuracy
        self.global_accuracy_tracker.update(y, preds)
        cum_acc = self.global_accuracy_tracker.result()
        
        # 3. Data Memory Footprint
        memory_mb = (X.nbytes + y.nbytes) / (1024 * 1024)
        
        # Record Logs
        self.history_["chunk_id"].append(self.chunk_counter)
        self.history_["chunk_size"].append(len(y))
        self.history_["memory_mb"].append(memory_mb)
        self.history_["chunk_accuracy"].append(chunk_acc)
        self.history_["cumulative_accuracy"].append(cum_acc)
        
        return chunk_acc

    def prequential_update(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Performs an Interleaved Test-Then-Train (Prequential) update.
        Scores the model on the unseen chunk, logs metrics, then fits the model to it.
        
        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray
        
        Returns
        -------
        float
            The chunk accuracy score prior to fitting.
        """
        # 1. Score unseen data (Test)
        # Note: If it's the very first chunk and the model isn't fitted, 
        # score_chunk will raise an error from the model. We handle this cleanly.
        try:
            chunk_acc = self.score_chunk(X, y)
        except RuntimeError:
            # First chunk fallback: Model is totally untrained, score is effectively 0
            chunk_acc = 0.0
            memory_mb = (X.nbytes + y.nbytes) / (1024 * 1024)
            self.history_["chunk_id"].append(self.chunk_counter)
            self.history_["chunk_size"].append(len(y))
            self.history_["memory_mb"].append(memory_mb)
            self.history_["chunk_accuracy"].append(0.0)
            
            # Cumulative tracker requires identical lengths, so we add all-wrong dummy zeros
            self.global_accuracy_tracker.update(np.ones(len(y)), np.zeros(len(y)))
            self.history_["cumulative_accuracy"].append(self.global_accuracy_tracker.result())
            
        # 2. Update model (Train)
        self.fit_chunk(X, y)
        
        return chunk_acc