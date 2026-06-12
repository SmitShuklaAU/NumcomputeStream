"""
test_stream_trainer.py
----------------------
Tests for the StreamTrainer loop logic and metric logging.

Run with:
    pytest tests/test_stream_trainer.py -v
"""

import numpy as np
import pytest
from numcompute_stream.stream import StreamTrainer

class MockPreprocessor:
    def __init__(self):
        self.fit_calls = 0
    def partial_fit(self, X, y=None):
        self.fit_calls += 1
        return self
    def transform(self, X):
        return X * 2

class MockModel:
    def __init__(self):
        self.fit_calls = 0
        self.fitted = False
    def partial_fit(self, X, y):
        self.fit_calls += 1
        self.fitted = True
        return self
    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("Not fitted")
        # Just return dummy predictions of class 0
        return np.zeros(X.shape[0])

class TestStreamTrainer:

    def test_fit_chunk_routing(self):
        """Ensure fit_chunk applies preprocessing and updates chunk counters."""
        prep = MockPreprocessor()
        model = MockModel()
        trainer = StreamTrainer(model, prep)
        
        X = np.ones((5, 2))
        y = np.ones(5)
        
        trainer.fit_chunk(X, y)
        
        assert prep.fit_calls == 1
        assert model.fit_calls == 1
        assert trainer.chunk_counter == 1

    def test_score_chunk_logging(self):
        """Ensure score_chunk correctly tracks cumulative metrics and memory."""
        prep = MockPreprocessor()
        model = MockModel()
        trainer = StreamTrainer(model, prep)
        
        # Fit to initialize the mock model
        trainer.fit_chunk(np.ones((2, 2)), np.zeros(2))
        
        X1 = np.ones((10, 2))
        y1 = np.zeros(10) # Model predicts 0s, so this is 100% correct
        trainer.score_chunk(X1, y1)
        
        X2 = np.ones((10, 2))
        y2 = np.ones(10) # Model predicts 0s, so this is 0% correct
        trainer.score_chunk(X2, y2)
        
        hist = trainer.history_
        
        assert len(hist["chunk_id"]) == 2
        
        # Accuracies
        assert hist["chunk_accuracy"][0] == 1.0  # 10/10 correct
        assert hist["chunk_accuracy"][1] == 0.0  # 0/10 correct
        assert hist["cumulative_accuracy"][1] == 0.5  # 10/20 correct overall
        
        # Memory Footprint checking (approximate for float64 arrays)
        assert hist["memory_mb"][0] > 0.0

    def test_prequential_update_fallback(self):
        """Ensure prequential update safely handles the untrained first chunk."""
        model = MockModel()
        trainer = StreamTrainer(model) # No preprocessor
        
        X = np.ones((5, 2))
        y = np.ones(5)
        
        # Model is completely blank. prequential_update should intercept the 
        # RuntimeError on .predict(), log a 0% accuracy, then .partial_fit().
        acc = trainer.prequential_update(X, y)
        
        assert acc == 0.0
        assert trainer.chunk_counter == 1
        assert model.fit_calls == 1
        assert trainer.history_["chunk_accuracy"][0] == 0.0