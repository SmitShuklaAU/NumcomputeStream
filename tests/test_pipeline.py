"""
test_stream_pipeline.py
-----------------------
Tests for streaming pipeline routing in numcompute_stream.pipeline

Run with:
    pytest tests/test_stream_pipeline.py -v
"""

import numpy as np
import pytest
from numcompute_stream.pipeline import StreamPipeline, StreamFeatureUnion
from numcompute.pipeline import Transformer, Estimator

# ---------------------------------------------------------------------------
# Streaming Stubs
# ---------------------------------------------------------------------------

class MockStreamTransformer(Transformer):
    def __init__(self, multiplier: float):
        self.multiplier = multiplier
        self.fit_calls = 0
        self.partial_fit_calls = 0

    def fit(self, X, y=None, **kw):
        self.fit_calls += 1
        return self

    def partial_fit(self, X, y=None, **kw):
        self.partial_fit_calls += 1
        return self

    def transform(self, X):
        return X * self.multiplier


class MockStreamEstimator(Estimator):
    def __init__(self):
        self.fit_calls = 0
        self.partial_fit_calls = 0
        self.last_X_shape = None

    def fit(self, X, y=None, **kw):
        self.fit_calls += 1
        return self

    def partial_fit(self, X, y=None, **kw):
        self.partial_fit_calls += 1
        self.last_X_shape = X.shape
        # Track the sum to verify transformations were applied before it arrived
        self.last_X_sum = np.sum(X) 
        return self

    def predict(self, X):
        return np.zeros(X.shape[0])


class NonStreamingStub(Transformer):
    """Stub missing the partial_fit method entirely."""
    def fit(self, X, y=None, **kw): return self
    def transform(self, X): return X

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStreamPipeline:

    def test_partial_fit_routing_and_transformation(self):
        """
        Verify that partial_fit calls partial_fit on every step, and that 
        intermediate steps correctly transform the data before passing it down.
        """
        t1 = MockStreamTransformer(multiplier=2.0)
        t2 = MockStreamTransformer(multiplier=3.0)
        est = MockStreamEstimator()
        
        pipe = StreamPipeline([
            ('t1', t1),
            ('t2', t2),
            ('model', est)
        ])
        
        X_chunk = np.ones((5, 2))
        pipe.partial_fit(X_chunk)
        
        assert t1.partial_fit_calls == 1
        assert t2.partial_fit_calls == 1
        assert est.partial_fit_calls == 1
        
        # Original sum = 10.
        # After t1 (x2) = 20.
        # After t2 (x3) = 60.
        assert est.last_X_sum == 60.0

    def test_partial_fit_raises_on_non_streaming_step(self):
        pipe = StreamPipeline([
            ('t1', NonStreamingStub()),
            ('model', MockStreamEstimator())
        ])
        
        X = np.ones((2, 2))
        with pytest.raises(AttributeError, match="does not implement partial_fit"):
            pipe.partial_fit(X)

    def test_fallback_to_standard_fit_predict(self):
        """Ensure inherited fit and predict still function perfectly."""
        t1 = MockStreamTransformer(multiplier=2.0)
        est = MockStreamEstimator()
        pipe = StreamPipeline([('t1', t1), ('model', est)])
        
        X = np.ones((2, 2))
        pipe.fit(X)
        
        assert t1.fit_calls == 1
        assert est.fit_calls == 1
        
        preds = pipe.predict(X)
        assert preds.shape == (2,)


class TestStreamFeatureUnion:

    def test_partial_fit_routing(self):
        t1 = MockStreamTransformer(multiplier=2.0)
        t2 = MockStreamTransformer(multiplier=3.0)
        
        union = StreamFeatureUnion([
            ('t1', t1),
            ('t2', t2)
        ])
        
        X = np.ones((5, 2))
        union.partial_fit(X)
        
        assert t1.partial_fit_calls == 1
        assert t2.partial_fit_calls == 1

    def test_partial_fit_raises_on_non_streaming_step(self):
        union = StreamFeatureUnion([
            ('t1', MockStreamTransformer(1.0)),
            ('t2', NonStreamingStub())
        ])
        
        X = np.ones((2, 2))
        with pytest.raises(AttributeError, match="does not implement partial_fit"):
            union.partial_fit(X)