"""
test_stream_preprocessing.py 
----------------------------
Tests for streaming preprocessing components in numcompute_stream.preprocessing

Run with:
    pytest tests/test_stream_preprocessing.py -v
"""

import numpy as np
import pytest
from numcompute_stream.preprocessing import (
    StreamStandardScaler, 
    StreamMinMaxScaler, 
    StreamOneHotEncoder, 
    StreamSimpleImputer
)

class TestStreamStandardScaler:
    
    def test_streaming_equivalence_to_batch(self):
        np.random.seed(42)
        full_data = np.random.rand(100, 3)
        
        # Batch fit
        batch_scaler = StreamStandardScaler().fit(full_data)
        
        # Stream fit in 4 chunks
        stream_scaler = StreamStandardScaler()
        for i in range(0, 100, 25):
            stream_scaler.partial_fit(full_data[i:i+25])
            
        np.testing.assert_allclose(stream_scaler.mean_, batch_scaler.mean_)
        np.testing.assert_allclose(stream_scaler.std_, batch_scaler.std_)
        
        # Test transforms are identical
        np.testing.assert_allclose(
            stream_scaler.transform(full_data),
            batch_scaler.transform(full_data)
        )

    def test_empty_chunk_ignored(self):
        scaler = StreamStandardScaler()
        scaler.partial_fit(np.array([[1.0, 2.0]]))
        scaler.partial_fit(np.array([[]]).reshape(0, 2)) # Empty chunk
        
        assert scaler.mean_ is not None
        np.testing.assert_allclose(scaler.mean_, [1.0, 2.0])


class TestStreamMinMaxScaler:
    
    def test_incremental_bounds_expansion(self):
        chunk1 = np.array([[2.0, 5.0], [3.0, 6.0]])
        chunk2 = np.array([[1.0, 7.0], [4.0, 4.0]])
        
        scaler = StreamMinMaxScaler(feature_range=(0, 1))
        scaler.partial_fit(chunk1)
        
        np.testing.assert_allclose(scaler.min_, [2.0, 5.0])
        np.testing.assert_allclose(scaler.max_, [3.0, 6.0])
        
        scaler.partial_fit(chunk2)
        
        # Bounds should expand to include [1.0, 7.0] and [4.0, 4.0] min/maxes
        np.testing.assert_allclose(scaler.min_, [1.0, 4.0])
        np.testing.assert_allclose(scaler.max_, [4.0, 7.0])

    def test_nan_ignoring_in_bounds(self):
        scaler = StreamMinMaxScaler()
        scaler.partial_fit(np.array([[np.nan, 5.0], [3.0, np.nan]]))
        scaler.partial_fit(np.array([[1.0, np.nan]]))
        
        np.testing.assert_allclose(scaler.min_, [1.0, 5.0])


class TestStreamOneHotEncoder:
    
    def test_dynamic_category_expansion(self):
        chunk1 = np.array([['red', 'square'], ['blue', 'circle']])
        chunk2 = np.array([['green', 'square'], ['red', 'triangle']])
        
        encoder = StreamOneHotEncoder()
        encoder.partial_fit(chunk1)
        
        # At this point, shape should be 4 (2 colors, 2 shapes)
        out1 = encoder.transform([['red', 'square']])
        assert out1.shape == (1, 4)
        
        encoder.partial_fit(chunk2)
        
        # Categories expanded: colors(blue, green, red), shapes(circle, square, triangle) -> shape 6
        out2 = encoder.transform([['green', 'triangle']])
        assert out2.shape == (1, 6)
        
        # Verify internal category tracking matches alphabetical sort
        assert np.array_equal(encoder.categories_[0], ['blue', 'green', 'red'])
        assert np.array_equal(encoder.categories_[1], ['circle', 'square', 'triangle'])


class TestStreamSimpleImputer:
    
    def test_streaming_mean_strategy(self):
        chunk1 = np.array([[1.0, np.nan], [3.0, 4.0]])
        chunk2 = np.array([[np.nan, 8.0], [5.0, 6.0]])
        
        imputer = StreamSimpleImputer(strategy="mean")
        imputer.partial_fit(chunk1)
        imputer.partial_fit(chunk2)
        
        # Col 0: [1.0, 3.0, 5.0] -> mean 3.0
        # Col 1: [4.0, 8.0, 6.0] -> mean 6.0
        np.testing.assert_allclose(imputer.fill_values_, [3.0, 6.0])
        
        test_chunk = np.array([[np.nan, np.nan]])
        out = imputer.transform(test_chunk)
        np.testing.assert_allclose(out, [[3.0, 6.0]])

    def test_streaming_constant_strategy(self):
        imputer = StreamSimpleImputer(strategy="constant", constant=-1.0)
        imputer.partial_fit(np.array([[np.nan, 2.0]]))
        
        out = imputer.transform(np.array([[np.nan, np.nan]]))
        np.testing.assert_allclose(out, [[-1.0, -1.0]])