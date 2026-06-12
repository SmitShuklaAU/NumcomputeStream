"""
test_stream_stats.py 
--------------------
Tests for streaming statistical updates in numcompute_stream.stats

Run with:
    pytest tests/test_stream_stats.py -v
"""

import numpy as np
import pytest
from numcompute_stream.stats import StreamStats


class TestStreamStats:
    
    def test_incremental_mean_variance(self):
        """Test if chunk-based Welford's algorithm matches batch numpy exactness."""
        np.random.seed(42)
        full_data = np.random.rand(1000, 3)
        
        stream_stats = StreamStats()
        
        # Feed data in chunks of 100
        for i in range(0, 1000, 100):
            stream_stats.update_stats(full_data[i:i+100])
            
        np.testing.assert_allclose(stream_stats.get_mean(), np.mean(full_data, axis=0))
        np.testing.assert_allclose(stream_stats.get_variance(), np.var(full_data, axis=0))
        np.testing.assert_allclose(stream_stats.min_, np.min(full_data, axis=0))
        np.testing.assert_allclose(stream_stats.max_, np.max(full_data, axis=0))

    def test_nan_handling_in_chunks(self):
        """Ensure NaNs in chunks don't corrupt the streaming computations."""
        chunk1 = np.array([[1.0, 2.0], [np.nan, 4.0]])
        chunk2 = np.array([[3.0, np.nan], [5.0, 6.0]])
        
        # Valid data for Col 0: [1.0, 3.0, 5.0] -> Mean: 3.0, Var: 2.666...
        # Valid data for Col 1: [2.0, 4.0, 6.0] -> Mean: 4.0, Var: 2.666...
        
        stream_stats = StreamStats()
        stream_stats.update_stats(chunk1)
        stream_stats.update_stats(chunk2)
        
        expected_mean = np.array([3.0, 4.0])
        expected_var = np.array([np.var([1, 3, 5]), np.var([2, 4, 6])])
        
        np.testing.assert_allclose(stream_stats.get_mean(), expected_mean)
        np.testing.assert_allclose(stream_stats.get_variance(), expected_var)

    def test_sliding_window_buffer(self):
        """Test if the sliding window accurately retains only the `window_size` elements."""
        stream_stats = StreamStats(window_size=5)
        
        chunk1 = np.array([[1], [2], [3], [4]]) # 4 elements
        stream_stats.update_stats(chunk1)
        
        # Buffer not full, should be [1, 2, 3, 4]
        assert stream_stats._buffer_idx == 4
        assert not stream_stats._buffer_full
        
        chunk2 = np.array([[5], [6], [7]]) # 3 elements
        stream_stats.update_stats(chunk2)
        
        # Buffer overflow. Elements should be [3, 4, 5, 6, 7]
        assert stream_stats._buffer_full
        
        # Compute max from window to verify recent data is retained
        # Quantile base function can handle the inner buffer call
        window_max = np.max(stream_stats._window_buffer)
        window_min = np.min(stream_stats._window_buffer)
        
        assert window_max == 7
        assert window_min == 3

    def test_transform_before_fit_raises(self):
        """Test safety mechanism when fetching uninitialized stats."""
        stream_stats = StreamStats()
        with pytest.raises(RuntimeError):
            stream_stats.get_mean()
        with pytest.raises(RuntimeError):
            stream_stats.get_quantiles(50.0)