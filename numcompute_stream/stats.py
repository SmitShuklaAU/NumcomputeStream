"""
stats.py
--------
Streaming statistical tracking for numcompute_stream.

Extends the 2.1 static statistical functions with chunk-based streaming
versions. We import the foundational 2.1 functions from `numcompute.stats` 
and expose them here alongside the new `StreamStats` engine.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Union, List, Any

# Re-exporting static functions from Assigment 2.1
# This assumes your 2.1 package 'numcompute' is installed or in the Python path
from numcompute.stats import (
    mean,
    var,
    median,
    histogram,
    quantile,
    minimum,
    maximum,
    std,
    summary,
)

__all__ = [
    "mean", "var", "median", "histogram", "quantile", 
    "minimum", "maximum", "std", "summary",
    "StreamStats"
]

class StreamStats:
    """
    Stateful streaming statistics tracker.
    
    Computes running mean, variance, min, and max using a parallelized 
    batched version of Welford's online algorithm. Maintains a sliding 
    window buffer for estimating quantiles and histograms over recent data.
    
    Attributes
    ----------
    n_samples_ : np.ndarray
        Total number of valid samples seen so far per feature.
    mean_ : np.ndarray
        Running mean per feature.
    var_ : np.ndarray
        Running variance per feature (accessible via .get_variance()).
    min_ : np.ndarray
        Running minimum per feature.
    max_ : np.ndarray
        Running maximum per feature.
    """
    
    def __init__(self, window_size: int = 10000, n_bins: int = 10) -> None:
        """
        Parameters
        ----------
        window_size : int, default=10000
            The size of the sliding window buffer for quantiles and histograms.
        n_bins : int, default=10
            Default number of bins for streaming histograms.
        """
        self.n_samples_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self._M2: np.ndarray | None = None  # Sum of squared differences for Welford
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None
        
        self.window_size: int = window_size
        self._window_buffer: np.ndarray | None = None
        self._buffer_idx: int = 0
        self._buffer_full: bool = False
        self.n_bins: int = n_bins

    def update_stats(self, X_chunk: np.ndarray) -> "StreamStats":
        """
        Updates the running statistics incrementally with a new chunk of data.
        
        Uses a parallel/batched Welford's algorithm to combine the statistics of
        the old data with the incoming chunk safely handling NaNs.
        
        Parameters
        ----------
        X_chunk : np.ndarray, shape (n_samples, n_features)
            The incoming data chunk.
            
        Returns
        -------
        self
        
        Complexity
        ----------
        Time : O(n_samples * n_features)
        Space : O(n_features) beyond the sliding window buffer
        """
        X_chunk = np.asarray(X_chunk, dtype=np.float64)
        if X_chunk.ndim == 1:
            X_chunk = X_chunk.reshape(-1, 1)
            
        n_chunk, n_features = X_chunk.shape
        if n_chunk == 0:
            return self

        # 1. Update Welford's Mean, Variance, Min, and Max
        with np.errstate(invalid='ignore', divide='ignore'):
            chunk_valid_counts = np.sum(~np.isnan(X_chunk), axis=0)
            chunk_mean = np.nanmean(X_chunk, axis=0)
            chunk_var = np.nanvar(X_chunk, axis=0)
            chunk_M2 = chunk_var * chunk_valid_counts
            
            chunk_min = np.nanmin(X_chunk, axis=0)
            chunk_max = np.nanmax(X_chunk, axis=0)

        # Zero out NaNs that appear in completely empty (all-NaN) columns 
        # so they don't corrupt the arithmetic. They will be ignored via update_mask.
        chunk_mean = np.where(np.isnan(chunk_mean), 0.0, chunk_mean)
        chunk_M2 = np.where(np.isnan(chunk_M2), 0.0, chunk_M2)

        if self.mean_ is None:
            # Initialize state on the first chunk
            self.n_samples_ = chunk_valid_counts.copy()
            self.mean_ = chunk_mean.copy()
            self._M2 = chunk_M2.copy()
            self.min_ = np.where(np.isnan(chunk_min), np.inf, chunk_min)
            self.max_ = np.where(np.isnan(chunk_max), -np.inf, chunk_max)
        else:
            if self.mean_.shape[0] != n_features:
                raise ValueError(f"Expected {self.mean_.shape[0]} features, got {n_features}")
                
            n_old = self.n_samples_.copy()
            n_new = chunk_valid_counts
            n_total = n_old + n_new
            
            update_mask = n_new > 0  # Only update columns that have valid new data
            
            if np.any(update_mask):
                delta = chunk_mean - self.mean_
                
                # Batched Welford update formulas:
                # new_mean = old_mean + (n_new / n_total) * delta
                self.mean_[update_mask] += (delta * n_new / n_total)[update_mask]
                
                # new_M2 = old_M2 + chunk_M2 + delta^2 * (n_old * n_new / n_total)
                self._M2[update_mask] += (chunk_M2 + (delta ** 2) * n_old * n_new / n_total)[update_mask]
                
                self.n_samples_ = n_total
                
                # Update Min/Max
                valid_chunk_min = np.where(np.isnan(chunk_min), np.inf, chunk_min)
                valid_chunk_max = np.where(np.isnan(chunk_max), -np.inf, chunk_max)
                self.min_ = np.minimum(self.min_, valid_chunk_min)
                self.max_ = np.maximum(self.max_, valid_chunk_max)

        # 2. Update Sliding Window Buffer (for quantiles/histograms)
        if self._window_buffer is None:
            self._window_buffer = np.full((self.window_size, n_features), np.nan)
            
        if n_chunk >= self.window_size:
            # Chunk overrides the entire window
            self._window_buffer[:] = X_chunk[-self.window_size:]
            self._buffer_idx = 0
            self._buffer_full = True
        else:
            end_idx = self._buffer_idx + n_chunk
            if end_idx <= self.window_size:
                self._window_buffer[self._buffer_idx:end_idx] = X_chunk
                self._buffer_idx = end_idx
                if self._buffer_idx == self.window_size:
                    self._buffer_idx = 0
                    self._buffer_full = True
            else:
                # Wrap around the cyclic buffer
                overflow = end_idx - self.window_size
                first_part = n_chunk - overflow
                self._window_buffer[self._buffer_idx:] = X_chunk[:first_part]
                self._window_buffer[:overflow] = X_chunk[first_part:]
                self._buffer_idx = overflow
                self._buffer_full = True

        return self

    def get_mean(self) -> np.ndarray:
        """Returns the streaming mean computed over all chunks."""
        if self.mean_ is None:
            raise RuntimeError("Call update_stats() before getting mean.")
        return self.mean_

    def get_variance(self) -> np.ndarray:
        """Returns the streaming population variance computed over all chunks."""
        if self._M2 is None or self.n_samples_ is None:
            raise RuntimeError("Call update_stats() before getting variance.")
        # Avoid division by zero warnings cleanly
        safe_n = np.where(self.n_samples_ == 0, 1, self.n_samples_)
        var_ = self._M2 / safe_n
        return np.where(self.n_samples_ == 0, np.nan, var_)

    def get_std(self) -> np.ndarray:
        """Returns the streaming population standard deviation."""
        return np.sqrt(self.get_variance())

    def get_quantiles(self, q: Union[float, List[float], np.ndarray]) -> np.ndarray:
        """
        Estimates quantiles using the recent sliding window buffer, natively 
        forwarding data to the static `quantile` function from Assignment 2.1.
        
        Parameters
        ----------
        q : float or array-like
            Percentile(s) to compute (0 to 100).
            
        Returns
        -------
        np.ndarray
            Quantiles computed over the recent sliding window.
        """
        if self._window_buffer is None:
            raise RuntimeError("Call update_stats() before computing quantiles.")
            
        valid_data = self._window_buffer if self._buffer_full else self._window_buffer[:self._buffer_idx]
        
        q_arr = np.atleast_1d(q)
        results = []
        for i in range(valid_data.shape[1]):
            col = valid_data[:, i]
            col_clean = col[~np.isnan(col)]
            
            if col_clean.size == 0:
                results.append(np.full(q_arr.shape, np.nan))
            else:
                # Dispatches to your 2.1 static quantile logic
                res = quantile(col_clean, q_arr) 
                results.append(np.atleast_1d(res))
                
        return np.column_stack(results)

    def get_histogram(self, feature_idx: int = 0, bins: Optional[int] = None) -> Any:
        """
        Computes a histogram for a specific feature over the sliding window 
        using the static `histogram` function from Assignment 2.1.
        
        Parameters
        ----------
        feature_idx : int
            Index of the feature.
        bins : int, optional
            Number of bins. Uses self.n_bins if None.
            
        Returns
        -------
        The return output format of your base `histogram` implementation.
        """
        if self._window_buffer is None:
            raise RuntimeError("Call update_stats() before computing histogram.")
            
        valid_data = self._window_buffer if self._buffer_full else self._window_buffer[:self._buffer_idx]
        col_data = valid_data[:, feature_idx]
        col_clean = col_data[~np.isnan(col_data)]
        
        b = bins or self.n_bins
        if col_clean.size == 0:
            return np.zeros(b)
            
        # Dispatch to 2.1 histogram function 
        return histogram(col_clean, n_bins=b)