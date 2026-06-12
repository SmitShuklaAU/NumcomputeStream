"""
preprocessing.py
----------------
Streaming data preprocessing for numcompute_stream.

Extends the 2.1 static preprocessing classes to support incremental 
`.partial_fit()` updates for online machine learning pipelines.
"""

from __future__ import annotations
import numpy as np
from numcompute_stream.stats import StreamStats

__all__ = [
    "StreamStandardScaler",
    "StreamMinMaxScaler",
    "StreamOneHotEncoder",
    "StreamSimpleImputer"
]

# ----------------------------------------------------------------------------
# StreamStandardScaler
# ----------------------------------------------------------------------------

class StreamStandardScaler:
    """
    Streaming Z-score standardization: z = (x - mean) / std
    
    Computes running mean and variance incrementally using Welford's algorithm
    via the modular `StreamStats` engine.
    
    Attributes
    ----------
    mean_ : np.ndarray, shape(n,)
    std_  : np.ndarray, shape(n,) 
    
    Complexity
    ----------
    Time  : O(n_samples * n_features) per chunk
    Space : O(n_features) for running states
    """
    
    def __init__(self, eps: float = 1e-12) -> None:
        self.eps: float = eps
        self._stats: StreamStats | None = None
    
    @property
    def mean_(self) -> np.ndarray | None:
        return self._stats.get_mean() if self._stats is not None else None

    @property
    def std_(self) -> np.ndarray | None:
        if self._stats is None:
            return None
        std = self._stats.get_std()
        return np.where(std < self.eps, 1.0, std)

    def partial_fit(self, x: np.ndarray) -> "StreamStandardScaler":
        """
        Incrementally update mean and std with a new chunk of data.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)  
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array, got {x.ndim}D")
        if x.size == 0:
            return self # Skip empty chunks safely

        if self._stats is None:
            self._stats = StreamStats()
            
        self._stats.update_stats(x)
        return self
        
    def fit(self, x: np.ndarray) -> "StreamStandardScaler":
        """Reset stats and fit entirely on the given data."""
        self._stats = None
        return self.partial_fit(x)
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply Z-score Standardization using the current running estimates.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Call partial_fit() or fit() before transform().")
        
        x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        if x.shape[1] != self.mean_.shape[0]:
            raise ValueError(f"expected {self.mean_.shape[0]} columns, got {x.shape[1]}")
            
        nan_mask = np.isnan(x)
        x = np.where(nan_mask, self.mean_, x)
        return (x - self.mean_) / self.std_
        
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


# ----------------------------------------------------------------------------
# StreamMinMaxScaler
# ----------------------------------------------------------------------------

class StreamMinMaxScaler:
    """
    Streaming Min-Max Scaler.
    Incrementally expands the global minimum and maximum bounds as chunks arrive.
    """
    def __init__(self, feature_range: tuple[int, int] = (0, 1), eps: float = 1e-12) -> None:
        if feature_range[0] >= feature_range[1]:
            raise ValueError("feature range must satisfy min < max")
        
        self.feature_range: tuple[int, int] = feature_range
        self.eps: float = eps
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None
        self.data_range_: np.ndarray | None = None
        
    def partial_fit(self, x: np.ndarray) -> "StreamMinMaxScaler":
        """Incrementally update global min and max."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array, got {x.ndim}D")
        if x.size == 0:
            return self

        with np.errstate(all='ignore'):
            chunk_min = np.nanmin(x, axis=0)
            chunk_max = np.nanmax(x, axis=0)
            
            # Mask out all-NaN columns in this chunk
            valid_min = ~np.isnan(chunk_min)
            valid_max = ~np.isnan(chunk_max)
        
        if self.min_ is None:
            self.min_ = np.where(valid_min, chunk_min, np.inf)
            self.max_ = np.where(valid_max, chunk_max, -np.inf)
        else:
            # Safely minimum/maximum skipping chunk NaNs
            self.min_[valid_min] = np.minimum(self.min_[valid_min], chunk_min[valid_min])
            self.max_[valid_max] = np.maximum(self.max_[valid_max], chunk_max[valid_max])
            
        data_range = self.max_ - self.min_
        self.data_range_ = np.where(data_range < self.eps, 1.0, data_range)
        
        return self

    def fit(self, x: np.ndarray) -> "StreamMinMaxScaler":
        self.min_ = None
        self.max_ = None
        self.data_range_ = None
        return self.partial_fit(x)
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.min_ is None:
            raise RuntimeError("Call partial_fit() before transform()")
            
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            
        if x.shape[1] != self.min_.shape[0]:
            raise ValueError(f"expected {self.min_.shape[0]} columns, got {x.shape[1]}")
       
        f_min, f_max = self.feature_range
        x_std = (x - self.min_) / self.data_range_
        return x_std * (f_max - f_min) + f_min
    
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


# ----------------------------------------------------------------------------
# StreamOneHotEncoder
# ----------------------------------------------------------------------------

class StreamOneHotEncoder:
    """
    Streaming One-Hot Encoder.
    Dynamically expands the category dictionary as new unique values are seen 
    in incoming chunks. Output dimensions grow over time.
    """
    def __init__(self, handle_unknown: str = "ignore") -> None:
        if handle_unknown not in ("ignore", "error"):
           raise ValueError("handle unknown must be 'ignore' or 'error'")
        self.handle_unknown: str = handle_unknown
        self.categories_: list[np.ndarray] | None = None 
       
    def partial_fit(self, x: np.ndarray) -> "StreamOneHotEncoder":
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.size == 0:
            return self
            
        if self.categories_ is None:
            self.categories_ = [np.unique(x[:, i]) for i in range(x.shape[1])]
        else:
            if len(self.categories_) != x.shape[1]:
                raise ValueError(f"Expected {len(self.categories_)} features, got {x.shape[1]}")
            
            for i in range(x.shape[1]):
                chunk_unique = np.unique(x[:, i])
                # Union the new unique values with the existing ones and sort
                self.categories_[i] = np.unique(np.concatenate((self.categories_[i], chunk_unique)))
                
        return self

    def fit(self, x: np.ndarray) -> "StreamOneHotEncoder":
        self.categories_ = None
        return self.partial_fit(x)
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.categories_ is None:
            raise RuntimeError("Call partial_fit() before transform()")
            
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            
        encoded = []
        for i, cats in enumerate(self.categories_):
            col = x[:, i]
            if self.handle_unknown == "error":
                unseen = ~np.isin(col, cats)
                if unseen.any():
                    raise ValueError(f"Column {i} contains unknown categories.")
            encoded.append((col[:, None] == cats).astype(float))        
        return np.hstack(encoded)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


# ----------------------------------------------------------------------------
# StreamSimpleImputer
# ----------------------------------------------------------------------------

class StreamSimpleImputer:
    """
    Streaming Simple Imputer.
    Updates missing value replacement estimates (mean, median) incrementally.
    """
    def __init__(self, strategy: str = "mean", constant: float = 0.0) -> None:
        if strategy not in ("constant", "mean", "median"):
            raise ValueError("Strategy must be 'constant', 'mean' or 'median'")
        self.strategy: str = strategy
        self.constant: float = constant
        
        self.fill_values_: np.ndarray | None = None
        self._stats: StreamStats | None = None
    
    def partial_fit(self, x: np.ndarray) -> "StreamSimpleImputer":
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array, got {x.ndim}D")
        if x.size == 0:
            return self

        if self.strategy == "constant":
            self.fill_values_ = np.full(x.shape[1], self.constant)
            return self

        # For mean/median, defer to StreamStats
        if self._stats is None:
            self._stats = StreamStats()
            
        self._stats.update_stats(x)
        
        if self.strategy == "mean":
            self.fill_values_ = self._stats.get_mean()
        elif self.strategy == "median":
            # get_quantiles returns shape (1, n_features), flatten it to (n_features,)
            self.fill_values_ = self._stats.get_quantiles(50.0).flatten()
            
        return self

    def fit(self, x: np.ndarray) -> "StreamSimpleImputer":
        self.fill_values_ = None
        self._stats = None
        return self.partial_fit(x)
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.fill_values_ is None:
            raise RuntimeError("Call partial_fit() before transform()")
        
        x = np.asarray(x, dtype=np.float64).copy()
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        if x.shape[1] != self.fill_values_.shape[0]:
            raise ValueError(f"expected {self.fill_values_.shape[0]} columns, got {x.shape[1]}")
        
        nan_mask = np.isnan(x)
        x = np.where(nan_mask, self.fill_values_, x)
        return x    
    
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)