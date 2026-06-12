""" stats.py 
- Basic descriptive statistics:
    Mean, median, standard deviation, min, max
- Histogram
- Quantiles (with NaN handling)
- Axis-wise stats with clear dimension/shape behaviour
"""
import numpy as np
from typing import NamedTuple, Optional, Sequence, Union

# Type aliases 
ArrayLike = Union[np.ndarray, list, tuple]
Axis = Optional[int]

class SummaryResult(NamedTuple):
    """
    Structured return type for summary function.
    
    Fields
    ------
    mean : np.ndarray, arithmetic mean of valid observations
    median : np.ndarray, median value
    std : np.ndarray, sample standard deviation (Bessel-corrected, ddof=1 by default)
    var : np.ndarray, sample variance
    min : np.ndarray, minimum value
    max : np.ndarray, maximum value
    """
    mean:   np.ndarray
    median: np.ndarray
    std:    np.ndarray
    var:    np.ndarray
    min:    np.ndarray
    max:    np.ndarray


# Internal helpers
def _to_float_array(X: ArrayLike) -> np.ndarray:
    """
    Convert X to a contiguous float64 NumPy array.

    Parameters
    ----------
    X : array-like                - Anything accepted by np.asarray
   
    Returns
    -------
    np.ndarray : contiguous view of X

    Complexity
    ----------
    Time : O(N) for a copy; O(1) if already float64 contiguous
    Space : O(N) worst-case (full copy)
    """
    arr = np.asarray(X, dtype=np.float64)
    return np.ascontiguousarray(arr)


def _keepdims_result(result: np.ndarray, 
    original_axis: Axis, 
    original_ndim: int, 
    keepdims: bool) -> np.ndarray:
    """
    Re-inserts the reduced axis as a size-1 dimension when keepdims is True.

    Parameters
    ----------
    result : np.ndarray           - the reduced array
    original_axis : int or None   - the axis that was reduced, None indicates full flattening reduction
    original_ndim : int.          - number of dimensions the input array had before reduction
    keepdims : bool               - if False result is returned unchanged

    Returns
    -------
        np.ndarray
            keepdims=False: result unchanged.
            keepdims=True:  shape has original_ndim dimensions; the reduced axis (or axes for axis=None) are size 1

    Complexity
    ----------
    Time complexity  : O(1) 
    Space complexity : O(1)
    """
    if not keepdims:
        return result
    if original_axis is None:
        return result.reshape((1,) * original_ndim)
    return np.expand_dims(result, axis=original_axis)


def _welford(X: np.ndarray, 
    axis: Axis, 
    skipna: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes single-pass mean and M2 accumulation via Welford's online algorithm.

    Parameters
    ----------
    X : np.ndarray                - input array, converted by _to_float_array          
    axis : int or None            - reduction axis, None ravels X first (full reduction)
    skipna : bool                 - True: maintain a per-element count; NaN observations are masked and 
                                    do not update mean_ or M2 for that element
                                    False: NaN in a slice propagates NaN to output via floating-point arithmetic
    Returns
    -------
    mean_ : np.ndarray, shape=out_shape  - arithmetic mean of valid observations
    M2 : np.ndarray, shape=out_shape     - sum of squared deviations from the running mean
    count : np.ndarray, shape=out_shape  - number of valid (non-NaN) observations per output element
    
    Complexity
    ----------
    Time : O(N)
    Space : O(out_shape) for accumulators; O(1) extra per step
    """
    if axis is None:
        X = X.ravel()
        axis = 0

    X = np.moveaxis(X, axis, 0)
    out_shape = X.shape[1:]

    mean_ = np.zeros(out_shape, dtype=np.float64)
    M2 = np.zeros(out_shape, dtype=np.float64)

    if skipna:
        count = np.zeros(out_shape, dtype=np.float64)

        for x in X:
            mask = ~np.isnan(x)
            count = np.where(mask, count + 1.0, count)
            safe_n = np.where(count > 0.0, count, 1.0)
            delta = np.where(mask, x - mean_, 0.0)
            mean_ = np.where(mask, mean_ + delta / safe_n, mean_)
            delta2 = np.where(mask, x - mean_, 0.0)
            M2 = np.where(mask, M2 + delta * delta2, M2)
    else:
        n_obs = X.shape[0]
        count = np.full(out_shape if out_shape else (1,), float(n_obs), dtype=np.float64)

        for i, x in enumerate(X):
            n = float(i + 1)
            delta = x - mean_
            mean_ += delta / n
            delta2 = x - mean_
            M2 += delta * delta2

        if out_shape == ():
            count = count.squeeze()

    if out_shape == ():
        mean_ = mean_.squeeze()
        M2 = M2.squeeze()

    if skipna:
        mean_ = np.where(count > 0, mean_, np.nan)

    return mean_, M2, count


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
  
def mean(X: ArrayLike, 
    *, 
    axis: Axis = None, 
    keepdims: bool = False, 
    skipna: bool = True) -> np.ndarray:
    """
    Computes arithmetic mean via Welford's algorithm.

    Parameters
    ----------
    X : array-like                - input data 
    axis : int or None            - axis along which the mean is computed (None flattens X before reducing)
    keepdims : bool               - if True, reduced axis is kept as a size-1 dimension.
    skipna : bool                 - True (default) NaN values ignored, False NaN in slice propagates NaN to output.

    Returns
    -------
    np.ndarray:
        axis=None: scalar (0-d array).
        axis=k: shape equals X.shape with dimension k removed or kept as 1 if keepdims is True.

    Raises
    ------
    np.AxisError      - if axis out of range for the dimensionality of X
  
    Complexity
    ----------
    Time : O(N) 
    Space : O(out_shape)
    """
    X = _to_float_array(X)
    mean_, _, _ = _welford(X, axis, skipna)
    return _keepdims_result(mean_, axis, X.ndim, keepdims)


def median(X: ArrayLike, 
    *, 
    axis: Axis = None, 
    keepdims: bool = False, 
    skipna: bool = True) -> np.ndarray:
    """
    Computes median value.

    Parameters
    -----------
    X : array-like                - input data (converted to float64 internally)
    axis : int or None            - axis along which the median is computed (None flattens first)
    keepdims :  bool              - if True, reduced axis kept as size-1 dimension
    skipna : bool                 - True (default) NaN values ignored, False NaN propagates

    Returns
    -------
    np.ndarray: axis=None: scalar (0-d array), 
                axis=k: shape equals X.shape with dimension k removed or kept as 1 if keepdims True

    Raises
    ------
    np.AxisError      - if axis is out of range
   
    Complexity
    ----------
    Time : O(N log N) 
    Space : O(N) 
    """
    X = _to_float_array(X)
    result = np.nanmedian(X, axis=axis) if skipna else np.median(X, axis=axis)
    return _keepdims_result(result, axis, X.ndim, keepdims)


def std(X: ArrayLike,
    *, 
    axis: Axis = None, 
    ddof: int = 1, 
    keepdims: bool = False, 
    skipna: bool = True) -> np.ndarray:
    """
    Computes sample standard deviation.

    Parameters
    -----------
    X : array-like
    axis : int or None
    ddof : int                    - Delta degrees of freedom, ddof=1 (default) sample std; ddof=0 population std
    keepdims : bool
    skipna : bool

    Returns
    -------
    np.ndarray: axis=None: scalar (0-d array)
                axis=k: shape equals X.shape with dimension k removed or kept as 1 if keepdims is True
                Returns ``nan`` when count <= ddof

    Raises
    ------
    np.AxisError      - if axis is out of range

    Complexity
    ----------
    Time : O(N)
    Space : O(out_shape)
    """
    return np.sqrt(var(X, axis=axis, ddof=ddof, keepdims=keepdims, skipna=skipna))

def var(X: ArrayLike,
    *,
    axis: Axis = None,
    ddof: int = 1,
    keepdims: bool = False, 
    skipna: bool = True) -> np.ndarray:
    """
    Computes sample variance via Welford's algorithm.

    Parameters
    -----------
    X : array-like                - input data (converted to float64 internally)
    axis : int or None            - reduction axis, None flattens first
    ddof : int                    - Delta degrees of freedom: var = M2 / (count - ddof), ddof=1 (default) gives the unbiased sample variance
    keepdims : bool               - if True, reduced axis kept as size-1 dimension
    skipna : bool                 - if True (default) NaN values ignored, if False NaN propagates

    Returns
    -------
    np.ndarray: axis=None: scalar (0-d array)
                axis=k: shape equals X.shape with dimension k removed or kept as 1 if keepdims True
                Returns nan for any element where count <= ddof

    Raises
    ------
    np.AxisError      - if axis is out of range

    Complexity
    ----------
    Time : O(N) 
    Space : O(out_shape)
    """
    X = _to_float_array(X)
    _, M2, count = _welford(X, axis, skipna)
    denom  = np.where(count - ddof > 0, count - ddof, np.nan)
    result = M2 / denom
    return _keepdims_result(result, axis, X.ndim, keepdims)


def minimum(X: ArrayLike, 
    *, 
    axis: Axis = None, 
    keepdims: bool = False, 
    skipna: bool = True) -> np.ndarray:
    """
    Computes minimum value along axis.

    Parameters
    ---------- 
    X : array-like                - input data (converted to float64 internally)
    axis : int or None            - Reduction axis. None flattens first
    keepdims : bool               - if True, the reduced axis is kept as a size-1 dimension
    skipna : bool                 - if True (default) NaN values ignored, if False NaN propagates

    Returns
    -------
    np.ndarray: axis=None: scalar (0-d array)
                axis=k: shape equals X.shape with dimension k removed or kept as 1 if keepdims True
    
    Raises
    ------
    ValueError:       - if the slice is all-NaN (with skipna=True) or zero-size
    np.AxisError      - if axis is out of range

    Complexity
    ----------
    Time complexity  : O(N)
    Space complexity : O(out_shape)
    """
    X = _to_float_array(X)
    result = np.nanmin(X, axis=axis) if skipna else np.min(X, axis=axis)
    return _keepdims_result(result, axis, X.ndim, keepdims)

def maximum(X: ArrayLike, 
    *, 
    axis: Axis = None, 
    keepdims: bool = False, 
    skipna: bool = True) -> np.ndarray:
    """
    Computes maximum value along axis.

    Parameters
    ----------
    X : array-like                - input data (converted to float64 internally)
    axis : int or None            - reduction axis, None flattens first
    keepdims : bool               - if True, reduced axis kept as a size-1 dimension
    skipna : bool                 - if True (default), NaN values are ignored, if False, NaN propagates

    Returns
    ------- 
    np.ndarray: axis=None: scalar (0-d array)
                axis=k: shape equals X.shape with dimension k removed or kept as 1 if keepdims True
    
    Raises
    ------
    ValueError:       - if the slice is all-NaN (with skipna=True) or zero-size
    np.AxisError      - if axis is out of range

    Time : O(N)
    Space : O(out_shape)
    """
    X = _to_float_array(X)
    result = np.nanmax(X, axis=axis) if skipna else np.max(X, axis=axis)
    return _keepdims_result(result, axis, X.ndim, keepdims)

# ----------------------------------------------------------------------------
# Quantiles
# ----------------------------------------------------------------------------

def quantile(X: ArrayLike, 
    q: Union[float, Sequence[float]],
    *, 
    axis: Axis = None, 
    keepdims: bool = False, 
    interpolation: str = "linear", 
    skipna: bool = True) -> np.ndarray:
    """
    Computes quantile(s).

    Parameters
    ----------
    X : array-like                   - input data (converted to float64 internally)
    q : float or sequence of floats  - quantile level(s) in the closed interval [0, 1]
                                       A scalar q returns an array with the same shape as the reduced X; 
                                       a sequence of m values prepends a dimension of size m
    axis : int or None               - reduction axis, None flattens first
    keepdims : bool                  - if True, reduced data axis kept as size-1 dimension leading quantile 
                                       dimension (for sequence q) is unaffected
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
                                     - Method used when desired quantile falls between two data points, 
                                       for tied values all methods give the same result
    skipna (bool):                   - if True (default) NaN values ignored, if False NaN propagates


    Returns
    -------
        np.ndarray:
            Scalar q, axis=None: scalar (0-d array).
            Scalar q, axis=k: shape = X.shape with dim k removed (or kept as 1 with keepdims).
            Sequence q of length m: shape = (m,) + <above>.

    Raises
    ------
    ValueError:       - if any value in q is outside [0, 1], or if X contains no valid (non-NaN) values
                        along the reduction axis when skipna=True
    np.AxisError      - if axis is out of range

    Time complexity  : O(N log N)
    Space complexity : O(N)
    """
    X = _to_float_array(X)
    q = np.asarray(q, dtype=np.float64)
    if np.any((q < 0.0) | (q > 1.0)):
        raise ValueError("All quantile values must be in the range [0, 1].")

    valid = X[~np.isnan(X)] if skipna else X
    if valid.size == 0:
        out_val = np.full(q.shape, np.nan, dtype=np.float64)
        return _keepdims_result(out_val, axis, X.ndim, keepdims)

    try:
        if skipna:
            result = np.nanquantile(X, q, axis=axis, method=interpolation)
        else:
            result = np.quantile(X, q, axis=axis, method=interpolation)
    except TypeError:
        if skipna:
            result = np.nanquantile(X, q, axis=axis, interpolation=interpolation)
        else:
            result = np.quantile(X, q, axis=axis, interpolation=interpolation)

    return _keepdims_result(result, axis, X.ndim, keepdims)

# ----------------------------------------------------------------------------
# Histogram 
# ----------------------------------------------------------------------------

def histogram(X: ArrayLike,
    *, 
    bins: Union[int, Sequence, str] = 10, 
    range: Optional[tuple] = None, 
    density: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes Histogram with NaN values silently dropped before binning.

    Parameters
    ----------
    X : array-like                                - input data (converted to float64 internally), flattened to 1D internally,
                                                    NaN values are removed so that they do not contribute to counts or skew automatic 
                                                    bin-edge selection
    bins : int, sequence of floats, or str        - int: number of equal-width bins over [min, max] of valid data
                                                    sequence: explicit monotonically increasing bin edges of length n_bins + 1; range is ignored
                                                    str: NumPy strategy, e.g. 'auto', 'fd', 'sturges'.
    range : ((float, float) or None):             - (lower, upper) of the bins. Values outside of this range fall into the boundary bins. 
                                                    Ignored when bins is a sequence. Defaults to [min(X), max(X)].
    density : bool                                - if True, return probability density (counts / bin_width / total) 

    Returns
    -------
    counts : np.ndarray, shape (n_bins,)) - bin counts (integers) or density values (floats if density)
    bin_edges : np.ndarray of shape (n_bins + 1,)) - monotonically increasing bin edge values

    Raises
    ------ 
    ValueError:       - if X is empty or all-NaN

    Complexity
    ----------
    Time : O(N log N) for string strategies; O(N log B) for fixed bins 
    """
    X = _to_float_array(X).ravel()
    X = X[~np.isnan(X)]
    if X.size == 0:
        raise ValueError(
            "histogram requires at least one finite value; "
            "input is empty or all-NaN."
        )
    counts, bin_edges = np.histogram(X, bins=bins, range=range, density=density)
    return counts, bin_edges

# ----------------------------------------------------------------------------
# Summary 
# ----------------------------------------------------------------------------

def summary(X: ArrayLike, 
    *, 
    axis: Axis = None, 
    ddof: int = 1, 
    keepdims: bool = False, 
    skipna: bool = True) -> SummaryResult:
    """
    Computes descriptive statistics returned as a typed class SummaryResult named tuple.

    Fields: mean, median, std, var, min, max.

    Parameters
    ----------
    X : array-like                - input data (converted to float64 internally)
    axis : int or None            - Reduction axis, None flattens first
    ddof : int                    - Delta degrees of freedom, ddof=1 (default)sample std; ddof=0 population std
    keepdims (bool):              - if True, every field has reduced axis kept as size-1 dimension
    skipna (bool):                - if True (default), NaN values ignored

    Returns
    -------
    SummaryResult : Named tuple with fields mean, median, std, var, min, max. 
        Each field follows the shape rules:
            axis=None: scalar (0-d array)
            axis=k: shape equals X.shape with dimension k removed or kept as 1 if keepdims is True      

    Raises
    ------
    np.AxisError      - if axis is out of range

    Complexity
    ----------
    Time complexity : O(N log N) 
    Space complexity : O(N) 
    """
    X = _to_float_array(X)
    ndim = X.ndim

    mean_, M2, count = _welford(X, axis, skipna)
    denom = np.where(count - ddof > 0, count - ddof, np.nan)
    variance = M2 / denom

    mean_out = _keepdims_result(mean_, axis, ndim, keepdims)
    var_out = _keepdims_result(variance, axis, ndim, keepdims)
    std_out = _keepdims_result(np.sqrt(variance), axis, ndim, keepdims)

    med_raw = np.nanmedian(X, axis=axis)if skipna else np.median(X, axis=axis)
    min_raw = np.nanmin(X, axis=axis)if skipna else np.min(X, axis=axis)
    max_raw = np.nanmax(X, axis=axis)if skipna else np.max(X, axis=axis)

    return SummaryResult(
        mean = mean_out,
        median = _keepdims_result(med_raw, axis, ndim, keepdims),
        std = std_out,
        var = var_out,
        min = _keepdims_result(min_raw, axis, ndim, keepdims),
        max = _keepdims_result(max_raw, axis, ndim, keepdims),
    )