"""
rank - Ranking with tie handling and percentile computation.

Vectorised ranking and percentile utilities built on NumPy.
Supports three tie-breaking methods (average, dense, ordinal) and
four interpolation strategies for percentiles.

Public API
----------
rank       - Assign ranks to elements with configurable tie handling.
percentile - Compute arbitrary percentiles with interpolation.

Examples
--------
>>> import numpy as np
>>> from numcompute.rank import rank, percentile
>>> rank(np.array([40, 10, 30, 10, 20]))
array([5. , 1.5, 4. , 1.5, 3. ])
>>> percentile(np.array([1, 2, 3, 4, 5]), 50)
3.0
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np

__all__ = [
    "rank",
    "percentile",
]



# Internal helpers
def _ensure_1d_numeric(a, name: str = "data") -> np.ndarray:
    """Convert `a` to a 1D float ndarray. Raises on bad input.

    Parameters
    ----------
    a : array_like
        Input data.
    name : str
        Label for error messages.

    Returns
    -------
    np.ndarray, 1D

    Raises
    ------
    TypeError
        If `a` can't be cast to a numeric ndarray.
    ValueError
        If result isn't 1D or has 0 elements.
    """
    try:
        arr = np.asarray(a, dtype=float)
    except (ValueError, TypeError) as exc:
        raise TypeError(
            f"`{name}` must be numeric array-like, got {type(a).__name__}."
        ) from exc
    if arr.ndim != 1:
        raise ValueError(
            f"`{name}` must be 1-D, got {arr.ndim}-D array with shape {arr.shape}."
        )
    if arr.size == 0:
        raise ValueError(f"`{name}` must be non-empty.")
    return arr



# Ranking
_RANK_METHODS = {"average", "dense", "ordinal"}


def rank(
    data,
    method: Literal["average", "dense", "ordinal"] = "average",
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> np.ndarray:
    """Assign numerical ranks to elements of a 1D array.

    Parameters
    ----------
    data : array_like, 1D
        Input values to rank.
    method : {'average', 'dense', 'ordinal'}, optional
        How to handle tied (equal) values:
        - 'average': tied values get the mean of their would-be ranks.
          e.g. two items sharing ranks 2 and 3 both get 2.5.
          This is the default, matching scipy.stats.rankdata behaviour.
        - 'dense': tied values get the same rank; the next distinct
          value gets rank+1 (no gaps in the sequence).
        - 'ordinal': every element gets a unique rank; ties are broken
          by position in the original array (stable sort order).
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        - 'propagate': NaN values receive rank NaN.
        - 'omit': NaN values are excluded from ranking; non-NaN elements
          are ranked as if the NaNs weren't there. Output shape matches
          input, with NaN positions filled with NaN.
        - 'raise': raise ValueError if any NaN is found.

    Returns
    -------
    np.ndarray, shape (n,), dtype float64
        Ranks starting at 1.

    Raises
    ------
    TypeError
        If `data` isn't numeric.
    ValueError
        If empty, not 1D, invalid method, or NaN with raise policy.

    Notes
    -----
    - Time:  O(n log n), dominated by the sort
    - Space: O(n)
    - Fully vectorised, no Python-level loops over elements

    Examples
    --------
    >>> rank([40, 10, 30, 10, 20])
    array([5. , 1.5, 4. , 1.5, 3. ])

    >>> rank([40, 10, 30, 10, 20], method='dense')
    array([4., 1., 3., 1., 2.])

    >>> rank([40, 10, 30, 10, 20], method='ordinal')
    array([5., 1., 4., 2., 3.])

    >>> rank([1.0, np.nan, 3.0], nan_policy='propagate')
    array([ 1., nan,  2.])
    """
    if method not in _RANK_METHODS:
        raise ValueError(
            f"`method` must be one of {_RANK_METHODS!r}, got {method!r}."
        )
    if nan_policy not in {"propagate", "omit", "raise"}:
        raise ValueError(
            f"`nan_policy` must be 'propagate', 'omit', or 'raise', "
            f"got {nan_policy!r}."
        )

    arr = _ensure_1d_numeric(data, "data")
    n = arr.shape[0]

    # --- Handle NaN values ---
    nan_mask = np.isnan(arr)
    has_nan = nan_mask.any()

    if has_nan and nan_policy == "raise":
        raise ValueError("`data` contains NaN values.")

    if has_nan and nan_policy == "omit":
        # Rank only non-NaN elements, scatter results back
        non_nan_idx = np.flatnonzero(~nan_mask)
        result = np.full(n, np.nan, dtype=np.float64)
        if non_nan_idx.size > 0:
            sub_ranks = _rank_core(arr[non_nan_idx], method)
            result[non_nan_idx] = sub_ranks
        return result

    # Either no NaN, or propagate mode
    result = _rank_core(arr, method)

    if has_nan:
        # Set NaN positions to NaN rank
        result[nan_mask] = np.nan

    return result


def _rank_core(arr: np.ndarray, method: str) -> np.ndarray:
    """Core ranking logic for a clean (no NaN) 1D array.

    Parameters
    ----------
    arr : np.ndarray, 1D
        NaN-free numeric array.
    method : str
        One of 'average', 'dense', 'ordinal'.

    Returns
    -------
    np.ndarray, shape (n,), dtype float64
    """
    n = arr.shape[0]
    # Stable argsort keeps ties in original order
    sorted_idx = np.argsort(arr, kind="stable")

    if method == "ordinal":
        # Just assign 1, 2, 3, ... based on sorted position
        ranks = np.empty(n, dtype=np.float64)
        ranks[sorted_idx] = np.arange(1, n + 1, dtype=np.float64)
        return ranks

    sorted_vals = arr[sorted_idx]

    if method == "average":
        # For each group of equal values, replace their ordinal ranks
        # with the group mean rank.
        ordinal_ranks = np.arange(1, n + 1, dtype=np.float64)

        # Find where consecutive sorted values differ (group boundaries)
        diff = np.concatenate(([True], sorted_vals[1:] != sorted_vals[:-1]))
        group_ids = np.cumsum(diff) - 1  # label groups 0, 1, 2, ...

        # Sum ranks per group and divide by group size -> average rank
        group_sum = np.bincount(group_ids, weights=ordinal_ranks)
        group_count = np.bincount(group_ids)
        group_avg = group_sum / group_count

        # Map back from sorted order to original positions
        avg_for_each = group_avg[group_ids]
        ranks = np.empty(n, dtype=np.float64)
        ranks[sorted_idx] = avg_for_each
        return ranks

    if method == "dense":
        # Ties get same rank, next distinct value gets rank+1 (no gaps)
        diff = np.concatenate(([True], sorted_vals[1:] != sorted_vals[:-1]))
        dense_ranks = np.cumsum(diff).astype(np.float64)
        ranks = np.empty(n, dtype=np.float64)
        ranks[sorted_idx] = dense_ranks
        return ranks

    # Shouldn't get here, but just in case
    raise ValueError(f"Unknown method {method!r}")  # pragma: no cover



# Percentile
_INTERP_METHODS = {"linear", "lower", "higher", "midpoint"}


def percentile(
    data,
    q: Union[float, int, list, np.ndarray],
    interpolation: Literal["linear", "lower", "higher", "midpoint"] = "linear",
    nan_policy: Literal["propagate", "omit", "raise"] = "omit",
) -> Union[float, np.ndarray]:
    """Compute the q-th percentile(s) of a 1D dataset.

    Parameters
    ----------
    data : array_like, 1D
        Input values.
    q : float, int, or array_like of float/int
        Percentile(s) to compute, each in [0, 100].
    interpolation : {'linear', 'lower', 'higher', 'midpoint'}, optional
        What to do when the desired percentile falls between two data
        points i and j:
        - 'linear':   i + (j - i) * fraction   (default)
        - 'lower':    just use i
        - 'higher':   just use j
        - 'midpoint': (i + j) / 2
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        - 'propagate': any NaN in data -> result is NaN
        - 'omit': drop NaN values before computing (default)
        - 'raise': raise ValueError if NaN is present

    Returns
    -------
    float or np.ndarray
        Scalar when q is scalar, array when q is array-like.

    Raises
    ------
    TypeError
        If `data` isn't numeric.
    ValueError
        If empty, not 1D, q outside [0, 100], invalid interpolation,
        or NaN with raise policy.

    Notes
    -----
    - Time:  O(n log n) for sorting
    - Space: O(n)
    - Uses the standard C=1 indexing scheme (same as np.percentile default)

    Examples
    --------
    >>> percentile([1, 2, 3, 4, 5], 50)
    3.0
    >>> percentile([1, 2, 3, 4, 5], [25, 50, 75])
    array([2., 3., 4.])
    >>> percentile([1, 2, 3, 4, 5], 30, interpolation='lower')
    2.0
    """
    if interpolation not in _INTERP_METHODS:
        raise ValueError(
            f"`interpolation` must be one of {_INTERP_METHODS!r}, "
            f"got {interpolation!r}."
        )
    if nan_policy not in {"propagate", "omit", "raise"}:
        raise ValueError(
            f"`nan_policy` must be 'propagate', 'omit', or 'raise', "
            f"got {nan_policy!r}."
        )

    arr = _ensure_1d_numeric(data, "data")

    # Validate q values
    q_arr = np.asarray(q, dtype=np.float64)
    scalar_q = q_arr.ndim == 0
    q_arr = np.atleast_1d(q_arr)
    if np.any((q_arr < 0) | (q_arr > 100)):
        raise ValueError("`q` values must be in the range [0, 100].")

    # --- Handle NaN values ---
    nan_mask = np.isnan(arr)
    has_nan = nan_mask.any()

    if has_nan:
        if nan_policy == "raise":
            raise ValueError("`data` contains NaN values.")
        elif nan_policy == "propagate":
            result = np.full(q_arr.shape, np.nan, dtype=np.float64)
            return float(result[0]) if scalar_q else result
        else:  # omit
            arr = arr[~nan_mask]
            if arr.size == 0:
                raise ValueError(
                    "All elements are NaN; cannot compute percentile."
                )

    sorted_arr = np.sort(arr, kind="stable")
    n = sorted_arr.shape[0]

    # Map percentile to a virtual index in [0, n-1]
    virtual_idx = q_arr / 100.0 * (n - 1)

    lower_idx = np.floor(virtual_idx).astype(int)
    upper_idx = np.ceil(virtual_idx).astype(int)
    lower_idx = np.clip(lower_idx, 0, n - 1)
    upper_idx = np.clip(upper_idx, 0, n - 1)

    lower_vals = sorted_arr[lower_idx]
    upper_vals = sorted_arr[upper_idx]

    if interpolation == "linear":
        fraction = virtual_idx - np.floor(virtual_idx)
        result = lower_vals + (upper_vals - lower_vals) * fraction
    elif interpolation == "lower":
        result = lower_vals.astype(np.float64)
    elif interpolation == "higher":
        result = upper_vals.astype(np.float64)
    elif interpolation == "midpoint":
        result = (lower_vals + upper_vals) / 2.0

    if scalar_q:
        return float(result[0])
    return result
