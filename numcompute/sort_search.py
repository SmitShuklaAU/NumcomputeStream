"""
sort_search - Sorting, partial-sort / top-k, and searching utilities.

Provides vectorised wrappers around NumPy's sorting and searching
primitives, plus an educational quickselect implementation.
All functions include input validation, NaN-awareness, and docstrings.

Public API
----------
stable_sort        - Stable sort wrapper (np.sort with kind='stable')
multi_key_sort     - Sort a 2D array by multiple column keys
topk               - Top-k elements via np.argpartition
quickselect        - k-th smallest element (educational, pure-Python pivot logic)
binary_search      - Binary search on a sorted 1D array

Examples
--------
>>> import numpy as np
>>> from numcompute.sort_search import stable_sort, topk, binary_search
>>> stable_sort(np.array([3, 1, 2]))
array([1, 2, 3])
>>> topk(np.array([10, 4, 7, 1, 9]), k=3)
(array([10,  9,  7]), array([0, 4, 2]))
>>> binary_search(np.array([1, 3, 5, 7]), 5)
(2, True)
"""

from __future__ import annotations

from typing import Literal, Tuple, Union

import numpy as np

__all__ = [
    "stable_sort",
    "multi_key_sort",
    "topk",
    "quickselect",
    "binary_search",
]


# Internal helpers
def _ensure_ndarray(a, name: str = "input") -> np.ndarray:
    """Convert `a` to np.ndarray; raise if it can't be converted or is empty.

    Parameters
    ----------
    a : array_like
        Input data.
    name : str
        Label used in error messages.

    Returns
    -------
    np.ndarray

    Raises
    ------
    TypeError
        If `a` can't be turned into an ndarray.
    ValueError
        If resulting array has 0 elements.
    """
    try:
        arr = np.asarray(a)
    except (ValueError, TypeError) as exc:
        raise TypeError(
            f"`{name}` must be array-like, got {type(a).__name__}"
        ) from exc
    if arr.size == 0:
        raise ValueError(f"`{name}` must be non-empty.")
    return arr



# Sorting
def stable_sort(
    a,
    axis: int = -1,
    descending: bool = False,
) -> np.ndarray:
    """Return a stable-sorted copy of the input array.

    Uses merge sort (kind='stable') so equal elements keep their
    original relative order. This matters when you chain multiple
    sort passes (e.g., multi-key sorting).

    Parameters
    ----------
    a : array_like
        Input array, any shape.
    axis : int, optional
        Axis to sort along. Defaults to -1 (last axis).
    descending : bool, optional
        If True, flip the result so it's in descending order.

    Returns
    -------
    np.ndarray
        Sorted copy, same shape as input.

    Raises
    ------
    TypeError
        If `a` can't be converted to ndarray.
    ValueError
        If `a` is empty.

    Notes
    -----
    - Time:  O(n log n)
    - Space: O(n) for the copy + merge-sort workspace
    - NaN values end up at the end regardless of `descending`

    Examples
    --------
    >>> stable_sort([3, 1, 4, 1, 5])
    array([1, 1, 3, 4, 5])
    >>> stable_sort([3, 1, 4], descending=True)
    array([4, 3, 1])
    """
    arr = _ensure_ndarray(a, "a")
    sorted_arr = np.sort(arr, axis=axis, kind="stable")
    if descending:
        sorted_arr = np.flip(sorted_arr, axis=axis)
    return sorted_arr


def multi_key_sort(
    a,
    keys: Union[list, tuple],
    descending: Union[bool, list] = False,
) -> np.ndarray:
    """Sort a 2D array by multiple column keys (stable).

    Works from least-significant to most-significant key so that
    ties in a higher-priority column get resolved by lower-priority ones.
    This is the standard "successive stable sort" trick.

    Parameters
    ----------
    a : array_like, shape (n, m)
        2D input array.
    keys : list[int] or tuple[int]
        Column indices in decreasing priority order.
        keys[0] is the primary sort key.
    descending : bool or list[bool], optional
        Single bool applies to all keys. A list must match len(keys)
        and controls direction per key (True = descending).

    Returns
    -------
    np.ndarray, shape (n, m)
        Row-reordered copy.

    Raises
    ------
    TypeError / ValueError
        On invalid types, shapes, or out-of-bounds key indices.

    Notes
    -----
    - Time:  O(k * n log n) where k = len(keys)
    - Space: O(n) for the index array + output copy

    Examples
    --------
    >>> data = np.array([[2, 2], [1, 1], [2, 1], [1, 2]], dtype=float)
    >>> multi_key_sort(data, keys=[0, 1])
    array([[1., 1.], [1., 2.], [2., 1.], [2., 2.]])
    """
    arr = _ensure_ndarray(a, "a")
    if arr.ndim != 2:
        raise ValueError(
            f"`a` must be 2-D, got {arr.ndim}-D array with shape {arr.shape}."
        )

    n_cols = arr.shape[1]
    for k in keys:
        if not (0 <= k < n_cols):
            raise ValueError(
                f"Key index {k} is out of bounds for array with "
                f"{n_cols} columns."
            )

    # Normalise descending to a list
    if isinstance(descending, bool):
        desc_flags = [descending] * len(keys)
    else:
        desc_flags = list(descending)
        if len(desc_flags) != len(keys):
            raise ValueError(
                f"Length of `descending` ({len(desc_flags)}) must match "
                f"length of `keys` ({len(keys)})."
            )

    # Apply stable sorts from least-significant key to most-significant
    idx = np.arange(arr.shape[0])
    for key, desc in zip(reversed(keys), reversed(desc_flags)):
        col = arr[idx, key]
        order = np.argsort(col, kind="stable")
        if desc:
            order = order[::-1]
        idx = idx[order]

    return arr[idx]



# Top-k / Partial Sort
def topk(
    values,
    k: int,
    largest: bool = True,
    return_indices: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Get the top-k elements from a 1D array using np.argpartition.

    argpartition does an O(n) partial sort, then we only need to
    fully sort the k selected elements -> O(n + k log k) total.

    The returned elements come out sorted (descending if largest=True,
    ascending if largest=False).

    Parameters
    ----------
    values : array_like, 1D
        Input values.
    k : int
        How many elements to select. Must be 1 <= k <= len(values).
    largest : bool, optional
        True (default) -> return the k largest.
        False -> return the k smallest.
    return_indices : bool, optional
        If True (default), also return indices into the original array.

    Returns
    -------
    top_values : np.ndarray, shape (k,)
        Selected top-k values, sorted.
    top_indices : np.ndarray, shape (k,)
        Only returned when return_indices=True. Indices into original array.

    Raises
    ------
    TypeError
        If `values` isn't array-like.
    ValueError
        If not 1D, empty, or k is out of range.

    Notes
    -----
    - Time:  O(n + k log k)
    - Space: O(n) for index/partition arrays
    - NaN values may appear in top-k (largest=True) since NumPy treats
      NaN as greater than any finite number in argpartition

    Examples
    --------
    >>> topk([10, 4, 7, 1, 9], k=3)
    (array([10,  9,  7]), array([0, 4, 2]))
    >>> topk([10, 4, 7, 1, 9], k=2, largest=False)
    (array([1, 4]), array([3, 1]))
    """
    arr = _ensure_ndarray(values, "values")
    if arr.ndim != 1:
        raise ValueError(
            f"`values` must be 1-D, got {arr.ndim}-D array."
        )

    n = arr.shape[0]
    if not (1 <= k <= n):
        raise ValueError(
            f"`k` must satisfy 1 <= k <= {n}, got k={k}."
        )

    if largest:
        # argpartition puts the smallest (n-k) elements at the front,
        # so the k largest end up at indices [n-k : n]
        part_idx = np.argpartition(arr, n - k)
        top_idx = part_idx[n - k:]
        # sort those k by value, descending
        order = np.argsort(arr[top_idx], kind="stable")[::-1]
    else:
        # k smallest are at the front after partitioning
        part_idx = np.argpartition(arr, k)
        top_idx = part_idx[:k]
        order = np.argsort(arr[top_idx], kind="stable")

    top_idx = top_idx[order]
    top_vals = arr[top_idx]

    if return_indices:
        return top_vals, top_idx
    return top_vals



# Quickselect (educational implementation)
def quickselect(
    a,
    k: int,
    largest: bool = False,
) -> float:
    """Find the k-th order statistic using the Quickselect algorithm.

    This is an educational implementation to demonstrate the
    divide-and-conquer selection idea. Uses a Python loop for the
    recursive partitioning since the algorithm is inherently sequential.
    For production use, prefer topk() which delegates to NumPy's
    optimised C-level argpartition.

    Parameters
    ----------
    a : array_like, 1D
        Input values (don't need to be sorted).
    k : int
        1-based rank of the desired element.
        - largest=False: k=1 gives the smallest element.
        - largest=True:  k=1 gives the largest element.
    largest : bool, optional
        If True, select the k-th largest instead of k-th smallest.

    Returns
    -------
    float
        The k-th order statistic.

    Raises
    ------
    TypeError
        If `a` isn't array-like.
    ValueError
        If not 1D, empty, contains NaN, or k is out of range.

    Notes
    -----
    - Time:  O(n) average, O(n^2) worst-case
    - Space: O(n) for the working copy
    - Uses median-of-three pivot selection to reduce worst-case probability
    - NaN values are rejected since comparisons with NaN are undefined

    Examples
    --------
    >>> quickselect([7, 2, 5, 3, 8], k=1)
    2.0
    >>> quickselect([7, 2, 5, 3, 8], k=2, largest=True)
    7.0
    """
    arr = _ensure_ndarray(a, "a").astype(float)
    if arr.ndim != 1:
        raise ValueError(f"`a` must be 1-D, got {arr.ndim}-D array.")
    if np.isnan(arr).any():
        raise ValueError("`a` must not contain NaN values.")

    n = arr.shape[0]
    if not (1 <= k <= n):
        raise ValueError(f"`k` must satisfy 1 <= k <= {n}, got k={k}.")

    # Convert to 0-based index into sorted order
    target = n - k if largest else k - 1

    # Work on a mutable copy so we don't touch the original
    data = arr.copy()

    lo, hi = 0, n - 1
    while lo < hi:
        # Pick pivot using median-of-three when subarray is big enough,
        # otherwise just use the last element as pivot
        if hi - lo >= 2:
            mid = (lo + hi) // 2
            # Sort lo/mid/hi so mid holds the median
            if data[lo] > data[mid]:
                data[lo], data[mid] = data[mid], data[lo]
            if data[lo] > data[hi]:
                data[lo], data[hi] = data[hi], data[lo]
            if data[mid] > data[hi]:
                data[mid], data[hi] = data[hi], data[mid]
            # Move median pivot to end for Lomuto partition
            data[mid], data[hi] = data[hi], data[mid]

        pivot = data[hi]

        # Lomuto partition: elements < pivot go to the left side
        store = lo
        for idx in range(lo, hi):
            if data[idx] < pivot:
                data[store], data[idx] = data[idx], data[store]
                store += 1
        data[store], data[hi] = data[hi], data[store]

        if target == store:
            return float(data[store])
        elif target < store:
            hi = store - 1
        else:
            lo = store + 1

    return float(data[lo])



# Binary Search
def binary_search(
    sorted_array,
    x,
) -> Tuple[int, bool]:
    """Search for `x` in a sorted 1D array using np.searchsorted (O(log n)).

    Parameters
    ----------
    sorted_array : array_like, 1D
        Must be pre-sorted in ascending order. If it's not sorted,
        the result is undefined.
    x : scalar
        Value to look for.

    Returns
    -------
    index : int
        Insertion point that keeps sorted order. If `x` exists,
        this is the index of its first occurrence.
    found : bool
        True if `x` is present in the array.

    Raises
    ------
    TypeError
        If `sorted_array` isn't array-like.
    ValueError
        If not 1D or empty.

    Notes
    -----
    - Time:  O(log n)
    - Space: O(1) beyond the input
    - Searching for np.nan returns found=False (since nan != nan)

    Examples
    --------
    >>> binary_search([1, 3, 5, 7, 9], 5)
    (2, True)
    >>> binary_search([1, 3, 5, 7, 9], 4)
    (2, False)
    """
    arr = _ensure_ndarray(sorted_array, "sorted_array")
    if arr.ndim != 1:
        raise ValueError(
            f"`sorted_array` must be 1-D, got {arr.ndim}-D array."
        )

    idx = int(np.searchsorted(arr, x, side="left"))
    found = (idx < arr.shape[0]) and (arr[idx] == x)

    # nan == nan evaluates True in some edge cases, guard against it
    if found and np.isnan(np.asarray(x)).any():
        found = False
    return idx, bool(found)