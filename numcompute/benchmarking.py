"""
benchmarking - Micro-benchmark harness for NumCompute.

Compares vectorised NumPy implementations in sort_search, rank,
and stats against equivalent pure-Python loop implementations to
show the performance gains of vectorisation.

Usage
-----
Run directly:
    python -m numcompute.benchmarking

Or import and use in a notebook / script:
    from numcompute.benchmarking import run_all, benchmark_stable_sort
    run_all()

Output
------
Prints a formatted performance table showing:
    - Operation name
    - Input size (n)
    - Python loop time (ms)
    - NumPy vectorised time (ms)
    - Speedup factor (x)
"""

from __future__ import annotations

import platform
import sys
import time
from typing import Callable

import numpy as np

from numcompute.sort_search import stable_sort, topk, binary_search
from numcompute.rank import rank, percentile

# stats module may not be merged yet — import conditionally
try:
    from numcompute.stats import mean as stats_mean, var as stats_var
    from numcompute.stats import median as stats_median, histogram as stats_histogram
    _HAS_STATS = True
except ImportError:
    _HAS_STATS = False


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------

def _time_fn(fn: Callable, *args, repeats: int = 5, warmup: int = 1,
             **kwargs) -> float:
    """Run fn(*args, **kwargs) several times and return the median
    wall-clock time in milliseconds.

    A few warmup calls are done first so the first timed run isn't
    penalised by import overhead, JIT compilation in BLAS, etc.

    Parameters
    ----------
    fn : Callable
        Function to time.
    *args
        Positional args forwarded to fn.
    repeats : int
        Number of timed runs (default 5). Median is reported.
    warmup : int
        Number of untimed warmup calls (default 1).
    **kwargs
        Keyword args forwarded to fn.

    Returns
    -------
    float
        Median execution time in milliseconds.
    """
    # warmup runs (not timed)
    for _ in range(warmup):
        fn(*args, **kwargs)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.median(times)) * 1000.0


def _print_table(results: list[dict]) -> None:
    """Print benchmark results as a nicely formatted table.

    Parameters
    ----------
    results : list[dict]
        Each dict has keys:
        'operation', 'n', 'loop_ms', 'numpy_ms', 'speedup'
    """
    header = (f"{'Operation':<30} {'n':>10} "
              f"{'Loop (ms)':>12} {'NumPy (ms)':>12} {'Speedup':>10}")
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['operation']:<30} "
            f"{r['n']:>10,} "
            f"{r['loop_ms']:>12.4f} "
            f"{r['numpy_ms']:>12.4f} "
            f"{r['speedup']:>9.1f}x"
        )
    print(sep + "\n")


def _print_env() -> None:
    """Print environment info for reproducibility."""
    print("=" * 65)
    print("  NumCompute Benchmark Suite")
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  NumPy       : {np.__version__}")
    print(f"  Platform    : {platform.system()} {platform.machine()}")
    print(f"  Processor   : {platform.processor() or 'unknown'}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Pure-Python loop baselines (intentionally naive)
# ---------------------------------------------------------------------------

def _loop_sort(arr: np.ndarray) -> list:
    """Insertion sort in pure Python. O(n^2) but faster than bubble sort
    so the benchmark finishes in a reasonable time on moderate n."""
    data = arr.tolist()
    n = len(data)
    for i in range(1, n):
        key = data[i]
        j = i - 1
        while j >= 0 and data[j] > key:
            data[j + 1] = data[j]
            j -= 1
        data[j + 1] = key
    return data


def _loop_topk(arr: np.ndarray, k: int) -> list:
    """Top-k via full Python sort + slice. O(n log n)."""
    return sorted(arr.tolist(), reverse=True)[:k]


def _loop_binary_search(arr: np.ndarray, x: float) -> tuple[int, bool]:
    """Standard binary search with a while loop."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == x:
            return mid, True
        elif arr[mid] < x:
            lo = mid + 1
        else:
            hi = mid - 1
    return lo, False


def _loop_rank(arr: np.ndarray) -> list:
    """Average ranking using nested loops. O(n^2)."""
    n = len(arr)
    ranks = []
    for i in range(n):
        r = 0
        count = 0
        for j in range(n):
            if arr[j] < arr[i]:
                r += 1
            elif arr[j] == arr[i]:
                r += 1
                count += 1
        ranks.append(r - count / 2 + 0.5)
    return ranks


def _loop_mean(arr: np.ndarray) -> float:
    """Arithmetic mean via a plain Python loop. O(n)."""
    total = 0.0
    for val in arr.tolist():
        total += val
    return total / len(arr)


def _loop_var(arr: np.ndarray) -> float:
    """Sample variance via two-pass Python loop. O(n)."""
    data = arr.tolist()
    n = len(data)
    m = sum(data) / n
    ss = 0.0
    for val in data:
        ss += (val - m) ** 2
    return ss / (n - 1)


def _loop_median(arr: np.ndarray) -> float:
    """Median via Python sort. O(n log n)."""
    s = sorted(arr.tolist())
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return float(s[mid])


def _loop_histogram(arr: np.ndarray, n_bins: int = 10) -> list:
    """Histogram via a Python loop over elements. O(n * bins)."""
    data = arr.tolist()
    lo, hi = min(data), max(data)
    width = (hi - lo) / n_bins if hi != lo else 1.0
    counts = [0] * n_bins
    for val in data:
        idx = int((val - lo) / width)
        idx = min(idx, n_bins - 1)  # clamp rightmost edge
        counts[idx] += 1
    return counts


def _loop_percentile(arr: np.ndarray, q: float) -> float:
    """Percentile via Python sort + linear interpolation."""
    sorted_data = sorted(arr.tolist())
    n = len(sorted_data)
    idx = (q / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * frac


# ---------------------------------------------------------------------------
# Individual benchmarks
#
# Each returns a dict with: operation, n, loop_ms, numpy_ms, speedup.
# Speedup is always computed on the SAME input size for a fair comparison.
# For O(n^2) loop baselines, we cap n to keep runtime reasonable.
# ---------------------------------------------------------------------------

def benchmark_stable_sort(n: int = 50_000, repeats: int = 5) -> dict:
    """Benchmark stable_sort vs insertion sort.

    Parameters
    ----------
    n : int
        Array size. The loop baseline caps at min(n, 5000) because
        insertion sort is O(n^2).
    repeats : int
        Timed repetitions.

    Returns
    -------
    dict
    """
    rng = np.random.default_rng(42)
    arr = rng.random(n)

    # Cap the loop size so the O(n^2) sort doesn't take forever
    loop_n = min(n, 5_000)
    small = arr[:loop_n].copy()

    loop_ms = _time_fn(_loop_sort, small, repeats=repeats)
    numpy_ms = _time_fn(stable_sort, small, repeats=repeats)
    speedup = loop_ms / numpy_ms if numpy_ms > 0 else float("inf")

    return {
        "operation": f"stable_sort (n={loop_n})",
        "n": loop_n,
        "loop_ms": loop_ms,
        "numpy_ms": numpy_ms,
        "speedup": speedup,
    }


def benchmark_topk(n: int = 100_000, k: int = 10, repeats: int = 5) -> dict:
    """Benchmark topk (argpartition) vs Python sorted()[:k].

    Parameters
    ----------
    n : int
        Array length.
    k : int
        Number of top elements.
    repeats : int
        Timed repetitions.

    Returns
    -------
    dict
    """
    rng = np.random.default_rng(42)
    arr = rng.random(n)

    loop_ms = _time_fn(_loop_topk, arr, k, repeats=repeats)
    numpy_ms = _time_fn(topk, arr, k, repeats=repeats)
    speedup = loop_ms / numpy_ms if numpy_ms > 0 else float("inf")

    return {
        "operation": f"topk (k={k})",
        "n": n,
        "loop_ms": loop_ms,
        "numpy_ms": numpy_ms,
        "speedup": speedup,
    }


def benchmark_binary_search(n: int = 1_000_000, repeats: int = 50) -> dict:
    """Benchmark binary_search (np.searchsorted) vs Python while loop.

    Uses more repeats since each individual call is sub-millisecond.

    Parameters
    ----------
    n : int
        Sorted array length.
    repeats : int
        Timed repetitions.

    Returns
    -------
    dict
    """
    arr = np.sort(np.random.default_rng(42).random(n))
    x = float(arr[n // 2])  # pick a value we know exists

    loop_ms = _time_fn(_loop_binary_search, arr, x, repeats=repeats)
    numpy_ms = _time_fn(binary_search, arr, x, repeats=repeats)
    speedup = loop_ms / numpy_ms if numpy_ms > 0 else float("inf")

    return {
        "operation": "binary_search",
        "n": n,
        "loop_ms": loop_ms,
        "numpy_ms": numpy_ms,
        "speedup": speedup,
    }


def benchmark_rank(n: int = 10_000, repeats: int = 5) -> dict:
    """Benchmark rank (vectorised) vs nested-loop ranking.

    Parameters
    ----------
    n : int
        Array size. Loop baseline caps at min(n, 2000) since it's O(n^2).
    repeats : int
        Timed repetitions.

    Returns
    -------
    dict
    """
    rng = np.random.default_rng(42)
    arr = rng.random(n)

    # O(n^2) loop — keep it small
    loop_n = min(n, 2_000)
    small = arr[:loop_n].copy()

    loop_ms = _time_fn(_loop_rank, small, repeats=repeats)
    numpy_ms = _time_fn(rank, small, repeats=repeats)
    speedup = loop_ms / numpy_ms if numpy_ms > 0 else float("inf")

    return {
        "operation": f"rank average (n={loop_n})",
        "n": loop_n,
        "loop_ms": loop_ms,
        "numpy_ms": numpy_ms,
        "speedup": speedup,
    }


def benchmark_percentile(n: int = 100_000, repeats: int = 5) -> dict:
    """Benchmark percentile (vectorised) vs Python sort + interpolation.

    Parameters
    ----------
    n : int
        Array length.
    repeats : int
        Timed repetitions.

    Returns
    -------
    dict
    """
    rng = np.random.default_rng(42)
    arr = rng.random(n)
    q = 75.0

    loop_ms = _time_fn(_loop_percentile, arr, q, repeats=repeats)
    numpy_ms = _time_fn(percentile, arr, q, repeats=repeats)
    speedup = loop_ms / numpy_ms if numpy_ms > 0 else float("inf")

    return {
        "operation": "percentile (q=75)",
        "n": n,
        "loop_ms": loop_ms,
        "numpy_ms": numpy_ms,
        "speedup": speedup,
    }


def benchmark_stats_mean(n: int = 100_000, repeats: int = 5) -> dict:
    """Benchmark stats.mean (Welford) vs pure-Python loop mean.

    Parameters
    ----------
    n : int
        Array length.
    repeats : int
        Timed repetitions.

    Returns
    -------
    dict
    """
    rng = np.random.default_rng(42)
    arr = rng.random(n)

    loop_ms = _time_fn(_loop_mean, arr, repeats=repeats)
    numpy_ms = _time_fn(stats_mean, arr, repeats=repeats)
    speedup = loop_ms / numpy_ms if numpy_ms > 0 else float("inf")

    return {
        "operation": "stats.mean (Welford)",
        "n": n,
        "loop_ms": loop_ms,
        "numpy_ms": numpy_ms,
        "speedup": speedup,
    }


def benchmark_stats_var(n: int = 100_000, repeats: int = 5) -> dict:
    """Benchmark stats.var (Welford) vs two-pass Python loop variance.

    Parameters
    ----------
    n : int
        Array length.
    repeats : int
        Timed repetitions.

    Returns
    -------
    dict
    """
    rng = np.random.default_rng(42)
    arr = rng.random(n)

    loop_ms = _time_fn(_loop_var, arr, repeats=repeats)
    numpy_ms = _time_fn(stats_var, arr, repeats=repeats)
    speedup = loop_ms / numpy_ms if numpy_ms > 0 else float("inf")

    return {
        "operation": "stats.var (Welford)",
        "n": n,
        "loop_ms": loop_ms,
        "numpy_ms": numpy_ms,
        "speedup": speedup,
    }


def benchmark_stats_median(n: int = 100_000, repeats: int = 5) -> dict:
    """Benchmark stats.median (np.nanmedian) vs Python sorted median.

    Parameters
    ----------
    n : int
        Array length.
    repeats : int
        Timed repetitions.

    Returns
    -------
    dict
    """
    rng = np.random.default_rng(42)
    arr = rng.random(n)

    loop_ms = _time_fn(_loop_median, arr, repeats=repeats)
    numpy_ms = _time_fn(stats_median, arr, repeats=repeats)
    speedup = loop_ms / numpy_ms if numpy_ms > 0 else float("inf")

    return {
        "operation": "stats.median",
        "n": n,
        "loop_ms": loop_ms,
        "numpy_ms": numpy_ms,
        "speedup": speedup,
    }


def benchmark_stats_histogram(n: int = 100_000, repeats: int = 5) -> dict:
    """Benchmark stats.histogram (np.histogram) vs a Python loop binning.

    Parameters
    ----------
    n : int
        Array length.
    repeats : int
        Timed repetitions.

    Returns
    -------
    dict
    """
    rng = np.random.default_rng(42)
    arr = rng.random(n)

    loop_ms = _time_fn(_loop_histogram, arr, repeats=repeats)
    numpy_ms = _time_fn(stats_histogram, arr, repeats=repeats)
    speedup = loop_ms / numpy_ms if numpy_ms > 0 else float("inf")

    return {
        "operation": "stats.histogram",
        "n": n,
        "loop_ms": loop_ms,
        "numpy_ms": numpy_ms,
        "speedup": speedup,
    }


# ---------------------------------------------------------------------------
# Generic harness for teammate modules
# ---------------------------------------------------------------------------

def benchmark_extra(
    fn_vectorised: Callable,
    fn_loop: Callable,
    arr: np.ndarray,
    operation_name: str,
    repeats: int = 5,
) -> dict:
    """Generic benchmark for any pair of (vectorised, loop) functions.

    Handy for plugging in teammates' modules once they're ready
    (stats.py, preprocessing.py, etc.).

    Parameters
    ----------
    fn_vectorised : Callable
        The vectorised NumPy implementation.
    fn_loop : Callable
        The equivalent pure-Python loop implementation.
    arr : np.ndarray
        Input data passed to both functions.
    operation_name : str
        Label shown in the results table.
    repeats : int
        Timed repetitions.

    Returns
    -------
    dict

    Examples
    --------
    >>> def loop_mean(a):
    ...     return sum(a.tolist()) / len(a)
    >>> result = benchmark_extra(np.mean, loop_mean,
    ...                          np.random.random(100_000), "numpy.mean")
    """
    loop_ms = _time_fn(fn_loop, arr, repeats=repeats)
    numpy_ms = _time_fn(fn_vectorised, arr, repeats=repeats)
    speedup = loop_ms / numpy_ms if numpy_ms > 0 else float("inf")

    return {
        "operation": operation_name,
        "n": arr.size,
        "loop_ms": loop_ms,
        "numpy_ms": numpy_ms,
        "speedup": speedup,
    }


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------

def run_all(repeats: int = 5) -> list[dict]:
    """Run the full benchmark suite and print a performance table.

    Parameters
    ----------
    repeats : int
        Number of timed repetitions per benchmark (default 5).
        Higher = more stable results but slower.

    Returns
    -------
    list[dict]
        Raw results for further processing (e.g. in a notebook).

    Notes
    -----
    - O(n^2) loop baselines (sort, rank) use capped input sizes.
    - Speedup is always computed on same-size inputs.
    - binary_search uses 50 repeats since each call is very fast.
    - Results vary by machine; run on a quiet system for best numbers.
    """
    _print_env()
    print(f"\n  Running benchmarks ({repeats} repeats each)...\n")

    results = [
        # sort_search module
        benchmark_stable_sort(n=50_000, repeats=repeats),
        benchmark_topk(n=100_000, k=10, repeats=repeats),
        benchmark_topk(n=100_000, k=100, repeats=repeats),
        benchmark_binary_search(n=1_000_000, repeats=50),
        # rank module
        benchmark_rank(n=10_000, repeats=repeats),
        benchmark_percentile(n=100_000, repeats=repeats),
    ]

    # stats module benchmarks — included automatically when available
    if _HAS_STATS:
        results.extend([
            benchmark_stats_mean(n=100_000, repeats=repeats),
            benchmark_stats_var(n=100_000, repeats=repeats),
            benchmark_stats_median(n=100_000, repeats=repeats),
            benchmark_stats_histogram(n=100_000, repeats=repeats),
        ])
    else:
        print("  [skipped] stats benchmarks (stats module not available yet)\n")

    _print_table(results)

    print("Notes:")
    print("  - stable_sort loop uses insertion sort (O(n^2)), capped at n=5000.")
    print("  - rank loop uses nested loops (O(n^2)), capped at n=2000.")
    if _HAS_STATS:
        print("  - stats.mean/var use Welford's streaming algorithm (loop over slices).")
    print("  - Speedup is measured on identical input sizes for fairness.")
    print("  - Use benchmark_extra() to add benchmarks for other modules.")
    print()

    return results


# Entry point
if __name__ == "__main__":
    run_all()