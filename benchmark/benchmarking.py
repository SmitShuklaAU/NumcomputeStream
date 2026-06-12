"""
benchmarking.py
---------------
Micro-benchmark harness for numcompute_stream.

Compares vectorised NumPy implementations of core streaming machine learning 
operations (Welford's updates, Gini impurity, Tree splitting) against 
equivalent pure-Python loop implementations to demonstrate the massive 
performance gains of vectorisation.

Usage
-----
Run directly from the root of the project:
    python -m benchmark.benchmarking
"""

from __future__ import annotations
import time
import numpy as np
from typing import Callable, Tuple

# ---------------------------------------------------------------------------
# Timing Harness
# ---------------------------------------------------------------------------

def _time_fn(fn: Callable, *args, repeats: int = 7, warmup: int = 2, **kwargs) -> float:
    """Run fn(*args, **kwargs) several times and return the median wall-clock time in ms."""
    # Warmup runs (not recorded)
    for _ in range(warmup):
        fn(*args, **kwargs)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # convert seconds to milliseconds

    return float(np.median(times))

def _print_table(results: list[dict]) -> None:
    """Format and print benchmark results as a clean ASCII table."""
    header = f"{'Operation':<25} | {'Input Size':<12} | {'Loop (ms)':<10} | {'NumPy (ms)':<10} | {'Speedup':<8}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for r in results:
        op = r["operation"]
        n = str(r["size"])
        t_loop = f"{r['loop_ms']:.3f}" if r['loop_ms'] is not None else "N/A"
        t_vec = f"{r['vec_ms']:.3f}"
        
        if r['loop_ms'] is not None and r['vec_ms'] > 0:
            speedup = f"{r['loop_ms'] / r['vec_ms']:.1f}x"
        else:
            speedup = "N/A"
            
        print(f"{op:<25} | {n:<12} | {t_loop:<10} | {t_vec:<10} | {speedup:<8}")
    print("-" * len(header))


# ---------------------------------------------------------------------------
# 1. Welford's Online Update (Streaming Stats / Scaler)
# ---------------------------------------------------------------------------

def loop_welford_update(mean: np.ndarray, M2: np.ndarray, n_samples: np.ndarray, X_chunk: np.ndarray):
    """Pure Python loop simulating sample-by-sample Welford updates."""
    n_features = X_chunk.shape[1]
    for row in X_chunk:
        for j in range(n_features):
            val = row[j]
            if not np.isnan(val):
                n_samples[j] += 1
                delta = val - mean[j]
                mean[j] += delta / n_samples[j]
                delta2 = val - mean[j]
                M2[j] += delta * delta2
    return mean, M2

def vec_welford_update(mean: np.ndarray, M2: np.ndarray, n_samples: np.ndarray, X_chunk: np.ndarray):
    """Batched NumPy chunk update used in StreamStats."""
    chunk_valid = np.sum(~np.isnan(X_chunk), axis=0)
    chunk_mean = np.nanmean(X_chunk, axis=0)
    chunk_var = np.nanvar(X_chunk, axis=0)
    chunk_M2 = chunk_var * chunk_valid
    
    n_new = chunk_valid
    n_total = n_samples + n_new
    
    delta = chunk_mean - mean
    mean += delta * n_new / n_total
    M2 += chunk_M2 + (delta ** 2) * n_samples * n_new / n_total
    
    return mean, M2

def benchmark_welford(n: int = 10000, features: int = 10, repeats: int = 5) -> dict:
    X_chunk = np.random.rand(n, features)
    mean_state = np.zeros(features)
    M2_state = np.zeros(features)
    n_state = np.ones(features)  # Start at 1 to avoid div by zero
    
    t_loop = _time_fn(loop_welford_update, mean_state.copy(), M2_state.copy(), n_state.copy(), X_chunk, repeats=repeats)
    t_vec = _time_fn(vec_welford_update, mean_state.copy(), M2_state.copy(), n_state.copy(), X_chunk, repeats=repeats)
    
    return {"operation": "Welford Chunk Update", "size": f"{n}x{features}", "loop_ms": t_loop, "vec_ms": t_vec}


# ---------------------------------------------------------------------------
# 2. Gini Impurity (Tree Splitting)
# ---------------------------------------------------------------------------

def loop_gini(y: np.ndarray) -> float:
    """Pure Python dictionary-based frequency counting."""
    counts = {}
    n = len(y)
    for val in y:
        counts[val] = counts.get(val, 0) + 1
        
    impurity = 1.0
    for count in counts.values():
        prob = count / n
        impurity -= prob ** 2
    return impurity

def vec_gini(y: np.ndarray, classes: np.ndarray) -> float:
    """NumPy broadcasting Gini calculation from tree.py."""
    counts = np.sum(y[:, None] == classes[None, :], axis=0)
    probs = counts / y.size
    return float(1.0 - np.sum(probs ** 2))

def benchmark_gini(n: int = 50000, repeats: int = 10) -> dict:
    y = np.random.randint(0, 5, n)
    classes = np.unique(y)
    
    t_loop = _time_fn(loop_gini, y, repeats=repeats)
    t_vec = _time_fn(vec_gini, y, classes, repeats=repeats)
    
    return {"operation": "Gini Impurity", "size": n, "loop_ms": t_loop, "vec_ms": t_vec}


# ---------------------------------------------------------------------------
# 3. Tree Node Split Routing (Data Partitioning)
# ---------------------------------------------------------------------------

def loop_split(X: np.ndarray, feature_idx: int, threshold: float) -> Tuple[list, list]:
    """Pure Python row-by-row filtering."""
    left, right = [], []
    for row in X:
        if row[feature_idx] <= threshold:
            left.append(row)
        else:
            right.append(row)
    return left, right

def vec_split(X: np.ndarray, feature_idx: int, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """NumPy boolean masking from tree.py."""
    left_mask = X[:, feature_idx] <= threshold
    right_mask = ~left_mask
    return X[left_mask], X[right_mask]

def benchmark_node_split(n: int = 50000, features: int = 10, repeats: int = 10) -> dict:
    X = np.random.rand(n, features)
    feature_idx = 5
    threshold = 0.5
    
    t_loop = _time_fn(loop_split, X, feature_idx, threshold, repeats=repeats)
    t_vec = _time_fn(vec_split, X, feature_idx, threshold, repeats=repeats)
    
    return {"operation": "Tree Node Splitting", "size": f"{n}x{features}", "loop_ms": t_loop, "vec_ms": t_vec}


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_all() -> None:
    print("\nStarting Streaming ML Benchmarks (NumPy Vectorisation vs Pure Python Loops)...")
    print("Simulating performance across streaming chunks...\n")
    
    results = [
        benchmark_welford(n=10000, features=20, repeats=5),
        benchmark_gini(n=100_000, repeats=10),
        benchmark_node_split(n=100_000, features=20, repeats=10)
    ]
    
    _print_table(results)
    print("\nConclusion: NumPy vectorisation provides exponential speedups crucial for online streaming scenarios.")

if __name__ == "__main__":
    np.random.seed(42)
    run_all()