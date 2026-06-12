
# NumCompute Stream: Real-Time Machine Learning Framework

**NumCompute Stream** is a bespoke, pure-NumPy online machine learning framework developed for real-time data streams. Built from scratch without relying on heavy external ML libraries (like `scikit-learn` or `pandas`), this framework focuses on algorithmic efficiency, memory-bounded incremental learning, and numerical stability.

This project was developed for the Artificial Intelligence and Machine Learning (AIML) Master's program at Adelaide University.

## Features

- **Online Learning:** All components support incremental chunk-wise updates via `.partial_fit()`.
- **Streaming Ensembles:** Implements a mini-batch bootstrapped Random Forest for real-time predictions.
- **Dynamic Preprocessing:** Includes stateful `StreamStandardScaler`, `StreamMinMaxScaler`, `StreamOneHotEncoder`, and `StreamSimpleImputer`.
- **Welford's Algorithm:** Mathematically stable, batched streaming statistics (mean, variance) without catastrophic cancellation.
- **Prequential Evaluation:** Interleaved test-then-train methodology for rigorous real-world metric tracking.
- **Pure NumPy:** Highly vectorized operations designed to outperform standard Python loops.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/numcompute-stream.git](https://github.com/YOUR_USERNAME/numcompute-stream.git)
   cd numcompute-stream

```

2. **Create a virtual environment (Recommended):**
```bash
python -m venv numcomputestreamenv
source numcomputestreamenv/bin/activate  # On Windows: numcomputestreamenv\Scripts\activate

```


3. **Install dependencies:**
The framework strictly relies on core scientific computing libraries.
```bash
pip install numpy matplotlib pytest

```



---

## Quick Start (Demo)

The easiest way to understand the framework is to run the interactive streaming demo, which simulates a live financial data stream and compares a single Decision Tree against a Random Forest Ensemble.

1. Open Jupyter Notebook (or VS Code).
2. Navigate to `demo/stream_demo.ipynb`.
3. Clear the outputs (**Kernel -> Restart Kernel and Clear All Outputs**).
4. Run all cells to watch the pipeline ingest data chunks, track memory footprints, and plot cumulative accuracy over time.

---

## Testing

The framework includes a comprehensive unit test suite (>40 tests) covering edge cases, NaN handling, dynamic bounds expansion, and mathematical equivalency to batch processes.

To execute the test suite, ensure your `PYTHONPATH` is set to the current directory:

```bash
# Mac/Linux
PYTHONPATH=. pytest tests/ -v

# Windows (PowerShell)
$env:PYTHONPATH="."; pytest tests/ -v

```

---

## Benchmarks

A core objective of this project is demonstrating the computational superiority of NumPy vectorisation over standard Python loops. The micro-benchmark suite tests Welford's streaming updates, Gini impurity calculations, and tree-node routing.

To run the benchmarks:

```bash
python -m benchmark.benchmarking

```


## License

Created for academic purposes.

```

```
