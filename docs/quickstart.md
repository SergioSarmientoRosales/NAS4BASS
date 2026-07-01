# Quickstart

This guide runs small smoke tests. They are intentionally tiny and are not meant to reproduce the full paper results.

## 1. Model-Based Search

This mode uses the surrogate predictors stored in `models/*.pkl`.

```bash
python main.py --eval model_based --search nsga3 --seed 1 --pop-size 10 --n-gen 2
```

Expected result:

- A folder like `outputs/seed_1/`
- A population CSV
- A non-dominated solutions CSV
- A cache summary CSV

## 2. Zero-Cost Search

This mode does not need the surrogate `.pkl` files.

```bash
python main.py --eval zero_cost --zc-metric param_score --search random --seed 1 --pop-size 10 --n-gen 2
```

Try a heavier zero-cost metric:

```bash
python main.py --eval zero_cost --zc-metric synflow --search random --seed 1 --pop-size 10 --n-gen 2
```

## 3. Full-Scale Search

The configuration constants define a default population size of 100 and the paper-style search can use hundreds of generations.

Example:

```bash
python main.py --eval model_based --search nsga3 --seed 1 --pop-size 100 --n-gen 500
```

Use `--outdir` if you want outputs somewhere else:

```bash
python main.py --eval model_based --search nsga3 --seed 1 --pop-size 100 --n-gen 500 --outdir outputs
```

## 4. Benchmark Zero-Cost Predictors

This is the full benchmark path, not a smoke test. It can take substantially longer than the small `main.py` commands above.

```bash
python Benchmark.py
python Plots.py
```

`Benchmark.py` creates raw benchmark results and tables. `Plots.py` uses those outputs to create compact paper-style tables and figures.
