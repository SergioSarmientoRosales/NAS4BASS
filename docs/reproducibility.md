# Reproducibility

This repository is organized so that a new user can run small smoke tests and regenerate outputs without committing generated artifacts.

## Detected Environment

- Python detected locally: 3.12.2
- Main framework: TensorFlow/Keras
- Surrogate model ecosystem: scikit-learn, XGBoost, joblib

TODO: add exact operating system, GPU, CUDA, cuDNN, and driver versions used for final experiments.

## Required Inputs

Model-based search expects surrogate predictors in:

```text
models/*.pkl
```

Zero-cost benchmark scripts expect:

```text
data/20_full_trained_models.csv
```

Other small CSVs in `data/` are retained as reference metadata or analysis inputs.

## Seeds

`main.py` exposes `--seed` and calls the project seed utility. Use explicit seeds for repeatable runs:

```bash
python main.py --eval model_based --search nsga3 --seed 1 --pop-size 100 --n-gen 500
```

`Benchmark.py` defines its own internal seed settings for repeated zero-cost predictor analysis.

## Generated Outputs

Search runs create CSV files under:

```text
outputs/seed_<seed>/
```

Benchmark scripts create files under:

```text
data/zero_cost_benchmark_outputs_paper_only/
```

Both locations are ignored by Git so that generated results do not pollute version control.

## Clean Reproduction Workflow

1. Create a fresh virtual environment.
2. Install `requirements.txt`.
3. Run a small smoke test from `docs/quickstart.md`.
4. Run the desired full search or benchmark.
5. Archive important outputs separately, for example in a release artifact or experiment storage.

## What Is Not Included

Large image datasets and full training pipelines are not currently packaged in this repository.

TODO: document dataset acquisition, preprocessing, and training-stage reproduction once the public release protocol is finalized.
