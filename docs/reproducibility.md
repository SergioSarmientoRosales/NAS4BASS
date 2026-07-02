# Reproducibility

This repository is organized so that a new user can run small smoke tests and regenerate outputs without committing generated artifacts.

## Detected Environment

- Python detected locally: 3.12.2
- Main framework: TensorFlow/Keras
- Surrogate model ecosystem: scikit-learn, XGBoost, joblib
- Important dependency constraint: use `numpy>=1.26,<2.0` with the pinned TensorFlow 2.16.x stack.

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

The serialized surrogate models were verified to load with the dependency versions pinned in `requirements.txt`. Pickle-based model artifacts can be version-sensitive, so dependency changes should be validated before reproducing experiments.

## Seeds

`main.py` exposes `--seed` and calls the project seed utility. Use explicit seeds for repeatable runs:

```bash
python main.py --eval model_based --search nsga3 --seed 1 --pop-size 100 --n-gen 500
```

`Benchmark.py` defines its own internal seed settings for repeated zero-cost predictor analysis.

For zero-cost NAS, `evaluators/zero_cost.py` also derives a stable TensorFlow seed from the run seed and decoded architecture. This reduces score changes caused only by candidate evaluation order. Use `--disable-deterministic-arch-seed` only when reproducing older behavior.

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

## Lightweight Validation Checks

Run the lightweight test suite:

```bash
python -m unittest discover -s tests
```

The zero-cost smoke tests instantiate representative BASS architectures, verify that searchable convolutional operations appear in the Keras graph, check that trainable convolutional parameters are visible to the metric collectors, and confirm that selected inexpensive zero-cost scores return scalar finite values.

Validate curated data and surrogate model artifacts:

```bash
python tools/validate_artifacts.py
```

The validator checks architecture length and value ranges, numeric fields, duplicate architecture rows, and whether each serialized surrogate model can predict from a 28-gene input. Duplicate architecture rows and historical non-finite `train_psnr` entries are reported as warnings by default.

The GitHub Actions workflow mirrors this in two stages:

- A lightweight job checks imports, `main.py --help`, compilation, lightweight tests, and CSV validation without TensorFlow.
- A full job installs `requirements.txt`, runs all tests, and validates both data and surrogate models.

## What Is Not Included

Large image datasets and full training pipelines are not currently packaged in this repository.

TODO: document dataset acquisition, preprocessing, and training-stage reproduction once the public release protocol is finalized.
