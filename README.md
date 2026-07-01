# NAS4BASS

NAS4BASS is a research-code repository for Neural Architecture Search (NAS) in super-resolution image restoration (SRIR). It explores BASS-style candidate neural networks and evaluates them with either surrogate machine-learning predictors or zero-cost predictors such as SynFlow-like scores.

The goal is simple: avoid fully training every candidate network when exploring many possible architectures. Instead, the code searches a modular architecture space and uses fast proxy signals to prioritize promising designs.

## Who This Repository Is For

This repository is useful for:

- Researchers working on NAS for image restoration.
- Students who want a practical NAS example without starting from scratch.
- Engineers who want to run a modular NSGA-III search over BASS-like architectures.
- Readers of the related paper who want to reproduce the search and zero-cost predictor analysis.

## Key Ideas In Plain Language

Neural Architecture Search (NAS) is a method for automatically exploring neural network designs instead of manually designing every architecture.

BASS is a search space designed for super-resolution image restoration, where candidate architectures are represented through modular design choices.

Super-resolution image restoration tries to reconstruct a higher-resolution image from a lower-resolution input.

Zero-cost predictors estimate architecture quality without fully training every model, reducing computational cost.

Why this matters: training every candidate architecture is expensive; therefore, efficient proxies can help prioritize promising architectures.

## Main Features

- Modular BASS-style architecture encoding and decoding.
- NSGA-III and random-search backends.
- Model-based evaluation with serialized surrogate predictors in `models/`.
- Zero-cost predictor evaluation, including SynFlow, Fisher, SNIP, GraSP, GradNorm, NWOT, Zen, ZiCo, Jacobian covariance, L2 norm, plain score, and parameter score variants.
- CSV outputs for population histories, non-dominated solutions, and cache summaries.
- Benchmark scripts for comparing zero-cost predictor behavior against trained-model reference data.

## Repository Structure

```text
NAS4BASS/
  main.py                 Main NAS entry point
  Benchmark.py            Zero-cost predictor benchmark pipeline
  Plots.py                Tables and figures for benchmark outputs
  config.py               Search-space constants and defaults
  core/                   NAS problem and registry wiring
  search/                 NSGA-III, random search, and operators
  search_space/           BASS encoding, decoding, and model builder
  evaluators/             Model-based and zero-cost evaluators
  predictors/             Surrogate model loading, selection, and ensembling
  models/                 Serialized surrogate predictors used by model-based search
  data/                   Small reference CSV files used by benchmarks
  docs/                   Installation, quickstart, reproducibility, and background
```

Generated outputs are intentionally ignored by Git. Running the project will create folders such as `outputs/` or `data/zero_cost_benchmark_outputs_paper_only/`.

## Installation

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For more detail, see [docs/installation.md](docs/installation.md).

## Quick Start

Run a small model-based NAS smoke test:

```bash
python main.py --eval model_based --search nsga3 --seed 1 --pop-size 10 --n-gen 2
```

Run a small zero-cost NAS smoke test without surrogate `.pkl` models:

```bash
python main.py --eval zero_cost --zc-metric param_score --search random --seed 1 --pop-size 10 --n-gen 2
```

The default output folder is `outputs/seed_<seed>/`.

## Main Pipeline

The main entry point is `main.py`.

Important arguments:

- `--eval model_based` uses surrogate predictors from `models/*.pkl`.
- `--eval zero_cost` evaluates architectures with a zero-cost metric.
- `--zc-metric` selects a zero-cost metric, for example `synflow`, `fisher`, `snip`, or `param_score`.
- `--search nsga3` runs NSGA-III.
- `--search random` runs random search.
- `--n-gen` is required unless `--early-stop` is enabled.
- `--outdir` controls where generated CSV files are written.

Example:

```bash
python main.py --eval model_based --search nsga3 --seed 1 --pop-size 100 --n-gen 500
```

## Zero-Cost Predictor Benchmark

`Benchmark.py` evaluates zero-cost predictors against reference trained-model data in:

```text
data/20_full_trained_models.csv
```

Run:

```bash
python Benchmark.py
```

This generates raw benchmark outputs and paper tables under:

```text
data/zero_cost_benchmark_outputs_paper_only/
```

Then run:

```bash
python Plots.py
```

to generate summary tables and the main benchmark figure from those outputs.

## Input Data Requirements

The repository includes small CSV reference files under `data/`. Large datasets are not included.

TODO: document the original training dataset locations and download/preprocessing steps when they are finalized for public release.

## Output Files

Typical outputs include:

- `ensemble_<method>_population_seed_<seed>_<timestamp>.csv`
- `non_dominated_seed_<seed>_<timestamp>.csv`
- `cache_summary_seed_<seed>_<timestamp>.csv`
- zero-cost benchmark tables and figures generated by `Benchmark.py` and `Plots.py`

Generated outputs should not be committed unless they are intentionally curated for a release or paper artifact.

## Citation

TODO: add paper title, authors, venue, DOI, and BibTeX entry after the citation details are finalized.

## Maintainer

Maintainer: Sergio Sarmiento Rosales

TODO: add preferred contact email or project contact link.
