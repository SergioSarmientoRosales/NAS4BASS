# BASS Architecture Sampling

Use `tools/sample_bass_architectures.py` to create the official 50-architecture
BASS sample for full training and later zero-cost predictor analysis.

The current official sample supersedes the deterministic max-min sample from
commit `fe0d379`.

## Design

The sampler uses:

1. A uniform random pool of 100,000 valid decoded BASS genes.
2. Five complexity strata computed as quantiles of `log_estimated_complexity`
   over that uniform pool.
3. Random selection within each stratum, with 10 architectures per band.
4. A selection RNG derived from the main seed but independent from pool
   generation.
5. No PSNR, predicted PSNR, zero-cost score, or training result in any
   architecture-selection decision.

The earlier fixed edges `0,6,8,10,12,inf` are still available through
`--complexity-strategy fixed`, but they are not the official default because a
uniform 100,000-architecture BASS pool produced no candidates in the first two
bands. Quantile bands preserve a uniform pool while making the 10-per-band
design feasible and statistically interpretable.

Generate the official sample:

```bash
python tools/sample_bass_architectures.py --save-pool-cache
```

The default command is equivalent to:

```bash
python tools/sample_bass_architectures.py \
  --n 50 \
  --pool-size 100000 \
  --seed 20260703 \
  --pool-policy uniform \
  --selection-method stratified_random \
  --complexity-strategy quantile \
  --complexity-bins 5 \
  --max-epochs 1000 \
  --save-pool-cache
```

The script writes:

```text
data/architectures/bass_50_sample_architectures.csv
data/architectures/bass_50_sample_metadata.json
data/architectures/bass_50_sample_trainer_manifest.csv
data/architectures/bass_50_sample_pool_cache.npz
data/architectures/bass_50_sample/genes/bass_0001.json
```

`params_real` is intentionally empty after sampling. It must be filled by the
TensorFlow validation pass before final analysis.

## Validation

Run the validator in a TensorFlow-compatible training environment:

```bash
python tools/validate_bass_sample.py \
  --sample-csv data/architectures/bass_50_sample_architectures.csv \
  --metadata-json data/architectures/bass_50_sample_metadata.json \
  --pool-cache data/architectures/bass_50_sample_pool_cache.npz
```

The validator instantiates each BASS model, computes `params_real`, detects
degenerate architectures, and replaces degenerate cases with the next unused
candidate from the same band's saved `draw_order`. It updates the CSV, gene JSONs,
trainer manifest, and metadata validation report. It also records a 5 by 5
proxy-alignment matrix between complexity band and quintiles of `log(params_real)`.

Workflow order:

```text
sampler -> validate_bass_sample.py -> final CSV -> full trainings -> zero-cost analysis
```

## Training

The trainer manifest contains one command per architecture. Each command uses
`--bass-gene-file`, so untrained `.keras` exports are not needed before training.

Training is full-training with early stopping, not a fixed-epoch protocol. The
manifest uses `--epochs 1000` only as a maximum cap and keeps:

```text
--lr-schedule plateau
--early-stopping-patience 20
--reduce-lr-patience 15
```

The same manifest includes `eval_extra_command`, which evaluates trained
`best.keras` checkpoints on Set5, Set14, and BSD100 using
`tools/eval_extra_datasets.py`.

Extra-dataset evaluation follows the trainer's internal convention: RGB/full-image
PSNR and SSIM, images normalized to `[0, 1]`, HR cropped to dimensions divisible by
scale, TensorFlow resize for LR generation, and no border shaving.

## Visualization

Create the official plots:

```bash
python tools/plot_bass_sample.py
```

This writes:

```text
figures/zerocost_50_stratified_random/bass_50_sample_architecture_space.png
figures/zerocost_50_stratified_random/bass_50_sample_complexity_distribution.png
figures/zerocost_50_stratified_random/bass_50_sample_architecture_space_coordinates.csv
```

To create the supplementary random-vs-max-min coverage figure, first generate a
temporary max-min sample with separate output paths, then pass it as
`--comparison-csv`. The official sample must remain the stratified-random CSV.

## Zero-Cost Analysis Plan

After all 50 architectures are fully trained and `params_real` is validated, use:

```bash
python tools/analyze_zerocost_sample.py \
  --sample-csv data/architectures/bass_50_sample_architectures.csv \
  --score-csv path/to/zero_cost_scores_seed_1.csv \
  --score-csv path/to/zero_cost_scores_seed_2.csv \
  --target-column valid_psnr \
  --params-column params_real
```

For every predictor and every transformation (`raw`, `div_params`, `neg_raw`,
`neg_div_params`), the analysis reports C-index, Spearman, Kendall tau-b, and
partial Spearman controlling `log(params_real)`. It separates score-seed
variability from architecture-bootstrap uncertainty and writes results under:

```text
results/zerocost_50_stratified_random/
figures/zerocost_50_stratified_random/
```

Band-level statistics are included as descriptive analyses because each band has
only 10 architectures.
