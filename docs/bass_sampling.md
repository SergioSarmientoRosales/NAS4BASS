# BASS Architecture Sampling

Use `tools/sample_bass_architectures.py` to create a reproducible BASS-space
sample for full training and later zero-cost comparison.

The default method is a hybrid diversity-aware sample:

1. Generate a large valid pool of decoded BASS genes.
2. Remove exact duplicates.
3. Keep possible overlap with the original 20 Pareto-front architectures, but
   report it in metadata.
4. Stratify the pool by fixed estimated structural-complexity bands.
5. Select the requested number of architectures with greedy max-min coverage in
   normalized encoding and structural descriptor space inside each stratum.
6. Do not use PSNR, predicted PSNR, or zero-cost scores for selection.

Generate the approved 50-architecture sample:

```bash
python tools/sample_bass_architectures.py \
  --n 50 \
  --pool-size 50000 \
  --seed 20260703 \
  --pool-policy mixed \
  --selection-method stratified_max_min \
  --complexity-strategy fixed \
  --complexity-edges 0,6,8,10,12,inf \
  --max-epochs 1000
```

The `mixed` pool policy combines uniform random genes with medium-complexity and
lightweight structural priors. This avoids a purely uniform pool that
under-samples compact BASS architectures with many identity operations, while
still avoiding PSNR, predicted PSNR, zero-cost scores, or training results.
The fixed complexity bands keep the final 50 from collapsing into the most
common high-complexity region of the random search space.

The script writes:

```text
data/architectures/bass_50_sample_architectures.csv
data/architectures/bass_50_sample_metadata.json
data/architectures/bass_50_sample_trainer_manifest.csv
data/architectures/bass_50_sample/genes/bass_0001.json
```

The trainer manifest contains one `python -m srir_training.train` command per
architecture. Each command uses `--bass-gene-file`, so there is no need to export
an untrained `.keras` model before training.

Example dynamic sample:

```bash
python tools/sample_bass_architectures.py \
  --n 75 \
  --pool-size 75000 \
  --seed 7 \
  --pool-policy mixed \
  --train-dir /data/DIV2K_train_HR \
  --val-dir /data/DIV2K_valid_HR \
  --max-epochs 1000
```

Train one sampled architecture:

```bash
python -m srir_training.train \
  --bass-gene-file data/architectures/bass_50_sample/genes/bass_0001.json \
  --directory-train /data/DIV2K_train_HR \
  --directory-val /data/DIV2K_valid_HR \
  --scale 2 \
  --epochs 1000 \
  --lr-schedule plateau \
  --early-stopping-patience 20 \
  --reduce-lr-patience 15 \
  --run-name bass_0001 \
  --output-dir srir_outputs/bass_50_sample
```

For full training, `--epochs` is only a maximum cap. The actual stop point is
controlled by the trainer callbacks: EarlyStopping monitors validation PSNR with
patience 20, and ReduceLROnPlateau waits for 15 consecutive non-improving
validation epochs before reducing the learning rate.

The full-training outputs can later be joined back to
`bass_50_sample_architectures.csv` by `sample_id`.

## Visualize The Sample

Create a PCA-style architecture-space figure:

```bash
python tools/plot_bass_sample.py
```

This writes:

```text
figures/zerocost_50_bass_sample/bass_50_sample_architecture_space.png
figures/zerocost_50_bass_sample/bass_50_sample_architecture_space.pdf
figures/zerocost_50_bass_sample/bass_50_sample_architecture_space_coordinates.csv
```

The visualization includes a PCA-style architecture-space view, a simple
descriptor scatter plot, and the selected-vs-pool complexity distribution. It
uses only architecture encodings and structural descriptors. It does not use
PSNR, predicted PSNR, training results, or zero-cost scores.
