# BASS Architecture Sampling

Use `tools/sample_bass_architectures.py` to create a reproducible BASS-space
sample for full training and later zero-cost comparison.

The default method is a hybrid diversity-aware sample:

1. Generate a large random pool of valid decoded BASS genes.
2. Remove exact duplicates.
3. Keep possible overlap with the original 20 Pareto-front architectures, but
   report it in metadata.
4. Select the requested number of architectures with greedy max-min coverage in
   normalized encoding and structural descriptor space.
5. Do not use PSNR, predicted PSNR, or zero-cost scores for selection.

Generate the approved 50-architecture sample:

```bash
python tools/sample_bass_architectures.py --n 50 --pool-size 10000 --seed 20260703
```

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
  --pool-size 20000 \
  --seed 7 \
  --train-dir /data/DIV2K_train_HR \
  --val-dir /data/DIV2K_valid_HR \
  --epochs 100
```

Train one sampled architecture:

```bash
python -m srir_training.train \
  --bass-gene-file data/architectures/bass_50_sample/genes/bass_0001.json \
  --directory-train /data/DIV2K_train_HR \
  --directory-val /data/DIV2K_valid_HR \
  --scale 2 \
  --epochs 100 \
  --run-name bass_0001 \
  --output-dir srir_outputs/bass_50_sample
```

The full-training outputs can later be joined back to
`bass_50_sample_architectures.csv` by `sample_id`.
