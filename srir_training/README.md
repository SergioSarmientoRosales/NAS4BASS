# SRIR Training Pipeline

This package provides a reusable TensorFlow/Keras training pipeline for CNN-based
super-resolution image restoration (SRIR). It is intentionally independent from
the NAS search code, but it is ready to accept BASS/NAS-generated Keras models.

## What It Supports

- Scale factors x2, x3, and x4.
- Paired LR/HR folders, including DIV2K-style names such as `0001x2.png` and `0001.png`.
- Direct DIV2K HR folders with on-the-fly LR generation by downsampling.
- GPU-aware automatic `patch_size` and `batch_size` defaults.
- Default lightweight residual SR model without BatchNorm.
- Custom user-provided Keras models through `--custom-model-path`.
- Normalized image range `[0, 1]`.
- SR-safe paired augmentations: horizontal flip, vertical flip, and 90-degree rotations.
- Charbonnier, L1/MAE, and MSE losses.
- AdamW or Adam optimizers with global gradient clipping.
- Dynamic learning-rate policies: plateau reduction, warmup cosine decay, or fixed LR.
- PSNR and SSIM metrics.
- Checkpointing, TensorBoard logs, CSV history, JSON config/runtime/metrics.
- Resume from `.keras` checkpoints.
- CPU fallback and optional GPU mixed precision.

## Folder Layout

```text
srir_training/
  train.py
  smoke_test.py
  config.py
  data.py
  augmentations.py
  models.py
  losses.py
  metrics.py
  callbacks.py
  utils.py
```

## Installation

From the repository root:

```bash
pip install -r requirements.txt
```

The local `requirements.txt` in this folder is a short reference file; the root
repository requirements remain the source of truth.

## DIV2K-Style Data

Two data modes are supported.

Direct HR-only DIV2K folders:

```text
directory_train = "/data/DIV2K_train_HR"
directory_val   = "/data/DIV2K_valid_HR"
```

In this mode, LR inputs are generated on the fly from HR crops using the selected
scale and `--downsample-method bicubic` by default. This is the shortest Docker
path style and avoids storing duplicated LR folders.

Paired LR/HR folder structure:

```text
DIV2K/
  train_HR/
    0001.png
    0002.png
  train_LR_bicubic/X2/
    0001x2.png
    0002x2.png
  valid_HR/
    0801.png
  valid_LR_bicubic/X2/
    0801x2.png
```

For x3 and x4, use `X3`/`X4` folders and filenames such as `0001x3.png` or
`0001x4.png`. Exact matching by stem is also supported.

## Smoke Test

The smoke test creates synthetic LR/HR batches, builds the model, checks output
shape, verifies finite gradients, and runs one tiny train/eval pass.

```bash
python -m srir_training.smoke_test
```

Force CPU:

```bash
python -m srir_training.smoke_test --cpu
```

Run one scale only:

```bash
python -m srir_training.smoke_test --scale 4
```

## Training

Charbonnier and Adam are not alternatives. Charbonnier is the training loss; it
measures the restoration error. Adam or AdamW is the optimizer; it decides how
weights are updated from gradients. The default is Charbonnier loss with AdamW,
which is a stable SRIR starting point.

`patch_size` and `batch_size` are automatic by default. The trainer checks the
available GPU memory when possible and writes the resolved values into the saved
`config.json`. Explicit `--patch-size` or `--batch-size` values still override
the automatic choice. Use `--max-batch-size` or `--max-patch-size` as safety caps
for unknown/custom models.

The default learning-rate policy is dynamic plateau reduction:

- Monitor: `val_psnr`
- Checkpoint mode: maximize
- Early stopping patience: 20 epochs
- ReduceLROnPlateau patience: 15 consecutive epochs without improvement
- Plateau counter reset: any `val_psnr` improvement larger than `min_delta` resets the 15-epoch counter
- Cooldown: 0, so the criterion is strictly consecutive

This means isolated fluctuations do not trigger a learning-rate drop. The
learning rate is reduced only after 15 validation epochs in a row fail to improve
the monitored metric. A non-finite loss still stops training through
`TerminateOnNaN`.

Example x2 training run with direct DIV2K HR folders:

```bash
python -m srir_training.train ^
  --directory-train C:\data\DIV2K_train_HR ^
  --directory-val C:\data\DIV2K_valid_HR ^
  --scale 2 ^
  --epochs 100 ^
  --output-dir srir_outputs
```

Linux/macOS:

```bash
python -m srir_training.train \
  --directory-train /data/DIV2K_train_HR \
  --directory-val /data/DIV2K_valid_HR \
  --scale 2 \
  --epochs 100 \
  --output-dir srir_outputs
```

Paired LR/HR folders remain available when you want fixed precomputed LR images:

```bash
python -m srir_training.train \
  --train-lr-dir /data/DIV2K_train_LR_bicubic/X2 \
  --train-hr-dir /data/DIV2K_train_HR \
  --val-lr-dir /data/DIV2K_valid_LR_bicubic/X2 \
  --val-hr-dir /data/DIV2K_valid_HR \
  --scale 2 \
  --epochs 100
```

Optional warmup cosine LR instead of plateau:

```bash
python -m srir_training.train \
  ... \
  --lr-schedule cosine \
  --warmup-epochs 5 \
  --cosine-alpha 0.1
```

Disable dynamic LR:

```bash
python -m srir_training.train ... --lr-schedule none
```

The HR patch size must be divisible by the scale. For example:

- x2: `--patch-size 96`
- x3: `--patch-size 96`
- x4: `--patch-size 128`

When omitted, the automatic resolver chooses a divisible patch size for the
selected scale.

## Resume

```bash
python -m srir_training.train \
  --config srir_outputs/<run>/config.json \
  --resume-from srir_outputs/<run>/checkpoints/last.keras
```

## Fine-Tuning Existing `best.keras` Models

Fine-tune every `best.keras` found recursively under a folder:

```bash
python -m srir_training.fine_tune \
  --models-dir srir_outputs \
  --directory-train /data/DIV2K_train_HR \
  --directory-val /data/DIV2K_valid_HR \
  --scale 2 \
  --epochs 50 \
  --learning-rate 2e-5 \
  --output-dir srir_finetune_outputs
```

The fine-tuning defaults are deliberately conservative: lower learning rate and
the same `val_psnr` checkpoint criterion. They still use early stopping patience
20 and plateau patience 15 consecutive validation epochs. Each source model gets
its own output folder with `checkpoints/best.keras`, `finetuned_final.keras`,
history, config, and final PSNR/SSIM metrics.

For a quick trial:

```bash
python -m srir_training.fine_tune --models-dir srir_outputs --max-models 1 ...
```

## Output Artifacts

Each run writes:

```text
srir_outputs/<run>/
  config.json
  runtime.json
  final_metrics.json
  history.json
  final.keras
  checkpoints/
    best.keras
    last.keras
  logs/
    history.csv
    tensorboard/
```

Training checkpoints use `val_psnr` by default. At the end, the best checkpoint
is evaluated and the final printed metrics include both PSNR and SSIM.

## Custom Keras Models

Use any compatible Keras model that maps:

```text
(batch, H, W, C) -> (batch, H*scale, W*scale, C)
```

Then train it with:

```bash
python -m srir_training.train --custom-model-path path/to/model.keras ...
```

## Future BASS/NAS Integration

BASS/NAS models can be connected without changing this training package by
building a Keras model before training and saving it as `.keras`, or by adding a
small adapter that calls:

```python
from search_space.search_space import decode
from search_space.model_builder import get_model

genotype = decode(decoded_architecture)
model = get_model(genotype, upscale_factor=scale)
model.save("candidate.keras")
```

Then pass `candidate.keras` with `--custom-model-path`.

## Notes

- Images are normalized to `[0, 1]`; PSNR and SSIM use `max_val=1.0`.
- BatchNorm is intentionally avoided in the default model because SR CNNs often
  train better without it.
- Mixed precision is enabled only when GPU support is available and
  `--mixed-precision auto` or `--mixed-precision on` is used.
- The code does not invent datasets or silently ignore missing LR/HR pairs.
