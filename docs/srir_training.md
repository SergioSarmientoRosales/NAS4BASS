# SRIR Training Pipeline

The `srir_training/` package is a standalone TensorFlow/Keras training pipeline
for super-resolution image restoration models. It is kept separate from the NAS
search logic so that training experiments can evolve without changing the search
implementation.

Use it for:

- x2, x3, or x4 SRIR training.
- DIV2K-style paired LR/HR folders.
- Direct DIV2K HR-only folders, for example `/data/DIV2K_train_HR` and
  `/data/DIV2K_valid_HR`, with LR generated on the fly.
- Stage 1 reference defaults (`patch_size=64`, `batch_size=64`) with optional
  GPU-aware automatic sizing through `--auto-size`.
- A default residual CNN baseline.
- User-provided Keras models.
- Future BASS/NAS-generated Keras models.

Default optimization uses MSE loss with AdamW to match the Stage 1 reference
trainer. MSE/Charbonnier are losses; Adam/AdamW performs the gradient updates.
Learning rate is dynamic by default through `ReduceLROnPlateau` on `val_psnr`,
with early stopping patience 20 and LR plateau patience 15 consecutive epochs
without improvement. Any real `val_psnr` improvement resets the plateau counter.

The standard protocol keeps only the Stage 1-compatible run. It does not launch
an automatic second Stage 2/p128 training pass. By default x2/x4 use
`patch_size=64` and `batch_size=64`; x3 uses `patch_size=96` if no explicit
patch size is provided, because 64 is not divisible by 3. Use `--auto-size` only
when GPU-aware sizing is preferred over strict reference comparability.

Compared with the reference `.py`, preprocessing is aligned for direct HR-only
DIV2K training: random HR crops, bicubic LR generation, normalized `[0, 1]`
images, paired augmentations, MSE loss, patch-count epoch sizing, and
sliding-window validation. The main documented differences are that Stage 2/p128
is not run automatically, x3 defaults to p96 because p64 is not divisible by 3,
and paired precomputed LR/HR validation still uses deterministic center crops.

Direct DIV2K HR usage:

```bash
python -m srir_training.train \
  --directory-train /data/DIV2K_train_HR \
  --directory-val /data/DIV2K_valid_HR \
  --scale 2
```

Quick smoke test:

```bash
python -m srir_training.smoke_test
```

Training entry point:

```bash
python -m srir_training.train --help
```

Fine-tuning entry point for folders containing `best.keras` files:

```bash
python -m srir_training.fine_tune --help
```

Full usage details are in [`srir_training/README.md`](../srir_training/README.md).
