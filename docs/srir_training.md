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
- GPU-aware automatic `patch_size` and `batch_size` defaults.
- A default residual CNN baseline.
- User-provided Keras models.
- Future BASS/NAS-generated Keras models.

Default optimization uses Charbonnier loss with AdamW. Charbonnier is the loss,
not an optimizer; Adam/AdamW performs the gradient updates. Learning rate is
dynamic by default through `ReduceLROnPlateau` on `val_psnr`, with early stopping
patience 20 and LR plateau patience 15 consecutive epochs without improvement.
Any real `val_psnr` improvement resets the plateau counter.

By default, `patch_size` and `batch_size` are resolved automatically from the
available GPU memory before training starts. The resolved values are saved in the
run `config.json`. You can still pass explicit values, or use `--max-batch-size`
and `--max-patch-size` as caps.

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
