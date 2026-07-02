# SRIR Training Pipeline

The `srir_training/` package is a standalone TensorFlow/Keras training pipeline
for super-resolution image restoration models. It is kept separate from the NAS
search logic so that training experiments can evolve without changing the search
implementation.

Use it for:

- x2, x3, or x4 SRIR training.
- DIV2K-style paired LR/HR folders.
- A default residual CNN baseline.
- User-provided Keras models.
- Future BASS/NAS-generated Keras models.

Default optimization uses Charbonnier loss with AdamW. Charbonnier is the loss,
not an optimizer; Adam/AdamW performs the gradient updates. Learning rate is
dynamic by default through `ReduceLROnPlateau` on `val_psnr`, with early stopping
patience 20 and LR plateau patience 15 consecutive epochs without improvement.
Any real `val_psnr` improvement resets the plateau counter.

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
