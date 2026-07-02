from __future__ import annotations

import argparse

import tensorflow as tf

from srir_training.config import DataConfig, ModelConfig, RuntimeConfig, TrainConfig, TrainingConfig
from srir_training.data import make_synthetic_dataset
from srir_training.losses import get_loss
from srir_training.metrics import default_metrics
from srir_training.models import load_or_build_model
from srir_training.train import compile_model
from srir_training.utils import configure_runtime, set_global_seed


def gradients_are_finite(model: tf.keras.Model, loss_fn, batch) -> bool:
    lr, hr = batch
    with tf.GradientTape() as tape:
        sr = model(lr, training=True)
        loss = loss_fn(hr, sr)
    grads = tape.gradient(loss, model.trainable_variables)
    grads = [g for g in grads if g is not None]
    if not grads:
        return False
    return bool(all(tf.reduce_all(tf.math.is_finite(g)).numpy() for g in grads))


def run_scale_smoke(scale: int, *, cpu: bool) -> None:
    cfg = TrainConfig(
        data=DataConfig(scale=scale, patch_size=24, channels=3, batch_size=2),
        model=ModelConfig(num_filters=8, num_res_blocks=1),
        training=TrainingConfig(epochs=1, learning_rate=1e-4, steps_per_epoch=1),
        runtime=RuntimeConfig(seed=123, cpu=cpu, mixed_precision="off"),
    )

    train_ds = make_synthetic_dataset(
        scale=scale,
        patch_size=cfg.data.patch_size,
        channels=cfg.data.channels,
        batch_size=cfg.data.batch_size,
        batches=2,
        seed=cfg.runtime.seed,
    )
    val_ds = make_synthetic_dataset(
        scale=scale,
        patch_size=cfg.data.patch_size,
        channels=cfg.data.channels,
        batch_size=cfg.data.batch_size,
        batches=1,
        seed=cfg.runtime.seed + 1,
    )

    model = load_or_build_model(cfg.model, cfg.data)
    compile_model(model, cfg)

    batch = next(iter(train_ds))
    lr, hr = batch
    sr = model(lr, training=False)
    expected_shape = tuple(hr.shape)
    got_shape = tuple(sr.shape)
    if got_shape != expected_shape:
        raise AssertionError(f"x{scale}: expected output shape {expected_shape}, got {got_shape}")

    loss_fn = get_loss(cfg.training.loss)
    if not gradients_are_finite(model, loss_fn, batch):
        raise AssertionError(f"x{scale}: gradients are not finite")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1,
        steps_per_epoch=1,
        validation_steps=1,
        verbose=0,
    )
    metrics = model.evaluate(val_ds, steps=1, return_dict=True, verbose=0)

    print(
        f"[SMOKE] x{scale} OK | output_shape={got_shape} | "
        f"loss={metrics['loss']:.6f} | psnr={metrics['psnr']:.4f} | ssim={metrics['ssim']:.4f}"
    )
    if not history.history:
        raise AssertionError(f"x{scale}: empty training history")


def parse_args():
    parser = argparse.ArgumentParser(description="Run a synthetic SRIR training smoke test.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument(
        "--scale",
        type=int,
        choices=[2, 3, 4],
        default=None,
        help="Run one scale only. By default x2, x3, and x4 are tested.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_global_seed(123, deterministic=False)
    configure_runtime(cpu=args.cpu, mixed_precision="off", enable_xla=False)

    scales = [args.scale] if args.scale is not None else [2, 3, 4]
    for scale in scales:
        run_scale_smoke(scale, cpu=args.cpu)

    print("[SMOKE] All requested scales passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
