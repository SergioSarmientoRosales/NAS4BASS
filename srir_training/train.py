from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf

from srir_training.callbacks import build_callbacks
from srir_training.config import config_from_args, save_config
from srir_training.data import build_train_val_datasets
from srir_training.losses import get_loss
from srir_training.metrics import default_metrics
from srir_training.models import load_or_build_model
from srir_training.utils import (
    build_optimizer,
    configure_runtime,
    ensure_dir,
    get_strategy,
    save_json,
    set_global_seed,
    timestamp,
)


def build_run_dir(output_dir: str, run_name: str) -> Path:
    name = run_name.strip() if run_name else f"srir_run_{timestamp()}"
    return ensure_dir(Path(output_dir) / name)


def compile_model(model: tf.keras.Model, cfg) -> None:
    optimizer = build_optimizer(
        name=cfg.training.optimizer,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        global_clipnorm=cfg.training.global_clipnorm,
    )
    loss = get_loss(
        cfg.training.loss,
        charbonnier_epsilon=cfg.training.charbonnier_epsilon,
    )
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=default_metrics(),
        steps_per_execution=cfg.training.steps_per_execution,
    )


def save_history_json(history: tf.keras.callbacks.History, path: str | Path) -> None:
    serializable = {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }
    save_json(serializable, path)


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(argv)
    run_dir = build_run_dir(cfg.runtime.output_dir, cfg.runtime.run_name)

    set_global_seed(cfg.runtime.seed, deterministic=cfg.runtime.deterministic)
    runtime_info = configure_runtime(
        cpu=cfg.runtime.cpu,
        mixed_precision=cfg.runtime.mixed_precision,
        enable_xla=cfg.runtime.enable_xla,
    )
    save_config(cfg, run_dir / "config.json")
    save_json(runtime_info, run_dir / "runtime.json")

    print("[DATA] Building DIV2K-style paired LR/HR datasets")
    train_ds, val_ds, train_info, val_info = build_train_val_datasets(
        cfg.data,
        seed=cfg.runtime.seed,
    )

    steps_per_epoch = cfg.training.steps_per_epoch or train_info.steps
    validation_steps = cfg.training.validation_steps

    print(
        "[DATA] train_pairs={0} train_examples_per_epoch={1} steps_per_epoch={2}".format(
            train_info.pairs,
            train_info.examples_per_epoch,
            steps_per_epoch,
        )
    )
    print(f"[DATA] val_pairs={val_info.pairs}")

    strategy = get_strategy()
    print(
        f"[RUNTIME] Strategy={type(strategy).__name__} "
        f"replicas={strategy.num_replicas_in_sync}"
    )

    with strategy.scope():
        if cfg.training.resume_from:
            print(f"[MODEL] Resuming model from {cfg.training.resume_from}")
            model = tf.keras.models.load_model(cfg.training.resume_from, compile=False)
        else:
            model = load_or_build_model(cfg.model, cfg.data)
        compile_model(model, cfg)

    model.summary()
    callbacks = build_callbacks(cfg.training, run_dir)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.training.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    final_model_path = run_dir / "final.keras"
    model.save(final_model_path)
    save_history_json(history, run_dir / "history.json")

    best_model_path = run_dir / "checkpoints" / "best.keras"
    eval_model = model
    if best_model_path.exists():
        eval_model = tf.keras.models.load_model(best_model_path, compile=False)
        compile_model(eval_model, cfg)

    final_metrics = eval_model.evaluate(
        val_ds,
        steps=validation_steps,
        return_dict=True,
        verbose=1,
    )
    final_metrics = {key: float(value) for key, value in final_metrics.items()}
    save_json(final_metrics, run_dir / "final_metrics.json")

    print("\n[RESULT] Final validation metrics from best checkpoint when available:")
    print(json.dumps(final_metrics, indent=2, sort_keys=True))
    print(f"[RESULT] Best checkpoint: {best_model_path}")
    print(f"[RESULT] Final model: {final_model_path}")
    print(f"[RESULT] Run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
