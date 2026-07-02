from __future__ import annotations

import math
from pathlib import Path

import tensorflow as tf

from srir_training.utils import get_current_lr, set_current_lr


class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["learning_rate"] = get_current_lr(self.model.optimizer)
        print(f"[LR] epoch={epoch + 1} learning_rate={logs['learning_rate']:.6e}")


class WarmupCosineLearningRate(tf.keras.callbacks.Callback):
    def __init__(
        self,
        *,
        initial_lr: float,
        min_lr: float,
        epochs: int,
        warmup_epochs: int,
        alpha: float,
    ):
        super().__init__()
        self.initial_lr = float(initial_lr)
        self.min_lr = float(min_lr)
        self.epochs = int(epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.alpha = float(alpha)

    def _lr_at_epoch(self, epoch: int) -> float:
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            return self.initial_lr * float(epoch + 1) / float(self.warmup_epochs)

        decay_epochs = max(1, self.epochs - self.warmup_epochs)
        decay_index = min(max(0, epoch - self.warmup_epochs), decay_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_index / decay_epochs))
        cosine_floor = self.initial_lr * self.alpha
        floor = max(self.min_lr, cosine_floor)
        return floor + (self.initial_lr - floor) * cosine

    def on_epoch_begin(self, epoch, logs=None):
        lr = self._lr_at_epoch(int(epoch))
        set_current_lr(self.model.optimizer, lr)
        print(f"[LR] warmup_cosine epoch={epoch + 1} learning_rate={lr:.6e}")


def build_lr_callbacks(training_cfg) -> list[tf.keras.callbacks.Callback]:
    if training_cfg.lr_schedule == "none":
        return []

    if training_cfg.lr_schedule == "cosine":
        return [
            WarmupCosineLearningRate(
                initial_lr=training_cfg.learning_rate,
                min_lr=training_cfg.min_learning_rate,
                epochs=training_cfg.epochs,
                warmup_epochs=training_cfg.warmup_epochs,
                alpha=training_cfg.cosine_alpha,
            )
        ]

    if training_cfg.lr_schedule == "plateau":
        return [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=training_cfg.checkpoint_monitor,
                mode=training_cfg.checkpoint_mode,
                factor=training_cfg.reduce_lr_factor,
                patience=training_cfg.reduce_lr_patience,
                min_delta=training_cfg.reduce_lr_min_delta,
                cooldown=0,
                min_lr=training_cfg.min_learning_rate,
                verbose=1,
            )
        ]

    raise RuntimeError(f"Unhandled lr_schedule '{training_cfg.lr_schedule}'")


def build_callbacks(training_cfg, run_dir: str | Path) -> list[tf.keras.callbacks.Callback]:
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    backup_dir = run_dir / "backup"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)

    monitor = training_cfg.checkpoint_monitor
    mode = training_cfg.checkpoint_mode

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / "best.keras"),
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / "last.keras"),
            save_best_only=False,
            verbose=0,
        ),
        tf.keras.callbacks.CSVLogger(str(log_dir / "history.csv"), append=False),
        tf.keras.callbacks.TensorBoard(log_dir=str(log_dir / "tensorboard")),
        tf.keras.callbacks.BackupAndRestore(backup_dir=str(backup_dir)),
    ]

    callbacks.extend(build_lr_callbacks(training_cfg))
    callbacks.extend(
        [
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=training_cfg.early_stopping_patience,
                min_delta=training_cfg.early_stopping_min_delta,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.TerminateOnNaN(),
            LearningRateLogger(),
        ]
    )

    return callbacks
