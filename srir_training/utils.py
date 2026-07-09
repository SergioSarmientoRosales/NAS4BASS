from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


@dataclass
class RuntimeInfo:
    python: str
    tensorflow: str
    keras: str
    platform: str
    devices: list[str]
    mixed_precision_policy: str
    git_commit: str | None


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception as exc:
            print(f"[WARN] Could not enable TensorFlow op determinism: {exc}")


def configure_runtime(*, cpu: bool, mixed_precision: str, enable_xla: bool) -> RuntimeInfo:
    if cpu:
        try:
            tf.config.set_visible_devices([], "GPU")
            print("[RUNTIME] CPU mode enabled")
        except Exception as exc:
            print(f"[WARN] Could not force CPU mode: {exc}")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus and not cpu:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as exc:
                print(f"[WARN] Could not enable memory growth for {gpu.name}: {exc}")

    if enable_xla:
        try:
            tf.config.optimizer.set_jit(True)
            print("[RUNTIME] XLA JIT enabled")
        except Exception as exc:
            print(f"[WARN] Could not enable XLA JIT: {exc}")

    use_mixed = mixed_precision == "on" or (mixed_precision == "auto" and bool(gpus) and not cpu)
    policy = "mixed_float16" if use_mixed else "float32"
    try:
        tf.keras.mixed_precision.set_global_policy(policy)
    except Exception as exc:
        print(f"[WARN] Could not set mixed precision policy '{policy}': {exc}")
        policy = str(tf.keras.mixed_precision.global_policy())

    devices = tf.config.list_logical_devices()
    print("[RUNTIME] TensorFlow:", tf.__version__)
    print("[RUNTIME] Devices:", ", ".join(d.name for d in devices) or "none")
    print("[RUNTIME] Mixed precision:", tf.keras.mixed_precision.global_policy())

    return RuntimeInfo(
        python=platform.python_version(),
        tensorflow=tf.__version__,
        keras=getattr(tf.keras, "__version__", "unknown"),
        platform=platform.platform(),
        devices=[d.name for d in devices],
        mixed_precision_policy=str(tf.keras.mixed_precision.global_policy()),
        git_commit=get_git_commit(Path.cwd()),
    )


def get_strategy() -> tf.distribute.Strategy:
    gpus = tf.config.list_logical_devices("GPU")
    if len(gpus) > 1:
        return tf.distribute.MirroredStrategy()
    return tf.distribute.get_strategy()


def get_git_commit(cwd: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(data, "__dataclass_fields__"):
        data = asdict(data)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def build_optimizer(
    *,
    name: str,
    learning_rate: float,
    weight_decay: float,
    epsilon: float,
    global_clipnorm: float | None,
) -> tf.keras.optimizers.Optimizer:
    kwargs = {
        "learning_rate": learning_rate,
        "epsilon": epsilon,
    }
    if global_clipnorm is not None and global_clipnorm > 0:
        kwargs["global_clipnorm"] = global_clipnorm

    if name == "adamw":
        try:
            return tf.keras.optimizers.AdamW(weight_decay=weight_decay, **kwargs)
        except AttributeError:
            print("[WARN] AdamW is unavailable in this TensorFlow build; falling back to Adam.")

    if name in {"adam", "adamw"}:
        return tf.keras.optimizers.Adam(**kwargs)

    raise ValueError(f"Unknown optimizer '{name}'")


def get_current_lr(optimizer: tf.keras.optimizers.Optimizer) -> float:
    lr = optimizer.learning_rate
    if callable(lr):
        lr = lr(optimizer.iterations)
    return float(tf.keras.backend.get_value(lr))


def set_current_lr(optimizer: tf.keras.optimizers.Optimizer, value: float) -> None:
    lr = optimizer.learning_rate
    if hasattr(lr, "assign"):
        lr.assign(float(value))
    else:
        optimizer.learning_rate = float(value)
