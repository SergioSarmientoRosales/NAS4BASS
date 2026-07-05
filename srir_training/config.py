from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    directory_train: str = ""
    directory_val: str = ""
    train_lr_dir: str = ""
    train_hr_dir: str = ""
    val_lr_dir: str = ""
    val_hr_dir: str = ""
    scale: int = 2
    patch_size: int | None = 64
    channels: int = 3
    batch_size: int | None = 64
    min_patch_size: int = 64
    max_patch_size: int = 128
    max_batch_size: int = 64
    repeats_per_image: int = 8
    validation_overlap: float = 0.1
    epoch_steps_mode: str = "patch_count"  # patch_count | repeat
    validation_mode: str = "sliding"  # sliding | center
    cache: bool = False
    shuffle_buffer: int = 512
    augment: bool = True
    lr_suffix: str | None = None
    downsample_method: str = "bicubic"


@dataclass
class ModelConfig:
    model_type: str = "residual_sr"
    custom_model_path: str = ""
    bass_gene: str = ""
    bass_gene_file: str = ""
    num_filters: int = 64
    num_res_blocks: int = 8
    residual_scale: float = 0.1
    activation: str = "relu"
    final_activation: str = "sigmoid"


@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 2e-4
    lr_schedule: str = "plateau"  # plateau | cosine | none
    warmup_epochs: int = 0
    cosine_alpha: float = 0.1
    optimizer: str = "adamw"
    weight_decay: float = 1e-8
    loss: str = "mse"
    charbonnier_epsilon: float = 1e-3
    global_clipnorm: float = 1.0
    steps_per_epoch: int | None = None
    validation_steps: int | None = None
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    reduce_lr_patience: int = 15
    reduce_lr_min_delta: float = 1e-4
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-7
    checkpoint_monitor: str = "val_psnr"
    checkpoint_mode: str = "max"
    steps_per_execution: int = 100
    resume_from: str = ""


@dataclass
class RuntimeConfig:
    output_dir: str = "srir_outputs"
    run_name: str = ""
    seed: int = 1
    deterministic: bool = False
    cpu: bool = False
    mixed_precision: str = "auto"  # auto | on | off
    enable_xla: bool = False


@dataclass
class TrainConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    runtime: RuntimeConfig

    @classmethod
    def defaults(cls) -> "TrainConfig":
        return cls(
            data=DataConfig(),
            model=ModelConfig(),
            training=TrainingConfig(),
            runtime=RuntimeConfig(),
        )

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainConfig":
        cfg = cls.defaults()
        _update_dataclass(cfg, values)
        validate_config(cfg)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _update_dataclass(instance: Any, values: dict[str, Any]) -> None:
    if not is_dataclass(instance):
        raise TypeError(f"Expected dataclass instance, got {type(instance)!r}")

    field_names = {field.name for field in fields(instance)}
    for key, value in values.items():
        if key not in field_names:
            raise ValueError(f"Unknown config field '{key}' for {type(instance).__name__}")

        current = getattr(instance, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"Config field '{key}' must be a dictionary")
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)


def load_config(path: str | None) -> TrainConfig:
    if not path:
        cfg = TrainConfig.defaults()
        validate_config(cfg)
        return cfg

    with open(path, "r", encoding="utf-8") as handle:
        values = json.load(handle)
    return TrainConfig.from_dict(values)


def validate_config(cfg: TrainConfig) -> None:
    if cfg.data.scale not in {2, 3, 4}:
        raise ValueError("scale must be one of 2, 3, or 4")
    if cfg.data.patch_size is not None:
        if cfg.data.patch_size <= 0 or cfg.data.patch_size % cfg.data.scale != 0:
            raise ValueError("patch_size must be positive and divisible by scale")
    if cfg.data.channels not in {1, 3}:
        raise ValueError("channels must be 1 or 3")
    if cfg.data.batch_size is not None and cfg.data.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if cfg.data.min_patch_size <= 0:
        raise ValueError("min_patch_size must be positive")
    if cfg.data.max_patch_size < cfg.data.min_patch_size:
        raise ValueError("max_patch_size must be greater than or equal to min_patch_size")
    if cfg.data.max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    if cfg.data.downsample_method not in {"area", "bicubic", "bilinear", "lanczos3", "lanczos5"}:
        raise ValueError("downsample_method must be one of: area, bicubic, bilinear, lanczos3, lanczos5")
    if cfg.data.repeats_per_image <= 0:
        raise ValueError("repeats_per_image must be positive")
    if not 0.0 <= cfg.data.validation_overlap < 1.0:
        raise ValueError("validation_overlap must be in [0, 1)")
    if cfg.data.epoch_steps_mode not in {"patch_count", "repeat"}:
        raise ValueError("epoch_steps_mode must be one of: patch_count, repeat")
    if cfg.data.validation_mode not in {"sliding", "center"}:
        raise ValueError("validation_mode must be one of: sliding, center")
    model_sources = [
        bool(cfg.model.custom_model_path),
        bool(cfg.model.bass_gene),
        bool(cfg.model.bass_gene_file),
    ]
    if sum(model_sources) > 1:
        raise ValueError("Use only one of custom_model_path, bass_gene, or bass_gene_file")
    if cfg.model.residual_scale <= 0:
        raise ValueError("residual_scale must be positive")
    if cfg.training.epochs <= 0:
        raise ValueError("epochs must be positive")
    if cfg.training.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if cfg.training.lr_schedule not in {"plateau", "cosine", "none"}:
        raise ValueError("lr_schedule must be one of: plateau, cosine, none")
    if cfg.training.warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")
    if cfg.training.warmup_epochs >= cfg.training.epochs:
        raise ValueError("warmup_epochs must be smaller than epochs")
    if not 0.0 <= cfg.training.cosine_alpha <= 1.0:
        raise ValueError("cosine_alpha must be in [0, 1]")
    if cfg.training.early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if cfg.training.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be non-negative")
    if cfg.training.reduce_lr_patience <= 0:
        raise ValueError("reduce_lr_patience must be positive")
    if cfg.training.reduce_lr_min_delta < 0:
        raise ValueError("reduce_lr_min_delta must be non-negative")
    if not 0.0 < cfg.training.reduce_lr_factor < 1.0:
        raise ValueError("reduce_lr_factor must be in (0, 1)")
    if cfg.training.min_learning_rate < 0:
        raise ValueError("min_learning_rate must be non-negative")
    if cfg.training.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if cfg.training.global_clipnorm is not None and cfg.training.global_clipnorm < 0:
        raise ValueError("global_clipnorm must be non-negative")
    if cfg.training.steps_per_epoch is not None and cfg.training.steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive when provided")
    if cfg.training.validation_steps is not None and cfg.training.validation_steps <= 0:
        raise ValueError("validation_steps must be positive when provided")
    if cfg.training.steps_per_execution <= 0:
        raise ValueError("steps_per_execution must be positive")
    if cfg.runtime.mixed_precision not in {"auto", "on", "off"}:
        raise ValueError("mixed_precision must be one of: auto, on, off")


def apply_scale_compatible_stage1_default(cfg: TrainConfig) -> None:
    if cfg.data.patch_size == 64 and cfg.data.scale == 3:
        cfg.data.patch_size = 96


def save_config(cfg: TrainConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(cfg.to_dict(), handle, indent=2, sort_keys=True)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train SRIR CNN models with TensorFlow/Keras.")
    parser.add_argument("--config", default=None, help="Optional JSON config file")
    parser.add_argument("--directory-train", "--train-dir", dest="directory_train", default=None)
    parser.add_argument("--directory-val", "--val-dir", dest="directory_val", default=None)
    parser.add_argument("--train-lr-dir", default=None)
    parser.add_argument("--train-hr-dir", default=None)
    parser.add_argument("--val-lr-dir", default=None)
    parser.add_argument("--val-hr-dir", default=None)
    parser.add_argument("--scale", type=int, choices=[2, 3, 4], default=None)
    parser.add_argument("--patch-size", type=int, default=None, help="HR patch size; default 64 for Stage 1 reference compatibility")
    parser.add_argument("--channels", type=int, choices=[1, 3], default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size; default 64 for Stage 1 reference compatibility")
    parser.add_argument("--auto-size", action="store_true", help="Resolve patch_size and batch_size from available GPU memory")
    parser.add_argument("--min-patch-size", type=int, default=None)
    parser.add_argument("--max-patch-size", type=int, default=None)
    parser.add_argument("--max-batch-size", type=int, default=None)
    parser.add_argument("--validation-overlap", type=float, default=None)
    parser.add_argument("--epoch-steps-mode", choices=["patch_count", "repeat"], default=None)
    parser.add_argument("--validation-mode", choices=["sliding", "center"], default=None)
    parser.add_argument("--downsample-method", choices=["area", "bicubic", "bilinear", "lanczos3", "lanczos5"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--lr-schedule", choices=["plateau", "cosine", "none"], default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--cosine-alpha", type=float, default=None)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default=None)
    parser.add_argument("--loss", choices=["charbonnier", "l1", "mae", "mse"], default=None)
    parser.add_argument("--global-clipnorm", type=float, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--validation-steps", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--reduce-lr-patience", type=int, default=None)
    parser.add_argument("--reduce-lr-min-delta", type=float, default=None)
    parser.add_argument("--reduce-lr-factor", type=float, default=None)
    parser.add_argument("--min-learning-rate", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--custom-model-path", default=None)
    parser.add_argument("--bass-gene", default=None, help="Decoded 28-gene BASS architecture, e.g. '[0, 1, ...]'")
    parser.add_argument("--bass-gene-file", default=None, help="JSON file containing a decoded BASS gene under key 'gene'")
    parser.add_argument("--mixed-precision", choices=["auto", "on", "off"], default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--enable-xla", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--lr-suffix", default=None)
    return parser.parse_args(argv)


def config_from_args(argv: list[str] | None = None) -> TrainConfig:
    args = parse_args(argv)
    cfg = load_config(args.config)

    overrides = {
        ("data", "directory_train"): args.directory_train,
        ("data", "directory_val"): args.directory_val,
        ("data", "train_lr_dir"): args.train_lr_dir,
        ("data", "train_hr_dir"): args.train_hr_dir,
        ("data", "val_lr_dir"): args.val_lr_dir,
        ("data", "val_hr_dir"): args.val_hr_dir,
        ("data", "scale"): args.scale,
        ("data", "patch_size"): args.patch_size,
        ("data", "channels"): args.channels,
        ("data", "batch_size"): args.batch_size,
        ("data", "min_patch_size"): args.min_patch_size,
        ("data", "max_patch_size"): args.max_patch_size,
        ("data", "max_batch_size"): args.max_batch_size,
        ("data", "validation_overlap"): args.validation_overlap,
        ("data", "epoch_steps_mode"): args.epoch_steps_mode,
        ("data", "validation_mode"): args.validation_mode,
        ("data", "downsample_method"): args.downsample_method,
        ("data", "lr_suffix"): args.lr_suffix,
        ("training", "epochs"): args.epochs,
        ("training", "learning_rate"): args.learning_rate,
        ("training", "lr_schedule"): args.lr_schedule,
        ("training", "warmup_epochs"): args.warmup_epochs,
        ("training", "cosine_alpha"): args.cosine_alpha,
        ("training", "optimizer"): args.optimizer,
        ("training", "loss"): args.loss,
        ("training", "global_clipnorm"): args.global_clipnorm,
        ("training", "steps_per_epoch"): args.steps_per_epoch,
        ("training", "validation_steps"): args.validation_steps,
        ("training", "early_stopping_patience"): args.early_stopping_patience,
        ("training", "reduce_lr_patience"): args.reduce_lr_patience,
        ("training", "reduce_lr_min_delta"): args.reduce_lr_min_delta,
        ("training", "reduce_lr_factor"): args.reduce_lr_factor,
        ("training", "min_learning_rate"): args.min_learning_rate,
        ("training", "resume_from"): args.resume_from,
        ("runtime", "output_dir"): args.output_dir,
        ("runtime", "run_name"): args.run_name,
        ("runtime", "seed"): args.seed,
        ("runtime", "mixed_precision"): args.mixed_precision,
        ("model", "custom_model_path"): args.custom_model_path,
        ("model", "bass_gene"): args.bass_gene,
        ("model", "bass_gene_file"): args.bass_gene_file,
    }

    for (section, key), value in overrides.items():
        if value is not None:
            setattr(getattr(cfg, section), key, value)

    if args.cpu:
        cfg.runtime.cpu = True
    if args.deterministic:
        cfg.runtime.deterministic = True
    if args.enable_xla:
        cfg.runtime.enable_xla = True
    if args.cache:
        cfg.data.cache = True
    if args.no_augment:
        cfg.data.augment = False
    if args.auto_size:
        cfg.data.patch_size = None
        cfg.data.batch_size = None
    elif args.config is None and args.patch_size is None:
        apply_scale_compatible_stage1_default(cfg)

    validate_config(cfg)
    return cfg
