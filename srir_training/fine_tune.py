from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import tensorflow as tf

from srir_training.autosize import resolve_auto_data_config
from srir_training.config import (
    TrainConfig,
    apply_scale_compatible_stage1_default,
    load_config,
    save_config,
    validate_config,
)
from srir_training.data import build_train_val_datasets
from srir_training.train import compile_model, save_history_json
from srir_training.utils import (
    configure_runtime,
    ensure_dir,
    get_strategy,
    save_json,
    set_global_seed,
    timestamp,
)


def discover_best_models(models_dir: str, pattern: str = "**/best.keras") -> list[Path]:
    root = Path(models_dir)
    if not root.exists():
        raise FileNotFoundError(f"Models directory does not exist: {models_dir}")

    if root.is_file():
        matches = [root] if root.name == "best.keras" else []
    else:
        matches = sorted(path for path in root.glob(pattern) if path.is_file())

    if not matches:
        raise FileNotFoundError(
            f"No fine-tuning source models found under {models_dir!r} with pattern {pattern!r}"
        )

    return matches


def safe_run_name(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path

    parent = relative.parent.as_posix() if relative.parent.as_posix() != "." else path.stem
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", parent).strip("_")
    return name or path.stem


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Fine-tune all best.keras SRIR models found in a folder."
    )
    parser.add_argument("--config", default=None, help="Optional JSON config file")
    parser.add_argument("--models-dir", required=True, help="Folder containing best.keras models")
    parser.add_argument("--pattern", default="**/best.keras", help="Glob pattern inside --models-dir")
    parser.add_argument("--max-models", type=int, default=None, help="Optional cap for quick trials")

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
    parser.add_argument("--mixed-precision", choices=["auto", "on", "off"], default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--enable-xla", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--lr-suffix", default=None)
    return parser.parse_args(argv)


def fine_tune_config_from_args(args) -> TrainConfig:
    cfg = load_config(args.config)

    if args.config is None:
        cfg.training.epochs = 50
        cfg.training.learning_rate = 2e-5
        cfg.training.early_stopping_patience = 20
        cfg.training.reduce_lr_patience = 15
        cfg.runtime.output_dir = "srir_finetune_outputs"

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
        ("runtime", "output_dir"): args.output_dir,
        ("runtime", "run_name"): args.run_name,
        ("runtime", "seed"): args.seed,
        ("runtime", "mixed_precision"): args.mixed_precision,
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


def load_finetune_model(path: Path) -> tf.keras.Model:
    # Import custom layers so Keras 3 can resolve registered SRIR/BASS objects.
    import srir_training.models  # noqa: F401

    try:
        import search_space.model_builder  # noqa: F401
    except Exception:
        pass

    return tf.keras.models.load_model(path, compile=False)


def fine_tune_one_model(
    *,
    model_path: Path,
    source_root: Path,
    base_run_dir: Path,
    cfg: TrainConfig,
    train_ds,
    val_ds,
    steps_per_epoch: int | None,
    validation_steps: int | None,
    strategy: tf.distribute.Strategy,
) -> dict[str, float | str]:
    run_name = safe_run_name(model_path, source_root)
    run_dir = ensure_dir(base_run_dir / run_name)

    print(f"\n[FINETUNE] Source model: {model_path}")
    print(f"[FINETUNE] Output dir: {run_dir}")

    source_info = {
        "source_model": str(model_path),
        "source_root": str(source_root),
    }
    save_json(source_info, run_dir / "source_model.json")
    save_config(cfg, run_dir / "config.json")

    with strategy.scope():
        model = load_finetune_model(model_path)
        compile_model(model, cfg)

    from srir_training.callbacks import build_callbacks

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

    final_model_path = run_dir / "finetuned_final.keras"
    model.save(final_model_path)
    save_history_json(history, run_dir / "history.json")

    best_model_path = run_dir / "checkpoints" / "best.keras"
    eval_model = model
    if best_model_path.exists():
        with strategy.scope():
            eval_model = load_finetune_model(best_model_path)
            compile_model(eval_model, cfg)

    metrics = eval_model.evaluate(
        val_ds,
        steps=validation_steps,
        return_dict=True,
        verbose=1,
    )
    metrics = {key: float(value) for key, value in metrics.items()}
    save_json(metrics, run_dir / "final_metrics.json")

    print("[FINETUNE] Final validation metrics:")
    print(json.dumps(metrics, indent=2, sort_keys=True))

    tf.keras.backend.clear_session()

    return {
        "source_model": str(model_path),
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_model_path),
        "final_model": str(final_model_path),
        **metrics,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = fine_tune_config_from_args(args)

    set_global_seed(cfg.runtime.seed, deterministic=cfg.runtime.deterministic)
    runtime_info = configure_runtime(
        cpu=cfg.runtime.cpu,
        mixed_precision=cfg.runtime.mixed_precision,
        enable_xla=cfg.runtime.enable_xla,
    )
    resolve_auto_data_config(cfg.data, cfg.model, cfg.runtime)
    validate_config(cfg)

    model_paths = discover_best_models(args.models_dir, pattern=args.pattern)
    if args.max_models is not None:
        model_paths = model_paths[: args.max_models]

    source_root = Path(args.models_dir).resolve()
    root_name = cfg.runtime.run_name or f"srir_finetune_{timestamp()}"
    base_run_dir = ensure_dir(Path(cfg.runtime.output_dir) / root_name)
    save_json(runtime_info, base_run_dir / "runtime.json")

    print(f"[FINETUNE] Found {len(model_paths)} model(s)")
    print("[DATA] Building DIV2K-style train/validation datasets")
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

    summary = []
    for model_path in model_paths:
        summary.append(
            fine_tune_one_model(
                model_path=model_path.resolve(),
                source_root=source_root,
                base_run_dir=base_run_dir,
                cfg=cfg,
                train_ds=train_ds,
                val_ds=val_ds,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                strategy=strategy,
            )
        )

    save_json(summary, base_run_dir / "fine_tune_summary.json")
    print("\n[FINETUNE] Summary:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[FINETUNE] Output root: {base_run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
