from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from srir_training.augmentations import (
    paired_center_crop,
    paired_random_crop,
    paired_stateless_augment,
)
from srir_training.config import DataConfig


AUTOTUNE = tf.data.AUTOTUNE
ALLOWED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


@dataclass(frozen=True)
class ImagePair:
    key: str
    lr_path: str
    hr_path: str


@dataclass(frozen=True)
class HRImage:
    key: str
    hr_path: str


@dataclass(frozen=True)
class DatasetInfo:
    pairs: int
    examples_per_epoch: int
    batch_size: int
    steps: int | None


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in ALLOWED_EXTS


def _canonical_stem(path: Path, *, scale: int, lr_suffix: str | None) -> str:
    stem = path.stem
    suffix = lr_suffix or f"x{scale}"
    if suffix and stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _list_images(directory: str) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    return sorted(path for path in root.iterdir() if _is_image(path))


def _sliding_step(cfg: DataConfig) -> int:
    return max(1, int(cfg.patch_size * (1.0 - cfg.validation_overlap)))


def count_patches_for_shape(height: int, width: int, cfg: DataConfig) -> int:
    if height < cfg.patch_size or width < cfg.patch_size:
        return 0
    step = _sliding_step(cfg)
    rows = ((height - cfg.patch_size) // step) + 1
    cols = ((width - cfg.patch_size) // step) + 1
    return int(rows * cols)


def get_image_hw(path: str, *, channels: int) -> tuple[int, int] | None:
    try:
        image = decode_image(tf.constant(path), channels=channels)
        shape = tf.shape(image)
        return int(shape[0].numpy()), int(shape[1].numpy())
    except Exception:
        return None


def count_sliding_patches(paths: list[str], cfg: DataConfig) -> tuple[int, int]:
    total_patches = 0
    images_used = 0
    for path in paths:
        hw = get_image_hw(path, channels=cfg.channels)
        if hw is None:
            continue
        patches = count_patches_for_shape(hw[0], hw[1], cfg)
        if patches > 0:
            total_patches += patches
            images_used += 1
    return total_patches, images_used


def _training_examples_per_epoch(paths: list[str], cfg: DataConfig) -> int:
    repeat_examples = len(paths) * cfg.repeats_per_image
    if cfg.epoch_steps_mode == "repeat":
        return repeat_examples

    total_patches, images_used = count_sliding_patches(paths, cfg)
    if total_patches > 0:
        print(
            "[DATA] patch_count epoch: images_used={0} patches={1}".format(
                images_used,
                total_patches,
            )
        )
        return total_patches

    print("[WARN] No sliding patches counted; falling back to repeats_per_image epoch sizing")
    return repeat_examples


def _require_resolved(cfg: DataConfig) -> None:
    if cfg.patch_size is None:
        raise ValueError("patch_size must be resolved before building datasets")
    if cfg.batch_size is None:
        raise ValueError("batch_size must be resolved before building datasets")


def collect_image_pairs(
    lr_dir: str,
    hr_dir: str,
    *,
    scale: int,
    lr_suffix: str | None = None,
) -> list[ImagePair]:
    lr_paths = _list_images(lr_dir)
    hr_paths = _list_images(hr_dir)

    if not lr_paths:
        raise ValueError(f"No LR images found in {lr_dir}")
    if not hr_paths:
        raise ValueError(f"No HR images found in {hr_dir}")

    lr_by_key = {}
    for path in lr_paths:
        key = _canonical_stem(path, scale=scale, lr_suffix=lr_suffix)
        if key in lr_by_key:
            raise ValueError(f"Duplicate LR key '{key}' in {lr_dir}")
        lr_by_key[key] = path

    hr_by_key = {}
    for path in hr_paths:
        key = _canonical_stem(path, scale=scale, lr_suffix=None)
        if key in hr_by_key:
            raise ValueError(f"Duplicate HR key '{key}' in {hr_dir}")
        hr_by_key[key] = path

    missing_hr = sorted(set(lr_by_key) - set(hr_by_key))
    missing_lr = sorted(set(hr_by_key) - set(lr_by_key))
    if missing_hr or missing_lr:
        raise ValueError(
            "LR/HR image pairs do not match. "
            f"Missing HR for {len(missing_hr)} LR files: {missing_hr[:5]}; "
            f"missing LR for {len(missing_lr)} HR files: {missing_lr[:5]}"
        )

    return [
        ImagePair(key=key, lr_path=str(lr_by_key[key]), hr_path=str(hr_by_key[key]))
        for key in sorted(lr_by_key)
    ]


def collect_hr_images(hr_dir: str) -> list[HRImage]:
    hr_paths = _list_images(hr_dir)
    if not hr_paths:
        raise ValueError(f"No HR images found in {hr_dir}")

    images = []
    seen = set()
    for path in hr_paths:
        key = path.stem
        if key in seen:
            raise ValueError(f"Duplicate HR key '{key}' in {hr_dir}")
        seen.add(key)
        images.append(HRImage(key=key, hr_path=str(path)))
    return images


def decode_image(path: tf.Tensor, *, channels: int) -> tf.Tensor:
    data = tf.io.read_file(path)
    image = tf.image.decode_image(data, channels=channels, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image.set_shape([None, None, channels])
    return image


def align_lr_hr(lr: tf.Tensor, hr: tf.Tensor, *, scale: int) -> tuple[tf.Tensor, tf.Tensor]:
    lr_shape = tf.shape(lr)
    hr_shape = tf.shape(hr)

    lr_h = tf.minimum(lr_shape[0], hr_shape[0] // scale)
    lr_w = tf.minimum(lr_shape[1], hr_shape[1] // scale)

    lr = lr[:lr_h, :lr_w, :]
    hr = hr[:lr_h * scale, :lr_w * scale, :]
    return lr, hr


def _load_pair(lr_path: tf.Tensor, hr_path: tf.Tensor, cfg: DataConfig):
    lr = decode_image(lr_path, channels=cfg.channels)
    hr = decode_image(hr_path, channels=cfg.channels)
    return align_lr_hr(lr, hr, scale=cfg.scale)


def _load_hr(hr_path: tf.Tensor, cfg: DataConfig):
    hr = decode_image(hr_path, channels=cfg.channels)
    shape = tf.shape(hr)
    hr_h = (shape[0] // cfg.scale) * cfg.scale
    hr_w = (shape[1] // cfg.scale) * cfg.scale
    return hr[:hr_h, :hr_w, :]


def _pair_is_large_enough(lr: tf.Tensor, cfg: DataConfig) -> tf.Tensor:
    lr_patch = cfg.patch_size // cfg.scale
    shape = tf.shape(lr)
    return tf.logical_and(shape[0] >= lr_patch, shape[1] >= lr_patch)


def _hr_is_large_enough(hr: tf.Tensor, cfg: DataConfig) -> tf.Tensor:
    shape = tf.shape(hr)
    return tf.logical_and(shape[0] >= cfg.patch_size, shape[1] >= cfg.patch_size)


def _dummy_pair(cfg: DataConfig) -> tuple[tf.Tensor, tf.Tensor]:
    lr_patch = cfg.patch_size // cfg.scale
    lr = tf.zeros([lr_patch, lr_patch, cfg.channels], dtype=tf.float32)
    hr = tf.zeros([cfg.patch_size, cfg.patch_size, cfg.channels], dtype=tf.float32)
    return lr, hr


def _set_pair_shapes(lr: tf.Tensor, hr: tf.Tensor, cfg: DataConfig):
    lr_patch = cfg.patch_size // cfg.scale
    lr.set_shape([lr_patch, lr_patch, cfg.channels])
    hr.set_shape([cfg.patch_size, cfg.patch_size, cfg.channels])
    return lr, hr


def _downsample_hr_crop(hr: tf.Tensor, cfg: DataConfig) -> tf.Tensor:
    lr_patch = cfg.patch_size // cfg.scale
    lr = tf.image.resize(
        hr,
        [lr_patch, lr_patch],
        method=cfg.downsample_method,
        antialias=True,
    )
    lr = tf.clip_by_value(lr, 0.0, 1.0)
    lr.set_shape([lr_patch, lr_patch, cfg.channels])
    return lr


def make_paired_dataset(
    pairs: list[ImagePair],
    cfg: DataConfig,
    *,
    training: bool,
    seed: int,
) -> tuple[tf.data.Dataset, DatasetInfo]:
    _require_resolved(cfg)
    if not pairs:
        raise ValueError("Cannot build a dataset from an empty pair list")

    lr_paths = [pair.lr_path for pair in pairs]
    hr_paths = [pair.hr_path for pair in pairs]

    ds = tf.data.Dataset.from_tensor_slices((lr_paths, hr_paths))
    examples_per_epoch = len(pairs)

    if training:
        ds = ds.shuffle(
            buffer_size=min(max(cfg.shuffle_buffer, 1), len(pairs)),
            seed=seed,
            reshuffle_each_iteration=True,
        )
        ds = ds.flat_map(
            lambda lr_path, hr_path: tf.data.Dataset.from_tensors((lr_path, hr_path)).repeat(
                cfg.repeats_per_image
            )
        )
        ds = ds.repeat()
        examples_per_epoch = _training_examples_per_epoch(hr_paths, cfg)

    counter = tf.data.experimental.Counter()
    ds = tf.data.Dataset.zip((ds, counter))

    def map_pair(paths, idx):
        lr_path, hr_path = paths
        lr, hr = _load_pair(lr_path, hr_path, cfg)
        ok = _pair_is_large_enough(lr, cfg)
        sample_seed = tf.stack(
            [tf.cast(seed, tf.int32), tf.cast(idx % (2**31 - 1), tf.int32)]
        )

        def make_real_pair():
            if training:
                lr_crop, hr_crop = paired_random_crop(
                    lr,
                    hr,
                    hr_patch_size=cfg.patch_size,
                    scale=cfg.scale,
                    seed=sample_seed,
                )
                if cfg.augment:
                    lr_crop, hr_crop = paired_stateless_augment(
                        lr_crop,
                        hr_crop,
                        seed=sample_seed + tf.constant([101, 313], dtype=tf.int32),
                    )
                return lr_crop, hr_crop

            return paired_center_crop(
                lr,
                hr,
                hr_patch_size=cfg.patch_size,
                scale=cfg.scale,
            )

        lr_out, hr_out = tf.cond(ok, make_real_pair, lambda: _dummy_pair(cfg))
        lr_out, hr_out = _set_pair_shapes(lr_out, hr_out, cfg)
        return lr_out, hr_out, ok

    ds = ds.map(map_pair, num_parallel_calls=AUTOTUNE)
    ds = ds.filter(lambda lr, hr, ok: ok)
    ds = ds.map(lambda lr, hr, ok: (lr, hr), num_parallel_calls=AUTOTUNE)

    if cfg.cache and not training:
        ds = ds.cache()

    ds = ds.batch(cfg.batch_size, drop_remainder=training)
    ds = ds.prefetch(AUTOTUNE)

    steps = max(1, examples_per_epoch // cfg.batch_size) if training else None
    info = DatasetInfo(
        pairs=len(pairs),
        examples_per_epoch=examples_per_epoch,
        batch_size=cfg.batch_size,
        steps=steps,
    )
    return ds, info


def make_hr_only_dataset(
    images: list[HRImage],
    cfg: DataConfig,
    *,
    training: bool,
    seed: int,
) -> tuple[tf.data.Dataset, DatasetInfo]:
    _require_resolved(cfg)
    if not images:
        raise ValueError("Cannot build a dataset from an empty HR image list")

    hr_paths = [image.hr_path for image in images]
    ds = tf.data.Dataset.from_tensor_slices(hr_paths)
    examples_per_epoch = len(images)

    if training:
        ds = ds.shuffle(
            buffer_size=min(max(cfg.shuffle_buffer, 1), len(images)),
            seed=seed,
            reshuffle_each_iteration=True,
        )
        ds = ds.flat_map(lambda hr_path: tf.data.Dataset.from_tensors(hr_path).repeat(cfg.repeats_per_image))
        ds = ds.repeat()
        examples_per_epoch = _training_examples_per_epoch(hr_paths, cfg)

    counter = tf.data.experimental.Counter()
    ds = tf.data.Dataset.zip((ds, counter))

    def map_hr(hr_path, idx):
        hr = _load_hr(hr_path, cfg)
        ok = _hr_is_large_enough(hr, cfg)
        sample_seed = tf.stack(
            [tf.cast(seed, tf.int32), tf.cast(idx % (2**31 - 1), tf.int32)]
        )

        def make_real_pair():
            if training:
                hr_shape = tf.shape(hr)
                max_y = hr_shape[0] - cfg.patch_size + 1
                max_x = hr_shape[1] - cfg.patch_size + 1
                y = tf.random.stateless_uniform(
                    [],
                    seed=sample_seed,
                    minval=0,
                    maxval=max_y,
                    dtype=tf.int32,
                )
                x = tf.random.stateless_uniform(
                    [],
                    seed=sample_seed + tf.constant([17, 29], dtype=tf.int32),
                    minval=0,
                    maxval=max_x,
                    dtype=tf.int32,
                )
                hr_crop = hr[y:y + cfg.patch_size, x:x + cfg.patch_size, :]
            else:
                hr_shape = tf.shape(hr)
                y = tf.maximum(0, (hr_shape[0] - cfg.patch_size) // 2)
                x = tf.maximum(0, (hr_shape[1] - cfg.patch_size) // 2)
                hr_crop = hr[y:y + cfg.patch_size, x:x + cfg.patch_size, :]

            lr_crop = _downsample_hr_crop(hr_crop, cfg)
            if training and cfg.augment:
                lr_crop, hr_crop = paired_stateless_augment(
                    lr_crop,
                    hr_crop,
                    seed=sample_seed + tf.constant([101, 313], dtype=tf.int32),
                )
            return lr_crop, hr_crop

        lr_out, hr_out = tf.cond(ok, make_real_pair, lambda: _dummy_pair(cfg))
        lr_out, hr_out = _set_pair_shapes(lr_out, hr_out, cfg)
        return lr_out, hr_out, ok

    ds = ds.map(map_hr, num_parallel_calls=AUTOTUNE)
    ds = ds.filter(lambda lr, hr, ok: ok)
    ds = ds.map(lambda lr, hr, ok: (lr, hr), num_parallel_calls=AUTOTUNE)

    if cfg.cache and not training:
        ds = ds.cache()

    ds = ds.batch(cfg.batch_size, drop_remainder=training)
    ds = ds.prefetch(AUTOTUNE)

    steps = max(1, examples_per_epoch // cfg.batch_size) if training else None
    info = DatasetInfo(
        pairs=len(images),
        examples_per_epoch=examples_per_epoch,
        batch_size=cfg.batch_size,
        steps=steps,
    )
    return ds, info


def _hr_sliding_val_generator(paths: list[str], cfg: DataConfig):
    patch_size = cfg.patch_size
    lr_patch = cfg.patch_size // cfg.scale
    step = _sliding_step(cfg)

    batch_lr = []
    batch_hr = []
    for path in paths:
        try:
            hr_image = decode_image(tf.constant(path), channels=cfg.channels).numpy()
        except Exception:
            continue

        height, width = hr_image.shape[:2]
        if height < patch_size or width < patch_size:
            continue

        for y in range(0, height - patch_size + 1, step):
            for x in range(0, width - patch_size + 1, step):
                hr = hr_image[y:y + patch_size, x:x + patch_size, :]
                lr = tf.image.resize(
                    hr,
                    [lr_patch, lr_patch],
                    method=cfg.downsample_method,
                    antialias=True,
                ).numpy()
                batch_lr.append(np.clip(lr, 0.0, 1.0).astype(np.float32))
                batch_hr.append(np.clip(hr, 0.0, 1.0).astype(np.float32))

                if len(batch_lr) == cfg.batch_size:
                    yield np.asarray(batch_lr, dtype=np.float32), np.asarray(batch_hr, dtype=np.float32)
                    batch_lr = []
                    batch_hr = []


def make_hr_only_sliding_val_dataset(
    images: list[HRImage],
    cfg: DataConfig,
) -> tuple[tf.data.Dataset, DatasetInfo]:
    _require_resolved(cfg)
    if not images:
        raise ValueError("Cannot build a dataset from an empty HR image list")

    hr_paths = [image.hr_path for image in images]
    total_patches, images_used = count_sliding_patches(hr_paths, cfg)
    steps = max(1, total_patches // cfg.batch_size)
    examples_per_epoch = steps * cfg.batch_size
    print(
        "[DATA] sliding validation: images_used={0} patches={1} steps={2}".format(
            images_used,
            total_patches,
            steps,
        )
    )

    lr_patch = cfg.patch_size // cfg.scale
    ds = tf.data.Dataset.from_generator(
        lambda: _hr_sliding_val_generator(hr_paths, cfg),
        output_signature=(
            tf.TensorSpec(
                shape=(cfg.batch_size, lr_patch, lr_patch, cfg.channels),
                dtype=tf.float32,
            ),
            tf.TensorSpec(
                shape=(cfg.batch_size, cfg.patch_size, cfg.patch_size, cfg.channels),
                dtype=tf.float32,
            ),
        ),
    )
    ds = ds.take(steps).prefetch(AUTOTUNE)

    info = DatasetInfo(
        pairs=len(images),
        examples_per_epoch=examples_per_epoch,
        batch_size=cfg.batch_size,
        steps=steps,
    )
    return ds, info


def build_train_val_datasets(cfg: DataConfig, *, seed: int):
    _require_resolved(cfg)
    train_hr_dir = cfg.train_hr_dir or cfg.directory_train
    val_hr_dir = cfg.val_hr_dir or cfg.directory_val
    paired_mode = bool(cfg.train_lr_dir or cfg.val_lr_dir)

    if paired_mode:
        if not (cfg.train_lr_dir and cfg.val_lr_dir and train_hr_dir and val_hr_dir):
            raise ValueError(
                "Paired LR/HR mode requires train_lr_dir, train_hr_dir, "
                "val_lr_dir, and val_hr_dir."
            )
        train_pairs = collect_image_pairs(
            cfg.train_lr_dir,
            train_hr_dir,
            scale=cfg.scale,
            lr_suffix=cfg.lr_suffix,
        )
        val_pairs = collect_image_pairs(
            cfg.val_lr_dir,
            val_hr_dir,
            scale=cfg.scale,
            lr_suffix=cfg.lr_suffix,
        )
        train_ds, train_info = make_paired_dataset(train_pairs, cfg, training=True, seed=seed)
        if cfg.validation_mode == "sliding":
            print("[WARN] Sliding validation is HR-only; using paired center-crop validation")
        val_ds, val_info = make_paired_dataset(val_pairs, cfg, training=False, seed=seed + 1)
        return train_ds, val_ds, train_info, val_info

    if not (train_hr_dir and val_hr_dir):
        raise ValueError(
            "HR-only mode requires directory_train/train_hr_dir and directory_val/val_hr_dir."
        )

    train_images = collect_hr_images(train_hr_dir)
    val_images = collect_hr_images(val_hr_dir)
    train_ds, train_info = make_hr_only_dataset(train_images, cfg, training=True, seed=seed)
    if cfg.validation_mode == "sliding":
        val_ds, val_info = make_hr_only_sliding_val_dataset(val_images, cfg)
    else:
        val_ds, val_info = make_hr_only_dataset(val_images, cfg, training=False, seed=seed + 1)
    return train_ds, val_ds, train_info, val_info


def make_synthetic_dataset(
    *,
    scale: int,
    patch_size: int,
    channels: int,
    batch_size: int,
    batches: int,
    seed: int,
) -> tf.data.Dataset:
    lr_patch = patch_size // scale

    def make_pair(idx):
        sample_seed = tf.stack([tf.cast(seed, tf.int32), tf.cast(idx, tf.int32)])
        hr = tf.random.stateless_uniform(
            [patch_size, patch_size, channels],
            seed=sample_seed,
            dtype=tf.float32,
        )
        lr = tf.image.resize(
            hr,
            [lr_patch, lr_patch],
            method="bicubic",
            antialias=True,
        )
        lr = tf.clip_by_value(lr, 0.0, 1.0)
        hr = tf.clip_by_value(hr, 0.0, 1.0)
        lr.set_shape([lr_patch, lr_patch, channels])
        hr.set_shape([patch_size, patch_size, channels])
        return lr, hr

    return (
        tf.data.Dataset.range(batch_size * batches)
        .map(make_pair, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
