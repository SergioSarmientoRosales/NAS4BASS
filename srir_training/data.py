from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

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


def _pair_is_large_enough(lr: tf.Tensor, cfg: DataConfig) -> tf.Tensor:
    lr_patch = cfg.patch_size // cfg.scale
    shape = tf.shape(lr)
    return tf.logical_and(shape[0] >= lr_patch, shape[1] >= lr_patch)


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


def make_paired_dataset(
    pairs: list[ImagePair],
    cfg: DataConfig,
    *,
    training: bool,
    seed: int,
) -> tuple[tf.data.Dataset, DatasetInfo]:
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
        examples_per_epoch = len(pairs) * cfg.repeats_per_image

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


def build_train_val_datasets(cfg: DataConfig, *, seed: int):
    train_pairs = collect_image_pairs(
        cfg.train_lr_dir,
        cfg.train_hr_dir,
        scale=cfg.scale,
        lr_suffix=cfg.lr_suffix,
    )
    val_pairs = collect_image_pairs(
        cfg.val_lr_dir,
        cfg.val_hr_dir,
        scale=cfg.scale,
        lr_suffix=cfg.lr_suffix,
    )

    train_ds, train_info = make_paired_dataset(train_pairs, cfg, training=True, seed=seed)
    val_ds, val_info = make_paired_dataset(val_pairs, cfg, training=False, seed=seed + 1)
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
