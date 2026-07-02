from __future__ import annotations

import tensorflow as tf


def paired_random_crop(
    lr: tf.Tensor,
    hr: tf.Tensor,
    *,
    hr_patch_size: int,
    scale: int,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    lr_patch_size = hr_patch_size // scale
    lr_shape = tf.shape(lr)
    max_y = lr_shape[0] - lr_patch_size + 1
    max_x = lr_shape[1] - lr_patch_size + 1

    y = tf.random.stateless_uniform([], seed=seed, minval=0, maxval=max_y, dtype=tf.int32)
    x = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([17, 29], dtype=tf.int32),
        minval=0,
        maxval=max_x,
        dtype=tf.int32,
    )

    lr_crop = lr[y:y + lr_patch_size, x:x + lr_patch_size, :]
    hr_y = y * scale
    hr_x = x * scale
    hr_crop = hr[hr_y:hr_y + hr_patch_size, hr_x:hr_x + hr_patch_size, :]
    return lr_crop, hr_crop


def paired_center_crop(
    lr: tf.Tensor,
    hr: tf.Tensor,
    *,
    hr_patch_size: int,
    scale: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    lr_patch_size = hr_patch_size // scale
    lr_shape = tf.shape(lr)

    y = tf.maximum(0, (lr_shape[0] - lr_patch_size) // 2)
    x = tf.maximum(0, (lr_shape[1] - lr_patch_size) // 2)

    lr_crop = lr[y:y + lr_patch_size, x:x + lr_patch_size, :]
    hr_y = y * scale
    hr_x = x * scale
    hr_crop = hr[hr_y:hr_y + hr_patch_size, hr_x:hr_x + hr_patch_size, :]
    return lr_crop, hr_crop


def paired_stateless_augment(
    lr: tf.Tensor,
    hr: tf.Tensor,
    *,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    lr = tf.image.stateless_random_flip_left_right(lr, seed=seed)
    hr = tf.image.stateless_random_flip_left_right(hr, seed=seed)

    seed2 = seed + tf.constant([1, 7], dtype=tf.int32)
    lr = tf.image.stateless_random_flip_up_down(lr, seed=seed2)
    hr = tf.image.stateless_random_flip_up_down(hr, seed=seed2)

    seed3 = seed + tf.constant([11, 19], dtype=tf.int32)
    k = tf.random.stateless_uniform([], seed=seed3, minval=0, maxval=4, dtype=tf.int32)
    lr = tf.image.rot90(lr, k=k)
    hr = tf.image.rot90(hr, k=k)
    return lr, hr
