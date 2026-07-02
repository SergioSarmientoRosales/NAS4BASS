from __future__ import annotations

import tensorflow as tf


def _to_float01(y):
    return tf.cast(tf.clip_by_value(y, 0.0, 1.0), tf.float32)


@tf.keras.utils.register_keras_serializable(package="srir")
def psnr(y_true, y_pred):
    y_true = _to_float01(y_true)
    y_pred = _to_float01(y_pred)
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


@tf.keras.utils.register_keras_serializable(package="srir")
def ssim(y_true, y_pred):
    y_true = _to_float01(y_true)
    y_pred = _to_float01(y_pred)
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


@tf.keras.utils.register_keras_serializable(package="srir")
def rmse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


@tf.keras.utils.register_keras_serializable(package="srir")
def mae(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def default_metrics():
    return [psnr, ssim, mae, rmse]
