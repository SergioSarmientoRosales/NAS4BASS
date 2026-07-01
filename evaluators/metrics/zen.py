from __future__ import annotations

import numpy as np
import tensorflow as tf


def _iter_layers_recursive(model: tf.keras.Model):
    layers_iter = (
        model._flatten_layers(include_self=False, recursive=True)
        if hasattr(model, "_flatten_layers")
        else model.layers
    )
    for layer in layers_iter:
        yield layer


def _gaussian_reinit(model: tf.keras.Model):
    """
    Minimal TensorFlow analogue of the PyTorch gaussian init used by ZEN.
    """
    group_norm_cls = getattr(tf.keras.layers, "GroupNormalization", None)

    for layer in _iter_layers_recursive(model):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                              tf.keras.layers.Conv2DTranspose,
                              tf.keras.layers.Dense)):
            if getattr(layer, "kernel", None) is not None:
                layer.kernel.assign(
                    tf.random.normal(shape=layer.kernel.shape, dtype=layer.kernel.dtype)
                )
            if getattr(layer, "bias", None) is not None:
                layer.bias.assign(tf.zeros_like(layer.bias))

        elif isinstance(layer, tf.keras.layers.BatchNormalization) or (
            group_norm_cls is not None and isinstance(layer, group_norm_cls)
        ):
            if getattr(layer, "gamma", None) is not None:
                layer.gamma.assign(tf.ones_like(layer.gamma))
            if getattr(layer, "beta", None) is not None:
                layer.beta.assign(tf.zeros_like(layer.beta))


def _get_feature_model(model: tf.keras.Model) -> tf.keras.Model:
    """
    Minimal SR analogue of forward_before_global_avg_pool.

    We use the penultimate layer output as a generic pre-reconstruction feature.
    For SR models, this usually corresponds to the feature map right before
    the final bounded reconstruction layer.
    """
    if not hasattr(model, "inputs") or model.inputs is None:
        raise ValueError("ZEN requires a Keras model with accessible model.inputs/model.output.")

    if len(model.layers) < 2:
        return tf.keras.Model(inputs=model.inputs, outputs=model.output)

    return tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)


def _resolve_zen_shape(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    input_shape: tuple[int, int, int] | None = None,
    batch_size: int | None = None,
) -> tuple[int, int, int, int]:
    """
    Resolve the batch/input shape used by ZEN.

    Priority:
    1) If `inputs` is provided, use its full shape
    2) Otherwise build the model from (`input_shape`, `batch_size`) and use that shape
    """
    if inputs is not None:
        if not model.built:
            _ = model(inputs[:1], training=False)
        return tuple(inputs.shape)

    if input_shape is None or batch_size is None:
        raise ValueError(
            "Either `inputs` or (`input_shape`, `batch_size`) must be provided."
        )

    dummy = tf.ones((batch_size,) + tuple(input_shape), dtype=tf.float32)
    if not model.built:
        _ = model(dummy[:1], training=False)

    return tuple(dummy.shape)


def compute_zen_score(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    targets=None,
    loss_fn=None,
    split_data: int = 1,
    repeat: int = 1,
    mixup_gamma: float = 1e-2,
    fp16: bool = False,
    input_shape: tuple[int, int, int] | None = None,
    batch_size: int | None = None,
) -> float:
    """
    TensorFlow/Keras adaptation of ZEN for SR.

    Compatible with:
    - benchmark mode: explicit `inputs`
    - modular NAS mode: `input_shape` + `batch_size`

    Notes
    -----
    - Uses gaussian re-initialization inside the method, as in the original.
    - Uses the penultimate layer output as the SR analogue of the pre-GAP feature.
    - Restores original weights at the end so the caller is not contaminated.
    """
    del targets, loss_fn, split_data  # unused for ZEN

    if repeat < 1:
        raise ValueError("repeat must be >= 1.")

    full_shape = _resolve_zen_shape(
        model=model,
        inputs=inputs,
        input_shape=input_shape,
        batch_size=batch_size,
    )

    dtype = tf.float16 if fp16 else tf.float32

    # Save original weights because the same model object may be reused
    # by the benchmark or evaluator.
    original_values = [tf.identity(v) for v in model.weights]

    feature_model = _get_feature_model(model)
    nas_score_list = []

    try:
        for _ in range(repeat):
            _gaussian_reinit(model)

            x1 = tf.random.normal(shape=full_shape, dtype=dtype)
            x2 = tf.random.normal(shape=full_shape, dtype=dtype)
            mixup_x = x1 + mixup_gamma * x2

            feat1 = feature_model(x1, training=False)
            feat2 = feature_model(mixup_x, training=False)

            feat1 = tf.cast(feat1, tf.float32)
            feat2 = tf.cast(feat2, tf.float32)

            axes = list(range(1, feat1.shape.rank))
            nas_score = tf.reduce_sum(tf.abs(feat1 - feat2), axis=axes)
            nas_score = tf.reduce_mean(nas_score)

            # BN scaling term
            log_bn_scaling_factor = tf.constant(0.0, dtype=tf.float32)
            for layer in _iter_layers_recursive(model):
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    if getattr(layer, "moving_variance", None) is not None:
                        bn_scaling_factor = tf.sqrt(tf.reduce_mean(layer.moving_variance))
                        log_bn_scaling_factor += tf.math.log(
                            tf.maximum(bn_scaling_factor, tf.constant(1e-12, dtype=tf.float32))
                        )

            nas_score = tf.math.log(tf.maximum(nas_score, tf.constant(1e-12, dtype=tf.float32)))
            nas_score = nas_score + log_bn_scaling_factor
            nas_score_list.append(float(nas_score.numpy()))

        return float(np.mean(nas_score_list))

    finally:
        for v, orig in zip(model.weights, original_values):
            v.assign(orig)