from __future__ import annotations

import tensorflow as tf


def compute_synflow(model: tf.keras.Model, input_shape: tuple = (64, 64, 3)) -> float:
    """TensorFlow/Keras adaptation of SynFlow (Zero-Cost NAS proxy)."""
    if not model.built:
        _ = model(tf.ones((1,) + input_shape, dtype=tf.float32), training=False)

    target_layers = [
        layer for layer in model.layers
        if hasattr(layer, "kernel") and layer.kernel is not None
    ]
    if not target_layers:
        return 0.0

    def linearize(layers):
        originals = [layer.kernel.numpy().copy() for layer in layers]
        for layer in layers:
            layer.kernel.assign(tf.abs(layer.kernel))
        return originals

    def nonlinearize(layers, originals):
        for layer, orig in zip(layers, originals):
            layer.kernel.assign(orig)

    originals = linearize(target_layers)

    try:
        input_ones = tf.ones((1,) + input_shape, dtype=tf.float32)
        kernels = [layer.kernel for layer in target_layers]

        with tf.GradientTape() as tape:
            output = model(input_ones, training=False)
            loss = tf.reduce_sum(output)

        grads = tape.gradient(loss, kernels)

        total_score = 0.0
        total_params = 0
        for layer, grad in zip(target_layers, grads):
            if grad is not None:
                total_score += float(tf.reduce_sum(tf.abs(layer.kernel * grad)).numpy())
            total_params += int(tf.size(layer.kernel).numpy())

    finally:
        nonlinearize(target_layers, originals)

    return total_score / max(total_params, 1)