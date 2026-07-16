from __future__ import annotations

import tensorflow as tf


def collect_kernel_weights(model: tf.keras.Model) -> list[tf.Variable]:
    """Collect each convolutional or dense kernel exactly once across Keras versions."""
    layers = (
        model._flatten_layers(include_self=False, recursive=True)
        if hasattr(model, "_flatten_layers")
        else model.layers
    )
    weights = []
    seen_ids: set[int] = set()
    for layer in layers:
        variable = None
        # In Keras 3, DepthwiseConv2D can satisfy the Conv2D check. Test the
        # specialized layers first so their depthwise kernel is not misclassified.
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            variable = getattr(layer, "depthwise_kernel", None)
            if variable is None:
                variable = getattr(layer, "kernel", None)
        elif isinstance(layer, tf.keras.layers.Conv2DTranspose):
            variable = getattr(layer, "kernel", None)
        elif isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            variable = getattr(layer, "kernel", None)
        if variable is None or id(variable) in seen_ids:
            continue
        seen_ids.add(id(variable))
        weights.append(variable)
    return weights
