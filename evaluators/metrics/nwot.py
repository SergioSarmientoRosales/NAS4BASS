from __future__ import annotations

import numpy as np
import tensorflow as tf


def _is_relu_activation(layer: tf.keras.layers.Layer) -> bool:
    act = getattr(layer, "activation", None)
    return act == tf.keras.activations.relu


def _collect_relu_outputs(model: tf.keras.Model):
    """
    Collect layers whose outputs should contribute to NWOT.

    We include:
    - explicit ReLU layers
    - Activation('relu') layers
    - layers with built-in activation='relu'
    """
    layers_iter = (
        model._flatten_layers(include_self=False, recursive=True)
        if hasattr(model, "_flatten_layers")
        else model.layers
    )

    relu_layers = []
    seen = set()

    for layer in layers_iter:
        use_layer = False

        if isinstance(layer, tf.keras.layers.ReLU):
            use_layer = True
        elif isinstance(layer, tf.keras.layers.Activation) and getattr(layer, "activation", None) == tf.keras.activations.relu:
            use_layer = True
        elif _is_relu_activation(layer):
            use_layer = True

        if not use_layer:
            continue

        key = layer.name
        if key in seen:
            continue
        seen.add(key)
        relu_layers.append(layer)

    return relu_layers


def _resolve_nwot_inputs(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    input_shape: tuple[int, int, int] | None = None,
    batch_size: int | None = None,
) -> tf.Tensor:
    """
    Resolve inputs for NWOT.

    Priority:
    1) Use explicit inputs if provided
    2) Otherwise build a random dummy batch from input_shape and batch_size
    """
    if inputs is not None:
        if not model.built:
            _ = model(inputs[:1], training=False)
        return inputs

    if input_shape is None or batch_size is None:
        raise ValueError(
            "Either `inputs` or (`input_shape`, `batch_size`) must be provided."
        )

    dummy_inputs = tf.random.uniform(
        shape=(batch_size,) + tuple(input_shape),
        minval=0.0,
        maxval=1.0,
        dtype=tf.float32,
    )

    if not model.built:
        _ = model(dummy_inputs[:1], training=False)

    return dummy_inputs


def compute_nwot(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    targets=None,
    loss_fn=None,
    split_data: int = 1,
    input_shape: tuple[int, int, int] | None = None,
    batch_size: int | None = None,
) -> float:
    """
    TensorFlow/Keras adaptation of NWOT.

    Compatible with:
    - benchmark mode: explicit `inputs`
    - modular NAS mode: `input_shape` + `batch_size`

    Uses the outputs of ReLU-activated layers and builds the binary kernel:
        K += X X^T + (1-X)(1-X)^T
    where X is the flattened binary activation pattern per sample.
    """
    del targets, loss_fn, split_data  # unused for NWOT

    try:
        inputs = _resolve_nwot_inputs(
            model=model,
            inputs=inputs,
            input_shape=input_shape,
            batch_size=batch_size,
        )

        relu_layers = _collect_relu_outputs(model)
        if not relu_layers:
            return np.nan

        if not hasattr(model, "inputs") or model.inputs is None:
            raise ValueError(
                "NWOT requires a Keras model with accessible model.inputs/model.output."
            )

        activation_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[layer.output for layer in relu_layers],
        )

        acts = activation_model(inputs, training=False)
        if not isinstance(acts, (list, tuple)):
            acts = [acts]

        n = int(inputs.shape[0])
        if n < 2:
            return np.nan

        K = np.zeros((n, n), dtype=np.float64)

        for act in acts:
            x = tf.reshape(act, [act.shape[0], -1])
            x = tf.cast(x > 0, tf.float32)

            x_np = x.numpy().astype(np.float64)
            K += x_np @ x_np.T + (1.0 - x_np) @ (1.0 - x_np).T

        sign, logdet = np.linalg.slogdet(K)
        if sign <= 0:
            return np.nan

        return float(logdet)

    except Exception as e:
        print(e)
        return np.nan