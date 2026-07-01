from __future__ import annotations

import tensorflow as tf


def _collect_target_weights(model: tf.keras.Model):
    """
    Minimal weight collector for SynFlow.

    Covers the weights that matter in the SR search space:
    - kernel
    - depthwise_kernel
    - pointwise_kernel
    """
    layers_iter = (
        model._flatten_layers(include_self=False, recursive=True)
        if hasattr(model, "_flatten_layers")
        else model.layers
    )

    items = []
    seen = set()

    for layer in layers_iter:
        candidates = []

        if hasattr(layer, "kernel"):
            candidates.append(getattr(layer, "kernel", None))
        if hasattr(layer, "depthwise_kernel"):
            candidates.append(getattr(layer, "depthwise_kernel", None))
        if hasattr(layer, "pointwise_kernel"):
            candidates.append(getattr(layer, "pointwise_kernel", None))

        for var in candidates:
            if var is None:
                continue

            key = getattr(var, "path", None) or getattr(var, "name", None) or str(id(var))
            if key in seen:
                continue
            seen.add(key)
            items.append(var)

    return items


def _resolve_synflow_shape(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    input_shape: tuple[int, int, int] | None = None,
    batch_size: int | None = None,
) -> tuple[int, ...]:
    """
    Resolve the non-batch input shape for SynFlow.

    Priority:
    1) Use explicit inputs if provided
    2) Otherwise use input_shape from modular NAS mode
    """
    if inputs is not None:
        if not model.built:
            _ = model(inputs[:1], training=False)

        if inputs.shape.rank is None or inputs.shape.rank < 2:
            raise ValueError("SynFlow requires an input tensor with rank >= 2.")

        if any(dim is None for dim in inputs.shape[1:]):
            raise ValueError(
                "SynFlow requires concrete non-batch input dimensions. "
                "Pass a real input batch with known shape."
            )

        return tuple(inputs.shape[1:])

    if input_shape is None:
        raise ValueError(
            "Either `inputs` or `input_shape` must be provided."
        )

    # batch_size is not actually needed by SynFlow, but we accept it
    # to keep the same general signature as the other methods.
    del batch_size

    dummy = tf.ones((1,) + tuple(input_shape), dtype=tf.float32)
    if not model.built:
        _ = model(dummy, training=False)

    return tuple(input_shape)


def _make_dummy_input_from_shape(input_shape: tuple[int, ...], dtype=tf.float32) -> tf.Tensor:
    """
    Pure SynFlow uses an all-ones input with batch size 1.
    """
    return tf.ones(shape=(1,) + tuple(input_shape), dtype=dtype)


def compute_synflow(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    targets: tf.Tensor | None = None,
    loss_fn=None,
    input_shape: tuple[int, int, int] | None = None,
    batch_size: int | None = None,
) -> float:
    """
    Pure SynFlow, compatible with:
    - benchmark mode: explicit `inputs`
    - modular NAS mode: `input_shape` (+ optional `batch_size`)

    Definition:
    - linearize weights with abs()
    - use all-ones input
    - objective = sum(output)
    - score = sum |w * grad|
    """
    del targets, loss_fn  # unused for pure SynFlow

    resolved_shape = _resolve_synflow_shape(
        model=model,
        inputs=inputs,
        input_shape=input_shape,
        batch_size=batch_size,
    )

    weights = _collect_target_weights(model)
    if not weights:
        return 0.0

    all_trainable = list(model.trainable_variables)
    if not all_trainable:
        return 0.0

    original_values = [tf.identity(v) for v in all_trainable]
    dummy_input = _make_dummy_input_from_shape(resolved_shape, dtype=tf.float32)

    try:
        for v in all_trainable:
            if getattr(v.dtype, "is_floating", False):
                v.assign(tf.abs(v))

        with tf.GradientTape() as tape:
            outputs = model(dummy_input, training=False)
            objective = tf.reduce_sum(tf.cast(outputs, tf.float32))

        grads = tape.gradient(objective, weights)

        total_score = 0.0
        for w, g in zip(weights, grads):
            if g is None:
                continue
            total_score += float(
                tf.reduce_sum(
                    tf.abs(tf.cast(w, tf.float64) * tf.cast(g, tf.float64))
                ).numpy()
            )

        return total_score

    finally:
        for v, orig in zip(all_trainable, original_values):
            v.assign(orig)


def compute_synflow_raw(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    targets: tf.Tensor | None = None,
    loss_fn=None,
    input_shape: tuple[int, int, int] | None = None,
    batch_size: int | None = None,
) -> float:
    """
    Backward-compatible alias to the pure SynFlow implementation.
    """
    return compute_synflow(
        model=model,
        inputs=inputs,
        targets=targets,
        loss_fn=loss_fn,
        input_shape=input_shape,
        batch_size=batch_size,
    )