from __future__ import annotations

import tensorflow as tf


def _collect_target_weights(model: tf.keras.Model):
    """
    Minimal weight collector for SNIP.

    Covers the layers that matter in the SR search space:
    - Conv2D              -> kernel
    - Conv2DTranspose     -> kernel
    - DepthwiseConv2D     -> depthwise_kernel
    - Dense               -> kernel
    """
    layers_iter = (
        model._flatten_layers(include_self=False, recursive=True)
        if hasattr(model, "_flatten_layers")
        else model.layers
    )

    weights = []
    seen = set()

    for layer in layers_iter:
        candidates = []

        if isinstance(layer, tf.keras.layers.Conv2D):
            candidates.append(getattr(layer, "kernel", None))
        elif isinstance(layer, tf.keras.layers.Conv2DTranspose):
            candidates.append(getattr(layer, "kernel", None))
        elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            candidates.append(getattr(layer, "depthwise_kernel", None))
        elif isinstance(layer, tf.keras.layers.Dense):
            candidates.append(getattr(layer, "kernel", None))

        for var in candidates:
            if var is None:
                continue

            key = getattr(var, "path", None) or getattr(var, "name", None) or str(id(var))
            if key in seen:
                continue
            seen.add(key)
            weights.append(var)

    return weights


def compute_snip_per_weight(
    model: tf.keras.Model,
    inputs: tf.Tensor,
    targets: tf.Tensor | None,
    loss_fn,
    split_data: int = 1,
):
    """
    TensorFlow/Keras adaptation of SNIP.

    Instead of explicitly introducing mask variables, we use the equivalent
    score at mask=1:
        abs(dL/dM) = abs(W * dL/dW)
    """
    if targets is None:
        raise ValueError("SNIP requires real targets.")
    if loss_fn is None:
        raise ValueError("SNIP requires a loss function.")
    if split_data < 1:
        raise ValueError("split_data must be >= 1.")

    if not model.built:
        _ = model(inputs[:1], training=False)

    weights = _collect_target_weights(model)
    if not weights:
        return []

    grad_accum = [None] * len(weights)

    N = int(inputs.shape[0])
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        x = inputs[st:en]
        y = targets[st:en]

        if x.shape[0] == 0:
            continue

        with tf.GradientTape() as tape:
            outputs = model(x, training=True)
            loss = loss_fn(y, outputs)
            loss = tf.reduce_mean(tf.cast(loss, tf.float32))

        grads = tape.gradient(loss, weights)

        for i, g in enumerate(grads):
            if g is None:
                continue
            g = tf.cast(g, tf.float32)
            grad_accum[i] = g if grad_accum[i] is None else grad_accum[i] + g

    per_weight = []
    for w, g in zip(weights, grad_accum):
        if g is not None:
            per_weight.append(tf.abs(tf.cast(w, tf.float32) * g))
        else:
            per_weight.append(tf.zeros_like(w, dtype=tf.float32))

    return per_weight


def compute_snip(
    model: tf.keras.Model,
    inputs: tf.Tensor,
    targets: tf.Tensor | None = None,
    loss_fn=None,
    split_data: int = 1,
) -> float:
    """
    Scalar SNIP score for the benchmark.
    """
    per_weight = compute_snip_per_weight(
        model=model,
        inputs=inputs,
        targets=targets,
        loss_fn=loss_fn,
        split_data=split_data,
    )

    if not per_weight:
        return 0.0

    return float(
        tf.add_n([tf.reduce_sum(tf.cast(t, tf.float64)) for t in per_weight]).numpy()
    )