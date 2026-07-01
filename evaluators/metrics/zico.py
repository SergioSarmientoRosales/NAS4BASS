from __future__ import annotations

import tensorflow as tf


def _collect_target_layers(model: tf.keras.Model):
    """
    Minimal layer collector for ZiCo / ZiCo-BC.

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

    items = []
    seen = set()

    for layer in layers_iter:
        candidates = []

        if isinstance(layer, tf.keras.layers.Conv2D):
            candidates.append(("Conv2D", layer, "kernel", getattr(layer, "kernel", None)))
        elif isinstance(layer, tf.keras.layers.Conv2DTranspose):
            candidates.append(("Conv2DTranspose", layer, "kernel", getattr(layer, "kernel", None)))
        elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            candidates.append(("DepthwiseConv2D", layer, "depthwise_kernel", getattr(layer, "depthwise_kernel", None)))
        elif isinstance(layer, tf.keras.layers.Dense):
            candidates.append(("Dense", layer, "kernel", getattr(layer, "kernel", None)))

        for layer_type, layer_obj, weight_name, var in candidates:
            if var is None:
                continue

            key = getattr(var, "path", None) or getattr(var, "name", None) or str(id(var))
            if key in seen:
                continue
            seen.add(key)

            items.append((layer_type, layer_obj, weight_name, var))

    return items


def _make_dummy_sr_batch(
    batch_size: int,
    input_shape: tuple[int, int, int],
    upscale_factor: int = 2,
    dtype: tf.dtypes.DType = tf.float32,
):
    h, w, c = input_shape
    hr_h = h * upscale_factor
    hr_w = w * upscale_factor

    lr = tf.random.uniform(
        shape=(batch_size, h, w, c),
        minval=0.0,
        maxval=1.0,
        dtype=dtype,
    )
    hr = tf.random.uniform(
        shape=(batch_size, hr_h, hr_w, c),
        minval=0.0,
        maxval=1.0,
        dtype=dtype,
    )
    return lr, hr


def _default_sr_loss():
    return tf.keras.losses.MeanSquaredError()


def _resolve_zico_context(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    targets: tf.Tensor | None = None,
    loss_fn=None,
    input_shape: tuple[int, int, int] | None = None,
    batch_size: int | None = None,
    upscale_factor: int = 2,
):
    """
    Resolve a supervised SR context for ZiCo / ZiCo-BC.

    Priority:
    1) Use explicit (inputs, targets, loss_fn) if provided
    2) Otherwise create a synthetic SR batch from (input_shape, batch_size)
    """
    if inputs is not None:
        if targets is None:
            raise ValueError("targets must be provided when inputs are provided.")
        if loss_fn is None:
            raise ValueError("loss_fn must be provided when inputs are provided.")

        if not model.built:
            _ = model(inputs[:1], training=False)

        return inputs, targets, loss_fn

    if input_shape is None or batch_size is None:
        raise ValueError(
            "Either (inputs, targets, loss_fn) or (input_shape, batch_size) must be provided."
        )

    dummy_inputs, dummy_targets = _make_dummy_sr_batch(
        batch_size=batch_size,
        input_shape=input_shape,
        upscale_factor=upscale_factor,
    )
    dummy_loss = _default_sr_loss()

    if not model.built:
        _ = model(dummy_inputs[:1], training=False)

    return dummy_inputs, dummy_targets, dummy_loss


def _compute_zico_base(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    targets: tf.Tensor | None = None,
    loss_fn=None,
    eps: float = 1e-8,
    input_shape: tuple[int, int, int] | None = None,
    batch_size: int | None = None,
    upscale_factor: int = 2,
) -> tuple[float, list[tuple[int, int, int]]]:
    """
    Computes the paper-style ZiCo base score and returns:
    - zico score
    - list of (H, W, C) tuples for bias correction
    """
    inputs, targets, loss_fn = _resolve_zico_context(
        model=model,
        inputs=inputs,
        targets=targets,
        loss_fn=loss_fn,
        input_shape=input_shape,
        batch_size=batch_size,
        upscale_factor=upscale_factor,
    )

    target_layers = _collect_target_layers(model)
    if not target_layers:
        return 0.0, []

    weights = [var for _, _, _, var in target_layers]
    batch_size_eff = int(inputs.shape[0])

    if batch_size_eff < 2:
        return float("nan"), []

    # Collect per-sample gradients
    per_sample_grads = [[] for _ in weights]

    for i in range(batch_size_eff):
        x = inputs[i:i + 1]
        y = targets[i:i + 1]

        with tf.GradientTape() as tape:
            outputs = model(x, training=True)
            loss = loss_fn(y, outputs)
            loss = tf.reduce_mean(tf.cast(loss, tf.float32))

        grads = tape.gradient(loss, weights)

        for j, g in enumerate(grads):
            if g is None:
                per_sample_grads[j].append(None)
            else:
                per_sample_grads[j].append(tf.cast(g, tf.float32))

    # ZiCo base score: sum_l log( sum |E[g]| / std(g) )
    zico_score = 0.0

    for grads_list in per_sample_grads:
        valid = [g for g in grads_list if g is not None]
        if len(valid) < 2:
            continue

        stacked = tf.stack(valid, axis=0)  # [B, ...]
        mean_g = tf.reduce_mean(stacked, axis=0)
        std_g = tf.math.reduce_std(stacked, axis=0)

        layer_score = tf.reduce_sum(tf.abs(mean_g) / (std_g + eps))
        zico_score += float(
            tf.math.log(tf.maximum(layer_score, tf.constant(eps, dtype=tf.float32))).numpy()
        )

    # Get concrete H, W, C for bias correction from actual activations
    if not hasattr(model, "inputs") or model.inputs is None:
        return zico_score, []

    activation_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[layer.output for _, layer, _, _ in target_layers],
    )

    acts = activation_model(inputs[:1], training=False)
    if not isinstance(acts, (list, tuple)):
        acts = [acts]

    hwc_list = []
    for act in acts:
        shape = act.shape
        if shape.rank == 4:
            H = int(shape[1])
            W = int(shape[2])
            C = int(shape[3])
        elif shape.rank == 2:
            H, W, C = 1, 1, int(shape[1])
        else:
            H, W, C = 1, 1, int(tf.size(act[0]).numpy())
        hwc_list.append((H, W, C))

    return zico_score, hwc_list


def compute_zico(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    targets: tf.Tensor | None = None,
    loss_fn=None,
    eps: float = 1e-8,
    input_shape: tuple[int, int, int] | None = None,
    batch_size: int | None = None,
    upscale_factor: int = 2,
) -> float:
    """
    Scalar ZiCo score, compatible with:
    - benchmark mode
    - modular NAS mode
    """
    score, _ = _compute_zico_base(
        model=model,
        inputs=inputs,
        targets=targets,
        loss_fn=loss_fn,
        eps=eps,
        input_shape=input_shape,
        batch_size=batch_size,
        upscale_factor=upscale_factor,
    )
    return float(score)


def compute_zico_bc(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    targets: tf.Tensor | None = None,
    loss_fn=None,
    beta: float = 1.0,
    eps: float = 1e-8,
    input_shape: tuple[int, int, int] | None = None,
    batch_size: int | None = None,
    upscale_factor: int = 2,
) -> float:
    """
    Scalar ZiCo-BC score, compatible with:
    - benchmark mode
    - modular NAS mode

    ZiCo-BC = ZiCo - beta * sum_l log(H_l * W_l * sqrt(C_l))
    """
    zico_score, hwc_list = _compute_zico_base(
        model=model,
        inputs=inputs,
        targets=targets,
        loss_fn=loss_fn,
        eps=eps,
        input_shape=input_shape,
        batch_size=batch_size,
        upscale_factor=upscale_factor,
    )

    if not hwc_list:
        return float(zico_score)

    penalty = 0.0
    for H, W, C in hwc_list:
        term = float(H) * float(W) * float(C) ** 0.5
        penalty += tf.math.log(
            tf.maximum(tf.constant(term, dtype=tf.float32), tf.constant(eps, dtype=tf.float32))
        ).numpy()

    return -float(zico_score - beta * penalty)