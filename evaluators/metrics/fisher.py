from __future__ import annotations

import tensorflow as tf


def _collect_target_layers(model: tf.keras.Model):
    """
    Minimal target-layer collector for Fisher.

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

            items.append((layer_type, layer_obj, weight_name))

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


def _resolve_fisher_context(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    targets: tf.Tensor | None = None,
    loss_fn=None,
    input_shape: tuple[int, int, int] | None = None,
    batch_size: int | None = None,
    upscale_factor: int = 2,
):
    """
    Resolve a supervised SR context for Fisher.

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


def _reduce_fisher_channel_metric(act: tf.Tensor, grad: tf.Tensor) -> tf.Tensor:
    """
    TensorFlow equivalent of the original PyTorch Fisher block:

        if len(act.shape) > 2:
            g_nk = torch.sum((act * grad), dims_spatial)
        else:
            g_nk = act * grad
        del_k = 0.5 * mean(g_nk^2, axis=0)

    Assumes channels-last tensors.
    """
    act = tf.cast(act, tf.float32)
    grad = tf.cast(grad, tf.float32)

    prod = act * grad

    if prod.shape.rank is None:
        raise ValueError("Activation rank must be known for Fisher computation.")

    if prod.shape.rank > 2:
        spatial_axes = list(range(1, prod.shape.rank - 1))
        g_nk = tf.reduce_sum(prod, axis=spatial_axes)
    else:
        g_nk = prod

    return 0.5 * tf.reduce_mean(tf.square(g_nk), axis=0)


def _broadcast_channel_metric(layer_type: str, channel_metric: tf.Tensor, kernel: tf.Tensor) -> tf.Tensor:
    """
    Broadcast channel Fisher values back to full weight tensor shape.
    """
    channel_metric = tf.cast(channel_metric, tf.float32)

    if layer_type == "Conv2D":
        return tf.broadcast_to(tf.reshape(channel_metric, [1, 1, 1, -1]), tf.shape(kernel))

    if layer_type == "Conv2DTranspose":
        return tf.broadcast_to(tf.reshape(channel_metric, [1, 1, -1, 1]), tf.shape(kernel))

    if layer_type == "Dense":
        return tf.broadcast_to(tf.reshape(channel_metric, [1, -1]), tf.shape(kernel))

    if layer_type == "DepthwiseConv2D":
        in_ch = int(kernel.shape[-2])
        depth_multiplier = int(kernel.shape[-1])
        expected = in_ch * depth_multiplier

        if int(channel_metric.shape[0]) != expected:
            raise ValueError(
                f"Depthwise Fisher metric length {int(channel_metric.shape[0])} "
                f"does not match expected {expected}."
            )

        return tf.broadcast_to(
            tf.reshape(channel_metric, [1, 1, in_ch, depth_multiplier]),
            tf.shape(kernel),
        )

    raise ValueError(f"Unsupported layer type: {layer_type}")


def compute_fisher_per_weight(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    targets: tf.Tensor | None = None,
    loss_fn=None,
    split_data: int = 1,
    input_shape: tuple[int, int, int] | None = None,
    batch_size: int | None = None,
    upscale_factor: int = 2,
):
    """
    TensorFlow/Keras Fisher adaptation with dual compatibility:
    - benchmark mode: explicit inputs/targets/loss_fn
    - modular NAS mode: input_shape + batch_size
    """
    if split_data < 1:
        raise ValueError("split_data must be >= 1.")

    inputs, targets, loss_fn = _resolve_fisher_context(
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
        return []

    if not hasattr(model, "inputs") or model.inputs is None:
        raise ValueError("Fisher requires a Keras model with accessible model.inputs/model.output.")

    activation_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[layer.output for _, layer, _ in target_layers] + [model.output],
    )

    fisher_accum = {}
    N = int(inputs.shape[0])

    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        x = inputs[st:en]
        y = targets[st:en]

        if x.shape[0] == 0:
            continue

        with tf.GradientTape(persistent=True) as tape:
            outputs = activation_model(x, training=True)
            activations = outputs[:-1]
            final_output = outputs[-1]

            loss = loss_fn(y, final_output)
            loss = tf.reduce_mean(tf.cast(loss, tf.float32))

        for (layer_type, layer, weight_name), act in zip(target_layers, activations):
            grad = tape.gradient(loss, act)
            if grad is None:
                continue

            del_k = _reduce_fisher_channel_metric(act, grad)

            if layer.name in fisher_accum:
                fisher_accum[layer.name] = fisher_accum[layer.name] + del_k
            else:
                fisher_accum[layer.name] = del_k

        del tape

    per_weight = []
    for layer_type, layer, weight_name in target_layers:
        kernel = getattr(layer, weight_name)

        if layer.name not in fisher_accum:
            per_weight.append(tf.zeros_like(kernel, dtype=tf.float32))
            continue

        channel_metric = tf.abs(fisher_accum[layer.name])
        per_weight.append(_broadcast_channel_metric(layer_type, channel_metric, kernel))

    return per_weight


def compute_fisher(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    targets: tf.Tensor | None = None,
    loss_fn=None,
    split_data: int = 1,
    input_shape: tuple[int, int, int] | None = None,
    batch_size: int | None = None,
    upscale_factor: int = 2,
) -> float:
    """
    Scalar Fisher score, compatible with:
    - benchmark mode
    - modular NAS mode
    """
    per_weight = compute_fisher_per_weight(
        model=model,
        inputs=inputs,
        targets=targets,
        loss_fn=loss_fn,
        split_data=split_data,
        input_shape=input_shape,
        batch_size=batch_size,
        upscale_factor=upscale_factor,
    )

    if not per_weight:
        return 0.0

    return float(
        tf.add_n([tf.reduce_sum(tf.abs(t)) for t in per_weight]).numpy()
    )