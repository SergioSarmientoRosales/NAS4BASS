from __future__ import annotations

import tensorflow as tf

from evaluators.metrics._weights import collect_kernel_weights


def _collect_target_weights(model: tf.keras.Model):
    """
    Minimal weight collector for GradNorm.

    Covers the layers that matter in the SR search space:
    - Conv2D              -> kernel
    - Conv2DTranspose     -> kernel
    - DepthwiseConv2D     -> depthwise_kernel
    - Dense               -> kernel
    """
    return collect_kernel_weights(model)


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


def _resolve_grad_norm_context(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    targets: tf.Tensor | None = None,
    loss_fn=None,
    input_shape: tuple[int, int, int] | None = None,
    batch_size: int | None = None,
    upscale_factor: int = 2,
):
    """
    Resolve a supervised SR context for GradNorm.

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


def get_grad_norm_arr(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    targets: tf.Tensor | None = None,
    loss_fn=None,
    split_data: int = 1,
    skip_grad: bool = False,
    input_shape: tuple[int, int, int] | None = None,
    batch_size: int | None = None,
    upscale_factor: int = 2,
):
    """
    TensorFlow adaptation of GradNorm with dual compatibility:
    - benchmark mode: explicit inputs/targets/loss_fn
    - modular NAS mode: input_shape + batch_size

    Returns
    -------
    list[tf.Tensor]
        One scalar grad norm per collected weight tensor.
    """
    del skip_grad  # kept only for API compatibility with the original

    if split_data < 1:
        raise ValueError("split_data must be >= 1.")

    inputs, targets, loss_fn = _resolve_grad_norm_context(
        model=model,
        inputs=inputs,
        targets=targets,
        loss_fn=loss_fn,
        input_shape=input_shape,
        batch_size=batch_size,
        upscale_factor=upscale_factor,
    )

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

    grad_norm_arr = []
    for g in grad_accum:
        if g is not None:
            grad_norm_arr.append(tf.norm(tf.cast(g, tf.float64), ord="euclidean"))
        else:
            grad_norm_arr.append(tf.constant(0.0, dtype=tf.float64))

    return grad_norm_arr


def compute_grad_norm(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    targets: tf.Tensor | None = None,
    loss_fn=None,
    split_data: int = 1,
    skip_grad: bool = False,
    input_shape: tuple[int, int, int] | None = None,
    batch_size: int | None = None,
    upscale_factor: int = 2,
) -> float:
    """
    Scalar GradNorm score, compatible with:
    - benchmark mode
    - modular NAS mode
    """
    grad_norm_arr = get_grad_norm_arr(
        model=model,
        inputs=inputs,
        targets=targets,
        loss_fn=loss_fn,
        split_data=split_data,
        skip_grad=skip_grad,
        input_shape=input_shape,
        batch_size=batch_size,
        upscale_factor=upscale_factor,
    )

    if not grad_norm_arr:
        return 0.0

    return float(tf.add_n(grad_norm_arr).numpy())
