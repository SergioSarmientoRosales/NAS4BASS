from __future__ import annotations

import tensorflow as tf

from evaluators.metrics._weights import collect_kernel_weights


def _collect_target_weights(model: tf.keras.Model):
    """
    Minimal weight collector for GraSP.

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


def _resolve_grasp_context(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    targets: tf.Tensor | None = None,
    loss_fn=None,
    input_shape: tuple[int, int, int] | None = None,
    batch_size: int | None = None,
    upscale_factor: int = 2,
):
    """
    Resolve a supervised SR context for GraSP.

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


def compute_grasp_per_weight(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    targets: tf.Tensor | None = None,
    loss_fn=None,
    T: float = 1.0,
    num_iters: int = 1,
    split_data: int = 1,
    input_shape: tuple[int, int, int] | None = None,
    batch_size: int | None = None,
    upscale_factor: int = 2,
):
    """
    TensorFlow/Keras adaptation of GraSP with dual compatibility:
    - benchmark mode: explicit inputs/targets/loss_fn
    - modular NAS mode: input_shape + batch_size

    Returns
    -------
    list[tf.Tensor]
        One per-weight GraSP tensor per collected weight tensor.
    """
    if split_data < 1:
        raise ValueError("split_data must be >= 1.")
    if num_iters < 1:
        raise ValueError("num_iters must be >= 1.")
    if T == 0:
        raise ValueError("T must be non-zero.")

    inputs, targets, loss_fn = _resolve_grasp_context(
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

    N = int(inputs.shape[0])

    # --------------------------------------------------------
    # Pass #1: accumulate grad_w
    # --------------------------------------------------------
    grad_w = [None] * len(weights)

    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        x = inputs[st:en]
        y = targets[st:en]

        if x.shape[0] == 0:
            continue

        for _ in range(num_iters):
            with tf.GradientTape() as tape:
                outputs = model(x, training=True) / T
                loss = loss_fn(y, outputs)
                loss = tf.reduce_mean(tf.cast(loss, tf.float32))

            grad_w_p = tape.gradient(loss, weights)

            for i, g in enumerate(grad_w_p):
                if g is None:
                    continue
                g = tf.cast(g, tf.float32)
                grad_w[i] = g if grad_w[i] is None else grad_w[i] + g

    # --------------------------------------------------------
    # Pass #2: compute Hg through z
    # --------------------------------------------------------
    grasp_hg = [None] * len(weights)

    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        x = inputs[st:en]
        y = targets[st:en]

        if x.shape[0] == 0:
            continue

        with tf.GradientTape() as outer_tape:
            with tf.GradientTape() as inner_tape:
                outputs = model(x, training=True) / T
                loss = loss_fn(y, outputs)
                loss = tf.reduce_mean(tf.cast(loss, tf.float32))

            grad_f = inner_tape.gradient(loss, weights)

            z_terms = []
            for gw, gf in zip(grad_w, grad_f):
                if gw is not None and gf is not None:
                    z_terms.append(
                        tf.reduce_sum(tf.stop_gradient(gw) * tf.cast(gf, tf.float32))
                    )

            if z_terms:
                z = tf.add_n(z_terms)
            else:
                z = tf.constant(0.0, dtype=tf.float32)

        grads_h = outer_tape.gradient(z, weights)

        for i, gh in enumerate(grads_h):
            if gh is None:
                continue
            gh = tf.cast(gh, tf.float32)
            grasp_hg[i] = gh if grasp_hg[i] is None else grasp_hg[i] + gh

    # --------------------------------------------------------
    # Final GraSP metric: -theta * Hg
    # --------------------------------------------------------
    grads = []
    for w, gh in zip(weights, grasp_hg):
        if gh is not None:
            grads.append(-tf.cast(w, tf.float32) * gh)
        else:
            grads.append(tf.zeros_like(w, dtype=tf.float32))

    return grads


def compute_grasp(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    targets: tf.Tensor | None = None,
    loss_fn=None,
    T: float = 1.0,
    num_iters: int = 1,
    split_data: int = 1,
    input_shape: tuple[int, int, int] | None = None,
    batch_size: int | None = None,
    upscale_factor: int = 2,
) -> float:
    """
    Scalar GraSP score, compatible with:
    - benchmark mode
    - modular NAS mode
    """
    grads = compute_grasp_per_weight(
        model=model,
        inputs=inputs,
        targets=targets,
        loss_fn=loss_fn,
        T=T,
        num_iters=num_iters,
        split_data=split_data,
        input_shape=input_shape,
        batch_size=batch_size,
        upscale_factor=upscale_factor,
    )

    if not grads:
        return 0.0

    return float(
        tf.add_n([tf.reduce_sum(tf.cast(g, tf.float64)) for g in grads]).numpy()
    )
