from __future__ import annotations

import tensorflow as tf

from evaluators.metrics._weights import collect_kernel_weights


def _collect_target_weights(model: tf.keras.Model):
    """
    Minimal weight collector for L2 norm.

    Covers the layers that matter in the SR search space:
    - Conv2D              -> kernel
    - Conv2DTranspose     -> kernel
    - DepthwiseConv2D     -> depthwise_kernel
    - Dense               -> kernel
    """
    return collect_kernel_weights(model)


def _resolve_l2_build_context(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    input_shape: tuple[int, int, int] | None = None,
):
    """
    L2 norm does not need data, but the model may need to be built.

    Priority:
    1) Use explicit inputs if provided
    2) Otherwise build from input_shape with a dummy tensor
    """
    if model.built:
        return

    if inputs is not None:
        _ = model(inputs[:1], training=False)
        return

    if input_shape is None:
        raise ValueError(
            "L2 norm requires either `inputs` or `input_shape` when the model is not built."
        )

    dummy = tf.ones((1,) + tuple(input_shape), dtype=tf.float32)
    _ = model(dummy, training=False)


def get_l2_norm_array(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    targets=None,
    mode: str = "param",
    split_data: int = 1,
    input_shape: tuple[int, int, int] | None = None,
    batch_size: int | None = None,
    **kwargs,
):
    """
    TensorFlow adaptation of the original L2 norm proxy.

    Compatible with:
    - benchmark mode: explicit `inputs`
    - modular NAS mode: `input_shape`

    Returns one scalar L2 norm per collected weight tensor.
    """
    del targets, split_data, batch_size, kwargs  # unused, kept for API compatibility

    if mode != "param":
        raise ValueError("L2 norm only supports mode='param' in this adaptation.")

    _resolve_l2_build_context(
        model=model,
        inputs=inputs,
        input_shape=input_shape,
    )

    weights = _collect_target_weights(model)
    return [tf.norm(tf.cast(w, tf.float64), ord="euclidean") for w in weights]


def compute_l2_norm(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    targets=None,
    loss_fn=None,
    input_shape: tuple[int, int, int] | None = None,
    batch_size: int | None = None,
) -> float:
    """
    Scalar L2 norm score, compatible with:
    - benchmark mode
    - modular NAS mode
    """
    del targets, loss_fn, batch_size  # unused, kept for signature compatibility

    norms = get_l2_norm_array(
        model=model,
        inputs=inputs,
        mode="param",
        input_shape=input_shape,
    )

    if not norms:
        return 0.0

    return float(tf.add_n(norms).numpy())
