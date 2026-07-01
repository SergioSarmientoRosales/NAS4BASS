from __future__ import annotations

import tensorflow as tf


def _resolve_param_score_build_context(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    input_shape: tuple[int, int, int] | None = None,
):
    """
    Param score does not need data, but the model may need to be built.

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
            "Param score requires either `inputs` or `input_shape` when the model is not built."
        )

    dummy = tf.ones((1,) + tuple(input_shape), dtype=tf.float32)
    _ = model(dummy, training=False)


def compute_param_score(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    targets=None,
    loss_fn=None,
    input_shape: tuple[int, int, int] | None = None,
    batch_size: int | None = None,
) -> float:
    """
    Parameter-count score, compatible with:
    - benchmark mode: explicit `inputs`
    - modular NAS mode: `input_shape`

    Returns the negative number of parameters so that
    smaller models receive larger scores.
    """
    del targets, loss_fn, batch_size  # unused, kept for signature compatibility

    _resolve_param_score_build_context(
        model=model,
        inputs=inputs,
        input_shape=input_shape,
    )

    return float(model.count_params())