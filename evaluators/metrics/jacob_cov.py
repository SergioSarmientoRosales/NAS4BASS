from __future__ import annotations

import numpy as np
import tensorflow as tf


def _make_dummy_inputs(
    batch_size: int,
    input_shape: tuple[int, int, int],
    dtype: tf.dtypes.DType = tf.float32,
) -> tf.Tensor:
    return tf.random.uniform(
        shape=(batch_size,) + tuple(input_shape),
        minval=0.0,
        maxval=1.0,
        dtype=dtype,
    )


def _resolve_jacob_inputs(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    input_shape: tuple[int, int, int] | None = None,
    batch_size: int | None = None,
) -> tf.Tensor:
    """
    Resolve inputs for Jacobian covariance.

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

    dummy_inputs = _make_dummy_inputs(
        batch_size=batch_size,
        input_shape=input_shape,
        dtype=tf.float32,
    )

    if not model.built:
        _ = model(dummy_inputs[:1], training=False)

    return dummy_inputs


def get_batch_jacobian(model: tf.keras.Model, x: tf.Tensor, target=None):
    """
    TensorFlow equivalent of:
    - enable gradients on input
    - forward
    - backward with ones-like output sum
    - collect d(sum(output)) / dx
    """
    del target  # kept only for API consistency

    x = tf.identity(x)
    x = tf.cast(x, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x, training=False)
        objective = tf.reduce_sum(tf.cast(y, tf.float32))

    jacob = tape.gradient(objective, x)
    return jacob, None


def eval_score(jacob: np.ndarray, labels=None) -> float:
    """
    Minimal NumPy adaptation of the original score, with slightly safer
    eigenvalue handling.
    """
    del labels  # unused, kept for API consistency

    corrs = np.corrcoef(jacob)

    # Numerical stabilization
    corrs = np.asarray(corrs, dtype=np.float64)
    corrs = 0.5 * (corrs + corrs.T)

    # For symmetric correlation matrices, eigvalsh is more stable than eig
    v = np.linalg.eigvalsh(corrs)

    k = 1e-5
    v = np.clip(v, a_min=k, a_max=None)

    return float(-np.sum(np.log(v) + 1.0 / v))


def compute_jacob_cov(
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    targets=None,
    split_data: int = 1,
    loss_fn=None,
    input_shape: tuple[int, int, int] | None = None,
    batch_size: int | None = None,
) -> float:
    """
    TensorFlow/Keras adaptation of jacob_cov.

    Compatible with:
    - benchmark mode: explicit `inputs`
    - modular NAS mode: `input_shape` + `batch_size`

    Returns
    -------
    float
        Scalar Jacobian covariance score.
    """
    del targets, loss_fn  # unused for jacob_cov

    try:
        if split_data < 1:
            raise ValueError("split_data must be >= 1.")

        inputs = _resolve_jacob_inputs(
            model=model,
            inputs=inputs,
            input_shape=input_shape,
            batch_size=batch_size,
        )

        N = int(inputs.shape[0])
        jacobs = []

        for sp in range(split_data):
            st = sp * N // split_data
            en = (sp + 1) * N // split_data

            x = inputs[st:en]
            if x.shape[0] == 0:
                continue

            jacob, _ = get_batch_jacobian(model, x, None)
            if jacob is None:
                continue

            jacobs.append(tf.reshape(jacob, [jacob.shape[0], -1]).numpy())

        if not jacobs:
            return np.nan

        jacobs = np.concatenate(jacobs, axis=0)

        if jacobs.shape[0] < 2:
            return np.nan

        return eval_score(jacobs)

    except Exception as e:
        print(e)
        return np.nan