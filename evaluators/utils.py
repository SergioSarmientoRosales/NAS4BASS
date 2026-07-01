from __future__ import annotations

import inspect
from typing import Any, Callable

import tensorflow as tf


def ensure_model_built(
    model: tf.keras.Model,
    *,
    inputs: tf.Tensor | None = None,
    input_shape: tuple[int, int, int] | None = None,
) -> None:
    """
    Build the model if needed.

    Priority:
    1) use provided inputs
    2) otherwise build from input_shape with a dummy batch of ones
    """
    if model.built:
        return

    if inputs is not None:
        _ = model(inputs[:1], training=False)
        return

    if input_shape is None:
        raise ValueError(
            "Model is not built and no inputs/input_shape were provided to build it."
        )

    dummy = tf.ones((1,) + tuple(input_shape), dtype=tf.float32)
    _ = model(dummy, training=False)


def make_dummy_sr_batch(
    *,
    batch_size: int,
    input_shape: tuple[int, int, int],
    upscale_factor: int = 2,
    dtype: tf.dtypes.DType = tf.float32,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Create a synthetic LR/HR batch for SR zero-cost metrics when explicit
    inputs/targets are not provided.
    """
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


def default_sr_loss() -> tf.keras.losses.Loss:
    return tf.keras.losses.MeanSquaredError()


def resolve_supervised_sr_context(
    *,
    model: tf.keras.Model,
    inputs: tf.Tensor | None = None,
    targets: tf.Tensor | None = None,
    loss_fn=None,
    input_shape: tuple[int, int, int] | None = None,
    batch_size: int | None = None,
    upscale_factor: int = 2,
) -> tuple[tf.Tensor, tf.Tensor, Any]:
    """
    Resolve a supervised SR context.

    Priority:
    1) Use explicit (inputs, targets, loss_fn) if provided.
    2) Otherwise synthesize a dummy SR batch from (input_shape, batch_size).

    This is the key piece that lets the same zero-cost method work in:
    - the benchmark (explicit inputs/targets/loss_fn)
    - the NAS modular evaluator (input_shape/batch_size only)
    """
    if inputs is not None:
        if targets is None:
            raise ValueError(
                "targets must be provided when inputs are provided for supervised zero-cost metrics."
            )
        if loss_fn is None:
            raise ValueError(
                "loss_fn must be provided when inputs are provided for supervised zero-cost metrics."
            )

        ensure_model_built(model, inputs=inputs)
        return inputs, targets, loss_fn

    if input_shape is None or batch_size is None:
        raise ValueError(
            "Either (inputs, targets, loss_fn) or (input_shape, batch_size) must be provided."
        )

    dummy_inputs, dummy_targets = make_dummy_sr_batch(
        batch_size=batch_size,
        input_shape=input_shape,
        upscale_factor=upscale_factor,
    )
    dummy_loss = default_sr_loss()

    ensure_model_built(model, inputs=dummy_inputs)
    return dummy_inputs, dummy_targets, dummy_loss


def call_metric_with_supported_kwargs(metric_fn: Callable, **kwargs):
    """
    Call a metric function while passing only the kwargs it actually supports.

    This is very useful when different zero-cost implementations still have
    slightly different signatures during refactoring.
    """
    sig = inspect.signature(metric_fn)
    supported = {}

    for name, param in sig.parameters.items():
        if name in kwargs:
            supported[name] = kwargs[name]

    return metric_fn(**supported)