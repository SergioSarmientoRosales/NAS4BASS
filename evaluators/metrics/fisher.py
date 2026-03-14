# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


# Super-Resolution zero-cost proxy inspired by the Fisher pruning criterion.
#
# The original implementation estimates per-channel Fisher statistics by
# registering backward hooks on nn.Identity dummy operators inserted after
# Conv2d / Linear layers. This provides access to both the activation and
# its gradient at the same point in the computational graph.
#
# TensorFlow/Keras does not provide an equally simple, non-intrusive, and
# model-agnostic mechanism for reproducing that exact hook-based workflow:
#
#   - Wrapping layer.__call__ is fragile and can interfere with Keras internals.
#   - layer.output is limited to graph-compatible Functional models.
#   - Intrusive alternatives modify the model behaviour or training objective.
#
# For this reason, this adaptation uses a weight-based diagonal Fisher
# approximation computed from gradients with respect to kernel tensors.
# Although this is not algebraically identical to the original activation-
# based formulation, it captures a closely related notion of sensitivity:
# how strongly a layer's parameters participate in the mapping induced by
# the chosen surrogate objective.
#
# Correspondence with the original:
#
#   Original (activation-based)           This adaptation (weight-based)
#   ─────────────────────────────         ────────────────────────────────
#   hook captures act, grad_act           tape.gradient(obj, layer.kernel)
#   Fisher accumulated per sub-batch      Fisher accumulated per sub-batch
#   training-time forward pass            training-time forward pass
#   supervised loss(outputs, targets)     target-free surrogate objective
#   per-channel importance                per-layer scalar importance
#   final downstream aggregation          mean over layers


from __future__ import annotations

import numpy as np
import tensorflow as tf


def _collect_target_layers(
    model: tf.keras.Model,
) -> list[tf.keras.layers.Layer]:
    """
    Collect layers whose kernel weights will be used for Fisher estimation.

    This mirrors the Conv2d / Linear targeting logic used in the original
    implementation, adapted here to TensorFlow/Keras layers that expose a
    `kernel` attribute, such as Conv2D and Dense.

    Parameters
    ----------
    model : tf.keras.Model
        Candidate architecture.

    Returns
    -------
    list[tf.keras.layers.Layer]
        Ordered list of scoreable layers.
    """
    return [
        layer
        for layer in model.layers
        if hasattr(layer, "kernel") and layer.kernel is not None
    ]


def _compute_weight_fisher(grad_w: tf.Tensor) -> tf.Tensor:
    """
    Compute a scalar diagonal Fisher approximation from a kernel gradient.

    The original activation-based formulation computes Fisher-style importance
    from activation-gradient interactions. In this TensorFlow/Keras SR
    adaptation, the score is computed directly in weight space as:

        fisher_w = 0.5 * mean(grad_w^2)

    This yields a diagonal Fisher-inspired sensitivity estimate for the layer
    without requiring explicit activation capture.

    Parameters
    ----------
    grad_w : tf.Tensor
        Gradient of the surrogate objective with respect to a layer kernel.
        Typical shapes include:
        - [H, W, C_in, C_out] for Conv2D
        - [C_in, C_out] for Dense

    Returns
    -------
    tf.Tensor
        Scalar Fisher mass for the layer.
    """
    fisher = 0.5 * tf.reduce_mean(tf.square(grad_w))
    return tf.abs(fisher)


def compute_fisher(
    model: tf.keras.Model,
    input_shape: tuple[int, int, int] = (64, 64, 3),
    batch_size: int = 32,
    split_data: int = 1,
) -> float:
    """
    Compute an SR-oriented Fisher-inspired zero-cost proxy.

    The procedure preserves the overall accumulation logic of the original
    Fisher implementation where feasible within TensorFlow/Keras:

      1. Generate a synthetic input batch.
      2. Split the batch into `split_data` sub-batches.
      3. For each sub-batch:
           a. run a forward pass with training=True,
           b. define a lightweight surrogate scalar objective,
           c. compute gradients with respect to layer kernels,
           d. accumulate diagonal Fisher-style layer sensitivities.
      4. Return the mean accumulated Fisher mass across contributing layers.

    Because this is a target-free SR zero-cost proxy, the supervised loss from
    the original implementation is replaced by a simple differentiable
    surrogate:

        scalar_objective = tf.reduce_sum(model(x))

    This surrogate is not intended to represent reconstruction quality
    directly; it is used only to probe sensitivity in a lightweight and
    architecture-agnostic way.

    Parameters
    ----------
    model : tf.keras.Model
        Candidate architecture to evaluate.
    input_shape : tuple[int, int, int], default=(64, 64, 3)
        LR input shape (H, W, C).
    batch_size : int, default=32
        Total number of synthetic samples used to estimate the proxy.
    split_data : int, default=1
        Number of sub-batches used for accumulation. Increasing this value can
        reduce peak memory usage when batch_size is large.

    Returns
    -------
    float
        Mean Fisher-inspired mass across contributing layers. Higher values
        indicate stronger aggregate weight sensitivity under the surrogate
        objective. Returns np.nan if no valid gradients can be computed.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if split_data < 1:
        raise ValueError("split_data must be >= 1.")

    dummy_inputs = tf.random.normal((batch_size,) + input_shape, dtype=tf.float32)

    # Ensure variables exist before gradient computation.
    if not model.built:
        _ = model(dummy_inputs[:1], training=False)

    target_layers = _collect_target_layers(model)
    if not target_layers:
        return 0.0

    layer_fisher: dict[str, tf.Tensor] = {}

    try:
        n_samples = batch_size

        for sp in range(split_data):
            st = sp * n_samples // split_data
            en = (sp + 1) * n_samples // split_data
            sub_inputs = dummy_inputs[st:en]

            kernel_vars = [layer.kernel for layer in target_layers]

            with tf.GradientTape() as tape:
                # training=True is used to remain closer to the original Fisher
                # workflow, which evaluates the network in training mode.
                final_output = model(sub_inputs, training=True)

                # Lightweight SR surrogate objective.
                scalar_objective = tf.reduce_sum(final_output)

            grads = tape.gradient(scalar_objective, kernel_vars)

            for layer, grad_w in zip(target_layers, grads):
                if grad_w is None:
                    # Skip disconnected or non-contributing layers.
                    continue

                fisher_val = _compute_weight_fisher(grad_w)

                if layer.name in layer_fisher:
                    layer_fisher[layer.name] = layer_fisher[layer.name] + fisher_val
                else:
                    layer_fisher[layer.name] = fisher_val

        if not layer_fisher:
            return float(np.nan)

        # Mean over layers to avoid trivially favouring deeper architectures
        # through raw accumulation alone.
        layer_scores = [float(v.numpy()) for v in layer_fisher.values()]
        return float(np.mean(layer_scores))

    except Exception:
        return float(np.nan)