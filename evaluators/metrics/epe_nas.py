# MIT License
#
# Copyright (c) 2021 VascoLopes
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# Super-Resolution zero-cost proxy inspired by EPE-NAS:
# Efficient Performance Estimation Without Training for Neural Architecture Search
# Paper: https://arxiv.org/abs/2102.08099
# Original reference implementation:
# https://github.com/VascoLopes/EPE-NAS/blob/main/search.py
#
# This implementation is NOT a direct reimplementation of the original EPE-NAS.
# The original method is class-conditioned and relies on per-class Jacobian
# correlation statistics for classification tasks. In contrast, this TensorFlow/
# Keras version is designed for Super-Resolution (SR), where categorical labels
# are typically unavailable. Accordingly, this module implements an SR-oriented
# Jacobian-correlation proxy inspired by the original EPE-NAS intuition:
# architectures are scored based on the diversity of their input-output Jacobian
# responses across a synthetic batch.


from __future__ import annotations

import numpy as np
import tensorflow as tf


def get_batch_jacobian(
    model: tf.keras.Model,
    inputs: tf.Tensor,
) -> tuple[np.ndarray, int]:
    """
    Compute a batch of flattened input-output Jacobian signatures.

    This helper computes the gradient of a scalar surrogate objective with
    respect to the input batch. For SR, model outputs are dense image tensors
    rather than class logits, so a task-agnostic scalar objective is required.
    Here, the sum of all output activations is used as a simple differentiable
    surrogate that exposes input-output sensitivity.

    Parameters
    ----------
    model : tf.keras.Model
        Candidate architecture to evaluate.
    inputs : tf.Tensor
        Input batch tensor with shape [B, H, W, C].

    Returns
    -------
    tuple[np.ndarray, int]
        - Flattened Jacobian matrix with shape [B, D], where D = H * W * C
        - Batch size B
    """
    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(inputs)
        outputs = model(inputs, training=False)

        # SR-oriented scalar surrogate.
        # This is not task-specific reconstruction loss; it is a lightweight
        # differentiable scalar used only to probe input-output sensitivity.
        scalar_objective = tf.reduce_sum(outputs)

    jacobian = tape.gradient(scalar_objective, inputs)

    if jacobian is None:
        raise ValueError(
            "Failed to compute input-output Jacobian. "
            "Ensure the model is differentiable and that inputs affect outputs."
        )

    batch_size = int(inputs.shape[0])
    jacobian = tf.reshape(jacobian, [batch_size, -1])

    return jacobian.numpy(), batch_size


def eval_score_from_jacobian(
    jacobian: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Compute an SR-oriented Jacobian diversity score.

    The original EPE-NAS uses class-aware correlation statistics. Since SR does
    not naturally provide semantic class labels, this variant instead computes
    pairwise similarity between sample-wise Jacobian signatures across the batch.

    The procedure is:
      1. Mean-center each Jacobian row.
      2. L2-normalize each row.
      3. Compute the pairwise cosine-like similarity matrix J J^T.
      4. Use only the upper triangle (excluding the diagonal).
      5. Convert similarity into a diversity score via -log(|sim| + eps).

    Interpretation:
      - High absolute similarity between samples => low contribution.
      - Low absolute similarity between samples => high contribution.
      - Therefore, higher scores indicate less redundant / more diverse
        Jacobian responses across the synthetic batch.

    Parameters
    ----------
    jacobian : np.ndarray
        Flattened Jacobian matrix with shape [B, D].
    eps : float, default=1e-8
        Numerical stabilizer used inside normalization and logarithms.

    Returns
    -------
    float
        Scalar zero-cost proxy score. Higher values indicate lower average
        pairwise similarity among sample-wise Jacobian signatures.
    """
    if jacobian.ndim != 2:
        raise ValueError(
            f"Expected a 2D Jacobian array [B, D], but got shape {jacobian.shape}."
        )

    batch_size = jacobian.shape[0]
    if batch_size < 2:
        return 0.0

    # Mean-center each sample-wise Jacobian signature.
    jacobian = jacobian - jacobian.mean(axis=1, keepdims=True)

    # L2-normalize rows so pairwise dot products reflect directional similarity
    # rather than raw gradient magnitude.
    norms = np.linalg.norm(jacobian, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    jacobian = jacobian / norms

    # Pairwise similarity matrix over samples.
    # Since rows are centered and normalized, this acts as a cosine-like
    # similarity over Jacobian signatures and is easier to interpret than a
    # second correlation pass via np.corrcoef.
    sim = jacobian @ jacobian.T  # shape [B, B]

    # Numerical safety: clip to valid similarity range.
    sim = np.clip(sim, -1.0, 1.0)
    sim = np.nan_to_num(sim, nan=0.0, posinf=1.0, neginf=-1.0)

    # Keep only unique off-diagonal pairwise similarities.
    upper = sim[np.triu_indices_from(sim, k=1)]

    # Convert similarity to diversity:
    # - if |sim| ~ 1, contribution ~ 0
    # - if |sim| ~ 0, contribution is large
    score = np.sum(-np.log(np.abs(upper) + eps))

    return float(score)


def compute_epe_nas(
    model: tf.keras.Model,
    input_shape: tuple[int, int, int] = (64, 64, 3),
    batch_size: int = 64,
    seed: int | None = 42,
) -> float:
    """
    Compute an SR-oriented zero-cost proxy inspired by EPE-NAS.

    This function follows the modular contract used in the project:
    it builds the model if needed, generates a synthetic input batch,
    computes the batch Jacobian, and returns a single scalar score.

    A random synthetic batch is used instead of a constant tensor. Constant
    inputs can lead to nearly identical Jacobian rows, which weakens the
    discriminative power of pairwise Jacobian diversity metrics.

    Parameters
    ----------
    model : tf.keras.Model
        Candidate architecture.
    input_shape : tuple[int, int, int], default=(64, 64, 3)
        LR input shape expected by the SR model (H, W, C).
    batch_size : int, default=64
        Number of synthetic samples used to estimate Jacobian diversity.
        Must be >= 2. A moderate default is preferred for NAS efficiency.
    seed : int or None, default=42
        Random seed for reproducibility. Set to None to disable deterministic
        synthetic input generation.

    Returns
    -------
    float
        Scalar zero-cost proxy score. Returns np.nan if the score cannot be
        computed (e.g., non-differentiable ops, OOM, degenerate architectures).
    """
    if batch_size < 2:
        raise ValueError("batch_size must be >= 2 for Jacobian diversity scoring.")

    if seed is not None:
        tf.random.set_seed(seed)

    # Random synthetic inputs improve sample diversity and make the Jacobian
    # similarity estimate more informative than a constant batch.
    dummy_inputs = tf.random.normal((batch_size,) + input_shape, dtype=tf.float32)

    # Ensure variables exist before gradient computation.
    if not model.built:
        _ = model(dummy_inputs[:1], training=False)

    try:
        jacobian, _ = get_batch_jacobian(model=model, inputs=dummy_inputs)
        score = eval_score_from_jacobian(jacobian=jacobian)
    except Exception:
        # Defensive behaviour consistent with zero-cost proxy pipelines:
        # failed candidates should not crash the NAS loop.
        score = np.nan

    return float(score)