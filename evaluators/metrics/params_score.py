from __future__ import annotations

import tensorflow as tf


def compute_param_score(model: tf.keras.Model) -> float:
    """
    Simple zero-cost placeholder metric.

    Returns the negative number of parameters so that larger is better
    from the evaluator perspective only if you want smaller models favored.

    For now we return the negative count to make it a meaningful proxy:
    fewer params -> higher score.
    """
    return -float(model.count_params())