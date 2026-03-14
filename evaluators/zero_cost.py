from __future__ import annotations

import tensorflow as tf

from evaluators.base import BaseEvaluator
from search_space.search_space import decode
from search_space.model_builder import get_model
from evaluators.metrics.params_score import compute_param_score
from evaluators.metrics.synflow import compute_synflow


class ZeroCostEvaluator(BaseEvaluator):
    def __init__(
        self,
        metric_name: str = "param_score",
        verbose: bool = True,
        input_shape: tuple[int, int, int] = (64, 64, 3),
    ):
        self.metric_name = metric_name
        self.verbose = verbose
        self.input_shape = input_shape

        valid_metrics = {"param_score", "synflow"}
        if self.metric_name not in valid_metrics:
            raise ValueError(
                f"Unknown zero-cost metric '{self.metric_name}'. "
                f"Available options: {sorted(valid_metrics)}"
            )

    def evaluate(self, decoded_ind: list[int], n_eval: int) -> dict:
        genotype = decode(decoded_ind)
        model = get_model(genotype)

        try:
            if self.metric_name == "param_score":
                score = compute_param_score(model)

            elif self.metric_name == "synflow":
                score = compute_synflow(
                    model=model,
                    input_shape=self.input_shape,
                )

            else:
                raise ValueError(f"Unknown zero-cost metric '{self.metric_name}'")

        finally:
            try:
                del model
            except Exception:
                pass
            tf.keras.backend.clear_session()

        return {
            "score": float(score),
            "details": {
                "metric_name": self.metric_name,
                "score_type": "zero_cost",
            },
        }