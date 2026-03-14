from __future__ import annotations

import numpy as np

from evaluators.base import BaseEvaluator
from predictors.loader import load_surrogate_models
from predictors.selectors import select_surrogate_models
from predictors.ensemble import SurrogateEnsemble


class ModelBasedEvaluator(BaseEvaluator):
    def __init__(
        self,
        model_paths: list[str],
        ensemble_method: str = "mean",
        ensemble_weights: list[float] | None = None,
        selected_model_names: list[str] | None = None,
        verbose: bool = True,
    ):
        models, model_names = load_surrogate_models(model_paths, verbose=verbose)

        models, model_names = select_surrogate_models(
            models=models,
            model_names=model_names,
            selected_names=selected_model_names,
        )

        self.backend = SurrogateEnsemble(
            models=models,
            model_names=model_names,
            method=ensemble_method,
            weights=ensemble_weights,
            verbose=verbose,
        )

    def evaluate(self, decoded_ind: list[int], n_eval: int) -> dict:
        x = np.array([decoded_ind], dtype=np.float32)
        pred_info = self.backend.predict_with_stats(x)

        return {
            "score": float(pred_info["ensemble_prediction"]),
            "details": pred_info,
        }