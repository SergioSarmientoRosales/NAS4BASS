from __future__ import annotations

import numpy as np


class SurrogateEnsemble:
    def __init__(
        self,
        models: list,
        model_names: list[str],
        method: str = "mean",
        weights: list[float] | None = None,
        verbose: bool = True,
    ):
        if not models:
            raise ValueError("models cannot be empty.")

        self.models = models
        self.model_names = model_names
        self.method = method
        self.weights = weights
        self.verbose = verbose

        self._validate()

    def _validate(self) -> None:
        valid_methods = {"mean", "median", "weighted_mean"}
        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid ensemble method '{self.method}'. Valid options: {sorted(valid_methods)}"
            )

        if self.method == "weighted_mean":
            if self.weights is None:
                raise ValueError("weights must be provided when method='weighted_mean'")
            if len(self.weights) != len(self.models):
                raise ValueError(
                    f"weights length ({len(self.weights)}) must match number of models ({len(self.models)})"
                )

            weights = np.asarray(self.weights, dtype=np.float64)
            if np.any(weights < 0):
                raise ValueError("weights must be non-negative.")
            if np.sum(weights) == 0:
                raise ValueError("At least one weight must be > 0.")

            self.weights = (weights / np.sum(weights)).tolist()

    def predict_all(self, x: np.ndarray) -> np.ndarray:
        preds = []

        for model in self.models:
            pred = model.predict(x)
            pred = np.asarray(pred).reshape(-1)

            if pred.size == 0:
                raise ValueError("A model returned an empty prediction.")

            preds.append(float(pred[0]))

        return np.asarray(preds, dtype=np.float64)

    def aggregate(self, preds: np.ndarray) -> float:
        if self.method == "mean":
            return float(np.mean(preds))
        if self.method == "median":
            return float(np.median(preds))
        if self.method == "weighted_mean":
            return float(np.average(preds, weights=np.asarray(self.weights, dtype=np.float64)))

        raise RuntimeError("Unexpected aggregation method.")

    def predict(self, x: np.ndarray) -> float:
        preds = self.predict_all(x)
        return self.aggregate(preds)

    def predict_with_stats(self, x: np.ndarray) -> dict:
        preds = self.predict_all(x)
        agg = self.aggregate(preds)

        return {
            "ensemble_prediction": float(agg),
            "mean": float(np.mean(preds)),
            "median": float(np.median(preds)),
            "std": float(np.std(preds)),
            "min": float(np.min(preds)),
            "max": float(np.max(preds)),
            "predictions": preds.tolist(),
            "model_names": self.model_names,
            "ensemble_method": self.method,
        }