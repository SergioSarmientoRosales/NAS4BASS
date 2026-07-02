from __future__ import annotations

import numpy as np


class SurrogateEnsemble:
    def __init__(
        self,
        models: list,
        model_names: list[str],
        method: str = "mean",
        weights: list[float] | dict[str, float] | None = None,
        verbose: bool = True,
    ):
        if not models:
            raise ValueError("models cannot be empty.")
        if len(model_names) != len(models):
            raise ValueError(
                f"model_names length ({len(model_names)}) must match "
                f"number of models ({len(models)})"
            )

        self.models = models
        self.model_names = model_names
        self.method = method
        self.weights = weights
        self.weight_map: dict[str, float] | None = None
        self.verbose = verbose

        self._validate()

    def _weights_to_array(self) -> np.ndarray:
        if isinstance(self.weights, dict):
            expected = set(self.model_names)
            provided = set(self.weights)
            missing = sorted(expected - provided)
            extra = sorted(provided - expected)

            if missing or extra:
                raise ValueError(
                    "Named ensemble weights must match the active model names. "
                    f"Missing: {missing}; extra: {extra}; "
                    f"active models: {self.model_names}"
                )

            return np.asarray(
                [self.weights[name] for name in self.model_names],
                dtype=np.float64,
            )

        return np.asarray(self.weights, dtype=np.float64)

    def _validate(self) -> None:
        valid_methods = {"mean", "median", "weighted_mean"}
        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid ensemble method '{self.method}'. Valid options: {sorted(valid_methods)}"
            )

        if self.method == "weighted_mean":
            if self.weights is None:
                raise ValueError("weights must be provided when method='weighted_mean'")

            weights = self._weights_to_array()
            if len(weights) != len(self.models):
                raise ValueError(
                    f"weights length ({len(weights)}) must match number of models ({len(self.models)})"
                )

            if np.any(weights < 0):
                raise ValueError("weights must be non-negative.")
            if np.sum(weights) == 0:
                raise ValueError("At least one weight must be > 0.")

            normalized = (weights / np.sum(weights)).tolist()
            self.weights = normalized
            self.weight_map = dict(zip(self.model_names, normalized))

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
            "ensemble_weights": self.weight_map,
        }
