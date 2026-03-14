from __future__ import annotations

import os
import joblib


def load_surrogate_models(model_paths: list[str], verbose: bool = True):
    models = []
    model_names = []

    for path in model_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        try:
            model = joblib.load(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{path}': {e}") from e

        models.append(model)
        model_names.append(os.path.splitext(os.path.basename(path))[0])

        if verbose:
            print(f"[INFO] Loaded model: {path}")

    return models, model_names