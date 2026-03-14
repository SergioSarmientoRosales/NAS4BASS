from __future__ import annotations

from search.nsga3 import NSGA3
from search.random_search import RandomSearch

from evaluators.model_based import ModelBasedEvaluator
from evaluators.zero_cost import ZeroCostEvaluator


SEARCH_REGISTRY = {
    "nsga3": NSGA3,
    "random": RandomSearch,
}

EVALUATOR_REGISTRY = {
    "model_based": ModelBasedEvaluator,
    "zero_cost": ZeroCostEvaluator,
}


def build_search_method(
    name: str,
    problem,
    pop_size: int,
    n_gen: int,
    verbose: bool = False,
    output_file: str | None = None,
):
    name = name.lower()

    if name not in SEARCH_REGISTRY:
        raise ValueError(
            f"Unknown search method '{name}'. "
            f"Available options: {sorted(SEARCH_REGISTRY.keys())}"
        )

    search_cls = SEARCH_REGISTRY[name]

    return search_cls(
        problem=problem,
        pop_size=pop_size,
        n_gen=n_gen,
        verbose=verbose,
        output_file=output_file,
    )


def build_evaluator(
    eval_name: str,
    model_paths: list[str] | None = None,
    ensemble_method: str = "mean",
    ensemble_weights: list[float] | None = None,
    selected_model_names: list[str] | None = None,
    zc_metric: str = "param_score",
    verbose: bool = True,
):
    eval_name = eval_name.lower()

    if eval_name not in EVALUATOR_REGISTRY:
        raise ValueError(
            f"Unknown evaluator '{eval_name}'. "
            f"Available options: {sorted(EVALUATOR_REGISTRY.keys())}"
        )

    if eval_name == "model_based":
        if model_paths is None or len(model_paths) == 0:
            raise ValueError("model_based evaluator requires model_paths")

        return ModelBasedEvaluator(
            model_paths=model_paths,
            ensemble_method=ensemble_method,
            ensemble_weights=ensemble_weights,
            selected_model_names=selected_model_names,
            verbose=verbose,
        )

    if eval_name == "zero_cost":
        return ZeroCostEvaluator(
            metric_name=zc_metric,
            verbose=verbose,
        )

    raise RuntimeError(f"Unhandled evaluator '{eval_name}'")