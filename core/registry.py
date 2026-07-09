from __future__ import annotations

SEARCH_REGISTRY = {
    "nsga3",
    "nsga2",
    "random",
    "mo_random",
    "lhs",
    "bayesopt",
    "successive_halving",
    "hyperband",
    "hill_climbing",
    "simulated_annealing",
    "cmopso",
    "imia",
    "sms_emoa",
}
SEARCH_ALIASES = {
    "sms-emoa": "sms_emoa",
    "latin_hypercube": "lhs",
    "quasi_random": "lhs",
    "bayesian": "bayesopt",
    "bayesian_optimization": "bayesopt",
    "sh": "successive_halving",
    "annealing": "simulated_annealing",
    "mo-random": "mo_random",
}
SEARCH_CLI_CHOICES = tuple(sorted(SEARCH_REGISTRY | set(SEARCH_ALIASES)))
EARLY_STOP_CAPABLE_SEARCHES = {"nsga3"}

EVALUATOR_REGISTRY = {"model_based", "zero_cost"}


def build_search_method(
    name: str,
    problem,
    pop_size: int,
    n_gen: int | None,
    verbose: bool = False,
    output_file: str | None = None,
    save_flush_every: int = 100,
    early_stop: bool = False,
    early_stop_min_gen: int = 200,
    early_stop_repeat_patience: int = 10,
    seed: int | None = None,
    max_evals: int | None = None,
    scalarization_weights: tuple[float, float] | None = None,
    eta: int = 3,
):
    name = name.lower()
    name = SEARCH_ALIASES.get(name, name)

    if name not in SEARCH_REGISTRY:
        raise ValueError(
            f"Unknown search method '{name}'. "
            f"Available options: {sorted(SEARCH_CLI_CHOICES)}"
        )

    if early_stop and name not in EARLY_STOP_CAPABLE_SEARCHES:
        raise ValueError(
            f"--early-stop is currently implemented only for "
            f"{sorted(EARLY_STOP_CAPABLE_SEARCHES)}; got '{name}'."
        )

    if name == "nsga3":
        from search.nsga3 import NSGA3

        search_cls = NSGA3
    elif name == "nsga2":
        from search.nsga2 import NSGA2

        search_cls = NSGA2
    elif name == "random":
        from search.random_search import RandomSearch

        search_cls = RandomSearch
    elif name == "mo_random":
        from search.mo_random import MultiObjectiveRandomSearch

        search_cls = MultiObjectiveRandomSearch
    elif name == "lhs":
        from search.lhs import LatinHypercubeSearch

        search_cls = LatinHypercubeSearch
    elif name == "bayesopt":
        from search.bayesopt import BayesianOptimizationSearch

        search_cls = BayesianOptimizationSearch
    elif name == "successive_halving":
        from search.successive_halving import SuccessiveHalvingSearch

        search_cls = SuccessiveHalvingSearch
    elif name == "hyperband":
        from search.hyperband import HyperbandSearch

        search_cls = HyperbandSearch
    elif name == "hill_climbing":
        from search.hill_climbing import HillClimbingSearch

        search_cls = HillClimbingSearch
    elif name == "simulated_annealing":
        from search.simulated_annealing import SimulatedAnnealingSearch

        search_cls = SimulatedAnnealingSearch
    elif name == "cmopso":
        from search.cmopso import CMOPSO

        search_cls = CMOPSO
    elif name == "imia":
        from search.imia import IMIA

        search_cls = IMIA
    elif name == "sms_emoa":
        from search.sms_emoa import SMSEMOA

        search_cls = SMSEMOA
    else:
        raise RuntimeError(f"Unhandled search method '{name}'")

    kwargs = {
        "problem": problem,
        "pop_size": pop_size,
        "n_gen": n_gen,
        "verbose": verbose,
        "output_file": output_file,
    }

    budgeted_names = {
        "nsga2",
        "random",
        "mo_random",
        "lhs",
        "bayesopt",
        "successive_halving",
        "hyperband",
        "hill_climbing",
        "simulated_annealing",
    }
    if name in budgeted_names:
        kwargs.update({"seed": seed, "max_evals": max_evals})

    if name in {"nsga3", "random"}:
        kwargs.update(
            {
                "save_flush_every": save_flush_every,
                "early_stop": early_stop,
                "early_stop_min_gen": early_stop_min_gen,
                "early_stop_repeat_patience": early_stop_repeat_patience,
            }
        )

    if name in {
        "bayesopt",
        "successive_halving",
        "hyperband",
        "hill_climbing",
        "simulated_annealing",
    }:
        kwargs["scalarization_weights"] = scalarization_weights

    if name in {"successive_halving", "hyperband"}:
        kwargs["eta"] = eta

    return search_cls(**kwargs)


def build_evaluator(
    eval_name: str,
    model_paths: list[str] | None = None,
    ensemble_method: str = "mean",
    ensemble_weights: list[float] | None = None,
    selected_model_names: list[str] | None = None,
    zc_metric: str = "synflow",
    zc_score_transform: str = "raw",
    seed: int = 1,
    deterministic_arch_seed: bool = True,
    verbose: bool = True,
):
    eval_name = eval_name.lower()

    if eval_name not in EVALUATOR_REGISTRY:
        raise ValueError(
            f"Unknown evaluator '{eval_name}'. "
            f"Available options: {sorted(EVALUATOR_REGISTRY)}"
        )

    if eval_name == "model_based":
        if model_paths is None or len(model_paths) == 0:
            raise ValueError("model_based evaluator requires model_paths")

        from evaluators.model_based import ModelBasedEvaluator

        return ModelBasedEvaluator(
            model_paths=model_paths,
            ensemble_method=ensemble_method,
            ensemble_weights=ensemble_weights,
            selected_model_names=selected_model_names,
            verbose=verbose,
        )

    if eval_name == "zero_cost":
        from evaluators.zero_cost import ZeroCostEvaluator

        return ZeroCostEvaluator(
            metric_name=zc_metric,
            score_transform=zc_score_transform,
            seed=seed,
            deterministic_arch_seed=deterministic_arch_seed,
            verbose=verbose,
        )

    raise RuntimeError(f"Unhandled evaluator '{eval_name}'")
