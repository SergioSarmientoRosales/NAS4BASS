from __future__ import annotations
import csv
import os
import glob
import time
import argparse
from datetime import datetime

from config import DEFAULT_POP_SIZE, DEFAULT_N_GEN, DEFAULT_ENSEMBLE_METHOD
from utils.seed import set_global_seed

from core.problem import NASProblem
from core.registry import build_search_method, build_evaluator


def parse_float_list(text: str | None) -> list[float] | None:
    if text is None:
        return None

    text = text.strip()
    if text == "":
        return None

    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: str | None) -> list[str] | None:
    if text is None:
        return None

    text = text.strip()
    if text == "":
        return None

    return [x.strip() for x in text.split(",") if x.strip()]


def discover_models(models_dir: str = "models") -> list[str]:
    model_paths = sorted(glob.glob(os.path.join(models_dir, "*.pkl")))

    if not model_paths:
        raise FileNotFoundError(
            f"No .pkl files were found inside '{models_dir}'. "
            f"Please place your surrogate models there."
        )

    return model_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run NSGA-III locally with modular model-based evaluation"
    )

    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory with surrogate .pkl models")
    parser.add_argument(
        "--model-paths",
        type=str,
        default=None,
        help="Comma-separated explicit model paths. If omitted, all models/*.pkl are loaded.",
    )
    parser.add_argument(
        "--selected-models",
        type=str,
        default=None,
        help="Comma-separated surrogate model names to use, e.g. 'xgb,rf,gbrt'",
    )

    parser.add_argument(
        "--ensemble-method",
        type=str,
        default=DEFAULT_ENSEMBLE_METHOD,
        choices=["mean", "median", "weighted_mean"],
        help="Ensemble aggregation strategy",
    )
    parser.add_argument(
        "--ensemble-weights",
        type=str,
        default=None,
        help="Comma-separated weights for weighted_mean",
    )

    parser.add_argument(
        "--search",
        type=str,
        default="nsga3",
        choices=["nsga3", "random"],
        help="Search method",
    )

    parser.add_argument(
        "--eval",
        type=str,
        default="model_based",
        choices=["model_based", "zero_cost"],
        help="Evaluation backend",
    )

    parser.add_argument(
        "--zc-metric",
        type=str,
        default="param_score",
        help="Zero-cost metric name",
    )

    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--pop-size", type=int, default=DEFAULT_POP_SIZE, help="Population size")
    parser.add_argument("--n-gen", type=int, default=DEFAULT_N_GEN, help="Number of generations")

    parser.add_argument("--verbose", action="store_true", help="Enable NSGA-III progress bar")
    parser.add_argument("--verbose-cache", action="store_true", help="Print cache hits")

    parser.add_argument("--disable-decode-cache", action="store_true", help="Disable decode cache")
    parser.add_argument("--disable-score-cache", action="store_true", help="Disable score cache")
    parser.add_argument("--disable-param-cache", action="store_true", help="Disable parameter cache")
    parser.add_argument("--disable-obj-cache", action="store_true", help="Disable objective cache")

    return parser.parse_args()


def resolve_model_paths(args) -> list[str]:
    if args.model_paths is not None and args.model_paths.strip() != "":
        model_paths = [x.strip() for x in args.model_paths.split(",") if x.strip()]
        if not model_paths:
            raise ValueError("No valid model paths were parsed from --model-paths")
        return model_paths

    return discover_models(args.models_dir)


def save_non_dominated_solutions(nds: dict, problem, filepath: str) -> None:
    import csv

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = [
            "individual_id",
            "primary_score",
            "params",
        ]
        header.extend([f"gene_{i}" for i in range(28)])
        writer.writerow(header)

        for idx, (x, obj) in enumerate(zip(nds["X"], nds["F"])):
            decoded_arch = problem.get_decoded_ind(x)

            row = [
                idx,
                -float(obj[0]),   # primary score
                int(obj[1]),
            ]
            row.extend(decoded_arch)

            writer.writerow(row)

def print_run_configuration(args, model_paths: list[str], selected_models: list[str] | None, seed_folder: str, output_file: str) -> None:
    print("=" * 80)
    print("[INFO] Modular NAS local run configuration")
    print("=" * 80)
    print(f"[INFO] Seed: {args.seed}")
    print(f"[INFO] Output folder: {seed_folder}")
    print(f"[INFO] Output file: {output_file}")
    print(f"[INFO] Population size: {args.pop_size}")
    print(f"[INFO] Generations: {args.n_gen}")
    print(f"[INFO] Ensemble method: {args.ensemble_method}")
    print(f"[INFO] Verbose progress: {args.verbose}")
    print(f"[INFO] Verbose cache: {args.verbose_cache}")
    print(f"[INFO] Decode cache enabled: {not args.disable_decode_cache}")
    print(f"[INFO] Score cache enabled: {not args.disable_score_cache}")
    print(f"[INFO] Parameter cache enabled: {not args.disable_param_cache}")
    print(f"[INFO] Objective cache enabled: {not args.disable_obj_cache}")
    print(f"[INFO] Selected models: {selected_models if selected_models is not None else 'ALL'}")
    print("[INFO] Model paths:")
    print(f"[INFO] Evaluation method: {args.eval}")
    if args.eval == "zero_cost":
        print(f"[INFO] Zero-cost metric: {args.zc_metric}")
    print(f"[INFO] Search method: {args.search}")
    for p in model_paths:
        print(f"  - {p}")

    if args.ensemble_method == "weighted_mean":
        print(f"[INFO] Ensemble weights: {args.ensemble_weights}")

    print("=" * 80)


def save_run_summary(summary: dict, filepath: str) -> None:
    file_exists = os.path.exists(filepath)

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(summary)


def main():
    start = time.time()
    args = parse_args()

    set_global_seed(args.seed)

    if args.eval == "model_based":
        model_paths = resolve_model_paths(args)
    else:
        model_paths = []

    selected_models = parse_str_list(args.selected_models)
    ensemble_weights = parse_float_list(args.ensemble_weights)

    if args.ensemble_method == "weighted_mean" and ensemble_weights is None:
        raise ValueError(
            "You selected --ensemble-method weighted_mean but did not provide "
            "--ensemble-weights"
        )

    if args.ensemble_method != "weighted_mean" and ensemble_weights is not None:
        print("[WARNING] Ensemble weights were provided but will be ignored because "
              "ensemble method is not 'weighted_mean'.")

    os.makedirs(args.outdir, exist_ok=True)
    seed_folder = os.path.join(args.outdir, f"seed_{args.seed}")
    os.makedirs(seed_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        seed_folder,
        f"ensemble_{args.ensemble_method}_population_seed_{args.seed}_{timestamp}.csv"
    )

    cache_summary_file = os.path.join(
        seed_folder,
        f"cache_summary_seed_{args.seed}_{timestamp}.csv"
    )

    print_run_configuration(args, model_paths, selected_models, seed_folder, output_file)

    # 1) Build evaluator
    evaluator = build_evaluator(
        eval_name=args.eval,
        model_paths=model_paths if args.eval == "model_based" else None,
        ensemble_method=args.ensemble_method,
        ensemble_weights=ensemble_weights,
        selected_model_names=selected_models,
        zc_metric=args.zc_metric,
        verbose=True,
    )

    # 2) Build problem
    problem = NASProblem(
        evaluator=evaluator,
        n_var=84,
        n_obj=2,
        use_decode_cache=not args.disable_decode_cache,
        use_score_cache=not args.disable_score_cache,
        use_param_cache=not args.disable_param_cache,
        use_obj_cache=not args.disable_obj_cache,
        verbose_cache=args.verbose_cache,
    )

    search_method = build_search_method(
        name=args.search,
        problem=problem,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        verbose=args.verbose,
        output_file=output_file,
    )

    final_population, non_dominated_solutions = search_method()

    print("\n[INFO] Optimization finished.")
    print(f"[INFO] Final population size: {len(final_population['X'])}")
    print(f"[INFO] Non-dominated solutions: {len(non_dominated_solutions['X'])}")

    end = time.time()
    runtime_minutes = (end - start) / 60.0

    cache_summary = problem.get_cache_summary()

    print("\n[INFO] Cache summary:")
    for k, v in cache_summary.items():
        print(f"  {k}: {v}")

    run_summary = {
        "timestamp": timestamp,
        "seed": args.seed,
        "search_method": args.search,
        "evaluation_method": args.eval,
        "zc_metric": args.zc_metric if args.eval == "zero_cost" else "",
        "ensemble_method": args.ensemble_method if args.eval == "model_based" else "",
        "selected_models": ",".join(selected_models) if selected_models is not None else "ALL",
        "pop_size": args.pop_size,
        "n_gen": args.n_gen,
        "final_population_size": len(final_population["X"]),
        "non_dominated_count": len(non_dominated_solutions["X"]),
        "runtime_minutes": runtime_minutes,
        **cache_summary,
    }

    save_run_summary(run_summary, cache_summary_file)
    print(f"\n[INFO] Run summary saved to: {cache_summary_file}")
    print(f"[INFO] Total runtime: {runtime_minutes:.2f} minutes")
    print(f"\n[INFO] Cache summary saved to: {cache_summary_file}")

    end = time.time()
    print(f"[INFO] Total runtime: {(end - start) / 60:.2f} minutes")

    nds_file = os.path.join(
        seed_folder,
        f"non_dominated_seed_{args.seed}_{timestamp}.csv"
    )

    save_non_dominated_solutions(
        nds=non_dominated_solutions,
        problem=problem,
        filepath=nds_file,
    )

    print(f"[INFO] Non-dominated solutions saved to: {nds_file}")

if __name__ == "__main__":
    main()