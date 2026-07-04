from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TRANSFORMS = ("raw", "div_params", "neg_raw", "neg_div_params")
ID_COLUMNS = {"sample_id", "Net", "gene", "architecture", "arch"}
NON_SCORE_COLUMNS = {
    "seed",
    "predictor",
    "metric",
    "zc_metric",
    "score_transform",
    "params",
    "params_real",
    "valid_psnr",
    "psnr",
    "target",
    "complexity_bin",
}


def read_csv_rows(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return math.nan
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom == 0:
        return math.nan
    return float(np.sum(x * y) / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return math.nan
    return pearson(rankdata(x), rankdata(y))


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 2:
        return math.nan

    concordant = discordant = ties_x = ties_y = 0
    for i in range(n - 1):
        dx = x[i] - x[i + 1:]
        dy = y[i] - y[i + 1:]
        prod = dx * dy
        concordant += int(np.sum(prod > 0))
        discordant += int(np.sum(prod < 0))
        ties_x += int(np.sum((dx == 0) & (dy != 0)))
        ties_y += int(np.sum((dx != 0) & (dy == 0)))

    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0:
        return math.nan
    return float((concordant - discordant) / denom)


def c_index(score: np.ndarray, target: np.ndarray) -> float:
    mask = np.isfinite(score) & np.isfinite(target)
    score = score[mask]
    target = target[mask]
    n = len(score)
    if n < 2:
        return math.nan

    concordance = 0.0
    comparable = 0
    for i in range(n - 1):
        dy = target[i] - target[i + 1:]
        ds = score[i] - score[i + 1:]
        non_tied_target = dy != 0
        comparable += int(np.sum(non_tied_target))
        concordance += float(np.sum((dy * ds > 0) & non_tied_target))
        concordance += 0.5 * float(np.sum((ds == 0) & non_tied_target))
    if comparable == 0:
        return math.nan
    return float(concordance / comparable)


def partial_spearman(score: np.ndarray, target: np.ndarray, log_params: np.ndarray) -> tuple[float, float]:
    rho_sy = spearman(score, target)
    rho_sp = spearman(score, log_params)
    rho_yp = spearman(target, log_params)
    denom = math.sqrt(max(0.0, (1.0 - rho_sp**2) * (1.0 - rho_yp**2)))
    if denom == 0 or not all(math.isfinite(v) for v in (rho_sy, rho_sp, rho_yp)):
        return math.nan, rho_sp
    return float((rho_sy - rho_sp * rho_yp) / denom), float(rho_sp)


def metric_bundle(score: np.ndarray, target: np.ndarray, params: np.ndarray) -> dict[str, float]:
    log_params = np.log(params)
    partial, score_params = partial_spearman(score, target, log_params)
    return {
        "c_index": c_index(score, target),
        "spearman": spearman(score, target),
        "kendall_tau_b": kendall_tau_b(score, target),
        "partial_spearman_log_params": partial,
        "spearman_score_params": score_params,
    }


def percentile_bootstrap(
    score: np.ndarray,
    target: np.ndarray,
    params: np.ndarray,
    *,
    metric_name: str,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    mask = np.isfinite(score) & np.isfinite(target) & np.isfinite(params) & (params > 0)
    score = score[mask]
    target = target[mask]
    params = params[mask]
    if len(score) < 3 or n_resamples <= 0:
        return math.nan, math.nan

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(score), size=len(score))
        metric_value = metric_bundle(score[idx], target[idx], params[idx]).get(metric_name, math.nan)
        if math.isfinite(metric_value):
            values.append(metric_value)
    if not values:
        return math.nan, math.nan
    return tuple(float(v) for v in np.percentile(values, [2.5, 97.5]))


def apply_transform(raw_score: float, params: float, transform: str) -> float:
    if not math.isfinite(raw_score) or not math.isfinite(params) or params <= 0:
        return math.nan
    if transform == "raw":
        return raw_score
    if transform == "div_params":
        return raw_score / params
    if transform == "neg_raw":
        return -raw_score
    if transform == "neg_div_params":
        return -raw_score / params
    raise ValueError(f"Unknown transform: {transform}")


def sample_key(row: dict) -> str:
    for key in ("sample_id", "Net", "gene", "architecture", "arch"):
        if row.get(key):
            return str(row[key])
    raise ValueError("Row lacks a sample identifier column")


def load_sample(path: str | Path, *, target_column: str, params_column: str) -> dict[str, dict]:
    samples: dict[str, dict] = {}
    for row in read_csv_rows(path):
        key = sample_key(row)
        target = to_float(row.get(target_column))
        params = to_float(row.get(params_column))
        if not math.isfinite(target):
            raise ValueError(f"Sample {key} has missing target column {target_column}")
        if not math.isfinite(params) or params <= 0:
            raise ValueError(f"Sample {key} has missing positive params column {params_column}")
        samples[key] = {
            "target": target,
            "params": params,
            "complexity_bin": int(float(row.get("complexity_bin", 0))),
        }
    return samples


def numeric_score_columns(row: dict) -> list[str]:
    out = []
    for key, value in row.items():
        if key in ID_COLUMNS or key in NON_SCORE_COLUMNS:
            continue
        if math.isfinite(to_float(value)):
            out.append(key)
    return out


def load_score_files(paths: list[str]) -> dict[tuple[str, str], dict[str, float]]:
    scores: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for path in paths:
        rows = read_csv_rows(path)
        file_seed = Path(path).stem
        for row in rows:
            key = sample_key(row)
            seed = str(row.get("seed") or file_seed)
            predictor = row.get("predictor") or row.get("metric") or row.get("zc_metric")
            if predictor:
                score_value = to_float(row.get("raw_score", row.get("score")))
                if math.isfinite(score_value):
                    scores[(str(predictor), seed)][key] = score_value
                continue

            for column in numeric_score_columns(row):
                scores[(column, seed)][key] = to_float(row[column])
    return scores


def iqr(values: list[float]) -> tuple[float, float, float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if len(finite) == 0:
        return math.nan, math.nan, math.nan
    q1, med, q3 = np.percentile(finite, [25, 50, 75])
    return float(med), float(q1), float(q3)


def aggregate_scores_by_architecture(seed_scores: list[dict[str, float]], keys: list[str]) -> np.ndarray:
    matrix = []
    for scores in seed_scores:
        matrix.append([scores.get(key, math.nan) for key in keys])
    return np.nanmedian(np.asarray(matrix, dtype=np.float64), axis=0)


def summarize_level(
    *,
    keys: list[str],
    samples: dict[str, dict],
    predictor: str,
    transform: str,
    seed_to_scores: dict[str, dict[str, float]],
    bootstrap_resamples: int,
    bootstrap_seed: int,
    level: str,
    band: int | str = "all",
) -> dict:
    target = np.asarray([samples[key]["target"] for key in keys], dtype=np.float64)
    params = np.asarray([samples[key]["params"] for key in keys], dtype=np.float64)

    seed_metric_values: dict[str, list[float]] = defaultdict(list)
    transformed_seed_scores = []
    for seed, raw_scores in sorted(seed_to_scores.items()):
        score = np.asarray(
            [
                apply_transform(raw_scores.get(key, math.nan), samples[key]["params"], transform)
                for key in keys
            ],
            dtype=np.float64,
        )
        transformed_seed_scores.append({key: value for key, value in zip(keys, score)})
        metrics = metric_bundle(score, target, params)
        for metric_name, metric_value in metrics.items():
            seed_metric_values[metric_name].append(metric_value)

    aggregate_score = aggregate_scores_by_architecture(transformed_seed_scores, keys)
    aggregate_metrics = metric_bundle(aggregate_score, target, params)

    row = {
        "level": level,
        "band": band,
        "predictor": predictor,
        "transform": transform,
        "n_architectures": len(keys),
        "n_seeds": len(seed_to_scores),
    }
    for metric_name in (
        "c_index",
        "spearman",
        "kendall_tau_b",
        "partial_spearman_log_params",
        "spearman_score_params",
    ):
        median, q1, q3 = iqr(seed_metric_values[metric_name])
        low, high = percentile_bootstrap(
            aggregate_score,
            target,
            params,
            metric_name=metric_name,
            n_resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
        row[f"{metric_name}_aggregate"] = aggregate_metrics[metric_name]
        row[f"{metric_name}_seed_median"] = median
        row[f"{metric_name}_seed_q1"] = q1
        row[f"{metric_name}_seed_q3"] = q3
        row[f"{metric_name}_boot_low"] = low
        row[f"{metric_name}_boot_high"] = high
    return row


def analyze(
    *,
    sample_csv: str,
    score_csvs: list[str],
    target_column: str,
    params_column: str,
    bootstrap_resamples: int,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    samples = load_sample(sample_csv, target_column=target_column, params_column=params_column)
    raw_scores = load_score_files(score_csvs)

    predictor_to_seed_scores: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (predictor, score_seed), scores in raw_scores.items():
        predictor_to_seed_scores[predictor][score_seed] = scores

    keys = sorted(samples)
    global_rows = []
    band_rows = []
    for predictor, seed_to_scores in sorted(predictor_to_seed_scores.items()):
        for transform in TRANSFORMS:
            global_rows.append(
                summarize_level(
                    keys=keys,
                    samples=samples,
                    predictor=predictor,
                    transform=transform,
                    seed_to_scores=seed_to_scores,
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_seed=seed,
                    level="global",
                )
            )

            bands = sorted({samples[key]["complexity_bin"] for key in keys})
            for band in bands:
                band_keys = [key for key in keys if samples[key]["complexity_bin"] == band]
                band_rows.append(
                    summarize_level(
                        keys=band_keys,
                        samples=samples,
                        predictor=predictor,
                        transform=transform,
                        seed_to_scores=seed_to_scores,
                        bootstrap_resamples=bootstrap_resamples,
                        bootstrap_seed=seed + int(band) + 1,
                        level="band_descriptive",
                        band=band,
                    )
                )

    summary = {
        "sample_csv": str(sample_csv),
        "score_csvs": list(score_csvs),
        "target_column": target_column,
        "params_column": params_column,
        "n_architectures": len(samples),
        "predictors": sorted(predictor_to_seed_scores),
        "transforms": list(TRANSFORMS),
        "bootstrap_resamples": bootstrap_resamples,
        "seed": seed,
        "band_results_note": "Band-level statistics are descriptive because each band has low n.",
    }
    return global_rows, band_rows, summary


def make_figures(global_rows: list[dict], band_rows: list[dict], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = [row for row in global_rows if row["transform"] == "raw"]
    if raw_rows:
        labels = [row["predictor"] for row in raw_rows]
        y = [float(row["spearman_aggregate"]) for row in raw_rows]
        low = [float(row["spearman_boot_low"]) for row in raw_rows]
        high = [float(row["spearman_boot_high"]) for row in raw_rows]
        yerr = np.asarray([[a - b for a, b in zip(y, low)], [b - a for a, b in zip(y, high)]])
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 4))
        ax.bar(labels, y, color="#2563eb", alpha=0.78)
        ax.errorbar(labels, y, yerr=yerr, fmt="none", ecolor="#111827", capsize=3)
        ax.set_ylabel("Spearman rho")
        ax.set_title("Global raw zero-cost correlation with 95% bootstrap CI")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, axis="y", color="#e5e7eb")
        fig.tight_layout()
        fig.savefig(output_dir / "global_raw_spearman_ci.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(
            [float(row["spearman_aggregate"]) for row in raw_rows],
            [float(row["partial_spearman_log_params_aggregate"]) for row in raw_rows],
            c="#2563eb",
        )
        for row in raw_rows:
            ax.annotate(row["predictor"], (float(row["spearman_aggregate"]), float(row["partial_spearman_log_params_aggregate"])))
        ax.set_xlabel("Crude Spearman")
        ax.set_ylabel("Partial Spearman controlling log(params_real)")
        ax.grid(True, color="#e5e7eb")
        fig.tight_layout()
        fig.savefig(output_dir / "partial_vs_crude_spearman.png", dpi=300)
        plt.close(fig)

    if band_rows:
        raw_band_rows = [row for row in band_rows if row["transform"] == "raw"]
        fig, ax = plt.subplots(figsize=(7, 4))
        for predictor in sorted({row["predictor"] for row in raw_band_rows}):
            rows = [row for row in raw_band_rows if row["predictor"] == predictor]
            rows.sort(key=lambda item: int(item["band"]))
            ax.plot(
                [int(row["band"]) for row in rows],
                [float(row["spearman_aggregate"]) for row in rows],
                marker="o",
                label=predictor,
            )
        ax.set_xlabel("Complexity band")
        ax.set_ylabel("Spearman rho")
        ax.set_title("Band-level raw correlations (descriptive)")
        ax.grid(True, color="#e5e7eb")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "band_raw_spearman.png", dpi=300)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze zero-cost scores on the 50-architecture BASS sample.")
    parser.add_argument("--sample-csv", required=True)
    parser.add_argument("--score-csv", action="append", required=True)
    parser.add_argument("--target-column", default="valid_psnr")
    parser.add_argument("--params-column", default="params_real")
    parser.add_argument("--output-dir", default="results/zerocost_50_stratified_random")
    parser.add_argument("--figure-dir", default="figures/zerocost_50_stratified_random")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global_rows, band_rows, summary = analyze(
        sample_csv=args.sample_csv,
        score_csvs=args.score_csv,
        target_column=args.target_column,
        params_column=args.params_column,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(global_rows[0]) if global_rows else []
    write_csv_rows(output_dir / "global_correlations.csv", global_rows, fieldnames)
    write_csv_rows(output_dir / "band_correlations.csv", band_rows, fieldnames)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if not args.no_figures:
        make_figures(global_rows, band_rows, Path(args.figure_dir))

    print(f"[ANALYZE] global rows={len(global_rows)}")
    print(f"[ANALYZE] band rows={len(band_rows)}")
    print(f"[ANALYZE] output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
