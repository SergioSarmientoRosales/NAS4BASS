from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, friedmanchisquare, kendalltau, rankdata, spearmanr, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "results" / "rework_50_architectures"
TRANSFORMS = ("raw", "div_params", "neg_raw", "neg_div_params")
TRANSFORM_ORDER = {name: index for index, name in enumerate(TRANSFORMS)}
DISPLAY_NAMES = {
    "l2_norm": "L2Norm",
    "param_score": "ParamScore",
    "nwot": "NWOT",
    "zen": "ZEN",
    "zico": "ZiCo",
    "jacob": "Jacov",
    "synflow": "SynFlow",
    "grad_norm": "GradNorm",
    "plain": "Plain",
    "snip": "SNIP",
    "fisher": "Fisher",
    "grasp": "GraSP",
}
TRANSFORM_NAMES = {
    "raw": "Original",
    "div_params": "Parameter-adjusted",
    "neg_raw": "Sign-inverted",
    "neg_div_params": "Sign-inverted parameter-adjusted",
}
APPENDIX_TRANSFORM_NAMES = {
    "raw": "Orig.",
    "div_params": "Param.-adj.",
    "neg_raw": "Inv.",
    "neg_div_params": "Inv.+param.-adj.",
}


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"scenario", "architecture_id", "valid_psnr", "params_real"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest {path} is missing columns: {sorted(missing)}")
    if frame["architecture_id"].duplicated().any():
        raise ValueError(f"Manifest {path} contains duplicate architecture IDs")
    return frame


def load_raw(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    raw = pd.concat(frames, ignore_index=True)
    required = {
        "scenario",
        "architecture_id",
        "seed",
        "proxy",
        "valid_psnr",
        "params_expected",
        "params_built",
        "raw_score",
        "validity_flag",
        "status",
        "proxy_time_ms",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Raw score files are missing columns: {sorted(missing)}")

    raw["completed_at_utc"] = raw.get("completed_at_utc", "")
    key_columns = ["scenario", "architecture_id", "seed", "proxy"]
    deduplicated_records = int(raw.duplicated(key_columns, keep="last").sum())
    raw = raw.sort_values("completed_at_utc").drop_duplicates(
        key_columns, keep="last"
    )
    raw.attrs["deduplicated_records"] = deduplicated_records
    raw["validity_flag"] = _bool_series(raw["validity_flag"])
    for column in (
        "seed",
        "valid_psnr",
        "params_built",
        "raw_score",
        "build_time_ms",
        "proxy_time_ms",
        "total_time_ms",
    ):
        if column in raw:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")
    return raw


def validate_raw(raw: pd.DataFrame, manifests: dict[str, pd.DataFrame]) -> dict[str, object]:
    checks: dict[str, object] = {
        "deduplicated_resume_records": int(raw.attrs.get("deduplicated_records", 0)),
        "scenarios": {},
    }
    for scenario, manifest in manifests.items():
        chunk = raw[raw["scenario"] == scenario]
        observed_ids = set(chunk["architecture_id"])
        expected_ids = set(manifest["architecture_id"])
        protocol_columns = [
            column
            for column in ("input_seed", "lr_patch_size", "batch_size", "upscale_factor")
            if column in chunk
        ]
        protocol_values = {
            column: sorted(chunk[column].dropna().astype(str).unique().tolist())
            for column in protocol_columns
        }
        checks["scenarios"][scenario] = {
            "expected_architectures": len(expected_ids),
            "observed_architectures": len(observed_ids),
            "missing_architectures": sorted(expected_ids - observed_ids),
            "unexpected_architectures": sorted(observed_ids - expected_ids),
            "seeds": sorted(int(seed) for seed in chunk["seed"].dropna().unique()),
            "proxies": sorted(chunk["proxy"].dropna().unique().tolist()),
            "rows": len(chunk),
            "valid_rows": int((chunk["validity_flag"] & chunk["raw_score"].notna()).sum()),
            "error_rows": int((chunk["status"] == "error").sum()),
            "invalid_rows": int((chunk["status"] == "invalid").sum()),
            "parameter_mismatch_rows": int(
                (
                    pd.to_numeric(chunk["params_expected"], errors="coerce")
                    != pd.to_numeric(chunk["params_built"], errors="coerce")
                ).sum()
            ),
            "nonfinite_raw_score_rows": int(
                (~np.isfinite(pd.to_numeric(chunk["raw_score"], errors="coerce"))).sum()
            ),
            "protocol_values": protocol_values,
        }
    return checks


def apply_transforms(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for transform in TRANSFORMS:
        chunk = raw.copy()
        params = chunk["params_built"].astype(float)
        score = chunk["raw_score"].astype(float)
        if transform == "raw":
            transformed = score
        elif transform == "div_params":
            transformed = score / params
        elif transform == "neg_raw":
            transformed = -score
        else:
            transformed = -score / params
        chunk["transformation"] = transform
        chunk["transformed_score"] = transformed.where(params > 0)
        chunk["transformed_valid"] = (
            chunk["validity_flag"]
            & np.isfinite(chunk["transformed_score"])
            & np.isfinite(params)
            & (params > 0)
        )
        rows.append(chunk)
    return pd.concat(rows, ignore_index=True)


def concordance_index(scores: np.ndarray, targets: np.ndarray) -> tuple[float, int]:
    concordant = 0.0
    comparable = 0
    for index in range(len(scores) - 1):
        target_delta = targets[index] - targets[index + 1 :]
        score_delta = scores[index] - scores[index + 1 :]
        mask = target_delta != 0
        comparable += int(mask.sum())
        concordant += float(((target_delta * score_delta > 0) & mask).sum())
        concordant += 0.5 * float(((score_delta == 0) & mask).sum())
    return (concordant / comparable, comparable) if comparable else (math.nan, 0)


def partial_spearman(scores: np.ndarray, targets: np.ndarray, params: np.ndarray) -> float:
    ranked_scores = rankdata(scores, method="average")
    ranked_targets = rankdata(targets, method="average")
    covariate = rankdata(np.log(params), method="average")
    design = np.column_stack((np.ones(len(covariate)), covariate))
    score_residual = ranked_scores - design @ np.linalg.lstsq(design, ranked_scores, rcond=None)[0]
    target_residual = ranked_targets - design @ np.linalg.lstsq(design, ranked_targets, rcond=None)[0]
    if np.std(score_residual) <= 1e-12 or np.std(target_residual) <= 1e-12:
        return math.nan
    return float(np.corrcoef(score_residual, target_residual)[0, 1])


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        return float(spearmanr(left, right).statistic)


def top_k_metrics(
    ids: np.ndarray, scores: np.ndarray, targets: np.ndarray, k: int
) -> dict[str, float]:
    requested_k = k
    effective_k = min(k, len(scores))
    selected = np.argsort(-scores, kind="mergesort")[:effective_k]
    truth = np.argsort(-targets, kind="mergesort")[:effective_k]
    selected_ids = set(ids[selected])
    truth_ids = set(ids[truth])
    target_ranks = rankdata(-targets, method="average")
    return {
        f"top{requested_k}_overlap": len(selected_ids & truth_ids) / effective_k,
        f"top{requested_k}_regret": float(np.max(targets) - np.max(targets[selected])),
        f"top{requested_k}_mean_true_rank": float(np.mean(target_ranks[selected])),
    }


def calculate_seed_metrics(
    transformed: pd.DataFrame, manifests: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = transformed.groupby(["scenario", "proxy", "transformation", "seed"], sort=True)
    for (scenario, proxy, transformation, seed), chunk in grouped:
        expected_ids = set(manifests[str(scenario)]["architecture_id"])
        valid = chunk[
            chunk["transformed_valid"]
            & chunk["architecture_id"].isin(expected_ids)
        ].drop_duplicates("architecture_id", keep="last")
        valid = valid.sort_values("architecture_id")
        complete = set(valid["architecture_id"]) == expected_ids
        row: dict[str, object] = {
            "scenario": scenario,
            "proxy": proxy,
            "transformation": transformation,
            "seed": int(seed),
            "n_expected": len(expected_ids),
            "n_valid": len(valid),
            "complete": complete,
        }
        if not complete or len(valid) < 2:
            rows.append(row)
            continue

        ids = valid["architecture_id"].to_numpy()
        scores = valid["transformed_score"].to_numpy(dtype=float)
        targets = valid["valid_psnr"].to_numpy(dtype=float)
        params = valid["params_built"].to_numpy(dtype=float)
        c_index, pairs = concordance_index(scores, targets)
        row.update(
            {
                "n_pairs": pairs,
                "c_index": c_index,
                "spearman": safe_spearman(scores, targets),
                "kendall_tau_b": float(kendalltau(scores, targets, variant="b").statistic),
                "partial_spearman_log_params": partial_spearman(scores, targets, params),
                "spearman_score_log_params": safe_spearman(scores, np.log(params)),
            }
        )
        manifest = manifests[str(scenario)]
        if "complexity_bin" in manifest.columns:
            bins = manifest[["architecture_id", "complexity_bin"]].copy()
            binned = valid.merge(bins, on="architecture_id", how="left", validate="one_to_one")
            within_bin_tau = []
            expected_bins = int(bins["complexity_bin"].nunique(dropna=False))
            for _, bin_chunk in binned.groupby("complexity_bin", sort=True, dropna=False):
                if len(bin_chunk) < 2:
                    continue
                value = kendalltau(
                    bin_chunk["transformed_score"].to_numpy(dtype=float),
                    bin_chunk["valid_psnr"].to_numpy(dtype=float),
                    variant="b",
                ).statistic
                if np.isfinite(value):
                    within_bin_tau.append(float(value))
            row["within_bin_kendall_tau_b"] = (
                float(np.mean(within_bin_tau))
                if len(within_bin_tau) == expected_bins
                else math.nan
            )
        for k in (1, 3, 5):
            row.update(top_k_metrics(ids, scores, targets, k))
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_resamples: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return math.nan, math.nan
    indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def summarize_seed_metrics(
    seed_metrics: pd.DataFrame, *, bootstrap_seed: int, bootstrap_resamples: int
) -> pd.DataFrame:
    metric_columns = [
        "c_index",
        "spearman",
        "kendall_tau_b",
        "partial_spearman_log_params",
        "spearman_score_log_params",
        "within_bin_kendall_tau_b",
        "top1_overlap",
        "top1_regret",
        "top1_mean_true_rank",
        "top3_overlap",
        "top3_regret",
        "top3_mean_true_rank",
        "top5_overlap",
        "top5_regret",
        "top5_mean_true_rank",
    ]
    rng = np.random.default_rng(bootstrap_seed)
    rows = []
    grouped = seed_metrics.groupby(["scenario", "proxy", "transformation"], sort=True)
    for (scenario, proxy, transformation), chunk in grouped:
        row: dict[str, object] = {
            "scenario": scenario,
            "proxy": proxy,
            "transformation": transformation,
            "n_seed_rows": len(chunk),
            "n_complete_seeds": int(chunk["complete"].fillna(False).sum()),
            "min_valid_architectures": int(chunk["n_valid"].min()),
        }
        for metric in metric_columns:
            if metric in chunk:
                values = pd.to_numeric(chunk[metric], errors="coerce").to_numpy(dtype=float)
            else:
                values = np.full(len(chunk), np.nan, dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(finite)) if len(finite) else math.nan
            row[f"{metric}_std"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0 if len(finite) else math.nan
            row[f"{metric}_median"] = float(np.median(finite)) if len(finite) else math.nan
            row[f"{metric}_q1"] = float(np.percentile(finite, 25)) if len(finite) else math.nan
            row[f"{metric}_q3"] = float(np.percentile(finite, 75)) if len(finite) else math.nan
            low, high = bootstrap_mean_ci(finite, rng, bootstrap_resamples)
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def select_preferred_transform(summary: pd.DataFrame) -> pd.DataFrame:
    valid = summary[summary["n_complete_seeds"] == summary["n_seed_rows"]].copy()
    valid = valid.sort_values(
        ["scenario", "proxy", "kendall_tau_b_mean", "c_index_mean", "spearman_mean"],
        ascending=[True, True, False, False, False],
    )
    return valid.drop_duplicates(["scenario", "proxy"], keep="first").reset_index(drop=True)


def transformation_stability(seed_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete = seed_metrics[seed_metrics["complete"].fillna(False)].copy()
    complete["_transform_order"] = complete["transformation"].map(TRANSFORM_ORDER)
    complete = complete.sort_values(
        ["scenario", "proxy", "seed", "kendall_tau_b", "c_index", "spearman", "_transform_order"],
        ascending=[True, True, True, False, False, False, True],
        na_position="last",
    )
    winners = complete.drop_duplicates(["scenario", "proxy", "seed"], keep="first")
    counts = (
        winners.groupby(["scenario", "proxy", "transformation"], sort=True)
        .size()
        .rename("winning_seeds")
        .reset_index()
    )
    totals = (
        winners.groupby(["scenario", "proxy"], sort=True)["seed"]
        .nunique()
        .rename("n_seeds")
        .reset_index()
    )
    grid = totals[["scenario", "proxy"]].merge(
        pd.DataFrame({"transformation": TRANSFORMS}), how="cross"
    )
    counts = grid.merge(counts, on=["scenario", "proxy", "transformation"], how="left")
    counts["winning_seeds"] = counts["winning_seeds"].fillna(0).astype(int)
    counts = counts.merge(totals, on=["scenario", "proxy"], how="left")
    counts["winning_seed_fraction"] = counts["winning_seeds"] / counts["n_seeds"]
    return winners.drop(columns="_transform_order"), counts


def pool_statistics(manifests: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for scenario, frame in manifests.items():
        params = frame["params_real"].to_numpy(dtype=float)
        psnr = frame["valid_psnr"].to_numpy(dtype=float)
        params_q1, params_q3 = np.percentile(params, [25, 75])
        psnr_q1, psnr_q3 = np.percentile(psnr, [25, 75])
        rows.append(
            {
                "scenario": scenario,
                "n_architectures": len(frame),
                "params_min": int(params.min()),
                "params_median": float(np.median(params)),
                "params_max": int(params.max()),
                "psnr_min": float(psnr.min()),
                "psnr_median": float(np.median(psnr)),
                "psnr_max": float(psnr.max()),
                "spearman_psnr_log_params": float(spearmanr(psnr, np.log(params)).statistic),
                "kendall_psnr_log_params": float(kendalltau(psnr, np.log(params), variant="b").statistic),
                "params_q1": float(params_q1),
                "params_q3": float(params_q3),
                "psnr_q1": float(psnr_q1),
                "psnr_q3": float(psnr_q3),
                "small_strong_count": int(((params <= params_q1) & (psnr >= psnr_q3)).sum()),
                "large_weak_count": int(((params >= params_q3) & (psnr <= psnr_q1)).sum()),
            }
        )
    return pd.DataFrame(rows)


def runtime_statistics(raw: pd.DataFrame) -> pd.DataFrame:
    valid = raw[raw["status"] == "ok"].copy()
    grouped = valid.groupby(["scenario", "proxy"], sort=True)
    return grouped.agg(
        n_evaluations=("proxy_time_ms", "count"),
        proxy_time_mean_ms=("proxy_time_ms", "mean"),
        proxy_time_median_ms=("proxy_time_ms", "median"),
        proxy_time_std_ms=("proxy_time_ms", "std"),
        build_time_median_ms=("build_time_ms", "median"),
        total_time_median_ms=("total_time_ms", "median"),
    ).reset_index()


def scenario_statistics(preferred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    expanded = preferred[preferred["scenario"] == "expanded50"].copy()
    pareto = preferred[preferred["scenario"] == "pareto20"].copy()
    detail = expanded.merge(pareto, on="proxy", suffixes=("_expanded50", "_pareto20"))
    if detail.empty:
        return detail, pd.DataFrame()
    detail["kendall_rank_expanded50"] = detail["kendall_tau_b_mean_expanded50"].rank(
        ascending=False, method="average"
    )
    detail["kendall_rank_pareto20"] = detail["kendall_tau_b_mean_pareto20"].rank(
        ascending=False, method="average"
    )
    detail["kendall_tau_b_delta_pareto_minus_expanded"] = (
        detail["kendall_tau_b_mean_pareto20"] - detail["kendall_tau_b_mean_expanded50"]
    )
    detail["preferred_transform_changed"] = (
        detail["transformation_expanded50"] != detail["transformation_pareto20"]
    )
    detail["top1_regret_delta_pareto_minus_expanded"] = (
        detail["top1_regret_mean_pareto20"] - detail["top1_regret_mean_expanded50"]
    )
    detail["top3_overlap_delta_pareto_minus_expanded"] = (
        detail["top3_overlap_mean_pareto20"] - detail["top3_overlap_mean_expanded50"]
    )
    change = detail["kendall_tau_b_delta_pareto_minus_expanded"]
    largest_increase_index = change.idxmax()
    largest_decrease_index = change.idxmin()
    smallest_absolute_change_index = change.abs().idxmin()
    summary = pd.DataFrame(
        [
            {
                "n_proxies": len(detail),
                "preferred_transform_changes": int(detail["preferred_transform_changed"].sum()),
                "spearman_method_ranks": safe_spearman(
                    detail["kendall_tau_b_mean_expanded50"].to_numpy(dtype=float),
                    detail["kendall_tau_b_mean_pareto20"].to_numpy(dtype=float),
                ),
                "kendall_method_ranks": float(
                    kendalltau(
                        detail["kendall_tau_b_mean_expanded50"],
                        detail["kendall_tau_b_mean_pareto20"],
                        variant="b",
                    ).statistic
                ),
                "mean_absolute_kendall_change": float(
                    change.abs().mean()
                ),
                "all_kendall_changes_positive": bool((change > 0).all()),
                "largest_increase_proxy": (
                    str(detail.loc[largest_increase_index, "proxy"])
                    if float(change.loc[largest_increase_index]) > 0
                    else ""
                ),
                "largest_decrease_proxy": (
                    str(detail.loc[largest_decrease_index, "proxy"])
                    if float(change.loc[largest_decrease_index]) < 0
                    else ""
                ),
                "smallest_absolute_change_proxy": str(
                    detail.loc[smallest_absolute_change_index, "proxy"]
                ),
            }
        ]
    )
    return detail, summary


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def paired_statistics(seed_metrics: pd.DataFrame, preferred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    omnibus_rows = []
    posthoc_rows = []
    for scenario, choices in preferred.groupby("scenario"):
        selected = seed_metrics.merge(
            choices[["scenario", "proxy", "transformation"]],
            on=["scenario", "proxy", "transformation"],
            how="inner",
        )
        pivot = selected.pivot(index="seed", columns="proxy", values="kendall_tau_b").dropna()
        if pivot.shape[0] < 2 or pivot.shape[1] < 3:
            omnibus_rows.append(
                {
                    "scenario": scenario,
                    "n_paired_seeds": pivot.shape[0],
                    "n_methods": pivot.shape[1],
                    "friedman_statistic": math.nan,
                    "friedman_p": math.nan,
                    "note": "Insufficient complete paired data",
                }
            )
            continue
        statistic, p_value = friedmanchisquare(*(pivot[column] for column in pivot.columns))
        omnibus_rows.append(
            {
                "scenario": scenario,
                "n_paired_seeds": pivot.shape[0],
                "n_methods": pivot.shape[1],
                "friedman_statistic": float(statistic),
                "friedman_p": float(p_value),
                "note": "Exploratory: each proxy uses its post-hoc preferred transformation",
            }
        )
        if p_value >= 0.05:
            continue
        comparisons = []
        columns = list(pivot.columns)
        for left_index, left in enumerate(columns[:-1]):
            for right in columns[left_index + 1 :]:
                delta = pivot[left].to_numpy() - pivot[right].to_numpy()
                if np.allclose(delta, 0.0):
                    test_stat, raw_p = 0.0, 1.0
                else:
                    test = wilcoxon(pivot[left], pivot[right], zero_method="pratt", alternative="two-sided")
                    test_stat, raw_p = float(test.statistic), float(test.pvalue)
                absolute_ranks = rankdata(np.abs(delta), method="average")
                positive_ranks = float(absolute_ranks[delta > 0].sum())
                negative_ranks = float(absolute_ranks[delta < 0].sum())
                ranked_total = positive_ranks + negative_ranks
                matched_rank_biserial = (
                    (positive_ranks - negative_ranks) / ranked_total if ranked_total else 0.0
                )
                comparisons.append(
                    {
                        "scenario": scenario,
                        "method_a": left,
                        "method_b": right,
                        "wilcoxon_statistic": test_stat,
                        "p_raw": raw_p,
                        "median_paired_difference": float(np.median(delta)),
                        "matched_rank_biserial": matched_rank_biserial,
                    }
                )
        adjusted = holm_adjust([float(item["p_raw"]) for item in comparisons])
        for item, p_adjusted in zip(comparisons, adjusted):
            item["p_holm"] = p_adjusted
            item["significant_0_05"] = p_adjusted < 0.05
            posthoc_rows.append(item)
    return pd.DataFrame(omnibus_rows), pd.DataFrame(posthoc_rows)


def _display_proxy(value: str) -> str:
    return DISPLAY_NAMES.get(value, value)


def configure_plot_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig, figure_dir: Path, stem: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.png", dpi=300, bbox_inches="tight")


def make_figures(
    manifests: dict[str, pd.DataFrame],
    seed_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    preferred: pd.DataFrame,
    runtime: pd.DataFrame,
    figure_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    configure_plot_style()
    outputs: list[str] = []

    primary_color = "#0072B2"
    secondary_color = "#D55E00"
    neutral_color = "#666666"
    grid_color = "#D9D9D9"
    palette = {"expanded50": primary_color, "pareto20": secondary_color}
    markers = {"expanded50": "o", "pareto20": "s"}
    fig, ax = plt.subplots(figsize=(5.2, 3.35))
    for scenario, frame in manifests.items():
        ax.scatter(
            frame["valid_psnr"],
            frame["params_real"],
            s=23,
            alpha=0.78,
            color=palette.get(scenario, neutral_color),
            marker=markers.get(scenario, "o"),
            edgecolors="white",
            linewidths=0.35,
            label="Strategic 50" if scenario == "expanded50" else "Pareto-conditioned 20",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Best validation PSNR (dB)")
    ax.set_ylabel("Number of parameters (log scale)")
    ax.grid(True, which="major", color=grid_color, linewidth=0.55)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    save_figure(fig, figure_dir, "pool_coverage_psnr_parameters")
    plt.close(fig)
    outputs.append("pool_coverage_psnr_parameters")

    primary = summary[summary["scenario"] == "expanded50"].copy()
    if not primary.empty:
        methods = [metric for metric in DISPLAY_NAMES if metric in set(primary["proxy"])]
        matrix = np.full((len(methods), len(TRANSFORMS)), np.nan)
        for row_index, proxy in enumerate(methods):
            for column_index, transform in enumerate(TRANSFORMS):
                rows = primary[(primary["proxy"] == proxy) & (primary["transformation"] == transform)]
                if not rows.empty:
                    matrix[row_index, column_index] = rows.iloc[0]["kendall_tau_b_mean"]
        fig, ax = plt.subplots(figsize=(5.4, 4.25))
        correlation_cmap = LinearSegmentedColormap.from_list(
            "strategic_blue_white_pareto_orange",
            [primary_color, "#F7F7F7", secondary_color],
        )
        image = ax.imshow(matrix, cmap=correlation_cmap, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(TRANSFORMS)), ["Orig.", "Param.-adj.", "Inv.", "Inv.+param."], rotation=25, ha="right")
        ax.set_yticks(range(len(methods)), [_display_proxy(method) for method in methods])
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                label = "--" if not np.isfinite(value) else f"{value:.2f}"
                color = "white" if np.isfinite(value) and abs(value) > 0.55 else "black"
                ax.text(column_index, row_index, label, ha="center", va="center", color=color, fontsize=7.3)
        colorbar = fig.colorbar(image, ax=ax, pad=0.02)
        colorbar.set_label("Mean Kendall's tau-b")
        fig.tight_layout()
        save_figure(fig, figure_dir, "expanded50_proxy_transformation_heatmap")
        plt.close(fig)
        outputs.append("expanded50_proxy_transformation_heatmap")

    primary_choices = preferred[preferred["scenario"] == "expanded50"]
    if not primary_choices.empty:
        selected = seed_metrics.merge(
            primary_choices[["scenario", "proxy", "transformation"]],
            on=["scenario", "proxy", "transformation"],
            how="inner",
        )
        order = (
            primary_choices.sort_values("kendall_tau_b_mean", ascending=False)["proxy"].tolist()
        )
        data = [selected.loc[selected["proxy"] == proxy, "kendall_tau_b"].dropna().to_numpy() for proxy in order]
        fig, ax = plt.subplots(figsize=(6.6, 3.45))
        boxes = ax.boxplot(
            data,
            patch_artist=True,
            showfliers=False,
            widths=0.62,
            medianprops={"color": "#222222", "linewidth": 1.0},
            whiskerprops={"color": neutral_color, "linewidth": 0.8},
            capprops={"color": neutral_color, "linewidth": 0.8},
        )
        for box in boxes["boxes"]:
            box.set_facecolor(primary_color)
            box.set_edgecolor(primary_color)
            box.set_alpha(0.72)
        ax.set_xticks(range(1, len(order) + 1), [_display_proxy(proxy) for proxy in order], rotation=32, ha="right")
        ax.set_ylabel("Kendall's tau-b")
        ax.axhline(0.0, color="#555555", linewidth=0.7)
        ax.grid(True, axis="y", color=grid_color, linewidth=0.55)
        fig.tight_layout()
        save_figure(fig, figure_dir, "expanded50_seed_distributions_preferred_transform")
        plt.close(fig)
        outputs.append("expanded50_seed_distributions_preferred_transform")

        merged = primary_choices.merge(runtime[runtime["scenario"] == "expanded50"], on=["scenario", "proxy"])
        fig, ax = plt.subplots(figsize=(5.1, 3.35))
        ax.scatter(
            merged["proxy_time_median_ms"],
            merged["kendall_tau_b_mean"],
            color=primary_color,
            edgecolors="white",
            linewidths=0.35,
            s=30,
        )
        runtime_offsets = {
            "plain": (7, -10),
            "snip": (7, 6),
            "grad_norm": (5, 6),
            "synflow": (5, 6),
            "zen": (7, -12),
            "zico": (7, 7),
            "jacob": (7, 5),
            "nwot": (7, 5),
            "fisher": (7, 5),
            "grasp": (7, 5),
        }
        for _, row in merged.iterrows():
            offset = runtime_offsets.get(row["proxy"], (4, 3))
            ax.annotate(
                _display_proxy(row["proxy"]),
                (row["proxy_time_median_ms"], row["kendall_tau_b_mean"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Median score-computation time (ms, log scale)")
        ax.set_ylabel("Mean Kendall's tau-b")
        ax.margins(y=0.10)
        ax.grid(True, which="major", color=grid_color, linewidth=0.55)
        fig.tight_layout()
        save_figure(fig, figure_dir, "expanded50_runtime_vs_ranking")
        plt.close(fig)
        outputs.append("expanded50_runtime_vs_ranking")

        fig, ax = plt.subplots(figsize=(5.0, 3.5))
        ax.scatter(
            primary_choices["spearman_score_log_params_mean"],
            primary_choices["spearman_mean"],
            color=primary_color,
            edgecolors="white",
            linewidths=0.35,
            s=30,
        )
        size_offsets = {
            "param_score": (-8, 18, "right"),
            "jacob": (-8, 8, "right"),
            "nwot": (-10, 5, "right"),
            "l2_norm": (-8, -2, "right"),
            "zico": (8, -2, "left"),
            "zen": (8, -20, "left"),
            "fisher": (8, -16, "left"),
            "plain": (5, -9, "left"),
            "snip": (-5, 7, "right"),
        }
        for _, row in primary_choices.iterrows():
            dx, dy, alignment = size_offsets.get(row["proxy"], (5, 4, "left"))
            ax.annotate(
                _display_proxy(row["proxy"]),
                (row["spearman_score_log_params_mean"], row["spearman_mean"]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=alignment,
                fontsize=7,
            )
        ax.set_xlabel("Spearman correlation with log(parameters)")
        ax.set_ylabel("Spearman correlation with PSNR")
        ax.axhline(0.0, color="#777777", linewidth=0.6)
        ax.axvline(0.0, color="#777777", linewidth=0.6)
        ax.margins(x=0.09, y=0.20)
        ax.grid(True, color=grid_color, linewidth=0.55)
        fig.tight_layout()
        save_figure(fig, figure_dir, "expanded50_size_bias_map")
        plt.close(fig)
        outputs.append("expanded50_size_bias_map")

    if set(preferred["scenario"]) >= {"expanded50", "pareto20"}:
        comparison = preferred.pivot(index="proxy", columns="scenario", values="kendall_tau_b_mean").dropna()
        comparison = comparison.sort_values("expanded50")
        y = np.arange(len(comparison))
        fig, ax = plt.subplots(figsize=(5.4, 4.1))
        for index, (_, row) in enumerate(comparison.iterrows()):
            ax.plot([row["expanded50"], row["pareto20"]], [index, index], color="#AAAAAA", linewidth=1)
        ax.scatter(comparison["expanded50"], y, color=palette["expanded50"], label="Strategic 50", zorder=3)
        ax.scatter(comparison["pareto20"], y, color=palette["pareto20"], marker="s", label="Pareto-conditioned 20", zorder=3)
        ax.set_yticks(y, [_display_proxy(proxy) for proxy in comparison.index])
        ax.set_xlabel("Mean Kendall's tau-b (post-hoc preferred transformation)")
        ax.axvline(0.0, color="#777777", linewidth=0.6)
        ax.grid(True, axis="x", color=grid_color, linewidth=0.55)
        ax.legend(frameon=False)
        fig.tight_layout()
        save_figure(fig, figure_dir, "scenario_dependence_kendall")
        plt.close(fig)
        outputs.append("scenario_dependence_kendall")

    return outputs


def latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_")


def write_latex_tables(
    pool_stats: pd.DataFrame,
    summary: pd.DataFrame,
    preferred: pd.DataFrame,
    transform_stability: pd.DataFrame,
    runtime: pd.DataFrame,
    omnibus: pd.DataFrame,
    table_dir: Path,
) -> list[str]:
    table_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    pool_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Scenario & $n$ & Parameters (min--median--max) & PSNR (min--median--max) & $\rho$(PSNR, $\log p$) \\",
        r"\midrule",
    ]
    for _, row in pool_stats.iterrows():
        label = "Strategic 50" if row["scenario"] == "expanded50" else "Pareto-conditioned 20"
        pool_lines.append(
            f"{label} & {int(row['n_architectures'])} & "
            f"{int(row['params_min']):,}--{int(row['params_median']):,}--{int(row['params_max']):,} & "
            f"{row['psnr_min']:.3f}--{row['psnr_median']:.3f}--{row['psnr_max']:.3f} & "
            f"{row['spearman_psnr_log_params']:.3f} \\\\"
        )
    pool_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (table_dir / "pool_characterization.tex").write_text("\n".join(pool_lines) + "\n", encoding="utf-8")
    files.append("pool_characterization.tex")

    primary = preferred[preferred["scenario"] == "expanded50"].copy()
    primary_stability = transform_stability[transform_stability["scenario"] == "expanded50"]
    primary = primary.merge(
        primary_stability[["proxy", "transformation", "winning_seed_fraction"]],
        on=["proxy", "transformation"],
        how="left",
    ).sort_values("kendall_tau_b_mean", ascending=False)
    main_lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Method & Transformation & C-index & Spearman & Kendall's $\tau_b$ & Win share \\",
        r"\midrule",
    ]
    for _, row in primary.iterrows():
        main_lines.append(
            f"{_display_proxy(row['proxy'])} & {TRANSFORM_NAMES[row['transformation']]} & "
            f"{row['c_index_mean']:.3f} $\\pm$ {row['c_index_std']:.3f} & "
            f"{row['spearman_mean']:.3f} $\\pm$ {row['spearman_std']:.3f} & "
            f"{row['kendall_tau_b_mean']:.3f} $\\pm$ {row['kendall_tau_b_std']:.3f} & "
            f"{row['winning_seed_fraction']:.2f} \\\\"
        )
    main_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (table_dir / "expanded50_preferred_transform_results.tex").write_text("\n".join(main_lines) + "\n", encoding="utf-8")
    files.append("expanded50_preferred_transform_results.tex")

    all_primary = summary[summary["scenario"] == "expanded50"].copy()
    all_primary["_transform_order"] = all_primary["transformation"].map(TRANSFORM_ORDER)
    all_primary = all_primary.sort_values(["proxy", "_transform_order"])
    all_lines = [
        r"\begin{tabular}{@{}lp{0.25\textwidth}*{3}{>{\centering\arraybackslash}p{0.15\textwidth}}@{}}",
        r"\toprule",
        r"Method & Transformation & C-index & Spearman & Kendall's $\tau_b$ \\",
        r"\midrule",
    ]
    for _, row in all_primary.iterrows():
        def metric_text(metric: str) -> str:
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            return "--" if not np.isfinite(mean) else f"{mean:.3f} $\\pm$ {std:.3f}"

        all_lines.append(
            f"{_display_proxy(row['proxy'])} & {APPENDIX_TRANSFORM_NAMES[row['transformation']]} & "
            f"{metric_text('c_index')} & {metric_text('spearman')} & {metric_text('kendall_tau_b')} \\\\"
        )
    all_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (table_dir / "expanded50_all_transformations.tex").write_text("\n".join(all_lines) + "\n", encoding="utf-8")
    files.append("expanded50_all_transformations.tex")

    pareto_all = summary[summary["scenario"] == "pareto20"].copy()
    pareto_all["_transform_order"] = pareto_all["transformation"].map(TRANSFORM_ORDER)
    pareto_all = pareto_all.sort_values(["proxy", "_transform_order"])
    pareto_lines = [
        r"\begin{tabular}{@{}lp{0.25\textwidth}*{3}{>{\centering\arraybackslash}p{0.15\textwidth}}@{}}",
        r"\toprule",
        r"Method & Transformation & C-index & Spearman & Kendall's $\tau_b$ \\",
        r"\midrule",
    ]
    for _, row in pareto_all.iterrows():
        def pareto_metric_text(metric: str) -> str:
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            return "--" if not np.isfinite(mean) else f"{mean:.3f} $\\pm$ {std:.3f}"

        pareto_lines.append(
            f"{_display_proxy(row['proxy'])} & {APPENDIX_TRANSFORM_NAMES[row['transformation']]} & "
            f"{pareto_metric_text('c_index')} & {pareto_metric_text('spearman')} & "
            f"{pareto_metric_text('kendall_tau_b')} \\\\"
        )
    pareto_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (table_dir / "pareto20_all_transformations.tex").write_text(
        "\n".join(pareto_lines) + "\n", encoding="utf-8"
    )
    files.append("pareto20_all_transformations.tex")

    topk_lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Method & Transform & Top-1 hit & Top-1 regret & Top-3 overlap & Top-3 regret & Top-5 overlap & Top-5 regret & Top-5 rank \\",
        r"\midrule",
    ]
    for _, row in primary.iterrows():
        topk_lines.append(
            f"{_display_proxy(row['proxy'])} & {TRANSFORM_NAMES[row['transformation']]} & "
            f"{row['top1_overlap_mean']:.3f} & {row['top1_regret_mean']:.3f} & "
            f"{row['top3_overlap_mean']:.3f} & "
            f"{row['top3_regret_mean']:.3f} & {row['top5_overlap_mean']:.3f} & "
            f"{row['top5_regret_mean']:.3f} & {row['top5_mean_true_rank_mean']:.2f} \\\\"
        )
    topk_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (table_dir / "expanded50_topk.tex").write_text("\n".join(topk_lines) + "\n", encoding="utf-8")
    files.append("expanded50_topk.tex")

    size_lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Method & Transform & $\rho$(score, $\log p$) & Partial $\rho$ & Within-bin $\tau_b$ & Overall $\tau_b$ \\",
        r"\midrule",
    ]
    for _, row in primary.iterrows():
        partial = row["partial_spearman_log_params_mean"]
        partial_text = "--" if not np.isfinite(partial) else f"{partial:.3f}"
        within_bin = row["within_bin_kendall_tau_b_mean"]
        within_bin_text = "--" if not np.isfinite(within_bin) else f"{within_bin:.3f}"
        size_lines.append(
            f"{_display_proxy(row['proxy'])} & {TRANSFORM_NAMES[row['transformation']]} & "
            f"{row['spearman_score_log_params_mean']:.3f} & {partial_text} & {within_bin_text} & "
            f"{row['kendall_tau_b_mean']:.3f} \\\\"
        )
    size_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (table_dir / "expanded50_size_bias.tex").write_text("\n".join(size_lines) + "\n", encoding="utf-8")
    files.append("expanded50_size_bias.tex")

    comparison = preferred.pivot(index="proxy", columns="scenario")
    comparison_lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & Strategic-50 transform & Strategic-50 $\tau_b$ & Pareto transform & Pareto $\tau_b$ & $\Delta\tau_b$ \\",
        r"\midrule",
    ]
    required_comparison_columns = {
        ("transformation", "expanded50"),
        ("transformation", "pareto20"),
        ("kendall_tau_b_mean", "expanded50"),
        ("kendall_tau_b_mean", "pareto20"),
    }
    comparable_proxies = comparison.index if required_comparison_columns <= set(comparison.columns) else []
    for proxy in sorted(comparable_proxies):
        expanded_transform = comparison.loc[proxy, ("transformation", "expanded50")]
        pareto_transform = comparison.loc[proxy, ("transformation", "pareto20")]
        expanded_tau = float(comparison.loc[proxy, ("kendall_tau_b_mean", "expanded50")])
        pareto_tau = float(comparison.loc[proxy, ("kendall_tau_b_mean", "pareto20")])
        comparison_lines.append(
            f"{_display_proxy(proxy)} & {TRANSFORM_NAMES[expanded_transform]} & {expanded_tau:.3f} & "
            f"{TRANSFORM_NAMES[pareto_transform]} & {pareto_tau:.3f} & {pareto_tau - expanded_tau:+.3f} \\\\"
        )
    comparison_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (table_dir / "scenario_comparison.tex").write_text(
        "\n".join(comparison_lines) + "\n", encoding="utf-8"
    )
    files.append("scenario_comparison.tex")

    runtime_primary = runtime[runtime["scenario"] == "expanded50"].sort_values("proxy_time_median_ms")
    def format_milliseconds(value: float) -> str:
        value = float(value)
        return f"{value:.3f}" if value < 0.1 else f"{value:.1f}"

    runtime_lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & Build median (ms) & Score median (ms) & Total median (ms) \\",
        r"\midrule",
    ]
    for _, row in runtime_primary.iterrows():
        runtime_lines.append(
            f"{_display_proxy(row['proxy'])} & {format_milliseconds(row['build_time_median_ms'])} & "
            f"{format_milliseconds(row['proxy_time_median_ms'])} & "
            f"{format_milliseconds(row['total_time_median_ms'])} \\\\"
        )
    runtime_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (table_dir / "expanded50_runtime.tex").write_text("\n".join(runtime_lines) + "\n", encoding="utf-8")
    files.append("expanded50_runtime.tex")

    omnibus_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Scenario & Paired seeds & Methods & Friedman $\chi^2$ & $p$-value \\",
        r"\midrule",
    ]
    for _, row in omnibus.iterrows():
        label = "Strategic 50" if row["scenario"] == "expanded50" else "Pareto-conditioned 20"
        statistic = "--" if not np.isfinite(row["friedman_statistic"]) else f"{row['friedman_statistic']:.3f}"
        if not np.isfinite(row["friedman_p"]):
            p_value = "--"
        elif row["friedman_p"] == 0:
            p_value = "$<10^{-300}$"
        elif row["friedman_p"] < 0.001:
            exponent = int(math.floor(math.log10(row["friedman_p"])))
            coefficient = row["friedman_p"] / (10 ** exponent)
            p_value = f"${coefficient:.2f}\\times10^{{{exponent}}}$"
        else:
            p_value = f"{row['friedman_p']:.3f}"
        omnibus_lines.append(
            f"{label} & {int(row['n_paired_seeds'])} & {int(row['n_methods'])} & "
            f"{statistic} & {p_value} \\\\"
        )
    omnibus_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (table_dir / "friedman_summary.tex").write_text("\n".join(omnibus_lines) + "\n", encoding="utf-8")
    files.append("friedman_summary.tex")
    return files


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze(args: argparse.Namespace) -> dict[str, object]:
    manifests = {
        "expanded50": load_manifest(args.expanded50_manifest),
        "pareto20": load_manifest(args.pareto20_manifest),
    }
    raw = load_raw(args.raw_scores)
    validation = validate_raw(raw, manifests)
    transformed = apply_transforms(raw)
    seed_metrics = calculate_seed_metrics(transformed, manifests)
    summary = summarize_seed_metrics(
        seed_metrics,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    preferred = select_preferred_transform(summary)
    transform_winners, transform_stability = transformation_stability(seed_metrics)
    pool_stats = pool_statistics(manifests)
    runtime = runtime_statistics(raw)
    scenario_detail, scenario_summary = scenario_statistics(preferred)
    omnibus, posthoc = paired_statistics(seed_metrics, preferred)

    processed_dir = args.output_root / "processed"
    write_csv(processed_dir / "transformed_scores_long.csv", transformed)
    write_csv(processed_dir / "ranking_metrics_by_seed.csv", seed_metrics)
    write_csv(processed_dir / "ranking_summary_all_transformations.csv", summary)
    write_csv(processed_dir / "preferred_transformation_by_proxy.csv", preferred)
    write_csv(processed_dir / "preferred_transformation_by_seed.csv", transform_winners)
    write_csv(processed_dir / "transformation_stability.csv", transform_stability)
    write_csv(processed_dir / "pool_characterization.csv", pool_stats)
    write_csv(processed_dir / "runtime_summary.csv", runtime)
    write_csv(processed_dir / "scenario_preferred_comparison.csv", scenario_detail)
    write_csv(processed_dir / "scenario_comparison_summary.csv", scenario_summary)
    write_csv(processed_dir / "friedman_preferred_transform.csv", omnibus)
    write_csv(processed_dir / "wilcoxon_holm_preferred_transform.csv", posthoc)
    (processed_dir / "validation_report.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8"
    )

    figure_stems = make_figures(
        manifests,
        seed_metrics,
        summary,
        preferred,
        runtime,
        args.figure_dir,
    )
    table_files = write_latex_tables(
        pool_stats,
        summary,
        preferred,
        transform_stability,
        runtime,
        omnibus,
        args.table_dir,
    )
    artifact_sources = {
        "figures/rework/pool_coverage_psnr_parameters.pdf": [
            "manifests/expanded50_manifest.csv",
            "manifests/pareto20_manifest.csv",
        ],
        "figures/rework/expanded50_proxy_transformation_heatmap.pdf": [
            "processed/ranking_summary_all_transformations.csv"
        ],
        "figures/rework/expanded50_seed_distributions_preferred_transform.pdf": [
            "processed/ranking_metrics_by_seed.csv",
            "processed/preferred_transformation_by_proxy.csv",
        ],
        "figures/rework/expanded50_runtime_vs_ranking.pdf": [
            "processed/runtime_summary.csv",
            "processed/preferred_transformation_by_proxy.csv",
        ],
        "figures/rework/expanded50_size_bias_map.pdf": [
            "processed/preferred_transformation_by_proxy.csv"
        ],
        "figures/rework/scenario_dependence_kendall.pdf": [
            "processed/preferred_transformation_by_proxy.csv",
            "processed/scenario_preferred_comparison.csv",
        ],
        "tables/rework/pool_characterization.tex": ["processed/pool_characterization.csv"],
        "tables/rework/expanded50_preferred_transform_results.tex": [
            "processed/preferred_transformation_by_proxy.csv",
            "processed/transformation_stability.csv",
        ],
        "tables/rework/expanded50_all_transformations.tex": [
            "processed/ranking_summary_all_transformations.csv"
        ],
        "tables/rework/pareto20_all_transformations.tex": [
            "processed/ranking_summary_all_transformations.csv"
        ],
        "tables/rework/expanded50_topk.tex": [
            "processed/preferred_transformation_by_proxy.csv"
        ],
        "tables/rework/expanded50_size_bias.tex": [
            "processed/preferred_transformation_by_proxy.csv"
        ],
        "tables/rework/scenario_comparison.tex": [
            "processed/preferred_transformation_by_proxy.csv",
            "processed/scenario_preferred_comparison.csv",
        ],
        "tables/rework/expanded50_runtime.tex": ["processed/runtime_summary.csv"],
        "tables/rework/friedman_summary.tex": ["processed/friedman_preferred_transform.csv"],
    }
    manifest = {
        "raw_score_files": [str(path.resolve()) for path in args.raw_scores],
        "processed_files": sorted(path.name for path in processed_dir.iterdir()),
        "figures": [f"{stem}.pdf" for stem in figure_stems],
        "figure_previews": [f"{stem}.png" for stem in figure_stems],
        "tables": table_files,
        "artifact_sources": {
            artifact: {
                "generator": "tools/analyze_zerocost_rework.py",
                "data": sources,
            }
            for artifact, sources in artifact_sources.items()
        },
        "sha256": {
            "raw_scores": {str(path.resolve()): sha256(path) for path in args.raw_scores},
            "processed": {
                path.name: sha256(path) for path in sorted(processed_dir.iterdir()) if path.is_file()
            },
            "figures": {
                path.name: sha256(path)
                for path in sorted(args.figure_dir.iterdir())
                if path.is_file() and path.suffix.lower() in {".pdf", ".png"}
            },
            "tables": {
                path.name: sha256(path)
                for path in sorted(args.table_dir.iterdir())
                if path.is_file() and path.suffix.lower() == ".tex"
            },
        },
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "preferred_transformation_note": (
            "Preferred transformations are selected post hoc by mean Kendall tau-b, "
            "with C-index and Spearman as tie-breakers; they are descriptive/oracle choices."
        ),
        "statistical_unit": "paired random-initialization seed",
    }
    (args.output_root / "result_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze both zero-cost rework scenarios and generate publication artifacts."
    )
    parser.add_argument(
        "--expanded50-manifest",
        type=Path,
        default=DEFAULT_ROOT / "manifests" / "expanded50_manifest.csv",
    )
    parser.add_argument(
        "--pareto20-manifest",
        type=Path,
        default=DEFAULT_ROOT / "manifests" / "pareto20_manifest.csv",
    )
    parser.add_argument("--raw-scores", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=REPO_ROOT / "Zero-cost paper" / "figures" / "rework",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=REPO_ROOT / "Zero-cost paper" / "tables" / "rework",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = analyze(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
