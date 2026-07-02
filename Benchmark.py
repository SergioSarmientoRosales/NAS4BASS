from __future__ import annotations

import ast
import math
import random
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from search_space.model_builder import get_model as build_model_from_genotype
from search_space.search_space import decode as decode_gene_to_genotype

try:
    from scipy.stats import kendalltau, spearmanr
except ImportError as e:
    raise ImportError(
        "This script requires scipy. Install it with: pip install scipy"
    ) from e

# ============================================================
# 0) Runtime / reproducibility
# ============================================================
def configure_runtime(cpu_mode: bool = False) -> None:
    if cpu_mode:
        tf.config.set_visible_devices([], "GPU")
        print("[RUNTIME] CPU mode enabled")
        return

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            print(f"[RUNTIME] GPU mode enabled | GPUs: {len(gpus)}")
        except Exception as e:
            print("[RUNTIME] GPU memory_growth warning:", e)
    else:
        print("[RUNTIME] No GPU found, using CPU")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# 1) Keras-serializable utilities
# ============================================================
@tf.keras.utils.register_keras_serializable(package="sr")
def psnr(y_true, y_pred):
    y_true = tf.cast(tf.clip_by_value(y_true, 0.0, 1.0), tf.float32)
    y_pred = tf.cast(tf.clip_by_value(y_pred, 0.0, 1.0), tf.float32)
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


# ============================================================
# 2) CSV loader
# ============================================================
def _parse_gene_field(value, expected_len: int = 28) -> List[int]:
    if isinstance(value, list):
        gene = [int(v) for v in value]
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Empty gene string.")
        if text.startswith("[") or text.startswith("("):
            gene = [int(v) for v in ast.literal_eval(text)]
        else:
            gene = [int(v.strip()) for v in text.split(",") if v.strip() != ""]
    else:
        raise TypeError(f"Unsupported gene type: {type(value)}")

    if expected_len is not None and len(gene) != expected_len:
        raise ValueError(f"Expected gene length {expected_len}, got {len(gene)}")
    return gene


def load_benchmark_models_csv(path: str, expected_len: int = 28) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")

    gene_candidates = ["Net", "gene", "Gene", "genotype", "architecture", "arch"]
    psnr_candidates = ["valid_psnr", "psnr", "PSNR", "Valid_PSNR"]
    id_candidates = ["model_id", "id", "name", "Model", "model_name"]

    gene_col = next((c for c in gene_candidates if c in df.columns), None)
    psnr_col = next((c for c in psnr_candidates if c in df.columns), None)
    id_col = next((c for c in id_candidates if c in df.columns), None)

    if gene_col is None:
        raise ValueError(f"Could not find a gene column. Tried: {gene_candidates}")
    if psnr_col is None:
        raise ValueError(f"Could not find a PSNR column. Tried: {psnr_candidates}")

    out = pd.DataFrame()
    out["model_index"] = np.arange(1, len(df) + 1)
    out["model_id"] = df[id_col].astype(str) if id_col is not None else out["model_index"].astype(str)
    out["gene"] = df[gene_col].apply(lambda x: _parse_gene_field(x, expected_len=expected_len))
    out["valid_psnr"] = pd.to_numeric(df[psnr_col], errors="coerce")

    bad_psnr = out["valid_psnr"].isna()
    if bad_psnr.any():
        bad_rows = out.loc[bad_psnr, "model_index"].tolist()
        raise ValueError(f"Invalid PSNR values in rows: {bad_rows}")

    return out


# ============================================================
# 4) Synthetic SR batch / loss
# ============================================================
def make_dummy_sr_batch(
    batch_size: int,
    lr_patch_size: int,
    ratio: int,
    dtype: tf.dtypes.DType = tf.float32,
) -> Tuple[tf.Tensor, tf.Tensor]:
    hr_patch_size = lr_patch_size * ratio

    lr = tf.random.uniform(
        shape=(batch_size, lr_patch_size, lr_patch_size, 3),
        minval=0.0,
        maxval=1.0,
        dtype=dtype,
    )
    hr = tf.random.uniform(
        shape=(batch_size, hr_patch_size, hr_patch_size, 3),
        minval=0.0,
        maxval=1.0,
        dtype=dtype,
    )
    return lr, hr


def get_sr_loss() -> tf.keras.losses.Loss:
    return tf.keras.losses.MeanSquaredError()


# ============================================================
# 5) Zero-cost registry
# ============================================================
def load_zero_cost_functions() -> Dict[str, Callable]:
    registry: Dict[str, Callable] = {}

    from evaluators.metrics.synflow import compute_synflow_raw
    from evaluators.metrics.fisher import compute_fisher
    from evaluators.metrics.grad_norm import compute_grad_norm
    from evaluators.metrics.grasp import compute_grasp
    from evaluators.metrics.jacob_cov import compute_jacob_cov
    from evaluators.metrics.l2_norm import compute_l2_norm
    from evaluators.metrics.plain import compute_plain
    from evaluators.metrics.params_score import compute_param_score
    from evaluators.metrics.snip import compute_snip
    from evaluators.metrics.nwot import compute_nwot
    from evaluators.metrics.zen import compute_zen_score
    from evaluators.metrics.zico import compute_zico

    registry["zico"] = compute_zico
    registry["zen"] = compute_zen_score
    registry["plain"] = compute_plain
    registry["nwot"] = compute_nwot
    registry["snip"] = compute_snip
    registry["param_score"] = compute_param_score
    registry["l2_norm"] = compute_l2_norm
    registry["jacob"] = compute_jacob_cov
    registry["grasp"] = compute_grasp
    registry["synflow"] = compute_synflow_raw
    registry["fisher"] = compute_fisher
    registry["grad_norm"] = compute_grad_norm

    return registry


# ============================================================
# 6) Score variants
# ============================================================
def safe_div_by_params(score: float, params: int) -> float:
    if params <= 0 or not np.isfinite(score):
        return np.nan
    return float(score) / float(params)


def safe_neg(score: float) -> float:
    if not np.isfinite(score):
        return np.nan
    return -float(score)


def safe_neg_div_by_params(score: float, params: int) -> float:
    if params <= 0 or not np.isfinite(score):
        return np.nan
    return -float(score) / float(params)


def get_score_variants(include_negative_variants: bool = False) -> Dict[str, Callable[[float, int], float]]:
    variants = {
        "raw": lambda score, params: float(score) if np.isfinite(score) else np.nan,
        "div_params": safe_div_by_params,
    }

    if include_negative_variants:
        variants["neg_raw"] = lambda score, params: safe_neg(score)
        variants["neg_div_params"] = safe_neg_div_by_params

    return variants


# ============================================================
# 7) Stats helpers
# ============================================================
def finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def finite_mean_std(values: np.ndarray) -> Tuple[float, float, int]:
    arr = finite_values(values)
    n = int(arr.size)

    if n == 0:
        return np.nan, np.nan, 0
    if n == 1:
        return float(arr[0]), 0.0, 1

    return float(np.mean(arr)), float(np.std(arr, ddof=1)), n


def finite_median(values: np.ndarray) -> float:
    arr = finite_values(values)
    if arr.size == 0:
        return np.nan
    return float(np.median(arr))


def finite_iqr(values: np.ndarray) -> float:
    arr = finite_values(values)
    if arr.size == 0:
        return np.nan
    q75, q25 = np.percentile(arr, [75, 25])
    return float(q75 - q25)


def finite_min(values: np.ndarray) -> float:
    arr = finite_values(values)
    if arr.size == 0:
        return np.nan
    return float(np.min(arr))


def finite_max(values: np.ndarray) -> float:
    arr = finite_values(values)
    if arr.size == 0:
        return np.nan
    return float(np.max(arr))


def ci95_from_std(std: float, n: int) -> float:
    if not np.isfinite(std) or n <= 1:
        return np.nan
    return 1.96 * float(std) / math.sqrt(n)


def coefficient_of_variation(mean: float, std: float) -> float:
    if not np.isfinite(mean) or not np.isfinite(std) or mean == 0.0:
        return np.nan
    return abs(float(std) / float(mean))


# ============================================================
# 8) Ranking metrics
# ============================================================
def concordance_index(
    scores: np.ndarray,
    targets: np.ndarray,
    tie_score: float = 0.5,
) -> Tuple[float, int]:
    """
    Pairwise concordance for reranking.
    - target ties are excluded
    - score ties contribute tie_score
    """
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    mask = np.isfinite(scores) & np.isfinite(targets)
    scores = scores[mask]
    targets = targets[mask]

    n = len(scores)
    if n < 2:
        return np.nan, 0

    concordant = 0.0
    comparable_pairs = 0

    for i in range(n - 1):
        for j in range(i + 1, n):
            dy = targets[i] - targets[j]
            if dy == 0:
                continue

            comparable_pairs += 1
            ds = scores[i] - scores[j]

            if ds == 0:
                concordant += tie_score
            elif ds * dy > 0:
                concordant += 1.0

    if comparable_pairs == 0:
        return np.nan, 0

    return float(concordant / comparable_pairs), int(comparable_pairs)


def _safe_corr_fn(fn: Callable, x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan
    if np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fn(x, y)

    if hasattr(result, "statistic"):
        return float(result.statistic)

    return float(result[0])


def compute_ranking_metrics(scores: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    mask = np.isfinite(scores) & np.isfinite(targets)
    scores = scores[mask]
    targets = targets[mask]

    n_models = int(len(scores))
    if n_models < 2:
        return {
            "n_models": n_models,
            "n_pairs": 0,
            "c_index": np.nan,
            "spearman": np.nan,
            "kendall_tau_b": np.nan,
        }

    c_idx, n_pairs = concordance_index(scores, targets, tie_score=0.5)

    return {
        "n_models": n_models,
        "n_pairs": int(n_pairs),
        "c_index": float(c_idx),
        "spearman": _safe_corr_fn(spearmanr, scores, targets),
        "kendall_tau_b": _safe_corr_fn(kendalltau, scores, targets),
    }


def detect_near_constant_scores(scores: np.ndarray, atol: float = 1e-12) -> bool:
    arr = finite_values(scores)
    if arr.size < 2:
        return True
    return np.max(arr) - np.min(arr) <= atol

# ============================================================
# 9) Runtime helpers
# ============================================================
def elapsed_ms(start_time: float, end_time: float) -> float:
    return float((end_time - start_time) * 1000.0)


# ============================================================
# 10) Paper-only benchmark execution
# ============================================================
def evaluate_zero_cost_scores_paper(
    benchmark_df: pd.DataFrame,
    zero_cost_fns: Dict[str, Callable],
    score_variants: Dict[str, Callable[[float, int], float]],
    loss_fn,
    n_seeds: int,
    base_seed: int,
    fixed_lr_batch: tf.Tensor,
    fixed_hr_batch: tf.Tensor,
    upscale_factor: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    metric_runs_df:
        one row per (model, seed, metric), including runtime
    variant_runs_df:
        one row per (model, seed, metric, variant), for ranking analysis
    """
    metric_rows = []
    variant_rows = []

    benchmark_df = benchmark_df.sort_values("model_index").reset_index(drop=True)

    for _, item in benchmark_df.iterrows():
        model_index = int(item["model_index"])
        model_id = str(item["model_id"])
        gene = item["gene"]
        valid_psnr = float(item["valid_psnr"])

        print(f"\n================ Model {model_index}/{len(benchmark_df)} | id={model_id} ================")

        for seed_offset in range(n_seeds):
            seed_value = base_seed + seed_offset

            tf.keras.backend.clear_session()
            set_seed(seed_value)

            t_build_start = time.perf_counter()
            genotype = decode_gene_to_genotype(gene)
            model = build_model_from_genotype(genotype, upscale_factor=upscale_factor)
            _ = model(fixed_lr_batch[:1], training=False)
            params = int(model.count_params())
            t_build_end = time.perf_counter()

            build_time_ms = elapsed_ms(t_build_start, t_build_end)

            print(f"[SEED {seed_value}] params={params} | build_time_ms={build_time_ms:.3f}")

            for metric_name, metric_fn in zero_cost_fns.items():
                raw_score = np.nan
                proxy_time_ms = np.nan
                status = "ok"
                error_msg = ""

                try:
                    t_metric_start = time.perf_counter()
                    score = metric_fn(
                        model=model,
                        inputs=fixed_lr_batch,
                        targets=fixed_hr_batch,
                        loss_fn=loss_fn,
                    )
                    t_metric_end = time.perf_counter()

                    raw_score = float(score) if np.isfinite(score) else np.nan
                    proxy_time_ms = elapsed_ms(t_metric_start, t_metric_end)

                    print(
                        f"  [{metric_name}] raw={raw_score:.6e} | "
                        f"proxy_time_ms={proxy_time_ms:.3f}"
                    )

                except Exception as e:
                    status = "error"
                    error_msg = str(e)
                    print(f"  [{metric_name}] ERROR -> {e}")

                total_eval_like_ms = (
                    build_time_ms + proxy_time_ms
                    if np.isfinite(build_time_ms) and np.isfinite(proxy_time_ms)
                    else np.nan
                )

                metric_rows.append({
                    "model_index": model_index,
                    "model_id": model_id,
                    "seed": seed_value,
                    "params": params,
                    "valid_psnr": valid_psnr,
                    "metric": metric_name,
                    "raw_score": raw_score,
                    "build_time_ms": build_time_ms,
                    "proxy_time_ms": proxy_time_ms,
                    "total_eval_like_ms": total_eval_like_ms,
                    "status": status,
                    "error_msg": error_msg,
                })

                for variant_name, variant_fn in score_variants.items():
                    variant_score = variant_fn(raw_score, params)
                    variant_rows.append({
                        "model_index": model_index,
                        "model_id": model_id,
                        "seed": seed_value,
                        "params": params,
                        "valid_psnr": valid_psnr,
                        "metric": metric_name,
                        "variant": variant_name,
                        "raw_score": raw_score,
                        "variant_score": variant_score,
                    })

            try:
                del model
            except Exception:
                pass
            tf.keras.backend.clear_session()

    return pd.DataFrame(metric_rows), pd.DataFrame(variant_rows)


# ============================================================
# 11) Ranking summaries
# ============================================================
def compute_ranking_per_seed(scores_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (metric, variant, seed), chunk in scores_df.groupby(["metric", "variant", "seed"]):
        chunk = chunk.sort_values("model_index")
        scores = chunk["variant_score"].to_numpy()

        if detect_near_constant_scores(scores):
            print(
                f"[WARNING] Near-constant scores detected | "
                f"metric={metric}, variant={variant}, seed={seed}"
            )

        metrics = compute_ranking_metrics(
            scores=scores,
            targets=chunk["valid_psnr"].to_numpy(),
        )
        rows.append({
            "metric": metric,
            "variant": variant,
            "seed": int(seed),
            **metrics,
        })

    return pd.DataFrame(rows)


def summarize_ranking_over_seeds(ranking_per_seed_df: pd.DataFrame) -> pd.DataFrame:
    stat_cols = ["c_index", "spearman", "kendall_tau_b"]

    rows = []
    for (metric, variant), chunk in ranking_per_seed_df.groupby(["metric", "variant"]):
        row = {
            "metric": metric,
            "variant": variant,
            "n_seed_runs": int(len(chunk)),
            "mean_n_models": float(np.nanmean(chunk["n_models"])),
            "mean_n_pairs": float(np.nanmean(chunk["n_pairs"])),
        }

        for col in stat_cols:
            values = chunk[col].to_numpy()
            mean, std, n = finite_mean_std(values)

            row[f"{col}_mean"] = mean
            row[f"{col}_median"] = finite_median(values)
            row[f"{col}_std"] = std
            row[f"{col}_iqr"] = finite_iqr(values)
            row[f"{col}_n"] = n
            row[f"{col}_ci95"] = ci95_from_std(std, n)

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 12) Runtime summary
# ============================================================
def summarize_runtime_overall(metric_runs_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for metric, chunk in metric_runs_df.groupby("metric"):
        proxy_values = chunk["proxy_time_ms"].to_numpy()
        total_values = chunk["total_eval_like_ms"].to_numpy()
        build_values = chunk["build_time_ms"].to_numpy()

        p_mean, p_std, p_n = finite_mean_std(proxy_values)
        t_mean, t_std, t_n = finite_mean_std(total_values)
        b_mean, b_std, b_n = finite_mean_std(build_values)

        rows.append({
            "metric": metric,

            "proxy_time_mean_ms": p_mean,
            "proxy_time_median_ms": finite_median(proxy_values),
            "proxy_time_std_ms": p_std,
            "proxy_time_iqr_ms": finite_iqr(proxy_values),
            "proxy_time_n": p_n,
            "proxy_time_ci95_ms": ci95_from_std(p_std, p_n),

            "build_time_mean_ms": b_mean,
            "build_time_median_ms": finite_median(build_values),
            "build_time_std_ms": b_std,
            "build_time_iqr_ms": finite_iqr(build_values),
            "build_time_n": b_n,

            "total_eval_like_mean_ms": t_mean,
            "total_eval_like_median_ms": finite_median(total_values),
            "total_eval_like_std_ms": t_std,
            "total_eval_like_iqr_ms": finite_iqr(total_values),
            "total_eval_like_n": t_n,
            "total_eval_like_ci95_ms": ci95_from_std(t_std, t_n),
        })

    return pd.DataFrame(rows)


# ============================================================
# 13) Selection / paper tables
# ============================================================
def select_best_variant_per_metric_from_seed_summary(
    ranking_summary_df: pd.DataFrame,
    primary_metric: str = "spearman_mean",
    tie_breakers: List[str] | None = None,
) -> pd.DataFrame:
    if tie_breakers is None:
        tie_breakers = ["kendall_tau_b_mean", "c_index_mean"]

    rows = []
    for metric, chunk in ranking_summary_df.groupby("metric"):
        sort_cols = [primary_metric] + tie_breakers
        ascending = [False] * len(sort_cols)
        best = chunk.sort_values(sort_cols, ascending=ascending).iloc[0].to_dict()
        rows.append(best)

    return pd.DataFrame(rows)


def round_numeric_df(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(decimals)
    return out


def save_df_csv_and_tex(df: pd.DataFrame, csv_path: Path, tex_path: Path, decimals: int = 4) -> None:
    rounded = round_numeric_df(df, decimals=decimals)
    rounded.to_csv(csv_path, index=False)
    rounded.to_latex(tex_path, index=False, escape=False)


# ============================================================
# 14) Main (paper-only outputs)
# ============================================================
def main():
    benchmark_csv = "./data/20_full_trained_models.csv"

    output_root = Path("./data/zero_cost_benchmark_outputs_paper_only")
    raw_runs_dir = output_root / "raw_runs"
    paper_tables_dir = output_root / "tables_for_paper"

    raw_runs_dir.mkdir(parents=True, exist_ok=True)
    paper_tables_dir.mkdir(parents=True, exist_ok=True)

    CPU_MODE = False
    BASE_SEED = 1
    N_SEEDS = 30
    INPUT_SEED = 12345

    UPSCALE_FACTOR = 2
    LR_PATCH_SIZE = 64
    BATCH_SIZE = 4

    configure_runtime(cpu_mode=CPU_MODE)

    print("TF:", tf.__version__)
    print("tf.keras:", tf.keras.__version__)
    print("GPUs:", len(tf.config.list_logical_devices("GPU")))

    benchmark_df = load_benchmark_models_csv(
        benchmark_csv,
        expected_len=28,
    )
    print("[BENCHMARK] total models:", len(benchmark_df))

    set_seed(INPUT_SEED)
    fixed_lr_batch, fixed_hr_batch = make_dummy_sr_batch(
        batch_size=BATCH_SIZE,
        lr_patch_size=LR_PATCH_SIZE,
        ratio=UPSCALE_FACTOR,
        dtype=tf.float32,
    )

    zero_cost_fns = load_zero_cost_functions()

    # ======================================================
    # FULL comparison: raw, div_params, neg_raw, neg_div_params
    # ======================================================
    score_variants = get_score_variants(include_negative_variants=True)

    loss_fn = get_sr_loss()

    print("[ZERO-COST]", list(zero_cost_fns.keys()))
    print("[VARIANTS]", list(score_variants.keys()))

    metric_runs_df, variant_runs_df = evaluate_zero_cost_scores_paper(
        benchmark_df=benchmark_df,
        zero_cost_fns=zero_cost_fns,
        score_variants=score_variants,
        loss_fn=loss_fn,
        n_seeds=N_SEEDS,
        base_seed=BASE_SEED,
        fixed_lr_batch=fixed_lr_batch,
        fixed_hr_batch=fixed_hr_batch,
        upscale_factor=UPSCALE_FACTOR,
    )

    # Raw long files
    metric_runs_df.to_csv(raw_runs_dir / "metric_runs_long.csv", index=False)
    variant_runs_df.to_csv(raw_runs_dir / "variant_runs_long.csv", index=False)

    ranking_per_seed_df = compute_ranking_per_seed(variant_runs_df)
    ranking_summary_df = summarize_ranking_over_seeds(ranking_per_seed_df)
    runtime_overall_df = summarize_runtime_overall(metric_runs_df)

    # ======================================================
    # TABLE 1: full ranking summary across ALL variants
    # ======================================================
    ranking_summary_for_paper = ranking_summary_df[
        [
            "metric", "variant",
            "c_index_mean", "c_index_std",
            "spearman_mean", "spearman_std",
            "kendall_tau_b_mean", "kendall_tau_b_std",
        ]
    ].copy()

    ranking_summary_for_paper = ranking_summary_for_paper.sort_values(
        by=["c_index_mean", "kendall_tau_b_mean", "spearman_mean"],
        ascending=False,
    ).reset_index(drop=True)

    # ======================================================
    # TABLE 2: runtime summary
    # runtime stays metric-level, not variant-level
    # ======================================================
    runtime_summary_for_paper = runtime_overall_df[
        [
            "metric",
            "proxy_time_mean_ms",
            "proxy_time_median_ms",
            "proxy_time_std_ms",
            "proxy_time_iqr_ms",
            "total_eval_like_mean_ms",
            "total_eval_like_median_ms",
        ]
    ].copy()

    runtime_summary_for_paper = runtime_summary_for_paper.sort_values(
        by="proxy_time_median_ms",
        ascending=True,
    ).reset_index(drop=True)

    # ======================================================
    # TABLE 3: best variant per metric
    # IMPORTANT: now selected by C-index, not Spearman
    # ======================================================
    best_variant_df = select_best_variant_per_metric_from_seed_summary(
        ranking_summary_df=ranking_summary_df,
        primary_metric="c_index_mean",
        tie_breakers=["kendall_tau_b_mean", "spearman_mean"],
    )

    best_variant_runtime_for_paper = best_variant_df.merge(
        runtime_overall_df[
            [
                "metric",
                "proxy_time_mean_ms",
                "proxy_time_median_ms",
                "proxy_time_std_ms",
                "proxy_time_iqr_ms",
                "total_eval_like_mean_ms",
                "total_eval_like_median_ms",
            ]
        ],
        on="metric",
        how="left",
    )[
        [
            "metric", "variant",
            "c_index_mean", "c_index_std",
            "spearman_mean", "spearman_std",
            "kendall_tau_b_mean", "kendall_tau_b_std",
            "proxy_time_mean_ms",
            "proxy_time_median_ms",
            "proxy_time_std_ms",
            "proxy_time_iqr_ms",
            "total_eval_like_mean_ms",
            "total_eval_like_median_ms",
        ]
    ].copy()

    best_variant_runtime_for_paper = best_variant_runtime_for_paper.sort_values(
        by=["c_index_mean", "kendall_tau_b_mean", "spearman_mean"],
        ascending=False,
    ).reset_index(drop=True)

    save_df_csv_and_tex(
        ranking_summary_for_paper,
        paper_tables_dir / "ranking_summary_for_paper_all_variants.csv",
        paper_tables_dir / "ranking_summary_for_paper_all_variants.tex",
        decimals=4,
    )

    save_df_csv_and_tex(
        runtime_summary_for_paper,
        paper_tables_dir / "runtime_summary_for_paper.csv",
        paper_tables_dir / "runtime_summary_for_paper.tex",
        decimals=4,
    )

    save_df_csv_and_tex(
        best_variant_runtime_for_paper,
        paper_tables_dir / "best_variant_runtime_for_paper_by_cindex.csv",
        paper_tables_dir / "best_variant_runtime_for_paper_by_cindex.tex",
        decimals=4,
    )

    print("\n================ DONE ================")
    print("Saved:", raw_runs_dir / "metric_runs_long.csv")
    print("Saved:", raw_runs_dir / "variant_runs_long.csv")
    print("Saved:", paper_tables_dir / "ranking_summary_for_paper_all_variants.csv")
    print("Saved:", paper_tables_dir / "ranking_summary_for_paper_all_variants.tex")
    print("Saved:", paper_tables_dir / "runtime_summary_for_paper.csv")
    print("Saved:", paper_tables_dir / "runtime_summary_for_paper.tex")
    print("Saved:", paper_tables_dir / "best_variant_runtime_for_paper_by_cindex.csv")
    print("Saved:", paper_tables_dir / "best_variant_runtime_for_paper_by_cindex.tex")

    print("\n=== Ranking summary for paper (all variants) ===")
    print(ranking_summary_for_paper.to_string(index=False))

    print("\n=== Runtime summary for paper ===")
    print(runtime_summary_for_paper.to_string(index=False))

    print("\n=== Best variant + runtime for paper (selected by C-index) ===")
    print(best_variant_runtime_for_paper.to_string(index=False))


if __name__ == "__main__":
    main()
