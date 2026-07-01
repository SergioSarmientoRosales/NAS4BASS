from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================
ROOT_CANDIDATES = [
    Path("./data/zero_cost_benchmark_outputs_paper_only"),
    Path("./data/zero_cost_benchmark_outputs"),
]

SAVE_PDF = True
SAVE_TEX = True
DPI = 400

KEEP_VARIANTS = {"raw", "div_params", "neg_raw", "neg_div_params"}

# If True, draw leader lines for the crowded cluster
USE_ARROWS_FOR_CROWDED_LABELS = True

# Optional proxy pretty names
LABEL_MAP = {
    "param_score": "ParamScore",
    "l2_norm": "L2Norm",
    "jacob": "Jacob",
    "grad_norm": "GradNorm",
    "fisher": "Fisher",
    "grasp": "GraSP",
    "snip": "SNIP",
    "nwot": "NWOT",
    "synflow": "SynFlow",
    "plain": "Plain",
    "zen": "ZEN",
    "zico": "ZiCo",
}

# Manual label offsets in points for the scatter
ANNOT_OFFSETS = {
    "ParamScore": (-15, 8),
    "L2Norm": (6, -4),
    "NWOT": (6, 8),
    "ZEN": (6, 2),
    "Jacob_cov": (6, 4),
    "ZiCo": (6, 6),
    "SynFlow": (-35,2),
    "Plain": (-25, -6),
    "GradNorm": (8, -2),
    "SNIP": (6, -6),
    "Fisher": (8, 2),
    "GraSP": (8, -6),
}

ARROW_PROXIES = {"SynFlow", "Plain", "GradNorm", "SNIP"}


# ============================================================
# STYLE
# ============================================================
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
})


# ============================================================
# HELPERS
# ============================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def round_numeric_df(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(decimals)
    return out


def save_csv_and_tex(df: pd.DataFrame, csv_path: Path, tex_path: Path, decimals: int = 4) -> None:
    out = round_numeric_df(df, decimals=decimals)
    out.to_csv(csv_path, index=False)

    if not SAVE_TEX:
        return

    try:
        latex_str = out.to_latex(index=False, escape=False)
        tex_path.write_text(latex_str, encoding="utf-8")
    except Exception as e:
        print(f"[WARNING] Could not save LaTeX table {tex_path.name}: {e}")


def save_fig(fig: plt.Figure, png_path: Path) -> None:
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")


def pretty_metric_name(metric: str) -> str:
    return LABEL_MAP.get(metric, metric)


def concordance_index(scores: np.ndarray, targets: np.ndarray, tie_score: float = 0.5) -> tuple[float, int]:
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


# ============================================================
# PATH RESOLUTION
# ============================================================
def resolve_root() -> Path:
    for root in ROOT_CANDIDATES:
        if root.exists():
            return root
    raise FileNotFoundError(
        "Could not find any benchmark output root. Tried:\n"
        + "\n".join(str(p) for p in ROOT_CANDIDATES)
    )


def first_existing(paths: list[Path], description: str) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {description}. Tried:\n" + "\n".join(str(p) for p in paths)
    )


def resolve_paths():
    root = resolve_root()
    raw_runs_dir = root / "raw_runs"
    tables_dir = root / "tables_for_paper"
    output_dir = root / "paper_main_clean"

    variant_runs_csv = first_existing(
        [
            raw_runs_dir / "variant_runs_long.csv",
        ],
        "variant runs CSV",
    )

    ranking_summary_csv = first_existing(
        [
            tables_dir / "ranking_summary_for_paper_all_variants.csv",
            tables_dir / "ranking_summary_for_paper.csv",
        ],
        "ranking summary CSV",
    )

    runtime_summary_csv = first_existing(
        [
            tables_dir / "runtime_summary_for_paper.csv",
        ],
        "runtime summary CSV",
    )

    return root, raw_runs_dir, tables_dir, output_dir, variant_runs_csv, ranking_summary_csv, runtime_summary_csv


# ============================================================
# LOAD
# ============================================================
def load_inputs(variant_runs_csv: Path, ranking_summary_csv: Path, runtime_summary_csv: Path):
    print("[INFO] Loading files...")
    print("  VARIANT_RUNS_CSV :", variant_runs_csv.resolve())
    print("  RANKING_SUMMARY  :", ranking_summary_csv.resolve())
    print("  RUNTIME_SUMMARY  :", runtime_summary_csv.resolve())

    variant_runs = pd.read_csv(variant_runs_csv)
    ranking_summary = pd.read_csv(ranking_summary_csv)
    runtime_summary = pd.read_csv(runtime_summary_csv)

    ranking_summary = ranking_summary[ranking_summary["variant"].isin(KEEP_VARIANTS)].copy()
    variant_runs = variant_runs[variant_runs["variant"].isin(KEEP_VARIANTS)].copy()

    return variant_runs, ranking_summary, runtime_summary


# ============================================================
# TABLE 1: ALL PROXY × VARIANT RESULTS
# ============================================================
def build_introductory_table(
    ranking_summary: pd.DataFrame,
    runtime_summary: pd.DataFrame,
) -> pd.DataFrame:
    intro = ranking_summary.merge(runtime_summary, on="metric", how="left")
    intro["proxy"] = intro["metric"].map(pretty_metric_name)

    intro = intro[
        [
            "proxy",
            "metric",
            "variant",
            "c_index_mean",
            "c_index_std",
            "spearman_mean",
            "spearman_std",
            "kendall_tau_b_mean",
            "kendall_tau_b_std",
            "proxy_time_mean_ms",
            "proxy_time_median_ms",
            "proxy_time_std_ms",
            "total_eval_like_mean_ms",
            "total_eval_like_median_ms",
        ]
    ].copy()

    intro = intro.sort_values(
        by=["c_index_mean", "kendall_tau_b_mean", "spearman_mean"],
        ascending=False,
    ).reset_index(drop=True)

    return intro


# ============================================================
# BEST VARIANT BY C-INDEX
# ============================================================
def select_best_variant_by_cindex(ranking_summary: pd.DataFrame) -> pd.DataFrame:
    best = (
        ranking_summary
        .sort_values(
            by=["metric", "c_index_mean", "kendall_tau_b_mean", "spearman_mean"],
            ascending=[True, False, False, False],
        )
        .groupby("metric", as_index=False)
        .first()
        .copy()
    )

    best["proxy"] = best["metric"].map(pretty_metric_name)
    return best


def build_best_variant_runtime_table(best_df: pd.DataFrame, runtime_summary: pd.DataFrame) -> pd.DataFrame:
    merged = best_df.merge(runtime_summary, on="metric", how="left")

    merged = merged[
        [
            "proxy",
            "metric",
            "variant",
            "c_index_mean",
            "c_index_std",
            "spearman_mean",
            "spearman_std",
            "kendall_tau_b_mean",
            "kendall_tau_b_std",
            "proxy_time_mean_ms",
            "proxy_time_median_ms",
            "proxy_time_std_ms",
            "total_eval_like_mean_ms",
            "total_eval_like_median_ms",
        ]
    ].copy()

    merged = merged.sort_values("c_index_mean", ascending=False).reset_index(drop=True)
    return merged


# ============================================================
# SEED-WISE C-INDEX FOR BOXPLOT
# ============================================================
def compute_seedwise_cindex(variant_runs: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (metric, variant, seed), chunk in variant_runs.groupby(["metric", "variant", "seed"]):
        chunk = chunk.sort_values("model_index")

        c_idx, n_pairs = concordance_index(
            scores=chunk["variant_score"].to_numpy(dtype=float),
            targets=chunk["valid_psnr"].to_numpy(dtype=float),
            tie_score=0.5,
        )

        rows.append({
            "metric": metric,
            "variant": variant,
            "seed": int(seed),
            "c_index": c_idx,
            "n_pairs": n_pairs,
        })

    out = pd.DataFrame(rows)
    out["proxy"] = out["metric"].map(pretty_metric_name)
    return out


def filter_seedwise_to_best_variants(seedwise_df: pd.DataFrame, best_df: pd.DataFrame) -> pd.DataFrame:
    best_map = dict(zip(best_df["metric"], best_df["variant"]))

    seed_best = seedwise_df[
        seedwise_df.apply(
            lambda row: best_map.get(row["metric"]) == row["variant"],
            axis=1,
        )
    ].copy()

    seed_best["proxy"] = seed_best["metric"].map(pretty_metric_name)
    return seed_best


# ============================================================
# MAIN FIGURE
# ============================================================
# ============================================================
# MAIN FIGURE
# ============================================================
def plot_main_subplot(best_runtime_df: pd.DataFrame, seed_best_df: pd.DataFrame) -> plt.Figure:
    order_proxies = best_runtime_df.sort_values("c_index_mean", ascending=False)["proxy"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    ax_left, ax_right = axes

    # Font sizes for paper readability
    LABEL_FONTSIZE = 18
    TICK_FONTSIZE = 16
    ANNOT_FONTSIZE = 9.5

    # --------------------------------
    # A) Runtime vs C-index
    # --------------------------------
    x = best_runtime_df["proxy_time_mean_ms"].astype(float).values
    y = best_runtime_df["c_index_mean"].astype(float).values

    ax_left.scatter(
        x,
        y,
        color="black",
        s=42,
        zorder=3,
    )

    for _, row in best_runtime_df.iterrows():
        proxy = row["proxy"]
        dx, dy = ANNOT_OFFSETS.get(proxy, (5, 5))

        arrowprops = None
        if USE_ARROWS_FOR_CROWDED_LABELS and proxy in ARROW_PROXIES:
            arrowprops = dict(arrowstyle="-", lw=0.65, color="gray")

        ax_left.annotate(
            proxy,
            xy=(float(row["proxy_time_mean_ms"]), float(row["c_index_mean"])),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=ANNOT_FONTSIZE,
            color="black",
            arrowprops=arrowprops,
        )

    ax_left.set_xlabel("Mean proxy evaluation time (ms)", fontsize=LABEL_FONTSIZE)
    ax_left.set_ylabel("Mean C-index", fontsize=LABEL_FONTSIZE)

    ax_left.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    ax_left.set_ylim(0.6, 1.0)
    ax_left.grid(True, alpha=0.25, linewidth=0.6)
    ax_left.set_axisbelow(True)

    # --------------------------------
    # B) Boxplot of seed-wise C-index
    # --------------------------------
    data = []
    labels = []
    for proxy in order_proxies:
        vals = (
            seed_best_df.loc[seed_best_df["proxy"] == proxy, "c_index"]
            .dropna()
            .astype(float)
            .values
        )
        if len(vals) > 0:
            data.append(vals)
            labels.append(proxy)

    ax_right.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.6,
        medianprops=dict(color="black", linewidth=1.0),
        whiskerprops=dict(color="black", linewidth=0.9),
        capprops=dict(color="black", linewidth=0.9),
        boxprops=dict(facecolor="#d9d9d9", edgecolor="black", linewidth=0.9),
        flierprops=dict(
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=3.5,
            linestyle="none",
        ),
    )

    ax_right.set_ylabel("C-index", fontsize=LABEL_FONTSIZE)
    ax_right.tick_params(axis="y", labelsize=TICK_FONTSIZE)

    ax_right.set_ylim(0.6, 1.0)
    ax_right.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax_right.set_axisbelow(True)

    for tick in ax_right.get_xticklabels():
        tick.set_rotation(45)
        tick.set_horizontalalignment("right")
        tick.set_fontsize(TICK_FONTSIZE)

    fig.tight_layout()
    return fig

# ============================================================
# MAIN
# ============================================================
def main():
    root, raw_runs_dir, tables_dir, output_dir, variant_runs_csv, ranking_summary_csv, runtime_summary_csv = resolve_paths()
    ensure_dir(output_dir)

    variant_runs, ranking_summary, runtime_summary = load_inputs(
        variant_runs_csv=variant_runs_csv,
        ranking_summary_csv=ranking_summary_csv,
        runtime_summary_csv=runtime_summary_csv,
    )

    # 1) Introductory table with all variants
    intro_table = build_introductory_table(ranking_summary, runtime_summary)
    save_csv_and_tex(
        intro_table,
        output_dir / "table_introductory_all_proxy_variant_results.csv",
        output_dir / "table_introductory_all_proxy_variant_results.tex",
        decimals=4,
    )

    # 2) Best variant per proxy by C-index
    best_df = select_best_variant_by_cindex(ranking_summary)
    best_runtime_df = build_best_variant_runtime_table(best_df, runtime_summary)

    save_csv_and_tex(
        best_runtime_df,
        output_dir / "table_best_variant_per_proxy_by_cindex.csv",
        output_dir / "table_best_variant_per_proxy_by_cindex.tex",
        decimals=4,
    )

    # 3) Seed-wise C-index for boxplot
    seedwise_df = compute_seedwise_cindex(variant_runs)
    seed_best_df = filter_seedwise_to_best_variants(seedwise_df, best_df)
    seed_best_df.to_csv(output_dir / "seedwise_cindex_best_variants.csv", index=False)

    # 4) Main figure
    fig_main = plot_main_subplot(best_runtime_df, seed_best_df)
    save_fig(fig_main, output_dir / "figure_main_subplot_runtime_vs_cindex_boxplot.png")
    plt.close(fig_main)

    print("\nDone.")
    print("ROOT:", root.resolve())
    print("Saved:")
    print(output_dir / "table_introductory_all_proxy_variant_results.csv")
    print(output_dir / "table_best_variant_per_proxy_by_cindex.csv")
    print(output_dir / "seedwise_cindex_best_variants.csv")
    print(output_dir / "figure_main_subplot_runtime_vs_cindex_boxplot.png")


if __name__ == "__main__":
    main()