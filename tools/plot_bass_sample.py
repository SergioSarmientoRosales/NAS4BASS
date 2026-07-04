from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.sample_bass_architectures import (  # noqa: E402
    architecture_descriptors,
    descriptor_matrix,
    generate_unique_pool,
    parse_gene,
)


def load_genes_from_csv(path: str | Path) -> list[tuple[int, ...]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        gene_col = next(
            (col for col in ("Net", "gene", "Gene", "genotype", "architecture", "arch") if col in (reader.fieldnames or [])),
            None,
        )
        if gene_col is None:
            raise ValueError(f"No architecture column found in {path}")
        return [parse_gene(row[gene_col]) for row in reader]


def load_pool(metadata: dict, *, pool_size: int | None, seed: int | None, pool_policy: str | None) -> list[tuple[int, ...]]:
    cache_path = Path(str(metadata.get("pool_cache", "")))
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        return [tuple(int(value) for value in row) for row in data["pool"]]

    resolved_pool_size = pool_size or int(metadata.get("pool_size_requested", 100000))
    resolved_seed = seed or int(metadata.get("seed", 20260703))
    resolved_policy = pool_policy or str(metadata.get("pool_policy", "uniform"))
    return generate_unique_pool(
        pool_size=resolved_pool_size,
        seed=resolved_seed,
        pool_policy=resolved_policy,
    )


def pca_2d(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = features - features.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    variance = singular_values**2
    explained = variance[:2] / variance.sum() if variance.sum() > 0 else np.zeros(2)
    return coords, explained


def downsample_pool(pool_genes: list[tuple[int, ...]], *, max_points: int, seed: int) -> list[tuple[int, ...]]:
    if not max_points or len(pool_genes) <= max_points:
        return pool_genes
    rng = np.random.default_rng(seed + 17)
    indices = np.sort(rng.choice(len(pool_genes), size=max_points, replace=False))
    return [pool_genes[int(idx)] for idx in indices]


def complexity_values(genes: list[tuple[int, ...]]) -> list[float]:
    return [architecture_descriptors(gene)["log_estimated_complexity"] for gene in genes]


def finite_edges(metadata: dict) -> list[float]:
    edges = []
    for raw in metadata.get("complexity_edges", []):
        if isinstance(raw, str) and raw.lower() == "inf":
            continue
        value = float(raw)
        if math.isfinite(value):
            edges.append(value)
    return edges


def scatter_panel(ax, coords, pool_slice, sample_slice, explained, *, pool_downsampled: bool):
    pool_label = "Uniform BASS pool"
    if pool_downsampled:
        pool_label += " (downsampled)"
    ax.scatter(
        coords[pool_slice, 0],
        coords[pool_slice, 1],
        s=6,
        c="#9ca3af",
        alpha=0.22,
        linewidths=0,
        label=pool_label,
    )
    ax.scatter(
        coords[sample_slice, 0],
        coords[sample_slice, 1],
        s=42,
        c="#2563eb",
        edgecolors="white",
        linewidths=0.5,
        label="Official stratified-random sample",
    )
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var.)")
    ax.set_title("Architecture-space coverage")
    ax.grid(True, color="#e5e7eb", linewidth=0.7)
    ax.legend(frameon=False, loc="best")


def descriptor_panel(ax, sample_genes):
    sample_desc = [architecture_descriptors(gene) for gene in sample_genes]
    ax.scatter(
        [desc["log_estimated_complexity"] for desc in sample_desc],
        [desc["identity_count"] for desc in sample_desc],
        s=42,
        c="#2563eb",
        edgecolors="white",
        linewidths=0.5,
    )
    ax.set_xlabel("log estimated structural complexity")
    ax.set_ylabel("identity operation count")
    ax.set_title("Selected-sample descriptors")
    ax.grid(True, color="#e5e7eb", linewidth=0.7)


def complexity_panel(ax, pool_genes, sample_genes, edges: list[float]):
    pool_values = complexity_values(pool_genes)
    sample_values = complexity_values(sample_genes)
    ax.hist(
        pool_values,
        bins=42,
        density=True,
        color="#9ca3af",
        alpha=0.35,
        linewidth=0,
        label="Pool density",
    )
    ax.hist(
        sample_values,
        bins=10,
        density=True,
        color="#2563eb",
        alpha=0.62,
        linewidth=0,
        label="Selected density",
    )
    for edge in edges[1:]:
        ax.axvline(edge, color="#111827", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel("log estimated structural complexity")
    ax.set_ylabel("density")
    ax.set_title("Complexity distribution and band edges")
    ax.grid(True, color="#e5e7eb", linewidth=0.7)
    ax.legend(frameon=False, loc="best")


def comparison_plot(output_dir: Path, prefix: str, pool_genes, sample_genes, comparison_genes, *, seed: int, dpi: int) -> None:
    plotted_pool = downsample_pool(pool_genes, max_points=15000, seed=seed)
    all_genes = plotted_pool + sample_genes + comparison_genes
    features = descriptor_matrix(all_genes)
    coords, explained = pca_2d(features)
    pool_slice = slice(0, len(plotted_pool))
    sample_slice = slice(len(plotted_pool), len(plotted_pool) + len(sample_genes))
    comparison_slice = slice(len(plotted_pool) + len(sample_genes), len(all_genes))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    ax.scatter(coords[pool_slice, 0], coords[pool_slice, 1], s=6, c="#9ca3af", alpha=0.18, linewidths=0, label="Uniform pool")
    ax.scatter(coords[sample_slice, 0], coords[sample_slice, 1], s=38, c="#2563eb", edgecolors="white", linewidths=0.5, label="Stratified random")
    ax.scatter(coords[comparison_slice, 0], coords[comparison_slice, 1], s=44, marker="^", c="#dc2626", edgecolors="white", linewidths=0.5, label="Stratified max-min")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var.)")
    ax.set_title("Supplementary coverage comparison")
    ax.grid(True, color="#e5e7eb", linewidth=0.7)
    ax.legend(frameon=False)
    png_path = output_dir / f"{prefix}_random_vs_maxmin.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] saved {png_path}")
    print(f"[PLOT] saved {png_path.with_suffix('.pdf')}")


def save_coords_csv(path: Path, coords: np.ndarray, labels: list[str], genes: list[tuple[int, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "pc1", "pc2", "Net"])
        for label, coord, gene in zip(labels, coords, genes):
            writer.writerow([label, float(coord[0]), float(coord[1]), str(list(gene))])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a BASS architecture sample.")
    parser.add_argument("--sample-csv", default="data/architectures/bass_50_sample_architectures.csv")
    parser.add_argument("--metadata-json", default="data/architectures/bass_50_sample_metadata.json")
    parser.add_argument("--comparison-csv", default=None)
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pool-policy", choices=["mixed", "uniform"], default=None)
    parser.add_argument("--max-pool-points", type=int, default=15000)
    parser.add_argument("--output-dir", default="figures/zerocost_50_stratified_random")
    parser.add_argument("--prefix", default="bass_50_sample")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = {}
    metadata_path = Path(args.metadata_json)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    seed = args.seed or int(metadata.get("seed", 20260703))
    sample_genes = load_genes_from_csv(args.sample_csv)
    pool_genes = load_pool(
        metadata,
        pool_size=args.pool_size,
        seed=seed,
        pool_policy=args.pool_policy,
    )
    plotted_pool = downsample_pool(pool_genes, max_points=args.max_pool_points, seed=seed)
    all_genes = plotted_pool + sample_genes
    labels = ["pool"] * len(plotted_pool) + ["selected_sample"] * len(sample_genes)
    features = descriptor_matrix(all_genes)
    coords, explained = pca_2d(features)

    pool_slice = slice(0, len(plotted_pool))
    sample_slice = slice(len(plotted_pool), len(all_genes))

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)
    scatter_panel(
        axes[0],
        coords,
        pool_slice,
        sample_slice,
        explained,
        pool_downsampled=len(plotted_pool) < len(pool_genes),
    )
    descriptor_panel(axes[1], sample_genes)
    complexity_panel(axes[2], plotted_pool, sample_genes, finite_edges(metadata))
    fig.suptitle("BASS 50-Architecture Stratified-Random Sample", fontsize=11)

    png_path = output_dir / f"{args.prefix}_architecture_space.png"
    pdf_path = png_path.with_suffix(".pdf")
    coords_path = output_dir / f"{args.prefix}_architecture_space_coordinates.csv"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    save_coords_csv(coords_path, coords, labels, all_genes)

    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    complexity_panel(ax, pool_genes, sample_genes, finite_edges(metadata))
    complexity_path = output_dir / f"{args.prefix}_complexity_distribution.png"
    fig.savefig(complexity_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(complexity_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    if args.comparison_csv:
        comparison_plot(
            output_dir,
            args.prefix,
            pool_genes,
            sample_genes,
            load_genes_from_csv(args.comparison_csv),
            seed=seed,
            dpi=args.dpi,
        )

    print(f"[PLOT] plotted pool points={len(plotted_pool)} of {len(pool_genes)}")
    print(f"[PLOT] saved {png_path}")
    print(f"[PLOT] saved {pdf_path}")
    print(f"[PLOT] saved {coords_path}")
    print(f"[PLOT] saved {complexity_path}")
    print(f"[PLOT] saved {complexity_path.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
