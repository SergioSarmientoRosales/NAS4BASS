from __future__ import annotations

import argparse
import csv
import json
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
    load_reference_genes,
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


def pca_2d(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = features - features.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    variance = singular_values**2
    explained = variance[:2] / variance.sum() if variance.sum() > 0 else np.zeros(2)
    return coords, explained


def scatter_panel(ax, coords, pool_slice, sample_slice, ref_slice, explained, *, pool_downsampled: bool):
    pool_label = "Random BASS pool"
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
        coords[ref_slice, 0],
        coords[ref_slice, 1],
        s=58,
        marker="x",
        c="#f97316",
        linewidths=1.6,
        label="20 Pareto-front set",
    )
    ax.scatter(
        coords[sample_slice, 0],
        coords[sample_slice, 1],
        s=38,
        c="#2563eb",
        edgecolors="white",
        linewidths=0.5,
        label="Selected BASS sample",
    )
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var.)")
    ax.set_title("Architecture-space coverage")
    ax.grid(True, color="#e5e7eb", linewidth=0.7)
    ax.legend(frameon=False, loc="best")


def descriptor_panel(ax, sample_genes, reference_genes):
    sample_desc = [architecture_descriptors(gene) for gene in sample_genes]
    ref_desc = [architecture_descriptors(gene) for gene in reference_genes]

    ax.scatter(
        [desc["log_estimated_complexity"] for desc in ref_desc],
        [desc["identity_count"] for desc in ref_desc],
        s=58,
        marker="x",
        c="#f97316",
        linewidths=1.6,
        label="20 Pareto-front set",
    )
    ax.scatter(
        [desc["log_estimated_complexity"] for desc in sample_desc],
        [desc["identity_count"] for desc in sample_desc],
        s=38,
        c="#2563eb",
        edgecolors="white",
        linewidths=0.5,
        label="Selected BASS sample",
    )
    ax.set_xlabel("log estimated structural complexity")
    ax.set_ylabel("identity operation count")
    ax.set_title("Simple descriptor view")
    ax.grid(True, color="#e5e7eb", linewidth=0.7)


def complexity_distribution_panel(ax, pool_genes, sample_genes, reference_genes):
    pool_values = [
        architecture_descriptors(gene)["log_estimated_complexity"] for gene in pool_genes
    ]
    sample_values = [
        architecture_descriptors(gene)["log_estimated_complexity"] for gene in sample_genes
    ]
    reference_values = [
        architecture_descriptors(gene)["log_estimated_complexity"] for gene in reference_genes
    ]

    ax.hist(
        pool_values,
        bins=32,
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
    ax.scatter(
        reference_values,
        np.zeros(len(reference_values)),
        marker="x",
        c="#f97316",
        linewidths=1.4,
        s=44,
        label="20 Pareto-front set",
    )
    ax.set_xlabel("log estimated structural complexity")
    ax.set_ylabel("density")
    ax.set_title("Complexity distribution")
    ax.grid(True, color="#e5e7eb", linewidth=0.7)
    ax.legend(frameon=False, loc="best")


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
    parser.add_argument("--reference-csv", default="data/20_full_trained_models.csv")
    parser.add_argument("--metadata-json", default="data/architectures/bass_50_sample_metadata.json")
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pool-policy", choices=["mixed", "uniform"], default=None)
    parser.add_argument("--max-pool-points", type=int, default=15000)
    parser.add_argument("--output-dir", default="figures/zerocost_50_bass_sample")
    parser.add_argument("--prefix", default="bass_50_sample")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = {}
    metadata_path = Path(args.metadata_json)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    pool_size = args.pool_size or int(metadata.get("pool_size_requested", 10000))
    seed = args.seed or int(metadata.get("seed", 20260703))
    pool_policy = args.pool_policy or str(metadata.get("pool_policy", "uniform"))

    sample_genes = load_genes_from_csv(args.sample_csv)
    reference_genes = sorted(load_reference_genes([args.reference_csv]))
    pool_genes = generate_unique_pool(
        pool_size=pool_size,
        seed=seed,
        pool_policy=pool_policy,
    )
    plotted_pool_genes = pool_genes
    if args.max_pool_points and len(pool_genes) > args.max_pool_points:
        rng = np.random.default_rng(seed + 17)
        selected_plot_indices = np.sort(
            rng.choice(len(pool_genes), size=args.max_pool_points, replace=False)
        )
        plotted_pool_genes = [pool_genes[int(idx)] for idx in selected_plot_indices]

    all_genes = plotted_pool_genes + sample_genes + reference_genes
    labels = (
        ["pool"] * len(plotted_pool_genes)
        + ["selected_sample"] * len(sample_genes)
        + ["pareto20"] * len(reference_genes)
    )
    features = descriptor_matrix(all_genes)
    coords, explained = pca_2d(features)

    pool_slice = slice(0, len(plotted_pool_genes))
    sample_slice = slice(len(plotted_pool_genes), len(plotted_pool_genes) + len(sample_genes))
    ref_slice = slice(len(plotted_pool_genes) + len(sample_genes), len(all_genes))

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)
    scatter_panel(
        axes[0],
        coords,
        pool_slice,
        sample_slice,
        ref_slice,
        explained,
        pool_downsampled=len(plotted_pool_genes) < len(pool_genes),
    )
    descriptor_panel(axes[1], sample_genes, reference_genes)
    complexity_distribution_panel(axes[2], plotted_pool_genes, sample_genes, reference_genes)
    fig.suptitle("BASS 50-Architecture Sample Visualization", fontsize=11)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{args.prefix}_architecture_space.png"
    pdf_path = png_path.with_suffix(".pdf")
    coords_path = output_dir / f"{args.prefix}_architecture_space_coordinates.csv"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    save_coords_csv(coords_path, coords, labels, all_genes)
    print(f"[PLOT] plotted pool points={len(plotted_pool_genes)} of {len(pool_genes)}")
    print(f"[PLOT] saved {png_path}")
    print(f"[PLOT] saved {pdf_path}")
    print(f"[PLOT] saved {coords_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
