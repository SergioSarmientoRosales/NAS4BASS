from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import CHANNELS, KERNELS, PRIMITIVES, REPEAT


DEFAULT_COMPLEXITY_EDGES = "0,6,8,10,12,inf"
SAMPLE_OUTPUT_FIELDS = [
    "sample_id",
    "selection_rank",
    "seed",
    "sampling_method",
    "Net",
    "bass_gene_file",
    "overlaps_reference",
    "selection_pool_index",
    "selection_min_distance",
    "selection_min_distance_raw",
    "complexity_bin",
    "params_real",
    "channel_idx",
    "channels",
    "identity_count",
    "trainable_unit_count",
    "estimated_complexity",
    "log_estimated_complexity",
    "op_counts_json",
] + [f"gene_{idx}" for idx in range(28)]


def parse_gene(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        text = value.strip()
        parsed = ast.literal_eval(text) if text.startswith(("[", "(")) else text.split(",")
        gene = tuple(int(item) for item in parsed)
    else:
        gene = tuple(int(item) for item in value)

    if len(gene) != 28:
        raise ValueError(f"Expected 28 genes, got {len(gene)}")
    bad_values = sorted({item for item in gene if item < 0 or item > 7})
    if bad_values:
        raise ValueError(f"Gene values must be in [0, 7], got {bad_values}")
    return gene


def weighted_choice(rng: np.random.Generator, values: list[int], weights: list[float]) -> int:
    weights_arr = np.asarray(weights, dtype=np.float64)
    weights_arr = weights_arr / weights_arr.sum()
    return int(rng.choice(np.asarray(values, dtype=np.int64), p=weights_arr))


def generate_gene(rng: np.random.Generator, *, policy: str) -> tuple[int, ...]:
    if policy == "uniform":
        return tuple(int(item) for item in rng.integers(0, 8, size=28))

    if policy != "mixed":
        raise ValueError(f"Unknown pool policy: {policy}")

    component = float(rng.random())
    if component < 0.45:
        return tuple(int(item) for item in rng.integers(0, 8, size=28))

    if component < 0.75:
        channel_idx = weighted_choice(
            rng,
            list(range(8)),
            [0.20, 0.14, 0.10, 0.06, 0.20, 0.14, 0.10, 0.06],
        )
        op_weights = [0.11, 0.10, 0.09, 0.08, 0.14, 0.10, 0.08, 0.30]
        kernel_weights = [0.22, 0.13, 0.08, 0.04, 0.22, 0.13, 0.08, 0.04]
        repeat_weights = [0.22, 0.13, 0.08, 0.04, 0.22, 0.13, 0.08, 0.04]
    else:
        channel_idx = weighted_choice(
            rng,
            list(range(8)),
            [0.30, 0.08, 0.04, 0.02, 0.30, 0.08, 0.04, 0.02],
        )
        op_weights = [0.08, 0.05, 0.04, 0.03, 0.10, 0.04, 0.03, 0.63]
        kernel_weights = [0.32, 0.08, 0.03, 0.01, 0.32, 0.08, 0.03, 0.01]
        repeat_weights = [0.32, 0.08, 0.03, 0.01, 0.32, 0.08, 0.03, 0.01]

    gene = [channel_idx]
    for _ in range(9):
        gene.append(weighted_choice(rng, list(range(8)), op_weights))
        gene.append(weighted_choice(rng, list(range(8)), kernel_weights))
        gene.append(weighted_choice(rng, list(range(8)), repeat_weights))
    return tuple(gene)


def load_reference_genes(paths: list[str]) -> set[tuple[int, ...]]:
    genes: set[tuple[int, ...]] = set()
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Reference architecture file not found: {path}")

        with path.open(newline="", encoding="utf-8-sig") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            has_header = any(name in sample.splitlines()[0] for name in ("Net", "gene", "architecture", "arch"))
            if has_header:
                reader = csv.DictReader(handle)
                gene_col = next(
                    (col for col in ("Net", "gene", "Gene", "genotype", "architecture", "arch") if col in (reader.fieldnames or [])),
                    None,
                )
                if gene_col is None:
                    raise ValueError(f"No architecture column found in {path}")
                for row in reader:
                    genes.add(parse_gene(row[gene_col]))
            else:
                reader = csv.reader(handle)
                for row in reader:
                    if row and any(cell.strip() for cell in row):
                        genes.add(parse_gene(",".join(row)))
    return genes


def generate_unique_pool(
    *,
    pool_size: int,
    seed: int,
    pool_policy: str = "uniform",
) -> list[tuple[int, ...]]:
    rng = np.random.default_rng(seed)
    pool: dict[tuple[int, ...], None] = {}
    attempts = 0
    max_attempts = max(pool_size * 100, pool_size + 1000)

    while len(pool) < pool_size and attempts < max_attempts:
        gene = generate_gene(rng, policy=pool_policy)
        pool.setdefault(gene, None)
        attempts += 1

    if len(pool) < pool_size:
        raise RuntimeError(
            f"Could only generate {len(pool)} unique architectures after {attempts} attempts"
        )

    return list(pool.keys())


def split_units(gene: tuple[int, ...]) -> tuple[int, list[tuple[int, int, int]]]:
    channel_idx = gene[0]
    rest = gene[1:]
    units = [(rest[i], rest[i + 1], rest[i + 2]) for i in range(0, len(rest), 3)]
    return channel_idx, units


def architecture_descriptors(gene: tuple[int, ...]) -> dict[str, float]:
    channel_idx, units = split_units(gene)
    channels = CHANNELS[channel_idx]
    op_counts = Counter(unit[0] for unit in units)
    kernel_counts = Counter(unit[1] for unit in units)
    repeat_counts = Counter(unit[2] for unit in units)

    complexity = 0.0
    trainable_units = 0
    identity_units = 0
    for op_idx, kernel_idx, repeat_idx in units:
        op_name = PRIMITIVES[op_idx]
        kernel = KERNELS[kernel_idx]
        repeat = REPEAT[repeat_idx]
        if op_name == "identity":
            identity_units += 1
            op_weight = 0.0
        elif op_name == "Dsep_conv":
            op_weight = 0.45
            trainable_units += 1
        elif op_name == "invert_Bot_Conv_E2":
            op_weight = 1.6
            trainable_units += 1
        elif op_name == "conv_transpose":
            op_weight = 1.2
            trainable_units += 1
        else:
            op_weight = 1.0
            trainable_units += 1
        complexity += op_weight * repeat * channels * channels * kernel * kernel

    out: dict[str, float] = {
        "channel_idx": float(channel_idx),
        "channels": float(channels),
        "identity_count": float(identity_units),
        "trainable_unit_count": float(trainable_units),
        "mean_kernel": float(np.mean([KERNELS[u[1]] for u in units])),
        "mean_repeat": float(np.mean([REPEAT[u[2]] for u in units])),
        "estimated_complexity": float(complexity),
        "log_estimated_complexity": float(math.log1p(complexity)),
    }
    for idx in range(8):
        out[f"op_{idx}_count"] = float(op_counts.get(idx, 0))
        out[f"kernel_{idx}_count"] = float(kernel_counts.get(idx, 0))
        out[f"repeat_{idx}_count"] = float(repeat_counts.get(idx, 0))
    return out


def descriptor_matrix(
    genes: list[tuple[int, ...]],
    *,
    standardize: bool = True,
) -> np.ndarray:
    """Return the 26-dimensional BASS composition descriptor matrix.

    Features are op counts / 9 (8), kernel counts / 9 (8), repeat counts / 9
    (8), channels / max(CHANNELS) (1), and log estimated structural complexity
    (1). The raw 28-gene vector is intentionally omitted because operation,
    kernel, and repeat indices are nominal rather than ordinal. This also avoids
    double-weighting counts through both raw genes and aggregate descriptors.
    Positional information about which unit owns which operation is therefore
    deliberately lost; for this benchmark, architecture composition is the
    intended coverage signal.
    """

    rows = []
    for gene in genes:
        desc = architecture_descriptors(gene)
        features = []
        for prefix in ("op", "kernel", "repeat"):
            features.extend(desc[f"{prefix}_{idx}_count"] / 9.0 for idx in range(8))
        features.append(desc["channels"] / max(CHANNELS))
        features.append(desc["log_estimated_complexity"])
        rows.append(features)

    matrix = np.asarray(rows, dtype=np.float64)
    if not standardize or matrix.size == 0:
        return matrix

    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    return (matrix - matrix.mean(axis=0)) / std


def greedy_max_min_select(
    genes: list[tuple[int, ...]],
    *,
    n_select: int,
    features: np.ndarray | None = None,
) -> tuple[list[int], list[float]]:
    """Select a max-min subset and return distances in the provided feature space.

    If ``features`` is supplied, it must already be standardized in the intended
    global space. Otherwise descriptors are computed and standardized on
    ``genes`` only, which is kept for backwards-compatible exploratory use.
    """

    if n_select > len(genes):
        raise ValueError(f"n_select={n_select} exceeds candidate pool size {len(genes)}")

    feature_matrix = np.asarray(features, dtype=np.float64) if features is not None else descriptor_matrix(genes)
    if len(feature_matrix) != len(genes):
        raise ValueError("features length must match genes length")

    centroid = feature_matrix.mean(axis=0)
    first_idx = int(np.argmin(np.linalg.norm(feature_matrix - centroid, axis=1)))

    selected = [first_idx]
    min_dist = np.linalg.norm(feature_matrix - feature_matrix[first_idx], axis=1)
    selected_distances = [0.0]
    min_dist[first_idx] = -np.inf

    while len(selected) < n_select:
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        selected_distances.append(float(min_dist[next_idx]))
        new_dist = np.linalg.norm(feature_matrix - feature_matrix[next_idx], axis=1)
        min_dist = np.minimum(min_dist, new_dist)
        min_dist[selected] = -np.inf

    return selected, selected_distances


def balanced_quotas(*, n_select: int, n_bins: int) -> list[int]:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    base, remainder = divmod(n_select, n_bins)
    return [base + (1 if idx < remainder else 0) for idx in range(n_bins)]


def parse_complexity_edges(value: str) -> list[float]:
    edges = []
    for raw_part in value.split(","):
        part = raw_part.strip().lower()
        if part in {"inf", "+inf", "infinity", "+infinity"}:
            edges.append(math.inf)
        else:
            edges.append(float(part))

    if len(edges) < 2:
        raise ValueError("--complexity-edges must contain at least two values")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("--complexity-edges must be strictly increasing")
    return edges


def complexity_bin_labels(
    genes: list[tuple[int, ...]],
    *,
    n_bins: int,
    strategy: str = "quantile",
    fixed_edges: list[float] | None = None,
) -> tuple[np.ndarray, list[float]]:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    log_complexity = np.asarray(
        [architecture_descriptors(gene)["log_estimated_complexity"] for gene in genes],
        dtype=np.float64,
    )

    if strategy == "fixed":
        if fixed_edges is None:
            raise ValueError("fixed complexity strategy requires fixed_edges")
        edges = np.asarray(fixed_edges, dtype=np.float64)
    elif strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(log_complexity, quantiles)
        edges[-1] = math.inf
    else:
        raise ValueError(f"Unknown complexity strategy: {strategy}")

    labels = np.searchsorted(edges[1:-1], log_complexity, side="right")
    labels = np.clip(labels, 0, len(edges) - 2)
    return labels.astype(int), [float(edge) for edge in edges]


def band_counts(labels: np.ndarray, *, n_bands: int) -> dict[int, int]:
    return {idx: int(np.sum(labels == idx)) for idx in range(n_bands)}


def band_proportions(counts: dict[int, int]) -> dict[int, float]:
    total = sum(counts.values())
    if total == 0:
        return {idx: 0.0 for idx in counts}
    return {idx: float(value / total) for idx, value in counts.items()}


def abort_if_underfilled(counts: dict[int, int], quotas: list[int]) -> None:
    underfilled = {
        band: {"available": counts.get(band, 0), "required": quota}
        for band, quota in enumerate(quotas)
        if counts.get(band, 0) < quota
    }
    if underfilled:
        raise RuntimeError(
            "Insufficient candidates for stratified sampling: "
            + json.dumps(underfilled, sort_keys=True)
        )


def stratified_random_select(
    labels: np.ndarray,
    *,
    n_select: int,
    n_bands: int,
    seed: int,
) -> tuple[list[int], list[int], dict[int, list[int]]]:
    """Randomly draw a balanced stratified sample from precomputed band labels."""

    quotas = balanced_quotas(n_select=n_select, n_bins=n_bands)
    counts = band_counts(labels, n_bands=n_bands)
    abort_if_underfilled(counts, quotas)

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1]))
    selected: list[int] = []
    selected_bins: list[int] = []
    draw_order: dict[int, list[int]] = {}

    for band_idx, quota in enumerate(quotas):
        candidate_indices = np.flatnonzero(labels == band_idx)
        order = rng.permutation(candidate_indices).astype(int).tolist()
        draw_order[band_idx] = order
        selected.extend(order[:quota])
        selected_bins.extend([band_idx] * quota)

    return selected, selected_bins, draw_order


def stratified_max_min_select(
    genes: list[tuple[int, ...]],
    *,
    n_select: int,
    labels: np.ndarray,
    n_bands: int,
    pool_features: np.ndarray,
) -> tuple[list[int], list[int], list[float]]:
    """Run max-min independently inside each band in pool-standardized space."""

    quotas = balanced_quotas(n_select=n_select, n_bins=n_bands)
    counts = band_counts(labels, n_bands=n_bands)
    abort_if_underfilled(counts, quotas)

    selected: list[int] = []
    selected_bins: list[int] = []
    selected_raw_distances: list[float] = []

    for band_idx, quota in enumerate(quotas):
        candidate_indices = np.flatnonzero(labels == band_idx).astype(int)
        local_genes = [genes[int(idx)] for idx in candidate_indices]
        local_features = pool_features[candidate_indices]
        local_selected, local_distances = greedy_max_min_select(
            local_genes,
            n_select=quota,
            features=local_features,
        )
        for local_idx, distance in zip(local_selected, local_distances):
            global_idx = int(candidate_indices[local_idx])
            selected.append(global_idx)
            selected_bins.append(band_idx)
            selected_raw_distances.append(float(distance))

    return selected, selected_bins, selected_raw_distances


def nearest_neighbor_distances(
    genes: list[tuple[int, ...]],
    *,
    features: np.ndarray | None = None,
) -> list[float]:
    """Return selected-set nearest-neighbor distances in the supplied feature space.

    For official samples this receives ``pool_features[selected_indices]``, so
    distances are in the single pool-standardized descriptor space.
    """

    if len(genes) <= 1:
        return [0.0] * len(genes)

    feature_matrix = np.asarray(features, dtype=np.float64) if features is not None else descriptor_matrix(genes)
    if len(feature_matrix) != len(genes):
        raise ValueError("features length must match genes length")

    distances = np.linalg.norm(feature_matrix[:, None, :] - feature_matrix[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    return [float(value) for value in distances.min(axis=1)]


def op_counts_json(gene: tuple[int, ...]) -> str:
    _, units = split_units(gene)
    counts = Counter(PRIMITIVES[unit[0]] for unit in units)
    return json.dumps(dict(sorted(counts.items())), sort_keys=True)


def build_sample_row(
    *,
    sample_id: str,
    rank: int,
    seed: int,
    sampling_method: str,
    gene: tuple[int, ...],
    reference_genes: set[tuple[int, ...]],
    selection_pool_index: int | str,
    selection_min_distance: float | str,
    selection_min_distance_raw: float | str,
    complexity_bin: int,
    params_real: int | str = "",
) -> dict:
    desc = architecture_descriptors(gene)
    row = {
        "sample_id": sample_id,
        "selection_rank": rank,
        "seed": seed,
        "sampling_method": sampling_method,
        "Net": str(list(gene)),
        "gene": list(gene),
        "overlaps_reference": gene in reference_genes,
        "selection_pool_index": selection_pool_index,
        "selection_min_distance": selection_min_distance,
        "selection_min_distance_raw": selection_min_distance_raw,
        "complexity_bin": int(complexity_bin),
        "params_real": params_real,
        "channel_idx": int(desc["channel_idx"]),
        "channels": int(desc["channels"]),
        "identity_count": int(desc["identity_count"]),
        "trainable_unit_count": int(desc["trainable_unit_count"]),
        "estimated_complexity": desc["estimated_complexity"],
        "log_estimated_complexity": desc["log_estimated_complexity"],
        "op_counts_json": op_counts_json(gene),
    }
    for idx, value in enumerate(gene):
        row[f"gene_{idx}"] = value
    return row


def write_gene_jsons(
    *,
    output_dir: Path,
    sample_rows: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in sample_rows:
        payload = {
            "sample_id": row["sample_id"],
            "gene": row["gene"],
            "sampling_method": row["sampling_method"],
            "seed": row["seed"],
            "selection_rank": row["selection_rank"],
            "complexity_bin": row["complexity_bin"],
            "selection_pool_index": row["selection_pool_index"],
        }
        path = output_dir / f"{row['sample_id']}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        row["bass_gene_file"] = str(path.as_posix())


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_train_command(
    *,
    row: dict,
    train_dir: str,
    val_dir: str,
    scale: int,
    output_dir: str,
    max_epochs: int,
    early_stopping_patience: int,
    reduce_lr_patience: int,
    lr_schedule: str,
) -> str:
    return (
        "python -m srir_training.train "
        f"--bass-gene-file {row['bass_gene_file']} "
        f"--directory-train {train_dir} "
        f"--directory-val {val_dir} "
        f"--scale {scale} "
        f"--epochs {max_epochs} "
        f"--lr-schedule {lr_schedule} "
        f"--early-stopping-patience {early_stopping_patience} "
        f"--reduce-lr-patience {reduce_lr_patience} "
        f"--run-name {row['sample_id']} "
        f"--output-dir {output_dir}"
    )


def build_manifest_rows(
    *,
    sample_rows: list[dict],
    train_dir: str,
    val_dir: str,
    scale: int,
    trainer_output_dir: str,
    max_epochs: int,
    early_stopping_patience: int,
    reduce_lr_patience: int,
    lr_schedule: str,
    extra_eval_command_template: str = "",
) -> list[dict]:
    manifest_rows = []
    for row in sample_rows:
        eval_command = ""
        if extra_eval_command_template:
            eval_command = extra_eval_command_template.format(
                sample_id=row["sample_id"],
                run_name=row["sample_id"],
                scale=scale,
                trainer_output_dir=trainer_output_dir,
                model_path=f"{trainer_output_dir}/{row['sample_id']}/checkpoints/best.keras",
            )
        manifest_rows.append(
            {
                "sample_id": row["sample_id"],
                "Net": row["Net"],
                "bass_gene_file": row["bass_gene_file"],
                "scale": scale,
                "run_name": row["sample_id"],
                "trainer_output_dir": trainer_output_dir,
                "train_command": build_train_command(
                    row=row,
                    train_dir=train_dir,
                    val_dir=val_dir,
                    scale=scale,
                    output_dir=trainer_output_dir,
                    max_epochs=max_epochs,
                    early_stopping_patience=early_stopping_patience,
                    reduce_lr_patience=reduce_lr_patience,
                    lr_schedule=lr_schedule,
                ),
                "eval_extra_command": eval_command,
            }
        )
    return manifest_rows


def manifest_fieldnames() -> list[str]:
    return [
        "sample_id",
        "Net",
        "bass_gene_file",
        "scale",
        "run_name",
        "trainer_output_dir",
        "train_command",
        "eval_extra_command",
    ]


def serialize_edges(edges: list[float]) -> list[float | str]:
    return ["inf" if math.isinf(edge) else float(edge) for edge in edges]


def metadata_description(*, pool_policy: str, selection_method: str, complexity_strategy: str) -> str:
    pool_text = "uniform random decoded-gene pool"
    if pool_policy == "mixed":
        pool_text = "experimental legacy mixed-prior decoded-gene pool"

    if selection_method == "stratified_random":
        selection_text = "random sampling within precomputed complexity strata"
    elif selection_method == "stratified_max_min":
        selection_text = "deterministic max-min coverage within precomputed complexity strata"
    else:
        selection_text = "deterministic global max-min coverage"

    return (
        f"Large {pool_text} followed by {selection_text}. Complexity strata are "
        f"defined with the {complexity_strategy} strategy. No PSNR, predicted PSNR, "
        "training result, or zero-cost score is used for architecture selection."
    )


def write_pool_cache(
    path: Path,
    *,
    pool: list[tuple[int, ...]],
    labels: np.ndarray,
    selected_indices: list[int],
    selected_bins: list[int],
    draw_order: dict[int, list[int]],
    edges: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pool": np.asarray(pool, dtype=np.int16),
        "complexity_labels": np.asarray(labels, dtype=np.int16),
        "selected_indices": np.asarray(selected_indices, dtype=np.int64),
        "selected_bins": np.asarray(selected_bins, dtype=np.int16),
        "complexity_edges": np.asarray(edges, dtype=np.float64),
    }
    for band_idx, order in draw_order.items():
        payload[f"draw_order_band_{band_idx}"] = np.asarray(order, dtype=np.int64)
    np.savez_compressed(path, **payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a reproducible diversity-aware BASS architecture set."
    )
    parser.add_argument("--n", type=int, default=50, help="Number of architectures to select")
    parser.add_argument("--pool-size", type=int, default=100000, help="Random valid pool size before stratified selection")
    parser.add_argument("--seed", type=int, default=20260703, help="Random seed")
    parser.add_argument(
        "--pool-policy",
        choices=["uniform", "mixed"],
        default="uniform",
        help="Candidate-pool generator. mixed is experimental/legacy and not the default benchmark policy.",
    )
    parser.add_argument(
        "--selection-method",
        choices=["stratified_random", "stratified_max_min", "greedy_max_min"],
        default="stratified_random",
        help="Selection criterion used after pool generation",
    )
    parser.add_argument(
        "--complexity-bins",
        type=int,
        default=5,
        help="Number of quantile strata used when --complexity-strategy quantile",
    )
    parser.add_argument(
        "--complexity-strategy",
        choices=["quantile", "fixed"],
        default="quantile",
        help="Use pool quantiles or fixed log-complexity bands for stratification",
    )
    parser.add_argument(
        "--complexity-edges",
        default=DEFAULT_COMPLEXITY_EDGES,
        help="Comma-separated fixed log-complexity edges used by --complexity-strategy fixed",
    )
    parser.add_argument("--prefix", default=None, help="Sample id prefix; default derives from --n")
    parser.add_argument(
        "--reference-csv",
        action="append",
        default=["data/20_full_trained_models.csv"],
        help="Reference architecture CSV used only for overlap reporting. Can be repeated.",
    )
    parser.add_argument(
        "--exclude-reference",
        action="store_true",
        help="Deprecated. References are report-only for the official benchmark sample.",
    )
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--gene-json-dir", default=None)
    parser.add_argument("--trainer-manifest", default=None)
    parser.add_argument("--metadata-json", default=None)
    parser.add_argument("--pool-cache", default=None)
    parser.add_argument("--save-pool-cache", action="store_true")
    parser.add_argument("--scale", type=int, choices=[2, 3, 4], default=2)
    parser.add_argument("--train-dir", default="/data/DIV2K_train_HR")
    parser.add_argument("--val-dir", default="/data/DIV2K_valid_HR")
    parser.add_argument("--trainer-output-dir", default=None)
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1000,
        help="Maximum epoch cap for full training; the trainer stops earlier by EarlyStopping",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Backward-compatible alias for --max-epochs",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--reduce-lr-patience", type=int, default=15)
    parser.add_argument("--lr-schedule", choices=["plateau", "cosine", "none"], default="plateau")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.exclude_reference:
        raise ValueError("Reference exclusion is deprecated; references are report-only.")
    if args.n <= 0:
        raise ValueError("--n must be positive")
    if args.pool_size < args.n:
        raise ValueError("--pool-size must be at least --n")
    if args.complexity_bins <= 0:
        raise ValueError("--complexity-bins must be positive")
    if args.early_stopping_patience <= 0:
        raise ValueError("--early-stopping-patience must be positive")
    if args.reduce_lr_patience <= 0:
        raise ValueError("--reduce-lr-patience must be positive")

    max_epochs = args.epochs if args.epochs is not None else args.max_epochs
    if max_epochs <= 0:
        raise ValueError("--max-epochs must be positive")
    fixed_edges = parse_complexity_edges(args.complexity_edges)

    sample_name = f"bass_{args.n}_sample"
    prefix = args.prefix or "bass"
    output_csv = Path(args.output_csv or f"data/architectures/{sample_name}_architectures.csv")
    gene_json_dir = Path(args.gene_json_dir or f"data/architectures/{sample_name}/genes")
    trainer_manifest = Path(args.trainer_manifest or f"data/architectures/{sample_name}_trainer_manifest.csv")
    metadata_json = Path(args.metadata_json or f"data/architectures/{sample_name}_metadata.json")
    trainer_output_dir = args.trainer_output_dir or f"srir_outputs/{sample_name}"
    pool_cache = Path(args.pool_cache or f"data/architectures/{sample_name}_pool_cache.npz")

    reference_genes = load_reference_genes(args.reference_csv)
    pool = generate_unique_pool(
        pool_size=args.pool_size,
        seed=args.seed,
        pool_policy=args.pool_policy,
    )
    generated_unique = len(pool)
    overlap_in_pool = sum(1 for gene in pool if gene in reference_genes)
    pool_features = descriptor_matrix(pool, standardize=True)

    labels, edges = complexity_bin_labels(
        pool,
        n_bins=args.complexity_bins,
        strategy=args.complexity_strategy,
        fixed_edges=fixed_edges if args.complexity_strategy == "fixed" else None,
    )
    n_bands = len(edges) - 1
    counts = band_counts(labels, n_bands=n_bands)
    proportions = band_proportions(counts)
    for band_idx in range(n_bands):
        print(f"[POOL] band_{band_idx}={counts[band_idx]}")

    draw_order: dict[int, list[int]] = {}
    selection_min_distance_raw: list[float | str]
    if args.selection_method == "stratified_random":
        selected_indices, complexity_bins, draw_order = stratified_random_select(
            labels,
            n_select=args.n,
            n_bands=n_bands,
            seed=args.seed,
        )
        selection_min_distance_raw = [""] * len(selected_indices)
        sampling_method = "stratified_random_fixed_bands" if args.complexity_strategy == "fixed" else "stratified_random_quantile_bands"
    elif args.selection_method == "stratified_max_min":
        selected_indices, complexity_bins, raw_distances = stratified_max_min_select(
            pool,
            n_select=args.n,
            labels=labels,
            n_bands=n_bands,
            pool_features=pool_features,
        )
        selection_min_distance_raw = raw_distances
        sampling_method = "stratified_max_min_fixed_bands" if args.complexity_strategy == "fixed" else "stratified_max_min_quantile_bands"
    else:
        selected_indices, raw_distances = greedy_max_min_select(
            pool,
            n_select=args.n,
            features=pool_features,
        )
        complexity_bins = [int(labels[idx]) for idx in selected_indices]
        selection_min_distance_raw = raw_distances
        sampling_method = "greedy_max_min_pool_standardized"

    selected_genes = [pool[idx] for idx in selected_indices]
    selected_distances = nearest_neighbor_distances(
        selected_genes,
        features=pool_features[np.asarray(selected_indices, dtype=np.int64)],
    )

    sample_rows: list[dict] = []
    for rank, (gene, distance, raw_distance, complexity_bin, pool_idx) in enumerate(
        zip(
            selected_genes,
            selected_distances,
            selection_min_distance_raw,
            complexity_bins,
            selected_indices,
        ),
        start=1,
    ):
        sample_rows.append(
            build_sample_row(
                sample_id=f"{prefix}_{rank:04d}",
                rank=rank,
                seed=args.seed,
                sampling_method=sampling_method,
                gene=gene,
                reference_genes=reference_genes,
                selection_pool_index=int(pool_idx),
                selection_min_distance=distance,
                selection_min_distance_raw=raw_distance,
                complexity_bin=int(complexity_bin),
                params_real="",
            )
        )

    write_gene_jsons(output_dir=gene_json_dir, sample_rows=sample_rows)
    write_csv(output_csv, sample_rows, SAMPLE_OUTPUT_FIELDS)

    manifest_rows = build_manifest_rows(
        sample_rows=sample_rows,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        scale=args.scale,
        trainer_output_dir=trainer_output_dir,
        max_epochs=max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        reduce_lr_patience=args.reduce_lr_patience,
        lr_schedule=args.lr_schedule,
    )
    write_csv(trainer_manifest, manifest_rows, manifest_fieldnames())

    if args.save_pool_cache or args.pool_cache:
        write_pool_cache(
            pool_cache,
            pool=pool,
            labels=labels,
            selected_indices=selected_indices,
            selected_bins=complexity_bins,
            draw_order=draw_order,
            edges=edges,
        )

    selected_overlap = sum(1 for gene in selected_genes if gene in reference_genes)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "obsolete_previous_sample_commit": "fe0d379",
        "sampling_method": sampling_method,
        "description": metadata_description(
            pool_policy=args.pool_policy,
            selection_method=args.selection_method,
            complexity_strategy=args.complexity_strategy,
        ),
        "n_selected": args.n,
        "pool_size_requested": args.pool_size,
        "pool_size_unique": generated_unique,
        "seed": args.seed,
        "pool_policy": args.pool_policy,
        "selection_method": args.selection_method,
        "complexity_strategy": args.complexity_strategy,
        "complexity_bins": args.complexity_bins,
        "complexity_edges": serialize_edges(edges),
        "fixed_complexity_edges_if_used": serialize_edges(fixed_edges),
        "pool_band_counts": {str(key): value for key, value in counts.items()},
        "pool_band_proportions": {str(key): value for key, value in proportions.items()},
        "draw_order": {str(key): value for key, value in draw_order.items()},
        "pool_cache": pool_cache.as_posix() if (args.save_pool_cache or args.pool_cache) else "",
        "reference_csv": args.reference_csv,
        "reference_gene_count": len(reference_genes),
        "reference_usage": "overlap reporting only; references are not used in sampling",
        "exclude_reference": False,
        "overlap_reference_in_pool": overlap_in_pool,
        "overlap_reference_selected": selected_overlap,
        "output_csv": output_csv.as_posix(),
        "gene_json_dir": gene_json_dir.as_posix(),
        "trainer_manifest": trainer_manifest.as_posix(),
        "trainer_output_dir": trainer_output_dir,
        "scale": args.scale,
        "train_dir": args.train_dir,
        "val_dir": args.val_dir,
        "max_epochs": max_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "reduce_lr_patience": args.reduce_lr_patience,
        "lr_schedule": args.lr_schedule,
        "training_stopping_policy": (
            "Full training uses the repository trainer callbacks. The epoch value is "
            "only a maximum cap; EarlyStopping monitors validation PSNR and restores "
            "the best weights."
        ),
        "params_real_status": "pending TensorFlow validation with tools/validate_bass_sample.py",
        "validation": {"validated": False, "degenerate_found": None, "replacements": []},
        "descriptor_notes": [
            "Descriptor matrix is 26-dimensional: op/kernel/repeat counts, channels, and log estimated complexity.",
            "Descriptor standardization for selection distances is computed once on the full pool.",
            "selection_min_distance is nearest-neighbor distance among the selected 50 in the pool-standardized descriptor space.",
            "selection_min_distance_raw is the max-min draw distance in the same pool-standardized space when max-min selection is used; it is empty for stratified_random.",
            "PCA is not used for selection and no performance label is used.",
        ],
    }
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[SAMPLE] selected={args.n} pool_unique={generated_unique} seed={args.seed}")
    print(f"[SAMPLE] reference_overlap_selected={selected_overlap}")
    print("[SAMPLE] params_real pending; run tools/validate_bass_sample.py in a TensorFlow-compatible environment")
    print(f"[SAMPLE] architectures: {output_csv}")
    print(f"[SAMPLE] genes: {gene_json_dir}")
    print(f"[SAMPLE] trainer manifest: {trainer_manifest}")
    print(f"[SAMPLE] metadata: {metadata_json}")
    if args.save_pool_cache or args.pool_cache:
        print(f"[SAMPLE] pool cache: {pool_cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
