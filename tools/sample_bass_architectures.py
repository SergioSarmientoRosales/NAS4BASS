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
) -> list[tuple[int, ...]]:
    rng = np.random.default_rng(seed)
    pool: dict[tuple[int, ...], None] = {}
    attempts = 0
    max_attempts = max(pool_size * 50, pool_size + 1000)

    while len(pool) < pool_size and attempts < max_attempts:
        gene = tuple(int(item) for item in rng.integers(0, 8, size=28))
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


def descriptor_matrix(genes: list[tuple[int, ...]]) -> np.ndarray:
    rows = []
    for gene in genes:
        desc = architecture_descriptors(gene)
        gene_features = [value / 7.0 for value in gene]
        struct_features = [
            desc["channels"] / max(CHANNELS),
            desc["identity_count"] / 9.0,
            desc["trainable_unit_count"] / 9.0,
            desc["mean_kernel"] / max(KERNELS),
            desc["mean_repeat"] / max(REPEAT),
            desc["log_estimated_complexity"],
        ]
        count_features = []
        for prefix in ("op", "kernel", "repeat"):
            count_features.extend(desc[f"{prefix}_{idx}_count"] / 9.0 for idx in range(8))
        rows.append(gene_features + struct_features + count_features)

    matrix = np.asarray(rows, dtype=np.float64)
    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    return (matrix - matrix.mean(axis=0)) / std


def greedy_max_min_select(
    genes: list[tuple[int, ...]],
    *,
    n_select: int,
) -> tuple[list[int], list[float]]:
    if n_select > len(genes):
        raise ValueError(f"n_select={n_select} exceeds candidate pool size {len(genes)}")

    features = descriptor_matrix(genes)
    centroid = features.mean(axis=0)
    first_idx = int(np.argmin(np.linalg.norm(features - centroid, axis=1)))

    selected = [first_idx]
    min_dist = np.linalg.norm(features - features[first_idx], axis=1)
    selected_distances = [0.0]
    min_dist[first_idx] = -np.inf

    while len(selected) < n_select:
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        selected_distances.append(float(min_dist[next_idx]))
        new_dist = np.linalg.norm(features - features[next_idx], axis=1)
        min_dist = np.minimum(min_dist, new_dist)
        min_dist[selected] = -np.inf

    return selected, selected_distances


def op_counts_json(gene: tuple[int, ...]) -> str:
    _, units = split_units(gene)
    counts = Counter(PRIMITIVES[unit[0]] for unit in units)
    return json.dumps(dict(sorted(counts.items())), sort_keys=True)


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
    epochs: int,
) -> str:
    return (
        "python -m srir_training.train "
        f"--bass-gene-file {row['bass_gene_file']} "
        f"--directory-train {train_dir} "
        f"--directory-val {val_dir} "
        f"--scale {scale} "
        f"--epochs {epochs} "
        f"--run-name {row['sample_id']} "
        f"--output-dir {output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a reproducible diversity-aware BASS architecture set."
    )
    parser.add_argument("--n", type=int, default=50, help="Number of architectures to select")
    parser.add_argument("--pool-size", type=int, default=10000, help="Random valid pool size before diversity selection")
    parser.add_argument("--seed", type=int, default=20260703, help="Random seed")
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
        help="Exclude architectures that overlap the reference CSV(s). Default keeps possible overlaps.",
    )
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--gene-json-dir", default=None)
    parser.add_argument("--trainer-manifest", default=None)
    parser.add_argument("--metadata-json", default=None)
    parser.add_argument("--scale", type=int, choices=[2, 3, 4], default=2)
    parser.add_argument("--train-dir", default="/data/DIV2K_train_HR")
    parser.add_argument("--val-dir", default="/data/DIV2K_valid_HR")
    parser.add_argument("--trainer-output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.n <= 0:
        raise ValueError("--n must be positive")
    if args.pool_size < args.n:
        raise ValueError("--pool-size must be at least --n")

    sample_name = f"bass_{args.n}_sample"
    prefix = args.prefix or "bass"
    output_csv = Path(args.output_csv or f"data/architectures/{sample_name}_architectures.csv")
    gene_json_dir = Path(args.gene_json_dir or f"data/architectures/{sample_name}/genes")
    trainer_manifest = Path(args.trainer_manifest or f"data/architectures/{sample_name}_trainer_manifest.csv")
    metadata_json = Path(args.metadata_json or f"data/architectures/{sample_name}_metadata.json")
    trainer_output_dir = args.trainer_output_dir or f"srir_outputs/{sample_name}"

    reference_genes = load_reference_genes(args.reference_csv)
    pool = generate_unique_pool(pool_size=args.pool_size, seed=args.seed)
    generated_unique = len(pool)
    overlap_in_pool = sum(1 for gene in pool if gene in reference_genes)

    if args.exclude_reference:
        pool = [gene for gene in pool if gene not in reference_genes]
        if len(pool) < args.n:
            raise RuntimeError(
                f"Only {len(pool)} candidates remain after reference exclusion; need {args.n}"
            )

    selected_indices, selected_distances = greedy_max_min_select(pool, n_select=args.n)
    selected_genes = [pool[idx] for idx in selected_indices]

    sample_rows: list[dict] = []
    for rank, (gene, distance) in enumerate(zip(selected_genes, selected_distances), start=1):
        sample_id = f"{prefix}_{rank:04d}"
        desc = architecture_descriptors(gene)
        row = {
            "sample_id": sample_id,
            "selection_rank": rank,
            "seed": args.seed,
            "sampling_method": "hybrid_random_pool_greedy_max_min",
            "Net": str(list(gene)),
            "gene": list(gene),
            "overlaps_reference": gene in reference_genes,
            "selection_min_distance": distance,
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
        sample_rows.append(row)

    write_gene_jsons(output_dir=gene_json_dir, sample_rows=sample_rows)

    output_fields = [
        "sample_id",
        "selection_rank",
        "seed",
        "sampling_method",
        "Net",
        "bass_gene_file",
        "overlaps_reference",
        "selection_min_distance",
        "channel_idx",
        "channels",
        "identity_count",
        "trainable_unit_count",
        "estimated_complexity",
        "log_estimated_complexity",
        "op_counts_json",
    ] + [f"gene_{idx}" for idx in range(28)]
    write_csv(output_csv, sample_rows, output_fields)

    manifest_rows = []
    for row in sample_rows:
        manifest_rows.append(
            {
                "sample_id": row["sample_id"],
                "Net": row["Net"],
                "bass_gene_file": row["bass_gene_file"],
                "scale": args.scale,
                "run_name": row["sample_id"],
                "trainer_output_dir": trainer_output_dir,
                "train_command": build_train_command(
                    row=row,
                    train_dir=args.train_dir,
                    val_dir=args.val_dir,
                    scale=args.scale,
                    output_dir=trainer_output_dir,
                    epochs=args.epochs,
                ),
            }
        )
    write_csv(
        trainer_manifest,
        manifest_rows,
        [
            "sample_id",
            "Net",
            "bass_gene_file",
            "scale",
            "run_name",
            "trainer_output_dir",
            "train_command",
        ],
    )

    selected_overlap = sum(1 for gene in selected_genes if gene in reference_genes)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sampling_method": "hybrid_random_pool_greedy_max_min",
        "description": (
            "Large random valid BASS decoded-gene pool followed by greedy max-min "
            "coverage selection in normalized encoding/structural descriptor space. "
            "No PSNR or performance labels are used."
        ),
        "n_selected": args.n,
        "pool_size_requested": args.pool_size,
        "pool_size_unique": generated_unique,
        "seed": args.seed,
        "reference_csv": args.reference_csv,
        "reference_gene_count": len(reference_genes),
        "exclude_reference": bool(args.exclude_reference),
        "overlap_reference_in_pool": overlap_in_pool,
        "overlap_reference_selected": selected_overlap,
        "output_csv": output_csv.as_posix(),
        "gene_json_dir": gene_json_dir.as_posix(),
        "trainer_manifest": trainer_manifest.as_posix(),
        "trainer_output_dir": trainer_output_dir,
        "scale": args.scale,
        "train_dir": args.train_dir,
        "val_dir": args.val_dir,
        "epochs": args.epochs,
        "descriptor_notes": [
            "Decoded 28-gene BASS encoding normalized by gene range.",
            "Structural descriptors include channel, operation/kernel/repeat counts, identity count, and estimated complexity.",
            "PCA is not used for selection and no performance label is used.",
        ],
    }
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[SAMPLE] selected={args.n} pool_unique={generated_unique} seed={args.seed}")
    print(f"[SAMPLE] reference_overlap_selected={selected_overlap}")
    print(f"[SAMPLE] architectures: {output_csv}")
    print(f"[SAMPLE] genes: {gene_json_dir}")
    print(f"[SAMPLE] trainer manifest: {trainer_manifest}")
    print(f"[SAMPLE] metadata: {metadata_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
