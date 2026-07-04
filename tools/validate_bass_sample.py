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

from tools.sample_bass_architectures import (
    SAMPLE_OUTPUT_FIELDS,
    build_eval_extra_command_template,
    build_manifest_rows,
    build_sample_row,
    descriptor_matrix,
    load_reference_genes,
    manifest_fieldnames,
    nearest_neighbor_distances,
    parse_gene,
    write_csv,
    write_gene_jsons,
)


def load_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_metadata(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def load_pool_cache(path: Path) -> tuple[list[tuple[int, ...]], np.ndarray, dict[int, list[int]]]:
    if not path.exists():
        raise FileNotFoundError(f"Pool cache not found: {path}")

    data = np.load(path, allow_pickle=False)
    pool = [tuple(int(value) for value in row) for row in data["pool"]]
    labels = data["complexity_labels"].astype(int)
    draw_order: dict[int, list[int]] = {}
    for key in data.files:
        if key.startswith("draw_order_band_"):
            band_idx = int(key.rsplit("_", 1)[1])
            draw_order[band_idx] = [int(value) for value in data[key]]
    if not draw_order:
        raise ValueError("Pool cache does not contain draw_order_band_* arrays")
    return pool, labels, draw_order


def import_tensorflow():
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - depends on local TF install
        raise RuntimeError(
            "TensorFlow is required to instantiate BASS models and compute params_real. "
            "Run this validator in the training environment where TensorFlow is installed "
            f"and compatible with NumPy. Import error: {exc}"
        ) from exc
    return tf


def count_params_for_gene(gene: tuple[int, ...], *, scale: int, channels: int) -> int:
    tf = import_tensorflow()
    from search_space.model_builder import get_model
    from search_space.search_space import decode

    try:
        model = get_model(decode(list(gene)), upscale_factor=scale, channels=channels)
        params_real = int(model.count_params())
    finally:
        tf.keras.backend.clear_session()
    return params_real


def validate_gene(gene: tuple[int, ...], *, scale: int, channels: int) -> tuple[bool, int | None, str]:
    from tools.sample_bass_architectures import architecture_descriptors

    desc = architecture_descriptors(gene)
    if int(desc["trainable_unit_count"]) == 0:
        return False, None, "trainable_unit_count == 0"

    try:
        params_real = count_params_for_gene(gene, scale=scale, channels=channels)
    except Exception as exc:  # pragma: no cover - depends on specific TF failures
        return False, None, f"model instantiation failed: {exc}"

    if params_real == 0:
        return False, params_real, "params_real == 0"
    return True, params_real, ""


def params_quintile_labels(params_real: list[int]) -> np.ndarray:
    log_params = np.log(np.asarray(params_real, dtype=np.float64))
    edges = np.quantile(log_params, np.linspace(0.0, 1.0, 6))
    labels = np.searchsorted(edges[1:-1], log_params, side="right")
    return np.clip(labels, 0, 4).astype(int)


def proxy_alignment_matrix(complexity_bins: list[int], params_real: list[int], *, n_bins: int = 5) -> list[list[int]]:
    params_labels = params_quintile_labels(params_real)
    matrix = np.zeros((n_bins, 5), dtype=int)
    for complexity_bin, params_bin in zip(complexity_bins, params_labels):
        matrix[int(complexity_bin), int(params_bin)] += 1
    return matrix.tolist()


def resolve_from_metadata(args: argparse.Namespace, metadata: dict, key: str, fallback):
    value = getattr(args, key)
    if value is not None:
        return value
    return metadata.get(key, fallback)


def replace_degenerate_indices(
    *,
    selected_indices: list[int],
    selected_bins: list[int],
    pool: list[tuple[int, ...]],
    draw_order: dict[int, list[int]],
    scale: int,
    channels: int,
) -> tuple[list[int], dict[int, int], list[dict]]:
    used_indices = set(selected_indices)
    blocked_indices = set(selected_indices)
    params_by_index: dict[int, int] = {}
    replacements: list[dict] = []

    for row_pos, (pool_idx, band_idx) in enumerate(list(zip(selected_indices, selected_bins))):
        gene = pool[pool_idx]
        is_valid, params_real, reason = validate_gene(gene, scale=scale, channels=channels)
        if is_valid:
            params_by_index[pool_idx] = int(params_real)
            continue

        replacement_record = {
            "row_position": row_pos,
            "old_pool_index": int(pool_idx),
            "complexity_bin": int(band_idx),
            "reason": reason,
            "new_pool_index": None,
        }

        replacement_found = False
        for candidate_idx in draw_order.get(int(band_idx), []):
            candidate_idx = int(candidate_idx)
            if candidate_idx in blocked_indices or candidate_idx in used_indices:
                continue
            candidate_gene = pool[candidate_idx]
            candidate_ok, candidate_params, candidate_reason = validate_gene(
                candidate_gene,
                scale=scale,
                channels=channels,
            )
            blocked_indices.add(candidate_idx)
            if not candidate_ok:
                replacements.append(
                    {
                        "row_position": row_pos,
                        "old_pool_index": int(pool_idx),
                        "candidate_pool_index": candidate_idx,
                        "complexity_bin": int(band_idx),
                        "reason": candidate_reason,
                        "new_pool_index": None,
                    }
                )
                continue

            selected_indices[row_pos] = candidate_idx
            used_indices.discard(pool_idx)
            used_indices.add(candidate_idx)
            params_by_index[candidate_idx] = int(candidate_params)
            replacement_record["new_pool_index"] = candidate_idx
            replacement_found = True
            break

        replacements.append(replacement_record)
        if not replacement_found:
            raise RuntimeError(
                f"No valid replacement found for row {row_pos} in complexity band {band_idx}"
            )

    return selected_indices, params_by_index, replacements


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a sampled BASS architecture CSV by instantiating models and computing params_real."
    )
    parser.add_argument("--sample-csv", default="data/architectures/bass_50_sample_architectures.csv")
    parser.add_argument("--metadata-json", default="data/architectures/bass_50_sample_metadata.json")
    parser.add_argument("--pool-cache", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--gene-json-dir", default=None)
    parser.add_argument("--trainer-manifest", default=None)
    parser.add_argument("--scale", type=int, choices=[2, 3, 4], default=None)
    parser.add_argument("--channels", type=int, choices=[1, 3], default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample_csv = Path(args.sample_csv)
    metadata_json = Path(args.metadata_json)
    metadata = load_metadata(metadata_json)

    pool_cache = Path(args.pool_cache or metadata.get("pool_cache") or "data/architectures/bass_50_sample_pool_cache.npz")
    output_csv = Path(args.output_csv or metadata.get("output_csv") or sample_csv)
    gene_json_dir = Path(args.gene_json_dir or metadata.get("gene_json_dir") or "data/architectures/bass_50_sample/genes")
    trainer_manifest = Path(args.trainer_manifest or metadata.get("trainer_manifest") or "data/architectures/bass_50_sample_trainer_manifest.csv")
    scale = int(args.scale or metadata.get("scale", 2))

    rows = load_csv_rows(sample_csv)
    pool, labels, draw_order = load_pool_cache(pool_cache)
    selected_indices = [int(row["selection_pool_index"]) for row in rows]
    selected_bins = [int(row["complexity_bin"]) for row in rows]

    selected_indices, params_by_index, replacements = replace_degenerate_indices(
        selected_indices=selected_indices,
        selected_bins=selected_bins,
        pool=pool,
        draw_order=draw_order,
        scale=scale,
        channels=args.channels,
    )

    pool_features = descriptor_matrix(pool, standardize=True)
    selected_genes = [pool[idx] for idx in selected_indices]
    selected_distances = nearest_neighbor_distances(
        selected_genes,
        features=pool_features[np.asarray(selected_indices, dtype=np.int64)],
    )
    reference_genes = load_reference_genes(metadata.get("reference_csv", []))

    updated_rows: list[dict] = []
    for original_row, pool_idx, distance in zip(rows, selected_indices, selected_distances):
        gene = pool[pool_idx]
        sample_id = original_row["sample_id"]
        rank = int(original_row["selection_rank"])
        raw_distance = original_row.get("selection_min_distance_raw", "")
        updated_rows.append(
            build_sample_row(
                sample_id=sample_id,
                rank=rank,
                seed=int(original_row.get("seed") or metadata.get("seed", 0)),
                sampling_method=original_row.get("sampling_method") or metadata.get("sampling_method", ""),
                gene=gene,
                reference_genes=reference_genes,
                selection_pool_index=int(pool_idx),
                selection_min_distance=distance,
                selection_min_distance_raw=raw_distance,
                complexity_bin=int(labels[pool_idx]),
                params_real=int(params_by_index[pool_idx]),
            )
        )

    write_gene_jsons(output_dir=gene_json_dir, sample_rows=updated_rows)
    if not args.dry_run:
        write_csv(output_csv, updated_rows, SAMPLE_OUTPUT_FIELDS)

    manifest_rows = build_manifest_rows(
        sample_rows=updated_rows,
        train_dir=metadata.get("train_dir", "/data/DIV2K_train_HR"),
        val_dir=metadata.get("val_dir", "/data/DIV2K_valid_HR"),
        scale=scale,
        trainer_output_dir=metadata.get("trainer_output_dir", "srir_outputs/bass_50_sample"),
        max_epochs=int(metadata.get("max_epochs", 1000)),
        early_stopping_patience=int(metadata.get("early_stopping_patience", 20)),
        reduce_lr_patience=int(metadata.get("reduce_lr_patience", 15)),
        lr_schedule=metadata.get("lr_schedule", "plateau"),
        extra_eval_command_template=build_eval_extra_command_template(
            set5_dir=metadata.get("extra_set5_dir", "/data/Set5"),
            set14_dir=metadata.get("extra_set14_dir", "/data/Set14"),
            bsd100_dir=metadata.get("extra_bsd100_dir", "/data/BSD100"),
            output_dir=metadata.get("extra_eval_output_dir", "results/zerocost_50_stratified_random/extra_datasets"),
        ),
    )
    if not args.dry_run:
        write_csv(trainer_manifest, manifest_rows, manifest_fieldnames())

    params_real = [int(row["params_real"]) for row in updated_rows]
    complexity_bins = [int(row["complexity_bin"]) for row in updated_rows]
    alignment = proxy_alignment_matrix(
        complexity_bins,
        params_real,
        n_bins=len(metadata.get("pool_band_counts", {})) or 5,
    )
    metadata["validation"] = {
        "validated": True,
        "degenerate_found": sum(1 for item in replacements if item.get("new_pool_index") is not None),
        "replacements": replacements,
        "params_real_min": int(min(params_real)),
        "params_real_max": int(max(params_real)),
        "params_real_median": float(np.median(params_real)),
    }
    metadata["params_real_status"] = "validated"
    metadata["proxy_alignment"] = {
        "rows": "complexity_bin",
        "columns": "quintiles of log(params_real)",
        "matrix": alignment,
    }
    if not args.dry_run:
        save_metadata(metadata_json, metadata)

    print(f"[VALIDATE] architectures={len(updated_rows)} validated=True")
    print(f"[VALIDATE] replacements={metadata['validation']['degenerate_found']}")
    print(f"[VALIDATE] params_real min={min(params_real)} median={np.median(params_real):.1f} max={max(params_real)}")
    print(f"[VALIDATE] output_csv={output_csv}")
    print(f"[VALIDATE] metadata={metadata_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
