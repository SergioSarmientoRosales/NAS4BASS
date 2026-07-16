from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "rework_50_architectures" / "manifests"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _parse_gene(value: object) -> list[int]:
    if isinstance(value, list):
        gene = [int(item) for item in value]
    else:
        parsed = ast.literal_eval(str(value))
        gene = [int(item) for item in parsed]
    if len(gene) != 28:
        raise ValueError(f"Expected a 28-value BASS gene, received {len(gene)} values")
    return gene


def _gene_digest(gene: list[int]) -> str:
    payload = json.dumps(gene, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _require_finite_positive(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be finite and positive, received {value!r}")
    return number


def build_primary_manifest(repo_root: Path) -> list[dict[str, object]]:
    sample_path = repo_root / "data" / "architectures" / "bass_50_sample_architectures.csv"
    summary_path = repo_root / "DNNs" / "Full Trained" / "strategic50" / "summary.csv"
    gene_dir = repo_root / "data" / "architectures" / "bass_50_sample" / "genes"

    sample_rows = _read_csv(sample_path)
    summary_rows = _read_csv(summary_path)
    if len(sample_rows) != 50 or len(summary_rows) != 50:
        raise ValueError(
            f"Primary scenario requires 50 sample and 50 summary rows; "
            f"found {len(sample_rows)} and {len(summary_rows)}"
        )

    summary_by_index = {int(row["gene_index"]): row for row in summary_rows}
    manifest: list[dict[str, object]] = []
    for sample_row in sample_rows:
        arch_id = sample_row["sample_id"].strip()
        index = int(arch_id.rsplit("_", 1)[1])
        summary_row = summary_by_index.get(index)
        if summary_row is None:
            raise ValueError(f"No full-training summary row for {arch_id}")

        gene_path = gene_dir / f"{arch_id}.json"
        gene_payload = json.loads(gene_path.read_text(encoding="utf-8"))
        gene = _parse_gene(gene_payload["gene"])
        sample_gene = _parse_gene(sample_row["Net"])
        if gene != sample_gene:
            raise ValueError(f"Gene mismatch between sample CSV and JSON for {arch_id}")

        params = int(_require_finite_positive(summary_row["params"], f"{arch_id} params"))
        psnr = _require_finite_positive(
            summary_row["best_overall_psnr"], f"{arch_id} best validation PSNR"
        )
        manifest.append(
            {
                "scenario": "expanded50",
                "architecture_id": arch_id,
                "architecture_index": index,
                "gene_json": json.dumps(gene, separators=(",", ":")),
                "gene_sha256": _gene_digest(gene),
                "gene_file": gene_path.relative_to(repo_root).as_posix(),
                "valid_psnr": psnr,
                "params_real": params,
                "complexity_bin": int(sample_row["complexity_bin"]),
                "sampling_method": sample_row["sampling_method"],
                "sampling_seed": int(sample_row["seed"]),
                "selection_pool_index": int(sample_row["selection_pool_index"]),
                "full_training_status": "complete",
            }
        )

    return sorted(manifest, key=lambda row: int(row["architecture_index"]))


def build_secondary_manifest(repo_root: Path) -> list[dict[str, object]]:
    benchmark_path = repo_root / "data" / "20_full_trained_models.csv"
    gene_dir = repo_root / "data" / "architectures" / "pareto20" / "genes"
    rows = _read_csv(benchmark_path)
    if len(rows) != 20:
        raise ValueError(f"Secondary scenario requires 20 rows; found {len(rows)}")

    manifest: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        arch_id = f"pareto_{index:04d}"
        gene = _parse_gene(row["Net"])
        gene_path = gene_dir / f"{arch_id}.json"
        if gene_path.exists():
            gene_payload = json.loads(gene_path.read_text(encoding="utf-8"))
            if _parse_gene(gene_payload["gene"]) != gene:
                raise ValueError(f"Gene mismatch between benchmark CSV and JSON for {arch_id}")

        manifest.append(
            {
                "scenario": "pareto20",
                "architecture_id": arch_id,
                "architecture_index": index,
                "gene_json": json.dumps(gene, separators=(",", ":")),
                "gene_sha256": _gene_digest(gene),
                "gene_file": gene_path.relative_to(repo_root).as_posix(),
                "valid_psnr": _require_finite_positive(
                    row["valid_psnr"], f"{arch_id} best validation PSNR"
                ),
                "params_real": int(_require_finite_positive(row["params"], f"{arch_id} params")),
                "complexity_bin": "",
                "sampling_method": "approximate_pareto_front",
                "sampling_seed": "",
                "selection_pool_index": "",
                "full_training_status": "complete",
            }
        )

    return manifest


def validate_manifests(
    primary: list[dict[str, object]], secondary: list[dict[str, object]]
) -> dict[str, object]:
    def duplicates(rows: list[dict[str, object]]) -> list[str]:
        seen: set[str] = set()
        duplicate_ids: list[str] = []
        for row in rows:
            digest = str(row["gene_sha256"])
            if digest in seen:
                duplicate_ids.append(str(row["architecture_id"]))
            seen.add(digest)
        return duplicate_ids

    primary_by_digest = {str(row["gene_sha256"]): str(row["architecture_id"]) for row in primary}
    overlaps = []
    for row in secondary:
        digest = str(row["gene_sha256"])
        if digest in primary_by_digest:
            overlaps.append(
                {
                    "expanded50_id": primary_by_digest[digest],
                    "pareto20_id": str(row["architecture_id"]),
                    "gene_sha256": digest,
                }
            )

    return {
        "expanded50_count": len(primary),
        "pareto20_count": len(secondary),
        "expanded50_duplicate_ids": duplicates(primary),
        "pareto20_duplicate_ids": duplicates(secondary),
        "cross_scenario_exact_overlaps": overlaps,
        "cross_scenario_overlap_count": len(overlaps),
    }


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(repo_root: Path, output_dir: Path) -> dict[str, object]:
    primary = build_primary_manifest(repo_root)
    secondary = build_secondary_manifest(repo_root)
    validation = validate_manifests(primary, secondary)
    if validation["expanded50_duplicate_ids"] or validation["pareto20_duplicate_ids"]:
        raise ValueError(f"Within-scenario duplicate genes detected: {validation}")

    primary_path = output_dir / "expanded50_manifest.csv"
    secondary_path = output_dir / "pareto20_manifest.csv"
    _write_manifest(primary_path, primary)
    _write_manifest(secondary_path, secondary)

    source_paths = [
        repo_root / "data" / "architectures" / "bass_50_sample_architectures.csv",
        repo_root / "data" / "architectures" / "bass_50_sample_metadata.json",
        repo_root / "DNNs" / "Full Trained" / "strategic50" / "summary.csv",
        repo_root / "data" / "20_full_trained_models.csv",
    ]
    validation["source_sha256"] = {
        path.relative_to(repo_root).as_posix(): _sha256(path) for path in source_paths
    }
    validation["manifest_sha256"] = {
        primary_path.name: _sha256(primary_path),
        secondary_path.name: _sha256(secondary_path),
    }
    validation_path = output_dir / "manifest_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create immutable-input manifests for the expanded50 and Pareto20 zero-cost scenarios."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validation = prepare(args.repo_root.resolve(), args.output_dir.resolve())
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
