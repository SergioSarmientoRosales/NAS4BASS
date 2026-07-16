from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


CHANNELS = (16, 32, 48, 64, 16, 32, 48, 64)
KERNELS = (1, 3, 5, 7, 1, 3, 5, 7)
REPEATS = (1, 2, 3, 4, 1, 2, 3, 4)
GENE_LENGTH = 28
MODEL_GLOB = "gene_*/stage1_p64_best.keras"

LayerDescriptor = tuple[str, int | None, tuple[int, ...], tuple[int, ...]]
ArchitectureSignature = tuple[int, tuple[tuple[LayerDescriptor, ...], ...]]


def _tensor_sources(value: Any) -> list[str]:
    sources: list[str] = []
    if isinstance(value, dict):
        if value.get("class_name") == "__keras_tensor__":
            sources.append(str(value["config"]["keras_history"][0]))
        else:
            for child in value.values():
                sources.extend(_tensor_sources(child))
    elif isinstance(value, list):
        for child in value:
            sources.extend(_tensor_sources(child))
    return sources


def _layer_descriptor(layer: dict[str, Any]) -> LayerDescriptor:
    config = layer.get("config", {})
    return (
        str(layer["class_name"]),
        config.get("filters"),
        tuple(config.get("kernel_size", ())),
        tuple(config.get("dilation_rate", ())),
    )


def read_keras_signature(path: Path) -> ArchitectureSignature:
    with zipfile.ZipFile(path) as archive:
        model_config = json.loads(archive.read("config.json"))["config"]

    layers = {layer["config"]["name"]: layer for layer in model_config["layers"]}
    if not {"stem", "merge"}.issubset(layers):
        raise ValueError(f"{path} is not a supported legacy BASS model")

    channels = int(layers["stem"]["config"]["filters"])
    branch_ends = _tensor_sources(layers["merge"].get("inbound_nodes", []))
    if len(branch_ends) != 3:
        raise ValueError(f"Expected three merge inputs in {path}, got {len(branch_ends)}")

    branches: list[tuple[LayerDescriptor, ...]] = []
    for branch_end in branch_ends:
        chain: list[LayerDescriptor] = []
        layer_name = branch_end
        while layer_name != "stem":
            layer = layers[layer_name]
            chain.append(_layer_descriptor(layer))
            sources = _tensor_sources(layer.get("inbound_nodes", []))
            if len(sources) != 1:
                raise ValueError(
                    f"Expected one predecessor for {layer_name} in {path}, got {sources}"
                )
            layer_name = sources[0]
        branches.append(tuple(reversed(chain)))

    return channels, tuple(branches)


def _conv(filters: int, kernel: int, dilation: int = 1) -> LayerDescriptor:
    return "Conv2D", filters, (kernel, kernel), (dilation, dilation)


def expand_unit(operation: int, kernel: int, repeat: int, channels: int) -> tuple[LayerDescriptor, ...]:
    if operation == 0:
        return (_conv(channels, kernel),) * repeat
    if operation in (1, 2, 3):
        return (_conv(channels, kernel, operation + 1),) * repeat
    if operation == 4:
        return tuple(
            layer
            for _ in range(repeat)
            for layer in (
                ("DepthwiseConv2D", None, (kernel, kernel), (1, 1)),
                _conv(channels, 1),
            )
        )
    if operation == 5:
        return tuple(
            layer
            for _ in range(repeat)
            for layer in (
                _conv(channels * 2, 1),
                ("DepthwiseConv2D", None, (kernel, kernel), (1, 1)),
                _conv(channels, kernel),
            )
        )
    if operation == 6:
        return (
            ("Conv2DTranspose", channels, (kernel, kernel), (1, 1)),
        ) * repeat
    if operation == 7:
        return (("Identity", None, (), ()),)
    raise ValueError(f"Unsupported operation index: {operation}")


def gene_signature(gene: list[int]) -> ArchitectureSignature:
    validate_gene(gene)
    channels = CHANNELS[gene[0]]
    branch_genes = gene[1:]
    branches: list[tuple[LayerDescriptor, ...]] = []
    for branch_index in range(3):
        layers: list[LayerDescriptor] = []
        offset = branch_index * 9
        for unit_index in range(3):
            start = offset + unit_index * 3
            operation, kernel_index, repeat_index = branch_genes[start : start + 3]
            layers.extend(
                expand_unit(
                    operation,
                    KERNELS[kernel_index],
                    REPEATS[repeat_index],
                    channels,
                )
            )
        branches.append(tuple(layers))
    return channels, tuple(branches)


def validate_gene(gene: list[int]) -> None:
    if len(gene) != GENE_LENGTH:
        raise ValueError(f"Expected {GENE_LENGTH} gene values, got {len(gene)}")
    invalid = sorted({value for value in gene if value < 0 or value > 7})
    if invalid:
        raise ValueError(f"Gene values must be in [0, 7], got {invalid}")


def _canonical_branch_gene(
    branch: tuple[LayerDescriptor, ...], channels: int
) -> tuple[int, ...]:
    units: list[tuple[tuple[int, int, int], tuple[LayerDescriptor, ...]]] = []
    for operation in range(8):
        for kernel_index in range(4):
            for repeat_index in range(4):
                if operation == 7 and (kernel_index or repeat_index):
                    continue
                units.append(
                    (
                        (operation, kernel_index, repeat_index),
                        expand_unit(
                            operation,
                            KERNELS[kernel_index],
                            REPEATS[repeat_index],
                            channels,
                        ),
                    )
                )

    solutions: list[tuple[int, ...]] = []

    def visit(position: int, unit_count: int, encoded: tuple[int, ...]) -> None:
        if unit_count == 3:
            if position == len(branch):
                solutions.append(encoded)
            return
        for unit_gene, expanded in units:
            end = position + len(expanded)
            if branch[position:end] == expanded:
                visit(end, unit_count + 1, encoded + unit_gene)

    visit(0, 0, ())
    if not solutions:
        raise ValueError("Could not represent a branch with three BASS units")
    return min(solutions)


def canonical_gene(signature: ArchitectureSignature) -> list[int]:
    channels, branches = signature
    channel_index = CHANNELS.index(channels)
    gene = [channel_index]
    for branch in branches:
        gene.extend(_canonical_branch_gene(branch, channels))
    validate_gene(gene)
    return gene


def _parse_gene(value: Any) -> list[int] | None:
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
        gene = [int(item) for item in parsed]
        validate_gene(gene)
        return gene
    except (SyntaxError, ValueError, TypeError):
        return None


def known_genes(
    data_dir: Path, *, exclude_dir: Path | None = None
) -> dict[tuple[int, ...], list[str]]:
    found: dict[tuple[int, ...], list[str]] = {}
    repo_root = data_dir.parent
    for path in sorted(data_dir.glob("*.csv")):
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for line_number, row in enumerate(csv.DictReader(handle), start=2):
                    gene = _parse_gene(row.get("Net"))
                    if gene is not None:
                        reference = f"{path.relative_to(repo_root).as_posix()}:{line_number}"
                        found.setdefault(tuple(gene), []).append(reference)
        except (csv.Error, UnicodeDecodeError):
            continue

    for path in sorted((data_dir / "architectures").glob("**/*.json")):
        if exclude_dir is not None and path.resolve().is_relative_to(exclude_dir.resolve()):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        gene = _parse_gene(payload.get("gene") if isinstance(payload, dict) else payload)
        if gene is not None:
            found.setdefault(tuple(gene), []).append(path.relative_to(repo_root).as_posix())
    return found


def _signature_digest(signature: ArchitectureSignature) -> str:
    encoded = json.dumps(signature, separators=(",", ":"), sort_keys=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_digest(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary_rows(summary_path: Path) -> dict[int, dict[str, str]]:
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {int(row["gene_index"]): row for row in csv.DictReader(handle)}


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def recover(repo_root: Path, model_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    model_dir = (repo_root / model_dir).resolve()
    output_dir = (repo_root / output_dir).resolve()
    gene_dir = output_dir / "genes"
    gene_dir.mkdir(parents=True, exist_ok=True)

    summaries = _summary_rows(model_dir / "summary.csv")
    repository_genes = known_genes(repo_root / "data", exclude_dir=output_dir)
    records: list[dict[str, Any]] = []

    for model_path in sorted(model_dir.glob(MODEL_GLOB)):
        legacy_id = model_path.parent.name
        legacy_index = int(legacy_id.rsplit("_", 1)[1])
        sample_id = f"big_{legacy_index:04d}"
        signature = read_keras_signature(model_path)

        matches = [gene for gene in repository_genes if gene_signature(list(gene)) == signature]
        if len(matches) == 1:
            gene = list(matches[0])
            method = "exact_match_from_repository_gene"
            references = repository_genes[matches[0]]
        else:
            gene = canonical_gene(signature)
            method = "canonical_equivalent_from_keras_config"
            references = []

        if gene_signature(gene) != signature:
            raise AssertionError(f"Recovered gene does not match {model_path}")

        summary = summaries[legacy_index]
        record = {
            "sample_id": sample_id,
            "gene": gene,
            "source_collection": "DNNs/Full Trained/Big Models",
            "source_model": _relative(model_path, repo_root),
            "source_model_sha256": _file_digest(model_path),
            "architecture_signature_sha256": _signature_digest(signature),
            "legacy_model_id": legacy_id,
            "params": int(summary["params"]),
            "scale": 2,
            "stage1_best_val_psnr": float(summary["st1_best_val_psnr"]),
            "reconstruction_method": method,
            "repository_gene_references": references,
        }
        output_path = gene_dir / f"{sample_id}.json"
        output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
        records.append(record)

    metadata = {
        "schema_version": 1,
        "collection": "big_models",
        "model_count": len(records),
        "gene_length": GENE_LENGTH,
        "gene_value_range": [0, 7],
        "default_scale": 2,
        "source_summary": _relative(model_dir / "summary.csv", repo_root),
        "gene_json_dir": _relative(gene_dir, repo_root),
        "recovery_tool": "tools/recover_big_model_genes.py",
        "recovery_command": "python tools/recover_big_model_genes.py",
        "canonicalization": (
            "Prefer the sole exact repository gene with the same expanded topology. "
            "Otherwise choose the lexicographically smallest decoded 28-value gene "
            "that reproduces the serialized three-branch Keras topology. Duplicate "
            "channel, kernel, and repeat aliases use indices 0-3; identity uses [7, 0, 0]."
        ),
        "scientific_note": (
            "Canonical-equivalent genes reproduce the serialized architecture exactly, "
            "but must not be represented as the unavailable original genotype."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    manifest_fields = (
        "sample_id",
        "legacy_model_id",
        "gene_file",
        "source_model",
        "params",
        "scale",
        "stage1_best_val_psnr",
        "reconstruction_method",
        "architecture_signature_sha256",
        "source_model_sha256",
    )
    with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **{field: record[field] for field in manifest_fields if field in record},
                    "gene_file": _relative(gene_dir / f"{record['sample_id']}.json", repo_root),
                }
            )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover deterministic BASS gene JSON files from legacy Big Models archives."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--model-dir", type=Path, default=Path("DNNs/Full Trained/Big Models")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/architectures/big_models")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = recover(args.repo_root.resolve(), args.model_dir, args.output_dir)
    exact = sum(record["reconstruction_method"].startswith("exact") for record in records)
    canonical = len(records) - exact
    print(
        f"Recovered {len(records)} genes "
        f"({exact} exact repository match, {canonical} canonical equivalents)."
    )


if __name__ == "__main__":
    main()
