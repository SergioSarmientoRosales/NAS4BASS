from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRATEGIC_DIR = REPO_ROOT / "DNNs" / "Full Trained" / "strategic50"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "rework_50_architectures" / "manifests"
DEFAULT_SAMPLE_CSV = REPO_ROOT / "data" / "architectures" / "bass_50_sample_architectures.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def match(pattern: str, text: str, *, cast=str, default=None):
    found = re.search(pattern, text, flags=re.MULTILINE)
    return cast(found.group(1)) if found else default


def unique_values(rows: list[dict[str, object]], key: str) -> list[object]:
    values = {row[key] for row in rows if row.get(key) not in (None, "")}
    return sorted(values, key=str)


def audit_architecture(directory: Path) -> dict[str, object]:
    architecture_id = directory.name.removesuffix("_x2")
    run_dir = directory / architecture_id
    result_path = run_dir / "result.json"
    log_path = directory / f"{directory.name}.console.log"
    if not result_path.is_file() or not log_path.is_file():
        raise FileNotFoundError(f"Missing result or console log for {architecture_id}")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    log = log_path.read_text(encoding="utf-8", errors="replace")
    attempt = re.search(
        r"\[ATTEMPT 1\].*?batch=(\d+) val_batch=(\d+) lr=([0-9.eE+-]+) "
        r"spe=(\d+) xla=(True|False)",
        log,
    )
    if attempt is None:
        raise ValueError(f"Could not parse attempt configuration for {architecture_id}")

    reduction_matches = re.findall(
        r"Epoch (\d+): ReduceLROnPlateau reducing learning rate to ([0-9.eE+-]+)", log
    )
    early_stop_epoch = match(r"Epoch (\d+): early stopping", log, cast=int)
    return {
        "architecture_id": architecture_id,
        "status": result.get("status"),
        "params": int(result["params"]),
        "best_val_psnr": float(result["best_val_psnr"]),
        "final_val_psnr": float(result["final_val_psnr"]),
        "epochs_ran": int(result["epochs_ran"]),
        "max_epochs": match(r"max_epochs=(\d+)", log, cast=int),
        "batch_size": int(attempt.group(1)),
        "val_batch_size": int(attempt.group(2)),
        "learning_rate": float(attempt.group(3)),
        "steps_per_execution": int(attempt.group(4)),
        "xla_enabled": attempt.group(5) == "True",
        "mixed_precision_policy": match(r"\[MP\] precision policy: <DTypePolicy \"([^\"]+)\">", log),
        "train_images": match(r"train_images=(\d+)", log, cast=int),
        "train_patches": int(result["train_patches"]),
        "train_steps": int(result["train_steps"]),
        "val_images": match(r"val_images=(\d+)", log, cast=int),
        "val_patches": int(result["val_patches"]),
        "val_steps": int(result["val_steps"]),
        "gpu_name": result.get("gpu_identity", {}).get("name"),
        "gpu_memory_total": result.get("gpu_identity", {}).get("memory_total"),
        "attempts": int(result.get("attempts", 0)),
        "lr_reduction_epochs": ";".join(epoch for epoch, _ in reduction_matches),
        "lr_reduction_values": ";".join(value for _, value in reduction_matches),
        "early_stop_epoch": early_stop_epoch,
        "result_sha256": sha256(result_path),
        "console_log_sha256": sha256(log_path),
    }


def sample_diversity(sample_csv: Path) -> dict[str, object]:
    with sample_csv.open(newline="", encoding="utf-8-sig") as handle:
        sample_rows = list(csv.DictReader(handle))
    if len(sample_rows) != 50:
        raise ValueError(f"Expected 50 architecture-sample rows, found {len(sample_rows)}")

    genes = [[int(value) for value in ast.literal_eval(row["Net"])] for row in sample_rows]
    if any(len(gene) != 28 for gene in genes):
        raise ValueError("Every sampled BASS gene must contain 28 integers")
    distances = [
        sum(left != right for left, right in zip(genes[first], genes[second]))
        for first in range(len(genes) - 1)
        for second in range(first + 1, len(genes))
    ]
    channel_counts: dict[str, int] = {}
    bin_counts: dict[str, int] = {}
    operation_counts: dict[str, int] = {}
    for row in sample_rows:
        channel = str(int(row["channels"]))
        channel_counts[channel] = channel_counts.get(channel, 0) + 1
        complexity_bin = str(int(row["complexity_bin"]))
        bin_counts[complexity_bin] = bin_counts.get(complexity_bin, 0) + 1
        for operation, count in json.loads(row["op_counts_json"]).items():
            operation_counts[operation] = operation_counts.get(operation, 0) + int(count)

    return {
        "unique_genes": len({tuple(gene) for gene in genes}),
        "pair_count": len(distances),
        "gene_hamming_distance": {
            "min": min(distances),
            "median": float(statistics.median(distances)),
            "mean": float(statistics.mean(distances)),
            "max": max(distances),
        },
        "channel_counts": dict(sorted(channel_counts.items(), key=lambda item: int(item[0]))),
        "complexity_bin_counts": dict(sorted(bin_counts.items(), key=lambda item: int(item[0]))),
        "operation_counts": dict(sorted(operation_counts.items())),
        "sampling_methods": sorted({row["sampling_method"] for row in sample_rows}),
        "sampling_seeds": sorted({int(row["seed"]) for row in sample_rows}),
        "sample_csv": str(sample_csv.resolve()),
        "sample_csv_sha256": sha256(sample_csv),
    }


def audit(strategic_dir: Path, output_dir: Path, sample_csv: Path) -> dict[str, object]:
    directories = sorted(
        path for path in strategic_dir.glob("bass_[0-9][0-9][0-9][0-9]_x2") if path.is_dir()
    )
    rows = [audit_architecture(path) for path in directories]
    if len(rows) != 50:
        raise ValueError(f"Expected 50 full-training directories, found {len(rows)}")
    if any(row["status"] != "complete" for row in rows):
        raise ValueError("At least one full-training result is not complete")
    if any(row["early_stop_epoch"] != row["epochs_ran"] for row in rows):
        raise ValueError("Early-stop epoch and result epochs_ran disagree")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "expanded50_full_training_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    params = [int(row["params"]) for row in rows]
    psnr = [float(row["best_val_psnr"]) for row in rows]
    epochs = [int(row["epochs_ran"]) for row in rows]
    summary = {
        "architecture_count": len(rows),
        "complete_count": sum(row["status"] == "complete" for row in rows),
        "attempt_count_values": unique_values(rows, "attempts"),
        "params": {"min": min(params), "median": float(statistics.median(params)), "max": max(params)},
        "best_val_psnr": {"min": min(psnr), "median": float(statistics.median(psnr)), "max": max(psnr)},
        "epochs_ran": {"min": min(epochs), "median": float(statistics.median(epochs)), "max": max(epochs)},
        "uniform_or_observed_values": {
            key: unique_values(rows, key)
            for key in (
                "max_epochs",
                "learning_rate",
                "val_batch_size",
                "steps_per_execution",
                "xla_enabled",
                "mixed_precision_policy",
                "train_images",
                "train_patches",
                "val_images",
                "val_patches",
                "gpu_name",
                "gpu_memory_total",
            )
        },
        "adaptive_batch_size_values": unique_values(rows, "batch_size"),
        "architecture_diversity": sample_diversity(sample_csv),
        "source_directory": str(strategic_dir.resolve()),
        "audit_csv": str(csv_path.resolve()),
        "audit_csv_sha256": sha256(csv_path),
        "note": (
            "Callback hyperparameters that are not serialized in result.json are not inferred here; "
            "the audit preserves observed LR-reduction and early-stop epochs instead."
        ),
    }
    json_path = output_dir / "expanded50_full_training_audit.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the immutable evidence for all 50 full-training runs.")
    parser.add_argument("--strategic-dir", type=Path, default=DEFAULT_STRATEGIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-csv", type=Path, default=DEFAULT_SAMPLE_CSV)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(
        json.dumps(
            audit(
                args.strategic_dir.resolve(),
                args.output_dir.resolve(),
                args.sample_csv.resolve(),
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
