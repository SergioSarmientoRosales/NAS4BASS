from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import platform
import random
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "rework_50_architectures" / "raw"
DEFAULT_METRICS = (
    "l2_norm",
    "param_score",
    "nwot",
    "zen",
    "zico",
    "jacob",
    "synflow",
    "grad_norm",
    "plain",
    "snip",
    "fisher",
    "grasp",
)

RAW_FIELDS = (
    "scenario",
    "architecture_id",
    "architecture_index",
    "gene_sha256",
    "seed",
    "proxy",
    "valid_psnr",
    "params_expected",
    "params_built",
    "raw_score",
    "validity_flag",
    "status",
    "error_type",
    "error_message",
    "build_time_ms",
    "proxy_time_ms",
    "total_time_ms",
    "input_seed",
    "lr_patch_size",
    "batch_size",
    "upscale_factor",
    "tensorflow_version",
    "keras_version",
    "device_summary",
    "completed_at_utc",
)


def parse_seed_spec(value: str) -> list[int]:
    seeds: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if ":" in token:
            fields = [int(item) for item in token.split(":")]
            if len(fields) not in (2, 3):
                raise ValueError(f"Invalid seed range: {token!r}")
            start, stop = fields[:2]
            step = fields[2] if len(fields) == 3 else 1
            if step == 0:
                raise ValueError("Seed range step cannot be zero")
            inclusive_stop = stop + (1 if step > 0 else -1)
            seeds.extend(range(start, inclusive_stop, step))
        else:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("At least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seed specification contains duplicates")
    return seeds


def parse_metrics(value: str) -> list[str]:
    metrics = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(metrics) - set(DEFAULT_METRICS))
    if unknown:
        raise ValueError(f"Unknown proxies: {unknown}; available={list(DEFAULT_METRICS)}")
    if not metrics:
        raise ValueError("At least one proxy is required")
    return metrics


def load_manifest(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty architecture manifest: {path}")

    parsed: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    seen_genes: set[str] = set()
    for row in rows:
        architecture_id = row["architecture_id"].strip()
        gene = [int(item) for item in ast.literal_eval(row["gene_json"])]
        if len(gene) != 28:
            raise ValueError(f"{architecture_id}: expected 28 gene values, received {len(gene)}")
        gene_digest = hashlib.sha256(
            json.dumps(gene, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        if gene_digest != row["gene_sha256"]:
            raise ValueError(f"{architecture_id}: gene digest does not match manifest")
        if architecture_id in seen_ids or gene_digest in seen_genes:
            raise ValueError(f"Duplicate architecture ID or gene in manifest: {architecture_id}")
        seen_ids.add(architecture_id)
        seen_genes.add(gene_digest)
        parsed.append(
            {
                **row,
                "architecture_id": architecture_id,
                "architecture_index": int(row["architecture_index"]),
                "gene": gene,
                "valid_psnr": float(row["valid_psnr"]),
                "params_real": int(float(row["params_real"])),
            }
        )
    return sorted(parsed, key=lambda item: int(item["architecture_index"]))


def load_registry() -> dict[str, Callable]:
    from evaluators.metrics.fisher import compute_fisher
    from evaluators.metrics.grad_norm import compute_grad_norm
    from evaluators.metrics.grasp import compute_grasp
    from evaluators.metrics.jacob_cov import compute_jacob_cov
    from evaluators.metrics.l2_norm import compute_l2_norm
    from evaluators.metrics.nwot import compute_nwot
    from evaluators.metrics.plain import compute_plain
    from evaluators.metrics.snip import compute_snip
    from evaluators.metrics.synflow import compute_synflow_raw
    from evaluators.metrics.zen import compute_zen_score
    from evaluators.metrics.zico import compute_zico

    return {
        "fisher": compute_fisher,
        "grad_norm": compute_grad_norm,
        "grasp": compute_grasp,
        "jacob": compute_jacob_cov,
        "l2_norm": compute_l2_norm,
        "nwot": compute_nwot,
        "plain": compute_plain,
        "snip": compute_snip,
        "synflow": compute_synflow_raw,
        "zen": compute_zen_score,
        "zico": compute_zico,
    }


def configure_environment(cpu_only: bool) -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def set_global_seed(seed: int, tf, keras) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)


def completed_keys(path: Path, retry_errors: bool) -> set[tuple[str, str, int, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str, int, str]] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if retry_errors and row.get("status") == "error":
                continue
            keys.add(
                (
                    row["scenario"],
                    row["architecture_id"],
                    int(row["seed"]),
                    row["proxy"],
                )
            )
    return keys


def append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _versions(tf, keras) -> tuple[str, str]:
    keras_version = getattr(keras, "__version__", "unknown")
    return str(tf.__version__), str(keras_version)


def _device_summary(tf) -> str:
    devices = []
    for device_type in ("GPU", "CPU"):
        for device in tf.config.list_physical_devices(device_type):
            devices.append({"type": device_type, "name": device.name})
    return json.dumps(devices, separators=(",", ":"))


def write_run_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    seeds: list[int],
    metrics: list[str],
    manifest: Path,
    tf,
    keras,
) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "tensorflow": str(tf.__version__),
        "keras": str(getattr(keras, "__version__", "unknown")),
        "devices": json.loads(_device_summary(tf)),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "seeds": seeds,
        "proxies": metrics,
        "transformations_applied_during_analysis": [
            "raw",
            "div_params",
            "neg_raw",
            "neg_div_params",
        ],
        "param_score_definition": "positive trainable and non-trainable model parameter count",
        "input_protocol": {
            "type": "fixed synthetic SR tensors",
            "input_seed": args.input_seed,
            "lr_patch_size": args.lr_patch_size,
            "batch_size": args.batch_size,
            "upscale_factor": args.upscale_factor,
            "loss": "mean_squared_error",
        },
        "determinism": {
            "python_random": True,
            "numpy": True,
            "tensorflow": True,
            "keras": True,
            "TF_DETERMINISTIC_OPS": os.environ.get("TF_DETERMINISTIC_OPS"),
            "TF_CUDNN_DETERMINISTIC": os.environ.get("TF_CUDNN_DETERMINISTIC"),
            "paired_model_initialization_across_proxies": True,
            "fresh_model_per_architecture_seed": True,
            "weight_snapshot_restored_before_and_after_each_proxy": True,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    configure_environment(args.cpu_only)
    import keras
    import tensorflow as tf

    from search_space.model_builder import get_model
    from search_space.search_space import decode

    if args.cpu_only:
        tf.config.set_visible_devices([], "GPU")
    else:
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    seeds = parse_seed_spec(args.seeds)
    metrics = parse_metrics(args.metrics)
    architectures = load_manifest(args.manifest)
    if args.max_architectures is not None:
        architectures = architectures[: args.max_architectures]
    if args.max_seeds is not None:
        seeds = seeds[: args.max_seeds]

    registry = load_registry()
    output_csv = args.output_dir / f"{args.scenario}_raw_scores.csv"
    metadata_path = args.output_dir / f"{args.scenario}_run_metadata.json"
    done = completed_keys(output_csv, retry_errors=args.retry_errors)
    write_run_metadata(
        metadata_path,
        args=args,
        seeds=seeds,
        metrics=metrics,
        manifest=args.manifest,
        tf=tf,
        keras=keras,
    )

    input_generator = tf.random.Generator.from_seed(args.input_seed)
    fixed_lr = input_generator.uniform(
        (args.batch_size, args.lr_patch_size, args.lr_patch_size, 3),
        minval=0.0,
        maxval=1.0,
        dtype=tf.float32,
    )
    hr_patch = args.lr_patch_size * args.upscale_factor
    fixed_hr = input_generator.uniform(
        (args.batch_size, hr_patch, hr_patch, 3),
        minval=0.0,
        maxval=1.0,
        dtype=tf.float32,
    )
    loss_fn = tf.keras.losses.MeanSquaredError()
    tf_version, keras_version = _versions(tf, keras)
    device_summary = _device_summary(tf)

    attempted = skipped = failures = 0
    total = len(architectures) * len(seeds) * len(metrics)
    print(
        f"[REWORK] scenario={args.scenario} architectures={len(architectures)} "
        f"seeds={len(seeds)} proxies={len(metrics)} combinations={total}"
    )

    for architecture in architectures:
        arch_id = str(architecture["architecture_id"])
        for seed in seeds:
            pending_metrics = []
            for metric_name in metrics:
                if (args.scenario, arch_id, seed, metric_name) in done:
                    skipped += 1
                else:
                    pending_metrics.append(metric_name)
            if not pending_metrics:
                continue

            tf.keras.backend.clear_session()
            set_global_seed(seed, tf, keras)
            model = None
            params_built = 0
            build_time_ms = math.nan
            build_error: Exception | None = None
            initial_values = []
            try:
                build_start = time.perf_counter()
                genotype = decode(list(architecture["gene"]))
                model = get_model(genotype, upscale_factor=args.upscale_factor)
                _ = model(fixed_lr[:1], training=False)
                params_built = int(model.count_params())
                build_time_ms = (time.perf_counter() - build_start) * 1000.0

                params_expected = int(architecture["params_real"])
                if params_built != params_expected and not args.allow_param_mismatch:
                    raise ValueError(
                        f"Parameter mismatch for {arch_id}: manifest={params_expected}, "
                        f"current builder={params_built}"
                    )
                initial_values = [tf.identity(variable) for variable in model.weights]
            except Exception as exc:
                build_error = exc
                if args.verbose_errors:
                    traceback.print_exc()

            for metric_name in pending_metrics:
                attempted += 1
                status = "error"
                validity = False
                raw_score = math.nan
                proxy_time_ms = math.nan
                error_type = error_message = ""
                try:
                    if build_error is not None:
                        raise build_error
                    for variable, initial_value in zip(model.weights, initial_values):
                        variable.assign(initial_value)
                    set_global_seed(seed, tf, keras)
                    proxy_start = time.perf_counter()
                    if metric_name == "param_score":
                        raw_score = float(params_built)
                    else:
                        raw_score = float(
                            registry[metric_name](
                                model=model,
                                inputs=fixed_lr,
                                targets=fixed_hr,
                                loss_fn=loss_fn,
                            )
                        )
                    proxy_time_ms = (time.perf_counter() - proxy_start) * 1000.0
                    validity = math.isfinite(raw_score)
                    status = "ok" if validity else "invalid"
                    if not validity:
                        error_type = "NonFiniteScore"
                        error_message = f"Proxy returned {raw_score!r}"
                except Exception as exc:  # every failure is retained in the raw audit table
                    failures += 1
                    error_type = type(exc).__name__
                    error_message = str(exc)
                    if args.verbose_errors:
                        traceback.print_exc()
                finally:
                    if model is not None and initial_values:
                        for variable, initial_value in zip(model.weights, initial_values):
                            variable.assign(initial_value)
                    total_time_ms = (
                        build_time_ms + proxy_time_ms
                        if math.isfinite(build_time_ms) and math.isfinite(proxy_time_ms)
                        else math.nan
                    )
                    row = {
                        "scenario": args.scenario,
                        "architecture_id": arch_id,
                        "architecture_index": architecture["architecture_index"],
                        "gene_sha256": architecture["gene_sha256"],
                        "seed": seed,
                        "proxy": metric_name,
                        "valid_psnr": architecture["valid_psnr"],
                        "params_expected": architecture["params_real"],
                        "params_built": params_built,
                        "raw_score": raw_score,
                        "validity_flag": validity,
                        "status": status,
                        "error_type": error_type,
                        "error_message": error_message,
                        "build_time_ms": build_time_ms,
                        "proxy_time_ms": proxy_time_ms,
                        "total_time_ms": total_time_ms,
                        "input_seed": args.input_seed,
                        "lr_patch_size": args.lr_patch_size,
                        "batch_size": args.batch_size,
                        "upscale_factor": args.upscale_factor,
                        "tensorflow_version": tf_version,
                        "keras_version": keras_version,
                        "device_summary": device_summary,
                        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    append_row(output_csv, row)

                print(
                    f"[{attempted + skipped}/{total}] {arch_id} seed={seed} "
                    f"proxy={metric_name} status={status} score={raw_score}"
                )
                if status == "error" and args.fail_on_error:
                    raise RuntimeError(f"Benchmark stopped after {arch_id}/{seed}/{metric_name}: {error_message}")

            if model is not None:
                del model
            tf.keras.backend.clear_session()

    print(
        f"[REWORK] complete attempted={attempted} skipped={skipped} "
        f"errors={failures} output={output_csv}"
    )
    return 1 if failures and args.fail_on_error else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the controlled, resumable zero-cost benchmark for one architecture scenario."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scenario", choices=("expanded50", "pareto20"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="1:30", help="Comma-separated seeds or inclusive ranges")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--input-seed", type=int, default=12345)
    parser.add_argument("--lr-patch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--upscale-factor", type=int, default=2)
    parser.add_argument("--max-architectures", type=int)
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--allow-param-mismatch", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--verbose-errors", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
