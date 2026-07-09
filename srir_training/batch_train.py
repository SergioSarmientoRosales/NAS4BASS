from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from srir_training.complexity import estimate_batch_size, inspect_model_complexity, round_batch
from srir_training.gpu import (
    available_cpu_count,
    discover_gpus,
    query_gpu_identity,
    query_gpu_memory_mb,
)
from srir_training.heartbeat import (
    heartbeat_age_sec,
    start_liveness_thread,
    stop_liveness_thread,
    write_heartbeat,
)


class DivergenceError(RuntimeError):
    pass


class DryRunError(RuntimeError):
    pass


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def load_gene_json(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if "gene" not in payload:
        raise ValueError(f"No 'gene' field found in {path}")
    gene = [int(value) for value in payload["gene"]]
    if len(gene) != 28:
        raise ValueError(f"Expected 28 BASS gene values in {path}, got {len(gene)}")
    payload["gene"] = gene
    return payload


def arch_id_from_gene_file(path: Path) -> str:
    return path.stem


def result_path(output_dir: Path, arch_id: str) -> Path:
    return output_dir / arch_id / "result.json"


def heartbeat_path(output_dir: Path, arch_id: str) -> Path:
    return output_dir / arch_id / "heartbeat.txt"


def is_complete(output_dir: Path, arch_id: str) -> bool:
    path = result_path(output_dir, arch_id)
    if not path.exists():
        return False
    try:
        return read_json(path).get("status") == "complete"
    except Exception:
        return False


def aggregate_results(output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("*/result.json")):
        try:
            payload = read_json(path)
        except Exception:
            continue
        rows.append(payload)

    if not rows:
        return

    preferred = [
        "arch_id",
        "status",
        "gpu_id",
        "attempts",
        "batch_size",
        "val_batch_size",
        "steps_per_execution",
        "best_val_psnr",
        "final_val_psnr",
        "final_val_ssim",
        "train_steps",
        "val_steps",
        "error_type",
        "error",
    ]
    extra = sorted({key for row in rows for key in row if key not in preferred})
    fields = preferred + extra
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def classify_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "resource exhausted" in text or "out of memory" in text or "oom" in text:
        return "oom"
    if "steps_per_execution" in text or "inaccessibletensor" in text:
        return "steps_per_execution"
    if "nan" in text or "inf" in text or "diverg" in text:
        return "divergence"
    if "xla" in text or "jit" in text:
        return "xla"
    if isinstance(exc, DryRunError):
        return "dry_run"
    return "unknown"


def terminate_process(proc: subprocess.Popen, log_path: Path, label: str) -> None:
    try:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{now()}] Terminating {label}\n")
    except Exception:
        pass

    try:
        if os.name == "nt":
            proc.terminate()
        else:
            os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            if os.name == "nt":
                proc.kill()
            else:
                os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def build_worker_command(args, *, gpu_id: str, gene_file: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "srir_training.batch_train",
        "--worker-mode",
        "--repo-dir",
        str(args.repo_dir),
        "--gene-file",
        str(gene_file),
        "--directory-train",
        args.directory_train,
        "--directory-val",
        args.directory_val,
        "--output-dir",
        str(args.output_dir),
        "--gpu-id",
        str(gpu_id),
        "--upscale-factor",
        str(args.upscale_factor),
        "--patch-size",
        str(args.patch_size),
        "--overlap-val",
        str(args.overlap_val),
        "--repeats-per-image",
        str(args.repeats_per_image),
        "--max-epochs",
        str(args.max_epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--adam-epsilon",
        str(args.adam_epsilon),
        "--min-lr",
        str(args.min_lr),
        "--lr-patience",
        str(args.lr_patience),
        "--lr-factor",
        str(args.lr_factor),
        "--es-patience",
        str(args.es_patience),
        "--es-min-delta",
        str(args.es_min_delta),
        "--initial-steps-per-execution",
        str(args.initial_steps_per_execution),
        "--min-steps-per-execution",
        str(args.min_steps_per_execution),
        "--precision",
        args.precision,
        "--global-clipnorm",
        str(args.global_clipnorm),
        "--vram-fraction",
        str(args.vram_fraction),
        "--min-batch",
        str(args.min_batch),
        "--max-batch",
        str(args.max_batch),
        "--max-val-batch",
        str(args.max_val_batch),
        "--validation-cache",
        args.validation_cache,
        "--max-retries",
        str(args.max_retries),
        "--retry-sleep-sec",
        str(args.retry_sleep_sec),
        "--heartbeat-every-steps",
        str(args.heartbeat_every_steps),
        "--process-heartbeat-sec",
        str(args.process_heartbeat_sec),
        "--dry-run-steps",
        str(args.dry_run_steps),
        "--dry-run-validation-steps",
        str(args.dry_run_validation_steps),
        "--dry-run-max-steps-per-execution",
        str(args.dry_run_max_steps_per_execution),
        "--base-seed",
        str(args.base_seed),
        "--verbose",
        str(args.verbose),
    ]
    if args.disable_xla:
        cmd.append("--disable-xla")
    if args.resume:
        cmd.append("--resume")
    if args.force:
        cmd.append("--force")
    if args.threads_per_worker is not None:
        cmd.extend(["--threads-per-worker", str(args.threads_per_worker)])
    return cmd


def parent_main(args) -> int:
    args.repo_dir = Path(args.repo_dir).resolve()
    args.output_dir = Path(args.output_dir).resolve()
    gene_dir = Path(args.gene_dir or args.repo_dir / "data" / "architectures" / "bass_50_sample" / "genes")
    gene_files = sorted(gene_dir.glob(args.gene_glob))
    if args.max_archs:
        gene_files = gene_files[: args.max_archs]
    if not gene_files:
        raise FileNotFoundError(f"No gene files found in {gene_dir} with glob {args.gene_glob}")

    gpus = discover_gpus(args.gpus)[: max(1, args.max_concurrent_gpus)]
    if args.threads_per_worker is None:
        args.threads_per_worker = max(1, available_cpu_count() // max(1, len(gpus)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for gene_file in gene_files:
        arch_id = arch_id_from_gene_file(gene_file)
        if is_complete(args.output_dir, arch_id) and not args.force:
            continue
        pending.append(gene_file)

    print(f"[PARENT] genes={len(gene_files)} pending={len(pending)} gpus={gpus}")
    running: dict[str, dict[str, Any]] = {}
    exit_code = 0

    try:
        while pending or running:
            for gpu_id in gpus:
                if not pending or gpu_id in running:
                    continue
                gene_file = pending.pop(0)
                arch_id = arch_id_from_gene_file(gene_file)
                arch_dir = args.output_dir / arch_id
                arch_dir.mkdir(parents=True, exist_ok=True)
                log_path = arch_dir / "worker_stdout.log"
                cmd = build_worker_command(args, gpu_id=gpu_id, gene_file=gene_file)
                log_handle = log_path.open("a", encoding="utf-8")
                log_handle.write(f"\n[{now()}] Launching {' '.join(cmd)}\n")
                log_handle.flush()
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(args.repo_dir),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=(os.name != "nt"),
                )
                running[gpu_id] = {
                    "proc": proc,
                    "gene_file": gene_file,
                    "arch_id": arch_id,
                    "start": time.time(),
                    "log_path": log_path,
                    "log_handle": log_handle,
                }
                print(f"[PARENT] started arch={arch_id} gpu={gpu_id} pid={proc.pid}")

            time.sleep(args.scheduler_poll_sec)

            for gpu_id, state in list(running.items()):
                proc = state["proc"]
                arch_id = state["arch_id"]
                if proc.poll() is not None:
                    state["log_handle"].close()
                    if proc.returncode != 0:
                        exit_code = 1
                        print(f"[PARENT] failed arch={arch_id} gpu={gpu_id} rc={proc.returncode}")
                    else:
                        print(f"[PARENT] complete arch={arch_id} gpu={gpu_id}")
                    del running[gpu_id]
                    aggregate_results(args.output_dir)
                    continue

                hb_path = heartbeat_path(args.output_dir, arch_id)
                age = heartbeat_age_sec(hb_path, fallback_start_time=state["start"])
                if age > args.heartbeat_timeout_sec:
                    exit_code = 1
                    terminate_process(proc, state["log_path"], f"{arch_id} on GPU {gpu_id}")
                    state["log_handle"].close()
                    write_json(
                        result_path(args.output_dir, arch_id),
                        {
                            "arch_id": arch_id,
                            "status": "failed",
                            "gpu_id": gpu_id,
                            "error_type": "heartbeat_timeout",
                            "error": f"No heartbeat for {age:.1f}s",
                            "finished_at": now(),
                        },
                    )
                    del running[gpu_id]
                    aggregate_results(args.output_dir)
    finally:
        for gpu_id, state in list(running.items()):
            terminate_process(state["proc"], state["log_path"], f"{state['arch_id']} on GPU {gpu_id}")
            state["log_handle"].close()

    aggregate_results(args.output_dir)
    return exit_code


def build_train_argv(args, *, arch_id: str, batch_size: int, steps_per_execution: int, xla_enabled: bool) -> list[str]:
    precision = "on" if args.precision == "mixed_float16" else "off"
    argv = [
        "--directory-train",
        args.directory_train,
        "--directory-val",
        args.directory_val,
        "--scale",
        str(args.upscale_factor),
        "--patch-size",
        str(args.patch_size),
        "--batch-size",
        str(batch_size),
        "--validation-overlap",
        str(args.overlap_val),
        "--validation-cache",
        args.validation_cache,
        "--epochs",
        str(args.max_epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--optimizer",
        "adamw",
        "--weight-decay",
        str(args.weight_decay),
        "--adam-epsilon",
        str(args.adam_epsilon),
        "--loss",
        "mse",
        "--global-clipnorm",
        str(args.global_clipnorm),
        "--lr-schedule",
        "plateau",
        "--reduce-lr-patience",
        str(args.lr_patience),
        "--reduce-lr-factor",
        str(args.lr_factor),
        "--min-learning-rate",
        str(args.min_lr),
        "--early-stopping-patience",
        str(args.es_patience),
        "--early-stopping-min-delta",
        str(args.es_min_delta),
        "--reduce-lr-min-delta",
        str(args.es_min_delta),
        "--bass-gene-file",
        args.gene_file,
        "--mixed-precision",
        precision,
        "--seed",
        str(args.base_seed + stable_arch_offset(arch_id)),
        "--steps-per-execution",
        str(steps_per_execution),
        "--run-name",
        arch_id,
        "--output-dir",
        args.output_dir,
    ]
    if xla_enabled:
        argv.append("--enable-xla")
    return argv


def stable_arch_offset(arch_id: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(arch_id)) % 1_000_000


def run_dry_run(train_argv: list[str], *, train_steps: int, val_steps: int) -> None:
    if train_steps <= 0 and val_steps <= 0:
        return

    import tensorflow as tf

    from srir_training.autosize import resolve_auto_data_config
    from srir_training.config import config_from_args, validate_config
    from srir_training.data import build_train_val_datasets
    from srir_training.models import load_or_build_model
    from srir_training.train import compile_model
    from srir_training.utils import set_global_seed

    cfg = config_from_args(train_argv)
    set_global_seed(cfg.runtime.seed, deterministic=False)
    resolve_auto_data_config(cfg.data, cfg.model, cfg.runtime)
    validate_config(cfg)
    train_ds, val_ds, _, _ = build_train_val_datasets(cfg.data, seed=cfg.runtime.seed)
    if cfg.data.validation_cache == "memory":
        val_ds = val_ds.cache()

    model = load_or_build_model(cfg.model, cfg.data)
    cfg.training.steps_per_execution = min(
        cfg.training.steps_per_execution,
        max(1, int(train_steps or cfg.training.steps_per_execution)),
    )
    compile_model(model, cfg)
    if train_steps > 0:
        model.fit(train_ds, epochs=1, steps_per_epoch=train_steps, verbose=0)
    if val_steps > 0:
        model.evaluate(val_ds, steps=val_steps, verbose=0)
    tf.keras.backend.clear_session()


def best_metric_from_history(run_dir: Path, key: str) -> float | None:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return None
    try:
        history = read_json(history_path)
        values = [float(value) for value in history.get(key, [])]
        values = [value for value in values if value == value and value not in {float("inf"), float("-inf")}]
        return max(values) if values else None
    except Exception:
        return None


def prepare_worker_environment(args, *, arch_id: str, attempt: int, steps_per_execution: int) -> Path:
    arch_dir = Path(args.output_dir) / arch_id
    hb_path = heartbeat_path(Path(args.output_dir), arch_id)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    if args.threads_per_worker:
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(args.threads_per_worker)
        os.environ["TF_NUM_INTEROP_THREADS"] = str(args.threads_per_worker)
        os.environ["OMP_NUM_THREADS"] = str(args.threads_per_worker)
    os.environ["SRIR_HEARTBEAT_PATH"] = str(hb_path)
    os.environ["SRIR_ARCH_ID"] = arch_id
    os.environ["SRIR_GPU_ID"] = str(args.gpu_id)
    os.environ["SRIR_HEARTBEAT_EVERY_STEPS"] = str(args.heartbeat_every_steps)
    os.environ["SRIR_BACKUP_NAME"] = f"backup_attempt{attempt:02d}_spe{steps_per_execution}"
    arch_dir.mkdir(parents=True, exist_ok=True)
    return hb_path


def estimate_initial_batch(args, *, arch_id: str) -> tuple[int, int, dict[str, Any]]:
    from srir_training.utils import configure_runtime
    from srir_training.models import load_bass_gene_file, build_bass_model

    configure_runtime(
        cpu=False,
        mixed_precision="on" if args.precision == "mixed_float16" else "off",
        enable_xla=not args.disable_xla,
    )

    try:
        memory = query_gpu_memory_mb(str(args.gpu_id))
        free_mb = memory.free_mb
    except Exception:
        memory = None
        free_mb = 8192

    gene = load_bass_gene_file(args.gene_file)
    model = build_bass_model(gene, scale=args.upscale_factor, channels=3)
    complexity = inspect_model_complexity(model)

    batch = estimate_batch_size(
        free_mb=free_mb,
        complexity=complexity,
        patch_size=args.patch_size,
        scale=args.upscale_factor,
        precision=args.precision,
        vram_fraction=args.vram_fraction,
        min_batch=args.min_batch,
        max_batch=args.max_batch,
    )
    val_batch = max(1, min(batch, args.max_val_batch))
    info = {
        "gpu_identity": query_gpu_identity(str(args.gpu_id)),
        "gpu_memory": asdict(memory) if memory is not None else None,
        "model_complexity": asdict(complexity),
        "arch_id": arch_id,
    }
    return batch, val_batch, info


def worker_main(args) -> int:
    arch_id = arch_id_from_gene_file(Path(args.gene_file))
    arch_dir = Path(args.output_dir) / arch_id
    arch_dir.mkdir(parents=True, exist_ok=True)
    log_path = arch_dir / "worker_events.log"

    def log(message: str) -> None:
        text = f"[{now()}] {message}"
        print(text, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    hb_path = prepare_worker_environment(args, arch_id=arch_id, attempt=0, steps_per_execution=args.initial_steps_per_execution)
    write_heartbeat(hb_path, arch_id=arch_id, gpu_id=str(args.gpu_id), event="worker_start")
    stop_event, hb_thread = start_liveness_thread(
        hb_path,
        arch_id=arch_id,
        gpu_id=str(args.gpu_id),
        interval_sec=args.process_heartbeat_sec,
        logger=log,
    )

    batch_size = args.max_batch
    val_batch_size = min(batch_size, args.max_val_batch)
    spe = max(args.min_steps_per_execution, args.initial_steps_per_execution)
    xla_enabled = not args.disable_xla
    attempts = 0
    metadata: dict[str, Any] = {}

    try:
        batch_size, val_batch_size, metadata = estimate_initial_batch(args, arch_id=arch_id)
        log(f"[INIT] batch={batch_size} val_batch={val_batch_size} spe={spe} xla={xla_enabled}")

        while attempts < args.max_retries:
            attempts += 1
            prepare_worker_environment(args, arch_id=arch_id, attempt=attempts, steps_per_execution=spe)
            train_argv = build_train_argv(
                args,
                arch_id=arch_id,
                batch_size=batch_size,
                steps_per_execution=spe,
                xla_enabled=xla_enabled,
            )

            try:
                if args.dry_run_steps or args.dry_run_validation_steps:
                    dry_argv = list(train_argv)
                    dry_spe = max(1, min(spe, args.dry_run_max_steps_per_execution))
                    dry_argv[dry_argv.index("--steps-per-execution") + 1] = str(dry_spe)
                    log(f"[DRY-RUN] train_steps={args.dry_run_steps} val_steps={args.dry_run_validation_steps}")
                    run_dry_run(
                        dry_argv,
                        train_steps=args.dry_run_steps,
                        val_steps=args.dry_run_validation_steps,
                    )

                from srir_training.train import main as train_main

                log(f"[TRAIN] attempt={attempts} batch={batch_size} spe={spe} xla={xla_enabled}")
                rc = train_main(train_argv)
                if rc != 0:
                    raise RuntimeError(f"train.main returned {rc}")

                run_dir = arch_dir
                best_val_psnr = best_metric_from_history(run_dir, "val_psnr")
                metrics_path = run_dir / "final_metrics.json"
                metrics = read_json(metrics_path) if metrics_path.exists() else {}
                final_val_psnr = metrics.get("psnr")
                if best_val_psnr is None and final_val_psnr is None:
                    raise DivergenceError("No finite validation PSNR found after training")

                payload = {
                    **metadata,
                    "arch_id": arch_id,
                    "status": "complete",
                    "gpu_id": args.gpu_id,
                    "attempts": attempts,
                    "batch_size": batch_size,
                    "val_batch_size": val_batch_size,
                    "steps_per_execution": spe,
                    "xla_enabled": xla_enabled,
                    "best_val_psnr": best_val_psnr,
                    "final_val_psnr": final_val_psnr,
                    "final_val_ssim": metrics.get("ssim"),
                    "run_dir": str(run_dir),
                    "finished_at": now(),
                }
                write_json(result_path(Path(args.output_dir), arch_id), payload)
                write_heartbeat(hb_path, arch_id=arch_id, gpu_id=str(args.gpu_id), event="worker_complete")
                return 0
            except BaseException as exc:
                err_type = classify_error(exc)
                log(f"[ERROR] attempt={attempts} type={err_type}: {exc}")
                log(traceback.format_exc())
                try:
                    import tensorflow as tf

                    tf.keras.backend.clear_session()
                except Exception:
                    pass
                if attempts >= args.max_retries:
                    write_json(
                        result_path(Path(args.output_dir), arch_id),
                        {
                            **metadata,
                            "arch_id": arch_id,
                            "status": "failed",
                            "gpu_id": args.gpu_id,
                            "attempts": attempts,
                            "batch_size": batch_size,
                            "val_batch_size": val_batch_size,
                            "steps_per_execution": spe,
                            "xla_enabled": xla_enabled,
                            "error_type": err_type,
                            "error": str(exc),
                            "finished_at": now(),
                        },
                    )
                    return 1

                if err_type == "oom" and batch_size > args.min_batch:
                    batch_size = round_batch(batch_size // 2, min_batch=args.min_batch, max_batch=args.max_batch)
                    val_batch_size = max(1, min(batch_size, args.max_val_batch))
                elif err_type == "steps_per_execution" and spe > args.min_steps_per_execution:
                    spe = max(args.min_steps_per_execution, spe // 2)
                elif err_type == "xla" and xla_enabled:
                    xla_enabled = False
                elif err_type == "divergence":
                    args.learning_rate *= 0.5
                else:
                    batch_size = round_batch(max(args.min_batch, batch_size // 2), min_batch=args.min_batch, max_batch=args.max_batch)
                    val_batch_size = max(1, min(batch_size, args.max_val_batch))

                time.sleep(args.retry_sleep_sec)
    finally:
        stop_liveness_thread(stop_event, hb_thread)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Train BASS architecture JSON files with one isolated worker per GPU."
    )
    parser.add_argument("--worker-mode", "--worker_mode", action="store_true")
    parser.add_argument("--gpu-id", "--gpu_id", dest="gpu_id", default=None)
    parser.add_argument("--repo-dir", "--repo_dir", dest="repo_dir", default=".")
    parser.add_argument("--gene-dir", "--gene_dir", dest="gene_dir", default=None)
    parser.add_argument("--gene-file", "--gene_file", dest="gene_file", default=None)
    parser.add_argument("--gene-glob", "--gene_glob", dest="gene_glob", default="bass_*.json")
    parser.add_argument("--max-archs", "--max_archs", dest="max_archs", type=int, default=50)
    parser.add_argument("--directory-train", "--directory_train", dest="directory_train", required=True)
    parser.add_argument("--directory-val", "--directory_val", dest="directory_val", required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--max-concurrent-gpus", "--max_concurrent_gpus", dest="max_concurrent_gpus", type=positive_int, default=4)
    parser.add_argument("--scheduler-poll-sec", "--scheduler_poll_sec", dest="scheduler_poll_sec", type=float, default=10.0)
    parser.add_argument("--heartbeat-timeout-sec", "--heartbeat_timeout_sec", dest="heartbeat_timeout_sec", type=float, default=3600.0)
    parser.add_argument("--upscale-factor", "--upscale_factor", dest="upscale_factor", type=int, choices=[2, 4], default=2)
    parser.add_argument("--patch-size", "--patch_size", dest="patch_size", type=int, default=64)
    parser.add_argument("--overlap-val", "--overlap_val", dest="overlap_val", type=float, default=0.1)
    parser.add_argument("--repeats-per-image", "--repeats_per_image", dest="repeats_per_image", type=positive_int, default=8)
    parser.add_argument("--max-epochs", "--max_epochs", dest="max_epochs", type=positive_int, default=200)
    parser.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1e-8)
    parser.add_argument("--adam-epsilon", "--adam_epsilon", dest="adam_epsilon", type=float, default=1e-7)
    parser.add_argument("--min-lr", "--min_lr", dest="min_lr", type=float, default=1e-6)
    parser.add_argument("--lr-patience", "--lr_patience", dest="lr_patience", type=positive_int, default=7)
    parser.add_argument("--lr-factor", "--lr_factor", dest="lr_factor", type=float, default=0.5)
    parser.add_argument("--es-patience", "--es_patience", dest="es_patience", type=positive_int, default=20)
    parser.add_argument("--es-min-delta", "--es_min_delta", dest="es_min_delta", type=float, default=0.02)
    parser.add_argument("--initial-steps-per-execution", "--initial_steps_per_execution", dest="initial_steps_per_execution", type=positive_int, default=128)
    parser.add_argument("--min-steps-per-execution", "--min_steps_per_execution", dest="min_steps_per_execution", type=positive_int, default=1)
    parser.add_argument("--disable-xla", "--disable_xla", dest="disable_xla", action="store_true")
    parser.add_argument("--precision", choices=["float32", "mixed_float16"], default="mixed_float16")
    parser.add_argument("--global-clipnorm", "--global_clipnorm", dest="global_clipnorm", type=float, default=5.0)
    parser.add_argument("--vram-fraction", "--vram_fraction", dest="vram_fraction", type=float, default=0.72)
    parser.add_argument("--min-batch", "--min_batch", dest="min_batch", type=positive_int, default=8)
    parser.add_argument("--max-batch", "--max_batch", dest="max_batch", type=positive_int, default=1024)
    parser.add_argument("--max-val-batch", "--max_val_batch", dest="max_val_batch", type=positive_int, default=256)
    parser.add_argument("--validation-cache", "--validation_cache", dest="validation_cache", choices=["memory", "disk", "none"], default="memory")
    parser.add_argument("--max-retries", "--max_retries", dest="max_retries", type=positive_int, default=6)
    parser.add_argument("--retry-sleep-sec", "--retry_sleep_sec", dest="retry_sleep_sec", type=float, default=5.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--heartbeat-every-steps", "--heartbeat_every_steps", dest="heartbeat_every_steps", type=positive_int, default=100)
    parser.add_argument("--process-heartbeat-sec", "--process_heartbeat_sec", dest="process_heartbeat_sec", type=float, default=60.0)
    parser.add_argument("--dry-run-steps", "--dry_run_steps", dest="dry_run_steps", type=int, default=2)
    parser.add_argument("--dry-run-validation-steps", "--dry_run_validation_steps", dest="dry_run_validation_steps", type=int, default=1)
    parser.add_argument("--dry-run-max-steps-per-execution", "--dry_run_max_steps_per_execution", dest="dry_run_max_steps_per_execution", type=positive_int, default=8)
    parser.add_argument("--base-seed", "--base_seed", dest="base_seed", type=int, default=20260703)
    parser.add_argument("--threads-per-worker", "--threads_per_worker", dest="threads_per_worker", type=int, default=None)
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
    args = parser.parse_args(argv)

    if args.patch_size != 64:
        raise ValueError("The multi-GPU BASS trainer fixes patch_size=64 for comparability")
    if args.patch_size % args.upscale_factor != 0:
        raise ValueError("patch_size must be divisible by upscale_factor")
    if args.worker_mode and not args.gpu_id:
        raise ValueError("--gpu-id is required in --worker-mode")
    if args.worker_mode and not args.gene_file:
        raise ValueError("--gene-file is required in --worker-mode")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_mode:
        return worker_main(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
