from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


@dataclass(frozen=True)
class GpuMemory:
    total_mb: int
    free_mb: int


def run_cmd(cmd: list[str], *, timeout: int = 30) -> tuple[int, str]:
    try:
        out = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        return 0, out.strip()
    except subprocess.CalledProcessError as exc:
        return int(exc.returncode), str(exc.output)
    except Exception as exc:
        return 1, str(exc)


def discover_gpus(gpus_arg: str) -> list[str]:
    if gpus_arg.lower() != "auto":
        gpus = [gpu.strip() for gpu in gpus_arg.split(",") if gpu.strip()]
        if not gpus:
            raise ValueError("--gpus must be 'auto' or a comma-separated list such as 0,1")
        return gpus

    rc, out = run_cmd(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        timeout=15,
    )
    if rc != 0:
        raise RuntimeError(f"Could not discover GPUs with nvidia-smi: {out}")

    gpus = [line.strip() for line in out.splitlines() if line.strip()]
    if not gpus:
        raise RuntimeError("No GPUs discovered by nvidia-smi")
    return gpus


def query_gpu_memory_mb(gpu_id: str) -> GpuMemory:
    rc, out = run_cmd(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        timeout=15,
    )
    if rc != 0:
        raise RuntimeError(f"Could not query GPU memory for GPU {gpu_id}: {out}")

    parts = [part.strip() for part in out.splitlines()[0].split(",")]
    if len(parts) < 2:
        raise RuntimeError(f"Unexpected nvidia-smi output for GPU {gpu_id}: {out}")
    return GpuMemory(total_mb=int(float(parts[0])), free_mb=int(float(parts[1])))


def query_gpu_identity(gpu_id: str) -> dict[str, str]:
    rc, out = run_cmd(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.free",
            "--format=csv,noheader",
        ],
        timeout=15,
    )
    if rc != 0:
        return {"error": out}

    keys = ["index", "uuid", "pci_bus_id", "name", "memory_total", "memory_free"]
    parts = [part.strip() for part in out.splitlines()[0].split(",")]
    return {key: parts[idx] if idx < len(parts) else "" for idx, key in enumerate(keys)}


def available_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)
