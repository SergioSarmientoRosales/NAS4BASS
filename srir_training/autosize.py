from __future__ import annotations

import subprocess
from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class AutoSizeDecision:
    patch_size: int
    batch_size: int
    gpu_memory_mib: int | None
    reason: str


def detect_gpu_memory_mib(*, cpu: bool = False) -> int | None:
    if cpu:
        return None

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return None

    detected: list[int] = []
    for gpu in gpus:
        try:
            details = tf.config.experimental.get_device_details(gpu)
        except Exception:
            details = {}
        memory = details.get("memory_size") or details.get("device_memory_size")
        if memory:
            detected.append(max(1, int(memory) // (1024 * 1024)))

    if detected:
        return min(detected)

    return _detect_nvidia_smi_memory_mib()


def _detect_nvidia_smi_memory_mib() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=5,
        )
    except Exception:
        return None

    values = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(float(line)))
        except ValueError:
            continue
    return min(values) if values else None


def resolve_auto_data_config(cfg, model_cfg, runtime_cfg) -> AutoSizeDecision:
    memory_mib = detect_gpu_memory_mib(cpu=runtime_cfg.cpu)

    if cfg.patch_size is None:
        cfg.patch_size = _choose_patch_size(
            scale=cfg.scale,
            min_patch=cfg.min_patch_size,
            max_patch=cfg.max_patch_size,
            memory_mib=memory_mib,
        )

    if cfg.batch_size is None:
        cfg.batch_size = _choose_batch_size(
            patch_size=cfg.patch_size,
            max_batch_size=cfg.max_batch_size,
            memory_mib=memory_mib,
            filters=getattr(model_cfg, "num_filters", 64),
            blocks=getattr(model_cfg, "num_res_blocks", 8),
            mixed_precision=runtime_cfg.mixed_precision,
            cpu=runtime_cfg.cpu,
        )

    reason = "CPU or unknown GPU memory" if memory_mib is None else f"GPU memory {memory_mib} MiB"
    print(
        "[AUTO-SIZE] {0}: patch_size={1}, batch_size={2}".format(
            reason,
            cfg.patch_size,
            cfg.batch_size,
        )
    )
    return AutoSizeDecision(
        patch_size=int(cfg.patch_size),
        batch_size=int(cfg.batch_size),
        gpu_memory_mib=memory_mib,
        reason=reason,
    )


def _choose_patch_size(
    *,
    scale: int,
    min_patch: int,
    max_patch: int,
    memory_mib: int | None,
) -> int:
    if memory_mib is None:
        target = min(max_patch, 96)
    elif memory_mib <= 4096:
        target = 64
    elif memory_mib <= 8192:
        target = 96
    elif memory_mib <= 16384:
        target = 96
    else:
        target = min(max_patch, 128)

    candidates = _patch_candidates(scale=scale, min_patch=min_patch, max_patch=max_patch)
    safe = [patch for patch in candidates if patch <= target]
    if safe:
        return max(safe)
    return min(candidates)


def _patch_candidates(*, scale: int, min_patch: int, max_patch: int) -> list[int]:
    base = [48, 64, 72, 96, 120, 128, 144, 160, 192, 256]
    candidates = [
        patch
        for patch in base
        if min_patch <= patch <= max_patch and patch % scale == 0
    ]
    if candidates:
        return candidates

    first = ((min_patch + scale - 1) // scale) * scale
    if first <= max_patch:
        return [first]
    raise ValueError("No valid auto patch size is divisible by scale within min/max bounds")


def _choose_batch_size(
    *,
    patch_size: int,
    max_batch_size: int,
    memory_mib: int | None,
    filters: int,
    blocks: int,
    mixed_precision: str,
    cpu: bool,
) -> int:
    if cpu or memory_mib is None:
        base = 4
    elif memory_mib <= 4096:
        base = 2
    elif memory_mib <= 6144:
        base = 4
    elif memory_mib <= 8192:
        base = 6
    elif memory_mib <= 12288:
        base = 8
    elif memory_mib <= 16384:
        base = 12
    else:
        base = 16

    patch_factor = (96 * 96) / float(patch_size * patch_size)
    width_factor = 64.0 / max(float(filters), 1.0)
    depth_factor = 8.0 / max(float(blocks), 1.0)
    precision_factor = (
        1.25
        if memory_mib is not None and mixed_precision in {"auto", "on"} and not cpu
        else 1.0
    )

    estimated = int(base * patch_factor * min(width_factor, 1.5) * min(depth_factor, 1.5) * precision_factor)
    return max(1, min(max_batch_size, max(1, estimated)))
