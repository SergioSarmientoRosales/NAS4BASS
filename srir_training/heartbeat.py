from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable


def now_payload(*, arch_id: str, gpu_id: str, event: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp": time.time(),
        "arch_id": arch_id,
        "gpu_id": gpu_id,
        "event": event,
    }
    payload.update(extra)
    return payload


def write_heartbeat(path: str | Path, *, arch_id: str, gpu_id: str, event: str, **extra: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = now_payload(arch_id=arch_id, gpu_id=gpu_id, event=event, **extra)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def heartbeat_age_sec(path: str | Path, *, fallback_start_time: float) -> float:
    try:
        return max(0.0, time.time() - Path(path).stat().st_mtime)
    except Exception:
        return max(0.0, time.time() - fallback_start_time)


def start_liveness_thread(
    path: str | Path,
    *,
    arch_id: str,
    gpu_id: str,
    interval_sec: float,
    logger: Callable[[str], None] | None = None,
):
    if interval_sec <= 0:
        return None, None

    stop_event = threading.Event()

    def loop() -> None:
        while not stop_event.wait(interval_sec):
            try:
                write_heartbeat(
                    path,
                    arch_id=arch_id,
                    gpu_id=gpu_id,
                    event="process_liveness",
                )
            except Exception as exc:
                if logger is not None:
                    logger(f"[HEARTBEAT] process liveness failed: {exc}")

    thread = threading.Thread(target=loop, name=f"heartbeat_{arch_id}", daemon=True)
    thread.start()
    return stop_event, thread


def stop_liveness_thread(stop_event, thread) -> None:
    try:
        if stop_event is not None:
            stop_event.set()
        if thread is not None:
            thread.join(timeout=2.0)
    except Exception:
        pass
