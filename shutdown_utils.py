from __future__ import annotations

from typing import Any, Iterable


def _as_list(items: Iterable[Any]) -> list[Any]:
    return [item for item in items if item is not None]


def process_is_alive(proc: Any) -> bool:
    try:
        return bool(proc.is_alive())
    except Exception:
        return False


def cleanup_processes(
    processes: Iterable[Any],
    *,
    graceful_timeout_s: float = 0.4,
    terminate_timeout_s: float = 0.2,
    close_handles: bool = True,
) -> dict[str, int]:
    procs = _as_list(processes)
    joined = 0
    terminated = 0
    closed = 0

    for proc in procs:
        try:
            proc.join(timeout=float(graceful_timeout_s))
            joined += 1
        except Exception:
            pass

    for proc in procs:
        if not process_is_alive(proc):
            continue
        try:
            proc.terminate()
            terminated += 1
        except Exception:
            pass

    for proc in procs:
        try:
            proc.join(timeout=float(terminate_timeout_s))
            joined += 1
        except Exception:
            pass

    if close_handles:
        for proc in procs:
            try:
                if not process_is_alive(proc) and hasattr(proc, "close"):
                    proc.close()
                    closed += 1
            except Exception:
                pass

    return {"joined": joined, "terminated": terminated, "closed": closed}


def close_queues(queues: Iterable[Any]) -> int:
    closed = 0
    for queue_obj in _as_list(queues):
        did_close = False
        try:
            queue_obj.close()
            did_close = True
        except Exception:
            pass
        try:
            queue_obj.join_thread()
        except Exception:
            pass
        if did_close:
            closed += 1
    return closed
