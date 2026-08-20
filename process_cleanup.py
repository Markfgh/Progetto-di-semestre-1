"""Primitive tolleranti agli errori per arrestare processi e code IPC.

Sono pensate per le routine di ``finally``: una risorsa già chiusa o un errore
di shutdown non deve impedire il rilascio delle risorse rimanenti.
"""

from __future__ import annotations

import threading
from typing import Any, Iterable


def _as_list(items: Iterable[Any]) -> list[Any]:
    return [item for item in items if item is not None]


def process_is_alive(proc: Any) -> bool:
    """Interroga un processo senza propagare errori di un handle già chiuso."""
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
    """Arresta processi in tre fasi: join, terminate dei superstiti, chiusura.

    Restituisce contatori diagnostici invece di sollevare eccezioni, perché è
    usata anche quando l'avvio parziale ha lasciato oggetti incompleti.
    """
    procs = _as_list(processes)
    join_attempts = 0
    terminated = 0
    closed = 0

    # Fase 1: concedere a tutti i processi il tempo di terminare da soli; un errore
    # su un handle non deve impedire il join degli altri processi ancora validi.
    for proc in procs:
        try:
            proc.join(timeout=float(graceful_timeout_s))
            join_attempts += 1
        except Exception:
            pass

    # Fase 2: soltanto i superstiti ricevono terminate(), così si preserva il
    # percorso cooperativo quando è possibile senza bloccare lo shutdown globale.
    for proc in procs:
        if not process_is_alive(proc):
            continue
        try:
            proc.terminate()
            terminated += 1
        except Exception:
            pass

    # Fase 3: dopo terminate() eseguiamo un secondo join per raccogliere il processo
    # terminato prima di rilasciare l'handle del sistema operativo.
    for proc in procs:
        try:
            proc.join(timeout=float(terminate_timeout_s))
            join_attempts += 1
        except Exception:
            pass

    if close_handles:
        for proc in procs:
            try:
                # multiprocessing permette close() solo a processo concluso; il controllo
                # difensivo evita che un singolo handle anomalo interrompa il cleanup.
                if not process_is_alive(proc) and hasattr(proc, "close"):
                    proc.close()
                    closed += 1
            except Exception:
                pass

    survivors = sum(1 for proc in procs if process_is_alive(proc))
    joined = len(procs) - survivors
    return {
        "joined": int(joined),
        "join_attempts": int(join_attempts),
        "terminated": int(terminated),
        "survivors": int(survivors),
        "closed": int(closed),
    }


def _join_queue_thread_with_timeout(queue_obj: Any, timeout_s: float) -> bool:
    completed = threading.Event()

    def _join() -> None:
        try:
            queue_obj.join_thread()
        except Exception:
            pass
        finally:
            completed.set()

    worker = threading.Thread(target=_join, name="queue-feeder-cleanup", daemon=True)
    worker.start()
    if completed.wait(timeout=max(0.0, float(timeout_s))):
        return True
    try:
        queue_obj.cancel_join_thread()
    except Exception:
        pass
    return False


def close_queues(queues: Iterable[Any], *, join_timeout_s: float = 0.2) -> int:
    """Chiude le code IPC e aspetta i loro feeder thread quando disponibili."""
    closed = 0
    for queue_obj in _as_list(queues):
        did_close = False
        try:
            # close() viene prima di join_thread(): segnala al feeder che non arriveranno
            # altri messaggi e permette di attendere la sua uscita senza lasciarlo sospeso.
            queue_obj.close()
            did_close = True
        except Exception:
            pass
        _join_queue_thread_with_timeout(queue_obj, join_timeout_s)
        if did_close:
            closed += 1
    return closed
