"""Protocollo di conferma delle catture SAR tra GUI e processi RX/logger."""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Any


class CaptureError(RuntimeError):
    """Una cattura non è stata accettata, completata o è stata annullata."""


@dataclass(frozen=True)
class CaptureTicket:
    """Riferimento a una richiesta prima che RX le assegni il session ID."""

    previous_session_id: int
    position_id: int
    position_mm: float | None
    position_microsteps: int | None


def _read_shared(value: Any) -> int:
    lock_getter = getattr(value, "get_lock", None)
    if callable(lock_getter):
        with lock_getter():
            return int(value.value)
    return int(value.value)


class CaptureSessionManager:
    """Serializza le catture e attende il flush del logger.

    ``cap_id`` è assegnato dal processo RX; ``cap_done_id`` viene aggiornato
    dal logger solo dopo chiusura e flush del file.  L'attesa quindi non
    dipende da transizioni ``cap_active`` che la GUI potrebbe non osservare.
    """

    def __init__(self, *, cmd_queue: Any, cap_id: Any, cap_done_id: Any, cap_result: Any) -> None:
        self._cmd_queue = cmd_queue
        self._cap_id = cap_id
        self._cap_done_id = cap_done_id
        self._cap_result = cap_result
        self._lock = threading.RLock()
        self._ticket: CaptureTicket | None = None

    @property
    def inflight(self) -> bool:
        with self._lock:
            self._reap_completed_locked()
            return self._ticket is not None

    def request(
        self,
        position_id: int,
        position_mm: float | None = None,
        position_microsteps: int | None = None,
    ) -> CaptureTicket:
        with self._lock:
            self._reap_completed_locked()
            if self._ticket is not None:
                raise CaptureError("Una cattura SAR è già in corso.")
            previous_session_id = _read_shared(self._cap_id)
            ticket = CaptureTicket(
                previous_session_id=previous_session_id,
                position_id=int(position_id),
                position_mm=None if position_mm is None else float(position_mm),
                position_microsteps=None if position_microsteps is None else int(position_microsteps),
            )
            metadata = {
                "carriage_position_mm": ticket.position_mm,
                "carriage_microsteps": ticket.position_microsteps,
            }
            try:
                self._cmd_queue.put_nowait(("CAPTURE", int(position_id), metadata))
            except queue.Full as exc:
                raise CaptureError("Coda catture piena: riprovare tra poco.") from exc
            except Exception as exc:
                raise CaptureError(f"Impossibile avviare la cattura: {exc}") from exc
            self._ticket = ticket
            return ticket

    def wait(self, ticket: CaptureTicket, timeout_seconds: float, cancel_event: threading.Event) -> None:
        deadline = time.monotonic() + max(0.001, float(timeout_seconds))
        session_id: int | None = None
        while True:
            if cancel_event.is_set():
                self.cancel()
                raise CaptureError("Cattura SAR annullata.")

            current_id = _read_shared(self._cap_id)
            if session_id is None and current_id != int(ticket.previous_session_id):
                session_id = current_id

            if session_id is not None and _read_shared(self._cap_done_id) == session_id:
                result = _read_shared(self._cap_result)
                with self._lock:
                    if self._ticket == ticket:
                        self._ticket = None
                if result == 1:
                    return
                if result == -1:
                    raise CaptureError("Cattura SAR annullata prima del completamento.")
                raise CaptureError(f"Cattura SAR terminata con stato non valido ({result}).")

            if time.monotonic() >= deadline:
                self.cancel()
                raise CaptureError("Timeout in attesa del completamento cattura SAR.")
            cancel_event.wait(0.01)

    def cancel(self) -> None:
        """Richiede a RX/logger di chiudere la cattura corrente.

        Il ticket rimane occupato finché il logger non pubblica il risultato:
        impedisce una nuova cattura che sovrascriverebbe i metadati condivisi.
        """
        with self._lock:
            if self._ticket is None:
                return
            try:
                self._cmd_queue.put_nowait(("CAPTURE_STOP",))
            except Exception:
                pass

    def _reap_completed_locked(self) -> None:
        ticket = self._ticket
        if ticket is None:
            return
        current_id = _read_shared(self._cap_id)
        if current_id == int(ticket.previous_session_id):
            return
        if _read_shared(self._cap_done_id) == current_id:
            self._ticket = None
