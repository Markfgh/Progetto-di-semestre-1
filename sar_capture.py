"""Protocollo di conferma delle catture SAR tra GUI e processi RX/logger."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Real
import queue
import threading
import time
from typing import Any, Mapping


class CaptureError(RuntimeError):
    """Una cattura non è stata accettata, completata o è stata annullata."""


CAPTURE_KIND_SAR = "sar"
CAPTURE_KIND_MANUAL_NO_STAGE = "manual_no_stage"
CAPTURE_KINDS = frozenset({CAPTURE_KIND_SAR, CAPTURE_KIND_MANUAL_NO_STAGE})


def _strict_integer(value: Any, *, field_name: str) -> int:
    """Convert an integer-valued scalar without truncating fractions or booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CaptureError(f"{field_name} must be an integer")
    value_f = float(value)
    if not math.isfinite(value_f) or not value_f.is_integer():
        raise CaptureError(f"{field_name} must be an integer")
    return int(value)


@dataclass(frozen=True)
class CaptureMetadataStore:
    """Piccolo blob JSON condiviso tra RX e logger per una cattura.

    Il ``session_id`` interno è pubblicato insieme al blob sotto ``lock``.
    RX scrive il blob prima di rendere visibile il nuovo ``cap_id``; il logger
    accetta quindi solo il metadata che corrisponde alla sessione osservata.
    """

    buffer: Any
    byte_count: Any
    session_id: Any
    lock: Any


def normalize_capture_metadata(capture_id: int, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Produce il blob canonico associato a un identificatore di cattura.

    ``capture_id`` identifica il file/cattura; ``acquisition_index`` è
    l'ordine temporale. ``position`` è l'indice della posizione lineare.
    """

    if metadata is None:
        raw: Mapping[str, Any] = {}
    elif isinstance(metadata, Mapping):
        raw = metadata
    else:
        raise CaptureError("capture metadata must be a mapping")

    try:
        capture_id_i = _strict_integer(capture_id, field_name="capture_id")
    except CaptureError as exc:
        raise CaptureError("capture_id must be a non-negative integer") from exc
    # Mantiene leggibili gli storici ``position_id`` lineari negativi quando
    # non è stato fornito un identificatore di cattura esplicito.
    if capture_id_i < 0 and "capture_id" in raw:
        raise CaptureError("capture_id must be a non-negative integer")
    supplied_capture_id = raw.get("capture_id", capture_id_i)
    try:
        supplied_capture_id_i = _strict_integer(supplied_capture_id, field_name="capture_id")
        acquisition_index = _strict_integer(
            raw.get("acquisition_index", capture_id_i), field_name="acquisition_index"
        )
        legacy_position = _strict_integer(raw.get("position", capture_id_i), field_name="position")
    except CaptureError as exc:
        raise CaptureError("capture_id, acquisition_index and position must be integers") from exc
    if supplied_capture_id_i != capture_id_i:
        raise CaptureError("capture metadata capture_id does not match the CAPTURE command")
    if acquisition_index < 0:
        raise CaptureError("acquisition_index must be non-negative")

    result: dict[str, Any] = {
        "capture_id": capture_id_i,
        "acquisition_index": acquisition_index,
        "position": legacy_position,
    }
    if raw.get("carriage_position_mm") is not None:
        try:
            carriage_position_mm = float(raw["carriage_position_mm"])
        except (TypeError, ValueError) as exc:
            raise CaptureError("carriage_position_mm must be numeric") from exc
        if not math.isfinite(carriage_position_mm):
            raise CaptureError("carriage_position_mm must be finite")
        result["carriage_position_mm"] = carriage_position_mm
    if raw.get("carriage_microsteps") is not None:
        result["carriage_microsteps"] = _strict_integer(
            raw["carriage_microsteps"], field_name="carriage_microsteps"
        )
    default_kind = (
        CAPTURE_KIND_SAR
        if result.get("carriage_position_mm") is not None
        else CAPTURE_KIND_MANUAL_NO_STAGE
    )
    capture_kind = str(raw.get("capture_kind", default_kind)).strip().lower()
    if capture_kind not in CAPTURE_KINDS:
        allowed = ", ".join(sorted(CAPTURE_KINDS))
        raise CaptureError(f"capture_kind must be one of: {allowed}")
    if capture_kind == CAPTURE_KIND_SAR and result.get("carriage_position_mm") is None:
        raise CaptureError("SAR capture metadata requires a finite carriage_position_mm")
    result["capture_kind"] = capture_kind
    return result


def write_capture_metadata(store: CaptureMetadataStore, session_id: int, metadata: Mapping[str, Any]) -> None:
    """Pubblica un blob metadata in modo atomico per una sessione RX."""

    try:
        payload = json.dumps(dict(metadata), allow_nan=False, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CaptureError("capture metadata must be JSON serializable") from exc
    if len(payload) > len(store.buffer):
        raise CaptureError(f"capture metadata exceeds shared buffer ({len(payload)} bytes)")
    with store.lock:
        store.buffer[: len(payload)] = payload
        store.byte_count.value = int(len(payload))
        # Scrivere il session ID per ultimo rende il blob completo prima che
        # il logger possa accettarlo per la sessione richiesta.
        store.session_id.value = int(session_id)


def read_capture_metadata(store: CaptureMetadataStore, session_id: int) -> dict[str, Any] | None:
    """Restituisce il metadata solo se appartiene alla sessione richiesta."""

    with store.lock:
        if int(store.session_id.value) != int(session_id):
            return None
        byte_count = int(store.byte_count.value)
        if byte_count < 0 or byte_count > len(store.buffer):
            raise CaptureError("shared capture metadata has an invalid length")
        payload = bytes(store.buffer[:byte_count])
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaptureError("shared capture metadata is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise CaptureError("shared capture metadata must be a JSON mapping")
    return decoded


@dataclass(frozen=True)
class CaptureTicket:
    """Riferimento a una richiesta prima che RX le assegni il session ID."""

    previous_session_id: int
    position_id: int
    position_mm: float | None
    position_microsteps: int | None
    capture_id: int
    acquisition_index: int
    capture_kind: str

    @property
    def position(self) -> int:
        """Indice della posizione lineare associata alla cattura."""
        return int(self.position_id)


def _read_shared(value: Any) -> int:
    lock_getter = getattr(value, "get_lock", None)
    if callable(lock_getter):
        with lock_getter():
            return int(value.value)
    return int(value.value)


class CaptureSessionManager:
    """Serializza i comandi di cattura e ne attende l'esito dal logger.

    Il ``session_id`` evita di scambiare una conferma tardiva della cattura
    precedente con il completamento della richiesta corrente.
    """

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
            return self._ticket is not None

    def request(
        self,
        position_id: int | None = None,
        position_mm: float | None = None,
        position_microsteps: int | None = None,
        *,
        capture_id: int | None = None,
        acquisition_index: int | None = None,
        capture_kind: str | None = None,
    ) -> CaptureTicket:
        """Invia una richiesta CAPTURE e restituisce il ticket da attendere."""
        with self._lock:
            if self._ticket is not None:
                raise CaptureError("Una cattura SAR è già in corso.")
            if position_id is None and capture_id is None:
                raise CaptureError("A capture requires capture_id or legacy position_id.")
            legacy_position = _strict_integer(
                capture_id if position_id is None else position_id,
                field_name="position_id",
            )
            effective_capture_id = _strict_integer(
                legacy_position if capture_id is None else capture_id,
                field_name="capture_id",
            )
            if effective_capture_id < 0 and capture_id is not None:
                raise CaptureError("capture_id must be non-negative")
            effective_acquisition_index = _strict_integer(
                effective_capture_id if acquisition_index is None else acquisition_index,
                field_name="acquisition_index",
            )
            if effective_acquisition_index < 0:
                raise CaptureError("acquisition_index must be non-negative")
            effective_capture_kind = str(
                capture_kind
                if capture_kind is not None
                else (
                    CAPTURE_KIND_SAR
                    if position_mm is not None
                    else CAPTURE_KIND_MANUAL_NO_STAGE
                )
            ).strip().lower()
            if effective_capture_kind not in CAPTURE_KINDS:
                allowed = ", ".join(sorted(CAPTURE_KINDS))
                raise CaptureError(f"capture_kind must be one of: {allowed}")
            if effective_capture_kind == CAPTURE_KIND_SAR and position_mm is None:
                raise CaptureError("SAR captures require a measured carriage position")
            # Il ticket ricorda l'ID osservato prima del comando. Solo un ID
            # successivo può appartenere a questa richiesta, non a una cattura
            # terminata in ritardo dalla sessione precedente.
            previous_session_id = _read_shared(self._cap_id)
            if position_mm is not None:
                try:
                    position_mm_f = float(position_mm)
                except (TypeError, ValueError) as exc:
                    raise CaptureError("position_mm must be a finite number") from exc
                if not math.isfinite(position_mm_f):
                    raise CaptureError("position_mm must be a finite number")
            else:
                position_mm_f = None
            position_microsteps_i = (
                None
                if position_microsteps is None
                else _strict_integer(position_microsteps, field_name="position_microsteps")
            )
            ticket = CaptureTicket(
                previous_session_id=previous_session_id,
                position_id=legacy_position,
                position_mm=position_mm_f,
                position_microsteps=position_microsteps_i,
                capture_id=effective_capture_id,
                acquisition_index=effective_acquisition_index,
                capture_kind=effective_capture_kind,
            )
            metadata = {
                "carriage_position_mm": ticket.position_mm,
                "carriage_microsteps": ticket.position_microsteps,
                "capture_id": int(ticket.capture_id),
                "acquisition_index": int(ticket.acquisition_index),
                "position": int(ticket.position_id),
                "capture_kind": str(ticket.capture_kind),
            }
            try:
                self._cmd_queue.put_nowait(("CAPTURE", int(ticket.capture_id), metadata))
            except queue.Full as exc:
                raise CaptureError("Coda catture piena: riprovare tra poco.") from exc
            except Exception as exc:
                raise CaptureError(f"Impossibile avviare la cattura: {exc}") from exc
            self._ticket = ticket
            return ticket

    def wait(self, ticket: CaptureTicket, timeout_seconds: float, cancel_event: threading.Event) -> None:
        """Attende esclusivamente il completamento della sessione del ticket."""
        deadline = time.monotonic() + max(0.001, float(timeout_seconds))
        while True:
            if cancel_event.is_set():
                self.cancel()
                raise CaptureError("Cattura SAR annullata.")
            if self.poll_completion(ticket):
                return

            if time.monotonic() >= deadline:
                self.cancel()
                raise CaptureError("Timeout in attesa del completamento cattura SAR.")
            cancel_event.wait(0.01)

    def poll_completion(self, ticket: CaptureTicket) -> bool:
        """Controlla senza bloccare se il logger ha chiuso la cattura.

        Restituisce ``False`` finché la sessione è pendente e ``True`` solo
        dopo flush/close riuscito. Gli esiti negativi vengono esposti come
        ``CaptureError`` invece di essere eliminati silenziosamente.
        """
        with self._lock:
            if self._ticket != ticket:
                raise CaptureError("Il ticket di cattura non è più quello attivo.")
            current_id = _read_shared(self._cap_id)
            if current_id == int(ticket.previous_session_id):
                return False
            if _read_shared(self._cap_done_id) != current_id:
                return False
            result = _read_shared(self._cap_result)
            self._ticket = None
        if result == 1:
            return True
        if result == -1:
            raise CaptureError("Cattura SAR annullata prima del completamento.")
        raise CaptureError(f"Cattura SAR terminata con stato non valido ({result}).")

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
