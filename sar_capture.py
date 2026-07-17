"""Protocollo di conferma delle catture SAR tra GUI e processi RX/logger."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import queue
import threading
import time
from typing import Any, Mapping


class CaptureError(RuntimeError):
    """Una cattura non è stata accettata, completata o è stata annullata."""


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


CYLINDRICAL_REQUIRED_FIELDS = (
    "angle_index",
    "height_index",
    "angle_count",
    "azimuth_rad",
    "height_m",
    "radius_m",
    "scene_center_m",
)


def _finite_float(value: Any, field: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise CaptureError(f"cylindrical.{field} must be a finite number") from exc
    if not math.isfinite(converted):
        raise CaptureError(f"cylindrical.{field} must be a finite number")
    return converted


def _nonnegative_int(value: Any, field: str, *, positive: bool = False) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise CaptureError(f"cylindrical.{field} must be an integer") from exc
    if converted < 0 or (positive and converted <= 0):
        qualifier = "positive" if positive else "non-negative"
        raise CaptureError(f"cylindrical.{field} must be {qualifier}")
    return converted


def normalize_cylindrical_metadata(cylindrical: Mapping[str, Any]) -> dict[str, Any]:
    """Valida e canonicalizza il metadata geometrico cylindrical v2.

    ``height_m`` è lo scostamento verticale dal terzo componente di
    ``scene_center_m``.  ``position_m`` viene sempre derivata dai parametri
    cilindrici per impedire che una seconda coordinata incoerente diventi
    autorevole.
    """

    if not isinstance(cylindrical, Mapping):
        raise CaptureError("cylindrical metadata must be a mapping")
    missing = [field for field in CYLINDRICAL_REQUIRED_FIELDS if field not in cylindrical]
    if missing:
        raise CaptureError(f"cylindrical metadata is missing: {', '.join(missing)}")

    center_raw = cylindrical["scene_center_m"]
    if isinstance(center_raw, (str, bytes)):
        raise CaptureError("cylindrical.scene_center_m must contain exactly three numbers")
    try:
        center_values = list(center_raw)
    except TypeError as exc:
        raise CaptureError("cylindrical.scene_center_m must contain exactly three numbers") from exc
    if len(center_values) != 3:
        raise CaptureError("cylindrical.scene_center_m must contain exactly three numbers")
    center = [_finite_float(value, "scene_center_m") for value in center_values]

    angle_index = _nonnegative_int(cylindrical["angle_index"], "angle_index")
    height_index = _nonnegative_int(cylindrical["height_index"], "height_index")
    angle_count = _nonnegative_int(cylindrical["angle_count"], "angle_count", positive=True)
    if angle_index >= angle_count:
        raise CaptureError("cylindrical.angle_index must be smaller than cylindrical.angle_count")
    azimuth_rad = _finite_float(cylindrical["azimuth_rad"], "azimuth_rad")
    if not 0.0 <= azimuth_rad < 2.0 * math.pi:
        raise CaptureError("cylindrical.azimuth_rad must be in [0, 2*pi)")
    height_m = _finite_float(cylindrical["height_m"], "height_m")
    radius_m = _finite_float(cylindrical["radius_m"], "radius_m")
    if radius_m <= 0.0:
        raise CaptureError("cylindrical.radius_m must be strictly positive")

    result: dict[str, Any] = {
        "angle_index": angle_index,
        "height_index": height_index,
        "angle_count": angle_count,
        "azimuth_rad": azimuth_rad,
        "height_m": height_m,
        "radius_m": radius_m,
        "scene_center_m": center,
        "position_m": [
            center[0] + radius_m * math.cos(azimuth_rad),
            center[1] + radius_m * math.sin(azimuth_rad),
            center[2] + height_m,
        ],
    }
    try:
        payload = json.dumps(result, allow_nan=False, ensure_ascii=True, separators=(",", ":"))
        decoded = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise CaptureError("cylindrical metadata must be JSON serializable") from exc
    if not isinstance(decoded, dict):  # Defensive: json.loads di un mapping lo è sempre.
        raise CaptureError("cylindrical metadata must be a JSON mapping")
    return decoded


def normalize_capture_metadata(capture_id: int, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Produce il blob canonico associato a un identificatore di cattura.

    ``capture_id`` identifica il file/cattura; ``acquisition_index`` è
    l'ordine temporale. ``position`` resta un campo legacy e non descrive la
    geometria v2.
    """

    if metadata is None:
        raw: Mapping[str, Any] = {}
    elif isinstance(metadata, Mapping):
        raw = metadata
    else:
        raise CaptureError("capture metadata must be a mapping")

    try:
        capture_id_i = int(capture_id)
    except (TypeError, ValueError) as exc:
        raise CaptureError("capture_id must be a non-negative integer") from exc
    # A v2 capture identifier is non-negative.  Keep historical negative
    # linear ``position_id`` values readable/writable when no v2 metadata was
    # requested; those values are legacy positions, not v2 capture IDs.
    if capture_id_i < 0 and (raw.get("cylindrical") is not None or "capture_id" in raw):
        raise CaptureError("capture_id must be a non-negative integer")
    supplied_capture_id = raw.get("capture_id", capture_id_i)
    try:
        if int(supplied_capture_id) != capture_id_i:
            raise CaptureError("capture metadata capture_id does not match the CAPTURE command")
        acquisition_index = int(raw.get("acquisition_index", capture_id_i))
        legacy_position = int(raw.get("position", capture_id_i))
    except (TypeError, ValueError) as exc:
        raise CaptureError("capture_id, acquisition_index and position must be integers") from exc
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
        if math.isfinite(carriage_position_mm):
            result["carriage_position_mm"] = carriage_position_mm
    if raw.get("carriage_microsteps") is not None:
        try:
            result["carriage_microsteps"] = int(raw["carriage_microsteps"])
        except (TypeError, ValueError) as exc:
            raise CaptureError("carriage_microsteps must be an integer") from exc
    if raw.get("cylindrical") is not None:
        result["cylindrical"] = normalize_cylindrical_metadata(raw["cylindrical"])
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
    cylindrical: dict[str, Any] | None

    @property
    def position(self) -> int:
        """Campo legacy; non descrive la geometria cylindrical v2."""
        return int(self.position_id)


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
        position_id: int | None = None,
        position_mm: float | None = None,
        position_microsteps: int | None = None,
        *,
        capture_id: int | None = None,
        acquisition_index: int | None = None,
        cylindrical: Mapping[str, Any] | None = None,
    ) -> CaptureTicket:
        with self._lock:
            self._reap_completed_locked()
            if self._ticket is not None:
                raise CaptureError("Una cattura SAR è già in corso.")
            if position_id is None and capture_id is None:
                raise CaptureError("A capture requires capture_id or legacy position_id.")
            legacy_position = int(capture_id if position_id is None else position_id)
            effective_capture_id = int(legacy_position if capture_id is None else capture_id)
            if effective_capture_id < 0 and (capture_id is not None or cylindrical is not None):
                raise CaptureError("capture_id must be non-negative")
            effective_acquisition_index = int(
                effective_capture_id if acquisition_index is None else acquisition_index
            )
            if effective_acquisition_index < 0:
                raise CaptureError("acquisition_index must be non-negative")
            cylindrical_normalized = (
                None if cylindrical is None else normalize_cylindrical_metadata(cylindrical)
            )
            previous_session_id = _read_shared(self._cap_id)
            ticket = CaptureTicket(
                previous_session_id=previous_session_id,
                position_id=legacy_position,
                position_mm=None if position_mm is None else float(position_mm),
                position_microsteps=None if position_microsteps is None else int(position_microsteps),
                capture_id=effective_capture_id,
                acquisition_index=effective_acquisition_index,
                cylindrical=cylindrical_normalized,
            )
            metadata = {
                "carriage_position_mm": ticket.position_mm,
                "carriage_microsteps": ticket.position_microsteps,
            }
            # Il comando legacy resta identico quando non viene richiesta una
            # geometria nuova: utile per compatibilità e test di regressione.
            if (
                ticket.cylindrical is not None
                or capture_id is not None
                or acquisition_index is not None
                or int(ticket.capture_id) != int(ticket.position_id)
            ):
                metadata.update(
                    {
                        "capture_id": int(ticket.capture_id),
                        "acquisition_index": int(ticket.acquisition_index),
                        "position": int(ticket.position_id),
                    }
                )
            if ticket.cylindrical is not None:
                metadata["cylindrical"] = ticket.cylindrical
            try:
                self._cmd_queue.put_nowait(("CAPTURE", int(ticket.capture_id), metadata))
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
