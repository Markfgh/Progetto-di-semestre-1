"""Coordinamento non-GUI di una scansione SAR a carrello.

Il modulo non conosce né Dear PyGui né il radar/Phidget: riceve piccole
callback sincrone per il movimento e per la cattura.  Questo mantiene la
sequenza ``cattura -> movimento -> assestamento`` testabile senza hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import threading
import time
from typing import Any, Callable


class ScanError(RuntimeError):
    """Errore recuperabile durante una scansione SAR."""


@dataclass(frozen=True)
class ScanPlan:
    """Parametri fisici e indici dei file di una scansione."""

    positions: int
    pitch_mm: float
    start_position_id: int = 1
    settling_seconds: float = 0.0
    motion_timeout_seconds: float = 120.0
    capture_timeout_seconds: float = 120.0

    def validate(self) -> None:
        if int(self.positions) <= 0:
            raise ScanError("Il numero di posizioni deve essere maggiore di zero.")
        if not math.isfinite(float(self.pitch_mm)) or float(self.pitch_mm) <= 0.0:
            raise ScanError("Il pitch della scansione deve essere maggiore di zero.")
        if int(self.start_position_id) <= 0:
            raise ScanError("L'ID iniziale della scansione deve essere positivo.")
        timing = (
            float(self.settling_seconds),
            float(self.motion_timeout_seconds),
            float(self.capture_timeout_seconds),
        )
        if not all(math.isfinite(value) for value in timing) or min(timing) < 0.0:
            raise ScanError("I timeout e il tempo di assestamento non possono essere negativi.")
        if float(self.motion_timeout_seconds) <= 0.0 or float(self.capture_timeout_seconds) <= 0.0:
            raise ScanError("I timeout di movimento e cattura devono essere maggiori di zero.")


@dataclass(frozen=True)
class ScanStatus:
    """Snapshot thread-safe consultato dalla GUI."""

    state: str = "idle"
    total: int = 0
    completed: int = 0
    position_id: int | None = None
    position_mm: float | None = None
    message: str = "Pronto"
    error: str = ""


def absolute_scan_targets_microsteps(
    *,
    start_microsteps: int,
    pitch_mm: float,
    mm_per_microstep: float,
    positions: int,
) -> tuple[int, ...]:
    """Restituisce target assoluti, evitando l'errore cumulativo del jog.

    Ogni target viene arrotondato rispetto alla posizione iniziale, non
    sommando a ogni ciclo un passo già arrotondato.
    """

    if int(positions) <= 0:
        raise ScanError("Il numero di posizioni deve essere maggiore di zero.")
    if (
        not math.isfinite(float(pitch_mm))
        or not math.isfinite(float(mm_per_microstep))
        or float(pitch_mm) <= 0.0
        or float(mm_per_microstep) <= 0.0
    ):
        raise ScanError("Pitch e risoluzione meccanica devono essere maggiori di zero.")
    pitch_microsteps = float(pitch_mm) / float(mm_per_microstep)
    start = int(start_microsteps)
    return tuple(start + int(round(index * pitch_microsteps)) for index in range(int(positions)))


class SarScanCoordinator:
    """Esegue una scansione sequenziale in un thread dedicato.

    Le callback ``move_to_microsteps`` e ``wait_capture`` devono completare
    solo quando rispettivamente il carrello è fermo e il file della cattura è
    stato chiuso/flushed.  ``cancel`` può essere chiamato dal thread GUI.
    """

    def __init__(
        self,
        *,
        begin_motion: Callable[[], None],
        finish_motion: Callable[[bool], None],
        get_position_microsteps: Callable[[], int],
        mm_from_microsteps: Callable[[int], float],
        mm_per_microstep: Callable[[], float],
        move_to_microsteps: Callable[[int, float, threading.Event], None],
        request_capture: Callable[[int, float, int], Any],
        wait_capture: Callable[[Any, float, threading.Event], None],
        cancel_capture: Callable[[], None],
        stop_motion: Callable[[], None],
    ) -> None:
        self._begin_motion = begin_motion
        self._finish_motion = finish_motion
        self._get_position_microsteps = get_position_microsteps
        self._mm_from_microsteps = mm_from_microsteps
        self._mm_per_microstep = mm_per_microstep
        self._move_to_microsteps = move_to_microsteps
        self._request_capture = request_capture
        self._wait_capture = wait_capture
        self._cancel_capture = cancel_capture
        self._stop_motion = stop_motion
        self._lock = threading.RLock()
        self._cancel_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._status = ScanStatus()

    def status(self) -> ScanStatus:
        with self._lock:
            return replace(self._status)

    @property
    def active(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self, plan: ScanPlan) -> None:
        plan.validate()
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise ScanError("Una scansione SAR è già in corso.")
            self._cancel_event = threading.Event()
            self._status = ScanStatus(
                state="starting",
                total=int(plan.positions),
                message="Preparazione scansione SAR",
            )
            self._thread = threading.Thread(
                target=self._run,
                args=(plan, self._cancel_event),
                name="sar-scan",
                daemon=True,
            )
            self._thread.start()

    def cancel(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                return
            # ``completed`` viene pubblicato soltanto dopo il cleanup; da quel
            # momento una richiesta tardiva non deve riscriverlo in
            # ``cancelling`` mentre il thread sta eseguendo il return finale.
            if self._status.state == "completed":
                return
            self._status = replace(self._status, state="cancelling", message="Annullamento scansione...")
        # Prima rendi sicuro il carrello e imposta i suoi flag STOP; soltanto
        # dopo sveglia il worker tramite cancel_event. In caso contrario il
        # worker potrebbe eseguire finish_motion(False) un istante prima di
        # stop_motion e il controller scambierebbe un cancel normale per FAULT.
        try:
            self._stop_motion()
        except Exception:
            pass
        finally:
            self._cancel_event.set()
        try:
            self._cancel_capture()
        except Exception:
            pass

    def join(self, timeout: float | None = None) -> None:
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def _set_status(self, **changes: Any) -> None:
        with self._lock:
            self._status = replace(self._status, **changes)

    @staticmethod
    def _wait_settling(cancel_event: threading.Event, seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, float(seconds))
        while True:
            if cancel_event.is_set():
                raise ScanError("Scansione annullata.")
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            cancel_event.wait(min(0.05, remaining))

    def _run(self, plan: ScanPlan, cancel_event: threading.Event) -> None:
        success = False
        try:
            self._begin_motion()
            start_steps = int(self._get_position_microsteps())
            targets = absolute_scan_targets_microsteps(
                start_microsteps=start_steps,
                pitch_mm=float(plan.pitch_mm),
                mm_per_microstep=float(self._mm_per_microstep()),
                positions=int(plan.positions),
            )

            for index, _target_steps in enumerate(targets):
                if cancel_event.is_set():
                    raise ScanError("Scansione annullata.")

                position_id = int(plan.start_position_id) + index
                actual_steps = int(self._get_position_microsteps())
                actual_mm = float(self._mm_from_microsteps(actual_steps))
                self._set_status(
                    state="capturing",
                    position_id=position_id,
                    position_mm=actual_mm,
                    message=f"Cattura posizione {index + 1}/{plan.positions}",
                )
                ticket = self._request_capture(position_id, actual_mm, actual_steps)
                self._wait_capture(ticket, float(plan.capture_timeout_seconds), cancel_event)
                if cancel_event.is_set():
                    raise ScanError("Scansione annullata.")
                self._set_status(completed=index + 1)

                if index + 1 >= len(targets):
                    break

                next_position_id = int(plan.start_position_id) + index + 1
                next_mm = float(self._mm_from_microsteps(int(targets[index + 1])))
                self._set_status(
                    state="moving",
                    position_id=next_position_id,
                    position_mm=next_mm,
                    message=f"Movimento verso posizione {index + 2}/{plan.positions}",
                )
                self._move_to_microsteps(
                    int(targets[index + 1]),
                    float(plan.motion_timeout_seconds),
                    cancel_event,
                )
                self._set_status(state="settling", message="Assestamento meccanico")
                self._wait_settling(cancel_event, float(plan.settling_seconds))

            if cancel_event.is_set():
                raise ScanError("Scansione annullata.")
            success = True
        except Exception as exc:
            if cancel_event.is_set() or str(exc) == "Scansione annullata.":
                self._set_status(state="cancelled", message="Scansione SAR annullata")
            else:
                self._set_status(state="failed", message="Scansione SAR interrotta", error=str(exc))
        finally:
            try:
                self._finish_motion(bool(success))
            except Exception as exc:
                if success:
                    self._set_status(state="failed", message="Errore chiusura scansione", error=str(exc))
            else:
                if success:
                    # Serializza la decisione finale con cancel(): se
                    # l'annullamento vince prima di questo punto non deve
                    # essere sovrascritto da un completamento tardivo.
                    with self._lock:
                        if cancel_event.is_set():
                            self._status = replace(
                                self._status,
                                state="cancelled",
                                message="Scansione SAR annullata",
                            )
                        else:
                            self._status = replace(
                                self._status,
                                state="completed",
                                message="Scansione SAR completata",
                            )
