"""Coordinamento non-GUI delle scansioni SAR lineare e cilindrica.

Il modulo non conosce né Dear PyGui né il radar/Phidget: riceve piccole
callback sincrone per il movimento e per la cattura.  Questo mantiene la
sequenza ``cattura -> movimento -> assestamento`` testabile senza hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import threading
import time
from typing import Any, Callable, Protocol

from sar_geometry import CylindricalCapture


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


class RotaryAxis(Protocol):
    """Interfaccia minima per l'asse rotativo di una scansione cilindrica.

    Il progetto non include un driver del rotatore.  Un adapter hardware deve
    implementare questo contratto senza esporre dettagli del proprio SDK al
    coordinatore. ``align_zero_and_wait`` porta il radar alla posa azimutale
    zero; ``move_relative_rad_and_wait`` avanza nel verso positivo richiesto
    dal piano. Entrambe le chiamate devono ritornare solo a moto concluso.
    """

    def begin_external_scan(self) -> None: ...

    def finish_external_scan(self, success: bool) -> None: ...

    def align_zero_and_wait(
        self,
        timeout_seconds: float,
        cancel_event: threading.Event,
    ) -> None: ...

    def move_relative_rad_and_wait(
        self,
        delta_rad: float,
        timeout_seconds: float,
        cancel_event: threading.Event,
    ) -> None: ...

    def stop(self) -> None: ...


@dataclass(frozen=True)
class CylindricalScanPlan:
    """Piano per un solo giro regolare per quota.

    L'origine HOME dell'asse verticale coincide in questa fase con
    ``scene_center_m[2]``: ``initial_height_m`` e ``vertical_step_m`` sono
    quindi sia le quote salvate negli header sia i target del verticale.
    Non viene introdotta alcuna calibrazione o trasformazione aggiuntiva.
    """

    angles_per_turn: int
    radius_m: float
    initial_height_m: float
    height_count: int
    vertical_step_m: float
    scene_center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    start_capture_id: int = 1
    angular_settling_seconds: float = 0.0
    vertical_settling_seconds: float = 0.0
    motion_timeout_seconds: float = 120.0
    capture_timeout_seconds: float = 120.0

    @property
    def total_captures(self) -> int:
        return int(self.angles_per_turn) * int(self.height_count)

    @property
    def angular_step_rad(self) -> float:
        return (2.0 * math.pi) / float(self.angles_per_turn)

    def height_m_for_index(self, height_index: int) -> float:
        if not 0 <= int(height_index) < int(self.height_count):
            raise ScanError("height_index fuori dal piano cilindrico.")
        return float(self.initial_height_m) + int(height_index) * float(self.vertical_step_m)

    def capture_for_indices(self, height_index: int, angle_index: int) -> CylindricalCapture:
        if not 0 <= int(angle_index) < int(self.angles_per_turn):
            raise ScanError("angle_index fuori dal piano cilindrico.")
        acquisition_index = int(height_index) * int(self.angles_per_turn) + int(angle_index)
        return CylindricalCapture(
            capture_id=int(self.start_capture_id) + acquisition_index,
            acquisition_index=acquisition_index,
            angle_index=int(angle_index),
            height_index=int(height_index),
            azimuth_rad=float(angle_index) * self.angular_step_rad,
            height_m=self.height_m_for_index(int(height_index)),
            radius_m=float(self.radius_m),
            scene_center_m=self.scene_center_m,
            angle_count=int(self.angles_per_turn),
            height_count=int(self.height_count),
        )

    def validate(self) -> None:
        if int(self.angles_per_turn) <= 0:
            raise ScanError("Il numero di angoli per giro deve essere maggiore di zero.")
        if int(self.height_count) <= 0:
            raise ScanError("Il numero di quote deve essere maggiore di zero.")
        if int(self.start_capture_id) < 0:
            raise ScanError("Il capture_id iniziale deve essere non negativo.")
        finite_positive = (float(self.radius_m), float(self.vertical_step_m))
        if not all(math.isfinite(value) and value > 0.0 for value in finite_positive):
            raise ScanError("Raggio e passo verticale devono essere maggiori di zero.")
        if not math.isfinite(float(self.initial_height_m)):
            raise ScanError("La quota iniziale deve essere finita.")
        try:
            center = tuple(float(value) for value in self.scene_center_m)
        except (TypeError, ValueError) as exc:
            raise ScanError("Il centro scena deve contenere tre coordinate finite.") from exc
        if len(center) != 3 or not all(math.isfinite(value) for value in center):
            raise ScanError("Il centro scena deve contenere tre coordinate finite.")
        timing = (
            float(self.angular_settling_seconds),
            float(self.vertical_settling_seconds),
            float(self.motion_timeout_seconds),
            float(self.capture_timeout_seconds),
        )
        if not all(math.isfinite(value) for value in timing) or min(timing) < 0.0:
            raise ScanError("I tempi della scansione cilindrica non possono essere negativi.")
        if float(self.motion_timeout_seconds) <= 0.0 or float(self.capture_timeout_seconds) <= 0.0:
            raise ScanError("I timeout di movimento e cattura devono essere maggiori di zero.")
        # Costruisce l'ultimo elemento: valida anche gli indici e il range
        # azimutale senza duplicare la logica del metadata v2.
        self.capture_for_indices(int(self.height_count) - 1, int(self.angles_per_turn) - 1)


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
    capture_id: int | None = None
    acquisition_index: int | None = None
    angle_index: int | None = None
    height_index: int | None = None
    azimuth_rad: float | None = None
    height_m: float | None = None


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


class CylindricalScanCoordinator:
    """Esegue un cilindro regolare coordinando rotatore, verticale e logger.

    Le callback dei due assi ritornano soltanto quando il moto è fermo.  La
    callback ``wait_capture`` ritorna invece soltanto dopo flush e chiusura del
    file; perciò nessun passo meccanico successivo può anticipare la persistenza
    dell'header ``rt_capture_v2`` e del suo payload.
    """

    def __init__(
        self,
        *,
        begin_vertical_scan: Callable[[], None],
        finish_vertical_scan: Callable[[bool], None],
        get_vertical_microsteps: Callable[[], int],
        height_m_from_microsteps: Callable[[int], float],
        microsteps_from_height_m: Callable[[float], int],
        move_vertical_to_microsteps: Callable[[int, float, threading.Event], None],
        stop_vertical: Callable[[], None],
        rotary_axis: RotaryAxis,
        request_capture: Callable[..., Any],
        wait_capture: Callable[[Any, float, threading.Event], None],
        cancel_capture: Callable[[], None],
    ) -> None:
        self._begin_vertical_scan = begin_vertical_scan
        self._finish_vertical_scan = finish_vertical_scan
        self._get_vertical_microsteps = get_vertical_microsteps
        self._height_m_from_microsteps = height_m_from_microsteps
        self._microsteps_from_height_m = microsteps_from_height_m
        self._move_vertical_to_microsteps = move_vertical_to_microsteps
        self._stop_vertical = stop_vertical
        self._rotary_axis = rotary_axis
        self._request_capture = request_capture
        self._wait_capture = wait_capture
        self._cancel_capture = cancel_capture
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

    def start(self, plan: CylindricalScanPlan) -> None:
        plan.validate()
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise ScanError("Una scansione cilindrica è già in corso.")
            self._cancel_event = threading.Event()
            self._status = ScanStatus(
                state="starting",
                total=int(plan.total_captures),
                message="Preparazione scansione cilindrica",
            )
            self._thread = threading.Thread(
                target=self._run,
                args=(plan, self._cancel_event),
                name="sar-cylindrical-scan",
                daemon=True,
            )
            self._thread.start()

    def cancel(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                return
            if self._status.state == "completed":
                return
            self._status = replace(
                self._status,
                state="cancelling",
                message="Annullamento scansione cilindrica...",
            )
        # Arresta entrambi gli assi prima di svegliare il worker: un callback
        # di movimento non deve poter osservare il cancel e chiudere la sua
        # prenotazione mentre l'altro asse è ancora in moto.
        for stop_axis in (self._stop_vertical, self._rotary_axis.stop):
            try:
                stop_axis()
            except Exception:
                pass
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
        SarScanCoordinator._wait_settling(cancel_event, seconds)

    def _move_vertical_to_height(
        self,
        height_m: float,
        timeout_seconds: float,
        cancel_event: threading.Event,
    ) -> tuple[int, float]:
        target_steps = int(self._microsteps_from_height_m(float(height_m)))
        self._move_vertical_to_microsteps(target_steps, float(timeout_seconds), cancel_event)
        if cancel_event.is_set():
            raise ScanError("Scansione annullata.")
        actual_steps = int(self._get_vertical_microsteps())
        actual_height_m = float(self._height_m_from_microsteps(actual_steps))
        resolution_m = abs(
            float(self._height_m_from_microsteps(1))
            - float(self._height_m_from_microsteps(0))
        )
        tolerance_m = max(1e-9, 0.51 * resolution_m)
        if not math.isfinite(actual_height_m) or abs(actual_height_m - float(height_m)) > tolerance_m:
            raise ScanError(
                "L'asse verticale non ha raggiunto la quota pianificata "
                f"({actual_height_m:.9f} m invece di {float(height_m):.9f} m)."
            )
        return actual_steps, actual_height_m

    def _capture(
        self,
        capture: CylindricalCapture,
        *,
        vertical_steps: int,
        vertical_height_m: float,
        plan: CylindricalScanPlan,
        cancel_event: threading.Event,
    ) -> None:
        if cancel_event.is_set():
            raise ScanError("Scansione annullata.")
        self._set_status(
            state="capturing",
            capture_id=int(capture.capture_id),
            acquisition_index=int(capture.acquisition_index),
            angle_index=int(capture.angle_index),
            height_index=int(capture.height_index),
            azimuth_rad=float(capture.azimuth_rad),
            height_m=float(capture.height_m),
            message=(
                f"Cattura quota {capture.height_index + 1}/{plan.height_count}, "
                f"angolo {capture.angle_index + 1}/{plan.angles_per_turn}"
            ),
        )
        ticket = self._request_capture(
            position_id=int(capture.capture_id),  # legacy only; not geometry v2
            position_mm=float(vertical_height_m) * 1000.0,
            position_microsteps=int(vertical_steps),
            capture_id=int(capture.capture_id),
            acquisition_index=int(capture.acquisition_index),
            cylindrical=capture.to_dict(),
        )
        self._wait_capture(ticket, float(plan.capture_timeout_seconds), cancel_event)
        if cancel_event.is_set():
            raise ScanError("Scansione annullata.")
        self._set_status(completed=int(capture.acquisition_index) + 1)

    def _run(self, plan: CylindricalScanPlan, cancel_event: threading.Event) -> None:
        success = False
        vertical_started = False
        rotary_started = False
        try:
            self._begin_vertical_scan()
            vertical_started = True
            self._rotary_axis.begin_external_scan()
            rotary_started = True

            for height_index in range(int(plan.height_count)):
                if cancel_event.is_set():
                    raise ScanError("Scansione annullata.")
                height_m = plan.height_m_for_index(height_index)
                self._set_status(
                    state="moving_vertical",
                    height_index=height_index,
                    height_m=height_m,
                    message=f"Movimento alla quota {height_index + 1}/{plan.height_count}",
                )
                vertical_steps, vertical_height_m = self._move_vertical_to_height(
                    height_m,
                    float(plan.motion_timeout_seconds),
                    cancel_event,
                )
                self._set_status(state="settling_vertical", message="Assestamento asse verticale")
                self._wait_settling(cancel_event, float(plan.vertical_settling_seconds))

                self._set_status(
                    state="aligning_rotary",
                    height_index=height_index,
                    height_m=height_m,
                    message=f"Allineamento azimutale quota {height_index + 1}/{plan.height_count}",
                )
                self._rotary_axis.align_zero_and_wait(float(plan.motion_timeout_seconds), cancel_event)

                for angle_index in range(int(plan.angles_per_turn)):
                    if cancel_event.is_set():
                        raise ScanError("Scansione annullata.")
                    if angle_index:
                        self._set_status(
                            state="moving_rotary",
                            angle_index=angle_index,
                            height_index=height_index,
                            azimuth_rad=float(angle_index) * plan.angular_step_rad,
                            height_m=height_m,
                            message=(
                                f"Movimento angolare {angle_index + 1}/{plan.angles_per_turn}, "
                                f"quota {height_index + 1}/{plan.height_count}"
                            ),
                        )
                        self._rotary_axis.move_relative_rad_and_wait(
                            plan.angular_step_rad,
                            float(plan.motion_timeout_seconds),
                            cancel_event,
                        )
                    self._set_status(state="settling_rotary", message="Assestamento asse rotativo")
                    self._wait_settling(cancel_event, float(plan.angular_settling_seconds))
                    self._capture(
                        plan.capture_for_indices(height_index, angle_index),
                        vertical_steps=vertical_steps,
                        vertical_height_m=vertical_height_m,
                        plan=plan,
                        cancel_event=cancel_event,
                    )

                # L'ultimo avanzamento chiude davvero il giro di 360° senza
                # duplicare la cattura a azimuth 0. Il metadata rimane sempre
                # nel range [0, 2*pi), come richiesto dal formato v2.
                self._set_status(
                    state="closing_turn",
                    height_index=height_index,
                    height_m=height_m,
                    message=f"Chiusura giro 360° quota {height_index + 1}/{plan.height_count}",
                )
                self._rotary_axis.move_relative_rad_and_wait(
                    plan.angular_step_rad,
                    float(plan.motion_timeout_seconds),
                    cancel_event,
                )

            if cancel_event.is_set():
                raise ScanError("Scansione annullata.")
            success = True
        except Exception as exc:
            if cancel_event.is_set() or str(exc) == "Scansione annullata.":
                self._set_status(state="cancelled", message="Scansione cilindrica annullata")
            else:
                self._set_status(
                    state="failed",
                    message="Scansione cilindrica interrotta",
                    error=str(exc),
                )
        finally:
            finish_error: Exception | None = None
            if rotary_started:
                try:
                    self._rotary_axis.finish_external_scan(bool(success))
                except Exception as exc:
                    finish_error = exc
            if vertical_started:
                try:
                    self._finish_vertical_scan(bool(success))
                except Exception as exc:
                    if finish_error is None:
                        finish_error = exc
            if success and finish_error is not None:
                self._set_status(
                    state="failed",
                    message="Errore chiusura scansione cilindrica",
                    error=str(finish_error),
                )
            elif success:
                with self._lock:
                    if cancel_event.is_set():
                        self._status = replace(
                            self._status,
                            state="cancelled",
                            message="Scansione cilindrica annullata",
                        )
                    else:
                        self._status = replace(
                            self._status,
                            state="completed",
                            message="Scansione cilindrica completata",
                        )


@dataclass(frozen=True)
class CylindricalDryRunResult:
    """Esito riproducibile della sequenza cilindrica senza driver hardware."""

    captures: tuple[CylindricalCapture, ...]
    events: tuple[tuple[Any, ...], ...]
    status: ScanStatus


def dry_run_cylindrical_scan(plan: CylindricalScanPlan) -> CylindricalDryRunResult:
    """Esegue l'intera sequenza con assi finti e valida ogni metadata v2.

    Il dry-run riusa lo stesso coordinatore dell'operatività reale; attese di
    settling e movimenti sono simulati istantaneamente, ma la sequenza e il
    contratto di cattura/flush sono identici.
    """

    plan.validate()
    events: list[tuple[Any, ...]] = []
    captures: list[CylindricalCapture] = []
    current_steps = int(round(float(plan.initial_height_m) * 1000.0))

    class _DryRunRotaryAxis:
        def begin_external_scan(self) -> None:
            events.append(("rotary_begin",))

        def finish_external_scan(self, success: bool) -> None:
            events.append(("rotary_finish", bool(success)))

        def align_zero_and_wait(self, _timeout: float, cancel_event: threading.Event) -> None:
            if cancel_event.is_set():
                raise ScanError("Scansione annullata.")
            events.append(("rotary_align_zero",))

        def move_relative_rad_and_wait(
            self,
            delta_rad: float,
            _timeout: float,
            cancel_event: threading.Event,
        ) -> None:
            if cancel_event.is_set():
                raise ScanError("Scansione annullata.")
            events.append(("rotary_move", float(delta_rad)))

        def stop(self) -> None:
            events.append(("rotary_stop",))

    def _move_vertical(target: int, _timeout: float, cancel_event: threading.Event) -> None:
        nonlocal current_steps
        if cancel_event.is_set():
            raise ScanError("Scansione annullata.")
        events.append(("vertical_move", int(target)))
        current_steps = int(target)

    def _request_capture(**kwargs: Any) -> CylindricalCapture:
        cylindrical = kwargs.get("cylindrical")
        if not isinstance(cylindrical, dict):
            raise ScanError("Dry-run senza metadata cylindrical.")
        capture = CylindricalCapture.from_dict(cylindrical)
        if int(kwargs.get("capture_id")) != capture.capture_id:
            raise ScanError("Dry-run con capture_id incoerente.")
        if int(kwargs.get("acquisition_index")) != capture.acquisition_index:
            raise ScanError("Dry-run con acquisition_index incoerente.")
        captures.append(capture)
        events.append(("capture", capture.capture_id, capture.acquisition_index))
        return capture

    def _wait_capture(ticket: CylindricalCapture, _timeout: float, cancel_event: threading.Event) -> None:
        if cancel_event.is_set():
            raise ScanError("Scansione annullata.")
        events.append(("flushed", ticket.capture_id))

    coordinator = CylindricalScanCoordinator(
        begin_vertical_scan=lambda: events.append(("vertical_begin",)),
        finish_vertical_scan=lambda success: events.append(("vertical_finish", bool(success))),
        get_vertical_microsteps=lambda: current_steps,
        height_m_from_microsteps=lambda steps: float(steps) / 1000.0,
        microsteps_from_height_m=lambda height_m: int(round(float(height_m) * 1000.0)),
        move_vertical_to_microsteps=_move_vertical,
        stop_vertical=lambda: events.append(("vertical_stop",)),
        rotary_axis=_DryRunRotaryAxis(),
        request_capture=_request_capture,
        wait_capture=_wait_capture,
        cancel_capture=lambda: events.append(("capture_cancel",)),
    )
    immediate_plan = replace(
        plan,
        angular_settling_seconds=0.0,
        vertical_settling_seconds=0.0,
    )
    coordinator.start(immediate_plan)
    coordinator.join(timeout=5.0)
    status = coordinator.status()
    if coordinator.active or status.state != "completed":
        raise ScanError(f"Dry-run cilindrico fallito: {status.error or status.message}")
    if len(captures) != int(plan.total_captures):
        raise ScanError("Dry-run cilindrico con numero di catture non coerente.")
    return CylindricalDryRunResult(tuple(captures), tuple(events), status)
