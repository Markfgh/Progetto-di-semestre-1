"""Controllo sicuro (non safety-rated) di un carrello con Phidget 1063.

Il 1063 e' un controller open-loop: il riferimento di posizione viene
stabilito esclusivamente dal finecorsa MIN durante HOME.  Il pulsante STOP
ferma il moto mantenendo la coppia; non sostituisce un arresto di emergenza
hardware cablato. ``PhidgetStepperController`` concentra la logica di
sicurezza e movimento; ``StepperGui`` la presenta in Dear PyGui. Il controller
può essere usato senza GUI, ad esempio dalla scansione SAR integrata.
"""

from __future__ import annotations

import copy
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import dearpygui.dearpygui as dpg
import yaml
from Phidget22.Devices.DigitalInput import DigitalInput
from Phidget22.Devices.Stepper import Stepper
from Phidget22.PhidgetException import PhidgetException


CONFIG_PATH = Path(__file__).with_name("phidget_stepper_config.yaml")
MICROSTEPS_PER_FULL_STEP = 16


class ConfigError(ValueError):
    """Configurazione assente o non utilizzabile."""


class MotionError(RuntimeError):
    """Comando di movimento non sicuro o non disponibile."""


class ControllerState(str, Enum):
    """Stati osservabili del controller, inclusi i blocchi di sicurezza."""

    DISCONNECTED = "Disconnesso"
    IDLE = "Collegato - da eseguire HOME"
    HOMING = "HOME in corso"
    READY = "Pronto"
    CYCLE_MOVING = "Ciclo: movimento"
    CYCLE_WAITING = "Ciclo: attesa"
    SCAN_MOVING = "Scansione SAR: movimento"
    SCAN_WAITING = "Scansione SAR: cattura/attesa"
    LIMIT_STOPPED = "Arrestato da finecorsa"
    STOPPED = "Arrestato"
    FAULT = "Errore"
    EMERGENCY = "Emergenza - motore disabilitato"


@dataclass
class DeviceConfig:
    """Indirizzamento del Phidget e dei due finecorsa cablati."""

    serial_number: int | None = None
    stepper_channel: int = 0
    limit_min_channel: int = 0
    limit_max_channel: int = 1
    # I DigitalInput del 1063 sono active-low: NO attivo = raw True,
    # NC attivo = raw False (anche un filo interrotto e' quindi sicuro).
    limit_min_polarity: str = "NO"
    limit_max_polarity: str = "NO"


@dataclass
class MechanicsConfig:
    """Conversione deterministica tra millimetri logici e microstep del motore."""

    belt_pitch_mm: float = 2.0
    pulley_teeth: int = 20
    motor_full_steps_per_rev: int = 200
    gear_reduction: float = 1.0

    @property
    def mm_per_microstep(self) -> float:
        return (
            self.belt_pitch_mm * self.pulley_teeth
            / (self.motor_full_steps_per_rev * MICROSTEPS_PER_FULL_STEP * self.gear_reduction)
        )

    def microsteps_from_mm(self, distance_mm: float) -> int:
        return int(round(distance_mm / self.mm_per_microstep))

    def mm_from_microsteps(self, microsteps: float) -> float:
        return microsteps * self.mm_per_microstep


@dataclass
class MotorConfig:
    velocity_microsteps_s: float = 800.0
    acceleration_microsteps_s2: float = 4000.0
    current_limit_a: float = 0.5


@dataclass
class HomingConfig:
    search_distance_mm: float = 1000.0
    release_distance_mm: float = 5.0
    velocity_microsteps_s: float = 400.0
    timeout_seconds: float = 30.0


@dataclass
class CycleConfig:
    step_mm: float = 1.0
    wait_seconds: float = 1.0
    cycles: int = 1


@dataclass
class AppConfig:
    """Configurazione completa del carrello, caricabile dal YAML locale."""

    device: DeviceConfig = field(default_factory=DeviceConfig)
    mechanics: MechanicsConfig = field(default_factory=MechanicsConfig)
    motor: MotorConfig = field(default_factory=MotorConfig)
    homing: HomingConfig = field(default_factory=HomingConfig)
    cycle: CycleConfig = field(default_factory=CycleConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AppConfig":
        if not isinstance(raw, dict):
            raise ConfigError("Il file YAML deve contenere una mappa.")
        try:
            motor_raw = dict(raw.get("motor", {}))
            # Il 1063 non supporta la corrente di mantenimento configurabile.
            # Ignora il campo delle configurazioni create dalle versioni precedenti.
            motor_raw.pop("holding_current_limit_a", None)
            config = cls(
                device=DeviceConfig(**raw.get("device", {})),
                mechanics=MechanicsConfig(**raw.get("mechanics", {})),
                motor=MotorConfig(**motor_raw),
                homing=HomingConfig(**raw.get("homing", {})),
                cycle=CycleConfig(**raw.get("cycle", {})),
            )
        except TypeError as exc:
            raise ConfigError(f"Campo YAML non valido: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        device = self.device
        mechanics = self.mechanics
        motor = self.motor
        homing = self.homing
        cycle = self.cycle
        if device.stepper_channel != 0:
            raise ConfigError("Il 1063 dispone del solo canale Stepper 0.")
        if device.limit_min_channel == device.limit_max_channel:
            raise ConfigError("MIN e MAX devono usare canali digitali diversi.")
        if not all(0 <= channel <= 3 for channel in (device.limit_min_channel, device.limit_max_channel)):
            raise ConfigError("I canali digitali del 1063 devono essere tra 0 e 3.")
        if device.limit_min_polarity not in {"NO", "NC"} or device.limit_max_polarity not in {"NO", "NC"}:
            raise ConfigError("La polarita' dei finecorsa deve essere NO o NC.")
        if min(mechanics.belt_pitch_mm, mechanics.pulley_teeth, mechanics.motor_full_steps_per_rev, mechanics.gear_reduction) <= 0:
            raise ConfigError("I parametri meccanici devono essere maggiori di zero.")
        if min(motor.velocity_microsteps_s, motor.acceleration_microsteps_s2, motor.current_limit_a) <= 0:
            raise ConfigError("Velocita', accelerazione e corrente devono essere maggiori di zero.")
        if min(homing.search_distance_mm, homing.release_distance_mm, homing.velocity_microsteps_s, homing.timeout_seconds) <= 0:
            raise ConfigError("I parametri HOME devono essere maggiori di zero.")
        if cycle.step_mm <= 0 or cycle.wait_seconds < 0 or cycle.cycles <= 0:
            raise ConfigError("Passo ciclo e numero cicli devono essere positivi; l'attesa puo' essere zero.")


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    """Carica e valida il YAML, fallendo con un errore orientato all'operatore."""
    if not path.exists():
        raise ConfigError(f"Configurazione non trovata: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML non valido: {exc}") from exc
    return AppConfig.from_dict(raw)


def save_config(config: AppConfig, path: Path = CONFIG_PATH) -> None:
    """Valida prima di salvare, per non lasciare una configurazione impossibile."""
    config.validate()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(asdict(config), handle, allow_unicode=True, sort_keys=False)


def direction_is_allowed(direction: int, limit_min_active: bool, limit_max_active: bool) -> bool:
    """Il moto di uscita da un finecorsa e' sempre consentito."""
    if direction < 0:
        return not limit_min_active
    if direction > 0:
        return not limit_max_active
    return True


@dataclass
class CycleProgress:
    """Stato puro del ciclo, indipendente da Phidget e GUI."""

    total: int
    completed: int = 0
    active: bool = False

    def start(self, homed: bool, max_active: bool) -> None:
        if not homed:
            raise MotionError("Eseguire HOME prima di avviare il ciclo.")
        if max_active:
            raise MotionError("Il finecorsa MAX e' attivo: il ciclo non puo' avanzare.")
        if self.total <= 0:
            raise MotionError("Il numero di cicli deve essere maggiore di zero.")
        self.completed = 0
        self.active = True

    def next_index(self) -> int | None:
        if not self.active or self.completed >= self.total:
            return None
        return self.completed + 1

    def mark_step_completed(self) -> bool:
        if not self.active:
            return True
        self.completed += 1
        if self.completed >= self.total:
            self.active = False
            return True
        return False

    def cancel(self) -> None:
        self.active = False


class PhidgetStepperController:
    """Backend thread-safe; nessuna chiamata Dear PyGui e' eseguita qui."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.stepper = Stepper()
        self.limit_min = DigitalInput()
        self.limit_max = DigitalInput()
        self.lock = threading.RLock()
        self.gui_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.connected = False
        self.homed = False
        self.state = ControllerState.DISCONNECTED
        self.limit_min_raw: bool | None = None
        self.limit_max_raw: bool | None = None
        self.limit_min_active = False
        self.limit_max_active = False
        self.command_direction = 0
        self.stop_requested = threading.Event()
        self.cancel_requested = threading.Event()
        self.stopped_event = threading.Event()
        self.min_active_event = threading.Event()
        self.min_released_event = threading.Event()
        self.home_contact_position: float | None = None
        self.cycle_progress: CycleProgress | None = None
        # Prenotazione impostata prima di avviare il thread HOME: evita che
        # due click ravvicinati creino due worker sullo stesso controller.
        self.homing_worker_active = False
        # La scansione SAR viene orchestrata dalla GUI radar, ma il backend
        # mantiene l'esclusione dei comandi manuali sul controller fisico.
        self.external_scan_active = False

    def _emit(self, event: str, value: Any) -> None:
        self.gui_queue.put((event, value))

    def log(self, message: str) -> None:
        self._emit("log", f"[{time.strftime('%H:%M:%S')}] {message}")

    def _set_state(self, state: ControllerState, detail: str | None = None) -> None:
        with self.lock:
            self.state = state
        self._emit("state", state.value if detail is None else f"{state.value}: {detail}")
        self._emit("snapshot", self.snapshot())

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            progress = self.cycle_progress
            return {
                "connected": self.connected,
                "homed": self.homed,
                "state": self.state.value,
                "min_raw": self.limit_min_raw,
                "max_raw": self.limit_max_raw,
                "min_active": self.limit_min_active,
                "max_active": self.limit_max_active,
                "completed": progress.completed if progress else 0,
                "total": progress.total if progress else 0,
                "cycle_active": progress.active if progress else False,
                "scan_active": self.external_scan_active,
                "motion_active": self.command_direction != 0,
            }

    @staticmethod
    def _limit_active(raw: bool, polarity: str) -> bool:
        return raw if polarity == "NO" else not raw

    def _configure_addressing(self) -> None:
        d = self.config.device
        for device in (self.stepper, self.limit_min, self.limit_max):
            if d.serial_number is not None:
                device.setDeviceSerialNumber(d.serial_number)
        self.stepper.setChannel(d.stepper_channel)
        self.limit_min.setChannel(d.limit_min_channel)
        self.limit_max.setChannel(d.limit_max_channel)

    def _configure_callbacks(self) -> None:
        self.stepper.setOnAttachHandler(self._on_attach)
        self.stepper.setOnDetachHandler(self._on_detach)
        self.stepper.setOnErrorHandler(self._on_error)
        self.stepper.setOnPositionChangeHandler(self._on_position_change)
        self.stepper.setOnStoppedHandler(self._on_stopped)
        self.limit_min.setOnStateChangeHandler(self._on_limit_min_change)
        self.limit_max.setOnStateChangeHandler(self._on_limit_max_change)

    def connect(self) -> None:
        if self.connected:
            return
        try:
            self.config.validate()
            self._configure_addressing()
            self._configure_callbacks()
            # Aprire prima gli input consente di conoscere subito una condizione limite.
            self.limit_min.openWaitForAttachment(5000)
            self.limit_max.openWaitForAttachment(5000)
            self.stepper.openWaitForAttachment(5000)
            with self.lock:
                self.limit_min_raw = bool(self.limit_min.getState())
                self.limit_max_raw = bool(self.limit_max.getState())
                self.limit_min_active = self._limit_active(self.limit_min_raw, self.config.device.limit_min_polarity)
                self.limit_max_active = self._limit_active(self.limit_max_raw, self.config.device.limit_max_polarity)
                if self.limit_min_active and self.limit_max_active:
                    raise MotionError("MIN e MAX risultano entrambi attivi: controllare cablaggio/polarita'.")
                self._apply_motor_settings()
                self.connected = True
                self.homed = False
            self._set_state(ControllerState.IDLE)
            self._emit("position", self.stepper.getPosition())
            self.log("Phidget 1063 collegato. Eseguire HOME prima del ciclo.")
        except Exception as exc:
            self._safe_close()
            self._set_state(ControllerState.FAULT, "connessione fallita")
            self.log(f"Errore di connessione: {exc}")

    def disconnect(self) -> None:
        self.cancel_motion(disengage=True)
        self._safe_close()
        with self.lock:
            self.connected = False
            self.homed = False
            self.command_direction = 0
            self.cycle_progress = None
            self.external_scan_active = False
        self._set_state(ControllerState.DISCONNECTED)
        self.log("Phidget disconnesso.")

    def _safe_close(self) -> None:
        for device in (self.stepper, self.limit_min, self.limit_max):
            try:
                device.close()
            except Exception:
                pass

    def _apply_motor_settings(self) -> None:
        motor = self.config.motor
        self.stepper.setAcceleration(motor.acceleration_microsteps_s2)
        self.stepper.setVelocityLimit(motor.velocity_microsteps_s)
        self.stepper.setCurrentLimit(motor.current_limit_a)

    def _require_connected(self) -> None:
        if not self.connected:
            raise MotionError("Phidget non collegato.")

    def _require_manual_motion_allowed(self) -> None:
        with self.lock:
            if not self.connected:
                raise MotionError("Phidget non collegato.")
            if self.state in {ControllerState.FAULT, ControllerState.EMERGENCY}:
                raise MotionError("Ripristinare la connessione e rieseguire HOME prima di muovere il carrello.")
            if self.state in {
                ControllerState.HOMING,
                ControllerState.CYCLE_MOVING,
                ControllerState.CYCLE_WAITING,
                ControllerState.SCAN_MOVING,
                ControllerState.SCAN_WAITING,
            } or self.external_scan_active or self.homing_worker_active:
                raise MotionError("Un'operazione automatica e' gia' in corso.")
            if self.command_direction != 0:
                raise MotionError("Attendere l'arresto del movimento corrente prima di inviare un altro comando.")

    def begin_external_scan(self) -> None:
        """Riserva il controller alla scansione SAR e blocca i comandi manuali."""
        """Riserva il 1063 per una scansione coordinata dal radar.

        Non fidarsi del solo stato READY: i jog manuali storici impostano
        READY subito dopo aver inviato il target, quindi controlliamo anche
        l'evento di moto ancora pendente.
        """
        self._require_connected()
        with self.lock:
            if self.external_scan_active:
                raise MotionError("Una scansione SAR e' gia' in corso.")
            if not self.homed:
                raise MotionError("Eseguire HOME prima di avviare la scansione SAR.")
            if self.state not in {ControllerState.READY, ControllerState.STOPPED}:
                raise MotionError("Il carrello deve essere fermo e pronto prima della scansione SAR.")
            if self.command_direction != 0:
                raise MotionError("Attendere il completamento del movimento manuale prima della scansione SAR.")
            if self.limit_min_active and self.limit_max_active:
                raise MotionError("Entrambi i finecorsa sono attivi: controllare cablaggio/polarita'.")
            self.external_scan_active = True
            self.stop_requested.clear()
            self.cancel_requested.clear()
        self._set_state(ControllerState.SCAN_WAITING, "pronto alla prima cattura")
        self.log("Scansione SAR riservata al controller radar.")

    def finish_external_scan(self, success: bool) -> None:
        """Rilascia la riserva SAR riportando il controller in uno stato coerente."""
        """Rilascia l'esclusione del carrello quando la scansione termina."""
        with self.lock:
            was_active = self.external_scan_active
            self.external_scan_active = False
            state = self.state
        if not was_active:
            return
        if state in {ControllerState.FAULT, ControllerState.EMERGENCY, ControllerState.LIMIT_STOPPED}:
            return
        if self.cancel_requested.is_set() or self.stop_requested.is_set():
            self._set_state(ControllerState.STOPPED, "scansione annullata - coppia mantenuta")
        elif success:
            self._set_state(ControllerState.READY, "scansione SAR completata - coppia mantenuta")
        else:
            self._set_state(ControllerState.FAULT, "scansione SAR fallita")

    def position_microsteps(self) -> int:
        """Quota attuale riferita al HOME, arrotondata al microstep."""
        self._require_connected()
        return int(round(self.stepper.getPosition()))

    def position_mm(self) -> float:
        return float(self.config.mechanics.mm_from_microsteps(self.position_microsteps()))

    def move_absolute_microsteps_and_wait(
        self,
        target_microsteps: int,
        timeout_seconds: float,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """Esegue un target assoluto e attende il segnale di arresto del motore."""
        """Muove un target di scansione e ritorna solo dopo ``Stopped``.

        Questo è il percorso usato dalla scansione SAR: non accumula arrotondamenti
        e non espone la GUI a una corsa ancora in movimento.
        """
        self._require_connected()
        with self.lock:
            if not self.external_scan_active:
                raise MotionError("Il movimento SAR richiede una scansione attiva.")
            if self.state not in {ControllerState.SCAN_WAITING, ControllerState.SCAN_MOVING}:
                raise MotionError("Il carrello non e' nello stato valido per il movimento SAR.")
        if cancel_event is not None and cancel_event.is_set():
            raise MotionError("Scansione annullata.")

        target = int(target_microsteps)
        current = self.position_microsteps()
        direction = (target > current) - (target < current)
        if direction == 0:
            self._emit("position", float(current))
            return

        target_mm = self.config.mechanics.mm_from_microsteps(target)
        self._set_state(ControllerState.SCAN_MOVING, f"target {target_mm:.3f} mm")
        self._issue_target(target, direction)

        deadline = time.monotonic() + max(0.001, float(timeout_seconds))
        while True:
            if self.cancel_requested.is_set() or self.stop_requested.is_set() or (
                cancel_event is not None and cancel_event.is_set()
            ):
                raise MotionError("Scansione annullata.")
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                # Un timeout non deve lasciare il target precedente attivo:
                # arresta mantenendo coppia e richiede un controllo operatore.
                self._stop_keep_torque()
                self._set_state(ControllerState.FAULT, "timeout movimento scansione SAR")
                raise TimeoutError("Timeout movimento scansione SAR.")
            if self.stopped_event.wait(min(0.05, remaining)):
                break

        if self.cancel_requested.is_set() or self.stop_requested.is_set() or (
            cancel_event is not None and cancel_event.is_set()
        ):
            raise MotionError("Scansione annullata.")
        actual = self.position_microsteps()
        self._emit("position", float(actual))
        self._set_state(ControllerState.SCAN_WAITING, f"arrivato a {self.position_mm():.3f} mm")

    def _on_attach(self, device: Stepper) -> None:
        self.log(f"Stepper rilevato, seriale {device.getDeviceSerialNumber()}.")

    def _on_detach(self, device: Stepper) -> None:
        with self.lock:
            self.connected = False
            self.homed = False
        self.cancel_requested.set()
        self._set_state(ControllerState.FAULT, "controller scollegato")
        self.log("Controller scollegato: ciclo annullato.")

    def _on_error(self, device: Stepper, error_code: int, description: str) -> None:
        self.cancel_requested.set()
        with self.lock:
            # Dopo un errore hardware non possiamo più garantire né la quota
            # open-loop né che il comando precedente sia ancora valido.
            self.homed = False
            self.command_direction = 0
        self.log(f"Errore Phidget {error_code}: {description}")
        try:
            self.stepper.setEngaged(False)
        except Exception:
            pass
        self._set_state(ControllerState.FAULT, "motore disabilitato")

    def _on_position_change(self, device: Stepper, position: float) -> None:
        self._emit("position", position)

    def _on_stopped(self, device: Stepper) -> None:
        self.command_direction = 0
        self.stopped_event.set()
        self._emit("motion_stopped", True)

    def _on_limit_min_change(self, device: DigitalInput, raw: bool) -> None:
        active = self._limit_active(bool(raw), self.config.device.limit_min_polarity)
        with self.lock:
            self.limit_min_raw = bool(raw)
            self.limit_min_active = active
        if active:
            self.min_active_event.set()
            self.log("Finecorsa MIN attivo.")
            if self.command_direction < 0:
                # Durante HOME il contatto MIN e' l'evento atteso: fermiamo
                # mantenendo coppia, ma non annulliamo il worker che deve
                # azzerare la quota dopo l'evento Stopped.
                if self.state == ControllerState.HOMING:
                    self.home_contact_position = self.stepper.getPosition()
                    self._stop_at_current()
                else:
                    self._limit_stop("MIN")
        else:
            self.min_released_event.set()
            self.log("Finecorsa MIN rilasciato.")
        self._emit("snapshot", self.snapshot())

    def _on_limit_max_change(self, device: DigitalInput, raw: bool) -> None:
        active = self._limit_active(bool(raw), self.config.device.limit_max_polarity)
        with self.lock:
            self.limit_max_raw = bool(raw)
            self.limit_max_active = active
        if active:
            self.log("Finecorsa MAX attivo.")
            if self.command_direction > 0:
                self._limit_stop("MAX")
        else:
            self.log("Finecorsa MAX rilasciato.")
        self._emit("snapshot", self.snapshot())

    def _limit_stop(self, which: str) -> None:
        self.cancel_requested.set()
        if self.cycle_progress:
            self.cycle_progress.cancel()
        try:
            self.stepper.setTargetPosition(self.stepper.getPosition())
        except PhidgetException as exc:
            self.log(f"Errore stop al finecorsa {which}: {exc}")
        self._set_state(ControllerState.LIMIT_STOPPED, which)

    def _stop_at_current(self) -> None:
        """Arresta secondo il profilo configurato senza cancellare HOME."""
        try:
            self.stepper.setTargetPosition(self.stepper.getPosition())
        except PhidgetException as exc:
            self.log(f"Errore arresto HOME: {exc}")

    def _issue_target(self, target: int, direction: int, velocity: float | None = None) -> None:
        self._require_connected()
        if not direction_is_allowed(direction, self.limit_min_active, self.limit_max_active):
            side = "MIN" if direction < 0 else "MAX"
            raise MotionError(f"Movimento verso {side} bloccato: finecorsa attivo.")
        self.stopped_event.clear()
        self.command_direction = direction
        self._apply_motor_settings()
        if velocity is not None:
            self.stepper.setVelocityLimit(velocity)
        self.stepper.setEngaged(True)
        self.stepper.setTargetPosition(target)

    def move_relative_mm(self, distance_mm: float) -> None:
        self._require_manual_motion_allowed()
        if distance_mm == 0:
            raise MotionError("Lo spostamento deve essere diverso da zero.")
        direction = 1 if distance_mm > 0 else -1
        microsteps = self.config.mechanics.microsteps_from_mm(distance_mm)
        if microsteps == 0:
            raise MotionError("Spostamento troppo piccolo per la risoluzione configurata.")
        target = int(round(self.stepper.getPosition())) + microsteps
        self.cancel_requested.clear()
        self._issue_target(target, direction)
        self._set_state(ControllerState.READY, f"jog {distance_mm:.3f} mm")
        self.log(f"Jog {distance_mm:.4f} mm ({microsteps} microstep).")

    def move_absolute_mm(self, target_mm: float) -> None:
        self._require_manual_motion_allowed()
        target = self.config.mechanics.microsteps_from_mm(target_mm)
        current = int(round(self.stepper.getPosition()))
        direction = (target > current) - (target < current)
        if direction == 0:
            return
        self.cancel_requested.clear()
        self._issue_target(target, direction)
        self._set_state(ControllerState.READY, f"target {target_mm:.3f} mm")

    def start_homing(self) -> None:
        """Prenota e avvia HOME in un worker, impedendo avvii concorrenti."""
        # Controllo e prenotazione devono avvenire sotto lo stesso lock: lo
        # stato HOMING impostato dal worker sarebbe troppo tardi per bloccare
        # un secondo click immediato.
        with self.lock:
            if not self.connected:
                raise MotionError("Phidget non collegato.")
            if self.state in {ControllerState.FAULT, ControllerState.EMERGENCY}:
                raise MotionError("Ripristinare la connessione e rieseguire HOME prima di muovere il carrello.")
            if self.state in {
                ControllerState.HOMING,
                ControllerState.CYCLE_MOVING,
                ControllerState.CYCLE_WAITING,
                ControllerState.SCAN_MOVING,
                ControllerState.SCAN_WAITING,
            } or self.external_scan_active or self.homing_worker_active:
                raise MotionError("Un'operazione automatica e' gia' in corso.")
            if self.command_direction != 0:
                raise MotionError("Attendere l'arresto del movimento corrente prima di eseguire HOME.")
            self.homing_worker_active = True
            self.state = ControllerState.HOMING

        self._emit("state", ControllerState.HOMING.value)
        self._emit("snapshot", self.snapshot())
        try:
            threading.Thread(target=self._homing_worker, name="phidget-home", daemon=True).start()
        except Exception:
            with self.lock:
                self.homing_worker_active = False
                if self.state == ControllerState.HOMING:
                    self.state = ControllerState.READY if self.homed else ControllerState.IDLE
            self._emit("snapshot", self.snapshot())
            raise

    def _wait_event(self, event: threading.Event, seconds: float, message: str) -> None:
        if not event.wait(seconds):
            raise TimeoutError(message)
        if self.stop_requested.is_set() or self.cancel_requested.is_set():
            raise MotionError("Operazione annullata.")

    def _homing_worker(self) -> None:
        """Cerca MIN, rilascia il contatto e azzera la coordinata open-loop."""
        try:
            self.stop_requested.clear()
            self.cancel_requested.clear()
            self.min_active_event.clear()
            self.min_released_event.clear()
            self.home_contact_position = None
            home = self.config.homing
            if self.limit_min_active:
                self.log("MIN gia' attivo: allontanamento prima di HOME.")
                release_steps = abs(self.config.mechanics.microsteps_from_mm(home.release_distance_mm))
                self._issue_target(int(round(self.stepper.getPosition())) + release_steps, 1, home.velocity_microsteps_s)
                self._wait_event(self.min_released_event, home.timeout_seconds, "Timeout rilascio MIN.")
                self._wait_event(self.stopped_event, home.timeout_seconds, "Timeout arresto rilascio MIN.")

            search_steps = abs(self.config.mechanics.microsteps_from_mm(home.search_distance_mm))
            self.min_active_event.clear()
            self._issue_target(int(round(self.stepper.getPosition())) - search_steps, -1, home.velocity_microsteps_s)
            self.log("Ricerca del finecorsa MIN.")
            self._wait_event(self.min_active_event, home.timeout_seconds, "Timeout HOME: MIN non raggiunto.")
            self._wait_event(self.stopped_event, home.timeout_seconds, "Timeout arresto HOME.")
            # Il 1063 non ha setPosition: l'offset rende zero la quota
            # registrata esattamente all'attivazione del finecorsa MIN.
            if self.home_contact_position is None:
                raise MotionError("HOME non ha registrato il contatto MIN.")
            self.stepper.addPositionOffset(-self.home_contact_position)
            self._emit("position", 0.0)

            # Lascia fisicamente il finecorsa dopo aver fissato lo zero al
            # contatto, cosi' MIN e' disponibile per un HOME successivo.
            release_steps = abs(self.config.mechanics.microsteps_from_mm(home.release_distance_mm))
            self.min_released_event.clear()
            self._issue_target(release_steps, 1, home.velocity_microsteps_s)
            self.log(f"Rilascio del finecorsa MIN di {home.release_distance_mm:.3f} mm.")
            self._wait_event(self.min_released_event, home.timeout_seconds, "Timeout rilascio MIN dopo HOME.")
            self._wait_event(self.stopped_event, home.timeout_seconds, "Timeout arresto rilascio MIN dopo HOME.")

            with self.lock:
                self.homed = True
            self._set_state(ControllerState.READY, "HOME completato")
            self.log("HOME completato: MIN impostato a 0.000 mm e finecorsa rilasciato.")
        except Exception as exc:
            self._stop_keep_torque()
            with self.lock:
                self.homed = False
            if not self.stop_requested.is_set():
                self._set_state(ControllerState.FAULT, "HOME fallito")
            self.log(f"Errore HOME: {exc}")
        finally:
            with self.lock:
                self.homing_worker_active = False

    def _stop_keep_torque(self) -> None:
        self.cancel_requested.set()
        if self.cycle_progress:
            self.cycle_progress.cancel()
        try:
            if self.connected:
                self.stepper.setTargetPosition(self.stepper.getPosition())
        except Exception as exc:
            self.log(f"Errore STOP: {exc}")

    def stop(self) -> None:
        self.stop_requested.set()
        self._stop_keep_torque()
        protected_states = {
            ControllerState.FAULT,
            ControllerState.EMERGENCY,
            ControllerState.LIMIT_STOPPED,
        }
        with self.lock:
            protected_state = self.state if self.state in protected_states else None
            if protected_state is None:
                self.state = ControllerState.STOPPED
        if protected_state is None:
            self._emit("state", f"{ControllerState.STOPPED.value}: coppia mantenuta")
            self._emit("snapshot", self.snapshot())
            self.log("STOP: moto annullato, coppia mantenuta.")
        else:
            self._emit("snapshot", self.snapshot())
            self.log(f"STOP richiesto; stato protetto preservato: {protected_state.value}.")

    def cancel_motion(self, disengage: bool = False) -> None:
        self.stop_requested.set()
        self._stop_keep_torque()
        if disengage:
            try:
                self.stepper.setEngaged(False)
            except Exception:
                pass

    def emergency_stop(self) -> None:
        """Disabilita il motore: richiede un nuovo HOME prima di qualsiasi moto."""
        self.stop_requested.set()
        self.cancel_requested.set()
        if self.cycle_progress:
            self.cycle_progress.cancel()
        try:
            self.stepper.setEngaged(False)
        except Exception as exc:
            self.log(f"Errore arresto emergenza: {exc}")
        with self.lock:
            self.homed = False
        self._set_state(ControllerState.EMERGENCY)
        self.log("EMERGENZA software: bobine disabilitate.")

    def start_cycle(self, cycle: CycleConfig) -> None:
        cycle_config = copy.deepcopy(cycle)
        candidate = copy.deepcopy(self.config)
        candidate.cycle = cycle_config
        candidate.validate()
        self._require_connected()
        if self.state not in {ControllerState.READY, ControllerState.STOPPED}:
            raise MotionError("Il ciclo puo' partire solo dopo HOME e con carrello fermo.")
        if self.limit_min_active and self.limit_max_active:
            raise MotionError("Entrambi i finecorsa sono attivi.")
        with self.lock:
            if self.cycle_progress and self.cycle_progress.active:
                raise MotionError("Un ciclo e' gia' in corso.")
            progress = CycleProgress(cycle_config.cycles)
            progress.start(self.homed, self.limit_max_active)
            self.cycle_progress = progress
        self.stop_requested.clear()
        self.cancel_requested.clear()
        threading.Thread(target=self._cycle_worker, args=(cycle_config,), name="phidget-cycle", daemon=True).start()

    def _cycle_worker(self, cycle: CycleConfig) -> None:
        try:
            assert self.cycle_progress is not None
            step_microsteps = self.config.mechanics.microsteps_from_mm(cycle.step_mm)
            if step_microsteps <= 0:
                raise MotionError("Il passo del ciclo non produce microstep positivi.")
            while self.cycle_progress.next_index() is not None:
                if self.cancel_requested.is_set():
                    return
                index = self.cycle_progress.next_index()
                self._set_state(ControllerState.CYCLE_MOVING, f"passo {index}/{cycle.cycles}")
                target = int(round(self.stepper.getPosition())) + step_microsteps
                self._issue_target(target, 1)
                # Ampio timeout: evita un worker sospeso in caso di perdita evento.
                self._wait_event(self.stopped_event, self.config.homing.timeout_seconds * 4, "Timeout movimento ciclo.")
                if self.cancel_requested.is_set():
                    return
                completed = self.cycle_progress.mark_step_completed()
                self._emit("snapshot", self.snapshot())
                if completed:
                    self._set_state(ControllerState.READY, "ciclo completato - coppia mantenuta")
                    self.log("Ciclo completato.")
                    return
                self._set_state(ControllerState.CYCLE_WAITING, f"attesa {cycle.wait_seconds:.2f} s")
                if self.cancel_requested.wait(cycle.wait_seconds):
                    return
        except Exception as exc:
            if not self.cancel_requested.is_set():
                self._stop_keep_torque()
                self._set_state(ControllerState.FAULT, "ciclo fallito")
                self.log(f"Errore ciclo: {exc}")

    def update_config(self, config: AppConfig) -> None:
        config.validate()
        if self.connected:
            raise MotionError("Disconnettere il 1063 prima di cambiare configurazione.")
        self.config = config


def safe_call(controller: PhidgetStepperController, callback, *args) -> None:
    try:
        callback(*args)
    except Exception as exc:
        controller.log(f"Errore comando: {exc}")


class StepperGui:
    def __init__(self, controller: PhidgetStepperController):
        self.controller = controller
        self.config = controller.config
        self.deadline: float | None = None

    def _config_from_ui(self) -> AppConfig:
        return AppConfig(
            device=DeviceConfig(
                serial_number=self._optional_int(dpg.get_value("serial_number")),
                stepper_channel=0,
                limit_min_channel=int(dpg.get_value("limit_min_channel")),
                limit_max_channel=int(dpg.get_value("limit_max_channel")),
                limit_min_polarity=dpg.get_value("limit_min_polarity"),
                limit_max_polarity=dpg.get_value("limit_max_polarity"),
            ),
            mechanics=MechanicsConfig(
                belt_pitch_mm=float(dpg.get_value("belt_pitch_mm")),
                pulley_teeth=int(dpg.get_value("pulley_teeth")),
                motor_full_steps_per_rev=int(dpg.get_value("motor_steps_rev")),
                gear_reduction=float(dpg.get_value("gear_reduction")),
            ),
            motor=MotorConfig(
                velocity_microsteps_s=float(dpg.get_value("velocity")),
                acceleration_microsteps_s2=float(dpg.get_value("acceleration")),
                current_limit_a=float(dpg.get_value("current_limit")),
            ),
            homing=HomingConfig(
                search_distance_mm=float(dpg.get_value("home_search_mm")),
                release_distance_mm=float(dpg.get_value("home_release_mm")),
                velocity_microsteps_s=float(dpg.get_value("home_velocity")),
                timeout_seconds=float(dpg.get_value("home_timeout")),
            ),
            cycle=self._cycle_from_ui(),
        )

    @staticmethod
    def _optional_int(value: str) -> int | None:
        value = value.strip()
        return int(value) if value else None

    def _cycle_from_ui(self) -> CycleConfig:
        return CycleConfig(
            step_mm=float(dpg.get_value("cycle_step_mm")),
            wait_seconds=float(dpg.get_value("cycle_wait_s")),
            cycles=int(dpg.get_value("cycle_count")),
        )

    def _save_settings(self) -> None:
        try:
            config = self._config_from_ui()
            self.controller.update_config(config)
            save_config(config)
            self.config = config
            dpg.set_value("mm_per_microstep", f"{config.mechanics.mm_per_microstep:.9f} mm/microstep")
            self.controller.log("Configurazione salvata. Connettere il 1063.")
        except Exception as exc:
            self.controller.log(f"Configurazione non salvata: {exc}")

    def _connect(self) -> None:
        threading.Thread(target=self.controller.connect, name="phidget-connect", daemon=True).start()

    def _home(self) -> None:
        safe_call(self.controller, self.controller.start_homing)

    def _start_cycle(self) -> None:
        try:
            cycle = self._cycle_from_ui()
            self.deadline = None
            self.controller.start_cycle(cycle)
        except Exception as exc:
            self.controller.log(f"Ciclo non avviato: {exc}")

    def _jog(self, sign: int) -> None:
        safe_call(self.controller, self.controller.move_relative_mm, sign * abs(float(dpg.get_value("jog_mm"))))

    def _absolute(self) -> None:
        safe_call(self.controller, self.controller.move_absolute_mm, float(dpg.get_value("target_mm")))

    def build(self) -> None:
        c = self.config
        dpg.create_context()
        with dpg.theme(tag="primary_button_theme"):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (37, 112, 188))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (55, 139, 222))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (26, 82, 145))
        with dpg.theme(tag="home_button_theme"):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (36, 133, 87))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (47, 161, 105))
        with dpg.theme(tag="stop_button_theme"):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (180, 103, 25))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (214, 129, 38))
        with dpg.theme(tag="emergency_button_theme"):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (178, 48, 48))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (220, 62, 62))

        with dpg.window(tag="main", label="Phidget 1063 - Carrello a passi", width=980, height=800):
            dpg.add_text("CONTROLLO CARRELLO", color=(100, 190, 255))
            dpg.add_text("1. Connetti   2. HOME   3. Avvia il ciclo", color=(210, 210, 210))
            dpg.add_separator()
            with dpg.child_window(height=92, border=True):
                with dpg.group(horizontal=True):
                    with dpg.group():
                        dpg.add_text("CONNESSIONE", color=(150, 150, 150))
                        dpg.add_text("disconnesso", tag="connection", color=(255, 210, 100))
                    dpg.add_spacer(width=38)
                    with dpg.group():
                        dpg.add_text("STATO", color=(150, 150, 150))
                        dpg.add_text(ControllerState.DISCONNECTED.value, tag="state")
                    dpg.add_spacer(width=38)
                    with dpg.group():
                        dpg.add_text("POSIZIONE", color=(150, 150, 150))
                        dpg.add_text("-- mm", tag="position", color=(100, 220, 255))
                    dpg.add_spacer(width=38)
                    with dpg.group():
                        dpg.add_text("RIFERIMENTO HOME", color=(150, 150, 150))
                        dpg.add_text("non eseguito", tag="homed")
                    dpg.add_spacer(width=38)
                    with dpg.group():
                        dpg.add_text("CICLO", color=(150, 150, 150))
                        dpg.add_text("0/0", tag="cycle_progress")
                        dpg.add_text("Attesa: --", tag="countdown")

            with dpg.tab_bar():
                with dpg.tab(label="OPERAZIONI"):
                    dpg.add_spacer(height=5)
                    with dpg.group(horizontal=True):
                        with dpg.child_window(width=455, height=345, border=True):
                            dpg.add_text("COMANDI PRINCIPALI", color=(100, 190, 255))
                            dpg.add_text("Usare HOME prima del ciclo automatico.")
                            dpg.add_spacer(height=6)
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="1. CONNETTI", width=205, height=42, callback=lambda: self._connect())
                                dpg.add_button(label="Disconnetti", width=205, height=42, callback=lambda: threading.Thread(target=self.controller.disconnect, daemon=True).start())
                            dpg.add_spacer(height=8)
                            dpg.add_button(label="2. HOME  (trova MIN e imposta zero)", tag="home_button", width=420, height=54, callback=lambda: self._home())
                            dpg.add_spacer(height=8)
                            dpg.add_button(label="STOP  -  ferma e mantiene la coppia", tag="stop_button", width=420, height=46, callback=lambda: safe_call(self.controller, self.controller.stop))
                            dpg.add_spacer(height=8)
                            dpg.add_button(label="ARRESTO DI EMERGENZA SOFTWARE", tag="emergency_button", width=420, height=52, callback=lambda: safe_call(self.controller, self.controller.emergency_stop))
                            dpg.add_text("Per la sicurezza fisica serve anche un E-stop cablato.", color=(255, 135, 135))
                        with dpg.child_window(width=455, height=345, border=True):
                            dpg.add_text("3. CICLO AUTOMATICO", color=(100, 190, 255))
                            dpg.add_text("Avanza verso MAX: si muove, arriva, attende e ripete.")
                            dpg.add_spacer(height=8)
                            dpg.add_input_float(label="Spostamento per ciclo  X  [mm]", tag="cycle_step_mm", default_value=c.cycle.step_mm, min_value=0.001, min_clamped=True, width=180)
                            dpg.add_input_float(label="Pausa dopo l'arrivo  Y  [s]", tag="cycle_wait_s", default_value=c.cycle.wait_seconds, min_value=0, min_clamped=True, width=180)
                            dpg.add_input_int(label="Numero di spostamenti  N", tag="cycle_count", default_value=c.cycle.cycles, min_value=1, min_clamped=True, width=180)
                            dpg.add_spacer(height=12)
                            dpg.add_button(label="AVVIA CICLO", tag="start_cycle_button", width=420, height=54, callback=lambda: self._start_cycle())
                            dpg.add_text("Il ciclo parte solo dopo HOME e con MAX libero.", color=(190, 190, 190))
                    dpg.add_spacer(height=8)
                    with dpg.group(horizontal=True):
                        with dpg.child_window(width=455, height=185, border=True):
                            dpg.add_text("MOVIMENTO MANUALE", color=(100, 190, 255))
                            dpg.add_input_float(label="Distanza jog [mm]", tag="jog_mm", default_value=c.cycle.step_mm, min_value=0.001, min_clamped=True, width=180)
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="JOG verso MIN  (-)", width=205, height=40, callback=lambda: self._jog(-1))
                                dpg.add_button(label="JOG verso MAX  (+)", width=205, height=40, callback=lambda: self._jog(1))
                        with dpg.child_window(width=455, height=185, border=True):
                            dpg.add_text("POSIZIONE ASSOLUTA", color=(100, 190, 255))
                            dpg.add_input_float(label="Target [mm]", tag="target_mm", default_value=0.0, width=180)
                            dpg.add_button(label="VAI ALLA POSIZIONE", width=420, height=40, callback=lambda: self._absolute())
                    dpg.add_spacer(height=8)
                    with dpg.collapsing_header(label="Diagnostica finecorsa", default_open=False):
                        dpg.add_text("raw = stato elettrico; interpretato = stato usato dal programma.")
                        with dpg.group(horizontal=True):
                            dpg.add_text("MIN raw:")
                            dpg.add_text("--", tag="min_raw")
                            dpg.add_text("     MIN:")
                            dpg.add_text("--", tag="min_active")
                            dpg.add_spacer(width=50)
                            dpg.add_text("MAX raw:")
                            dpg.add_text("--", tag="max_raw")
                            dpg.add_text("     MAX:")
                            dpg.add_text("--", tag="max_active")
                    dpg.add_separator()
                    dpg.add_text("Log eventi")
                    dpg.add_input_text(tag="log", multiline=True, readonly=True, width=-1, height=120)

                with dpg.tab(label="IMPOSTAZIONI"):
                    dpg.add_text("Modificare e salvare le impostazioni soltanto a 1063 disconnesso.", color=(255, 210, 100))
                    with dpg.group(horizontal=True):
                        with dpg.child_window(width=455, height=310, border=True):
                            dpg.add_text("CONNESSIONE E FINECORSA", color=(100, 190, 255))
                            dpg.add_input_text(label="Seriale (vuoto = unico 1063)", tag="serial_number", default_value="" if c.device.serial_number is None else str(c.device.serial_number), width=190)
                            dpg.add_separator()
                            dpg.add_text("MIN")
                            with dpg.group(horizontal=True):
                                dpg.add_input_int(label="Canale", tag="limit_min_channel", default_value=c.device.limit_min_channel, min_value=0, max_value=3, min_clamped=True, max_clamped=True, width=100)
                                dpg.add_combo(["NO", "NC"], label="Polarita'", tag="limit_min_polarity", default_value=c.device.limit_min_polarity, width=100)
                            dpg.add_text("MAX")
                            with dpg.group(horizontal=True):
                                dpg.add_input_int(label="Canale", tag="limit_max_channel", default_value=c.device.limit_max_channel, min_value=0, max_value=3, min_clamped=True, max_clamped=True, width=100)
                                dpg.add_combo(["NO", "NC"], label="Polarita'", tag="limit_max_polarity", default_value=c.device.limit_max_polarity, width=100)
                            dpg.add_text("Verificare i valori raw nella scheda Operazioni.", color=(190, 190, 190))
                        with dpg.child_window(width=455, height=310, border=True):
                            dpg.add_text("MECCANICA", color=(100, 190, 255))
                            dpg.add_input_float(label="Passo cinghia [mm]", tag="belt_pitch_mm", default_value=c.mechanics.belt_pitch_mm, min_value=0.001, min_clamped=True, width=150)
                            dpg.add_input_int(label="Denti puleggia", tag="pulley_teeth", default_value=c.mechanics.pulley_teeth, min_value=1, min_clamped=True, width=150)
                            dpg.add_input_int(label="Passi motore/giro", tag="motor_steps_rev", default_value=c.mechanics.motor_full_steps_per_rev, min_value=1, min_clamped=True, width=150)
                            dpg.add_input_float(label="Riduzione", tag="gear_reduction", default_value=c.mechanics.gear_reduction, format="%.5f", min_value=0.001, min_clamped=True, width=150)
                            dpg.add_separator()
                            dpg.add_text("Risoluzione calcolata:")
                            dpg.add_text(f"{c.mechanics.mm_per_microstep:.9f} mm/microstep", tag="mm_per_microstep", color=(100, 220, 255))
                    with dpg.group(horizontal=True):
                        with dpg.child_window(width=455, height=270, border=True):
                            dpg.add_text("MOTORE", color=(100, 190, 255))
                            dpg.add_input_float(label="Velocita' [microstep/s]", tag="velocity", default_value=c.motor.velocity_microsteps_s, min_value=1, min_clamped=True, width=170)
                            dpg.add_input_float(label="Accelerazione [microstep/s²]", tag="acceleration", default_value=c.motor.acceleration_microsteps_s2, min_value=1, min_clamped=True, width=170)
                            dpg.add_input_float(label="Corrente massima [A]", tag="current_limit", default_value=c.motor.current_limit_a, min_value=0.01, min_clamped=True, width=170)
                        with dpg.child_window(width=455, height=270, border=True):
                            dpg.add_text("HOME", color=(100, 190, 255))
                            dpg.add_input_float(label="Corsa di ricerca [mm]", tag="home_search_mm", default_value=c.homing.search_distance_mm, min_value=0.001, min_clamped=True, width=170)
                            dpg.add_input_float(label="Rilascio MIN [mm]", tag="home_release_mm", default_value=c.homing.release_distance_mm, min_value=0.001, min_clamped=True, width=170)
                            dpg.add_input_float(label="Velocita' HOME", tag="home_velocity", default_value=c.homing.velocity_microsteps_s, min_value=1, min_clamped=True, width=170)
                            dpg.add_input_float(label="Timeout HOME [s]", tag="home_timeout", default_value=c.homing.timeout_seconds, min_value=0.1, min_clamped=True, width=170)
                    dpg.add_button(label="SALVA IMPOSTAZIONI", tag="save_settings_button", width=300, height=45, callback=lambda: self._save_settings())

        dpg.bind_item_theme("home_button", "home_button_theme")
        dpg.bind_item_theme("start_cycle_button", "primary_button_theme")
        dpg.bind_item_theme("save_settings_button", "primary_button_theme")
        dpg.bind_item_theme("stop_button", "stop_button_theme")
        dpg.bind_item_theme("emergency_button", "emergency_button_theme")
        dpg.create_viewport(title="Phidget 1063 - Carrello", width=1010, height=840)
        dpg.setup_dearpygui()
        dpg.set_primary_window("main", True)
        dpg.set_exit_callback(self.close)
        dpg.show_viewport()

    def _render_snapshot(self, value: dict[str, Any]) -> None:
        dpg.set_value("connection", "COLLEGATO" if value["connected"] else "disconnesso")
        dpg.set_value("homed", "eseguito" if value["homed"] else "non eseguito")
        dpg.set_value("min_raw", str(value["min_raw"]))
        dpg.set_value("max_raw", str(value["max_raw"]))
        dpg.set_value("min_active", "ATTIVO" if value["min_active"] else "libero")
        dpg.set_value("max_active", "ATTIVO" if value["max_active"] else "libero")
        dpg.set_value("cycle_progress", f"{value['completed']}/{value['total']}")

    def process_events(self) -> None:
        try:
            while True:
                event, value = self.controller.gui_queue.get_nowait()
                if event == "log":
                    current = dpg.get_value("log")
                    lines = (current + "\n" + value).strip().splitlines()
                    dpg.set_value("log", "\n".join(lines[-250:]))
                elif event == "state":
                    dpg.set_value("state", value)
                    if "attesa" in value.lower():
                        self.deadline = time.monotonic() + float(dpg.get_value("cycle_wait_s"))
                    else:
                        self.deadline = None
                elif event == "position":
                    mm = self.controller.config.mechanics.mm_from_microsteps(value)
                    dpg.set_value("position", f"{mm:.4f} mm ({value:.0f} microstep)")
                elif event == "snapshot":
                    self._render_snapshot(value)
        except queue.Empty:
            pass
        if self.deadline is not None:
            remaining = max(0.0, self.deadline - time.monotonic())
            dpg.set_value("countdown", f"{remaining:.1f} s")
        else:
            dpg.set_value("countdown", "--")

    def run(self) -> None:
        self.build()
        while dpg.is_dearpygui_running():
            self.process_events()
            dpg.render_dearpygui_frame()
        self.close()
        dpg.destroy_context()

    def close(self) -> None:
        try:
            self.controller.disconnect()
        except Exception:
            pass


def main() -> None:
    try:
        config = load_config()
    except ConfigError as exc:
        raise SystemExit(f"Configurazione non valida: {exc}") from exc
    StepperGui(PhidgetStepperController(config)).run()


if __name__ == "__main__":
    main()
