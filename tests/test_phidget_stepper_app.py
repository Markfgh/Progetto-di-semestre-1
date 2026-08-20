"""Verifica conversioni, limiti e stati sicuri del controller Phidget."""

from pathlib import Path

import pytest

from phidget_stepper_app import (
    AppConfig,
    ConfigError,
    ControllerState,
    CycleProgress,
    MotionError,
    PhidgetStepperController,
    direction_is_allowed,
    load_config,
    save_config,
)


def test_mechanics_converts_mm_and_microsteps_without_cumulative_rounding():
    config = AppConfig()
    # Cinghia 2 mm, puleggia 20T, motore 200 passi: 40 mm/giro.
    assert config.mechanics.mm_per_microstep == pytest.approx(0.0125)
    assert config.mechanics.microsteps_from_mm(1.0) == 80
    assert config.mechanics.mm_from_microsteps(80) == pytest.approx(1.0)


def test_config_rejects_invalid_limit_channels():
    config = AppConfig()
    config.device.limit_max_channel = 0
    with pytest.raises(ConfigError, match="canali digitali diversi"):
        config.validate()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_config_rejects_nonfinite_motor_values(invalid: float):
    config = AppConfig()
    config.motor.velocity_microsteps_s = invalid
    with pytest.raises(ConfigError, match="Velocita"):
        config.validate()


def test_legacy_holding_current_setting_is_ignored():
    config = AppConfig.from_dict({"motor": {"holding_current_limit_a": 0.6}})
    assert config.motor.current_limit_a == pytest.approx(0.5)


def test_yaml_round_trip_and_invalid_yaml_shape(tmp_path: Path):
    path = tmp_path / "stepper.yaml"
    config = AppConfig()
    config.cycle.step_mm = 3.25
    save_config(config, path)
    assert load_config(path).cycle.step_mm == pytest.approx(3.25)
    path.write_text("- not-a-map\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mappa"):
        load_config(path)


def test_cycle_requires_home_and_advances_only_after_completion():
    cycle = CycleProgress(total=2)
    with pytest.raises(MotionError, match="HOME"):
        cycle.start(homed=False, max_active=False)
    cycle.start(homed=True, max_active=False)
    assert cycle.next_index() == 1
    assert cycle.mark_step_completed() is False
    assert cycle.next_index() == 2
    assert cycle.mark_step_completed() is True
    assert cycle.next_index() is None


def test_limit_policy_blocks_only_toward_active_limit():
    assert not direction_is_allowed(-1, limit_min_active=True, limit_max_active=False)
    assert direction_is_allowed(1, limit_min_active=True, limit_max_active=False)
    assert not direction_is_allowed(1, limit_min_active=False, limit_max_active=True)
    assert direction_is_allowed(-1, limit_min_active=False, limit_max_active=True)


def test_manual_command_is_blocked_while_previous_motion_is_active():
    controller = PhidgetStepperController(AppConfig())
    controller.connected = True
    controller.state = ControllerState.READY
    controller.command_direction = 1

    with pytest.raises(MotionError, match="movimento corrente"):
        controller.move_relative_mm(1.0)


def test_home_is_reserved_before_worker_start(monkeypatch):
    controller = PhidgetStepperController(AppConfig())
    controller.connected = True
    controller.state = ControllerState.READY

    class DeferredThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr("phidget_stepper_app.threading.Thread", DeferredThread)

    controller.start_homing()
    assert controller.homing_worker_active
    assert controller.state == ControllerState.HOMING
    with pytest.raises(MotionError, match="operazione automatica"):
        controller.start_homing()


class _FakeStepper:
    def __init__(self):
        self.engaged = True
        self.position = 123.0

    def setEngaged(self, value):
        self.engaged = bool(value)

    def getPosition(self):
        return self.position

    def setTargetPosition(self, value):
        self.position = float(value)

    def setAcceleration(self, _value):
        pass

    def setVelocityLimit(self, _value):
        pass

    def setCurrentLimit(self, _value):
        pass


def test_phidget_error_invalidates_home_and_motion_state():
    controller = PhidgetStepperController(AppConfig())
    controller.stepper = _FakeStepper()
    controller.connected = True
    controller.homed = True
    controller.command_direction = 1

    controller._on_error(controller.stepper, 99, "errore test")

    assert not controller.homed
    assert controller.command_direction == 0
    assert controller.state == ControllerState.FAULT
    assert not controller.stepper.engaged


def test_issue_target_failure_clears_stale_motion_and_invalidates_home():
    class FailingTargetStepper(_FakeStepper):
        def setTargetPosition(self, _value):
            raise RuntimeError("target rejected")

    controller = PhidgetStepperController(AppConfig())
    controller.stepper = FailingTargetStepper()
    controller.connected = True
    controller.homed = True
    controller.state = ControllerState.READY

    with pytest.raises(RuntimeError, match="target rejected"):
        controller._issue_target(200, 1)

    assert controller.command_direction == 0
    assert not controller.homed
    assert controller.stopped_event.is_set()


def test_detach_clears_motion_and_releases_waiters():
    controller = PhidgetStepperController(AppConfig())
    controller.stepper = _FakeStepper()
    controller.connected = True
    controller.homed = True
    controller.command_direction = -1
    controller.external_scan_active = True

    controller._on_detach(controller.stepper)

    assert not controller.connected
    assert not controller.homed
    assert controller.command_direction == 0
    assert not controller.external_scan_active
    assert controller.cancel_requested.is_set()
    assert controller.stopped_event.is_set()
    assert controller.min_active_event.is_set()
    assert controller.min_released_event.is_set()
    assert controller.state == ControllerState.FAULT


@pytest.mark.parametrize(
    "protected_state",
    [ControllerState.FAULT, ControllerState.EMERGENCY, ControllerState.LIMIT_STOPPED],
)
def test_stop_preserves_protected_controller_states(protected_state):
    controller = PhidgetStepperController(AppConfig())
    controller.stepper = _FakeStepper()
    controller.connected = True
    controller.state = protected_state

    controller.stop()

    assert controller.state == protected_state
