from pathlib import Path

import pytest

from phidget_stepper_gui import (
    AppConfig,
    ConfigError,
    CycleProgress,
    MotionError,
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
