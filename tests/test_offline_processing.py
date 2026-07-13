from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from offline_processing import _read_bp_runtime_cfg


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _fallback_capture_cfg() -> dict:
    return {
        "radar": {
            "c": 3.0e8,
            "fc": 77.0e9,
        },
        "capture": {
            "tx": 2,
            "rx": 4,
        },
    }


def _offline_bp_cfg(**bp_overrides: object) -> dict:
    bp_cfg = {
        "mode": "mimo_sar",
        "phase_sign": -1,
        "motion_mode": "static_zero_doppler",
        "tx_offsets_lambda": None,
        "rx_offsets_lambda": None,
        "tx_offsets_m": None,
        "rx_offsets_m": None,
        "coherent_sum": True,
    }
    bp_cfg.update(bp_overrides)
    return {"bp": bp_cfg}


def _offline_reconstruction_cfg(*, algorithm: str = "backprojection", **bp_overrides: object) -> dict:
    payload = _offline_bp_cfg(**bp_overrides)
    payload["reconstruction"] = {"algorithm": algorithm}
    return payload


def test_read_bp_runtime_cfg_mimo_sar_rejects_legacy_motion_modes(tmp_path: Path) -> None:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "Config.yaml"
    _write_yaml(offline_cfg, _offline_bp_cfg(motion_mode="all_doppler_incoherent"))
    _write_yaml(fallback_cfg, _fallback_capture_cfg())

    with pytest.raises(ValueError, match="static_zero_doppler"):
        _read_bp_runtime_cfg(offline_cfg, fallback_cfg)


def test_read_bp_runtime_cfg_rejects_invalid_reconstruction_algorithm(tmp_path: Path) -> None:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "Config.yaml"
    _write_yaml(offline_cfg, _offline_reconstruction_cfg(algorithm="wrong_mode"))
    _write_yaml(fallback_cfg, _fallback_capture_cfg())

    with pytest.raises(ValueError, match="synthetic_range_angle"):
        _read_bp_runtime_cfg(offline_cfg, fallback_cfg)


def test_read_bp_runtime_cfg_mimo_sar_does_not_depend_on_avg_mode(tmp_path: Path) -> None:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "Config.yaml"
    _write_yaml(offline_cfg, _offline_bp_cfg(avg_mode="none"))
    _write_yaml(fallback_cfg, _fallback_capture_cfg())

    runtime = _read_bp_runtime_cfg(offline_cfg, fallback_cfg)

    assert runtime["mode"] == "mimo_sar"
    assert runtime["motion_mode"] == "static_zero_doppler"
    assert runtime["algorithm"] == "backprojection"
    assert "avg_mode" not in runtime


def test_read_bp_runtime_cfg_mimo_sar_ignores_legacy_virtual_array_flags(tmp_path: Path) -> None:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "Config.yaml"
    _write_yaml(
        offline_cfg,
        _offline_bp_cfg(
            avg_mode="loop",
            virtual_ant_pitch_m=0.123,
            use_virtual_antennas=False,
        ),
    )
    _write_yaml(fallback_cfg, _fallback_capture_cfg())

    runtime = _read_bp_runtime_cfg(offline_cfg, fallback_cfg)

    assert runtime["mode"] == "mimo_sar"
    assert runtime["motion_mode"] == "static_zero_doppler"
    assert runtime["coherent_sum"] is True
    assert "avg_mode" not in runtime
    assert "virtual_ant_pitch_m" not in runtime
    assert "use_virtual_antennas" not in runtime
    assert runtime["tx_offsets_m"] is None
    assert runtime["rx_offsets_m"] is None
    assert isinstance(runtime["warnings"], list)
    assert not any("virtual_ant_pitch_m" in warning for warning in runtime["warnings"])
    assert not any("use_virtual_antennas" in warning for warning in runtime["warnings"])


def test_read_bp_runtime_cfg_mimo_sar_keeps_physical_offset_overrides(tmp_path: Path) -> None:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "Config.yaml"
    tx_offsets = [0.0, 0.01]
    rx_offsets = [0.0, 0.002, 0.004, 0.006]
    _write_yaml(
        offline_cfg,
        _offline_bp_cfg(
            tx_offsets_m=tx_offsets,
            rx_offsets_m=rx_offsets,
        ),
    )
    _write_yaml(fallback_cfg, _fallback_capture_cfg())

    runtime = _read_bp_runtime_cfg(offline_cfg, fallback_cfg)

    np.testing.assert_allclose(runtime["tx_offsets_m"], np.asarray(tx_offsets, dtype=np.float32))
    np.testing.assert_allclose(runtime["rx_offsets_m"], np.asarray(rx_offsets, dtype=np.float32))


def test_read_bp_runtime_cfg_parses_synthetic_range_angle_settings(tmp_path: Path) -> None:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "Config.yaml"
    _write_yaml(
        offline_cfg,
        {
            **_offline_reconstruction_cfg(algorithm="synthetic_range_angle"),
            "offline_sar_range_angle": {
                "use_realtime_filters": True,
                "window_range": "blackman",
                "window_doppler": "hamming",
                "window_angle": "hanning",
                "zero_after_range_fft_bins": 7,
                "background_subtraction": {
                    "enabled": True,
                    "mode": "frozen",
                    "init_frames": 5,
                },
                "angle_processing": {
                    "mode": "mvdr",
                    "mvdr_diagonal_loading": 0.05,
                },
            },
        },
    )
    _write_yaml(
        fallback_cfg,
        {
            **_fallback_capture_cfg(),
            "fft": {"nfft_angle": 128},
            "display": {"projection_mode": "cartesian", "projection_interp": "bilinear"},
            "dsp": {
                "window_range": "hanning",
                "window_doppler": "hanning",
                "window_angle": "hanning",
                "zero_after_range_fft_bins": 0,
                "display_filters": {},
                "angle_processing": {"mode": "bartlett"},
            },
        },
    )

    runtime = _read_bp_runtime_cfg(offline_cfg, fallback_cfg)

    assert runtime["algorithm"] == "synthetic_range_angle"
    range_angle_cfg = runtime["range_angle"]
    assert range_angle_cfg.use_realtime_filters is True
    assert range_angle_cfg.window_range == "blackman"
    assert range_angle_cfg.window_doppler == "hamming"
    assert range_angle_cfg.window_angle == "hanning"
    assert range_angle_cfg.zero_after_range_fft_bins == 7
    assert range_angle_cfg.post_range_fft_filters.background_subtraction.enabled is True
    assert range_angle_cfg.post_range_fft_filters.background_subtraction.mode == "frozen"
    assert range_angle_cfg.angle_processing.mode == "mvdr"
    assert range_angle_cfg.nfft_angle == 128
