"""Verifica reader, configurazione e worker della ricostruzione SAR offline."""

from __future__ import annotations

from pathlib import Path
import json
from multiprocessing import Event, Process, Queue, shared_memory
import struct
import time

import numpy as np
import pytest
import yaml

from offline_processing import (
    OfflineBPRuntime,
    OfflineSARConfig,
    SARReader,
    _backprojection_viewport_max_bin,
    _apply_offline_backprojection_aperture_window,
    _mirror_offline_image_x,
    _offline_reader_worker,
    _read_bp_runtime_cfg,
    _read_offline_sar_range_angle_cfg,
    _read_x_pitch_m,
    _build_bp_frame_position_errors,
    _compact_bp_frame_positions,
    _sample_bp_frame_position_errors,
    _resolve_position_interval,
    _subtract_reference_background,
    _validate_background_reference_layout,
    _validate_offline_memory_budget,
    _viewport_from_cmd_payload,
    estimate_offline_processing_peak_bytes,
    offline_map_bounds_from_yaml_dict,
    offline_mirror_x_from_yaml_dict,
    resolve_offline_synthetic_angle_mode,
)
from realtime_dsp import build_display_viewport


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _fallback_capture_cfg() -> dict:
    return {
        "radar": {
            "c": 3.0e8,
            "fs": 10.0e6,
            "slope": 60.0e12,
            "fc": 77.0e9,
        },
        "capture": {
            "tx": 2,
            "rx": 4,
        },
    }


def test_backprojection_aperture_window_weights_position_antenna_aperture() -> None:
    snapshots = np.ones((2, 3, 4, 5), dtype=np.complex64)

    windowed = _apply_offline_backprojection_aperture_window(
        snapshots,
        window_type="hanning",
        enabled=True,
    )

    expected = np.broadcast_to(
        np.hanning(8).astype(np.float32).reshape(2, 1, 4, 1),
        snapshots.shape,
    )
    np.testing.assert_allclose(windowed, expected)
    np.testing.assert_array_equal(snapshots, np.ones_like(snapshots))


def _offline_bp_cfg(**bp_overrides: object) -> dict:
    bp_cfg = {
        "phase_sign": -1,
        "tx_offsets_lambda": None,
        "rx_offsets_lambda": None,
        "tx_offsets_m": None,
        "rx_offsets_m": None,
    }
    bp_cfg.update(bp_overrides)
    return {"bp": bp_cfg}


def _offline_reconstruction_cfg(*, algorithm: str = "backprojection", **bp_overrides: object) -> dict:
    payload = _offline_bp_cfg(**bp_overrides)
    payload["reconstruction"] = {"algorithm": algorithm}
    return payload


def _write_capture(
    path: Path,
    *,
    position: object,
    frames: int = 2,
    samples: int = 8,
    chirps: int = 4,
    rx: int = 4,
    tx: int = 2,
    raw_i16_value: int | None = None,
    stage_position_mm: float | None = None,
    include_stage: bool = True,
    format_name: str = "rt_capture_v1",
    radar: dict[str, float] | None = None,
) -> None:
    capture = {
        "samples": samples,
        "chirps": chirps,
        "rx": rx,
        "tx": tx,
        "frames_per_position": frames,
    }
    header_payload = {
        "format": format_name,
        "position": position,
        "radar": (
            {"c": 3.0e8, "fs": 10.0e6, "slope": 60.0e12, "fc": 77.0e9}
            if radar is None
            else dict(radar)
        ),
        "capture": capture,
    }
    if include_stage:
        header_payload["stage"] = {
            "position_mm": float(position * 10 if stage_position_mm is None else stage_position_mm),
        }
    header = json.dumps(
        header_payload,
        separators=(",", ":"),
    ).encode("utf-8")
    i16_count = frames * chirps * samples * rx * 2
    if raw_i16_value is None:
        raw_i16 = np.arange(i16_count, dtype=np.int16) + np.int16(position * 7)
    else:
        raw_i16 = np.full(i16_count, int(raw_i16_value), dtype=np.int16)
    raw = raw_i16.tobytes()
    path.write_bytes(b"RTPBIN1\x00" + struct.pack("<I", len(header)) + header + raw)


def _reader_config(run_dir: Path, *, x_end: int = 1) -> OfflineSARConfig:
    return OfflineSARConfig.from_mapping(
        {
            "data": {"input_dir": str(run_dir)},
            "capture": {"frames_per_position": 2},
            "scan": {"x_start": 1, "x_end": x_end, "x_step": 1},
        }
    )


def test_reader_requires_complete_radar_profile_in_every_header(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_missing_radar"
    run_dir.mkdir()
    _write_capture(run_dir / "capture_pos1.bin", position=1, radar={})

    with pytest.raises(ValueError, match=r"header\.radar\.c"):
        SARReader(config=_reader_config(run_dir)).describe_stream()


def test_reader_rejects_inconsistent_radar_profiles_between_positions(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_mixed_profile"
    run_dir.mkdir()
    _write_capture(run_dir / "capture_pos1.bin", position=1)
    _write_capture(
        run_dir / "capture_pos2.bin",
        position=2,
        radar={"c": 3e8, "fs": 15e6, "slope": 60e12, "fc": 77e9},
    )

    with pytest.raises(ValueError, match="profilo radar incoerente"):
        SARReader(config=_reader_config(run_dir, x_end=2)).describe_stream()


def test_background_layout_rejects_different_radar_profile(tmp_path: Path) -> None:
    target_dir = tmp_path / "target_profile"
    reference_dir = tmp_path / "reference_profile"
    target_dir.mkdir()
    reference_dir.mkdir()
    _write_capture(target_dir / "capture_pos1.bin", position=1)
    _write_capture(
        reference_dir / "capture_pos1.bin",
        position=1,
        radar={"c": 3e8, "fs": 12e6, "slope": 60e12, "fc": 77e9},
    )
    target = SARReader(config=_reader_config(target_dir)).describe_stream()
    reference = SARReader(config=_reader_config(reference_dir)).describe_stream()

    with pytest.raises(ValueError, match="profilo radar diverso"):
        _validate_background_reference_layout(target, reference)


def test_source_resolution_ignores_unrelated_bin_files(tmp_path: Path) -> None:
    (tmp_path / "diagnostic.bin").write_bytes(b"not a capture")
    # This name matches the broad capture_pos*.bin glob but not the exact
    # capture_pos<integer>.bin convention.
    (tmp_path / "capture_position.bin").write_bytes(b"not a capture")
    run_dir = tmp_path / "run_20260101_010101"
    run_dir.mkdir()
    _write_capture(run_dir / "capture_pos1.bin", position=1)

    layout = SARReader(config=_reader_config(tmp_path)).describe_stream()

    assert layout.source_dir == run_dir


def test_reader_rejects_fractional_header_position(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_fractional_position"
    run_dir.mkdir()
    _write_capture(run_dir / "capture_pos1.bin", position=1.5)

    with pytest.raises(ValueError, match="header 'position' non valido"):
        SARReader(config=_reader_config(run_dir)).describe_stream()


@pytest.mark.parametrize("raw", ["false", "off", "0"])
def test_offline_filter_boolean_strings_are_parsed_strictly(raw: str) -> None:
    cfg = _read_offline_sar_range_angle_cfg(
        {"offline_sar_range_angle": {"use_realtime_filters": raw}},
        {**_fallback_capture_cfg(), "fft": {"nfft_range": 64, "nfft_angle": 32}},
    )
    assert cfg.use_realtime_filters is False


def test_offline_filter_boolean_rejects_non_boolean_integer() -> None:
    with pytest.raises(ValueError, match="use_realtime_filters"):
        _read_offline_sar_range_angle_cfg(
            {"offline_sar_range_angle": {"use_realtime_filters": 2}},
            {**_fallback_capture_cfg(), "fft": {"nfft_range": 64, "nfft_angle": 32}},
        )


@pytest.mark.parametrize("raw", [0, -4, "invalid", 4.5])
def test_offline_fft_size_rejects_invalid_values(raw: object) -> None:
    with pytest.raises(ValueError, match="nfft_range"):
        _read_offline_sar_range_angle_cfg(
            {"offline_sar_range_angle": {"nfft_range": raw}},
            {**_fallback_capture_cfg(), "fft": {"nfft_angle": 32}},
        )


@pytest.mark.parametrize("raw", [-1, 1.5, True, "invalid"])
def test_zeroed_range_bin_count_is_strict(raw: object) -> None:
    with pytest.raises(ValueError, match="zero_after_range_fft_bins"):
        _read_offline_sar_range_angle_cfg(
            {"offline_sar_range_angle": {"zero_after_range_fft_bins": raw}},
            {**_fallback_capture_cfg(), "fft": {"nfft_range": 64, "nfft_angle": 32}},
        )


def test_x_pitch_rejects_non_finite_values(tmp_path: Path) -> None:
    cfg_path = tmp_path / "offline.yaml"
    _write_yaml(cfg_path, {"scan": {"x_pitch_m": float("nan")}})

    with pytest.raises(ValueError, match="finito"):
        _read_x_pitch_m(cfg_path)


def test_offline_mirror_x_is_strict_and_defaults_to_disabled() -> None:
    assert offline_mirror_x_from_yaml_dict({}) is False
    assert offline_mirror_x_from_yaml_dict({"reconstruction": {"mirror_x": True}}) is True
    with pytest.raises(ValueError, match="mirror_x"):
        offline_mirror_x_from_yaml_dict({"reconstruction": {"mirror_x": 2}})


def test_offline_mirror_x_reverses_only_the_output_columns() -> None:
    image = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    unchanged = _mirror_offline_image_x(image, mirror_x=False)
    mirrored = _mirror_offline_image_x(image, mirror_x=True)

    np.testing.assert_array_equal(unchanged, image)
    np.testing.assert_array_equal(mirrored, image[:, ::-1])
    np.testing.assert_array_equal(image, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_sparse_position_interval_without_capture_is_rejected() -> None:
    positions = np.asarray([2, 4, 6], dtype=np.int32)

    assert _resolve_position_interval(positions, 3, 3) is None
    assert _resolve_position_interval(positions, 3, 4) == (3, 4)


def test_read_bp_runtime_cfg_rejects_invalid_reconstruction_algorithm(tmp_path: Path) -> None:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "realtime_config.yaml"
    _write_yaml(offline_cfg, _offline_reconstruction_cfg(algorithm="wrong_mode"))
    _write_yaml(fallback_cfg, _fallback_capture_cfg())

    with pytest.raises(ValueError, match="synthetic_range_angle"):
        _read_bp_runtime_cfg(offline_cfg, fallback_cfg)


def test_read_bp_runtime_cfg_parses_memory_cap(tmp_path: Path) -> None:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "realtime_config.yaml"
    _write_yaml(
        offline_cfg,
        {"data": {"max_memory_gib": 0.5}, **_offline_reconstruction_cfg()},
    )
    _write_yaml(fallback_cfg, _fallback_capture_cfg())

    runtime = _read_bp_runtime_cfg(offline_cfg, fallback_cfg)

    assert runtime["max_memory_bytes"] == 512 * 1024 * 1024


@pytest.mark.parametrize("raw", [0, -1, float("nan"), "invalid"])
def test_read_bp_runtime_cfg_rejects_invalid_memory_cap(
    tmp_path: Path,
    raw: object,
) -> None:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "realtime_config.yaml"
    _write_yaml(
        offline_cfg,
        {"data": {"max_memory_gib": raw}, **_offline_reconstruction_cfg()},
    )
    _write_yaml(fallback_cfg, _fallback_capture_cfg())

    with pytest.raises(ValueError, match="max_memory_gib"):
        _read_bp_runtime_cfg(offline_cfg, fallback_cfg)


def test_memory_estimate_and_configured_cap_use_cropped_shared_cube() -> None:
    estimate = estimate_offline_processing_peak_bytes(
        n_positions=150,
        n_frames=2,
        n_antennas=8,
        input_samples=256,
        chirps=256,
        rx=4,
        nfft_range=16384,
        retained_range_bins=2048,
        nfft_angle=4096,
        algorithm="backprojection",
        angle_mode="fft",
    )

    assert estimate["shared_bytes"] == 150 * 2 * 8 * 2048 * 8
    assert estimate["peak_bytes"] >= estimate["shared_bytes"]
    with pytest.raises(MemoryError, match="data.max_memory_gib"):
        _validate_offline_memory_budget(
            estimate,
            configured_limit_bytes=int(estimate["peak_bytes"]) - 1,
        )


def test_synthetic_memory_mode_uses_measured_nonuniform_geometry() -> None:
    effective = resolve_offline_synthetic_angle_mode(
        stage_positions_m=np.asarray([0.0, 0.01, 0.021], dtype=np.float32),
        x_tx_ant_m=np.asarray([0.0], dtype=np.float32),
        x_rx_ant_m=np.asarray([0.0], dtype=np.float32),
        c_m_s=3.0e8,
        fc_hz=77.0e9,
        requested_mode="fft",
    )

    assert effective == "bartlett"


def test_synthetic_memory_estimate_accounts_for_steering_and_reference_batch() -> None:
    common = dict(
        n_positions=20,
        n_frames=2,
        n_antennas=8,
        input_samples=256,
        chirps=256,
        rx=4,
        nfft_range=4096,
        retained_range_bins=512,
        nfft_angle=2048,
        algorithm="synthetic_range_angle",
        image_h=128,
        image_w=128,
    )
    fft_estimate = estimate_offline_processing_peak_bytes(
        **common,
        angle_mode="fft",
    )
    bartlett_estimate = estimate_offline_processing_peak_bytes(
        **common,
        angle_mode="bartlett",
        max_position_frames=32,
    )
    full_loop_estimate = estimate_offline_processing_peak_bytes(
        **common,
        angle_mode="fft",
        full_loop_range_fft=True,
    )

    assert bartlett_estimate["shared_bytes"] == fft_estimate["shared_bytes"]
    assert bartlett_estimate["reader_peak_bytes"] > fft_estimate["reader_peak_bytes"]
    assert bartlett_estimate["dsp_peak_bytes"] > fft_estimate["dsp_peak_bytes"]
    assert full_loop_estimate["reader_peak_bytes"] > fft_estimate["reader_peak_bytes"]


def test_read_bp_runtime_cfg_mimo_sar_keeps_physical_offset_overrides(tmp_path: Path) -> None:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "realtime_config.yaml"
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


def test_read_bp_runtime_cfg_reads_signed_range_offset_for_backprojection(tmp_path: Path) -> None:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "realtime_config.yaml"
    _write_yaml(offline_cfg, _offline_reconstruction_cfg(range_offset_m=-0.0125))
    _write_yaml(fallback_cfg, _fallback_capture_cfg())

    runtime = _read_bp_runtime_cfg(offline_cfg, fallback_cfg)

    assert runtime["range_offset_m"] == pytest.approx(-0.0125)


def test_read_bp_runtime_cfg_reads_seeded_uniform_frame_position_error(tmp_path: Path) -> None:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "realtime_config.yaml"
    _write_yaml(
        offline_cfg,
        _offline_reconstruction_cfg(
            frame_position_error={
                "enabled": True,
                "max_abs_mm": 1.0,
                "seed": 20260810,
            }
        ),
    )
    _write_yaml(fallback_cfg, _fallback_capture_cfg())

    runtime = _read_bp_runtime_cfg(offline_cfg, fallback_cfg)
    error_cfg = runtime["frame_position_error"]
    errors = _sample_bp_frame_position_errors(
        error_cfg,
        n_positions=3,
        n_frames=2,
    )
    expected = np.random.default_rng(20260810).uniform(
        -0.001,
        0.001,
        size=(3, 2),
    ).astype(np.float32)

    assert error_cfg.enabled is True
    assert error_cfg.max_abs_m == pytest.approx(0.001)
    assert error_cfg.seed == 20260810
    np.testing.assert_array_equal(errors, expected)
    assert np.all(errors >= -0.001)
    assert np.all(errors <= 0.001)


def test_static_step_error_changes_every_reconstructed_pitch_by_fixed_amount(
    tmp_path: Path,
) -> None:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "realtime_config.yaml"
    _write_yaml(
        offline_cfg,
        _offline_reconstruction_cfg(
            frame_position_error={
                "enabled": True,
                "mode": "static_step",
                "static_step_error_mm": 2.0,
            }
        ),
    )
    _write_yaml(fallback_cfg, _fallback_capture_cfg())

    error_cfg = _read_bp_runtime_cfg(offline_cfg, fallback_cfg)["frame_position_error"]
    stage_positions_m = np.asarray([0.0, 0.008, 0.016, 0.024], dtype=np.float32)
    errors_m = _build_bp_frame_position_errors(
        error_cfg,
        stage_positions_m=stage_positions_m,
        n_frames=2,
    )
    bp_positions_m = stage_positions_m[:, None] + errors_m

    assert error_cfg.mode == "static_step"
    assert error_cfg.static_step_error_m == pytest.approx(0.002)
    np.testing.assert_allclose(errors_m[:, 0], [-0.003, -0.001, 0.001, 0.003])
    np.testing.assert_allclose(errors_m[:, 0], errors_m[:, 1])
    np.testing.assert_allclose(np.diff(bp_positions_m[:, 0]), 0.010)


def test_static_frame_positions_are_compacted_but_random_positions_are_not() -> None:
    static_positions = np.asarray(
        [[-0.003, -0.003], [0.007, 0.007], [0.017, 0.017]],
        dtype=np.float32,
    )
    dynamic_positions = static_positions.copy()
    dynamic_positions[1, 1] += np.float32(0.0005)

    compact = _compact_bp_frame_positions(static_positions)
    dynamic = _compact_bp_frame_positions(dynamic_positions)

    assert compact.shape == (3,)
    np.testing.assert_array_equal(compact, static_positions[:, 0])
    assert dynamic.shape == (3, 2)
    assert dynamic is dynamic_positions


@pytest.mark.parametrize("max_abs_mm", [-0.001, float("nan"), float("inf")])
def test_read_bp_runtime_cfg_rejects_invalid_frame_position_error_amplitude(
    tmp_path: Path,
    max_abs_mm: float,
) -> None:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "realtime_config.yaml"
    _write_yaml(
        offline_cfg,
        _offline_reconstruction_cfg(
            frame_position_error={"enabled": True, "max_abs_mm": max_abs_mm}
        ),
    )
    _write_yaml(fallback_cfg, _fallback_capture_cfg())

    with pytest.raises(ValueError, match="frame_position_error.max_abs_mm"):
        _read_bp_runtime_cfg(offline_cfg, fallback_cfg)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "not-a-number"])
def test_read_bp_runtime_cfg_rejects_non_finite_range_offset(tmp_path: Path, value: object) -> None:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "realtime_config.yaml"
    _write_yaml(offline_cfg, _offline_reconstruction_cfg(range_offset_m=value))
    _write_yaml(fallback_cfg, _fallback_capture_cfg())

    with pytest.raises(ValueError, match="range_offset_m"):
        _read_bp_runtime_cfg(offline_cfg, fallback_cfg)


@pytest.mark.parametrize(("configured", "expected"), [("off", 0), ("+", 1), ("-", -1)])
def test_read_bp_runtime_cfg_parses_residual_video_phase(
    tmp_path: Path,
    configured: str,
    expected: int,
) -> None:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "realtime_config.yaml"
    _write_yaml(offline_cfg, _offline_reconstruction_cfg(residual_video_phase=configured))
    _write_yaml(fallback_cfg, _fallback_capture_cfg())

    runtime = _read_bp_runtime_cfg(offline_cfg, fallback_cfg)

    assert runtime["residual_video_phase"] == expected


def test_offline_map_bounds_are_loaded_from_reconstruction_config(tmp_path: Path) -> None:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "realtime_config.yaml"
    _write_yaml(
        offline_cfg,
        {
            "reconstruction": {
                "algorithm": "backprojection",
                "map_bounds": {
                    "x_min_m": -2.0,
                    "x_max_m": 4.0,
                    "y_min_m": 1.5,
                    "y_max_m": 8.0,
                },
            }
        },
    )
    _write_yaml(fallback_cfg, _fallback_capture_cfg())

    runtime = _read_bp_runtime_cfg(offline_cfg, fallback_cfg)

    assert runtime["map_bounds"].x_min_m == pytest.approx(-2.0)
    assert runtime["map_bounds"].x_max_m == pytest.approx(4.0)
    assert runtime["map_bounds"].y_min_m == pytest.approx(1.5)
    assert runtime["map_bounds"].y_max_m == pytest.approx(8.0)


@pytest.mark.parametrize(
    "bounds, error",
    [
        ({"x_min_m": 1.0, "x_max_m": 1.0, "y_min_m": 0.0, "y_max_m": 2.0}, "x_max_m"),
        ({"x_min_m": -1.0, "x_max_m": 1.0, "y_min_m": -0.1, "y_max_m": 2.0}, "y_min_m"),
        ({"x_min_m": -1.0, "x_max_m": 1.0, "y_min_m": 2.0, "y_max_m": 2.0}, "y_max_m"),
    ],
)
def test_offline_map_bounds_validate_physical_rectangle(bounds: dict, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        offline_map_bounds_from_yaml_dict(
            {"reconstruction": {"map_bounds": bounds}},
            _fallback_capture_cfg(),
        )


def test_read_bp_runtime_cfg_parses_empty_scene_reference_background(tmp_path: Path) -> None:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "realtime_config.yaml"
    _write_yaml(
        offline_cfg,
        {
            **_offline_reconstruction_cfg(algorithm="backprojection"),
            "offline_background": {
                "enabled": True,
                "reference_dir": "empty_scene",
                "scale": 0.75,
            },
        },
    )
    _write_yaml(fallback_cfg, _fallback_capture_cfg())

    runtime = _read_bp_runtime_cfg(offline_cfg, fallback_cfg)

    background = runtime["background_reference"]
    assert background.enabled is True
    assert background.reference_dir == (tmp_path / "empty_scene").resolve()
    assert background.scale == pytest.approx(0.75)


@pytest.mark.parametrize(
    "background, error",
    [
        ({"enabled": True}, "reference_dir"),
        ({"enabled": True, "reference_dir": "empty", "scale": -0.1}, "scale"),
    ],
)
def test_read_bp_runtime_cfg_rejects_invalid_empty_scene_reference_background(
    tmp_path: Path,
    background: dict,
    error: str,
) -> None:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "realtime_config.yaml"
    _write_yaml(
        offline_cfg,
        {
            **_offline_reconstruction_cfg(algorithm="backprojection"),
            "offline_background": background,
        },
    )
    _write_yaml(fallback_cfg, _fallback_capture_cfg())

    with pytest.raises(ValueError, match=error):
        _read_bp_runtime_cfg(offline_cfg, fallback_cfg)


def test_reference_background_is_supported_for_synthetic_range_angle(tmp_path: Path) -> None:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "realtime_config.yaml"
    _write_yaml(
        offline_cfg,
        {
            **_offline_reconstruction_cfg(algorithm="synthetic_range_angle"),
            "offline_background": {"enabled": True, "reference_dir": "empty"},
        },
    )
    _write_yaml(fallback_cfg, _fallback_capture_cfg())

    runtime = _read_bp_runtime_cfg(offline_cfg, fallback_cfg)

    assert runtime["algorithm"] == "synthetic_range_angle"
    assert runtime["background_reference"].enabled is True


def test_subtract_reference_background_is_complex_and_does_not_mutate_input() -> None:
    target = np.array(
        [
            [[3.0 + 2.0j, 5.0 - 1.0j], [2.0 + 4.0j, -1.0 + 3.0j]],
            [[1.0 - 2.0j, 4.0 + 3.0j], [7.0 + 0.0j, 2.0 - 5.0j]],
        ],
        dtype=np.complex64,
    )
    reference = np.array(
        [[1.0 + 1.0j, 2.0 - 2.0j], [3.0 + 0.0j, -2.0 + 1.0j]],
        dtype=np.complex64,
    )
    original = target.copy()

    residual = _subtract_reference_background(target, reference, scale=0.5)

    expected = target - np.complex64(0.5) * reference[None, :, :]
    assert residual.shape == target.shape
    np.testing.assert_allclose(residual, expected)
    np.testing.assert_array_equal(target, original)
    assert residual.dtype == np.complex64


def test_subtract_reference_background_scale_zero_preserves_independent_frames() -> None:
    scene = np.array(
        [
            [[1.0 + 2.0j, 3.0 - 4.0j]],
            [[5.0 - 1.0j, -2.0 + 6.0j]],
        ],
        dtype=np.complex64,
    )
    reference_mean = np.zeros_like(scene[0])

    residual = _subtract_reference_background(scene, reference_mean, scale=0.0)

    assert residual.shape == scene.shape
    np.testing.assert_array_equal(residual, scene)


def test_subtract_reference_background_validates_antenna_and_range_shape() -> None:
    target = np.zeros((2, 8, 16), dtype=np.complex64)
    reference = np.zeros((7, 16), dtype=np.complex64)

    with pytest.raises(ValueError, match="non coerente"):
        _subtract_reference_background(target, reference, scale=1.0)


def test_read_bp_runtime_cfg_parses_synthetic_range_angle_settings(tmp_path: Path) -> None:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "realtime_config.yaml"
    _write_yaml(
        offline_cfg,
        {
            **_offline_reconstruction_cfg(algorithm="synthetic_range_angle"),
            "offline_sar_range_angle": {
                "use_realtime_filters": True,
                "window_range": "blackman",
                "window_doppler": "hamming",
                "window_angle": "hanning",
                "nfft_range": 512,
                "nfft_angle": 64,
                "zero_after_range_fft_bins": 7,
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
    assert range_angle_cfg.angle_processing.mode == "mvdr"
    assert range_angle_cfg.nfft_range == 512
    assert range_angle_cfg.nfft_angle == 64


def test_offline_synthetic_range_angle_does_not_inherit_realtime_slow_time_filter(tmp_path: Path) -> None:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "realtime_config.yaml"
    _write_yaml(
        offline_cfg,
        {
            **_offline_reconstruction_cfg(algorithm="synthetic_range_angle"),
        },
    )
    _write_yaml(
        fallback_cfg,
        {
            **_fallback_capture_cfg(),
            "dsp": {
                "display_filters": {
                    "slow_time": {"enabled": True, "mode": "highpass"},
                }
            },
        },
    )

    runtime = _read_bp_runtime_cfg(offline_cfg, fallback_cfg)

    slow_time = runtime["range_angle"].post_range_fft_filters.slow_time
    assert slow_time.enabled is False
    assert slow_time.mode == "none"


def test_read_bp_runtime_cfg_inherits_fft_sizes_when_not_overridden(tmp_path: Path) -> None:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "realtime_config.yaml"
    _write_yaml(offline_cfg, _offline_reconstruction_cfg(algorithm="synthetic_range_angle"))
    _write_yaml(
        fallback_cfg,
        {
            **_fallback_capture_cfg(),
            "fft": {"nfft_range": 512, "nfft_angle": 128},
            "dsp": {"display_filters": {}, "angle_processing": {"mode": "bartlett"}},
        },
    )

    runtime = _read_bp_runtime_cfg(offline_cfg, fallback_cfg)

    assert runtime["range_angle"].nfft_range == 512
    assert runtime["range_angle"].nfft_angle == 128


def test_offline_viewport_keeps_exact_requested_roi_for_backprojection_grid() -> None:
    home = build_display_viewport(
        x_min_m=-5.0,
        x_max_m=5.0,
        y_min_m=0.0,
        y_max_m=10.0,
        dr_m=0.05,
    )

    viewport = _viewport_from_cmd_payload(
        {
            "x_min_m": -2.0,
            "x_max_m": 2.0,
            "y_min_m": 2.0,
            "y_max_m": 4.0,
            "seq": 7,
        },
        home_viewport=home,
        output_width=256,
        output_height=256,
        dr_m=0.05,
    )

    assert viewport is not None
    assert viewport.x_min_m == -2.0
    assert viewport.x_max_m == 2.0
    assert viewport.y_min_m == 2.0
    assert viewport.y_max_m == 4.0
    assert viewport.seq == 7


def test_backprojection_roi_reads_only_bins_reachable_by_selected_geometry() -> None:
    viewport = build_display_viewport(
        x_min_m=-2.0,
        x_max_m=2.0,
        y_min_m=0.0,
        y_max_m=4.0,
        dr_m=1.0,
    )

    max_bin = _backprojection_viewport_max_bin(
        viewport,
        x_pos_m=np.asarray([3.0], dtype=np.float32),
        x_tx_ant_m=np.asarray([0.0], dtype=np.float32),
        x_rx_ant_m=np.asarray([0.0], dtype=np.float32),
        dr_m=1.0,
        available_bins=128,
    )

    # Farthest point is x=-2 m from a sensor at x=3 m and y=4 m:
    # ceil(hypot(5, 4)) plus two cubic-interpolation guard bins.
    assert max_bin == 9

    offset_max_bin = _backprojection_viewport_max_bin(
        viewport,
        x_pos_m=np.asarray([3.0], dtype=np.float32),
        x_tx_ant_m=np.asarray([0.0], dtype=np.float32),
        x_rx_ant_m=np.asarray([0.0], dtype=np.float32),
        dr_m=1.0,
        available_bins=128,
        range_offset_m=0.75,
    )

    # A positive range calibration samples farther FFT bins, so the retained
    # spectrum must cover it as well.
    assert offset_max_bin == 10


def test_stream_reader_validates_without_loading_full_cube(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    _write_capture(run_dir / "capture_pos1.bin", position=1)
    _write_capture(run_dir / "capture_pos2.bin", position=2)
    offline_cfg = tmp_path / "offline_config.yaml"
    _write_yaml(
        offline_cfg,
        {
            "data": {"input_dir": str(run_dir)},
            "capture": {"frames_per_position": 2},
            "scan": {"x_start": 1, "x_end": 2, "x_step": 1, "x_pitch_m": 0.01},
        },
    )
    reader = SARReader(offline_cfg)
    layout = reader.describe_stream()
    streamed = list(reader.iter_iq_positions(layout))

    assert layout.positions.tolist() == [1, 2]
    assert layout.n_frames_per_position == 2
    assert [pos for pos, _iq in streamed] == [1, 2]
    assert streamed[0][1].shape == (2, 2, 8, 8)


def test_stream_reader_uses_measured_stage_positions_for_linear_geometry(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_stage_positions"
    run_dir.mkdir()
    _write_capture(
        run_dir / "capture_pos1.bin",
        position=1,
        stage_position_mm=12.5,
    )
    _write_capture(
        run_dir / "capture_pos2.bin",
        position=2,
        stage_position_mm=28.0,
    )
    offline_cfg = tmp_path / "offline_config.yaml"
    _write_yaml(
        offline_cfg,
        {
            "data": {"input_dir": str(run_dir)},
            # Deliberately unlike the measured 15.5 mm separation.
            "scan": {"x_start": 1, "x_end": 2, "x_step": 1, "x_pitch_m": 0.01},
        },
    )

    layout = SARReader(offline_cfg).describe_stream()

    assert layout.stage_positions_m is not None
    np.testing.assert_allclose(layout.stage_positions_m, [0.0125, 0.0280])


def test_stream_reader_rejects_linear_capture_without_measured_stage_position(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_without_stage"
    run_dir.mkdir()
    _write_capture(run_dir / "capture_pos1.bin", position=1, include_stage=False)
    offline_cfg = tmp_path / "offline_config.yaml"
    _write_yaml(
        offline_cfg,
        {
            "data": {"input_dir": str(run_dir)},
            "scan": {"x_start": 1, "x_end": 1, "x_step": 1, "x_pitch_m": 0.01},
        },
    )

    with pytest.raises(ValueError, match="stage"):
        SARReader(offline_cfg).describe_stream()


def test_stream_reader_uses_requested_prefix_of_available_frames(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_capture(run_dir / "capture_pos1.bin", position=1, frames=3)
    _write_capture(run_dir / "capture_pos2.bin", position=2, frames=3)
    offline_cfg = tmp_path / "offline_config.yaml"
    _write_yaml(
        offline_cfg,
        {
            "data": {"input_dir": str(run_dir)},
            "capture": {"frames_per_position": 2},
            "scan": {"x_start": 1, "x_end": 2, "x_step": 1, "x_pitch_m": 0.01},
        },
    )

    reader = SARReader(offline_cfg)
    layout = reader.describe_stream()
    streamed = list(reader.iter_iq_positions(layout))

    assert layout.available_frames_per_position == 3
    assert layout.n_frames_per_position == 2
    assert [iq.shape[0] for _position, iq in streamed] == [2, 2]


def test_stream_reader_allows_regular_position_subsampling(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for position in range(1, 7):
        _write_capture(run_dir / f"capture_pos{position}.bin", position=position, frames=2)
    offline_cfg = tmp_path / "offline_config.yaml"
    _write_yaml(
        offline_cfg,
        {
            "data": {"input_dir": str(run_dir)},
            "capture": {"frames_per_position": 1},
            "scan": {"x_start": 2, "x_end": 6, "x_step": 2, "x_pitch_m": 0.01},
        },
    )

    layout = SARReader(offline_cfg).describe_stream()

    assert layout.positions.tolist() == [2, 4, 6]
    assert [path.name for path in layout.files] == [
        "capture_pos2.bin",
        "capture_pos4.bin",
        "capture_pos6.bin",
    ]


def test_stream_reader_rejects_more_frames_than_available(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_capture(run_dir / "capture_pos1.bin", position=1, frames=2)
    offline_cfg = tmp_path / "offline_config.yaml"
    _write_yaml(
        offline_cfg,
        {
            "data": {"input_dir": str(run_dir)},
            "capture": {"frames_per_position": 3},
            "scan": {"x_start": 1, "x_end": 1, "x_step": 1, "x_pitch_m": 0.01},
        },
    )

    with pytest.raises(ValueError, match="frames_per_position richiesto=3"):
        SARReader(offline_cfg).describe_stream()


def test_stream_reader_rejects_v2_capture_header(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_v2"
    run_dir.mkdir()
    _write_capture(
        run_dir / "capture_pos1.bin",
        position=1,
        format_name="rt_capture_v2",
    )
    offline_cfg = tmp_path / "offline_config.yaml"
    _write_yaml(
        offline_cfg,
        {
            "data": {"input_dir": str(run_dir)},
            "scan": {"x_start": 1, "x_end": 1, "x_step": 1, "x_pitch_m": 0.01},
        },
    )

    with pytest.raises(ValueError, match="supporta solo 'rt_capture_v1'"):
        SARReader(offline_cfg).describe_stream()


@pytest.mark.parametrize("nfft_range", [48, 64, 96])
def test_offline_reader_worker_publishes_compact_zero_doppler_cube(
    tmp_path: Path,
    nfft_range: int,
) -> None:
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    _write_capture(run_dir / "capture_pos1.bin", position=1)
    _write_capture(run_dir / "capture_pos2.bin", position=2)
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "realtime_config.yaml"
    _write_yaml(
        offline_cfg,
        {
            "data": {"input_dir": str(run_dir)},
            "capture": {"frames_per_position": 2},
            "scan": {"x_start": 1, "x_end": 2, "x_step": 1, "x_pitch_m": 0.01},
            "reconstruction": {
                "algorithm": "backprojection",
                "map_bounds": {
                    "x_min_m": -0.1,
                    "x_max_m": 0.1,
                    "y_min_m": 0.0,
                    "y_max_m": 0.5,
                },
            },
            "bp": {},
            "offline_sar_range_angle": {
                "use_realtime_filters": True,
                "window_range": "hanning",
                "window_doppler": "hanning",
                "window_angle": "hanning",
                "nfft_range": nfft_range,
                "nfft_angle": 32,
            },
        },
    )
    fallback_payload = _fallback_capture_cfg()
    fallback_payload["capture"].update({"samples": 8, "chirps": 4})
    fallback_payload["fft"] = {"workers": 1}
    _write_yaml(fallback_cfg, fallback_payload)

    data_q: Queue = Queue()
    status_q: Queue = Queue()
    stop_evt = Event()
    worker = Process(
        target=_offline_reader_worker,
        args=(str(offline_cfg), str(fallback_cfg), nfft_range, data_q, status_q, stop_evt),
    )
    worker.start()
    msg = data_q.get(timeout=2.0)
    assert msg.get("type") == "data", msg
    shm = shared_memory.SharedMemory(name=str(msg["range_fft_shm_name"]))
    try:
        shape = tuple(int(v) for v in msg["range_fft_shape"])
        snapshots = np.ndarray(shape, dtype=np.complex64, buffer=shm.buf)
        assert shape == (2, 2, 8, int(msg["range_fft_bins_stored"]))
        assert 1 <= shape[-1] < nfft_range
        assert np.all(np.isfinite(snapshots))
        np.testing.assert_allclose(msg["bp_x_pos_m"], [0.01, 0.02])
    finally:
        shm.close()
        stop_evt.set()
        worker.join(timeout=2.0)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=1.0)
        try:
            shm.unlink()
        except FileNotFoundError:
            pass
        data_q.close()
        status_q.close()


def test_offline_reader_worker_injects_seeded_position_error_per_frame(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_position_error"
    run_dir.mkdir()
    _write_capture(run_dir / "capture_pos1.bin", position=1)
    _write_capture(run_dir / "capture_pos2.bin", position=2)
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "realtime_config.yaml"
    _write_yaml(
        offline_cfg,
        {
            "data": {"input_dir": str(run_dir)},
            "capture": {"frames_per_position": 2},
            "scan": {"x_start": 1, "x_end": 2, "x_step": 1},
            "reconstruction": {
                "algorithm": "backprojection",
                "map_bounds": {
                    "x_min_m": -0.1,
                    "x_max_m": 0.1,
                    "y_min_m": 0.0,
                    "y_max_m": 0.5,
                },
            },
            "bp": {
                "frame_position_error": {
                    "enabled": True,
                    "max_abs_mm": 1.0,
                    "seed": 20260810,
                }
            },
            "offline_sar_range_angle": {"nfft_range": 48, "nfft_angle": 32},
        },
    )
    fallback_payload = _fallback_capture_cfg()
    fallback_payload["capture"].update({"samples": 8, "chirps": 4})
    fallback_payload["fft"] = {"workers": 1}
    _write_yaml(fallback_cfg, fallback_payload)

    data_q: Queue = Queue()
    status_q: Queue = Queue()
    stop_evt = Event()
    worker = Process(
        target=_offline_reader_worker,
        args=(str(offline_cfg), str(fallback_cfg), 48, data_q, status_q, stop_evt),
    )
    worker.start()
    msg = data_q.get(timeout=2.0)
    assert msg.get("type") == "data", msg
    shm = shared_memory.SharedMemory(name=str(msg["range_fft_shm_name"]))
    try:
        expected_errors = np.random.default_rng(20260810).uniform(
            -0.001,
            0.001,
            size=(2, 2),
        ).astype(np.float32)
        expected_x = np.asarray([0.01, 0.02], dtype=np.float32)[:, None] + expected_errors
        assert msg["bp_frame_position_error_enabled"] is True
        assert msg["bp_frame_position_error_seed"] == 20260810
        np.testing.assert_array_equal(msg["bp_x_pos_m"], expected_x)
        assert float(msg["bp_frame_position_error_min_m"]) >= -0.001
        assert float(msg["bp_frame_position_error_max_m"]) <= 0.001
    finally:
        shm.close()
        stop_evt.set()
        worker.join(timeout=2.0)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=1.0)
        try:
            shm.unlink()
        except FileNotFoundError:
            pass
        data_q.close()
        status_q.close()


def test_offline_reader_rejects_insufficient_bp_oversampling_before_allocation(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_low_oversampling"
    run_dir.mkdir()
    _write_capture(run_dir / "capture_pos1.bin", position=1, samples=8)
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "realtime_config.yaml"
    _write_yaml(
        offline_cfg,
        {
            "data": {"input_dir": str(run_dir)},
            "capture": {"frames_per_position": 2},
            "scan": {"x_start": 1, "x_end": 1, "x_step": 1},
            "reconstruction": {"algorithm": "backprojection"},
            "offline_sar_range_angle": {"nfft_range": 40, "nfft_angle": 16},
        },
    )
    fallback = _fallback_capture_cfg()
    fallback["capture"].update({"samples": 8, "chirps": 4})
    _write_yaml(fallback_cfg, fallback)
    data_q: Queue = Queue()
    status_q: Queue = Queue()
    stop_evt = Event()
    worker = Process(
        target=_offline_reader_worker,
        args=(str(offline_cfg), str(fallback_cfg), 40, data_q, status_q, stop_evt),
    )
    worker.start()
    try:
        msg = data_q.get(timeout=3.0)
        assert msg["type"] == "error"
        assert "oversampling" in str(msg["error"])
        assert "range_fft_shm_name" not in msg
    finally:
        stop_evt.set()
        worker.join(timeout=2.0)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=1.0)
        data_q.close()
        status_q.close()


def test_offline_reader_worker_rejects_capture_profile_different_from_fallback(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_profile_mismatch"
    run_dir.mkdir()
    _write_capture(run_dir / "capture_pos1.bin", position=1)
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "realtime_config.yaml"
    _write_yaml(
        offline_cfg,
        {
            "data": {"input_dir": str(run_dir)},
            "capture": {"frames_per_position": 2},
            "scan": {"x_start": 1, "x_end": 1, "x_step": 1, "x_pitch_m": 0.01},
            "reconstruction": {"algorithm": "backprojection"},
            "offline_sar_range_angle": {"nfft_range": 32, "nfft_angle": 16},
        },
    )
    fallback = _fallback_capture_cfg()
    fallback["radar"]["fs"] = 15e6
    fallback["capture"].update({"samples": 8, "chirps": 4})
    fallback["fft"] = {"workers": 1}
    _write_yaml(fallback_cfg, fallback)
    data_q: Queue = Queue()
    status_q: Queue = Queue()
    stop_evt = Event()
    worker = Process(
        target=_offline_reader_worker,
        args=(str(offline_cfg), str(fallback_cfg), 32, data_q, status_q, stop_evt),
    )
    worker.start()
    try:
        msg = data_q.get(timeout=3.0)
        assert msg["type"] == "error"
        assert "fs_hz" in str(msg["error"])
    finally:
        stop_evt.set()
        worker.join(timeout=2.0)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=1.0)
        data_q.close()
        status_q.close()


def test_offline_reader_subtracts_empty_scene_reference_with_different_frame_count(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "target"
    reference_dir = tmp_path / "empty_scene"
    target_dir.mkdir()
    reference_dir.mkdir()
    for position in (1, 2):
        # Deliberately use a different number of reference frames.  A constant
        # IQ payload makes both per-run means identical, so the residual must
        # be zero after complex subtraction.
        _write_capture(
            target_dir / f"capture_pos{position}.bin",
            position=position,
            frames=2,
            raw_i16_value=17,
        )
        _write_capture(
            reference_dir / f"capture_pos{position}.bin",
            position=position,
            frames=3,
            raw_i16_value=17,
        )

    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "realtime_config.yaml"
    _write_yaml(
        offline_cfg,
        {
            "data": {"input_dir": str(target_dir)},
            "capture": {"frames_per_position": 2},
            "scan": {"x_start": 1, "x_end": 2, "x_step": 1, "x_pitch_m": 0.01},
            "reconstruction": {"algorithm": "backprojection"},
            "bp": {},
            "offline_background": {
                "enabled": True,
                "reference_dir": str(reference_dir),
                "scale": 1.0,
            },
            "offline_sar_range_angle": {
                "use_realtime_filters": True,
                "window_range": "hanning",
                "window_doppler": "rectangular",
                "window_angle": "hanning",
                "nfft_range": 48,
            },
        },
    )
    fallback_payload = _fallback_capture_cfg()
    fallback_payload["capture"].update({"samples": 8, "chirps": 4})
    fallback_payload["fft"] = {"workers": 1}
    _write_yaml(fallback_cfg, fallback_payload)

    data_q: Queue = Queue()
    status_q: Queue = Queue()
    stop_evt = Event()
    worker = Process(
        target=_offline_reader_worker,
        args=(str(offline_cfg), str(fallback_cfg), 48, data_q, status_q, stop_evt),
    )
    worker.start()
    msg = data_q.get(timeout=3.0)
    assert msg.get("type") == "data", msg
    assert msg["background_reference_enabled"] is True
    assert msg["background_reference_frames"] == 3
    assert msg["background_reference_scale"] == pytest.approx(1.0)

    shm = shared_memory.SharedMemory(name=str(msg["range_fft_shm_name"]))
    try:
        shape = tuple(int(v) for v in msg["range_fft_shape"])
        snapshots = np.ndarray(shape, dtype=np.complex64, buffer=shm.buf)
        assert shape == (2, 2, 8, int(msg["range_fft_bins_stored"]))
        np.testing.assert_allclose(snapshots, 0.0, atol=1e-5)
    finally:
        shm.close()
        stop_evt.set()
        worker.join(timeout=2.0)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=1.0)
        try:
            shm.unlink()
        except FileNotFoundError:
            pass
        data_q.close()
        status_q.close()
