from __future__ import annotations

from pathlib import Path
import json
from multiprocessing import Event, Process, Queue, shared_memory
import struct

import numpy as np
import pytest
import yaml

from offline_processing import (
    SARReader,
    _backprojection_viewport_max_bin,
    _offline_reader_worker,
    _read_bp_runtime_cfg,
    _subtract_reference_background,
    _viewport_from_cmd_payload,
    offline_map_bounds_from_yaml_dict,
)
from realtime_dsp import build_display_viewport


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
    position: int,
    frames: int = 2,
    samples: int = 8,
    chirps: int = 4,
    rx: int = 4,
    tx: int = 2,
    raw_i16_value: int | None = None,
) -> None:
    capture = {
        "samples": samples,
        "chirps": chirps,
        "rx": rx,
        "tx": tx,
        "frames_per_position": frames,
    }
    header = json.dumps(
        {"format": "rt_capture_v1", "position": position, "capture": capture},
        separators=(",", ":"),
    ).encode("utf-8")
    i16_count = frames * chirps * samples * rx * 2
    if raw_i16_value is None:
        raw_i16 = np.arange(i16_count, dtype=np.int16) + np.int16(position * 7)
    else:
        raw_i16 = np.full(i16_count, int(raw_i16_value), dtype=np.int16)
    raw = raw_i16.tobytes()
    path.write_bytes(b"RTPBIN1\x00" + struct.pack("<I", len(header)) + header + raw)


def test_read_bp_runtime_cfg_rejects_invalid_reconstruction_algorithm(tmp_path: Path) -> None:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "Config.yaml"
    _write_yaml(offline_cfg, _offline_reconstruction_cfg(algorithm="wrong_mode"))
    _write_yaml(fallback_cfg, _fallback_capture_cfg())

    with pytest.raises(ValueError, match="synthetic_range_angle"):
        _read_bp_runtime_cfg(offline_cfg, fallback_cfg)


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


def test_offline_map_bounds_are_loaded_from_reconstruction_config(tmp_path: Path) -> None:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "Config.yaml"
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
    fallback_cfg = tmp_path / "Config.yaml"
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
    fallback_cfg = tmp_path / "Config.yaml"
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
    fallback_cfg = tmp_path / "Config.yaml"
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

    np.testing.assert_allclose(residual, target - np.complex64(0.5) * reference[None, :, :])
    np.testing.assert_array_equal(target, original)
    assert residual.dtype == np.complex64


def test_subtract_reference_background_validates_antenna_and_range_shape() -> None:
    target = np.zeros((2, 8, 16), dtype=np.complex64)
    reference = np.zeros((7, 16), dtype=np.complex64)

    with pytest.raises(ValueError, match="non coerente"):
        _subtract_reference_background(target, reference, scale=1.0)


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
    fallback_cfg = tmp_path / "Config.yaml"
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
    fallback_cfg = tmp_path / "Config.yaml"
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


@pytest.mark.parametrize("nfft_range", [4, 8, 16])
def test_offline_reader_worker_publishes_compact_zero_doppler_cube(
    tmp_path: Path,
    nfft_range: int,
) -> None:
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    _write_capture(run_dir / "capture_pos1.bin", position=1)
    _write_capture(run_dir / "capture_pos2.bin", position=2)
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "Config.yaml"
    _write_yaml(
        offline_cfg,
        {
            "data": {"input_dir": str(run_dir)},
            "capture": {"frames_per_position": 2},
            "scan": {"x_start": 1, "x_end": 2, "x_step": 1, "x_pitch_m": 0.01},
            "reconstruction": {"algorithm": "backprojection"},
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
        assert shape == (2, 2, 8, nfft_range)
        assert np.all(np.isfinite(snapshots))
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
    fallback_cfg = tmp_path / "Config.yaml"
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
                "nfft_range": 8,
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
        args=(str(offline_cfg), str(fallback_cfg), 8, data_q, status_q, stop_evt),
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
        assert shape == (2, 2, 8, 8)
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
