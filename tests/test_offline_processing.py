from __future__ import annotations

from pathlib import Path
import json
from multiprocessing import Event, Process, Queue, shared_memory
import struct

import numpy as np
import pytest
import yaml

from offline_processing import SARReader, _offline_reader_worker, _read_bp_runtime_cfg, _viewport_from_cmd_payload
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


def _write_capture(
    path: Path,
    *,
    position: int,
    frames: int = 2,
    samples: int = 8,
    chirps: int = 4,
    rx: int = 4,
    tx: int = 2,
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
    raw = (np.arange(i16_count, dtype=np.int16) + np.int16(position * 7)).tobytes()
    path.write_bytes(b"RTPBIN1\x00" + struct.pack("<I", len(header)) + header + raw)


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
                "nfft_range": 512,
                "nfft_angle": 64,
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
    assert range_angle_cfg.nfft_range == 512
    assert range_angle_cfg.nfft_angle == 64


def test_offline_synthetic_range_angle_ignores_legacy_slow_time_filter(tmp_path: Path) -> None:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "Config.yaml"
    _write_yaml(
        offline_cfg,
        {
            **_offline_reconstruction_cfg(algorithm="synthetic_range_angle"),
            "offline_sar_range_angle": {
                "slow_time": {"enabled": True, "mode": "mean_subtraction"},
            },
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


def test_stream_reader_validates_without_loading_full_cube(tmp_path: Path) -> None:
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
        },
    )
    _write_yaml(fallback_cfg, _fallback_capture_cfg())

    reader = SARReader(offline_cfg, fallback_cfg)
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
            "bp": {"mode": "mimo_sar", "motion_mode": "static_zero_doppler"},
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
        assert msg["data_stage"] == "zero_doppler_snapshots"
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
