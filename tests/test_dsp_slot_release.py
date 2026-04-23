from __future__ import annotations

import multiprocessing as mp
import queue
import threading
from types import SimpleNamespace

import numpy as np

import realtime_dsp


class OverwritingFreeSlots:
    def __init__(self, shm_frames: bytearray, overwrite_i16: np.ndarray) -> None:
        self._shm_frames = shm_frames
        self._overwrite_i16 = np.asarray(overwrite_i16, dtype=np.int16)
        self.released: list[int] = []

    def put_nowait(self, slot: int) -> None:
        self.released.append(int(slot))
        shm_i16 = np.frombuffer(self._shm_frames, dtype=np.int16)
        shm_i16[:] = self._overwrite_i16


def test_dsp_releases_slot_only_after_copy(monkeypatch) -> None:
    dsp_cfg = realtime_dsp.RealtimeDSPConfig(
        c=3.0e8,
        fs=1.0,
        slope=1.0,
        samples=2,
        chirps=1,
        rx=4,
        tx=1,
        x_frames=1,
        bytes_per_frame=32,
        nfft_range=4,
        nfft_angle=4,
        range_max_display=1.0,
        range_profile_count=1,
        virtual_ant=4,
        fft_workers=1,
        debug_stats=False,
    )

    original_i16 = np.array(
        [1, 2, 3, 4, 10, 20, 30, 40, 5, 6, 7, 8, 50, 60, 70, 80],
        dtype=np.int16,
    )
    overwrite_i16 = np.full(original_i16.shape, 777, dtype=np.int16)
    expected_complex = np.array(
        [1 + 10j, 2 + 20j, 3 + 30j, 4 + 40j, 5 + 50j, 6 + 60j, 7 + 70j, 8 + 80j],
        dtype=np.complex64,
    )

    shm_frames = bytearray(original_i16.tobytes())
    free_slots = OverwritingFreeSlots(shm_frames, overwrite_i16)
    dsp_ready_queue: queue.Queue[tuple[int, int]] = queue.Queue()
    dsp_cmd_queue: queue.Queue[dict[str, str]] = queue.Queue()
    dsp_ready_queue.put((1, 0))

    slot_state = mp.Array("i", [1])
    slot_ok = mp.Array("i", [1])
    slot_usemask = mp.Array("i", [1])
    slot_pub_seq = mp.Array("i", [1])
    publish_lock = threading.Lock()

    stop_evt = threading.Event()
    captured: dict[str, np.ndarray] = {}

    monkeypatch.setattr(realtime_dsp, "selection_from_yaml_dict", lambda cfg: realtime_dsp.DspSelection("none", "none", "none"))
    monkeypatch.setattr(
        realtime_dsp,
        "mean_selections_from_yaml_dict",
        lambda cfg: (realtime_dsp.MeanSelection(enabled=False), realtime_dsp.MeanSelection(enabled=False)),
    )
    monkeypatch.setattr(realtime_dsp, "slow_time_from_yaml_dict", lambda cfg: SimpleNamespace(enabled=False, mode="none"))
    monkeypatch.setattr(
        realtime_dsp,
        "background_subtraction_from_yaml_dict",
        lambda cfg: realtime_dsp.BackgroundSubtractionConfig(enabled=False),
    )
    monkeypatch.setattr(
        realtime_dsp,
        "loop_average_after_background_from_yaml_dict",
        lambda cfg: SimpleNamespace(enabled=False),
    )
    monkeypatch.setattr(realtime_dsp, "angle_processing_from_yaml_dict", lambda cfg: SimpleNamespace())
    monkeypatch.setattr(realtime_dsp, "heatmap_ema_from_yaml_dict", lambda cfg: SimpleNamespace(enabled=False))
    monkeypatch.setattr(realtime_dsp, "heatmap_spatial_filter_from_yaml_dict", lambda cfg: SimpleNamespace())
    monkeypatch.setattr(
        realtime_dsp,
        "display_projection_from_yaml_dict",
        lambda cfg: SimpleNamespace(projection_mode="cartesian", projection_interp="nearest"),
    )
    monkeypatch.setattr(realtime_dsp, "detection_static_from_yaml_dict", lambda cfg: SimpleNamespace(enabled=False))
    monkeypatch.setattr(
        realtime_dsp,
        "detection_moving_from_yaml_dict",
        lambda cfg: SimpleNamespace(enabled=False, doppler_fft_shift=False),
    )
    monkeypatch.setattr(realtime_dsp, "fusion_from_yaml_dict", lambda cfg: SimpleNamespace(enabled=False))
    monkeypatch.setattr(realtime_dsp, "tracking_from_yaml_dict", lambda cfg: SimpleNamespace(enabled=False))
    monkeypatch.setattr(realtime_dsp, "tracker_from_yaml_dict", lambda cfg: SimpleNamespace())
    monkeypatch.setattr(
        realtime_dsp,
        "build_windows",
        lambda selection, samples, n_loops, virtual_ant: (
            np.ones((1, 1, 1, samples, 1), dtype=np.float32),
            np.ones((1, n_loops, 1, 1, 1), dtype=np.float32),
            np.ones((1, 1, 1, virtual_ant), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        realtime_dsp,
        "build_angle_steering_matrix",
        lambda virtual_ant, nfft_angle, geometry=None: np.zeros((nfft_angle, virtual_ant), dtype=np.complex64),
    )
    monkeypatch.setattr(realtime_dsp, "build_angle_axis_deg", lambda nfft_angle, geometry=None: np.zeros(nfft_angle, dtype=np.float32))
    monkeypatch.setattr(realtime_dsp, "resolve_display_crossrange_max_m", lambda display_y_max_m, angle_axis_deg, display_projection_cfg: display_y_max_m)
    monkeypatch.setattr(realtime_dsp, "build_display_projection_lut", lambda **kwargs: None)
    monkeypatch.setattr(realtime_dsp, "build_doppler_axis_mps", lambda cfg_dict, dsp_cfg, n_doppler, doppler_fft_shift: np.zeros(n_doppler, dtype=np.float32))

    def fake_process_buffer(raw_buffer, n_frames, *args, **kwargs):
        captured["raw_buffer"] = np.array(raw_buffer[: expected_complex.size], copy=True)
        captured["display_heatmap_mode"] = kwargs.get("display_heatmap_mode")
        stop_evt.set()
        cal_vector = np.asarray(kwargs.get("cal_vector", np.ones(dsp_cfg.virtual_ant, dtype=np.complex64)), dtype=np.complex64)
        return None, [], cal_vector

    monkeypatch.setattr(realtime_dsp, "process_buffer", fake_process_buffer)

    gui_dbuf = bytearray(2 * 1 * 1 * 4)
    gui_prof_dbuf = bytearray(2 * dsp_cfg.range_profile_count * 1 * 4)
    tracks_xy_dbuf = bytearray(4 * 4)
    tracks_meta_dbuf = bytearray(4 * 4)
    tracks_state_dbuf = bytearray(2 * 4)
    tracks_stop_xy_dbuf = bytearray(2 * 4)

    realtime_dsp.dsp_worker(
        free_slots=free_slots,
        dsp_ready_queue=dsp_ready_queue,
        dsp_cmd_queue=dsp_cmd_queue,
        shm_frames=shm_frames,
        slot_state=slot_state,
        slot_ok=slot_ok,
        slot_usemask=slot_usemask,
        slot_pub_seq=slot_pub_seq,
        publish_lock=publish_lock,
        gui_dbuf=gui_dbuf,
        gui_prof_dbuf=gui_prof_dbuf,
        gui_h=1,
        fft_plot_h=1,
        gui_w=1,
        gui_latest_idx=mp.Value("i", 0),
        gui_latest_seq=mp.Value("i", 0),
        gui_lock=threading.Lock(),
        tracks_xy_dbuf=tracks_xy_dbuf,
        tracks_meta_dbuf=tracks_meta_dbuf,
        tracks_state_dbuf=tracks_state_dbuf,
        tracks_stop_xy_dbuf=tracks_stop_xy_dbuf,
        tracks_count=mp.Value("i", 0),
        tracks_seq=mp.Value("i", 0),
        tracks_lock=threading.Lock(),
        dsp_skip=mp.Value("i", 0),
        dsp_ms_avg=mp.Value("d", 0.0),
        dsp_ms_p95=mp.Value("d", 0.0),
        norm_to_peak=mp.Value("i", 1),
        stat_raw_min_db=mp.Value("d", 0.0),
        stat_raw_max_db=mp.Value("d", 0.0),
        stat_norm_min_db=mp.Value("d", 0.0),
        stat_norm_max_db=mp.Value("d", 0.0),
        stop_evt=stop_evt,
        cfg_dict={},
        dsp_cfg=dsp_cfg,
        heatmap_view_mode=mp.Value("i", 1),
    )

    assert free_slots.released == [0]
    np.testing.assert_array_equal(captured["raw_buffer"], expected_complex)
    assert captured["display_heatmap_mode"] == "range_angle_moving"

