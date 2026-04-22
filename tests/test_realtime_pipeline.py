from __future__ import annotations

import multiprocessing as mp
import threading

import numpy as np

import realtime_dsp


def _geometry() -> realtime_dsp.VirtualArrayGeometry:
    return realtime_dsp.VirtualArrayGeometry(
        order_flat=np.arange(8, dtype=np.int32),
        phase_centers_lambda=np.arange(8, dtype=np.float32) * np.float32(0.25),
        identity_order=True,
        uniform_half_lambda=False,
        uniform_spacing_lambda=0.25,
        angle_axis_sign=1.0,
        angle_u_to_sin_scale=2.0,
    )


def _snapshot(angle_deg: float) -> np.ndarray:
    geometry = _geometry()
    u_target = np.float32(2.0 * np.sin(np.deg2rad(np.float32(angle_deg))))
    return np.exp((-1j * np.float32(2.0 * np.pi)) * geometry.phase_centers_lambda * u_target).astype(np.complex64)


def _dsp_cfg(samples: int = 32, loops: int = 4) -> realtime_dsp.RealtimeDSPConfig:
    return realtime_dsp.RealtimeDSPConfig(
        c=1.0,
        fs=1.0,
        slope=0.15625,
        samples=samples,
        chirps=loops * 2,
        rx=4,
        tx=2,
        x_frames=1,
        bytes_per_frame=0,
        nfft_range=samples,
        nfft_angle=256,
        range_max_display=2.0,
        range_profile_count=8,
        virtual_ant=8,
        fft_workers=1,
        debug_stats=True,
    )


def test_process_buffer_synthetic_static_target_outputs_detection_and_finite_views() -> None:
    samples = 32
    loops = 4
    n_frames = 1
    range_bin = 9
    angle_deg = 25.0
    dsp_cfg = _dsp_cfg(samples=samples, loops=loops)
    geometry = _geometry()
    angle_axis = realtime_dsp.build_angle_axis_deg(dsp_cfg.nfft_angle, geometry=geometry)
    steering = realtime_dsp.build_angle_steering_matrix(dsp_cfg.virtual_ant, dsp_cfg.nfft_angle, geometry=geometry)

    sample_phase = np.exp(1j * 2.0 * np.pi * range_bin * np.arange(samples, dtype=np.float32) / samples)
    snapshot = _snapshot(angle_deg).reshape(2, 4)
    raw = (
        sample_phase.reshape(1, 1, 1, samples, 1)
        * snapshot.reshape(1, 1, 2, 1, 4)
        * np.complex64(30.0 + 0.0j)
    )
    raw = np.tile(raw, (n_frames, loops, 1, 1, 1)).astype(np.complex64, copy=False).reshape(-1)

    gui_h = 32
    gui_w = 32
    gui_heat_views = (
        np.full(gui_h * gui_w, -120.0, dtype=np.float32),
        np.full(gui_h * gui_w, -120.0, dtype=np.float32),
    )
    gui_profile_views = (
        np.full(dsp_cfg.range_profile_count * samples, -120.0, dtype=np.float32),
        np.full(dsp_cfg.range_profile_count * samples, -120.0, dtype=np.float32),
    )
    profiles_out = np.empty((dsp_cfg.range_profile_count, samples), dtype=np.float32)

    heatmap_ema, detections, cal_vector = realtime_dsp.process_buffer(
        raw,
        n_frames,
        np.ones((1, 1, 1, samples, 1), dtype=np.float32),
        np.ones((1, loops, 1, 1, 1), dtype=np.float32),
        np.ones((1, 1, 1, dsp_cfg.virtual_ant), dtype=np.float32),
        False,
        False,
        False,
        realtime_dsp.MeanSelection(enabled=False),
        realtime_dsp.PostRangeFftFilterConfig(),
        realtime_dsp.PostRangeFftFilterConfig(),
        realtime_dsp.PostRangeFftFilterConfig(),
        False,
        False,
        realtime_dsp.AngleProcessingConfig(mode="bartlett"),
        realtime_dsp.HeatmapEMAConfig(enabled=False),
        realtime_dsp.HeatmapSpatialFilterConfig(enabled=False),
        realtime_dsp.DisplayProjectionConfig(projection_mode="cartesian", projection_interp="nearest"),
        geometry,
        steering,
        angle_axis,
        None,
        2.0,
        1.5,
        None,
        None,
        realtime_dsp.DetectionConfigStatic(
            enabled=True,
            threshold_mode="relative",
            threshold_db=-6.0,
            localmax_range_bins=1,
            localmax_angle_bins=3,
            min_power_db=-120.0,
            max_detections=1,
        ),
        realtime_dsp.DetectionConfigMoving(enabled=False),
        realtime_dsp.FusionConfig(enabled=True),
        realtime_dsp.BackgroundSubtractionState(),
        realtime_dsp.BackgroundSubtractionState(),
        realtime_dsp.BackgroundSubtractionState(),
        None,
        False,
        np.ones(dsp_cfg.virtual_ant, dtype=np.complex64),
        False,
        None,
        None,
        None,
        None,
        None,
        gui_h,
        gui_w,
        gui_heat_views,
        gui_profile_views,
        mp.Value("i", 0),
        mp.Value("i", 0),
        threading.Lock(),
        True,
        profiles_out,
        mp.Value("d", 0.0),
        mp.Value("d", 0.0),
        mp.Value("d", 0.0),
        mp.Value("d", 0.0),
        dsp_cfg,
        20,
        20,
    )

    assert heatmap_ema is not None
    assert heatmap_ema.shape == (20, dsp_cfg.nfft_angle)
    assert len(detections) == 1
    det = detections[0]
    assert abs(det.range_bin - range_bin) <= 1
    assert abs(det.angle_deg - angle_deg) <= 1.0
    assert np.all(np.isfinite(gui_heat_views[1]))
    assert np.all(np.isfinite(gui_profile_views[1]))
    assert np.all(np.isfinite(profiles_out))
    np.testing.assert_allclose(cal_vector, np.ones(dsp_cfg.virtual_ant, dtype=np.complex64))
