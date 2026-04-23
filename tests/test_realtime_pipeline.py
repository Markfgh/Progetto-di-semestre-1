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


def _raw_target(
    *,
    samples: int,
    loops: int,
    range_bin: int,
    angle_deg: float,
    doppler_bin: int = 0,
    amplitude: float = 30.0,
) -> np.ndarray:
    sample_phase = np.exp(1j * 2.0 * np.pi * range_bin * np.arange(samples, dtype=np.float32) / samples)
    doppler_cycles = np.float32(float(doppler_bin) / float(loops))
    loop_phase = np.exp(1j * 2.0 * np.pi * doppler_cycles * np.arange(loops, dtype=np.float32))
    tx_phase = np.exp(1j * 2.0 * np.pi * doppler_cycles * (np.arange(2, dtype=np.float32) / np.float32(2.0)))
    snapshot = _snapshot(angle_deg).reshape(2, 4) * tx_phase.reshape(2, 1)
    raw = (
        sample_phase.reshape(1, 1, 1, samples, 1)
        * loop_phase.reshape(1, loops, 1, 1, 1)
        * snapshot.reshape(1, 1, 2, 1, 4)
        * np.complex64(amplitude + 0.0j)
    )
    return raw.astype(np.complex64, copy=False).reshape(-1)


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


def _run_display_process(
    *,
    display_heatmap_mode: str | None = "power_xy",
    normalize_to_peak: bool = True,
    doppler_bin: int = 1,
    amplitude: float = 30.0,
    detection_moving_pre_doppler_filters: realtime_dsp.PostRangeFftFilterConfig | None = None,
    display_post_range_fft_filters: realtime_dsp.PostRangeFftFilterConfig | None = None,
    display_projection_cfg: realtime_dsp.DisplayProjectionConfig | None = None,
    range_angle_moving_cfg: realtime_dsp.RangeAngleMovingConfig | None = None,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    samples = 32
    loops = 8
    n_frames = 1
    range_bin = 9
    angle_deg = 0.0
    dsp_cfg = _dsp_cfg(samples=samples, loops=loops)
    if range_angle_moving_cfg is not None:
        dsp_cfg = realtime_dsp.RealtimeDSPConfig(
            **{
                **dsp_cfg.__dict__,
                "range_angle_moving": range_angle_moving_cfg,
            }
        )
    geometry = _geometry()
    angle_axis = realtime_dsp.build_angle_axis_deg(dsp_cfg.nfft_angle, geometry=geometry)
    steering = realtime_dsp.build_angle_steering_matrix(dsp_cfg.virtual_ant, dsp_cfg.nfft_angle, geometry=geometry)
    gui_h = 32
    gui_w = dsp_cfg.nfft_angle if display_heatmap_mode in {"range_angle_moving", "velocity_xy"} else 65
    gui_heat_views = (
        np.full(gui_h * gui_w, -120.0, dtype=np.float32),
        np.full(gui_h * gui_w, -120.0, dtype=np.float32),
    )
    gui_heat_alpha_views = (
        np.zeros(gui_h * gui_w, dtype=np.float32),
        np.zeros(gui_h * gui_w, dtype=np.float32),
    )
    gui_profile_views = (
        np.full(dsp_cfg.range_profile_count * samples, -120.0, dtype=np.float32),
        np.full(dsp_cfg.range_profile_count * samples, -120.0, dtype=np.float32),
    )
    profiles_out = np.empty((dsp_cfg.range_profile_count, samples), dtype=np.float32)
    doppler_axis = (np.fft.fftshift(np.fft.fftfreq(loops, d=1.0)) * 4.0).astype(np.float32)

    process_kwargs = {}
    if display_heatmap_mode is not None:
        process_kwargs["display_heatmap_mode"] = display_heatmap_mode

    heatmap_ema, _, _ = realtime_dsp.process_buffer(
        _raw_target(
            samples=samples,
            loops=loops,
            range_bin=range_bin,
            angle_deg=angle_deg,
            doppler_bin=doppler_bin,
            amplitude=amplitude,
        ),
        n_frames,
        np.ones((1, 1, 1, samples, 1), dtype=np.float32),
        np.ones((1, loops, 1, 1, 1), dtype=np.float32),
        np.ones((1, 1, 1, dsp_cfg.virtual_ant), dtype=np.float32),
        False,
        False,
        False,
        realtime_dsp.MeanSelection(enabled=False),
        realtime_dsp.PostRangeFftFilterConfig(),
        detection_moving_pre_doppler_filters or realtime_dsp.PostRangeFftFilterConfig(),
        display_post_range_fft_filters or realtime_dsp.PostRangeFftFilterConfig(),
        False,
        False,
        realtime_dsp.AngleProcessingConfig(mode="mvdr"),
        realtime_dsp.HeatmapEMAConfig(enabled=False),
        realtime_dsp.HeatmapSpatialFilterConfig(enabled=False),
        display_projection_cfg or realtime_dsp.DisplayProjectionConfig(projection_mode="cartesian", projection_interp="nearest"),
        geometry,
        steering,
        angle_axis,
        None,
        2.0,
        1.5,
        doppler_axis,
        realtime_dsp.build_tdm_mimo_doppler_compensation_table(loops, dsp_cfg.tx, doppler_fft_shift=True),
        realtime_dsp.DetectionConfigStatic(enabled=False),
        realtime_dsp.DetectionConfigMoving(enabled=False, doppler_fft_shift=True),
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
        np.empty((n_frames, loops, dsp_cfg.tx, 20, dsp_cfg.rx), dtype=np.complex64),
        None,
        np.empty((gui_h, gui_w), dtype=np.float32),
        gui_h,
        gui_w,
        gui_heat_views,
        gui_profile_views,
        mp.Value("i", 0),
        mp.Value("i", 0),
        threading.Lock(),
        normalize_to_peak,
        profiles_out,
        mp.Value("d", 0.0),
        mp.Value("d", 0.0),
        mp.Value("d", 0.0),
        mp.Value("d", 0.0),
        dsp_cfg,
        20,
        20,
        gui_heat_alpha_views=gui_heat_alpha_views,
        **process_kwargs,
    )
    return (
        heatmap_ema,
        gui_heat_views[1].reshape(gui_h, gui_w),
        gui_heat_alpha_views[1].reshape(gui_h, gui_w),
        doppler_axis,
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


def test_range_angle_moving_reports_signed_doppler_and_alpha_for_synthetic_target() -> None:
    doppler_bin = 1
    _, velocity_view, alpha_view, doppler_axis = _run_display_process(
        display_heatmap_mode="range_angle_moving",
        normalize_to_peak=True,
        doppler_bin=doppler_bin,
    )

    expected = float(doppler_axis[(doppler_axis.size // 2) + doppler_bin])
    assert expected > 0.0
    peak = np.unravel_index(int(np.argmax(alpha_view)), alpha_view.shape)
    assert alpha_view[peak] > np.float32(0.0)
    assert velocity_view[peak] == np.float32(expected)
    assert np.count_nonzero(alpha_view > 0.0) > 0
    assert np.all(alpha_view[(alpha_view <= 0.0)] == np.float32(0.0))


def test_process_buffer_power_xy_default_matches_explicit_power_mode() -> None:
    heatmap_default, view_default, _, _ = _run_display_process(
        display_heatmap_mode=None,
        normalize_to_peak=True,
    )
    heatmap_explicit, view_explicit, _, _ = _run_display_process(
        display_heatmap_mode="power_xy",
        normalize_to_peak=True,
    )

    assert heatmap_default is not None
    assert heatmap_explicit is not None
    np.testing.assert_allclose(heatmap_default, heatmap_explicit, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(view_default, view_explicit, rtol=0.0, atol=0.0)


def test_range_angle_moving_ignores_normalize_to_peak() -> None:
    _, view_norm_on, alpha_norm_on, _ = _run_display_process(
        display_heatmap_mode="range_angle_moving",
        normalize_to_peak=True,
    )
    _, view_norm_off, alpha_norm_off, _ = _run_display_process(
        display_heatmap_mode="range_angle_moving",
        normalize_to_peak=False,
    )

    np.testing.assert_allclose(view_norm_on, view_norm_off, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(alpha_norm_on, alpha_norm_off, rtol=0.0, atol=0.0)


def test_range_angle_moving_distinguishes_zero_velocity_from_invisible_cells() -> None:
    _, velocity_view, alpha_view, _ = _run_display_process(
        display_heatmap_mode="range_angle_moving",
        doppler_bin=0,
    )

    valid = alpha_view > np.float32(0.0)
    assert np.count_nonzero(valid) > 0
    assert np.all(np.isclose(velocity_view[valid], 0.0))
    assert np.count_nonzero(~valid) > 0


def test_range_angle_moving_uses_moving_pre_doppler_filters() -> None:
    moving_filters = realtime_dsp.PostRangeFftFilterConfig(
        slow_time=realtime_dsp.SlowTimeConfig(enabled=True, mode="mean_subtraction")
    )
    _, _, alpha_view, _ = _run_display_process(
        display_heatmap_mode="range_angle_moving",
        doppler_bin=0,
        detection_moving_pre_doppler_filters=moving_filters,
    )

    assert np.count_nonzero(alpha_view > 0.0) == 0


def test_range_angle_moving_alpha_is_zero_below_min_power_floor() -> None:
    _, _, alpha_view, _ = _run_display_process(
        display_heatmap_mode="range_angle_moving",
        amplitude=0.01,
        range_angle_moving_cfg=realtime_dsp.RangeAngleMovingConfig(
            relative_power_floor_db=-80.0,
            min_power_db=60.0,
            min_dominance_ratio=0.0,
        ),
    )

    assert np.count_nonzero(alpha_view > 0.0) == 0


def test_range_angle_moving_projects_velocity_and_alpha_to_xy(monkeypatch) -> None:
    original_projection = realtime_dsp.project_heatmap_for_display
    projection_modes: list[str] = []

    def spy_projection(*args, **kwargs):
        projection_modes.append(str(kwargs.get("projection_mode", "")))
        return original_projection(*args, **kwargs)

    monkeypatch.setattr(realtime_dsp, "project_heatmap_for_display", spy_projection)
    _, _, alpha_view, _ = _run_display_process(display_heatmap_mode="range_angle_moving")

    assert np.count_nonzero(alpha_view > 0.0) > 0
    assert projection_modes.count("cartesian") >= 2


def test_cartesian_projection_bilinear_is_nan_aware() -> None:
    angle_axis = np.asarray([0.0, 90.0], dtype=np.float32)
    heatmap = np.asarray(
        [
            [2.0, np.nan],
            [np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    supported = realtime_dsp.project_heatmap_for_display(
        heatmap,
        angle_axis_deg=angle_axis,
        dr_m=1.0,
        gui_h=1,
        gui_w=1,
        y_max_m=1.0,
        x_max_m=0.0,
        projection_mode="cartesian",
        projection_interp="bilinear",
        fill_value=float("nan"),
    )
    unsupported = realtime_dsp.project_heatmap_for_display(
        heatmap,
        angle_axis_deg=angle_axis,
        dr_m=1.0,
        gui_h=1,
        gui_w=1,
        y_max_m=1.98,
        x_max_m=0.0,
        projection_mode="cartesian",
        projection_interp="bilinear",
        fill_value=float("nan"),
    )

    assert supported[0, 0] == np.float32(2.0)
    assert np.isnan(unsupported[0, 0])
