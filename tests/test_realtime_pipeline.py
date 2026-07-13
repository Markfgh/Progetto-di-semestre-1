from __future__ import annotations

import math
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
    gui_h: int = 32,
    gui_w: int | None = None,
    angle_processing: realtime_dsp.AngleProcessingConfig | None = None,
    display_viewport: realtime_dsp.DisplayViewport | None = None,
    display_zoom_cfg: realtime_dsp.DisplayZoomConfig | None = None,
    display_zoom_runtime: realtime_dsp.DisplayZoomRuntime | None = None,
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
    if gui_w is None:
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
    if display_viewport is not None:
        process_kwargs["display_viewport"] = display_viewport
    if display_zoom_cfg is not None:
        process_kwargs["display_zoom_cfg"] = display_zoom_cfg
    if display_zoom_runtime is not None:
        process_kwargs["display_zoom_runtime"] = display_zoom_runtime

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
        angle_processing or realtime_dsp.AngleProcessingConfig(mode="mvdr"),
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


def _build_home_and_zoom_viewports(
    *,
    dsp_cfg: realtime_dsp.RealtimeDSPConfig,
    gui_h: int,
    gui_w: int,
    x_max_m: float = 1.5,
    y_max_m: float = 2.0,
) -> tuple[realtime_dsp.DisplayViewport, realtime_dsp.DisplayViewport]:
    dr_m = dsp_cfg.c * dsp_cfg.fs / (2.0 * dsp_cfg.slope * dsp_cfg.nfft_range)
    home_viewport = realtime_dsp.build_display_viewport(
        x_min_m=-x_max_m,
        x_max_m=x_max_m,
        y_min_m=0.0,
        y_max_m=y_max_m,
        dr_m=dr_m,
        seq=0,
    )
    zoom_viewport = realtime_dsp.clamp_display_viewport(
        x_min_m=-0.45,
        x_max_m=0.45,
        y_min_m=0.70,
        y_max_m=1.35,
        home_viewport=home_viewport,
        output_width=gui_w,
        output_height=gui_h,
        dr_m=dr_m,
        seq=1,
    )
    return home_viewport, zoom_viewport


def test_process_buffer_publishes_angle_and_doppler_diagnostics():
    samples = 32
    loops = 8
    n_frames = 1
    range_bin = 9
    dsp_cfg = _dsp_cfg(samples=samples, loops=loops)
    geometry = _geometry()
    angle_axis = realtime_dsp.build_angle_axis_deg(dsp_cfg.nfft_angle, geometry=geometry)
    steering = realtime_dsp.build_angle_steering_matrix(dsp_cfg.virtual_ant, dsp_cfg.nfft_angle, geometry=geometry)
    gui_h = 16
    gui_w = 32
    angle_diag = np.full((samples, dsp_cfg.nfft_angle), -120.0, dtype=np.float32)
    doppler_diag = np.full((samples, loops), -120.0, dtype=np.float32)
    gui_angle_diag_views = (
        np.full(angle_diag.size, -120.0, dtype=np.float32),
        np.full(angle_diag.size, -120.0, dtype=np.float32),
    )
    gui_doppler_diag_views = (
        np.full(doppler_diag.size, -120.0, dtype=np.float32),
        np.full(doppler_diag.size, -120.0, dtype=np.float32),
    )
    gui_heat_views = (
        np.full(gui_h * gui_w, -120.0, dtype=np.float32),
        np.full(gui_h * gui_w, -120.0, dtype=np.float32),
    )
    gui_profile_views = (
        np.full(dsp_cfg.range_profile_count * samples, -120.0, dtype=np.float32),
        np.full(dsp_cfg.range_profile_count * samples, -120.0, dtype=np.float32),
    )
    gui_latest_idx = mp.Value("i", 0)
    gui_latest_seq = mp.Value("i", 0)

    realtime_dsp.process_buffer(
        _raw_target(
            samples=samples,
            loops=loops,
            range_bin=range_bin,
            angle_deg=15.0,
            doppler_bin=1,
            amplitude=30.0,
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
        realtime_dsp.PostRangeFftFilterConfig(),
        realtime_dsp.PostRangeFftFilterConfig(),
        False,
        False,
        realtime_dsp.AngleProcessingConfig(mode="fft"),
        realtime_dsp.HeatmapEMAConfig(enabled=False),
        realtime_dsp.HeatmapSpatialFilterConfig(enabled=False),
        realtime_dsp.DisplayProjectionConfig(projection_mode="cartesian", projection_interp="nearest"),
        geometry,
        steering,
        angle_axis,
        None,
        2.0,
        1.5,
        (np.fft.fftshift(np.fft.fftfreq(loops, d=1.0)) * 4.0).astype(np.float32),
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
        np.empty((n_frames, loops, 20, dsp_cfg.tx, dsp_cfg.rx), dtype=np.complex64),
        np.empty((n_frames, loops, 20, dsp_cfg.virtual_ant), dtype=np.complex64),
        np.empty((n_frames, loops, dsp_cfg.tx, 20, dsp_cfg.rx), dtype=np.complex64),
        np.empty((dsp_cfg.tx, samples, dsp_cfg.rx), dtype=np.float32),
        np.empty((gui_h, gui_w), dtype=np.float32),
        gui_h,
        gui_w,
        gui_heat_views,
        gui_profile_views,
        gui_latest_idx,
        gui_latest_seq,
        threading.Lock(),
        True,
        np.empty((dsp_cfg.range_profile_count, samples), dtype=np.float32),
        mp.Value("d", 0.0),
        mp.Value("d", 0.0),
        mp.Value("d", 0.0),
        mp.Value("d", 0.0),
        dsp_cfg,
        20,
        20,
        gui_angle_diag_views=gui_angle_diag_views,
        gui_doppler_diag_views=gui_doppler_diag_views,
        angle_diag_out_buf=angle_diag,
        doppler_diag_out_buf=doppler_diag,
    )

    published_idx = int(gui_latest_idx.value)
    published_angle = gui_angle_diag_views[published_idx].reshape(angle_diag.shape)
    published_doppler = gui_doppler_diag_views[published_idx].reshape(doppler_diag.shape)
    assert np.isfinite(published_angle).all()
    assert np.isfinite(published_doppler).all()
    assert float(np.max(published_angle[range_bin])) > -120.0
    assert float(np.max(published_doppler[range_bin])) > -120.0


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
    home_viewport, _ = _build_home_and_zoom_viewports(dsp_cfg=dsp_cfg, gui_h=gui_h, gui_w=gui_w)
    assert heatmap_ema.shape == (int(math.ceil(float(home_viewport.range_max_bin_f))), dsp_cfg.nfft_angle)
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


def test_power_xy_zoom_keeps_fixed_texture_size_and_tracks_applied_viewport() -> None:
    gui_h = 48
    gui_w = 96
    dsp_cfg = _dsp_cfg()
    home_viewport, zoom_viewport = _build_home_and_zoom_viewports(dsp_cfg=dsp_cfg, gui_h=gui_h, gui_w=gui_w)
    zoom_cfg = realtime_dsp.DisplayZoomConfig(
        enabled=True,
        output_width=gui_w,
        output_height=gui_h,
        max_zoom_nfft_range=dsp_cfg.nfft_range * 4,
        max_zoom_nfft_angle=dsp_cfg.nfft_angle * 2,
        max_update_hz=0.0,
        dsp_budget_ms=1000.0,
        fallback_mode="baseline_projection",
    )
    full_runtime = realtime_dsp.DisplayZoomRuntime(home_viewport=home_viewport)
    zoom_runtime = realtime_dsp.DisplayZoomRuntime(home_viewport=home_viewport)

    full_heatmap, full_view, _, _ = _run_display_process(
        gui_h=gui_h,
        gui_w=gui_w,
        display_viewport=home_viewport,
        display_zoom_cfg=zoom_cfg,
        display_zoom_runtime=full_runtime,
    )
    zoom_heatmap, zoom_view, _, _ = _run_display_process(
        gui_h=gui_h,
        gui_w=gui_w,
        display_viewport=zoom_viewport,
        display_zoom_cfg=zoom_cfg,
        display_zoom_runtime=zoom_runtime,
    )

    assert full_heatmap is not None
    assert zoom_heatmap is not None
    assert full_view.shape == (gui_h, gui_w)
    assert zoom_view.shape == (gui_h, gui_w)
    assert full_runtime.last_applied_meta is not None
    assert zoom_runtime.last_applied_meta is not None
    assert realtime_dsp.display_viewport_signature(full_runtime.last_applied_meta) == realtime_dsp.display_viewport_signature(home_viewport)
    assert realtime_dsp.display_viewport_signature(zoom_runtime.last_applied_meta) == realtime_dsp.display_viewport_signature(zoom_viewport)
    assert not zoom_runtime.last_applied_meta.fallback_used
    assert zoom_runtime.last_compute_t_s > 0.0
    assert zoom_heatmap.shape[0] > full_heatmap.shape[0]
    assert not np.allclose(full_view, zoom_view)


def test_display_zoom_over_budget_retries_periodically() -> None:
    gui_h = 48
    gui_w = 96
    dsp_cfg = _dsp_cfg()
    home_viewport, zoom_viewport = _build_home_and_zoom_viewports(dsp_cfg=dsp_cfg, gui_h=gui_h, gui_w=gui_w)
    zoom_cfg = realtime_dsp.DisplayZoomConfig(
        enabled=True,
        output_width=gui_w,
        output_height=gui_h,
        max_update_hz=15.0,
        dsp_budget_ms=6.0,
    )
    runtime = realtime_dsp.DisplayZoomRuntime(home_viewport=home_viewport)
    runtime.last_viewport_signature = realtime_dsp.display_viewport_signature(zoom_viewport)
    runtime.last_mode = "power_xy"
    runtime.last_compute_ms = 9.5
    runtime.last_compute_t_s = 10.0

    assert not realtime_dsp.should_recompute_display_zoom(
        runtime,
        active_viewport_sig=realtime_dsp.display_viewport_signature(zoom_viewport),
        display_mode="power_xy",
        display_zoom_cfg=zoom_cfg,
        now_s=10.10,
    )
    assert realtime_dsp.should_recompute_display_zoom(
        runtime,
        active_viewport_sig=realtime_dsp.display_viewport_signature(zoom_viewport),
        display_mode="power_xy",
        display_zoom_cfg=zoom_cfg,
        now_s=10.30,
    )
    assert realtime_dsp.should_recompute_display_zoom(
        runtime,
        active_viewport_sig=realtime_dsp.display_viewport_signature(home_viewport),
        display_mode="power_xy",
        display_zoom_cfg=zoom_cfg,
        now_s=10.10,
    )


def test_display_zoom_runtime_home_viewport_update_resets_stale_state() -> None:
    dsp_cfg = _dsp_cfg()
    gui_h = 48
    gui_w = 96
    dr_m = dsp_cfg.c * dsp_cfg.fs / (2.0 * dsp_cfg.slope * dsp_cfg.nfft_range)
    old_home, _ = _build_home_and_zoom_viewports(dsp_cfg=dsp_cfg, gui_h=gui_h, gui_w=gui_w)
    new_home = realtime_dsp.build_display_viewport(
        x_min_m=-0.75,
        x_max_m=0.75,
        y_min_m=0.0,
        y_max_m=1.2,
        dr_m=dr_m,
        seq=3,
    )
    runtime = realtime_dsp.DisplayZoomRuntime(home_viewport=old_home)
    runtime.last_view_db = np.ones((gui_h, gui_w), dtype=np.float32)
    runtime.last_view_alpha = np.ones((gui_h, gui_w), dtype=np.float32)
    runtime.last_applied_meta = realtime_dsp.applied_viewport_meta_from_viewport(
        old_home,
        fallback_used=True,
        frame_seq=17,
    )
    runtime.last_viewport_signature = realtime_dsp.display_viewport_signature(old_home)
    runtime.last_compute_ms = 11.0
    runtime.last_compute_t_s = 42.0
    runtime.last_mode = "range_angle_moving"

    changed = realtime_dsp.update_display_zoom_runtime_home_viewport(runtime, new_home)

    assert changed
    assert realtime_dsp.display_viewport_signature(runtime.home_viewport) == realtime_dsp.display_viewport_signature(new_home)
    assert runtime.last_view_db is None
    assert runtime.last_view_alpha is None
    assert runtime.last_applied_meta is None
    assert runtime.last_viewport_signature is None
    assert runtime.last_compute_ms == 0.0
    assert runtime.last_compute_t_s == 0.0
    assert runtime.last_mode == "power_xy"


def test_requested_viewport_matching_updated_home_disables_zoom() -> None:
    gui_h = 48
    gui_w = 96
    dsp_cfg = _dsp_cfg()
    dr_m = dsp_cfg.c * dsp_cfg.fs / (2.0 * dsp_cfg.slope * dsp_cfg.nfft_range)
    old_home, _ = _build_home_and_zoom_viewports(dsp_cfg=dsp_cfg, gui_h=gui_h, gui_w=gui_w)
    new_home = realtime_dsp.build_display_viewport(
        x_min_m=-0.75,
        x_max_m=0.75,
        y_min_m=0.0,
        y_max_m=1.2,
        dr_m=dr_m,
        seq=5,
    )
    stale_requested = realtime_dsp.clamp_display_viewport(
        x_min_m=new_home.x_min_m,
        x_max_m=new_home.x_max_m,
        y_min_m=new_home.y_min_m,
        y_max_m=new_home.y_max_m,
        home_viewport=old_home,
        output_width=gui_w,
        output_height=gui_h,
        dr_m=dr_m,
        seq=5,
    )
    assert stale_requested.zoom_level > 1.0

    runtime = realtime_dsp.DisplayZoomRuntime(home_viewport=old_home)
    realtime_dsp.update_display_zoom_runtime_home_viewport(runtime, new_home)
    requested_home = realtime_dsp.clamp_display_viewport(
        x_min_m=new_home.x_min_m,
        x_max_m=new_home.x_max_m,
        y_min_m=new_home.y_min_m,
        y_max_m=new_home.y_max_m,
        home_viewport=runtime.home_viewport,
        output_width=gui_w,
        output_height=gui_h,
        dr_m=dr_m,
        seq=5,
    )
    zoom_cfg = realtime_dsp.DisplayZoomConfig(
        enabled=True,
        output_width=gui_w,
        output_height=gui_h,
        max_zoom_nfft_range=dsp_cfg.nfft_range * 4,
        max_zoom_nfft_angle=dsp_cfg.nfft_angle * 2,
        max_update_hz=0.0,
        dsp_budget_ms=1000.0,
        fallback_mode="baseline_projection",
    )

    _, view_display, _, _ = _run_display_process(
        gui_h=gui_h,
        gui_w=gui_w,
        display_viewport=requested_home,
        display_zoom_cfg=zoom_cfg,
        display_zoom_runtime=runtime,
    )

    assert requested_home.zoom_level == 1.0
    assert runtime.last_compute_t_s == 0.0
    assert runtime.last_applied_meta is not None
    assert not runtime.last_applied_meta.fallback_used
    assert runtime.last_applied_meta.zoom_level == 1.0
    assert realtime_dsp.display_viewport_signature(runtime.home_viewport) == realtime_dsp.display_viewport_signature(new_home)
    assert realtime_dsp.display_viewport_signature(runtime.last_applied_meta) == realtime_dsp.display_viewport_signature(requested_home)
    assert view_display.shape == (gui_h, gui_w)


def test_process_buffer_zoom_args_leave_tracking_detections_unchanged() -> None:
    samples = 32
    loops = 4
    n_frames = 1
    range_bin = 9
    angle_deg = 25.0
    gui_h = 48
    gui_w = 96
    dsp_cfg = _dsp_cfg(samples=samples, loops=loops)
    geometry = _geometry()
    angle_axis = realtime_dsp.build_angle_axis_deg(dsp_cfg.nfft_angle, geometry=geometry)
    steering = realtime_dsp.build_angle_steering_matrix(dsp_cfg.virtual_ant, dsp_cfg.nfft_angle, geometry=geometry)
    home_viewport, zoom_viewport = _build_home_and_zoom_viewports(dsp_cfg=dsp_cfg, gui_h=gui_h, gui_w=gui_w)
    zoom_cfg = realtime_dsp.DisplayZoomConfig(
        enabled=True,
        output_width=gui_w,
        output_height=gui_h,
        max_zoom_nfft_range=dsp_cfg.nfft_range * 4,
        max_zoom_nfft_angle=dsp_cfg.nfft_angle * 2,
        max_update_hz=0.0,
        dsp_budget_ms=1000.0,
        fallback_mode="baseline_projection",
    )

    sample_phase = np.exp(1j * 2.0 * np.pi * range_bin * np.arange(samples, dtype=np.float32) / samples)
    snapshot = _snapshot(angle_deg).reshape(2, 4)
    raw = (
        sample_phase.reshape(1, 1, 1, samples, 1)
        * snapshot.reshape(1, 1, 2, 1, 4)
        * np.complex64(30.0 + 0.0j)
    )
    raw = np.tile(raw, (n_frames, loops, 1, 1, 1)).astype(np.complex64, copy=False).reshape(-1)

    def _run_detection(
        *,
        display_viewport: realtime_dsp.DisplayViewport,
        display_zoom_cfg: realtime_dsp.DisplayZoomConfig,
        display_zoom_runtime: realtime_dsp.DisplayZoomRuntime,
    ) -> tuple[np.ndarray | None, list[realtime_dsp.Detection], np.ndarray]:
        gui_heat_views = (
            np.full(gui_h * gui_w, -120.0, dtype=np.float32),
            np.full(gui_h * gui_w, -120.0, dtype=np.float32),
        )
        gui_profile_views = (
            np.full(dsp_cfg.range_profile_count * samples, -120.0, dtype=np.float32),
            np.full(dsp_cfg.range_profile_count * samples, -120.0, dtype=np.float32),
        )
        profiles_out = np.empty((dsp_cfg.range_profile_count, samples), dtype=np.float32)
        return realtime_dsp.process_buffer(
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
            display_viewport=display_viewport,
            display_zoom_cfg=display_zoom_cfg,
            display_zoom_runtime=display_zoom_runtime,
        )

    _, detections_base, cal_base = _run_detection(
        display_viewport=home_viewport,
        display_zoom_cfg=realtime_dsp.DisplayZoomConfig(
            enabled=False,
            output_width=gui_w,
            output_height=gui_h,
        ),
        display_zoom_runtime=realtime_dsp.DisplayZoomRuntime(home_viewport=home_viewport),
    )
    _, detections_zoom, cal_zoom = _run_detection(
        display_viewport=zoom_viewport,
        display_zoom_cfg=zoom_cfg,
        display_zoom_runtime=realtime_dsp.DisplayZoomRuntime(home_viewport=home_viewport),
    )

    assert detections_base == detections_zoom
    np.testing.assert_allclose(cal_base, cal_zoom, rtol=0.0, atol=0.0)


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


def test_cartesian_projection_bilinear_keeps_last_half_range_bin_supported() -> None:
    angle_axis = np.asarray([0.0], dtype=np.float32)
    heatmap = np.asarray(
        [
            [1.0],
            [7.0],
        ],
        dtype=np.float32,
    )
    projected = realtime_dsp.project_heatmap_for_display(
        heatmap,
        angle_axis_deg=angle_axis,
        dr_m=1.0,
        gui_h=1,
        gui_w=1,
        y_max_m=1.5,
        x_max_m=0.0,
        projection_mode="cartesian",
        projection_interp="bilinear",
        fill_value=float("nan"),
        y_min_m=1.0,
    )

    assert projected[0, 0] == np.float32(7.0)
