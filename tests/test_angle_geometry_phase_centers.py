from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from offline_dsp import build_mimo_geometry
import realtime_dsp


def _phase_center_geometry(angle_axis_sign: float = 1.0) -> realtime_dsp.VirtualArrayGeometry:
    return realtime_dsp.VirtualArrayGeometry(
        order_flat=np.arange(8, dtype=np.int32),
        phase_centers_lambda=(np.arange(8, dtype=np.float32) * np.float32(0.25)).astype(np.float32, copy=False),
        identity_order=True,
        uniform_half_lambda=False,
        uniform_spacing_lambda=0.25,
        angle_axis_sign=float(angle_axis_sign),
        angle_u_to_sin_scale=2.0,
    )


def _dsp_cfg(nfft_angle: int = 512) -> realtime_dsp.RealtimeDSPConfig:
    return realtime_dsp.RealtimeDSPConfig(
        c=3.0e8,
        fs=1.0,
        slope=1.0,
        samples=1,
        chirps=1,
        rx=4,
        tx=2,
        x_frames=1,
        bytes_per_frame=16,
        nfft_range=64,
        nfft_angle=int(nfft_angle),
        range_max_display=4.0,
        range_profile_count=1,
        virtual_ant=8,
        fft_workers=1,
        debug_stats=False,
    )


def _snapshot_for_angle(geometry: realtime_dsp.VirtualArrayGeometry, angle_deg: float) -> np.ndarray:
    u_target = np.float32(2.0 * np.sin(np.deg2rad(np.float32(angle_deg))))
    phase_centers = geometry.phase_centers_lambda.astype(np.float32, copy=False)
    return np.exp((-1j * np.float32(2.0 * np.pi)) * phase_centers * u_target).astype(
        np.complex64,
        copy=False,
    )


def _estimate_angle(mode: str, angle_deg: float) -> float:
    geometry = _phase_center_geometry()
    dsp_cfg = _dsp_cfg()
    angle_cfg = realtime_dsp.AngleProcessingConfig(
        mode=mode,
        mvdr_diagonal_loading=0.02,
        aggregation="frame_loop",
    )
    angle_axis = realtime_dsp.build_angle_axis_deg(dsp_cfg.nfft_angle, geometry=geometry)
    angle_steering = realtime_dsp.build_angle_steering_matrix(
        virtual_ant=8,
        nfft_angle=dsp_cfg.nfft_angle,
        geometry=geometry,
    )
    snapshot = _snapshot_for_angle(geometry, angle_deg)
    virtual_array = np.tile(snapshot.reshape(1, 1, 1, 8), (4, 4, 1, 1)).astype(np.complex64, copy=False)
    heatmap = realtime_dsp.compute_angle_heatmap(
        virtual_array,
        angle_cfg=angle_cfg,
        dsp_cfg=dsp_cfg,
        angle_steering=angle_steering,
        geometry=geometry,
        ant_spacing=geometry.uniform_spacing_lambda,
    )
    peak_idx = int(np.argmax(heatmap[0]))
    return float(angle_axis[peak_idx])


def _legacy_mvdr_heatmap(
    virtual_array: np.ndarray,
    *,
    angle_cfg: realtime_dsp.AngleProcessingConfig,
    dsp_cfg: realtime_dsp.RealtimeDSPConfig,
    angle_steering: np.ndarray,
) -> np.ndarray:
    num_frames, num_loops, num_range, num_ant = virtual_array.shape
    snapshots = np.asarray(virtual_array, dtype=np.complex64).reshape(num_frames * num_loops, num_range, num_ant)
    x_by_range = snapshots.transpose(1, 2, 0).astype(np.complex128, copy=False)
    steering = np.asarray(angle_steering, dtype=np.complex128)
    eye = np.eye(num_ant, dtype=np.complex128)
    load_factor = np.float64(max(0.0, float(angle_cfg.mvdr_diagonal_loading)))
    heatmap = np.empty((num_range, int(dsp_cfg.nfft_angle)), dtype=np.float32)
    for range_idx in range(num_range):
        x = x_by_range[range_idx]
        cov = (x @ x.conj().T) / np.float64(max(1, snapshots.shape[0]))
        trace_r = float(np.trace(cov).real)
        if np.isfinite(trace_r) and trace_r > 0.0:
            cov = cov + ((load_factor * trace_r / np.float64(max(1, num_ant))) * eye)
        cov_inv = np.linalg.pinv(cov)
        den_left = cov_inv @ steering
        den = np.einsum("ak,ak->k", np.conj(steering), den_left, optimize=True).real.astype(np.float32, copy=False)
        np.maximum(den, np.float32(1e-8), out=den)
        heatmap[range_idx, :] = np.float32(1.0) / den
    return heatmap


def _bartlett_reference_range(snapshot_range: np.ndarray, steering: np.ndarray) -> np.ndarray:
    snapshots = np.asarray(snapshot_range, dtype=np.complex128).reshape(-1, int(steering.shape[0]))
    steering_c = np.asarray(steering, dtype=np.complex128)
    beam = np.einsum("ak,sa->sk", np.conj(steering_c), snapshots, optimize=True)
    power = (beam.real * beam.real + beam.imag * beam.imag).mean(axis=0, dtype=np.float64)
    return power.astype(np.float32, copy=False)


def test_default_virtual_geometry_uses_bistatic_phase_centers() -> None:
    geometry, warnings = realtime_dsp.build_virtual_array_geometry_from_yaml_dict(
        {"antenna": {"virtual_array_order": list(range(8))}},
        SimpleNamespace(tx=2, rx=4, virtual_ant=8),
    )

    np.testing.assert_allclose(
        geometry.phase_centers_lambda,
        np.asarray([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75], dtype=np.float32),
        atol=1e-6,
        rtol=0.0,
    )
    assert geometry.uniform_spacing_lambda == 0.25
    assert geometry.angle_u_to_sin_scale == 2.0
    assert warnings == []


def test_realtime_phase_centers_match_offline_bistatic_geometry_up_to_offset() -> None:
    geometry = _phase_center_geometry()
    x_tx, x_rx = build_mimo_geometry(2, 4, fc_hz=77e9, c_m_s=3e8)
    lambda_m = np.float32(3e8 / 77e9)

    realtime_centers = geometry.phase_centers_lambda.astype(np.float32, copy=False)
    realtime_centered = (realtime_centers - np.mean(realtime_centers, dtype=np.float32)).astype(
        np.float32,
        copy=False,
    )
    offline_centered = (np.float32(0.5) * (x_tx + x_rx) / lambda_m).astype(np.float32, copy=False)

    np.testing.assert_allclose(realtime_centered, offline_centered, atol=1e-6, rtol=0.0)


def test_build_angle_axis_deg_uses_asin_u_over_two() -> None:
    geometry = _phase_center_geometry()

    angle_axis_deg = realtime_dsp.build_angle_axis_deg(8, geometry=geometry)

    expected = np.asarray(
        [-90.0, -48.59038, -30.0, -14.477512, 0.0, 14.477512, 30.0, 48.59038],
        dtype=np.float32,
    )
    np.testing.assert_allclose(angle_axis_deg, expected, atol=1e-4, rtol=0.0)


def test_angle_fft_keeps_u_between_one_and_two_for_phase_centers() -> None:
    geometry = _phase_center_geometry()
    dsp_cfg = realtime_dsp.RealtimeDSPConfig(
        c=3.0e8,
        fs=1.0,
        slope=1.0,
        samples=1,
        chirps=1,
        rx=4,
        tx=2,
        x_frames=1,
        bytes_per_frame=16,
        nfft_range=1,
        nfft_angle=8,
        range_max_display=1.0,
        range_profile_count=1,
        virtual_ant=8,
        fft_workers=1,
        debug_stats=False,
    )
    angle_cfg = realtime_dsp.AngleProcessingConfig(mode="fft")
    angle_steering = realtime_dsp.build_angle_steering_matrix(virtual_ant=8, nfft_angle=8, geometry=geometry)

    u_target = np.float32(1.5)
    phase_centers = geometry.phase_centers_lambda.astype(np.float32, copy=False)
    snapshot = np.exp((-1j * 2.0 * np.pi) * phase_centers * u_target).astype(np.complex64, copy=False)
    virtual_array = snapshot.reshape(1, 1, 1, 8)

    heatmap = realtime_dsp.compute_angle_heatmap(
        virtual_array,
        angle_cfg=angle_cfg,
        dsp_cfg=dsp_cfg,
        angle_steering=angle_steering,
        geometry=geometry,
        ant_spacing=geometry.uniform_spacing_lambda,
    )

    assert heatmap.shape == (1, 8)
    peak_idx = int(np.argmax(heatmap[0]))
    u_axis = realtime_dsp._build_angle_u_axis(dsp_cfg.nfft_angle, spacing_lambda=geometry.uniform_spacing_lambda)
    assert np.isclose(abs(float(u_axis[peak_idx])), 1.5, atol=1e-6, rtol=0.0)
    assert abs(float(u_axis[peak_idx])) > 1.0
    assert float(heatmap[0, peak_idx]) > 0.0


def test_realtime_synthetic_aoa_modes_use_same_physical_sign() -> None:
    for target_angle in (30.0, -30.0, 10.0, 50.0):
        estimates = {
            mode: _estimate_angle(mode, target_angle)
            for mode in ("fft", "bartlett", "mvdr")
        }
        for estimated_angle in estimates.values():
            assert abs(estimated_angle - target_angle) <= 0.75
        assert max(estimates.values()) - min(estimates.values()) <= 0.75


def test_mvdr_solve_matches_legacy_pinv_on_well_conditioned_input() -> None:
    geometry = _phase_center_geometry()
    dsp_cfg = _dsp_cfg(nfft_angle=64)
    angle_cfg = realtime_dsp.AngleProcessingConfig(mode="mvdr", mvdr_diagonal_loading=0.05)
    angle_steering = realtime_dsp.build_angle_steering_matrix(
        virtual_ant=8,
        nfft_angle=dsp_cfg.nfft_angle,
        geometry=geometry,
    )
    rng = np.random.default_rng(1234)
    virtual_array = (
        rng.normal(size=(4, 4, 3, 8)) + 1j * rng.normal(size=(4, 4, 3, 8))
    ).astype(np.complex64)

    heatmap = realtime_dsp.compute_angle_heatmap(
        virtual_array,
        angle_cfg=angle_cfg,
        dsp_cfg=dsp_cfg,
        angle_steering=angle_steering,
        geometry=geometry,
        ant_spacing=geometry.uniform_spacing_lambda,
    )
    legacy = _legacy_mvdr_heatmap(
        virtual_array,
        angle_cfg=angle_cfg,
        dsp_cfg=dsp_cfg,
        angle_steering=angle_steering,
    )

    assert getattr(realtime_dsp.compute_angle_heatmap, "_mvdr_total_bins") == 3
    assert getattr(realtime_dsp.compute_angle_heatmap, "_mvdr_fallback_bins") == 0
    np.testing.assert_allclose(heatmap, legacy, atol=1e-4, rtol=1e-4)


def test_mvdr_falls_back_to_bartlett_for_only_singular_range_bin() -> None:
    geometry = _phase_center_geometry()
    dsp_cfg = _dsp_cfg(nfft_angle=64)
    angle_cfg = realtime_dsp.AngleProcessingConfig(mode="mvdr", mvdr_diagonal_loading=0.0)
    angle_steering = realtime_dsp.build_angle_steering_matrix(
        virtual_ant=8,
        nfft_angle=dsp_cfg.nfft_angle,
        geometry=geometry,
    )
    virtual_array = np.zeros((4, 4, 2, 8), dtype=np.complex64)
    virtual_array[:, :, 0, :] = _snapshot_for_angle(geometry, 20.0).reshape(1, 1, 8)
    rng = np.random.default_rng(4321)
    virtual_array[:, :, 1, :] = (
        rng.normal(size=(4, 4, 8)) + 1j * rng.normal(size=(4, 4, 8))
    ).astype(np.complex64)

    heatmap = realtime_dsp.compute_angle_heatmap(
        virtual_array,
        angle_cfg=angle_cfg,
        dsp_cfg=dsp_cfg,
        angle_steering=angle_steering,
        geometry=geometry,
        ant_spacing=geometry.uniform_spacing_lambda,
    )
    bartlett_bin0 = _bartlett_reference_range(virtual_array[:, :, 0, :], angle_steering)

    assert getattr(realtime_dsp.compute_angle_heatmap, "_mvdr_total_bins") == 2
    assert getattr(realtime_dsp.compute_angle_heatmap, "_mvdr_fallback_bins") == 1
    np.testing.assert_allclose(heatmap[0], bartlett_bin0, atol=1e-5, rtol=1e-5)
    assert np.all(np.isfinite(heatmap[1]))


def test_non_mvdr_modes_do_not_use_mvdr_linear_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    geometry = _phase_center_geometry()
    dsp_cfg = _dsp_cfg(nfft_angle=64)
    angle_steering = realtime_dsp.build_angle_steering_matrix(
        virtual_ant=8,
        nfft_angle=dsp_cfg.nfft_angle,
        geometry=geometry,
    )
    virtual_array = np.tile(
        _snapshot_for_angle(geometry, 30.0).reshape(1, 1, 1, 8),
        (2, 2, 1, 1),
    ).astype(np.complex64)

    def _boom(*args, **kwargs):
        raise AssertionError("MVDR linear solver should not run for non-MVDR modes")

    monkeypatch.setattr(realtime_dsp.np.linalg, "solve", _boom)
    for mode in ("fft", "bartlett"):
        heatmap = realtime_dsp.compute_angle_heatmap(
            virtual_array.copy(),
            angle_cfg=realtime_dsp.AngleProcessingConfig(mode=mode),
            dsp_cfg=dsp_cfg,
            angle_steering=angle_steering,
            geometry=geometry,
            ant_spacing=geometry.uniform_spacing_lambda,
        )
        assert heatmap.shape == (1, int(dsp_cfg.nfft_angle))
        assert np.all(np.isfinite(heatmap))


def test_mvdr_returns_finite_output_for_empty_or_weak_input() -> None:
    geometry = _phase_center_geometry()
    dsp_cfg = _dsp_cfg(nfft_angle=64)
    angle_cfg = realtime_dsp.AngleProcessingConfig(mode="mvdr", mvdr_diagonal_loading=0.0)
    angle_steering = realtime_dsp.build_angle_steering_matrix(
        virtual_ant=8,
        nfft_angle=dsp_cfg.nfft_angle,
        geometry=geometry,
    )
    virtual_array = np.zeros((2, 2, 3, 8), dtype=np.complex64)
    virtual_array[:, :, 1, :] = np.complex64(1e-20 + 0.0j)

    heatmap = realtime_dsp.compute_angle_heatmap(
        virtual_array,
        angle_cfg=angle_cfg,
        dsp_cfg=dsp_cfg,
        angle_steering=angle_steering,
        geometry=geometry,
        ant_spacing=geometry.uniform_spacing_lambda,
    )

    assert heatmap.shape == (3, int(dsp_cfg.nfft_angle))
    assert np.all(np.isfinite(heatmap))


def test_realtime_detection_xy_uses_physical_angle_axis() -> None:
    geometry = _phase_center_geometry()
    dsp_cfg = _dsp_cfg()
    angle_axis = realtime_dsp.build_angle_axis_deg(dsp_cfg.nfft_angle, geometry=geometry)
    angle_steering = realtime_dsp.build_angle_steering_matrix(
        virtual_ant=8,
        nfft_angle=dsp_cfg.nfft_angle,
        geometry=geometry,
    )

    range_bin_m = 0.05
    range_bin = 20
    target_angle = 30.0
    target_range = range_bin * range_bin_m
    snapshot = _snapshot_for_angle(geometry, target_angle)
    range_fft = np.zeros((1, 1, 2, 64, 4), dtype=np.complex64)
    range_fft[0, 0, :, range_bin, :] = snapshot.reshape(2, 4)

    detections, _ = realtime_dsp.detect_static_targets(
        range_fft,
        static_cfg=realtime_dsp.DetectionConfigStatic(
            threshold_mode="relative",
            threshold_db=-3.0,
            min_power_db=-120.0,
            max_detections=1,
        ),
        angle_cfg=realtime_dsp.AngleProcessingConfig(mode="bartlett"),
        dsp_cfg=dsp_cfg,
        virtual_array_geometry=geometry,
        w_angle=np.ones((1, 1, 1, 8), dtype=np.float32),
        angle_steering=angle_steering,
        angle_axis_deg=angle_axis,
        range_bin_m=range_bin_m,
        max_bin=64,
        apply_angle_window=False,
    )

    assert len(detections) == 1
    det = detections[0]
    assert abs(det.angle_deg - target_angle) <= 0.75
    np.testing.assert_allclose(det.x_m, target_range * np.sin(np.deg2rad(target_angle)), atol=0.02, rtol=0.0)
    np.testing.assert_allclose(det.y_m, target_range * np.cos(np.deg2rad(target_angle)), atol=0.02, rtol=0.0)


def test_cartesian_projection_places_peak_at_expected_xy() -> None:
    geometry = _phase_center_geometry()
    angle_axis = realtime_dsp.build_angle_axis_deg(8, geometry=geometry)
    dr_m = 0.05
    target_range = 2.0
    target_angle = 30.0
    range_bin = int(round(target_range / dr_m))
    angle_bin = int(np.argmin(np.abs(angle_axis - np.float32(target_angle))))
    heatmap = np.zeros((80, 8), dtype=np.float32)
    heatmap[range_bin, angle_bin] = np.float32(1.0)

    view = realtime_dsp.project_heatmap_for_display(
        heatmap,
        angle_axis_deg=angle_axis,
        dr_m=dr_m,
        gui_h=151,
        gui_w=151,
        y_max_m=3.0,
        x_max_m=2.0,
        projection_mode="cartesian",
        projection_interp="nearest",
    )
    lut = realtime_dsp.build_display_projection_lut(
        gui_h=151,
        gui_w=151,
        x_max_m=2.0,
        y_max_m=3.0,
        dr_m=dr_m,
        angle_axis_deg=angle_axis,
        projection_mode="cartesian",
        projection_interp="nearest",
    )
    x_axis = np.asarray(lut["x_axis_m"], dtype=np.float32)
    y_axis = np.asarray(lut["y_axis_m"], dtype=np.float32)
    expected_x = target_range * np.sin(np.deg2rad(target_angle))
    expected_y = target_range * np.cos(np.deg2rad(target_angle))

    max_value = float(np.max(view))
    assert max_value > 0.0
    peak_rows, peak_cols = np.where(view >= np.float32(max_value))
    distances = np.hypot(x_axis[peak_cols] - np.float32(expected_x), y_axis[peak_rows] - np.float32(expected_y))
    assert float(np.min(distances)) <= max(float(x_axis[1] - x_axis[0]), float(y_axis[1] - y_axis[0])) + 0.06


def test_projection_lut_accepts_explicit_zoomed_extents() -> None:
    angle_axis = np.asarray([-30.0, 0.0, 30.0], dtype=np.float32)
    lut = realtime_dsp.build_display_projection_lut(
        gui_h=5,
        gui_w=7,
        x_max_m=0.6,
        y_max_m=2.0,
        dr_m=0.1,
        angle_axis_deg=angle_axis,
        projection_mode="cartesian",
        projection_interp="nearest",
        x_min_m=-0.2,
        y_min_m=1.0,
    )
    np.testing.assert_allclose(lut["x_axis_m"][0], np.float32(-0.2 + (0.8 / 7.0) * 0.5), atol=1e-6)
    np.testing.assert_allclose(lut["y_axis_m"][0], np.float32(1.0 + (1.0 / 5.0) * 0.5), atol=1e-6)

    heatmap = np.ones((32, 3), dtype=np.float32)
    view = realtime_dsp.project_heatmap_for_display(
        heatmap,
        angle_axis_deg=angle_axis,
        dr_m=0.1,
        gui_h=5,
        gui_w=7,
        y_max_m=2.0,
        x_max_m=0.6,
        projection_mode="cartesian",
        projection_interp="nearest",
        x_min_m=-0.2,
        y_min_m=1.0,
        precomputed_lut=lut,
    )

    assert view.shape == (5, 7)
    assert np.all(np.isfinite(view))
