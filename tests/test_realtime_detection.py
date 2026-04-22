from __future__ import annotations

import numpy as np

import realtime_dsp


def _dsp_cfg(nfft_range: int = 64, nfft_angle: int = 512, loops: int = 8) -> realtime_dsp.RealtimeDSPConfig:
    return realtime_dsp.RealtimeDSPConfig(
        c=3.0e8,
        fs=1.0,
        slope=1.0,
        samples=nfft_range,
        chirps=loops * 2,
        rx=4,
        tx=2,
        x_frames=1,
        bytes_per_frame=0,
        nfft_range=nfft_range,
        nfft_angle=nfft_angle,
        range_max_display=4.0,
        range_profile_count=8,
        virtual_ant=8,
        fft_workers=1,
        debug_stats=False,
    )


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


def _snapshot(geometry: realtime_dsp.VirtualArrayGeometry, angle_deg: float) -> np.ndarray:
    u_target = np.float32(2.0 * np.sin(np.deg2rad(np.float32(angle_deg))))
    return np.exp((-1j * np.float32(2.0 * np.pi)) * geometry.phase_centers_lambda * u_target).astype(np.complex64)


def _static_range_fft(range_bin: int, angle_deg: float, *, amplitude: float = 100.0) -> np.ndarray:
    geometry = _geometry()
    range_fft = np.zeros((1, 4, 2, 64, 4), dtype=np.complex64)
    range_fft[:, :, :, range_bin, :] = (amplitude * _snapshot(geometry, angle_deg)).reshape(1, 1, 2, 4)
    return range_fft


def test_static_detection_finds_synthetic_target_with_physical_xy() -> None:
    dsp_cfg = _dsp_cfg()
    geometry = _geometry()
    angle_axis = realtime_dsp.build_angle_axis_deg(dsp_cfg.nfft_angle, geometry=geometry)
    steering = realtime_dsp.build_angle_steering_matrix(dsp_cfg.virtual_ant, dsp_cfg.nfft_angle, geometry=geometry)
    range_bin_m = 0.10
    range_bin = 24
    angle_deg = -20.0

    detections, heatmap = realtime_dsp.detect_static_targets(
        _static_range_fft(range_bin, angle_deg),
        static_cfg=realtime_dsp.DetectionConfigStatic(
            threshold_mode="relative",
            threshold_db=-6.0,
            min_power_db=-120.0,
            max_detections=1,
            localmax_range_bins=2,
            localmax_angle_bins=4,
        ),
        angle_cfg=realtime_dsp.AngleProcessingConfig(mode="bartlett"),
        dsp_cfg=dsp_cfg,
        virtual_array_geometry=geometry,
        w_angle=np.ones((1, 1, 1, dsp_cfg.virtual_ant), dtype=np.float32),
        angle_steering=steering,
        angle_axis_deg=angle_axis,
        range_bin_m=range_bin_m,
        max_bin=48,
        apply_angle_window=False,
    )

    assert heatmap.shape == (48, dsp_cfg.nfft_angle)
    assert len(detections) == 1
    det = detections[0]
    assert abs(det.range_bin - range_bin) <= 1
    assert abs(det.angle_deg - angle_deg) <= 1.0
    expected_range = range_bin * range_bin_m
    np.testing.assert_allclose(det.x_m, expected_range * np.sin(np.deg2rad(angle_deg)), atol=0.03)
    np.testing.assert_allclose(det.y_m, expected_range * np.cos(np.deg2rad(angle_deg)), atol=0.03)


def test_static_nms_keeps_stronger_of_nearby_angle_peaks() -> None:
    power_db = np.zeros((8, 12), dtype=np.float32)
    power_db[4, 5] = 20.0
    power_db[4, 7] = 30.0

    peaks = realtime_dsp.extract_detection_peaks_2d(
        power_db,
        threshold_db=1.0,
        win_row=1,
        win_col=3,
        max_peaks=4,
    )

    assert peaks.tolist() == [[4, 7]]


def test_range_doppler_finds_known_bin_and_excludes_zero_doppler() -> None:
    loops = 8
    dsp_cfg = _dsp_cfg(loops=loops)
    range_bin = 13
    doppler_bin = 2
    range_fft = np.zeros((1, loops, 2, 32, 4), dtype=np.complex64)
    phase = np.exp(1j * 2.0 * np.pi * doppler_bin * np.arange(loops, dtype=np.float32) / loops)
    range_fft[:, :, :, range_bin, :] = phase.reshape(1, loops, 1, 1)

    _, rd_map = realtime_dsp.compute_range_doppler(
        range_fft,
        max_bin=24,
        dsp_cfg=dsp_cfg,
        moving_cfg=realtime_dsp.DetectionConfigMoving(
            doppler_fft_shift=False,
            zero_doppler_exclusion_bins=1,
        ),
        w_doppler=np.ones((1, loops, 1, 1, 1), dtype=np.float32),
        apply_doppler_window=False,
    )

    assert rd_map.shape == (24, loops)
    assert int(np.argmax(rd_map[range_bin])) == doppler_bin
    np.testing.assert_allclose(rd_map[:, 0:2], 0.0, atol=1e-6)


def test_doppler_axis_and_tdm_compensation_are_physical_and_finite() -> None:
    dsp_cfg = _dsp_cfg(loops=8)
    axis = realtime_dsp.build_doppler_axis_mps(
        {"radar": {"fc": 77.0e9, "chirp_period_s": 50.0e-6}},
        dsp_cfg,
        8,
        doppler_fft_shift=True,
    )

    assert axis is not None
    assert axis.shape == (8,)
    assert np.all(np.isfinite(axis))
    assert axis[4] == 0.0
    np.testing.assert_allclose(axis[5], -axis[3], rtol=1e-6, atol=1e-6)

    table = realtime_dsp.build_tdm_mimo_doppler_compensation_table(8, 2, doppler_fft_shift=False)
    snapshot = np.ones((1, 2, 4), dtype=np.complex64)
    compensated = realtime_dsp.apply_tdm_mimo_doppler_compensation(snapshot, doppler_bin=2, compensation_table=table)

    np.testing.assert_allclose(compensated[:, 0, :], 1.0 + 0.0j)
    np.testing.assert_allclose(compensated[:, 1, :], table[2, 1])


def test_fusion_merges_overlapping_static_and_moving_and_cleaning_dedups() -> None:
    static = realtime_dsp.Detection(10, 20, None, 1.0, 5.0, None, 0.10, 0.99, 20.0, 13.0, "static")
    moving = realtime_dsp.Detection(10, 21, 6, 1.01, 5.5, 0.4, 0.11, 1.00, 10.0, 10.0, "moving")
    far = realtime_dsp.Detection(30, 22, None, 3.0, -10.0, None, -0.5, 2.9, 5.0, 7.0, "static")
    bad = realtime_dsp.Detection(0, None, None, float("nan"), float("nan"), float("inf"), float("nan"), 0.0, float("nan"), float("nan"), "static")
    cfg = realtime_dsp.FusionConfig(enabled=True, merge_xy_m=0.20, merge_range_m=0.20, merge_angle_deg=2.0)

    fused = realtime_dsp.fuse_detections([static, far], [moving], cfg)
    cleaned = realtime_dsp.clean_detections_for_tracking(fused + [bad, moving], cfg)

    assert len(fused) == 2
    assert fused[0].source == "fused"
    assert fused[0].doppler_mps == 0.4
    assert len(cleaned) == 2
    assert all(np.isfinite([det.x_m, det.y_m, det.range_m, det.angle_deg, det.power_lin, det.power_db]).all() for det in cleaned)
