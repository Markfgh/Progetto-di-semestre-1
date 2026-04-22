from __future__ import annotations

import numpy as np

import realtime_dsp


def _dsp_cfg() -> realtime_dsp.RealtimeDSPConfig:
    return realtime_dsp.RealtimeDSPConfig(
        c=3.0e8,
        fs=2.0e6,
        slope=30.0e12,
        samples=16,
        chirps=8,
        rx=4,
        tx=2,
        x_frames=1,
        bytes_per_frame=0,
        nfft_range=32,
        nfft_angle=64,
        range_max_display=4.0,
        range_profile_count=8,
        virtual_ant=8,
        fft_workers=1,
        debug_stats=False,
    )


def test_config_parsers_sanitize_invalid_values_and_legacy_aliases() -> None:
    cfg = {
        "dsp": {
            "window_range": "not-a-window",
            "window_doppler": "rectangular",
            "angle_processing": {
                "mode": "mvdr",
                "mvdr_diagonal_loading": "-2.0",
                "aggregation": "bad",
                "frame_index": "-9",
            },
            "display_filters": {
                "slow_time": {"enabled": True, "mode": "doppler_fft"},
                "mean_after_range_fft": {"enabled": True, "axes": ["sample", "loop", "loop"]},
            },
            "detection_filters": {
                "background_subtraction": {"enabled": True, "clamp_positive_only": True},
                "loop_average_after_background": {"enabled": True},
            },
            "detection_moving_pre_doppler_filters": {
                "slow_time": {"enabled": True, "mode": "doppler_fft", "doppler_zero_notch": True},
                "loop_average_after_background": {"enabled": True},
            },
        },
        "detection_static": {
            "threshold_mode": "bad",
            "localmax_range_bins": -4,
            "max_detections": 0,
        },
        "detection_moving": {
            "threshold_mode": "os_cfar",
            "zero_doppler_exclusion_bins": -3,
            "max_detections": "0",
        },
        "fusion": {
            "merge_xy_m": "-1",
            "merge_range_m": "nan",
        },
        "tracking": {
            "frame_dt_s": "0.04",
            "max_tracks": 0,
            "min_confirmed_hits": 2,
            "max_missed_frames": 4,
            "model": "unsupported",
            "gating_xy_m": "-2",
        },
    }

    selection = realtime_dsp.selection_from_yaml_dict(cfg)
    assert selection.window_range == "blackman"
    assert selection.window_doppler == "rectangular"

    angle = realtime_dsp.angle_processing_from_yaml_dict(cfg)
    assert angle.mode == "mvdr"
    assert angle.mvdr_diagonal_loading == 0.0
    assert angle.aggregation == "frame_loop"
    assert angle.frame_index == 0

    static_cfg = realtime_dsp.detection_static_from_yaml_dict(cfg)
    moving_cfg = realtime_dsp.detection_moving_from_yaml_dict(cfg)
    fusion_cfg = realtime_dsp.fusion_from_yaml_dict(cfg)
    tracking_cfg = realtime_dsp.tracking_from_yaml_dict(cfg)
    tracker_cfg = realtime_dsp.tracker_from_yaml_dict(cfg)

    assert static_cfg.threshold_mode == "relative"
    assert static_cfg.localmax_range_bins == 0
    assert static_cfg.max_detections == 1
    assert moving_cfg.threshold_mode == "os_cfar"
    assert moving_cfg.zero_doppler_exclusion_bins == 0
    assert moving_cfg.max_detections == 1
    assert fusion_cfg.merge_xy_m == 0.0
    assert fusion_cfg.merge_range_m == 0.0
    assert tracking_cfg.dt_s == 0.04
    assert tracking_cfg.max_tracks == 1
    assert tracking_cfg.min_hits_to_confirm == 2
    assert tracking_cfg.max_missed_confirmed == 4
    assert tracker_cfg.model == "kalman_cv_2d"
    assert tracker_cfg.gating_xy_m == 0.0

    moving_filters = realtime_dsp.detection_moving_pre_doppler_filters_from_yaml_dict(cfg)
    moving_filters, moving_warnings = realtime_dsp.sanitize_detection_moving_pre_doppler_filters(moving_filters)
    assert not moving_filters.slow_time.enabled
    assert not moving_filters.loop_average_after_background.enabled
    assert any("unsupported" in warning for warning in moving_warnings)

    static_filters = realtime_dsp.detection_static_post_range_fft_filters_from_yaml_dict(cfg)
    static_filters, static_warnings = realtime_dsp.sanitize_detection_static_post_range_fft_filters(static_filters)
    assert static_filters.loop_average_after_background.enabled
    assert not static_filters.background_subtraction.clamp_positive_only
    assert any("display-only" in warning for warning in static_warnings)

    display_filters = realtime_dsp.display_post_range_fft_filters_from_yaml_dict(cfg)
    display_filters, display_warnings = realtime_dsp.sanitize_display_post_range_fft_filters(display_filters)
    assert display_filters.mean_after_range_fft.axes == ("range_bin",)
    assert any("removing 'loop'" in warning for warning in display_warnings)

    assert realtime_dsp.cfar_numba_from_yaml_dict({}) == realtime_dsp.CfarNumbaConfig()
    assert realtime_dsp.dsp_diagnostics_from_yaml_dict({}) == realtime_dsp.DspDiagnosticsConfig()
    cfar_numba = realtime_dsp.cfar_numba_from_yaml_dict(
        {"dsp": {"cfar_numba": {"enabled": "true", "warmup_on_start": "yes", "self_check_on_start": "on"}}}
    )
    assert cfar_numba.enabled
    assert cfar_numba.warmup_on_start
    assert cfar_numba.self_check_on_start


def test_virtual_array_geometry_falls_back_on_invalid_user_geometry() -> None:
    geometry, warnings = realtime_dsp.build_virtual_array_geometry_from_yaml_dict(
        {
            "antenna": {
                "virtual_array_order": [0, 0, 2],
                "virtual_array_phase_centers_lambda": [0.0, np.nan],
                "angle_axis_sign": 0,
            }
        },
        _dsp_cfg(),
    )

    np.testing.assert_array_equal(geometry.order_flat, np.arange(8, dtype=np.int32))
    np.testing.assert_allclose(geometry.phase_centers_lambda, np.arange(8, dtype=np.float32) * 0.25)
    assert geometry.angle_axis_sign == 1.0
    assert len(warnings) == 3


def test_window_shapes_and_identity_windows_are_stable() -> None:
    w_range, w_doppler, w_angle = realtime_dsp.build_windows(
        realtime_dsp.DspSelection("none", "hamming", "blackman"),
        samples=5,
        n_loops=4,
        virtual_ant=3,
    )

    assert w_range.shape == (1, 1, 1, 5, 1)
    assert w_doppler.shape == (1, 4, 1, 1, 1)
    assert w_angle.shape == (1, 1, 1, 3)
    np.testing.assert_array_equal(w_range, np.ones_like(w_range))
    assert np.all(np.isfinite(w_doppler))
    assert np.all(np.isfinite(w_angle))
