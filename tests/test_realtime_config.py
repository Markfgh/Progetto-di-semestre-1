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
    velocity_xy_cfg = realtime_dsp.velocity_xy_from_yaml_dict(
        {
            "dsp": {
                "velocity_xy": {
                    "relative_power_floor_db": "nan",
                    "median_power_floor_scale": -1.0,
                    "min_dominance_ratio": 2.0,
                }
            }
        }
    )
    range_angle_moving_cfg = realtime_dsp.range_angle_moving_from_yaml_dict(
        {
            "dsp": {
                "range_angle_moving": {
                    "relative_power_floor_db": "nan",
                    "min_power_db": "nan",
                    "min_dominance_ratio": 2.0,
                    "velocity_dead_zone": "nan",
                }
            }
        }
    )

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
    assert velocity_xy_cfg.relative_power_floor_db == -20.0
    assert velocity_xy_cfg.median_power_floor_scale == 8.0
    assert velocity_xy_cfg.min_dominance_ratio == 1.0
    assert range_angle_moving_cfg.relative_power_floor_db == -12.0
    assert range_angle_moving_cfg.min_power_db == 6.0
    assert range_angle_moving_cfg.min_dominance_ratio == 1.0
    assert range_angle_moving_cfg.velocity_dead_zone == 0.08

    velocity_xy_cfg = realtime_dsp.velocity_xy_from_yaml_dict(
        {
            "dsp": {
                "velocity_xy": {
                    "relative_power_floor_db": -18.0,
                    "median_power_floor_scale": 5.0,
                    "min_dominance_ratio": 0.42,
                }
            }
        }
    )
    assert velocity_xy_cfg.relative_power_floor_db == -18.0
    assert velocity_xy_cfg.median_power_floor_scale == 5.0
    assert velocity_xy_cfg.min_dominance_ratio == 0.42

    range_angle_moving_cfg = realtime_dsp.range_angle_moving_from_yaml_dict(
        {
            "dsp": {
                "range_angle_moving": {
                    "relative_power_floor_db": -18.0,
                    "min_power_db": 2.5,
                    "min_dominance_ratio": 0.42,
                    "velocity_dead_zone": 0.21,
                }
            }
        }
    )
    assert range_angle_moving_cfg.relative_power_floor_db == -18.0
    assert range_angle_moving_cfg.min_power_db == 2.5
    assert range_angle_moving_cfg.min_dominance_ratio == 0.42
    assert range_angle_moving_cfg.velocity_dead_zone == 0.21

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


def test_display_zoom_parser_and_viewport_clamps_are_stable() -> None:
    cfg = {
        "radar": {"c": 3.0e8, "fs": 2.0e6, "slope": 30.0e12},
        "fft": {"nfft_range": 64, "nfft_angle": 128},
        "display": {"range_max": 4.0},
        "display_zoom": {
            "enabled": "true",
            "output_width": -10,
            "output_height": "bad",
            "max_zoom_nfft_range": 16,
            "max_zoom_nfft_angle": 32,
            "max_update_hz": -5.0,
            "dsp_budget_ms": "nan",
            "fallback_mode": "cached_frame",
        },
    }

    zoom_cfg = realtime_dsp.display_zoom_from_yaml_dict(cfg)
    assert zoom_cfg.enabled
    assert zoom_cfg.output_width == 1
    assert zoom_cfg.output_height > 0
    assert zoom_cfg.max_zoom_nfft_range == 64
    assert zoom_cfg.max_zoom_nfft_angle == 128
    assert zoom_cfg.max_update_hz == 15.0
    assert zoom_cfg.dsp_budget_ms == 6.0
    assert zoom_cfg.fallback_mode == "cached_frame"

    home = realtime_dsp.build_display_viewport(
        x_min_m=-2.0,
        x_max_m=2.0,
        y_min_m=0.0,
        y_max_m=4.0,
        dr_m=0.1,
        seq=0,
    )
    clamped = realtime_dsp.clamp_display_viewport(
        x_min_m=-9.0,
        x_max_m=0.2,
        y_min_m=-1.0,
        y_max_m=6.5,
        home_viewport=home,
        output_width=128,
        output_height=64,
        dr_m=0.1,
        seq=3,
    )

    assert clamped.x_min_m >= home.x_min_m
    assert clamped.x_max_m <= home.x_max_m
    assert clamped.y_min_m >= home.y_min_m
    assert clamped.y_max_m <= home.y_max_m
    assert clamped.seq == 3
    assert clamped.zoom_level >= 1.0


def test_display_zoom_method_warnings_are_explicit_for_unsupported_values() -> None:
    warnings = realtime_dsp.display_zoom_method_warnings(
        realtime_dsp.DisplayZoomConfig(
            enabled=True,
            realtime_method="fft_zoom",
            offline_method="adaptive",
        )
    )

    assert any("display_zoom.realtime_method='fft_zoom'" in warning for warning in warnings)
    assert any("display_zoom.offline_method='adaptive'" in warning for warning in warnings)
    assert realtime_dsp.display_zoom_method_warnings(realtime_dsp.DisplayZoomConfig()) == []


def test_display_viewport_clamp_expands_to_cover_requested_roi() -> None:
    home = realtime_dsp.build_display_viewport(
        x_min_m=-2.0,
        x_max_m=2.0,
        y_min_m=0.0,
        y_max_m=3.0,
        dr_m=0.09763672265546891,
        seq=0,
    )

    clamped = realtime_dsp.clamp_display_viewport(
        x_min_m=-0.2,
        x_max_m=0.2,
        y_min_m=0.0,
        y_max_m=0.5,
        home_viewport=home,
        output_width=256,
        output_height=128,
        dr_m=0.09763672265546891,
        seq=1,
    )

    assert clamped.x_min_m <= -0.2
    assert clamped.x_max_m >= 0.2
    assert clamped.y_min_m <= 0.0
    assert clamped.y_max_m >= 0.5
