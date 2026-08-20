"""Test dei parser di configurazione e delle invarianti della GUI realtime."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import radar_app
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


def test_retained_positive_range_bins_includes_last_bin_centre_and_caps_half_fft() -> None:
    current_dr = 3.0e8 * 10.0e6 / (2.0 * 39.01e12 * 512)
    assert realtime_dsp.retained_positive_range_bins(18.0, current_dr, 512) == 240
    assert realtime_dsp.retained_positive_range_bins(0.0, 0.1, 512) == 1
    assert realtime_dsp.retained_positive_range_bins(100.0, 0.1, 512) == 256


@pytest.mark.parametrize("invalid", [True, 512.5])
def test_realtime_configuration_rejects_non_integer_capture_sizes(invalid: object) -> None:
    config = radar_app.yaml.safe_load(radar_app.CFG_PATH.read_text(encoding="utf-8"))
    config["capture"]["samples"] = invalid
    with pytest.raises(ValueError, match="capture.samples must be a positive integer"):
        radar_app.validate_realtime_configuration(config)


def test_nested_filter_booleans_and_nonfinite_coefficients_are_strict() -> None:
    filters = realtime_dsp.display_post_range_fft_filters_from_yaml_dict(
        {"dsp": {"display_filters": {"mean_after_range_fft": {"enabled": "false"}}}}
    )
    assert not filters.mean_after_range_fft.enabled

    with pytest.raises(ValueError, match="heatmap_ema.alpha"):
        realtime_dsp.heatmap_ema_from_yaml_dict({"dsp": {"heatmap_ema": {"alpha": "nan"}}})
    with pytest.raises(ValueError, match="slow_time.highpass_beta"):
        realtime_dsp.display_post_range_fft_filters_from_yaml_dict(
            {"dsp": {"display_filters": {"slow_time": {"highpass_beta": "nan"}}}}
        )
    with pytest.raises(ValueError, match="background_subtraction.alpha"):
        realtime_dsp.display_post_range_fft_filters_from_yaml_dict(
            {"dsp": {"display_filters": {"background_subtraction": {"alpha": "inf"}}}}
        )
    with pytest.raises(ValueError, match="mvdr_diagonal_loading"):
        realtime_dsp.angle_processing_from_yaml_dict(
            {"dsp": {"angle_processing": {"mvdr_diagonal_loading": "inf"}}}
        )


EXPECTED_TUNING_PATHS = {
    "dsp.window_range",
    "dsp.window_doppler",
    "dsp.window_angle",
    "dsp.zero_after_range_fft_bins",
    "dsp.range_angle_moving.relative_power_floor_db",
    "dsp.range_angle_moving.min_power_db",
    "dsp.range_angle_moving.min_dominance_ratio",
    "dsp.range_angle_moving.velocity_dead_zone",
    "dsp.range_angle_moving.min_opacity",
    "dsp.display_filters.background_subtraction.enabled",
    "dsp.display_filters.background_subtraction.mode",
    "dsp.display_filters.background_subtraction.alpha",
    "dsp.display_filters.background_subtraction.init_frames",
    "dsp.display_filters.background_subtraction.window_frames",
    "dsp.display_filters.background_subtraction.clamp_positive_only",
    "dsp.display_filters.slow_time.enabled",
    "dsp.display_filters.slow_time.mode",
    "dsp.display_filters.slow_time.highpass_beta",
    "dsp.display_filters.mean_after_range_fft.enabled",
    "dsp.display_filters.loop_average_after_background.enabled",
    "dsp.angle_processing.mode",
    "dsp.angle_processing.mvdr_diagonal_loading",
    "dsp.angle_processing.aggregation",
    "dsp.heatmap_ema.enabled",
    "dsp.heatmap_ema.alpha",
    "dsp.heatmap_spatial_filter.enabled",
    "dsp.heatmap_spatial_filter.mode",
    "tracking.enabled",
    "tracking.max_tracks",
    "tracking.min_hits_to_confirm",
    "tracking.max_missed_tentative",
    "tracking.max_missed_confirmed",
    "tracking.max_track_age",
    "tracking.gating_xy_m",
    "tracking.gating_doppler_mps",
    "tracking.birth_min_separation_m",
    "tracking.use_doppler_in_cost",
    "tracking.process_noise_pos",
    "tracking.process_noise_vel",
    "tracking.measurement_noise_xy",
    "tracking.moving_speed_threshold_mps",
    "tracking.stopped_speed_threshold_mps",
    "tracking.doppler_moving_threshold_mps",
    "tracking.motion_confirm_frames_moving",
    "tracking.motion_confirm_frames_stopped",
    "tracking.stopped_memory_s",
    "tracking.stopped_resume_gate_m",
    "tracking.stop_position_alpha",
    "detection_static.enabled",
    "detection_static.threshold_mode",
    "detection_static.threshold_db",
    "detection_static.min_power_db",
    "detection_static.max_detections",
    "detection_static.localmax_range_bins",
    "detection_static.localmax_angle_bins",
    "detection_static.cfar_train_range_bins",
    "detection_static.cfar_guard_range_bins",
    "detection_static.cfar_train_col_bins",
    "detection_static.cfar_guard_col_bins",
    "detection_static.cfar_threshold_db",
    "detection_static.os_cfar_rank",
    "detection_moving.enabled",
    "detection_moving.threshold_mode",
    "detection_moving.threshold_db",
    "detection_moving.min_power_db",
    "detection_moving.max_detections",
    "detection_moving.localmax_range_bins",
    "detection_moving.localmax_doppler_bins",
    "detection_moving.zero_doppler_exclusion_bins",
    "detection_moving.cfar_train_range_bins",
    "detection_moving.cfar_guard_range_bins",
    "detection_moving.cfar_train_col_bins",
    "detection_moving.cfar_guard_col_bins",
    "detection_moving.cfar_threshold_db",
    "detection_moving.os_cfar_rank",
    "dsp.detection_moving_pre_doppler_filters.slow_time.enabled",
    "dsp.detection_moving_pre_doppler_filters.slow_time.mode",
    "dsp.detection_moving_pre_doppler_filters.slow_time.highpass_beta",
    "fusion.enabled",
    "fusion.merge_xy_m",
    "fusion.merge_range_m",
    "fusion.merge_angle_deg",
    "fusion.prefer_moving_when_doppler_valid",
}


def _extract_tuning_paths_from_gui_source() -> set[str]:
    source = Path("radar_app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "TUNING_FIELD_SPECS" for target in node.targets):
            continue
        if not isinstance(node.value, ast.List):
            raise AssertionError("TUNING_FIELD_SPECS deve essere una lista")
        paths: set[str] = set()
        for spec in node.value.elts:
            if not isinstance(spec, ast.Dict):
                continue
            for key, value in zip(spec.keys, spec.values):
                if isinstance(key, ast.Constant) and key.value == "path" and isinstance(value, ast.Constant):
                    paths.add(str(value.value))
        return paths
    raise AssertionError("TUNING_FIELD_SPECS non trovato in radar_app.py")


def test_tuning_gui_paths_are_applied_by_runtime_config_parsers() -> None:
    assert _extract_tuning_paths_from_gui_source() == EXPECTED_TUNING_PATHS

    cfg = {
        "dsp": {
            "window_range": "hamming",
            "window_doppler": "blackman",
            "window_angle": "rectangular",
            "zero_after_range_fft_bins": 3,
            "range_angle_moving": {
                "relative_power_floor_db": -17.5,
                "min_power_db": 4.25,
                "min_dominance_ratio": 0.55,
                "velocity_dead_zone": 0.12,
                "min_opacity": 0.35,
            },
            "display_filters": {
                "background_subtraction": {
                    "enabled": False,
                    "mode": "window_mean",
                    "alpha": 0.33,
                    "init_frames": 7,
                    "window_frames": 9,
                    "clamp_positive_only": True,
                },
                "slow_time": {
                    "enabled": True,
                    "mode": "highpass",
                    "highpass_beta": 0.77,
                },
                "mean_after_range_fft": {"enabled": True},
                "loop_average_after_background": {"enabled": True},
            },
            "angle_processing": {
                "mode": "mvdr",
                "mvdr_diagonal_loading": 0.04,
                "aggregation": "loop",
            },
            "heatmap_ema": {"enabled": True, "alpha": 0.27},
            "heatmap_spatial_filter": {"enabled": True, "mode": "gaussian_3x3"},
            "detection_moving_pre_doppler_filters": {
                "slow_time": {
                    "enabled": True,
                    "mode": "highpass",
                    "highpass_beta": 0.66,
                }
            },
        },
        "tracking": {
            "enabled": False,
            "max_tracks": 6,
            "min_hits_to_confirm": 4,
            "max_missed_tentative": 5,
            "max_missed_confirmed": 11,
            "max_track_age": 13,
            "gating_xy_m": 1.2,
            "gating_doppler_mps": 0.45,
            "birth_min_separation_m": 0.62,
            "use_doppler_in_cost": False,
            "process_noise_pos": 0.44,
            "process_noise_vel": 0.88,
            "measurement_noise_xy": 0.19,
            "moving_speed_threshold_mps": 0.31,
            "stopped_speed_threshold_mps": 0.09,
            "doppler_moving_threshold_mps": 0.21,
            "motion_confirm_frames_moving": 3,
            "motion_confirm_frames_stopped": 5,
            "stopped_memory_s": 7.5,
            "stopped_resume_gate_m": 0.71,
            "stop_position_alpha": 0.41,
        },
        "detection_static": {
            "enabled": False,
            "threshold_mode": "ca_cfar",
            "threshold_db": -8.0,
            "min_power_db": 3.0,
            "max_detections": 7,
            "localmax_range_bins": 2,
            "localmax_angle_bins": 3,
            "cfar_train_range_bins": 9,
            "cfar_guard_range_bins": 4,
            "cfar_train_col_bins": 10,
            "cfar_guard_col_bins": 5,
            "cfar_threshold_db": 11.0,
            "os_cfar_rank": 6,
        },
        "detection_moving": {
            "enabled": False,
            "threshold_mode": "os_cfar",
            "threshold_db": -6.0,
            "min_power_db": 4.0,
            "max_detections": 8,
            "localmax_range_bins": 3,
            "localmax_doppler_bins": 4,
            "zero_doppler_exclusion_bins": 2,
            "cfar_train_range_bins": 11,
            "cfar_guard_range_bins": 5,
            "cfar_train_col_bins": 6,
            "cfar_guard_col_bins": 2,
            "cfar_threshold_db": 12.5,
            "os_cfar_rank": 7,
        },
        "fusion": {
            "enabled": False,
            "merge_xy_m": 0.61,
            "merge_range_m": 0.42,
            "merge_angle_deg": 9.5,
            "prefer_moving_when_doppler_valid": True,
        },
    }

    selection = realtime_dsp.selection_from_yaml_dict(cfg)
    assert selection == realtime_dsp.DspSelection("hamming", "blackman", "rectangular")
    assert int((cfg["dsp"] or {})["zero_after_range_fft_bins"]) == 3

    range_angle = realtime_dsp.range_angle_moving_from_yaml_dict(cfg)
    assert range_angle.relative_power_floor_db == -17.5
    assert range_angle.min_power_db == 4.25
    assert range_angle.min_dominance_ratio == 0.55
    assert range_angle.velocity_dead_zone == 0.12
    assert range_angle.min_opacity == 0.35

    display_filters = realtime_dsp.display_post_range_fft_filters_from_yaml_dict(cfg)
    assert not display_filters.background_subtraction.enabled
    assert display_filters.background_subtraction.mode == "window_mean"
    assert display_filters.background_subtraction.alpha == 0.33
    assert display_filters.background_subtraction.init_frames == 7
    assert display_filters.background_subtraction.window_frames == 9
    assert display_filters.background_subtraction.clamp_positive_only
    assert display_filters.slow_time.enabled
    assert display_filters.slow_time.mode == "highpass"
    assert display_filters.slow_time.highpass_beta == 0.77
    assert display_filters.mean_after_range_fft.enabled
    assert display_filters.loop_average_after_background.enabled

    angle = realtime_dsp.angle_processing_from_yaml_dict(cfg)
    assert angle.mode == "mvdr"
    assert angle.mvdr_diagonal_loading == 0.04
    assert angle.aggregation == "loop"

    assert realtime_dsp.heatmap_ema_from_yaml_dict(cfg) == realtime_dsp.HeatmapEMAConfig(True, 0.27)
    assert realtime_dsp.heatmap_spatial_filter_from_yaml_dict(cfg) == realtime_dsp.HeatmapSpatialFilterConfig(
        True,
        "gaussian_3x3",
    )

    moving_filters = realtime_dsp.detection_moving_pre_doppler_filters_from_yaml_dict(cfg)
    assert moving_filters.slow_time.enabled
    assert moving_filters.slow_time.mode == "highpass"
    assert moving_filters.slow_time.highpass_beta == 0.66

    tracking = realtime_dsp.tracking_from_yaml_dict(cfg)
    tracker = realtime_dsp.tracker_from_yaml_dict(cfg)
    assert not tracking.enabled
    assert tracking.max_tracks == 6
    assert tracking.min_hits_to_confirm == 4
    assert tracking.max_missed_tentative == 5
    assert tracking.max_missed_confirmed == 11
    assert tracking.max_track_age == 13
    assert tracker.gating_xy_m == 1.2
    assert tracker.gating_doppler_mps == 0.45
    assert tracker.birth_min_separation_m == 0.62
    assert not tracker.use_doppler_in_cost
    assert tracker.process_noise_pos == 0.44
    assert tracker.process_noise_vel == 0.88
    assert tracker.measurement_noise_xy == 0.19
    assert tracker.moving_speed_threshold_mps == 0.31
    assert tracker.stopped_speed_threshold_mps == 0.09
    assert tracker.doppler_moving_threshold_mps == 0.21
    assert tracker.motion_confirm_frames_moving == 3
    assert tracker.motion_confirm_frames_stopped == 5
    assert tracker.stopped_memory_s == 7.5
    assert tracker.stopped_resume_gate_m == 0.71
    assert tracker.stop_position_alpha == 0.41

    static = realtime_dsp.detection_static_from_yaml_dict(cfg)
    assert not static.enabled
    assert static.threshold_mode == "ca_cfar"
    assert static.threshold_db == -8.0
    assert static.min_power_db == 3.0
    assert static.max_detections == 7
    assert static.localmax_range_bins == 2
    assert static.localmax_angle_bins == 3
    assert static.cfar_train_range_bins == 9
    assert static.cfar_guard_range_bins == 4
    assert static.cfar_train_col_bins == 10
    assert static.cfar_guard_col_bins == 5
    assert static.cfar_threshold_db == 11.0
    assert static.os_cfar_rank == 6

    moving = realtime_dsp.detection_moving_from_yaml_dict(cfg)
    assert not moving.enabled
    assert moving.threshold_mode == "os_cfar"
    assert moving.threshold_db == -6.0
    assert moving.min_power_db == 4.0
    assert moving.max_detections == 8
    assert moving.localmax_range_bins == 3
    assert moving.localmax_doppler_bins == 4
    assert moving.zero_doppler_exclusion_bins == 2
    assert moving.cfar_train_range_bins == 11
    assert moving.cfar_guard_range_bins == 5
    assert moving.cfar_train_col_bins == 6
    assert moving.cfar_guard_col_bins == 2
    assert moving.cfar_threshold_db == 12.5
    assert moving.os_cfar_rank == 7

    fusion = realtime_dsp.fusion_from_yaml_dict(cfg)
    assert not fusion.enabled
    assert fusion.merge_xy_m == 0.61
    assert fusion.merge_range_m == 0.42
    assert fusion.merge_angle_deg == 9.5
    assert fusion.prefer_moving_when_doppler_valid


def test_runtime_config_patch_merges_nested_blocks_without_dropping_siblings() -> None:
    base = {
        "dsp": {
            "window_range": "hanning",
            "display_filters": {
                "background_subtraction": {
                    "enabled": True,
                    "mode": "frozen",
                    "alpha": 0.02,
                },
                "slow_time": {
                    "enabled": False,
                    "mode": "none",
                },
            },
        },
        "detection_static": {
            "threshold_mode": "relative",
            "threshold_db": -10.0,
        },
    }
    patch = {
        "dsp": {
            "display_filters": {
                "background_subtraction": {
                    "alpha": 0.15,
                },
            },
        },
        "detection_static": {
            "threshold_db": -7.0,
        },
    }

    merged = realtime_dsp._deep_merge_dict(base, patch)

    assert merged["dsp"]["window_range"] == "hanning"
    assert merged["dsp"]["display_filters"]["background_subtraction"]["enabled"] is True
    assert merged["dsp"]["display_filters"]["background_subtraction"]["mode"] == "frozen"
    assert merged["dsp"]["display_filters"]["background_subtraction"]["alpha"] == 0.15
    assert merged["dsp"]["display_filters"]["slow_time"]["mode"] == "none"
    assert merged["detection_static"]["threshold_mode"] == "relative"
    assert merged["detection_static"]["threshold_db"] == -7.0
    assert base["dsp"]["display_filters"]["background_subtraction"]["alpha"] == 0.02


def test_config_parsers_sanitize_invalid_values() -> None:
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
            "detection_static_filters": {
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
            "dt_s": "0.04",
            "max_tracks": 0,
            "min_hits_to_confirm": 2,
            "max_missed_confirmed": 4,
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
    assert range_angle_moving_cfg.relative_power_floor_db == -12.0
    assert range_angle_moving_cfg.min_power_db == 6.0
    assert range_angle_moving_cfg.min_dominance_ratio == 1.0
    assert range_angle_moving_cfg.velocity_dead_zone == 0.08

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
    assert static_filters.background_subtraction.clamp_positive_only
    assert not any("display-only" in warning for warning in static_warnings)

    display_filters = realtime_dsp.display_post_range_fft_filters_from_yaml_dict(cfg)
    display_filters, display_warnings = realtime_dsp.sanitize_display_post_range_fft_filters(display_filters)
    assert display_filters.mean_after_range_fft.axes == ("range_bin",)
    assert any("removing 'loop'" in warning for warning in display_warnings)

    assert realtime_dsp.cfar_numba_from_yaml_dict({}) == realtime_dsp.CfarNumbaConfig()
    assert realtime_dsp.angle_power_numba_from_yaml_dict({}) == realtime_dsp.AnglePowerNumbaConfig()
    assert realtime_dsp.dsp_diagnostics_from_yaml_dict({}) == realtime_dsp.DspDiagnosticsConfig()
    cfar_numba = realtime_dsp.cfar_numba_from_yaml_dict(
        {"dsp": {"cfar_numba": {"enabled": "true", "warmup_on_start": "yes", "self_check_on_start": "on"}}}
    )
    assert cfar_numba.enabled
    assert cfar_numba.warmup_on_start
    assert cfar_numba.self_check_on_start

    angle_power_numba = realtime_dsp.angle_power_numba_from_yaml_dict(
        {"dsp": {"angle_power_numba": {"enabled": "yes", "threads": "4"}}}
    )
    assert angle_power_numba == realtime_dsp.AnglePowerNumbaConfig(enabled=True, threads=4)
    assert realtime_dsp.angle_power_numba_from_yaml_dict(
        {"dsp": {"angle_power_numba": {"threads": -6}}}
    ).threads == 0


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
            "max_zoom_nfft_range": 16,
            "max_zoom_nfft_angle": 32,
            "max_update_hz": -5.0,
            "dsp_budget_ms": "nan",
            "fallback_mode": "cached_frame",
        },
    }

    zoom_cfg = realtime_dsp.display_zoom_from_yaml_dict(cfg)
    assert zoom_cfg.enabled
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


def test_display_image_resolutions_are_independent_and_validated() -> None:
    base_cfg = {
        "fft": {"nfft_range": 32768, "nfft_angle": 64},
        "display": {"range_max": 4.0},
    }
    defaults = realtime_dsp.display_image_resolutions_from_yaml_dict(base_cfg)
    assert defaults.realtime == realtime_dsp.DisplayImageResolution(width=128, height=128)
    assert defaults.offline == realtime_dsp.DisplayImageResolution(width=128, height=128)

    explicit_cfg = {
        **base_cfg,
        "display": {
            "range_max": 4.0,
            "image_resolution": {
                "realtime": {"width": 256, "height": 128},
                "offline": {"width": 512, "height": 256},
            },
        },
    }
    explicit = realtime_dsp.display_image_resolutions_from_yaml_dict(explicit_cfg)
    assert explicit.realtime == realtime_dsp.DisplayImageResolution(width=256, height=128)
    assert explicit.offline == realtime_dsp.DisplayImageResolution(width=512, height=256)

    invalid = realtime_dsp.display_image_resolutions_from_yaml_dict(
        {
            **base_cfg,
            "display": {
                "range_max": 4.0,
                "image_resolution": {"realtime": {"width": -7, "height": "not-a-number"}},
            },
        }
    )
    assert invalid.realtime == realtime_dsp.DisplayImageResolution(width=1, height=128)
    assert invalid.offline == realtime_dsp.DisplayImageResolution(width=128, height=128)


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
