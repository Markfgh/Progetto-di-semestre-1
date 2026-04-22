from __future__ import annotations

import numpy as np
import pytest

import realtime_dsp


def test_relative_threshold_matches_legacy_scalar_behavior() -> None:
    power_db = np.asarray([[0.0, 4.0], [8.0, 2.0]], dtype=np.float32)

    threshold = realtime_dsp.compute_detection_threshold_db(
        power_db,
        threshold_mode="relative",
        threshold_db=-3.0,
        min_power_db=1.0,
    )

    assert threshold == 5.0


def test_ca_cfar_detects_isolated_target_and_rejects_edges() -> None:
    power = np.ones((9, 9), dtype=np.float32)
    power[4, 4] = np.float32(100.0)

    mask, threshold_map = realtime_dsp.compute_cfar_candidate_mask(
        power,
        threshold_mode="ca_cfar",
        train_range_bins=1,
        guard_range_bins=1,
        train_col_bins=1,
        guard_col_bins=1,
        threshold_offset_db=6.0,
        min_power_db=-120.0,
        os_cfar_rank=0,
    )

    assert bool(mask[4, 4])
    assert not bool(mask[0, 0])
    assert not np.isfinite(float(threshold_map[0, 0]))


def test_ca_cfar_numba_self_check_when_available() -> None:
    status = realtime_dsp.cfar_numba_runtime_status()
    if not status["available"]:
        pytest.skip("Numba is optional and not installed")

    try:
        realtime_dsp.configure_cfar_numba_runtime(
            realtime_dsp.CfarNumbaConfig(enabled=True, warmup_on_start=True, self_check_on_start=True),
            log=False,
        )
        status = realtime_dsp.cfar_numba_runtime_status()
        assert status["enabled"]
        assert status["self_checked"]
    finally:
        realtime_dsp.configure_cfar_numba_runtime(realtime_dsp.CfarNumbaConfig(enabled=False), log=False)


def test_os_cfar_uses_rank_so_training_outlier_does_not_hide_target() -> None:
    power = np.full((7, 7), np.float32(10.0), dtype=np.float32)
    power[3, 3] = np.float32(80.0)
    power[2, 2] = np.float32(1000.0)

    ca_mask, _ = realtime_dsp.compute_cfar_candidate_mask(
        power,
        threshold_mode="ca_cfar",
        train_range_bins=1,
        guard_range_bins=0,
        train_col_bins=1,
        guard_col_bins=0,
        threshold_offset_db=0.0,
        min_power_db=-120.0,
        os_cfar_rank=0,
    )
    os_mask, _ = realtime_dsp.compute_cfar_candidate_mask(
        power,
        threshold_mode="os_cfar",
        train_range_bins=1,
        guard_range_bins=0,
        train_col_bins=1,
        guard_col_bins=0,
        threshold_offset_db=0.0,
        min_power_db=-120.0,
        os_cfar_rank=4,
    )

    assert not bool(ca_mask[3, 3])
    assert bool(os_mask[3, 3])


def test_cfar_mask_flows_through_localmax_and_max_detections() -> None:
    power_db = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 30.0, 20.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    mask = np.zeros(power_db.shape, dtype=bool)
    mask[1, 1] = True
    mask[1, 2] = True

    peaks = realtime_dsp.extract_detection_peaks_2d(
        power_db,
        threshold_db=999.0,
        win_row=0,
        win_col=0,
        max_peaks=1,
        candidates_mask=mask,
    )

    assert peaks.tolist() == [[1, 1]]


def test_moving_ca_cfar_range_doppler_detection() -> None:
    rd_map = np.ones((9, 9), dtype=np.float32)
    rd_map[4, 5] = np.float32(100.0)
    doppler_axis = np.linspace(-2.0, 2.0, 9, dtype=np.float32)
    cfg = realtime_dsp.DetectionConfigMoving(
        threshold_mode="ca_cfar",
        localmax_range_bins=1,
        localmax_doppler_bins=1,
        min_power_db=-120.0,
        max_detections=4,
        cfar_train_range_bins=1,
        cfar_guard_range_bins=1,
        cfar_train_col_bins=1,
        cfar_guard_col_bins=1,
        cfar_threshold_db=6.0,
    )

    detections = realtime_dsp.detect_moving_targets(
        rd_map,
        moving_cfg=cfg,
        range_bin_m=0.25,
        doppler_axis_mps=doppler_axis,
    )

    assert len(detections) == 1
    assert detections[0].range_bin == 4
    assert detections[0].doppler_bin == 5
    assert detections[0].doppler_mps == float(doppler_axis[5])
