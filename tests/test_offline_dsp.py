from __future__ import annotations

import numpy as np
import pytest

from offline_dsp import (
    _interp_cubic_complex,
    _interp_linear_complex,
    build_mimo_back_projection_plan,
    back_projection_power_mimo_frames,
    back_projection_power_mimo_snapshot,
    back_projection_image_mimo,
    build_mimo_geometry,
    motion_mode_normalize,
    prepare_synthetic_aperture_data,
    prepare_mimo_snapshots,
)


def _target_profile(n_bins: int, bin_float: float, phase: complex) -> np.ndarray:
    bins = np.arange(int(n_bins), dtype=np.float32)
    return (np.sinc(bins - np.float32(bin_float)).astype(np.complex64) * np.complex64(phase)).astype(np.complex64, copy=False)


def _target_cell(x_axis: np.ndarray, y_axis: np.ndarray, target_x: float, target_y: float) -> tuple[int, int]:
    col = int(np.argmin(np.abs(np.asarray(x_axis, dtype=np.float32) - np.float32(target_x))))
    row = int(np.argmin(np.abs(np.asarray(y_axis, dtype=np.float32) - np.float32(target_y))))
    return row, col


def _physical_iwr1443_geometry(lambda_m: np.float32) -> tuple[np.ndarray, np.ndarray]:
    tx_base = (np.asarray([0.0, 2.0], dtype=np.float32) * lambda_m).astype(np.float32, copy=False)
    rx_base = (np.asarray([0.0, 0.5, 1.0, 1.5], dtype=np.float32) * lambda_m).astype(np.float32, copy=False)
    x_tx_phys = np.repeat(tx_base, 4).astype(np.float32, copy=False)
    x_rx_phys = np.tile(rx_base, 2).astype(np.float32, copy=False)
    center_m = (
        np.float32(0.5)
        * np.mean((x_tx_phys + x_rx_phys).astype(np.float32, copy=False), dtype=np.float32)
    ).astype(np.float32, copy=False)
    x_tx_phys = (x_tx_phys - center_m).astype(np.float32, copy=False)
    x_rx_phys = (x_rx_phys - center_m).astype(np.float32, copy=False)
    return x_tx_phys, x_rx_phys


def test_default_geometry_matches_ula() -> None:
    x_tx, x_rx = build_mimo_geometry(2, 4, fc_hz=77e9, c_m_s=3e8)
    lambda_m = np.float32(3e8 / 77e9)
    expected_pair_sums = (np.arange(8, dtype=np.float32) - np.float32(3.5)) * np.float32(0.5) * lambda_m
    phase_centers = (np.float32(0.5) * (x_tx + x_rx)).astype(np.float32, copy=False)
    expected_phase_centers = (np.float32(0.5) * expected_pair_sums).astype(np.float32, copy=False)
    np.testing.assert_allclose(x_tx + x_rx, expected_pair_sums, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(phase_centers, expected_phase_centers, atol=1e-6, rtol=0.0)


def test_bistatic_bp_point_target() -> None:
    c_m_s = np.float32(3e8)
    fc_hz = np.float32(77e9)
    dr_m = np.float32(0.02)
    n_bins = 128
    lambda_m = (c_m_s / fc_hz).astype(np.float32, copy=False)
    x_tx_phys, x_rx_phys = _physical_iwr1443_geometry(lambda_m)
    x_tx_ant_m, x_rx_ant_m = build_mimo_geometry(2, 4, fc_hz=float(fc_hz), c_m_s=float(c_m_s))
    np.testing.assert_allclose(x_tx_ant_m, x_tx_phys, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(x_rx_ant_m, x_rx_phys, atol=1e-6, rtol=0.0)

    x_pos_m = np.linspace(-0.14, 0.14, 13, dtype=np.float32)
    target_x = np.float32(0.3)
    target_y = np.float32(2.0)
    k_bistatic = np.float32((2.0 * np.pi * float(fc_hz)) / float(c_m_s))

    snapshots = np.zeros((int(x_pos_m.size), 1, 8, n_bins), dtype=np.complex64)
    for pos_i, xpos in enumerate(x_pos_m):
        for ant_i in range(8):
            r_tx = np.hypot(target_x - (xpos + x_tx_phys[ant_i]), target_y).astype(np.float32, copy=False)
            r_rx = np.hypot(target_x - (xpos + x_rx_phys[ant_i]), target_y).astype(np.float32, copy=False)
            r_total = (r_tx + r_rx).astype(np.float32, copy=False)
            r_eq = (np.float32(0.5) * r_total).astype(np.float32, copy=False)
            snapshots[pos_i, 0, ant_i, :] = _target_profile(
                n_bins=n_bins,
                bin_float=float(r_eq / dr_m),
                phase=np.exp(-1j * np.float32(k_bistatic * r_total)),
            )

    x_axis = np.linspace(-0.6, 0.6, 256, dtype=np.float32)
    y_axis = np.linspace(0.8, 2.8, 256, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    img_db = back_projection_image_mimo(
        snapshots,
        x_pos_m,
        x_tx_ant_m,
        x_rx_ant_m,
        x_grid,
        y_grid,
        dr_m=float(dr_m),
        fc_hz=float(fc_hz),
        c_m_s=float(c_m_s),
        max_bin=n_bins,
        motion_mode="static_zero_doppler",
        phase_sign=1,
    )

    peak_row, peak_col = np.unravel_index(int(np.argmax(img_db)), img_db.shape)
    target_row, target_col = _target_cell(x_axis, y_axis, float(target_x), float(target_y))
    assert abs(int(peak_row) - int(target_row)) <= 2
    assert abs(int(peak_col) - int(target_col)) <= 2
    assert float(img_db[target_row, target_col]) >= float(np.mean(img_db, dtype=np.float32)) + 6.0


def test_bistatic_bp_point_target_at_30deg() -> None:
    c_m_s = np.float32(3e8)
    fc_hz = np.float32(77e9)
    dr_m = np.float32(0.02)
    n_bins = 256
    lambda_m = (c_m_s / fc_hz).astype(np.float32, copy=False)
    x_tx_phys, x_rx_phys = _physical_iwr1443_geometry(lambda_m)
    x_tx_ant_m, x_rx_ant_m = build_mimo_geometry(2, 4, fc_hz=float(fc_hz), c_m_s=float(c_m_s))

    target_range = np.float32(2.0)
    target_angle = np.float32(30.0)
    target_x = (target_range * np.sin(np.deg2rad(target_angle))).astype(np.float32, copy=False)
    target_y = (target_range * np.cos(np.deg2rad(target_angle))).astype(np.float32, copy=False)
    x_pos_m = np.linspace(-0.30, 0.30, 31, dtype=np.float32)
    k_bistatic = np.float32((2.0 * np.pi * float(fc_hz)) / float(c_m_s))

    snapshots = np.zeros((int(x_pos_m.size), 1, 8, n_bins), dtype=np.complex64)
    for pos_i, xpos in enumerate(x_pos_m):
        for ant_i in range(8):
            r_tx = np.hypot(target_x - (xpos + x_tx_phys[ant_i]), target_y).astype(np.float32, copy=False)
            r_rx = np.hypot(target_x - (xpos + x_rx_phys[ant_i]), target_y).astype(np.float32, copy=False)
            r_total = (r_tx + r_rx).astype(np.float32, copy=False)
            r_eq = (np.float32(0.5) * r_total).astype(np.float32, copy=False)
            snapshots[pos_i, 0, ant_i, :] = _target_profile(
                n_bins=n_bins,
                bin_float=float(r_eq / dr_m),
                phase=np.exp(-1j * np.float32(k_bistatic * r_total)),
            )

    x_axis = np.linspace(0.4, 1.6, 121, dtype=np.float32)
    y_axis = np.linspace(1.2, 2.2, 121, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    img_db = back_projection_image_mimo(
        snapshots,
        x_pos_m,
        x_tx_ant_m,
        x_rx_ant_m,
        x_grid,
        y_grid,
        dr_m=float(dr_m),
        fc_hz=float(fc_hz),
        c_m_s=float(c_m_s),
        max_bin=n_bins,
        motion_mode="static_zero_doppler",
        phase_sign=1,
    )

    peak_row, peak_col = np.unravel_index(int(np.argmax(img_db)), img_db.shape)
    target_row, target_col = _target_cell(x_axis, y_axis, float(target_x), float(target_y))
    assert abs(int(peak_row) - int(target_row)) <= 3
    assert abs(int(peak_col) - int(target_col)) <= 3
    assert float(img_db[target_row, target_col]) >= float(np.mean(img_db, dtype=np.float32)) + 6.0


def test_incoherent_mimo_bp_sums_power_not_amplitude() -> None:
    snapshots = np.zeros((1, 2, 8), dtype=np.complex64)
    snapshots[0, 0, 3] = np.complex64(3.0 + 0.0j)
    snapshots[0, 1, 3] = np.complex64(0.0 + 4.0j)

    img_power = back_projection_power_mimo_snapshot(
        snapshots,
        np.asarray([0.0], dtype=np.float32),
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([[0.0]], dtype=np.float32),
        np.asarray([[3.0]], dtype=np.float32),
        dr_m=1.0,
        fc_hz=77e9,
        c_m_s=3e8,
        max_bin=8,
        coherent_sum=False,
    )

    np.testing.assert_allclose(img_power, np.asarray([[25.0]], dtype=np.float32), atol=1e-5, rtol=0.0)


def test_mimo_bp_plan_matches_direct_path() -> None:
    snapshots = np.zeros((2, 2, 16), dtype=np.complex64)
    snapshots[0, 0, 4] = np.complex64(1.0 + 2.0j)
    snapshots[0, 1, 5] = np.complex64(0.5 - 0.25j)
    snapshots[1, 0, 6] = np.complex64(2.0 - 1.0j)
    snapshots[1, 1, 4] = np.complex64(-0.75 + 0.5j)

    x_pos_m = np.asarray([-0.02, 0.02], dtype=np.float32)
    x_tx_ant_m = np.asarray([0.0, 0.01], dtype=np.float32)
    x_rx_ant_m = np.asarray([0.0, 0.015], dtype=np.float32)
    x_axis = np.linspace(-0.05, 0.05, 7, dtype=np.float32)
    y_axis = np.linspace(0.15, 0.45, 9, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)

    plan = build_mimo_back_projection_plan(
        x_pos_m,
        x_tx_ant_m,
        x_rx_ant_m,
        x_grid,
        y_grid,
        dr_m=0.05,
        fc_hz=77e9,
        c_m_s=3e8,
        max_bin=16,
        phase_sign=-1,
    )
    direct = back_projection_power_mimo_snapshot(
        snapshots,
        x_pos_m,
        x_tx_ant_m,
        x_rx_ant_m,
        x_grid,
        y_grid,
        dr_m=0.05,
        fc_hz=77e9,
        c_m_s=3e8,
        max_bin=16,
        phase_sign=-1,
        coherent_sum=True,
    )
    planned = back_projection_power_mimo_snapshot(
        snapshots,
        x_pos_m,
        x_tx_ant_m,
        x_rx_ant_m,
        x_grid,
        y_grid,
        dr_m=0.05,
        fc_hz=77e9,
        c_m_s=3e8,
        max_bin=16,
        phase_sign=-1,
        coherent_sum=True,
        bp_plan=plan,
    )

    np.testing.assert_allclose(planned, direct, atol=1e-5, rtol=1e-5)


def test_motion_mode_normalize_accepts_static_zero_doppler() -> None:
    assert motion_mode_normalize("static_zero_doppler") == "static_zero_doppler"


def test_motion_mode_normalize_rejects_legacy_all_doppler_incoherent() -> None:
    with pytest.raises(ValueError, match="static_zero_doppler"):
        motion_mode_normalize("all_doppler_incoherent")


def test_prepare_mimo_snapshots_returns_4d_and_uses_all_virtual_antennas() -> None:
    n_pos, n_frames, n_loops, n_tx, n_rx, n_bins = 2, 3, 8, 2, 4, 16
    raw = np.zeros((n_pos, n_frames, n_loops, n_tx, n_rx, n_bins), dtype=np.complex64)

    prepared = prepare_mimo_snapshots(raw, n_tx=n_tx, motion_mode="static_zero_doppler")

    assert prepared.shape == (n_pos, n_frames, n_tx * n_rx, n_bins)


@pytest.mark.parametrize("window_name", ["rectangular", "hanning"])
def test_prepare_mimo_snapshots_matches_explicit_zero_doppler_fft(window_name: str) -> None:
    rng = np.random.default_rng(4242)
    shape = (2, 3, 16, 2, 4, 11)
    raw = (
        rng.standard_normal(shape, dtype=np.float32)
        + 1j * rng.standard_normal(shape, dtype=np.float32)
    ).astype(np.complex64)
    window = (
        np.ones(shape[2], dtype=np.float32)
        if window_name == "rectangular"
        else np.hanning(shape[2]).astype(np.float32)
    )

    explicit = np.fft.fft(
        raw * window.reshape(1, 1, shape[2], 1, 1, 1),
        axis=2,
    ).astype(np.complex64, copy=False)[:, :, 0]
    explicit = explicit.reshape(shape[0], shape[1], shape[3] * shape[4], shape[5])
    prepared = prepare_mimo_snapshots(
        raw,
        n_tx=shape[3],
        motion_mode="static_zero_doppler",
        window_doppler=window,
    )

    np.testing.assert_allclose(prepared, explicit, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("coherent_sum", [True, False])
@pytest.mark.parametrize("phase_sign", [-1, 1])
def test_multi_frame_backprojection_matches_legacy_frame_loop(
    coherent_sum: bool,
    phase_sign: int,
) -> None:
    rng = np.random.default_rng(909)
    snapshots = (
        rng.standard_normal((3, 4, 2, 24), dtype=np.float32)
        + 1j * rng.standard_normal((3, 4, 2, 24), dtype=np.float32)
    ).astype(np.complex64)
    x_pos_m = np.asarray([-0.03, 0.0, 0.03], dtype=np.float32)
    x_tx_ant_m = np.asarray([-0.004, 0.004], dtype=np.float32)
    x_rx_ant_m = np.asarray([-0.002, 0.002], dtype=np.float32)
    x_grid, y_grid = np.meshgrid(
        np.linspace(-0.08, 0.08, 9, dtype=np.float32),
        np.linspace(0.2, 0.7, 11, dtype=np.float32),
    )

    legacy = np.zeros_like(x_grid, dtype=np.float32)
    for frame_i in range(snapshots.shape[1]):
        legacy += back_projection_power_mimo_snapshot(
            snapshots[:, frame_i],
            x_pos_m,
            x_tx_ant_m,
            x_rx_ant_m,
            x_grid,
            y_grid,
            dr_m=0.04,
            fc_hz=77e9,
            c_m_s=3e8,
            max_bin=24,
            phase_sign=phase_sign,
            coherent_sum=coherent_sum,
        )
    optimized = back_projection_power_mimo_frames(
        snapshots,
        x_pos_m,
        x_tx_ant_m,
        x_rx_ant_m,
        x_grid,
        y_grid,
        dr_m=0.04,
        fc_hz=77e9,
        c_m_s=3e8,
        max_bin=24,
        phase_sign=phase_sign,
        coherent_sum=coherent_sum,
    )

    np.testing.assert_allclose(optimized, legacy, atol=3e-4, rtol=2e-5)


def test_prepare_synthetic_aperture_data_flattens_selected_positions_and_antennas() -> None:
    zero_doppler = np.arange(4 * 2 * 8 * 3, dtype=np.float32).reshape(4, 2, 8, 3).astype(np.complex64)
    selected_positions = np.asarray([0, 2, 5, 9], dtype=np.int32)
    x_tx_ant_m, x_rx_ant_m = build_mimo_geometry(2, 4, fc_hz=77e9, c_m_s=3e8)

    synthetic = prepare_synthetic_aperture_data(
        zero_doppler,
        selected_positions=selected_positions,
        x_pitch_m=0.1,
        x_tx_ant_m=x_tx_ant_m,
        x_rx_ant_m=x_rx_ant_m,
    )

    expected_cube = np.transpose(zero_doppler, (1, 3, 0, 2)).reshape(2, 1, 3, 32)
    expected_phase_centers = (np.float32(0.5) * (x_tx_ant_m + x_rx_ant_m)).astype(np.float32, copy=False)
    expected_positions_m = selected_positions.astype(np.float32) * np.float32(0.1)
    expected_x = (expected_positions_m[:, None] + expected_phase_centers[None, :]).reshape(-1)

    assert synthetic.snapshot_cube.shape == (2, 1, 3, 32)
    np.testing.assert_allclose(synthetic.snapshot_cube, expected_cube, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(synthetic.x_position_m, expected_positions_m, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(synthetic.x_element_m, expected_x, atol=1e-6, rtol=0.0)


def test_prepare_synthetic_aperture_data_preserves_noncontiguous_physical_spacing() -> None:
    zero_doppler = np.zeros((2, 1, 8, 2), dtype=np.complex64)
    selected_positions = np.asarray([1, 4], dtype=np.int32)
    x_tx_ant_m, x_rx_ant_m = build_mimo_geometry(2, 4, fc_hz=77e9, c_m_s=3e8)

    synthetic = prepare_synthetic_aperture_data(
        zero_doppler,
        selected_positions=selected_positions,
        x_pitch_m=0.05,
        x_tx_ant_m=x_tx_ant_m,
        x_rx_ant_m=x_rx_ant_m,
    )

    x_by_pos = synthetic.x_element_m.reshape(2, 8)
    np.testing.assert_allclose(
        x_by_pos[1, 0] - x_by_pos[0, 0],
        np.float32(0.15),
        atol=1e-6,
        rtol=0.0,
    )


def test_tdm_compensation_static_target_survives_zero_bin() -> None:
    n_pos, n_frames, n_loops, n_tx, n_rx, n_bins, target_bin = 1, 1, 32, 2, 4, 64, 18

    raw = np.zeros((n_pos, n_frames, n_loops, n_tx, n_rx, n_bins), dtype=np.complex64)
    raw[:, :, :, :, :, target_bin] = np.complex64(1.0 + 0.0j)

    zero_bin = prepare_mimo_snapshots(raw, n_tx=n_tx, motion_mode="static_zero_doppler")
    assert zero_bin.shape == (n_pos, n_frames, n_tx * n_rx, n_bins)

    zero_energy = float(np.sum(np.abs(zero_bin[:, :, :, target_bin]) ** 2, dtype=np.float32))
    expected_energy = float((n_tx * n_rx) * (n_loops**2))
    assert zero_energy >= 0.99 * expected_energy


def test_tdm_compensation_moving_target_attenuated_in_zero_bin() -> None:
    n_pos, n_frames, n_loops, n_tx, n_rx, n_bins, target_bin = 1, 1, 32, 2, 4, 64, 18
    target_doppler_bin = 3
    doppler_cycles = np.float32(target_doppler_bin / n_loops)

    raw = np.zeros((n_pos, n_frames, n_loops, n_tx, n_rx, n_bins), dtype=np.complex64)
    loop_idx = np.arange(n_loops, dtype=np.float32)
    for tx_i in range(n_tx):
        phase = np.exp(
            1j * np.float32(2.0 * np.pi)
            * doppler_cycles
            * (loop_idx + (np.float32(tx_i) / np.float32(n_tx)))
        ).astype(np.complex64, copy=False)
        raw[:, :, :, tx_i, :, target_bin] = phase.reshape(1, 1, n_loops, 1)

    zero_bin = prepare_mimo_snapshots(raw, n_tx=n_tx, motion_mode="static_zero_doppler")

    zero_energy = float(np.sum(np.abs(zero_bin[:, :, :, target_bin]) ** np.float32(2.0), dtype=np.float32))
    expected_peak_energy = float((n_tx * n_rx) * (n_loops**2))
    zero_db = 10.0 * np.log10(max(zero_energy, 1e-12))
    correct_db = 10.0 * np.log10(max(expected_peak_energy, 1e-12))
    assert zero_db <= correct_db - 15.0


def test_back_projection_image_mimo_rejects_5d_input() -> None:
    range_fft_sel = np.zeros((2, 1, 3, 8, 16), dtype=np.complex64)
    x_pos_m = np.asarray([-0.01, 0.01], dtype=np.float32)
    x_tx_ant_m = np.zeros(8, dtype=np.float32)
    x_rx_ant_m = np.zeros(8, dtype=np.float32)
    x_grid = np.zeros((4, 4), dtype=np.float32)
    y_grid = np.ones((4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="atteso \\[pos, frame, ant, bin\\]"):
        back_projection_image_mimo(
            range_fft_sel,
            x_pos_m,
            x_tx_ant_m,
            x_rx_ant_m,
            x_grid,
            y_grid,
            dr_m=0.05,
            fc_hz=77e9,
            c_m_s=3e8,
            max_bin=16,
            motion_mode="static_zero_doppler",
        )


def test_cubic_vs_linear_interpolation() -> None:
    n_bins = 128
    base_bin = 27.0
    fractions = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
    omega = np.float32(0.21)
    bins = np.arange(n_bins, dtype=np.float32)
    spec = np.exp(1j * omega * bins).astype(np.complex64, copy=False)

    err_linear_sq = np.float32(0.0)
    err_cubic_sq = np.float32(0.0)
    for frac in fractions:
        bin_float = np.float32(base_bin) + frac
        truth = np.exp(1j * omega * bin_float).astype(np.complex64, copy=False)
        linear = _interp_linear_complex(spec, np.asarray([bin_float], dtype=np.float32), n_bins)[0]
        cubic = _interp_cubic_complex(spec, np.asarray([bin_float], dtype=np.float32), n_bins)[0]
        err_linear_sq += np.float32(np.abs(linear - truth) ** np.float32(2.0))
        err_cubic_sq += np.float32(np.abs(cubic - truth) ** np.float32(2.0))

    rms_linear = np.sqrt(err_linear_sq / np.float32(fractions.size)).astype(np.float32, copy=False)
    rms_cubic = np.sqrt(err_cubic_sq / np.float32(fractions.size)).astype(np.float32, copy=False)
    assert float(rms_cubic) <= float(rms_linear / np.float32(3.0))
