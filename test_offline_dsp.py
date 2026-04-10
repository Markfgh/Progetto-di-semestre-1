from __future__ import annotations

import numpy as np

from offline_dsp import (
    _interp_cubic_complex,
    _interp_linear_complex,
    back_projection_image_mimo,
    build_mimo_geometry,
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
    n_bins = 256
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

    x_axis = np.linspace(-0.6, 0.6, 161, dtype=np.float32)
    y_axis = np.linspace(0.8, 2.8, 161, dtype=np.float32)
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


def test_tdm_compensation_static_target_survives_zero_bin() -> None:
    n_pos, n_frames, n_loops, n_tx, n_rx, n_bins, target_bin = 1, 1, 32, 2, 4, 64, 18

    raw = np.zeros((n_pos, n_frames, n_loops, n_tx, n_rx, n_bins), dtype=np.complex64)
    raw[:, :, :, :, :, target_bin] = np.complex64(1.0 + 0.0j)

    zero_bin = prepare_mimo_snapshots(raw, n_tx=n_tx, motion_mode="static_zero_doppler")
    all_bins = prepare_mimo_snapshots(raw, n_tx=n_tx, motion_mode="all_doppler_incoherent")

    zero_energy = float(np.sum(np.abs(zero_bin[:, :, :, target_bin]) ** 2))
    total_energy = float(np.sum(np.abs(all_bins[:, :, :, :, target_bin]) ** 2))
    assert zero_energy / max(total_energy, 1e-12) >= 0.85


def test_tdm_compensation_moving_target_attenuated_in_zero_bin() -> None:
    n_pos, n_frames, n_loops, n_tx, n_rx, n_bins, target_bin = 1, 1, 32, 2, 4, 64, 18
    c_m_s = np.float32(3e8)
    fc_hz = np.float32(77e9)
    lambda_m = c_m_s / fc_hz
    chirp_period_s = np.float32(160e-6)
    velocity_m_s = np.float32(0.5)

    raw = np.zeros((n_pos, n_frames, n_loops, n_tx, n_rx, n_bins), dtype=np.complex64)
    loop_idx = np.arange(n_loops, dtype=np.float32)
    for tx_i in range(n_tx):
        t_chirp = (loop_idx * np.float32(n_tx) + np.float32(tx_i)) * chirp_period_s
        phase = np.exp(
            1j * np.float32(2.0 * np.pi)
            * (np.float32(2.0) * velocity_m_s / lambda_m)
            * t_chirp
        ).astype(np.complex64, copy=False)
        raw[:, :, :, tx_i, :, target_bin] = phase.reshape(1, 1, n_loops, 1)

    zero_bin = prepare_mimo_snapshots(raw, n_tx=n_tx, motion_mode="static_zero_doppler")
    all_bins = prepare_mimo_snapshots(raw, n_tx=n_tx, motion_mode="all_doppler_incoherent")

    zero_energy = float(np.sum(np.abs(zero_bin[:, :, :, target_bin]) ** 2))
    doppler_energy = np.sum(np.abs(all_bins[:, :, :, :, target_bin]) ** np.float32(2.0), axis=(0, 1, 3), dtype=np.float32)
    correct_doppler_energy = float(np.max(doppler_energy))
    zero_db = 10.0 * np.log10(max(zero_energy, 1e-12))
    correct_db = 10.0 * np.log10(max(correct_doppler_energy, 1e-12))
    assert zero_db <= correct_db - 15.0

    total_doppler = float(np.sum(doppler_energy))
    assert correct_doppler_energy / max(total_doppler, 1e-12) >= 0.70


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
