"""Verifica numericamente la back-projection SAR e le sue varianti MIMO."""

from __future__ import annotations

import numpy as np
import pytest

from offline_dsp import (
    _interp_cubic_complex,
    _interp_linear_complex,
    back_projection_power_mimo_geometry,
    back_projection_power_mimo_frames,
    back_projection_image_mimo,
    build_mimo_geometry,
    prepare_synthetic_aperture_data,
    prepare_mimo_snapshots,
    residual_video_phase_sign_normalize,
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


def _linear_world_geometry(
    x_pos_m: np.ndarray,
    x_tx_ant_m: np.ndarray,
    x_rx_ant_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_pos = int(x_pos_m.size)
    n_ant = int(x_tx_ant_m.size)
    tx_global = np.zeros((n_pos, n_ant, 3), dtype=np.float32)
    rx_global = np.zeros((n_pos, n_ant, 3), dtype=np.float32)
    tx_global[..., 0] = x_pos_m[:, None] + x_tx_ant_m[None, :]
    rx_global[..., 0] = x_pos_m[:, None] + x_rx_ant_m[None, :]
    return tx_global, rx_global


def _plane_voxel_xyz(x_grid: np.ndarray, y_grid: np.ndarray, z_m: float = 0.0) -> np.ndarray:
    voxel_xyz = np.empty(x_grid.shape + (3,), dtype=np.float32)
    voxel_xyz[..., 0] = x_grid
    voxel_xyz[..., 1] = y_grid
    voxel_xyz[..., 2] = np.float32(z_m)
    return voxel_xyz


def _legacy_linear_bp_reference(
    snapshots: np.ndarray,
    x_pos_m: np.ndarray,
    x_tx_ant_m: np.ndarray,
    x_rx_ant_m: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    dr_m: float,
    fc_hz: float,
    c_m_s: float,
    max_bin: int,
    phase_sign: int,
    chunk_size: int,
) -> np.ndarray:
    """Pre-generalization linear BP, retained only as a regression oracle."""
    n_pos, n_frames, n_ant, n_bins_avail = (int(v) for v in snapshots.shape)
    max_bin_eff = max(0, min(int(max_bin), n_bins_avail))
    if max_bin_eff < 2 or n_frames <= 0:
        return np.zeros_like(x_grid, dtype=np.float32)

    x_flat = x_grid.reshape(-1).astype(np.float32, copy=False)
    y_flat = y_grid.reshape(-1).astype(np.float32, copy=False)
    y_sq = (y_flat * y_flat).astype(np.float32, copy=False)
    phase_scale = np.float32(float(phase_sign)) * np.float32(
        (2.0 * np.pi * float(fc_hz)) / float(c_m_s)
    )
    inv_dr = np.float32(1.0 / float(dr_m))
    frame_acc = np.zeros((n_frames, int(x_flat.size)), dtype=np.complex64)

    for pos_i in range(n_pos):
        x_pos = np.float32(x_pos_m[pos_i])
        for ant_i in range(n_ant):
            x_tx = x_pos + np.float32(x_tx_ant_m[ant_i])
            x_rx = x_pos + np.float32(x_rx_ant_m[ant_i])
            dx_tx = x_flat - x_tx
            dx_rx = x_flat - x_rx
            r_tx = np.sqrt(dx_tx * dx_tx + y_sq).astype(np.float32, copy=False)
            r_rx = np.sqrt(dx_rx * dx_rx + y_sq).astype(np.float32, copy=False)
            r_total = (r_tx + r_rx).astype(np.float32, copy=False)
            bins = (np.float32(0.5) * r_total * inv_dr).astype(np.float32, copy=False)
            valid_idx = np.flatnonzero(
                np.isfinite(bins)
                & (bins >= 0.0)
                & (bins < np.float32(max_bin_eff - 1))
            )
            for start in range(0, int(valid_idx.size), max(1, int(chunk_size))):
                idx = valid_idx[start : start + max(1, int(chunk_size))]
                if idx.size == 0:
                    continue
                phase = np.exp(1j * (phase_scale * r_total[idx])).astype(np.complex64, copy=False)
                for frame_i in range(n_frames):
                    interp = _interp_cubic_complex(snapshots[pos_i, frame_i, ant_i], bins[idx], max_bin_eff)
                    frame_acc[frame_i, idx] += interp * phase

    total_power = np.sum(
        frame_acc.real.astype(np.float32, copy=False) ** np.float32(2.0)
        + frame_acc.imag.astype(np.float32, copy=False) ** np.float32(2.0),
        axis=0,
        dtype=np.float32,
    )
    return total_power.reshape(x_grid.shape).astype(np.float32, copy=False)


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
        phase_sign=1,
    )

    peak_row, peak_col = np.unravel_index(int(np.argmax(img_db)), img_db.shape)
    target_row, target_col = _target_cell(x_axis, y_axis, float(target_x), float(target_y))
    assert abs(int(peak_row) - int(target_row)) <= 3
    assert abs(int(peak_col) - int(target_col)) <= 3
    assert float(img_db[target_row, target_col]) >= float(np.mean(img_db, dtype=np.float32)) + 6.0


@pytest.mark.parametrize("phase_sign", [-1, 1])
def test_geometry_bp_matches_pre_generalization_linear_reference(phase_sign: int) -> None:
    """The legacy adapter must retain the original X-only reconstruction."""
    rng = np.random.default_rng(739)
    snapshots = (
        rng.standard_normal((3, 2, 3, 32), dtype=np.float32)
        + 1j * rng.standard_normal((3, 2, 3, 32), dtype=np.float32)
    ).astype(np.complex64)
    x_pos_m = np.asarray([-0.07, 0.01, 0.09], dtype=np.float32)
    x_tx_ant_m = np.asarray([-0.006, 0.0, 0.006], dtype=np.float32)
    x_rx_ant_m = np.asarray([-0.004, 0.002, 0.008], dtype=np.float32)
    x_grid, y_grid = np.meshgrid(
        np.linspace(-0.12, 0.14, 17, dtype=np.float32),
        np.linspace(0.18, 0.72, 19, dtype=np.float32),
    )
    kwargs = {
        "dr_m": 0.04,
        "fc_hz": 77e9,
        "c_m_s": 3e8,
        "max_bin": 32,
        "phase_sign": phase_sign,
        "chunk_size": 37,
    }

    reference = _legacy_linear_bp_reference(
        snapshots,
        x_pos_m,
        x_tx_ant_m,
        x_rx_ant_m,
        x_grid,
        y_grid,
        **kwargs,
    )
    adapter_reference = back_projection_power_mimo_frames(
        snapshots,
        x_pos_m,
        x_tx_ant_m,
        x_rx_ant_m,
        x_grid,
        y_grid,
        use_numba=False,
        **kwargs,
    )
    adapter_accelerated = back_projection_power_mimo_frames(
        snapshots,
        x_pos_m,
        x_tx_ant_m,
        x_rx_ant_m,
        x_grid,
        y_grid,
        use_numba=True,
        **kwargs,
    )
    tx_global, rx_global = _linear_world_geometry(x_pos_m, x_tx_ant_m, x_rx_ant_m)
    core = back_projection_power_mimo_geometry(
        snapshots,
        tx_global,
        rx_global,
        _plane_voxel_xyz(x_grid, y_grid),
        **kwargs,
    )

    # ``use_numba=False`` is the preserved pre-Numba reference path.  The
    # parallel calculation follows the same formula; only last-bit float32
    # rounding may differ because scalar libm is used for the phase.
    np.testing.assert_array_equal(adapter_reference, reference)
    np.testing.assert_allclose(adapter_accelerated, reference, atol=3e-4, rtol=2e-5)
    assert np.unravel_index(
        int(np.argmax(adapter_accelerated)), adapter_accelerated.shape
    ) == np.unravel_index(int(np.argmax(reference)), reference.shape)
    np.testing.assert_allclose(core, reference, atol=3e-4, rtol=2e-5)


def test_geometry_bp_numba_matches_numpy_reference_with_frames_weights_and_rvp() -> None:
    """The accelerated physical BP must retain the pre-Numba calculation."""
    import offline_dsp

    if offline_dsp._back_projection_power_mimo_geometry_numba_kernel is None:
        pytest.skip("Numba non installato")

    rng = np.random.default_rng(20260722)
    snapshots = (
        rng.standard_normal((4, 3, 3, 40), dtype=np.float32)
        + 1j * rng.standard_normal((4, 3, 3, 40), dtype=np.float32)
    ).astype(np.complex64)
    x_pos_m = np.asarray([-0.08, -0.02, 0.04, 0.10], dtype=np.float32)
    x_tx_ant_m = np.asarray([-0.006, 0.0, 0.006], dtype=np.float32)
    x_rx_ant_m = np.asarray([-0.004, 0.002, 0.008], dtype=np.float32)
    tx_global, rx_global = _linear_world_geometry(x_pos_m, x_tx_ant_m, x_rx_ant_m)
    x_grid, y_grid = np.meshgrid(
        np.linspace(-0.16, 0.18, 23, dtype=np.float32),
        np.linspace(0.24, 0.88, 29, dtype=np.float32),
    )
    kwargs = {
        "dr_m": 0.04,
        "fc_hz": 77e9,
        "c_m_s": 3e8,
        "max_bin": 40,
        "phase_sign": -1,
        "residual_video_phase": "+",
        "slope_hz_s": 6e13,
        "chunk_size": 47,
        "pose_weights": np.asarray([1.0, 0.8, 0.6, 0.9], dtype=np.float32),
        "channel_weights": np.asarray([0.7, 1.0, 0.5], dtype=np.float32),
    }

    reference = back_projection_power_mimo_geometry(
        snapshots,
        tx_global,
        rx_global,
        _plane_voxel_xyz(x_grid, y_grid),
        use_numba=False,
        **kwargs,
    )
    accelerated = back_projection_power_mimo_geometry(
        snapshots,
        tx_global,
        rx_global,
        _plane_voxel_xyz(x_grid, y_grid),
        use_numba=True,
        **kwargs,
    )

    np.testing.assert_allclose(accelerated, reference, atol=3e-4, rtol=2e-5)
    assert np.unravel_index(int(np.argmax(accelerated)), accelerated.shape) == np.unravel_index(
        int(np.argmax(reference)), reference.shape
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [("off", 0), ("+", 1), ("-", -1), (0, 0), (1, 1), (-1, -1)],
)
def test_residual_video_phase_sign_normalize(value: object, expected: int) -> None:
    assert residual_video_phase_sign_normalize(value) == expected


def test_residual_video_phase_refocuses_matching_fmcw_snapshots() -> None:
    """The selected RVP sign must restore coherent summation across the aperture."""
    c_m_s = 3e8
    fc_hz = 77e9
    # Deliberately magnified to make this a robust unit test.  The production
    # slope is much lower, so its expected improvement is subtler.
    slope_hz_s = 5e15
    dr_m = 0.1
    x_pos_m = np.linspace(-0.775, 0.775, 201, dtype=np.float32)
    x_tx_ant_m = np.zeros(1, dtype=np.float32)
    x_rx_ant_m = np.zeros(1, dtype=np.float32)
    target_x = np.float32(15.0)
    target_y = np.float32(30.0)
    n_bins = 512
    snapshots = np.zeros((x_pos_m.size, 1, 1, n_bins), dtype=np.complex64)
    carrier_scale = (2.0 * np.pi * fc_hz) / c_m_s
    rvp_scale = np.pi * slope_hz_s / (c_m_s * c_m_s)
    for pos_i, x_pos in enumerate(x_pos_m):
        r_total = 2.0 * np.hypot(float(target_x - x_pos), float(target_y))
        range_bin = 0.5 * r_total / dr_m
        snapshots[pos_i, 0, 0] = _target_profile(
            n_bins,
            range_bin,
            np.exp(-1j * (carrier_scale * r_total + rvp_scale * r_total * r_total)),
        )

    x_grid = np.asarray([[target_x]], dtype=np.float32)
    y_grid = np.asarray([[target_y]], dtype=np.float32)
    common = dict(
        dr_m=dr_m,
        fc_hz=fc_hz,
        c_m_s=c_m_s,
        max_bin=n_bins,
        phase_sign=1,
        slope_hz_s=slope_hz_s,
    )
    uncorrected = back_projection_power_mimo_frames(
        snapshots,
        x_pos_m,
        x_tx_ant_m,
        x_rx_ant_m,
        x_grid,
        y_grid,
        residual_video_phase="off",
        **common,
    )
    corrected = back_projection_power_mimo_frames(
        snapshots,
        x_pos_m,
        x_tx_ant_m,
        x_rx_ant_m,
        x_grid,
        y_grid,
        residual_video_phase="+",
        **common,
    )

    assert float(corrected[0, 0]) > float(uncorrected[0, 0]) * 20.0


def test_geometry_bp_uniform_default_weights_match_explicit_ones() -> None:
    rng = np.random.default_rng(209)
    snapshots = (
        rng.standard_normal((2, 1, 3, 28), dtype=np.float32)
        + 1j * rng.standard_normal((2, 1, 3, 28), dtype=np.float32)
    ).astype(np.complex64)
    x_pos_m = np.asarray([-0.04, 0.04], dtype=np.float32)
    x_tx_ant_m = np.asarray([-0.003, 0.0, 0.003], dtype=np.float32)
    x_rx_ant_m = np.asarray([-0.002, 0.001, 0.004], dtype=np.float32)
    tx_global, rx_global = _linear_world_geometry(x_pos_m, x_tx_ant_m, x_rx_ant_m)
    x_grid, y_grid = np.meshgrid(
        np.linspace(-0.1, 0.1, 15, dtype=np.float32),
        np.linspace(0.15, 0.6, 13, dtype=np.float32),
    )
    voxel_xyz = _plane_voxel_xyz(x_grid, y_grid)
    kwargs = {
        "dr_m": 0.04,
        "fc_hz": 77e9,
        "c_m_s": 3e8,
        "max_bin": 28,
        "phase_sign": -1,
        "chunk_size": 41,
    }

    default_weights = back_projection_power_mimo_geometry(
        snapshots,
        tx_global,
        rx_global,
        voxel_xyz,
        **kwargs,
    )
    explicit_uniform_weights = back_projection_power_mimo_geometry(
        snapshots,
        tx_global,
        rx_global,
        voxel_xyz,
        pose_weights=np.ones(snapshots.shape[0], dtype=np.float32),
        channel_weights=np.ones(snapshots.shape[2], dtype=np.float32),
        **kwargs,
    )

    np.testing.assert_array_equal(default_weights, explicit_uniform_weights)


def test_prepare_mimo_snapshots_returns_4d_and_uses_all_virtual_antennas() -> None:
    n_pos, n_frames, n_loops, n_tx, n_rx, n_bins = 2, 3, 8, 2, 4, 16
    raw = np.zeros((n_pos, n_frames, n_loops, n_tx, n_rx, n_bins), dtype=np.complex64)

    prepared = prepare_mimo_snapshots(raw, n_tx=n_tx)

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
        window_doppler=window,
    )

    np.testing.assert_allclose(prepared, explicit, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("phase_sign", [-1, 1])
def test_multi_frame_backprojection_matches_single_frame_batches(
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

    frame_batches = np.zeros_like(x_grid, dtype=np.float32)
    for frame_i in range(snapshots.shape[1]):
        frame_batches += back_projection_power_mimo_frames(
            snapshots[:, frame_i : frame_i + 1],
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
    )

    np.testing.assert_allclose(optimized, frame_batches, atol=3e-4, rtol=2e-5)


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


def test_prepare_synthetic_aperture_data_prefers_measured_stage_coordinates() -> None:
    zero_doppler = np.arange(2 * 2 * 8 * 3, dtype=np.float32).reshape(2, 2, 8, 3).astype(
        np.complex64
    )
    selected_positions = np.asarray([1, 4], dtype=np.int32)
    measured_positions_m = np.asarray([-0.012, 0.0175], dtype=np.float32)
    x_tx_ant_m, x_rx_ant_m = build_mimo_geometry(2, 4, fc_hz=77e9, c_m_s=3e8)

    synthetic = prepare_synthetic_aperture_data(
        zero_doppler,
        selected_positions=selected_positions,
        x_pitch_m=0.05,
        x_tx_ant_m=x_tx_ant_m,
        x_rx_ant_m=x_rx_ant_m,
        measured_positions_m=measured_positions_m,
    )

    np.testing.assert_allclose(synthetic.x_position_m, measured_positions_m, atol=0.0, rtol=0.0)
    assert not np.shares_memory(synthetic.snapshot_cube, zero_doppler)
    expected_cube = np.transpose(zero_doppler, (1, 3, 0, 2)).reshape(2, 1, 3, 16)
    np.testing.assert_array_equal(synthetic.snapshot_cube, expected_cube)


def test_tdm_compensation_static_target_survives_zero_bin() -> None:
    n_pos, n_frames, n_loops, n_tx, n_rx, n_bins, target_bin = 1, 1, 32, 2, 4, 64, 18

    raw = np.zeros((n_pos, n_frames, n_loops, n_tx, n_rx, n_bins), dtype=np.complex64)
    raw[:, :, :, :, :, target_bin] = np.complex64(1.0 + 0.0j)

    zero_bin = prepare_mimo_snapshots(raw, n_tx=n_tx)
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

    zero_bin = prepare_mimo_snapshots(raw, n_tx=n_tx)

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


@pytest.mark.parametrize("fraction", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_phase_aware_cubic_interpolation_recovers_fractional_real_tone(
    fraction: float,
) -> None:
    """A causal real tone must retain amplitude and phase between FFT bins."""
    samples_used = 128
    nfft = 6 * samples_used
    bin_float = 27.0 + float(fraction)
    sample_index = np.arange(samples_used, dtype=np.float64)
    signal = np.cos(
        (2.0 * np.pi * bin_float / float(nfft)) * sample_index
    ).astype(np.float32)
    spectrum = np.fft.fft(signal, n=nfft).astype(np.complex64, copy=False)
    truth = np.sum(
        signal.astype(np.float64, copy=False)
        * np.exp(-2j * np.pi * bin_float * sample_index / float(nfft)),
        dtype=np.complex128,
    )

    interpolated = _interp_cubic_complex(
        spectrum,
        np.asarray([bin_float], dtype=np.float32),
        nfft,
        samples_used=samples_used,
        nfft=nfft,
    )[0]
    ratio = np.complex128(interpolated) / truth
    amplitude_error_db = 20.0 * np.log10(abs(ratio))
    phase_error_rad = float(np.angle(ratio))

    assert abs(float(amplitude_error_db)) < 0.1
    assert abs(phase_error_rad) < 5e-4


def test_phase_aware_interpolation_rejects_insufficient_oversampling() -> None:
    spectrum = np.ones(256, dtype=np.complex64)
    bins = np.asarray([20.5], dtype=np.float32)

    with pytest.raises(ValueError, match="oversampling.*insufficiente"):
        _interp_cubic_complex(
            spectrum,
            bins,
            spectrum.size,
            samples_used=128,
            nfft=5 * 128,
        )


def test_phase_aware_linear_dc_boundary_stays_within_point_one_db() -> None:
    samples_used = 128
    nfft = 6 * samples_used
    bin_float = 0.5
    sample_index = np.arange(samples_used, dtype=np.float64)
    signal = np.exp(2j * np.pi * bin_float * sample_index / float(nfft))
    spectrum = np.fft.fft(signal, n=nfft).astype(np.complex64)
    truth = np.sum(
        signal * np.exp(-2j * np.pi * bin_float * sample_index / float(nfft)),
        dtype=np.complex128,
    )

    interpolated = _interp_cubic_complex(
        spectrum,
        np.asarray([bin_float], dtype=np.float32),
        nfft,
        samples_used=samples_used,
        nfft=nfft,
    )[0]
    amplitude_error_db = 20.0 * np.log10(abs(np.complex128(interpolated) / truth))

    assert abs(float(amplitude_error_db)) < 0.1


def test_phase_aware_interpolation_rejects_single_bin_fft() -> None:
    with pytest.raises(ValueError, match="nfft deve essere >= 2"):
        _interp_cubic_complex(
            np.ones(1, dtype=np.complex64),
            np.asarray([0.0], dtype=np.float32),
            1,
            samples_used=1,
            nfft=1,
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"samples_used": 128},
        {"nfft": 384},
    ],
)
def test_phase_aware_interpolation_requires_complete_fft_metadata(
    metadata: dict[str, int],
) -> None:
    with pytest.raises(ValueError, match="specificati insieme"):
        _interp_cubic_complex(
            np.ones(384, dtype=np.complex64),
            np.asarray([20.5], dtype=np.float32),
            384,
            **metadata,
        )


def test_phase_aware_geometry_bp_numba_matches_numpy_on_fractional_tone() -> None:
    import offline_dsp

    if offline_dsp._back_projection_power_mimo_geometry_numba_kernel is None:
        pytest.skip("Numba non installato")

    samples_used = 128
    nfft = 6 * samples_used
    bin_float = 27.5
    sample_index = np.arange(samples_used, dtype=np.float64)
    signal = np.cos(
        (2.0 * np.pi * bin_float / float(nfft)) * sample_index
    ).astype(np.float32)
    spectrum = np.fft.fft(signal, n=nfft).astype(np.complex64, copy=False)
    truth = np.sum(
        signal.astype(np.float64, copy=False)
        * np.exp(-2j * np.pi * bin_float * sample_index / float(nfft)),
        dtype=np.complex128,
    )
    snapshots = spectrum.reshape(1, 1, 1, nfft)
    tx_global = np.zeros((1, 1, 3), dtype=np.float32)
    rx_global = np.zeros((1, 1, 3), dtype=np.float32)
    voxel_xyz = np.asarray([[bin_float, 0.0, 0.0]], dtype=np.float32)
    kwargs = {
        "dr_m": 1.0,
        "fc_hz": 77e9,
        "c_m_s": 3e8,
        "max_bin": nfft,
        "range_samples_used": samples_used,
        "nfft_range": nfft,
    }

    reference = back_projection_power_mimo_geometry(
        snapshots,
        tx_global,
        rx_global,
        voxel_xyz,
        use_numba=False,
        **kwargs,
    )
    accelerated = back_projection_power_mimo_geometry(
        snapshots,
        tx_global,
        rx_global,
        voxel_xyz,
        use_numba=True,
        **kwargs,
    )

    np.testing.assert_allclose(accelerated, reference, atol=1e-2, rtol=2e-5)
    amplitude_error_db = 10.0 * np.log10(float(reference[0]) / (abs(truth) ** 2))
    assert abs(amplitude_error_db) < 0.1
