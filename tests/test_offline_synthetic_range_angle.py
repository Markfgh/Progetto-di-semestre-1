"""Casi sintetici per la ricostruzione offline range-angolo."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import offline_processing
import realtime_dsp
from offline_dsp import build_mimo_geometry, prepare_mimo_snapshots
from offline_processing import (
    OfflineSyntheticRangeAngleConfig,
    _apply_offline_backprojection_aperture_window,
    _apply_offline_backprojection_range_window,
    _select_offline_range_fft_input,
    _apply_offline_sar_range_angle_pre_filters,
    _compute_synthetic_range_angle_image,
    _compute_angle_heatmap_bounded,
    _prepare_offline_zero_doppler_position,
)


def _range_angle_cfg(**overrides) -> OfflineSyntheticRangeAngleConfig:
    base = OfflineSyntheticRangeAngleConfig(
        use_realtime_filters=False,
        window_range="hanning",
        window_doppler="hanning",
        window_angle="hanning",
        zero_after_range_fft_bins=0,
        post_range_fft_filters=realtime_dsp.PostRangeFftFilterConfig(
            mean_after_range_fft=realtime_dsp.MeanSelection(enabled=False),
        ),
        angle_processing=realtime_dsp.AngleProcessingConfig(mode="fft"),
        nfft_range=64,
        nfft_angle=16,
        projection=realtime_dsp.DisplayProjectionConfig(projection_mode="cartesian", projection_interp="nearest"),
        filter_warnings=(),
    )
    return replace(base, **overrides)


def _viewport() -> realtime_dsp.DisplayViewport:
    return realtime_dsp.build_display_viewport(
        x_min_m=-1.5,
        x_max_m=1.5,
        y_min_m=0.0,
        y_max_m=3.0,
        dr_m=0.05,
        seq=0,
    )


def _geometry() -> tuple[np.ndarray, np.ndarray]:
    return build_mimo_geometry(2, 4, fc_hz=77e9, c_m_s=3e8)


def test_zero_doppler_reduction_precedes_range_fft_without_changing_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(41)
    iq = (
        rng.normal(size=(2, 4, 8, 8))
        + 1j * rng.normal(size=(2, 4, 8, 8))
    ).astype(np.complex64)
    nfft = 64
    window = np.hanning(4).astype(np.float32)
    cfg = _range_angle_cfg(use_realtime_filters=False, nfft_range=nfft)

    full_fft = offline_processing.fft.fft(iq, n=nfft, axis=-1, workers=1).astype(np.complex64)
    expected = prepare_mimo_snapshots(
        full_fft.reshape(1, 2, 4, 2, 4, nfft),
        n_tx=2,
        window_doppler=window,
        log_info=False,
    )[0]

    real_fft = offline_processing.fft.fft
    observed_shapes: list[tuple[int, ...]] = []

    def recording_fft(value, *args, **kwargs):
        observed_shapes.append(tuple(int(v) for v in np.asarray(value).shape))
        return real_fft(value, *args, **kwargs)

    monkeypatch.setattr(offline_processing.fft, "fft", recording_fft)
    actual = _prepare_offline_zero_doppler_position(
        iq,
        nfft_range=nfft,
        chirps=8,
        tx=2,
        rx=4,
        algorithm="backprojection",
        range_angle_cfg=cfg,
        fft_workers=1,
        doppler_window=window,
    )

    assert observed_shapes == [(2, 2, 4, 8)]
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_angle_heatmap_is_processed_in_bounded_range_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cube = np.ones((2, 1, 11, 8), dtype=np.complex64)
    geometry, _ = offline_processing._build_synthetic_virtual_array_geometry(
        np.arange(8, dtype=np.float32) * np.float32(0.001),
        wavelength_m=0.004,
    )
    cfg = _range_angle_cfg(nfft_angle=16)
    dsp_cfg = offline_processing._build_synthetic_angle_dsp_cfg(
        c_m_s=3e8,
        fs_hz=10e6,
        slope_hz_s=60e12,
        nfft_range=64,
        nfft_angle=16,
        range_max_m=3.0,
        synthetic_ant=8,
        fft_workers=1,
        frames_like=2,
    )
    calls: list[int] = []

    def fake_heatmap(value, **_kwargs):
        calls.append(int(value.shape[2]))
        return np.full((int(value.shape[2]), 16), float(len(calls)), dtype=np.float32)

    monkeypatch.setattr(offline_processing, "compute_angle_heatmap", fake_heatmap)
    monkeypatch.setattr(offline_processing, "_OFFLINE_ANGLE_FFT_TARGET_BYTES", 256)
    heatmap, chunk_bins = _compute_angle_heatmap_bounded(
        cube,
        angle_cfg=cfg.angle_processing,
        dsp_cfg=dsp_cfg,
        angle_steering=np.empty((0, 0), dtype=np.complex64),
        geometry=geometry,
    )

    assert chunk_bins == 1
    assert calls == [1] * 11
    assert heatmap.shape == (11, 16)


def test_offline_pre_filters_ignore_slow_time_even_if_supplied() -> None:
    raw_mimo = np.ones((1, 1, 8, 2, 4, 3), dtype=np.complex64)
    filters_cfg = realtime_dsp.PostRangeFftFilterConfig(
        mean_after_range_fft=realtime_dsp.MeanSelection(enabled=False),
        slow_time=realtime_dsp.SlowTimeConfig(enabled=True, mode="mean_subtraction"),
    )

    out = _apply_offline_sar_range_angle_pre_filters(raw_mimo, filters_cfg=filters_cfg)

    np.testing.assert_allclose(out, raw_mimo, atol=0.0, rtol=0.0)


def test_backprojection_range_window_applies_on_fast_time_only() -> None:
    signal = np.ones((2, 1, 3, 4, 8), dtype=np.complex64) * np.complex64(1.0 + 2.0j)

    out = _apply_offline_backprojection_range_window(
        signal,
        window_type="hamming",
        enabled=True,
    )

    expected = signal * np.hamming(8).astype(np.float32).reshape(1, 1, 1, 1, 8)
    np.testing.assert_allclose(out, expected, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(
        _apply_offline_backprojection_range_window(signal, window_type="hamming", enabled=False),
        signal,
        atol=0.0,
        rtol=0.0,
    )


def test_backprojection_aperture_window_uses_position_major_antenna_order() -> None:
    snapshots = np.ones((2, 3, 4, 5), dtype=np.complex64)

    out = _apply_offline_backprojection_aperture_window(
        snapshots,
        window_type="hamming",
        enabled=True,
    )

    expected = snapshots * np.hamming(8).astype(np.float32).reshape(2, 1, 4, 1)
    np.testing.assert_allclose(out, expected, atol=1e-6, rtol=0.0)


def test_synthetic_range_angle_uses_one_combined_aperture_and_projects_to_gui(monkeypatch: pytest.MonkeyPatch) -> None:
    x_tx_ant_m, x_rx_ant_m = _geometry()
    range_fft_sel = np.ones((4, 2, 8, 6), dtype=np.complex64)
    selected_positions = np.asarray([0, 1, 2, 3], dtype=np.int32)
    captured: dict[str, object] = {}

    def fake_heatmap(virtual_array, **kwargs):
        captured["virtual_array_shape"] = tuple(int(v) for v in virtual_array.shape)
        return np.ones((6, 16), dtype=np.float32)

    monkeypatch.setattr(offline_processing, "compute_angle_heatmap", fake_heatmap)

    img_db, meta = _compute_synthetic_range_angle_image(
        range_fft_sel,
        selected_positions=selected_positions,
        x_pitch_m=0.1,
        tx_i=2,
        rx_i=4,
        x_tx_ant_m=x_tx_ant_m,
        x_rx_ant_m=x_rx_ant_m,
        range_angle_cfg=_range_angle_cfg(use_realtime_filters=False),
        c_m_s=3.0e8,
        fs_hz=10.0e6,
        slope_hz_s=60.0e12,
        fc_hz=77.0e9,
        fft_workers=1,
        nfft_range=64,
        gui_h=33,
        gui_w=41,
        viewport=_viewport(),
    )

    assert captured["virtual_array_shape"] == (2, 1, 6, 32)
    assert img_db.shape == (33, 41)
    assert np.all(np.isfinite(img_db))
    assert meta["synthetic_antennas"] == 32


def test_synthetic_range_angle_accepts_prepared_zero_doppler_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_tx_ant_m, x_rx_ant_m = _geometry()
    prepared = np.ones((2, 3, 8, 6), dtype=np.complex64)
    captured: dict[str, object] = {}

    def fail_prepare(*args, **kwargs):
        raise AssertionError("prepared input must not run Doppler preparation again")

    def fake_heatmap(virtual_array, **kwargs):
        captured["shape"] = tuple(int(v) for v in virtual_array.shape)
        return np.ones((6, 16), dtype=np.float32)

    monkeypatch.setattr(offline_processing, "_prepare_mimo_snapshots", fail_prepare)
    monkeypatch.setattr(offline_processing, "compute_angle_heatmap", fake_heatmap)

    img_db, meta = _compute_synthetic_range_angle_image(
        prepared,
        selected_positions=np.asarray([0, 1], dtype=np.int32),
        x_pitch_m=0.1,
        tx_i=2,
        rx_i=4,
        x_tx_ant_m=x_tx_ant_m,
        x_rx_ant_m=x_rx_ant_m,
        range_angle_cfg=_range_angle_cfg(use_realtime_filters=False),
        c_m_s=3.0e8,
        fs_hz=10.0e6,
        slope_hz_s=60.0e12,
        fc_hz=77.0e9,
        fft_workers=1,
        nfft_range=64,
        gui_h=17,
        gui_w=19,
        viewport=_viewport(),
    )

    assert captured["shape"] == (3, 1, 6, 16)
    assert img_db.shape == (17, 19)
    assert meta["synthetic_antennas"] == 16


def test_synthetic_range_angle_applies_window_angle_over_flattened_aperture(monkeypatch: pytest.MonkeyPatch) -> None:
    x_tx_ant_m, x_rx_ant_m = _geometry()
    captured: dict[str, np.ndarray] = {}

    def fake_heatmap(virtual_array, **kwargs):
        captured["snapshot"] = np.asarray(virtual_array, dtype=np.complex64).copy()
        return np.ones((1, 16), dtype=np.float32)

    monkeypatch.setattr(offline_processing, "compute_angle_heatmap", fake_heatmap)

    cfg = _range_angle_cfg(
        use_realtime_filters=True,
        post_range_fft_filters=realtime_dsp.PostRangeFftFilterConfig(
            mean_after_range_fft=realtime_dsp.MeanSelection(enabled=False),
            slow_time=realtime_dsp.SlowTimeConfig(enabled=False),
        ),
        window_angle="hanning",
    )
    _compute_synthetic_range_angle_image(
        np.ones((2, 1, 8, 1), dtype=np.complex64),
        selected_positions=np.asarray([0, 1], dtype=np.int32),
        x_pitch_m=0.1,
        tx_i=2,
        rx_i=4,
        x_tx_ant_m=x_tx_ant_m,
        x_rx_ant_m=x_rx_ant_m,
        range_angle_cfg=cfg,
        c_m_s=3.0e8,
        fs_hz=10.0e6,
        slope_hz_s=60.0e12,
        fc_hz=77.0e9,
        fft_workers=1,
        nfft_range=64,
        gui_h=9,
        gui_w=9,
        viewport=_viewport(),
    )

    expected_window = offline_processing._build_window_1d("hanning", 16)
    snapshot = captured["snapshot"]
    assert snapshot.shape == (1, 1, 1, 16)
    np.testing.assert_allclose(np.abs(snapshot.reshape(-1)), expected_window, atol=1e-6, rtol=0.0)


def test_uniform_synthetic_geometry_keeps_fft_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    x_tx_ant_m, x_rx_ant_m = _geometry()
    captured: dict[str, object] = {}

    def fake_heatmap(virtual_array, *, angle_cfg, **kwargs):
        captured["mode"] = angle_cfg.mode
        return np.ones((4, 16), dtype=np.float32)

    monkeypatch.setattr(offline_processing, "compute_angle_heatmap", fake_heatmap)

    _compute_synthetic_range_angle_image(
        np.ones((2, 1, 8, 4), dtype=np.complex64),
        selected_positions=np.asarray([0, 1], dtype=np.int32),
        x_pitch_m=float(2.0 * (3.0e8 / 77.0e9)),
        tx_i=2,
        rx_i=4,
        x_tx_ant_m=x_tx_ant_m,
        x_rx_ant_m=x_rx_ant_m,
        range_angle_cfg=_range_angle_cfg(angle_processing=realtime_dsp.AngleProcessingConfig(mode="fft")),
        c_m_s=3.0e8,
        fs_hz=10.0e6,
        slope_hz_s=60.0e12,
        fc_hz=77.0e9,
        fft_workers=1,
        nfft_range=64,
        gui_h=9,
        gui_w=9,
        viewport=_viewport(),
    )

    assert captured["mode"] == "fft"


def test_range_fft_input_is_truncated_or_kept_for_zero_padding() -> None:
    signal = np.arange(8, dtype=np.float32).astype(np.complex64).reshape(1, 8)

    truncated = _select_offline_range_fft_input(signal, nfft_range=5)
    padded_input = _select_offline_range_fft_input(signal, nfft_range=16)

    assert truncated.shape == (1, 5)
    np.testing.assert_array_equal(truncated, signal[:, :5])
    assert padded_input.shape == (1, 8)
    np.testing.assert_array_equal(padded_input, signal)


def test_measured_motor_pitch_is_uniform_and_angle_fft_supports_padding_and_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_tx_ant_m, x_rx_ant_m = _geometry()
    n_positions = 194
    captured: list[dict[str, int | str]] = []

    def fake_heatmap(virtual_array, *, angle_cfg, dsp_cfg, **kwargs):
        captured.append({
            "mode": str(angle_cfg.mode),
            "nfft_angle": int(dsp_cfg.nfft_angle),
            "synthetic_antennas": int(virtual_array.shape[-1]),
        })
        return np.ones((4, int(dsp_cfg.nfft_angle)), dtype=np.float32)

    monkeypatch.setattr(offline_processing, "compute_angle_heatmap", fake_heatmap)

    _, padded_meta = _compute_synthetic_range_angle_image(
        np.ones((n_positions, 1, 8, 4), dtype=np.complex64),
        selected_positions=np.arange(1, n_positions + 1, dtype=np.int32),
        x_pitch_m=0.007792000193148851,
        tx_i=2,
        rx_i=4,
        x_tx_ant_m=x_tx_ant_m,
        x_rx_ant_m=x_rx_ant_m,
        range_angle_cfg=_range_angle_cfg(
            nfft_angle=2048,
            angle_processing=realtime_dsp.AngleProcessingConfig(mode="fft"),
        ),
        c_m_s=3.0e8,
        fs_hz=10.0e6,
        slope_hz_s=14.967e12,
        fc_hz=77.0e9,
        fft_workers=1,
        nfft_range=1024,
        gui_h=9,
        gui_w=9,
        viewport=_viewport(),
    )

    _, truncated_meta = _compute_synthetic_range_angle_image(
        np.ones((n_positions, 1, 8, 4), dtype=np.complex64),
        selected_positions=np.arange(1, n_positions + 1, dtype=np.int32),
        x_pitch_m=0.007792000193148851,
        tx_i=2,
        rx_i=4,
        x_tx_ant_m=x_tx_ant_m,
        x_rx_ant_m=x_rx_ant_m,
        range_angle_cfg=_range_angle_cfg(
            nfft_angle=256,
            angle_processing=realtime_dsp.AngleProcessingConfig(mode="fft"),
        ),
        c_m_s=3.0e8,
        fs_hz=10.0e6,
        slope_hz_s=14.967e12,
        fc_hz=77.0e9,
        fft_workers=1,
        nfft_range=1024,
        gui_h=9,
        gui_w=9,
        viewport=_viewport(),
    )

    assert captured[0] == {
        "mode": "fft",
        "nfft_angle": 2048,
        "synthetic_antennas": 1552,
    }
    assert captured[1] == {
        "mode": "fft",
        "nfft_angle": 256,
        "synthetic_antennas": 256,
    }
    assert padded_meta["fft_uniform_geometry"] is True
    assert padded_meta["angle_input_elements"] == 1552
    assert padded_meta["angle_elements_used"] == 1552
    assert padded_meta["nfft_angle_requested"] == 2048
    assert padded_meta["nfft_angle_effective"] == 2048
    assert truncated_meta["fft_uniform_geometry"] is True
    assert truncated_meta["angle_input_elements"] == 1552
    assert truncated_meta["angle_elements_used"] == 256
    assert truncated_meta["nfft_angle_requested"] == 256
    assert truncated_meta["nfft_angle_effective"] == 256


def test_nonuniform_synthetic_geometry_falls_back_to_bartlett_with_warning(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    x_tx_ant_m, x_rx_ant_m = _geometry()
    captured: dict[str, object] = {}

    def fake_heatmap(virtual_array, *, angle_cfg, **kwargs):
        captured["mode"] = angle_cfg.mode
        return np.ones((4, 16), dtype=np.float32)

    monkeypatch.setattr(offline_processing, "compute_angle_heatmap", fake_heatmap)

    _compute_synthetic_range_angle_image(
        np.ones((2, 1, 8, 4), dtype=np.complex64),
        selected_positions=np.asarray([0, 2], dtype=np.int32),
        x_pitch_m=float(2.0 * (3.0e8 / 77.0e9)),
        tx_i=2,
        rx_i=4,
        x_tx_ant_m=x_tx_ant_m,
        x_rx_ant_m=x_rx_ant_m,
        range_angle_cfg=_range_angle_cfg(angle_processing=realtime_dsp.AngleProcessingConfig(mode="fft")),
        c_m_s=3.0e8,
        fs_hz=10.0e6,
        slope_hz_s=60.0e12,
        fc_hz=77.0e9,
        fft_workers=1,
        nfft_range=64,
        gui_h=9,
        gui_w=9,
        viewport=_viewport(),
    )

    assert captured["mode"] == "bartlett"
    assert "falling back to bartlett" in capsys.readouterr().out


def test_bartlett_steering_uses_physical_synthetic_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    x_tx_ant_m, x_rx_ant_m = _geometry()
    captured: dict[str, np.ndarray] = {}
    wavelength_m = np.float32(3.0e8 / 77.0e9)
    selected_positions = np.asarray([1, 3], dtype=np.int32)

    def fake_steering(virtual_ant, nfft_angle, geometry):
        captured["phase_centers_lambda"] = np.asarray(geometry.phase_centers_lambda, dtype=np.float32).copy()
        return np.ones((int(virtual_ant), int(nfft_angle)), dtype=np.complex64)

    monkeypatch.setattr(offline_processing, "build_angle_steering_matrix", fake_steering)
    monkeypatch.setattr(
        offline_processing,
        "compute_angle_heatmap",
        lambda virtual_array, **kwargs: np.ones((3, 16), dtype=np.float32),
    )

    _compute_synthetic_range_angle_image(
        np.ones((2, 1, 8, 3), dtype=np.complex64),
        selected_positions=selected_positions,
        x_pitch_m=0.05,
        tx_i=2,
        rx_i=4,
        x_tx_ant_m=x_tx_ant_m,
        x_rx_ant_m=x_rx_ant_m,
        range_angle_cfg=_range_angle_cfg(angle_processing=realtime_dsp.AngleProcessingConfig(mode="bartlett")),
        c_m_s=3.0e8,
        fs_hz=10.0e6,
        slope_hz_s=60.0e12,
        fc_hz=77.0e9,
        fft_workers=1,
        nfft_range=64,
        gui_h=9,
        gui_w=9,
        viewport=_viewport(),
    )

    expected_phase_centers = (
        (
            selected_positions.astype(np.float32)[:, None] * np.float32(0.05)
            + (np.float32(0.5) * (x_tx_ant_m + x_rx_ant_m))[None, :]
        ).reshape(-1)
        / wavelength_m
    ).astype(np.float32, copy=False)
    np.testing.assert_allclose(captured["phase_centers_lambda"], expected_phase_centers, atol=1e-6, rtol=0.0)


def test_synthetic_range_angle_heatmap_peaks_at_expected_range_and_angle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_tx_ant_m, x_rx_ant_m = _geometry()
    wavelength_m = np.float32(3.0e8 / 77.0e9)
    x_pitch_m = float(2.0 * wavelength_m)
    selected_positions = np.asarray([0, 1, 2, 3], dtype=np.int32)
    target_bin = 7
    target_angle_deg = np.float32(20.0)
    nfft_angle = 128
    captured: dict[str, np.ndarray] = {}

    phase_centers_m = (np.float32(0.5) * (x_tx_ant_m + x_rx_ant_m)).astype(np.float32, copy=False)
    x_element_m = (
        selected_positions.astype(np.float32)[:, None] * np.float32(x_pitch_m)
        + phase_centers_m[None, :]
    ).astype(np.float32, copy=False)
    geometry, _ = offline_processing._build_synthetic_virtual_array_geometry(
        x_element_m.reshape(-1),
        wavelength_m=float(wavelength_m),
    )
    u_target = np.float32(np.sin(np.deg2rad(target_angle_deg))) * np.float32(geometry.angle_u_to_sin_scale)
    steering = np.exp(
        (-1j * np.float32(2.0 * np.pi))
        * (x_element_m.reshape(-1) / wavelength_m)
        * u_target
    ).astype(np.complex64, copy=False)
    zero_doppler = np.zeros((4, 1, 8, 16), dtype=np.complex64)
    zero_doppler[:, 0, :, target_bin] = steering.reshape(4, 8)

    def fake_project(heatmap_lin, **kwargs):
        captured["heatmap"] = np.asarray(heatmap_lin, dtype=np.float32).copy()
        captured["angle_axis_deg"] = np.asarray(kwargs["angle_axis_deg"], dtype=np.float32).copy()
        return np.zeros((int(kwargs["gui_h"]), int(kwargs["gui_w"])), dtype=np.float32)

    monkeypatch.setattr(offline_processing, "project_heatmap_for_display", fake_project)

    _compute_synthetic_range_angle_image(
        zero_doppler,
        selected_positions=selected_positions,
        x_pitch_m=x_pitch_m,
        tx_i=2,
        rx_i=4,
        x_tx_ant_m=x_tx_ant_m,
        x_rx_ant_m=x_rx_ant_m,
        range_angle_cfg=_range_angle_cfg(
            use_realtime_filters=False,
            angle_processing=realtime_dsp.AngleProcessingConfig(mode="bartlett"),
            nfft_angle=nfft_angle,
        ),
        c_m_s=3.0e8,
        fs_hz=10.0e6,
        slope_hz_s=60.0e12,
        fc_hz=77.0e9,
        fft_workers=1,
        nfft_range=64,
        gui_h=9,
        gui_w=9,
        viewport=_viewport(),
    )

    heatmap = captured["heatmap"]
    angle_axis_deg = captured["angle_axis_deg"]
    peak_row, peak_col = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)

    assert peak_row == target_bin
    assert abs(float(angle_axis_deg[peak_col]) - float(target_angle_deg)) <= 3.0
