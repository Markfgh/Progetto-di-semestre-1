"""Test unitari dei filtri applicati prima e dopo la range FFT."""

from __future__ import annotations

import numpy as np

import realtime_dsp


def test_subtract_selected_mean_removes_requested_axes_in_place() -> None:
    data = np.arange(2 * 3 * 2 * 4 * 2, dtype=np.float32).reshape(2, 3, 2, 4, 2).astype(np.complex64)
    original_id = id(data)

    out = realtime_dsp.subtract_selected_mean(
        data,
        realtime_dsp.MeanSelection(enabled=True, axes=("frame", "tx")),
    )

    assert id(out) == original_id
    np.testing.assert_allclose(out.mean(axis=(0, 2)), 0.0, atol=1e-6)


def test_slow_time_mean_subtraction_and_zero_doppler_notch() -> None:
    loops = 8
    data = np.ones((1, loops, 1, 2, 1), dtype=np.complex64)
    data[:, :, :, 1, :] = np.exp(
        1j * 2.0 * np.pi * np.arange(loops, dtype=np.float32) / loops
    ).reshape(1, loops, 1, 1)

    mean_removed = realtime_dsp.apply_slow_time_filter(
        data,
        realtime_dsp.SlowTimeConfig(enabled=True, mode="mean_subtraction"),
    )
    np.testing.assert_allclose(mean_removed[:, :, :, 0, :], 0.0, atol=1e-6)

    doppler = realtime_dsp.apply_slow_time_filter(
        data,
        realtime_dsp.SlowTimeConfig(
            enabled=True,
            mode="doppler_fft",
            doppler_fft_shift=True,
            doppler_zero_notch=True,
        ),
    )
    assert doppler.shape == data.shape
    np.testing.assert_allclose(doppler[:, loops // 2, :, :, :], 0.0, atol=1e-6)
    assert float(np.abs(doppler[:, loops // 2 + 1, :, 1, :]).max()) > 7.5


def test_background_subtraction_ema_initializes_then_subtracts_model() -> None:
    state = realtime_dsp.BackgroundSubtractionState()
    cfg = realtime_dsp.BackgroundSubtractionConfig(enabled=True, mode="ema", alpha=0.5, init_frames=2)
    batch1 = np.ones((1, 2, 1, 3, 1), dtype=np.complex64)
    batch2 = np.full_like(batch1, 3.0 + 0.0j)
    batch3 = np.full_like(batch1, 7.0 + 0.0j)

    out1 = realtime_dsp.apply_background_subtraction(batch1.copy(), cfg, state)
    out2 = realtime_dsp.apply_background_subtraction(batch2.copy(), cfg, state)
    out3 = realtime_dsp.apply_background_subtraction(batch3.copy(), cfg, state)

    np.testing.assert_allclose(out1, batch1)
    np.testing.assert_allclose(out2, 1.0 + 0.0j)
    np.testing.assert_allclose(out3, 5.0 + 0.0j)
    np.testing.assert_allclose(state.model, 4.5 + 0.0j)


def test_background_subtraction_clamp_positive_only_preserves_negative_residual_magnitude() -> None:
    state = realtime_dsp.BackgroundSubtractionState(model=np.full((1, 1, 2, 1), 10.0 + 0.0j, dtype=np.complex64))
    cfg = realtime_dsp.BackgroundSubtractionConfig(
        enabled=True,
        mode="frozen",
        init_frames=1,
        clamp_positive_only=True,
    )
    data = np.full((1, 1, 1, 2, 1), 3.0 + 0.0j, dtype=np.complex64)

    out = realtime_dsp.apply_background_subtraction(data, cfg, state)

    np.testing.assert_allclose(out, 0.0 + 0.0j, atol=1e-6)


def test_disabled_background_subtraction_ignores_existing_model() -> None:
    state = realtime_dsp.BackgroundSubtractionState(
        model=np.full((1, 1, 2, 1), 10.0 + 0.0j, dtype=np.complex64)
    )
    cfg = realtime_dsp.BackgroundSubtractionConfig(enabled=False, mode="frozen", init_frames=1)
    data = np.full((1, 1, 1, 2, 1), 3.0 + 0.0j, dtype=np.complex64)

    out = realtime_dsp.apply_background_subtraction(data, cfg, state)

    assert out is data
    np.testing.assert_allclose(out, data)


def test_post_range_filters_can_loop_average_after_background() -> None:
    data = np.ones((2, 4, 1, 3, 1), dtype=np.complex64)
    data[:, 2:, :, :, :] = 3.0 + 0.0j
    filters = realtime_dsp.PostRangeFftFilterConfig(
        mean_after_range_fft=realtime_dsp.MeanSelection(enabled=False),
        slow_time=realtime_dsp.SlowTimeConfig(enabled=False),
        background_subtraction=realtime_dsp.BackgroundSubtractionConfig(enabled=False),
    )

    out = realtime_dsp.apply_post_range_fft_filters(
        data,
        filters,
        bg_state=realtime_dsp.BackgroundSubtractionState(),
        fft_workers=1,
        apply_loop_average_after_background=True,
    )

    assert out.shape == (2, 1, 1, 3, 1)
    np.testing.assert_allclose(out, 2.0 + 0.0j)
