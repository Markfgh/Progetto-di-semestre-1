"""Readable GUI status and memory formatting."""

from __future__ import annotations

import radar_app


def test_memory_formatter_keeps_small_allocations_visible() -> None:
    assert radar_app.format_memory_bytes(0) == "0 B"
    assert radar_app.format_memory_bytes(512 * 1024) == "512.0 KiB"
    assert radar_app.format_memory_bytes(12 * 1024**2) == "12.0 MiB"
    assert radar_app.format_memory_bytes(1.5 * 1024**3) == "1.50 GiB"


def test_realtime_pipeline_error_has_one_worker_per_line() -> None:
    text = radar_app.format_realtime_pipeline_error(
        [
            ("RX", "OSError: socket closed"),
            ("DSP", "exited unexpectedly (code 1)"),
        ]
    )

    assert text.splitlines() == [
        "REALTIME PIPELINE ERROR",
        "RX: OSError: socket closed",
        "DSP: exited unexpectedly (code 1)",
    ]


def test_backprojection_summary_does_not_advertise_angle_fft() -> None:
    text = radar_app.format_offline_algorithm_summary("backprojection", "fft")

    assert text == "Algorithm: BACKPROJECTION"
    assert "FFT" not in text


def test_synthetic_range_angle_summary_includes_angle_method() -> None:
    text = radar_app.format_offline_algorithm_summary("synthetic_range_angle", "mvdr")

    assert text == "Algorithm: SYNTHETIC RANGE-ANGLE | Angle: MVDR"
