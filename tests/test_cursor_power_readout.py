"""Cursor sampling and power-readout helpers."""

from __future__ import annotations

import radar_app


def test_resolve_display_power_db_recovers_raw_and_normalized_values():
    raw_db, normalized_db = radar_app.resolve_display_power_db(
        -18.25,
        normalization_reference_db=42.5,
        displayed_is_normalized=True,
    )

    assert raw_db == 24.25
    assert normalized_db == -18.25

    raw_db, normalized_db = radar_app.resolve_display_power_db(
        24.25,
        normalization_reference_db=42.5,
        displayed_is_normalized=False,
    )

    assert raw_db == 24.25
    assert normalized_db == -18.25


def test_format_power_cursor_readout_includes_position_and_both_power_scales():
    readout = radar_app.format_power_cursor_readout(
        x_m=-1.25,
        y_m=3.5,
        raw_power_db=24.25,
        normalized_power_db=-18.25,
    )

    assert "X: -1.250 m | Y: +3.500 m" in readout
    assert "Power raw: +24.25 dB" in readout
    assert "Power norm: -18.25 dB" in readout
