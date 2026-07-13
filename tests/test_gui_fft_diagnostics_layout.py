from __future__ import annotations

from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "main_refactory.py"


def _source() -> str:
    return SOURCE.read_text(encoding="utf-8")


def _section(src: str, start_marker: str, end_marker: str) -> str:
    start = src.index(start_marker)
    end = src.index(end_marker, start)
    return src[start:end]


def test_angle_diagnostics_controls_are_below_plot_and_colorbar():
    src = _source()
    section = _section(src, 'with dpg.tab(label="Angle"):', 'with dpg.tab(label="Doppler"):')

    plot_idx = section.index("ANGLEFFT_HEAT_PLOT_TAG")
    colorbar_idx = section.index("ANGLEFFT_CMAP_SCALE_TAG")
    controls_idx = section.index("CHK_ANGLE_SINGLE_BIN")

    assert plot_idx < controls_idx
    assert colorbar_idx < controls_idx
    assert "with dpg.group(horizontal=True)" not in section


def test_doppler_diagnostics_controls_are_below_plot_and_colorbar():
    src = _source()
    section = _section(src, 'with dpg.tab(label="Doppler"):', 'with dpg.table(header_row=False, resizable=True, policy=dpg.mvTable_SizingFixedFit, parent=TAB_PROCESSED_TAG):')

    plot_idx = section.index("DOPPLERFFT_HEAT_PLOT_TAG")
    colorbar_idx = section.index("DOPPLERFFT_CMAP_SCALE_TAG")
    controls_idx = section.index("CHK_DOPPLER_SINGLE_BIN")

    assert plot_idx < controls_idx
    assert colorbar_idx < controls_idx
    assert "with dpg.group(horizontal=True)" not in section
