"""Test del validatore standalone per i file RTPBIN v1."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import struct

import pytest

from capture_file_inspector import HEADER_MAGIC, main, read_capture_header


def _valid_metadata(*, position: int = 7) -> dict:
    return {
        "format": "rt_capture_v1",
        "position": position,
        "radar": {
            "c": 3.0e8,
            "fs": 10.0e6,
            "slope": 29.982e12,
            "fc": 77.0e9,
        },
        "capture": {
            "samples": 8,
            "chirps": 4,
            "rx": 4,
            "tx": 2,
            "x_frames": 1,
            "frames_per_position": 1,
        },
        "stage": {"position_mm": -12.5},
    }


def _write_capture(path: Path, metadata: dict, *, payload: bytes = b"\x00") -> None:
    header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    path.write_bytes(HEADER_MAGIC + struct.pack("<I", len(header)) + header + payload)


def test_valid_header_accepts_signed_finite_stage_and_cli_counts_frames(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "capture_pos7.bin"
    metadata = _valid_metadata()
    bytes_per_frame = 4 * 8 * 4 * 4
    _write_capture(path, metadata, payload=bytes(bytes_per_frame))

    parsed, data_offset = read_capture_header(path)

    assert parsed == metadata
    assert data_offset < path.stat().st_size
    assert main([str(path)]) == 0
    output = capsys.readouterr().out
    assert f"bytes_per_frame: {bytes_per_frame}" in output
    assert "Frame nel payload: 1" in output


@pytest.mark.parametrize("section", ["radar", "capture", "stage"])
def test_required_metadata_objects_are_rejected(tmp_path: Path, section: str) -> None:
    metadata = _valid_metadata()
    del metadata[section]
    path = tmp_path / "capture_pos7.bin"
    _write_capture(path, metadata)

    with pytest.raises(ValueError, match=section):
        read_capture_header(path)


@pytest.mark.parametrize("position", [None, float("nan"), float("inf")])
def test_header_position_is_required_and_must_be_an_integer(
    tmp_path: Path,
    position: object,
) -> None:
    metadata = _valid_metadata()
    if position is None:
        del metadata["position"]
    else:
        metadata["position"] = position
    path = tmp_path / "arbitrary_name.bin"
    _write_capture(path, metadata)

    with pytest.raises(ValueError, match=r"header\.position"):
        read_capture_header(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("c", 0.0, r"radar\.c deve essere > 0"),
        ("fs", -1.0, r"radar\.fs deve essere > 0"),
        ("slope", float("nan"), r"radar\.slope deve essere finito"),
        ("fc", float("inf"), r"radar\.fc deve essere finito"),
    ],
)
def test_radar_values_must_be_finite_and_positive(
    tmp_path: Path,
    field: str,
    value: float,
    message: str,
) -> None:
    metadata = _valid_metadata()
    metadata["radar"][field] = value
    path = tmp_path / "capture_pos7.bin"
    _write_capture(path, metadata)

    with pytest.raises(ValueError, match=message):
        read_capture_header(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("samples", 0),
        ("chirps", -1),
        ("rx", 0),
        ("tx", 0),
        ("x_frames", 0),
        ("frames_per_position", -2),
    ],
)
def test_capture_counts_must_be_positive(
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    metadata = _valid_metadata()
    metadata["capture"][field] = value
    path = tmp_path / "capture_pos7.bin"
    _write_capture(path, metadata)

    with pytest.raises(ValueError, match=rf"capture\.{field} deve essere > 0"):
        read_capture_header(path)


@pytest.mark.parametrize("value", [True, 1.5, "1.5"])
def test_capture_counts_must_be_integers(tmp_path: Path, value: object) -> None:
    metadata = _valid_metadata()
    metadata["capture"]["samples"] = value
    path = tmp_path / "capture_pos7.bin"
    _write_capture(path, metadata)

    with pytest.raises(ValueError, match=r"capture\.samples"):
        read_capture_header(path)


def test_chirps_must_be_a_multiple_of_tx(tmp_path: Path) -> None:
    metadata = _valid_metadata()
    metadata["capture"]["chirps"] = 5
    path = tmp_path / "capture_pos7.bin"
    _write_capture(path, metadata)

    with pytest.raises(ValueError, match="chirps deve essere multiplo di tx"):
        read_capture_header(path)


@pytest.mark.parametrize("position_mm", [float("nan"), float("inf")])
def test_stage_position_mm_must_be_finite(
    tmp_path: Path,
    position_mm: float,
) -> None:
    metadata = _valid_metadata()
    metadata["stage"]["position_mm"] = position_mm
    path = tmp_path / "capture_pos7.bin"
    _write_capture(path, metadata)

    with pytest.raises(ValueError, match=r"stage\.position_mm deve essere finito"):
        read_capture_header(path)


def test_stage_position_mm_is_required(tmp_path: Path) -> None:
    metadata = _valid_metadata()
    del metadata["stage"]["position_mm"]
    path = tmp_path / "capture_pos7.bin"
    _write_capture(path, metadata)

    with pytest.raises(ValueError, match=r"stage\.position_mm mancante"):
        read_capture_header(path)


def test_filename_header_position_mismatch_is_fatal_for_api_and_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "capture_pos8.bin"
    _write_capture(path, _valid_metadata(position=7))

    with pytest.raises(ValueError, match=r"nome=8, header=7"):
        read_capture_header(path)

    assert main([str(path)]) == 2
    assert "[ERR] Header non valido" in capsys.readouterr().out


def test_zero_layout_exits_before_payload_modulo(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata = copy.deepcopy(_valid_metadata())
    metadata["capture"]["samples"] = 0
    path = tmp_path / "capture_pos7.bin"
    _write_capture(path, metadata)

    assert main([str(path)]) == 2
    output = capsys.readouterr().out
    assert "capture.samples deve essere > 0" in output
