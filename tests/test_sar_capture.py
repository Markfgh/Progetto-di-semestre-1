"""Verifica il protocollo IPC delle catture SAR e i relativi metadati."""

from __future__ import annotations

import queue
import threading
import json
import struct
from pathlib import Path

import pytest

import main_refactory
from sar_capture import (
    CaptureError,
    CaptureMetadataStore,
    CaptureSessionManager,
    normalize_capture_metadata,
    read_capture_metadata,
    write_capture_metadata,
)


class Shared:
    def __init__(self, value: int = 0) -> None:
        self.value = value


def test_capture_waits_for_matching_completed_session() -> None:
    commands: queue.Queue = queue.Queue()
    cap_id = Shared(10)
    cap_done_id = Shared(9)  # completione vecchia: non deve sbloccare la nuova
    cap_result = Shared(1)
    manager = CaptureSessionManager(
        cmd_queue=commands,
        cap_id=cap_id,
        cap_done_id=cap_done_id,
        cap_result=cap_result,
    )

    ticket = manager.request(4, 12.5, 321)
    assert commands.get_nowait() == (
        "CAPTURE",
        4,
        {"carriage_position_mm": 12.5, "carriage_microsteps": 321},
    )
    assert manager.inflight

    cap_id.value = 11
    cap_done_id.value = 10
    with pytest.raises(CaptureError, match="Timeout"):
        manager.wait(ticket, timeout_seconds=0.02, cancel_event=threading.Event())

    # Il ticket resta occupato fino al risultato della stessa sessione.
    assert manager.inflight
    cap_done_id.value = 11
    cap_result.value = 1
    manager.wait(ticket, timeout_seconds=0.2, cancel_event=threading.Event())
    assert not manager.inflight


def test_capture_header_and_offline_scan_config_include_stage_coordinates(tmp_path: Path) -> None:
    header = main_refactory._build_capture_file_header(
        7,
        carriage_position_mm=12.375,
        carriage_microsteps=439,
    )
    prefix_len = len(main_refactory.CAPTURE_HEADER_MAGIC) + 4
    assert header.startswith(main_refactory.CAPTURE_HEADER_MAGIC)
    header_len = struct.unpack("<I", header[len(main_refactory.CAPTURE_HEADER_MAGIC) : prefix_len])[0]
    payload = json.loads(header[prefix_len : prefix_len + header_len].decode("utf-8"))
    assert payload["format"] == "rt_capture_v1"
    assert payload["position"] == 7
    assert payload["capture"]["frames_per_position"] == main_refactory.FRAMES_PER_POSITION
    assert payload["stage"] == {
        "reference": "phidget_home_min",
        "position_mm": 12.375,
        "position_microsteps": 439,
    }

    offline_config = tmp_path / "offline_config.yaml"
    offline_config.write_text(
        "# commento da conservare\n"
        "data:\n  input_dir: logs\n"
        "scan:\n  x_start: 1\n  x_end: 1\n  x_step: 1\n"
        "  x_pitch_m: 0.007792 # 7.792 mm pitch\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "logs" / "run_test"
    output_dir.mkdir(parents=True)
    pitch_mm = main_refactory.configure_offline_scan_for_run(
        offline_config,
        output_dir=output_dir,
        start_position_id=3,
        positions=4,
        frames_per_position=main_refactory.FRAMES_PER_POSITION,
    )
    with offline_config.open("r", encoding="utf-8") as handle:
        saved = main_refactory.yaml.safe_load(handle)
    assert pitch_mm == pytest.approx(7.792)
    assert saved["scan"] == {"x_start": 3, "x_end": 6, "x_step": 1, "x_pitch_m": 0.007792}
    assert saved["capture"]["frames_per_position"] == main_refactory.FRAMES_PER_POSITION
    assert saved["data"]["input_dir"] == "logs\\run_test" or saved["data"]["input_dir"] == "logs/run_test"
    saved_text = offline_config.read_text(encoding="utf-8")
    assert "# commento da conservare" in saved_text
    assert "x_pitch_m: 0.007792 # 7.792 mm pitch" in saved_text


def test_manual_yaml_scalar_update_preserves_comments_and_nested_structure(tmp_path: Path) -> None:
    config_path = tmp_path / "offline_config.yaml"
    config_path.write_text(
        "# nota generale\n"
        "data:\n  input_dir: logs # cartella acquisizioni\n"
        "scan:\n  x_pitch_m: 0.007792 # vecchio commento\n"
        "outer:\n  nested:\n    enabled: false # flag importante\n",
        encoding="utf-8",
    )

    main_refactory._update_existing_yaml_scalar_paths(
        config_path,
        {
            "data.input_dir": "logs/run_new",
            "scan.x_pitch_m": 0.01,
            "outer.nested.enabled": True,
        },
        inline_comments={"scan.x_pitch_m": "10 mm pitch"},
    )

    saved_text = config_path.read_text(encoding="utf-8")
    saved = main_refactory.yaml.safe_load(saved_text)
    assert saved["data"]["input_dir"] == "logs/run_new"
    assert saved["scan"]["x_pitch_m"] == pytest.approx(0.01)
    assert saved["outer"]["nested"]["enabled"] is True
    assert "# nota generale" in saved_text
    assert "# cartella acquisizioni" in saved_text
    assert "# flag importante" in saved_text
    assert "x_pitch_m: 0.01 # 10 mm pitch" in saved_text
