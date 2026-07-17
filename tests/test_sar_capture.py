from __future__ import annotations

import queue
import threading
import json
import math
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


def test_yaml_scalar_update_materializes_v2_view_without_removing_legacy_plane(tmp_path: Path) -> None:
    config_path = tmp_path / "offline_config.yaml"
    config_path.write_text(
        "data:\n  input_dir: logs\n"
        "reconstruction:\n"
        "  cylindrical_plane:\n"
        "    x_min_m: -1.0\n    x_max_m: 1.0\n"
        "    y_min_m: -1.0\n    y_max_m: 1.0\n    z_m: 0.2\n",
        encoding="utf-8",
    )

    main_refactory._update_existing_yaml_scalar_paths(
        config_path,
        {
            "data.input_dir": "logs/run_v2",
            "reconstruction.cylindrical_view.bounds.x_min_m": -0.5,
            "reconstruction.cylindrical_view.bounds.x_max_m": 0.5,
            "reconstruction.cylindrical_view.bounds.y_min_m": -0.4,
            "reconstruction.cylindrical_view.bounds.y_max_m": 0.4,
            "reconstruction.cylindrical_view.section.plane": "xy",
            "reconstruction.cylindrical_view.section.coordinate_m": 0.2,
        },
    )

    saved = main_refactory.yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["data"]["input_dir"] == "logs/run_v2"
    assert saved["reconstruction"]["cylindrical_plane"]["z_m"] == pytest.approx(0.2)
    assert saved["reconstruction"]["cylindrical_view"]["bounds"]["y_min_m"] == pytest.approx(-0.4)
    assert saved["reconstruction"]["cylindrical_view"]["section"] == {
        "plane": "xy",
        "coordinate_m": 0.2,
    }


def test_cylindrical_run_updates_only_offline_directory_and_required_frame_count(tmp_path: Path) -> None:
    config_path = tmp_path / "offline_config.yaml"
    config_path.write_text(
        "data:\n  input_dir: logs/old\n"
        "capture:\n  frames_per_position: 3\n"
        "scan:\n  x_start: 7\n  x_end: 19\n  x_step: 2\n  x_pitch_m: 0.123\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "logs" / "cylinder"
    output_dir.mkdir(parents=True)

    main_refactory.configure_offline_cylindrical_scan_for_run(
        config_path,
        output_dir=output_dir,
        frames_per_position=8,
    )

    saved = main_refactory.yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["data"]["input_dir"] in {"logs/cylinder", "logs\\cylinder"}
    assert saved["capture"]["frames_per_position"] == 8
    # I campi ``scan`` sono esclusivamente legacy lineari e non vengono
    # riscritti dalla finalizzazione di una run rt_capture_v2.
    assert saved["scan"] == {"x_start": 7, "x_end": 19, "x_step": 2, "x_pitch_m": 0.123}


def test_capture_header_v2_cylindrical_preserves_legacy_position_and_derives_world_position() -> None:
    header = main_refactory._build_capture_file_header(
        91,
        capture_id=41,
        acquisition_index=7,
        cylindrical={
            "angle_index": 3,
            "height_index": 2,
            "angle_count": 12,
            "azimuth_rad": float(0.5 * 3.141592653589793),
            "height_m": 0.4,
            "radius_m": 2.0,
            "scene_center_m": [1.0, -2.0, 0.25],
        },
    )
    prefix_len = len(main_refactory.CAPTURE_HEADER_MAGIC) + 4
    header_len = struct.unpack("<I", header[len(main_refactory.CAPTURE_HEADER_MAGIC) : prefix_len])[0]
    payload = json.loads(header[prefix_len : prefix_len + header_len].decode("utf-8"))

    assert payload["format"] == "rt_capture_v2"
    assert payload["capture_id"] == 41
    assert payload["acquisition_index"] == 7
    assert payload["position"] == 91  # Legacy only; v2 geometry is cylindrical.
    assert payload["cylindrical"] == {
        "angle_index": 3,
        "height_index": 2,
        "angle_count": 12,
        "azimuth_rad": pytest.approx(0.5 * 3.141592653589793),
        "height_m": 0.4,
        "radius_m": 2.0,
        "scene_center_m": [1.0, -2.0, 0.25],
        "position_m": pytest.approx([1.0, 0.0, 0.65]),
    }


def test_cylindrical_capture_command_and_shared_metadata_are_bound_to_session() -> None:
    commands: queue.Queue = queue.Queue()
    cap_id = Shared(10)
    manager = CaptureSessionManager(
        cmd_queue=commands,
        cap_id=cap_id,
        cap_done_id=Shared(0),
        cap_result=Shared(0),
    )
    cylindrical = {
        "angle_index": 1,
        "height_index": 4,
        "angle_count": 8,
        "azimuth_rad": 0.25,
        "height_m": 1.2,
        "radius_m": 0.9,
        "scene_center_m": [0.0, 0.0, 0.0],
    }

    ticket = manager.request(
        position_id=999,
        position_mm=12.5,
        position_microsteps=321,
        capture_id=44,
        acquisition_index=6,
        cylindrical=cylindrical,
    )
    command = commands.get_nowait()
    assert command[0:2] == ("CAPTURE", 44)
    assert command[2]["capture_id"] == 44
    assert command[2]["acquisition_index"] == 6
    assert command[2]["position"] == 999
    assert command[2]["cylindrical"]["position_m"] == pytest.approx(
        [0.9 * math.cos(0.25), 0.9 * math.sin(0.25), 1.2]
    )
    assert ticket.capture_id == 44
    assert ticket.acquisition_index == 6
    assert ticket.position == 999

    store = CaptureMetadataStore(
        buffer=bytearray(4096),
        byte_count=Shared(0),
        session_id=Shared(0),
        lock=threading.RLock(),
    )
    metadata = normalize_capture_metadata(command[1], command[2])
    write_capture_metadata(store, session_id=11, metadata=metadata)
    assert read_capture_metadata(store, session_id=10) is None
    persisted = read_capture_metadata(store, session_id=11)
    assert persisted is not None
    assert persisted["capture_id"] == 44
    assert persisted["acquisition_index"] == 6
    assert persisted["position"] == 999
    assert persisted["cylindrical"] == command[2]["cylindrical"]


def test_cylindrical_metadata_requires_regular_positive_radius_and_wrapped_azimuth() -> None:
    common = {
        "angle_index": 0,
        "height_index": 0,
        "angle_count": 1,
        "height_m": 0.0,
        "scene_center_m": [0.0, 0.0, 0.0],
    }
    with pytest.raises(CaptureError, match="strictly positive"):
        main_refactory._build_capture_file_header(
            1,
            cylindrical={**common, "azimuth_rad": 0.0, "radius_m": 0.0},
        )
    with pytest.raises(CaptureError, match=r"\[0, 2\*pi\)"):
        main_refactory._build_capture_file_header(
            1,
            cylindrical={**common, "azimuth_rad": 2.0 * 3.141592653589793, "radius_m": 1.0},
        )
    with pytest.raises(CaptureError, match="capture_id"):
        normalize_capture_metadata(-1, {"cylindrical": {**common, "azimuth_rad": 0.0, "radius_m": 1.0}})
