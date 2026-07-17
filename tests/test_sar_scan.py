from __future__ import annotations

import threading

import pytest

from sar_capture import normalize_cylindrical_metadata
from sar_scan import (
    CylindricalScanCoordinator,
    CylindricalScanPlan,
    ScanPlan,
    SarScanCoordinator,
    absolute_scan_targets_microsteps,
    dry_run_cylindrical_scan,
)


def test_absolute_targets_do_not_accumulate_per_step_rounding() -> None:
    # Configurazione attuale: 7.792 mm / 0.0281602715934975 mm per microstep.
    targets = absolute_scan_targets_microsteps(
        start_microsteps=10_000,
        pitch_mm=7.792,
        mm_per_microstep=0.0281602715934975,
        positions=4,
    )

    assert targets == (10_000, 10_277, 10_553, 10_830)
    # Sommare sempre 277 avrebbe prodotto 10_554 e 10_831: deriva evitata.
    assert targets[-1] != 10_831


def test_coordinator_captures_before_each_move_and_skips_final_move() -> None:
    events: list[tuple] = []
    current_steps = 1_000

    def begin_motion() -> None:
        events.append(("begin",))

    def finish_motion(success: bool) -> None:
        events.append(("finish", success))

    def get_position() -> int:
        return current_steps

    def mm_from_steps(steps: int) -> float:
        return steps * 0.5

    def move(target: int, _timeout: float, _cancel: threading.Event) -> None:
        nonlocal current_steps
        events.append(("move", target))
        current_steps = target

    def request_capture(pos_id: int, pos_mm: float, pos_steps: int) -> int:
        events.append(("capture", pos_id, pos_mm, pos_steps))
        return pos_id

    def wait_capture(ticket: int, _timeout: float, _cancel: threading.Event) -> None:
        events.append(("saved", ticket))

    coordinator = SarScanCoordinator(
        begin_motion=begin_motion,
        finish_motion=finish_motion,
        get_position_microsteps=get_position,
        mm_from_microsteps=mm_from_steps,
        mm_per_microstep=lambda: 0.5,
        move_to_microsteps=move,
        request_capture=request_capture,
        wait_capture=wait_capture,
        cancel_capture=lambda: events.append(("cancel_capture",)),
        stop_motion=lambda: events.append(("stop",)),
    )

    coordinator.start(ScanPlan(positions=3, pitch_mm=1.25, settling_seconds=0.0))
    coordinator.join(timeout=1.0)

    assert coordinator.status().state == "completed"
    assert events == [
        ("begin",),
        ("capture", 1, 500.0, 1_000),
        ("saved", 1),
        ("move", 1_002),
        ("capture", 2, 501.0, 1_002),
        ("saved", 2),
        ("move", 1_005),
        ("capture", 3, 502.5, 1_005),
        ("saved", 3),
        ("finish", True),
    ]


def test_cancel_after_capture_confirmation_cannot_become_completed() -> None:
    events: list[tuple] = []
    coordinator: SarScanCoordinator

    def wait_capture(ticket: int, _timeout: float, _cancel: threading.Event) -> None:
        events.append(("saved", ticket))
        coordinator.cancel()

    coordinator = SarScanCoordinator(
        begin_motion=lambda: events.append(("begin",)),
        finish_motion=lambda success: events.append(("finish", success)),
        get_position_microsteps=lambda: 100,
        mm_from_microsteps=lambda steps: float(steps),
        mm_per_microstep=lambda: 1.0,
        move_to_microsteps=lambda *_args: events.append(("move",)),
        request_capture=lambda pos_id, _mm, _steps: pos_id,
        wait_capture=wait_capture,
        cancel_capture=lambda: events.append(("cancel_capture",)),
        stop_motion=lambda: events.append(("stop",)),
    )

    coordinator.start(ScanPlan(positions=1, pitch_mm=1.0))
    coordinator.join(timeout=1.0)

    status = coordinator.status()
    assert status.state == "cancelled"
    assert status.completed == 0
    assert events == [
        ("begin",),
        ("saved", 1),
        ("stop",),
        ("cancel_capture",),
        ("finish", False),
    ]


def test_completed_is_published_only_after_finish_motion_returns() -> None:
    states_seen_during_finish: list[str] = []
    coordinator: SarScanCoordinator

    def finish_motion(success: bool) -> None:
        assert success
        states_seen_during_finish.append(coordinator.status().state)

    coordinator = SarScanCoordinator(
        begin_motion=lambda: None,
        finish_motion=finish_motion,
        get_position_microsteps=lambda: 100,
        mm_from_microsteps=lambda steps: float(steps),
        mm_per_microstep=lambda: 1.0,
        move_to_microsteps=lambda *_args: None,
        request_capture=lambda pos_id, _mm, _steps: pos_id,
        wait_capture=lambda *_args: None,
        cancel_capture=lambda: None,
        stop_motion=lambda: None,
    )

    coordinator.start(ScanPlan(positions=1, pitch_mm=1.0))
    coordinator.join(timeout=1.0)

    assert states_seen_during_finish == ["capturing"]
    assert coordinator.status().state == "completed"


def test_finish_motion_failure_never_reports_completed() -> None:
    coordinator: SarScanCoordinator

    def finish_motion(_success: bool) -> None:
        assert coordinator.status().state != "completed"
        raise RuntimeError("chiusura motore fallita")

    coordinator = SarScanCoordinator(
        begin_motion=lambda: None,
        finish_motion=finish_motion,
        get_position_microsteps=lambda: 100,
        mm_from_microsteps=lambda steps: float(steps),
        mm_per_microstep=lambda: 1.0,
        move_to_microsteps=lambda *_args: None,
        request_capture=lambda pos_id, _mm, _steps: pos_id,
        wait_capture=lambda *_args: None,
        cancel_capture=lambda: None,
        stop_motion=lambda: None,
    )

    coordinator.start(ScanPlan(positions=1, pitch_mm=1.0))
    coordinator.join(timeout=1.0)

    status = coordinator.status()
    assert status.state == "failed"
    assert status.error == "chiusura motore fallita"


def test_cylindrical_dry_run_end_to_end_two_heights_generates_v2_metadata_in_order() -> None:
    plan = CylindricalScanPlan(
        angles_per_turn=4,
        radius_m=1.25,
        initial_height_m=0.20,
        height_count=2,
        vertical_step_m=0.15,
        scene_center_m=(1.0, -2.0, 0.5),
        start_capture_id=40,
        angular_settling_seconds=0.25,
        vertical_settling_seconds=0.5,
    )

    result = dry_run_cylindrical_scan(plan)

    assert result.status.state == "completed"
    assert [capture.capture_id for capture in result.captures] == list(range(40, 48))
    assert [capture.acquisition_index for capture in result.captures] == list(range(8))
    assert [capture.angle_index for capture in result.captures] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert [capture.height_index for capture in result.captures] == [0, 0, 0, 0, 1, 1, 1, 1]
    expected_azimuth = [0.0, 0.5 * 3.141592653589793, 3.141592653589793, 1.5 * 3.141592653589793]
    assert [capture.azimuth_rad for capture in result.captures] == pytest.approx(expected_azimuth * 2)
    assert [capture.height_m for capture in result.captures] == pytest.approx([0.20] * 4 + [0.35] * 4)
    for capture in result.captures:
        normalized = normalize_cylindrical_metadata(capture.to_dict())
        assert normalized["position_m"] == pytest.approx(capture.position_world_m.tolist())

    capture_events = [event for event in result.events if event[0] == "capture"]
    flushed_events = [event for event in result.events if event[0] == "flushed"]
    assert [event[1] for event in capture_events] == list(range(40, 48))
    assert [event[1] for event in flushed_events] == list(range(40, 48))
    # Quattro passi positivi (tre campioni successivi + chiusura) per ciascun
    # giro, per un avanzamento fisico totale di 2*pi senza duplicare azimuth 0.
    rotary_moves = [event[1] for event in result.events if event[0] == "rotary_move"]
    assert rotary_moves == pytest.approx([plan.angular_step_rad] * 8)
    assert sum(rotary_moves[:4]) == pytest.approx(2.0 * 3.141592653589793)
    assert sum(rotary_moves[4:]) == pytest.approx(2.0 * 3.141592653589793)


def test_cylindrical_cancel_stops_vertical_and_rotary_axes() -> None:
    events: list[tuple] = []
    aligned = threading.Event()
    release = threading.Event()
    current_steps = 0

    class BlockingRotary:
        def begin_external_scan(self) -> None:
            events.append(("rotary_begin",))

        def finish_external_scan(self, success: bool) -> None:
            events.append(("rotary_finish", success))

        def align_zero_and_wait(self, _timeout: float, cancel_event: threading.Event) -> None:
            events.append(("rotary_align",))
            aligned.set()
            release.wait(1.0)
            if cancel_event.is_set():
                raise RuntimeError("Scansione annullata.")

        def move_relative_rad_and_wait(self, *_args) -> None:
            events.append(("rotary_move",))

        def stop(self) -> None:
            events.append(("rotary_stop",))
            release.set()

    def move_vertical(target: int, _timeout: float, _cancel: threading.Event) -> None:
        nonlocal current_steps
        current_steps = target
        events.append(("vertical_move", target))

    coordinator = CylindricalScanCoordinator(
        begin_vertical_scan=lambda: events.append(("vertical_begin",)),
        finish_vertical_scan=lambda success: events.append(("vertical_finish", success)),
        get_vertical_microsteps=lambda: current_steps,
        height_m_from_microsteps=lambda steps: steps / 1000.0,
        microsteps_from_height_m=lambda height_m: int(round(height_m * 1000.0)),
        move_vertical_to_microsteps=move_vertical,
        stop_vertical=lambda: events.append(("vertical_stop",)),
        rotary_axis=BlockingRotary(),
        request_capture=lambda **_kwargs: None,
        wait_capture=lambda *_args: None,
        cancel_capture=lambda: events.append(("capture_cancel",)),
    )
    coordinator.start(
        CylindricalScanPlan(
            angles_per_turn=4,
            radius_m=1.0,
            initial_height_m=0.0,
            height_count=2,
            vertical_step_m=0.1,
        )
    )
    assert aligned.wait(1.0)
    coordinator.cancel()
    coordinator.join(timeout=1.0)

    assert coordinator.status().state == "cancelled"
    assert ("vertical_stop",) in events
    assert ("rotary_stop",) in events
    assert ("capture_cancel",) in events
    assert ("rotary_finish", False) in events
    assert ("vertical_finish", False) in events
