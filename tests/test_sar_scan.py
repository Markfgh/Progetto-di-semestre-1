"""Casi di successo, annullamento ed errore della macchina a stati SAR."""

from __future__ import annotations

import threading

from sar_scan import (
    ScanEvent,
    ScanPlan,
    ScanState,
    SarScanCoordinator,
    absolute_scan_targets_microsteps,
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

    status = coordinator.status()
    assert status.state is ScanState.COMPLETED
    assert status.event is ScanEvent.COMPLETED
    assert status.completed == 3
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
    cancellation_status: tuple[ScanState, ScanEvent] | None = None
    coordinator: SarScanCoordinator

    def wait_capture(ticket: int, _timeout: float, _cancel: threading.Event) -> None:
        events.append(("saved", ticket))
        coordinator.cancel()

    def stop_motion() -> None:
        nonlocal cancellation_status
        status = coordinator.status()
        cancellation_status = (status.state, status.event)
        events.append(("stop",))

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
        stop_motion=stop_motion,
    )

    coordinator.start(ScanPlan(positions=1, pitch_mm=1.0))
    coordinator.join(timeout=1.0)

    status = coordinator.status()
    assert status.state is ScanState.CANCELLED
    assert status.event is ScanEvent.CANCELLED
    assert cancellation_status == (ScanState.CANCELLING, ScanEvent.CANCELLATION_REQUESTED)
    assert status.completed == 0
    assert events == [
        ("begin",),
        ("saved", 1),
        ("stop",),
        ("cancel_capture",),
        ("finish", False),
    ]


def test_completed_is_published_only_after_finish_motion_returns() -> None:
    states_seen_during_finish: list[ScanState] = []
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

    assert states_seen_during_finish == [ScanState.CAPTURING]
    status = coordinator.status()
    assert status.state is ScanState.COMPLETED
    assert status.event is ScanEvent.COMPLETED


def test_finish_motion_failure_never_reports_completed() -> None:
    coordinator: SarScanCoordinator

    def finish_motion(_success: bool) -> None:
        assert coordinator.status().state is not ScanState.COMPLETED
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
    assert status.state is ScanState.FAILED
    assert status.event is ScanEvent.FINALIZATION_FAILED
    assert status.error == "chiusura motore fallita"


def test_operational_failure_publishes_interrupted_event() -> None:
    def request_capture(*_args: object) -> None:
        raise RuntimeError("logger unavailable")

    coordinator = SarScanCoordinator(
        begin_motion=lambda: None,
        finish_motion=lambda _success: None,
        get_position_microsteps=lambda: 100,
        mm_from_microsteps=lambda steps: float(steps),
        mm_per_microstep=lambda: 1.0,
        move_to_microsteps=lambda *_args: None,
        request_capture=request_capture,
        wait_capture=lambda *_args: None,
        cancel_capture=lambda: None,
        stop_motion=lambda: None,
    )

    coordinator.start(ScanPlan(positions=1, pitch_mm=1.0))
    coordinator.join(timeout=1.0)

    status = coordinator.status()
    assert status.state is ScanState.FAILED
    assert status.event is ScanEvent.INTERRUPTED
    assert status.error == "logger unavailable"


def test_status_uses_event_codes_and_separate_position_data() -> None:
    """Il coordinatore non produce messaggi localizzati per gli stati SAR."""

    events_seen: list[tuple[ScanState, ScanEvent, int | None, float | None]] = []
    current_steps = 100
    coordinator: SarScanCoordinator

    def wait_capture(_ticket: int, _timeout: float, _cancel: threading.Event) -> None:
        status = coordinator.status()
        events_seen.append((status.state, status.event, status.position_id, status.position_mm))

    def move(_target: int, _timeout: float, _cancel: threading.Event) -> None:
        nonlocal current_steps
        status = coordinator.status()
        events_seen.append((status.state, status.event, status.position_id, status.position_mm))
        current_steps = _target

    coordinator = SarScanCoordinator(
        begin_motion=lambda: None,
        finish_motion=lambda _success: None,
        get_position_microsteps=lambda: current_steps,
        mm_from_microsteps=lambda steps: float(steps) / 10.0,
        mm_per_microstep=lambda: 1.0,
        move_to_microsteps=move,
        request_capture=lambda pos_id, _mm, _steps: pos_id,
        wait_capture=wait_capture,
        cancel_capture=lambda: None,
        stop_motion=lambda: None,
    )

    coordinator.start(ScanPlan(positions=2, pitch_mm=1.0))
    coordinator.join(timeout=1.0)

    assert events_seen == [
        (ScanState.CAPTURING, ScanEvent.CAPTURING_POSITION, 1, 10.0),
        (ScanState.MOVING, ScanEvent.MOVING_TO_POSITION, 2, 10.1),
        (ScanState.CAPTURING, ScanEvent.CAPTURING_POSITION, 2, 10.1),
    ]
    status = coordinator.status()
    assert not hasattr(status, "message")
