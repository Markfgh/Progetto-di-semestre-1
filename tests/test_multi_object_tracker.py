"""Scenari sintetici per associazione, ciclo di vita e stato moto dei track."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from multi_object_tracker import MultiObjectTracker, TrackerConfig, TrackingConfig


def _det(
    x: float,
    y: float,
    *,
    doppler: float | None = None,
    source: str = "moving",
    power: float = 10.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        x_m=float(x),
        y_m=float(y),
        range_m=float(np.hypot(x, y)),
        angle_deg=float(np.rad2deg(np.arctan2(x, max(y, 1e-6)))),
        doppler_mps=doppler,
        power_lin=float(power),
        power_db=float(power),
        source=source,
    )


def _tracker(
    *,
    max_tracks: int = 10,
    min_hits: int = 2,
    max_missed_tentative: int = 1,
    max_missed_confirmed: int = 2,
    gating_xy_m: float = 0.5,
    gating_doppler_mps: float = 0.4,
    birth_min_separation_m: float = 0.15,
    motion_confirm_frames_stopped: int = 2,
    stopped_memory_s: float = 0.5,
) -> MultiObjectTracker:
    return MultiObjectTracker(
        TrackingConfig(
            enabled=True,
            dt_s=0.1,
            max_tracks=max_tracks,
            min_hits_to_confirm=min_hits,
            max_missed_tentative=max_missed_tentative,
            max_missed_confirmed=max_missed_confirmed,
        ),
        TrackerConfig(
            gating_xy_m=gating_xy_m,
            gating_doppler_mps=gating_doppler_mps,
            process_noise_pos=0.05,
            process_noise_vel=0.2,
            measurement_noise_xy=0.05,
            moving_speed_threshold_mps=0.20,
            stopped_speed_threshold_mps=0.05,
            doppler_moving_threshold_mps=0.12,
            motion_confirm_frames_moving=1,
            motion_confirm_frames_stopped=motion_confirm_frames_stopped,
            stopped_memory_s=stopped_memory_s,
            birth_min_separation_m=birth_min_separation_m,
        ),
    )


def test_tracker_associates_near_measurements_and_confirms_track() -> None:
    tracker = _tracker()

    tracks = tracker.step([_det(0.0, 2.0, doppler=0.3)], timestamp_s=0.0)
    first_id = tracks[0].track_id
    tracks = tracker.step([_det(0.02, 2.03, doppler=0.31)], timestamp_s=0.1)

    assert len(tracks) == 1
    assert tracks[0].track_id == first_id
    assert tracks[0].confirmed
    assert tracks[0].hits == 2
    assert tracker.last_debug_snapshot.matches == 1


def test_tracker_xy_and_doppler_gating_create_separate_tracks() -> None:
    tracker = _tracker(
        max_tracks=5,
        min_hits=1,
        gating_xy_m=0.25,
        gating_doppler_mps=0.2,
        birth_min_separation_m=0.0,
    )

    tracker.step([_det(0.0, 2.0, doppler=0.2)], timestamp_s=0.0)
    tracker.step([_det(1.0, 2.0, doppler=0.2)], timestamp_s=0.1)
    assert len(tracker.tracks) == 2
    assert tracker.last_debug_snapshot.gating_reject_xy >= 1

    tracker.step([_det(1.02, 2.02, doppler=-0.8)], timestamp_s=0.2)
    assert len(tracker.tracks) == 3
    assert tracker.last_debug_snapshot.gating_reject_doppler >= 1


def test_tracker_lifecycle_deletes_missed_tentative_and_confirmed_tracks() -> None:
    tentative = _tracker(min_hits=3, max_missed_tentative=0)
    tentative.step([_det(0.0, 1.0)], timestamp_s=0.0)
    tracks = tentative.step([], timestamp_s=0.1)
    assert tracks == []
    assert tentative.last_debug_snapshot.deleted_tracks == 1

    confirmed = _tracker(
        min_hits=1,
        max_missed_confirmed=1,
        motion_confirm_frames_stopped=99,
        stopped_memory_s=0.0,
    )
    confirmed.step([_det(0.0, 1.0)], timestamp_s=0.0)
    assert len(confirmed.step([], timestamp_s=0.1)) == 1
    assert confirmed.step([], timestamp_s=0.2) == []
    assert confirmed.last_debug_snapshot.deleted_tracks == 1


def test_tracker_stopped_then_resume_uses_stop_anchor_and_motion_classification() -> None:
    tracker = _tracker(
        min_hits=1,
        gating_xy_m=0.4,
        gating_doppler_mps=0.8,
        motion_confirm_frames_stopped=1,
    )

    tracker.step([_det(0.0, 2.0, doppler=0.0, source="static")], timestamp_s=0.0)
    tracker.step([_det(0.01, 2.0, doppler=0.0, source="static")], timestamp_s=0.1)
    stopped = tracker.tracks[0]
    assert stopped.motion_state == "stopped"
    assert stopped.classification == "static"
    assert stopped.stop_x_m is not None
    stop_id = stopped.track_id

    tracks = tracker.step([_det(0.05, 2.02, doppler=0.5, source="moving")], timestamp_s=0.2)
    assert len(tracks) == 1
    assert tracks[0].track_id == stop_id
    assert tracks[0].motion_state == "moving"
    assert tracks[0].classification == "dynamic"


def test_tracker_max_tracks_birth_order_prefers_moving_and_filters_invalid() -> None:
    tracker = _tracker(max_tracks=2, min_hits=1)
    invalid = SimpleNamespace(x_m=np.nan, y_m=0.0, range_m=0.0, angle_deg=0.0, doppler_mps=None, power_lin=1.0, power_db=0.0, source="static")

    tracks = tracker.step(
        [
            _det(0.0, 1.0, source="static", power=100.0),
            _det(2.0, 1.0, source="moving", power=1.0),
            _det(4.0, 1.0, source="moving", power=2.0),
            invalid,
        ],
        timestamp_s=0.0,
    )

    assert len(tracks) == 2
    assert tracker.last_debug_snapshot.invalid_detections == 1
    assert [track.last_detection_source for track in tracks] == ["moving", "moving"]
