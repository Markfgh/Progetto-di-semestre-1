from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import time
from typing import Any, Literal, Protocol

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment
except Exception:
    _linear_sum_assignment = None


TrackLifecycleState = Literal["tentative", "confirmed", "deleted"]
TrackClassification = Literal["static", "dynamic", "unknown"]

_EPS = 1e-6
_LARGE_COST = 1e6


class DetectionLike(Protocol):
    range_m: float
    angle_deg: float
    doppler_mps: float | None
    x_m: float
    y_m: float
    power_lin: float
    power_db: float
    source: str


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool = True
    dt_s: float | None = None
    max_tracks: int = 30
    min_hits_to_confirm: int = 3
    max_missed_tentative: int = 2
    max_missed_confirmed: int = 8
    max_track_age: int = 0


@dataclass(frozen=True)
class TrackerConfig:
    model: str = "kalman_cv_2d"
    gating_xy_m: float = 0.75
    gating_doppler_mps: float = 0.50
    process_noise_pos: float = 0.20
    process_noise_vel: float = 1.00
    measurement_noise_xy: float = 0.25
    static_speed_threshold_mps: float = 0.08
    dynamic_speed_threshold_mps: float = 0.20
    doppler_static_threshold_mps: float = 0.10
    classification_confirm_frames: int = 2
    use_doppler_in_cost: bool = True
    use_detection_class_in_cost: bool = True
    history_len: int = 10
    debug_log: bool = False


@dataclass
class Track:
    track_id: int
    state_vector: np.ndarray = field(repr=False)
    covariance: np.ndarray = field(repr=False)
    age: int = 1
    hits: int = 1
    missed: int = 0
    state: TrackLifecycleState = "tentative"
    classification: TrackClassification = "unknown"
    last_update_time_s: float | None = None
    last_predict_dt_s: float = 0.0
    range_m: float = 0.0
    angle_deg: float = 0.0
    doppler_mps: float | None = None
    last_detection_source: str | None = None
    history: deque[tuple[float, float]] = field(default_factory=deque, repr=False)
    static_frames: int = 0
    dynamic_frames: int = 0

    @property
    def x_m(self) -> float:
        return float(self.state_vector[0])

    @x_m.setter
    def x_m(self, value: float) -> None:
        self.state_vector[0] = float(value)

    @property
    def y_m(self) -> float:
        return float(self.state_vector[1])

    @y_m.setter
    def y_m(self, value: float) -> None:
        self.state_vector[1] = float(value)

    @property
    def vx_mps(self) -> float:
        return float(self.state_vector[2])

    @vx_mps.setter
    def vx_mps(self, value: float) -> None:
        self.state_vector[2] = float(value)

    @property
    def vy_mps(self) -> float:
        return float(self.state_vector[3])

    @vy_mps.setter
    def vy_mps(self, value: float) -> None:
        self.state_vector[3] = float(value)

    @property
    def missed_frames(self) -> int:
        return int(self.missed)

    @missed_frames.setter
    def missed_frames(self, value: int) -> None:
        self.missed = int(value)

    @property
    def confirmed(self) -> bool:
        return self.state == "confirmed"

    @property
    def speed_mps(self) -> float:
        return float(math.hypot(self.vx_mps, self.vy_mps))

    @property
    def heading_deg(self) -> float | None:
        if self.speed_mps <= _EPS:
            return None
        return float(math.degrees(math.atan2(self.vx_mps, self.vy_mps)))

    @property
    def radial_velocity_mps(self) -> float | None:
        if self.range_m <= _EPS:
            return None
        return float((self.x_m * self.vx_mps + self.y_m * self.vy_mps) / max(self.range_m, _EPS))

    def to_public_dict(self) -> dict[str, Any]:
        doppler_mps = self.doppler_mps
        if doppler_mps is None:
            doppler_mps = self.radial_velocity_mps
        return {
            "id": int(self.track_id),
            "x_m": float(self.x_m),
            "y_m": float(self.y_m),
            "vx_mps": float(self.vx_mps),
            "vy_mps": float(self.vy_mps),
            "speed_mps": float(self.speed_mps),
            "range_m": float(self.range_m),
            "angle_deg": float(self.angle_deg),
            "heading_deg": None if self.heading_deg is None else float(self.heading_deg),
            "doppler_mps": None if doppler_mps is None else float(doppler_mps),
            "classification": str(self.classification),
            "state": str(self.state),
            "confirmed": bool(self.confirmed),
            "age": int(self.age),
            "hits": int(self.hits),
            "missed": int(self.missed),
        }


@dataclass(frozen=True)
class Measurement:
    x_m: float
    y_m: float
    range_m: float
    angle_deg: float
    doppler_mps: float | None
    power_lin: float
    power_db: float
    source: str
    motion_hint: TrackClassification


@dataclass(frozen=True)
class TrackerDebugSnapshot:
    assignment: str
    detections_in: int
    detections_used: int
    tracks_before: int
    tracks_after: int
    matches: int
    new_tracks: int
    deleted_tracks: int
    unmatched_tracks: int
    unmatched_detections: int
    gating_reject_xy: int
    gating_reject_doppler: int
    invalid_detections: int


class MultiObjectTracker:
    def __init__(self, tracking_cfg: TrackingConfig, tracker_cfg: TrackerConfig):
        self.tracking_cfg = tracking_cfg
        self.tracker_cfg = tracker_cfg
        self.tracks: list[Track] = []
        self.next_track_id = 1
        self._last_timestamp_s: float | None = None
        self._last_dt_s = self._fallback_dt_s()
        self.last_debug_snapshot = TrackerDebugSnapshot(
            assignment="none",
            detections_in=0,
            detections_used=0,
            tracks_before=0,
            tracks_after=0,
            matches=0,
            new_tracks=0,
            deleted_tracks=0,
            unmatched_tracks=0,
            unmatched_detections=0,
            gating_reject_xy=0,
            gating_reject_doppler=0,
            invalid_detections=0,
        )

    def reset(self) -> None:
        self.tracks.clear()
        self.next_track_id = 1
        self._last_timestamp_s = None
        self._last_dt_s = self._fallback_dt_s()

    def step(self, detections: list[DetectionLike], timestamp_s: float | None = None) -> list[Track]:
        if not self.tracking_cfg.enabled:
            self.reset()
            return []

        now_s = float(timestamp_s if timestamp_s is not None else time.perf_counter())
        dt_s = self._resolve_dt(now_s)
        measurements, invalid_count = self._sanitize_detections(detections)

        for track in self.tracks:
            self._predict_track(track, dt_s, now_s)

        matches, unmatched_tracks, unmatched_measurements, assoc_debug = self._associate(measurements)

        for track_idx, meas_idx in matches:
            self._update_track(self.tracks[track_idx], measurements[meas_idx], now_s)

        deleted_tracks = 0
        for track_idx in unmatched_tracks:
            if self._mark_unmatched(self.tracks[track_idx]):
                deleted_tracks += 1

        births = self._spawn_tracks(measurements, unmatched_measurements, now_s)
        self._finalize_tracks()

        self.last_debug_snapshot = TrackerDebugSnapshot(
            assignment=assoc_debug["assignment"],
            detections_in=int(len(detections)),
            detections_used=int(len(measurements)),
            tracks_before=int(assoc_debug["tracks_before"]),
            tracks_after=int(len(self.tracks)),
            matches=int(len(matches)),
            new_tracks=int(births),
            deleted_tracks=int(deleted_tracks),
            unmatched_tracks=int(len(unmatched_tracks)),
            unmatched_detections=int(len(unmatched_measurements)),
            gating_reject_xy=int(assoc_debug["gating_reject_xy"]),
            gating_reject_doppler=int(assoc_debug["gating_reject_doppler"]),
            invalid_detections=int(invalid_count),
        )
        if self.tracker_cfg.debug_log:
            dbg = self.last_debug_snapshot
            print(
                "[TRACK] "
                f"tracks={dbg.tracks_after} matches={dbg.matches} new={dbg.new_tracks} "
                f"deleted={dbg.deleted_tracks} det={dbg.detections_used}/{dbg.detections_in} "
                f"unmatched_tr={dbg.unmatched_tracks} unmatched_det={dbg.unmatched_detections} "
                f"reject_xy={dbg.gating_reject_xy} reject_doppler={dbg.gating_reject_doppler} "
                f"invalid={dbg.invalid_detections} assign={dbg.assignment}"
            )
        return list(self.tracks)

    def _fallback_dt_s(self) -> float:
        cfg_dt_s = self.tracking_cfg.dt_s
        if cfg_dt_s is not None and math.isfinite(cfg_dt_s) and cfg_dt_s > 0.0:
            return float(cfg_dt_s)
        return 0.05

    def _resolve_dt(self, timestamp_s: float) -> float:
        cfg_dt_s = self.tracking_cfg.dt_s
        if cfg_dt_s is not None and math.isfinite(cfg_dt_s) and cfg_dt_s > 0.0:
            dt_s = float(cfg_dt_s)
        elif self._last_timestamp_s is not None:
            dt_s = float(timestamp_s - self._last_timestamp_s)
        else:
            dt_s = self._last_dt_s
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            dt_s = self._fallback_dt_s()
        dt_s = float(min(max(dt_s, 1e-3), 1.0))
        self._last_timestamp_s = float(timestamp_s)
        self._last_dt_s = dt_s
        return dt_s

    def _sanitize_detections(self, detections: list[DetectionLike]) -> tuple[list[Measurement], int]:
        measurements: list[Measurement] = []
        invalid_count = 0
        for det in detections:
            meas = self._measurement_from_detection(det)
            if meas is None:
                invalid_count += 1
                continue
            measurements.append(meas)
        return measurements, invalid_count

    def _measurement_from_detection(self, det: DetectionLike) -> Measurement | None:
        try:
            x_m = float(getattr(det, "x_m"))
            y_m = float(getattr(det, "y_m"))
        except Exception:
            return None
        if not (math.isfinite(x_m) and math.isfinite(y_m)):
            return None
        range_m = float(getattr(det, "range_m", math.hypot(x_m, y_m)))
        if not math.isfinite(range_m) or range_m < 0.0:
            range_m = float(math.hypot(x_m, y_m))
        angle_deg = float(getattr(det, "angle_deg", math.degrees(math.atan2(x_m, max(y_m, _EPS)))))
        if not math.isfinite(angle_deg):
            angle_deg = float(math.degrees(math.atan2(x_m, max(y_m, _EPS))))

        doppler_raw = getattr(det, "doppler_mps", None)
        doppler_mps: float | None
        if doppler_raw is None:
            doppler_mps = None
        else:
            try:
                doppler_val = float(doppler_raw)
            except Exception:
                doppler_val = float("nan")
            doppler_mps = float(doppler_val) if math.isfinite(doppler_val) else None

        power_lin = float(getattr(det, "power_lin", 0.0))
        if not math.isfinite(power_lin):
            power_lin = 0.0
        power_db = float(getattr(det, "power_db", 0.0))
        if not math.isfinite(power_db):
            power_db = 0.0

        source = str(getattr(det, "source", "unknown") or "unknown").strip().lower()
        motion_hint = "unknown"
        abs_doppler = None if doppler_mps is None else abs(float(doppler_mps))
        if source == "moving" or (abs_doppler is not None and abs_doppler >= self.tracker_cfg.dynamic_speed_threshold_mps):
            motion_hint = "dynamic"
        elif source == "static" or (abs_doppler is not None and abs_doppler <= self.tracker_cfg.doppler_static_threshold_mps):
            motion_hint = "static"

        return Measurement(
            x_m=float(x_m),
            y_m=float(y_m),
            range_m=float(range_m),
            angle_deg=float(angle_deg),
            doppler_mps=doppler_mps,
            power_lin=float(power_lin),
            power_db=float(power_db),
            source=source,
            motion_hint=motion_hint,
        )

    def _predict_track(self, track: Track, dt_s: float, now_s: float) -> None:
        f_mat = np.array(
            [
                [1.0, 0.0, dt_s, 0.0],
                [0.0, 1.0, 0.0, dt_s],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        q_mat = self._build_process_noise(dt_s)
        track.state_vector = f_mat @ track.state_vector
        track.covariance = f_mat @ track.covariance @ f_mat.T + q_mat
        track.covariance = 0.5 * (track.covariance + track.covariance.T)
        track.age += 1
        track.missed += 1
        track.last_predict_dt_s = float(dt_s)
        track.last_update_time_s = now_s if track.last_update_time_s is None else track.last_update_time_s
        self._refresh_track_geometry(track)

    def _build_process_noise(self, dt_s: float) -> np.ndarray:
        dt2 = dt_s * dt_s
        dt3 = dt2 * dt_s
        dt4 = dt2 * dt2
        q_vel = max(self.tracker_cfg.process_noise_vel, 1e-3) ** 2
        q_pos = max(self.tracker_cfg.process_noise_pos, 1e-3) ** 2
        q_cv = q_vel * np.array(
            [
                [0.25 * dt4, 0.0, 0.5 * dt3, 0.0],
                [0.0, 0.25 * dt4, 0.0, 0.5 * dt3],
                [0.5 * dt3, 0.0, dt2, 0.0],
                [0.0, 0.5 * dt3, 0.0, dt2],
            ],
            dtype=np.float64,
        )
        q_diag = np.diag(
            [
                q_pos * max(dt2, 1e-3),
                q_pos * max(dt2, 1e-3),
                q_pos * max(dt_s, 1e-3),
                q_pos * max(dt_s, 1e-3),
            ]
        ).astype(np.float64, copy=False)
        return q_cv + q_diag

    def _associate(
        self, measurements: list[Measurement]
    ) -> tuple[list[tuple[int, int]], list[int], list[int], dict[str, Any]]:
        n_tracks = len(self.tracks)
        n_meas = len(measurements)
        if n_tracks <= 0 or n_meas <= 0:
            return [], list(range(n_tracks)), list(range(n_meas)), {
                "assignment": "none",
                "tracks_before": n_tracks,
                "gating_reject_xy": 0,
                "gating_reject_doppler": 0,
            }

        cost_matrix = np.full((n_tracks, n_meas), _LARGE_COST, dtype=np.float64)
        reject_xy = 0
        reject_doppler = 0
        for track_idx, track in enumerate(self.tracks):
            track_doppler = self._expected_track_doppler(track)
            for meas_idx, meas in enumerate(measurements):
                d_xy = float(math.hypot(meas.x_m - track.x_m, meas.y_m - track.y_m))
                if d_xy > self.tracker_cfg.gating_xy_m:
                    reject_xy += 1
                    continue
                if (
                    self.tracker_cfg.gating_doppler_mps > 0.0
                    and meas.doppler_mps is not None
                    and track_doppler is not None
                ):
                    if abs(float(meas.doppler_mps) - float(track_doppler)) > self.tracker_cfg.gating_doppler_mps:
                        reject_doppler += 1
                        continue
                cost_matrix[track_idx, meas_idx] = self._association_cost(track, meas, d_xy, track_doppler)

        if _linear_sum_assignment is not None:
            row_idx, col_idx = _linear_sum_assignment(cost_matrix)
            assignment = "hungarian"
            matched_pairs = [
                (int(row), int(col))
                for row, col in zip(row_idx, col_idx)
                if float(cost_matrix[int(row), int(col)]) < _LARGE_COST
            ]
        else:
            assignment = "greedy_fallback"
            matched_pairs = self._greedy_assign(cost_matrix)

        matched_track_idx = {track_idx for track_idx, _ in matched_pairs}
        matched_meas_idx = {meas_idx for _, meas_idx in matched_pairs}
        unmatched_tracks = [idx for idx in range(n_tracks) if idx not in matched_track_idx]
        unmatched_measurements = [idx for idx in range(n_meas) if idx not in matched_meas_idx]
        return matched_pairs, unmatched_tracks, unmatched_measurements, {
            "assignment": assignment,
            "tracks_before": n_tracks,
            "gating_reject_xy": reject_xy,
            "gating_reject_doppler": reject_doppler,
        }

    def _greedy_assign(self, cost_matrix: np.ndarray) -> list[tuple[int, int]]:
        finite_pairs = np.argwhere(cost_matrix < _LARGE_COST)
        ranked_pairs = sorted(
            (
                (float(cost_matrix[row_idx, col_idx]), int(row_idx), int(col_idx))
                for row_idx, col_idx in finite_pairs
            ),
            key=lambda item: item[0],
        )
        used_tracks: set[int] = set()
        used_measurements: set[int] = set()
        matches: list[tuple[int, int]] = []
        for _, track_idx, meas_idx in ranked_pairs:
            if track_idx in used_tracks or meas_idx in used_measurements:
                continue
            used_tracks.add(track_idx)
            used_measurements.add(meas_idx)
            matches.append((track_idx, meas_idx))
        return matches

    def _association_cost(
        self,
        track: Track,
        meas: Measurement,
        d_xy: float,
        track_doppler: float | None,
    ) -> float:
        gate_xy = max(self.tracker_cfg.gating_xy_m, 1e-3)
        cost = d_xy / gate_xy
        if (
            self.tracker_cfg.use_doppler_in_cost
            and self.tracker_cfg.gating_doppler_mps > 0.0
            and meas.doppler_mps is not None
            and track_doppler is not None
        ):
            cost += 0.35 * (
                abs(float(meas.doppler_mps) - float(track_doppler))
                / max(self.tracker_cfg.gating_doppler_mps, 1e-3)
            )
        if self.tracker_cfg.use_detection_class_in_cost:
            meas_class = meas.motion_hint
            if (
                track.classification != "unknown"
                and meas_class != "unknown"
                and meas_class != track.classification
            ):
                cost += 0.40
        return float(cost)

    def _update_track(self, track: Track, meas: Measurement, now_s: float) -> None:
        h_mat = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        r_var = max(self.tracker_cfg.measurement_noise_xy, 1e-3) ** 2
        r_mat = np.diag([r_var, r_var]).astype(np.float64, copy=False)
        z = np.array([meas.x_m, meas.y_m], dtype=np.float64)
        innovation = z - (h_mat @ track.state_vector)
        pht = track.covariance @ h_mat.T
        s_mat = h_mat @ pht + r_mat
        try:
            s_inv = np.linalg.inv(s_mat)
        except np.linalg.LinAlgError:
            s_inv = np.linalg.pinv(s_mat)
        k_mat = pht @ s_inv
        track.state_vector = track.state_vector + (k_mat @ innovation)
        i_mat = np.eye(4, dtype=np.float64)
        i_kh = i_mat - (k_mat @ h_mat)
        track.covariance = i_kh @ track.covariance @ i_kh.T + (k_mat @ r_mat @ k_mat.T)
        track.covariance = 0.5 * (track.covariance + track.covariance.T)
        track.hits += 1
        track.missed = 0
        track.last_update_time_s = float(now_s)
        self._refresh_track_measurement(track, meas)
        self._update_track_classification(track, meas)
        self._stabilize_static_track(track)
        if track.state == "tentative" and track.hits >= self.tracking_cfg.min_hits_to_confirm:
            track.state = "confirmed"

    def _mark_unmatched(self, track: Track) -> bool:
        max_age = int(max(0, self.tracking_cfg.max_track_age))
        if max_age > 0 and track.age > max_age:
            track.state = "deleted"
            return True
        if track.state == "tentative":
            if track.missed > int(self.tracking_cfg.max_missed_tentative):
                track.state = "deleted"
                return True
        else:
            if track.missed > int(self.tracking_cfg.max_missed_confirmed):
                track.state = "deleted"
                return True
        return False

    def _spawn_tracks(self, measurements: list[Measurement], unmatched_measurements: list[int], now_s: float) -> int:
        live_count = sum(1 for track in self.tracks if track.state != "deleted")
        available_slots = max(0, int(self.tracking_cfg.max_tracks) - live_count)
        if available_slots <= 0 or not unmatched_measurements:
            return 0
        ranked = sorted(
            unmatched_measurements,
            key=lambda meas_idx: measurements[meas_idx].power_lin,
            reverse=True,
        )
        births = 0
        for meas_idx in ranked[:available_slots]:
            meas = measurements[meas_idx]
            new_track = self._create_track(meas, now_s)
            self.tracks.append(new_track)
            self.next_track_id += 1
            births += 1
        return births

    def _create_track(self, meas: Measurement, now_s: float) -> Track:
        vx_init, vy_init = self._initial_velocity(meas)
        r_var = max(self.tracker_cfg.measurement_noise_xy, 1e-3) ** 2
        v_var = max(self.tracker_cfg.process_noise_vel, 0.25) ** 2
        classification, static_frames, dynamic_frames = self._initial_classification(meas)
        state = "confirmed" if self.tracking_cfg.min_hits_to_confirm <= 1 else "tentative"
        track = Track(
            track_id=int(self.next_track_id),
            state_vector=np.array([meas.x_m, meas.y_m, vx_init, vy_init], dtype=np.float64),
            covariance=np.diag([r_var, r_var, v_var, v_var]).astype(np.float64, copy=False),
            age=1,
            hits=1,
            missed=0,
            state=state,
            classification=classification,
            last_update_time_s=float(now_s),
            last_predict_dt_s=0.0,
            range_m=float(meas.range_m),
            angle_deg=float(meas.angle_deg),
            doppler_mps=meas.doppler_mps,
            last_detection_source=str(meas.source),
            history=deque(maxlen=max(1, int(self.tracker_cfg.history_len))),
            static_frames=static_frames,
            dynamic_frames=dynamic_frames,
        )
        self._refresh_track_geometry(track)
        return track

    def _initial_velocity(self, meas: Measurement) -> tuple[float, float]:
        if meas.doppler_mps is None or meas.range_m <= _EPS:
            return 0.0, 0.0
        scale = float(meas.doppler_mps) / max(float(meas.range_m), _EPS)
        return float(meas.x_m * scale), float(meas.y_m * scale)

    def _initial_classification(self, meas: Measurement) -> tuple[TrackClassification, int, int]:
        confirm_frames = max(1, int(self.tracker_cfg.classification_confirm_frames))
        if meas.motion_hint == "dynamic":
            return "dynamic", 0, confirm_frames
        if meas.motion_hint == "static":
            return "static", confirm_frames, 0
        return "unknown", 0, 0

    def _refresh_track_measurement(self, track: Track, meas: Measurement) -> None:
        track.doppler_mps = meas.doppler_mps
        track.last_detection_source = meas.source
        self._refresh_track_geometry(track)

    def _refresh_track_geometry(self, track: Track) -> None:
        track.range_m = float(math.hypot(track.x_m, track.y_m))
        track.angle_deg = float(math.degrees(math.atan2(track.x_m, max(track.y_m, _EPS))))
        if track.doppler_mps is None:
            track.doppler_mps = track.radial_velocity_mps

    def _append_history(self, track: Track) -> None:
        track.history.append((float(track.x_m), float(track.y_m)))

    def _expected_track_doppler(self, track: Track) -> float | None:
        return track.radial_velocity_mps

    def _update_track_classification(self, track: Track, meas: Measurement) -> None:
        confirm_frames = max(1, int(self.tracker_cfg.classification_confirm_frames))
        abs_doppler = None if meas.doppler_mps is None else abs(float(meas.doppler_mps))
        speed_mps = float(track.speed_mps)

        dynamic_candidate = bool(
            meas.motion_hint == "dynamic"
            or speed_mps >= self.tracker_cfg.dynamic_speed_threshold_mps
            or (abs_doppler is not None and abs_doppler >= self.tracker_cfg.dynamic_speed_threshold_mps)
        )
        static_candidate = bool(
            meas.motion_hint == "static"
            or (
                speed_mps <= self.tracker_cfg.static_speed_threshold_mps
                and (abs_doppler is None or abs_doppler <= self.tracker_cfg.doppler_static_threshold_mps)
            )
        )

        if dynamic_candidate and not static_candidate:
            track.dynamic_frames = min(confirm_frames, track.dynamic_frames + 1)
            track.static_frames = max(0, track.static_frames - 1)
        elif static_candidate and not dynamic_candidate:
            track.static_frames = min(confirm_frames, track.static_frames + 1)
            track.dynamic_frames = max(0, track.dynamic_frames - 1)
        elif dynamic_candidate and static_candidate:
            if speed_mps >= self.tracker_cfg.dynamic_speed_threshold_mps:
                track.dynamic_frames = min(confirm_frames, track.dynamic_frames + 1)
            else:
                track.static_frames = min(confirm_frames, track.static_frames + 1)
        else:
            track.dynamic_frames = max(0, track.dynamic_frames - 1)
            track.static_frames = max(0, track.static_frames - 1)

        if track.dynamic_frames >= confirm_frames:
            track.classification = "dynamic"
        elif track.static_frames >= confirm_frames:
            track.classification = "static"
        elif track.classification not in {"static", "dynamic"}:
            track.classification = "unknown"

    def _stabilize_static_track(self, track: Track) -> None:
        if track.classification != "static":
            return
        speed_mps = track.speed_mps
        if speed_mps <= self.tracker_cfg.static_speed_threshold_mps:
            track.vx_mps = 0.0
            track.vy_mps = 0.0
            track.doppler_mps = 0.0
        else:
            track.vx_mps *= 0.5
            track.vy_mps *= 0.5
            track.doppler_mps = track.radial_velocity_mps

    def _finalize_tracks(self) -> None:
        live_tracks: list[Track] = []
        for track in self.tracks:
            if track.state == "deleted":
                continue
            if track.doppler_mps is None:
                track.doppler_mps = track.radial_velocity_mps
            self._refresh_track_geometry(track)
            self._append_history(track)
            live_tracks.append(track)
        live_tracks.sort(key=lambda track: track.track_id)
        self.tracks = live_tracks
