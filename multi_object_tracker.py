"""Tracciamento multi-oggetto delle detection prodotte dal DSP realtime.

Il modulo separa le detection grezze dai track persistenti: predice lo stato
con un filtro di Kalman, associa le misure tramite gating spaziale/Doppler e
gestisce il ciclo di vita ``tentative -> confirmed -> deleted``.
"""

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
TrackMotionState = Literal["unknown", "moving", "stopped"]
MotionHint = Literal["unknown", "moving", "stopped"]

_EPS = 1e-6
_LARGE_COST = 1e6


class DetectionLike(Protocol):
    """Contratto minimo della detection consumata dal tracker.

    Le implementazioni concrete arrivano dal DSP; il protocollo evita che il
    tracker dipenda dalla loro specifica dataclass.
    """

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
    """Regole di vita dei track, indipendenti dal modello di moto."""

    enabled: bool = True
    dt_s: float | None = None
    max_tracks: int = 30
    min_hits_to_confirm: int = 3
    max_missed_tentative: int = 2
    max_missed_confirmed: int = 8
    max_track_age: int = 0


@dataclass(frozen=True)
class TrackerConfig:
    """Parametri del filtro di Kalman, del gating e della classificazione moto."""

    model: str = "kalman_cv_2d"
    gating_xy_m: float = 0.75
    gating_doppler_mps: float = 0.50
    process_noise_pos: float = 0.20
    process_noise_vel: float = 1.00
    measurement_noise_xy: float = 0.25
    moving_speed_threshold_mps: float = 0.20
    stopped_speed_threshold_mps: float = 0.08
    doppler_moving_threshold_mps: float = 0.12
    motion_confirm_frames_moving: int = 2
    motion_confirm_frames_stopped: int = 3
    stopped_memory_s: float = 3.0
    stopped_resume_gate_m: float = 0.90
    stop_position_alpha: float = 0.25
    birth_min_separation_m: float = 0.20
    use_doppler_in_cost: bool = True
    history_len: int = 12
    debug_log: bool = False


@dataclass
class Track:
    """Stato persistente di un bersaglio, con vettore ``[x, y, vx, vy]``.

    ``state_vector`` e ``covariance`` restano interni al filtro; le proprietà
    espongono le unità fisiche usate dal resto dell'applicazione.
    """

    track_id: int
    state_vector: np.ndarray = field(repr=False)
    covariance: np.ndarray = field(repr=False)
    age: int = 1
    hits: int = 1
    missed: int = 0
    state: TrackLifecycleState = "tentative"
    motion_state: TrackMotionState = "unknown"
    last_update_time_s: float | None = None
    last_predict_dt_s: float = 0.0
    range_m: float = 0.0
    angle_deg: float = 0.0
    doppler_mps: float | None = None
    last_detection_source: str | None = None
    history: deque[tuple[float, float]] = field(default_factory=deque, repr=False)
    moving_evidence: int = 0
    stopped_evidence: int = 0
    stop_x_m: float | None = None
    stop_y_m: float | None = None
    stop_timestamp_s: float | None = None
    last_motion_change_s: float | None = None

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

    @property
    def classification(self) -> str:
        # Compatibility alias: dynamic/static/unknown.
        if self.motion_state == "moving":
            return "dynamic"
        if self.motion_state == "stopped":
            return "static"
        return "unknown"

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
            "motion_state": str(self.motion_state),
            "state": str(self.state),
            "confirmed": bool(self.confirmed),
            "age": int(self.age),
            "hits": int(self.hits),
            "missed": int(self.missed),
            "stop_x_m": None if self.stop_x_m is None else float(self.stop_x_m),
            "stop_y_m": None if self.stop_y_m is None else float(self.stop_y_m),
            "stop_timestamp_s": None if self.stop_timestamp_s is None else float(self.stop_timestamp_s),
        }


@dataclass(frozen=True)
class Measurement:
    """Copia normalizzata di una detection valida, pronta per l'associazione."""

    x_m: float
    y_m: float
    range_m: float
    angle_deg: float
    doppler_mps: float | None
    power_lin: float
    power_db: float
    source: str
    motion_hint: MotionHint


@dataclass(frozen=True)
class TrackerDebugSnapshot:
    """Contatori dell'ultimo ciclo, utili per log e diagnostica della GUI."""

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


def _clamp(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


class MultiObjectTracker:
    """Tracker a velocità costante 2D con associazione one-to-one delle misure."""

    def __init__(self, tracking_cfg: TrackingConfig, tracker_cfg: TrackerConfig):
        self.tracking_cfg = tracking_cfg
        self.tracker_cfg = tracker_cfg
        self.tracks: list[Track] = []
        self.next_track_id = 1
        self._last_timestamp_s: float | None = None
        self._last_dt_s = self._fallback_dt_s()
        self._h_mat = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        self._i4 = np.eye(4, dtype=np.float64)
        cfg_dt_s = self.tracking_cfg.dt_s
        if cfg_dt_s is not None and math.isfinite(cfg_dt_s) and cfg_dt_s > 0.0:
            self._nominal_dt_s: float | None = float(cfg_dt_s)
        else:
            self._nominal_dt_s = None
        self._cached_predict_dt_s: float | None = None
        self._cached_f_mat: np.ndarray | None = None
        self._cached_q_mat: np.ndarray | None = None
        r_var = max(self.tracker_cfg.measurement_noise_xy, 1e-3) ** 2
        self._r_mat = np.diag([r_var, r_var]).astype(np.float64, copy=False)
        self._z_vec = np.zeros(2, dtype=np.float64)
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
        """Avanza il tracker di un frame e restituisce i track ancora vivi.

        L'ordine è intenzionale: prima predizione, poi associazione/aggiornamento,
        quindi gestione delle assenze e nascita di nuovi track.  In questo modo
        una detection non può aggiornare due track nello stesso frame.
        """
        if not self.tracking_cfg.enabled:
            self.reset()
            return []

        now_s = float(timestamp_s if timestamp_s is not None else time.perf_counter())
        dt_s = self._resolve_dt(now_s)
        measurements, invalid_count = self._sanitize_detections(detections)

        # La predizione porta tutti i track allo stesso istante della misura.
        self._predict_stage(dt_s, now_s)
        matches, unmatched_tracks, unmatched_measurements, assoc_debug = self._associate_stage(measurements, now_s)
        self._update_stage(matches, measurements, now_s)

        deleted_tracks = 0
        for track_idx in unmatched_tracks:
            if self._mark_unmatched(self.tracks[track_idx], now_s):
                deleted_tracks += 1

        births = self._allocate_stage(measurements, unmatched_measurements, now_s)
        self._finalize_tracks(now_s)

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
            stopped_tracks = sum(1 for track in self.tracks if track.motion_state == "stopped")
            moving_tracks = sum(1 for track in self.tracks if track.motion_state == "moving")
            print(
                "[TRACK] "
                f"tracks={dbg.tracks_after} moving={moving_tracks} stopped={stopped_tracks} "
                f"matches={dbg.matches} new={dbg.new_tracks} deleted={dbg.deleted_tracks} "
                f"det={dbg.detections_used}/{dbg.detections_in} "
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
        if self._last_timestamp_s is not None:
            dt_s = float(timestamp_s - self._last_timestamp_s)
        elif self._nominal_dt_s is not None:
            dt_s = float(self._nominal_dt_s)
        else:
            dt_s = self._last_dt_s
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            dt_s = self._fallback_dt_s()
        # Evita che un timestamp duplicato o una lunga pausa della GUI faccia
        # collassare o esplodere la covarianza del filtro a velocità costante.
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
        abs_doppler = None if doppler_mps is None else abs(float(doppler_mps))
        moving_hint = bool(
            source == "moving"
            or (
                abs_doppler is not None
                and abs_doppler >= max(self.tracker_cfg.doppler_moving_threshold_mps, self.tracker_cfg.moving_speed_threshold_mps)
            )
        )
        stopped_hint = bool(
            source == "static"
            or (
                abs_doppler is not None
                and abs_doppler <= max(self.tracker_cfg.stopped_speed_threshold_mps, 1e-3)
            )
        )
        motion_hint: MotionHint = "unknown"
        if moving_hint and not stopped_hint:
            motion_hint = "moving"
        elif stopped_hint and not moving_hint:
            motion_hint = "stopped"

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

    def _build_state_transition(self, dt_s: float) -> np.ndarray:
        # Modello CV per stato [x, y, vx, vy]: la posizione avanza di v*dt,
        # mentre la velocità resta costante fino alla prossima misura.
        return np.array(
            [
                [1.0, 0.0, dt_s, 0.0],
                [0.0, 1.0, 0.0, dt_s],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _get_predict_mats(self, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
        dt_use = float(dt_s)
        if (
            self._cached_f_mat is None
            or self._cached_q_mat is None
            or self._cached_predict_dt_s != dt_use
        ):
            self._cached_f_mat = self._build_state_transition(dt_use)
            self._cached_q_mat = self._build_process_noise(dt_use)
            self._cached_predict_dt_s = dt_use
        return self._cached_f_mat, self._cached_q_mat

    def _predict_stage(self, dt_s: float, now_s: float) -> None:
        f_mat, q_mat = self._get_predict_mats(dt_s)
        for track in self.tracks:
            self._predict_track(track, dt_s, now_s, f_mat, q_mat)

    def _predict_track(
        self,
        track: Track,
        dt_s: float,
        now_s: float,
        f_mat: np.ndarray,
        q_mat: np.ndarray,
    ) -> None:
        track.state_vector = f_mat @ track.state_vector
        track.covariance = f_mat @ track.covariance @ f_mat.T + q_mat
        track.covariance = 0.5 * (track.covariance + track.covariance.T)
        track.age += 1
        track.missed += 1
        track.last_predict_dt_s = float(dt_s)
        track.last_update_time_s = now_s if track.last_update_time_s is None else track.last_update_time_s
        if track.motion_state == "stopped":
            self._hold_track_at_stop(track)
        self._refresh_track_geometry(track)

    def _build_process_noise(self, dt_s: float) -> np.ndarray:
        dt2 = dt_s * dt_s
        dt3 = dt2 * dt_s
        dt4 = dt2 * dt2
        q_vel = max(self.tracker_cfg.process_noise_vel, 1e-3) ** 2
        q_pos = max(self.tracker_cfg.process_noise_pos, 1e-3) ** 2
        # Rumore di accelerazione discreto del modello CV più un floor
        # posizionale: il filtro può adattarsi a manovre e a misura rumorosa.
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

    def _associate_stage(
        self, measurements: list[Measurement], now_s: float
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

        # I pair fuori gate mantengono il costo sentinella e non potranno essere
        # selezionati né dall'algoritmo ungherese né dal fallback greedy.
        cost_matrix = np.full((n_tracks, n_meas), _LARGE_COST, dtype=np.float64)
        reject_xy = 0
        reject_doppler = 0
        for track_idx, track in enumerate(self.tracks):
            gate_xy = self._track_gate_xy(track)
            track_doppler = self._expected_track_doppler(track)
            for meas_idx, meas in enumerate(measurements):
                d_xy = float(math.hypot(meas.x_m - track.x_m, meas.y_m - track.y_m))
                if d_xy > gate_xy:
                    reject_xy += 1
                    continue

                d_stop: float | None = None
                stopped_resume_candidate = False
                if track.motion_state == "stopped" and track.stop_x_m is not None and track.stop_y_m is not None:
                    d_stop = float(math.hypot(meas.x_m - track.stop_x_m, meas.y_m - track.stop_y_m))
                    if d_stop > max(gate_xy, float(self.tracker_cfg.stopped_resume_gate_m)):
                        reject_xy += 1
                        continue
                    stopped_resume_candidate = bool(
                        meas.motion_hint == "moving"
                        or (
                            meas.doppler_mps is not None
                            and abs(float(meas.doppler_mps))
                            >= float(self.tracker_cfg.doppler_moving_threshold_mps)
                        )
                    )

                if (
                    not stopped_resume_candidate
                    and self.tracker_cfg.gating_doppler_mps > 0.0
                    and meas.doppler_mps is not None
                    and track_doppler is not None
                    and abs(float(meas.doppler_mps) - float(track_doppler)) > self.tracker_cfg.gating_doppler_mps
                ):
                    reject_doppler += 1
                    continue

                cost_matrix[track_idx, meas_idx] = self._association_cost(
                    track,
                    meas,
                    d_xy=d_xy,
                    gate_xy=gate_xy,
                    track_doppler=track_doppler,
                    d_stop=d_stop,
                )

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
        *,
        d_xy: float,
        gate_xy: float,
        track_doppler: float | None,
        d_stop: float | None,
    ) -> float:
        cost = d_xy / max(gate_xy, 1e-3)
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
        if d_stop is not None:
            cost += 0.20 * (d_stop / max(self.tracker_cfg.stopped_resume_gate_m, 1e-3))
        if track.motion_state == "moving" and meas.motion_hint == "stopped":
            cost += 0.05
        return float(cost)

    def _update_stage(
        self,
        matches: list[tuple[int, int]],
        measurements: list[Measurement],
        now_s: float,
    ) -> None:
        for track_idx, meas_idx in matches:
            self._update_track(self.tracks[track_idx], measurements[meas_idx], now_s)

    def _update_track(self, track: Track, meas: Measurement, now_s: float) -> None:
        z = self._z_vec
        z[0] = float(meas.x_m)
        z[1] = float(meas.y_m)
        innovation = z - (self._h_mat @ track.state_vector)
        pht = track.covariance @ self._h_mat.T
        s_mat = self._h_mat @ pht + self._r_mat
        try:
            s_inv = np.linalg.inv(s_mat)
        except np.linalg.LinAlgError:
            s_inv = np.linalg.pinv(s_mat)
        k_mat = pht @ s_inv
        track.state_vector = track.state_vector + (k_mat @ innovation)
        i_kh = self._i4 - (k_mat @ self._h_mat)
        # Forma di Joseph: mantiene la covarianza simmetrica e semidefinita
        # positiva anche con errori numerici delle matrici in float64.
        track.covariance = i_kh @ track.covariance @ i_kh.T + (k_mat @ self._r_mat @ k_mat.T)
        track.covariance = 0.5 * (track.covariance + track.covariance.T)
        track.hits += 1
        track.missed = 0
        track.last_update_time_s = float(now_s)
        self._refresh_track_measurement(track, meas)
        self._update_motion_state(track, meas, now_s)
        if track.state == "tentative" and track.hits >= self.tracking_cfg.min_hits_to_confirm:
            track.state = "confirmed"

    def _mark_unmatched(self, track: Track, now_s: float) -> bool:
        """Applica la politica di sopravvivenza a un track senza misura associata."""
        max_age = int(max(0, self.tracking_cfg.max_track_age))
        if max_age > 0 and track.age > max_age:
            track.state = "deleted"
            return True

        if track.state == "tentative":
            if track.missed > int(self.tracking_cfg.max_missed_tentative):
                track.state = "deleted"
                return True
            return False

        self._decay_unmatched_track_motion(track, now_s)

        max_missed = int(self.tracking_cfg.max_missed_confirmed)
        if track.motion_state == "stopped":
            max_missed = max(max_missed, self._stopped_hold_frames())
            if track.stop_timestamp_s is None:
                track.stop_timestamp_s = float(now_s)
            self._hold_track_at_stop(track)

        if track.missed > max_missed:
            track.state = "deleted"
            return True
        return False

    def _decay_unmatched_track_motion(self, track: Track, now_s: float) -> None:
        if track.motion_state == "stopped":
            return

        # In moving-only mode a target can disappear as soon as it reaches near-zero Doppler.
        # Decay the latent velocity on unmatched confirmed tracks so they can still settle to
        # the stopped state and use the stopped-memory hold instead of being deleted early.
        track.vx_mps *= 0.35
        track.vy_mps *= 0.35

        stop_confirm = max(1, int(self.tracker_cfg.motion_confirm_frames_stopped))
        speed_mps = float(track.speed_mps)
        if speed_mps <= float(self.tracker_cfg.stopped_speed_threshold_mps):
            track.stopped_evidence = min(stop_confirm, track.stopped_evidence + 1)
            track.moving_evidence = max(0, track.moving_evidence - 1)
        else:
            track.moving_evidence = max(0, track.moving_evidence - 1)

        if track.stopped_evidence >= stop_confirm:
            prev_state = track.motion_state
            track.motion_state = "stopped"
            if prev_state != "stopped":
                track.last_motion_change_s = float(now_s)
                self._enter_stopped_state(track, now_s)
            else:
                self._update_stop_anchor(track, now_s)

    def _stopped_hold_frames(self) -> int:
        hold_s = max(0.0, float(self.tracker_cfg.stopped_memory_s))
        if hold_s <= 0.0:
            return int(max(0, self.tracking_cfg.max_missed_confirmed))
        return int(max(0, round(hold_s / max(self._last_dt_s, 1e-3))))

    def _allocate_stage(self, measurements: list[Measurement], unmatched_measurements: list[int], now_s: float) -> int:
        live_count = sum(1 for track in self.tracks if track.state != "deleted")
        available_slots = max(0, int(self.tracking_cfg.max_tracks) - live_count)
        if available_slots <= 0 or not unmatched_measurements:
            return 0

        # In caso di capacità limitata, privilegia prima i target mobili e poi
        # le detection più energetiche: i nuovi track meno affidabili restano
        # fuori finché non si libera uno slot.
        ranked = sorted(
            unmatched_measurements,
            key=lambda meas_idx: (
                measurements[meas_idx].source != "moving",
                -float(measurements[meas_idx].power_lin),
            ),
        )
        births = 0
        for meas_idx in ranked:
            if births >= available_slots:
                break
            meas = measurements[meas_idx]
            if self._is_too_close_to_live_track(meas):
                continue
            self.tracks.append(self._create_track(meas, now_s))
            self.next_track_id += 1
            births += 1
        return births

    def _is_too_close_to_live_track(self, meas: Measurement) -> bool:
        min_sep = max(0.0, float(self.tracker_cfg.birth_min_separation_m))
        if min_sep <= 0.0:
            return False
        for track in self.tracks:
            if track.state == "deleted":
                continue
            tx = float(track.x_m)
            ty = float(track.y_m)
            if track.motion_state == "stopped" and track.stop_x_m is not None and track.stop_y_m is not None:
                tx = float(track.stop_x_m)
                ty = float(track.stop_y_m)
            if math.hypot(meas.x_m - tx, meas.y_m - ty) < min_sep:
                return True
        return False

    def _create_track(self, meas: Measurement, now_s: float) -> Track:
        vx_init, vy_init = self._initial_velocity(meas)
        r_var = max(self.tracker_cfg.measurement_noise_xy, 1e-3) ** 2
        v_var = max(self.tracker_cfg.process_noise_vel, 0.25) ** 2

        move_confirm = max(1, int(self.tracker_cfg.motion_confirm_frames_moving))
        stop_confirm = max(1, int(self.tracker_cfg.motion_confirm_frames_stopped))
        moving_evidence = move_confirm if meas.motion_hint == "moving" and move_confirm <= 1 else 0
        stopped_evidence = stop_confirm if meas.motion_hint == "stopped" and stop_confirm <= 1 else 0
        motion_state: TrackMotionState = "unknown"
        if moving_evidence >= move_confirm:
            motion_state = "moving"
        elif stopped_evidence >= stop_confirm:
            motion_state = "stopped"

        state = "confirmed" if self.tracking_cfg.min_hits_to_confirm <= 1 else "tentative"
        track = Track(
            track_id=int(self.next_track_id),
            state_vector=np.array([meas.x_m, meas.y_m, vx_init, vy_init], dtype=np.float64),
            covariance=np.diag([r_var, r_var, v_var, v_var]).astype(np.float64, copy=False),
            age=1,
            hits=1,
            missed=0,
            state=state,
            motion_state=motion_state,
            last_update_time_s=float(now_s),
            last_predict_dt_s=0.0,
            range_m=float(meas.range_m),
            angle_deg=float(meas.angle_deg),
            doppler_mps=meas.doppler_mps,
            last_detection_source=str(meas.source),
            history=deque(maxlen=max(1, int(self.tracker_cfg.history_len))),
            moving_evidence=moving_evidence,
            stopped_evidence=stopped_evidence,
            stop_x_m=None,
            stop_y_m=None,
            stop_timestamp_s=None,
            last_motion_change_s=float(now_s),
        )
        if track.motion_state == "stopped":
            track.stop_x_m = float(track.x_m)
            track.stop_y_m = float(track.y_m)
            track.stop_timestamp_s = float(now_s)
            self._hold_track_at_stop(track)
        self._refresh_track_geometry(track)
        return track

    def _initial_velocity(self, meas: Measurement) -> tuple[float, float]:
        # Il Doppler non determina la componente tangenziale, ma fornisce una
        # stima iniziale affidabile della componente radiale. Inizializzarla a
        # zero rendeva il gate Doppler incoerente già dal secondo/terzo frame.
        if meas.doppler_mps is None or not math.isfinite(float(meas.doppler_mps)):
            return 0.0, 0.0
        range_m = float(math.hypot(meas.x_m, meas.y_m))
        if range_m <= _EPS:
            return 0.0, 0.0
        radial = float(meas.doppler_mps)
        return radial * float(meas.x_m) / range_m, radial * float(meas.y_m) / range_m

    def _refresh_track_measurement(self, track: Track, meas: Measurement) -> None:
        track.doppler_mps = meas.doppler_mps
        track.last_detection_source = meas.source
        self._refresh_track_geometry(track)

    def _refresh_track_geometry(self, track: Track) -> None:
        track.range_m = float(math.hypot(track.x_m, track.y_m))
        track.angle_deg = float(math.degrees(math.atan2(track.x_m, max(track.y_m, _EPS))))
        if track.motion_state == "stopped":
            track.doppler_mps = 0.0
        elif track.doppler_mps is None:
            track.doppler_mps = track.radial_velocity_mps

    def _append_history(self, track: Track) -> None:
        track.history.append((float(track.x_m), float(track.y_m)))

    def _expected_track_doppler(self, track: Track) -> float | None:
        if track.motion_state == "stopped":
            return 0.0
        # Il filtro cartesiano assimila solo x/y; la sua velocità radiale può
        # quindi convergere più lentamente del Doppler misurato. Per il gate
        # duro usiamo l'ultima misura Doppler valida e lasciamo la velocità del
        # Kalman come fallback.
        if track.doppler_mps is not None:
            return float(track.doppler_mps)
        return track.radial_velocity_mps

    def _update_motion_state(self, track: Track, meas: Measurement, now_s: float) -> None:
        """Conferma i passaggi moving/stopped con evidenza su più frame.

        La doppia soglia e i contatori evitano che rumore Doppler o una singola
        detection statica facciano oscillare la classificazione.
        """
        move_confirm = max(1, int(self.tracker_cfg.motion_confirm_frames_moving))
        stop_confirm = max(1, int(self.tracker_cfg.motion_confirm_frames_stopped))
        speed_mps = float(track.speed_mps)
        abs_doppler = None if meas.doppler_mps is None else abs(float(meas.doppler_mps))
        explicit_static_observation = bool(meas.motion_hint == "stopped" and meas.source != "moving")

        moving_candidate = bool(
            meas.motion_hint == "moving"
            or speed_mps >= self.tracker_cfg.moving_speed_threshold_mps
            or (abs_doppler is not None and abs_doppler >= self.tracker_cfg.doppler_moving_threshold_mps)
        )
        stopped_candidate = bool(
            meas.motion_hint == "stopped"
            or (
                speed_mps <= self.tracker_cfg.stopped_speed_threshold_mps
                and (abs_doppler is None or abs_doppler <= self.tracker_cfg.stopped_speed_threshold_mps)
            )
        )

        if explicit_static_observation:
            # When the static branch repeatedly sees the same target, damp velocity quickly
            # so the track can transition to stopped instead of lingering in moving.
            track.vx_mps *= 0.5
            track.vy_mps *= 0.5
            speed_mps = float(track.speed_mps)
            moving_candidate = bool(
                (abs_doppler is not None and abs_doppler >= (1.25 * self.tracker_cfg.doppler_moving_threshold_mps))
                and speed_mps >= (1.25 * self.tracker_cfg.moving_speed_threshold_mps)
            )
            stopped_candidate = True

        if moving_candidate and not stopped_candidate:
            track.moving_evidence = min(move_confirm, track.moving_evidence + 1)
            track.stopped_evidence = max(0, track.stopped_evidence - 1)
        elif stopped_candidate and not moving_candidate:
            track.stopped_evidence = min(stop_confirm, track.stopped_evidence + 1)
            track.moving_evidence = max(0, track.moving_evidence - 1)
        elif moving_candidate and stopped_candidate:
            if (not explicit_static_observation) and speed_mps >= self.tracker_cfg.moving_speed_threshold_mps:
                track.moving_evidence = min(move_confirm, track.moving_evidence + 1)
                track.stopped_evidence = max(0, track.stopped_evidence - 1)
            else:
                track.stopped_evidence = min(stop_confirm, track.stopped_evidence + 1)
                track.moving_evidence = max(0, track.moving_evidence - 1)
        else:
            track.moving_evidence = max(0, track.moving_evidence - 1)
            track.stopped_evidence = max(0, track.stopped_evidence - 1)

        prev_state = track.motion_state
        if track.moving_evidence >= move_confirm:
            track.motion_state = "moving"
        elif track.stopped_evidence >= stop_confirm:
            track.motion_state = "stopped"
        elif track.motion_state not in {"moving", "stopped"}:
            track.motion_state = "unknown"

        if track.motion_state != prev_state:
            track.last_motion_change_s = float(now_s)
            if track.motion_state == "stopped":
                self._enter_stopped_state(track, now_s)
            elif track.motion_state == "moving":
                if track.doppler_mps is None:
                    track.doppler_mps = track.radial_velocity_mps
        elif track.motion_state == "stopped":
            self._update_stop_anchor(track, now_s)

    def _enter_stopped_state(self, track: Track, now_s: float) -> None:
        alpha = _clamp(float(self.tracker_cfg.stop_position_alpha), 0.0, 1.0)
        if track.stop_x_m is None or track.stop_y_m is None:
            track.stop_x_m = float(track.x_m)
            track.stop_y_m = float(track.y_m)
        else:
            track.stop_x_m = float((1.0 - alpha) * track.stop_x_m + alpha * track.x_m)
            track.stop_y_m = float((1.0 - alpha) * track.stop_y_m + alpha * track.y_m)
        track.stop_timestamp_s = float(now_s)
        self._hold_track_at_stop(track)

    def _update_stop_anchor(self, track: Track, now_s: float) -> None:
        alpha = _clamp(float(self.tracker_cfg.stop_position_alpha), 0.0, 1.0)
        if track.stop_x_m is None or track.stop_y_m is None:
            track.stop_x_m = float(track.x_m)
            track.stop_y_m = float(track.y_m)
        else:
            track.stop_x_m = float((1.0 - alpha) * track.stop_x_m + alpha * track.x_m)
            track.stop_y_m = float((1.0 - alpha) * track.stop_y_m + alpha * track.y_m)
        track.stop_timestamp_s = float(now_s)
        self._hold_track_at_stop(track)

    def _hold_track_at_stop(self, track: Track) -> None:
        if track.stop_x_m is not None and track.stop_y_m is not None:
            track.x_m = float(track.stop_x_m)
            track.y_m = float(track.stop_y_m)
        track.vx_mps = 0.0
        track.vy_mps = 0.0
        track.doppler_mps = 0.0

    def _track_gate_xy(self, track: Track) -> float:
        gate = max(self.tracker_cfg.gating_xy_m, 1e-3)
        if track.state == "tentative":
            gate *= 1.15
        if track.motion_state == "stopped":
            gate = max(gate, self.tracker_cfg.stopped_resume_gate_m)
        return float(gate)

    def _finalize_tracks(self, now_s: float) -> None:
        live_tracks: list[Track] = []
        for track in self.tracks:
            if track.state == "deleted":
                continue
            if track.motion_state == "stopped":
                if track.stop_timestamp_s is None:
                    track.stop_timestamp_s = float(now_s)
                self._hold_track_at_stop(track)
            elif track.doppler_mps is None:
                track.doppler_mps = track.radial_velocity_mps
            self._refresh_track_geometry(track)
            self._append_history(track)
            live_tracks.append(track)
        live_tracks.sort(key=lambda track: track.track_id)
        self.tracks = live_tracks
