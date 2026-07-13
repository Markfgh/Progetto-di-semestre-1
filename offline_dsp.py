from __future__ import annotations

from dataclasses import dataclass
import time
import warnings
from typing import Any, Literal

import numpy as np

AvgMode = Literal["none", "loop", "frame", "both"]
BpMode = Literal["sar_only", "mimo_sar"]
MotionMode = Literal["static_zero_doppler"]

_AVG_MODES = {"none", "loop", "frame", "both"}
_BP_MODES = {"sar_only", "mimo_sar"}
_MOTION_MODES = {"static_zero_doppler"}


@dataclass(frozen=True)
class MimoBackProjectionPlanEntry:
    pos_i: int
    ant_i: int
    valid_idx: np.ndarray
    bin_float: np.ndarray
    coherent_phase: np.ndarray


@dataclass(frozen=True)
class MimoBackProjectionPlan:
    image_shape: tuple[int, int]
    n_pos: int
    n_ant: int
    max_bin_eff: int
    dr_m: float
    fc_hz: float
    c_m_s: float
    phase_sign: int
    entries: tuple[MimoBackProjectionPlanEntry, ...]


@dataclass(frozen=True)
class SyntheticApertureData:
    snapshot_cube: np.ndarray
    x_position_m: np.ndarray
    x_phase_center_m: np.ndarray
    x_element_m: np.ndarray


def avg_mode_normalize(mode: str | None) -> AvgMode:
    mode_s = str(mode or "both").strip().lower()
    if mode_s not in _AVG_MODES:
        return "both"
    return mode_s  # type: ignore[return-value]


def bp_mode_normalize(mode: str | None) -> BpMode:
    mode_s = str(mode or "sar_only").strip().lower()
    if mode_s not in _BP_MODES:
        return "sar_only"
    return mode_s  # type: ignore[return-value]


def motion_mode_normalize(mode: str | None) -> MotionMode:
    mode_s = str(mode or "static_zero_doppler").strip().lower()
    if mode_s not in _MOTION_MODES:
        raise ValueError(
            "motion_mode non valido per mimo_sar: "
            f"{mode!r}. Valore supportato: 'static_zero_doppler'."
        )
    return mode_s  # type: ignore[return-value]


def phase_sign_normalize(value: Any, *, field_name: str = "phase_sign") -> int:
    try:
        phase_sign_i = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} non valido: {value!r}") from exc
    if phase_sign_i not in (-1, 1):
        raise ValueError(f"{field_name} deve essere +1 o -1, trovato: {value!r}")
    return int(phase_sign_i)


def _image_to_db(img_lin: np.ndarray) -> np.ndarray:
    out = np.asarray(img_lin, dtype=np.float32).copy()
    np.add(out, np.float32(1e-6), out=out)
    np.log10(out, out=out)
    out *= np.float32(20.0)
    return out.astype(np.float32, copy=False)


def power_image_to_db(img_power: np.ndarray) -> np.ndarray:
    out = np.asarray(img_power, dtype=np.float32).copy()
    np.add(out, np.float32(1e-12), out=out)
    np.log10(out, out=out)
    out *= np.float32(10.0)
    return out.astype(np.float32, copy=False)


def build_virtual_array_x_offsets(
    virtual_antennas: int,
    *,
    pitch_m: float,
    use_virtual_antennas: bool = True,
) -> np.ndarray:
    """Deprecated legacy phase-center helper.

    MIMO-SAR processing uses the physical bistatic TX/RX geometry from
    build_mimo_geometry(); this helper is kept only for older SAR-only callers.
    """
    warnings.warn(
        "build_virtual_array_x_offsets is deprecated for MIMO-SAR; use build_mimo_geometry "
        "and phase centers 0.5 * (x_tx + x_rx) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    n_ant = int(virtual_antennas)
    if n_ant <= 0:
        raise ValueError(f"virtual_antennas deve essere > 0, trovato: {virtual_antennas!r}")
    if not bool(use_virtual_antennas):
        return np.zeros(n_ant, dtype=np.float32)

    pitch = float(pitch_m)
    if pitch <= 0.0:
        raise ValueError(f"pitch_m deve essere > 0, trovato: {pitch_m!r}")

    idx = np.arange(n_ant, dtype=np.float32) - np.float32(0.5 * (n_ant - 1))
    return (idx * np.float32(pitch)).astype(np.float32, copy=False)


def reduce_avg_mode(range_fft: np.ndarray, avg_mode: AvgMode) -> np.ndarray:
    if range_fft.ndim == 4:
        if avg_mode == "both":
            out = range_fft.mean(axis=(1, 2), dtype=np.complex64)
            return out[:, np.newaxis, :].astype(np.complex64, copy=False)
        if avg_mode == "frame":
            out = range_fft.mean(axis=1, dtype=np.complex64)
            return out.astype(np.complex64, copy=False)
        if avg_mode == "loop":
            out = range_fft.mean(axis=2, dtype=np.complex64)
            return out.astype(np.complex64, copy=False)

        n_pos, n_frames, n_loops, n_bins = range_fft.shape
        return range_fft.reshape(n_pos, n_frames * n_loops, n_bins)

    if range_fft.ndim == 5:
        if avg_mode == "both":
            out = range_fft.mean(axis=(1, 2), dtype=np.complex64)
            return out[:, np.newaxis, :, :].astype(np.complex64, copy=False)
        if avg_mode == "frame":
            out = range_fft.mean(axis=1, dtype=np.complex64)
            return out.astype(np.complex64, copy=False)
        if avg_mode == "loop":
            out = range_fft.mean(axis=2, dtype=np.complex64)
            return out.astype(np.complex64, copy=False)

        n_pos, n_frames, n_loops, n_ant, n_bins = range_fft.shape
        return range_fft.reshape(n_pos, n_frames * n_loops, n_ant, n_bins)

    raise ValueError(f"range_fft shape non valido: {range_fft.shape!r}")


def _parse_offsets(
    value: Any,
    *,
    expected_len: int,
    field_name: str,
) -> np.ndarray:
    if value is None:
        raise ValueError(f"{field_name} mancante")
    try:
        arr = np.asarray(list(value), dtype=np.float32).reshape(-1)
    except Exception as exc:
        raise ValueError(f"{field_name} non valido: {value!r}") from exc
    if arr.size != int(expected_len):
        raise ValueError(f"{field_name} size={arr.size}, atteso {expected_len}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{field_name} contiene valori non finiti")
    return arr.astype(np.float32, copy=False)


def build_mimo_geometry(
    n_tx: int,
    n_rx: int,
    fc_hz: float,
    c_m_s: float,
    tx_offsets_lambda: Any = None,
    rx_offsets_lambda: Any = None,
    tx_offsets_m: Any = None,
    rx_offsets_m: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_tx_i = int(n_tx)
    n_rx_i = int(n_rx)
    if n_tx_i <= 0 or n_rx_i <= 0:
        raise ValueError(f"n_tx e n_rx devono essere > 0, trovati {n_tx!r}/{n_rx!r}")

    fc_f = float(fc_hz)
    c_f = float(c_m_s)
    if not np.isfinite(fc_f) or fc_f <= 0.0:
        raise ValueError(f"fc_hz non valido: {fc_hz!r}")
    if not np.isfinite(c_f) or c_f <= 0.0:
        raise ValueError(f"c_m_s non valido: {c_m_s!r}")
    wavelength_m = np.float32(c_f / fc_f)

    has_lambda_override = tx_offsets_lambda is not None or rx_offsets_lambda is not None
    has_meter_override = tx_offsets_m is not None or rx_offsets_m is not None
    if has_lambda_override and has_meter_override:
        raise ValueError("Usa override in lambda oppure in metri, non entrambi")

    if has_meter_override:
        if tx_offsets_m is None or rx_offsets_m is None:
            raise ValueError("tx_offsets_m e rx_offsets_m devono essere entrambi presenti")
        tx_base_m = _parse_offsets(tx_offsets_m, expected_len=n_tx_i, field_name="tx_offsets_m")
        rx_base_m = _parse_offsets(rx_offsets_m, expected_len=n_rx_i, field_name="rx_offsets_m")
    elif has_lambda_override:
        if tx_offsets_lambda is None or rx_offsets_lambda is None:
            raise ValueError("tx_offsets_lambda e rx_offsets_lambda devono essere entrambi presenti")
        tx_base_lambda = _parse_offsets(tx_offsets_lambda, expected_len=n_tx_i, field_name="tx_offsets_lambda")
        rx_base_lambda = _parse_offsets(rx_offsets_lambda, expected_len=n_rx_i, field_name="rx_offsets_lambda")
        tx_base_m = (tx_base_lambda * wavelength_m).astype(np.float32, copy=False)
        rx_base_m = (rx_base_lambda * wavelength_m).astype(np.float32, copy=False)
    else:
        if n_tx_i != 2 or n_rx_i != 4:
            raise ValueError(
                "build_mimo_geometry richiede override tx/rx per configurazioni diverse "
                f"dal default IWR1443BOOST 2Tx/4Rx (trovati n_tx={n_tx_i}, n_rx={n_rx_i})"
            )
        tx_base_m = (np.asarray([0.0, 2.0], dtype=np.float32) * wavelength_m).astype(np.float32, copy=False)
        rx_base_m = (np.asarray([0.0, 0.5, 1.0, 1.5], dtype=np.float32) * wavelength_m).astype(np.float32, copy=False)

    tx_idx = np.repeat(np.arange(n_tx_i, dtype=np.int32), n_rx_i)
    rx_idx = np.tile(np.arange(n_rx_i, dtype=np.int32), n_tx_i)
    x_tx_ant_m = tx_base_m[tx_idx].astype(np.float32, copy=False)
    x_rx_ant_m = rx_base_m[rx_idx].astype(np.float32, copy=False)

    center_m = (np.mean((x_tx_ant_m + x_rx_ant_m).astype(np.float32, copy=False), dtype=np.float32) * np.float32(0.5)).astype(np.float32, copy=False)
    x_tx_ant_m = (x_tx_ant_m - center_m).astype(np.float32, copy=False)
    x_rx_ant_m = (x_rx_ant_m - center_m).astype(np.float32, copy=False)
    return x_tx_ant_m, x_rx_ant_m


def _interp_linear_complex(spec: np.ndarray, b_chunk: np.ndarray, max_bin_local: int) -> np.ndarray:
    b0_chunk = np.floor(b_chunk).astype(np.int32, copy=False)
    np.clip(b0_chunk, 0, max_bin_local - 2, out=b0_chunk)
    frac = (b_chunk - b0_chunk.astype(np.float32, copy=False)).astype(np.float32, copy=False)
    s0 = spec[b0_chunk]
    s1 = spec[b0_chunk + 1]
    return (s0 + (s1 - s0) * frac).astype(np.complex64, copy=False)


def _interp_cubic_complex(spec: np.ndarray, b_chunk: np.ndarray, max_bin_local: int) -> np.ndarray:
    out = _interp_linear_complex(spec, b_chunk, max_bin_local)
    if max_bin_local < 4 or out.size <= 0:
        return out

    b0_chunk = np.floor(b_chunk).astype(np.int32, copy=False)
    cubic_mask = (b0_chunk >= 1) & (b0_chunk <= (max_bin_local - 3))
    if not np.any(cubic_mask):
        return out

    idx = np.flatnonzero(cubic_mask)
    if idx.size <= 0:
        return out

    b0_valid = b0_chunk[idx]
    frac = (b_chunk[idx] - b0_valid.astype(np.float32, copy=False)).astype(np.float32, copy=False)

    p0 = spec[b0_valid - 1]
    p1 = spec[b0_valid]
    p2 = spec[b0_valid + 1]
    p3 = spec[b0_valid + 2]

    frac2 = (frac * frac).astype(np.float32, copy=False)
    frac3 = (frac2 * frac).astype(np.float32, copy=False)

    cubic = (
        np.complex64(0.5 + 0.0j)
        * (
            np.complex64(2.0 + 0.0j) * p1
            + (-p0 + p2) * frac
            + (
                np.complex64(2.0 + 0.0j) * p0
                - np.complex64(5.0 + 0.0j) * p1
                + np.complex64(4.0 + 0.0j) * p2
                - p3
            )
            * frac2
            + (-p0 + np.complex64(3.0 + 0.0j) * p1 - np.complex64(3.0 + 0.0j) * p2 + p3) * frac3
        )
    ).astype(np.complex64, copy=False)
    out[idx] = cubic
    return out.astype(np.complex64, copy=False)


def _build_doppler_window(window_doppler: np.ndarray | None, n_loops: int) -> np.ndarray:
    if window_doppler is None:
        return np.ones(int(n_loops), dtype=np.float32)
    win = np.asarray(window_doppler, dtype=np.float32).reshape(-1)
    if win.size != int(n_loops):
        raise ValueError(f"window_doppler size={win.size}, atteso {n_loops}")
    if not np.all(np.isfinite(win)):
        raise ValueError("window_doppler contiene valori non finiti")
    return win.astype(np.float32, copy=False)


def _build_doppler_bin_cycles_axis(n_doppler: int, *, doppler_fft_shift: bool = True) -> np.ndarray:
    n_doppler_i = max(1, int(n_doppler))
    doppler_cycles = np.fft.fftfreq(n_doppler_i, d=1.0).astype(np.float32, copy=False)
    if doppler_fft_shift:
        doppler_cycles = np.fft.fftshift(doppler_cycles)
    return doppler_cycles.astype(np.float32, copy=False)


def _apply_tdm_post_fft_compensation(doppler_cube: np.ndarray, n_tx: int) -> np.ndarray:
    """Compensate per-TX TDM phase after the loop FFT.

    This keeps static energy at zero-Doppler and only aligns TX phases for each
    already-estimated Doppler bin. Input shape: [pos, frames, doppler, tx, rx, bins].
    """
    n_doppler = int(doppler_cube.shape[2])
    n_tx_i = int(n_tx)
    if n_doppler <= 0 or n_tx_i <= 1:
        return doppler_cube.astype(np.complex64, copy=False)

    out = np.asarray(doppler_cube, dtype=np.complex64)
    doppler_cycles = _build_doppler_bin_cycles_axis(n_doppler, doppler_fft_shift=True)
    tx_delay_in_loops = (np.arange(n_tx_i, dtype=np.float32) / np.float32(n_tx_i)).astype(
        np.float32,
        copy=False,
    )
    phase = np.exp(
        (-1j * np.float32(2.0 * np.pi))
        * doppler_cycles[:, None].astype(np.float64, copy=False)
        * tx_delay_in_loops[None, :].astype(np.float64, copy=False)
    ).astype(np.complex64, copy=False)
    phase[:, 0] = np.complex64(1.0 + 0.0j)
    out *= phase.reshape(1, 1, n_doppler, n_tx_i, 1, 1)
    return out.astype(np.complex64, copy=False)


def build_mimo_back_projection_plan(
    x_pos_m: np.ndarray,
    x_tx_ant_m: np.ndarray,
    x_rx_ant_m: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    dr_m: float,
    fc_hz: float,
    c_m_s: float,
    max_bin: int,
    phase_sign: int = -1,
) -> MimoBackProjectionPlan:
    if x_pos_m.ndim != 1:
        raise ValueError(f"x_pos_m shape non valido: {x_pos_m.shape!r}")
    if x_tx_ant_m.ndim != 1 or x_rx_ant_m.ndim != 1:
        raise ValueError("x_tx_ant_m e x_rx_ant_m devono essere vettori 1D")
    if x_tx_ant_m.size != x_rx_ant_m.size:
        raise ValueError("x_tx_ant_m e x_rx_ant_m devono avere stessa size")
    if x_grid.shape != y_grid.shape:
        raise ValueError("x_grid e y_grid devono avere stessa shape")
    if float(dr_m) <= 0.0:
        raise ValueError("dr_m deve essere > 0")

    phase_sign_i = phase_sign_normalize(phase_sign, field_name="phase_sign")
    n_pos = int(x_pos_m.size)
    n_ant = int(x_tx_ant_m.size)
    max_bin_eff = max(0, int(max_bin))
    if n_pos <= 0:
        raise ValueError("Nessuna posizione selezionata per back projection")
    if n_ant <= 0:
        raise ValueError("Nessuna antenna selezionata per back projection")
    if max_bin_eff < 2:
        return MimoBackProjectionPlan(
            image_shape=tuple(int(v) for v in x_grid.shape),
            n_pos=n_pos,
            n_ant=n_ant,
            max_bin_eff=max_bin_eff,
            dr_m=float(dr_m),
            fc_hz=float(fc_hz),
            c_m_s=float(c_m_s),
            phase_sign=int(phase_sign_i),
            entries=(),
        )

    x_flat = x_grid.reshape(-1).astype(np.float32, copy=False)
    y_flat = y_grid.reshape(-1).astype(np.float32, copy=False)
    y_sq = (y_flat * y_flat).astype(np.float32, copy=False)
    k = np.float32((2.0 * np.pi * float(fc_hz)) / float(c_m_s))
    phase_scale = np.float32(float(phase_sign_i)) * k
    inv_dr = np.float32(1.0 / float(dr_m))
    entries: list[MimoBackProjectionPlanEntry] = []

    for pos_i in range(n_pos):
        x_pos = np.float32(x_pos_m[pos_i])
        for ant_i in range(n_ant):
            x_tx = x_pos + np.float32(x_tx_ant_m[ant_i])
            x_rx = x_pos + np.float32(x_rx_ant_m[ant_i])

            dx_tx = x_flat - x_tx
            dx_rx = x_flat - x_rx
            r_tx = np.sqrt(dx_tx * dx_tx + y_sq).astype(np.float32, copy=False)
            r_rx = np.sqrt(dx_rx * dx_rx + y_sq).astype(np.float32, copy=False)
            r_total = (r_tx + r_rx).astype(np.float32, copy=False)
            b = (np.float32(0.5) * r_total * inv_dr).astype(np.float32, copy=False)

            valid = np.isfinite(b) & (b >= 0.0) & (b < np.float32(max_bin_eff - 1))
            valid_idx = np.flatnonzero(valid)
            if valid_idx.size == 0:
                continue

            b_valid = b[valid_idx].astype(np.float32, copy=True)
            phase_valid = np.exp(1j * (phase_scale * r_total[valid_idx])).astype(
                np.complex64,
                copy=False,
            )
            entries.append(
                MimoBackProjectionPlanEntry(
                    pos_i=int(pos_i),
                    ant_i=int(ant_i),
                    valid_idx=valid_idx.astype(np.int32, copy=False),
                    bin_float=b_valid,
                    coherent_phase=phase_valid,
                )
            )

    return MimoBackProjectionPlan(
        image_shape=tuple(int(v) for v in x_grid.shape),
        n_pos=n_pos,
        n_ant=n_ant,
        max_bin_eff=max_bin_eff,
        dr_m=float(dr_m),
        fc_hz=float(fc_hz),
        c_m_s=float(c_m_s),
        phase_sign=int(phase_sign_i),
        entries=tuple(entries),
    )


def prepare_mimo_snapshots(
    raw_range_fft: np.ndarray,
    n_tx: int,
    motion_mode: MotionMode,
    window_doppler: np.ndarray | None = None,
) -> np.ndarray:
    """Prepare offline MIMO-SAR snapshots for a static scene using zero Doppler.

    Pipeline:
    range FFT -> Doppler FFT sui loop -> compensazione TDM-MIMO ->
    selezione Doppler zero -> output [pos, frame, ant, bin].
    """
    raw = np.asarray(raw_range_fft, dtype=np.complex64)
    if raw.ndim != 6:
        raise ValueError(f"raw_range_fft shape non valido: {raw.shape!r}")

    n_pos, n_frames, n_loops, n_tx_eff, n_rx, n_bins = raw.shape
    n_tx_i = int(n_tx)
    if n_tx_i <= 0:
        raise ValueError(f"n_tx deve essere > 0, trovato {n_tx!r}")
    if int(n_tx_eff) != n_tx_i:
        raise ValueError(f"raw_range_fft asse tx={n_tx_eff}, atteso n_tx={n_tx_i}")

    motion_mode_normalize(motion_mode)
    win = _build_doppler_window(window_doppler, int(n_loops))
    t0 = time.perf_counter()

    doppler_in = np.array(raw, dtype=np.complex64, copy=True)
    doppler_in *= win.reshape(1, 1, int(n_loops), 1, 1, 1).astype(np.complex64, copy=False)

    doppler_cube = np.fft.fft(doppler_in, axis=2).astype(np.complex64, copy=False)
    doppler_cube = np.fft.fftshift(doppler_cube, axes=2).astype(np.complex64, copy=False)
    doppler_cube = _apply_tdm_post_fft_compensation(doppler_cube, n_tx_i)

    zero_idx = int(n_loops // 2)
    out = doppler_cube[:, :, zero_idx, :, :, :].reshape(
        int(n_pos),
        int(n_frames),
        int(n_tx_i * n_rx),
        int(n_bins),
    ).astype(np.complex64, copy=False)
    prep_ms = np.float32((time.perf_counter() - t0) * 1000.0)
    est_bp_snapshots = int(n_frames)
    print(
        "[OFFLINE INFO] prepare_mimo_snapshots "
        "motion_mode=static_zero_doppler doppler_bins_used=1 "
        f"prep_ms={float(prep_ms):.1f} est_bp_snapshots={est_bp_snapshots}"
    )
    return out


def prepare_synthetic_aperture_data(
    zero_doppler_snapshots: np.ndarray,
    *,
    selected_positions: np.ndarray,
    x_pitch_m: float,
    x_tx_ant_m: np.ndarray,
    x_rx_ant_m: np.ndarray,
) -> SyntheticApertureData:
    """Flatten selected SAR positions and MIMO phase centers into one aperture.

    Input zero_doppler_snapshots shape: [pos, frame, ant, range_bin]
    Output snapshot_cube shape: [frame, 1, range_bin, synthetic_ant]
    """
    snapshots = np.asarray(zero_doppler_snapshots, dtype=np.complex64)
    if snapshots.ndim != 4:
        raise ValueError(
            "zero_doppler_snapshots shape non valido per synthetic aperture: "
            f"{snapshots.shape!r}; atteso [pos, frame, ant, bin]."
        )

    n_pos, n_frames, n_ant, n_bins = snapshots.shape
    pos_values = np.asarray(selected_positions, dtype=np.float32).reshape(-1)
    x_tx = np.asarray(x_tx_ant_m, dtype=np.float32).reshape(-1)
    x_rx = np.asarray(x_rx_ant_m, dtype=np.float32).reshape(-1)
    if pos_values.size != int(n_pos):
        raise ValueError(
            f"selected_positions size={pos_values.size} != aperture positions={int(n_pos)}"
        )
    if x_tx.size != int(n_ant) or x_rx.size != int(n_ant):
        raise ValueError(
            f"x_tx_ant_m/x_rx_ant_m size non coerente con asse antenna: "
            f"{x_tx.size}/{x_rx.size} vs {int(n_ant)}"
        )
    pitch = float(x_pitch_m)
    if not np.isfinite(pitch) or pitch <= 0.0:
        raise ValueError(f"x_pitch_m non valido: {x_pitch_m!r}")

    x_position_m = (pos_values * np.float32(pitch)).astype(np.float32, copy=False)
    x_phase_center_m = (
        np.float32(0.5) * (x_tx + x_rx).astype(np.float32, copy=False)
    ).astype(np.float32, copy=False)
    x_element_m = (
        x_position_m[:, None] + x_phase_center_m[None, :]
    ).astype(np.float32, copy=False)
    flattened = np.transpose(snapshots, (1, 3, 0, 2)).reshape(
        int(n_frames),
        int(n_bins),
        int(n_pos * n_ant),
    )
    snapshot_cube = flattened[:, None, :, :].astype(np.complex64, copy=False)
    return SyntheticApertureData(
        snapshot_cube=snapshot_cube,
        x_position_m=x_position_m.astype(np.float32, copy=False),
        x_phase_center_m=x_phase_center_m.astype(np.float32, copy=False),
        x_element_m=x_element_m.reshape(-1).astype(np.float32, copy=False),
    )


def synthetic_aperture_uniform_spacing_lambda(
    x_element_m: np.ndarray,
    *,
    wavelength_m: float,
    atol: float = 1e-6,
) -> float | None:
    phase_centers_lambda = (
        np.asarray(x_element_m, dtype=np.float32).reshape(-1) / np.float32(float(wavelength_m))
    ).astype(np.float32, copy=False)
    if phase_centers_lambda.size <= 1:
        return 0.0
    diffs = np.diff(phase_centers_lambda.astype(np.float64, copy=False))
    if diffs.size <= 0:
        return 0.0
    ref = float(diffs[0])
    if not np.isfinite(ref) or abs(ref) <= float(atol):
        return None
    if not np.all(np.isfinite(diffs)):
        return None
    if not np.allclose(diffs, ref, rtol=0.0, atol=float(atol)):
        return None
    return float(ref)


def back_projection_image(
    range_fft_sel: np.ndarray,
    x_pos_m: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    dr_m: float,
    fc_hz: float,
    c_m_s: float,
    max_bin: int,
    phase_sign: int = -1,
    chunk_size: int = 16384,
) -> np.ndarray:
    if range_fft_sel.ndim != 3:
        raise ValueError(f"range_fft_sel shape non valido: {range_fft_sel.shape!r}")
    if x_pos_m.ndim != 1:
        raise ValueError(f"x_pos_m shape non valido: {x_pos_m.shape!r}")
    if x_grid.shape != y_grid.shape:
        raise ValueError("x_grid e y_grid devono avere stessa shape")
    if float(dr_m) <= 0.0:
        raise ValueError("dr_m deve essere > 0")
    phase_sign_i = phase_sign_normalize(phase_sign, field_name="phase_sign")

    n_pos = int(range_fft_sel.shape[0])
    if n_pos <= 0:
        raise ValueError("Nessuna posizione selezionata per back projection")
    if x_pos_m.size != n_pos:
        raise ValueError("x_pos_m size != numero posizioni range_fft_sel")

    x_flat = x_grid.reshape(-1).astype(np.float32, copy=False)
    y_flat = y_grid.reshape(-1).astype(np.float32, copy=False)
    y_sq = (y_flat * y_flat).astype(np.float32, copy=False)
    img_flat = np.zeros(x_flat.shape[0], dtype=np.complex64)
    k = np.float32((4.0 * np.pi * float(fc_hz)) / float(c_m_s))
    phase_scale = np.float32(float(phase_sign_i)) * k
    inv_dr = np.float32(1.0 / float(dr_m))
    n_bins_avail = int(range_fft_sel.shape[2])
    max_bin_eff = max(0, min(int(max_bin), n_bins_avail))
    if max_bin_eff < 2:
        img_mag = np.abs(img_flat).reshape(x_grid.shape).astype(np.float32, copy=False)
        return _image_to_db(img_mag)
    chunk_n = max(1, int(chunk_size))

    for pos_i in range(n_pos):
        dx = x_flat - np.float32(x_pos_m[pos_i])
        rr = np.sqrt(dx * dx + y_sq).astype(np.float32, copy=False)
        b = rr * inv_dr
        valid = np.isfinite(b) & (b >= 0.0) & (b < np.float32(max_bin_eff - 1))
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            continue

        spec = range_fft_sel[pos_i]
        if spec.ndim != 2 or spec.shape[0] <= 0:
            continue
        s_mean = spec.mean(axis=0, dtype=np.complex64)
        max_bin_local = min(int(s_mean.size), int(max_bin_eff))
        if max_bin_local < 2:
            continue

        for start in range(0, int(valid_idx.size), chunk_n):
            idx = valid_idx[start : start + chunk_n]
            if idx.size == 0:
                continue
            s_interp = _interp_cubic_complex(s_mean, b[idx], max_bin_local)
            phase = np.exp(1j * (phase_scale * rr[idx])).astype(np.complex64, copy=False)
            img_flat[idx] += s_interp * phase

    img_mag = np.abs(img_flat).reshape(x_grid.shape).astype(np.float32, copy=False)
    return _image_to_db(img_mag)


def back_projection_power_mimo_snapshot(
    snapshot_ant_range: np.ndarray,
    x_pos_m: np.ndarray,
    x_tx_ant_m: np.ndarray,
    x_rx_ant_m: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    dr_m: float,
    fc_hz: float,
    c_m_s: float,
    max_bin: int,
    phase_sign: int = -1,
    chunk_size: int = 16384,
    coherent_sum: bool = True,
    bp_plan: MimoBackProjectionPlan | None = None,
) -> np.ndarray:
    if snapshot_ant_range.ndim != 3:
        raise ValueError(f"snapshot_ant_range shape non valido: {snapshot_ant_range.shape!r}")
    if x_pos_m.ndim != 1:
        raise ValueError(f"x_pos_m shape non valido: {x_pos_m.shape!r}")
    if x_tx_ant_m.ndim != 1 or x_rx_ant_m.ndim != 1:
        raise ValueError("x_tx_ant_m e x_rx_ant_m devono essere vettori 1D")
    if x_grid.shape != y_grid.shape:
        raise ValueError("x_grid e y_grid devono avere stessa shape")
    if float(dr_m) <= 0.0:
        raise ValueError("dr_m deve essere > 0")
    phase_sign_i = phase_sign_normalize(phase_sign, field_name="phase_sign")

    n_pos, n_ant, n_bins_avail = snapshot_ant_range.shape
    if n_pos <= 0:
        raise ValueError("Nessuna posizione selezionata per back projection")
    if x_pos_m.size != n_pos:
        raise ValueError("x_pos_m size != numero posizioni snapshot_ant_range")
    if x_tx_ant_m.size != n_ant or x_rx_ant_m.size != n_ant:
        raise ValueError("x_tx_ant_m/x_rx_ant_m size != numero antenne snapshot_ant_range")

    x_flat = x_grid.reshape(-1).astype(np.float32, copy=False)
    y_flat = y_grid.reshape(-1).astype(np.float32, copy=False)
    y_sq = (y_flat * y_flat).astype(np.float32, copy=False)
    k = np.float32((2.0 * np.pi * float(fc_hz)) / float(c_m_s))
    phase_scale = np.float32(float(phase_sign_i)) * k
    inv_dr = np.float32(1.0 / float(dr_m))
    max_bin_eff = max(0, min(int(max_bin), int(n_bins_avail)))
    if max_bin_eff < 2:
        return np.zeros_like(x_grid, dtype=np.float32)
    chunk_n = max(1, int(chunk_size))
    coherent = bool(coherent_sum)

    if coherent:
        img_flat_c = np.zeros(x_flat.shape[0], dtype=np.complex64)
    else:
        img_flat_f = np.zeros(x_flat.shape[0], dtype=np.float32)

    if (
        bp_plan is not None
        and tuple(int(v) for v in bp_plan.image_shape) == tuple(int(v) for v in x_grid.shape)
        and int(bp_plan.n_pos) == int(n_pos)
        and int(bp_plan.n_ant) == int(n_ant)
        and int(bp_plan.max_bin_eff) == int(max_bin_eff)
        and int(bp_plan.phase_sign) == int(phase_sign_i)
        and np.isclose(float(bp_plan.dr_m), float(dr_m), rtol=0.0, atol=1e-12)
        and np.isclose(float(bp_plan.fc_hz), float(fc_hz), rtol=0.0, atol=1e-3)
        and np.isclose(float(bp_plan.c_m_s), float(c_m_s), rtol=0.0, atol=1e-3)
    ):
        for entry in bp_plan.entries:
            pos_i = int(entry.pos_i)
            ant_i = int(entry.ant_i)
            if pos_i < 0 or pos_i >= n_pos or ant_i < 0 or ant_i >= n_ant:
                continue
            spec = snapshot_ant_range[pos_i, ant_i]
            if int(spec.size) < int(max_bin_eff):
                continue

            valid_idx = np.asarray(entry.valid_idx, dtype=np.int32)
            b_valid = np.asarray(entry.bin_float, dtype=np.float32)
            if valid_idx.size != b_valid.size:
                continue
            phase_valid = np.asarray(entry.coherent_phase, dtype=np.complex64)
            if coherent and phase_valid.size != b_valid.size:
                continue

            for start in range(0, int(valid_idx.size), chunk_n):
                stop = start + chunk_n
                idx = valid_idx[start:stop]
                if idx.size == 0:
                    continue
                s_interp = _interp_cubic_complex(spec, b_valid[start:stop], max_bin_eff)
                if coherent:
                    img_flat_c[idx] += s_interp * phase_valid[start:stop]
                else:
                    img_flat_f[idx] += (
                        s_interp.real.astype(np.float32, copy=False) ** np.float32(2.0)
                        + s_interp.imag.astype(np.float32, copy=False) ** np.float32(2.0)
                    )

        if coherent:
            img_lin = (
                img_flat_c.real.astype(np.float32, copy=False) ** np.float32(2.0)
                + img_flat_c.imag.astype(np.float32, copy=False) ** np.float32(2.0)
            )
        else:
            img_lin = img_flat_f.astype(np.float32, copy=False)
        return img_lin.reshape(x_grid.shape).astype(np.float32, copy=False)

    for pos_i in range(n_pos):
        x_pos = np.float32(x_pos_m[pos_i])
        for ant_i in range(n_ant):
            x_tx = x_pos + np.float32(x_tx_ant_m[ant_i])
            x_rx = x_pos + np.float32(x_rx_ant_m[ant_i])

            dx_tx = x_flat - x_tx
            dx_rx = x_flat - x_rx
            r_tx = np.sqrt(dx_tx * dx_tx + y_sq).astype(np.float32, copy=False)
            r_rx = np.sqrt(dx_rx * dx_rx + y_sq).astype(np.float32, copy=False)
            r_total = (r_tx + r_rx).astype(np.float32, copy=False)
            r_eq = (np.float32(0.5) * r_total).astype(np.float32, copy=False)
            b = r_eq * inv_dr

            valid = np.isfinite(b) & (b >= 0.0) & (b < np.float32(max_bin_eff - 1))
            valid_idx = np.flatnonzero(valid)
            if valid_idx.size == 0:
                continue

            spec = snapshot_ant_range[pos_i, ant_i]
            max_bin_local = min(int(spec.size), int(max_bin_eff))
            if max_bin_local < 2:
                continue

            for start in range(0, int(valid_idx.size), chunk_n):
                idx = valid_idx[start : start + chunk_n]
                if idx.size == 0:
                    continue
                s_interp = _interp_cubic_complex(spec, b[idx], max_bin_local)
                if coherent:
                    phase = np.exp(1j * (phase_scale * r_total[idx])).astype(np.complex64, copy=False)
                    img_flat_c[idx] += s_interp * phase
                else:
                    img_flat_f[idx] += (
                        s_interp.real.astype(np.float32, copy=False) ** np.float32(2.0)
                        + s_interp.imag.astype(np.float32, copy=False) ** np.float32(2.0)
                    )

    if coherent:
        img_lin = (
            img_flat_c.real.astype(np.float32, copy=False) ** np.float32(2.0)
            + img_flat_c.imag.astype(np.float32, copy=False) ** np.float32(2.0)
        )
    else:
        img_lin = img_flat_f.astype(np.float32, copy=False)
    return img_lin.reshape(x_grid.shape).astype(np.float32, copy=False)


def back_projection_image_mimo(
    range_fft_sel: np.ndarray,
    x_pos_m: np.ndarray,
    x_tx_ant_m: np.ndarray,
    x_rx_ant_m: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    dr_m: float,
    fc_hz: float,
    c_m_s: float,
    max_bin: int,
    motion_mode: MotionMode,
    phase_sign: int = -1,
    chunk_size: int = 16384,
    coherent_sum: bool = True,
    bp_plan: MimoBackProjectionPlan | None = None,
) -> np.ndarray:
    motion_mode_normalize(motion_mode)
    if range_fft_sel.ndim != 4:
        raise ValueError(
            "range_fft_sel shape non valido per mimo_sar static_zero_doppler: "
            f"{range_fft_sel.shape!r}; atteso [pos, frame, ant, bin]."
        )

    total_power = np.zeros_like(x_grid, dtype=np.float32)
    n_frames = int(range_fft_sel.shape[1])
    for frame_i in range(n_frames):
        total_power += back_projection_power_mimo_snapshot(
            range_fft_sel[:, frame_i, :, :],
            x_pos_m,
            x_tx_ant_m,
            x_rx_ant_m,
            x_grid,
            y_grid,
            dr_m=dr_m,
            fc_hz=fc_hz,
            c_m_s=c_m_s,
            max_bin=max_bin,
            phase_sign=phase_sign,
            chunk_size=chunk_size,
            coherent_sum=coherent_sum,
            bp_plan=bp_plan,
        )
    return power_image_to_db(total_power)
