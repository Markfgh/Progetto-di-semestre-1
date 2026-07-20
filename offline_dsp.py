from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

@dataclass(frozen=True)
class SyntheticApertureData:
    snapshot_cube: np.ndarray
    x_position_m: np.ndarray
    x_phase_center_m: np.ndarray
    x_element_m: np.ndarray


def phase_sign_normalize(value: Any, *, field_name: str = "phase_sign") -> int:
    try:
        phase_sign_i = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} non valido: {value!r}") from exc
    if phase_sign_i not in (-1, 1):
        raise ValueError(f"{field_name} deve essere +1 o -1, trovato: {value!r}")
    return int(phase_sign_i)


def residual_video_phase_sign_normalize(
    value: Any,
    *,
    field_name: str = "residual_video_phase",
) -> int:
    """Normalize the optional FMCW residual-video-phase correction sign.

    ``0``/``off`` disables the correction.  ``+`` and ``-`` add respectively
    ``+/- pi * slope * (R_bi / c)^2`` to the backprojection phase.  The
    appropriate sign depends on the dechirp/IQ convention of the capture.
    """
    if isinstance(value, str):
        normalized = value.strip().lower()
        aliases = {
            "off": 0,
            "none": 0,
            "0": 0,
            "+": 1,
            "+1": 1,
            "1": 1,
            "-": -1,
            "-1": -1,
        }
        if normalized in aliases:
            return int(aliases[normalized])
    try:
        sign_i = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} non valido: {value!r}; usa off, + o -") from exc
    if sign_i not in (-1, 0, 1):
        raise ValueError(f"{field_name} deve essere off, + o -; trovato: {value!r}")
    return int(sign_i)


def power_image_to_db(img_power: np.ndarray) -> np.ndarray:
    out = np.asarray(img_power, dtype=np.float32).copy()
    np.add(out, np.float32(1e-12), out=out)
    np.log10(out, out=out)
    out *= np.float32(10.0)
    return out.astype(np.float32, copy=False)


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


def prepare_mimo_snapshots(
    raw_range_fft: np.ndarray,
    n_tx: int,
    window_doppler: np.ndarray | None = None,
    log_info: bool = True,
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

    win = _build_doppler_window(window_doppler, int(n_loops))
    t0 = time.perf_counter()

    # The centered zero-Doppler bin is FFT bin k=0 after fftshift.  Computing
    # only that coefficient is exactly the windowed sum over the loop axis and
    # avoids materializing the complete Doppler cube.  TDM compensation is one
    # at zero Doppler, so no per-TX phase correction is required here.
    zero_doppler = np.einsum(
        "pfltrb,l->pftrb",
        raw,
        win,
        dtype=np.complex64,
        optimize=True,
    ).astype(np.complex64, copy=False)
    out = zero_doppler.reshape(
        int(n_pos),
        int(n_frames),
        int(n_tx_i * n_rx),
        int(n_bins),
    ).astype(np.complex64, copy=False)
    prep_ms = np.float32((time.perf_counter() - t0) * 1000.0)
    est_bp_snapshots = int(n_frames)
    if bool(log_info):
        print(
            "[OFFLINE INFO] prepare_mimo_snapshots "
            "doppler_bins_used=1 "
            f"prep_ms={float(prep_ms):.1f} est_bp_snapshots={est_bp_snapshots}"
        )
    return out


def _back_projection_weights(
    value: np.ndarray | None,
    *,
    expected_size: int,
    field_name: str,
) -> np.ndarray:
    """Return a real coherent weight for each pose or physical channel."""
    if value is None:
        return np.ones(int(expected_size), dtype=np.float32)

    weights = np.asarray(value, dtype=np.float32)
    if weights.ndim != 1 or int(weights.size) != int(expected_size):
        raise ValueError(f"{field_name} non coerente: atteso vettore di {int(expected_size)} elementi")
    if not np.all(np.isfinite(weights)):
        raise ValueError(f"{field_name} contiene valori non finiti")
    return weights.astype(np.float32, copy=False)


def back_projection_power_mimo_geometry(
    snapshot_frame_ant_range: np.ndarray,
    tx_global_m: np.ndarray,
    rx_global_m: np.ndarray,
    voxel_xyz: np.ndarray,
    *,
    dr_m: float,
    fc_hz: float,
    c_m_s: float,
    max_bin: int,
    phase_sign: int = -1,
    residual_video_phase: int | str = 0,
    slope_hz_s: float | None = None,
    chunk_size: int = 16384,
    pose_weights: np.ndarray | None = None,
    channel_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Backproject physical bistatic TX/RX coordinates into an arbitrary 3-D grid.

    ``snapshot_frame_ant_range`` has shape ``[pose, frame, antenna, range_bin]``.
    ``tx_global_m`` and ``rx_global_m`` have shape ``[pose, antenna, 3]`` and
    contain the *physical* transmitter and receiver coordinates in the same
    world frame as ``voxel_xyz``.  The output has shape ``voxel_xyz.shape[:-1]``.

    The coherent sum is evaluated independently for each frame and the frame
    powers are summed, as in the legacy linear implementation.  ``pose_weights``
    and ``channel_weights`` are independent real coherent weights; omitted
    weights are uniform ones.  No position-times-channel aperture window is
    applied here.
    """
    snapshots = np.asarray(snapshot_frame_ant_range, dtype=np.complex64)
    if snapshots.ndim != 4:
        raise ValueError(
            "snapshot_frame_ant_range shape non valido: "
            f"{snapshots.shape!r}; atteso [pos, frame, ant, bin]."
        )

    n_pos, n_frames, n_ant, n_bins_avail = (int(v) for v in snapshots.shape)
    expected_geometry_shape = (n_pos, n_ant, 3)
    tx_global = np.asarray(tx_global_m, dtype=np.float32)
    rx_global = np.asarray(rx_global_m, dtype=np.float32)
    if tx_global.shape != expected_geometry_shape or rx_global.shape != expected_geometry_shape:
        raise ValueError(
            "tx_global_m e rx_global_m devono avere shape "
            f"{expected_geometry_shape!r}; trovate {tx_global.shape!r}/{rx_global.shape!r}"
        )
    if not np.all(np.isfinite(tx_global)) or not np.all(np.isfinite(rx_global)):
        raise ValueError("tx_global_m e rx_global_m devono contenere coordinate finite")

    voxels = np.asarray(voxel_xyz, dtype=np.float32)
    if voxels.ndim < 1 or int(voxels.shape[-1]) != 3:
        raise ValueError("voxel_xyz deve avere shape [..., 3]")
    if not np.all(np.isfinite(voxels)):
        raise ValueError("voxel_xyz deve contenere coordinate finite")

    dr_f = float(dr_m)
    fc_f = float(fc_hz)
    c_f = float(c_m_s)
    if not np.isfinite(dr_f) or dr_f <= 0.0:
        raise ValueError("dr_m deve essere > 0")
    if not np.isfinite(fc_f) or fc_f <= 0.0:
        raise ValueError("fc_hz deve essere > 0")
    if not np.isfinite(c_f) or c_f <= 0.0:
        raise ValueError("c_m_s deve essere > 0")

    phase_sign_i = phase_sign_normalize(phase_sign, field_name="phase_sign")
    rvp_sign_i = residual_video_phase_sign_normalize(
        residual_video_phase,
        field_name="residual_video_phase",
    )
    pose_weight = _back_projection_weights(
        pose_weights,
        expected_size=n_pos,
        field_name="pose_weights",
    )
    channel_weight = _back_projection_weights(
        channel_weights,
        expected_size=n_ant,
        field_name="channel_weights",
    )

    output_shape = voxels.shape[:-1]
    max_bin_eff = max(0, min(int(max_bin), n_bins_avail))
    if max_bin_eff < 2 or n_frames <= 0:
        return np.zeros(output_shape, dtype=np.float32)

    voxel_flat = voxels.reshape(-1, 3).astype(np.float32, copy=False)
    k = np.float32((2.0 * np.pi * fc_f) / c_f)
    phase_scale = np.float32(float(phase_sign_i)) * k
    rvp_phase_scale = np.float32(0.0)
    if rvp_sign_i != 0:
        if slope_hz_s is None or not np.isfinite(float(slope_hz_s)) or float(slope_hz_s) <= 0.0:
            raise ValueError("slope_hz_s deve essere > 0 quando residual_video_phase e' attiva")
        rvp_phase_scale = np.float32(
            float(rvp_sign_i) * np.pi * float(slope_hz_s) / (c_f * c_f)
        )
    inv_dr = np.float32(1.0 / dr_f)
    chunk_n = max(1, int(chunk_size))
    frame_acc = np.zeros((n_frames, int(voxel_flat.shape[0])), dtype=np.complex64)

    for pos_i in range(n_pos):
        current_pose_weight = np.float32(pose_weight[pos_i])
        if current_pose_weight == np.float32(0.0):
            continue
        for ant_i in range(n_ant):
            coherent_weight = np.float32(current_pose_weight * channel_weight[ant_i])
            if coherent_weight == np.float32(0.0):
                continue

            tx = tx_global[pos_i, ant_i]
            rx = rx_global[pos_i, ant_i]
            for start in range(0, int(voxel_flat.shape[0]), chunk_n):
                stop = min(start + chunk_n, int(voxel_flat.shape[0]))
                voxel_chunk = voxel_flat[start:stop]
                if voxel_chunk.size == 0:
                    continue

                delta_tx = (voxel_chunk - tx).astype(np.float32, copy=False)
                delta_rx = (voxel_chunk - rx).astype(np.float32, copy=False)
                r_tx = np.sqrt(
                    delta_tx[:, 0] * delta_tx[:, 0]
                    + delta_tx[:, 1] * delta_tx[:, 1]
                    + delta_tx[:, 2] * delta_tx[:, 2]
                ).astype(np.float32, copy=False)
                r_rx = np.sqrt(
                    delta_rx[:, 0] * delta_rx[:, 0]
                    + delta_rx[:, 1] * delta_rx[:, 1]
                    + delta_rx[:, 2] * delta_rx[:, 2]
                ).astype(np.float32, copy=False)
                r_total = (r_tx + r_rx).astype(np.float32, copy=False)
                bins = (np.float32(0.5) * r_total * inv_dr).astype(np.float32, copy=False)
                valid = np.isfinite(bins) & (bins >= 0.0) & (bins < np.float32(max_bin_eff - 1))
                valid_idx = np.flatnonzero(valid)
                if valid_idx.size == 0:
                    continue

                valid_bins = bins[valid_idx]
                phase_argument = phase_scale * r_total[valid_idx]
                if rvp_sign_i != 0:
                    phase_argument = (
                        phase_argument
                        + rvp_phase_scale * r_total[valid_idx] * r_total[valid_idx]
                    )
                phase = np.exp(1j * phase_argument).astype(np.complex64, copy=False)
                output_idx = (start + valid_idx).astype(np.intp, copy=False)
                for frame_i in range(n_frames):
                    interp = _interp_cubic_complex(
                        snapshots[pos_i, frame_i, ant_i],
                        valid_bins,
                        max_bin_eff,
                    )
                    contribution = interp * phase
                    if coherent_weight != np.float32(1.0):
                        contribution = contribution * coherent_weight
                    frame_acc[frame_i, output_idx] += contribution

    total_power = np.sum(
        frame_acc.real.astype(np.float32, copy=False) ** np.float32(2.0)
        + frame_acc.imag.astype(np.float32, copy=False) ** np.float32(2.0),
        axis=0,
        dtype=np.float32,
    )
    return total_power.reshape(output_shape).astype(np.float32, copy=False)


def back_projection_power_mimo_frames(
    snapshot_frame_ant_range: np.ndarray,
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
    residual_video_phase: int | str = 0,
    slope_hz_s: float | None = None,
    chunk_size: int = 16384,
) -> np.ndarray:
    """Reference-compatible adapter for the historical linear z=0 BP.

    The 3-D kernel above is the implementation for new cylindrical captures.
    This legacy entry point intentionally retains the exact arithmetic and
    chunk traversal of the previous linear implementation: existing linear
    files therefore remain a numerical reference, rather than merely being
    approximately equivalent after a coordinate conversion.
    """
    snapshots = np.asarray(snapshot_frame_ant_range, dtype=np.complex64)
    if snapshots.ndim != 4:
        raise ValueError(
            "snapshot_frame_ant_range shape non valido: "
            f"{snapshots.shape!r}; atteso [pos, frame, ant, bin]."
        )
    if x_pos_m.ndim != 1 or int(x_pos_m.size) != int(snapshots.shape[0]):
        raise ValueError("x_pos_m non coerente con asse posizione")
    if x_tx_ant_m.ndim != 1 or x_rx_ant_m.ndim != 1:
        raise ValueError("x_tx_ant_m e x_rx_ant_m devono essere vettori 1D")
    if int(x_tx_ant_m.size) != int(snapshots.shape[2]) or int(x_rx_ant_m.size) != int(snapshots.shape[2]):
        raise ValueError("geometria TX/RX non coerente con asse antenna")
    if x_grid.shape != y_grid.shape:
        raise ValueError("x_grid e y_grid devono avere stessa shape")
    if float(dr_m) <= 0.0:
        raise ValueError("dr_m deve essere > 0")

    phase_sign_i = phase_sign_normalize(phase_sign, field_name="phase_sign")
    rvp_sign_i = residual_video_phase_sign_normalize(
        residual_video_phase,
        field_name="residual_video_phase",
    )
    n_pos, n_frames, n_ant, n_bins_avail = (int(v) for v in snapshots.shape)
    max_bin_eff = max(0, min(int(max_bin), int(n_bins_avail)))
    if max_bin_eff < 2 or n_frames <= 0:
        return np.zeros_like(x_grid, dtype=np.float32)

    x_flat = x_grid.reshape(-1).astype(np.float32, copy=False)
    y_flat = y_grid.reshape(-1).astype(np.float32, copy=False)
    y_sq = (y_flat * y_flat).astype(np.float32, copy=False)
    k = np.float32((2.0 * np.pi * float(fc_hz)) / float(c_m_s))
    phase_scale = np.float32(float(phase_sign_i)) * k
    rvp_phase_scale = np.float32(0.0)
    if rvp_sign_i != 0:
        if slope_hz_s is None or not np.isfinite(float(slope_hz_s)) or float(slope_hz_s) <= 0.0:
            raise ValueError("slope_hz_s deve essere > 0 quando residual_video_phase e' attiva")
        rvp_phase_scale = np.float32(
            float(rvp_sign_i) * np.pi * float(slope_hz_s) / (float(c_m_s) * float(c_m_s))
        )
    inv_dr = np.float32(1.0 / float(dr_m))
    chunk_n = max(1, int(chunk_size))
    frame_acc = np.zeros((n_frames, int(x_flat.size)), dtype=np.complex64)

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

            for start in range(0, int(valid_idx.size), chunk_n):
                idx = valid_idx[start : start + chunk_n]
                if idx.size == 0:
                    continue
                bins = b[idx]
                phase_argument = phase_scale * r_total[idx]
                if rvp_sign_i != 0:
                    phase_argument = phase_argument + rvp_phase_scale * r_total[idx] * r_total[idx]
                phase = np.exp(1j * phase_argument).astype(np.complex64, copy=False)
                for frame_i in range(n_frames):
                    interp = _interp_cubic_complex(
                        snapshots[pos_i, frame_i, ant_i],
                        bins,
                        max_bin_eff,
                    )
                    frame_acc[frame_i, idx] += interp * phase

    total_power = np.sum(
        frame_acc.real.astype(np.float32, copy=False) ** np.float32(2.0)
        + frame_acc.imag.astype(np.float32, copy=False) ** np.float32(2.0),
        axis=0,
        dtype=np.float32,
    )
    return total_power.reshape(x_grid.shape).astype(np.float32, copy=False)


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
    atol: float = 1e-4,
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
    phase_sign: int = -1,
    residual_video_phase: int | str = 0,
    slope_hz_s: float | None = None,
    chunk_size: int = 16384,
) -> np.ndarray:
    """Return a dB image shaped exactly like the supplied spatial grids.

    The sampling density of ``x_grid``/``y_grid`` controls image resolution;
    the range FFT only supplies the interpolation samples addressed by
    ``max_bin``.
    """
    if range_fft_sel.ndim != 4:
        raise ValueError(
            "range_fft_sel shape non valido per mimo_sar static_zero_doppler: "
            f"{range_fft_sel.shape!r}; atteso [pos, frame, ant, bin]."
        )

    total_power = back_projection_power_mimo_frames(
        range_fft_sel,
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
        residual_video_phase=residual_video_phase,
        slope_hz_s=slope_hz_s,
        chunk_size=chunk_size,
    )
    return power_image_to_db(total_power)
