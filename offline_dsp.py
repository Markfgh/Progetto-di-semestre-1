from __future__ import annotations

from typing import Any, Literal

import numpy as np

AvgMode = Literal["none", "loop", "frame", "both"]
BpMode = Literal["sar_only", "mimo_sar"]
_AVG_MODES = {"none", "loop", "frame", "both"}
_BP_MODES = {"sar_only", "mimo_sar"}


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


def build_virtual_array_x_offsets(
    virtual_antennas: int,
    *,
    pitch_m: float,
    use_virtual_antennas: bool = True,
) -> np.ndarray:
    n_ant = int(virtual_antennas)
    if n_ant <= 0:
        raise ValueError(f"virtual_antennas deve essere > 0, trovato: {virtual_antennas!r}")
    if not bool(use_virtual_antennas):
        return np.zeros(n_ant, dtype=np.float32)

    pitch = float(pitch_m)
    if pitch <= 0.0:
        raise ValueError(f"pitch_m deve essere > 0, trovato: {pitch_m!r}")

    # Prima approssimazione: array lineare uniforme centrato in x=0.
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
        # Input MIMO: [pos, frame, loop, ant, bin]
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
        b0 = np.floor(b).astype(np.int32, copy=False)
        valid = np.isfinite(b) & (b0 >= 0) & (b0 < (max_bin_eff - 1))
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            continue

        spec = range_fft_sel[pos_i]
        if spec.ndim != 2 or spec.shape[0] <= 0:
            continue
        s_mean = spec.mean(axis=0, dtype=np.complex64)
        if s_mean.size < max_bin_eff:
            max_bin_local = int(s_mean.size)
            if max_bin_local < 2:
                continue
        else:
            max_bin_local = max_bin_eff

        for start in range(0, int(valid_idx.size), chunk_n):
            idx = valid_idx[start : start + chunk_n]
            if idx.size == 0:
                continue
            b_chunk = b[idx]
            b0_chunk = np.floor(b_chunk).astype(np.int32, copy=False)
            np.clip(b0_chunk, 0, max_bin_local - 2, out=b0_chunk)
            frac = (b_chunk - b0_chunk.astype(np.float32, copy=False)).astype(np.float32, copy=False)

            s0 = s_mean[b0_chunk]
            s1 = s_mean[b0_chunk + 1]
            s_interp = s0 + (s1 - s0) * frac

            phase = np.exp(1j * (phase_scale * rr[idx])).astype(np.complex64, copy=False)
            img_flat[idx] += s_interp * phase

    img_mag = np.abs(img_flat).reshape(x_grid.shape).astype(np.float32, copy=False)
    return _image_to_db(img_mag)


def back_projection_image_mimo(
    range_fft_sel: np.ndarray,
    x_pos_m: np.ndarray,
    x_ant_m: np.ndarray,
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
) -> np.ndarray:
    if range_fft_sel.ndim != 4:
        raise ValueError(f"range_fft_sel shape non valido: {range_fft_sel.shape!r}")
    if x_pos_m.ndim != 1:
        raise ValueError(f"x_pos_m shape non valido: {x_pos_m.shape!r}")
    if x_ant_m.ndim != 1:
        raise ValueError(f"x_ant_m shape non valido: {x_ant_m.shape!r}")
    if x_grid.shape != y_grid.shape:
        raise ValueError("x_grid e y_grid devono avere stessa shape")
    if float(dr_m) <= 0.0:
        raise ValueError("dr_m deve essere > 0")
    phase_sign_i = phase_sign_normalize(phase_sign, field_name="phase_sign")

    n_pos, _, n_ant, n_bins_avail = range_fft_sel.shape
    if n_pos <= 0:
        raise ValueError("Nessuna posizione selezionata per back projection")
    if x_pos_m.size != n_pos:
        raise ValueError("x_pos_m size != numero posizioni range_fft_sel")
    if x_ant_m.size != n_ant:
        raise ValueError("x_ant_m size != numero antenne range_fft_sel")

    x_flat = x_grid.reshape(-1).astype(np.float32, copy=False)
    y_flat = y_grid.reshape(-1).astype(np.float32, copy=False)
    y_sq = (y_flat * y_flat).astype(np.float32, copy=False)
    k = np.float32((4.0 * np.pi * float(fc_hz)) / float(c_m_s))
    phase_scale = np.float32(float(phase_sign_i)) * k
    inv_dr = np.float32(1.0 / float(dr_m))
    max_bin_eff = max(0, min(int(max_bin), int(n_bins_avail)))
    if max_bin_eff < 2:
        return _image_to_db(np.zeros_like(x_grid, dtype=np.float32))
    chunk_n = max(1, int(chunk_size))
    coherent = bool(coherent_sum)

    if coherent:
        img_flat_c = np.zeros(x_flat.shape[0], dtype=np.complex64)
    else:
        img_flat_f = np.zeros(x_flat.shape[0], dtype=np.float32)

    for pos_i in range(n_pos):
        for ant_i in range(n_ant):
            x_eff = np.float32(x_pos_m[pos_i]) + np.float32(x_ant_m[ant_i])
            dx = x_flat - x_eff
            rr = np.sqrt(dx * dx + y_sq).astype(np.float32, copy=False)
            b = rr * inv_dr
            b0 = np.floor(b).astype(np.int32, copy=False)
            valid = np.isfinite(b) & (b0 >= 0) & (b0 < (max_bin_eff - 1))
            valid_idx = np.flatnonzero(valid)
            if valid_idx.size == 0:
                continue

            spec = range_fft_sel[pos_i, :, ant_i, :]
            if spec.ndim != 2 or spec.shape[0] <= 0:
                continue

            if coherent:
                spec_red = spec.mean(axis=0, dtype=np.complex64)
                max_bin_local = min(int(spec_red.size), int(max_bin_eff))
                if max_bin_local < 2:
                    continue
                for start in range(0, int(valid_idx.size), chunk_n):
                    idx = valid_idx[start : start + chunk_n]
                    if idx.size == 0:
                        continue
                    b_chunk = b[idx]
                    b0_chunk = np.floor(b_chunk).astype(np.int32, copy=False)
                    np.clip(b0_chunk, 0, max_bin_local - 2, out=b0_chunk)
                    frac = (b_chunk - b0_chunk.astype(np.float32, copy=False)).astype(np.float32, copy=False)

                    s0 = spec_red[b0_chunk]
                    s1 = spec_red[b0_chunk + 1]
                    s_interp = s0 + (s1 - s0) * frac

                    phase = np.exp(1j * (phase_scale * rr[idx])).astype(np.complex64, copy=False)
                    img_flat_c[idx] += s_interp * phase
            else:
                spec_red = np.abs(spec).mean(axis=0, dtype=np.float32)
                max_bin_local = min(int(spec_red.size), int(max_bin_eff))
                if max_bin_local < 2:
                    continue
                for start in range(0, int(valid_idx.size), chunk_n):
                    idx = valid_idx[start : start + chunk_n]
                    if idx.size == 0:
                        continue
                    b_chunk = b[idx]
                    b0_chunk = np.floor(b_chunk).astype(np.int32, copy=False)
                    np.clip(b0_chunk, 0, max_bin_local - 2, out=b0_chunk)
                    frac = (b_chunk - b0_chunk.astype(np.float32, copy=False)).astype(np.float32, copy=False)

                    s0 = spec_red[b0_chunk]
                    s1 = spec_red[b0_chunk + 1]
                    s_interp = s0 + (s1 - s0) * frac
                    img_flat_f[idx] += s_interp.astype(np.float32, copy=False)

    if coherent:
        img_lin = np.abs(img_flat_c).reshape(x_grid.shape).astype(np.float32, copy=False)
    else:
        img_lin = img_flat_f.reshape(x_grid.shape).astype(np.float32, copy=False)
    return _image_to_db(img_lin)
