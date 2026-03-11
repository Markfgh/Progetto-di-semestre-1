from __future__ import annotations

from typing import Any, Literal

import numpy as np

AvgMode = Literal["none", "loop", "frame", "both"]
_AVG_MODES = {"none", "loop", "frame", "both"}


def avg_mode_normalize(mode: str | None) -> AvgMode:
    mode_s = str(mode or "both").strip().lower()
    if mode_s not in _AVG_MODES:
        return "both"
    return mode_s  # type: ignore[return-value]


def phase_sign_normalize(value: Any, *, field_name: str = "phase_sign") -> int:
    try:
        phase_sign_i = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} non valido: {value!r}") from exc
    if phase_sign_i not in (-1, 1):
        raise ValueError(f"{field_name} deve essere +1 o -1, trovato: {value!r}")
    return int(phase_sign_i)


def reduce_avg_mode(range_fft_4d: np.ndarray, avg_mode: AvgMode) -> np.ndarray:
    if range_fft_4d.ndim != 4:
        raise ValueError(f"range_fft_4d shape non valido: {range_fft_4d.shape!r}")

    if avg_mode == "both":
        out = range_fft_4d.mean(axis=(1, 2), dtype=np.complex64)
        return out[:, np.newaxis, :].astype(np.complex64, copy=False)
    if avg_mode == "frame":
        out = range_fft_4d.mean(axis=1, dtype=np.complex64)
        return out.astype(np.complex64, copy=False)
    if avg_mode == "loop":
        out = range_fft_4d.mean(axis=2, dtype=np.complex64)
        return out.astype(np.complex64, copy=False)

    n_pos, n_frames, n_loops, n_bins = range_fft_4d.shape
    return range_fft_4d.reshape(n_pos, n_frames * n_loops, n_bins)


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
        img_mag = np.abs(img_flat).reshape(x_grid.shape)
        return (20.0 * np.log10(img_mag + 1e-6)).astype(np.float32, copy=False)
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

    img_mag = np.abs(img_flat).reshape(x_grid.shape)
    img_db = (20.0 * np.log10(img_mag + 1e-6)).astype(np.float32, copy=False)
    return img_db
