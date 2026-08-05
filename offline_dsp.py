"""Operazioni DSP per ricostruzioni SAR offline.

Qui vivono la preparazione dei cubi MIMO e la back-projection bistatica.  Il
modulo non gestisce file o processi: riceve array numerici già validati da
``offline_processing`` e restituisce immagini di potenza.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

try:
    import numba as _numba
except Exception:  # pragma: no cover - Numba is an optional local dependency
    _numba = None


if _numba is not None:

    @_numba.njit(cache=True, fastmath=False, parallel=True)
    def _back_projection_power_mimo_geometry_numba_kernel(
        snapshots,
        tx_global,
        rx_global,
        voxel_flat,
        max_bin_eff,
        inv_dr,
        range_offset_m,
        phase_scale,
        rvp_phase_scale,
        range_phase_step,
        pose_weight,
        channel_weight,
    ):
        """Parallel scalar form of the physical bistatic BP accumulation.

        One worker owns one output voxel, so it can coherently accumulate all
        poses/channels without locks.  The per-frame accumulation remains
        separate, preserving the legacy ``sum(abs(coherent_frame_sum)**2)``
        result instead of coherently mixing independent radar frames.
        """
        n_pos = snapshots.shape[0]
        n_frames = snapshots.shape[1]
        n_ant = snapshots.shape[2]
        n_voxels = voxel_flat.shape[0]
        frame_real = np.empty((n_voxels, n_frames), dtype=np.float32)
        frame_imag = np.empty((n_voxels, n_frames), dtype=np.float32)
        output = np.empty(n_voxels, dtype=np.float32)

        half = np.float32(0.5)
        two = np.float32(2.0)
        three = np.float32(3.0)
        four = np.float32(4.0)
        five = np.float32(5.0)
        zero = np.float32(0.0)
        phase_aware = range_phase_step != zero
        step_cos = np.cos(range_phase_step)
        step_sin = np.sin(range_phase_step)
        step_2_cos = np.cos(two * range_phase_step)
        step_2_sin = np.sin(two * range_phase_step)

        for voxel_i in _numba.prange(n_voxels):
            for frame_i in range(n_frames):
                frame_real[voxel_i, frame_i] = zero
                frame_imag[voxel_i, frame_i] = zero

            vx = voxel_flat[voxel_i, 0]
            vy = voxel_flat[voxel_i, 1]
            vz = voxel_flat[voxel_i, 2]
            for pos_i in range(n_pos):
                current_pose_weight = pose_weight[pos_i]
                if current_pose_weight == zero:
                    continue
                for ant_i in range(n_ant):
                    coherent_weight = current_pose_weight * channel_weight[ant_i]
                    if coherent_weight == zero:
                        continue

                    tx_dx = vx - tx_global[pos_i, ant_i, 0]
                    tx_dy = vy - tx_global[pos_i, ant_i, 1]
                    tx_dz = vz - tx_global[pos_i, ant_i, 2]
                    rx_dx = vx - rx_global[pos_i, ant_i, 0]
                    rx_dy = vy - rx_global[pos_i, ant_i, 1]
                    rx_dz = vz - rx_global[pos_i, ant_i, 2]
                    r_tx = np.sqrt(tx_dx * tx_dx + tx_dy * tx_dy + tx_dz * tx_dz)
                    r_rx = np.sqrt(rx_dx * rx_dx + rx_dy * rx_dy + rx_dz * rx_dz)
                    r_total = r_tx + r_rx
                    # ``range_offset_m`` is an equivalent one-way range
                    # calibration.  It moves only the FFT sample addressed by
                    # BP; the propagation phase below remains geometric.
                    range_bin = (half * r_total + range_offset_m) * inv_dr
                    if not (range_bin >= zero and range_bin < np.float32(max_bin_eff - 1)):
                        continue

                    b0 = int(range_bin)
                    frac = range_bin - np.float32(b0)
                    phase_argument = phase_scale * r_total
                    if rvp_phase_scale != zero:
                        phase_argument = phase_argument + rvp_phase_scale * r_total * r_total
                    phase_real = np.cos(phase_argument)
                    phase_imag = np.sin(phase_argument)

                    for frame_i in range(n_frames):
                        spectrum = snapshots[pos_i, frame_i, ant_i]
                        p1 = spectrum[b0]
                        p2 = spectrum[b0 + 1]
                        p1_real = p1.real
                        p1_imag = p1.imag
                        if phase_aware:
                            # Remove the causal DFT phase relative to b0.  The
                            # common integer-bin phase cancels in the final
                            # restoration, leaving constant neighbour rotations.
                            p2_real = p2.real * step_cos - p2.imag * step_sin
                            p2_imag = p2.real * step_sin + p2.imag * step_cos
                        else:
                            p2_real = p2.real
                            p2_imag = p2.imag
                        if b0 >= 1 and b0 <= max_bin_eff - 3:
                            p0 = spectrum[b0 - 1]
                            p3 = spectrum[b0 + 2]
                            if phase_aware:
                                p0_real = p0.real * step_cos + p0.imag * step_sin
                                p0_imag = p0.imag * step_cos - p0.real * step_sin
                                p3_real = p3.real * step_2_cos - p3.imag * step_2_sin
                                p3_imag = p3.real * step_2_sin + p3.imag * step_2_cos
                            else:
                                p0_real = p0.real
                                p0_imag = p0.imag
                                p3_real = p3.real
                                p3_imag = p3.imag
                            frac2 = frac * frac
                            frac3 = frac2 * frac
                            interp_real = half * (
                                two * p1_real
                                + (-p0_real + p2_real) * frac
                                + (two * p0_real - five * p1_real + four * p2_real - p3_real) * frac2
                                + (-p0_real + three * p1_real - three * p2_real + p3_real) * frac3
                            )
                            interp_imag = half * (
                                two * p1_imag
                                + (-p0_imag + p2_imag) * frac
                                + (two * p0_imag - five * p1_imag + four * p2_imag - p3_imag) * frac2
                                + (-p0_imag + three * p1_imag - three * p2_imag + p3_imag) * frac3
                            )
                        else:
                            interp_real = p1_real + (p2_real - p1_real) * frac
                            interp_imag = p1_imag + (p2_imag - p1_imag) * frac

                        if phase_aware:
                            restore_argument = range_phase_step * frac
                            restore_cos = np.cos(restore_argument)
                            restore_sin = np.sin(restore_argument)
                            restored_real = interp_real * restore_cos + interp_imag * restore_sin
                            restored_imag = interp_imag * restore_cos - interp_real * restore_sin
                            interp_real = restored_real
                            interp_imag = restored_imag

                        contribution_real = (
                            interp_real * phase_real - interp_imag * phase_imag
                        ) * coherent_weight
                        contribution_imag = (
                            interp_real * phase_imag + interp_imag * phase_real
                        ) * coherent_weight
                        frame_real[voxel_i, frame_i] += contribution_real
                        frame_imag[voxel_i, frame_i] += contribution_imag

            total_power = zero
            for frame_i in range(n_frames):
                value_real = frame_real[voxel_i, frame_i]
                value_imag = frame_imag[voxel_i, frame_i]
                total_power += value_real * value_real + value_imag * value_imag
            output[voxel_i] = total_power

        return output


else:
    _back_projection_power_mimo_geometry_numba_kernel = None

@dataclass(frozen=True)
class SyntheticApertureData:
    """Apertura SAR/MIMO appiattita, pronta per il beamforming range-angolo."""

    snapshot_cube: np.ndarray
    x_position_m: np.ndarray
    x_phase_center_m: np.ndarray
    x_element_m: np.ndarray


def phase_sign_normalize(value: Any, *, field_name: str = "phase_sign") -> int:
    """Valida il segno della fase di rifocalizzazione bistatica (solo ±1)."""
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
    """Converte potenza lineare in dB, proteggendo il logaritmo dallo zero."""
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
    """Restituisce le coordinate TX/RX per canale virtuale, centrate sull'array.

    Per la geometria IWR1443 2Tx/4Rx usa il layout fisico noto; altre
    configurazioni richiedono offset espliciti per non introdurre assunzioni
    geometriche silenziose.
    """
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

    # L'ordine TX-major/RX-minor deve coincidere con il reshape dei dati MIMO:
    # ogni canale virtuale conserva così le coordinate fisiche TX e RX corrette.
    tx_idx = np.repeat(np.arange(n_tx_i, dtype=np.int32), n_rx_i)
    rx_idx = np.tile(np.arange(n_rx_i, dtype=np.int32), n_tx_i)
    x_tx_ant_m = tx_base_m[tx_idx].astype(np.float32, copy=False)
    x_rx_ant_m = rx_base_m[rx_idx].astype(np.float32, copy=False)

    # Trasla TX e RX con lo stesso riferimento. La fase relativa resta invariata
    # mentre la geometria è espressa attorno all'origine della scena SAR.
    center_m = (np.mean((x_tx_ant_m + x_rx_ant_m).astype(np.float32, copy=False), dtype=np.float32) * np.float32(0.5)).astype(np.float32, copy=False)
    x_tx_ant_m = (x_tx_ant_m - center_m).astype(np.float32, copy=False)
    x_rx_ant_m = (x_rx_ant_m - center_m).astype(np.float32, copy=False)
    return x_tx_ant_m, x_rx_ant_m


_MIN_RANGE_FFT_OVERSAMPLING = 6.0


def _positive_integer(value: Any, *, field_name: str) -> int:
    """Return a strictly positive integer without silently truncating floats."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} deve essere un intero > 0, trovato: {value!r}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} deve essere un intero > 0, trovato: {value!r}") from exc
    if not np.isfinite(numeric) or numeric <= 0.0 or not numeric.is_integer():
        raise ValueError(f"{field_name} deve essere un intero > 0, trovato: {value!r}")
    return int(numeric)


def _range_fft_interpolation_phase_step(
    *,
    samples_used: int | None,
    nfft: int | None,
) -> np.float32:
    """Validate Range-FFT metadata and return its causal DFT phase step.

    A finite record whose non-zero samples occupy indices ``0..M-1`` has a
    linear DFT phase associated with the temporal centre ``(M - 1) / 2``.
    Removing that phase before interpolation avoids cancellation between
    adjacent complex FFT bins.  A zero return value preserves the historical
    interpolation when the caller supplies neither piece of metadata.

    Sixfold FFT oversampling keeps the worst-case amplitude error of an
    unwindowed tone below 0.1 dB, including the first interval next to DC
    where cubic interpolation falls back to its linear boundary rule.  Lower
    ratios are rejected instead of returning a silently attenuated sample.
    """
    if samples_used is None and nfft is None:
        return np.float32(0.0)
    if samples_used is None or nfft is None:
        raise ValueError(
            "Interpolazione Range FFT phase-aware: samples_used e nfft "
            "devono essere specificati insieme"
        )

    samples_i = _positive_integer(samples_used, field_name="samples_used")
    nfft_i = _positive_integer(nfft, field_name="nfft")
    if samples_i > nfft_i:
        raise ValueError(
            "Interpolazione Range FFT phase-aware: "
            f"samples_used={samples_i} supera nfft={nfft_i}"
        )

    if nfft_i < 2:
        raise ValueError(
            "Interpolazione Range FFT phase-aware: nfft deve essere >= 2"
        )

    oversampling = float(nfft_i) / float(samples_i)
    if samples_i > 1 and oversampling < _MIN_RANGE_FFT_OVERSAMPLING:
        raise ValueError(
            "Interpolazione Range FFT phase-aware: rapporto di oversampling "
            f"nfft/samples_used={oversampling:.3f} insufficiente; "
            f"richiesto >= {_MIN_RANGE_FFT_OVERSAMPLING:.1f}"
        )

    sample_center = 0.5 * float(samples_i - 1)
    return np.float32((2.0 * np.pi * sample_center) / float(nfft_i))


def _interp_linear_complex_impl(
    spec: np.ndarray,
    b_chunk: np.ndarray,
    max_bin_local: int,
    phase_step: np.float32,
) -> np.ndarray:
    b0_chunk = np.floor(b_chunk).astype(np.int32, copy=False)
    np.clip(b0_chunk, 0, max_bin_local - 2, out=b0_chunk)
    frac = (b_chunk - b0_chunk.astype(np.float32, copy=False)).astype(np.float32, copy=False)
    s0 = spec[b0_chunk]
    s1 = spec[b0_chunk + 1]
    if phase_step == np.float32(0.0):
        return (s0 + (s1 - s0) * frac).astype(np.complex64, copy=False)

    # Work relative to b0.  The common exp(+j*phase_step*b0) term cancels
    # against the final phase restoration, so only one constant bin rotation
    # and one fractional rotation are needed.
    rotate_next = np.complex64(np.exp(1j * phase_step))
    s1_centered = (s1 * rotate_next).astype(np.complex64, copy=False)
    centered = (s0 + (s1_centered - s0) * frac).astype(np.complex64, copy=False)
    restore = np.exp(-1j * phase_step * frac).astype(np.complex64, copy=False)
    return (centered * restore).astype(np.complex64, copy=False)


def _interp_linear_complex(
    spec: np.ndarray,
    b_chunk: np.ndarray,
    max_bin_local: int,
    *,
    samples_used: int | None = None,
    nfft: int | None = None,
) -> np.ndarray:
    """Interpolate complex FFT bins, optionally compensating causal DFT phase."""
    phase_step = _range_fft_interpolation_phase_step(
        samples_used=samples_used,
        nfft=nfft,
    )
    return _interp_linear_complex_impl(spec, b_chunk, max_bin_local, phase_step)


def _interp_cubic_complex_impl(
    spec: np.ndarray,
    b_chunk: np.ndarray,
    max_bin_local: int,
    phase_step: np.float32,
) -> np.ndarray:
    out = _interp_linear_complex_impl(spec, b_chunk, max_bin_local, phase_step)
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

    if phase_step != np.float32(0.0):
        rotate_previous = np.complex64(np.exp(-1j * phase_step))
        rotate_next = np.complex64(np.exp(1j * phase_step))
        rotate_next_2 = np.complex64(np.exp(2j * phase_step))
        p0 = (p0 * rotate_previous).astype(np.complex64, copy=False)
        p2 = (p2 * rotate_next).astype(np.complex64, copy=False)
        p3 = (p3 * rotate_next_2).astype(np.complex64, copy=False)

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
    if phase_step != np.float32(0.0):
        restore = np.exp(-1j * phase_step * frac).astype(np.complex64, copy=False)
        cubic = (cubic * restore).astype(np.complex64, copy=False)
    out[idx] = cubic
    return out.astype(np.complex64, copy=False)


def _interp_cubic_complex(
    spec: np.ndarray,
    b_chunk: np.ndarray,
    max_bin_local: int,
    *,
    samples_used: int | None = None,
    nfft: int | None = None,
) -> np.ndarray:
    """Catmull-Rom FFT interpolation with optional causal-phase correction."""
    phase_step = _range_fft_interpolation_phase_step(
        samples_used=samples_used,
        nfft=nfft,
    )
    return _interp_cubic_complex_impl(spec, b_chunk, max_bin_local, phase_step)


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
    # I frame restano separati: la backprojection somma prima in modo coerente
    # per ciascun frame e combina le rispettive potenze solo alla fine.
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
    range_offset_m: float = 0.0,
    phase_sign: int = -1,
    residual_video_phase: int | str = 0,
    slope_hz_s: float | None = None,
    chunk_size: int = 16384,
    pose_weights: np.ndarray | None = None,
    channel_weights: np.ndarray | None = None,
    use_numba: bool = True,
    range_samples_used: int | None = None,
    nfft_range: int | None = None,
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
    applied here.  When locally available, Numba parallelises independent
    output voxels; pass ``use_numba=False`` only for numerical reference tests.
    Pass ``range_samples_used`` and ``nfft_range`` together to enable the
    phase-aware interpolation required for a causal, zero-padded Range FFT.
    Omitting both retains the historical interpolation for API compatibility.
    ``range_offset_m`` is an optional signed equivalent one-way range bias:
    it is added only while addressing the range FFT bins, leaving physical
    voxel geometry and propagation-phase compensation unchanged.
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

    try:
        range_offset_f = float(range_offset_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("range_offset_m deve essere un numero finito") from exc
    if not np.isfinite(range_offset_f):
        raise ValueError("range_offset_m deve essere un numero finito")

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
    range_phase_step = _range_fft_interpolation_phase_step(
        samples_used=range_samples_used,
        nfft=nfft_range,
    )
    if nfft_range is not None and int(nfft_range) < n_bins_avail:
        raise ValueError(
            f"nfft_range={int(nfft_range)} inferiore ai bin disponibili={n_bins_avail}"
        )

    output_shape = voxels.shape[:-1]
    max_bin_eff = max(0, min(int(max_bin), n_bins_avail))
    if max_bin_eff < 2 or n_frames <= 0:
        return np.zeros(output_shape, dtype=np.float32)

    # Per un percorso bistatico il bin di range corrisponde a metà della
    # distanza TX→voxel→RX: bin = ((r_tx + r_rx) / 2 + offset) / dr_m.
    # ``range_offset_m`` calibra il riferimento della Range FFT (ritardo
    # fisso di catena/cavo o zero-range); non altera la geometria né la fase
    # di propagazione, che devono restare quelle fisiche del voxel.
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

    # Il kernel JIT e il percorso NumPy ricevono gli stessi pesi e la stessa
    # convenzione di fase; ``use_numba=False`` serve quindi come riferimento.
    if bool(use_numba) and _back_projection_power_mimo_geometry_numba_kernel is not None:
        total_power = _back_projection_power_mimo_geometry_numba_kernel(
            snapshots,
            tx_global,
            rx_global,
            voxel_flat,
            int(max_bin_eff),
            inv_dr,
            np.float32(range_offset_f),
            phase_scale,
            rvp_phase_scale,
            range_phase_step,
            pose_weight,
            channel_weight,
        )
        return total_power.reshape(output_shape).astype(np.float32, copy=False)

    # Accumula pose e canali coerentemente all'interno di ogni frame. I frame
    # vengono poi sommati in potenza per non imporre coerenza tra acquisizioni.
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
                # Il chunking limita soltanto la memoria temporanea dei voxel;
                # non cambia la somma coerente su pose, canali o frame.
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
                bins = (
                    (np.float32(0.5) * r_total + np.float32(range_offset_f)) * inv_dr
                ).astype(np.float32, copy=False)
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
                    interp = _interp_cubic_complex_impl(
                        snapshots[pos_i, frame_i, ant_i],
                        valid_bins,
                        max_bin_eff,
                        range_phase_step,
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
    range_offset_m: float = 0.0,
    phase_sign: int = -1,
    residual_video_phase: int | str = 0,
    slope_hz_s: float | None = None,
    chunk_size: int = 16384,
    use_numba: bool = True,
    range_samples_used: int | None = None,
    nfft_range: int | None = None,
) -> np.ndarray:
    """Reference-compatible adapter for the historical linear z=0 BP.

    The general geometry kernel above is also used by the linear path through
    this adapter and evaluated in parallel. ``use_numba=False`` retains the
    original X-only loop/chunk traversal as a numerical reference
    implementation.  ``range_offset_m`` has the same signed equivalent
    one-way range convention as the general geometry function.
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

    try:
        range_offset_f = float(range_offset_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("range_offset_m deve essere un numero finito") from exc
    if not np.isfinite(range_offset_f):
        raise ValueError("range_offset_m deve essere un numero finito")

    phase_sign_i = phase_sign_normalize(phase_sign, field_name="phase_sign")
    rvp_sign_i = residual_video_phase_sign_normalize(
        residual_video_phase,
        field_name="residual_video_phase",
    )
    range_phase_step = _range_fft_interpolation_phase_step(
        samples_used=range_samples_used,
        nfft=nfft_range,
    )
    n_pos, n_frames, n_ant, n_bins_avail = (int(v) for v in snapshots.shape)
    if nfft_range is not None and int(nfft_range) < n_bins_avail:
        raise ValueError(
            f"nfft_range={int(nfft_range)} inferiore ai bin disponibili={n_bins_avail}"
        )
    max_bin_eff = max(0, min(int(max_bin), int(n_bins_avail)))
    if max_bin_eff < 2 or n_frames <= 0:
        return np.zeros_like(x_grid, dtype=np.float32)

    x_flat = x_grid.reshape(-1).astype(np.float32, copy=False)
    y_flat = y_grid.reshape(-1).astype(np.float32, copy=False)

    if bool(use_numba) and _back_projection_power_mimo_geometry_numba_kernel is not None:
        x_pos = np.asarray(x_pos_m, dtype=np.float32).reshape(-1)
        x_tx = np.asarray(x_tx_ant_m, dtype=np.float32).reshape(-1)
        x_rx = np.asarray(x_rx_ant_m, dtype=np.float32).reshape(-1)
        tx_global = np.zeros((n_pos, n_ant, 3), dtype=np.float32)
        rx_global = np.zeros((n_pos, n_ant, 3), dtype=np.float32)
        tx_global[:, :, 0] = x_pos[:, None] + x_tx[None, :]
        rx_global[:, :, 0] = x_pos[:, None] + x_rx[None, :]
        voxel_flat = np.zeros((int(x_flat.size), 3), dtype=np.float32)
        voxel_flat[:, 0] = x_flat
        voxel_flat[:, 1] = y_flat
        return back_projection_power_mimo_geometry(
            snapshots,
            tx_global,
            rx_global,
            voxel_flat,
            dr_m=float(dr_m),
            fc_hz=float(fc_hz),
            c_m_s=float(c_m_s),
            max_bin=int(max_bin_eff),
            range_offset_m=range_offset_f,
            phase_sign=phase_sign_i,
            residual_video_phase=rvp_sign_i,
            slope_hz_s=slope_hz_s,
            chunk_size=chunk_size,
            use_numba=True,
            range_samples_used=range_samples_used,
            nfft_range=nfft_range,
        ).reshape(x_grid.shape).astype(np.float32, copy=False)

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
            b = (
                (np.float32(0.5) * r_total + np.float32(range_offset_f)) * inv_dr
            ).astype(np.float32, copy=False)
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
                    interp = _interp_cubic_complex_impl(
                        snapshots[pos_i, frame_i, ant_i],
                        bins,
                        max_bin_eff,
                        range_phase_step,
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
    measured_positions_m: np.ndarray | None = None,
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
    if measured_positions_m is None:
        pitch = float(x_pitch_m)
        if not np.isfinite(pitch) or pitch <= 0.0:
            raise ValueError(f"x_pitch_m non valido: {x_pitch_m!r}")
        x_position_m = (pos_values * np.float32(pitch)).astype(np.float32, copy=False)
    else:
        x_position_m = np.asarray(measured_positions_m, dtype=np.float32).reshape(-1)
        if x_position_m.size != int(n_pos) or not np.all(np.isfinite(x_position_m)):
            raise ValueError(
                "measured_positions_m non coerente con l'apertura sintetica: "
                f"size={x_position_m.size}, atteso={int(n_pos)}"
            )
    # Il centro di fase virtuale è il punto medio TX/RX; la posizione meccanica
    # della slitta lo trasla per costruire l'apertura sintetica completa.
    x_phase_center_m = (
        np.float32(0.5) * (x_tx + x_rx).astype(np.float32, copy=False)
    ).astype(np.float32, copy=False)
    x_element_m = (
        x_position_m[:, None] + x_phase_center_m[None, :]
    ).astype(np.float32, copy=False)
    # Da [pos, frame, ant, bin] a [frame, 1, bin, synthetic_ant].  Fill the
    # final cube directly so a non-contiguous transpose does not create an
    # additional full-size temporary copy.
    snapshot_cube = np.empty(
        (int(n_frames), 1, int(n_bins), int(n_pos * n_ant)),
        dtype=np.complex64,
    )
    for pos_i in range(int(n_pos)):
        first = int(pos_i * n_ant)
        last = int(first + n_ant)
        snapshot_cube[:, 0, :, first:last] = snapshots[pos_i].transpose(0, 2, 1)
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
    """Restituisce il passo uniforme dell'apertura in lambda, o ``None`` se irregolare."""
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
    range_offset_m: float = 0.0,
    phase_sign: int = -1,
    residual_video_phase: int | str = 0,
    slope_hz_s: float | None = None,
    chunk_size: int = 16384,
    use_numba: bool = True,
    range_samples_used: int | None = None,
    nfft_range: int | None = None,
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
        range_offset_m=range_offset_m,
        phase_sign=phase_sign,
        residual_video_phase=residual_video_phase,
        slope_hz_s=slope_hz_s,
        chunk_size=chunk_size,
        use_numba=use_numba,
        range_samples_used=range_samples_used,
        nfft_range=nfft_range,
    )
    return power_image_to_db(total_power)
