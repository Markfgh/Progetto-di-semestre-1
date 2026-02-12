"""dsp_processing.py
DSP pipeline + selezione finestre (ex dsp_selection.py).
"""

from __future__ import annotations

import queue as pyqueue
from dataclasses import dataclass
from typing import Any, Dict, Literal, Tuple

import numpy as np
import scipy.fft as fft
from multiprocessing.sharedctypes import Synchronized


# ---------------------------- WINDOW SELECTION ----------------------------

WindowType = Literal["rectangular", "hanning", "hamming", "blackman"]


@dataclass(frozen=True)
class DspSelection:
    window_range: WindowType = "blackman"
    window_angle: WindowType = "hanning"


def _get_window_1d(win_type: str, size: int) -> np.ndarray:
    wt = win_type.lower()
    if wt == "rectangular":
        return np.ones(size, dtype=np.float32)
    if wt == "hanning":
        return np.hanning(size).astype(np.float32, copy=False)
    if wt == "hamming":
        return np.hamming(size).astype(np.float32, copy=False)
    if wt == "blackman":
        return np.blackman(size).astype(np.float32, copy=False)
    raise ValueError(f"Unknown window type: {win_type!r}. Use: rectangular|hanning|hamming|blackman")


def build_windows(selection: DspSelection, samples: int, virtual_ant: int) -> Tuple[np.ndarray, np.ndarray]:
    """Finestre già reshape() per broadcasting nel tuo reshape/transpose."""
    w_range = _get_window_1d(selection.window_range, samples).reshape(1, 1, 1, 1, samples)
    w_angle = _get_window_1d(selection.window_angle, virtual_ant).reshape(1, 1, 1, virtual_ant)
    return w_range, w_angle


def selection_from_yaml_dict(cfg: Dict[str, Any]) -> DspSelection:
    """Atteso YAML:
    dsp:
      window_range: blackman
      window_angle: hanning
    """
    dsp = cfg.get("dsp", {}) or {}
    return DspSelection(
        window_range=str(dsp.get("window_range", "blackman")),
        window_angle=str(dsp.get("window_angle", "hanning")),
    )


# ---------------------------- DSP WORKER ----------------------------

def dsp_worker(
    frame_queue,
    free_slots,
    shm_frames,
    gui_queue,
    gui_put_drops: Synchronized,
    frame_get_ok: Synchronized,
    gui_put_ok: Synchronized,
    cfg: Dict[str, Any],
    params: Dict[str, Any],
) -> None:
    """Consuma frame dal ring (slot index), fa DSP e push heatmap verso GUI."""

    debug_stats = bool(params.get("debug_stats", False))

    # parametri
    samples = int(params["samples"])
    chirps = int(params["chirps"])
    rx = int(params["rx"])
    tx = int(params["tx"])
    x_frames = int(params["x_frames"])
    bytes_per_frame = int(params["bytes_per_frame"])
    virtual_ant = int(params["virtual_ant"])

    # finestre DSP
    selection = selection_from_yaml_dict(cfg)
    window_range, window_angle = build_windows(selection, samples=samples, virtual_ant=virtual_ant)

    heatmap_ema = None
    batch_slots = []
    shm_view = memoryview(shm_frames).cast("B")

    # buffer locale contiguo per X_FRAMES
    i16_per_frame = bytes_per_frame // 2
    i16_batch = np.empty((x_frames, i16_per_frame), dtype=np.int16)

    while True:
        try:
            slot = frame_queue.get(timeout=0.5)
            if debug_stats:
                with frame_get_ok.get_lock():
                    frame_get_ok.value += 1
        except pyqueue.Empty:
            continue

        # Se X_FRAMES==1, tieni solo l'ultimo frame disponibile (bassa latenza)
        if x_frames == 1:
            last = slot
            while True:
                try:
                    old = frame_queue.get_nowait()
                    if debug_stats:
                        with frame_get_ok.get_lock():
                            frame_get_ok.value += 1
                    free_slots.put(old)
                    last = old
                except pyqueue.Empty:
                    break
            slot = last

        batch_slots.append(slot)
        if len(batch_slots) < x_frames:
            continue

        if len(batch_slots) > x_frames:
            to_free = batch_slots[:-x_frames]
            for s in to_free:
                free_slots.put(s)
            batch_slots = batch_slots[-x_frames:]

        # copia ring -> buffer numpy contiguo
        for k, s in enumerate(batch_slots):
            base = s * bytes_per_frame
            frame_mv = shm_view[base: base + bytes_per_frame]
            i16_batch[k, :] = np.frombuffer(frame_mv, dtype=np.int16, count=i16_per_frame)

        i16 = i16_batch.reshape(-1)  # contiguo

        n_blocks = i16.size // 8
        if n_blocks == 0:
            for s in batch_slots:
                free_slots.put(s)
            batch_slots.clear()
            continue

        i16 = i16[:n_blocks * 8].reshape(n_blocks, 8)

        # i16 -> complex64 (re,im)
        re_i16 = i16[:, :4]
        im_i16 = i16[:, 4:]
        complex_data = np.empty(re_i16.size, dtype=np.complex64)
        complex_data.real = re_i16.reshape(-1)
        complex_data.imag = im_i16.reshape(-1)

        total_samples_needed = x_frames * chirps * samples * rx
        if complex_data.size < total_samples_needed:
            for s in batch_slots:
                free_slots.put(s)
            batch_slots.clear()
            continue

        raw_buffer = complex_data[:total_samples_needed]

        heatmap_ema = process_buffer(
            raw_buffer=raw_buffer,
            w_range=window_range,
            w_angle=window_angle,
            heatmap_ema=heatmap_ema,
            alpha=float(params.get("ema_alpha", 0.2)),
            gui_queue=gui_queue,
            gui_put_drops=gui_put_drops,
            gui_put_ok=gui_put_ok,
            params=params,
        )

        for s in batch_slots:
            free_slots.put(s)
        batch_slots.clear()


def process_buffer(
    raw_buffer: np.ndarray,
    w_range: np.ndarray,
    w_angle: np.ndarray,
    heatmap_ema: np.ndarray | None,
    alpha: float,
    gui_queue,
    gui_put_drops: Synchronized,
    gui_put_ok: Synchronized,
    params: Dict[str, Any],
):
    """DSP core: range FFT + angle FFT + EMA + log + push heatmap."""

    debug_stats = bool(params.get("debug_stats", False))

    # fisica/FFT
    c = float(params["c"])
    fs = float(params["fs"])
    slope = float(params["slope"])
    nfft_range = int(params["nfft_range"])
    nfft_angle = int(params["nfft_angle"])
    range_max_display = float(params["range_max_display"])
    fft_workers = int(params.get("fft_workers", 6))

    # acquisizione
    samples = int(params["samples"])
    chirps = int(params["chirps"])
    rx = int(params["rx"])
    tx = int(params["tx"])
    x_frames = int(params["x_frames"])
    virtual_ant = int(params["virtual_ant"])

    # range bins da visualizzare
    dr = c * fs / (2.0 * slope * nfft_range)
    max_bin = int(np.floor(range_max_display / dr))
    max_bin = max(1, min(max_bin, nfft_range // 2))

    try:
        # A) reshape + transpose
        data = raw_buffer.reshape(x_frames, chirps // tx, tx, samples, rx).transpose(0, 1, 2, 4, 3)
        if not data.flags["C_CONTIGUOUS"]:
            data = np.ascontiguousarray(data)

        # B) preprocess + window
        data -= data.mean(axis=-1, keepdims=True, dtype=np.complex64)
        data *= w_range

        # C) range FFT
        range_fft = fft.fft(data, n=nfft_range, axis=-1, workers=fft_workers, overwrite_x=True)

        # D) virtual array + window angle
        virtual_array = range_fft.transpose(0, 1, 4, 2, 3).reshape(x_frames, chirps // tx, nfft_range, virtual_ant)
        virtual_array *= w_angle

        # E) angle FFT
        angle_fft = fft.fft(virtual_array, n=nfft_angle, axis=-1, workers=fft_workers, overwrite_x=True)
        angle_fft = fft.fftshift(angle_fft, axes=-1)

        # F) power map (mean su frame e chirp-loop)
        re = angle_fft.real
        im = angle_fft.imag
        heatmap = (re * re + im * im).mean(axis=(0, 1))

        # G) EMA
        if heatmap_ema is None:
            heatmap_ema = heatmap
        else:
            heatmap_ema *= (1.0 - alpha)
            heatmap_ema += (alpha * heatmap)

        # H) dB + normalizzazione
        heatmap_db = 10.0 * np.log10(heatmap_ema + 1e-12)
        view_db = heatmap_db[:max_bin, :]
        if view_db.size > 0:
            view_db -= np.max(view_db)

        # I) push verso GUI
        try:
            gui_queue.put_nowait(view_db)
            if debug_stats:
                with gui_put_ok.get_lock():
                    gui_put_ok.value += 1
        except pyqueue.Full:
            if debug_stats:
                with gui_put_drops.get_lock():
                    gui_put_drops.value += 1

        return heatmap_ema

    except Exception as e:
        print(f"[DSP ERR] {e}")
        return heatmap_ema
