from __future__ import annotations

from dataclasses import dataclass
import queue as pyqueue
import time
from multiprocessing.sharedctypes import Synchronized
from typing import Any, Literal

import numpy as np
import scipy.fft as fft

# Realtime DSP only: window setup, batch processing, and worker loop.
WindowType = Literal["none", "rectangular", "hanning", "hamming", "blackman"]
_VALID_WINDOWS = {"none", "rectangular", "hanning", "hamming", "blackman"}

MeanAxis = Literal["frame", "loop", "tx", "sample", "range_bin", "rx"]
BackgroundMode = Literal["ema", "running_mean", "window_mean", "frozen"]
AngleProcessingMode = Literal["fft", "bartlett", "mvdr"]
HeatmapSpatialFilterMode = Literal["none", "gaussian_3x3"]
SlowTimeMode = Literal["none", "mean_subtraction", "highpass", "doppler_fft"]
_VALID_MEAN_AXES = {"frame", "loop", "tx", "sample", "range_bin", "rx"}
_VALID_BACKGROUND_MODES = {"ema", "running_mean", "window_mean", "frozen"}
_VALID_ANGLE_PROCESSING_MODES = {"fft", "bartlett", "mvdr"}
_VALID_HEATMAP_SPATIAL_FILTER_MODES = {"none", "gaussian_3x3"}
_VALID_SLOW_TIME_MODES = {"none", "mean_subtraction", "highpass", "doppler_fft"}

_MEAN_AXIS_INDEX = {
    "frame": 0,
    "loop": 1,
    "tx": 2,
    "sample": 3,
    "range_bin": 3,
    "rx": 4,
}

# The DSP config is passed as a single packed struct to avoid relying on globals in the worker process.
@dataclass(frozen=True)
class DspSelection:
    window_range: WindowType = "blackman"
    window_angle: WindowType = "hanning"


@dataclass(frozen=True)
class MeanSelection:
    axes: tuple[MeanAxis, ...] = ("tx",)
    enabled: bool = False


@dataclass(frozen=True)
class BackgroundSubtractionConfig:
    enabled: bool = False
    mode: BackgroundMode = "ema"
    alpha: float = 0.02
    init_frames: int = 20
    window_frames: int = 20
    clamp_positive_only: bool = False


@dataclass
class BackgroundSubtractionState:
    model: np.ndarray | None = None
    init_sum: np.ndarray | None = None
    init_count: int = 0
    running_sum: np.ndarray | None = None
    running_count: int = 0
    window_ring: np.ndarray | None = None
    window_sum: np.ndarray | None = None
    window_count: int = 0
    window_head: int = 0


@dataclass(frozen=True)
class LoopAverageConfig:
    enabled: bool = False


@dataclass(frozen=True)
class AngleProcessingConfig:
    mode: AngleProcessingMode = "fft"
    mvdr_diagonal_loading: float = 0.01


@dataclass(frozen=True)
class HeatmapEMAConfig:
    enabled: bool = True
    alpha: float = 0.2


@dataclass(frozen=True)
class HeatmapSpatialFilterConfig:
    enabled: bool = False
    mode: HeatmapSpatialFilterMode = "gaussian_3x3"


@dataclass(frozen=True)
class SlowTimeConfig:
    enabled: bool = False
    mode: SlowTimeMode = "none"
    highpass_beta: float = 0.9
    doppler_fft_shift: bool = True
    doppler_zero_notch: bool = False


@dataclass(frozen=True)
class RealtimeDSPConfig:
    # Packed config passed to the DSP process to avoid relying on globals.
    c: float
    fs: float
    slope: float
    samples: int
    chirps: int
    rx: int
    tx: int
    x_frames: int
    bytes_per_frame: int
    nfft_range: int
    nfft_angle: int
    range_max_display: float
    range_profile_count: int
    virtual_ant: int
    fft_workers: int
    debug_stats: bool

# Return a 1D FFT window of the requested type as float32.
def _get_window_1d(win_type: str, size: int) -> np.ndarray:
    wt = win_type.lower()
    if wt == "none":
        return np.ones(size, dtype=np.float32)
    if wt == "rectangular":
        return np.ones(size, dtype=np.float32)
    if wt == "hanning":
        return np.hanning(size).astype(np.float32, copy=False)
    if wt == "hamming":
        return np.hamming(size).astype(np.float32, copy=False)
    if wt == "blackman":
        return np.blackman(size).astype(np.float32, copy=False)
    raise ValueError(f"Unknown window type: {win_type!r}. Use: none|rectangular|hanning|hamming|blackman")


# Build range and angle windows with shapes ready for NumPy broadcasting
def build_windows(selection: DspSelection, samples: int, virtual_ant: int) -> tuple[np.ndarray, np.ndarray]:
    # Pre-shaped for NumPy broadcasting on range and virtual-array axes.
    w_range = _get_window_1d(selection.window_range, samples).reshape(1, 1, 1, samples, 1)
    w_angle = _get_window_1d(selection.window_angle, virtual_ant).reshape(1, 1, 1, virtual_ant)
    return w_range, w_angle


# Normalize a config value to a valid window type, falling back to the default.
def window_type_normalize(value: Any, default: WindowType = "hanning") -> WindowType:
    v = str(value or default).strip().lower()
    if v not in _VALID_WINDOWS:
        return default
    return v  # type: ignore[return-value]


def mean_axes_normalize(value: Any, default: tuple[MeanAxis, ...] = ("tx",)) -> tuple[MeanAxis, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        raw_axes = [value]
    else:
        try:
            raw_axes = list(value)
        except TypeError:
            return default

    axes_out: list[MeanAxis] = []
    seen: set[str] = set()
    for raw_axis in raw_axes:
        axis = str(raw_axis).strip().lower()
        if axis not in _VALID_MEAN_AXES or axis in seen:
            continue
        axes_out.append(axis)  # type: ignore[arg-type]
        seen.add(axis)
    return tuple(axes_out) if axes_out else default



# Build DSP window selection from the YAML config with validated defaults.
def selection_from_yaml_dict(cfg: dict[str, Any]) -> DspSelection:
    dsp = cfg.get("dsp", {}) or {}
    return DspSelection(
        window_range=window_type_normalize(dsp.get("window_range"), "blackman"),
        window_angle=window_type_normalize(dsp.get("window_angle"), "hanning"),
    )


def _mean_selection_from_yaml_dict(dsp: dict[str, Any], key: str, default_axes: tuple[MeanAxis, ...]) -> MeanSelection:
    block = dsp.get(key, {}) or {}
    return MeanSelection(
        axes=mean_axes_normalize(block.get("axes"), default_axes),
        enabled=bool(block.get("enabled", False)),
    )


def mean_selections_from_yaml_dict(cfg: dict[str, Any]) -> tuple[MeanSelection, MeanSelection]:
    dsp = cfg.get("dsp", {}) or {}
    return (
        _mean_selection_from_yaml_dict(dsp, "mean_before_range_fft", ("tx",)),
        _mean_selection_from_yaml_dict(dsp, "mean_after_range_fft", ("tx",)),
    )


def background_subtraction_from_yaml_dict(cfg: dict[str, Any]) -> BackgroundSubtractionConfig:
    dsp = cfg.get("dsp", {}) or {}
    bg = dsp.get("background_subtraction", {}) or {}
    mode = str(bg.get("mode", "ema")).strip().lower()
    if mode not in _VALID_BACKGROUND_MODES:
        mode = "ema"
    try:
        alpha = float(bg.get("alpha", 0.02))
    except (TypeError, ValueError):
        alpha = 0.02
    alpha = min(max(alpha, 0.0), 1.0)
    try:
        init_frames = int(bg.get("init_frames", 20))
    except (TypeError, ValueError):
        init_frames = 20
    try:
        window_frames = int(bg.get("window_frames", 20))
    except (TypeError, ValueError):
        window_frames = 20
    return BackgroundSubtractionConfig(
        enabled=bool(bg.get("enabled", False)),
        mode=mode,  # type: ignore[arg-type]
        alpha=alpha,
        init_frames=max(1, init_frames),
        window_frames=max(1, window_frames),
        clamp_positive_only=bool(bg.get("clamp_positive_only", False)),
    )


def loop_average_after_background_from_yaml_dict(cfg: dict[str, Any]) -> LoopAverageConfig:
    dsp = cfg.get("dsp", {}) or {}
    block = dsp.get("loop_average_after_background", {}) or {}
    return LoopAverageConfig(enabled=bool(block.get("enabled", False)))


def angle_processing_from_yaml_dict(cfg: dict[str, Any]) -> AngleProcessingConfig:
    dsp = cfg.get("dsp", {}) or {}
    block = dsp.get("angle_processing", {}) or {}
    mode_raw = str(block.get("mode", "fft")).strip().lower()
    mode: AngleProcessingMode = "fft"
    if mode_raw in _VALID_ANGLE_PROCESSING_MODES:
        if mode_raw == "bartlett":
            mode = "bartlett"
        elif mode_raw == "mvdr":
            mode = "mvdr"
    try:
        mvdr_diagonal_loading = float(block.get("mvdr_diagonal_loading", 0.01))
    except (TypeError, ValueError):
        mvdr_diagonal_loading = 0.01
    return AngleProcessingConfig(
        mode=mode,
        mvdr_diagonal_loading=max(0.0, mvdr_diagonal_loading),
    )


def heatmap_ema_from_yaml_dict(cfg: dict[str, Any]) -> HeatmapEMAConfig:
    dsp = cfg.get("dsp", {}) or {}
    block = dsp.get("heatmap_ema", {}) or {}
    try:
        alpha = float(block.get("alpha", 0.2))
    except (TypeError, ValueError):
        alpha = 0.2
    return HeatmapEMAConfig(
        enabled=bool(block.get("enabled", True)),
        alpha=min(max(alpha, 0.0), 1.0),
    )


def heatmap_spatial_filter_from_yaml_dict(cfg: dict[str, Any]) -> HeatmapSpatialFilterConfig:
    dsp = cfg.get("dsp", {}) or {}
    block = dsp.get("heatmap_spatial_filter", {}) or {}
    mode_raw = str(block.get("mode", "gaussian_3x3")).strip().lower()
    mode: HeatmapSpatialFilterMode = "gaussian_3x3"
    if mode_raw in _VALID_HEATMAP_SPATIAL_FILTER_MODES and mode_raw == "none":
        mode = "none"
    return HeatmapSpatialFilterConfig(
        enabled=bool(block.get("enabled", False)),
        mode=mode,
    )


def slow_time_from_yaml_dict(cfg: dict[str, Any]) -> SlowTimeConfig:
    dsp = cfg.get("dsp", {}) or {}
    block = dsp.get("slow_time", {}) or {}
    mode_raw = str(block.get("mode", "none")).strip().lower()
    mode: SlowTimeMode = "none"
    if mode_raw in _VALID_SLOW_TIME_MODES:
        if mode_raw == "mean_subtraction":
            mode = "mean_subtraction"
        elif mode_raw == "highpass":
            mode = "highpass"
        elif mode_raw == "doppler_fft":
            mode = "doppler_fft"
    try:
        highpass_beta = float(block.get("highpass_beta", 0.9))
    except (TypeError, ValueError):
        highpass_beta = 0.9
    return SlowTimeConfig(
        enabled=bool(block.get("enabled", False)),
        mode=mode,
        highpass_beta=min(max(highpass_beta, 0.0), 1.0),
        doppler_fft_shift=bool(block.get("doppler_fft_shift", True)),
        doppler_zero_notch=bool(block.get("doppler_zero_notch", False)),
    )


def subtract_selected_mean(data: np.ndarray, selection: MeanSelection) -> np.ndarray:
    if not selection.enabled or not selection.axes:
        return data
    axes = tuple(_MEAN_AXIS_INDEX[axis] for axis in selection.axes)
    mean = data.mean(axis=axes, keepdims=True, dtype=np.complex64)
    np.subtract(data, mean, out=data)
    return data


def apply_heatmap_spatial_filter(
    heatmap: np.ndarray,
    filter_cfg: HeatmapSpatialFilterConfig,
) -> np.ndarray:
    if not filter_cfg.enabled or filter_cfg.mode == "none":
        return heatmap
    if heatmap.ndim != 2 or heatmap.shape[0] < 1 or heatmap.shape[1] < 1:
        return heatmap
    # 3x3 Gaussian blur for display smoothing only.
    kernel = np.array(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        dtype=np.float32,
    ) / np.float32(16.0)
    padded = np.pad(heatmap.astype(np.float32, copy=False), ((1, 1), (1, 1)), mode="edge")
    out = (
        padded[:-2, :-2] * kernel[0, 0]
        + padded[:-2, 1:-1] * kernel[0, 1]
        + padded[:-2, 2:] * kernel[0, 2]
        + padded[1:-1, :-2] * kernel[1, 0]
        + padded[1:-1, 1:-1] * kernel[1, 1]
        + padded[1:-1, 2:] * kernel[1, 2]
        + padded[2:, :-2] * kernel[2, 0]
        + padded[2:, 1:-1] * kernel[2, 1]
        + padded[2:, 2:] * kernel[2, 2]
    )
    return out.astype(np.float32, copy=False)


def apply_slow_time_filter(
    data: np.ndarray,
    slow_time_cfg: SlowTimeConfig,
    *,
    fft_workers: int = 1,
) -> np.ndarray:
    if not slow_time_cfg.enabled or slow_time_cfg.mode == "none":
        return data
    if data.ndim != 5 or int(data.shape[1]) <= 0:
        return data

    if slow_time_cfg.mode == "mean_subtraction":
        return data - data.mean(axis=1, keepdims=True, dtype=np.complex64)

    if slow_time_cfg.mode == "highpass":
        out = np.empty_like(data)
        out[:, :1, :, :, :] = data[:, :1, :, :, :]
        beta = np.float32(slow_time_cfg.highpass_beta)
        # Slow-time high-pass IIR to emphasize chirp-to-chirp changes and suppress static clutter.
        for k in range(1, int(data.shape[1])):
            out[:, k : k + 1, :, :, :] = (
                beta * out[:, k - 1 : k, :, :, :]
                + data[:, k : k + 1, :, :, :]
                - data[:, k - 1 : k, :, :, :]
            )
        return out

    out = fft.fft(
        data,
        n=int(data.shape[1]),
        axis=1,
        workers=int(fft_workers),
        overwrite_x=False,
    )
    if slow_time_cfg.doppler_fft_shift:
        out = np.fft.fftshift(out, axes=1)
    if slow_time_cfg.doppler_zero_notch and out.shape[1] > 0:
        zero_idx = int(out.shape[1] // 2) if slow_time_cfg.doppler_fft_shift else 0
        out[:, zero_idx : zero_idx + 1, :, :, :] = 0
    return out.astype(np.complex64, copy=False)


def build_angle_steering_matrix(virtual_ant: int, nfft_angle: int) -> np.ndarray:
    ant_idx = np.arange(int(virtual_ant), dtype=np.float32)
    # Spatial frequency grid in fftshift order: u in [-1, 1).
    u = np.linspace(-1.0, 1.0, int(nfft_angle), endpoint=False, dtype=np.float32)
    phase = (-1j * np.pi) * ant_idx[:, None] * u[None, :]
    steering = np.exp(phase).astype(np.complex64, copy=False)
    col_energy = (steering.real * steering.real + steering.imag * steering.imag).sum(axis=0, dtype=np.float32)
    col_norm = np.sqrt(np.maximum(col_energy, np.float32(1e-8))).astype(np.float32, copy=False)
    steering /= col_norm[np.newaxis, :].astype(np.complex64, copy=False)
    return steering


def compute_angle_heatmap(
    virtual_array: np.ndarray,
    *,
    angle_cfg: AngleProcessingConfig,
    dsp_cfg: RealtimeDSPConfig,
    angle_steering: np.ndarray,
) -> np.ndarray:
    if angle_cfg.mode == "fft":
        angle_fft = fft.fft(
            virtual_array,
            n=dsp_cfg.nfft_angle,
            axis=-1,
            workers=dsp_cfg.fft_workers,
            overwrite_x=True,
        )
        re = angle_fft.real
        im = angle_fft.imag
        heatmap = (re * re + im * im).mean(axis=(0, 1), dtype=np.float32)
        return np.fft.fftshift(heatmap, axes=-1).astype(np.float32, copy=False)

    x = virtual_array.transpose(2, 0, 1, 3)
    n_range = int(x.shape[0])
    n_snap = int(x.shape[1] * x.shape[2])
    n_ant = int(x.shape[3])
    x = x.reshape(n_range, n_snap, n_ant).astype(np.complex64, copy=False)
    steering = angle_steering[:n_ant, :]

    if angle_cfg.mode == "bartlett":
        y = np.einsum("rsm,mk->rsk", x, np.conj(steering), optimize=True)
        yr = y.real
        yi = y.imag
        return (yr * yr + yi * yi).mean(axis=1, dtype=np.float32).astype(np.float32, copy=False)

    cov = np.einsum("rsm,rsn->rmn", np.conj(x), x, optimize=True).astype(np.complex64, copy=False)
    if n_snap > 0:
        cov /= np.float32(n_snap)
    if angle_cfg.mvdr_diagonal_loading > 0.0:
        ant = int(cov.shape[-1])
        tr = np.trace(cov, axis1=-2, axis2=-1).real.astype(np.float32, copy=False)
        tr /= np.float32(max(1, ant))
        eye = np.eye(ant, dtype=np.complex64)[None, :, :]
        load = (np.float32(angle_cfg.mvdr_diagonal_loading) * tr)[:, None, None].astype(np.complex64, copy=False)
        cov = cov + (load * eye)
    cov_inv = np.linalg.pinv(cov).astype(np.complex64, copy=False)
    den_left = np.einsum("rmn,nk->rmk", cov_inv, steering, optimize=True)
    den = np.einsum("mk,rmk->rk", np.conj(steering), den_left, optimize=True).real.astype(np.float32, copy=False)
    np.maximum(den, np.float32(1e-8), out=den)
    return (np.float32(1.0) / den).astype(np.float32, copy=False)


def apply_background_subtraction(
    range_fft: np.ndarray,
    bg_cfg: BackgroundSubtractionConfig,
    bg_state: BackgroundSubtractionState,
) -> np.ndarray:
    if not bg_cfg.enabled:
        return range_fft

    frame_sum = range_fft.sum(axis=0, dtype=np.complex64)
    batch_frames = int(range_fft.shape[0])
    model_initialized_now = False

    def _window_mean_push_batch() -> None:
        if batch_frames <= 0:
            return
        frame_shape = tuple(int(x) for x in range_fft.shape[1:])
        window_frames = int(max(1, bg_cfg.window_frames))
        ring = bg_state.window_ring
        sum_buf = bg_state.window_sum
        if (
            ring is None
            or sum_buf is None
            or int(ring.shape[0]) != window_frames
            or tuple(int(x) for x in ring.shape[1:]) != frame_shape
        ):
            ring = np.empty((window_frames,) + frame_shape, dtype=np.complex64)
            sum_buf = np.zeros(frame_shape, dtype=np.complex64)
            bg_state.window_count = 0
            bg_state.window_head = 0
            bg_state.window_ring = ring
            bg_state.window_sum = sum_buf

        assert ring is not None
        assert sum_buf is not None
        count = int(bg_state.window_count)
        head = int(bg_state.window_head)
        cap = int(ring.shape[0])
        for frame in range_fft:
            if count < cap:
                ring[head, ...] = frame
                sum_buf += ring[head, ...]
                count += 1
            else:
                sum_buf -= ring[head, ...]
                ring[head, ...] = frame
                sum_buf += ring[head, ...]
            head += 1
            if head >= cap:
                head = 0
        bg_state.window_count = int(count)
        bg_state.window_head = int(head)

    if bg_state.model is None:
        if bg_cfg.mode == "window_mean":
            _window_mean_push_batch()
            bg_state.init_count += batch_frames
            if bg_state.init_count < bg_cfg.init_frames:
                return range_fft
            if bg_state.window_sum is None or bg_state.window_count <= 0:
                return range_fft
            bg_state.model = np.asarray(
                bg_state.window_sum / np.float32(max(1, bg_state.window_count)),
                dtype=np.complex64,
            )
            model_initialized_now = True
        else:
            if bg_state.init_sum is None:
                bg_state.init_sum = np.zeros_like(frame_sum, dtype=np.complex64)
            bg_state.init_sum += frame_sum
            bg_state.init_count += batch_frames
            if bg_state.init_count < bg_cfg.init_frames:
                return range_fft
            bg_state.model = np.asarray(
                bg_state.init_sum / np.float32(bg_state.init_count),
                dtype=np.complex64,
            )
            if bg_cfg.mode == "running_mean":
                bg_state.running_sum = np.array(bg_state.init_sum, dtype=np.complex64, copy=True)
                bg_state.running_count = bg_state.init_count
            bg_state.init_sum = None
            model_initialized_now = True

    model = bg_state.model
    if model is None:
        return range_fft
    bg_broadcast = model.reshape((1,) + model.shape)
    if bg_cfg.clamp_positive_only:
        current_mag = np.abs(range_fft)
        bg_mag = np.abs(bg_broadcast)
        out_mag = np.maximum(current_mag - bg_mag, 0.0).astype(np.float32, copy=False)
        phase = np.exp(1j * np.angle(range_fft)).astype(np.complex64, copy=False)
        range_fft_out = (out_mag * phase).astype(np.complex64, copy=False)
    else:
        range_fft_out = range_fft - bg_broadcast
    # Avoid double counting the batch that completed background initialization.
    if model_initialized_now:
        return range_fft_out
    frame_mean = frame_sum / np.float32(max(1, batch_frames))
    if bg_cfg.mode == "ema":
        model *= np.float32(1.0 - bg_cfg.alpha)
        model += np.float32(bg_cfg.alpha) * frame_mean
        bg_state.model = model
    elif bg_cfg.mode == "running_mean":
        if bg_state.running_sum is None:
            bg_state.running_sum = np.array(frame_sum, dtype=np.complex64, copy=True)
            bg_state.running_count = batch_frames
        else:
            bg_state.running_sum += frame_sum
            bg_state.running_count += batch_frames
        bg_state.model = np.asarray(
            bg_state.running_sum / np.float32(max(1, bg_state.running_count)),
            dtype=np.complex64,
        )
    elif bg_cfg.mode == "window_mean":
        _window_mean_push_batch()
        if bg_state.window_sum is not None and bg_state.window_count > 0:
            bg_state.model = np.asarray(
                bg_state.window_sum / np.float32(max(1, bg_state.window_count)),
                dtype=np.complex64,
            )
    return range_fft_out


def _convert_rx4_iiiiqqqq_to_complex64(dst_frame: np.ndarray, src_i16_frame: np.ndarray) -> None:
    block_view = src_i16_frame.reshape(-1, 8)
    dst_frame.real[:] = block_view[:, :4].reshape(-1)  # type: ignore[index]
    dst_frame.imag[:] = block_view[:, 4:].reshape(-1)  # type: ignore[index]


# Process one DSP batch and publish the updated heatmap/profile views to the GUI buffers.
def process_buffer(
    raw_buffer: np.ndarray,
    n_frames: int,
    w_range: np.ndarray,
    w_angle: np.ndarray,
    mean_before_range_fft: MeanSelection,
    mean_after_range_fft: MeanSelection,
    slow_time_cfg: SlowTimeConfig,
    bg_subtraction: BackgroundSubtractionConfig,
    apply_loop_average_after_background: bool,
    angle_processing: AngleProcessingConfig,
    heatmap_ema_cfg: HeatmapEMAConfig,
    heatmap_spatial_filter_cfg: HeatmapSpatialFilterConfig,
    angle_steering: np.ndarray,
    bg_state: BackgroundSubtractionState,
    heatmap_ema: np.ndarray | None,
    gui_dbuf,
    gui_prof_dbuf,
    gui_h: int,
    gui_w: int,
    gui_latest_idx: Synchronized,
    gui_latest_seq: Synchronized,
    gui_lock,
    max_bin: int,
    normalize_to_peak: bool,
    profiles_out_buf: np.ndarray,
    stat_raw_min_db: Synchronized,
    stat_raw_max_db: Synchronized,
    stat_norm_min_db: Synchronized,
    stat_norm_max_db: Synchronized,
    dsp_cfg: RealtimeDSPConfig,
) -> np.ndarray | None:
    try:

        # Raw complex stream -> Reshape -> radar tensor [frame, loop, tx, sample, rx].
        data = raw_buffer.reshape(
            n_frames,
            dsp_cfg.chirps // dsp_cfg.tx,
            dsp_cfg.tx,
            dsp_cfg.samples,
            dsp_cfg.rx,
        )

        data = subtract_selected_mean(data, mean_before_range_fft)
        data *= w_range


        # Compute the range FFT along the sample axis.
        range_fft = fft.fft(data,n=dsp_cfg.nfft_range,axis=3,workers=dsp_cfg.fft_workers,overwrite_x=True,)
        # Apply the selected slow-time filter on the loop axis after the range FFT.
        range_fft = apply_slow_time_filter(
            range_fft,
            slow_time_cfg,
            fft_workers=dsp_cfg.fft_workers,
        )
        range_fft = subtract_selected_mean(range_fft, mean_after_range_fft)
        range_fft = apply_background_subtraction(
            range_fft,
            bg_subtraction,
            bg_state,
        )

        if apply_loop_average_after_background:
            # Collapse the loop dimension while preserving the axis for the downstream pipeline.
            range_fft = range_fft.mean(axis=1, keepdims=True, dtype=np.complex64)

        # Build the virtual array after trimming range bins to limit memory traffic.
        range_fft = range_fft[:, :, :, :max_bin, :]
        va = range_fft.transpose(0, 1, 3, 2, 4)
        va = np.ascontiguousarray(va)
        virtual_array = va.reshape(
            n_frames,
            dsp_cfg.chirps // dsp_cfg.tx,
            max_bin,
            dsp_cfg.virtual_ant,
        )

        prof_re = virtual_array.real
        prof_im = virtual_array.imag
        profiles_pow = (prof_re * prof_re + prof_im * prof_im).mean(axis=(0, 1))
        profiles_db = np.array(profiles_pow, dtype=np.float32, copy=True)
        np.add(profiles_db, np.float32(1e-12), out=profiles_db)
        np.log10(profiles_db, out=profiles_db)
        profiles_db *= np.float32(10.0)
        profiles_db_va = profiles_db.transpose(1, 0)
        profiles_out = profiles_out_buf
        profiles_out.fill(np.float32(-120.0))
        copy_rows = min(int(dsp_cfg.range_profile_count), int(profiles_db_va.shape[0]))
        if copy_rows > 0:
            profiles_out[:copy_rows, :] = profiles_db_va[:copy_rows, :].astype(np.float32, copy=False)


        virtual_array *= w_angle

        heatmap = compute_angle_heatmap(
            virtual_array,
            angle_cfg=angle_processing,
            dsp_cfg=dsp_cfg,
            angle_steering=angle_steering,
        )

        if not heatmap_ema_cfg.enabled: #bypass
            heatmap_ema = heatmap
        elif heatmap_ema is None: #initialize
            heatmap_ema = heatmap
        else: # update
            heatmap_ema *= (1.0 - heatmap_ema_cfg.alpha)
            heatmap_ema += (heatmap_ema_cfg.alpha * heatmap)
        heatmap_ema = apply_heatmap_spatial_filter(heatmap_ema, heatmap_spatial_filter_cfg)

        # Convert to dB and optionally normalize to the peak for display.
        heatmap_db = np.array(heatmap_ema, dtype=np.float32, copy=True)
        np.add(heatmap_db, np.float32(1e-12), out=heatmap_db)
        np.log10(heatmap_db, out=heatmap_db)
        heatmap_db *= np.float32(10.0)

        view_db = heatmap_db
        if view_db.size > 0:
            raw_max = float(np.max(view_db))
            if dsp_cfg.debug_stats:
                raw_min = float(np.min(view_db))
                norm_max = 0.0
                norm_min = float(raw_min - raw_max)
                try:
                    with stat_raw_min_db.get_lock():
                        stat_raw_min_db.value = raw_min
                    with stat_raw_max_db.get_lock():
                        stat_raw_max_db.value = raw_max
                    with stat_norm_min_db.get_lock():
                        stat_norm_min_db.value = norm_min
                    with stat_norm_max_db.get_lock():
                        stat_norm_max_db.value = norm_max
                except Exception:
                    pass
            if normalize_to_peak:
                view_db -= raw_max

        # Latest-wins publish to the GUI double buffer.
        with gui_lock:
            prev_idx = int(gui_latest_idx.value)
            next_idx = 1 if prev_idx == 0 else 0
            base = next_idx * int(gui_h) * int(gui_w)
            dst = np.frombuffer(gui_dbuf, dtype=np.float32, count=int(gui_h) * int(gui_w), offset=base * 4)
            dst.fill(-120.0)
            flat = view_db.astype(np.float32, copy=False).reshape(-1)
            n = min(dst.size, flat.size)
            if n > 0:
                dst[:n] = flat[:n]

            prof_base = next_idx * int(dsp_cfg.range_profile_count) * int(gui_h)
            dst_prof = np.frombuffer(
                gui_prof_dbuf,
                dtype=np.float32,
                count=int(dsp_cfg.range_profile_count) * int(gui_h),
                offset=prof_base * 4,
            )
            dst_prof.fill(-120.0)
            prof_flat = profiles_out.reshape(-1)
            n_prof = min(dst_prof.size, prof_flat.size)
            if n_prof > 0:
                dst_prof[:n_prof] = prof_flat[:n_prof]
            gui_latest_idx.value = next_idx
            gui_latest_seq.value = int(gui_latest_seq.value) + 1
        return heatmap_ema

    except Exception as e:
        print(f"[DSP ERR] {e}")
        return heatmap_ema


def dsp_worker(
    free_slots,
    dsp_ready_queue,
    shm_frames,
    slot_state,
    slot_ok,
    slot_usemask,
    slot_pub_seq,
    publish_lock,
    gui_dbuf,
    gui_prof_dbuf,
    gui_h: int,
    gui_w: int,
    gui_latest_idx: Synchronized,
    gui_latest_seq: Synchronized,
    gui_lock,
    dsp_skip: Synchronized,
    dsp_ms_avg: Synchronized,
    dsp_ms_p95: Synchronized,
    norm_to_peak: Synchronized,
    stat_raw_min_db: Synchronized,
    stat_raw_max_db: Synchronized,
    stat_norm_min_db: Synchronized,
    stat_norm_max_db: Synchronized,
    stop_evt,
    cfg_dict: dict[str, Any],
    dsp_cfg: RealtimeDSPConfig,
) -> None:
    selection = selection_from_yaml_dict(cfg_dict)
    mean_before_range_fft, mean_after_range_fft = mean_selections_from_yaml_dict(cfg_dict)
    slow_time_cfg = slow_time_from_yaml_dict(cfg_dict)
    bg_subtraction = background_subtraction_from_yaml_dict(cfg_dict)
    loop_average_after_background = loop_average_after_background_from_yaml_dict(cfg_dict)
    angle_processing = angle_processing_from_yaml_dict(cfg_dict)
    heatmap_ema_cfg = heatmap_ema_from_yaml_dict(cfg_dict)
    heatmap_spatial_filter_cfg = heatmap_spatial_filter_from_yaml_dict(cfg_dict)
    window_range, window_angle = build_windows(
        selection,
        samples=dsp_cfg.samples,
        virtual_ant=dsp_cfg.virtual_ant,
    )
    angle_steering = build_angle_steering_matrix(
        virtual_ant=dsp_cfg.virtual_ant,
        nfft_angle=dsp_cfg.nfft_angle,
    )

    dr = dsp_cfg.c * dsp_cfg.fs / (2.0 * dsp_cfg.slope * dsp_cfg.nfft_range)
    max_bin = int(np.floor(dsp_cfg.range_max_display / dr))
    max_bin = max(1, min(max_bin, dsp_cfg.nfft_range // 2))

    i16_per_frame = dsp_cfg.bytes_per_frame // 2
    total_samples_needed = dsp_cfg.x_frames * dsp_cfg.chirps * dsp_cfg.samples * dsp_cfg.rx
    complex_per_frame = dsp_cfg.chirps * dsp_cfg.samples * dsp_cfg.rx

    complex_data = np.zeros(total_samples_needed, dtype=np.complex64)
    profiles_out_buf = np.empty((dsp_cfg.range_profile_count, max_bin), dtype=np.float32)

    shm_view = memoryview(shm_frames).cast("B")
    n_slots = len(slot_state)
    heatmap_ema = None
    bg_state = BackgroundSubtractionState()
    warned_loop_average_after_doppler = False
    dsp_ms_samples: list[float] = []
    dsp_stat_last_flush = time.perf_counter()
    dsp_ms_acc_sum = 0.0
    dsp_ms_acc_count = 0
    slot_i16_views = [
        np.frombuffer(
            shm_view[s * dsp_cfg.bytes_per_frame : (s + 1) * dsp_cfg.bytes_per_frame],
            dtype=np.int16,
        )
        for s in range(n_slots)
    ]

    if dsp_cfg.rx != 4:
        raise ValueError("DSP: conversione I/Q attuale assume RX=4 (packing IIIIQQQQ).")

    def _release_slots_dsp(slots) -> None:
        # Release the DSP ownership bit; the slot is recycled only when no consumer needs it.
        if not slots:
            return
        to_free = []
        try:
            with publish_lock:
                for s in slots:
                    si = int(s)
                    m = int(slot_usemask[si]) & (~1)
                    slot_usemask[si] = m
                    if m == 0:
                        slot_state[si] = 0
                        to_free.append(si)
        except Exception:
            return
        for si in to_free:
            try:
                free_slots.put_nowait(int(si))
            except Exception:
                pass

    while True:
        if stop_evt.is_set():
            break

        drained = []
        try:
            first_item = dsp_ready_queue.get(timeout=0.001)
        except pyqueue.Empty:
            continue
        drained.append(first_item)
        while True:
            try:
                drained.append(dsp_ready_queue.get_nowait())
            except pyqueue.Empty:
                break

        ready = []
        with publish_lock:
            for item in drained:
                try:
                    seq = int(item[0])
                    s = int(item[1])
                except Exception:
                    continue
                if s < 0 or s >= n_slots:
                    continue
                if int(slot_state[s]) != 1:
                    continue
                if int(slot_pub_seq[s]) != seq:
                    continue
                if (int(slot_usemask[s]) & 1) == 0:
                    continue
                ready.append((seq, s, int(slot_ok[s])))

        if not ready:
            continue

        # Keep only the newest frames so the display stays real-time under load.
        drop_slots = []
        if len(ready) > dsp_cfg.x_frames:
            drop_slots = [int(s) for _, s, _ in ready[:-dsp_cfg.x_frames]]
            ready = ready[-dsp_cfg.x_frames :]

        if drop_slots:
            if dsp_cfg.debug_stats and dsp_skip is not None:
                with dsp_skip.get_lock():
                    dsp_skip.value += len(drop_slots)
            _release_slots_dsp(drop_slots)

        proc_slots = []
        bad_slots = []
        for _, s, ok in ready:
            if ok == 1:
                proc_slots.append(int(s))
            else:
                bad_slots.append(int(s))
        if bad_slots:
            _release_slots_dsp(bad_slots)
        if not proc_slots:
            continue

        n_proc = min(len(proc_slots), dsp_cfg.x_frames)
        slots_to_process = proc_slots[:n_proc]

        # Release slots early so RX/logger can keep moving while DSP computes.
        _release_slots_dsp(slots_to_process)

        # Current packing is IIIIQQQQ for RX=4, converted in-place to complex64.
        n_cplx = n_proc * complex_per_frame
        complex_view = complex_data[:n_cplx]
        for k, s in enumerate(slots_to_process):
            start = int(k * complex_per_frame)
            end = int(start + complex_per_frame)
            _convert_rx4_iiiiqqqq_to_complex64(
                complex_view[start:end],
                slot_i16_views[int(s)],
            )

        t0_proc = time.perf_counter()
        try:
            with norm_to_peak.get_lock():
                normalize_to_peak = bool(norm_to_peak.value)
        except Exception:
            normalize_to_peak = True
        apply_loop_average_after_background = bool(loop_average_after_background.enabled)
        if slow_time_cfg.enabled and slow_time_cfg.mode == "doppler_fft" and apply_loop_average_after_background:
            if not warned_loop_average_after_doppler:
                print("[DSP WARN] loop_average_after_background skipped because slow_time.mode=doppler_fft.")
                warned_loop_average_after_doppler = True
            apply_loop_average_after_background = False
        heatmap_ema = process_buffer(
            complex_view,
            n_proc,
            window_range,
            window_angle,
            mean_before_range_fft,
            mean_after_range_fft,
            slow_time_cfg,
            bg_subtraction,
            apply_loop_average_after_background,
            angle_processing,
            heatmap_ema_cfg,
            heatmap_spatial_filter_cfg,
            angle_steering,
            bg_state,
            heatmap_ema,
            gui_dbuf,
            gui_prof_dbuf,
            gui_h,
            gui_w,
            gui_latest_idx,
            gui_latest_seq,
            gui_lock,
            max_bin,
            normalize_to_peak,
            profiles_out_buf,
            stat_raw_min_db,
            stat_raw_max_db,
            stat_norm_min_db,
            stat_norm_max_db,
            dsp_cfg,
        )
        t1_proc = time.perf_counter()
        if n_proc > 0 and dsp_cfg.debug_stats:
            ms_per_frame = ((t1_proc - t0_proc) * 1000.0) / float(n_proc)
            dsp_ms_acc_sum += ms_per_frame * float(n_proc)
            dsp_ms_acc_count += int(n_proc)
            dsp_ms_samples.extend([ms_per_frame] * int(n_proc))
            if len(dsp_ms_samples) > 4096:
                dsp_ms_samples = dsp_ms_samples[-4096:]
            t_stats_now = time.perf_counter()
            if (t_stats_now - dsp_stat_last_flush) >= 1.0 and dsp_ms_acc_count > 0:
                avg_now = float(dsp_ms_acc_sum / float(dsp_ms_acc_count))
                p95_now = avg_now
                if dsp_ms_samples:
                    p95_now = float(np.percentile(dsp_ms_samples, 95))
                with dsp_ms_avg.get_lock():
                    dsp_ms_avg.value = avg_now
                with dsp_ms_p95.get_lock():
                    dsp_ms_p95.value = p95_now
                dsp_stat_last_flush = t_stats_now
                dsp_ms_acc_sum = 0.0
                dsp_ms_acc_count = 0
                dsp_ms_samples.clear()
