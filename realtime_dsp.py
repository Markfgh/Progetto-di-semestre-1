from __future__ import annotations

from dataclasses import dataclass, replace
import math
import queue as pyqueue
import time
from multiprocessing.sharedctypes import Synchronized
from typing import Any, Literal


import numpy as np
import pyfftw
import scipy.fft as fft

from tracker import MultiObjectTracker, Track, TrackerConfig, TrackingConfig

# Diciamo a SciPy di usare il motore di FFTW sotto il cofano
pyfftw.interfaces.cache.enable()
fft.set_global_backend(pyfftw.interfaces.scipy_fft)

# Realtime DSP only: window setup, batch processing, and worker loop.
WindowType = Literal["none", "rectangular", "hanning", "hamming", "blackman"]
_VALID_WINDOWS = {"none", "rectangular", "hanning", "hamming", "blackman"}

MeanAxis = Literal["frame", "loop", "tx", "sample", "range_bin", "rx"]
BackgroundMode = Literal["ema", "running_mean", "window_mean", "frozen"]
AngleProcessingMode = Literal["fft", "bartlett", "mvdr"]
HeatmapSpatialFilterMode = Literal["none", "gaussian_3x3"]
DisplayProjectionMode = Literal["polar_stretched", "cartesian"]
DisplayProjectionInterp = Literal["nearest", "bilinear"]
SlowTimeMode = Literal["none", "mean_subtraction", "highpass", "doppler_fft"]
DetectionThresholdMode = Literal["relative", "absolute"]
DetectionSource = Literal["static", "moving", "fused"]
_VALID_MEAN_AXES = {"frame", "loop", "tx", "sample", "range_bin", "rx"}
_VALID_BACKGROUND_MODES = {"ema", "running_mean", "window_mean", "frozen"}
_VALID_ANGLE_PROCESSING_MODES = {"fft", "bartlett", "mvdr"}
_VALID_HEATMAP_SPATIAL_FILTER_MODES = {"none", "gaussian_3x3"}
_VALID_DISPLAY_PROJECTION_MODES = {"polar_stretched", "cartesian"}
_VALID_DISPLAY_PROJECTION_INTERPS = {"nearest", "bilinear"}
_VALID_SLOW_TIME_MODES = {"none", "mean_subtraction", "highpass", "doppler_fft"}
_VALID_THRESHOLD_MODES = {"relative", "absolute"}

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
    window_doppler: WindowType = "hanning"
    window_angle: WindowType = "hanning"


@dataclass(frozen=True)
class CalibrationConfig:
    enabled: bool = True
    vector_real: tuple[float, ...] = (1.0,) * 8
    vector_imag: tuple[float, ...] = (0.0,) * 8


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
    aggregation: str = "frame_loop"
    frame_index: int = 0
    loop_index: int = 0


@dataclass(frozen=True)
class HeatmapEMAConfig:
    enabled: bool = True
    alpha: float = 0.2


@dataclass(frozen=True)
class HeatmapSpatialFilterConfig:
    enabled: bool = False
    mode: HeatmapSpatialFilterMode = "gaussian_3x3"


@dataclass(frozen=True)
class DisplayProjectionConfig:
    projection_mode: DisplayProjectionMode = "polar_stretched"
    projection_interp: DisplayProjectionInterp = "nearest"
    crossrange_max_m: float | None = None
    crossrange_auto: bool = False


@dataclass(frozen=True)
class SlowTimeConfig:
    enabled: bool = False
    mode: SlowTimeMode = "none"
    highpass_beta: float = 0.9
    doppler_fft_shift: bool = True
    doppler_zero_notch: bool = False


@dataclass(frozen=True)
class PostRangeFftFilterConfig:
    mean_after_range_fft: MeanSelection = MeanSelection(enabled=False)
    slow_time: SlowTimeConfig = SlowTimeConfig(enabled=False)
    background_subtraction: BackgroundSubtractionConfig = BackgroundSubtractionConfig(enabled=False)
    loop_average_after_background: LoopAverageConfig = LoopAverageConfig(enabled=False)


@dataclass(frozen=True)
class DetectionConfigStatic:
    enabled: bool = True
    threshold_mode: DetectionThresholdMode = "relative"
    threshold_db: float = -10.0
    localmax_range_bins: int = 2
    localmax_angle_bins: int = 2
    min_power_db: float = 5.0
    max_detections: int = 64


@dataclass(frozen=True)
class DetectionConfigMoving:
    enabled: bool = True
    threshold_mode: DetectionThresholdMode = "relative"
    threshold_db: float = -12.0
    localmax_range_bins: int = 2
    localmax_doppler_bins: int = 1
    zero_doppler_exclusion_bins: int = 1
    min_power_db: float = 5.0
    doppler_fft_shift: bool = True
    max_detections: int = 64


@dataclass(frozen=True)
class FusionConfig:
    enabled: bool = True
    merge_xy_m: float = 0.40
    merge_range_m: float = 0.30
    merge_angle_deg: float = 5.0
    prefer_moving_when_doppler_valid: bool = True


@dataclass
class Detection:
    range_bin: int
    angle_bin: int | None
    doppler_bin: int | None
    range_m: float
    angle_deg: float
    doppler_mps: float | None
    x_m: float
    y_m: float
    power_lin: float
    power_db: float
    source: DetectionSource


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
    range_max_processing_m: float = 0.0
    calibration: CalibrationConfig = CalibrationConfig()


@dataclass(frozen=True)
class VirtualArrayGeometry:
    order_flat: np.ndarray
    phase_centers_lambda: np.ndarray
    identity_order: bool
    uniform_half_lambda: bool
    uniform_spacing_lambda: float | None

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
def build_windows(
    selection: DspSelection,
    samples: int,
    n_loops: int,
    virtual_ant: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Pre-shaped for NumPy broadcasting on range and virtual-array axes.
    w_range = _get_window_1d(selection.window_range, samples).reshape(1, 1, 1, samples, 1)
    w_doppler = _get_window_1d(selection.window_doppler, n_loops).reshape(1, n_loops, 1, 1, 1)
    w_angle = _get_window_1d(selection.window_angle, virtual_ant).reshape(1, 1, 1, virtual_ant)
    return w_range, w_doppler, w_angle


def _window_is_identity(window_type: WindowType) -> bool:
    return str(window_type).strip().lower() in {"none", "rectangular"}


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
        window_doppler=window_type_normalize(dsp.get("window_doppler"), "hanning"),
        window_angle=window_type_normalize(dsp.get("window_angle"), "hanning"),
    )


def _calibration_vector_from_yaml(raw_value: Any, *, size: int, default: float) -> tuple[float, ...]:
    size = max(1, int(size))
    if isinstance(raw_value, (list, tuple)):
        raw_items = list(raw_value)
    else:
        raw_items = []
    out: list[float] = []
    for raw_item in raw_items[:size]:
        try:
            val = float(raw_item)
        except (TypeError, ValueError):
            val = float(default)
        out.append(val)
    if len(out) < size:
        out.extend([float(default)] * (size - len(out)))
    return tuple(out)


def calibration_from_yaml_dict(cfg: dict[str, Any], *, virtual_ant: int = 8) -> CalibrationConfig:
    dsp = cfg.get("dsp", {}) or {}
    block = dsp.get("calibration", {}) or {}
    size = max(1, int(virtual_ant))
    return CalibrationConfig(
        enabled=bool(block.get("enabled", True)),
        vector_real=_calibration_vector_from_yaml(block.get("vector_real"), size=size, default=1.0),
        vector_imag=_calibration_vector_from_yaml(block.get("vector_imag"), size=size, default=0.0),
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
    aggregation_raw = str(block.get("aggregation", "frame_loop")).strip().lower()
    if aggregation_raw not in {"frame_loop", "frame", "loop", "none"}:
        aggregation_raw = "frame_loop"
    try:
        frame_index = int(block.get("frame_index", 0))
    except (TypeError, ValueError):
        frame_index = 0
    try:
        loop_index = int(block.get("loop_index", 0))
    except (TypeError, ValueError):
        loop_index = 0
    return AngleProcessingConfig(
        mode=mode,
        mvdr_diagonal_loading=max(0.0, mvdr_diagonal_loading),
        aggregation=aggregation_raw,
        frame_index=max(0, frame_index),
        loop_index=max(0, loop_index),
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


def display_projection_from_yaml_dict(cfg: dict[str, Any]) -> DisplayProjectionConfig:
    display = cfg.get("display", {}) or {}
    mode_raw = str(display.get("projection_mode", "polar_stretched")).strip().lower()
    mode: DisplayProjectionMode = "polar_stretched"
    if mode_raw in _VALID_DISPLAY_PROJECTION_MODES and mode_raw == "cartesian":
        mode = "cartesian"

    interp_raw = str(display.get("projection_interp", "nearest")).strip().lower()
    interp: DisplayProjectionInterp = "nearest"
    if interp_raw in _VALID_DISPLAY_PROJECTION_INTERPS and interp_raw == "bilinear":
        interp = "bilinear"

    crossrange_raw = display.get("crossrange_max_m", display.get("crossrange_max", None))
    crossrange_auto = isinstance(crossrange_raw, str) and crossrange_raw.strip().lower() == "auto"
    crossrange_max_m = None
    if not crossrange_auto:
        crossrange_max_m = _to_optional_float(crossrange_raw)
        if crossrange_max_m is not None and crossrange_max_m <= 0.0:
            crossrange_max_m = None

    return DisplayProjectionConfig(
        projection_mode=mode,
        projection_interp=interp,
        crossrange_max_m=crossrange_max_m,
        crossrange_auto=bool(crossrange_auto),
    )


def _slow_time_from_block(block: dict[str, Any]) -> SlowTimeConfig:
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


def _background_subtraction_from_block(block: dict[str, Any]) -> BackgroundSubtractionConfig:
    mode = str(block.get("mode", "ema")).strip().lower()
    if mode not in _VALID_BACKGROUND_MODES:
        mode = "ema"
    try:
        alpha = float(block.get("alpha", 0.02))
    except (TypeError, ValueError):
        alpha = 0.02
    alpha = min(max(alpha, 0.0), 1.0)
    try:
        init_frames = int(block.get("init_frames", 20))
    except (TypeError, ValueError):
        init_frames = 20
    try:
        window_frames = int(block.get("window_frames", 20))
    except (TypeError, ValueError):
        window_frames = 20
    return BackgroundSubtractionConfig(
        enabled=bool(block.get("enabled", False)),
        mode=mode,  # type: ignore[arg-type]
        alpha=alpha,
        init_frames=max(1, init_frames),
        window_frames=max(1, window_frames),
        clamp_positive_only=bool(block.get("clamp_positive_only", False)),
    )


def _loop_average_after_background_from_block(block: dict[str, Any]) -> LoopAverageConfig:
    return LoopAverageConfig(enabled=bool(block.get("enabled", False)))


def slow_time_from_yaml_dict(cfg: dict[str, Any]) -> SlowTimeConfig:
    # Legacy compatibility wrapper: maps to the display branch filters.
    return display_post_range_fft_filters_from_yaml_dict(cfg).slow_time


def background_subtraction_from_yaml_dict(cfg: dict[str, Any]) -> BackgroundSubtractionConfig:
    # Legacy compatibility wrapper: maps to the display branch filters.
    return display_post_range_fft_filters_from_yaml_dict(cfg).background_subtraction


def loop_average_after_background_from_yaml_dict(cfg: dict[str, Any]) -> LoopAverageConfig:
    # Legacy compatibility wrapper: maps to the display branch filters.
    return display_post_range_fft_filters_from_yaml_dict(cfg).loop_average_after_background


def display_post_range_fft_filters_from_yaml_dict(cfg: dict[str, Any]) -> PostRangeFftFilterConfig:
    dsp = cfg.get("dsp", {}) or {}
    branch = dsp.get("display_filters", {}) or {}
    return PostRangeFftFilterConfig(
        mean_after_range_fft=_mean_selection_from_yaml_dict(
            {"mean_after_range_fft": branch.get("mean_after_range_fft", dsp.get("mean_after_range_fft", {}))},
            "mean_after_range_fft",
            ("tx",),
        ),
        slow_time=_slow_time_from_block(branch.get("slow_time", dsp.get("slow_time", {})) or {}),
        background_subtraction=_background_subtraction_from_block(
            branch.get("background_subtraction", dsp.get("background_subtraction", {})) or {}
        ),
        loop_average_after_background=_loop_average_after_background_from_block(
            branch.get("loop_average_after_background", dsp.get("loop_average_after_background", {})) or {}
        ),
    )


def detection_static_post_range_fft_filters_from_yaml_dict(cfg: dict[str, Any]) -> PostRangeFftFilterConfig:
    dsp = cfg.get("dsp", {}) or {}
    # `detection_filters` is kept as a legacy alias for backward compatibility.
    branch = dsp.get("detection_static_filters", dsp.get("detection_filters", {})) or {}
    return PostRangeFftFilterConfig(
        mean_after_range_fft=_mean_selection_from_yaml_dict(
            {"mean_after_range_fft": branch.get("mean_after_range_fft", {})},
            "mean_after_range_fft",
            ("tx",),
        ),
        slow_time=_slow_time_from_block(branch.get("slow_time", {}) or {}),
        background_subtraction=_background_subtraction_from_block(branch.get("background_subtraction", {}) or {}),
        loop_average_after_background=_loop_average_after_background_from_block(
            branch.get("loop_average_after_background", {}) or {}
        ),
    )


def detection_moving_pre_doppler_filters_from_yaml_dict(cfg: dict[str, Any]) -> PostRangeFftFilterConfig:
    dsp = cfg.get("dsp", {}) or {}
    branch = dsp.get("detection_moving_pre_doppler_filters", {}) or {}
    return PostRangeFftFilterConfig(
        mean_after_range_fft=_mean_selection_from_yaml_dict(
            {"mean_after_range_fft": branch.get("mean_after_range_fft", {})},
            "mean_after_range_fft",
            ("tx",),
        ),
        slow_time=_slow_time_from_block(branch.get("slow_time", {}) or {}),
        background_subtraction=_background_subtraction_from_block(branch.get("background_subtraction", {}) or {}),
        loop_average_after_background=_loop_average_after_background_from_block(
            branch.get("loop_average_after_background", {}) or {}
        ),
    )


def _sanitize_post_range_fft_mean_selection(
    selection: MeanSelection,
    *,
    branch_label: str,
    slow_time_cfg: SlowTimeConfig,
) -> tuple[MeanSelection, list[str]]:
    if not selection.enabled or not selection.axes:
        return selection, []

    warnings: list[str] = []
    axes_out: list[MeanAxis] = []
    seen: set[str] = set()
    normalized_sample = False
    removed_loop = False

    for axis in selection.axes:
        axis_eff = axis
        if axis_eff == "sample":
            axis_eff = "range_bin"
            normalized_sample = True
        if slow_time_cfg.enabled and slow_time_cfg.mode == "doppler_fft" and axis_eff == "loop":
            removed_loop = True
            continue
        if axis_eff in seen:
            continue
        axes_out.append(axis_eff)
        seen.add(axis_eff)

    if normalized_sample:
        warnings.append(
            f"{branch_label}.mean_after_range_fft.axes uses 'sample' after range FFT; normalizing it to 'range_bin'."
        )
    if removed_loop:
        warnings.append(
            f"{branch_label}.mean_after_range_fft.axes contains 'loop' but slow_time.mode=doppler_fft turns that axis into Doppler bins; removing 'loop'."
        )

    if not axes_out:
        warnings.append(
            f"{branch_label}.mean_after_range_fft has no coherent axes left after sanitization; disabling mean_after_range_fft."
        )
        return replace(selection, enabled=False), warnings

    axes_tuple = tuple(axes_out)
    if axes_tuple == selection.axes:
        return selection, warnings
    return replace(selection, axes=axes_tuple), warnings


def _sanitize_post_range_fft_filters(
    filters_cfg: PostRangeFftFilterConfig,
    *,
    branch_label: str,
    allow_doppler_fft: bool,
    detection_safe_background: bool,
    allow_loop_average_after_background: bool,
) -> tuple[PostRangeFftFilterConfig, list[str]]:
    sanitized = filters_cfg
    warnings: list[str] = []

    if sanitized.slow_time.enabled and sanitized.slow_time.mode == "doppler_fft" and not allow_doppler_fft:
        warnings.append(
            f"{branch_label}.slow_time.mode=doppler_fft is unsupported in this branch; forcing slow_time off."
        )
        sanitized = replace(
            sanitized,
            slow_time=replace(sanitized.slow_time, enabled=False, mode="none", doppler_zero_notch=False),
        )

    if sanitized.loop_average_after_background.enabled and not allow_loop_average_after_background:
        warnings.append(
            f"{branch_label}.loop_average_after_background is unsupported in this branch; ignoring it."
        )
        sanitized = replace(sanitized, loop_average_after_background=LoopAverageConfig(enabled=False))

    mean_selection, mean_warnings = _sanitize_post_range_fft_mean_selection(
        sanitized.mean_after_range_fft,
        branch_label=branch_label,
        slow_time_cfg=sanitized.slow_time,
    )
    warnings.extend(mean_warnings)
    if mean_selection != sanitized.mean_after_range_fft:
        sanitized = replace(sanitized, mean_after_range_fft=mean_selection)

    if detection_safe_background and sanitized.background_subtraction.clamp_positive_only:
        warnings.append(
            f"{branch_label}.background_subtraction.clamp_positive_only is display-only and unsafe for detection; forcing it off."
        )
        sanitized = replace(
            sanitized,
            background_subtraction=replace(sanitized.background_subtraction, clamp_positive_only=False),
        )

    return sanitized, warnings


def sanitize_detection_moving_pre_doppler_filters(
    filters_cfg: PostRangeFftFilterConfig,
) -> tuple[PostRangeFftFilterConfig, list[str]]:
    return _sanitize_post_range_fft_filters(
        filters_cfg,
        branch_label="detection_moving_pre_doppler_filters",
        allow_doppler_fft=False,
        detection_safe_background=True,
        allow_loop_average_after_background=False,
    )


def sanitize_detection_static_post_range_fft_filters(
    filters_cfg: PostRangeFftFilterConfig,
) -> tuple[PostRangeFftFilterConfig, list[str]]:
    return _sanitize_post_range_fft_filters(
        filters_cfg,
        branch_label="detection_static_filters",
        allow_doppler_fft=True,
        detection_safe_background=True,
        allow_loop_average_after_background=True,
    )


def sanitize_display_post_range_fft_filters(
    filters_cfg: PostRangeFftFilterConfig,
) -> tuple[PostRangeFftFilterConfig, list[str]]:
    return _sanitize_post_range_fft_filters(
        filters_cfg,
        branch_label="display_filters",
        allow_doppler_fft=True,
        detection_safe_background=False,
        allow_loop_average_after_background=True,
    )


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return float(parsed)


def resolve_processing_range_max_m(cfg: dict[str, Any]) -> float:
    processing = cfg.get("processing", {}) or {}
    detection = cfg.get("detection", {}) or {}
    display = cfg.get("display", {}) or {}
    raw_value = None
    for block, key in (
        (processing, "range_max_m"),
        (detection, "range_max_m"),
        (processing, "range_max"),
        (detection, "range_max"),
    ):
        if key in block:
            raw_value = block.get(key)
            break
    if raw_value is None:
        raw_value = display.get("range_max_m", display.get("range_max", None))
    parsed = _to_optional_float(raw_value)
    if parsed is not None and parsed > 0.0:
        return float(parsed)
    display_fallback = _to_optional_float(display.get("range_max_m", display.get("range_max", None)))
    if display_fallback is not None and display_fallback > 0.0:
        return float(display_fallback)
    return 0.0


def _default_virtual_array_order_flat(virtual_ant: int) -> np.ndarray:
    return np.arange(max(0, int(virtual_ant)), dtype=np.int32)


def _default_virtual_array_phase_centers_lambda(virtual_ant: int) -> np.ndarray:
    return (np.float32(0.5) * np.arange(max(0, int(virtual_ant)), dtype=np.float32)).astype(np.float32, copy=False)


def _parse_virtual_array_order_entry(
    entry: Any,
    *,
    tx: int,
    rx: int,
    virtual_ant: int,
) -> int:
    if isinstance(entry, dict):
        if "flat" in entry:
            idx = int(entry["flat"])
        else:
            idx = int(entry["tx"]) * int(rx) + int(entry["rx"])
    elif isinstance(entry, (list, tuple)):
        if len(entry) == 2:
            idx = int(entry[0]) * int(rx) + int(entry[1])
        elif len(entry) == 1:
            idx = int(entry[0])
        else:
            raise ValueError(f"virtual_array_order entry non valido: {entry!r}")
    else:
        idx = int(entry)
    if idx < 0 or idx >= int(virtual_ant):
        raise ValueError(f"virtual_array_order index fuori range: {idx}")
    return int(idx)


def build_virtual_array_geometry_from_yaml_dict(
    cfg: dict[str, Any],
    dsp_cfg: RealtimeDSPConfig,
) -> tuple[VirtualArrayGeometry, list[str]]:
    block = cfg.get("antenna", {}) or cfg.get("virtual_array", {}) or {}
    virtual_ant = int(dsp_cfg.virtual_ant)
    default_order = _default_virtual_array_order_flat(virtual_ant)
    default_phase_centers = _default_virtual_array_phase_centers_lambda(virtual_ant)
    warnings: list[str] = []

    order_flat = default_order
    order_raw = block.get("virtual_array_order", block.get("order", None))
    if order_raw is not None:
        try:
            order_list = list(order_raw)
            if len(order_list) != virtual_ant:
                raise ValueError(
                    f"virtual_array_order size={len(order_list)} != virtual_ant={virtual_ant}"
                )
            parsed = np.asarray(
                [
                    _parse_virtual_array_order_entry(
                        item,
                        tx=int(dsp_cfg.tx),
                        rx=int(dsp_cfg.rx),
                        virtual_ant=virtual_ant,
                    )
                    for item in order_list
                ],
                dtype=np.int32,
            )
            if np.unique(parsed).size != virtual_ant:
                raise ValueError("virtual_array_order contiene duplicati")
            order_flat = parsed
        except Exception as exc:
            warnings.append(
                f"antenna.virtual_array_order non valido ({exc}); using legacy tx-major/rx order."
            )
            order_flat = default_order

    phase_centers_lambda = default_phase_centers
    phase_centers_lambda_raw = block.get(
        "virtual_array_phase_centers_lambda",
        block.get("phase_centers_lambda", None),
    )
    phase_centers_m_raw = block.get(
        "virtual_array_phase_centers_m",
        block.get("phase_centers_m", None),
    )
    if phase_centers_lambda_raw is not None or phase_centers_m_raw is not None:
        try:
            raw_values = phase_centers_lambda_raw if phase_centers_lambda_raw is not None else phase_centers_m_raw
            parsed = np.asarray(list(raw_values), dtype=np.float32)
            if parsed.size != virtual_ant or not np.all(np.isfinite(parsed)):
                raise ValueError(
                    f"phase center size/values incoerenti: size={parsed.size}, virtual_ant={virtual_ant}"
                )
            if phase_centers_lambda_raw is not None:
                phase_centers_lambda = parsed.astype(np.float32, copy=False)
            else:
                radar = cfg.get("radar", {}) or {}
                c_mps = _to_float(radar.get("c", 3e8), 3e8)
                fc_hz = _to_float(radar.get("fc", None), float("nan"))
                if not np.isfinite(fc_hz) or fc_hz <= 0.0:
                    raise ValueError("radar.fc mancante/invalid per convertire phase_centers_m in lambda")
                wavelength_m = float(c_mps) / float(fc_hz)
                if not np.isfinite(wavelength_m) or wavelength_m <= 0.0:
                    raise ValueError("wavelength_m non valida")
                phase_centers_lambda = (parsed / np.float32(wavelength_m)).astype(np.float32, copy=False)
        except Exception as exc:
            warnings.append(
                f"antenna.virtual_array_phase_centers_* non valido ({exc}); using legacy half-lambda ULA."
            )
            phase_centers_lambda = default_phase_centers

    identity_order = bool(np.array_equal(order_flat, default_order))
    uniform_spacing_lambda: float | None = None
    if phase_centers_lambda.size <= 1:
        uniform_half_lambda = True
        uniform_spacing_lambda = 0.5
    else:
        spacing = np.diff(phase_centers_lambda.astype(np.float64, copy=False))
        is_uniform_spacing = bool(
            np.all(np.isfinite(spacing))
            and np.allclose(spacing, spacing[0], atol=1e-4, rtol=0.0)
        )
        if is_uniform_spacing:
            uniform_spacing_lambda = float(spacing[0])
        uniform_half_lambda = bool(
            is_uniform_spacing
            and np.isclose(float(spacing[0]), 0.5, atol=1e-3, rtol=0.0)
        )

    geometry = VirtualArrayGeometry(
        order_flat=order_flat,
        phase_centers_lambda=phase_centers_lambda.astype(np.float32, copy=False),
        identity_order=identity_order,
        uniform_half_lambda=uniform_half_lambda,
        uniform_spacing_lambda=uniform_spacing_lambda,
    )
    return geometry, warnings


def resolve_display_crossrange_max_m(
    y_max_m: float,
    angle_axis_deg: np.ndarray,
    projection_cfg: DisplayProjectionConfig,
) -> float:
    if projection_cfg.crossrange_max_m is not None and projection_cfg.crossrange_max_m > 0.0:
        return float(projection_cfg.crossrange_max_m)
    if not projection_cfg.crossrange_auto:
        return max(0.0, float(y_max_m))

    angle_axis = np.asarray(angle_axis_deg, dtype=np.float32).reshape(-1)
    if angle_axis.size <= 0:
        return max(0.0, float(y_max_m))
    finite = np.isfinite(angle_axis)
    if not np.any(finite):
        return max(0.0, float(y_max_m))

    sin_abs = np.abs(np.sin(np.deg2rad(angle_axis[finite].astype(np.float64, copy=False))))
    if sin_abs.size <= 0:
        return max(0.0, float(y_max_m))
    return max(0.0, float(y_max_m) * float(np.max(sin_abs)))


def _threshold_mode_from_value(value: Any, default: DetectionThresholdMode = "relative") -> DetectionThresholdMode:
    mode_raw = str(value or default).strip().lower()
    if mode_raw not in _VALID_THRESHOLD_MODES:
        return default
    return mode_raw  # type: ignore[return-value]


def _tracking_cfg_value(
    tracking_block: dict[str, Any],
    tracker_block: dict[str, Any],
    *keys: str,
    default: Any,
) -> Any:
    for key in keys:
        if key in tracking_block:
            return tracking_block.get(key)
    for key in keys:
        if key in tracker_block:
            return tracker_block.get(key)
    return default


def detection_static_from_yaml_dict(cfg: dict[str, Any]) -> DetectionConfigStatic:
    block = cfg.get("detection_static", {}) or {}
    return DetectionConfigStatic(
        enabled=bool(block.get("enabled", True)),
        threshold_mode=_threshold_mode_from_value(block.get("threshold_mode"), "relative"),
        threshold_db=_to_float(block.get("threshold_db", -10.0), -10.0),
        localmax_range_bins=max(0, _to_int(block.get("localmax_range_bins", 2), 2)),
        localmax_angle_bins=max(0, _to_int(block.get("localmax_angle_bins", 2), 2)),
        min_power_db=_to_float(block.get("min_power_db", 5.0), 5.0),
        max_detections=max(1, _to_int(block.get("max_detections", 64), 64)),
    )


def detection_moving_from_yaml_dict(cfg: dict[str, Any]) -> DetectionConfigMoving:
    block = cfg.get("detection_moving", {}) or {}
    return DetectionConfigMoving(
        enabled=bool(block.get("enabled", True)),
        threshold_mode=_threshold_mode_from_value(block.get("threshold_mode"), "relative"),
        threshold_db=_to_float(block.get("threshold_db", -12.0), -12.0),
        localmax_range_bins=max(0, _to_int(block.get("localmax_range_bins", 2), 2)),
        localmax_doppler_bins=max(0, _to_int(block.get("localmax_doppler_bins", 1), 1)),
        zero_doppler_exclusion_bins=max(0, _to_int(block.get("zero_doppler_exclusion_bins", 1), 1)),
        min_power_db=_to_float(block.get("min_power_db", 5.0), 5.0),
        doppler_fft_shift=bool(block.get("doppler_fft_shift", True)),
        max_detections=max(1, _to_int(block.get("max_detections", 64), 64)),
    )


def fusion_from_yaml_dict(cfg: dict[str, Any]) -> FusionConfig:
    block = cfg.get("fusion", {}) or {}
    return FusionConfig(
        enabled=bool(block.get("enabled", True)),
        merge_xy_m=max(0.0, _to_float(block.get("merge_xy_m", 0.40), 0.40)),
        merge_range_m=max(0.0, _to_float(block.get("merge_range_m", 0.30), 0.30)),
        merge_angle_deg=max(0.0, _to_float(block.get("merge_angle_deg", 5.0), 5.0)),
        prefer_moving_when_doppler_valid=bool(block.get("prefer_moving_when_doppler_valid", True)),
    )


def tracking_from_yaml_dict(cfg: dict[str, Any]) -> TrackingConfig:
    tracking_block = cfg.get("tracking", {}) or {}
    tracker_block = cfg.get("tracker", {}) or {}
    dt_s = _to_optional_float(
        _tracking_cfg_value(
            tracking_block,
            tracker_block,
            "dt_s",
            "frame_dt_s",
            default=None,
        )
    )
    return TrackingConfig(
        enabled=bool(_tracking_cfg_value(tracking_block, tracker_block, "enabled", default=True)),
        dt_s=None if dt_s is None else max(1e-3, float(dt_s)),
        max_tracks=max(1, _to_int(_tracking_cfg_value(tracking_block, tracker_block, "max_tracks", default=30), 30)),
        min_hits_to_confirm=max(
            1,
            _to_int(
                _tracking_cfg_value(
                    tracking_block,
                    tracker_block,
                    "min_hits_to_confirm",
                    "min_confirmed_hits",
                    default=3,
                ),
                3,
            ),
        ),
        max_missed_tentative=max(
            0,
            _to_int(
                _tracking_cfg_value(
                    tracking_block,
                    tracker_block,
                    "max_missed_tentative",
                    default=2,
                ),
                2,
            ),
        ),
        max_missed_confirmed=max(
            0,
            _to_int(
                _tracking_cfg_value(
                    tracking_block,
                    tracker_block,
                    "max_missed_confirmed",
                    "max_missed_frames",
                    default=8,
                ),
                8,
            ),
        ),
        max_track_age=max(
            0,
            _to_int(
                _tracking_cfg_value(
                    tracking_block,
                    tracker_block,
                    "max_track_age",
                    default=0,
                ),
                0,
            ),
        ),
    )


def tracker_from_yaml_dict(cfg: dict[str, Any]) -> TrackerConfig:
    tracking_block = cfg.get("tracking", {}) or {}
    tracker_block = cfg.get("tracker", {}) or {}
    model = str(
        _tracking_cfg_value(
            tracking_block,
            tracker_block,
            "model",
            default="kalman_cv_2d",
        )
        or "kalman_cv_2d"
    ).strip().lower()
    if model != "kalman_cv_2d":
        print(f"[TRACK WARN] tracking.model={model!r} unsupported; forcing 'kalman_cv_2d'.")
        model = "kalman_cv_2d"
    return TrackerConfig(
        model=model,
        gating_xy_m=max(
            0.0,
            _to_float(
                _tracking_cfg_value(tracking_block, tracker_block, "gating_xy_m", default=0.75),
                0.75,
            ),
        ),
        gating_doppler_mps=max(
            0.0,
            _to_float(
                _tracking_cfg_value(tracking_block, tracker_block, "gating_doppler_mps", default=0.50),
                0.50,
            ),
        ),
        process_noise_pos=max(
            1e-4,
            _to_float(
                _tracking_cfg_value(
                    tracking_block,
                    tracker_block,
                    "process_noise_pos",
                    "process_noise_xy",
                    default=0.20,
                ),
                0.20,
            ),
        ),
        process_noise_vel=max(
            1e-4,
            _to_float(
                _tracking_cfg_value(
                    tracking_block,
                    tracker_block,
                    "process_noise_vel",
                    "process_noise_v",
                    default=1.00,
                ),
                1.00,
            ),
        ),
        measurement_noise_xy=max(
            1e-4,
            _to_float(
                _tracking_cfg_value(
                    tracking_block,
                    tracker_block,
                    "measurement_noise_xy",
                    default=0.25,
                ),
                0.25,
            ),
        ),
        moving_speed_threshold_mps=max(
            0.0,
            _to_float(
                _tracking_cfg_value(
                    tracking_block,
                    tracker_block,
                    "moving_speed_threshold_mps",
                    "dynamic_speed_threshold_mps",
                    default=0.20,
                ),
                0.20,
            ),
        ),
        stopped_speed_threshold_mps=max(
            0.0,
            _to_float(
                _tracking_cfg_value(
                    tracking_block,
                    tracker_block,
                    "stopped_speed_threshold_mps",
                    "static_speed_threshold_mps",
                    default=0.08,
                ),
                0.08,
            ),
        ),
        doppler_moving_threshold_mps=max(
            0.0,
            _to_float(
                _tracking_cfg_value(
                    tracking_block,
                    tracker_block,
                    "doppler_moving_threshold_mps",
                    "doppler_static_threshold_mps",
                    "dynamic_speed_threshold_mps",
                    default=0.12,
                ),
                0.12,
            ),
        ),
        motion_confirm_frames_moving=max(
            1,
            _to_int(
                _tracking_cfg_value(
                    tracking_block,
                    tracker_block,
                    "motion_confirm_frames_moving",
                    "classification_confirm_frames",
                    default=2,
                ),
                2,
            ),
        ),
        motion_confirm_frames_stopped=max(
            1,
            _to_int(
                _tracking_cfg_value(
                    tracking_block,
                    tracker_block,
                    "motion_confirm_frames_stopped",
                    "classification_confirm_frames",
                    default=3,
                ),
                3,
            ),
        ),
        stopped_memory_s=max(
            0.0,
            _to_float(
                _tracking_cfg_value(
                    tracking_block,
                    tracker_block,
                    "stopped_memory_s",
                    "stopped_hold_s",
                    "stopped_keepalive_s",
                    default=3.0,
                ),
                3.0,
            ),
        ),
        stopped_resume_gate_m=max(
            0.0,
            _to_float(
                _tracking_cfg_value(
                    tracking_block,
                    tracker_block,
                    "stopped_resume_gate_m",
                    "resume_xy_m",
                    default=0.90,
                ),
                0.90,
            ),
        ),
        stop_position_alpha=min(
            1.0,
            max(
                0.0,
                _to_float(
                    _tracking_cfg_value(
                        tracking_block,
                        tracker_block,
                        "stop_position_alpha",
                        default=0.25,
                    ),
                    0.25,
                ),
            ),
        ),
        birth_min_separation_m=max(
            0.0,
            _to_float(
                _tracking_cfg_value(
                    tracking_block,
                    tracker_block,
                    "birth_min_separation_m",
                    default=0.20,
                ),
                0.20,
            ),
        ),
        use_doppler_in_cost=bool(
            _tracking_cfg_value(
                tracking_block,
                tracker_block,
                "use_doppler_in_cost",
                default=True,
            )
        ),
        history_len=max(
            1,
            _to_int(
                _tracking_cfg_value(
                    tracking_block,
                    tracker_block,
                    "history_len",
                    default=12,
                ),
                12,
            ),
        ),
        debug_log=bool(
            _tracking_cfg_value(
                tracking_block,
                tracker_block,
                "debug_log",
                default=False,
            )
        ),
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


def apply_post_range_fft_filters(
    data: np.ndarray,
    filters_cfg: PostRangeFftFilterConfig,
    *,
    bg_state: BackgroundSubtractionState,
    fft_workers: int,
    apply_loop_average_after_background: bool,
) -> np.ndarray:
    out = apply_slow_time_filter(data,filters_cfg.slow_time,fft_workers=fft_workers,)
    out = subtract_selected_mean(out, filters_cfg.mean_after_range_fft)
    out = apply_background_subtraction(out, filters_cfg.background_subtraction, bg_state)
    if apply_loop_average_after_background:
        # Collapse the loop dimension while preserving the axis for the downstream pipeline.
        out = out.mean(axis=1, keepdims=True, dtype=np.complex64)
    return out


def _branch_needs_copy(filters_cfg: PostRangeFftFilterConfig) -> bool:
    if not filters_cfg.mean_after_range_fft.enabled or not filters_cfg.mean_after_range_fft.axes:
        return False
    return not filters_cfg.slow_time.enabled or filters_cfg.slow_time.mode == "none"


def _build_angle_u_axis(nfft_angle: int, spacing_lambda: float = 0.5) -> np.ndarray:
    nfft_angle = max(1, int(nfft_angle))
    if nfft_angle == 1:
        return np.asarray([0.0], dtype=np.float32)
    spacing = float(spacing_lambda)
    if not np.isfinite(spacing) or abs(spacing) <= 1e-8:
        spacing = 0.5
    u = (np.fft.fftshift(np.fft.fftfreq(nfft_angle, d=1.0)) / np.float32(spacing)).astype(
        np.float32,
        copy=False,
    )
    return u


def build_angle_steering_matrix(
    virtual_ant: int,
    nfft_angle: int,
    geometry: VirtualArrayGeometry | None = None,
) -> np.ndarray:
    if geometry is None:
        phase_centers_lambda = _default_virtual_array_phase_centers_lambda(int(virtual_ant))
    else:
        phase_centers_lambda = np.asarray(geometry.phase_centers_lambda, dtype=np.float32).reshape(-1)
    if phase_centers_lambda.size != int(virtual_ant):
        raise ValueError(
            f"phase_centers_lambda size={phase_centers_lambda.size} != virtual_ant={int(virtual_ant)}"
        )
    spacing_lambda = 0.5
    if geometry is not None and geometry.uniform_spacing_lambda is not None:
        spacing_lambda = float(geometry.uniform_spacing_lambda)
    u = _build_angle_u_axis(nfft_angle, spacing_lambda=spacing_lambda)
    phase = (-1j * np.float32(2.0 * np.pi)) * phase_centers_lambda[:, None] * u[None, :]
    steering = np.exp(phase).astype(np.complex64, copy=False)
    col_energy = (steering.real * steering.real + steering.imag * steering.imag).sum(axis=0, dtype=np.float32)
    col_norm = np.sqrt(np.maximum(col_energy, np.float32(1e-8))).astype(np.float32, copy=False)
    steering /= col_norm[np.newaxis, :].astype(np.complex64, copy=False)
    return steering


#[frame, loop, range_bin, virtual_ant] -> [range_bin, angle_bin] with angle_bin either FFT or Bartlett/MVDR output
def compute_angle_heatmap(
    virtual_array: np.ndarray, # dati effettivi di forma [frame, loop, range_bin, virtual_ant]
    *,
    angle_cfg: AngleProcessingConfig, 
    dsp_cfg: RealtimeDSPConfig,
    angle_steering: np.ndarray, # matrice di steering per Bartlett/MVDR, già normalizzata
    geometry: VirtualArrayGeometry | None = None,
) -> np.ndarray:
    # in [frame, loop, range_bin, angle_bin] -> [range_bin, angle_bin] con eventuale aggregazione su frame/loop a seconda di angle_cfg
    def _aggregate_angle_power(power: np.ndarray) -> np.ndarray:
        if power.ndim != 4:
            return np.asarray(power, dtype=np.float32)
        mode = str(getattr(angle_cfg, "aggregation", "frame_loop")).strip().lower()
        if mode == "frame":
            return power.mean(axis=0, dtype=np.float32)
        if mode == "loop":
            return power.mean(axis=1, dtype=np.float32)
        if mode == "none":
            frame_idx = min(max(0, int(getattr(angle_cfg, "frame_index", 0))), int(power.shape[0]) - 1)
            loop_idx = min(max(0, int(getattr(angle_cfg, "loop_index", 0))), int(power.shape[1]) - 1)
            return power[frame_idx, loop_idx, :, :].astype(np.float32, copy=False)
        return power.mean(axis=(0, 1), dtype=np.float32)

    # calculate angle heatmap with FFT; output is [range_bin, angle_bin]
    if angle_cfg.mode == "fft":
        angle_fft = fft.fft(
            virtual_array,
            n=dsp_cfg.nfft_angle,
            axis=3,
            workers=dsp_cfg.fft_workers,
            overwrite_x=True,
        )
        # power calculation
        re = angle_fft.real 
        im = angle_fft.imag
        power = (re * re + im * im).astype(np.float32, copy=False) 
        # shift zero angle to center bin
        power  = np.fft.fftshift(power , axes=-1)
        spacing_lambda = 0.5
        if geometry is not None and geometry.uniform_spacing_lambda is not None:
            spacing_lambda = float(geometry.uniform_spacing_lambda)
        valid_u = np.abs(
            _build_angle_u_axis(dsp_cfg.nfft_angle, spacing_lambda=spacing_lambda).astype(np.float64, copy=False)
        ) <= 1.0
        if np.any(~valid_u):
            power[..., ~valid_u] = np.float32(0.0)
        # aggregate according to angle_cfg and return
        heatmap = _aggregate_angle_power(power)
        return heatmap.astype(np.float32, copy=False)

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
        power = (yr * yr + yi * yi).astype(np.float32, copy=False)
        return power.mean(axis=1, dtype=np.float32).astype(np.float32, copy=False)

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


def build_angle_axis_deg(
    nfft_angle: int,
    geometry: VirtualArrayGeometry | None = None,
) -> np.ndarray:
    spacing_lambda = 0.5
    if geometry is not None and geometry.uniform_spacing_lambda is not None:
        spacing_lambda = float(geometry.uniform_spacing_lambda)
    u = _build_angle_u_axis(nfft_angle, spacing_lambda=spacing_lambda)
    angle_axis = np.full(u.shape, np.float32(np.nan), dtype=np.float32)
    valid = np.abs(u.astype(np.float64, copy=False)) <= 1.0
    if np.any(valid):
        angle_axis[valid] = np.rad2deg(
            np.arcsin(u[valid].astype(np.float64, copy=False))
        ).astype(np.float32, copy=False)
    return angle_axis


def _normalize_display_projection_mode(value: Any) -> DisplayProjectionMode:
    mode_raw = str(value or "polar_stretched").strip().lower()
    if mode_raw not in _VALID_DISPLAY_PROJECTION_MODES:
        return "polar_stretched"
    return mode_raw  # type: ignore[return-value]


def _normalize_display_projection_interp(value: Any) -> DisplayProjectionInterp:
    interp_raw = str(value or "nearest").strip().lower()
    if interp_raw not in _VALID_DISPLAY_PROJECTION_INTERPS:
        return "nearest"
    return interp_raw  # type: ignore[return-value]


def _build_display_axis(min_m: float, max_m: float, size: int) -> np.ndarray:
    size = int(size)
    if size <= 0:
        return np.empty((0,), dtype=np.float32)
    min_v = float(min_m)
    max_v = float(max_m)
    if size == 1:
        return np.asarray([(min_v + max_v) * 0.5], dtype=np.float32)
    step = (max_v - min_v) / float(size)
    return (min_v + (np.arange(size, dtype=np.float32) + np.float32(0.5)) * np.float32(step)).astype(
        np.float32,
        copy=False,
    )


def build_display_projection_lut(
    gui_h: int,
    gui_w: int,
    x_max_m: float,
    y_max_m: float,
    dr_m: float,
    angle_axis_deg: np.ndarray,
    projection_mode: str,
    projection_interp: str,
) -> dict[str, Any]:
    gui_h = max(0, int(gui_h))
    gui_w = max(0, int(gui_w))
    mode = _normalize_display_projection_mode(projection_mode)
    interp = _normalize_display_projection_interp(projection_interp)
    angle_axis = np.asarray(angle_axis_deg, dtype=np.float32).reshape(-1)

    lut: dict[str, Any] = {
        "projection_mode": mode,
        "projection_interp": interp,
        "output_shape": (gui_h, gui_w),
        "x_max_m": float(x_max_m),
        "y_max_m": float(y_max_m),
        "dr_m": float(dr_m),
        "angle_count": int(angle_axis.size),
        "x_axis_m": _build_display_axis(-float(x_max_m), +float(x_max_m), gui_w),
        "y_axis_m": _build_display_axis(0.0, float(y_max_m), gui_h),
    }
    if mode != "cartesian" or gui_h <= 0 or gui_w <= 0 or not np.isfinite(dr_m) or float(dr_m) <= 0.0:
        return lut

    finite_cols = np.flatnonzero(np.isfinite(angle_axis))
    if finite_cols.size <= 0:
        lut["valid_theta"] = np.zeros(gui_h * gui_w, dtype=bool)
        return lut

    angle_sorted_order = np.argsort(angle_axis[finite_cols], kind="mergesort")
    angle_src_idx = finite_cols[angle_sorted_order].astype(np.int32, copy=False)
    angle_sorted_deg = angle_axis[angle_src_idx].astype(np.float32, copy=False)

    x_axis_m = lut["x_axis_m"]
    y_axis_m = lut["y_axis_m"]
    x_grid_m, y_grid_m = np.meshgrid(x_axis_m, y_axis_m, indexing="xy")
    radius_idx = (np.hypot(x_grid_m, y_grid_m) / np.float32(dr_m)).astype(np.float32, copy=False).reshape(-1)
    theta_deg = np.rad2deg(np.arctan2(x_grid_m, y_grid_m)).astype(np.float32, copy=False).reshape(-1)

    if angle_sorted_deg.size == 1:
        angle_low = np.zeros(theta_deg.shape, dtype=np.int32)
        angle_high = np.zeros(theta_deg.shape, dtype=np.int32)
        angle_weight = np.zeros(theta_deg.shape, dtype=np.float32)
        valid_theta = np.isclose(theta_deg, angle_sorted_deg[0], atol=1e-4)
    else:
        upper = np.searchsorted(angle_sorted_deg, theta_deg, side="right")
        angle_low = np.clip(upper - 1, 0, angle_sorted_deg.size - 1).astype(np.int32, copy=False)
        angle_high = np.clip(upper, 0, angle_sorted_deg.size - 1).astype(np.int32, copy=False)
        denom = angle_sorted_deg[angle_high] - angle_sorted_deg[angle_low]
        angle_weight = np.zeros(theta_deg.shape, dtype=np.float32)
        good = np.abs(denom) > np.float32(1e-6)
        angle_weight[good] = (
            (theta_deg[good] - angle_sorted_deg[angle_low[good]]) / denom[good]
        ).astype(np.float32, copy=False)
        np.clip(angle_weight, 0.0, 1.0, out=angle_weight)
        valid_theta = (theta_deg >= angle_sorted_deg[0]) & (theta_deg <= angle_sorted_deg[-1])

    lut.update(
        {
            "valid_theta": valid_theta.astype(bool, copy=False),
            "range_pos": radius_idx,
            "theta_deg": theta_deg,
            "angle_sorted_deg": angle_sorted_deg,
            "angle_low_src": angle_src_idx[angle_low],
            "angle_high_src": angle_src_idx[angle_high],
            "angle_weight": angle_weight.astype(np.float32, copy=False),
        }
    )

    if interp == "nearest":
        lut["range_nearest"] = np.rint(radius_idx).astype(np.int32, copy=False)
        nearest_sel = np.where(angle_weight <= np.float32(0.5), angle_low, angle_high)
        lut["angle_nearest_src"] = angle_src_idx[nearest_sel].astype(np.int32, copy=False)
    else:
        range_low = np.floor(radius_idx).astype(np.int32, copy=False)
        lut["range_low"] = range_low
        lut["range_high"] = (range_low + 1).astype(np.int32, copy=False)
        lut["range_weight"] = (radius_idx - range_low.astype(np.float32, copy=False)).astype(
            np.float32,
            copy=False,
        )
    return lut


def project_heatmap_for_display(
    heatmap_lin: np.ndarray, # matrice di potenza lineare [range_bin, angle_bin]
    angle_axis_deg: np.ndarray, # asse degli angoli in gradi corrispondente alla dimensione angle_bin di heatmap_lin
    dr_m: float, # risoluzione in metri per bin di range
    gui_h: int, # altezza in pixel dell'area di visualizzazione
    gui_w: int, # larghezza in pixel dell'area di visualizzazione
    y_max_m: float, # distanza massima in metri da visualizzare sull'asse y (range)
    x_max_m: float, # distanza massima in metri da visualizzare sull'asse x (angle); per la modalità "polar_stretched" questo è anche il raggio massimo visualizzato
    projection_mode: str = "polar_stretched", # modalità di proiezione per visualizzare il heatmap; "polar_stretched" per una proiezione polare con stretching lineare dell'asse x, "cartesian" per una proiezione cartesiana senza stretching (ma comunque con interpolazione se projection_interp != "nearest")
    projection_interp: str = "nearest", # metodo di interpolazione da usare se la modalità di proiezione richiede una mappatura non 1:1 tra pixel e bin di heatmap; "nearest" per nearest neighbor, "bilinear" per interpolazione bilineare
    out: np.ndarray | None = None, # array opzionale di output preallocato con shape (gui_h, gui_w) e dtype float32; se fornito e compatibile, verrà usato per evitare l'allocazione interna di un nuovo array
    fill_value: float = 0.0, # valore di potenza lineare da usare per i pixel dell'output che non corrispondono a nessun bin valido del heatmap; di default 0.0 per visualizzare come nero, ma può essere impostato a un valore più alto per evidenziare le aree fuori range/angolo
    precomputed_lut: dict[str, Any] | None = None, # lookup table opzionale precomputata per la proiezione, ottenibile con build_display_projection_lut; se fornita e compatibile, verrà usata per evitare il calcolo interno della LUT e velocizzare la proiezione soprattutto in caso di proiezioni complesse o output di grandi dimensioni
) -> np.ndarray:
    # round to int and sanitize inputs
    gui_h = max(0, int(gui_h)) 
    gui_w = max(0, int(gui_w))
    # la modalità di proiezione e il metodo di interpolazione vengono normalizzati e validati; 
    # se non validi, si usano i valori di default "polar_stretched" e "nearest"
    mode = _normalize_display_projection_mode(projection_mode)
    interp = _normalize_display_projection_interp(projection_interp)

    if out is not None and out.shape == (gui_h, gui_w) and out.dtype == np.float32:
        dst = out #heatmap out [gui_h, gui_w] già preallocato e compatibile
    else:
        dst = np.empty((gui_h, gui_w), dtype=np.float32) #heatmap out [gui_h, gui_w] da allocare
    dst.fill(np.float32(fill_value)) #fill con fill_value per i pixel non validi

    #check input heatmap e angle_axis e sanity check per dimensioni e valori; se non validi, restituisco dst già fill
    src = np.asarray(heatmap_lin, dtype=np.float32) 
    if src.ndim != 2 or gui_h <= 0 or gui_w <= 0:
        return dst

    # Source heatmap size: rows=range bins, cols=angle bins.
    src_rows = int(src.shape[0]) 
    src_cols = int(src.shape[1])
    if src_rows <= 0 or src_cols <= 0:
        return dst

    # Non-cartesian mode copies the polar heatmap directly; cartesian mode needs a valid angle axis.
    if mode != "cartesian":
        # Copy only the overlapping source area when display mode keeps the polar grid unchanged.
        copy_rows = min(gui_h, src_rows)
        copy_cols = min(gui_w, src_cols)
        if copy_rows > 0 and copy_cols > 0:
            dst[:copy_rows, :copy_cols] = src[:copy_rows, :copy_cols]
        return dst

    # Use only the angle columns that have a matching valid angle-axis entry.
    angle_axis = np.asarray(angle_axis_deg, dtype=np.float32).reshape(-1)
    eff_cols = min(src_cols, int(angle_axis.size))
    if eff_cols <= 0:
        return dst

    # Rebuild the projection LUT whenever the cached one does not match the current display geometry.
    lut = precomputed_lut
    if (
        lut is None
        or tuple(lut.get("output_shape", ())) != (gui_h, gui_w)
        or str(lut.get("projection_mode", "")) != mode
        or str(lut.get("projection_interp", "")) != interp
        or int(lut.get("angle_count", -1)) != eff_cols
        or not np.isclose(float(lut.get("x_max_m", np.nan)), float(x_max_m))
        or not np.isclose(float(lut.get("y_max_m", np.nan)), float(y_max_m))
        or not np.isclose(float(lut.get("dr_m", np.nan)), float(dr_m))
    ):
        lut = build_display_projection_lut(
            gui_h=gui_h,
            gui_w=gui_w,
            x_max_m=x_max_m,
            y_max_m=y_max_m,
            dr_m=dr_m,
            angle_axis_deg=angle_axis[:eff_cols],
            projection_mode=mode,
            projection_interp=interp,
        )

    #Abort if the LUT provides no valid angle coverage for the current output grid.
    valid_theta = np.asarray(lut.get("valid_theta", np.zeros(gui_h * gui_w, dtype=bool)), dtype=bool)
    if valid_theta.size != gui_h * gui_w or not np.any(valid_theta):
        return dst

    dst_flat = dst.reshape(-1)
    src_view = src[:, :eff_cols]

    if interp == "nearest":
        range_nearest = np.asarray(lut.get("range_nearest", ()), dtype=np.int32)
        angle_nearest_src = np.asarray(lut.get("angle_nearest_src", ()), dtype=np.int32)
        if range_nearest.size != dst_flat.size or angle_nearest_src.size != dst_flat.size:
            return dst
        valid = (
            valid_theta
            & (range_nearest >= 0)
            & (range_nearest < src_rows)
            & (angle_nearest_src >= 0)
            & (angle_nearest_src < eff_cols)
        )
        if np.any(valid):
            dst_flat[valid] = src_view[range_nearest[valid], angle_nearest_src[valid]]
        return dst

    range_pos = np.asarray(lut.get("range_pos", ()), dtype=np.float32)
    range_low = np.asarray(lut.get("range_low", ()), dtype=np.int32)
    range_high = np.asarray(lut.get("range_high", ()), dtype=np.int32)
    range_weight = np.asarray(lut.get("range_weight", ()), dtype=np.float32)
    angle_low_src = np.asarray(lut.get("angle_low_src", ()), dtype=np.int32)
    angle_high_src = np.asarray(lut.get("angle_high_src", ()), dtype=np.int32)
    angle_weight = np.asarray(lut.get("angle_weight", ()), dtype=np.float32)
    if (
        range_pos.size != dst_flat.size
        or range_low.size != dst_flat.size
        or range_high.size != dst_flat.size
        or range_weight.size != dst_flat.size
        or angle_low_src.size != dst_flat.size
        or angle_high_src.size != dst_flat.size
        or angle_weight.size != dst_flat.size
    ):
        return dst

    max_range_idx = float(src_rows - 1)
    valid = (
        valid_theta
        & (range_pos >= 0.0)
        & (range_pos <= max_range_idx)
        & (angle_low_src >= 0)
        & (angle_low_src < eff_cols)
        & (angle_high_src >= 0)
        & (angle_high_src < eff_cols)
    )
    if not np.any(valid):
        return dst

    r0 = np.clip(range_low[valid], 0, max(0, src_rows - 1))
    r1 = np.clip(range_high[valid], 0, max(0, src_rows - 1))
    c0 = angle_low_src[valid]
    c1 = angle_high_src[valid]
    wr = range_weight[valid]
    wa = angle_weight[valid]

    v00 = src_view[r0, c0]
    v01 = src_view[r0, c1]
    v10 = src_view[r1, c0]
    v11 = src_view[r1, c1]
    top = v00 + ((v01 - v00) * wa)
    bottom = v10 + ((v11 - v10) * wa)
    dst_flat[valid] = top + ((bottom - top) * wr)
    return dst


def _resolve_chirp_period_s(cfg: dict[str, Any]) -> float | None:
    radar = cfg.get("radar", {}) or {}
    capture = cfg.get("capture", {}) or {}
    for key in (
        "chirp_period_s",
        "chirp_repetition_s",
        "pri_s",
        "t_chirp_s",
        "tc_s",
        "chirp_time_s",
    ):
        val = _to_float(radar.get(key, None), float("nan"))
        if np.isfinite(val) and val > 0.0:
            return float(val)
    for key in ("chirp_period_s", "chirp_repetition_s", "pri_s"):
        val = _to_float(capture.get(key, None), float("nan"))
        if np.isfinite(val) and val > 0.0:
            return float(val)
    return None


def resolve_nominal_frame_period_s(
    cfg: dict[str, Any],
    dsp_cfg: RealtimeDSPConfig,
    tracking_cfg: TrackingConfig | None = None,
) -> float | None:
    tracking_dt_s = None if tracking_cfg is None else getattr(tracking_cfg, "dt_s", None)
    if tracking_dt_s is not None:
        dt_s = float(tracking_dt_s)
        if np.isfinite(dt_s) and dt_s > 0.0:
            return float(dt_s)
    chirp_period_s = _resolve_chirp_period_s(cfg)
    if chirp_period_s is None or chirp_period_s <= 0.0:
        return None
    frame_period_s = float(chirp_period_s) * float(max(1, int(dsp_cfg.chirps)))
    if not np.isfinite(frame_period_s) or frame_period_s <= 0.0:
        return None
    return float(frame_period_s)


def _build_doppler_bin_cycles_axis(
    n_doppler: int,
    *,
    doppler_fft_shift: bool,
) -> np.ndarray:
    n_doppler = max(1, int(n_doppler))
    doppler_cycles = np.fft.fftfreq(n_doppler, d=1.0).astype(np.float32, copy=False)
    if doppler_fft_shift:
        doppler_cycles = np.fft.fftshift(doppler_cycles)
    return doppler_cycles.astype(np.float32, copy=False)


def build_doppler_axis_mps(
    cfg: dict[str, Any],
    dsp_cfg: RealtimeDSPConfig,
    n_doppler: int,
    *,
    doppler_fft_shift: bool,
) -> np.ndarray | None:
    radar = cfg.get("radar", {}) or {}
    fc_hz = _to_float(radar.get("fc", None), float("nan"))
    if not np.isfinite(fc_hz) or fc_hz <= 0.0:
        print("[DSP WARN] Doppler axis disabled: invalid radar.fc in config (doppler_mps=None).")
        return None
    chirp_period_s = _resolve_chirp_period_s(cfg)
    if chirp_period_s is None or chirp_period_s <= 0.0:
        print("[DSP WARN] Doppler axis disabled: missing/invalid chirp_period_s or pri_s (doppler_mps=None).")
        return None
    tx_count = max(1, int(dsp_cfg.tx))
    if tx_count != 2:
        print(
            f"[DSP WARN] Doppler axis generalized to uniform TDM with tx={tx_count}; "
            "verify against the real chirp schedule if TX delays are non-uniform."
        )
    effective_pri_s = float(chirp_period_s) * float(tx_count)
    wavelength_m = float(dsp_cfg.c) / float(fc_hz)
    doppler_cycles = _build_doppler_bin_cycles_axis(n_doppler, doppler_fft_shift=doppler_fft_shift)
    fd_hz = (doppler_cycles / np.float32(effective_pri_s)).astype(np.float32, copy=False)
    return (fd_hz * np.float32(wavelength_m * 0.5)).astype(np.float32, copy=False)


def build_tdm_mimo_doppler_compensation_table(
    n_doppler: int,
    tx_count: int,
    *,
    doppler_fft_shift: bool,
) -> np.ndarray:
    n_doppler = max(0, int(n_doppler))
    tx_count = max(0, int(tx_count))
    if n_doppler <= 0 or tx_count <= 0:
        return np.empty((0, 0), dtype=np.complex64)
    if tx_count == 1:
        return np.ones((n_doppler, 1), dtype=np.complex64)

    doppler_cycles = _build_doppler_bin_cycles_axis(n_doppler, doppler_fft_shift=doppler_fft_shift)
    tx_delay_in_loops = (np.arange(tx_count, dtype=np.float32) / np.float32(tx_count)).astype(np.float32, copy=False)
    # numpy/scipy FFT sign convention: a target at Doppler bin k carries an extra phase
    # exp(+j 2*pi*f_d*t_tx) on TX acquired t_tx later. Undo it before virtual-array flattening.
    # TODO(hardware): validate sign against real board chirp ordering / I-Q convention; if needed,
    # conjugate this table once the hardware sign is confirmed.
    phase = np.exp(
        (-1j * 2.0 * np.pi)
        * doppler_cycles[:, None].astype(np.float64, copy=False)
        * tx_delay_in_loops[None, :].astype(np.float64, copy=False)
    ).astype(np.complex64, copy=False)
    phase[:, 0] = np.complex64(1.0 + 0.0j)
    return phase


def apply_tdm_mimo_doppler_compensation(
    snapshot_tx_rx: np.ndarray,
    *,
    doppler_bin: int,
    compensation_table: np.ndarray | None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compensate TDM-MIMO per-TX Doppler phase on snapshots with trailing axes [tx, rx].

    Leading axes are preserved (typically [frame, tx, rx]); output has the same axes/order
    as the input and is ready for virtual-array flattening in the configured antenna order.
    """
    snapshot = np.asarray(snapshot_tx_rx, dtype=np.complex64)
    if out is not None and out.shape == snapshot.shape and out.dtype == np.complex64:
        compensated = out
        np.copyto(compensated, snapshot, casting="unsafe")
    else:
        compensated = np.array(snapshot, dtype=np.complex64, copy=True)
    if compensated.ndim < 2:
        return compensated
    if compensation_table is None or compensation_table.size <= 0:
        return compensated
    dbin = int(doppler_bin)
    if dbin < 0 or dbin >= int(compensation_table.shape[0]):
        return compensated

    phase_row = np.asarray(compensation_table[dbin], dtype=np.complex64).reshape(-1)
    if phase_row.size != int(compensated.shape[-2]):
        return compensated
    phase_shape = (1,) * max(0, compensated.ndim - 2) + (int(phase_row.size), 1)
    np.multiply(compensated, phase_row.reshape(phase_shape), out=compensated)
    return compensated


def _build_virtual_array_from_range_fft(
    range_fft: np.ndarray,
    *,
    max_bin: int,
    dsp_cfg: RealtimeDSPConfig,
    geometry: VirtualArrayGeometry,
    work_buf: np.ndarray | None = None,
    flat_work_buf: np.ndarray | None = None,
) -> np.ndarray:
    trimmed = range_fft[:, :, :, :max_bin, :]
    va_src = trimmed.transpose(0, 1, 3, 2, 4)
    if (
        work_buf is not None
        and work_buf.shape == va_src.shape
        and work_buf.dtype == np.complex64
        and work_buf.flags.c_contiguous
    ):
        np.copyto(work_buf, va_src, casting="unsafe")
        va = work_buf
    else:
        va = np.ascontiguousarray(va_src)
    va_flat = va.reshape(
        int(range_fft.shape[0]),
        int(range_fft.shape[1]),
        int(max_bin),
        int(dsp_cfg.virtual_ant),
    )
    if geometry.identity_order:
        return va_flat
    if (
        flat_work_buf is not None
        and flat_work_buf.shape == va_flat.shape
        and flat_work_buf.dtype == np.complex64
        and flat_work_buf.flags.c_contiguous
    ):
        np.take(va_flat, geometry.order_flat, axis=-1, out=flat_work_buf)
        return flat_work_buf
    return np.take(va_flat, geometry.order_flat, axis=-1).astype(np.complex64, copy=False)


def _build_complex_calibration_vector(calibration_cfg: CalibrationConfig, virtual_ant: int) -> np.ndarray:
    size = max(1, int(virtual_ant))
    real = np.asarray(calibration_cfg.vector_real[:size], dtype=np.float32)
    imag = np.asarray(calibration_cfg.vector_imag[:size], dtype=np.float32)
    if real.size < size:
        real = np.pad(real, (0, size - real.size), constant_values=np.float32(1.0))
    if imag.size < size:
        imag = np.pad(imag, (0, size - imag.size), constant_values=np.float32(0.0))
    return (real + (1j * imag)).astype(np.complex64, copy=False)


def _apply_calibration_vector(virtual_array: np.ndarray, cal_vector: np.ndarray | None) -> np.ndarray:
    if cal_vector is None or virtual_array.size <= 0 or virtual_array.ndim <= 0:
        return virtual_array
    cal = np.asarray(cal_vector, dtype=np.complex64).reshape(-1)
    if cal.size != int(virtual_array.shape[-1]):
        return virtual_array
    cal_shape = (1,) * max(0, virtual_array.ndim - 1) + (int(cal.size),)
    np.multiply(virtual_array, cal.reshape(cal_shape), out=virtual_array)
    return virtual_array


def _estimate_boresight_calibration_vector(virtual_array: np.ndarray) -> tuple[np.ndarray | None, int | None]:
    if virtual_array.ndim != 4 or virtual_array.shape[2] <= 0 or virtual_array.shape[3] <= 0:
        return None, None
    avg_snapshot = virtual_array.mean(axis=(0, 1), dtype=np.complex64)
    if avg_snapshot.ndim != 2 or avg_snapshot.shape[0] <= 0 or avg_snapshot.shape[1] <= 0:
        return None, None
    re = avg_snapshot.real
    im = avg_snapshot.imag
    energy = (re * re + im * im).sum(axis=-1, dtype=np.float32)
    if energy.size <= 0 or not np.any(np.isfinite(energy)):
        return None, None
    target_bin = int(np.nanargmax(energy))
    antenna_row = np.asarray(avg_snapshot[target_bin, :], dtype=np.complex64).reshape(-1)
    if antenna_row.size <= 0:
        return None, target_bin
    ref = np.complex64(antenna_row[0])
    if abs(ref) <= 1e-12:
        return None, target_bin
    relative_phases = np.angle(antenna_row * np.conj(ref)).astype(np.float32, copy=False)
    cal_vector = np.exp((-1j) * relative_phases.astype(np.float64, copy=False)).astype(np.complex64, copy=False)
    cal_vector[0] = np.complex64(1.0 + 0.0j)
    return cal_vector, target_bin


def _format_calibration_vector_for_log(cal_vector: np.ndarray) -> tuple[str, str]:
    vec = np.asarray(cal_vector, dtype=np.complex64).reshape(-1)
    real_str = ", ".join(f"{float(v):.6f}" for v in vec.real)
    imag_str = ", ".join(f"{float(v):.6f}" for v in vec.imag)
    return real_str, imag_str


def _power_to_db(power_lin: np.ndarray) -> np.ndarray:
    out = np.array(power_lin, dtype=np.float32, copy=True)
    np.add(out, np.float32(1e-12), out=out)
    np.log10(out, out=out)
    out *= np.float32(10.0)
    return out


def _format_debug_top_peaks_range_angle(
    heatmap_lin: np.ndarray,
    *,
    angle_axis_deg: np.ndarray,
    range_bin_m: float,
    top_k: int,
    range_min_m: float | None = None,
    range_max_m: float | None = None,
    angle_min_deg: float | None = None,
    angle_max_deg: float | None = None,
) -> str:
    src = np.asarray(heatmap_lin, dtype=np.float32)
    if src.ndim != 2 or src.size <= 0:
        return f"[DSP DEBUG] range_angle_top_peaks: shape={src.shape} (no 2D data)"
    row_idx = np.arange(int(src.shape[0]), dtype=np.float32) * np.float32(range_bin_m)
    angle_axis = np.asarray(angle_axis_deg, dtype=np.float32).reshape(-1)
    valid_rows = np.ones(int(src.shape[0]), dtype=bool)
    valid_cols = np.ones(int(src.shape[1]), dtype=bool)
    if range_min_m is not None and np.isfinite(float(range_min_m)):
        valid_rows &= row_idx >= np.float32(float(range_min_m))
    if range_max_m is not None and np.isfinite(float(range_max_m)):
        valid_rows &= row_idx <= np.float32(float(range_max_m))
    eff_cols = min(int(src.shape[1]), int(angle_axis.size))
    if eff_cols < int(src.shape[1]):
        valid_cols[eff_cols:] = False
    if angle_min_deg is not None and np.isfinite(float(angle_min_deg)):
        valid_cols[:eff_cols] &= angle_axis[:eff_cols] >= np.float32(float(angle_min_deg))
    if angle_max_deg is not None and np.isfinite(float(angle_max_deg)):
        valid_cols[:eff_cols] &= angle_axis[:eff_cols] <= np.float32(float(angle_max_deg))
    if not np.any(valid_rows):
        return (
            f"[DSP DEBUG] range_angle_top_peaks: shape={src.shape}, "
            f"empty range filter [{range_min_m}, {range_max_m}] m"
        )
    if not np.any(valid_cols):
        return (
            f"[DSP DEBUG] range_angle_top_peaks: shape={src.shape}, "
            f"empty angle filter [{angle_min_deg}, {angle_max_deg}] deg"
        )
    filtered = np.where(valid_rows[:, None] & valid_cols[None, :], src, np.float32(-np.inf))
    finite_mask = np.isfinite(filtered)
    if not np.any(finite_mask):
        return f"[DSP DEBUG] range_angle_top_peaks: shape={src.shape}, no finite values in active filters"
    k = max(1, min(int(top_k), int(np.count_nonzero(finite_mask))))
    flat = filtered.reshape(-1)
    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    lines = [
        f"[DSP DEBUG] range_angle_top_peaks: shape={src.shape}, top_k={k}, "
        f"range_filter=[{range_min_m}, {range_max_m}] m, "
        f"angle_filter=[{angle_min_deg}, {angle_max_deg}] deg"
    ]
    for rank, flat_idx in enumerate(idx, start=1):
        rb, ab = np.unravel_index(int(flat_idx), src.shape)
        angle_deg = float(angle_axis[ab]) if 0 <= ab < int(angle_axis.size) else float("nan")
        range_m = float(rb) * float(range_bin_m)
        lines.append(
            f"  {rank:02d}. rb={rb:03d} ({range_m:6.3f} m)  ab={ab:03d} ({angle_deg:7.2f} deg)  power={float(src[rb, ab]):.3e}"
        )
    return "\n".join(lines)


def _format_debug_top_peaks_xy(
    view_db: np.ndarray,
    *,
    x_max_m: float,
    y_max_m: float,
    top_k: int,
    range_min_m: float | None = None,
    range_max_m: float | None = None,
    angle_min_deg: float | None = None,
    angle_max_deg: float | None = None,
) -> str:
    src = np.asarray(view_db, dtype=np.float32)
    if src.ndim != 2 or src.size <= 0:
        return f"[DSP DEBUG] xy_top_peaks: shape={src.shape} (no 2D data)"
    y_axis = _build_display_axis(0.0, float(y_max_m), int(src.shape[0]))
    x_axis = _build_display_axis(-float(x_max_m), float(x_max_m), int(src.shape[1]))
    valid_rows = np.ones(int(src.shape[0]), dtype=bool)
    x_grid_m, y_grid_m = np.meshgrid(x_axis, y_axis, indexing="xy")
    theta_deg = np.rad2deg(np.arctan2(x_grid_m, np.maximum(y_grid_m, np.float32(1e-6)))).astype(np.float32, copy=False)
    valid_angles = np.ones(src.shape, dtype=bool)
    if range_min_m is not None and np.isfinite(float(range_min_m)):
        valid_rows &= y_axis >= np.float32(float(range_min_m))
    if range_max_m is not None and np.isfinite(float(range_max_m)):
        valid_rows &= y_axis <= np.float32(float(range_max_m))
    if angle_min_deg is not None and np.isfinite(float(angle_min_deg)):
        valid_angles &= theta_deg >= np.float32(float(angle_min_deg))
    if angle_max_deg is not None and np.isfinite(float(angle_max_deg)):
        valid_angles &= theta_deg <= np.float32(float(angle_max_deg))
    if not np.any(valid_rows):
        return (
            f"[DSP DEBUG] xy_top_peaks: shape={src.shape}, "
            f"empty range filter [{range_min_m}, {range_max_m}] m"
        )
    if not np.any(valid_angles):
        return (
            f"[DSP DEBUG] xy_top_peaks: shape={src.shape}, "
            f"empty angle filter [{angle_min_deg}, {angle_max_deg}] deg"
        )
    filtered = np.where(valid_rows[:, None] & valid_angles, src, np.float32(-np.inf))
    finite_mask = np.isfinite(filtered)
    if not np.any(finite_mask):
        return f"[DSP DEBUG] xy_top_peaks: shape={src.shape}, no finite values in active filters"
    k = max(1, min(int(top_k), int(np.count_nonzero(finite_mask))))
    flat = filtered.reshape(-1)
    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    lines = [
        f"[DSP DEBUG] xy_top_peaks: shape={src.shape}, top_k={k}, "
        f"range_filter=[{range_min_m}, {range_max_m}] m, "
        f"angle_filter=[{angle_min_deg}, {angle_max_deg}] deg"
    ]
    for rank, flat_idx in enumerate(idx, start=1):
        yb, xb = np.unravel_index(int(flat_idx), src.shape)
        x_m = float(x_axis[xb]) if 0 <= xb < int(x_axis.size) else float("nan")
        y_m = float(y_axis[yb]) if 0 <= yb < int(y_axis.size) else float("nan")
        angle_deg = float(theta_deg[yb, xb])
        lines.append(
            f"  {rank:02d}. yb={yb:03d} ({y_m:6.3f} m)  xb={xb:03d} ({x_m:7.3f} m)  angle={angle_deg:7.2f} deg  db={float(src[yb, xb]):7.2f}"
        )
    return "\n".join(lines)


def _resolve_detection_threshold_db(
    power_db: np.ndarray,
    *,
    threshold_mode: DetectionThresholdMode,
    threshold_db: float,
    min_power_db: float,
) -> float:
    if power_db.size <= 0:
        return float(min_power_db)
    if threshold_mode == "relative":
        ref_db = float(np.max(power_db))
        return max(float(min_power_db), ref_db + float(threshold_db))
    return max(float(min_power_db), float(threshold_db))


def compute_detection_power_db_map(power_lin: np.ndarray) -> np.ndarray:
    return _power_to_db(power_lin)


def compute_detection_threshold_db(
    power_db: np.ndarray,
    *,
    threshold_mode: DetectionThresholdMode,
    threshold_db: float,
    min_power_db: float,
) -> float:
    # TODO(CFAR): replace this global threshold with CA-CFAR / OS-CFAR while keeping
    # the contract "power map in -> scalar/decision threshold out -> shared peak extractor".
    return _resolve_detection_threshold_db(
        power_db,
        threshold_mode=threshold_mode,
        threshold_db=threshold_db,
        min_power_db=min_power_db,
    )


def extract_detection_peaks_2d(
    power_db: np.ndarray,
    *,
    threshold_db: float,
    win_row: int,
    win_col: int,
    max_peaks: int,
) -> np.ndarray:
    return _select_localmax_2d(
        power_db,
        threshold_db=threshold_db,
        win_row=win_row,
        win_col=win_col,
        max_peaks=max_peaks,
    )


def _select_localmax_2d(
    power_db: np.ndarray,
    *,
    threshold_db: float,
    win_row: int,
    win_col: int,
    max_peaks: int,
) -> np.ndarray:
    if power_db.size <= 0 or max_peaks <= 0:
        return np.empty((0, 2), dtype=np.int32)
    candidates_mask = power_db >= np.float32(threshold_db)
    if not np.any(candidates_mask):
        return np.empty((0, 2), dtype=np.int32)
    candidates = np.argwhere(candidates_mask)
    strengths = power_db[candidates_mask]
    order = np.argsort(strengths)[::-1]
    suppressed = np.zeros(power_db.shape, dtype=bool)
    selected: list[tuple[int, int]] = []
    n_rows, n_cols = int(power_db.shape[0]), int(power_db.shape[1])
    row_pad = int(max(0, win_row))
    col_pad = int(max(0, win_col))
    for idx in order:
        r = int(candidates[idx, 0])
        c = int(candidates[idx, 1])
        if suppressed[r, c]:
            continue
        selected.append((r, c))
        r0 = max(0, r - row_pad)
        r1 = min(n_rows, r + row_pad + 1)
        c0 = max(0, c - col_pad)
        c1 = min(n_cols, c + col_pad + 1)
        suppressed[r0:r1, c0:c1] = True
        if len(selected) >= max_peaks:
            break
    if not selected:
        return np.empty((0, 2), dtype=np.int32)
    return np.asarray(selected, dtype=np.int32)



# Main detection function for static targets using angle heatmap peak picking.
def detect_static_targets(
    range_fft: np.ndarray,
    *,
    static_cfg: DetectionConfigStatic,
    angle_cfg: AngleProcessingConfig,
    dsp_cfg: RealtimeDSPConfig,
    virtual_array_geometry: VirtualArrayGeometry,
    w_angle: np.ndarray,
    angle_steering: np.ndarray,
    angle_axis_deg: np.ndarray,
    range_bin_m: float,
    max_bin: int,
    apply_angle_window: bool,
    calibration_enabled: bool = False,
    cal_vector: np.ndarray | None = None,
    virtual_array_work_buf: np.ndarray | None = None,
    virtual_array_flat_work_buf: np.ndarray | None = None,
) -> tuple[list[Detection], np.ndarray]:
    if not static_cfg.enabled:
        return [], np.empty((0, 0), dtype=np.float32)
    if range_fft.ndim != 5 or max_bin <= 0:
        return [], np.empty((0, 0), dtype=np.float32)
    if int(range_fft.shape[0]) <= 0 or int(range_fft.shape[1]) <= 0:
        return [], np.empty((0, 0), dtype=np.float32)

    virtual_array = _build_virtual_array_from_range_fft(
        range_fft,
        max_bin=max_bin,
        dsp_cfg=dsp_cfg,
        geometry=virtual_array_geometry,
        work_buf=virtual_array_work_buf,
        flat_work_buf=virtual_array_flat_work_buf,
    )
    if calibration_enabled:
        _apply_calibration_vector(virtual_array, cal_vector)
    if apply_angle_window:
        virtual_array *= w_angle
    static_heatmap = compute_angle_heatmap(
        virtual_array,
        angle_cfg=angle_cfg,
        dsp_cfg=dsp_cfg,
        angle_steering=angle_steering,
        geometry=virtual_array_geometry,
    )
    static_db = compute_detection_power_db_map(static_heatmap)
    threshold_db = compute_detection_threshold_db(
        static_db,
        threshold_mode=static_cfg.threshold_mode,
        threshold_db=static_cfg.threshold_db,
        min_power_db=static_cfg.min_power_db,
    )
    peak_idx = extract_detection_peaks_2d(
        static_db,
        threshold_db=threshold_db,
        win_row=static_cfg.localmax_range_bins,
        win_col=static_cfg.localmax_angle_bins,
        max_peaks=static_cfg.max_detections,
    )
    detections: list[Detection] = []
    for r_bin, a_bin in peak_idx:
        rb = int(r_bin)
        ab = int(a_bin)
        range_m = float(rb) * float(range_bin_m)
        angle_deg = float(angle_axis_deg[ab]) if 0 <= ab < int(angle_axis_deg.size) else 0.0
        angle_rad = np.deg2rad(np.float32(angle_deg))
        x_m = float(range_m * float(np.sin(angle_rad)))
        y_m = float(range_m * float(np.cos(angle_rad)))
        p_lin = float(static_heatmap[rb, ab])
        p_db = float(static_db[rb, ab])
        detections.append(
            Detection(
                range_bin=rb,
                angle_bin=ab,
                doppler_bin=None,
                range_m=range_m,
                angle_deg=angle_deg,
                doppler_mps=None,
                x_m=x_m,
                y_m=y_m,
                power_lin=p_lin,
                power_db=p_db,
                source="static",
            )
        )
    return detections, static_heatmap


def compute_range_doppler(
    range_fft: np.ndarray,
    *,
    max_bin: int,
    dsp_cfg: RealtimeDSPConfig,
    moving_cfg: DetectionConfigMoving,
    w_doppler: np.ndarray,
    apply_doppler_window: bool,
    doppler_work_buf: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if range_fft.ndim != 5 or max_bin <= 0:
        return np.empty((0, 0, 0, 0, 0), dtype=np.complex64), np.empty((0, 0), dtype=np.float32)
    n_loops = int(range_fft.shape[1])
    if n_loops <= 0:
        return np.empty((0, 0, 0, 0, 0), dtype=np.complex64), np.empty((0, 0), dtype=np.float32)

    trimmed = range_fft[:, :, :, :max_bin, :]
    trimmed_work = doppler_work_buf
    if (
        trimmed_work is not None
        and (
            trimmed_work.shape != trimmed.shape
            or trimmed_work.dtype != np.complex64
            or (not trimmed_work.flags.c_contiguous)
        )
    ):
        trimmed_work = None
    if apply_doppler_window:
        if trimmed_work is not None:
            np.multiply(trimmed, w_doppler, out=trimmed_work)
            doppler_in = trimmed_work
        else:
            doppler_in = np.ascontiguousarray(trimmed * w_doppler)
    else:
        if trimmed.flags.c_contiguous:
            doppler_in = trimmed
        elif trimmed_work is not None:
            np.copyto(trimmed_work, trimmed, casting="unsafe")
            doppler_in = trimmed_work
        else:
            doppler_in = np.ascontiguousarray(trimmed)
    doppler_cube = fft.fft(
        doppler_in,
        n=n_loops,
        axis=1,
        workers=dsp_cfg.fft_workers,
        overwrite_x=False,
    )
    if moving_cfg.doppler_fft_shift:
        doppler_cube = np.fft.fftshift(doppler_cube, axes=1)
    doppler_cube = doppler_cube.astype(np.complex64, copy=False)

    re = doppler_cube.real
    im = doppler_cube.imag
    rd_pow = (re * re + im * im).mean(axis=(0, 2, 4), dtype=np.float32)
    range_doppler_map = rd_pow.transpose(1, 0).astype(np.float32, copy=False)

    excl = int(max(0, moving_cfg.zero_doppler_exclusion_bins))
    if excl > 0 and range_doppler_map.shape[1] > 0:
        zero_idx = int(range_doppler_map.shape[1] // 2) if moving_cfg.doppler_fft_shift else 0
        d0 = max(0, zero_idx - excl)
        d1 = min(int(range_doppler_map.shape[1]), zero_idx + excl + 1)
        range_doppler_map[:, d0:d1] = 0.0
    return doppler_cube, range_doppler_map


def detect_moving_targets(
    range_doppler_map: np.ndarray,
    *,
    moving_cfg: DetectionConfigMoving,
    range_bin_m: float,
    doppler_axis_mps: np.ndarray | None,
) -> list[Detection]:
    if not moving_cfg.enabled or range_doppler_map.size <= 0:
        return []
    moving_db = compute_detection_power_db_map(range_doppler_map)
    threshold_db = compute_detection_threshold_db(
        moving_db,
        threshold_mode=moving_cfg.threshold_mode,
        threshold_db=moving_cfg.threshold_db,
        min_power_db=moving_cfg.min_power_db,
    )
    peak_idx = extract_detection_peaks_2d(
        moving_db,
        threshold_db=threshold_db,
        win_row=moving_cfg.localmax_range_bins,
        win_col=moving_cfg.localmax_doppler_bins,
        max_peaks=moving_cfg.max_detections,
    )
    detections: list[Detection] = []
    for r_bin, d_bin in peak_idx:
        rb = int(r_bin)
        dbin = int(d_bin)
        range_m = float(rb) * float(range_bin_m)
        doppler_mps = None
        if doppler_axis_mps is not None and 0 <= dbin < int(doppler_axis_mps.size):
            doppler_mps = float(doppler_axis_mps[dbin])
        p_lin = float(range_doppler_map[rb, dbin])
        p_db = float(moving_db[rb, dbin])
        detections.append(
            Detection(
                range_bin=rb,
                angle_bin=None,
                doppler_bin=dbin,
                range_m=range_m,
                angle_deg=0.0,
                doppler_mps=doppler_mps,
                x_m=0.0,
                y_m=range_m,
                power_lin=p_lin,
                power_db=p_db,
                source="moving",
            )
        )
    return detections


def estimate_angle_for_moving_detections(
    detections: list[Detection],
    doppler_cube: np.ndarray,
    *,
    w_angle: np.ndarray,
    apply_angle_window: bool,
    angle_cfg: AngleProcessingConfig,
    dsp_cfg: RealtimeDSPConfig,
    virtual_array_geometry: VirtualArrayGeometry,
    angle_steering: np.ndarray,
    angle_axis_deg: np.ndarray,
    calibration_enabled: bool = False,
    cal_vector: np.ndarray | None = None,
    tdm_mimo_compensation_table: np.ndarray | None = None,
) -> list[Detection]:
    if not detections or doppler_cube.ndim != 5:
        return detections
    n_frames = int(doppler_cube.shape[0])
    n_doppler = int(doppler_cube.shape[1])
    n_range = int(doppler_cube.shape[3])
    snapshot_work = np.empty((n_frames, int(dsp_cfg.tx), int(dsp_cfg.rx)), dtype=np.complex64)
    va = np.empty((n_frames, 1, 1, int(dsp_cfg.virtual_ant)), dtype=np.complex64)
    va_view = va[:, 0, 0, :]
    for det in detections:
        if det.doppler_bin is None:
            continue
        dbin = int(det.doppler_bin)
        rbin = int(det.range_bin)
        if dbin < 0 or dbin >= n_doppler or rbin < 0 or rbin >= n_range:
            continue
        snapshot = doppler_cube[:, dbin, :, rbin, :]
        snapshot_comp = apply_tdm_mimo_doppler_compensation(
            snapshot,
            doppler_bin=dbin,
            compensation_table=tdm_mimo_compensation_table,
            out=snapshot_work,
        )
        snapshot_flat = snapshot_comp.reshape(n_frames, int(dsp_cfg.virtual_ant))
        if virtual_array_geometry.identity_order:
            np.copyto(va_view, snapshot_flat, casting="unsafe")
        else:
            np.take(snapshot_flat, virtual_array_geometry.order_flat, axis=-1, out=va_view)
        if calibration_enabled:
            _apply_calibration_vector(va, cal_vector)
        if apply_angle_window:
            va *= w_angle
        angle_pow = compute_angle_heatmap(
            va,
            angle_cfg=angle_cfg,
            dsp_cfg=dsp_cfg,
            angle_steering=angle_steering,
            geometry=virtual_array_geometry,
        )
        if angle_pow.size <= 0:
            continue
        angle_spec = angle_pow[0]
        angle_bin = int(np.argmax(angle_spec))
        angle_deg = float(angle_axis_deg[angle_bin]) if 0 <= angle_bin < int(angle_axis_deg.size) else 0.0
        angle_rad = np.deg2rad(np.float32(angle_deg))
        det.angle_bin = angle_bin
        det.angle_deg = angle_deg
        det.x_m = float(det.range_m * float(np.sin(angle_rad)))
        det.y_m = float(det.range_m * float(np.cos(angle_rad)))
    return detections


def _merge_two_detections(
    det_static: Detection,
    det_moving: Detection,
    fusion_cfg: FusionConfig,
) -> Detection:
    use_moving_primary = bool(
        fusion_cfg.prefer_moving_when_doppler_valid and det_moving.doppler_mps is not None
    )
    if use_moving_primary:
        range_bin = int(det_moving.range_bin)
        angle_bin = det_moving.angle_bin if det_moving.angle_bin is not None else det_static.angle_bin
        range_m = float(det_moving.range_m)
        angle_deg = float(det_moving.angle_deg if det_moving.angle_bin is not None else det_static.angle_deg)
        x_m = float(det_moving.x_m if det_moving.angle_bin is not None else det_static.x_m)
        y_m = float(det_moving.y_m if det_moving.angle_bin is not None else det_static.y_m)
    else:
        w_static = max(float(det_static.power_lin), 1e-6)
        w_moving = max(float(det_moving.power_lin), 1e-6)
        w_sum = w_static + w_moving
        x_m = (det_static.x_m * w_static + det_moving.x_m * w_moving) / w_sum
        y_m = (det_static.y_m * w_static + det_moving.y_m * w_moving) / w_sum
        range_m = float(np.hypot(x_m, y_m))
        angle_deg = float(np.rad2deg(np.arctan2(np.float32(x_m), np.float32(max(y_m, 1e-6)))))
        range_bin = int(round((det_static.range_bin * w_static + det_moving.range_bin * w_moving) / w_sum))
        angle_bin = det_moving.angle_bin if det_moving.angle_bin is not None else det_static.angle_bin
    return Detection(
        range_bin=range_bin,
        angle_bin=angle_bin,
        doppler_bin=det_moving.doppler_bin,
        range_m=range_m,
        angle_deg=angle_deg,
        doppler_mps=det_moving.doppler_mps,
        x_m=x_m,
        y_m=y_m,
        power_lin=max(float(det_static.power_lin), float(det_moving.power_lin)),
        power_db=max(float(det_static.power_db), float(det_moving.power_db)),
        source="fused",
    )


def fuse_detections(
    detections_static: list[Detection],
    detections_moving: list[Detection],
    fusion_cfg: FusionConfig,
) -> list[Detection]:
    if not detections_static and not detections_moving:
        return []
    if not fusion_cfg.enabled:
        out = list(detections_moving) + list(detections_static)
        out.sort(key=lambda d: d.power_lin, reverse=True)
        return out

    fused: list[Detection] = []
    used_static: set[int] = set()
    moving_sorted = sorted(detections_moving, key=lambda d: d.power_lin, reverse=True)
    for moving_det in moving_sorted:
        best_idx = -1
        best_xy = float("inf")
        for si, static_det in enumerate(detections_static):
            if si in used_static:
                continue
            dx = float(moving_det.x_m - static_det.x_m)
            dy = float(moving_det.y_m - static_det.y_m)
            d_xy = float(np.hypot(dx, dy))
            if d_xy > fusion_cfg.merge_xy_m:
                continue
            if abs(float(moving_det.range_m - static_det.range_m)) > fusion_cfg.merge_range_m:
                continue
            if abs(float(moving_det.angle_deg - static_det.angle_deg)) > fusion_cfg.merge_angle_deg:
                continue
            if d_xy < best_xy:
                best_xy = d_xy
                best_idx = si
        if best_idx >= 0:
            used_static.add(best_idx)
            fused.append(_merge_two_detections(detections_static[best_idx], moving_det, fusion_cfg))
        else:
            fused.append(moving_det)

    for si, static_det in enumerate(detections_static):
        if si not in used_static:
            fused.append(static_det)
    fused.sort(key=lambda d: d.power_lin, reverse=True)
    return fused


def clean_detections_for_tracking(
    detections: list[Detection],
    fusion_cfg: FusionConfig,
) -> list[Detection]:
    if not detections:
        return []

    cleaned: list[Detection] = []
    for det in detections:
        x_m = float(det.x_m)
        y_m = float(det.y_m)
        if not (math.isfinite(x_m) and math.isfinite(y_m)):
            continue

        range_m = float(det.range_m)
        if not math.isfinite(range_m) or range_m < 0.0:
            range_m = float(math.hypot(x_m, y_m))

        angle_deg = float(det.angle_deg)
        if not math.isfinite(angle_deg):
            angle_deg = float(np.rad2deg(np.arctan2(np.float32(x_m), np.float32(max(y_m, 1e-6)))))

        doppler_mps = det.doppler_mps
        if doppler_mps is not None:
            doppler_val = float(doppler_mps)
            doppler_mps = doppler_val if math.isfinite(doppler_val) else None

        power_lin = float(det.power_lin)
        if not math.isfinite(power_lin):
            power_lin = 0.0
        power_db = float(det.power_db)
        if not math.isfinite(power_db):
            power_db = 0.0

        cleaned.append(
            Detection(
                range_bin=int(det.range_bin),
                angle_bin=None if det.angle_bin is None else int(det.angle_bin),
                doppler_bin=None if det.doppler_bin is None else int(det.doppler_bin),
                range_m=range_m,
                angle_deg=angle_deg,
                doppler_mps=doppler_mps,
                x_m=x_m,
                y_m=y_m,
                power_lin=power_lin,
                power_db=power_db,
                source=str(det.source),
            )
        )

    if len(cleaned) <= 1 or not fusion_cfg.enabled:
        return cleaned

    dedup_xy_m = max(0.05, 0.5 * float(fusion_cfg.merge_xy_m))
    dedup_range_m = max(0.10, 0.5 * float(fusion_cfg.merge_range_m))
    deduped: list[Detection] = []
    for det in sorted(cleaned, key=lambda d: d.power_lin, reverse=True):
        duplicate = False
        for kept in deduped:
            if (
                math.hypot(det.x_m - kept.x_m, det.y_m - kept.y_m) <= dedup_xy_m
                and abs(det.range_m - kept.range_m) <= dedup_range_m
            ):
                duplicate = True
                break
        if not duplicate:
            deduped.append(det)
    return deduped


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
    w_doppler: np.ndarray,
    w_angle: np.ndarray,
    apply_range_window: bool,
    apply_doppler_window: bool,
    apply_angle_window: bool,
    mean_before_range_fft: MeanSelection,
    detection_static_post_range_fft_filters: PostRangeFftFilterConfig,
    detection_moving_pre_doppler_filters: PostRangeFftFilterConfig,
    display_post_range_fft_filters: PostRangeFftFilterConfig,
    apply_detection_loop_average_after_background: bool,
    apply_display_loop_average_after_background: bool,
    angle_processing: AngleProcessingConfig,
    heatmap_ema_cfg: HeatmapEMAConfig,
    heatmap_spatial_filter_cfg: HeatmapSpatialFilterConfig,
    display_projection_cfg: DisplayProjectionConfig,
    virtual_array_geometry: VirtualArrayGeometry,
    angle_steering: np.ndarray,
    angle_axis_deg: np.ndarray,
    display_projection_lut: dict[str, Any] | None,
    display_y_max_m: float,
    display_x_max_m: float,
    doppler_axis_mps: np.ndarray | None,
    tdm_mimo_compensation_table: np.ndarray | None,
    detection_static_cfg: DetectionConfigStatic,
    detection_moving_cfg: DetectionConfigMoving,
    fusion_cfg: FusionConfig,
    detection_static_bg_state: BackgroundSubtractionState,
    detection_moving_bg_state: BackgroundSubtractionState,
    display_bg_state: BackgroundSubtractionState,
    heatmap_ema: np.ndarray | None,
    calibration_enabled: bool,
    cal_vector: np.ndarray,
    pending_calibration: bool,
    virtual_array_work_buf: np.ndarray | None,
    virtual_array_flat_work_buf: np.ndarray | None,
    doppler_work_buf: np.ndarray | None,
    profiles_db_work_buf: np.ndarray | None,
    heatmap_db_work_buf: np.ndarray | None,
    gui_h: int,
    gui_w: int,
    gui_heat_views: tuple[np.ndarray, np.ndarray],
    gui_profile_views: tuple[np.ndarray, np.ndarray],
    gui_latest_idx: Synchronized,
    gui_latest_seq: Synchronized,
    gui_lock,
    normalize_to_peak: bool,
    profiles_out_buf: np.ndarray,
    stat_raw_min_db: Synchronized,
    stat_raw_max_db: Synchronized,
    stat_norm_min_db: Synchronized,
    stat_norm_max_db: Synchronized,
    dsp_cfg: RealtimeDSPConfig,
    processing_max_bin: int,
    display_max_bin: int,
    debug_print_top_peaks: bool = False,
    debug_top_peaks_count: int = 10,
    debug_top_peaks_range_min_m: float | None = None,
    debug_top_peaks_range_max_m: float | None = None,
    debug_top_peaks_angle_min_deg: float | None = None,
    debug_top_peaks_angle_max_deg: float | None = None,
) -> tuple[np.ndarray | None, list[Detection], np.ndarray]:
    try:
        cal_vector_out = np.asarray(cal_vector, dtype=np.complex64).reshape(-1)

        # Raw complex stream -> Reshape -> radar tensor [frame, loop, tx, sample, rx].
        data = raw_buffer.reshape(n_frames,dsp_cfg.chirps // dsp_cfg.tx,dsp_cfg.tx,dsp_cfg.samples,dsp_cfg.rx,)

        # Range-FFT pre-processing: static mean subtraction 
        data = subtract_selected_mean(data, mean_before_range_fft)

        # Apply range window (broadcasting over all non-range dimensions) to reduce sidelobes before the range FFT.
        if apply_range_window:
            data *= w_range

        # Range FFT with optional zero-padding and in-place computation to save memory.
        range_fft_common = fft.fft(data,n=dsp_cfg.nfft_range,axis=3,workers=dsp_cfg.fft_workers,overwrite_x=True,)

        # Range bin to physical range conversion factor (meters per bin).
        range_bin_m = dsp_cfg.c * dsp_cfg.fs / (2.0 * dsp_cfg.slope * dsp_cfg.nfft_range)

        if pending_calibration:
            calibration_max_bin = max(1, min(max(int(processing_max_bin), int(display_max_bin)), dsp_cfg.nfft_range // 2))
            calibration_virtual_array = _build_virtual_array_from_range_fft(
                range_fft_common,
                max_bin=calibration_max_bin,
                dsp_cfg=dsp_cfg,
                geometry=virtual_array_geometry,
                work_buf=(
                    None
                    if virtual_array_work_buf is None
                    else virtual_array_work_buf[
                        :n_frames,
                        : int(range_fft_common.shape[1]),
                        :calibration_max_bin,
                        :,
                        :,
                    ]
                ),
                flat_work_buf=(
                    None
                    if virtual_array_flat_work_buf is None
                    else virtual_array_flat_work_buf[
                        :n_frames,
                        : int(range_fft_common.shape[1]),
                        :calibration_max_bin,
                        :,
                    ]
                ),
            )
            updated_cal_vector, target_bin = _estimate_boresight_calibration_vector(calibration_virtual_array)
            if updated_cal_vector is None:
                print("[DSP CAL] boresight calibration skipped: target/reference antenna not valid on current batch.")
            else:
                cal_vector_out = updated_cal_vector
                real_str, imag_str = _format_calibration_vector_for_log(cal_vector_out)
                print(f"[DSP CAL] boresight updated on range_bin={int(target_bin or 0)}")
                print(f"[DSP CAL] vector_real: [{real_str}]")
                print(f"[DSP CAL] vector_imag: [{imag_str}]")


        # Split the shared range-FFT cube into three logical branches:
        # static detection, moving detection, and display.
        # Each branch reuses the common range-FFT output unless its own
        # post-processing filters require an isolated copy.
        range_fft_detection_static = range_fft_common
        if detection_static_cfg.enabled:
            range_fft_detection_static_in = (
                range_fft_common.copy()
                if _branch_needs_copy(detection_static_post_range_fft_filters)
                else range_fft_common
            )
            range_fft_detection_static = apply_post_range_fft_filters(
                range_fft_detection_static_in,
                detection_static_post_range_fft_filters,
                bg_state=detection_static_bg_state,
                fft_workers=dsp_cfg.fft_workers,
                apply_loop_average_after_background=apply_detection_loop_average_after_background,
            )

        range_fft_detection_moving = range_fft_common
        if detection_moving_cfg.enabled:
            range_fft_detection_moving_in = (
                range_fft_common.copy()
                if _branch_needs_copy(detection_moving_pre_doppler_filters)
                else range_fft_common
            )
            range_fft_detection_moving = apply_post_range_fft_filters(
                range_fft_detection_moving_in,
                detection_moving_pre_doppler_filters,
                bg_state=detection_moving_bg_state,
                fft_workers=dsp_cfg.fft_workers,
                apply_loop_average_after_background=False,
            )

        range_fft_display_in = (
            range_fft_common.copy()
            if _branch_needs_copy(display_post_range_fft_filters)
            else range_fft_common
        )
        # chain of filters that are executed pn the Range FFT for the display branch.
        # range_fft_in -> slow_time_filter -> subtract_selected_mean -> background_subtraction -> loop_average -> range_fft_display
        range_fft_display = apply_post_range_fft_filters(
            range_fft_display_in,
            display_post_range_fft_filters,
            bg_state=display_bg_state,
            fft_workers=dsp_cfg.fft_workers,
            apply_loop_average_after_background=apply_display_loop_average_after_background,
        )

        # Tracking path (physical data): no display EMA/blur/normalization.
        detections_static: list[Detection] = []
        detections_moving: list[Detection] = []
        # If enabled, detect static targets from the range-FFT cube and estimate their angle/range.
        if detection_static_cfg.enabled:
            detections_static, _ = detect_static_targets(
                range_fft_detection_static,
                static_cfg=detection_static_cfg,
                angle_cfg=angle_processing,
                dsp_cfg=dsp_cfg,
                virtual_array_geometry=virtual_array_geometry,
                w_angle=w_angle,
                angle_steering=angle_steering,
                angle_axis_deg=angle_axis_deg,
                range_bin_m=range_bin_m,
                max_bin=processing_max_bin,
                apply_angle_window=apply_angle_window,
                calibration_enabled=calibration_enabled,
                cal_vector=cal_vector_out,
                virtual_array_work_buf=(
                    None
                    if virtual_array_work_buf is None
                    else virtual_array_work_buf[
                        :n_frames,
                        : int(range_fft_detection_static.shape[1]),
                        :processing_max_bin,
                        :,
                        :,
                    ]
                ),
                virtual_array_flat_work_buf=(
                    None
                    if virtual_array_flat_work_buf is None
                    else virtual_array_flat_work_buf[
                        :n_frames,
                        : int(range_fft_detection_static.shape[1]),
                        :processing_max_bin,
                        :,
                    ]
                ),
            )
        
        if detection_moving_cfg.enabled:
            doppler_cube, range_doppler_map = compute_range_doppler(
                range_fft_detection_moving,
                max_bin=processing_max_bin,
                dsp_cfg=dsp_cfg,
                moving_cfg=detection_moving_cfg,
                w_doppler=w_doppler,
                apply_doppler_window=apply_doppler_window,
                doppler_work_buf=(
                    None
                    if doppler_work_buf is None
                    else doppler_work_buf[:n_frames, : int(range_fft_detection_moving.shape[1]), :, :, :]
                ),
            )
            detections_moving = detect_moving_targets(
                range_doppler_map,
                moving_cfg=detection_moving_cfg,
                range_bin_m=range_bin_m,
                doppler_axis_mps=doppler_axis_mps,
            )
            detections_moving = estimate_angle_for_moving_detections(
                detections_moving,
                doppler_cube,
                w_angle=w_angle,
                apply_angle_window=apply_angle_window,
                angle_cfg=angle_processing,
                dsp_cfg=dsp_cfg,
                virtual_array_geometry=virtual_array_geometry,
                angle_steering=angle_steering,
                angle_axis_deg=angle_axis_deg,
                calibration_enabled=calibration_enabled,
                cal_vector=cal_vector_out,
                tdm_mimo_compensation_table=tdm_mimo_compensation_table,
            )
        fused_detections = fuse_detections(detections_static, detections_moving, fusion_cfg)
        tracking_detections = clean_detections_for_tracking(fused_detections, fusion_cfg)

        fft_profile_bins = min(int(range_fft_display.shape[3]), int(profiles_out_buf.shape[1]))
        profiles_out = profiles_out_buf
        profiles_out.fill(np.float32(-120.0))
        if fft_profile_bins > 0:
            # Keep the Range FFT plot decoupled from the physical max_bin used by the radar logic.
            profiles_src = range_fft_display[:, :, :, :fft_profile_bins, :]
            prof_re = profiles_src.real
            prof_im = profiles_src.imag
            profiles_pow = (prof_re * prof_re + prof_im * prof_im).mean(axis=(0, 1), dtype=np.float32)
            if (
                profiles_db_work_buf is not None
                and profiles_db_work_buf.shape == profiles_pow.shape
                and profiles_db_work_buf.dtype == np.float32
            ):
                profiles_db = profiles_db_work_buf
                np.copyto(profiles_db, profiles_pow, casting="unsafe")
            else:
                profiles_db = np.array(profiles_pow, dtype=np.float32, copy=True)
            np.add(profiles_db, np.float32(1e-12), out=profiles_db)
            np.log10(profiles_db, out=profiles_db)
            profiles_db *= np.float32(10.0)
            profiles_db_va = profiles_db.transpose(0, 2, 1).reshape(int(dsp_cfg.virtual_ant), int(fft_profile_bins))
            copy_rows = min(int(dsp_cfg.range_profile_count), int(profiles_db_va.shape[0]))
            if copy_rows > 0:
                profiles_out[:copy_rows, :fft_profile_bins] = profiles_db_va[:copy_rows, :].astype(np.float32, copy=False)

        # Build the virtual array after trimming range bins to limit memory traffic.
        # [frame, loop, range_bin, virtual_ant]
        virtual_array = _build_virtual_array_from_range_fft(
            range_fft_display,
            max_bin=display_max_bin,
            dsp_cfg=dsp_cfg,
            geometry=virtual_array_geometry,
            work_buf=(
                None
                if virtual_array_work_buf is None
                else virtual_array_work_buf[
                    :n_frames,
                    : int(range_fft_display.shape[1]),
                    :display_max_bin,
                    :,
                    :,
                ]
            ),
            flat_work_buf=(
                None
                if virtual_array_flat_work_buf is None
                else virtual_array_flat_work_buf[
                    :n_frames,
                    : int(range_fft_display.shape[1]),
                    :display_max_bin,
                    :,
                ]
            ),
        )

        if calibration_enabled:
            _apply_calibration_vector(virtual_array, cal_vector_out)
        if apply_angle_window:
            virtual_array *= w_angle

        heatmap = compute_angle_heatmap(
            virtual_array,
            angle_cfg=angle_processing,
            dsp_cfg=dsp_cfg,
            angle_steering=angle_steering,
            geometry=virtual_array_geometry,
        )

        if not heatmap_ema_cfg.enabled: #bypass
            heatmap_ema = heatmap
        elif heatmap_ema is None: #initialize
            heatmap_ema = heatmap
        else: # update
            heatmap_ema *= (1.0 - heatmap_ema_cfg.alpha)
            heatmap_ema += (heatmap_ema_cfg.alpha * heatmap)
        heatmap_ema = apply_heatmap_spatial_filter(heatmap_ema, heatmap_spatial_filter_cfg)

        # Display path only: project the linear polar heatmap onto the GUI grid, then convert to dB.
        view_db = project_heatmap_for_display(
            heatmap_ema,
            angle_axis_deg=angle_axis_deg,
            dr_m=range_bin_m,
            gui_h=gui_h,
            gui_w=gui_w,
            y_max_m=display_y_max_m,
            x_max_m=display_x_max_m,
            projection_mode=display_projection_cfg.projection_mode,
            projection_interp=display_projection_cfg.projection_interp,
            out=heatmap_db_work_buf,
            fill_value=0.0,
            precomputed_lut=display_projection_lut,
        )
        np.add(view_db, np.float32(1e-12), out=view_db)
        np.log10(view_db, out=view_db)
        view_db *= np.float32(10.0)

        if debug_print_top_peaks:
            print(
                _format_debug_top_peaks_range_angle(
                    heatmap_ema,
                    angle_axis_deg=angle_axis_deg,
                    range_bin_m=range_bin_m,
                    top_k=debug_top_peaks_count,
                    range_min_m=debug_top_peaks_range_min_m,
                    range_max_m=debug_top_peaks_range_max_m,
                    angle_min_deg=debug_top_peaks_angle_min_deg,
                    angle_max_deg=debug_top_peaks_angle_max_deg,
                )
            )
            print(
                _format_debug_top_peaks_xy(
                    view_db,
                    x_max_m=display_x_max_m,
                    y_max_m=display_y_max_m,
                    top_k=debug_top_peaks_count,
                    range_min_m=debug_top_peaks_range_min_m,
                    range_max_m=debug_top_peaks_range_max_m,
                    angle_min_deg=debug_top_peaks_angle_min_deg,
                    angle_max_deg=debug_top_peaks_angle_max_deg,
                )
            )

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
            dst = gui_heat_views[next_idx]
            dst.fill(-120.0)
            flat = view_db.reshape(-1)
            n = min(dst.size, flat.size)
            if n > 0:
                dst[:n] = flat[:n]

            dst_prof = gui_profile_views[next_idx]
            dst_prof.fill(-120.0)
            prof_flat = profiles_out.reshape(-1)
            n_prof = min(dst_prof.size, prof_flat.size)
            if n_prof > 0:
                dst_prof[:n_prof] = prof_flat[:n_prof]
            gui_latest_idx.value = next_idx
            gui_latest_seq.value = int(gui_latest_seq.value) + 1
        return heatmap_ema, tracking_detections, cal_vector_out

    except Exception as e:
        print(f"[DSP ERR] {e}")
        return heatmap_ema, [], np.asarray(cal_vector, dtype=np.complex64).reshape(-1)


def dsp_worker(
    free_slots,
    dsp_ready_queue,
    dsp_cmd_queue,
    shm_frames,
    slot_state,
    slot_ok,
    slot_usemask,
    slot_pub_seq,
    publish_lock,
    gui_dbuf,
    gui_prof_dbuf,
    gui_h: int,
    fft_plot_h: int,
    gui_w: int,
    gui_latest_idx: Synchronized,
    gui_latest_seq: Synchronized,
    gui_lock,
    tracks_xy_dbuf,
    tracks_meta_dbuf,
    tracks_state_dbuf,
    tracks_stop_xy_dbuf,
    tracks_count: Synchronized,
    tracks_seq: Synchronized,
    tracks_lock,
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
    calibration_cfg = getattr(dsp_cfg, "calibration", calibration_from_yaml_dict(cfg_dict, virtual_ant=int(dsp_cfg.virtual_ant)))
    mean_before_range_fft, _ = mean_selections_from_yaml_dict(cfg_dict)
    detection_static_post_range_fft_filters = detection_static_post_range_fft_filters_from_yaml_dict(cfg_dict)
    detection_static_post_range_fft_filters, detection_static_filter_warnings = sanitize_detection_static_post_range_fft_filters(
        detection_static_post_range_fft_filters
    )
    detection_moving_pre_doppler_filters = detection_moving_pre_doppler_filters_from_yaml_dict(cfg_dict)
    detection_moving_pre_doppler_filters, detection_moving_filter_warnings = sanitize_detection_moving_pre_doppler_filters(
        detection_moving_pre_doppler_filters
    )
    display_post_range_fft_filters = display_post_range_fft_filters_from_yaml_dict(cfg_dict)
    display_post_range_fft_filters, display_filter_warnings = sanitize_display_post_range_fft_filters(
        display_post_range_fft_filters
    )
    angle_processing = angle_processing_from_yaml_dict(cfg_dict)
    heatmap_ema_cfg = heatmap_ema_from_yaml_dict(cfg_dict)
    heatmap_spatial_filter_cfg = heatmap_spatial_filter_from_yaml_dict(cfg_dict)
    display_projection_cfg = display_projection_from_yaml_dict(cfg_dict)
    detection_static_cfg = detection_static_from_yaml_dict(cfg_dict)
    detection_moving_cfg = detection_moving_from_yaml_dict(cfg_dict)
    fusion_cfg = fusion_from_yaml_dict(cfg_dict)
    tracking_cfg = tracking_from_yaml_dict(cfg_dict)
    tracker_cfg = tracker_from_yaml_dict(cfg_dict)
    debug_block = cfg_dict.get("debug", {}) or {}
    debug_print_top_peaks = bool(debug_block.get("print_top_peaks", False))
    debug_top_peaks_count = max(1, int(_to_int(debug_block.get("print_top_peaks_count", 10), 10)))
    range_min_raw = debug_block.get("print_top_peaks_range_min_m", None)
    range_max_raw = debug_block.get("print_top_peaks_range_max_m", None)
    angle_min_raw = debug_block.get("print_top_peaks_angle_min_deg", None)
    angle_max_raw = debug_block.get("print_top_peaks_angle_max_deg", None)
    debug_top_peaks_range_min_m = None if range_min_raw is None else _to_float(range_min_raw, float("nan"))
    debug_top_peaks_range_max_m = None if range_max_raw is None else _to_float(range_max_raw, float("nan"))
    debug_top_peaks_angle_min_deg = None if angle_min_raw is None else _to_float(angle_min_raw, float("nan"))
    debug_top_peaks_angle_max_deg = None if angle_max_raw is None else _to_float(angle_max_raw, float("nan"))
    if debug_top_peaks_range_min_m is not None and not np.isfinite(float(debug_top_peaks_range_min_m)):
        debug_top_peaks_range_min_m = None
    if debug_top_peaks_range_max_m is not None and not np.isfinite(float(debug_top_peaks_range_max_m)):
        debug_top_peaks_range_max_m = None
    if debug_top_peaks_angle_min_deg is not None and not np.isfinite(float(debug_top_peaks_angle_min_deg)):
        debug_top_peaks_angle_min_deg = None
    if debug_top_peaks_angle_max_deg is not None and not np.isfinite(float(debug_top_peaks_angle_max_deg)):
        debug_top_peaks_angle_max_deg = None
    virtual_array_geometry, virtual_array_warnings = build_virtual_array_geometry_from_yaml_dict(cfg_dict, dsp_cfg)
    tracking_runtime_enabled = bool(getattr(tracking_cfg, "enabled", False))
    if tracking_runtime_enabled and int(dsp_cfg.x_frames) > 1:
        print(
            f"[TRACK WARN] tracking disabled because capture.x_frames={int(dsp_cfg.x_frames)}. "
            "Current tracker expects x_frames=1."
        )
        tracking_runtime_enabled = False
    for warn_msg in (
        detection_static_filter_warnings
        + detection_moving_filter_warnings
        + display_filter_warnings
        + virtual_array_warnings
    ):
        print(f"[DSP WARN] {warn_msg}")
    window_range, window_doppler, window_angle = build_windows(
        selection,
        samples=dsp_cfg.samples,
        n_loops=dsp_cfg.chirps // dsp_cfg.tx,
        virtual_ant=dsp_cfg.virtual_ant,
    )
    apply_range_window = not _window_is_identity(selection.window_range)
    apply_doppler_window = not _window_is_identity(selection.window_doppler)
    apply_angle_window = not _window_is_identity(selection.window_angle)
    angle_steering = build_angle_steering_matrix(
        virtual_ant=dsp_cfg.virtual_ant,
        nfft_angle=dsp_cfg.nfft_angle,
        geometry=virtual_array_geometry,
    )
    angle_axis_deg = build_angle_axis_deg(dsp_cfg.nfft_angle, geometry=virtual_array_geometry)
    if (
        str(getattr(angle_processing, "mode", "fft")).strip().lower() == "fft"
        and virtual_array_geometry.uniform_spacing_lambda is None
    ):
        print(
            "[DSP WARN] angle_processing.mode='fft' requires a uniformly spaced virtual array; "
            "for non-uniform phase centers only bartlett/mvdr use the geometry exactly."
        )
    display_y_max_m = float(dsp_cfg.range_max_display)
    display_x_max_m = resolve_display_crossrange_max_m(display_y_max_m, angle_axis_deg, display_projection_cfg)
    display_projection_lut = build_display_projection_lut(
        gui_h=gui_h,
        gui_w=gui_w,
        x_max_m=display_x_max_m,
        y_max_m=display_y_max_m,
        dr_m=dsp_cfg.c * dsp_cfg.fs / (2.0 * dsp_cfg.slope * dsp_cfg.nfft_range),
        angle_axis_deg=angle_axis_deg,
        projection_mode=display_projection_cfg.projection_mode,
        projection_interp=display_projection_cfg.projection_interp,
    )
    n_doppler = int(dsp_cfg.chirps // dsp_cfg.tx)
    doppler_axis_mps = build_doppler_axis_mps(
        cfg_dict,
        dsp_cfg,
        n_doppler,
        doppler_fft_shift=detection_moving_cfg.doppler_fft_shift,
    )
    tdm_mimo_compensation_table = build_tdm_mimo_doppler_compensation_table(
        n_doppler,
        int(dsp_cfg.tx),
        doppler_fft_shift=detection_moving_cfg.doppler_fft_shift,
    )

    dr = dsp_cfg.c * dsp_cfg.fs / (2.0 * dsp_cfg.slope * dsp_cfg.nfft_range)
    processing_range_max_m = float(
        dsp_cfg.range_max_processing_m if dsp_cfg.range_max_processing_m > 0.0 else dsp_cfg.range_max_display
    )
    processing_max_bin = int(np.floor(processing_range_max_m / dr))
    processing_max_bin = max(1, min(processing_max_bin, dsp_cfg.nfft_range // 2))
    display_max_bin = int(np.floor(display_y_max_m / dr))
    display_max_bin = max(1, min(display_max_bin, dsp_cfg.nfft_range // 2))
    max_virtual_array_bin = max(processing_max_bin, display_max_bin)

    total_samples_needed = dsp_cfg.x_frames * dsp_cfg.chirps * dsp_cfg.samples * dsp_cfg.rx
    complex_per_frame = dsp_cfg.chirps * dsp_cfg.samples * dsp_cfg.rx

    complex_data = np.zeros(total_samples_needed, dtype=np.complex64)
    profiles_out_buf = np.empty((dsp_cfg.range_profile_count, int(fft_plot_h)), dtype=np.float32)
    virtual_array_work_buf = np.empty(
        (dsp_cfg.x_frames, n_doppler, max_virtual_array_bin, dsp_cfg.tx, dsp_cfg.rx),
        dtype=np.complex64,
    )
    virtual_array_flat_work_buf = np.empty(
        (dsp_cfg.x_frames, n_doppler, max_virtual_array_bin, dsp_cfg.virtual_ant),
        dtype=np.complex64,
    )
    doppler_work_buf = np.empty(
        (dsp_cfg.x_frames, n_doppler, dsp_cfg.tx, processing_max_bin, dsp_cfg.rx),
        dtype=np.complex64,
    )
    profiles_db_work_buf = np.empty((dsp_cfg.tx, int(fft_plot_h), dsp_cfg.rx), dtype=np.float32)
    heatmap_db_work_buf = np.empty((int(gui_h), int(gui_w)), dtype=np.float32)
    gui_heat_size = int(gui_h) * int(gui_w)
    gui_heat_views = (
        np.frombuffer(gui_dbuf, dtype=np.float32, count=gui_heat_size, offset=0),
        np.frombuffer(gui_dbuf, dtype=np.float32, count=gui_heat_size, offset=gui_heat_size * 4),
    )
    gui_prof_size = int(dsp_cfg.range_profile_count) * int(fft_plot_h)
    gui_profile_views = (
        np.frombuffer(gui_prof_dbuf, dtype=np.float32, count=gui_prof_size, offset=0),
        np.frombuffer(gui_prof_dbuf, dtype=np.float32, count=gui_prof_size, offset=gui_prof_size * 4),
    )

    shm_view = memoryview(shm_frames).cast("B")
    n_slots = len(slot_state)
    heatmap_ema = None
    calibration_enabled = bool(calibration_cfg.enabled)
    cal_vector = _build_complex_calibration_vector(calibration_cfg, int(dsp_cfg.virtual_ant))
    pending_calibration = False
    detection_static_bg_state = BackgroundSubtractionState()
    detection_moving_bg_state = BackgroundSubtractionState()
    display_bg_state = BackgroundSubtractionState()
    tracker: MultiObjectTracker | None
    if tracking_runtime_enabled:
        tracker = MultiObjectTracker(tracking_cfg=tracking_cfg, tracker_cfg=tracker_cfg)
    else:
        tracker = None
    tracker_nominal_frame_dt_s = resolve_nominal_frame_period_s(cfg_dict, dsp_cfg, tracking_cfg)
    tracker_time_s: float | None = None
    last_tracker_seq: int | None = None
    tracks_xy_view = np.frombuffer(tracks_xy_dbuf, dtype=np.float32)
    tracks_meta_view = np.frombuffer(tracks_meta_dbuf, dtype=np.int32)
    tracks_state_view = np.frombuffer(tracks_state_dbuf, dtype=np.int32)
    tracks_stop_xy_view = np.frombuffer(tracks_stop_xy_dbuf, dtype=np.float32)
    tracks_capacity = int(
        min(
            tracks_xy_view.size // 4,
            tracks_meta_view.size // 4,
            tracks_state_view.size // 2,
            tracks_stop_xy_view.size // 2,
        )
    )
    warned_display_loop_average_after_doppler = False
    warned_detection_loop_average_after_doppler = False
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

    def _publish_tracks_latest_wins(active_tracks: list[Track]) -> None:
        if tracks_capacity <= 0:
            return
        n_pub = min(int(len(active_tracks)), tracks_capacity)
        with tracks_lock:
            for idx_tr in range(n_pub):
                tr = active_tracks[idx_tr]
                base = idx_tr * 4
                base2 = idx_tr * 2
                tracks_xy_view[base + 0] = np.float32(tr.x_m)
                tracks_xy_view[base + 1] = np.float32(tr.y_m)
                tracks_xy_view[base + 2] = np.float32(tr.vx_mps)
                tracks_xy_view[base + 3] = np.float32(tr.vy_mps)
                tracks_meta_view[base + 0] = int(tr.track_id)
                tracks_meta_view[base + 1] = 1 if tr.confirmed else 0
                tracks_meta_view[base + 2] = int(tr.age)
                tracks_meta_view[base + 3] = int(tr.missed_frames)
                motion_state = str(getattr(tr, "motion_state", "unknown") or "unknown").strip().lower()
                if motion_state == "moving":
                    motion_state_code = 1
                elif motion_state == "stopped":
                    motion_state_code = 2
                else:
                    motion_state_code = 0
                stop_x = getattr(tr, "stop_x_m", None)
                stop_y = getattr(tr, "stop_y_m", None)
                has_stop = int(
                    stop_x is not None
                    and stop_y is not None
                    and math.isfinite(float(stop_x))
                    and math.isfinite(float(stop_y))
                )
                tracks_state_view[base2 + 0] = int(motion_state_code)
                tracks_state_view[base2 + 1] = int(has_stop)
                tracks_stop_xy_view[base2 + 0] = np.float32(float(stop_x) if has_stop else np.nan)
                tracks_stop_xy_view[base2 + 1] = np.float32(float(stop_y) if has_stop else np.nan)
            tracks_count.value = n_pub
            tracks_seq.value = int(tracks_seq.value) + 1

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

    def _poll_dsp_commands() -> None:
        nonlocal pending_calibration
        while True:
            try:
                cmd = dsp_cmd_queue.get_nowait()
            except pyqueue.Empty:
                break
            if not isinstance(cmd, dict):
                continue
            cmd_type = str(cmd.get("type", "")).strip().lower()
            if cmd_type == "calibrate_boresight":
                pending_calibration = True
                print("[DSP CAL] boresight calibration requested; waiting for next processed batch.")

    while True:
        _poll_dsp_commands()
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
        proc_seqs: list[int] = []
        bad_slots = []
        for seq, s, ok in ready:
            if ok == 1:
                proc_slots.append(int(s))
                proc_seqs.append(int(seq))
            else:
                bad_slots.append(int(s))
        if bad_slots:
            _release_slots_dsp(bad_slots)
        if not proc_slots:
            continue

        n_proc = min(len(proc_slots), dsp_cfg.x_frames)
        slots_to_process = proc_slots[:n_proc]
        proc_seqs = proc_seqs[:n_proc]

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
        _release_slots_dsp(slots_to_process)

        t0_proc = time.perf_counter()
        try:
            with norm_to_peak.get_lock():
                normalize_to_peak = bool(norm_to_peak.value)
        except Exception:
            normalize_to_peak = True
        apply_display_loop_average_after_background = bool(
            display_post_range_fft_filters.loop_average_after_background.enabled
        )
        if (
            display_post_range_fft_filters.slow_time.enabled
            and display_post_range_fft_filters.slow_time.mode == "doppler_fft"
            and apply_display_loop_average_after_background
        ):
            if not warned_display_loop_average_after_doppler:
                print("[DSP WARN] display_filters.loop_average_after_background skipped because display_filters.slow_time.mode=doppler_fft.")
                warned_display_loop_average_after_doppler = True
            apply_display_loop_average_after_background = False
        apply_detection_loop_average_after_background = bool(
            detection_static_post_range_fft_filters.loop_average_after_background.enabled
        )
        if (
            detection_static_post_range_fft_filters.slow_time.enabled
            and detection_static_post_range_fft_filters.slow_time.mode == "doppler_fft"
            and apply_detection_loop_average_after_background
        ):
            if not warned_detection_loop_average_after_doppler:
                print("[DSP WARN] detection_static_filters.loop_average_after_background skipped because detection_static_filters.slow_time.mode=doppler_fft.")
                warned_detection_loop_average_after_doppler = True
            apply_detection_loop_average_after_background = False
        requested_calibration = bool(pending_calibration)
        heatmap_ema, tracking_detections, cal_vector = process_buffer(
            complex_view,
            n_proc,
            window_range,
            window_doppler,
            window_angle,
            apply_range_window,
            apply_doppler_window,
            apply_angle_window,
            mean_before_range_fft,
            detection_static_post_range_fft_filters,
            detection_moving_pre_doppler_filters,
            display_post_range_fft_filters,
            apply_detection_loop_average_after_background,
            apply_display_loop_average_after_background,
            angle_processing,
            heatmap_ema_cfg,
            heatmap_spatial_filter_cfg,
            display_projection_cfg,
            virtual_array_geometry,
            angle_steering,
            angle_axis_deg,
            display_projection_lut,
            display_y_max_m,
            display_x_max_m,
            doppler_axis_mps,
            tdm_mimo_compensation_table,
            detection_static_cfg,
            detection_moving_cfg,
            fusion_cfg,
            detection_static_bg_state,
            detection_moving_bg_state,
            display_bg_state,
            heatmap_ema,
            calibration_enabled,
            cal_vector,
            requested_calibration,
            virtual_array_work_buf[:n_proc, :, :, :, :],
            virtual_array_flat_work_buf[:n_proc, :, :, :],
            doppler_work_buf[:n_proc, :, :, :, :],
            profiles_db_work_buf,
            heatmap_db_work_buf,
            gui_h,
            gui_w,
            gui_heat_views,
            gui_profile_views,
            gui_latest_idx,
            gui_latest_seq,
            gui_lock,
            normalize_to_peak,
            profiles_out_buf,
            stat_raw_min_db,
            stat_raw_max_db,
            stat_norm_min_db,
            stat_norm_max_db,
            dsp_cfg,
            processing_max_bin,
            display_max_bin,
            debug_print_top_peaks,
            debug_top_peaks_count,
            debug_top_peaks_range_min_m,
            debug_top_peaks_range_max_m,
            debug_top_peaks_angle_min_deg,
            debug_top_peaks_angle_max_deg,
        )
        if requested_calibration:
            pending_calibration = False
        if tracker is not None and tracking_runtime_enabled:
            tracker_timestamp_s: float | None
            if tracker_nominal_frame_dt_s is not None and proc_seqs:
                latest_proc_seq = int(proc_seqs[-1])
                if last_tracker_seq is None:
                    tracker_time_s = float(latest_proc_seq) * float(tracker_nominal_frame_dt_s)
                else:
                    seq_delta = int(latest_proc_seq) - int(last_tracker_seq)
                    if seq_delta <= 0:
                        seq_delta = 1
                    tracker_time_s = (
                        float(tracker_time_s) if tracker_time_s is not None else float(last_tracker_seq) * float(tracker_nominal_frame_dt_s)
                    ) + (float(seq_delta) * float(tracker_nominal_frame_dt_s))
                last_tracker_seq = latest_proc_seq
                tracker_timestamp_s = tracker_time_s
            else:
                tracker_timestamp_s = time.perf_counter()
            active_tracks = tracker.step(tracking_detections, timestamp_s=tracker_timestamp_s)
        else:
            active_tracks = []
        _publish_tracks_latest_wins(active_tracks)
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
