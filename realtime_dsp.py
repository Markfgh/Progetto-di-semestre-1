"""Pipeline DSP realtime per acquisizioni radar FMCW MIMO.

Il modulo converte frame IQ in mappe range-angolo/range-Doppler, estrae e
fonde le detection e prepara le immagini per la GUI.  Le funzioni pure sono
separate dal worker multiprocessing, che coordina memoria condivisa e comandi.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field, replace
import io
import math
import os
import platform
import queue as pyqueue
import sys
import time
from multiprocessing.sharedctypes import Synchronized
from typing import Any, Literal


import numpy as np
import pyfftw
import scipy.fft as fft

try:
    import numba as _numba
except Exception as _numba_import_exc:  # pragma: no cover - depends on optional local install
    _numba = None
    _NUMBA_IMPORT_ERROR = _numba_import_exc
else:  # pragma: no cover - exercised only when numba is installed
    _NUMBA_IMPORT_ERROR = None

from multi_object_tracker import MultiObjectTracker, Track, TrackerConfig, TrackingConfig

# Diciamo a SciPy di usare il motore di FFTW sotto il cofano
pyfftw.interfaces.cache.enable()
fft.set_global_backend(pyfftw.interfaces.scipy_fft)
_PYFFTW_BACKEND_ACTIVE = True

# Realtime DSP only: window setup, batch processing, and worker loop.
WindowType = Literal["none", "rectangular", "hanning", "hamming", "blackman"]
_VALID_WINDOWS = {"none", "rectangular", "hanning", "hamming", "blackman"}

MeanAxis = Literal["frame", "loop", "tx", "sample", "range_bin", "rx"]
BackgroundMode = Literal["ema", "running_mean", "window_mean", "frozen"]
AngleProcessingMode = Literal["fft", "bartlett", "mvdr"]
HeatmapSpatialFilterMode = Literal["none", "gaussian_3x3"]
DisplayProjectionMode = Literal["polar_stretched", "cartesian"]
DisplayProjectionInterp = Literal["nearest", "bilinear"]
DisplayHeatmapMode = Literal["power_xy", "range_angle_moving"]
DisplayZoomFallbackMode = Literal["baseline_projection", "cached_frame"]
SlowTimeMode = Literal["none", "mean_subtraction", "highpass", "doppler_fft"]
DetectionThresholdMode = Literal["relative", "absolute", "ca_cfar", "os_cfar"]
DetectionSource = Literal["static", "moving", "fused"]
_VALID_MEAN_AXES = {"frame", "loop", "tx", "sample", "range_bin", "rx"}
_VALID_BACKGROUND_MODES = {"ema", "running_mean", "window_mean", "frozen"}
_VALID_ANGLE_PROCESSING_MODES = {"fft", "bartlett", "mvdr"}
_VALID_HEATMAP_SPATIAL_FILTER_MODES = {"none", "gaussian_3x3"}
_VALID_DISPLAY_PROJECTION_MODES = {"polar_stretched", "cartesian"}
_VALID_DISPLAY_PROJECTION_INTERPS = {"nearest", "bilinear"}
_VALID_DISPLAY_HEATMAP_MODES = {"power_xy", "range_angle_moving"}
_VALID_DISPLAY_ZOOM_FALLBACK_MODES = {"baseline_projection", "cached_frame"}
_VALID_SLOW_TIME_MODES = {"none", "mean_subtraction", "highpass", "doppler_fft"}
_VALID_THRESHOLD_MODES = {"relative", "absolute", "ca_cfar", "os_cfar"}
_DISPLAY_ZOOM_OVER_BUDGET_RETRY_S = 0.25
_GAUSSIAN_3X3_KERNEL = (
    np.array(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        dtype=np.float32,
    )
    / np.float32(16.0)
)

# Invariante della pipeline: [frame, loop, tx, sample/range_bin, rx]. L'asse
# 3 non cambia posizione dopo la Range FFT, quindi sample e range_bin condividono
# lo stesso indice nella configurazione dei filtri.
_MEAN_AXIS_INDEX = {
    "frame": 0,
    "loop": 1,
    "tx": 2,
    "sample": 3,
    "range_bin": 3,
    "rx": 4,
}

_CFAR_NUMBA_ENABLED = False
_CFAR_NUMBA_SELF_CHECKED = False
_CFAR_NUMBA_DISABLED_REASON = "not configured"
_CFAR_NUMBA_LAST_ERROR = ""
# The angle-power reduction is called for every realtime frame.  Unlike the
# optional CFAR JIT, this kernel uses ``prange`` and therefore needs an
# explicit runtime limit: otherwise Numba defaults to every logical CPU.
_ANGLE_POWER_NUMBA_ENABLED = True
_ANGLE_POWER_NUMBA_THREADS = 0
_ANGLE_POWER_NUMBA_LAST_ERROR = ""
_DSP_RUNTIME_DIAGNOSTICS_LOGGED = False


if _numba is not None:

    @_numba.njit(cache=True, fastmath=False)
    def _ca_cfar_threshold_db_map_numba_kernel(
        power,
        threshold_map,
        train_row,
        guard_row,
        train_col,
        guard_col,
        offset_lin,
        min_lin,
    ):
        n_rows = power.shape[0]
        n_cols = power.shape[1]
        row_margin = train_row + guard_row
        col_margin = train_col + guard_col
        r_guard0_off = train_row
        r_guard1_off = train_row + (2 * guard_row) + 1
        c_guard0_off = train_col
        c_guard1_off = train_col + (2 * guard_col) + 1

        for row in range(row_margin, n_rows - row_margin):
            r0 = row - row_margin
            for col in range(col_margin, n_cols - col_margin):
                c0 = col - col_margin
                total = np.float32(0.0)
                count = 0
                win_rows = (2 * row_margin) + 1
                win_cols = (2 * col_margin) + 1
                for wr in range(win_rows):
                    in_guard_row = r_guard0_off <= wr < r_guard1_off
                    rr = r0 + wr
                    for wc in range(win_cols):
                        if in_guard_row and c_guard0_off <= wc < c_guard1_off:
                            continue
                        value = power[rr, c0 + wc]
                        if np.isfinite(value):
                            total += value
                            count += 1
                if count <= 0:
                    continue

                noise_lin = np.float32(total / np.float32(count))
                threshold_lin = float(min_lin)
                candidate_lin = float(noise_lin) * float(offset_lin)
                if candidate_lin > threshold_lin:
                    threshold_lin = candidate_lin

                threshold_for_log = threshold_lin
                if threshold_for_log == threshold_for_log and threshold_for_log < 1e-12:
                    threshold_for_log = 1e-12
                threshold_map[row, col] = np.float32(10.0 * math.log10(threshold_for_log))


    @_numba.njit(cache=True, fastmath=False, parallel=True)
    def _angle_power_frame_loop_numba_kernel(angle_fft, heatmap):
        """Collapse angle-FFT power over frame/loop without a full power cube."""
        n_frames = angle_fft.shape[0]
        n_loops = angle_fft.shape[1]
        n_ranges = angle_fft.shape[2]
        n_angles = angle_fft.shape[3]
        sample_count = max(1, n_frames * n_loops)
        scale = np.float32(1.0) / np.float32(sample_count)

        for range_bin in _numba.prange(n_ranges):
            for angle_bin in range(n_angles):
                total = np.float32(0.0)
                for frame_idx in range(n_frames):
                    for loop_idx in range(n_loops):
                        value = angle_fft[frame_idx, loop_idx, range_bin, angle_bin]
                        total += (value.real * value.real) + (value.imag * value.imag)
                heatmap[range_bin, angle_bin] = total * scale

else:
    _ca_cfar_threshold_db_map_numba_kernel = None
    _angle_power_frame_loop_numba_kernel = None


def warmup_angle_power_numba() -> bool:
    """Compile the angle-power reduction before the first realtime frame."""
    if not _ANGLE_POWER_NUMBA_ENABLED or _angle_power_frame_loop_numba_kernel is None:
        return False
    angle_fft = np.zeros((1, 1, 1, 1), dtype=np.complex64)
    heatmap = np.empty((1, 1), dtype=np.float32)
    _angle_power_frame_loop_numba_kernel(angle_fft, heatmap)
    return True


# La configurazione è passata come struct al processo DSP: con ``spawn`` non
# bisogna affidarsi a globali inizializzate soltanto nel processo della GUI.
@dataclass(frozen=True)
class DspSelection:
    """Finestre applicate alle tre dimensioni FFT della pipeline."""

    window_range: WindowType = "blackman"
    window_doppler: WindowType = "hanning"
    window_angle: WindowType = "hanning"


@dataclass(frozen=True)
class CfarNumbaConfig:
    enabled: bool = False
    warmup_on_start: bool = False
    self_check_on_start: bool = False


@dataclass(frozen=True)
class AnglePowerNumbaConfig:
    """Runtime policy for the parallel angle-power reduction.

    ``threads=0`` retains Numba's runtime default.  A positive value caps the
    Numba worker pool for the DSP process, which prevents this small,
    per-frame reduction from monopolising a CPU shared with FFTW and the UI.
    """

    enabled: bool = True
    threads: int = 0


@dataclass(frozen=True)
class DspDiagnosticsConfig:
    log_cpu_runtime: bool = False


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
    """Memoria mutabile del modello di sfondo EMA o a finestra mobile."""

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
class DisplayImageResolution:
    """Fixed raster size used by one rendered image stream."""

    width: int = 128
    height: int = 128


@dataclass(frozen=True)
class DisplayImageResolutions:
    """Independent fixed output grids for realtime and offline images."""

    realtime: DisplayImageResolution = field(default_factory=DisplayImageResolution)
    offline: DisplayImageResolution = field(default_factory=DisplayImageResolution)


@dataclass(frozen=True)
class DisplayZoomConfig:
    enabled: bool = False
    max_zoom_nfft_range: int = 0
    max_zoom_nfft_angle: int = 0
    max_update_hz: float = 15.0
    dsp_budget_ms: float = 6.0
    fallback_mode: DisplayZoomFallbackMode = "baseline_projection"


@dataclass(frozen=True)
class DisplayViewport:
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    range_min_bin_f: float
    range_max_bin_f: float
    angle_min_deg: float
    angle_max_deg: float
    doppler_min_mps: float | None = None
    doppler_max_mps: float | None = None
    zoom_level: float = 1.0
    seq: int = 0


@dataclass(frozen=True)
class AppliedViewportMeta:
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    range_min_bin_f: float
    range_max_bin_f: float
    angle_min_deg: float
    angle_max_deg: float
    doppler_min_mps: float | None = None
    doppler_max_mps: float | None = None
    zoom_level: float = 1.0
    seq: int = 0
    fallback_used: bool = False
    frame_seq: int = 0


@dataclass
class DisplayZoomRuntime:
    home_viewport: DisplayViewport
    lut_cache: dict[tuple[Any, ...], dict[str, Any]] = field(default_factory=dict)
    steering_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    display_bg_state: BackgroundSubtractionState = field(default_factory=BackgroundSubtractionState)
    moving_bg_state: BackgroundSubtractionState = field(default_factory=BackgroundSubtractionState)
    display_bg_state_cache: dict[tuple[int, ...], BackgroundSubtractionState] = field(default_factory=dict)
    moving_bg_state_cache: dict[tuple[int, ...], BackgroundSubtractionState] = field(default_factory=dict)
    heatmap_ema: np.ndarray | None = None
    last_view_db: np.ndarray | None = None
    last_view_alpha: np.ndarray | None = None
    last_fill_value: float = -120.0
    last_applied_meta: AppliedViewportMeta | None = None
    last_viewport_signature: tuple[Any, ...] | None = None
    last_compute_ms: float = 0.0
    last_compute_t_s: float = 0.0
    last_mode: str = "power_xy"
    last_error_signature: tuple[str, str] | None = None


@dataclass(frozen=True)
class SlowTimeConfig:
    enabled: bool = False
    mode: SlowTimeMode = "none"
    highpass_beta: float = 0.9
    doppler_fft_shift: bool = True
    doppler_zero_notch: bool = False


@dataclass(frozen=True)
class PostRangeFftFilterConfig:
    """Filtri di una branca applicati dopo la range FFT.

    Le branche static, moving e display possono usare configurazioni diverse:
    un filtro estetico della GUI non deve cambiare le detection.
    """

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
    cfar_train_range_bins: int = 8
    cfar_guard_range_bins: int = 2
    cfar_train_col_bins: int = 8
    cfar_guard_col_bins: int = 2
    cfar_threshold_db: float = 12.0
    os_cfar_rank: int = 0


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
    cfar_train_range_bins: int = 8
    cfar_guard_range_bins: int = 2
    cfar_train_col_bins: int = 4
    cfar_guard_col_bins: int = 1
    cfar_threshold_db: float = 12.0
    os_cfar_rank: int = 0


@dataclass(frozen=True)
class FusionConfig:
    enabled: bool = True
    merge_xy_m: float = 0.40
    merge_range_m: float = 0.30
    merge_angle_deg: float = 5.0
    prefer_moving_when_doppler_valid: bool = True


@dataclass
class Detection:
    """Picco radar espresso sia in bin sia in coordinate fisiche.

    ``source`` identifica la branca che l'ha prodotto: ``static`` oppure
    ``moving``; dopo la fusione può anche conservare il Doppler della branca
    mobile insieme all'angolo della branca statica.
    """

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
class RangeAngleMovingConfig:
    relative_power_floor_db: float = -12.0
    min_power_db: float = 6.0
    min_dominance_ratio: float = 0.65
    velocity_dead_zone: float = 0.08
    min_opacity: float = 0.35


@dataclass(frozen=True)
class RealtimeDSPConfig:
    """Parametri geometrici e dimensionali immutabili di una sessione realtime."""

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
    normalize_skip_range_bins: int = 0
    zero_after_range_fft_bins: int = 0
    range_angle_moving: RangeAngleMovingConfig = field(default_factory=RangeAngleMovingConfig)
    display_zoom: DisplayZoomConfig = field(default_factory=DisplayZoomConfig)


@dataclass(frozen=True)
class VirtualArrayGeometry:
    """Ordine e centri di fase dell'array virtuale TDM-MIMO, in lunghezze d'onda."""

    order_flat: np.ndarray
    phase_centers_lambda: np.ndarray
    identity_order: bool
    uniform_half_lambda: bool
    uniform_spacing_lambda: float | None
    angle_axis_sign: float = 1.0
    angle_u_to_sin_scale: float = 2.0

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
    """Crea finestre ``float32`` già sagomate per il broadcasting sugli array IQ."""
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


def _mean_selection_from_yaml_dict(dsp: dict[str, Any], key: str, default_axes: tuple[MeanAxis, ...]) -> MeanSelection:
    block = dsp.get(key, {}) or {}
    return MeanSelection(
        axes=mean_axes_normalize(block.get("axes"), default_axes),
        enabled=bool(block.get("enabled", False)),
    )


def mean_before_range_fft_from_yaml_dict(cfg: dict[str, Any]) -> MeanSelection:
    dsp = cfg.get("dsp", {}) or {}
    return _mean_selection_from_yaml_dict(dsp, "mean_before_range_fft", ("tx",))


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


def display_zoom_from_yaml_dict(cfg: dict[str, Any]) -> DisplayZoomConfig:
    block = cfg.get("display_zoom", {}) or {}
    fft_cfg = cfg.get("fft", {}) or {}

    base_nfft_range = max(1, _to_int(fft_cfg.get("nfft_range", 128), 128))
    base_nfft_angle = max(1, _to_int(fft_cfg.get("nfft_angle", 128), 128))

    max_zoom_nfft_range = max(base_nfft_range, _to_int(block.get("max_zoom_nfft_range", base_nfft_range), base_nfft_range))
    max_zoom_nfft_angle = max(base_nfft_angle, _to_int(block.get("max_zoom_nfft_angle", base_nfft_angle), base_nfft_angle))

    max_update_hz = _to_float(block.get("max_update_hz", 15.0), 15.0)
    if not np.isfinite(max_update_hz) or max_update_hz < 0.0:
        max_update_hz = 15.0
    dsp_budget_ms = _to_float(block.get("dsp_budget_ms", 6.0), 6.0)
    if not np.isfinite(dsp_budget_ms) or dsp_budget_ms < 0.0:
        dsp_budget_ms = 6.0

    fallback_mode_raw = str(block.get("fallback_mode", "baseline_projection") or "baseline_projection").strip().lower()
    fallback_mode: DisplayZoomFallbackMode = "baseline_projection"
    if fallback_mode_raw in _VALID_DISPLAY_ZOOM_FALLBACK_MODES and fallback_mode_raw == "cached_frame":
        fallback_mode = "cached_frame"

    return DisplayZoomConfig(
        enabled=_to_bool(block.get("enabled", False), False),
        max_zoom_nfft_range=int(max_zoom_nfft_range),
        max_zoom_nfft_angle=int(max_zoom_nfft_angle),
        max_update_hz=float(max_update_hz),
        dsp_budget_ms=float(dsp_budget_ms),
        fallback_mode=fallback_mode,
    )


def display_image_resolutions_from_yaml_dict(cfg: dict[str, Any]) -> DisplayImageResolutions:
    """Read fixed image grids without coupling them to either FFT size."""
    default_resolution = DisplayImageResolution()
    display_cfg = cfg.get("display", {}) or {}
    root = display_cfg.get("image_resolution", {}) if isinstance(display_cfg, dict) else {}
    root = root if isinstance(root, dict) else {}

    def _parse_stream(name: str) -> DisplayImageResolution:
        block = root.get(name, {})
        block = block if isinstance(block, dict) else {}
        return DisplayImageResolution(
            width=max(1, _to_int(block.get("width", default_resolution.width), default_resolution.width)),
            height=max(1, _to_int(block.get("height", default_resolution.height), default_resolution.height)),
        )

    return DisplayImageResolutions(
        realtime=_parse_stream("realtime"),
        offline=_parse_stream("offline"),
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


def display_post_range_fft_filters_from_yaml_dict(cfg: dict[str, Any]) -> PostRangeFftFilterConfig:
    dsp = cfg.get("dsp", {}) or {}
    branch = dsp.get("display_filters", {}) or {}
    return PostRangeFftFilterConfig(
        mean_after_range_fft=_mean_selection_from_yaml_dict(
            {"mean_after_range_fft": branch.get("mean_after_range_fft", {})},
            "mean_after_range_fft",
            ("tx",),
        ),
        slow_time=_slow_time_from_block(branch.get("slow_time", {}) or {}),
        background_subtraction=_background_subtraction_from_block(branch.get("background_subtraction", {}) or {}),
        loop_average_after_background=_loop_average_after_background_from_block(branch.get("loop_average_after_background", {}) or {}),
    )


def detection_static_post_range_fft_filters_from_yaml_dict(cfg: dict[str, Any]) -> PostRangeFftFilterConfig:
    dsp = cfg.get("dsp", {}) or {}
    branch = dsp.get("detection_static_filters", {}) or {}
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


def range_angle_moving_from_yaml_dict(cfg: dict[str, Any]) -> RangeAngleMovingConfig:
    dsp = cfg.get("dsp", {}) or {}
    block = dsp.get("range_angle_moving", {}) or {}
    defaults = RangeAngleMovingConfig()

    relative_power_floor_db = _to_float(
        block.get("relative_power_floor_db", defaults.relative_power_floor_db),
        defaults.relative_power_floor_db,
    )
    if not np.isfinite(relative_power_floor_db):
        relative_power_floor_db = defaults.relative_power_floor_db

    min_power_db = _to_float(
        block.get("min_power_db", defaults.min_power_db),
        defaults.min_power_db,
    )
    if not np.isfinite(min_power_db):
        min_power_db = defaults.min_power_db

    min_dominance_ratio = _to_float(
        block.get("min_dominance_ratio", defaults.min_dominance_ratio),
        defaults.min_dominance_ratio,
    )
    if not np.isfinite(min_dominance_ratio):
        min_dominance_ratio = defaults.min_dominance_ratio
    min_dominance_ratio = min(1.0, max(0.0, float(min_dominance_ratio)))

    velocity_dead_zone = _to_float(block.get("velocity_dead_zone", defaults.velocity_dead_zone), defaults.velocity_dead_zone)
    if not np.isfinite(velocity_dead_zone):
        velocity_dead_zone = defaults.velocity_dead_zone
    velocity_dead_zone = min(0.99, max(0.0, float(velocity_dead_zone)))

    min_opacity = _to_float(block.get("min_opacity", defaults.min_opacity), defaults.min_opacity)
    if not np.isfinite(min_opacity):
        min_opacity = defaults.min_opacity
    min_opacity = min(1.0, max(0.0, float(min_opacity)))

    return RangeAngleMovingConfig(
        relative_power_floor_db=float(relative_power_floor_db),
        min_power_db=float(min_power_db),
        min_dominance_ratio=float(min_dominance_ratio),
        velocity_dead_zone=float(velocity_dead_zone),
        min_opacity=float(min_opacity),
    )


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return bool(default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
    return bool(value)


def _deep_merge_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out.get(key, {}) or {}, value)
        else:
            out[key] = value
    return out


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


def cfar_numba_from_yaml_dict(cfg: dict[str, Any]) -> CfarNumbaConfig:
    dsp = cfg.get("dsp", {}) or {}
    block = dsp.get("cfar_numba", {}) or {}
    if not isinstance(block, dict):
        block = {}
    return CfarNumbaConfig(
        enabled=_to_bool(block.get("enabled", False), False),
        warmup_on_start=_to_bool(block.get("warmup_on_start", False), False),
        self_check_on_start=_to_bool(block.get("self_check_on_start", False), False),
    )


def angle_power_numba_from_yaml_dict(cfg: dict[str, Any]) -> AnglePowerNumbaConfig:
    dsp = cfg.get("dsp", {}) or {}
    block = dsp.get("angle_power_numba", {}) or {}
    if not isinstance(block, dict):
        block = {}
    return AnglePowerNumbaConfig(
        enabled=_to_bool(block.get("enabled", True), True),
        threads=max(0, _to_int(block.get("threads", 0), 0)),
    )


def dsp_diagnostics_from_yaml_dict(cfg: dict[str, Any]) -> DspDiagnosticsConfig:
    dsp = cfg.get("dsp", {}) or {}
    block = dsp.get("diagnostics", {}) or {}
    if not isinstance(block, dict):
        block = {}
    return DspDiagnosticsConfig(
        log_cpu_runtime=_to_bool(block.get("log_cpu_runtime", False), False),
    )


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
    return (np.float32(0.25) * np.arange(max(0, int(virtual_ant)), dtype=np.float32)).astype(np.float32, copy=False)


def _default_angle_u_to_sin_scale() -> float:
    return 2.0


def _resolve_angle_u_to_sin_scale(geometry: VirtualArrayGeometry | None = None) -> float:
    scale = float(_default_angle_u_to_sin_scale())
    if geometry is not None:
        try:
            scale = float(getattr(geometry, "angle_u_to_sin_scale", scale))
        except (TypeError, ValueError):
            scale = float(_default_angle_u_to_sin_scale())
    if not np.isfinite(scale) or abs(scale) <= 1e-8:
        scale = float(_default_angle_u_to_sin_scale())
    return abs(scale)


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
    # Riordina i canali acquisiti TX-major/RX-minor nell'ordine fisico
    # dell'apertura; i phase center devono seguire lo stesso ordine di uscita.
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
                f"antenna.virtual_array_order non valido ({exc}); using default tx-major/rx order."
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
                f"antenna.virtual_array_phase_centers_* non valido ({exc}); using default bistatic phase centers."
            )
            phase_centers_lambda = default_phase_centers

    angle_axis_sign = 1.0
    angle_sign_raw = block.get("angle_axis_sign", block.get("angle_sign", 1))
    try:
        angle_axis_sign = float(angle_sign_raw)
        if not np.isfinite(angle_axis_sign) or angle_axis_sign == 0.0:
            raise ValueError("deve essere +/-1")
        angle_axis_sign = 1.0 if angle_axis_sign > 0.0 else -1.0
    except Exception as exc:
        warnings.append(
            f"antenna.angle_axis_sign non valido ({exc}); using +1."
        )
        angle_axis_sign = 1.0

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

    if (
        int(dsp_cfg.tx) == 2
        and int(dsp_cfg.rx) == 4
        and uniform_spacing_lambda is not None
        and not np.isclose(float(uniform_spacing_lambda), 0.25, atol=1e-3, rtol=0.0)
    ):
        warnings.append(
            "antenna.virtual_array_phase_centers_* uses spacing "
            f"{float(uniform_spacing_lambda):.3f} lambda; "
            "bistatic phase-center xWR14xx 2Tx/4Rx azimuth uses a quarter-lambda virtual ULA. "
            "Angle estimates may be distorted."
        )

    geometry = VirtualArrayGeometry(
        order_flat=order_flat,
        phase_centers_lambda=phase_centers_lambda.astype(np.float32, copy=False),
        identity_order=identity_order,
        uniform_half_lambda=uniform_half_lambda,
        uniform_spacing_lambda=uniform_spacing_lambda,
        angle_axis_sign=float(angle_axis_sign),
        angle_u_to_sin_scale=float(_default_angle_u_to_sin_scale()),
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


def display_viewport_signature(viewport: DisplayViewport | AppliedViewportMeta | None) -> tuple[Any, ...] | None:
    if viewport is None:
        return None
    doppler_min = None if viewport.doppler_min_mps is None else round(float(viewport.doppler_min_mps), 6)
    doppler_max = None if viewport.doppler_max_mps is None else round(float(viewport.doppler_max_mps), 6)
    return (
        round(float(viewport.x_min_m), 6),
        round(float(viewport.x_max_m), 6),
        round(float(viewport.y_min_m), 6),
        round(float(viewport.y_max_m), 6),
        round(float(viewport.range_min_bin_f), 6),
        round(float(viewport.range_max_bin_f), 6),
        round(float(viewport.angle_min_deg), 6),
        round(float(viewport.angle_max_deg), 6),
        doppler_min,
        doppler_max,
    )


def build_display_viewport(
    *,
    x_min_m: float,
    x_max_m: float,
    y_min_m: float,
    y_max_m: float,
    dr_m: float,
    seq: int = 0,
    home_viewport: DisplayViewport | None = None,
    doppler_min_mps: float | None = None,
    doppler_max_mps: float | None = None,
) -> DisplayViewport:
    x0 = float(min(x_min_m, x_max_m))
    x1 = float(max(x_min_m, x_max_m))
    y0 = max(0.0, float(min(y_min_m, y_max_m)))
    y1 = max(y0, float(max(y_min_m, y_max_m)))
    dr_f = float(dr_m)
    if not np.isfinite(dr_f) or dr_f <= 0.0:
        span_y = max(y1 - y0, 1e-3)
        dr_f = span_y / 128.0

    x_near = 0.0 if x0 <= 0.0 <= x1 else min(abs(x0), abs(x1))
    range_min_m = float(np.hypot(x_near, y0))
    corner_ranges = np.asarray(
        [
            np.hypot(x0, y0),
            np.hypot(x0, y1),
            np.hypot(x1, y0),
            np.hypot(x1, y1),
        ],
        dtype=np.float64,
    )
    finite_corner_ranges = corner_ranges[np.isfinite(corner_ranges)]
    range_max_m = float(np.max(finite_corner_ranges)) if finite_corner_ranges.size > 0 else range_min_m

    y_angle_near = max(y0, dr_f * 0.5, 1e-6)
    y_angle_far = max(y1, y_angle_near)
    angle_samples = np.asarray(
        [
            np.rad2deg(np.arctan2(x0, y_angle_near)),
            np.rad2deg(np.arctan2(x0, y_angle_far)),
            np.rad2deg(np.arctan2(x1, y_angle_near)),
            np.rad2deg(np.arctan2(x1, y_angle_far)),
        ],
        dtype=np.float64,
    )
    finite_angles = angle_samples[np.isfinite(angle_samples)]
    if finite_angles.size <= 0:
        angle_min_deg = 0.0
        angle_max_deg = 0.0
    else:
        angle_min_deg = float(np.min(finite_angles))
        angle_max_deg = float(np.max(finite_angles))
    if angle_max_deg < angle_min_deg:
        angle_min_deg, angle_max_deg = angle_max_deg, angle_min_deg

    zoom_level = 1.0
    if home_viewport is not None:
        home_w = max(1e-6, float(home_viewport.x_max_m - home_viewport.x_min_m))
        home_h = max(1e-6, float(home_viewport.y_max_m - home_viewport.y_min_m))
        cur_w = max(1e-6, float(x1 - x0))
        cur_h = max(1e-6, float(y1 - y0))
        zoom_level = max(float(home_w / cur_w), float(home_h / cur_h), 1.0)

    return DisplayViewport(
        x_min_m=float(x0),
        x_max_m=float(x1),
        y_min_m=float(y0),
        y_max_m=float(y1),
        range_min_bin_f=float(range_min_m / dr_f),
        range_max_bin_f=float(range_max_m / dr_f),
        angle_min_deg=float(angle_min_deg),
        angle_max_deg=float(angle_max_deg),
        doppler_min_mps=None if doppler_min_mps is None else float(doppler_min_mps),
        doppler_max_mps=None if doppler_max_mps is None else float(doppler_max_mps),
        zoom_level=float(zoom_level),
        seq=int(seq),
    )


def clamp_display_viewport(
    *,
    x_min_m: float,
    x_max_m: float,
    y_min_m: float,
    y_max_m: float,
    home_viewport: DisplayViewport,
    output_width: int,
    output_height: int,
    dr_m: float,
    seq: int = 0,
    doppler_min_mps: float | None = None,
    doppler_max_mps: float | None = None,
    quantize: bool = True,
) -> DisplayViewport:
    home_x0 = float(home_viewport.x_min_m)
    home_x1 = float(home_viewport.x_max_m)
    home_y0 = float(home_viewport.y_min_m)
    home_y1 = float(home_viewport.y_max_m)
    home_w = max(1e-6, home_x1 - home_x0)
    home_h = max(1e-6, home_y1 - home_y0)
    x_step = max(home_w / float(max(1, int(output_width))), 1e-6)
    y_step = max(home_h / float(max(1, int(output_height))), max(float(dr_m), 1e-6))

    x0 = max(home_x0, min(home_x1, float(min(x_min_m, x_max_m))))
    x1 = max(home_x0, min(home_x1, float(max(x_min_m, x_max_m))))
    y0 = max(home_y0, min(home_y1, float(min(y_min_m, y_max_m))))
    y1 = max(home_y0, min(home_y1, float(max(y_min_m, y_max_m))))

    if x1 <= x0:
        x1 = min(home_x1, x0 + x_step)
    if y1 <= y0:
        y1 = min(home_y1, y0 + y_step)

    def _quantize_floor(value: float, origin: float, step: float) -> float:
        return float(origin + math.floor((value - origin) / step) * step)

    def _quantize_ceil(value: float, origin: float, step: float) -> float:
        return float(origin + math.ceil((value - origin) / step) * step)

    if quantize:
        # Quantize outward so the applied viewport always covers the user's
        # requested ROI instead of shrinking it and clipping the display edges.
        x0 = _quantize_floor(x0, home_x0, x_step)
        x1 = _quantize_ceil(x1, home_x0, x_step)
        y0 = _quantize_floor(y0, home_y0, y_step)
        y1 = _quantize_ceil(y1, home_y0, y_step)

    x0 = max(home_x0, min(home_x1, x0))
    x1 = max(home_x0, min(home_x1, x1))
    y0 = max(home_y0, min(home_y1, y0))
    y1 = max(home_y0, min(home_y1, y1))
    if x1 <= x0:
        if x0 >= home_x1:
            x0 = max(home_x0, home_x1 - x_step)
            x1 = home_x1
        else:
            x1 = min(home_x1, x0 + x_step)
    if y1 <= y0:
        if y0 >= home_y1:
            y0 = max(home_y0, home_y1 - y_step)
            y1 = home_y1
        else:
            y1 = min(home_y1, y0 + y_step)

    return build_display_viewport(
        x_min_m=float(x0),
        x_max_m=float(x1),
        y_min_m=float(y0),
        y_max_m=float(y1),
        dr_m=float(dr_m),
        seq=int(seq),
        home_viewport=home_viewport,
        doppler_min_mps=doppler_min_mps,
        doppler_max_mps=doppler_max_mps,
    )


def applied_viewport_meta_from_viewport(
    viewport: DisplayViewport,
    *,
    fallback_used: bool,
    frame_seq: int,
) -> AppliedViewportMeta:
    return AppliedViewportMeta(
        x_min_m=float(viewport.x_min_m),
        x_max_m=float(viewport.x_max_m),
        y_min_m=float(viewport.y_min_m),
        y_max_m=float(viewport.y_max_m),
        range_min_bin_f=float(viewport.range_min_bin_f),
        range_max_bin_f=float(viewport.range_max_bin_f),
        angle_min_deg=float(viewport.angle_min_deg),
        angle_max_deg=float(viewport.angle_max_deg),
        doppler_min_mps=None if viewport.doppler_min_mps is None else float(viewport.doppler_min_mps),
        doppler_max_mps=None if viewport.doppler_max_mps is None else float(viewport.doppler_max_mps),
        zoom_level=float(viewport.zoom_level),
        seq=int(viewport.seq),
        fallback_used=bool(fallback_used),
        frame_seq=int(frame_seq),
    )


def update_display_zoom_runtime_home_viewport(
    runtime: DisplayZoomRuntime,
    home_viewport: DisplayViewport,
) -> bool:
    prev_sig = display_viewport_signature(runtime.home_viewport)
    next_sig = display_viewport_signature(home_viewport)
    runtime.home_viewport = home_viewport
    if prev_sig == next_sig:
        return False
    runtime.last_view_db = None
    runtime.last_view_alpha = None
    runtime.last_applied_meta = None
    runtime.last_viewport_signature = None
    runtime.last_compute_ms = 0.0
    runtime.last_compute_t_s = 0.0
    runtime.last_mode = "power_xy"
    return True


def should_recompute_display_zoom(
    runtime: DisplayZoomRuntime | None,
    *,
    active_viewport_sig: tuple[Any, ...] | None,
    display_mode: str,
    display_zoom_cfg: DisplayZoomConfig,
    now_s: float | None = None,
) -> bool:
    if runtime is None:
        return False
    min_interval_s = 0.0
    if float(display_zoom_cfg.max_update_hz) > 0.0:
        min_interval_s = 1.0 / float(display_zoom_cfg.max_update_hz)
    same_view = runtime.last_viewport_signature == active_viewport_sig
    same_mode = str(runtime.last_mode) == str(display_mode)
    if not same_view or not same_mode:
        return True
    retry_interval_s = float(min_interval_s)
    if (
        float(display_zoom_cfg.dsp_budget_ms) > 0.0
        and float(runtime.last_compute_ms) > float(display_zoom_cfg.dsp_budget_ms)
    ):
        retry_interval_s = max(retry_interval_s, _DISPLAY_ZOOM_OVER_BUDGET_RETRY_S)
    if retry_interval_s <= 0.0:
        return True
    now_value = time.perf_counter() if now_s is None else float(now_s)
    return (now_value - float(runtime.last_compute_t_s)) >= retry_interval_s


def _threshold_mode_from_value(value: Any, default: DetectionThresholdMode = "relative") -> DetectionThresholdMode:
    mode_raw = str(value or default).strip().lower()
    if mode_raw not in _VALID_THRESHOLD_MODES:
        return default
    return mode_raw  # type: ignore[return-value]


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
        cfar_train_range_bins=max(0, _to_int(block.get("cfar_train_range_bins", 8), 8)),
        cfar_guard_range_bins=max(0, _to_int(block.get("cfar_guard_range_bins", 2), 2)),
        cfar_train_col_bins=max(0, _to_int(block.get("cfar_train_col_bins", 8), 8)),
        cfar_guard_col_bins=max(0, _to_int(block.get("cfar_guard_col_bins", 2), 2)),
        cfar_threshold_db=_to_float(block.get("cfar_threshold_db", 12.0), 12.0),
        os_cfar_rank=max(0, _to_int(block.get("os_cfar_rank", 0), 0)),
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
        cfar_train_range_bins=max(0, _to_int(block.get("cfar_train_range_bins", 8), 8)),
        cfar_guard_range_bins=max(0, _to_int(block.get("cfar_guard_range_bins", 2), 2)),
        cfar_train_col_bins=max(0, _to_int(block.get("cfar_train_col_bins", 4), 4)),
        cfar_guard_col_bins=max(0, _to_int(block.get("cfar_guard_col_bins", 1), 1)),
        cfar_threshold_db=_to_float(block.get("cfar_threshold_db", 12.0), 12.0),
        os_cfar_rank=max(0, _to_int(block.get("os_cfar_rank", 0), 0)),
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
    block = cfg.get("tracking", {}) or {}
    dt_s = _to_optional_float(block.get("dt_s"))
    return TrackingConfig(
        enabled=bool(block.get("enabled", True)),
        dt_s=None if dt_s is None else max(1e-3, float(dt_s)),
        max_tracks=max(1, _to_int(block.get("max_tracks", 30), 30)),
        min_hits_to_confirm=max(
            1,
            _to_int(block.get("min_hits_to_confirm", 3), 3),
        ),
        max_missed_tentative=max(
            0,
            _to_int(block.get("max_missed_tentative", 2), 2),
        ),
        max_missed_confirmed=max(
            0,
            _to_int(block.get("max_missed_confirmed", 8), 8),
        ),
        max_track_age=max(
            0,
            _to_int(block.get("max_track_age", 0), 0),
        ),
    )


def tracker_from_yaml_dict(cfg: dict[str, Any]) -> TrackerConfig:
    block = cfg.get("tracking", {}) or {}
    model = str(block.get("model", "kalman_cv_2d") or "kalman_cv_2d").strip().lower()
    if model != "kalman_cv_2d":
        print(f"[TRACK WARN] tracking.model={model!r} unsupported; forcing 'kalman_cv_2d'.")
        model = "kalman_cv_2d"
    return TrackerConfig(
        model=model,
        gating_xy_m=max(
            0.0,
            _to_float(block.get("gating_xy_m", 0.75), 0.75),
        ),
        gating_doppler_mps=max(
            0.0,
            _to_float(block.get("gating_doppler_mps", 0.50), 0.50),
        ),
        process_noise_pos=max(
            1e-4,
            _to_float(block.get("process_noise_pos", 0.20), 0.20),
        ),
        process_noise_vel=max(
            1e-4,
            _to_float(block.get("process_noise_vel", 1.00), 1.00),
        ),
        measurement_noise_xy=max(
            1e-4,
            _to_float(block.get("measurement_noise_xy", 0.25), 0.25),
        ),
        moving_speed_threshold_mps=max(
            0.0,
            _to_float(block.get("moving_speed_threshold_mps", 0.20), 0.20),
        ),
        stopped_speed_threshold_mps=max(
            0.0,
            _to_float(block.get("stopped_speed_threshold_mps", 0.08), 0.08),
        ),
        doppler_moving_threshold_mps=max(
            0.0,
            _to_float(block.get("doppler_moving_threshold_mps", 0.12), 0.12),
        ),
        motion_confirm_frames_moving=max(
            1,
            _to_int(block.get("motion_confirm_frames_moving", 2), 2),
        ),
        motion_confirm_frames_stopped=max(
            1,
            _to_int(block.get("motion_confirm_frames_stopped", 3), 3),
        ),
        stopped_memory_s=max(
            0.0,
            _to_float(block.get("stopped_memory_s", 3.0), 3.0),
        ),
        stopped_resume_gate_m=max(
            0.0,
            _to_float(block.get("stopped_resume_gate_m", 0.90), 0.90),
        ),
        stop_position_alpha=min(
            1.0,
            max(
                0.0,
                _to_float(block.get("stop_position_alpha", 0.25), 0.25),
            ),
        ),
        birth_min_separation_m=max(
            0.0,
            _to_float(block.get("birth_min_separation_m", 0.20), 0.20),
        ),
        use_doppler_in_cost=bool(block.get("use_doppler_in_cost", True)),
        history_len=max(
            1,
            _to_int(block.get("history_len", 12), 12),
        ),
        debug_log=bool(block.get("debug_log", False)),
    )


def subtract_selected_mean(data: np.ndarray, selection: MeanSelection) -> np.ndarray:
    """Rimuove in-place la media lungo gli assi richiesti dalla configurazione."""
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
    padded = np.pad(np.asarray(heatmap, dtype=np.float32), ((1, 1), (1, 1)), mode="edge")
    out = (
        padded[:-2, :-2] * _GAUSSIAN_3X3_KERNEL[0, 0]
        + padded[:-2, 1:-1] * _GAUSSIAN_3X3_KERNEL[0, 1]
        + padded[:-2, 2:] * _GAUSSIAN_3X3_KERNEL[0, 2]
        + padded[1:-1, :-2] * _GAUSSIAN_3X3_KERNEL[1, 0]
        + padded[1:-1, 1:-1] * _GAUSSIAN_3X3_KERNEL[1, 1]
        + padded[1:-1, 2:] * _GAUSSIAN_3X3_KERNEL[1, 2]
        + padded[2:, :-2] * _GAUSSIAN_3X3_KERNEL[2, 0]
        + padded[2:, 1:-1] * _GAUSSIAN_3X3_KERNEL[2, 1]
        + padded[2:, 2:] * _GAUSSIAN_3X3_KERNEL[2, 2]
    )
    return out.astype(np.float32, copy=False)


def apply_slow_time_filter(
    data: np.ndarray,
    slow_time_cfg: SlowTimeConfig,
    *,
    fft_workers: int = 1,
) -> np.ndarray:
    """Filtra l'asse dei loop/chirp dopo la range FFT.

    I modi disponibili sopprimono il clutter statico oppure trasformano già
    l'asse lento in Doppler; il chiamante decide quale branca possa usarli.
    """
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
    """Applica la sequenza canonica di filtri di una branca DSP."""
    # Ordine intenzionale: slow-time, rimozione media, background e infine
    # media sui loop. Invertirlo cambia il clutter stimato e il cubo a valle.
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


def _build_angle_u_axis(nfft_angle: int, spacing_lambda: float = 0.25) -> np.ndarray:
    nfft_angle = max(1, int(nfft_angle))
    if nfft_angle == 1:
        return np.asarray([0.0], dtype=np.float32)
    spacing = float(spacing_lambda)
    if not np.isfinite(spacing) or abs(spacing) <= 1e-8:
        spacing = 0.25
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
    """Costruisce steering normalizzato per beamforming Bartlett/MVDR."""
    if geometry is None:
        phase_centers_lambda = _default_virtual_array_phase_centers_lambda(int(virtual_ant))
    else:
        phase_centers_lambda = np.asarray(geometry.phase_centers_lambda, dtype=np.float32).reshape(-1)
    if phase_centers_lambda.size != int(virtual_ant):
        raise ValueError(
            f"phase_centers_lambda size={phase_centers_lambda.size} != virtual_ant={int(virtual_ant)}"
        )
    spacing_lambda = 0.25
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
    ant_spacing: float | None = None,
) -> np.ndarray:
    """Riduce l'array virtuale a una mappa ``[range_bin, angle_bin]``.

    La modalità FFT è la più economica; Bartlett e MVDR usano i centri di fase
    fisici. MVDR aggiunge carico diagonale e degrada localmente a Bartlett se
    una covarianza non è risolvibile, invece di invalidare l'intera mappa.
    """
    # in [frame, loop, range_bin, angle_bin] -> [range_bin, angle_bin] con eventuale aggregazione su frame/loop a seconda di angle_cfg
    def _aggregate_angle_power(power: np.ndarray) -> np.ndarray:
        if power.ndim != 4:
            return np.asarray(power, dtype=np.float32)
        num_frames = int(power.shape[0])
        num_loops = int(power.shape[1])
        mode = str(getattr(angle_cfg, "aggregation", "frame_loop")).strip().lower()
        frame_idx = min(max(0, int(getattr(angle_cfg, "frame_index", 0))), max(0, num_frames - 1))
        loop_idx = min(max(0, int(getattr(angle_cfg, "loop_index", 0))), max(0, num_loops - 1))
        if mode == "frame":
            # Average frames while keeping a single selected loop snapshot.
            return power[:, loop_idx, :, :].mean(axis=0, dtype=np.float32)
        if mode == "loop":
            # Average loops while keeping a single selected frame snapshot.
            return power[frame_idx, :, :, :].mean(axis=0, dtype=np.float32)
        if mode == "none":
            return power[frame_idx, loop_idx, :, :].astype(np.float32, copy=False)
        return power.mean(axis=(0, 1), dtype=np.float32)

    def _build_uniform_steering(n_ant: int, n_angle: int, spacing_lambda: float) -> np.ndarray:
        spacing = float(spacing_lambda)
        if not np.isfinite(spacing) or abs(spacing) <= 1e-8:
            spacing = 0.25
        phase_centers_lambda = (
            np.arange(max(0, int(n_ant)), dtype=np.float32) * np.float32(spacing)
        ).astype(np.float32, copy=False)
        u = _build_angle_u_axis(n_angle, spacing_lambda=spacing)
        phase = (-1j * np.float32(2.0 * np.pi)) * phase_centers_lambda[:, None] * u[None, :]
        steering = np.exp(phase).astype(np.complex64, copy=False)
        col_energy = (steering.real * steering.real + steering.imag * steering.imag).sum(axis=0, dtype=np.float32)
        col_norm = np.sqrt(np.maximum(col_energy, np.float32(1e-8))).astype(np.float32, copy=False)
        steering /= col_norm[np.newaxis, :].astype(np.complex64, copy=False)
        return steering

    def _resolve_bartlett_mvdr_steering(n_ant: int, n_angle: int) -> np.ndarray:
        spacing_lambda: float | None = None
        if ant_spacing is not None:
            try:
                spacing_candidate = float(ant_spacing)
            except (TypeError, ValueError):
                spacing_candidate = float("nan")
            if np.isfinite(spacing_candidate) and abs(spacing_candidate) > 1e-8:
                spacing_lambda = spacing_candidate
        if spacing_lambda is not None:
            return _build_uniform_steering(n_ant, n_angle, spacing_lambda)

        steering = np.asarray(angle_steering, dtype=np.complex64)
        if steering.ndim == 2 and steering.shape[0] >= n_ant and steering.shape[1] >= n_angle:
            return steering[:n_ant, :n_angle]

        if geometry is not None and geometry.uniform_spacing_lambda is not None:
            return _build_uniform_steering(n_ant, n_angle, float(geometry.uniform_spacing_lambda))

        phase_centers_lambda = _default_virtual_array_phase_centers_lambda(n_ant)
        if geometry is not None:
            phase_centers_raw = np.asarray(geometry.phase_centers_lambda, dtype=np.float32).reshape(-1)
            if phase_centers_raw.size >= n_ant:
                phase_centers_lambda = phase_centers_raw[:n_ant]
        u = _build_angle_u_axis(n_angle, spacing_lambda=0.25)
        phase = (-1j * np.float32(2.0 * np.pi)) * phase_centers_lambda[:, None] * u[None, :]
        steering = np.exp(phase).astype(np.complex64, copy=False)
        col_energy = (steering.real * steering.real + steering.imag * steering.imag).sum(axis=0, dtype=np.float32)
        col_norm = np.sqrt(np.maximum(col_energy, np.float32(1e-8))).astype(np.float32, copy=False)
        steering /= col_norm[np.newaxis, :].astype(np.complex64, copy=False)
        return steering

    def _resolve_frame_index(num_frames: int) -> int:
        if num_frames <= 0:
            return 0
        return min(max(0, int(getattr(angle_cfg, "frame_index", 0))), num_frames - 1)

    # calculate angle heatmap with FFT; output is [range_bin, angle_bin]
    if angle_cfg.mode == "fft":
        # Steering/Bartlett/MVDR use the physical snapshot convention
        # exp(-j 2*pi*x*u). Use the matching inverse-DFT sign here so the
        # FFT branch reports the same positive angle as the beamformers.
        angle_fft = fft.ifft(
            virtual_array,
            n=dsp_cfg.nfft_angle,
            axis=3,
            workers=dsp_cfg.fft_workers,
            overwrite_x=True,
        )
        angle_fft *= np.float32(max(1, int(dsp_cfg.nfft_angle)))
        spacing_lambda = 0.25
        if geometry is not None and geometry.uniform_spacing_lambda is not None:
            spacing_lambda = float(geometry.uniform_spacing_lambda)
        valid_u = np.abs(
            _build_angle_u_axis(dsp_cfg.nfft_angle, spacing_lambda=spacing_lambda).astype(np.float64, copy=False)
        ) <= float(_resolve_angle_u_to_sin_scale(geometry))
        # The common frame_loop path uses a compiled reduction that writes the
        # final [range, angle] map directly, avoiding the large power cube.
        # Other aggregation modes keep the NumPy implementation unchanged.
        if (
            _ANGLE_POWER_NUMBA_ENABLED
            and _angle_power_frame_loop_numba_kernel is not None
            and angle_fft.ndim == 4
            and angle_fft.flags.c_contiguous
            and str(getattr(angle_cfg, "aggregation", "frame_loop")).strip().lower() == "frame_loop"
        ):
            heatmap = np.empty((angle_fft.shape[2], angle_fft.shape[3]), dtype=np.float32)
            _angle_power_frame_loop_numba_kernel(angle_fft, heatmap)
        else:
            # power calculation
            re = angle_fft.real
            im = angle_fft.imag
            power = (re * re + im * im).astype(np.float32, copy=False)
            # Aggregate before shifting: fftshift acts only on the angle axis,
            # so it commutes with frame/loop aggregation while avoiding a
            # large full-cube copy on every heatmap.
            heatmap = _aggregate_angle_power(power)
        heatmap = np.fft.fftshift(heatmap, axes=-1)
        if np.any(~valid_u):
            heatmap[..., ~valid_u] = np.float32(0.0)
        return heatmap.astype(np.float32, copy=False)

    if (
        virtual_array.ndim != 4
        or int(virtual_array.shape[0]) <= 0
        or int(virtual_array.shape[1]) <= 0
        or int(virtual_array.shape[2]) <= 0
        or int(virtual_array.shape[3]) <= 0
    ):
        return np.empty((0, 0), dtype=np.float32)
    num_frames = int(virtual_array.shape[0])
    num_loops = int(virtual_array.shape[1])
    num_range = int(virtual_array.shape[2])
    num_ant = int(virtual_array.shape[3])

    if angle_cfg.mode == "bartlett":
        steering = _resolve_bartlett_mvdr_steering(
            num_ant,
            int(dsp_cfg.nfft_angle),
        ).astype(np.complex64, copy=False)
        snapshots = np.asarray(virtual_array, dtype=np.complex64)
        # Match the FFT path: beamform each snapshot first, then integrate the power incoherently.
        beam = np.einsum("ak,flra->flrk", np.conj(steering), snapshots, optimize=True)
        power = (beam.real * beam.real + beam.imag * beam.imag).astype(np.float32, copy=False)
        heatmap = _aggregate_angle_power(power)
        # Steering columns are unit norm, while the FFT branch retains the
        # coherent array gain. Preserve the established FFT dB calibration.
        heatmap *= np.float32(max(1, num_ant))
        return heatmap.astype(np.float32, copy=False)

    # MVDR always uses every available frame/loop snapshot in the batch to estimate covariance.
    snapshots = np.asarray(virtual_array, dtype=np.complex64).reshape(num_frames * num_loops, num_range, num_ant)
    num_snapshots = int(snapshots.shape[0])
    range_fft_data = snapshots.transpose(0, 2, 1).astype(np.complex128, copy=False)  # [snapshot, ant, range]
    steering = _resolve_bartlett_mvdr_steering(num_ant, int(dsp_cfg.nfft_angle)).astype(np.complex128, copy=False)

    if num_snapshots < num_ant and not getattr(compute_angle_heatmap, "_mvdr_snapshot_warned", False):
        print(
            f"[DSP WARN] MVDR requires more snapshots than antennas; got snapshots={num_snapshots}, antennas={num_ant}."
        )
        compute_angle_heatmap._mvdr_snapshot_warned = True

    x_by_range = range_fft_data.transpose(2, 1, 0).astype(np.complex128, copy=False)  # [range, ant, snapshot]
    heatmap = np.empty((num_range, int(dsp_cfg.nfft_angle)), dtype=np.float32)
    eye = np.eye(num_ant, dtype=np.complex128)
    load_factor = np.float64(max(0.0, float(getattr(angle_cfg, "mvdr_diagonal_loading", 0.0))))
    empty_trace_eps = np.float64(1e-12)
    min_empty_loading = np.float64(1e-12)
    den_floor = np.float32(1e-8)
    mvdr_fallback_bins = 0

    def _bartlett_range_power(x_range: np.ndarray) -> np.ndarray:
        beam = np.einsum("ak,as->sk", np.conj(steering), x_range, optimize=True)
        power = (beam.real * beam.real + beam.imag * beam.imag).mean(axis=0, dtype=np.float64)
        return power.astype(np.float32, copy=False)

    def _solve_loaded_covariance(cov_loaded: np.ndarray) -> np.ndarray:
        try:
            chol = np.linalg.cholesky(cov_loaded)
            y = np.linalg.solve(chol, steering)
            return np.linalg.solve(chol.conj().T, y)
        except np.linalg.LinAlgError:
            return np.linalg.solve(cov_loaded, steering)

    for range_idx in range(num_range):
        x = x_by_range[range_idx]
        cov = (x @ x.conj().T) / np.float64(max(1, num_snapshots))
        cov = np.float64(0.5) * (cov + cov.conj().T)
        trace_r = float(np.trace(cov).real)
        if not np.isfinite(trace_r):
            heatmap[range_idx, :] = _bartlett_range_power(x)
            mvdr_fallback_bins += 1
            continue

        load_diag = np.float64(0.0)
        if trace_r > 0.0:
            load_diag = load_factor * np.float64(trace_r) / np.float64(max(1, num_ant))
            # Near-empty bins need a tiny absolute floor; otherwise R can collapse to exactly zero.
            if np.float64(trace_r) <= empty_trace_eps:
                load_diag = max(load_diag, min_empty_loading)
        else:
            load_diag = min_empty_loading
        if load_diag > 0.0:
            cov = cov + (load_diag * eye)

        try:
            den_left = _solve_loaded_covariance(cov)
            den = np.einsum("ak,ak->k", np.conj(steering), den_left, optimize=True).real
        except np.linalg.LinAlgError:
            heatmap[range_idx, :] = _bartlett_range_power(x)
            mvdr_fallback_bins += 1
            continue

        if not np.all(np.isfinite(den)) or np.any(den <= 0.0):
            heatmap[range_idx, :] = _bartlett_range_power(x)
            mvdr_fallback_bins += 1
            continue

        den = den.astype(np.float32, copy=False)
        np.maximum(den, den_floor, out=den)
        heatmap[range_idx, :] = np.float32(1.0) / den
    compute_angle_heatmap._mvdr_total_bins = int(num_range)
    compute_angle_heatmap._mvdr_fallback_bins = int(mvdr_fallback_bins)
    if mvdr_fallback_bins > 0 and bool(getattr(dsp_cfg, "debug_stats", False)):
        report_key = (int(num_range), int(mvdr_fallback_bins))
        if getattr(compute_angle_heatmap, "_mvdr_fallback_last_report", None) != report_key:
            print(f"[DSP WARN] MVDR local Bartlett fallback bins={mvdr_fallback_bins}/{num_range}")
            compute_angle_heatmap._mvdr_fallback_last_report = report_key
    heatmap *= np.float32(max(1, num_ant))
    return heatmap.astype(np.float32, copy=False)


def build_angle_axis_deg(
    nfft_angle: int,
    geometry: VirtualArrayGeometry | None = None,
) -> np.ndarray:
    spacing_lambda = 0.25
    angle_axis_sign = 1.0
    if geometry is not None and geometry.uniform_spacing_lambda is not None:
        spacing_lambda = float(geometry.uniform_spacing_lambda)
    if geometry is not None:
        try:
            angle_axis_sign = float(getattr(geometry, "angle_axis_sign", 1.0))
        except (TypeError, ValueError):
            angle_axis_sign = 1.0
        if not np.isfinite(angle_axis_sign) or angle_axis_sign == 0.0:
            angle_axis_sign = 1.0
        angle_axis_sign = 1.0 if angle_axis_sign > 0.0 else -1.0
    u = _build_angle_u_axis(nfft_angle, spacing_lambda=spacing_lambda)
    sin_scale = np.float64(_resolve_angle_u_to_sin_scale(geometry))
    angle_axis = np.full(u.shape, np.float32(np.nan), dtype=np.float32)
    valid = np.abs(u.astype(np.float64, copy=False)) <= sin_scale
    if np.any(valid):
        angle_axis[valid] = np.rad2deg(
            np.arcsin(u[valid].astype(np.float64, copy=False) / sin_scale)
        ).astype(np.float32, copy=False) * np.float32(angle_axis_sign)
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


def _normalize_display_heatmap_mode(value: Any) -> DisplayHeatmapMode:
    mode_raw = str(value or "power_xy").strip().lower()
    if mode_raw not in _VALID_DISPLAY_HEATMAP_MODES:
        return "power_xy"
    return mode_raw  # type: ignore[return-value]


def _warn_velocity_display_once(message: str) -> None:
    key = str(message)
    warned = getattr(_warn_velocity_display_once, "_warned", set())
    if key in warned:
        return
    print(f"[DSP WARN] {message}")
    warned.add(key)
    _warn_velocity_display_once._warned = warned


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
    *,
    x_min_m: float | None = None,
    y_min_m: float | None = None,
) -> dict[str, Any]:
    gui_h = max(0, int(gui_h))
    gui_w = max(0, int(gui_w))
    mode = _normalize_display_projection_mode(projection_mode)
    interp = _normalize_display_projection_interp(projection_interp)
    angle_axis = np.asarray(angle_axis_deg, dtype=np.float32).reshape(-1)
    x_min_v = float(-float(x_max_m) if x_min_m is None else x_min_m)
    x_max_v = float(x_max_m)
    y_min_v = float(0.0 if y_min_m is None else y_min_m)
    y_max_v = float(y_max_m)

    lut: dict[str, Any] = {
        "projection_mode": mode,
        "projection_interp": interp,
        "output_shape": (gui_h, gui_w),
        "x_min_m": float(x_min_v),
        "x_max_m": float(x_max_m),
        "y_min_m": float(y_min_v),
        "y_max_m": float(y_max_m),
        "dr_m": float(dr_m),
        "angle_count": int(angle_axis.size),
        "x_axis_m": _build_display_axis(float(x_min_v), float(x_max_v), gui_w),
        "y_axis_m": _build_display_axis(float(y_min_v), float(y_max_v), gui_h),
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

    # Inverse mapping pixel→radar: per ogni pixel cartesiano calcola range e
    # angolo, poi conserva gli indici/interpolanti per non rifare la geometria
    # a ogni frame della GUI.
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
    *,
    x_min_m: float | None = None,
    y_min_m: float | None = None,
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
    x_min_v = float(-float(x_max_m) if x_min_m is None else x_min_m)
    y_min_v = float(0.0 if y_min_m is None else y_min_m)
    if (
        lut is None
        or tuple(lut.get("output_shape", ())) != (gui_h, gui_w)
        or str(lut.get("projection_mode", "")) != mode
        or str(lut.get("projection_interp", "")) != interp
        or int(lut.get("angle_count", -1)) != eff_cols
        or not np.isclose(float(lut.get("x_min_m", np.nan)), float(x_min_v))
        or not np.isclose(float(lut.get("x_max_m", np.nan)), float(x_max_m))
        or not np.isclose(float(lut.get("y_min_m", np.nan)), float(y_min_v))
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
            x_min_m=x_min_v,
            y_min_m=y_min_v,
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
    # Bilinear interpolation is centered on discrete range bins, so the
    # supported physical extent includes half a bin beyond the first/last
    # sample. At the borders we fall back to one-sided interpolation by
    # clipping r0/r1 below, instead of marking the final half-bin as empty.
    min_range_support = np.float32(-0.5)
    max_range_support = np.float32(max_range_idx + 0.5)
    valid = (
        valid_theta
        & (range_pos >= min_range_support)
        & (range_pos <= max_range_support)
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
    finite00 = np.isfinite(v00)
    finite01 = np.isfinite(v01)
    finite10 = np.isfinite(v10)
    finite11 = np.isfinite(v11)
    if bool(np.all(finite00) and np.all(finite01) and np.all(finite10) and np.all(finite11)):
        top = v00 + ((v01 - v00) * wa)
        bottom = v10 + ((v11 - v10) * wa)
        dst_flat[valid] = top + ((bottom - top) * wr)
        return dst

    w00 = (np.float32(1.0) - wr) * (np.float32(1.0) - wa)
    w01 = (np.float32(1.0) - wr) * wa
    w10 = wr * (np.float32(1.0) - wa)
    w11 = wr * wa
    weight_sum = (
        np.where(finite00, w00, np.float32(0.0))
        + np.where(finite01, w01, np.float32(0.0))
        + np.where(finite10, w10, np.float32(0.0))
        + np.where(finite11, w11, np.float32(0.0))
    ).astype(np.float32, copy=False)
    weighted = (
        np.where(finite00, v00 * w00, np.float32(0.0))
        + np.where(finite01, v01 * w01, np.float32(0.0))
        + np.where(finite10, v10 * w10, np.float32(0.0))
        + np.where(finite11, v11 * w11, np.float32(0.0))
    ).astype(np.float32, copy=False)
    supported = weight_sum >= np.float32(0.25)
    valid_idx = np.flatnonzero(valid)
    if np.any(supported):
        dst_flat[valid_idx[supported]] = (weighted[supported] / weight_sum[supported]).astype(np.float32, copy=False)
    return dst


def _quantize_zoom_nfft(base_nfft: int, max_nfft: int, scale: float) -> int:
    base = max(1, int(base_nfft))
    limit = max(base, int(max_nfft))
    target = max(float(base), float(base) * max(1.0, float(scale)))
    nfft = base
    while float(nfft) < target and nfft < limit:
        nfft <<= 1
    return int(min(max(base, nfft), limit))


def _viewport_angle_roi_indices(
    angle_axis_deg: np.ndarray,
    viewport: DisplayViewport,
) -> np.ndarray:
    axis = np.asarray(angle_axis_deg, dtype=np.float32).reshape(-1)
    if axis.size <= 0:
        return np.empty((0,), dtype=np.int32)
    valid = np.isfinite(axis)
    if not np.any(valid):
        return np.empty((0,), dtype=np.int32)
    finite_axis = axis[valid]
    if finite_axis.size <= 0:
        return np.empty((0,), dtype=np.int32)
    if finite_axis.size <= 2:
        return np.flatnonzero(valid).astype(np.int32, copy=False)
    step = float(np.nanmedian(np.abs(np.diff(finite_axis.astype(np.float64, copy=False)))))
    if not np.isfinite(step) or step <= 0.0:
        step = 0.5
    margin = max(step * 2.0, 0.5)
    roi = valid & (axis >= np.float32(float(viewport.angle_min_deg) - margin)) & (axis <= np.float32(float(viewport.angle_max_deg) + margin))
    idx = np.flatnonzero(roi)
    if idx.size <= 0:
        idx = np.flatnonzero(valid)
    if idx.size <= 0:
        return np.empty((0,), dtype=np.int32)
    start = max(0, int(idx[0]) - 1)
    stop = min(int(axis.size), int(idx[-1]) + 2)
    return np.arange(start, stop, dtype=np.int32)


def _display_zoom_cache_key_for_geometry(
    geometry: VirtualArrayGeometry,
    nfft_angle: int,
) -> tuple[Any, ...]:
    phase_centers = np.asarray(geometry.phase_centers_lambda, dtype=np.float32).reshape(-1)
    return (
        int(nfft_angle),
        bytes(phase_centers.tobytes()),
        None if geometry.uniform_spacing_lambda is None else round(float(geometry.uniform_spacing_lambda), 8),
        round(float(getattr(geometry, "angle_axis_sign", 1.0)), 8),
        round(float(getattr(geometry, "angle_u_to_sin_scale", 2.0)), 8),
    )


def _get_cached_angle_axis_and_steering(
    runtime: DisplayZoomRuntime,
    *,
    geometry: VirtualArrayGeometry,
    virtual_ant: int,
    nfft_angle: int,
) -> tuple[np.ndarray, np.ndarray]:
    cache_key = _display_zoom_cache_key_for_geometry(geometry, nfft_angle)
    cached = runtime.steering_cache.get(cache_key)
    if cached is not None:
        return cached
    angle_axis = build_angle_axis_deg(int(nfft_angle), geometry=geometry)
    steering = build_angle_steering_matrix(int(virtual_ant), int(nfft_angle), geometry=geometry)
    runtime.steering_cache[cache_key] = (angle_axis, steering)
    return angle_axis, steering


def _get_cached_projection_lut(
    runtime: DisplayZoomRuntime,
    *,
    gui_h: int,
    gui_w: int,
    viewport: DisplayViewport,
    dr_m: float,
    angle_axis_deg: np.ndarray,
    projection_mode: str,
    projection_interp: str,
) -> dict[str, Any]:
    angle_axis = np.asarray(angle_axis_deg, dtype=np.float32).reshape(-1)
    cache_key = (
        int(gui_h),
        int(gui_w),
        round(float(viewport.x_min_m), 6),
        round(float(viewport.x_max_m), 6),
        round(float(viewport.y_min_m), 6),
        round(float(viewport.y_max_m), 6),
        round(float(dr_m), 9),
        int(angle_axis.size),
        str(_normalize_display_projection_mode(projection_mode)),
        str(_normalize_display_projection_interp(projection_interp)),
    )
    cached = runtime.lut_cache.get(cache_key)
    if cached is not None:
        return cached
    lut = build_display_projection_lut(
        gui_h=int(gui_h),
        gui_w=int(gui_w),
        x_max_m=float(viewport.x_max_m),
        y_max_m=float(viewport.y_max_m),
        dr_m=float(dr_m),
        angle_axis_deg=angle_axis,
        projection_mode=projection_mode,
        projection_interp=projection_interp,
        x_min_m=float(viewport.x_min_m),
        y_min_m=float(viewport.y_min_m),
    )
    runtime.lut_cache[cache_key] = lut
    return lut


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
    # In TDM lo stesso TX ricompare ogni ``tx_count`` chirp: il PRI effettivo
    # del canale virtuale è quindi più lungo del PRI dello stream ADC.
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
    # L'acquisizione usa [frame, loop, tx, range, rx].  Portare ``range``
    # accanto a ``loop`` consente di appiattire TX×RX senza cambiare l'ordine
    # TX-major/RX-minor atteso da ``geometry.order_flat``.
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
    # La geometria può correggere l'ordine di cablaggio/fase, ma non altera
    # né i campioni né la forma della matrice virtuale.
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


def _virtual_array_may_alias_range_fft(
    range_fft: np.ndarray,
    *,
    max_bin: int,
    geometry: VirtualArrayGeometry,
    work_buf: np.ndarray | None = None,
) -> bool:
    """Return whether the virtual-array builder can hand out a source view.

    Normally the transpose from ``[frame, loop, tx, range, rx]`` forces a
    contiguous copy.  Degenerate dimensions (notably ``tx == 1``) can make
    that transpose already contiguous, however.  With identity geometry and
    no valid work buffer, the returned virtual array can then share memory
    with ``range_fft`` and downstream in-place angle processing may alter it.

    This is only needed to decide whether static/display branch reuse is safe;
    it intentionally mirrors the allocation choices in
    :func:`_build_virtual_array_from_range_fft` without allocating an array.
    """
    if range_fft.ndim != 5 or int(max_bin) <= 0:
        return False
    available_bins = int(range_fft.shape[3])
    if int(max_bin) > available_bins:
        return False

    trimmed = range_fft[:, :, :, : int(max_bin), :]
    va_src = trimmed.transpose(0, 1, 3, 2, 4)
    work_is_valid = bool(
        work_buf is not None
        and work_buf.shape == va_src.shape
        and work_buf.dtype == np.complex64
        and work_buf.flags.c_contiguous
    )
    if work_is_valid:
        return False

    # A non-identity antenna order always uses ``np.take`` (or its dedicated
    # output buffer), both of which produce a separate virtual array.
    if not geometry.identity_order:
        return False

    return bool(va_src.flags.c_contiguous)


def _power_to_db(power_lin: np.ndarray) -> np.ndarray:
    out = np.array(power_lin, dtype=np.float32, copy=True)
    np.add(out, np.float32(1e-12), out=out)
    np.log10(out, out=out)
    out *= np.float32(10.0)
    return out


def _write_diagnostic_power_db(source_power: np.ndarray, out: np.ndarray) -> None:
    """Publish a linear-power diagnostic map as dB without touching its source.

    The source maps are also consumed by detection or the display path.  The
    previous diagnostic path converted a freshly recomputed map in-place; when
    the same map is reused, conversion must instead happen only in the
    diagnostic output buffer.
    """
    out.fill(np.float32(-120.0))
    source = np.asarray(source_power, dtype=np.float32)
    if source.ndim != 2 or out.ndim != 2:
        return
    rows = min(int(out.shape[0]), int(source.shape[0]))
    cols = min(int(out.shape[1]), int(source.shape[1]))
    if rows <= 0 or cols <= 0:
        return
    target = out[:rows, :cols]
    np.copyto(target, source[:rows, :cols], casting="unsafe")
    np.add(target, np.float32(1e-12), out=target)
    np.log10(target, out=target)
    target *= np.float32(10.0)


def _normalization_reference_db(
    heatmap_lin: np.ndarray,
    *,
    skip_range_bins: int,
) -> float | None:
    src = np.asarray(heatmap_lin, dtype=np.float32)
    if src.ndim != 2 or src.size <= 0:
        return None
    start_bin = max(0, int(skip_range_bins))
    if start_bin >= int(src.shape[0]):
        start_bin = 0
    active = src[start_bin:, :]
    if active.size <= 0:
        return None
    ref_lin = float(np.max(active))
    if not np.isfinite(ref_lin) or ref_lin <= 0.0:
        return None
    return float(np.float32(10.0) * np.log10(np.float32(max(ref_lin, 1e-12))))


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
    return _resolve_detection_threshold_db(
        power_db,
        threshold_mode=threshold_mode,
        threshold_db=threshold_db,
        min_power_db=min_power_db,
    )


def _cfar_training_values_for_cell(
    power_lin: np.ndarray,
    *,
    row: int,
    col: int,
    train_row: int,
    guard_row: int,
    train_col: int,
    guard_col: int,
) -> np.ndarray | None:
    n_rows, n_cols = int(power_lin.shape[0]), int(power_lin.shape[1])
    r0 = int(row) - guard_row - train_row
    r1 = int(row) + guard_row + train_row + 1
    c0 = int(col) - guard_col - train_col
    c1 = int(col) + guard_col + train_col + 1
    if r0 < 0 or c0 < 0 or r1 > n_rows or c1 > n_cols:
        # Ai bordi non esiste una finestra CFAR completa: una soglia infinita
        # è preferibile a una stima del rumore con un numero variabile di celle.
        return None

    window = power_lin[r0:r1, c0:c1]
    training_mask = np.ones(window.shape, dtype=bool)
    gr0 = train_row
    gr1 = train_row + (2 * guard_row) + 1
    gc0 = train_col
    gc1 = train_col + (2 * guard_col) + 1
    # Escludi CUT e celle di guardia: l'energia del possibile bersaglio non
    # deve entrare nella stima del rumore locale.
    training_mask[gr0:gr1, gc0:gc1] = False
    training = window[training_mask]
    if training.size <= 0:
        return None
    training = training[np.isfinite(training)]
    if training.size <= 0:
        return None
    return training


def _compute_cfar_threshold_db_map_python(
    power_lin: np.ndarray,
    *,
    threshold_mode: DetectionThresholdMode,
    train_range_bins: int,
    guard_range_bins: int,
    train_col_bins: int,
    guard_col_bins: int,
    threshold_offset_db: float,
    min_power_db: float,
    os_cfar_rank: int,
) -> np.ndarray:
    power = np.asarray(power_lin, dtype=np.float32)
    threshold_map = np.full(power.shape, np.float32(np.inf), dtype=np.float32)
    if power.ndim != 2 or power.size <= 0 or threshold_mode not in {"ca_cfar", "os_cfar"}:
        return threshold_map

    train_row = int(max(0, train_range_bins))
    guard_row = int(max(0, guard_range_bins))
    train_col = int(max(0, train_col_bins))
    guard_col = int(max(0, guard_col_bins))
    if train_row <= 0 and train_col <= 0:
        return threshold_map

    n_rows, n_cols = int(power.shape[0]), int(power.shape[1])
    row_margin = train_row + guard_row
    col_margin = train_col + guard_col
    offset_lin = np.float32(10.0 ** (float(threshold_offset_db) / 10.0))
    min_lin = np.float32(10.0 ** (float(min_power_db) / 10.0))

    for row in range(row_margin, n_rows - row_margin):
        for col in range(col_margin, n_cols - col_margin):
            training = _cfar_training_values_for_cell(
                power,
                row=row,
                col=col,
                train_row=train_row,
                guard_row=guard_row,
                train_col=train_col,
                guard_col=guard_col,
            )
            if training is None:
                continue
            if threshold_mode == "ca_cfar":
                noise_lin = np.float32(np.mean(training, dtype=np.float32))
            else:
                rank = int(os_cfar_rank)
                if rank <= 0:
                    rank = int(math.ceil(0.75 * float(training.size)))
                rank = min(max(1, rank), int(training.size))
                noise_lin = np.float32(np.partition(training, rank - 1)[rank - 1])
            threshold_lin = max(float(min_lin), float(noise_lin) * float(offset_lin))
            threshold_map[row, col] = np.float32(10.0 * math.log10(max(threshold_lin, 1e-12)))
    return threshold_map


def _compute_ca_cfar_threshold_db_map_numba(
    power: np.ndarray,
    *,
    train_row: int,
    guard_row: int,
    train_col: int,
    guard_col: int,
    threshold_offset_db: float,
    min_power_db: float,
) -> np.ndarray | None:
    global _CFAR_NUMBA_ENABLED, _CFAR_NUMBA_DISABLED_REASON, _CFAR_NUMBA_LAST_ERROR

    if not _CFAR_NUMBA_ENABLED or _ca_cfar_threshold_db_map_numba_kernel is None:
        return None
    threshold_map = np.full(power.shape, np.float32(np.inf), dtype=np.float32)
    if power.ndim != 2 or power.size <= 0:
        return threshold_map
    if train_row <= 0 and train_col <= 0:
        return threshold_map

    offset_lin = np.float32(10.0 ** (float(threshold_offset_db) / 10.0))
    min_lin = np.float32(10.0 ** (float(min_power_db) / 10.0))
    try:
        _ca_cfar_threshold_db_map_numba_kernel(
            power,
            threshold_map,
            int(train_row),
            int(guard_row),
            int(train_col),
            int(guard_col),
            offset_lin,
            min_lin,
        )
        return threshold_map
    except Exception as exc:  # pragma: no cover - defensive fallback for local JIT/runtime issues
        _CFAR_NUMBA_ENABLED = False
        _CFAR_NUMBA_DISABLED_REASON = f"runtime failure: {exc}"
        _CFAR_NUMBA_LAST_ERROR = str(exc)
        print(f"[DSP NUMBA WARN] CA-CFAR JIT disabled; falling back to Python ({exc})")
        return None


def _candidate_mask_from_threshold_map(power_lin: np.ndarray, threshold_map: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        power_db = compute_detection_power_db_map(power_lin)
    return np.isfinite(power_db) & np.isfinite(threshold_map) & (power_db >= threshold_map)


def self_check_ca_cfar_numba() -> tuple[bool, str]:
    if _numba is None or _ca_cfar_threshold_db_map_numba_kernel is None:
        return False, f"numba unavailable: {_NUMBA_IMPORT_ERROR}"

    rng = np.random.default_rng(20260422)
    cases: list[tuple[np.ndarray, tuple[int, int, int, int], float, float]] = []

    small = np.ones((4, 5), dtype=np.float32)
    small[0, 0] = np.nan
    cases.append((small, (3, 1, 2, 1), 6.0, -120.0))

    edge = np.arange(81, dtype=np.float32).reshape(9, 9) + np.float32(1.0)
    edge[4, 4] = np.float32(200.0)
    edge[2, 7] = np.inf
    edge[7, 2] = -np.inf
    cases.append((edge, (1, 1, 1, 1), 3.0, -110.0))

    mixed = rng.lognormal(mean=1.0, sigma=0.7, size=(17, 23)).astype(np.float32)
    mixed[3, 5] = np.nan
    mixed[8, 12] = np.inf
    mixed[12, 3] = -np.inf
    cases.append((mixed, (2, 1, 3, 1), 9.5, -80.0))

    large = rng.lognormal(mean=2.0, sigma=0.9, size=(96, 128)).astype(np.float32)
    large[20, 30] = np.float32(1e6)
    large[40, 60] = np.nan
    large[70, 80] = np.inf
    cases.append((large, (8, 2, 4, 1), 12.0, 8.0))

    for idx, (power, params, offset_db, min_db) in enumerate(cases, start=1):
        train_row, guard_row, train_col, guard_col = params
        py_map = _compute_cfar_threshold_db_map_python(
            power,
            threshold_mode="ca_cfar",
            train_range_bins=train_row,
            guard_range_bins=guard_row,
            train_col_bins=train_col,
            guard_col_bins=guard_col,
            threshold_offset_db=offset_db,
            min_power_db=min_db,
            os_cfar_rank=0,
        )
        jit_map = np.full(power.shape, np.float32(np.inf), dtype=np.float32)
        offset_lin = np.float32(10.0 ** (float(offset_db) / 10.0))
        min_lin = np.float32(10.0 ** (float(min_db) / 10.0))
        try:
            _ca_cfar_threshold_db_map_numba_kernel(
                np.asarray(power, dtype=np.float32),
                jit_map,
                int(train_row),
                int(guard_row),
                int(train_col),
                int(guard_col),
                offset_lin,
                min_lin,
            )
        except Exception as exc:
            return False, f"case {idx} JIT failed: {exc}"
        if jit_map.dtype != np.float32:
            return False, f"case {idx} dtype changed: {jit_map.dtype}"
        if not np.allclose(py_map, jit_map, rtol=2e-5, atol=2e-4, equal_nan=True):
            diff = np.nanmax(np.abs(py_map.astype(np.float64) - jit_map.astype(np.float64)))
            return False, f"case {idx} threshold mismatch max_abs_diff={float(diff):.6g}"
        py_mask = _candidate_mask_from_threshold_map(power, py_map)
        jit_mask = _candidate_mask_from_threshold_map(power, jit_map)
        if not np.array_equal(py_mask, jit_mask):
            return False, f"case {idx} candidate_mask mismatch"
    return True, f"{len(cases)} deterministic cases passed"


def configure_cfar_numba_runtime(
    cfg: CfarNumbaConfig,
    *,
    log: bool = True,
) -> None:
    global _CFAR_NUMBA_ENABLED, _CFAR_NUMBA_SELF_CHECKED, _CFAR_NUMBA_DISABLED_REASON, _CFAR_NUMBA_LAST_ERROR

    _CFAR_NUMBA_SELF_CHECKED = False
    _CFAR_NUMBA_LAST_ERROR = ""
    if not cfg.enabled:
        _CFAR_NUMBA_ENABLED = False
        _CFAR_NUMBA_DISABLED_REASON = "disabled by config"
        if log:
            print("[DSP NUMBA] CA-CFAR JIT disabled by config; Python CFAR path active.")
        return
    if _numba is None or _ca_cfar_threshold_db_map_numba_kernel is None:
        _CFAR_NUMBA_ENABLED = False
        _CFAR_NUMBA_DISABLED_REASON = f"numba unavailable: {_NUMBA_IMPORT_ERROR}"
        if log:
            print(f"[DSP NUMBA WARN] CA-CFAR JIT requested but unavailable ({_NUMBA_IMPORT_ERROR}); Python fallback active.")
        return

    _CFAR_NUMBA_ENABLED = True
    _CFAR_NUMBA_DISABLED_REASON = ""
    if cfg.self_check_on_start:
        ok, msg = self_check_ca_cfar_numba()
        _CFAR_NUMBA_SELF_CHECKED = True
        if not ok:
            _CFAR_NUMBA_ENABLED = False
            _CFAR_NUMBA_DISABLED_REASON = f"self-check failed: {msg}"
            _CFAR_NUMBA_LAST_ERROR = msg
            if log:
                print(f"[DSP NUMBA WARN] CA-CFAR JIT self-check failed; Python fallback active ({msg})")
            return
        if log:
            print(f"[DSP NUMBA] CA-CFAR JIT enabled; self-check OK ({msg}).")
    elif cfg.warmup_on_start:
        warmup = np.ones((9, 9), dtype=np.float32)
        maybe_map = _compute_ca_cfar_threshold_db_map_numba(
            warmup,
            train_row=1,
            guard_row=1,
            train_col=1,
            guard_col=1,
            threshold_offset_db=6.0,
            min_power_db=-120.0,
        )
        if maybe_map is None:
            if log:
                print(f"[DSP NUMBA WARN] CA-CFAR JIT warmup failed; Python fallback active ({_CFAR_NUMBA_DISABLED_REASON})")
            return
        if log:
            print("[DSP NUMBA] CA-CFAR JIT enabled; warmup OK.")
    elif log:
        print("[DSP NUMBA] CA-CFAR JIT enabled; first CA-CFAR call will compile if cache is cold.")


def cfar_numba_runtime_status() -> dict[str, Any]:
    return {
        "enabled": bool(_CFAR_NUMBA_ENABLED),
        "available": bool(_numba is not None and _ca_cfar_threshold_db_map_numba_kernel is not None),
        "self_checked": bool(_CFAR_NUMBA_SELF_CHECKED),
        "disabled_reason": str(_CFAR_NUMBA_DISABLED_REASON),
        "last_error": str(_CFAR_NUMBA_LAST_ERROR),
        "numba_version": None if _numba is None else str(getattr(_numba, "__version__", "unknown")),
    }


def configure_angle_power_numba_runtime(
    cfg: AnglePowerNumbaConfig,
    *,
    log: bool = True,
) -> None:
    """Apply the bounded Numba policy before the realtime frame loop starts."""
    global _ANGLE_POWER_NUMBA_ENABLED, _ANGLE_POWER_NUMBA_THREADS, _ANGLE_POWER_NUMBA_LAST_ERROR

    _ANGLE_POWER_NUMBA_LAST_ERROR = ""
    if not cfg.enabled:
        _ANGLE_POWER_NUMBA_ENABLED = False
        _ANGLE_POWER_NUMBA_THREADS = 0
        if log:
            print("[DSP NUMBA] angle-power JIT disabled by config; NumPy reduction active.")
        return
    if _numba is None or _angle_power_frame_loop_numba_kernel is None:
        _ANGLE_POWER_NUMBA_ENABLED = False
        _ANGLE_POWER_NUMBA_THREADS = 0
        _ANGLE_POWER_NUMBA_LAST_ERROR = f"numba unavailable: {_NUMBA_IMPORT_ERROR}"
        if log:
            print(f"[DSP NUMBA WARN] angle-power JIT unavailable ({_NUMBA_IMPORT_ERROR}); NumPy reduction active.")
        return

    try:
        runtime_max = max(1, int(getattr(_numba.config, "NUMBA_NUM_THREADS", _numba.get_num_threads())))
        requested = int(cfg.threads)
        effective = runtime_max if requested <= 0 else min(max(1, requested), runtime_max)
        _numba.set_num_threads(int(effective))
        _ANGLE_POWER_NUMBA_ENABLED = True
        _ANGLE_POWER_NUMBA_THREADS = int(_numba.get_num_threads())
        if log:
            print(
                "[DSP NUMBA] "
                f"angle-power JIT enabled; threads={_ANGLE_POWER_NUMBA_THREADS} "
                f"(requested={'auto' if requested <= 0 else requested}, max={runtime_max})."
            )
    except Exception as exc:  # pragma: no cover - depends on the local Numba runtime
        _ANGLE_POWER_NUMBA_ENABLED = False
        _ANGLE_POWER_NUMBA_THREADS = 0
        _ANGLE_POWER_NUMBA_LAST_ERROR = str(exc)
        if log:
            print(f"[DSP NUMBA WARN] angle-power JIT disabled; cannot set thread limit ({exc}).")


def angle_power_numba_runtime_status() -> dict[str, Any]:
    return {
        "enabled": bool(_ANGLE_POWER_NUMBA_ENABLED),
        "available": bool(_numba is not None and _angle_power_frame_loop_numba_kernel is not None),
        "threads": int(_ANGLE_POWER_NUMBA_THREADS),
        "last_error": str(_ANGLE_POWER_NUMBA_LAST_ERROR),
    }


def compute_cfar_threshold_db_map(
    power_lin: np.ndarray,
    *,
    threshold_mode: DetectionThresholdMode,
    train_range_bins: int,
    guard_range_bins: int,
    train_col_bins: int,
    guard_col_bins: int,
    threshold_offset_db: float,
    min_power_db: float,
    os_cfar_rank: int,
) -> np.ndarray:
    power = np.asarray(power_lin, dtype=np.float32)
    if threshold_mode == "ca_cfar":
        train_row = int(max(0, train_range_bins))
        guard_row = int(max(0, guard_range_bins))
        train_col = int(max(0, train_col_bins))
        guard_col = int(max(0, guard_col_bins))
        maybe_map = _compute_ca_cfar_threshold_db_map_numba(
            power,
            train_row=train_row,
            guard_row=guard_row,
            train_col=train_col,
            guard_col=guard_col,
            threshold_offset_db=threshold_offset_db,
            min_power_db=min_power_db,
        )
        if maybe_map is not None:
            return maybe_map
    return _compute_cfar_threshold_db_map_python(
        power,
        threshold_mode=threshold_mode,
        train_range_bins=train_range_bins,
        guard_range_bins=guard_range_bins,
        train_col_bins=train_col_bins,
        guard_col_bins=guard_col_bins,
        threshold_offset_db=threshold_offset_db,
        min_power_db=min_power_db,
        os_cfar_rank=os_cfar_rank,
    )


def compute_cfar_candidate_mask(
    power_lin: np.ndarray,
    *,
    threshold_mode: DetectionThresholdMode,
    train_range_bins: int,
    guard_range_bins: int,
    train_col_bins: int,
    guard_col_bins: int,
    threshold_offset_db: float,
    min_power_db: float,
    os_cfar_rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    threshold_map = compute_cfar_threshold_db_map(
        power_lin,
        threshold_mode=threshold_mode,
        train_range_bins=train_range_bins,
        guard_range_bins=guard_range_bins,
        train_col_bins=train_col_bins,
        guard_col_bins=guard_col_bins,
        threshold_offset_db=threshold_offset_db,
        min_power_db=min_power_db,
        os_cfar_rank=os_cfar_rank,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        power_db = compute_detection_power_db_map(power_lin)
    mask = np.isfinite(power_db) & np.isfinite(threshold_map) & (power_db >= threshold_map)
    return mask, threshold_map


def extract_detection_peaks_2d(
    power_db: np.ndarray,
    *,
    threshold_db: float,
    win_row: int,
    win_col: int,
    max_peaks: int,
    candidates_mask: np.ndarray | None = None,
) -> np.ndarray:
    return _select_localmax_2d(
        power_db,
        threshold_db=threshold_db,
        win_row=win_row,
        win_col=win_col,
        max_peaks=max_peaks,
        candidates_mask=candidates_mask,
    )


def _select_localmax_2d(
    power_db: np.ndarray,
    *,
    threshold_db: float,
    win_row: int,
    win_col: int,
    max_peaks: int,
    candidates_mask: np.ndarray | None = None,
) -> np.ndarray:
    if power_db.size <= 0 or max_peaks <= 0:
        return np.empty((0, 2), dtype=np.int32)
    if candidates_mask is None:
        candidates_mask_eff = power_db >= np.float32(threshold_db)
    else:
        candidates_mask_eff = np.asarray(candidates_mask, dtype=bool)
        if candidates_mask_eff.shape != power_db.shape:
            candidates_mask_eff = np.zeros(power_db.shape, dtype=bool)
        else:
            candidates_mask_eff = candidates_mask_eff & np.isfinite(power_db)
    if not np.any(candidates_mask_eff):
        return np.empty((0, 2), dtype=np.int32)
    candidates = np.argwhere(candidates_mask_eff)
    strengths = power_db[candidates_mask_eff]
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
        # Selezione greedy dal picco più forte: la finestra soppressa evita
        # che i bin della stessa lobatura producano detection duplicate.
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
    virtual_array_work_buf: np.ndarray | None = None,
    virtual_array_flat_work_buf: np.ndarray | None = None,
) -> tuple[list[Detection], np.ndarray]:
    """Estrae bersagli statici dalla mappa range-angolo.

    Restituisce anche la mappa lineare da cui sono stati scelti i picchi: la
    stessa mappa può quindi essere riusata per diagnosi e visualizzazione.
    """
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
    if apply_angle_window:
        virtual_array *= w_angle
    static_heatmap = compute_angle_heatmap(
        virtual_array,
        angle_cfg=angle_cfg,
        dsp_cfg=dsp_cfg,
        angle_steering=angle_steering,
        geometry=virtual_array_geometry,
        ant_spacing=virtual_array_geometry.uniform_spacing_lambda,
    )
    static_db = compute_detection_power_db_map(static_heatmap)
    candidates_mask: np.ndarray | None = None
    if static_cfg.threshold_mode in {"ca_cfar", "os_cfar"}:
        # CFAR decide quali celle sono statisticamente interessanti; il NMS
        # sottostante conserva poi un solo massimo per ciascun bersaglio.
        candidates_mask, _ = compute_cfar_candidate_mask(
            static_heatmap,
            threshold_mode=static_cfg.threshold_mode,
            train_range_bins=static_cfg.cfar_train_range_bins,
            guard_range_bins=static_cfg.cfar_guard_range_bins,
            train_col_bins=static_cfg.cfar_train_col_bins,
            guard_col_bins=static_cfg.cfar_guard_col_bins,
            threshold_offset_db=static_cfg.cfar_threshold_db,
            min_power_db=static_cfg.min_power_db,
            os_cfar_rank=static_cfg.os_cfar_rank,
        )
        threshold_db = float(static_cfg.min_power_db)
    else:
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
        candidates_mask=candidates_mask,
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


def _compute_doppler_cube(
    range_fft: np.ndarray,
    *,
    max_bin: int,
    dsp_cfg: RealtimeDSPConfig,
    w_doppler: np.ndarray,
    apply_doppler_window: bool,
    doppler_fft_shift: bool,
    doppler_work_buf: np.ndarray | None = None,
) -> np.ndarray:
    if range_fft.ndim != 5 or max_bin <= 0:
        return np.empty((0, 0, 0, 0, 0), dtype=np.complex64)
    n_loops = int(range_fft.shape[1])
    if n_loops <= 0:
        return np.empty((0, 0, 0, 0, 0), dtype=np.complex64)

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
            doppler_in = np.empty(trimmed.shape, dtype=trimmed.dtype)
            np.multiply(trimmed, w_doppler, out=doppler_in)
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
    if doppler_fft_shift:
        doppler_cube = np.fft.fftshift(doppler_cube, axes=1)
    return doppler_cube.astype(np.complex64, copy=False)


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
    """Calcola il cubo Doppler e la sua proiezione di potenza range-Doppler."""
    doppler_cube = _compute_doppler_cube(
        range_fft,
        max_bin=max_bin,
        dsp_cfg=dsp_cfg,
        w_doppler=w_doppler,
        apply_doppler_window=apply_doppler_window,
        doppler_fft_shift=bool(moving_cfg.doppler_fft_shift),
        doppler_work_buf=doppler_work_buf,
    )
    if doppler_cube.size <= 0:
        return doppler_cube, np.empty((0, 0), dtype=np.float32)

    re = doppler_cube.real
    im = doppler_cube.imag
    rd_pow = (re * re + im * im).mean(axis=(0, 2, 4), dtype=np.float32)
    range_doppler_map = rd_pow.transpose(1, 0).astype(np.float32, copy=False)

    excl = int(max(0, moving_cfg.zero_doppler_exclusion_bins))
    if excl > 0 and range_doppler_map.shape[1] > 0:
        # Lo zero-Doppler viene azzerato solo nella mappa di detection: il
        # cubo complesso resta integro per la successiva stima angolare.
        n_doppler = int(range_doppler_map.shape[1])
        if moving_cfg.doppler_fft_shift:
            zero_idx = n_doppler // 2
            d0 = max(0, zero_idx - excl)
            d1 = min(n_doppler, zero_idx + excl + 1)
            range_doppler_map[:, d0:d1] = 0.0
        else:
            # Unshifted FFT is circular: negative bins nearest zero live at
            # the end of the array, not next to index zero in memory.
            range_doppler_map[:, : min(n_doppler, excl + 1)] = 0.0
            if excl < n_doppler:
                range_doppler_map[:, max(0, n_doppler - excl) :] = 0.0
    return doppler_cube, range_doppler_map


def compute_range_angle_moving_velocity_map(
    range_fft_doppler: np.ndarray,
    *,
    max_bin: int,
    dsp_cfg: RealtimeDSPConfig,
    w_doppler: np.ndarray,
    w_angle: np.ndarray,
    apply_doppler_window: bool,
    apply_angle_window: bool,
    doppler_fft_shift: bool,
    doppler_axis_mps: np.ndarray | None,
    tdm_mimo_compensation_table: np.ndarray | None,
    virtual_array_geometry: VirtualArrayGeometry,
    angle_steering: np.ndarray,
    angle_axis_deg: np.ndarray | None = None,
    doppler_work_buf: np.ndarray | None = None,
    virtual_array_work_buf: np.ndarray | None = None,
    virtual_array_flat_work_buf: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build collapsed range-angle moving maps for display.

    The input is the moving-branch pre-Doppler range FFT, so this view stays
    Doppler-consistent without inheriting display-only filtering.

    The output pair is (dominant signed velocity m/s, normalized moving-power
    alpha). Beamforming is deliberately Bartlett for a stable Doppler/range/angle
    cube without adding MVDR covariance cost to every display frame.
    """
    n_range = max(0, int(max_bin))
    n_angle = max(1, int(dsp_cfg.nfft_angle))
    # Zero è un valore fisico ambiguo nella mappa velocità. ``alpha_ra`` è
    # dunque anche la maschera di validità: alpha=0 significa nessuna stima
    # affidabile, non necessariamente velocità nulla.
    velocity_ra = np.zeros((n_range, n_angle), dtype=np.float32)
    alpha_ra = np.zeros((n_range, n_angle), dtype=np.float32)
    if n_range <= 0:
        return velocity_ra, alpha_ra

    n_loops = int(range_fft_doppler.shape[1]) if getattr(range_fft_doppler, "ndim", 0) == 5 else 0
    doppler_axis = None if doppler_axis_mps is None else np.asarray(doppler_axis_mps, dtype=np.float32).reshape(-1)
    if doppler_axis is None or int(doppler_axis.size) != n_loops or n_loops <= 0:
        _warn_velocity_display_once(
            "velocity heatmap disabled: Doppler velocity axis is unavailable or incompatible; publishing no-data map."
        )
        return velocity_ra, alpha_ra

    doppler_cube = _compute_doppler_cube(
        range_fft_doppler,
        max_bin=n_range,
        dsp_cfg=dsp_cfg,
        w_doppler=w_doppler,
        apply_doppler_window=apply_doppler_window,
        doppler_fft_shift=bool(doppler_fft_shift),
        doppler_work_buf=doppler_work_buf,
    )
    if doppler_cube.size <= 0 or doppler_cube.ndim != 5:
        return velocity_ra, alpha_ra

    n_frames = int(doppler_cube.shape[0])
    n_doppler = int(doppler_cube.shape[1])
    n_tx = int(doppler_cube.shape[2])
    n_rx = int(doppler_cube.shape[4])
    n_virtual = int(dsp_cfg.virtual_ant)
    if n_tx * n_rx != n_virtual:
        _warn_velocity_display_once(
            f"velocity heatmap disabled: tx*rx={n_tx * n_rx} does not match virtual_ant={n_virtual}."
        )
        return velocity_ra, alpha_ra

    comp_table = None if tdm_mimo_compensation_table is None else np.asarray(tdm_mimo_compensation_table, dtype=np.complex64)
    if comp_table is not None and comp_table.ndim == 2 and comp_table.shape[0] >= n_doppler and comp_table.shape[1] >= n_tx:
        # In TDM i TX sono campionati in istanti diversi. Compensare la fase
        # per bin Doppler prima di appiattire TX×RX evita un angolo falso per
        # i bersagli in movimento.
        phase = comp_table[:n_doppler, :n_tx].reshape(1, n_doppler, n_tx, 1, 1)
        np.multiply(doppler_cube, phase, out=doppler_cube)

    va_src = doppler_cube.transpose(0, 1, 3, 2, 4)
    if (
        virtual_array_work_buf is not None
        and virtual_array_work_buf.shape == va_src.shape
        and virtual_array_work_buf.dtype == np.complex64
        and virtual_array_work_buf.flags.c_contiguous
    ):
        np.copyto(virtual_array_work_buf, va_src, casting="unsafe")
        va_tx_rx = virtual_array_work_buf
    else:
        va_tx_rx = np.ascontiguousarray(va_src)
    va_flat = va_tx_rx.reshape(n_frames, n_doppler, n_range, n_virtual)
    if not virtual_array_geometry.identity_order:
        if (
            virtual_array_flat_work_buf is not None
            and virtual_array_flat_work_buf.shape == va_flat.shape
            and virtual_array_flat_work_buf.dtype == np.complex64
            and virtual_array_flat_work_buf.flags.c_contiguous
        ):
            np.take(va_flat, virtual_array_geometry.order_flat, axis=-1, out=virtual_array_flat_work_buf)
            va_flat = virtual_array_flat_work_buf
        else:
            va_flat = np.take(va_flat, virtual_array_geometry.order_flat, axis=-1).astype(np.complex64, copy=False)

    if apply_angle_window:
        va_flat *= w_angle

    steering = np.asarray(angle_steering, dtype=np.complex64)
    if steering.ndim != 2 or steering.shape[0] < n_virtual or steering.shape[1] < n_angle:
        steering = build_angle_steering_matrix(n_virtual, n_angle, geometry=virtual_array_geometry)
    steering = steering[:n_virtual, :n_angle].astype(np.complex64, copy=False)

    beam = np.einsum("ak,fdra->fdrk", np.conj(steering), va_flat, optimize=True)
    beam_power = (beam.real * beam.real + beam.imag * beam.imag).astype(np.float32, copy=False)
    if n_frames > 1:
        power_dra = beam_power.mean(axis=0, dtype=np.float32)
    else:
        power_dra = beam_power[0]
    if power_dra.size <= 0:
        return velocity_ra, alpha_ra

    max_power = np.max(power_dra, axis=0)
    dominant_doppler = np.argmax(power_dra, axis=0)
    finite_power = np.isfinite(max_power)
    if not np.any(finite_power):
        return velocity_ra, alpha_ra

    peak_power = float(np.max(max_power[finite_power]))
    if not np.isfinite(peak_power) or peak_power <= 0.0:
        return velocity_ra, alpha_ra
    velocity_cfg = getattr(dsp_cfg, "range_angle_moving", RangeAngleMovingConfig())
    rel_floor_db = float(getattr(velocity_cfg, "relative_power_floor_db", -12.0))
    if not np.isfinite(rel_floor_db):
        rel_floor_db = -12.0
    min_power_db = float(getattr(velocity_cfg, "min_power_db", 6.0))
    if not np.isfinite(min_power_db):
        min_power_db = 6.0
    min_dominance = float(getattr(velocity_cfg, "min_dominance_ratio", 0.65))
    if not np.isfinite(min_dominance):
        min_dominance = 0.65
    min_dominance = min(1.0, max(0.0, min_dominance))

    rel_floor = peak_power * float(10.0 ** (rel_floor_db / 10.0))
    abs_floor = float(10.0 ** (min_power_db / 10.0))
    power_floor = np.float32(max(rel_floor, abs_floor, 1e-12))

    sum_power = power_dra.sum(axis=0, dtype=np.float32)
    dominance = max_power / np.maximum(sum_power, np.float32(1e-12))
    reliable = finite_power & (max_power >= power_floor) & (dominance >= np.float32(min_dominance))
    if angle_axis_deg is not None:
        angle_axis = np.asarray(angle_axis_deg, dtype=np.float32).reshape(-1)
        finite_angle = np.zeros(n_angle, dtype=bool)
        angle_count = min(n_angle, int(angle_axis.size))
        if angle_count > 0:
            finite_angle[:angle_count] = np.isfinite(angle_axis[:angle_count])
        reliable &= finite_angle.reshape(1, n_angle)
    if np.any(reliable):
        velocity_ra[reliable] = doppler_axis[dominant_doppler[reliable]]
        alpha_ra[reliable] = (max_power[reliable] / np.float32(peak_power)).astype(np.float32, copy=False)
        np.clip(alpha_ra, 0.0, 1.0, out=alpha_ra)
    return velocity_ra, alpha_ra


def detect_moving_targets(
    range_doppler_map: np.ndarray,
    *,
    moving_cfg: DetectionConfigMoving,
    range_bin_m: float,
    doppler_axis_mps: np.ndarray | None,
) -> list[Detection]:
    """Estrae picchi mobili e associa a ogni bin la velocità Doppler fisica."""
    if not moving_cfg.enabled or range_doppler_map.size <= 0:
        return []
    moving_db = compute_detection_power_db_map(range_doppler_map)
    candidates_mask: np.ndarray | None = None
    if moving_cfg.threshold_mode in {"ca_cfar", "os_cfar"}:
        candidates_mask, _ = compute_cfar_candidate_mask(
            range_doppler_map,
            threshold_mode=moving_cfg.threshold_mode,
            train_range_bins=moving_cfg.cfar_train_range_bins,
            guard_range_bins=moving_cfg.cfar_guard_range_bins,
            train_col_bins=moving_cfg.cfar_train_col_bins,
            guard_col_bins=moving_cfg.cfar_guard_col_bins,
            threshold_offset_db=moving_cfg.cfar_threshold_db,
            min_power_db=moving_cfg.min_power_db,
            os_cfar_rank=moving_cfg.os_cfar_rank,
        )
        threshold_db = float(moving_cfg.min_power_db)
    else:
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
        candidates_mask=candidates_mask,
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
        if apply_angle_window:
            va *= w_angle
        angle_pow = compute_angle_heatmap(
            va,
            angle_cfg=angle_cfg,
            dsp_cfg=dsp_cfg,
            angle_steering=angle_steering,
            geometry=virtual_array_geometry,
            ant_spacing=virtual_array_geometry.uniform_spacing_lambda,
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
        # Le due branche hanno guadagni algoritmici differenti, quindi i raw
        # power_lin non sono pesi confrontabili. "Prefer static" mantiene la
        # geometria statica e aggiunge soltanto il Doppler della branca mobile.
        range_bin = int(det_static.range_bin)
        angle_bin = det_static.angle_bin
        range_m = float(det_static.range_m)
        angle_deg = float(det_static.angle_deg)
        x_m = float(det_static.x_m)
        y_m = float(det_static.y_m)
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
    """Unisce detection statiche e mobili che descrivono lo stesso bersaglio.

    La fusione conserva la geometria angolare della branca statica e, quando
    disponibile, il Doppler della branca mobile.
    """
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
    """Normalizza valori non finiti e rimuove duplicati prima del tracking."""
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

    frame_shape = tuple(int(x) for x in range_fft.shape[1:])
    state_shapes = [
        tuple(int(x) for x in value.shape)
        for value in (bg_state.model, bg_state.init_sum, bg_state.running_sum, bg_state.window_sum)
        if value is not None
    ]
    if bg_state.window_ring is not None:
        state_shapes.append(tuple(int(x) for x in bg_state.window_ring.shape[1:]))
    if any(shape != frame_shape for shape in state_shapes):
        # NFFT/ROI can change at runtime. Never combine a background learned
        # on one cube geometry with another one.
        bg_state.model = None
        bg_state.init_sum = None
        bg_state.init_count = 0
        bg_state.running_sum = None
        bg_state.running_count = 0
        bg_state.window_ring = None
        bg_state.window_sum = None
        bg_state.window_count = 0
        bg_state.window_head = 0

    mode = str(bg_cfg.mode)
    batch_frames = int(range_fft.shape[0])
    frame_sum: np.ndarray | None = None
    if (bg_state.model is None and mode != "window_mean") or mode in {"ema", "running_mean"}:
        frame_sum = range_fft.sum(axis=0, dtype=np.complex64)
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
        if mode == "window_mean":
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
            assert frame_sum is not None
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
        # Questo ramo sottrae le magnitudini e conserva la fase corrente: è un
        # floor visivo/non coerente, non una cancellazione complessa del clutter.
        current_mag = np.abs(range_fft)
        bg_mag = np.abs(bg_broadcast)
        # Per x != 0 vale exp(j*angle(x)) == x / abs(x).  Usiamo quindi il
        # guadagno radiale max(abs(x) - abs(bg), 0) / abs(x): conserva
        # matematicamente la fase di x (entro l'arrotondamento float), ma evita
        # angle() ed exp() per ogni cella.
        # Quando abs(x) è zero il numeratore è già zero; ``where`` lascia il
        # guadagno a zero e evita la divisione per zero.
        gain = np.maximum(current_mag - bg_mag, 0.0).astype(np.float32, copy=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(gain, current_mag, out=gain, where=current_mag > np.float32(0.0))
        range_fft_out = (range_fft * gain).astype(np.complex64, copy=False)
    else:
        range_fft_out = range_fft - bg_broadcast
    # Avoid double counting the batch that completed background initialization.
    if model_initialized_now:
        return range_fft_out
    if mode == "ema":
        assert frame_sum is not None
        frame_mean = frame_sum / np.float32(max(1, batch_frames))
        model *= np.float32(1.0 - bg_cfg.alpha)
        model += np.float32(bg_cfg.alpha) * frame_mean
        bg_state.model = model
    elif mode == "running_mean":
        assert frame_sum is not None
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
    elif mode == "window_mean":
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


def _format_cpu_list(cpus: list[int] | None) -> str:
    if cpus is None:
        return "unavailable"
    if not cpus:
        return "[]"
    if len(cpus) <= 32:
        return str(cpus)
    return f"{cpus[:16]} ... {cpus[-8:]} (count={len(cpus)})"


def _effective_process_affinity() -> tuple[list[int] | None, str]:
    try:
        if hasattr(os, "sched_getaffinity"):
            return sorted(int(c) for c in os.sched_getaffinity(0)), "os.sched_getaffinity"
    except Exception:
        pass
    try:
        import psutil  # type: ignore

        return sorted(int(c) for c in psutil.Process(os.getpid()).cpu_affinity()), "psutil"
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes
            import ctypes.wintypes as wt

            kernel32 = ctypes.windll.kernel32
            GetCurrentProcess = kernel32.GetCurrentProcess
            GetCurrentProcess.restype = wt.HANDLE
            GetProcessAffinityMask = kernel32.GetProcessAffinityMask
            GetProcessAffinityMask.argtypes = [wt.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
            GetProcessAffinityMask.restype = wt.BOOL
            proc_mask = ctypes.c_size_t(0)
            sys_mask = ctypes.c_size_t(0)
            ok = GetProcessAffinityMask(GetCurrentProcess(), ctypes.byref(proc_mask), ctypes.byref(sys_mask))
            if ok:
                mask = int(proc_mask.value)
                cpus = [idx for idx in range(max(1, mask.bit_length())) if mask & (1 << idx)]
                return cpus, "WinAPI"
        except Exception:
            pass
    return None, "unavailable"


def _effective_process_priority() -> tuple[str, str]:
    try:
        import psutil  # type: ignore

        return str(psutil.Process(os.getpid()).nice()), "psutil"
    except Exception:
        pass
    try:
        if hasattr(os, "getpriority") and hasattr(os, "PRIO_PROCESS"):
            return str(os.getpriority(os.PRIO_PROCESS, 0)), "os.getpriority"
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes
            import ctypes.wintypes as wt

            kernel32 = ctypes.windll.kernel32
            GetCurrentProcess = kernel32.GetCurrentProcess
            GetCurrentProcess.restype = wt.HANDLE
            GetPriorityClass = kernel32.GetPriorityClass
            GetPriorityClass.argtypes = [wt.HANDLE]
            GetPriorityClass.restype = wt.DWORD
            value = int(GetPriorityClass(GetCurrentProcess()))
            names = {
                0x00000040: "idle",
                0x00004000: "below_normal",
                0x00000020: "normal",
                0x00008000: "above_normal",
                0x00000080: "high",
                0x00000100: "realtime",
            }
            return names.get(value, f"0x{value:x}"), "WinAPI"
        except Exception:
            pass
    return "unavailable", "unavailable"


def _capture_callable_text(fn) -> tuple[str, str | None]:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            result = fn()
    except Exception as exc:
        return "", str(exc)
    text = buf.getvalue().strip()
    if not text and result is not None:
        text = repr(result)
    return text, None


def _print_compact_diagnostic_block(label: str, text: str, *, max_lines: int = 48, max_chars: int = 260) -> None:
    lines = [line.rstrip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        print(f"[DSP RUNTIME] {label}: unavailable")
        return
    print(f"[DSP RUNTIME] {label}:")
    for line in lines[:max_lines]:
        clean = line.strip()
        if len(clean) > max_chars:
            clean = clean[: max_chars - 3] + "..."
        print(f"[DSP RUNTIME]   {clean}")
    if len(lines) > max_lines:
        print(f"[DSP RUNTIME]   ... ({len(lines) - max_lines} more lines omitted)")


def _log_dsp_runtime_diagnostics_once(
    cfg_dict: dict[str, Any],
    dsp_cfg: RealtimeDSPConfig,
    diagnostics_cfg: DspDiagnosticsConfig,
) -> None:
    global _DSP_RUNTIME_DIAGNOSTICS_LOGGED
    if _DSP_RUNTIME_DIAGNOSTICS_LOGGED or not diagnostics_cfg.log_cpu_runtime:
        return
    _DSP_RUNTIME_DIAGNOSTICS_LOGGED = True

    try:
        logical_cpus = int(os.cpu_count() or 1)
    except Exception:
        logical_cpus = 1
    affinity_cpus, affinity_source = _effective_process_affinity()
    visible_cpus = len(affinity_cpus) if affinity_cpus is not None else logical_cpus
    priority_value, priority_source = _effective_process_priority()
    process_cfg = cfg_dict.get("process", {}) or {}
    affinity_cfg = process_cfg.get("affinity", {}) or {}
    priority_cfg = process_cfg.get("priority", {}) or {}
    pyfftw_cache = "unknown"
    try:
        is_enabled = getattr(pyfftw.interfaces.cache, "is_enabled", None)
        pyfftw_cache = str(bool(is_enabled())) if callable(is_enabled) else "enabled"
    except Exception:
        pass

    # Runtime diagnostics verify what the local Ryzen process can actually use
    # (affinity, FFT workers, SIMD-dispatched NumPy code, JIT), instead of
    # assuming AVX512 or all cores are exposed by the OS/configuration.
    try:
        platform_text = platform.platform()
    except Exception:
        platform_text = "unavailable"
    print(
        "[DSP RUNTIME] "
        f"platform={platform_text} os={os.name} python={sys.version.split()[0]} "
        f"pid={os.getpid()}"
    )
    print(
        "[DSP RUNTIME] "
        f"logical_cpus={logical_cpus} visible_by_affinity={visible_cpus} "
        f"effective_affinity={_format_cpu_list(affinity_cpus)} source={affinity_source}"
    )
    print(
        "[DSP RUNTIME] "
        f"fft_workers={int(getattr(dsp_cfg, 'fft_workers', 1))} "
        f"pyfftw_backend_active={bool(_PYFFTW_BACKEND_ACTIVE)} "
        f"pyfftw_version={getattr(pyfftw, '__version__', 'unknown')} "
        f"pyfftw_cache={pyfftw_cache}"
    )
    print(
        "[DSP RUNTIME] "
        f"process.affinity enabled={_to_bool(affinity_cfg.get('enabled', False), False)} "
        f"requested_dsp={affinity_cfg.get('dsp', 'auto')} "
        f"process.priority enabled={_to_bool(priority_cfg.get('enabled', False), False)} "
        f"requested_dsp={priority_cfg.get('dsp', 'normal')} "
        f"effective_priority={priority_value} source={priority_source}"
    )
    status = cfar_numba_runtime_status()
    angle_power_status = angle_power_numba_runtime_status()
    if _numba is None:
        print(f"[DSP RUNTIME] numba=unavailable error={_NUMBA_IMPORT_ERROR}")
    else:
        num_threads = "unknown"
        threading_layer = "unknown"
        try:
            num_threads = str(_numba.get_num_threads())
        except Exception:
            pass
        try:
            threading_layer = str(_numba.threading_layer())
        except Exception as exc:
            threading_layer = f"uninitialized ({exc})"
        numba_config = getattr(_numba, "config", None)
        cpu_name = getattr(numba_config, "CPU_NAME", None) if numba_config is not None else None
        cpu_features = getattr(numba_config, "CPU_FEATURES", None) if numba_config is not None else None
        print(
            "[DSP RUNTIME] "
            f"numba_version={status['numba_version']} numba_threads={num_threads} "
            f"threading_layer={threading_layer} cpu_name={cpu_name or 'auto'} "
            f"cpu_features={cpu_features or 'auto'} cfar_jit_enabled={status['enabled']} "
            f"cfar_self_checked={status['self_checked']} "
            f"angle_power_jit_enabled={angle_power_status['enabled']} "
            f"angle_power_threads={angle_power_status['threads']} "
            f"cfar_disabled_reason={status['disabled_reason'] or 'none'}"
        )

    show_runtime = getattr(np, "show_runtime", None)
    if callable(show_runtime):
        runtime_text, runtime_err = _capture_callable_text(show_runtime)
        if runtime_err is None:
            _print_compact_diagnostic_block("numpy.show_runtime", runtime_text, max_lines=32)
        else:
            print(f"[DSP RUNTIME] numpy.show_runtime unavailable: {runtime_err}")
    else:
        print("[DSP RUNTIME] numpy.show_runtime unavailable")

    show_config = getattr(np, "show_config", None)
    if callable(show_config):
        config_text, config_err = _capture_callable_text(show_config)
        if config_err is None:
            _print_compact_diagnostic_block("numpy.show_config", config_text, max_lines=64)
        else:
            print(f"[DSP RUNTIME] numpy.show_config unavailable: {config_err}")
    else:
        print("[DSP RUNTIME] numpy.show_config unavailable")


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
    display_heatmap_mode: str = "power_xy",
    gui_heat_alpha_views: tuple[np.ndarray, np.ndarray] | None = None,
    display_viewport: DisplayViewport | None = None,
    display_zoom_cfg: DisplayZoomConfig | None = None,
    display_zoom_runtime: DisplayZoomRuntime | None = None,
    frame_seq: int = 0,
    gui_angle_diag_views: tuple[np.ndarray, np.ndarray] | None = None,
    gui_doppler_diag_views: tuple[np.ndarray, np.ndarray] | None = None,
    angle_diag_out_buf: np.ndarray | None = None,
    doppler_diag_out_buf: np.ndarray | None = None,
    display_normalization_reference_db_out: Synchronized | None = None,
    display_power_normalized_out: Synchronized | None = None,
    publish_applied_viewport=None,
) -> tuple[np.ndarray | None, list[Detection]]:
    """Elabora un batch IQ e pubblica l'ultima immagine nella GUI.

    Il batch percorre tre branche indipendenti dopo la range FFT: detection
    statica, detection mobile e display. La funzione restituisce l'EMA della
    heatmap e le detection pulite, ma la GUI legge i double buffer condivisi.
    """
    try:
        display_mode = _normalize_display_heatmap_mode(display_heatmap_mode)
        display_zoom_cfg_eff = display_zoom_cfg or getattr(dsp_cfg, "display_zoom", DisplayZoomConfig())
        home_viewport = (
            display_zoom_runtime.home_viewport
            if display_zoom_runtime is not None
            else build_display_viewport(
                x_min_m=-float(display_x_max_m),
                x_max_m=float(display_x_max_m),
                y_min_m=0.0,
                y_max_m=float(display_y_max_m),
                dr_m=float(dsp_cfg.c * dsp_cfg.fs / (2.0 * dsp_cfg.slope * dsp_cfg.nfft_range)),
                seq=0,
            )
        )
        active_viewport = display_viewport or home_viewport
        active_viewport_sig = display_viewport_signature(active_viewport)
        home_viewport_sig = display_viewport_signature(home_viewport)
        zoom_active = bool(
            display_zoom_cfg_eff.enabled
            and active_viewport_sig is not None
            and active_viewport_sig != home_viewport_sig
            and float(active_viewport.zoom_level) > 1.0001
        )
        if (
            display_zoom_runtime is not None
            and display_mode != "range_angle_moving"
            and display_zoom_runtime.last_viewport_signature is not None
            and display_zoom_runtime.last_viewport_signature != active_viewport_sig
        ):
            heatmap_ema = None
        home_width_m = max(1e-6, float(home_viewport.x_max_m - home_viewport.x_min_m))
        home_height_m = max(1e-6, float(home_viewport.y_max_m - home_viewport.y_min_m))
        cur_width_m = max(1e-6, float(active_viewport.x_max_m - active_viewport.x_min_m))
        cur_height_m = max(1e-6, float(active_viewport.y_max_m - active_viewport.y_min_m))
        zoom_width_scale = float(home_width_m / cur_width_m)
        zoom_height_scale = float(home_height_m / cur_height_m)
        zoom_range_nfft = int(dsp_cfg.nfft_range)
        zoom_angle_nfft = int(dsp_cfg.nfft_angle)

        # Raw complex stream -> Reshape -> radar tensor [frame, loop, tx, sample, rx].
        data = raw_buffer.reshape(n_frames,dsp_cfg.chirps // dsp_cfg.tx,dsp_cfg.tx,dsp_cfg.samples,dsp_cfg.rx,)

        # Range-FFT pre-processing: static mean subtraction 
        data = subtract_selected_mean(data, mean_before_range_fft)

        # Apply range window (broadcasting over all non-range dimensions) to reduce sidelobes before the range FFT.
        if apply_range_window:
            data *= w_range

        zoom_range_fft_input: np.ndarray | None = None
        if zoom_active:
            zoom_range_nfft = _quantize_zoom_nfft(
                int(dsp_cfg.nfft_range),
                int(display_zoom_cfg_eff.max_zoom_nfft_range),
                zoom_height_scale,
            )
            zoom_angle_nfft = _quantize_zoom_nfft(
                int(dsp_cfg.nfft_angle),
                int(display_zoom_cfg_eff.max_zoom_nfft_angle),
                zoom_width_scale,
            )
            if zoom_range_nfft > int(dsp_cfg.nfft_range):
                zoom_range_fft_input = np.array(data, dtype=np.complex64, copy=True)

        # Range FFT with optional zero-padding and in-place computation to save memory.
        range_fft_common = fft.fft(data,n=dsp_cfg.nfft_range,axis=3,workers=dsp_cfg.fft_workers,overwrite_x=True,)
        zero_after_range_fft_bins = int(getattr(dsp_cfg, "zero_after_range_fft_bins", 0))
        if zero_after_range_fft_bins > 0:
            zero_bins = min(zero_after_range_fft_bins, int(range_fft_common.shape[3]))
            range_fft_common[:, :, :, :zero_bins, :] = np.complex64(0.0)

        # Range bin to physical range conversion factor (meters per bin).
        range_bin_m = dsp_cfg.c * dsp_cfg.fs / (2.0 * dsp_cfg.slope * dsp_cfg.nfft_range)
        zoom_range_bin_m = dsp_cfg.c * dsp_cfg.fs / (2.0 * dsp_cfg.slope * zoom_range_nfft)

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
        need_moving_doppler_branch = bool(detection_moving_cfg.enabled or display_mode == "range_angle_moving")
        if need_moving_doppler_branch:
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

        static_virtual_array_work_buf = None
        static_virtual_array_flat_work_buf = None
        static_virtual_array_may_alias_source = False
        if detection_static_cfg.enabled:
            static_virtual_array_work_buf = (
                None
                if virtual_array_work_buf is None
                else virtual_array_work_buf[
                    :n_frames,
                    : int(range_fft_detection_static.shape[1]),
                    :processing_max_bin,
                    :,
                    :,
                ]
            )
            static_virtual_array_flat_work_buf = (
                None
                if virtual_array_flat_work_buf is None
                else virtual_array_flat_work_buf[
                    :n_frames,
                    : int(range_fft_detection_static.shape[1]),
                    :processing_max_bin,
                    :,
                ]
            )
            static_virtual_array_may_alias_source = _virtual_array_may_alias_range_fft(
                range_fft_detection_static,
                max_bin=processing_max_bin,
                geometry=virtual_array_geometry,
                work_buf=static_virtual_array_work_buf,
            )

        # A post-range branch with no background state is deterministic for a
        # frame.  When static detection and display request that exact branch,
        # calculate it once and let the per-frame angle-map cache reuse its
        # result below.  Background subtraction deliberately stays isolated:
        # its state belongs to a branch even if two configurations look alike.
        # The whole reuse is also disabled when the static virtual array can
        # alias its source: in that rare layout, the angular path may process
        # the cube in place, so preserving the two historical branches is the
        # only result-invariant behaviour.
        share_static_display_branch = bool(
            detection_static_cfg.enabled
            and detection_static_post_range_fft_filters == display_post_range_fft_filters
            and bool(apply_detection_loop_average_after_background)
            == bool(apply_display_loop_average_after_background)
            and not detection_static_post_range_fft_filters.background_subtraction.enabled
            and not static_virtual_array_may_alias_source
        )
        if share_static_display_branch:
            range_fft_display = range_fft_detection_static
        else:
            range_fft_display_in = (
                range_fft_common.copy()
                if _branch_needs_copy(display_post_range_fft_filters)
                else range_fft_common
            )
            # range_fft_in -> slow-time_filter -> selected mean -> background
            # subtraction -> optional loop average -> range_fft_display.
            range_fft_display = apply_post_range_fft_filters(
                range_fft_display_in,
                display_post_range_fft_filters,
                bg_state=display_bg_state,
                fft_workers=dsp_cfg.fft_workers,
                apply_loop_average_after_background=apply_display_loop_average_after_background,
            )

        # The cache exists only for this batch.  An entry is valid exclusively
        # when both users hold the very same post-filter cube; object identity
        # deliberately prevents sharing results from branches with independent
        # background states or different filters.
        angle_heatmap_frame_cache: list[tuple[np.ndarray, int, np.ndarray]] = []

        def _cache_angle_heatmap(source_cube: np.ndarray, max_bin: int, heatmap: np.ndarray) -> None:
            if heatmap.ndim != 2 or int(max_bin) <= 0:
                return
            angle_heatmap_frame_cache.append((source_cube, int(max_bin), heatmap))

        def _cached_angle_heatmap(source_cube: np.ndarray, max_bin: int) -> np.ndarray | None:
            required_bins = max(0, int(max_bin))
            for cached_cube, cached_bins, cached_heatmap in angle_heatmap_frame_cache:
                if cached_cube is not source_cube or cached_bins < required_bins:
                    continue
                if cached_heatmap.shape[0] < required_bins:
                    continue
                return cached_heatmap[:required_bins, :]
            return None

        # Tracking path (physical data): no display EMA/blur/normalization.
        detections_static: list[Detection] = []
        detections_moving: list[Detection] = []
        range_doppler_detection: np.ndarray | None = None
        # If enabled, detect static targets from the range-FFT cube and estimate their angle/range.
        if detection_static_cfg.enabled:
            detections_static, static_detection_heatmap = detect_static_targets(
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
                virtual_array_work_buf=static_virtual_array_work_buf,
                virtual_array_flat_work_buf=static_virtual_array_flat_work_buf,
            )
            _cache_angle_heatmap(
                range_fft_detection_static,
                processing_max_bin,
                static_detection_heatmap,
            )
        
        if detection_moving_cfg.enabled:
            doppler_cube, range_doppler_detection = compute_range_doppler(
                range_fft_detection_moving,
                max_bin=processing_max_bin,
                dsp_cfg=dsp_cfg,
                moving_cfg=detection_moving_cfg,
                w_doppler=w_doppler,
                apply_doppler_window=apply_doppler_window,
                doppler_work_buf=(
                    None
                    if doppler_work_buf is None
                    else doppler_work_buf[:n_frames, : int(range_fft_detection_moving.shape[1]), :, :processing_max_bin, :]
                ),
            )
            detections_moving = detect_moving_targets(
                range_doppler_detection,
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

        publish_fill_value = np.float32(-120.0)
        fallback_used = False
        view_alpha: np.ndarray | None = None
        normalization_reference_db = float("nan")
        power_display_normalized = False
        viewport_range_max_m = max(0.0, float(active_viewport.range_max_bin_f) * float(range_bin_m))

        def _viewport_max_bin_for(dr_local: float, available_bins: int) -> int:
            available = max(1, int(available_bins))
            if not np.isfinite(dr_local) or float(dr_local) <= 0.0:
                return max(
                    1,
                    min(
                        int(math.ceil(float(active_viewport.range_max_bin_f))),
                        available,
                    ),
                )
            return max(
                1,
                min(
                    int(math.ceil(float(viewport_range_max_m) / float(dr_local))),
                    available,
                ),
            )

        active_display_max_bin = _viewport_max_bin_for(range_bin_m, int(range_fft_display.shape[3]))

        def _projection_lut_for(
            angle_axis_local: np.ndarray,
            dr_local: float,
            projection_mode_local: str,
        ) -> dict[str, Any] | None:
            if display_zoom_runtime is not None:
                return _get_cached_projection_lut(
                    display_zoom_runtime,
                    gui_h=gui_h,
                    gui_w=gui_w,
                    viewport=active_viewport,
                    dr_m=dr_local,
                    angle_axis_deg=angle_axis_local,
                    projection_mode=projection_mode_local,
                    projection_interp=display_projection_cfg.projection_interp,
                )
            if (
                projection_mode_local == display_projection_cfg.projection_mode
                and display_viewport_signature(active_viewport) == home_viewport_sig
            ):
                return display_projection_lut
            return build_display_projection_lut(
                gui_h=gui_h,
                gui_w=gui_w,
                x_max_m=float(active_viewport.x_max_m),
                y_max_m=float(active_viewport.y_max_m),
                dr_m=float(dr_local),
                angle_axis_deg=angle_axis_local,
                projection_mode=projection_mode_local,
                projection_interp=display_projection_cfg.projection_interp,
                x_min_m=float(active_viewport.x_min_m),
                y_min_m=float(active_viewport.y_min_m),
            )

        def _project_view(
            src: np.ndarray,
            *,
            angle_axis_local: np.ndarray,
            dr_local: float,
            projection_mode_local: str,
            out_buf: np.ndarray | None,
            fill_value_local: float,
        ) -> np.ndarray:
            return project_heatmap_for_display(
                src,
                angle_axis_deg=angle_axis_local,
                dr_m=dr_local,
                gui_h=gui_h,
                gui_w=gui_w,
                y_max_m=float(active_viewport.y_max_m),
                x_max_m=float(active_viewport.x_max_m),
                projection_mode=projection_mode_local,
                projection_interp=display_projection_cfg.projection_interp,
                out=out_buf,
                fill_value=float(fill_value_local),
                precomputed_lut=_projection_lut_for(angle_axis_local, dr_local, projection_mode_local),
                x_min_m=float(active_viewport.x_min_m),
                y_min_m=float(active_viewport.y_min_m),
            )

        zoom_can_improve = bool(
            zoom_active
            and (
                int(zoom_range_nfft) > int(dsp_cfg.nfft_range)
                or int(zoom_angle_nfft) > int(dsp_cfg.nfft_angle)
            )
        )
        zoom_should_recompute = False
        if zoom_can_improve:
            zoom_should_recompute = should_recompute_display_zoom(
                display_zoom_runtime,
                active_viewport_sig=active_viewport_sig,
                display_mode=display_mode,
                display_zoom_cfg=display_zoom_cfg_eff,
                now_s=time.perf_counter(),
            )

        # This is the un-smoothed base display map.  The angle diagnostic is
        # defined from exactly this map, not from the optional EMA/blur or a
        # zoomed replacement.
        display_angle_heatmap_raw: np.ndarray | None = None

        if display_mode == "range_angle_moving":
            moving_display_max_bin = max(
                1,
                min(active_display_max_bin, max(1, int(range_fft_detection_moving.shape[3]))),
            )
            velocity_ra, alpha_ra = compute_range_angle_moving_velocity_map(
                range_fft_detection_moving,
                max_bin=moving_display_max_bin,
                dsp_cfg=dsp_cfg,
                w_doppler=w_doppler,
                w_angle=w_angle,
                apply_doppler_window=apply_doppler_window,
                apply_angle_window=apply_angle_window,
                doppler_fft_shift=bool(detection_moving_cfg.doppler_fft_shift),
                doppler_axis_mps=doppler_axis_mps,
                tdm_mimo_compensation_table=tdm_mimo_compensation_table,
                virtual_array_geometry=virtual_array_geometry,
                angle_steering=angle_steering,
                angle_axis_deg=angle_axis_deg,
                doppler_work_buf=(
                    None
                    if doppler_work_buf is None
                    else doppler_work_buf[:n_frames, : int(range_fft_detection_moving.shape[1]), :, :moving_display_max_bin, :]
                ),
                virtual_array_work_buf=(
                    None
                    if virtual_array_work_buf is None
                    else virtual_array_work_buf[
                        :n_frames,
                        : int(range_fft_detection_moving.shape[1]),
                        :moving_display_max_bin,
                        :,
                        :,
                    ]
                ),
                virtual_array_flat_work_buf=(
                    None
                    if virtual_array_flat_work_buf is None
                    else virtual_array_flat_work_buf[
                        :n_frames,
                        : int(range_fft_detection_moving.shape[1]),
                        :moving_display_max_bin,
                        :,
                    ]
                ),
            )
            velocity_ra_for_projection = velocity_ra.astype(np.float32, copy=True)
            velocity_ra_for_projection[alpha_ra <= np.float32(0.0)] = np.float32(np.nan)
            view_display = _project_view(
                velocity_ra_for_projection,
                angle_axis_local=angle_axis_deg,
                dr_local=range_bin_m,
                projection_mode_local="cartesian",
                out_buf=heatmap_db_work_buf,
                fill_value_local=0.0,
            )
            view_alpha = _project_view(
                alpha_ra,
                angle_axis_local=angle_axis_deg,
                dr_local=range_bin_m,
                projection_mode_local="cartesian",
                out_buf=None,
                fill_value_local=0.0,
            )
            publish_fill_value = np.float32(0.0)

            if zoom_can_improve and not zoom_should_recompute:
                fallback_used = True
                if (
                    display_zoom_cfg_eff.fallback_mode == "cached_frame"
                    and display_zoom_runtime is not None
                    and display_zoom_runtime.last_view_db is not None
                    and display_zoom_runtime.last_viewport_signature == active_viewport_sig
                    and str(display_zoom_runtime.last_mode) == str(display_mode)
                    and display_zoom_runtime.last_view_db.shape == view_display.shape
                ):
                    np.copyto(view_display, display_zoom_runtime.last_view_db, casting="unsafe")
                    if view_alpha is not None and display_zoom_runtime.last_view_alpha is not None and display_zoom_runtime.last_view_alpha.shape == view_alpha.shape:
                        np.copyto(view_alpha, display_zoom_runtime.last_view_alpha, casting="unsafe")
            elif zoom_should_recompute and display_zoom_runtime is not None:
                zoom_t0 = time.perf_counter()
                try:
                    if zoom_range_fft_input is not None:
                        zoom_range_fft_common = fft.fft(
                            zoom_range_fft_input,
                            n=int(zoom_range_nfft),
                            axis=3,
                            workers=dsp_cfg.fft_workers,
                            overwrite_x=True,
                        )
                        if zero_after_range_fft_bins > 0:
                            zero_bins = min(zero_after_range_fft_bins, int(zoom_range_fft_common.shape[3]))
                            zoom_range_fft_common[:, :, :, :zero_bins, :] = np.complex64(0.0)
                        zoom_range_fft_detection_in = (
                            zoom_range_fft_common.copy()
                            if _branch_needs_copy(detection_moving_pre_doppler_filters)
                            else zoom_range_fft_common
                        )
                        moving_bg_key = tuple(int(x) for x in zoom_range_fft_detection_in.shape[1:])
                        zoom_moving_bg_state = display_zoom_runtime.moving_bg_state_cache.setdefault(
                            moving_bg_key,
                            BackgroundSubtractionState(),
                        )
                        zoom_range_fft_detection = apply_post_range_fft_filters(
                            zoom_range_fft_detection_in,
                            detection_moving_pre_doppler_filters,
                            bg_state=zoom_moving_bg_state,
                            fft_workers=dsp_cfg.fft_workers,
                            apply_loop_average_after_background=False,
                        )
                    else:
                        zoom_range_fft_detection = range_fft_detection_moving

                    zoom_angle_axis_full, zoom_steering_full = _get_cached_angle_axis_and_steering(
                        display_zoom_runtime,
                        geometry=virtual_array_geometry,
                        virtual_ant=int(dsp_cfg.virtual_ant),
                        nfft_angle=int(zoom_angle_nfft),
                    )
                    roi_idx = _viewport_angle_roi_indices(zoom_angle_axis_full, active_viewport)
                    if roi_idx.size > 0 and roi_idx.size < int(zoom_angle_axis_full.size):
                        zoom_angle_axis = zoom_angle_axis_full[roi_idx].astype(np.float32, copy=False)
                        zoom_steering = zoom_steering_full[:, roi_idx].astype(np.complex64, copy=False)
                        zoom_dsp_cfg = replace(dsp_cfg, nfft_angle=int(zoom_angle_axis.size))
                    else:
                        zoom_angle_axis = zoom_angle_axis_full
                        zoom_steering = zoom_steering_full
                        zoom_dsp_cfg = replace(dsp_cfg, nfft_angle=int(zoom_angle_nfft))
                    zoom_dsp_cfg = replace(zoom_dsp_cfg, nfft_range=int(zoom_range_nfft))

                    zoom_display_max_bin = max(
                        1,
                        _viewport_max_bin_for(zoom_range_bin_m, int(zoom_range_fft_detection.shape[3])),
                    )
                    velocity_zoom, alpha_zoom = compute_range_angle_moving_velocity_map(
                        zoom_range_fft_detection,
                        max_bin=zoom_display_max_bin,
                        dsp_cfg=zoom_dsp_cfg,
                        w_doppler=w_doppler,
                        w_angle=w_angle,
                        apply_doppler_window=apply_doppler_window,
                        apply_angle_window=apply_angle_window,
                        doppler_fft_shift=bool(detection_moving_cfg.doppler_fft_shift),
                        doppler_axis_mps=doppler_axis_mps,
                        tdm_mimo_compensation_table=tdm_mimo_compensation_table,
                        virtual_array_geometry=virtual_array_geometry,
                        angle_steering=zoom_steering,
                        angle_axis_deg=zoom_angle_axis,
                        doppler_work_buf=None,
                        virtual_array_work_buf=None,
                        virtual_array_flat_work_buf=None,
                    )
                    velocity_zoom_for_projection = velocity_zoom.astype(np.float32, copy=True)
                    velocity_zoom_for_projection[alpha_zoom <= np.float32(0.0)] = np.float32(np.nan)
                    view_display = _project_view(
                        velocity_zoom_for_projection,
                        angle_axis_local=zoom_angle_axis,
                        dr_local=zoom_range_bin_m,
                        projection_mode_local="cartesian",
                        out_buf=heatmap_db_work_buf,
                        fill_value_local=0.0,
                    )
                    view_alpha = _project_view(
                        alpha_zoom,
                        angle_axis_local=zoom_angle_axis,
                        dr_local=zoom_range_bin_m,
                        projection_mode_local="cartesian",
                        out_buf=None,
                        fill_value_local=0.0,
                    )
                    display_zoom_runtime.last_compute_ms = float((time.perf_counter() - zoom_t0) * 1000.0)
                    display_zoom_runtime.last_compute_t_s = time.perf_counter()
                except Exception as exc:
                    fallback_used = True
                    error_signature = (type(exc).__name__, str(exc))
                    if display_zoom_runtime.last_error_signature != error_signature:
                        print(f"[DSP WARN] moving display zoom fallback: {type(exc).__name__}: {exc}")
                        display_zoom_runtime.last_error_signature = error_signature
            if view_display.size > 0 and dsp_cfg.debug_stats:
                try:
                    valid_alpha = view_alpha > np.float32(0.0) if view_alpha is not None else np.ones(view_display.shape, dtype=bool)
                    finite_view = view_display[valid_alpha]
                    raw_min = float(np.min(finite_view)) if finite_view.size > 0 else float("nan")
                    raw_max = float(np.max(finite_view)) if finite_view.size > 0 else float("nan")
                    with stat_raw_min_db.get_lock():
                        stat_raw_min_db.value = raw_min
                    with stat_raw_max_db.get_lock():
                        stat_raw_max_db.value = raw_max
                    with stat_norm_min_db.get_lock():
                        stat_norm_min_db.value = raw_min
                    with stat_norm_max_db.get_lock():
                        stat_norm_max_db.value = raw_max
                except Exception:
                    pass
        else:
            # Build the virtual array after trimming range bins to limit memory traffic.
            # [frame, loop, range_bin, virtual_ant]
            debug_angle_axis = angle_axis_deg
            heatmap = (
                _cached_angle_heatmap(range_fft_display, active_display_max_bin)
                if share_static_display_branch
                else None
            )
            if heatmap is None:
                virtual_array = _build_virtual_array_from_range_fft(
                    range_fft_display,
                    max_bin=active_display_max_bin,
                    dsp_cfg=dsp_cfg,
                    geometry=virtual_array_geometry,
                    work_buf=(
                        None
                        if virtual_array_work_buf is None
                        else virtual_array_work_buf[
                            :n_frames,
                            : int(range_fft_display.shape[1]),
                            :active_display_max_bin,
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
                            :active_display_max_bin,
                            :,
                        ]
                    ),
                )

                if apply_angle_window:
                    virtual_array *= w_angle

                heatmap = compute_angle_heatmap(
                    virtual_array,
                    angle_cfg=angle_processing,
                    dsp_cfg=replace(dsp_cfg, nfft_angle=int(dsp_cfg.nfft_angle)),
                    angle_steering=angle_steering,
                    geometry=virtual_array_geometry,
                    ant_spacing=virtual_array_geometry.uniform_spacing_lambda,
                )
                _cache_angle_heatmap(range_fft_display, active_display_max_bin, heatmap)
            display_angle_heatmap_raw = heatmap

            if not heatmap_ema_cfg.enabled:
                heatmap_ema = heatmap
            elif heatmap_ema is None:
                heatmap_ema = heatmap
            else:
                heatmap_ema *= (1.0 - heatmap_ema_cfg.alpha)
                heatmap_ema += (heatmap_ema_cfg.alpha * heatmap)
            heatmap_ema = apply_heatmap_spatial_filter(heatmap_ema, heatmap_spatial_filter_cfg)
            debug_heatmap = heatmap_ema

            view_display = _project_view(
                heatmap_ema,
                angle_axis_local=angle_axis_deg,
                dr_local=range_bin_m,
                projection_mode_local=display_projection_cfg.projection_mode,
                out_buf=heatmap_db_work_buf,
                fill_value_local=0.0,
            )
            np.add(view_display, np.float32(1e-12), out=view_display)
            np.log10(view_display, out=view_display)
            view_display *= np.float32(10.0)

            if zoom_can_improve and not zoom_should_recompute:
                fallback_used = True
                if (
                    display_zoom_cfg_eff.fallback_mode == "cached_frame"
                    and display_zoom_runtime is not None
                    and display_zoom_runtime.last_view_db is not None
                    and display_zoom_runtime.last_viewport_signature == active_viewport_sig
                    and str(display_zoom_runtime.last_mode) == str(display_mode)
                    and display_zoom_runtime.last_view_db.shape == view_display.shape
                ):
                    np.copyto(view_display, display_zoom_runtime.last_view_db, casting="unsafe")
            elif zoom_should_recompute and display_zoom_runtime is not None:
                zoom_t0 = time.perf_counter()
                try:
                    if zoom_range_fft_input is not None:
                        zoom_range_fft_common = fft.fft(
                            zoom_range_fft_input,
                            n=int(zoom_range_nfft),
                            axis=3,
                            workers=dsp_cfg.fft_workers,
                            overwrite_x=True,
                        )
                        if zero_after_range_fft_bins > 0:
                            zero_bins = min(zero_after_range_fft_bins, int(zoom_range_fft_common.shape[3]))
                            zoom_range_fft_common[:, :, :, :zero_bins, :] = np.complex64(0.0)
                        zoom_range_fft_display_in = (
                            zoom_range_fft_common.copy()
                            if _branch_needs_copy(display_post_range_fft_filters)
                            else zoom_range_fft_common
                        )
                        display_bg_key = tuple(int(x) for x in zoom_range_fft_display_in.shape[1:])
                        zoom_display_bg_state = display_zoom_runtime.display_bg_state_cache.setdefault(
                            display_bg_key,
                            BackgroundSubtractionState(),
                        )
                        zoom_range_fft_display = apply_post_range_fft_filters(
                            zoom_range_fft_display_in,
                            display_post_range_fft_filters,
                            bg_state=zoom_display_bg_state,
                            fft_workers=dsp_cfg.fft_workers,
                            apply_loop_average_after_background=apply_display_loop_average_after_background,
                        )
                    else:
                        zoom_range_fft_display = range_fft_display

                    zoom_dsp_cfg = replace(dsp_cfg, nfft_range=int(zoom_range_nfft), nfft_angle=int(zoom_angle_nfft))
                    zoom_display_max_bin = max(
                        1,
                        _viewport_max_bin_for(zoom_range_bin_m, int(zoom_range_fft_display.shape[3])),
                    )
                    zoom_virtual_array = _build_virtual_array_from_range_fft(
                        zoom_range_fft_display,
                        max_bin=zoom_display_max_bin,
                        dsp_cfg=zoom_dsp_cfg,
                        geometry=virtual_array_geometry,
                        work_buf=None,
                        flat_work_buf=None,
                    )
                    if apply_angle_window:
                        zoom_virtual_array *= w_angle

                    if angle_processing.mode == "fft":
                        zoom_angle_axis, zoom_steering = _get_cached_angle_axis_and_steering(
                            display_zoom_runtime,
                            geometry=virtual_array_geometry,
                            virtual_ant=int(dsp_cfg.virtual_ant),
                            nfft_angle=int(zoom_angle_nfft),
                        )
                        zoom_heatmap = compute_angle_heatmap(
                            zoom_virtual_array,
                            angle_cfg=angle_processing,
                            dsp_cfg=zoom_dsp_cfg,
                            angle_steering=zoom_steering,
                            geometry=virtual_array_geometry,
                            ant_spacing=virtual_array_geometry.uniform_spacing_lambda,
                        )
                    else:
                        zoom_angle_axis_full, zoom_steering_full = _get_cached_angle_axis_and_steering(
                            display_zoom_runtime,
                            geometry=virtual_array_geometry,
                            virtual_ant=int(dsp_cfg.virtual_ant),
                            nfft_angle=int(zoom_angle_nfft),
                        )
                        roi_idx = _viewport_angle_roi_indices(zoom_angle_axis_full, active_viewport)
                        if roi_idx.size > 0 and roi_idx.size < int(zoom_angle_axis_full.size):
                            zoom_angle_axis = zoom_angle_axis_full[roi_idx].astype(np.float32, copy=False)
                            zoom_steering = zoom_steering_full[:, roi_idx].astype(np.complex64, copy=False)
                        else:
                            zoom_angle_axis = zoom_angle_axis_full
                            zoom_steering = zoom_steering_full
                        zoom_heatmap = compute_angle_heatmap(
                            zoom_virtual_array,
                            angle_cfg=angle_processing,
                            dsp_cfg=replace(zoom_dsp_cfg, nfft_angle=int(zoom_angle_axis.size)),
                            angle_steering=zoom_steering,
                            geometry=virtual_array_geometry,
                            ant_spacing=virtual_array_geometry.uniform_spacing_lambda,
                        )
                    debug_angle_axis = zoom_angle_axis

                    zoom_heatmap_ema = display_zoom_runtime.heatmap_ema
                    if not heatmap_ema_cfg.enabled:
                        zoom_heatmap_ema = zoom_heatmap
                    elif zoom_heatmap_ema is None or zoom_heatmap_ema.shape != zoom_heatmap.shape:
                        zoom_heatmap_ema = np.array(zoom_heatmap, dtype=np.float32, copy=True)
                    else:
                        zoom_heatmap_ema *= (1.0 - heatmap_ema_cfg.alpha)
                        zoom_heatmap_ema += (heatmap_ema_cfg.alpha * zoom_heatmap)
                    zoom_heatmap_ema = apply_heatmap_spatial_filter(zoom_heatmap_ema, heatmap_spatial_filter_cfg)
                    display_zoom_runtime.heatmap_ema = zoom_heatmap_ema
                    debug_heatmap = zoom_heatmap_ema
                    view_display = _project_view(
                        zoom_heatmap_ema,
                        angle_axis_local=zoom_angle_axis,
                        dr_local=zoom_range_bin_m,
                        projection_mode_local=display_projection_cfg.projection_mode,
                        out_buf=heatmap_db_work_buf,
                        fill_value_local=0.0,
                    )
                    np.add(view_display, np.float32(1e-12), out=view_display)
                    np.log10(view_display, out=view_display)
                    view_display *= np.float32(10.0)
                    display_zoom_runtime.last_compute_ms = float((time.perf_counter() - zoom_t0) * 1000.0)
                    display_zoom_runtime.last_compute_t_s = time.perf_counter()
                except Exception as exc:
                    fallback_used = True
                    error_signature = (type(exc).__name__, str(exc))
                    if display_zoom_runtime.last_error_signature != error_signature:
                        print(f"[DSP WARN] power display zoom fallback: {type(exc).__name__}: {exc}")
                        display_zoom_runtime.last_error_signature = error_signature

            if debug_print_top_peaks:
                print(
                    _format_debug_top_peaks_range_angle(
                        debug_heatmap,
                        angle_axis_deg=debug_angle_axis,
                        range_bin_m=(zoom_range_bin_m if debug_angle_axis is not angle_axis_deg else range_bin_m),
                        top_k=debug_top_peaks_count,
                        range_min_m=debug_top_peaks_range_min_m,
                        range_max_m=debug_top_peaks_range_max_m,
                        angle_min_deg=debug_top_peaks_angle_min_deg,
                        angle_max_deg=debug_top_peaks_angle_max_deg,
                    )
                )
                print(
                    _format_debug_top_peaks_xy(
                        view_display,
                        x_max_m=float(active_viewport.x_max_m),
                        y_max_m=float(active_viewport.y_max_m),
                        top_k=debug_top_peaks_count,
                        range_min_m=debug_top_peaks_range_min_m,
                        range_max_m=debug_top_peaks_range_max_m,
                        angle_min_deg=debug_top_peaks_angle_min_deg,
                        angle_max_deg=debug_top_peaks_angle_max_deg,
                    )
                )

            if view_display.size > 0:
                norm_ref_db = _normalization_reference_db(
                    heatmap_ema,
                    skip_range_bins=int(getattr(dsp_cfg, "normalize_skip_range_bins", 0)),
                )
                raw_max = float(np.max(view_display))
                normalization_reference_db = (
                    raw_max if norm_ref_db is None else float(norm_ref_db)
                )
                if dsp_cfg.debug_stats:
                    raw_min = float(np.min(view_display))
                    norm_max = 0.0
                    norm_peak = normalization_reference_db
                    norm_min = float(raw_min - norm_peak)
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
                    view_display -= normalization_reference_db
                    power_display_normalized = True

        # The normal power display and its angle diagnostic consume the same
        # un-smoothed range-angle map.  Reuse it whenever its FFT grid matches
        # the diagnostic grid.  Moving-velocity display has no such map, and
        # non-standard diagnostic widths retain the original independent path.
        if angle_diag_out_buf is not None:
            diag_range_bins = min(
                int(active_display_max_bin),
                int(angle_diag_out_buf.shape[0]),
                int(range_fft_display.shape[3]),
            )
            diag_angle_power: np.ndarray | None = None
            if (
                display_angle_heatmap_raw is not None
                and int(display_angle_heatmap_raw.shape[1]) == int(angle_diag_out_buf.shape[1])
                and int(display_angle_heatmap_raw.shape[0]) >= diag_range_bins
            ):
                diag_angle_power = display_angle_heatmap_raw[:diag_range_bins, :]
            elif diag_range_bins > 0:
                diag_virtual_array = _build_virtual_array_from_range_fft(
                    range_fft_display,
                    max_bin=diag_range_bins,
                    dsp_cfg=dsp_cfg,
                    geometry=virtual_array_geometry,
                    work_buf=(
                        None
                        if virtual_array_work_buf is None
                        else virtual_array_work_buf[
                            :n_frames,
                            : int(range_fft_display.shape[1]),
                            :diag_range_bins,
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
                            :diag_range_bins,
                            :,
                        ]
                    ),
                )
                if apply_angle_window:
                    diag_virtual_array *= w_angle
                diag_angle_power = compute_angle_heatmap(
                    diag_virtual_array,
                    angle_cfg=angle_processing,
                    dsp_cfg=replace(dsp_cfg, nfft_angle=int(angle_diag_out_buf.shape[1])),
                    angle_steering=angle_steering,
                    geometry=virtual_array_geometry,
                    ant_spacing=virtual_array_geometry.uniform_spacing_lambda,
                )
            if diag_angle_power is None:
                angle_diag_out_buf.fill(np.float32(-120.0))
            else:
                _write_diagnostic_power_db(diag_angle_power, angle_diag_out_buf)

        # Moving detection already computes this exact range-Doppler map.  Its
        # range rows are independent, so the visual diagnostic can safely use
        # a prefix of the detection map.  If moving detection is disabled (or
        # has a smaller processing range), preserve the standalone diagnostic.
        if doppler_diag_out_buf is not None:
            diag_range_bins = min(
                int(active_display_max_bin),
                int(doppler_diag_out_buf.shape[0]),
                int(range_fft_detection_moving.shape[3]),
            )
            diag_range_doppler: np.ndarray | None = None
            if (
                range_doppler_detection is not None
                and int(range_doppler_detection.shape[0]) >= diag_range_bins
            ):
                diag_range_doppler = range_doppler_detection[:diag_range_bins, :]
            elif diag_range_bins > 0:
                _, diag_range_doppler = compute_range_doppler(
                    range_fft_detection_moving,
                    max_bin=diag_range_bins,
                    dsp_cfg=dsp_cfg,
                    moving_cfg=detection_moving_cfg,
                    w_doppler=w_doppler,
                    apply_doppler_window=apply_doppler_window,
                    doppler_work_buf=(
                        None
                        if doppler_work_buf is None
                        else doppler_work_buf[
                            :n_frames,
                            : int(range_fft_detection_moving.shape[1]),
                            :,
                            :diag_range_bins,
                            :,
                        ]
                    ),
                )
            if diag_range_doppler is None:
                doppler_diag_out_buf.fill(np.float32(-120.0))
            else:
                _write_diagnostic_power_db(diag_range_doppler, doppler_diag_out_buf)

        if display_zoom_runtime is not None:
            display_zoom_runtime.last_viewport_signature = active_viewport_sig
            display_zoom_runtime.last_mode = str(display_mode)
            display_zoom_runtime.last_fill_value = float(publish_fill_value)
            display_zoom_runtime.last_applied_meta = applied_viewport_meta_from_viewport(
                active_viewport,
                fallback_used=bool(fallback_used),
                frame_seq=int(frame_seq),
            )
            display_zoom_runtime.last_view_db = np.array(view_display, dtype=np.float32, copy=True)
            if view_alpha is None:
                display_zoom_runtime.last_view_alpha = None
            else:
                display_zoom_runtime.last_view_alpha = np.array(view_alpha, dtype=np.float32, copy=True)

        # Latest-wins publish to the GUI double buffer.
        with gui_lock:
            if publish_applied_viewport is not None and display_zoom_runtime is not None:
                publish_applied_viewport(display_zoom_runtime.last_applied_meta)
            if display_normalization_reference_db_out is not None:
                try:
                    with display_normalization_reference_db_out.get_lock():
                        display_normalization_reference_db_out.value = float(normalization_reference_db)
                except Exception:
                    pass
            if display_power_normalized_out is not None:
                try:
                    with display_power_normalized_out.get_lock():
                        display_power_normalized_out.value = 1 if power_display_normalized else 0
                except Exception:
                    pass
            prev_idx = int(gui_latest_idx.value)
            next_idx = 1 if prev_idx == 0 else 0
            dst = gui_heat_views[next_idx]
            dst.fill(publish_fill_value)
            flat = view_display.reshape(-1)
            n = min(dst.size, flat.size)
            if n > 0:
                dst[:n] = flat[:n]

            if gui_heat_alpha_views is not None:
                dst_alpha = gui_heat_alpha_views[next_idx]
                if view_alpha is None:
                    dst_alpha.fill(np.float32(1.0))
                else:
                    dst_alpha.fill(np.float32(0.0))
                    alpha_flat = view_alpha.reshape(-1)
                    n_alpha = min(dst_alpha.size, alpha_flat.size)
                    if n_alpha > 0:
                        dst_alpha[:n_alpha] = alpha_flat[:n_alpha]

            dst_prof = gui_profile_views[next_idx]
            dst_prof.fill(-120.0)
            prof_flat = profiles_out.reshape(-1)
            n_prof = min(dst_prof.size, prof_flat.size)
            if n_prof > 0:
                dst_prof[:n_prof] = prof_flat[:n_prof]
            if gui_angle_diag_views is not None and angle_diag_out_buf is not None:
                dst_angle = gui_angle_diag_views[next_idx]
                dst_angle.fill(-120.0)
                angle_flat = angle_diag_out_buf.reshape(-1)
                n_angle = min(dst_angle.size, angle_flat.size)
                if n_angle > 0:
                    dst_angle[:n_angle] = angle_flat[:n_angle]
            if gui_doppler_diag_views is not None and doppler_diag_out_buf is not None:
                dst_doppler = gui_doppler_diag_views[next_idx]
                dst_doppler.fill(-120.0)
                doppler_flat = doppler_diag_out_buf.reshape(-1)
                n_doppler_diag = min(dst_doppler.size, doppler_flat.size)
                if n_doppler_diag > 0:
                    dst_doppler[:n_doppler_diag] = doppler_flat[:n_doppler_diag]
            gui_latest_idx.value = next_idx
            gui_latest_seq.value = int(gui_latest_seq.value) + 1
        return heatmap_ema, tracking_detections

    except Exception as e:
        print(f"[DSP ERR] {e}")
        return heatmap_ema, []


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
    heatmap_view_mode: Synchronized | None = None,
    gui_alpha_dbuf=None,
    display_viewport_request=None,
    display_viewport_request_seq: Synchronized | None = None,
    display_viewport_request_lock=None,
    display_viewport_applied=None,
    display_viewport_applied_seq: Synchronized | None = None,
    display_viewport_applied_frame_seq: Synchronized | None = None,
    display_viewport_applied_fallback: Synchronized | None = None,
    display_viewport_applied_lock=None,
    display_home_viewport=None,
    display_home_viewport_lock=None,
    gui_angle_diag_dbuf=None,
    gui_doppler_diag_dbuf=None,
    angle_diag_w: int = 0,
    doppler_diag_w: int = 0,
    display_normalization_reference_db: Synchronized | None = None,
    display_power_normalized: Synchronized | None = None,
    slot_frame_time_s=None,
    ready_evt=None,
) -> None:
    """Loop del processo DSP: consuma slot RX, elabora e rilascia lo slot.

    La regola di proprietà è cruciale: uno slot torna a ``free_slots`` solo
    dopo che i dati sono stati copiati/elaborati, così RX non sovrascrive il
    frame ancora in uso dal DSP.
    """
    dsp_block = cfg_dict.get("dsp", {}) or {}
    selection = selection_from_yaml_dict(cfg_dict)
    cfar_numba_cfg = cfar_numba_from_yaml_dict(cfg_dict)
    angle_power_numba_cfg = angle_power_numba_from_yaml_dict(cfg_dict)
    diagnostics_cfg = dsp_diagnostics_from_yaml_dict(cfg_dict)
    configure_cfar_numba_runtime(cfar_numba_cfg, log=("cfar_numba" in dsp_block))
    configure_angle_power_numba_runtime(angle_power_numba_cfg, log=("angle_power_numba" in dsp_block))
    if warmup_angle_power_numba():
        print("[DSP NUMBA] angle-power reduction JIT warmed up.")
    mean_before_range_fft = mean_before_range_fft_from_yaml_dict(cfg_dict)
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
    display_zoom_cfg = display_zoom_from_yaml_dict(cfg_dict)
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
    dr = dsp_cfg.c * dsp_cfg.fs / (2.0 * dsp_cfg.slope * dsp_cfg.nfft_range)
    home_viewport = build_display_viewport(
        x_min_m=-float(display_x_max_m),
        x_max_m=float(display_x_max_m),
        y_min_m=0.0,
        y_max_m=float(display_y_max_m),
        dr_m=float(dr),
        seq=0,
    )

    def _read_home_viewport(default_viewport: DisplayViewport) -> DisplayViewport:
        if display_home_viewport is None:
            return default_viewport
        arr = None
        try:
            if display_home_viewport_lock is not None:
                with display_home_viewport_lock:
                    arr = np.asarray(display_home_viewport[:], dtype=np.float64)
            else:
                arr = np.asarray(display_home_viewport[:], dtype=np.float64)
        except Exception:
            return default_viewport
        if arr is None or int(arr.size) < 4:
            return default_viewport
        return build_display_viewport(
            x_min_m=float(arr[0]),
            x_max_m=float(arr[1]),
            y_min_m=float(arr[2]),
            y_max_m=float(arr[3]),
            dr_m=float(dr),
            seq=int(default_viewport.seq),
        )

    home_viewport = _read_home_viewport(home_viewport)
    display_projection_lut = build_display_projection_lut(
        gui_h=gui_h,
        gui_w=gui_w,
        x_max_m=display_x_max_m,
        y_max_m=display_y_max_m,
        dr_m=dr,
        angle_axis_deg=angle_axis_deg,
        projection_mode=display_projection_cfg.projection_mode,
        projection_interp=display_projection_cfg.projection_interp,
        x_min_m=float(home_viewport.x_min_m),
        y_min_m=float(home_viewport.y_min_m),
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

    processing_range_max_m = float(
        dsp_cfg.range_max_processing_m if dsp_cfg.range_max_processing_m > 0.0 else dsp_cfg.range_max_display
    )
    processing_max_bin = int(np.floor(processing_range_max_m / dr))
    processing_max_bin = max(1, min(processing_max_bin, dsp_cfg.nfft_range // 2))
    display_max_bin = int(math.ceil(float(home_viewport.range_max_bin_f)))
    display_max_bin = max(1, min(display_max_bin, dsp_cfg.nfft_range // 2))
    max_virtual_array_bin = max(processing_max_bin, display_max_bin)

    total_samples_needed = dsp_cfg.x_frames * dsp_cfg.chirps * dsp_cfg.samples * dsp_cfg.rx
    complex_per_frame = dsp_cfg.chirps * dsp_cfg.samples * dsp_cfg.rx

    complex_data = np.zeros(total_samples_needed, dtype=np.complex64)
    profiles_out_buf = np.empty((dsp_cfg.range_profile_count, int(fft_plot_h)), dtype=np.float32)
    angle_diag_out_buf = (
        np.empty((int(fft_plot_h), max(1, int(angle_diag_w))), dtype=np.float32)
        if gui_angle_diag_dbuf is not None and int(angle_diag_w) > 0
        else None
    )
    doppler_diag_out_buf = (
        np.empty((int(fft_plot_h), max(1, int(doppler_diag_w))), dtype=np.float32)
        if gui_doppler_diag_dbuf is not None and int(doppler_diag_w) > 0
        else None
    )
    virtual_array_work_buf = np.empty(
        (dsp_cfg.x_frames, n_doppler, max_virtual_array_bin, dsp_cfg.tx, dsp_cfg.rx),
        dtype=np.complex64,
    )
    virtual_array_flat_work_buf = np.empty(
        (dsp_cfg.x_frames, n_doppler, max_virtual_array_bin, dsp_cfg.virtual_ant),
        dtype=np.complex64,
    )
    doppler_work_buf = np.empty(
        (dsp_cfg.x_frames, n_doppler, dsp_cfg.tx, max_virtual_array_bin, dsp_cfg.rx),
        dtype=np.complex64,
    )
    profiles_db_work_buf = np.empty((dsp_cfg.tx, int(fft_plot_h), dsp_cfg.rx), dtype=np.float32)
    heatmap_db_work_buf = np.empty((int(gui_h), int(gui_w)), dtype=np.float32)
    gui_heat_size = int(gui_h) * int(gui_w)
    gui_heat_views = (
        np.frombuffer(gui_dbuf, dtype=np.float32, count=gui_heat_size, offset=0),
        np.frombuffer(gui_dbuf, dtype=np.float32, count=gui_heat_size, offset=gui_heat_size * 4),
    )
    gui_heat_alpha_views: tuple[np.ndarray, np.ndarray] | None = None
    if gui_alpha_dbuf is not None:
        gui_heat_alpha_views = (
            np.frombuffer(gui_alpha_dbuf, dtype=np.float32, count=gui_heat_size, offset=0),
            np.frombuffer(gui_alpha_dbuf, dtype=np.float32, count=gui_heat_size, offset=gui_heat_size * 4),
        )
    gui_prof_size = int(dsp_cfg.range_profile_count) * int(fft_plot_h)
    gui_profile_views = (
        np.frombuffer(gui_prof_dbuf, dtype=np.float32, count=gui_prof_size, offset=0),
        np.frombuffer(gui_prof_dbuf, dtype=np.float32, count=gui_prof_size, offset=gui_prof_size * 4),
    )
    gui_angle_diag_views: tuple[np.ndarray, np.ndarray] | None = None
    if gui_angle_diag_dbuf is not None and int(angle_diag_w) > 0:
        gui_angle_diag_size = int(fft_plot_h) * max(1, int(angle_diag_w))
        gui_angle_diag_views = (
            np.frombuffer(gui_angle_diag_dbuf, dtype=np.float32, count=gui_angle_diag_size, offset=0),
            np.frombuffer(gui_angle_diag_dbuf, dtype=np.float32, count=gui_angle_diag_size, offset=gui_angle_diag_size * 4),
        )
    gui_doppler_diag_views: tuple[np.ndarray, np.ndarray] | None = None
    if gui_doppler_diag_dbuf is not None and int(doppler_diag_w) > 0:
        gui_doppler_diag_size = int(fft_plot_h) * max(1, int(doppler_diag_w))
        gui_doppler_diag_views = (
            np.frombuffer(gui_doppler_diag_dbuf, dtype=np.float32, count=gui_doppler_diag_size, offset=0),
            np.frombuffer(gui_doppler_diag_dbuf, dtype=np.float32, count=gui_doppler_diag_size, offset=gui_doppler_diag_size * 4),
        )

    shm_view = memoryview(shm_frames).cast("B")
    n_slots = len(slot_state)
    heatmap_ema = None
    display_zoom_runtime = DisplayZoomRuntime(home_viewport=home_viewport)
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

    _log_dsp_runtime_diagnostics_once(cfg_dict, dsp_cfg, diagnostics_cfg)

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

    def _reset_runtime_processing_state(*, reset_tracker: bool) -> None:
        nonlocal detection_static_bg_state, detection_moving_bg_state, display_bg_state
        nonlocal heatmap_ema, tracker, tracker_time_s, last_tracker_seq
        nonlocal warned_display_loop_average_after_doppler, warned_detection_loop_average_after_doppler

        detection_static_bg_state = BackgroundSubtractionState()
        detection_moving_bg_state = BackgroundSubtractionState()
        display_bg_state = BackgroundSubtractionState()
        heatmap_ema = None
        warned_display_loop_average_after_doppler = False
        warned_detection_loop_average_after_doppler = False
        if display_zoom_runtime is not None:
            display_zoom_runtime.display_bg_state = BackgroundSubtractionState()
            display_zoom_runtime.moving_bg_state = BackgroundSubtractionState()
            display_zoom_runtime.display_bg_state_cache.clear()
            display_zoom_runtime.moving_bg_state_cache.clear()
            display_zoom_runtime.heatmap_ema = None
            display_zoom_runtime.last_view_db = None
            display_zoom_runtime.last_view_alpha = None
            display_zoom_runtime.last_viewport_signature = None
            display_zoom_runtime.last_applied_meta = None
            display_zoom_runtime.last_mode = "power_xy"
        if reset_tracker:
            tracker = (
                MultiObjectTracker(tracking_cfg=tracking_cfg, tracker_cfg=tracker_cfg)
                if tracking_runtime_enabled
                else None
            )
            tracker_time_s = None
            last_tracker_seq = None

    def _apply_runtime_config_patch(cfg_patch: dict[str, Any], *, reset_runtime_state: bool = False) -> None:
        nonlocal cfg_dict, dsp_cfg
        nonlocal selection, window_range, window_doppler, window_angle
        nonlocal apply_range_window, apply_doppler_window, apply_angle_window
        nonlocal mean_before_range_fft
        nonlocal detection_static_post_range_fft_filters, detection_moving_pre_doppler_filters
        nonlocal display_post_range_fft_filters, angle_processing
        nonlocal heatmap_ema_cfg, heatmap_spatial_filter_cfg, heatmap_ema
        nonlocal detection_static_cfg, detection_moving_cfg, fusion_cfg
        nonlocal tracking_cfg, tracker_cfg, tracking_runtime_enabled, tracker
        nonlocal tracker_nominal_frame_dt_s, tracker_time_s, last_tracker_seq
        nonlocal detection_static_bg_state, detection_moving_bg_state, display_bg_state
        nonlocal warned_display_loop_average_after_doppler, warned_detection_loop_average_after_doppler

        if not isinstance(cfg_patch, dict) or not cfg_patch:
            return

        # Un controllo GUI invia solo una patch: il deep-merge preserva le
        # chiavi sorelle della configurazione corrente invece di sostituire
        # l'intero blocco YAML.
        next_cfg_dict = _deep_merge_dict(cfg_dict, cfg_patch)
        next_selection = selection_from_yaml_dict(next_cfg_dict)
        next_mean_before_range_fft = mean_before_range_fft_from_yaml_dict(next_cfg_dict)
        next_detection_static_filters = detection_static_post_range_fft_filters_from_yaml_dict(next_cfg_dict)
        next_detection_static_filters, detection_static_filter_warnings = sanitize_detection_static_post_range_fft_filters(
            next_detection_static_filters
        )
        next_detection_moving_filters = detection_moving_pre_doppler_filters_from_yaml_dict(next_cfg_dict)
        next_detection_moving_filters, detection_moving_filter_warnings = sanitize_detection_moving_pre_doppler_filters(
            next_detection_moving_filters
        )
        next_display_filters = display_post_range_fft_filters_from_yaml_dict(next_cfg_dict)
        next_display_filters, display_filter_warnings = sanitize_display_post_range_fft_filters(next_display_filters)
        next_angle_processing = angle_processing_from_yaml_dict(next_cfg_dict)
        next_heatmap_ema_cfg = heatmap_ema_from_yaml_dict(next_cfg_dict)
        next_heatmap_spatial_filter_cfg = heatmap_spatial_filter_from_yaml_dict(next_cfg_dict)
        next_detection_static_cfg = detection_static_from_yaml_dict(next_cfg_dict)
        next_detection_moving_cfg = detection_moving_from_yaml_dict(next_cfg_dict)
        next_fusion_cfg = fusion_from_yaml_dict(next_cfg_dict)
        next_tracking_cfg = tracking_from_yaml_dict(next_cfg_dict)
        if int(next_tracking_cfg.max_tracks) > tracks_capacity:
            print(
                f"[TRACK WARN] max_tracks={int(next_tracking_cfg.max_tracks)} exceeds shared GUI capacity "
                f"{tracks_capacity}; clamped until pipeline restart."
            )
            next_tracking_cfg = replace(next_tracking_cfg, max_tracks=max(1, tracks_capacity))
        next_tracker_cfg = tracker_from_yaml_dict(next_cfg_dict)
        next_range_angle_moving_cfg = range_angle_moving_from_yaml_dict(next_cfg_dict)
        next_zero_after = max(0, _to_int((next_cfg_dict.get("dsp", {}) or {}).get("zero_after_range_fft_bins", 0), 0))

        if next_selection != selection:
            window_range, window_doppler, window_angle = build_windows(
                next_selection,
                samples=dsp_cfg.samples,
                n_loops=dsp_cfg.chirps // dsp_cfg.tx,
                virtual_ant=dsp_cfg.virtual_ant,
            )
            apply_range_window = not _window_is_identity(next_selection.window_range)
            apply_doppler_window = not _window_is_identity(next_selection.window_doppler)
            apply_angle_window = not _window_is_identity(next_selection.window_angle)

        # EMA e modelli di clutter dipendono dai parametri che li hanno
        # generati; azzerarli evita di mescolare due regimi di filtraggio.
        if next_detection_static_filters != detection_static_post_range_fft_filters:
            detection_static_bg_state = BackgroundSubtractionState()
            warned_detection_loop_average_after_doppler = False
        if next_detection_moving_filters != detection_moving_pre_doppler_filters:
            detection_moving_bg_state = BackgroundSubtractionState()
            if display_zoom_runtime is not None:
                display_zoom_runtime.moving_bg_state = BackgroundSubtractionState()
                display_zoom_runtime.moving_bg_state_cache.clear()
        if next_display_filters != display_post_range_fft_filters:
            display_bg_state = BackgroundSubtractionState()
            if display_zoom_runtime is not None:
                display_zoom_runtime.display_bg_state = BackgroundSubtractionState()
                display_zoom_runtime.display_bg_state_cache.clear()
            warned_display_loop_average_after_doppler = False
        if next_heatmap_ema_cfg != heatmap_ema_cfg:
            heatmap_ema = None
            if display_zoom_runtime is not None:
                display_zoom_runtime.heatmap_ema = None

        next_tracking_runtime_enabled = bool(getattr(next_tracking_cfg, "enabled", False))
        if next_tracking_runtime_enabled and int(dsp_cfg.x_frames) > 1:
            print(
                f"[TRACK WARN] runtime tracking disabled because capture.x_frames={int(dsp_cfg.x_frames)}. "
                "Current tracker expects x_frames=1."
            )
            next_tracking_runtime_enabled = False
        if (
            next_tracking_runtime_enabled != tracking_runtime_enabled
            or next_tracking_cfg != tracking_cfg
            or next_tracker_cfg != tracker_cfg
        ):
            tracker = (
                MultiObjectTracker(tracking_cfg=next_tracking_cfg, tracker_cfg=next_tracker_cfg)
                if next_tracking_runtime_enabled
                else None
            )
            tracker_time_s = None
            last_tracker_seq = None

        for warn_msg in detection_static_filter_warnings + detection_moving_filter_warnings + display_filter_warnings:
            print(f"[DSP WARN] runtime config: {warn_msg}")

        cfg_dict = next_cfg_dict
        selection = next_selection
        mean_before_range_fft = next_mean_before_range_fft
        detection_static_post_range_fft_filters = next_detection_static_filters
        detection_moving_pre_doppler_filters = next_detection_moving_filters
        display_post_range_fft_filters = next_display_filters
        angle_processing = next_angle_processing
        heatmap_ema_cfg = next_heatmap_ema_cfg
        heatmap_spatial_filter_cfg = next_heatmap_spatial_filter_cfg
        detection_static_cfg = next_detection_static_cfg
        detection_moving_cfg = next_detection_moving_cfg
        fusion_cfg = next_fusion_cfg
        tracking_cfg = next_tracking_cfg
        tracker_cfg = next_tracker_cfg
        tracking_runtime_enabled = next_tracking_runtime_enabled
        tracker_nominal_frame_dt_s = resolve_nominal_frame_period_s(cfg_dict, dsp_cfg, tracking_cfg)
        dsp_cfg = replace(
            dsp_cfg,
            zero_after_range_fft_bins=int(next_zero_after),
            range_angle_moving=next_range_angle_moving_cfg,
        )
        if display_zoom_runtime is not None:
            display_zoom_runtime.last_view_db = None
            display_zoom_runtime.last_view_alpha = None
            display_zoom_runtime.last_viewport_signature = None
        if reset_runtime_state:
            _reset_runtime_processing_state(reset_tracker=True)
        print("[DSP CFG] runtime config updated from GUI.")

    def _poll_dsp_commands() -> None:
        while True:
            try:
                cmd = dsp_cmd_queue.get_nowait()
            except pyqueue.Empty:
                break
            if not isinstance(cmd, dict):
                continue
            cmd_type = str(cmd.get("type", "")).strip().lower()
            if cmd_type == "update_runtime_config":
                cfg_patch = cmd.get("cfg_patch", cmd.get("patch", {}))
                try:
                    _apply_runtime_config_patch(
                        cfg_patch,
                        reset_runtime_state=bool(cmd.get("reset_runtime_state", False)),
                    )
                except Exception as e:
                    print(f"[DSP CFG WARN] runtime config update failed: {e}")
            elif cmd_type == "reset_runtime_state":
                _reset_runtime_processing_state(reset_tracker=True)
                print("[DSP CFG] runtime processing state reset from GUI.")

    def _read_requested_viewport() -> DisplayViewport:
        if display_viewport_request is None:
            return home_viewport
        arr = None
        try:
            if display_viewport_request_lock is not None:
                with display_viewport_request_lock:
                    arr = np.asarray(display_viewport_request[:], dtype=np.float64)
            else:
                arr = np.asarray(display_viewport_request[:], dtype=np.float64)
        except Exception:
            return home_viewport
        if arr is None or int(arr.size) < 4:
            return home_viewport
        try:
            seq_value = int(display_viewport_request_seq.value) if display_viewport_request_seq is not None else 0
        except Exception:
            seq_value = 0
        return clamp_display_viewport(
            x_min_m=float(arr[0]),
            x_max_m=float(arr[1]),
            y_min_m=float(arr[2]),
            y_max_m=float(arr[3]),
            home_viewport=home_viewport,
            output_width=int(gui_w),
            output_height=int(gui_h),
            dr_m=float(dr),
            seq=int(seq_value),
        )

    def _sync_home_viewport() -> None:
        nonlocal home_viewport, display_max_bin
        next_home_viewport = _read_home_viewport(home_viewport)
        if display_zoom_runtime is not None:
            changed = update_display_zoom_runtime_home_viewport(display_zoom_runtime, next_home_viewport)
        else:
            changed = display_viewport_signature(next_home_viewport) != display_viewport_signature(home_viewport)
        home_viewport = next_home_viewport
        if changed:
            display_max_bin = int(math.ceil(float(home_viewport.range_max_bin_f)))
            display_max_bin = max(1, min(display_max_bin, dsp_cfg.nfft_range // 2))

    def _write_applied_viewport(meta: AppliedViewportMeta | None) -> None:
        if meta is None or display_viewport_applied is None:
            return
        values = [
            float(meta.x_min_m),
            float(meta.x_max_m),
            float(meta.y_min_m),
            float(meta.y_max_m),
            float(meta.range_min_bin_f),
            float(meta.range_max_bin_f),
            float(meta.angle_min_deg),
            float(meta.angle_max_deg),
            float(meta.zoom_level),
        ]
        try:
            if display_viewport_applied_lock is not None:
                with display_viewport_applied_lock:
                    for idx_val, value in enumerate(values):
                        display_viewport_applied[idx_val] = float(value)
            else:
                for idx_val, value in enumerate(values):
                    display_viewport_applied[idx_val] = float(value)
        except Exception:
            return
        try:
            if display_viewport_applied_seq is not None:
                display_viewport_applied_seq.value = int(meta.seq)
        except Exception:
            pass
        try:
            if display_viewport_applied_frame_seq is not None:
                display_viewport_applied_frame_seq.value = int(meta.frame_seq)
        except Exception:
            pass
        try:
            if display_viewport_applied_fallback is not None:
                display_viewport_applied_fallback.value = 1 if bool(meta.fallback_used) else 0
        except Exception:
            pass

    if ready_evt is not None:
        ready_evt.set()

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
                frame_time_s = 0.0
                if slot_frame_time_s is not None:
                    try:
                        frame_time_s = float(slot_frame_time_s[s])
                    except Exception:
                        frame_time_s = 0.0
                ready.append((seq, s, int(slot_ok[s]), frame_time_s))

        if not ready:
            continue

        # Keep only the newest frames so the display stays real-time under load.
        drop_slots = []
        if len(ready) > dsp_cfg.x_frames:
            drop_slots = [int(s) for _, s, _, _ in ready[:-dsp_cfg.x_frames]]
            ready = ready[-dsp_cfg.x_frames :]

        if drop_slots:
            if dsp_cfg.debug_stats and dsp_skip is not None:
                with dsp_skip.get_lock():
                    dsp_skip.value += len(drop_slots)
            _release_slots_dsp(drop_slots)

        proc_slots = []
        proc_seqs: list[int] = []
        proc_frame_times_s: list[float] = []
        bad_slots = []
        for seq, s, ok, frame_time_s in ready:
            if ok == 1:
                proc_slots.append(int(s))
                proc_seqs.append(int(seq))
                proc_frame_times_s.append(float(frame_time_s))
            else:
                bad_slots.append(int(s))
        if bad_slots:
            _release_slots_dsp(bad_slots)
        if not proc_slots:
            continue

        n_proc = min(len(proc_slots), dsp_cfg.x_frames)
        slots_to_process = proc_slots[:n_proc]
        proc_seqs = proc_seqs[:n_proc]
        proc_frame_times_s = proc_frame_times_s[:n_proc]

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
        try:
            if heatmap_view_mode is None:
                heatmap_mode_value = 0
            else:
                with heatmap_view_mode.get_lock():
                    heatmap_mode_value = int(heatmap_view_mode.value)
        except Exception:
            heatmap_mode_value = 0
        display_heatmap_mode = "range_angle_moving" if int(heatmap_mode_value) == 1 else "power_xy"
        if display_heatmap_mode == "range_angle_moving":
            normalize_to_peak = False
        _sync_home_viewport()
        requested_viewport = _read_requested_viewport()
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
        heatmap_ema, tracking_detections = process_buffer(
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
            display_heatmap_mode=display_heatmap_mode,
            gui_heat_alpha_views=gui_heat_alpha_views,
            display_viewport=requested_viewport,
            display_zoom_cfg=display_zoom_cfg,
            display_zoom_runtime=display_zoom_runtime,
            frame_seq=(int(proc_seqs[-1]) if proc_seqs else 0),
            gui_angle_diag_views=gui_angle_diag_views,
            gui_doppler_diag_views=gui_doppler_diag_views,
            angle_diag_out_buf=angle_diag_out_buf,
            doppler_diag_out_buf=doppler_diag_out_buf,
            display_normalization_reference_db_out=display_normalization_reference_db,
            display_power_normalized_out=display_power_normalized,
            publish_applied_viewport=_write_applied_viewport,
        )
        if tracker is not None and tracking_runtime_enabled:
            tracker_timestamp_s: float | None
            latest_frame_time_s = float(proc_frame_times_s[-1]) if proc_frame_times_s else 0.0
            if np.isfinite(latest_frame_time_s) and latest_frame_time_s > 0.0:
                tracker_timestamp_s = latest_frame_time_s
                tracker_time_s = latest_frame_time_s
                if proc_seqs:
                    last_tracker_seq = int(proc_seqs[-1])
            elif tracker_nominal_frame_dt_s is not None and proc_seqs:
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
