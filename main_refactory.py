"""Applicazione principale: GUI, acquisizione realtime, SAR e post-processing.

Questo è il punto di integrazione del progetto.  Le routine esterne ai worker
gestiscono la GUI Dear PyGui; ``radar_rx`` e ``logger_worker`` possiedono
rispettivamente lo stream UDP e la scrittura delle catture su disco.
"""

import socket
import time
import queue as pyqueue
import atexit
import threading
from pathlib import Path
from multiprocessing import Process, Queue, Value
from array import array
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray, Synchronized
import yaml
import numpy as np
import dearpygui.dearpygui as dpg
import os
import json
import re
import struct
from datetime import datetime

from sar_capture import (
    CaptureError,
    CaptureMetadataStore,
    CaptureSessionManager,
    normalize_capture_metadata,
    read_capture_metadata,
    write_capture_metadata,
)
from sar_scan import (
    SarScanCoordinator,
    ScanError,
    ScanEvent,
    ScanPlan,
    ScanState,
)

try:
    # Il backend non crea una finestra: la GUI standalone viene istanziata solo
    # sotto il suo ``if __name__ == '__main__'``.  Main resta quindi l'unico
    # proprietario del 1063 durante una scansione SAR.
    from phidget_stepper_gui import PhidgetStepperController, load_config as load_stepper_config
except Exception as exc:  # Il radar può restare utilizzabile senza SDK Phidget.
    PhidgetStepperController = None
    load_stepper_config = None
    _PHIDGET_BACKEND_ERROR = str(exc)
else:
    _PHIDGET_BACKEND_ERROR = ""

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

def _cpu_percent_pid_win(pid: int, cpu_state: dict) -> float:
    """Windows fallback CPU% for a pid using GetProcessTimes."""
    if os.name != "nt" or pid is None or pid <= 0:
        return float("nan")
    try:
        import ctypes
        import ctypes.wintypes as wt

        class FILETIME(ctypes.Structure):
            _fields_ = [("dwLowDateTime", wt.DWORD), ("dwHighDateTime", wt.DWORD)]

        def _ft_to_int(ft: FILETIME) -> int:
            return (int(ft.dwHighDateTime) << 32) | int(ft.dwLowDateTime)

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        PROCESS_QUERY_INFORMATION = 0x0400

        kernel32 = ctypes.windll.kernel32
        OpenProcess = kernel32.OpenProcess
        OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
        OpenProcess.restype = wt.HANDLE

        GetProcessTimes = kernel32.GetProcessTimes
        GetProcessTimes.argtypes = [
            wt.HANDLE,
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
        ]
        GetProcessTimes.restype = wt.BOOL

        CloseHandle = kernel32.CloseHandle
        CloseHandle.argtypes = [wt.HANDLE]
        CloseHandle.restype = wt.BOOL

        h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_QUERY_INFORMATION, False, int(pid))
        if not h:
            return 0.0
        try:
            ft_create = FILETIME()
            ft_exit = FILETIME()
            ft_kernel = FILETIME()
            ft_user = FILETIME()
            ok = GetProcessTimes(h, ft_create, ft_exit, ft_kernel, ft_user)
            if not ok:
                return 0.0
            cpu_now_100ns = _ft_to_int(ft_kernel) + _ft_to_int(ft_user)
            t_now = time.perf_counter()
            key = ("win_cpu", int(pid))
            prev = cpu_state.get(key)
            cpu_state[key] = (t_now, cpu_now_100ns)
            if prev is None:
                return 0.0
            dt = float(t_now - prev[0])
            dc = int(cpu_now_100ns - prev[1])
            if dt <= 0.0 or dc < 0:
                return 0.0
            # 100ns ticks -> seconds; 100% means one full CPU core.
            return float((dc * 1e-7 / dt) * 100.0)
        finally:
            try:
                CloseHandle(h)
            except Exception:
                pass
    except Exception:
        return float("nan")

def _cpu_percent_pid(pid: int, cpu_state: dict) -> float:
    """Best-effort process CPU% using psutil, non-blocking."""
    if pid is None or pid <= 0:
        return 0.0
    if psutil is not None:
        p = cpu_state.get(int(pid))
        if p is None:
            try:
                p = psutil.Process(int(pid))
                p.cpu_percent(None)
                cpu_state[int(pid)] = p
                return 0.0
            except Exception:
                pass
        else:
            try:
                return float(p.cpu_percent(None))
            except Exception:
                cpu_state.pop(int(pid), None)
    # Fallback if psutil is missing or fails for this pid.
    return _cpu_percent_pid_win(int(pid), cpu_state)


def _normalize_priority_level(level: str) -> str:
    return str(level).strip().lower()


def _set_process_priority(pid: int, level: str):
    if pid is None or int(pid) <= 0:
        return False, "invalid pid"

    level_n = _normalize_priority_level(level)
    allowed = {"idle", "below_normal", "normal", "above_normal", "high", "realtime"}
    if level_n not in allowed:
        return False, f"unknown priority level '{level}'"

    if os.name == "nt":
        win_map = {
            "idle": 0x00000040,
            "below_normal": 0x00004000,
            "normal": 0x00000020,
            "above_normal": 0x00008000,
            "high": 0x00000080,
            "realtime": 0x00000100,
        }
        # Try psutil first (if present), then fallback to WinAPI.
        try:
            import psutil  # type: ignore

            p = psutil.Process(int(pid))
            p.nice(int(win_map[level_n]))
            return True, level_n
        except Exception:
            try:
                import ctypes
                import ctypes.wintypes as wt

                PROCESS_SET_INFORMATION = 0x0200
                PROCESS_QUERY_INFORMATION = 0x0400
                kernel32 = ctypes.windll.kernel32
                OpenProcess = kernel32.OpenProcess
                OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
                OpenProcess.restype = wt.HANDLE

                SetPriorityClass = kernel32.SetPriorityClass
                SetPriorityClass.argtypes = [wt.HANDLE, wt.DWORD]
                SetPriorityClass.restype = wt.BOOL

                CloseHandle = kernel32.CloseHandle
                CloseHandle.argtypes = [wt.HANDLE]
                CloseHandle.restype = wt.BOOL

                h = OpenProcess(PROCESS_SET_INFORMATION | PROCESS_QUERY_INFORMATION, False, int(pid))
                if not h:
                    return False, "OpenProcess failed"
                ok = SetPriorityClass(h, int(win_map[level_n]))
                CloseHandle(h)
                if not ok:
                    return False, "SetPriorityClass failed"
                return True, level_n
            except Exception as e:
                return False, f"windows priority error: {e}"

    # Posix fallback via nice value.
    if not hasattr(os, "setpriority") or not hasattr(os, "PRIO_PROCESS"):
        return False, "os.setpriority unavailable on this platform"

    nice_map = {
        "idle": 19,
        "below_normal": 10,
        "normal": 0,
        "above_normal": -5,
        "high": -10,
        "realtime": -15,
    }
    try:
        os.setpriority(os.PRIO_PROCESS, int(pid), int(nice_map[level_n]))
        return True, level_n
    except PermissionError:
        return False, f"permission denied for level '{level_n}'"
    except Exception as e:
        return False, f"setpriority error: {e}"


def _apply_process_priority(label: str, pid: int, level: str, enabled: bool) -> None:
    if not enabled:
        return
    ok, msg = _set_process_priority(int(pid), str(level))
    if ok:
        print(f"[PRIO] {label} pid={int(pid)} level={msg}")
    else:
        print(f"[PRIO WARN] {label} pid={int(pid)} requested={level} ({msg})")


def _parse_cpu_set(raw_value, logical_count: int):
    """Parse CPU set from list[int] or string like '0-7,10,12-15'."""
    if logical_count <= 0:
        return []
    out = set()
    if isinstance(raw_value, (list, tuple)):
        for x in raw_value:
            try:
                c = int(x)
            except Exception:
                continue
            if 0 <= c < logical_count:
                out.add(c)
        return sorted(out)
    if isinstance(raw_value, str):
        txt = raw_value.strip()
        if not txt:
            return []
        for part in txt.split(","):
            p = part.strip()
            if not p:
                continue
            if "-" in p:
                lr = p.split("-", 1)
                if len(lr) != 2:
                    continue
                try:
                    lo = int(lr[0].strip())
                    hi = int(lr[1].strip())
                except Exception:
                    continue
                if hi < lo:
                    lo, hi = hi, lo
                lo = max(0, lo)
                hi = min(logical_count - 1, hi)
                for c in range(lo, hi + 1):
                    out.add(c)
            else:
                try:
                    c = int(p)
                except Exception:
                    continue
                if 0 <= c < logical_count:
                    out.add(c)
        return sorted(out)
    return []


def _default_affinity_sets(logical_count: int):
    """
    Auto partition logical CPUs:
      - DSP gets most cores
      - MAIN/RX/LOG stay on a reserved tail set
    """
    if logical_count <= 1:
        return {"main": [0], "rx": [0], "log": [0], "dsp": [0]}

    if logical_count >= 24:
        reserve = 8
    elif logical_count >= 16:
        reserve = 4
    else:
        reserve = 2
    reserve = min(max(1, reserve), logical_count - 1)

    dsp = list(range(0, logical_count - reserve))
    tail = list(range(logical_count - reserve, logical_count))
    if not dsp:
        dsp = [0]
    if not tail:
        tail = [logical_count - 1]

    rx = [tail[0]]
    log = [tail[1]] if len(tail) >= 2 else [tail[0]]
    main = tail
    return {"main": main, "rx": rx, "log": log, "dsp": dsp}


def _apply_process_affinity(label: str, pid: int, cpus, enabled: bool) -> None:
    if not enabled:
        return
    if pid is None or int(pid) <= 0:
        print(f"[AFF WARN] {label} skipped: invalid pid")
        return
    if not cpus:
        print(f"[AFF WARN] {label} skipped: empty cpu set")
        return
    if psutil is not None:
        try:
            p = psutil.Process(int(pid))
            p.cpu_affinity(list(cpus))
            print(f"[AFF] {label} pid={int(pid)} cpus={list(cpus)} (psutil)")
            return
        except Exception as e:
            print(f"[AFF WARN] {label} psutil failed ({e}), trying WinAPI fallback")

    if os.name != "nt":
        print(f"[AFF WARN] {label} skipped: affinity fallback only implemented on Windows")
        return

    try:
        import ctypes
        import ctypes.wintypes as wt

        PROCESS_SET_INFORMATION = 0x0200
        PROCESS_QUERY_INFORMATION = 0x0400

        mask = 0
        for c in cpus:
            cc = int(c)
            if 0 <= cc < 64:
                mask |= (1 << cc)
        if mask == 0:
            print(f"[AFF WARN] {label} skipped: cpu mask is zero")
            return

        kernel32 = ctypes.windll.kernel32
        OpenProcess = kernel32.OpenProcess
        OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
        OpenProcess.restype = wt.HANDLE

        SetProcessAffinityMask = kernel32.SetProcessAffinityMask
        SetProcessAffinityMask.argtypes = [wt.HANDLE, ctypes.c_size_t]
        SetProcessAffinityMask.restype = wt.BOOL

        CloseHandle = kernel32.CloseHandle
        CloseHandle.argtypes = [wt.HANDLE]
        CloseHandle.restype = wt.BOOL

        h = OpenProcess(PROCESS_SET_INFORMATION | PROCESS_QUERY_INFORMATION, False, int(pid))
        if not h:
            print(f"[AFF WARN] {label} OpenProcess failed")
            return
        try:
            ok = SetProcessAffinityMask(h, ctypes.c_size_t(mask))
            if not ok:
                print(f"[AFF WARN] {label} SetProcessAffinityMask failed")
                return
            print(f"[AFF] {label} pid={int(pid)} cpus={list(cpus)} (winapi)")
        finally:
            try:
                CloseHandle(h)
            except Exception:
                pass
    except Exception as e:
        print(f"[AFF WARN] {label} fallback failed ({e})")


from offline_processing import (
    OfflineBPRuntime,
    OfflineSARConfig,
    SARReader,
    offline_map_bounds_from_yaml_dict,
)
from mmwave_studio_bridge import DCA1000Config, MmwaveStudioBridge, MmwaveStudioError, RadarConnectionConfig
from shutdown_utils import cleanup_processes, close_queues
from realtime_dsp import (
    AppliedViewportMeta,
    DisplayViewport,
    RealtimeDSPConfig,
    applied_viewport_meta_from_viewport,
    build_angle_axis_deg,
    build_doppler_axis_mps,
    build_display_viewport,
    clamp_display_viewport,
    display_image_resolutions_from_yaml_dict,
    display_projection_from_yaml_dict,
    display_viewport_signature,
    display_zoom_from_yaml_dict,
    dsp_worker,
    range_angle_moving_from_yaml_dict,
    resolve_processing_range_max_m,
    resolve_display_crossrange_max_m,
)
#import dpnp as dp


# --- CONFIGURAZIONE ---
CFG_PATH = Path(__file__).with_name("Config.yaml")  # <-- nome esatto del tuo file
with CFG_PATH.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


# --- CONFIGURAZIONE FISICA ---
C = float(cfg["radar"]["c"])
FS = float(cfg["radar"]["fs"])
SLOPE = float(cfg["radar"]["slope"])
FC = float(cfg["radar"]["fc"])

# --- CONFIGURAZIONE ACQUISIZIONE ---
SAMPLES = int(cfg["capture"]["samples"])
CHIRPS = int(cfg["capture"]["chirps"])
RX = int(cfg["capture"]["rx"])
TX = int(cfg["capture"]["tx"])
X_FRAMES = int(cfg["capture"]["x_frames"])

VIRTUAL_ANT = TX * RX
BYTES_PER_FRAME = CHIRPS * SAMPLES * RX * 4

# --- FFT ---
NFFT_RANGE = int(cfg["fft"]["nfft_range"])
NFFT_ANGLE = int(cfg["fft"]["nfft_angle"])

# --- DISPLAY ---
# vmin/vmax: scala di base per modalita NON normalizzata (NON hard limit)
VMIN_RAW = float(cfg["display"]["vmin"])
VMAX_RAW = float(cfg["display"]["vmax"])
# vmin_norm/vmax_norm: scala di base per modalita normalizzata (fallback su raw)
VMIN_NORM = float(cfg.get("display", {}).get("vmin_norm", VMIN_RAW))
VMAX_NORM = float(cfg.get("display", {}).get("vmax_norm", VMAX_RAW))

# range_max / crossrange_max: HARD LIMIT (0 .. max config)
RMAX_HARD_MAX = float(cfg["display"]["range_max"])
RANGE_MAX_DISPLAY = float(cfg["display"]["range_max"])
RANGE_MAX_PROCESSING = float(resolve_processing_range_max_m(cfg))
display_projection_cfg = display_projection_from_yaml_dict(cfg)
display_zoom_cfg = display_zoom_from_yaml_dict(cfg)
display_image_resolutions_cfg = display_image_resolutions_from_yaml_dict(cfg)
HEATMAP_CROSSRANGE_MAX_DISPLAY = float(
    resolve_display_crossrange_max_m(
        RANGE_MAX_DISPLAY,
        build_angle_axis_deg(NFFT_ANGLE),
        display_projection_cfg,
    )
)
HEATMAP_XMAX_HARD_MAX = float(HEATMAP_CROSSRANGE_MAX_DISPLAY)

crossrange_cfg_raw = cfg.get("display", {}).get("crossrange_max_m", cfg.get("display", {}).get("crossrange_max", None))
try:
    CROSSRANGE_MAX_DISPLAY = float(crossrange_cfg_raw)
except (TypeError, ValueError):
    CROSSRANGE_MAX_DISPLAY = float(RANGE_MAX_DISPLAY)
if not np.isfinite(CROSSRANGE_MAX_DISPLAY) or CROSSRANGE_MAX_DISPLAY <= 0.0:
    CROSSRANGE_MAX_DISPLAY = float(RANGE_MAX_DISPLAY)
XMAX_HARD_MAX = float(CROSSRANGE_MAX_DISPLAY)

# valori iniziali di visualizzazione (partono dai limiti config, ma poi l'utente puÃ² ridurli fino a 0)
RANGEFFT_DB_MIN = float(cfg.get("display", {}).get("rangefft_db_min", VMIN_RAW))
RANGEFFT_DB_MAX = float(cfg.get("display", {}).get("rangefft_db_max", VMAX_RAW))
if RANGEFFT_DB_MAX <= RANGEFFT_DB_MIN:
    RANGEFFT_DB_MAX = RANGEFFT_DB_MIN + 1.0
RANGEFFT_LIN_MIN = float(cfg.get("display", {}).get("rangefft_lin_min", 10.0 ** (RANGEFFT_DB_MIN / 10.0)))
RANGEFFT_LIN_MAX = float(cfg.get("display", {}).get("rangefft_lin_max", 10.0 ** (RANGEFFT_DB_MAX / 10.0)))
if RANGEFFT_LIN_MAX <= RANGEFFT_LIN_MIN:
    RANGEFFT_LIN_MAX = RANGEFFT_LIN_MIN + 1e-6
_range_profile_count_raw = cfg.get("display", {}).get("range_profile_count", VIRTUAL_ANT)
try:
    RANGE_PROFILE_COUNT = int(_range_profile_count_raw)
except (TypeError, ValueError):
    RANGE_PROFILE_COUNT = int(VIRTUAL_ANT)
RANGE_PROFILE_COUNT = max(1, min(int(RANGE_PROFILE_COUNT), int(VIRTUAL_ANT)))
RANGEFFT_PLOT_COUNT = int(TX)
RANGEFFT_LINES_PER_PLOT = int(RX)
ANGLEFFT_BINS = max(1, int(NFFT_ANGLE))
DOPPLERFFT_BINS = max(1, int(CHIRPS // max(1, TX)))


def _fallback_heatmap_velocity_scale_mps() -> tuple[float, float]:
    radar_cfg = cfg.get("radar", {}) or {}
    capture_cfg = cfg.get("capture", {}) or {}
    chirp_period_s = None
    for block in (radar_cfg, capture_cfg):
        for key in (
            "chirp_period_s",
            "chirp_repetition_s",
            "pri_s",
            "t_chirp_s",
            "tc_s",
            "chirp_time_s",
        ):
            try:
                value = float(block.get(key, float("nan")))
            except (TypeError, ValueError):
                value = float("nan")
            if np.isfinite(value) and value > 0.0:
                chirp_period_s = value
                break
        if chirp_period_s is not None:
            break
    if chirp_period_s is None:
        return -1.0, 1.0
    try:
        fc_hz = float(FC)
    except (TypeError, ValueError):
        fc_hz = float("nan")
    if not np.isfinite(fc_hz) or fc_hz <= 0.0:
        return -1.0, 1.0
    n_doppler = max(1, int(CHIRPS // max(1, TX)))
    cycles = np.fft.fftshift(np.fft.fftfreq(n_doppler, d=1.0)).astype(np.float32, copy=False)
    wavelength_m = float(C) / fc_hz
    effective_pri_s = float(chirp_period_s) * float(max(1, TX))
    axis_mps = cycles * np.float32(wavelength_m * 0.5 / effective_pri_s)
    vmax = float(np.max(np.abs(axis_mps))) if axis_mps.size > 0 else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    return -vmax, vmax


def _default_heatmap_velocity_scale_mps() -> tuple[float, float]:
    fallback_vmin, fallback_vmax = _fallback_heatmap_velocity_scale_mps()
    display_cfg = cfg.get("display", {}) or {}

    def _optional_finite_float(raw_value):
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    cfg_vmin = _optional_finite_float(display_cfg.get("velocity_vmin_mps", None))
    cfg_vmax = _optional_finite_float(display_cfg.get("velocity_vmax_mps", None))
    vmin = float(cfg_vmin) if cfg_vmin is not None else float(fallback_vmin)
    vmax = float(cfg_vmax) if cfg_vmax is not None else float(fallback_vmax)
    if not np.isfinite(vmin):
        vmin = -1.0
    if not np.isfinite(vmax):
        vmax = 1.0
    if vmax <= vmin:
        fallback_span = max(float(fallback_vmax - fallback_vmin), 1e-3)
        if cfg_vmin is not None and cfg_vmax is None:
            vmax = vmin + fallback_span
        elif cfg_vmax is not None and cfg_vmin is None:
            vmin = vmax - fallback_span
        else:
            vmax = vmin + 1e-3
    return float(vmin), float(vmax)


HEATMAP_VELOCITY_VMIN_MPS, HEATMAP_VELOCITY_VMAX_MPS = _default_heatmap_velocity_scale_mps()
# --- CODE QUEUE ---
DEBUG_STATS = bool(cfg["debug"]["debug_stats"])

# --- WORKERS FFT ---
LOGICAL_CPUS = int(os.cpu_count() or 1)
_fft_workers_raw = cfg.get("dsp", {}).get("fft_workers", cfg.get("fft", {}).get("workers", None))
if _fft_workers_raw is None:
    FFT_WORKERS = max(1, min(16, LOGICAL_CPUS - 4))
else:
    FFT_WORKERS = int(_fft_workers_raw)
FFT_WORKERS = max(1, min(FFT_WORKERS, LOGICAL_CPUS))
RANGE_ANGLE_MOVING_CFG = range_angle_moving_from_yaml_dict(cfg)
HEATMAP_VELOCITY_DEAD_ZONE = float(getattr(RANGE_ANGLE_MOVING_CFG, "velocity_dead_zone", 0.08))
HEATMAP_VELOCITY_MIN_OPACITY = float(getattr(RANGE_ANGLE_MOVING_CFG, "min_opacity", 0.35))
if not np.isfinite(HEATMAP_VELOCITY_MIN_OPACITY):
    HEATMAP_VELOCITY_MIN_OPACITY = 0.35
HEATMAP_VELOCITY_MIN_OPACITY = min(1.0, max(0.0, HEATMAP_VELOCITY_MIN_OPACITY))
REALTIME_DSP_CFG = RealtimeDSPConfig(
    c=float(C),
    fs=float(FS),
    slope=float(SLOPE),
    samples=int(SAMPLES),
    chirps=int(CHIRPS),
    rx=int(RX),
    tx=int(TX),
    x_frames=int(X_FRAMES),
    bytes_per_frame=int(BYTES_PER_FRAME),
    nfft_range=int(NFFT_RANGE),
    nfft_angle=int(NFFT_ANGLE),
    range_max_display=float(RANGE_MAX_DISPLAY),
    range_profile_count=int(RANGE_PROFILE_COUNT),
    virtual_ant=int(VIRTUAL_ANT),
    fft_workers=int(FFT_WORKERS),
    debug_stats=bool(DEBUG_STATS),
    range_max_processing_m=float(RANGE_MAX_PROCESSING),
    normalize_skip_range_bins=max(0, int(cfg.get("display", {}).get("normalize_skip_range_bins", 0))),
    zero_after_range_fft_bins=max(0, int(cfg.get("dsp", {}).get("zero_after_range_fft_bins", 0))),
    range_angle_moving=RANGE_ANGLE_MOVING_CFG,
    display_zoom=display_zoom_cfg,
)

# --- PROCESS PRIORITY ---
proc_cfg = cfg.get("process", {}) or {}
prio_cfg = proc_cfg.get("priority", {}) or {}
PRIO_ENABLED = bool(prio_cfg.get("enabled", False))
PRIO_MAIN = str(prio_cfg.get("main", "normal"))
PRIO_RX = str(prio_cfg.get("rx", "normal"))
PRIO_LOG = str(prio_cfg.get("logger", "normal"))
PRIO_DSP = str(prio_cfg.get("dsp", "normal"))

# --- PROCESS AFFINITY ---
aff_cfg = proc_cfg.get("affinity", {}) or {}
AFF_ENABLED = bool(aff_cfg.get("enabled", False))
auto_sets = _default_affinity_sets(LOGICAL_CPUS)
AFF_MAIN = _parse_cpu_set(aff_cfg.get("main", auto_sets["main"]), LOGICAL_CPUS)
AFF_RX = _parse_cpu_set(aff_cfg.get("rx", auto_sets["rx"]), LOGICAL_CPUS)
AFF_LOG = _parse_cpu_set(aff_cfg.get("logger", auto_sets["log"]), LOGICAL_CPUS)
AFF_DSP = _parse_cpu_set(aff_cfg.get("dsp", auto_sets["dsp"]), LOGICAL_CPUS)

# --- SAR capture-only parameters ---
sar_cfg = cfg.get("sar", {}) or {}
SETTLING_DELAY_S = float(sar_cfg.get("settling_delay_s", 0.4))
FRAMES_PER_POSITION = int(sar_cfg.get("frames_per_position", 8))

# Binary header prepended to each capture_pos*.bin:
# [8-byte magic][4-byte little-endian header_len][header_json_utf8]
CAPTURE_HEADER_MAGIC = b"RTPBIN1\x00"
CAPTURE_METADATA_BUFFER_BYTES = 16 * 1024


def _build_capture_file_header(
    pos_id: int,
    *,
    carriage_position_mm: float | None = None,
    carriage_microsteps: int | None = None,
) -> bytes:
    """Serializza l'header RTPBIN v1 che precede i frame raw nel file catturato."""
    header = {
        "format": "rt_capture_v1",
        "position": int(pos_id),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "radar": {
            "c": float(C),
            "fs": float(FS),
            "slope": float(SLOPE),
            "fc": float(FC),
        },
        "capture": {
            "samples": int(SAMPLES),
            "chirps": int(CHIRPS),
            "rx": int(RX),
            "tx": int(TX),
            "x_frames": int(X_FRAMES),  # Realtime batch size.
            "frames_per_position": int(FRAMES_PER_POSITION),
        },
    }
    if carriage_position_mm is not None or carriage_microsteps is not None:
        stage: dict[str, float | int | str] = {"reference": "phidget_home_min"}
        if carriage_position_mm is not None and np.isfinite(float(carriage_position_mm)):
            stage["position_mm"] = float(carriage_position_mm)
        if carriage_microsteps is not None:
            stage["position_microsteps"] = int(carriage_microsteps)
        header["stage"] = stage
    payload = json.dumps(header, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return CAPTURE_HEADER_MAGIC + struct.pack("<I", len(payload)) + payload


def read_offline_scan_settings(path: Path) -> tuple[int, int, float]:
    """Legge numero di frame, posizioni e pitch che la GUI mostrerà per il run."""
    """Legge ``(start_position_id, default_positions, pitch_mm)`` dal YAML."""
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("offline_config.yaml must contain a YAML mapping")
    scan = raw.get("scan", {})
    if not isinstance(scan, dict):
        raise ValueError("offline_config.yaml.scan must contain a mapping")
    start = int(scan.get("x_start", 1))
    end = int(scan.get("x_end", start))
    step = int(scan.get("x_step", 1))
    pitch_m = float(scan.get("x_pitch_m"))
    if start <= 0 or end < start or step <= 0 or not np.isfinite(pitch_m) or pitch_m <= 0.0:
        raise ValueError("invalid scan.x_start/x_end/x_step/x_pitch_m")
    positions = ((end - start) // step) + 1
    return int(start), int(positions), float(pitch_m * 1000.0)


def _yaml_scalar_literal(value) -> str:
    """Serializza un singolo scalare senza riscrivere l'intero YAML."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value_f = float(value)
        if not np.isfinite(value_f):
            raise ValueError("Valore YAML non finito")
        return repr(value_f)
    # Una stringa JSON e' anche uno scalare YAML valido e protegge path,
    # due punti e caratteri '#'.
    return json.dumps(str(value), ensure_ascii=False)


def _split_yaml_inline_comment(value_text: str) -> tuple[str, str]:
    """Separa un commento YAML non racchiuso tra apici dal valore."""
    quote = ""
    escaped = False
    for index, char in enumerate(value_text):
        if escaped:
            escaped = False
            continue
        if quote == '"' and char == "\\":
            escaped = True
            continue
        if char in {'"', "'"}:
            if not quote:
                quote = char
            elif quote == char:
                quote = ""
            continue
        if char == "#" and not quote and (index == 0 or value_text[index - 1].isspace()):
            comment_start = index
            while comment_start > 0 and value_text[comment_start - 1] in " \t":
                comment_start -= 1
            return value_text[:comment_start], value_text[comment_start:]
    return value_text, ""


def _update_top_level_yaml_scalars(
    path: Path,
    updates: dict[tuple[str, str], object],
) -> None:
    """Aggiorna chiavi dirette di sezioni YAML preservando commenti e ordine.

    ``yaml.safe_dump`` eliminerebbe il commento che documenta il pitch fisico.
    Questa routine modifica soltanto le poche chiavi di run controllate dalla
    scansione automatica e inserisce una sezione/chiave se ancora assente.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        original = handle.read()
    newline = "\r\n" if "\r\n" in original else "\n"
    lines = original.splitlines(keepends=True)

    def _section_bounds(section: str) -> tuple[int, int] | None:
        section_pattern = re.compile(rf"^{re.escape(section)}:[ \t]*(?:#.*)?(?:\r?\n)?$")
        start = next((i for i, line in enumerate(lines) if section_pattern.match(line)), None)
        if start is None:
            return None
        end = len(lines)
        for i in range(start + 1, len(lines)):
            if re.match(r"^[^\s#][^:]*:", lines[i]):
                end = i
                break
        return start, end

    for (section, key), value in updates.items():
        bounds = _section_bounds(section)
        literal = _yaml_scalar_literal(value)
        if bounds is None:
            if lines and lines[-1].strip():
                lines.append(newline)
            lines.extend((f"{section}:{newline}", f"  {key}: {literal}{newline}"))
            continue

        start, end = bounds
        key_pattern = re.compile(rf"^(  {re.escape(key)}:[ \t]*)(.*?)(\r?\n)?$")
        replaced = False
        for index in range(start + 1, end):
            match = key_pattern.match(lines[index])
            if match is None:
                continue
            _old_value, comment = _split_yaml_inline_comment(match.group(2))
            eol = match.group(3) or newline
            prefix = match.group(1)
            if not prefix.endswith((" ", "\t")):
                prefix += " "
            lines[index] = f"{prefix}{literal}{comment}{eol}"
            replaced = True
            break
        if replaced:
            continue

        insert_at = end
        while insert_at > start + 1 and not lines[insert_at - 1].strip():
            insert_at -= 1
        lines.insert(insert_at, f"  {key}: {literal}{newline}")

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("".join(lines))
    tmp_path.replace(path)


def _update_existing_yaml_scalar_paths(
    path: Path,
    updates: dict[str, object],
    *,
    inline_comments: dict[str, str] | None = None,
) -> None:
    """Aggiorna percorsi YAML esistenti, inclusi quelli annidati, senza dump.

    Il file di tuning contiene note operative utili. Il salvataggio manuale
    modifica quindi solo gli scalari esposti dalla GUI e conserva struttura,
    ordine e commenti di tutte le altre righe.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        original = handle.read()
    newline = "\r\n" if "\r\n" in original else "\n"
    lines = original.splitlines(keepends=True)
    stack: list[tuple[int, str]] = []
    updated: set[str] = set()
    comments = inline_comments or {}
    mapping_line = re.compile(
        r"^(?P<indent> *)(?P<key>[A-Za-z0-9_]+):(?P<rest>.*?)(?P<eol>\r?\n)?$"
    )

    for index, line in enumerate(lines):
        match = mapping_line.match(line)
        if match is None:
            continue
        indent = len(match.group("indent"))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        key = match.group("key")
        yaml_path = ".".join([part for _depth, part in stack] + [key])
        value_text, old_comment = _split_yaml_inline_comment(match.group("rest"))
        if not value_text.strip():
            # Mappa a blocchi (eventualmente seguita da un commento).
            stack.append((indent, key))
            continue
        if yaml_path not in updates:
            continue
        comment = old_comment
        if yaml_path in comments:
            requested = str(comments[yaml_path]).strip()
            comment = f" # {requested}" if requested else ""
        eol = match.group("eol") or newline
        lines[index] = (
            f"{match.group('indent')}{key}: {_yaml_scalar_literal(updates[yaml_path])}{comment}{eol}"
        )
        updated.add(yaml_path)

    missing = sorted(set(updates) - updated)
    if missing:
        # A manually shortened configuration can lack one or more tuning
        # paths. In that case materialise the exact requested mapping
        # atomically; subsequent saves remain scalar-only and preserve the
        # existing layout/comments as before.
        try:
            migrated = yaml.safe_load(original) or {}
            if not isinstance(migrated, dict):
                raise ValueError("la radice YAML deve essere una mappa")
            for yaml_path, value in updates.items():
                node = migrated
                parts = yaml_path.split(".")
                for part in parts[:-1]:
                    child = node.get(part)
                    if not isinstance(child, dict):
                        child = {}
                        node[part] = child
                    node = child
                node[parts[-1]] = value
            rendered = yaml.safe_dump(migrated, sort_keys=False, allow_unicode=True)
        except Exception as exc:
            raise ValueError(
                "Missing YAML tuning keys (file not modified): " + ", ".join(missing)
            ) from exc
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(rendered)
        tmp_path.replace(path)
        return

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("".join(lines))
    tmp_path.replace(path)


def configure_offline_scan_for_run(
    path: Path,
    *,
    output_dir: Path,
    start_position_id: int,
    positions: int,
    frames_per_position: int | None = None,
) -> float:
    """Aggiorna solo i valori di scansione nel YAML conservando commenti e ordine.

    Il ritorno è il pitch in millimetri, utile alla GUI per aggiornare subito
    l'etichetta dopo che la sessione SAR ha scelto la sua cartella di output.
    """
    """Aggiorna il set di file che l'offline dovrà leggere dopo la scansione.

    Il pitch non viene modificato: resta l'unica sorgente fisica della
    configurazione offline e viene restituito in millimetri al coordinatore.
    """
    if int(positions) <= 0 or int(start_position_id) <= 0:
        raise ValueError("start position and number of positions must be positive")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("offline_config.yaml must contain a YAML mapping")
    scan = raw.setdefault("scan", {})
    if not isinstance(scan, dict):
        raise ValueError("offline_config.yaml.scan must contain a mapping")
    try:
        pitch_m = float(scan["x_pitch_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("scan.x_pitch_m must be a positive number") from exc
    if not np.isfinite(pitch_m) or pitch_m <= 0.0:
        raise ValueError("scan.x_pitch_m must be a positive number")

    x_start = int(start_position_id)
    x_end = int(start_position_id) + int(positions) - 1
    data = raw.setdefault("data", {})
    if not isinstance(data, dict):
        raise ValueError("offline_config.yaml.data must contain a mapping")
    try:
        relative_output = output_dir.resolve().relative_to(path.parent.resolve())
        input_dir = str(relative_output)
    except Exception:
        input_dir = str(output_dir.resolve())
    scalar_updates: dict[tuple[str, str], object] = {
        ("data", "input_dir"): input_dir,
        ("scan", "x_start"): x_start,
        ("scan", "x_end"): x_end,
        # I file prodotti dalla GUI sono sempre consecutivi: capture_posN.bin.
        ("scan", "x_step"): 1,
    }
    if frames_per_position is not None:
        if int(frames_per_position) <= 0:
            raise ValueError("frames_per_position must be greater than zero")
        capture = raw.setdefault("capture", {})
        if not isinstance(capture, dict):
            raise ValueError("offline_config.yaml.capture must contain a mapping")
        scalar_updates[("capture", "frames_per_position")] = int(frames_per_position)

    _update_top_level_yaml_scalars(path, scalar_updates)
    return float(pitch_m * 1000.0)


# ----------------------------
# FUNZIONI DI ELABORAZIONE
# ----------------------------
def radar_rx(
    cmd_queue: Queue,
    free_slots: Queue,
    dsp_ready_queue: Queue,
    shm_frames,
    slot_state,
    slot_ok,
    slot_pos_id,
    slot_usemask,
    slot_pub_seq,
    publish_lock,
    cap_active: Synchronized,
    cap_pos_id: Synchronized,
    cap_id: Synchronized,
    cap_saved: Synchronized,
    capture_metadata: CaptureMetadataStore,
    cap_cancel_id: Synchronized,
    cap_done_id: Synchronized,
    cap_result: Synchronized,
    lost_pkts: Synchronized,
    rx_pkts: Synchronized,
    rx_last_packet_time_s: Synchronized,
    rx_put_drops: Synchronized,
    rx_frames_ok: Synchronized,
    stall_events: Synchronized,
    stream_resets: Synchronized,
    stop_evt,
    settling_delay_s: float,
):
    """
    RX UDP (DCA1000 RAW mode).

    Header RAW:
      - seq:        uint32 little-endian  [0:4]
      - byte_count: uint48 little-endian  [4:10]
      - payload:    [10:]

    IntegritÃ :
      - gap/reset sequenza (seq): il frame attraversato Ã¨ corrotto -> SCARTATO
      - incoerenza byte_count non spiegata da seq_gap: hard resync sul boundary reale del frame

    Performance:
      - zero-copy interprocess: RX scrive raw frame in shared memory (ring di slot)
      - enqueue leggero verso DSP con (seq, slot) per evitare scansione completa ring
      - logger continua a leggere dal ring condiviso durante capture
      - se non ci sono slot liberi: drop (no block), ma manteniamo l'allineamento consumando i byte

    Capture:
      - trigger via cmd_queue ("CAPTURE", capture_id, metadata)
      - dopo settling_delay_s: cap_active=1 e RX tagga slot_pos_id
      - il logger salva X frame validi per capture_id (puÃ² perdere frame, quindi dura piÃ¹ a lungo se necessario)
      - metadata è un blob JSON atomico associato al session ``cap_id``
    """
    PC_IP = "192.168.33.30"
    PORT = 4098
    HEADER_LEN = 10
    RCVBUF_BYTES = 256 * 1024 * 1024
    STALL_TIMEOUT_S = float(cfg.get("rx", {}).get("stall_timeout_s", 2.0))
    if STALL_TIMEOUT_S <= 0.0:
        STALL_TIMEOUT_S = 2.0
    SEQ_REORDER_MAX_BACK = int(cfg.get("rx", {}).get("seq_reorder_max_back", 32))
    if SEQ_REORDER_MAX_BACK < 0:
        SEQ_REORDER_MAX_BACK = 0
    U32_MOD = 1 << 32
    U32_HALF = 1 << 31

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RCVBUF_BYTES)
        sock.bind((PC_IP, PORT))
        sock.settimeout(0.2)
    except Exception:
        try:
            sock.close()
        except Exception:
            pass
        raise

    packet_buf = bytearray(2048)
    packet_mv = memoryview(packet_buf)
    packet_view = packet_mv.cast("B")

    shm_view = memoryview(shm_frames).cast("B")

    # Stato frame corrente
    curr_slot = None
    frame_view = None
    have_slot = False  # True se stiamo scrivendo in SHM, False se stiamo "droppando" (no slot)
    w = 0

    payload_len_ref = None
    last_seq = None

    # byte_count: 48-bit modulo 2^48
    last_byte_count = None
    MOD48 = 1 << 48

    frame_ok = True
    pkts_local = 0
    t_flush = time.perf_counter()
    last_packet_perf = t_flush
    publish_seq = 0
    RX_RUNNING = 1
    RX_STALLED = 0
    rx_state = RX_RUNNING

    # Capture control
    pending = False
    pending_start_t = 0.0
    pending_pos_id = 0

    def _bump_counter(counter: Synchronized) -> None:
        try:
            with counter.get_lock():
                counter.value += 1
        except Exception:
            pass

    def _next_capture_session_id() -> int:
        """Calcola un nuovo session ID senza pubblicarlo ancora al logger."""
        with cap_id.get_lock():
            next_cap_id = (int(cap_id.value) + 1) & 0xFFFFFFFF
        # Zero è riservato a "nessuna sessione" nel protocollo logger.
        return 1 if next_cap_id == 0 else int(next_cap_id)

    def _publish_capture_setup_failure(capture_file_id: int, session_id: int) -> None:
        """Sblocca il chiamante anche se il metadata non è pubblicabile."""
        with cap_saved.get_lock():
            cap_saved.value = 0
        with cap_pos_id.get_lock():
            cap_pos_id.value = int(capture_file_id)
        with cap_cancel_id.get_lock():
            cap_cancel_id.value = 0
        with cap_result.get_lock():
            cap_result.value = -1
        with cap_done_id.get_lock():
            cap_done_id.value = int(session_id)
        # Pubblicare cap_id per ultimo evita che il manager/logger osservino
        # una sessione per cui non sia già disponibile un esito terminale.
        with cap_id.get_lock():
            cap_id.value = int(session_id)
        with publish_lock:
            cap_active.value = 0

    def _poll_commands(now_perf: float) -> None:
        nonlocal pending, pending_start_t, pending_pos_id
        while True:
            try:
                cmd = cmd_queue.get_nowait()
            except pyqueue.Empty:
                break
            if not cmd:
                continue
            cmd_type = str(cmd[0]).strip().upper()
            if cmd_type == "CAPTURE_STOP":
                pending = False
                # La transizione e il tagging dei frame condividono lo stesso
                # lock: dopo questo punto RX non puo' pubblicare nuovi slot
                # appartenenti alla sessione annullata.
                with publish_lock:
                    cap_active.value = 0
                with cap_id.get_lock():
                    cancelled_id = int(cap_id.value)
                if cancelled_id != 0:
                    # Il logger pubblica cap_done_id solo dopo aver chiuso il
                    # file (anche se non era ancora riuscito ad aprirlo).
                    with cap_cancel_id.get_lock():
                        cap_cancel_id.value = int(cancelled_id)
                continue
            if cmd_type == "CAPTURE":
                capture_file_id = int(cmd[1])
                metadata_raw = cmd[2] if len(cmd) >= 3 and isinstance(cmd[2], dict) else {}
                try:
                    metadata = normalize_capture_metadata(capture_file_id, metadata_raw)
                    next_cap_id = _next_capture_session_id()
                    # Il blob è completo e legato al session ID prima di
                    # rendere visibile il nuovo cap_id al logger.
                    write_capture_metadata(capture_metadata, next_cap_id, metadata)
                except CaptureError:
                    _publish_capture_setup_failure(capture_file_id, _next_capture_session_id())
                    continue

                pending_pos_id = int(capture_file_id)
                pending_start_t = now_perf + max(0.0, float(settling_delay_s))
                pending = True

                # reset capture counters shared
                with cap_saved.get_lock():
                    cap_saved.value = 0
                with cap_pos_id.get_lock():
                    cap_pos_id.value = pending_pos_id

                # Pubblica la sessione solo dopo la scrittura atomica del
                # metadata condiviso (vedi write_capture_metadata sopra).
                with cap_id.get_lock():
                    cap_id.value = int(next_cap_id)
                with cap_cancel_id.get_lock():
                    cap_cancel_id.value = 0
                with cap_result.get_lock():
                    cap_result.value = 0

                # not active until settling passes
                with publish_lock:
                    cap_active.value = 0

    def _maybe_enable_capture(now_perf: float) -> None:
        nonlocal pending
        # Si arma soltanto tra due frame. In questo modo il primo frame salvato
        # e' stato acquisito interamente dopo il tempo di assestamento.
        if pending and w == 0 and (now_perf >= pending_start_t):
            pending = False
            with publish_lock:
                cap_active.value = 1

    def _free_current_slot():
        nonlocal curr_slot, frame_view, have_slot
        # Drop only local references; do not release published slots here.
        curr_slot = None
        frame_view = None
        have_slot = False

    def _discard_partial_frame() -> None:
        """Discard current in-progress frame and release acquired slot if any."""
        nonlocal w, frame_ok
        if have_slot and (curr_slot is not None):
            slot = int(curr_slot)
            slot_ok[slot] = 0
            slot_usemask[slot] = 0
            slot_pub_seq[slot] = 0
            slot_state[slot] = 0
            slot_pos_id[slot] = -1
            try:
                free_slots.put_nowait(slot)
            except Exception:
                pass
        w = 0
        frame_ok = True
        _free_current_slot()

    def _soft_reset_stream() -> None:
        """Reset RX continuity trackers and release partial local resources."""
        nonlocal payload_len_ref, last_seq, last_byte_count
        _discard_partial_frame()
        payload_len_ref = None
        last_seq = None
        last_byte_count = None
        _bump_counter(stream_resets)

    def _hard_resync_from_byte_count(byte_count: int) -> None:
        """
        Realign the local frame cursor to the absolute stream position carried by byte_count.

        If the packet starts mid-frame we logically consume the missing prefix and suppress
        publication until the next physical frame boundary.
        """
        nonlocal w, frame_ok
        frame_offset = int(byte_count % BYTES_PER_FRAME)
        _discard_partial_frame()
        if frame_offset > 0:
            w = frame_offset
            frame_ok = False
        _bump_counter(stream_resets)

    def _acquire_slot_nonblocking() -> bool:
        """Prende uno slot senza mai bloccare. Se non disponibile -> drop frame (have_slot=False)."""
        nonlocal curr_slot, frame_view, have_slot
        try:
            curr_slot = free_slots.get_nowait()
        except pyqueue.Empty:
            curr_slot = None
            frame_view = None
            have_slot = False
            return False
        base = int(curr_slot) * BYTES_PER_FRAME
        frame_view = shm_view[base : base + BYTES_PER_FRAME]
        have_slot = True
        return True

    def _ensure_slot() -> None:
        """Assicura che ci sia un contesto di frame (slot o drop-mode) quando w==0.""" 
        nonlocal w
        if w != 0:
            return
        # prova ad allocare slot; se non c'Ã¨, restiamo in drop-mode ma manteniamo l'allineamento
        _acquire_slot_nonblocking()

    def _publish_frame() -> None:
        nonlocal w, frame_ok, publish_seq
        # frame completo (w==BYTES_PER_FRAME)
        if have_slot and (curr_slot is not None):
            if frame_ok:
                slot = int(curr_slot)
                # --- ATOMIC PUBLISH (coherent READY+slot+seq) ---
                next_seq = 0
                with publish_lock:
                    # Leggere cap_active sotto lo stesso lock usato dal logger
                    # per il drain impedisce che uno slot riceva LOGGER_BIT
                    # subito dopo la chiusura di una cattura.
                    m = 1  # DSP always consumes
                    if int(cap_active.value) == 1:
                        slot_pos_id[slot] = int(cap_pos_id.value)
                        m |= 2  # LOGGER also consumes during capture
                    else:
                        slot_pos_id[slot] = -1
                    next_seq = int(publish_seq) + 1
                    slot_ok[slot] = 1
                    slot_usemask[slot] = m
                    slot_pub_seq[slot] = next_seq
                    slot_state[slot] = 1  # READY (set before seq bump)
                    publish_seq = next_seq
                try:
                    dsp_ready_queue.put_nowait((int(next_seq), int(slot)))
                except Exception:
                    pass
                if DEBUG_STATS and rx_frames_ok is not None:
                    with rx_frames_ok.get_lock():
                        rx_frames_ok.value += 1
            else:
                # corrotto: scarta
                slot = int(curr_slot)
                slot_ok[slot] = 0
                slot_usemask[slot] = 0
                slot_pub_seq[slot] = 0
                slot_state[slot] = 0
                try:
                    free_slots.put_nowait(slot)
                except Exception:
                    pass
        else:
            if rx_put_drops is not None:
                with rx_put_drops.get_lock():
                    rx_put_drops.value += 1
        # se non avevamo slot: era drop-mode, non pubblichiamo nulla

        # reset stato frame
        w = 0
        frame_ok = True
        _free_current_slot()
        # Il deadline puo' scadere nel mezzo di un pacchetto UDP: questo e' il
        # primo boundary sicuro da cui iniziare a taggare la cattura.
        _maybe_enable_capture(time.perf_counter())

    try:
        while not stop_evt.is_set():
            now_perf = time.perf_counter()
            _poll_commands(now_perf)
            _maybe_enable_capture(now_perf)
    
            try:
                n_bytes, _ = sock.recvfrom_into(packet_mv)
                pkts_local += 1
                recv_perf = time.perf_counter()
                try:
                    with rx_last_packet_time_s.get_lock():
                        rx_last_packet_time_s.value = time.time()
                except Exception:
                    pass
                if recv_perf - t_flush >= 0.1:
                    with rx_pkts.get_lock():
                        rx_pkts.value += pkts_local
                    pkts_local = 0
                    t_flush = recv_perf
            except socket.timeout:
                if rx_state == RX_RUNNING and (time.perf_counter() - last_packet_perf) >= STALL_TIMEOUT_S:
                    rx_state = RX_STALLED
                    _bump_counter(stall_events)
                continue
    
            if n_bytes <= HEADER_LEN:
                continue
    
            last_packet_perf = time.perf_counter()
    
            if rx_state == RX_STALLED:
                _soft_reset_stream()
                rx_state = RX_RUNNING
    
            current_payload_len = n_bytes - HEADER_LEN
            if payload_len_ref is None:
                payload_len_ref = current_payload_len
    
            # --- parse header ---
            seq = int.from_bytes(packet_view[0:4], "little", signed=False)
    
            # uint48 little-endian
            bc = int.from_bytes(packet_view[4:10], "little", signed=False) & (MOD48 - 1)
    
            # --- seq continuity: forward, reorder, restart ---
            seq_gap_pkts = 0
            if last_seq is not None:
                delta = (int(seq) - int(last_seq)) & (U32_MOD - 1)
                if delta == 0:
                    # duplicate packet
                    continue
                if delta < U32_HALF:
                    # forward progression (also handles uint32 wrap-around)
                    seq_gap_pkts = int(delta - 1)
                else:
                    # Out-of-order packet: a small jump is reordering; a large one is a stream restart.
                    back = (int(last_seq) - int(seq)) & (U32_MOD - 1)
                    if back <= SEQ_REORDER_MAX_BACK:
                        continue
                    _soft_reset_stream()
                    payload_len_ref = current_payload_len
                    seq_gap_pkts = 0
    
            if seq_gap_pkts > 0 and payload_len_ref is not None:
                with lost_pkts.get_lock():
                    lost_pkts.value += int(seq_gap_pkts)
    
            # --- byte_count check (absolute stream alignment) ---
            hard_resync = False
            if last_byte_count is not None and payload_len_ref is not None:
                expected = (last_byte_count + ((int(seq_gap_pkts) + 1) * int(payload_len_ref))) % MOD48
                if bc != expected:
                    _hard_resync_from_byte_count(bc)
                    payload_len_ref = current_payload_len
                    hard_resync = True
            last_byte_count = bc
    
            # --- GAP handling (consume missing bytes to keep frame alignment) ---
            if (not hard_resync) and seq_gap_pkts > 0 and payload_len_ref is not None:
                frame_ok = False
    
                bytes_missing = int(seq_gap_pkts) * int(payload_len_ref)
                while bytes_missing > 0 and (not stop_evt.is_set()):
                    _ensure_slot()
                    take = min(bytes_missing, BYTES_PER_FRAME - w)
                    # consume without writing (keep alignment)
                    w += take
                    bytes_missing -= take
                    if w == BYTES_PER_FRAME:
                        _publish_frame()
    
            last_seq = seq
    
            # --- process current payload ---
            off = HEADER_LEN
            current_payload_len = n_bytes - off
            payload_cursor = 0
    
            while payload_cursor < current_payload_len and (not stop_evt.is_set()):
                _ensure_slot()
                chunk_size = min(current_payload_len - payload_cursor, BYTES_PER_FRAME - w)
    
                if have_slot and (frame_view is not None):
                    start_src = off + payload_cursor
                    frame_view[w : w + chunk_size] = packet_view[start_src : start_src + chunk_size]
    
                w += chunk_size
                payload_cursor += chunk_size
    
                if w == BYTES_PER_FRAME:
                    _publish_frame()
    
    
    finally:
        try:
            _discard_partial_frame()
        except Exception:
            pass
        payload_len_ref = None
        last_seq = None
        last_byte_count = None
        try:
            sock.close()
        except Exception:
            pass

def _read_shared_text(buffer) -> str:
    """Legge una stringa terminata da NUL da un ``multiprocessing.Array``."""
    lock_getter = getattr(buffer, "get_lock", None)
    if callable(lock_getter):
        with lock_getter():
            chars = buffer[:]
    else:
        chars = buffer[:]
    return "".join(chars).split("\x00", 1)[0]


def _write_shared_text(buffer, value: str) -> None:
    """Scrive una stringa in un ``multiprocessing.Array`` senza usare ``.value``.

    ``Array('u', ...)`` restituisce un ``SynchronizedArray`` su Windows e non
    espone l'attributo ``value`` dei ``Value``: usare esplicitamente lo slice
    lo rende portabile anche nel processo figlio del logger.
    """
    text = str(value)
    if "\x00" in text or len(text) >= len(buffer):
        raise ValueError("output folder path is invalid or too long")
    lock_getter = getattr(buffer, "get_lock", None)
    if callable(lock_getter):
        with lock_getter():
            buffer[:] = "\x00" * len(buffer)
            buffer[:len(text)] = text
    else:
        buffer[:] = "\x00" * len(buffer)
        buffer[:len(text)] = text


def logger_worker(
    free_slots: Queue,
    shm_frames,
    slot_state,
    slot_ok,
    slot_pos_id,
    slot_usemask,
    publish_lock,
    cap_active: Synchronized,
    cap_pos_id: Synchronized,
    cap_id: Synchronized,
    cap_saved: Synchronized,
    capture_metadata: CaptureMetadataStore,
    cap_cancel_id: Synchronized,
    cap_done_id: Synchronized,
    cap_result: Synchronized,
    log_bytes: Synchronized,
    stop_evt,
    out_dir_shared,
    frames_per_position: int,
    block_frames: int = 16,
    ready_evt=None,
):
    """
    Logger capture-only (reworked for performance):
      - produce 1 file .bin per click: capture_pos{pos_id}.bin
      - prepend header (magic+json) con metadata posizione/data/radar/capture
      - scrive a blocchi (block_frames) per ridurre overhead I/O
      - salva SOLO frame validi (slot_ok==1) e SOLO quando cap_active==1
      - deve arrivare a frames_per_position frame completi per la posizione corrente.
        Se perde frame (ring overwrite / race), semplicemente continua finchÃ© non raggiunge X.

    ``capture_metadata`` è il blob JSON atomico associato al session ID.
    """
    # RX e logger condividono lo stesso ring: ogni slot porta un bit per
    # ciascun consumatore. Il logger libera solo il proprio bit dopo la copia.
    def _current_output_dir() -> Path:
        """Legge la run selezionata dalla GUI prima di aprire ogni file."""
        value = _read_shared_text(out_dir_shared)
        if not value:
            raise RuntimeError("capture output folder is not configured")
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        return path

    _current_output_dir()

    shm_view = memoryview(shm_frames).cast("B")

    last_seen_cap_id = int(cap_id.value)  # avoid auto-start at process boot
    pending_cap_id = None
    pending_pos_id = None

    fbin = None
    buf = None
    buf_used = 0
    saved_local = 0
    pos_local = -1
    file_cap_id: int | None = None

    # scan pointer to avoid always starting at 0
    scan_i = 0
    n_slots = len(slot_state)
    LOGGER_BIT = 2
    LOGGER_BUSY_BIT = 0x80

    def _mark_capture_finished(session_id: int | None, result: int) -> None:
        if session_id is None or int(session_id) == 0:
            return
        with cap_result.get_lock():
            cap_result.value = int(result)
        with cap_done_id.get_lock():
            cap_done_id.value = int(session_id)

    def _close_file() -> tuple[int | None, bool]:
        nonlocal fbin, buf, buf_used, saved_local, pos_local, file_cap_id
        closed_cap_id = file_cap_id
        io_ok = True
        if fbin is not None:
            try:
                if buf_used and buf is not None:
                    fbin.write(buf[:buf_used])
                    if log_bytes is not None:
                        with log_bytes.get_lock():
                            log_bytes.value += int(buf_used)
            except Exception:
                io_ok = False
            try:
                fbin.flush()
            except Exception:
                io_ok = False
            try:
                fbin.close()
            except Exception:
                io_ok = False
        fbin = None
        buf = None
        buf_used = 0
        saved_local = 0
        pos_local = -1
        file_cap_id = None
        return closed_cap_id, io_ok

    def _open_file(capture_file_id: int, session_id: int):
        nonlocal fbin, buf, buf_used, saved_local, pos_local, file_cap_id
        _close_file()
        metadata = read_capture_metadata(capture_metadata, int(session_id))
        if metadata is None:
            raise RuntimeError(f"missing capture metadata for session {int(session_id)}")
        header_blob = _build_capture_file_header(
            int(metadata["position"]),
            carriage_position_mm=metadata.get("carriage_position_mm"),
            carriage_microsteps=metadata.get("carriage_microsteps"),
        )
        p = _current_output_dir() / f"capture_pos{int(capture_file_id)}.bin"
        opened_file = open(p, "wb", buffering=1024 * 1024)
        try:
            opened_file.write(header_blob)
        except Exception:
            opened_file.close()
            raise
        fbin = opened_file
        pos_local = int(capture_file_id)
        file_cap_id = int(session_id)
        saved_local = 0
        buf_used = 0
        buf = bytearray(BYTES_PER_FRAME * int(block_frames))
        if log_bytes is not None:
            with log_bytes.get_lock():
                log_bytes.value += int(len(header_blob))
        with cap_saved.get_lock():
            cap_saved.value = 0

    def _claim_slot_for_logger(posid: int):
        """Atomically claim one slot for logger copy without releasing it yet."""
        nonlocal scan_i
        for _ in range(n_slots):
            s = scan_i
            scan_i = (scan_i + 1) % n_slots
            with publish_lock:
                if int(slot_state[s]) != 1:
                    continue
                if int(slot_pos_id[s]) != int(posid):
                    continue
                m = int(slot_usemask[s])
                if (m & LOGGER_BIT) == 0:
                    continue
                if (m & LOGGER_BUSY_BIT) != 0:
                    continue
                ok = int(slot_ok[s])
                slot_usemask[s] = m | LOGGER_BUSY_BIT
                return s, ok
        return -1, 0

    def _finalize_logger_slot(s: int) -> None:
        """Clear logger bits and free slot only when no consumer still needs it."""
        to_free = False
        with publish_lock:
            m = int(slot_usemask[s])
            m &= ~(LOGGER_BIT | LOGGER_BUSY_BIT)
            slot_usemask[s] = m
            if m == 0:
                slot_state[s] = 0
                to_free = True
        if to_free:
            try:
                free_slots.put_nowait(int(s))
            except Exception:
                pass

    def _stop_capture_and_release_logger_slots() -> None:
        """Ferma il tagging RX e rilascia tutti gli slot rimasti al logger.

        RX può pubblicare più frame di quanti ne servano mentre il logger sta
        copiando/flushando l'ultimo blocco. Senza questo drain il bit LOGGER
        residuo impedirebbe al DSP di riciclare quegli slot.
        """
        to_free: list[int] = []
        with publish_lock:
            cap_active.value = 0
            for s in range(n_slots):
                m = int(slot_usemask[s])
                if (m & (LOGGER_BIT | LOGGER_BUSY_BIT)) == 0:
                    continue
                m &= ~(LOGGER_BIT | LOGGER_BUSY_BIT)
                slot_usemask[s] = m
                slot_pos_id[s] = -1
                if m == 0:
                    slot_state[s] = 0
                    to_free.append(int(s))
        for s in to_free:
            try:
                free_slots.put_nowait(int(s))
            except Exception:
                pass

    if ready_evt is not None:
        ready_evt.set()

    try:
        while not stop_evt.is_set():
            capid = int(cap_id.value)
            posid = int(cap_pos_id.value)
            cancelled_id = int(cap_cancel_id.value)
            completed_id = int(cap_done_id.value)
    
            # Detect a new CAPTURE click (cap_id increments in RX when GUI sends CAPTURE)
            if capid != last_seen_cap_id:
                last_seen_cap_id = capid
                pending_cap_id = capid
                pending_pos_id = posid
                # cap_active will be set to 1 by RX after settling_delay_s
                # (logger must NOT enable capture by itself)
                with cap_saved.get_lock():
                    cap_saved.value = 0
    
            # Start logging only when capture is ACTIVE (set by RX) and we have a pending click
            if fbin is None:
                # RX può pubblicare un esito immediato se il metadata non è
                # serializzabile o non entra nel blob condiviso.
                if pending_cap_id is not None and int(completed_id) == int(pending_cap_id):
                    pending_cap_id = None
                    pending_pos_id = None
                    time.sleep(0.002)
                    continue
                if pending_cap_id is not None and int(cancelled_id) == int(pending_cap_id):
                    _stop_capture_and_release_logger_slots()
                    _mark_capture_finished(int(pending_cap_id), -1)
                    pending_cap_id = None
                    pending_pos_id = None
                    time.sleep(0.002)
                    continue
                if pending_cap_id is None or int(cap_active.value) != 1:
                    time.sleep(0.002)
                    continue
                try:
                    _open_file(int(pending_pos_id), int(pending_cap_id))
                except Exception:
                    _stop_capture_and_release_logger_slots()
                    _mark_capture_finished(int(pending_cap_id), -1)
                    pending_cap_id = None
                    pending_pos_id = None
                    time.sleep(0.002)
                    continue
                pending_cap_id = None
                pending_pos_id = None

            # If capture was externally stopped, close the file and idle
            if int(cancelled_id) == int(file_cap_id or -1) or int(cap_active.value) != 1:
                _stop_capture_and_release_logger_slots()
                closed_id, _close_ok = _close_file()
                _mark_capture_finished(closed_id, -1)
                time.sleep(0.002)
                continue
    
            # scan a bounded number of slots per tick to keep CPU low
            did_work = False
            while saved_local < int(frames_per_position):
                s, ok = _claim_slot_for_logger(int(posid))
                if s < 0:
                    break
    
                try:
                    if ok == 1:
                        base = int(s) * BYTES_PER_FRAME
                        assert buf is not None and fbin is not None
                        buf[buf_used : buf_used + BYTES_PER_FRAME] = shm_view[base : base + BYTES_PER_FRAME]
                        buf_used += BYTES_PER_FRAME
                        saved_local += 1
                        did_work = True

                        with cap_saved.get_lock():
                            cap_saved.value = int(saved_local)

                        if buf_used >= BYTES_PER_FRAME * int(block_frames):
                            fbin.write(buf[:buf_used])
                            if log_bytes is not None:
                                with log_bytes.get_lock():
                                    log_bytes.value += int(buf_used)
                            buf_used = 0
                finally:
                    # always release logger ownership, even for corrupt slot
                    _finalize_logger_slot(int(s))
    
            # stop condition: reached target
            if saved_local >= int(frames_per_position):
                _stop_capture_and_release_logger_slots()
                closed_id, close_ok = _close_file()
                # cap_done_id e' una conferma di persistenza: un errore di
                # write/flush/close non puo' essere pubblicato come successo.
                _mark_capture_finished(closed_id, 1 if close_ok else -1)
                continue
    
    
            if not did_work:
                time.sleep(0.002)
    
    
    finally:
        _stop_capture_and_release_logger_slots()
        closed_id, _close_ok = _close_file()
        _mark_capture_finished(closed_id, -1)

def main():
    """Crea la GUI e orchestra le risorse delle pipeline realtime e SAR.

    Tutte le risorse esterne vengono registrate in ``shutdown_resources`` al
    momento della creazione: il callback ``_shutdown`` può così ripulire anche
    un avvio parziale o un'eccezione durante l'inizializzazione della GUI.
    """
    shutdown_state = {
        "in_progress": False,
        "done": False,
        "dpg_context_created": False,
        "dpg_context_destroyed": False,
    }
    shutdown_resources = {
        "processes": [],
        "queues": [],
        "mp_refs": [],
        "stop_evt": None,
        "offline_runtime": None,
        "mmwave_bridge": None,
        "motor_controller": None,
        "sar_scan": None,
    }
    mmwave_auto_rearm_armed = False

    def _track_process(proc):
        if proc is not None:
            shutdown_resources["processes"].append(proc)
        return proc

    def _track_queue(queue_obj):
        if queue_obj is not None:
            shutdown_resources["queues"].append(queue_obj)
        return queue_obj

    def _track_mp_ref(obj):
        if obj is not None:
            shutdown_resources["mp_refs"].append(obj)
        return obj

    def _shutdown():
        """Arresto idempotente in ordine: scansione/hardware, worker, GUI."""
        nonlocal mmwave_auto_rearm_armed
        if shutdown_state["done"] or shutdown_state["in_progress"]:
            return
        shutdown_state["in_progress"] = True
        try:
            mmwave_auto_rearm_armed = False

            scan = shutdown_resources.get("sar_scan")
            if scan is not None:
                try:
                    scan.cancel()
                    scan.join(timeout=0.5)
                except Exception:
                    pass

            motor_controller = shutdown_resources.get("motor_controller")
            if motor_controller is not None:
                try:
                    motor_controller.disconnect()
                except Exception:
                    pass

            bridge = shutdown_resources.get("mmwave_bridge")
            if bridge is not None:
                try:
                    if bridge.is_streaming:
                        bridge.stop_streaming(stop_delay_s=0.25)
                except Exception:
                    pass
                try:
                    bridge.disconnect_hardware(stop_delay_s=0.25)
                except TypeError:
                    try:
                        bridge.disconnect_hardware()
                    except Exception:
                        pass
                except Exception:
                    pass

            stop_evt_obj = shutdown_resources.get("stop_evt")
            if stop_evt_obj is not None:
                try:
                    stop_evt_obj.set()
                except Exception:
                    pass

            offline_runtime_obj = shutdown_resources.get("offline_runtime")
            if offline_runtime_obj is not None:
                try:
                    offline_runtime_obj.stop()
                except Exception:
                    pass

            cleanup_processes(
                shutdown_resources.get("processes", ()),
                graceful_timeout_s=0.4,
                terminate_timeout_s=0.2,
                close_handles=True,
            )
            close_queues(shutdown_resources.get("queues", ()))

            shutdown_resources["processes"].clear()
            shutdown_resources["queues"].clear()
            shutdown_resources["mp_refs"].clear()
            shutdown_resources["stop_evt"] = None
            shutdown_resources["offline_runtime"] = None
            shutdown_resources["mmwave_bridge"] = None
            shutdown_resources["motor_controller"] = None
            shutdown_resources["sar_scan"] = None

            if shutdown_state["dpg_context_created"] and not shutdown_state["dpg_context_destroyed"]:
                try:
                    dpg.destroy_context()
                except Exception:
                    pass
                shutdown_state["dpg_context_destroyed"] = True
        finally:
            shutdown_state["done"] = True
            shutdown_state["in_progress"] = False

    atexit.register(_shutdown)

    # --- SETUP CODE E PROCESSI ---
    free_slots = _track_queue(Queue())
    dsp_ready_queue = _track_queue(Queue())

    # Ring slots (user requested 64)
    N_SLOTS = 64
    shm_frames = _track_mp_ref(RawArray("B", N_SLOTS * BYTES_PER_FRAME))
    for i in range(N_SLOTS):
        free_slots.put(i)

    # Slot metadata (shared)
    # slot_state: 0=FREE, 1=READY
    slot_state = _track_mp_ref(RawArray("b", N_SLOTS))
    slot_ok = _track_mp_ref(RawArray("b", N_SLOTS))
    slot_pos_id = _track_mp_ref(RawArray("i", N_SLOTS))
    for i in range(N_SLOTS):
        slot_state[i] = 0
        slot_ok[i] = 0
        slot_pos_id[i] = -1

    # slot_usemask: bit0=DSP, bit1=LOGGER (capture)
    slot_usemask = _track_mp_ref(RawArray("B", N_SLOTS))
    slot_pub_seq = _track_mp_ref(RawArray("Q", N_SLOTS))
    for i in range(N_SLOTS):
        slot_usemask[i] = 0
        slot_pub_seq[i] = 0

    publish_lock = _track_mp_ref(mp.Lock())  # atomic publish lock (seq+slot+state)


    # Capture shared state
    cap_active = _track_mp_ref(Value("i", 0))   # 1 while capturing
    cap_pos_id = _track_mp_ref(Value("i", 0))   # current position id (from GUI)
    cap_id = _track_mp_ref(Value("I", 0))       # increments each CAPTURE command
    cap_saved = _track_mp_ref(Value("i", 0))    # frames saved for current position
    # Un solo blob JSON protetto da lock evita che logger e RX associno al
    # medesimo cap_id una combinazione parziale di metadata scollegati.
    capture_metadata = CaptureMetadataStore(
        buffer=_track_mp_ref(RawArray("B", CAPTURE_METADATA_BUFFER_BYTES)),
        byte_count=_track_mp_ref(Value("I", 0)),
        session_id=_track_mp_ref(Value("I", 0)),
        lock=_track_mp_ref(mp.Lock()),
    )
    cap_cancel_id = _track_mp_ref(Value("I", 0))
    # Il logger pubblica questo ID solo dopo close+flush: è il segnale usato
    # dalla scansione prima di comandare il movimento successivo.
    cap_done_id = _track_mp_ref(Value("I", 0))
    cap_result = _track_mp_ref(Value("i", 0))  # 1=ok, -1=annullata, 0=in corso

    dr_shared = C * FS / (2.0 * SLOPE * NFFT_RANGE)
    # The realtime and offline rasters are fixed for the lifetime of their
    # respective shared buffers/textures.  They intentionally do not follow
    # range/angle NFFT dimensions.
    gui_h = max(1, int(display_image_resolutions_cfg.realtime.height))
    fft_plot_h = max(1, int(NFFT_RANGE))
    gui_w = max(1, int(display_image_resolutions_cfg.realtime.width))
    offline_gui_h = max(1, int(display_image_resolutions_cfg.offline.height))
    offline_gui_w = max(1, int(display_image_resolutions_cfg.offline.width))
    gui_dbuf = _track_mp_ref(RawArray("f", 2 * gui_h * gui_w))
    gui_alpha_dbuf = _track_mp_ref(RawArray("f", 2 * gui_h * gui_w))
    gui_prof_dbuf = _track_mp_ref(RawArray("f", 2 * RANGE_PROFILE_COUNT * fft_plot_h))
    gui_angle_diag_dbuf = _track_mp_ref(RawArray("f", 2 * fft_plot_h * ANGLEFFT_BINS))
    gui_doppler_diag_dbuf = _track_mp_ref(RawArray("f", 2 * fft_plot_h * DOPPLERFFT_BINS))
    gui_latest_idx = _track_mp_ref(Value("i", -1))
    gui_latest_seq = _track_mp_ref(Value("Q", 0))
    gui_lock = _track_mp_ref(mp.Lock())
    rt_home_viewport = build_display_viewport(
        x_min_m=-float(HEATMAP_CROSSRANGE_MAX_DISPLAY),
        x_max_m=float(HEATMAP_CROSSRANGE_MAX_DISPLAY),
        y_min_m=0.0,
        y_max_m=float(RANGE_MAX_DISPLAY),
        dr_m=float(dr_shared),
        seq=0,
    )
    rt_home_viewport_shared = _track_mp_ref(RawArray("d", 4))
    rt_home_viewport_lock = _track_mp_ref(mp.Lock())
    rt_requested_viewport = _track_mp_ref(RawArray("d", 4))
    rt_requested_viewport_seq = _track_mp_ref(Value("Q", 0))
    rt_requested_viewport_lock = _track_mp_ref(mp.Lock())
    rt_applied_viewport = _track_mp_ref(RawArray("d", 9))
    rt_applied_viewport_seq = _track_mp_ref(Value("Q", 0))
    rt_applied_viewport_frame_seq = _track_mp_ref(Value("Q", 0))
    rt_applied_viewport_fallback = _track_mp_ref(Value("b", 0))
    rt_applied_viewport_lock = _track_mp_ref(mp.Lock())
    with rt_home_viewport_lock:
        rt_home_viewport_shared[0] = float(rt_home_viewport.x_min_m)
        rt_home_viewport_shared[1] = float(rt_home_viewport.x_max_m)
        rt_home_viewport_shared[2] = float(rt_home_viewport.y_min_m)
        rt_home_viewport_shared[3] = float(rt_home_viewport.y_max_m)
    with rt_requested_viewport_lock:
        rt_requested_viewport[0] = float(rt_home_viewport.x_min_m)
        rt_requested_viewport[1] = float(rt_home_viewport.x_max_m)
        rt_requested_viewport[2] = float(rt_home_viewport.y_min_m)
        rt_requested_viewport[3] = float(rt_home_viewport.y_max_m)
    with rt_applied_viewport_lock:
        rt_applied_viewport[0] = float(rt_home_viewport.x_min_m)
        rt_applied_viewport[1] = float(rt_home_viewport.x_max_m)
        rt_applied_viewport[2] = float(rt_home_viewport.y_min_m)
        rt_applied_viewport[3] = float(rt_home_viewport.y_max_m)
        rt_applied_viewport[4] = float(rt_home_viewport.range_min_bin_f)
        rt_applied_viewport[5] = float(rt_home_viewport.range_max_bin_f)
        rt_applied_viewport[6] = float(rt_home_viewport.angle_min_deg)
        rt_applied_viewport[7] = float(rt_home_viewport.angle_max_deg)
        rt_applied_viewport[8] = float(rt_home_viewport.zoom_level)
    track_cfg = cfg.get("tracking", {}) or {}
    track_max_shared = max(1, int(track_cfg.get("max_tracks", 30)))
    gui_tracks_xy_dbuf = _track_mp_ref(RawArray("f", track_max_shared * 4))   # x, y, vx, vy
    gui_tracks_meta_dbuf = _track_mp_ref(RawArray("i", track_max_shared * 4))  # id, confirmed, age, missed
    gui_tracks_state_dbuf = _track_mp_ref(RawArray("i", track_max_shared * 2))  # motion_state_code, has_stop
    gui_tracks_stop_xy_dbuf = _track_mp_ref(RawArray("f", track_max_shared * 2))  # stop_x, stop_y
    gui_tracks_count = _track_mp_ref(Value("i", 0))
    gui_tracks_seq = _track_mp_ref(Value("Q", 0))
    gui_tracks_lock = _track_mp_ref(mp.Lock())

    cmd_q = _track_queue(Queue(maxsize=16))
    dsp_cmd_q = _track_queue(Queue(maxsize=16))

    stop_evt = _track_mp_ref(mp.Event())
    logger_ready_evt = _track_mp_ref(mp.Event())
    shutdown_resources["stop_evt"] = stop_evt
    sar_pos_counter = _track_mp_ref(Value("L", 0))  # GUI-only counter (pos id generator)
    capture_sessions = CaptureSessionManager(
        cmd_queue=cmd_q,
        cap_id=cap_id,
        cap_done_id=cap_done_id,
        cap_result=cap_result,
    )
    mmwave_bridge = MmwaveStudioBridge()
    shutdown_resources["mmwave_bridge"] = mmwave_bridge
    mmwave_radar_cfg = RadarConnectionConfig(
        uart_com_port=3,
        baudrate=921600,
        timeout_ms=1000,
    )
    mmwave_dca_cfg = DCA1000Config(
        pc_ip="192.168.33.30",
        capture_card_ip="192.168.33.180",
        capture_card_mac="12:34:56:78:90:12",
        mode_device_type=1,
        adc_data_path=Path(r"C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\PostProc\adc_data.bin"),
    )

    # Stats
    lost_pkts = _track_mp_ref(Value("L", 0))
    rx_pkts = _track_mp_ref(Value("L", 0))
    rx_last_packet_time_s = _track_mp_ref(Value("d", time.time()))
    rx_put_drops = _track_mp_ref(Value("L", 0))
    rx_frames_ok = _track_mp_ref(Value("L", 0))
    rx_stall_events = _track_mp_ref(Value("L", 0))
    rx_stream_resets = _track_mp_ref(Value("L", 0))
    dsp_skip = _track_mp_ref(Value("L", 0))
    dsp_ms_avg = _track_mp_ref(Value("d", 0.0))
    dsp_ms_p95 = _track_mp_ref(Value("d", 0.0))
    log_bytes = _track_mp_ref(Value("L", 0))
    norm_to_peak = _track_mp_ref(Value("b", 1))
    heatmap_view_mode = _track_mp_ref(Value("i", 0))  # 0=power XY, 1=projected XY moving velocity
    stat_raw_min_db = _track_mp_ref(Value("d", float("nan")))
    stat_raw_max_db = _track_mp_ref(Value("d", float("nan")))
    stat_norm_min_db = _track_mp_ref(Value("d", float("nan")))
    stat_norm_max_db = _track_mp_ref(Value("d", float("nan")))

    out_root = Path(__file__).with_name("logs")

    def _create_output_run() -> Path:
        """Crea una nuova run senza rischiare di riusare un nome nello stesso secondo."""
        out_root.mkdir(parents=True, exist_ok=True)
        stem = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        for suffix in range(10_000):
            name = stem if suffix == 0 else f"{stem}_{suffix:02d}"
            candidate = out_root / name
            try:
                candidate.mkdir(exist_ok=False)
                return candidate
            except FileExistsError:
                continue
        raise RuntimeError("Unable to create a new acquisition folder")

    out_dir = _create_output_run()
    # Il logger è un processo separato: legge questa stringa condivisa prima
    # di ogni nuova cattura, così la GUI può avviare una nuova run senza
    # fermare radar, DCA1000 o applicazione.
    output_dir_shared = _track_mp_ref(mp.Array("u", 1024, lock=True))
    _write_shared_text(output_dir_shared, str(out_dir))

    # --- AVVIO PROCESSI ---
    p_rx = Process(
        target=radar_rx,
        args=(
            cmd_q,
            free_slots,
            dsp_ready_queue,
            shm_frames,
            slot_state,
            slot_ok,
            slot_pos_id,
            slot_usemask,
            slot_pub_seq,
            publish_lock,
            cap_active,
            cap_pos_id,
            cap_id,
            cap_saved,
            capture_metadata,
            cap_cancel_id,
            cap_done_id,
            cap_result,
            lost_pkts,
            rx_pkts,
            rx_last_packet_time_s,
            rx_put_drops,
            rx_frames_ok,
            rx_stall_events,
            rx_stream_resets,
            stop_evt,
            SETTLING_DELAY_S,
        ),
    )
    _track_process(p_rx)

    p_log = Process(
        target=logger_worker,
        args=(
            free_slots,
            shm_frames,
            slot_state,
            slot_ok,
            slot_pos_id,
            slot_usemask,
            publish_lock,
            cap_active,
            cap_pos_id,
            cap_id,
            cap_saved,
            capture_metadata,
            cap_cancel_id,
            cap_done_id,
            cap_result,
            log_bytes,
            stop_evt,
            output_dir_shared,
            FRAMES_PER_POSITION,
            16,  # block_frames
            logger_ready_evt,
        ),
    )
    _track_process(p_log)

    p_dsp = Process(
        target=dsp_worker,
        args=(
            free_slots,
            dsp_ready_queue,
            dsp_cmd_q,
            shm_frames,
            slot_state,
            slot_ok,
            slot_usemask,
            slot_pub_seq,
            publish_lock,
            gui_dbuf,
            gui_prof_dbuf,
            gui_h,
            fft_plot_h,
            gui_w,
            gui_latest_idx,
            gui_latest_seq,
            gui_lock,
            gui_tracks_xy_dbuf,
            gui_tracks_meta_dbuf,
            gui_tracks_state_dbuf,
            gui_tracks_stop_xy_dbuf,
            gui_tracks_count,
            gui_tracks_seq,
            gui_tracks_lock,
            dsp_skip,
            dsp_ms_avg,
            dsp_ms_p95,
            norm_to_peak,
            stat_raw_min_db,
            stat_raw_max_db,
            stat_norm_min_db,
            stat_norm_max_db,
            stop_evt,
            cfg,
            REALTIME_DSP_CFG,
            heatmap_view_mode,
            gui_alpha_dbuf,
            rt_requested_viewport,
            rt_requested_viewport_seq,
            rt_requested_viewport_lock,
            rt_applied_viewport,
            rt_applied_viewport_seq,
            rt_applied_viewport_frame_seq,
            rt_applied_viewport_fallback,
            rt_applied_viewport_lock,
            rt_home_viewport_shared,
            rt_home_viewport_lock,
        ),
        kwargs={
            "gui_angle_diag_dbuf": gui_angle_diag_dbuf,
            "gui_doppler_diag_dbuf": gui_doppler_diag_dbuf,
            "angle_diag_w": int(ANGLEFFT_BINS),
            "doppler_diag_w": int(DOPPLERFFT_BINS),
        },
    )
    _track_process(p_dsp)

    p_rx.daemon = True
    p_log.daemon = True
    p_dsp.daemon = True
    p_rx.start()
    p_log.start()
    p_dsp.start()
    if not logger_ready_evt.wait(timeout=5.0):
        print("[LOGGER WARN] processo logger non pronto entro 5 s; catture temporaneamente non affidabili.")

    _apply_process_priority("MAIN", os.getpid(), PRIO_MAIN, PRIO_ENABLED)
    _apply_process_priority("RX", int(p_rx.pid or 0), PRIO_RX, PRIO_ENABLED)
    _apply_process_priority("LOG", int(p_log.pid or 0), PRIO_LOG, PRIO_ENABLED)
    _apply_process_priority("DSP", int(p_dsp.pid or 0), PRIO_DSP, PRIO_ENABLED)
    _apply_process_affinity("MAIN", os.getpid(), AFF_MAIN, AFF_ENABLED)
    _apply_process_affinity("RX", int(p_rx.pid or 0), AFF_RX, AFF_ENABLED)
    _apply_process_affinity("LOG", int(p_log.pid or 0), AFF_LOG, AFF_ENABLED)
    _apply_process_affinity("DSP", int(p_dsp.pid or 0), AFF_DSP, AFF_ENABLED)

    print(f"[LOGGER] dir: {out_dir}")
    print(f"[FFT] workers={FFT_WORKERS} logical_cpus={LOGICAL_CPUS}")
    print(f"[PRIO] enabled={PRIO_ENABLED} main={PRIO_MAIN} rx={PRIO_RX} log={PRIO_LOG} dsp={PRIO_DSP}")
    print(f"[AFF] enabled={AFF_ENABLED} main={AFF_MAIN} rx={AFF_RX} log={AFF_LOG} dsp={AFF_DSP}")
    if DEBUG_STATS and psutil is None:
        print("[STATS] psutil not available: using Windows fallback for CPU%.")

    # Il 1063 viene aperto soltanto da questa istanza di main.  La GUI
    # standalone resta utile per configurare phidget_stepper_config.yaml, ma
    # una volta iniziata la scansione non deve essere eseguita in parallelo.
    stepper_controller = None
    stepper_error = _PHIDGET_BACKEND_ERROR
    if PhidgetStepperController is not None and load_stepper_config is not None:
        try:
            stepper_controller = PhidgetStepperController(load_stepper_config())
            shutdown_resources["motor_controller"] = stepper_controller
        except Exception as exc:
            stepper_error = f"Phidget configuration unavailable: {exc}"
            print(f"[PHIDGET WARN] {stepper_error}")

    sar_scan = None
    if stepper_controller is not None:
        sar_scan = SarScanCoordinator(
            begin_motion=stepper_controller.begin_external_scan,
            finish_motion=stepper_controller.finish_external_scan,
            get_position_microsteps=stepper_controller.position_microsteps,
            mm_from_microsteps=stepper_controller.config.mechanics.mm_from_microsteps,
            mm_per_microstep=lambda: stepper_controller.config.mechanics.mm_per_microstep,
            move_to_microsteps=stepper_controller.move_absolute_microsteps_and_wait,
            request_capture=capture_sessions.request,
            wait_capture=capture_sessions.wait,
            cancel_capture=capture_sessions.cancel,
            stop_motion=stepper_controller.stop,
        )
        shutdown_resources["sar_scan"] = sar_scan

    def _create_offline_runtime():
        """Build an isolated offline runtime from the persisted configuration."""
        return OfflineBPRuntime(
            offline_config_path=Path(__file__).with_name("offline_config.yaml"),
            fallback_capture_cfg=Path(__file__).with_name("Config.yaml"),
            c_m_s=float(C),
            fs_hz=float(FS),
            slope_hz_s=float(SLOPE),
            fc_hz=float(FC),
            nfft_range=int(NFFT_RANGE),
            image_h=int(offline_gui_h),
            image_w=int(offline_gui_w),
        )

    offline_runtime = None
    offline_error = ""
    offline_info = {}
    offline_boot_cfg: dict = {}
    offline_map_bounds_boot = None
    offline_config_boot_path = Path(__file__).with_name("offline_config.yaml")
    try:
        with offline_config_boot_path.open("r", encoding="utf-8") as handle:
            offline_boot_cfg = yaml.safe_load(handle) or {}
        if not isinstance(offline_boot_cfg, dict):
            raise ValueError("offline_config.yaml must contain a YAML mapping")
        offline_map_bounds_boot = offline_map_bounds_from_yaml_dict(offline_boot_cfg, cfg)
        scan_boot = offline_boot_cfg.get("scan", {}) or {}
        x_start_boot = int(scan_boot.get("x_start", 1))
        x_end_boot = int(scan_boot.get("x_end", x_start_boot))
        if x_end_boot < x_start_boot:
            x_start_boot, x_end_boot = x_end_boot, x_start_boot
        offline_info = {
            "pos_min": int(x_start_boot),
            "pos_max": int(x_end_boot),
            "x_start": int(x_start_boot),
            "x_end": int(x_end_boot),
            "algorithm": str((offline_boot_cfg.get("reconstruction", {}) or {}).get("algorithm", "backprojection")),
        }
        print(
            f"[OFFLINE] deferred load configured positions={x_start_boot}..{x_end_boot}; "
            "press LOAD / CALCULATE OFFLINE in the GUI"
        )
    except Exception as exc:
        offline_error = str(exc)
        print(f"[OFFLINE WARN] {offline_error}")
    shutdown_resources["offline_runtime"] = None

    # =========================================================================
    # GUI SETUP (RESPONSIVE - CLEAN)
    # =========================================================================

    # display state (mode 0: cartesian X-Y power, mode 1: projected X-Y moving velocity)
    norm_enabled_init = bool(norm_to_peak.value)
    heatmap_mode_init = int(heatmap_view_mode.value)
    if heatmap_mode_init not in (0, 1):
        heatmap_mode_init = 0
    dr_plot = float(dr_shared)
    if heatmap_mode_init == 1:
        vis_vmin = float(HEATMAP_VELOCITY_VMIN_MPS)
        vis_vmax = float(HEATMAP_VELOCITY_VMAX_MPS)
    elif norm_enabled_init:
        vis_vmin = float(VMIN_NORM)
        vis_vmax = float(VMAX_NORM)
    else:
        vis_vmin = float(VMIN_RAW)
        vis_vmax = float(VMAX_RAW)
    if vis_vmax <= vis_vmin:
        vis_vmax = vis_vmin + 1.0
    vis_rmax = float(RANGE_MAX_DISPLAY)
    vis_xmax = float(HEATMAP_CROSSRANGE_MAX_DISPLAY)
    vis_heatmap_mode = int(heatmap_mode_init)
    vis_fft_mode_db = True
    vis_fft_view_full = False
    vis_fft_xmin = 0.0
    vis_fft_xmax = float(min(int(fft_plot_h), max(1, int(np.ceil(float(RANGE_MAX_DISPLAY) / max(float(dr_plot), 1e-9)))))) * float(dr_plot)
    vis_fft_vmin = float(RANGEFFT_DB_MIN)
    vis_fft_vmax = float(RANGEFFT_DB_MAX)
    if vis_fft_vmax <= vis_fft_vmin:
        vis_fft_vmax = vis_fft_vmin + 1.0
    fft_mode_db = bool(vis_fft_mode_db)
    fft_view_full = bool(vis_fft_view_full)
    angle_single_bin = False
    doppler_single_bin = False
    angle_selected_bin = 0
    doppler_selected_bin = 0
    vis_angle_diag_vmin = -50.0
    vis_angle_diag_vmax = 0.0
    vis_doppler_diag_vmin = -50.0
    vis_doppler_diag_vmax = 0.0
    angle_diag_norm_to_peak = True
    doppler_diag_norm_to_peak = True
    angle_axis_diag = build_angle_axis_deg(ANGLEFFT_BINS)
    angle_axis_finite = angle_axis_diag[np.isfinite(angle_axis_diag)]
    if angle_axis_finite.size > 0:
        angle_axis_min = float(np.min(angle_axis_finite))
        angle_axis_max = float(np.max(angle_axis_finite))
    else:
        angle_axis_min = float(-ANGLEFFT_BINS // 2)
        angle_axis_max = float(ANGLEFFT_BINS // 2)
    doppler_axis_diag = build_doppler_axis_mps(
        cfg,
        REALTIME_DSP_CFG,
        DOPPLERFFT_BINS,
        doppler_fft_shift=bool((cfg.get("detection_moving", {}) or {}).get("doppler_fft_shift", True)),
    )
    if doppler_axis_diag is None or int(np.asarray(doppler_axis_diag).size) != int(DOPPLERFFT_BINS):
        doppler_axis_diag = np.linspace(-1.0, 1.0, int(DOPPLERFFT_BINS), dtype=np.float32)
    else:
        doppler_axis_diag = np.asarray(doppler_axis_diag, dtype=np.float32)
    doppler_axis_min = float(np.min(doppler_axis_diag)) if doppler_axis_diag.size > 0 else -1.0
    doppler_axis_max = float(np.max(doppler_axis_diag)) if doppler_axis_diag.size > 0 else 1.0

    ui_dirty = True
    ui_dirty_t = 0.0
    ui_pending = {
        "vmin": vis_vmin,
        "vmax": vis_vmax,
        "rmax": vis_rmax,
        "xmax": vis_xmax,
        "fft_xmin": float(vis_fft_xmin),
        "fft_xmax": float(vis_fft_xmax),
        "fft_vmin": vis_fft_vmin,
        "fft_vmax": vis_fft_vmax,
        "fft_mode_db": bool(fft_mode_db),
        "fft_view_full": bool(fft_view_full),
        "heatmap_mode": int(vis_heatmap_mode),
        "reset_view": True,
    }

    offline_scan_config_path = Path(__file__).with_name("offline_config.yaml")
    try:
        scan_start_id_default, scan_positions_default, scan_pitch_mm_default = read_offline_scan_settings(
            offline_scan_config_path
        )
        scan_config_error = ""
    except Exception as exc:
        scan_start_id_default, scan_positions_default, scan_pitch_mm_default = 1, 1, float("nan")
        scan_config_error = str(exc)

    # 2) DearPyGui Init
    dpg.create_context()
    shutdown_state["dpg_context_created"] = True

    # Font
    font_mono = None
    with dpg.font_registry():
        try:
            font_ui = dpg.add_font(r"C:\Windows\Fonts\segoeui.ttf", 18)
            font_mono = dpg.add_font(r"C:\Windows\Fonts\consola.ttf", 16)
            dpg.bind_font(font_ui)
        except Exception:
            pass

    # 3) Tags (NO container/resize tags)
    TAG_MAIN_WINDOW = "primary_window"
    TAG_MAIN_TABBAR = "main_tabbar"
    TAB_REALTIME_TAG = "tab_tempo_reale"
    TAB_PROCESSED_TAG = "tab_dati_processati"
    TAB_TUNING_TAG = "tab_tuning_dsp"
    TAB_OFFLINE_TUNING_TAG = "tab_tuning_offline"
    TAG_SIDEBAR     = "sidebar_col"
    TAG_CBAR_COL    = "cbar_col"
    TAG_RANGEFFT_COL = "rangefft_col"

    TXT_PIPELINE_CONFIG_TAG = "txt_pipeline_config"
    TXT_DISPLAY_DIAG_TAG = "txt_display_diagnostics"
    DEBUG_STAT_VALUE_TAGS = {
        "udp_rx": "debug_stat_udp_rx",
        "frames": "debug_stat_frames",
        "ring": "debug_stat_ring",
        "drops": "debug_stat_drops",
        "dsp_frame": "debug_stat_dsp_frame",
        "dsp_stale": "debug_stat_dsp_stale",
        "image_updates": "debug_stat_image_updates",
        "cpu": "debug_stat_cpu",
        "logger": "debug_stat_logger",
        "stalls": "debug_stat_stalls",
        "resyncs": "debug_stat_resyncs",
    }
    TXT_TUNING_STATUS_TAG = "txt_tuning_status"
    IN_VMIN, IN_VMAX = "in_vmin", "in_vmax"
    IN_RMAX, IN_XMAX = "in_rmax", "in_xmax"
    IN_FFT_XMIN, IN_FFT_XMAX = "in_fft_xmin", "in_fft_xmax"
    IN_FFT_VMIN, IN_FFT_VMAX = "in_fft_vmin", "in_fft_vmax"
    TXT_POS_TAG = "txt_pos_counter"
    TXT_CAPTURE_STATUS_TAG = "txt_capture_status"
    TXT_MOTOR_STATUS_TAG = "txt_motor_status"
    TXT_MOTOR_POSITION_TAG = "txt_motor_position"
    TXT_MOTOR_LOG_TAG = "txt_motor_log"
    TXT_SCAN_STATUS_TAG = "txt_scan_status"
    TXT_SCAN_PITCH_TAG = "txt_sar_scan_pitch"
    TXT_RUN_DIR_TAG = "txt_sar_run_dir"
    IN_MOTOR_JOG_TAG = "in_motor_jog_mm"
    IN_SCAN_POSITIONS_TAG = "in_sar_scan_positions"
    BTN_CAPTURE_TAG = "btn_capture_frame"
    BTN_NEW_SESSION_TAG = "btn_new_sar_session"
    BTN_SCAN_START_TAG = "btn_sar_scan_start"
    BTN_SCAN_CANCEL_TAG = "btn_sar_scan_cancel"
    BTN_NORM_TAG = "btn_norm_toggle"
    BTN_HEATMAP_MODE_TAG = "btn_heatmap_mode"
    BTN_FFT_VIEW_TAG = "btn_fft_view"
    BTN_FFT_MODE_TAG = "btn_fft_mode"
    BTN_MMWAVE_CONNECT_TAG = "btn_mmwave_connect"
    BTN_MMWAVE_STREAM_TAG = "btn_mmwave_stream"
    TXT_MMWAVE_STATUS_TAG = "txt_mmwave_status"
    TXT_MMWAVE_LINKS_TAG = "txt_mmwave_links"

    TEX_TAG = "heat_tex"
    HEAT_PLOT_TAG = "heat_plot"
    XAXIS_TAG, YAXIS_TAG = "xaxis", "yaxis"
    IMG_SERIES_TAG = "img_series"
    GUIDE_POS20_TAG = "guide_pos20"
    GUIDE_NEG20_TAG = "guide_neg20"
    TRACK_SCATTER_CONF_TAG = "track_scatter_confirmed"
    TRACK_SCATTER_UNCONF_TAG = "track_scatter_unconfirmed"
    TRACK_SCATTER_MOVING_TAG = "track_scatter_moving"
    TRACK_SCATTER_STOPPED_TAG = "track_scatter_stopped"
    TRACK_SCATTER_UNKNOWN_TAG = "track_scatter_unknown"
    TRACK_STOP_MARKER_TAG = "track_stop_marker"
    TRACK_VEL_SERIES_TAG = "track_velocity_series"
    TRACK_ANN_PREFIX = "track_ann_"
    TRACK_VEL_SCALE = 0.25
    PROC_TEX_TAG = "proc_heat_tex"
    PROC_HEAT_PLOT_TAG = "proc_heat_plot"
    PROC_XAXIS_TAG, PROC_YAXIS_TAG = "proc_xaxis", "proc_yaxis"
    PROC_IMG_SERIES_TAG = "proc_img_series"
    PROC_IN_XSTART = "proc_in_xstart"
    PROC_IN_XEND = "proc_in_xend"
    PROC_GRP_LINEAR_POSITION_SELECTION = "proc_grp_linear_position_selection"
    PROC_BTN_LOAD_OFFLINE = "proc_btn_load_offline"
    PROC_TXT_MEMORY_ESTIMATE = "proc_txt_memory_estimate"
    PROC_IN_VMIN = "proc_in_vmin"
    PROC_IN_VMAX = "proc_in_vmax"
    PROC_BTN_NORM = "proc_btn_norm"
    PROC_IN_ZOOM_XMIN = "proc_in_zoom_xmin"
    PROC_IN_ZOOM_XMAX = "proc_in_zoom_xmax"
    PROC_IN_ZOOM_YMIN = "proc_in_zoom_ymin"
    PROC_IN_ZOOM_YMAX = "proc_in_zoom_ymax"
    PROC_TXT_ZOOM_STATUS = "proc_txt_zoom_status"
    PROC_CMAP_SCALE_TAG = "proc_cmap_scale"
    PROC_CMAP_NUM_FMT = "%+6.1f"
    TXT_OFFLINE_TUNING_STATUS_TAG = "txt_offline_tuning_status"
    TXT_OFFLINE_TUNING_INFO_TAG = "txt_offline_tuning_info"
    BTN_OFFLINE_INSPECT_RUN = "btn_offline_inspect_run"
    TXT_OFFLINE_RUN_FORMAT = "txt_offline_run_format"
    OFFLINE_TUNING_GRP_LINEAR_SCAN = "offline_tuning_grp_linear_scan"
    OFFLINE_TUNING_GRP_RECONSTRUCTION = "offline_tuning_grp_reconstruction"
    OFFLINE_TUNING_GRP_MAP_BOUNDS = "offline_tuning_grp_map_bounds"
    CMAP_SCALE_TAG = "cmap_scale"
    CMAP_NUM_FMT = "%+6.1f"
    CMAP_VELOCITY_NUM_FMT = "%+5.2f"
    CMAP_VELOCITY_TAG = "cmap_velocity_bwr"
    RANGEFFT_PLOT_TAG = "rangefft_plot"
    RANGEFFT_XAXIS_TAG = "rangefft_xaxis"
    RANGEFFT_YAXIS_TAG = "rangefft_yaxis"
    RANGEFFT_LINE_TAGS = [f"rangefft_line_ant{ant_i}" for ant_i in range(RANGE_PROFILE_COUNT)]
    FFT_DIAG_TABBAR_TAG = "fft_diag_tabbar"
    ANGLEFFT_TEX_TAG = "anglefft_tex"
    DOPPLERFFT_TEX_TAG = "dopplerfft_tex"
    ANGLEFFT_HEAT_PLOT_TAG = "anglefft_heat_plot"
    ANGLEFFT_PROFILE_PLOT_TAG = "anglefft_profile_plot"
    ANGLEFFT_XAXIS_TAG = "anglefft_xaxis"
    ANGLEFFT_YAXIS_TAG = "anglefft_yaxis"
    ANGLEFFT_PROFILE_XAXIS_TAG = "anglefft_profile_xaxis"
    ANGLEFFT_PROFILE_YAXIS_TAG = "anglefft_profile_yaxis"
    ANGLEFFT_IMG_SERIES_TAG = "anglefft_img_series"
    ANGLEFFT_PROFILE_LINE_TAG = "anglefft_profile_line"
    ANGLEFFT_CMAP_SCALE_TAG = "anglefft_cmap_scale"
    DOPPLERFFT_HEAT_PLOT_TAG = "dopplerfft_heat_plot"
    DOPPLERFFT_PROFILE_PLOT_TAG = "dopplerfft_profile_plot"
    DOPPLERFFT_XAXIS_TAG = "dopplerfft_xaxis"
    DOPPLERFFT_YAXIS_TAG = "dopplerfft_yaxis"
    DOPPLERFFT_PROFILE_XAXIS_TAG = "dopplerfft_profile_xaxis"
    DOPPLERFFT_PROFILE_YAXIS_TAG = "dopplerfft_profile_yaxis"
    DOPPLERFFT_IMG_SERIES_TAG = "dopplerfft_img_series"
    DOPPLERFFT_PROFILE_LINE_TAG = "dopplerfft_profile_line"
    DOPPLERFFT_CMAP_SCALE_TAG = "dopplerfft_cmap_scale"
    CHK_ANGLE_SINGLE_BIN = "chk_angle_single_bin"
    CHK_DOPPLER_SINGLE_BIN = "chk_doppler_single_bin"
    CHK_ANGLE_NORM = "chk_angle_norm"
    CHK_DOPPLER_NORM = "chk_doppler_norm"
    IN_ANGLE_BIN = "in_angle_bin"
    IN_DOPPLER_BIN = "in_doppler_bin"
    IN_ANGLE_VMIN = "in_angle_vmin"
    IN_ANGLE_VMAX = "in_angle_vmax"
    IN_DOPPLER_VMIN = "in_doppler_vmin"
    IN_DOPPLER_VMAX = "in_doppler_vmax"
    RANGEFFT_LINE_COLORS = [
        (0, 255, 255, 255),    # Ciano
        (255, 255, 0, 255),    # Giallo
        (57, 255, 20, 255),    # Verde neon
        (255, 140, 0, 255),    # Arancione
        (255, 0, 255, 255),    # Magenta
        (255, 255, 255, 255),  # Bianco
        (255, 0, 0, 255),      # Rosso
        (135, 206, 250, 255),  # Azzurro chiaro
    ]
    rangefft_line_themes = []
    for line_color in RANGEFFT_LINE_COLORS:
        with dpg.theme() as line_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, line_color, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2.0, category=dpg.mvThemeCat_Plots)
        rangefft_line_themes.append(line_theme)
    with dpg.theme() as track_conf_theme:
        with dpg.theme_component(dpg.mvScatterSeries):
            dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, (255, 255, 255, 0), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, (255, 255, 255, 220), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 8.0, category=dpg.mvThemeCat_Plots)
    with dpg.theme() as track_unconf_theme:
        with dpg.theme_component(dpg.mvScatterSeries):
            dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, (255, 255, 255, 0), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, (255, 255, 255, 130), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Diamond, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 7.0, category=dpg.mvThemeCat_Plots)
    with dpg.theme() as track_vel_theme:
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 255, 255, 170), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 1.0, category=dpg.mvThemeCat_Plots)
    with dpg.theme() as guide_line_theme:
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 230, 80, 190), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 1.5, category=dpg.mvThemeCat_Plots)
    with dpg.theme() as track_moving_theme:
        with dpg.theme_component(dpg.mvScatterSeries):
            dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, (0, 235, 180, 255), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, (0, 235, 180, 255), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 7.0, category=dpg.mvThemeCat_Plots)
    with dpg.theme() as track_stopped_theme:
        with dpg.theme_component(dpg.mvScatterSeries):
            dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, (255, 175, 0, 255), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, (255, 175, 0, 255), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Square, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 7.0, category=dpg.mvThemeCat_Plots)
    with dpg.theme() as track_unknown_theme:
        with dpg.theme_component(dpg.mvScatterSeries):
            dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, (155, 155, 155, 255), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, (155, 155, 155, 255), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 6.0, category=dpg.mvThemeCat_Plots)
    with dpg.theme() as track_stop_marker_theme:
        with dpg.theme_component(dpg.mvScatterSeries):
            dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, (255, 230, 80, 255), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, (255, 230, 80, 255), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Cross, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 8.0, category=dpg.mvThemeCat_Plots)

    # 4) Texture setup (1:1 con il buffer GUI, senza resampling)
    tex_w, tex_h = int(gui_w), int(gui_h)
    tex_buf = array("f", [0.0]) * (tex_w * tex_h * 4)
    tex_np = np.frombuffer(tex_buf, dtype=np.float32)
    angle_diag_tex_w, angle_diag_tex_h = int(ANGLEFFT_BINS), int(fft_plot_h)
    angle_diag_tex_buf = array("f", [0.0]) * (angle_diag_tex_w * angle_diag_tex_h * 4)
    angle_diag_tex_np = np.frombuffer(angle_diag_tex_buf, dtype=np.float32)
    doppler_diag_tex_w, doppler_diag_tex_h = int(DOPPLERFFT_BINS), int(fft_plot_h)
    doppler_diag_tex_buf = array("f", [0.0]) * (doppler_diag_tex_w * doppler_diag_tex_h * 4)
    doppler_diag_tex_np = np.frombuffer(doppler_diag_tex_buf, dtype=np.float32)
    proc_tex_w, proc_tex_h = int(offline_gui_w), int(offline_gui_h)
    proc_tex_buf = array("f", [0.0]) * (proc_tex_w * proc_tex_h * 4)
    proc_tex_np = np.frombuffer(proc_tex_buf, dtype=np.float32)

    off_pos_min = int(offline_info.get("pos_min", 0)) if offline_info else 0
    off_pos_max = int(offline_info.get("pos_max", 0)) if offline_info else 0
    if off_pos_max < off_pos_min:
        off_pos_min, off_pos_max = off_pos_max, off_pos_min
    off_x_start = int(offline_info.get("x_start", off_pos_min)) if offline_info else off_pos_min
    off_x_end = int(offline_info.get("x_end", off_pos_max)) if offline_info else off_pos_max
    if off_x_end < off_x_start:
        off_x_start, off_x_end = off_x_end, off_x_start

    def _estimate_offline_memory(cfg_source: dict | None = None) -> dict:
        """Estimate offline allocations using file sizes only; never read capture payloads."""
        source = cfg_source if isinstance(cfg_source, dict) else {}
        fft_cfg = source.get("offline_sar_range_angle", {}) or {}
        reader_cfg = OfflineSARConfig.from_mapping(
            source,
            base_dir=offline_config_boot_path.parent,
        )
        layout = SARReader(config=reader_cfg).describe_stream()
        source_dir = layout.source_dir
        selected_files = list(layout.files)
        capture_bytes = sum(int(path.stat().st_size) for path in selected_files)
        samples_input = max(1, int(layout.samples))
        chirps_input = max(1, int(layout.chirps))
        rx_input = max(1, int(layout.rx))
        tx_input = max(1, int(layout.tx))
        frames_input = max(1, int(layout.n_frames_per_position))
        # The reader loads only the first configured frames of each file.
        # Keep the estimate aligned with that bounded read rather than with
        # the complete capture file stored on disk.
        max_position_bytes = int(frames_input) * int(layout.bytes_per_frame)
        nfft_range_est = max(1, int(fft_cfg.get("nfft_range", NFFT_RANGE)))
        # Streaming pipeline: only one capture payload and its complex64
        # conversion are resident while the compact zero-Doppler snapshot cube
        # for every configured position is filled directly in shared memory.
        iq_bytes = int(max_position_bytes) * 2
        samples_used = min(int(samples_input), int(nfft_range_est))
        range_window = str(fft_cfg.get("window_range", "none")).strip().lower()
        preprocessing = bool(fft_cfg.get("use_realtime_filters", True))
        window_copy_bytes = (
            int(iq_bytes * samples_used / samples_input)
            if preprocessing and range_window not in {"none", "rectangular"}
            else 0
        )
        fft_bytes = int(iq_bytes * nfft_range_est / samples_input)
        prepared_bytes = int(
            len(selected_files)
            * int(frames_input)
            * int(tx_input)
            * int(rx_input)
            * int(nfft_range_est)
            * np.dtype(np.complex64).itemsize
        )
        reader_peak_bytes = int(
            prepared_bytes
            + max_position_bytes
            + iq_bytes
            + window_copy_bytes
            + fft_bytes
        )
        display_pixels = int(offline_gui_h) * int(offline_gui_w)
        bp_workspace_bytes = int(
            int(frames_input) * display_pixels * np.dtype(np.complex64).itemsize
            + 8 * display_pixels * np.dtype(np.float32).itemsize
        )
        dsp_peak_bytes = int((2 * prepared_bytes) + bp_workspace_bytes)
        peak_bytes = max(reader_peak_bytes, dsp_peak_bytes)
        return {
            "source_dir": str(source_dir),
            "files": int(len(selected_files)),
            "samples_input": int(samples_input),
            "nfft_range": int(nfft_range_est),
            "capture_bytes": int(capture_bytes),
            "iq_bytes": int(iq_bytes),
            "fft_bytes": int(fft_bytes),
            "prepared_bytes": int(prepared_bytes),
            "peak_bytes": int(peak_bytes),
        }

    def _format_offline_memory_estimate(cfg_source: dict | None = None) -> str:
        try:
            estimate = _estimate_offline_memory(cfg_source)
            gib = float(1024 ** 3)
            padding_note = (
                "zero-padding"
                if int(estimate["nfft_range"]) > int(estimate["samples_input"])
                else "no range zero-padding"
            )
            return (
                f"Memory estimate (payload not loaded): {estimate['files']} files | "
                f"Range {estimate['samples_input']} -> {estimate['nfft_range']} ({padding_note})\n"
                f"Per-position IQ {estimate['iq_bytes'] / gib:.2f} GiB | "
                f"Range FFT workspace {estimate['fft_bytes'] / gib:.2f} GiB | "
                f"Doppler-zero shared {estimate['prepared_bytes'] / gib:.2f} GiB | "
                f"streaming peak about {estimate['peak_bytes'] / gib:.2f} GiB"
            )
        except Exception as exc:
            return f"Memory estimate unavailable: {exc}"

    offline_memory_estimate_text = _format_offline_memory_estimate(offline_boot_cfg)
    offline_frame_valid = False
    # A runtime is "ready" after its workers have started, but its first
    # reconstruction is published later.  Keep these states separate so the
    # GUI never announces completion before its texture contains that result.
    offline_recalculation_completion_pending = None
    off_norm_enabled = True
    if off_norm_enabled:
        off_vmin = float(VMIN_NORM)
        off_vmax = float(VMAX_NORM)
    else:
        off_vmin = float(VMIN_RAW)
        off_vmax = float(VMAX_RAW)
    if off_vmax <= off_vmin:
        off_vmax = off_vmin + 1.0
    if offline_map_bounds_boot is None:
        offline_map_bounds_boot = offline_map_bounds_from_yaml_dict({}, cfg)
    off_map_bounds_current = offline_map_bounds_boot
    rt_home_viewport_current = rt_home_viewport
    rt_requested_viewport_current = rt_home_viewport
    rt_applied_meta_current = applied_viewport_meta_from_viewport(
        rt_home_viewport,
        fallback_used=False,
        frame_seq=0,
    )
    off_home_viewport_current = build_display_viewport(
        x_min_m=float(off_map_bounds_current.x_min_m),
        x_max_m=float(off_map_bounds_current.x_max_m),
        y_min_m=float(off_map_bounds_current.y_min_m),
        y_max_m=float(off_map_bounds_current.y_max_m),
        dr_m=float(dr_plot),
        seq=0,
    )
    off_requested_viewport_current = off_home_viewport_current
    off_applied_meta_current = applied_viewport_meta_from_viewport(
        off_home_viewport_current,
        fallback_used=False,
        frame_seq=0,
    )
    rt_applied_seq_local = 0

    guide_angle_deg = 20.0
    guide_slope = float(np.tan(np.deg2rad(np.float32(guide_angle_deg))))

    def _guide_line_points(angle_sign: float, y_max_m: float) -> list[list[float]]:
        y_top = max(0.0, float(y_max_m))
        x_top = float(angle_sign) * guide_slope * y_top
        return [[0.0, x_top], [0.0, y_top]]

    def _configure_image_bounds(series_tag: str, meta: AppliedViewportMeta | DisplayViewport) -> None:
        if not dpg.does_item_exist(series_tag):
            return
        dpg.configure_item(
            series_tag,
            bounds_min=(float(meta.x_min_m), float(meta.y_min_m)),
            bounds_max=(float(meta.x_max_m), float(meta.y_max_m)),
        )

    def _write_realtime_requested_viewport(viewport: DisplayViewport) -> None:
        with rt_requested_viewport_lock:
            rt_requested_viewport[0] = float(viewport.x_min_m)
            rt_requested_viewport[1] = float(viewport.x_max_m)
            rt_requested_viewport[2] = float(viewport.y_min_m)
            rt_requested_viewport[3] = float(viewport.y_max_m)
        with rt_requested_viewport_seq.get_lock():
            rt_requested_viewport_seq.value = int(viewport.seq)

    def _write_realtime_home_viewport(viewport: DisplayViewport) -> None:
        with rt_home_viewport_lock:
            rt_home_viewport_shared[0] = float(viewport.x_min_m)
            rt_home_viewport_shared[1] = float(viewport.x_max_m)
            rt_home_viewport_shared[2] = float(viewport.y_min_m)
            rt_home_viewport_shared[3] = float(viewport.y_max_m)

    def _read_realtime_applied_meta() -> AppliedViewportMeta:
        with rt_applied_viewport_lock:
            arr = np.asarray(rt_applied_viewport[:], dtype=np.float64)
        try:
            seq_val = int(rt_applied_viewport_seq.value)
        except Exception:
            seq_val = 0
        try:
            frame_seq_val = int(rt_applied_viewport_frame_seq.value)
        except Exception:
            frame_seq_val = 0
        try:
            fallback_used = bool(rt_applied_viewport_fallback.value)
        except Exception:
            fallback_used = False
        return AppliedViewportMeta(
            x_min_m=float(arr[0]),
            x_max_m=float(arr[1]),
            y_min_m=float(arr[2]),
            y_max_m=float(arr[3]),
            range_min_bin_f=float(arr[4]),
            range_max_bin_f=float(arr[5]),
            angle_min_deg=float(arr[6]),
            angle_max_deg=float(arr[7]),
            zoom_level=float(arr[8]),
            seq=int(seq_val),
            fallback_used=bool(fallback_used),
            frame_seq=int(frame_seq_val),
        )

    def _try_get_axis_limits(x_axis_tag: str, y_axis_tag: str) -> tuple[float, float, float, float] | None:
        if not dpg.does_item_exist(x_axis_tag) or not dpg.does_item_exist(y_axis_tag):
            return None
        try:
            x_limits = dpg.get_axis_limits(x_axis_tag)
            y_limits = dpg.get_axis_limits(y_axis_tag)
        except Exception:
            return None
        if x_limits is None or y_limits is None:
            return None
        try:
            x0, x1 = float(x_limits[0]), float(x_limits[1])
            y0, y1 = float(y_limits[0]), float(y_limits[1])
        except Exception:
            return None
        if not (np.isfinite(x0) and np.isfinite(x1) and np.isfinite(y0) and np.isfinite(y1)):
            return None
        return (x0, x1, y0, y1)

    def _poll_requested_viewport(
        *,
        x_axis_tag: str,
        y_axis_tag: str,
        home_viewport: DisplayViewport,
        current_viewport: DisplayViewport,
    ) -> DisplayViewport:
        limits = _try_get_axis_limits(x_axis_tag, y_axis_tag)
        if limits is None:
            return current_viewport
        x0, x1, y0, y1 = limits
        return clamp_display_viewport(
            x_min_m=float(x0),
            x_max_m=float(x1),
            y_min_m=float(y0),
            y_max_m=float(y1),
            home_viewport=home_viewport,
            output_width=int(gui_w),
            output_height=int(gui_h),
            dr_m=float(dr_plot),
            seq=int(current_viewport.seq) + 1,
        )

    off_ui_dirty = False
    off_ui_dirty_t = 0.0
    # DearPyGui may defer GPU creation for textures contained in a hidden tab.
    # Re-upload the cached offline texture whenever the main tab changes.
    off_texture_upload_requested = True
    off_ui_pending = {
        "x_start": int(off_x_start),
        "x_end": int(off_x_end),
        "vmin": float(off_vmin),
        "vmax": float(off_vmax),
        "norm_enabled": bool(off_norm_enabled),
        "reset_view": True,
        "reconstruct": True,
        "display_refresh": True,
    }
    offline_summary_state = f"ERROR: {offline_error}" if offline_error else ""
    offline_last_calculation_ms: float | None = None
    # Populated by "Inspect run".  The bound is the minimum physical frame
    # count across the selected capture files, so one value is safe for every
    # position used by the reconstruction.
    offline_frame_limit: dict[str, object] = {"available": None, "input_dir": ""}

    def _render_offline_summary(
        *,
        state: str | None = None,
        calculation_ms: float | None = None,
    ) -> None:
        """Render the one yellow offline summary without duplicate status text."""
        nonlocal offline_summary_state, offline_last_calculation_ms
        if state is not None:
            offline_summary_state = str(state).strip()
        if calculation_ms is not None:
            elapsed = float(calculation_ms)
            offline_last_calculation_ms = elapsed if np.isfinite(elapsed) and elapsed >= 0.0 else None

        lines: list[str] = []
        if offline_summary_state:
            lines.append(offline_summary_state)
        if offline_last_calculation_ms is not None:
            lines.append(f"Last reconstruction: {offline_last_calculation_ms:.1f} ms")
        lines.append(offline_memory_estimate_text)
        if dpg.does_item_exist(PROC_TXT_MEMORY_ESTIMATE):
            dpg.set_value(PROC_TXT_MEMORY_ESTIMATE, "\n".join(lines))

    def _on_main_tab_changed(sender=None, app_data=None):
        nonlocal off_texture_upload_requested
        off_texture_upload_requested = True

    with dpg.texture_registry(show=False):
        dpg.add_dynamic_texture(
            width=tex_w,
            height=tex_h,
            default_value=[0.0, 0.0, 0.0, 1.0] * tex_w * tex_h,
            tag=TEX_TAG,
        )
        dpg.add_dynamic_texture(
            width=angle_diag_tex_w,
            height=angle_diag_tex_h,
            default_value=[0.0, 0.0, 0.0, 1.0] * angle_diag_tex_w * angle_diag_tex_h,
            tag=ANGLEFFT_TEX_TAG,
        )
        dpg.add_dynamic_texture(
            width=doppler_diag_tex_w,
            height=doppler_diag_tex_h,
            default_value=[0.0, 0.0, 0.0, 1.0] * doppler_diag_tex_w * doppler_diag_tex_h,
            tag=DOPPLERFFT_TEX_TAG,
        )
        dpg.add_dynamic_texture(
            width=proc_tex_w,
            height=proc_tex_h,
            default_value=[0.0, 0.0, 0.0, 1.0] * proc_tex_w * proc_tex_h,
            tag=PROC_TEX_TAG,
        )

    # 5) Callbacks
    dpg.set_exit_callback(_shutdown)

    def _build_jet_lut(size: int = 2048):
        x = np.linspace(0.0, 1.0, int(size), dtype=np.float32)
        r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
        a = np.ones_like(x, dtype=np.float32)
        return np.stack((r, g, b, a), axis=-1).astype(np.float32, copy=False)

    def _build_velocity_lut(size: int = 2048):
        x = np.linspace(0.0, 1.0, int(size), dtype=np.float32)
        signed = (x - np.float32(0.5)) * np.float32(2.0)
        dead_zone_value = float(HEATMAP_VELOCITY_DEAD_ZONE)
        if not np.isfinite(dead_zone_value):
            dead_zone_value = 0.08
        dead_zone = np.float32(min(0.99, max(0.0, dead_zone_value)))
        mag = (np.abs(signed) - dead_zone) / (np.float32(1.0) - dead_zone)
        np.clip(mag, 0.0, 1.0, out=mag)
        mag = np.power(mag, np.float32(0.70), dtype=np.float32)

        center = np.asarray([0.015, 0.015, 0.018], dtype=np.float32)
        neg_mid = np.asarray([52.0, 92.0, 255.0], dtype=np.float32) / np.float32(255.0)
        neg_hi = np.asarray([0.0, 235.0, 255.0], dtype=np.float32) / np.float32(255.0)
        pos_mid = np.asarray([80.0, 230.0, 95.0], dtype=np.float32) / np.float32(255.0)
        pos_hi = np.asarray([255.0, 190.0, 0.0], dtype=np.float32) / np.float32(255.0)

        mid_mix = np.clip(mag / np.float32(0.55), 0.0, 1.0)
        hi_mix = np.clip((mag - np.float32(0.55)) / np.float32(0.45), 0.0, 1.0)
        neg_rgb = center[None, :] + (neg_mid[None, :] - center[None, :]) * mid_mix[:, None]
        neg_rgb = neg_rgb + (neg_hi[None, :] - neg_rgb) * hi_mix[:, None]
        pos_rgb = center[None, :] + (pos_mid[None, :] - center[None, :]) * mid_mix[:, None]
        pos_rgb = pos_rgb + (pos_hi[None, :] - pos_rgb) * hi_mix[:, None]
        rgb = np.where(signed[:, None] < np.float32(0.0), neg_rgb, pos_rgb)
        rgb[mag <= np.float32(0.0), :] = center
        a = np.ones((int(size), 1), dtype=np.float32)
        return np.concatenate((rgb, a), axis=1).astype(np.float32, copy=False)

    try:
        velocity_cmap_preview = _build_velocity_lut(256)
        velocity_cmap_colors = [
            tuple(int(round(float(c) * 255.0)) for c in rgba)
            for rgba in velocity_cmap_preview
        ]
        with dpg.colormap_registry():
            dpg.add_colormap(
                velocity_cmap_colors,
                False,
                label="Velocity Blue-Black-Green",
                tag=CMAP_VELOCITY_TAG,
            )
    except Exception:
        pass

    def _fft_mode_label(mode_db: bool) -> str:
        return "FFT SCALE: dB" if mode_db else "FFT SCALE: LINEAR"

    def _fft_view_label(view_full: bool) -> str:
        return "FFT VIEW: FULL" if view_full else "FFT VIEW: HALF"

    def _fft_plot_title(view_full: bool) -> str:
        if view_full:
            return f"Range FFT | diagnostic full | TX 1-{RANGEFFT_PLOT_COUNT} (RX 1-{RANGEFFT_LINES_PER_PLOT})"
        return f"Range FFT | TX 1-{RANGEFFT_PLOT_COUNT} (RX 1-{RANGEFFT_LINES_PER_PLOT})"

    def _fft_axis_label(mode_db: bool) -> str:
        return "dB" if mode_db else "Linear"

    def _fft_visible_range_from_view(rmax_m: float, view_full: bool):
        if view_full:
            max_r_bin = int(fft_plot_h)
            max_r_m = float(max_r_bin) * float(dr_plot)
        else:
            max_r_m = max(0.0, min(float(rmax_m), float(fft_plot_h) * float(dr_plot)))
            max_r_bin = int(np.ceil(float(max_r_m) / float(dr_plot))) if dr_plot > 1e-12 else int(fft_plot_h)
            max_r_bin = max(1, min(max_r_bin, int(fft_plot_h)))
        return max_r_bin, max_r_m

    def _set_fft_x_ticks_window(xmin_m: float, xmax_m: float):
        if not dpg.does_item_exist(RANGEFFT_XAXIS_TAG):
            return
        tick_start = max(0, int(np.ceil(float(xmin_m) - 1e-6)))
        tick_end = max(tick_start, int(np.floor(float(xmax_m) + 1e-6)))
        ticks = tuple((str(x), float(x)) for x in range(tick_start, tick_end + 1, 1))
        dpg.set_axis_ticks(RANGEFFT_XAXIS_TAG, ticks)

    def _clamp_fft_x_window(xmin_m: float, xmax_m: float, rmax_m: float, view_full: bool):
        _, view_max_m = _fft_visible_range_from_view(rmax_m, view_full)
        x0 = max(0.0, min(float(xmin_m), float(view_max_m)))
        x1 = max(0.0, min(float(xmax_m), float(view_max_m)))
        eps = max(float(dr_plot), 1e-3)
        if x1 <= x0:
            x1 = min(float(view_max_m), x0 + eps)
        if x1 <= x0:
            x0 = max(0.0, float(view_max_m) - eps)
            x1 = float(view_max_m)
        return float(x0), float(x1), float(view_max_m)

    def _auto_expand_fft_x_window_for_rmax_growth(
        *,
        prev_rmax_m: float,
        new_rmax_m: float,
        fft_xmax_m: float,
        view_full: bool,
    ) -> float:
        if view_full or float(new_rmax_m) <= float(prev_rmax_m):
            return float(fft_xmax_m)
        _, prev_view_max_m = _fft_visible_range_from_view(prev_rmax_m, view_full)
        _, new_view_max_m = _fft_visible_range_from_view(new_rmax_m, view_full)
        eps = max(float(dr_plot), 1e-3)
        if abs(float(fft_xmax_m) - float(prev_view_max_m)) <= eps:
            return float(new_view_max_m)
        return float(fft_xmax_m)

    def _base_fft_scale_for_mode(mode_db: bool):
        if mode_db:
            y_min = float(RANGEFFT_DB_MIN)
            y_max = float(RANGEFFT_DB_MAX)
        else:
            y_min = float(RANGEFFT_LIN_MIN)
            y_max = float(RANGEFFT_LIN_MAX)
            y_min = float(np.floor(y_min))
            y_max = float(np.ceil(y_max))
        eps = 1.0
        if y_max <= y_min:
            y_max = y_min + eps
        return y_min, y_max

    def _update_fft_scale_input_labels(mode_db: bool):
        unit = "dB" if mode_db else "lin"
        if dpg.does_item_exist(IN_FFT_VMIN):
            dpg.configure_item(IN_FFT_VMIN, label=f"FFT Ymin ({unit})")
        if dpg.does_item_exist(IN_FFT_VMAX):
            dpg.configure_item(IN_FFT_VMAX, label=f"FFT Ymax ({unit})")

    def _apply_params(sender=None, app_data=None):
        nonlocal ui_dirty, ui_pending, fft_mode_db, fft_view_full
        # callback ultraleggera: salva valori e setta flag
        # Nota:
        #  - vmin/vmax sono "base values" (non hard limit). Li lasciamo liberi.
        #  - rmax/xmax sono HARD LIMIT: 0 .. (range_max/crossrange_max da config)
        try:
            vmin = float(dpg.get_value(IN_VMIN))
            vmax = float(dpg.get_value(IN_VMAX))
            rmax = float(dpg.get_value(IN_RMAX))
            xmax = float(dpg.get_value(IN_XMAX))
            fft_xmin = float(dpg.get_value(IN_FFT_XMIN))
            fft_xmax = float(dpg.get_value(IN_FFT_XMAX))
            fft_vmin = float(dpg.get_value(IN_FFT_VMIN))
            fft_vmax = float(dpg.get_value(IN_FFT_VMAX))
        except (ValueError, TypeError):
            return  # mentre scrivi '-', '.', '' ecc.

        heatmap_mode_now = _get_heatmap_mode()
        vmin, vmax = _sanitize_heatmap_scale_inputs(heatmap_mode_now, vmin, vmax, sender=sender)
        if dpg.does_item_exist(IN_VMIN) and float(dpg.get_value(IN_VMIN)) != float(vmin):
            dpg.set_value(IN_VMIN, vmin)
        if dpg.does_item_exist(IN_VMAX) and float(dpg.get_value(IN_VMAX)) != float(vmax):
            dpg.set_value(IN_VMAX, vmax)

        # HARD clamp su Rmax/Xmax
        rmax_cl = max(0.0, min(rmax, RMAX_HARD_MAX))
        xmax_cl = max(0.0, min(xmax, XMAX_HARD_MAX))

        if rmax_cl != rmax:
            dpg.set_value(IN_RMAX, rmax_cl)
        if xmax_cl != xmax:
            dpg.set_value(IN_XMAX, xmax_cl)
        fft_eps = 1.0
        if fft_vmax <= fft_vmin:
            fft_vmax = fft_vmin + fft_eps
            if dpg.does_item_exist(IN_FFT_VMAX):
                dpg.set_value(IN_FFT_VMAX, fft_vmax)

        prev_rmax = float(ui_pending.get("rmax", rmax_cl))
        prev_xmax = float(ui_pending.get("xmax", xmax_cl))
        fft_xmax = _auto_expand_fft_x_window_for_rmax_growth(
            prev_rmax_m=prev_rmax,
            new_rmax_m=rmax_cl,
            fft_xmax_m=fft_xmax,
            view_full=fft_view_full,
        )
        fft_xmin_cl, fft_xmax_cl, _ = _clamp_fft_x_window(fft_xmin, fft_xmax, rmax_cl, fft_view_full)
        if fft_xmin_cl != fft_xmin and dpg.does_item_exist(IN_FFT_XMIN):
            dpg.set_value(IN_FFT_XMIN, fft_xmin_cl)
        if fft_xmax_cl != fft_xmax and dpg.does_item_exist(IN_FFT_XMAX):
            dpg.set_value(IN_FFT_XMAX, fft_xmax_cl)
        ui_pending["vmin"] = vmin
        ui_pending["vmax"] = vmax
        ui_pending["rmax"] = rmax_cl
        ui_pending["xmax"] = xmax_cl
        ui_pending["fft_xmin"] = float(fft_xmin_cl)
        ui_pending["fft_xmax"] = float(fft_xmax_cl)
        ui_pending["fft_vmin"] = fft_vmin
        ui_pending["fft_vmax"] = fft_vmax
        ui_pending["fft_mode_db"] = bool(fft_mode_db)
        ui_pending["fft_view_full"] = bool(fft_view_full)
        ui_pending["heatmap_mode"] = int(heatmap_mode_now)
        ui_pending["reset_view"] = bool(
            ui_pending.get("reset_view", False)
            or abs(float(prev_rmax) - float(rmax_cl)) > 1e-6
            or abs(float(prev_xmax) - float(xmax_cl)) > 1e-6
        )

        ui_dirty = True


    # La configurazione offline della run viene confermata soltanto dopo il
    # completamento di tutti i file. Su cancel/failure rimane quindi puntata
    # all'ultima acquisizione completa.
    scan_pending_run: dict[str, int | None] = {
        "start_position_id": None,
        "positions": None,
    }

    def _set_scan_status(text: str) -> None:
        if dpg.does_item_exist(TXT_SCAN_STATUS_TAG):
            dpg.set_value(TXT_SCAN_STATUS_TAG, str(text))

    def _any_scan_active() -> bool:
        return bool(sar_scan is not None and sar_scan.active)

    def _active_or_pending_scan():
        return sar_scan

    def _update_scan_pitch_label(pitch_mm: float | None = None, error: str = "") -> None:
        if not dpg.does_item_exist(TXT_SCAN_PITCH_TAG):
            return
        if pitch_mm is not None and np.isfinite(float(pitch_mm)) and float(pitch_mm) > 0.0:
            label = f"Pitch from offline_config: {float(pitch_mm):.6f} mm"
        else:
            label = f"offline_config error: {error or 'invalid pitch'}"
        dpg.set_value(TXT_SCAN_PITCH_TAG, label)

    def _motor_metadata_for_capture() -> tuple[float | None, int | None]:
        if stepper_controller is None:
            return None, None
        try:
            snapshot = stepper_controller.snapshot()
            if not bool(snapshot.get("connected")) or not bool(snapshot.get("homed")):
                return None, None
            if bool(snapshot.get("motion_active")):
                raise CaptureError("Wait for the carriage to stop before capturing.")
            steps = int(stepper_controller.position_microsteps())
            return float(stepper_controller.config.mechanics.mm_from_microsteps(steps)), steps
        except CaptureError:
            raise
        except Exception:
            return None, None

    def _on_capture():
        if _any_scan_active():
            _set_scan_status("Manual capture is disabled during a SAR scan.")
            return
        try:
            position_mm, position_steps = _motor_metadata_for_capture()
            with sar_pos_counter.get_lock():
                next_pid = int(sar_pos_counter.value) + 1
            capture_sessions.request(next_pid, position_mm, position_steps)
            with sar_pos_counter.get_lock():
                sar_pos_counter.value = int(next_pid)
            dpg.set_value(TXT_POS_TAG, f"Last position: {next_pid} | next ID: {next_pid + 1}")
            _set_scan_status(f"Manual capture for position {next_pid} is running")
        except Exception as exc:
            _set_scan_status(f"Capture did not start: {exc}")

    def _on_new_sar_session() -> None:
        """Passa a una cartella di acquisizione vuota senza riavviare la GUI."""
        nonlocal out_dir
        if capture_sessions.inflight:
            _set_scan_status("Wait for the current capture to complete or be cancelled.")
            return
        if _any_scan_active():
            _set_scan_status("A session cannot be changed during a SAR scan.")
            return
        if scan_pending_run["start_position_id"] is not None:
            _set_scan_status("Wait for the previous scan offline finalization.")
            return
        try:
            new_dir = _create_output_run()
            _write_shared_text(output_dir_shared, str(new_dir))
            out_dir = new_dir
            with sar_pos_counter.get_lock():
                sar_pos_counter.value = 0
            if dpg.does_item_exist(TXT_POS_TAG):
                dpg.set_value(TXT_POS_TAG, "Next ID: 1")
            if dpg.does_item_exist(TXT_RUN_DIR_TAG):
                dpg.set_value(TXT_RUN_DIR_TAG, f"Current run: {out_dir.name}")
            _set_scan_status(f"New session ready: {out_dir.name}")
        except Exception as exc:
            _set_scan_status(f"New session was not created: {exc}")

    def _on_motor_connect() -> None:
        if stepper_controller is None:
            _set_scan_status(f"Phidget unavailable: {stepper_error or 'backend not loaded'}")
            return
        threading.Thread(target=stepper_controller.connect, name="phidget-connect-main", daemon=True).start()

    def _on_motor_disconnect() -> None:
        if stepper_controller is None:
            return
        if _any_scan_active():
            _set_scan_status("Cancel the SAR scan before disconnecting the carriage.")
            return
        if capture_sessions.inflight:
            _set_scan_status("Wait for the current capture to complete or be cancelled.")
            return
        threading.Thread(
            target=stepper_controller.disconnect,
            name="phidget-disconnect-main",
            daemon=True,
        ).start()

    def _on_motor_home() -> None:
        if stepper_controller is None:
            return
        try:
            stepper_controller.start_homing()
        except Exception as exc:
            stepper_controller.log(f"HOME did not start: {exc}")

    def _on_motor_jog(sign: int) -> None:
        if stepper_controller is None:
            return
        try:
            distance = abs(float(dpg.get_value(IN_MOTOR_JOG_TAG))) * int(sign)
            stepper_controller.move_relative_mm(distance)
        except Exception as exc:
            stepper_controller.log(f"Jog did not start: {exc}")

    def _on_motor_stop() -> None:
        active_scan = _active_or_pending_scan()
        if active_scan is not None and active_scan.active:
            active_scan.cancel()
            return
        if capture_sessions.inflight:
            capture_sessions.cancel()
            _set_scan_status("Capture cancellation requested...")
            return
        if stepper_controller is not None:
            try:
                stepper_controller.stop()
            except Exception as exc:
                stepper_controller.log(f"STOP failed: {exc}")

    def _on_start_sar_scan() -> None:
        if sar_scan is None or stepper_controller is None:
            _set_scan_status(f"Scan unavailable: {stepper_error or 'Phidget not configured'}")
            return
        if capture_sessions.inflight:
            _set_scan_status("Wait for the current capture to complete or be cancelled.")
            return
        if _any_scan_active():
            _set_scan_status("A SAR scan is already active.")
            return
        if scan_pending_run["start_position_id"] is not None:
            _set_scan_status("Wait for the previous scan offline finalization.")
            return
        try:
            motor_snapshot = stepper_controller.snapshot()
            if not bool(motor_snapshot.get("connected")):
                raise ScanError("Connect the Phidget before starting a scan.")
            if not bool(motor_snapshot.get("homed")):
                raise ScanError("Run HOME and manually set the carriage start position before scanning.")
            if bool(motor_snapshot.get("motion_active")):
                raise ScanError("Wait for manual motion to stop before scanning.")
            radar_state = mmwave_bridge.get_gui_state()
            if not radar_state.connected:
                raise ScanError("Connect the radar and DCA1000 first with mmWave: Connect.")
            if not radar_state.streaming:
                raise ScanError("Start the radar first with Radar: Start.")
            with rx_pkts.get_lock():
                received_packets = int(rx_pkts.value)
            with rx_last_packet_time_s.get_lock():
                udp_idle_s = max(0.0, time.time() - float(rx_last_packet_time_s.value))
            if received_packets <= 0 or udp_idle_s > 2.0:
                raise ScanError("Streaming is enabled but UDP is inactive: wait for radar data before scanning.")
            n_positions = int(dpg.get_value(IN_SCAN_POSITIONS_TAG))
            start_id, _default_positions, pitch_mm = read_offline_scan_settings(offline_scan_config_path)
            _update_scan_pitch_label(pitch_mm)
            if n_positions <= 0:
                raise ScanError("The number of positions must be greater than zero.")
            existing = [
                out_dir / f"capture_pos{position_id}.bin"
                for position_id in range(start_id, start_id + n_positions)
                if (out_dir / f"capture_pos{position_id}.bin").exists()
            ]
            if existing:
                raise ScanError(
                    "The current run already contains captures (start a new session to avoid overwriting them)."
                )
            with sar_pos_counter.get_lock():
                sar_pos_counter.value = int(start_id) - 1
            plan = ScanPlan(
                positions=n_positions,
                pitch_mm=float(pitch_mm),
                start_position_id=int(start_id),
                # radar_rx applica già SETTLING_DELAY_S dopo che il carrello è
                # fermo; non duplicare inutilmente l'attesa qui.
                settling_seconds=0.0,
                motion_timeout_seconds=max(1.0, float(sar_cfg.get("scan_motion_timeout_s", 120.0))),
                capture_timeout_seconds=max(1.0, float(sar_cfg.get("scan_capture_timeout_s", 120.0))),
            )
            sar_scan.start(plan)
            scan_pending_run.update(
                {
                    "start_position_id": int(start_id),
                    "positions": int(n_positions),
                }
            )
            _set_scan_status(
                f"Linear scan started: {n_positions} positions, pitch {pitch_mm:.6f} mm"
            )
        except Exception as exc:
            _set_scan_status(f"Scan did not start: {exc}")

    def _on_cancel_sar_scan() -> None:
        if sar_scan is None or not sar_scan.active:
            _set_scan_status("No SAR scan is active.")
            return
        sar_scan.cancel()
        _set_scan_status("Scan cancellation requested...")

    motor_log_lines: list[str] = []

    def _scan_event_label(event: ScanEvent) -> str:
        """Renderizza l'evento SAR strutturato nella lingua della GUI."""
        labels = {
            ScanEvent.READY: "Ready",
            ScanEvent.PREPARING: "Preparing SAR scan",
            ScanEvent.CAPTURING_POSITION: "Capturing position",
            ScanEvent.MOVING_TO_POSITION: "Moving to position",
            ScanEvent.MECHANICAL_SETTLING: "Mechanical settling",
            ScanEvent.CANCELLATION_REQUESTED: "Cancelling scan...",
            ScanEvent.CANCELLED: "SAR scan cancelled",
            ScanEvent.COMPLETED: "SAR scan completed",
            ScanEvent.INTERRUPTED: "SAR scan interrupted",
            ScanEvent.FINALIZATION_FAILED: "Scan finalization error",
        }
        return labels.get(event, "SAR scan status unavailable")

    def _drain_motor_events() -> None:
        if stepper_controller is None:
            return
        try:
            while True:
                event, value = stepper_controller.gui_queue.get_nowait()
                if event == "log":
                    motor_log_lines.append(str(value))
                elif event == "position":
                    try:
                        mm = stepper_controller.config.mechanics.mm_from_microsteps(float(value))
                        if dpg.does_item_exist(TXT_MOTOR_POSITION_TAG):
                            dpg.set_value(TXT_MOTOR_POSITION_TAG, f"Position: {mm:.4f} mm ({float(value):.0f} microsteps)")
                    except Exception:
                        pass
        except pyqueue.Empty:
            pass
        except Exception as exc:
            motor_log_lines.append(f"[motor event error] {exc}")

        try:
            snapshot = stepper_controller.snapshot()
            state = str(snapshot.get("state", "--"))
            connected = "connected" if bool(snapshot.get("connected")) else "disconnected"
            homed = "HOME complete" if bool(snapshot.get("homed")) else "HOME required"
            limits = f"MIN={'ACTIVE' if snapshot.get('min_active') else 'clear'} MAX={'ACTIVE' if snapshot.get('max_active') else 'clear'}"
            if dpg.does_item_exist(TXT_MOTOR_STATUS_TAG):
                dpg.set_value(TXT_MOTOR_STATUS_TAG, f"Carriage: {connected} | {state}\n{homed} | {limits}")
        except Exception:
            pass

        if dpg.does_item_exist(TXT_MOTOR_LOG_TAG):
            dpg.set_value(TXT_MOTOR_LOG_TAG, "\n".join(motor_log_lines[-4:]))

    def _refresh_sar_scan_ui() -> ScanState:
        """Aggiorna i controlli dal thread GUI e restituisce lo stato scan."""
        state = ScanState.IDLE
        active = False
        if sar_scan is not None:
            status = sar_scan.status()
            state = status.state
            active = sar_scan.active
            # Non sovrascrivere un errore di pre-avvio con lo stato ``idle``.
            # Prima il callback mostrava "Scan did not start: ...", ma il
            # frame GUI seguente lo rimpiazzava subito con "Ready".
            if active or state in {ScanState.COMPLETED, ScanState.CANCELLED, ScanState.FAILED}:
                detail = _scan_event_label(status.event)
                if status.position_id is not None:
                    detail += f" | id={int(status.position_id)}"
                if status.position_mm is not None:
                    detail += f" | x={float(status.position_mm):.4f} mm"
                if status.total:
                    detail += f" | {int(status.completed)}/{int(status.total)}"
                if status.error:
                    detail += f" | ERROR: {status.error}"
                if dpg.does_item_exist(TXT_SCAN_STATUS_TAG):
                    dpg.set_value(TXT_SCAN_STATUS_TAG, detail)
            if state is ScanState.CAPTURING and status.position_id is not None:
                with sar_pos_counter.get_lock():
                    sar_pos_counter.value = int(status.position_id)
                if dpg.does_item_exist(TXT_POS_TAG):
                    dpg.set_value(TXT_POS_TAG, f"Scan: position ID {int(status.position_id)}")

        busy_capture = bool(capture_sessions.inflight)
        if dpg.does_item_exist(TXT_CAPTURE_STATUS_TAG):
            if active:
                capture_detail = "Automatic capture is managed by the active scan."
            elif busy_capture:
                capture_detail = (
                    f"Manual capture in progress | position {int(cap_pos_id.value)} | "
                    f"{int(cap_saved.value)}/{int(FRAMES_PER_POSITION)} frame"
                )
            else:
                capture_detail = "Ready for a manual capture in the current run."
            dpg.set_value(TXT_CAPTURE_STATUS_TAG, capture_detail)
        if dpg.does_item_exist(BTN_CAPTURE_TAG):
            dpg.configure_item(BTN_CAPTURE_TAG, enabled=not active and not busy_capture)
        if dpg.does_item_exist(BTN_NEW_SESSION_TAG):
            dpg.configure_item(
                BTN_NEW_SESSION_TAG,
                enabled=not active and not busy_capture and scan_pending_run["start_position_id"] is None,
            )
        if dpg.does_item_exist(BTN_SCAN_START_TAG):
            dpg.configure_item(
                BTN_SCAN_START_TAG,
                enabled=(
                    sar_scan is not None
                    and not active
                    and not busy_capture
                    and scan_pending_run["start_position_id"] is None
                ),
            )
        if dpg.does_item_exist(BTN_SCAN_CANCEL_TAG):
            dpg.configure_item(BTN_SCAN_CANCEL_TAG, enabled=active)
        return state

    def _mmwave_connect_label(connected: bool) -> str:
        return "mmWave: Disconnect" if connected else "mmWave: Connect"

    def _mmwave_stream_label(streaming: bool) -> str:
        return "Radar: Stop" if streaming else "Radar: Start"

    mmwave_last_rx_pkt_t = time.time()
    mmwave_outage_dca_done = False
    mmwave_outage_heavy_done = False
    mmwave_auto_rearm_armed = False
    mmwave_connected_since_perf: float | None = None
    MMWAVE_CONNECT_SETTLE_S = 3.0
    MMWAVE_RX_IDLE_DCA_REARM_S = 1.00
    MMWAVE_RX_IDLE_HEAVY_REARM_S = 3.0

    def _sync_mmwave_udp_activity(now_wall: float | None = None) -> float:
        nonlocal mmwave_last_rx_pkt_t, mmwave_outage_dca_done, mmwave_outage_heavy_done
        try:
            with rx_last_packet_time_s.get_lock():
                shared_last = float(rx_last_packet_time_s.value)
        except Exception:
            shared_last = float(mmwave_last_rx_pkt_t)
        if now_wall is None:
            now_wall = time.time()
        if shared_last > float(mmwave_last_rx_pkt_t):
            mmwave_last_rx_pkt_t = float(shared_last)
            mmwave_outage_dca_done = False
            mmwave_outage_heavy_done = False
        return float(now_wall - mmwave_last_rx_pkt_t)

    def _refresh_mmwave_controls() -> None:
        state = mmwave_bridge.get_gui_state()
        if dpg.does_item_exist(BTN_MMWAVE_CONNECT_TAG):
            dpg.configure_item(BTN_MMWAVE_CONNECT_TAG, label=_mmwave_connect_label(state.connected))
        if dpg.does_item_exist(BTN_MMWAVE_STREAM_TAG):
            dpg.configure_item(BTN_MMWAVE_STREAM_TAG, label=_mmwave_stream_label(state.streaming))
            connect_wait_remaining = 0.0
            if state.connected and not state.streaming and mmwave_connected_since_perf is not None:
                connect_wait_remaining = max(
                    0.0,
                    MMWAVE_CONNECT_SETTLE_S - (time.perf_counter() - mmwave_connected_since_perf),
                )
            dpg.configure_item(
                BTN_MMWAVE_STREAM_TAG,
                enabled=bool(state.connected) and (state.streaming or connect_wait_remaining <= 0.0),
            )
        if dpg.does_item_exist(TXT_MMWAVE_STATUS_TAG):
            status_text = state.last_error if state.last_error else state.last_message
            if not status_text:
                status_text = "mmWave Studio bridge idle"
            if state.connected and not state.streaming and mmwave_connected_since_perf is not None:
                remaining = max(
                    0.0,
                    MMWAVE_CONNECT_SETTLE_S - (time.perf_counter() - mmwave_connected_since_perf),
                )
                if remaining > 0.0:
                    status_text = f"Radar connected: wait {remaining:.1f} s before START"
            dpg.set_value(TXT_MMWAVE_STATUS_TAG, status_text)
        if dpg.does_item_exist(TXT_MMWAVE_LINKS_TAG):
            udp_idle_s = max(0.0, _sync_mmwave_udp_activity())
            udp_receiving = bool(state.streaming and udp_idle_s <= MMWAVE_RX_IDLE_DCA_REARM_S)
            link_lines = [
                "LINK STATUS",
                f"{'RSTD':<18}{'ON' if state.rstd_connected else 'OFF'}",
                f"{'Radar link':<18}{'ON' if state.radar_connected else 'OFF'}",
                f"{'DCA ready':<18}{'ON' if state.dca_ready else 'OFF'}",
                f"{'Streaming req':<18}{'ON' if state.streaming else 'OFF'}",
                f"{'UDP receiving':<18}{'ON' if udp_receiving else 'OFF'}",
                f"{'UDP idle s':<18}{udp_idle_s:>6.1f}",
                f"{'Last rearm s':<18}{float(state.last_rearm_s):>6.1f}",
            ]
            dpg.set_value(TXT_MMWAVE_LINKS_TAG, "\n".join(link_lines))

    def _maybe_rearm_mmwave_stream(now_perf: float) -> None:
        nonlocal mmwave_last_rx_pkt_t, mmwave_outage_dca_done, mmwave_outage_heavy_done, mmwave_auto_rearm_armed
        state = mmwave_bridge.get_gui_state()
        if not mmwave_auto_rearm_armed or not state.connected or not state.streaming:
            return
        _ = now_perf
        idle_s = max(0.0, _sync_mmwave_udp_activity())
        if (not mmwave_outage_dca_done) and idle_s >= MMWAVE_RX_IDLE_DCA_REARM_S:
            mmwave_outage_dca_done = True
            print(f"[gui] mmWave auto-rearm (DCA ARM): UDP idle for {idle_s:.3f}s", flush=True)
            try:
                mmwave_bridge.rearm_dca_only(
                    mmwave_dca_cfg.adc_data_path,
                    capture_mode=1,
                    arm_delay_s=0.10,
                )
            except MmwaveStudioError as exc:
                mmwave_bridge.set_status(error=f"Auto DCA-arm failed: {exc}")
            except Exception as exc:
                mmwave_bridge.set_status(error=f"Unexpected auto DCA-arm error: {exc}")
            _refresh_mmwave_controls()
            return
        if mmwave_outage_dca_done and (not mmwave_outage_heavy_done) and idle_s >= MMWAVE_RX_IDLE_HEAVY_REARM_S:
            mmwave_outage_heavy_done = True
            print(f"[gui] mmWave auto-rearm (heavy): UDP idle for {idle_s:.3f}s", flush=True)
            try:
                mmwave_bridge.rearm_streaming(
                    mmwave_dca_cfg.adc_data_path,
                    capture_mode=1,
                    arm_delay_s=0.25,
                    stop_delay_s=0.25,
                )
            except MmwaveStudioError as exc:
                mmwave_bridge.set_status(error=f"Auto heavy rearm failed: {exc}")
            except Exception as exc:
                mmwave_bridge.set_status(error=f"Unexpected auto heavy rearm error: {exc}")
            _refresh_mmwave_controls()

    def _on_mmwave_connect_toggle():
        nonlocal mmwave_last_rx_pkt_t, mmwave_outage_dca_done, mmwave_outage_heavy_done
        nonlocal mmwave_auto_rearm_armed, mmwave_connected_since_perf
        if dpg.does_item_exist(TXT_MMWAVE_STATUS_TAG):
            dpg.set_value(TXT_MMWAVE_STATUS_TAG, "mmWave connect/disconnect in progress...")
        print("[gui] mmWave connect button pressed", flush=True)
        try:
            was_connected = bool(mmwave_bridge.get_gui_state().connected)
            new_state = mmwave_bridge.toggle_connection(radar=mmwave_radar_cfg, dca=mmwave_dca_cfg)
            mmwave_last_rx_pkt_t = time.time()
            mmwave_outage_dca_done = False
            mmwave_outage_heavy_done = False
            if new_state.connected and not was_connected:
                mmwave_connected_since_perf = time.perf_counter()
            elif not new_state.connected:
                mmwave_connected_since_perf = None
                mmwave_auto_rearm_armed = False
        except MmwaveStudioError as exc:
            state = mmwave_bridge.get_gui_state()
            mmwave_bridge.set_status(
                message=state.last_message or "mmWave connect failed",
                error=str(exc),
            )
        except Exception as exc:
            mmwave_bridge.set_status(error=f"Unexpected mmWave error: {exc}")
        _refresh_mmwave_controls()

    def _on_mmwave_stream_toggle():
        nonlocal mmwave_last_rx_pkt_t, mmwave_outage_dca_done, mmwave_outage_heavy_done, mmwave_auto_rearm_armed
        if dpg.does_item_exist(TXT_MMWAVE_STATUS_TAG):
            dpg.set_value(TXT_MMWAVE_STATUS_TAG, "Radar start/stop in progress...")
        print("[gui] Radar start/stop button pressed", flush=True)
        try:
            state_before = mmwave_bridge.get_gui_state()
            if state_before.connected and not state_before.streaming and mmwave_connected_since_perf is not None:
                remaining = MMWAVE_CONNECT_SETTLE_S - (
                    time.perf_counter() - mmwave_connected_since_perf
                )
                if remaining > 0.0:
                    mmwave_bridge.set_status(
                        message=f"Wait another {remaining:.1f} s before starting the radar"
                    )
                    _refresh_mmwave_controls()
                    return
            mmwave_bridge.toggle_streaming(
                mmwave_dca_cfg.adc_data_path,
                capture_mode=1,
                arm_delay_s=1.0,
                stop_delay_s=2.0,
            )
            mmwave_last_rx_pkt_t = time.time()
            mmwave_outage_dca_done = False
            mmwave_outage_heavy_done = False
            mmwave_auto_rearm_armed = bool(mmwave_bridge.get_gui_state().streaming)
        except MmwaveStudioError as exc:
            error_message = str(exc)
            if not mmwave_bridge.is_hw_connected:
                mmwave_bridge.set_status(
                    message="Connect hardware before starting the radar",
                    error=error_message,
                )
            else:
                mmwave_bridge.set_status(error=error_message)
        except Exception as exc:
            mmwave_bridge.set_status(error=f"Unexpected radar streaming error: {exc}")
        _refresh_mmwave_controls()

    def _norm_toggle_label(enabled: bool) -> str:
        return "NORM: ON" if enabled else "NORM: OFF"

    def _heatmap_mode_label(mode: int) -> str:
        return "HEATMAP: XY MOVING VELOCITY" if int(mode) == 1 else "HEATMAP: POWER"

    def _heatmap_norm_label(mode: int, enabled: bool) -> str:
        if int(mode) == 1:
            return "NORM: OFF (VELOCITY)"
        return _norm_toggle_label(enabled)

    def _heatmap_vscale_for_mode(mode: int, norm_enabled: bool):
        if int(mode) == 1:
            vmin = float(HEATMAP_VELOCITY_VMIN_MPS)
            vmax = float(HEATMAP_VELOCITY_VMAX_MPS)
            if not np.isfinite(vmin):
                vmin = -1.0
            if not np.isfinite(vmax):
                vmax = 1.0
            if vmax <= vmin:
                vmax = vmin + 1e-3
            return vmin, vmax
        return _base_vscale_for_mode(norm_enabled)

    def _sanitize_heatmap_scale_inputs(mode: int, vmin: float, vmax: float, *, sender=None):
        if int(mode) == 1:
            default_vmin, default_vmax = _heatmap_vscale_for_mode(1, False)
            if not np.isfinite(vmin):
                vmin = float(default_vmin)
            if not np.isfinite(vmax):
                vmax = float(default_vmax)
            if vmax <= vmin:
                vmax = vmin + 1e-3
            return float(vmin), float(vmax)
        if vmax <= vmin:
            vmax = vmin + 1.0
        return float(vmin), float(vmax)

    def _get_heatmap_mode() -> int:
        try:
            with heatmap_view_mode.get_lock():
                mode = int(heatmap_view_mode.value)
        except Exception:
            mode = 0
        return 1 if mode == 1 else 0

    def _update_heatmap_scale_input_labels(mode: int):
        unit = "m/s" if int(mode) == 1 else "dB"
        if dpg.does_item_exist(IN_VMIN):
            dpg.configure_item(IN_VMIN, label=f"Vmin ({unit})")
        if dpg.does_item_exist(IN_VMAX):
            dpg.configure_item(IN_VMAX, label=f"Vmax ({unit})")
        if dpg.does_item_exist(IN_XMAX):
            dpg.configure_item(IN_XMAX, label="Xmax (m)", enabled=True)

    def _set_realtime_xy_overlays_visible(visible: bool):
        show = bool(visible)
        for tag in (
            GUIDE_NEG20_TAG,
            GUIDE_POS20_TAG,
            TRACK_SCATTER_CONF_TAG,
            TRACK_SCATTER_UNCONF_TAG,
            TRACK_SCATTER_MOVING_TAG,
            TRACK_SCATTER_STOPPED_TAG,
            TRACK_SCATTER_UNKNOWN_TAG,
            TRACK_STOP_MARKER_TAG,
            TRACK_VEL_SERIES_TAG,
        ):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, show=show)
        if not show and supports_plot_annotation and track_annotation_tags:
            for ann_tag in track_annotation_tags:
                if dpg.does_item_exist(ann_tag):
                    dpg.configure_item(ann_tag, show=False)

    def _apply_heatmap_plot_geometry(mode: int, rmax: float, xmax: float, *, reset_limits: bool = False):
        _configure_image_bounds(IMG_SERIES_TAG, rt_applied_meta_current)
        if dpg.does_item_exist(XAXIS_TAG):
            dpg.configure_item(XAXIS_TAG, label="X (m)")
            if reset_limits:
                dpg.set_axis_limits(XAXIS_TAG, -float(xmax), +float(xmax))
        if dpg.does_item_exist(YAXIS_TAG):
            dpg.configure_item(YAXIS_TAG, label="Y (m)")
            if reset_limits:
                dpg.set_axis_limits(YAXIS_TAG, 0.0, float(rmax))
        _set_realtime_xy_overlays_visible(True)

    def _apply_heatmap_mode_controls(mode: int):
        norm_enabled = _get_norm_enabled()
        if dpg.does_item_exist(BTN_HEATMAP_MODE_TAG):
            dpg.configure_item(BTN_HEATMAP_MODE_TAG, label=_heatmap_mode_label(mode))
        if dpg.does_item_exist(BTN_NORM_TAG):
            dpg.configure_item(
                BTN_NORM_TAG,
                label=_heatmap_norm_label(mode, norm_enabled),
                enabled=(int(mode) == 0),
            )
        _update_heatmap_scale_input_labels(mode)
        if dpg.does_item_exist(CMAP_SCALE_TAG):
            fmt = CMAP_VELOCITY_NUM_FMT if int(mode) == 1 else CMAP_NUM_FMT
            cmap = CMAP_VELOCITY_TAG if int(mode) == 1 and dpg.does_item_exist(CMAP_VELOCITY_TAG) else dpg.mvPlotColormap_Jet
            dpg.configure_item(CMAP_SCALE_TAG, format=fmt, colormap=cmap)
        _set_realtime_xy_overlays_visible(True)

    def _base_vscale_for_mode(enabled: bool):
        if enabled:
            vmin = float(VMIN_NORM)
            vmax = float(VMAX_NORM)
        else:
            vmin = float(VMIN_RAW)
            vmax = float(VMAX_RAW)
        if vmax <= vmin:
            vmax = vmin + 1.0
        return vmin, vmax

    def _get_norm_enabled() -> bool:
        try:
            with norm_to_peak.get_lock():
                return bool(norm_to_peak.value)
        except Exception:
            return True

    def _toggle_norm(sender=None, app_data=None):
        if _get_heatmap_mode() == 1:
            if dpg.does_item_exist(BTN_NORM_TAG):
                dpg.configure_item(BTN_NORM_TAG, label=_heatmap_norm_label(1, False), enabled=False)
            return
        try:
            with norm_to_peak.get_lock():
                norm_to_peak.value = 0 if int(norm_to_peak.value) else 1
                enabled = bool(norm_to_peak.value)
        except Exception:
            enabled = True
        if dpg.does_item_exist(BTN_NORM_TAG):
            dpg.configure_item(BTN_NORM_TAG, label=_heatmap_norm_label(0, enabled), enabled=True)
        base_vmin, base_vmax = _heatmap_vscale_for_mode(0, enabled)
        if dpg.does_item_exist(IN_VMIN):
            dpg.set_value(IN_VMIN, base_vmin)
        if dpg.does_item_exist(IN_VMAX):
            dpg.set_value(IN_VMAX, base_vmax)
        _apply_params()

    def _toggle_heatmap_mode(sender=None, app_data=None):
        try:
            with heatmap_view_mode.get_lock():
                heatmap_view_mode.value = 0 if int(heatmap_view_mode.value) == 1 else 1
                mode = int(heatmap_view_mode.value)
        except Exception:
            mode = 0
        mode = 1 if mode == 1 else 0
        base_vmin, base_vmax = _heatmap_vscale_for_mode(mode, _get_norm_enabled())
        if dpg.does_item_exist(IN_VMIN):
            dpg.set_value(IN_VMIN, base_vmin)
        if dpg.does_item_exist(IN_VMAX):
            dpg.set_value(IN_VMAX, base_vmax)
        _apply_heatmap_mode_controls(mode)
        ui_pending["reset_view"] = True
        _apply_params()

    def _toggle_fft_mode(sender=None, app_data=None):
        nonlocal fft_mode_db
        fft_mode_db = not bool(fft_mode_db)
        if dpg.does_item_exist(BTN_FFT_MODE_TAG):
            dpg.configure_item(BTN_FFT_MODE_TAG, label=_fft_mode_label(fft_mode_db))
        _update_fft_scale_input_labels(fft_mode_db)
        fft_vmin, fft_vmax = _base_fft_scale_for_mode(fft_mode_db)
        if dpg.does_item_exist(IN_FFT_VMIN):
            dpg.set_value(IN_FFT_VMIN, fft_vmin)
        if dpg.does_item_exist(IN_FFT_VMAX):
            dpg.set_value(IN_FFT_VMAX, fft_vmax)
        _apply_params()

    def _toggle_fft_view(sender=None, app_data=None):
        nonlocal fft_view_full, ui_dirty
        fft_view_full = not bool(fft_view_full)
        try:
            rmax_now = float(dpg.get_value(IN_RMAX))
        except Exception:
            rmax_now = float(vis_rmax)
        try:
            fft_xmin_now = float(dpg.get_value(IN_FFT_XMIN))
        except Exception:
            fft_xmin_now = float(ui_pending.get("fft_xmin", 0.0))
        _, view_max_m = _fft_visible_range_from_view(rmax_now, fft_view_full)
        fft_xmin_now = max(0.0, min(float(fft_xmin_now), float(view_max_m)))
        if fft_xmin_now >= float(view_max_m):
            fft_xmin_now = 0.0
        fft_xmax_now = float(view_max_m)
        if dpg.does_item_exist(BTN_FFT_VIEW_TAG):
            dpg.configure_item(BTN_FFT_VIEW_TAG, label=_fft_view_label(fft_view_full))
        if dpg.does_item_exist(IN_FFT_XMIN):
            dpg.set_value(IN_FFT_XMIN, fft_xmin_now)
        if dpg.does_item_exist(IN_FFT_XMAX):
            dpg.set_value(IN_FFT_XMAX, fft_xmax_now)
        ui_pending["fft_xmin"] = float(fft_xmin_now)
        ui_pending["fft_xmax"] = float(fft_xmax_now)
        ui_pending["fft_view_full"] = bool(fft_view_full)
        ui_dirty = True

    def _on_angle_single_bin(sender=None, app_data=None):
        nonlocal angle_single_bin
        angle_single_bin = bool(app_data)
        if dpg.does_item_exist(ANGLEFFT_HEAT_PLOT_TAG):
            dpg.configure_item(ANGLEFFT_HEAT_PLOT_TAG, show=not angle_single_bin)
        if dpg.does_item_exist(ANGLEFFT_PROFILE_PLOT_TAG):
            dpg.configure_item(ANGLEFFT_PROFILE_PLOT_TAG, show=angle_single_bin)

    def _on_doppler_single_bin(sender=None, app_data=None):
        nonlocal doppler_single_bin
        doppler_single_bin = bool(app_data)
        if dpg.does_item_exist(DOPPLERFFT_HEAT_PLOT_TAG):
            dpg.configure_item(DOPPLERFFT_HEAT_PLOT_TAG, show=not doppler_single_bin)
        if dpg.does_item_exist(DOPPLERFFT_PROFILE_PLOT_TAG):
            dpg.configure_item(DOPPLERFFT_PROFILE_PLOT_TAG, show=doppler_single_bin)

    def _read_diag_scale(vmin_tag: str, vmax_tag: str, default_vmin: float, default_vmax: float) -> tuple[float, float]:
        try:
            vmin = float(dpg.get_value(vmin_tag)) if dpg.does_item_exist(vmin_tag) else float(default_vmin)
            vmax = float(dpg.get_value(vmax_tag)) if dpg.does_item_exist(vmax_tag) else float(default_vmax)
        except (TypeError, ValueError):
            return float(default_vmin), float(default_vmax)
        if not np.isfinite(vmin):
            vmin = float(default_vmin)
        if not np.isfinite(vmax):
            vmax = float(default_vmax)
        if vmax <= vmin:
            vmax = vmin + 1.0
        return float(vmin), float(vmax)

    def _diag_norm_enabled(tag: str, default_value: bool) -> bool:
        try:
            return bool(dpg.get_value(tag)) if dpg.does_item_exist(tag) else bool(default_value)
        except Exception:
            return bool(default_value)

    def _update_diag_colormap_scale(tag: str, vmin: float, vmax: float) -> None:
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, min_scale=float(vmin), max_scale=float(vmax), format=CMAP_NUM_FMT)

    def _normalize_diag_map_to_lut(
        src_db: np.ndarray,
        out_norm: np.ndarray,
        *,
        vmin: float,
        vmax: float,
        norm_to_peak: bool,
    ) -> float:
        peak = 0.0
        if bool(norm_to_peak) and src_db.size > 0:
            finite_src = src_db[np.isfinite(src_db)]
            peak = float(np.max(finite_src)) if finite_src.size > 0 else 0.0
            if peak <= -119.0:
                peak = 0.0
            np.subtract(src_db, peak, out=out_norm)
        else:
            out_norm[:, :] = src_db
        denom_local = float(vmax - vmin)
        if denom_local < 1e-6:
            denom_local = 1e-6
        np.subtract(out_norm, float(vmin), out=out_norm)
        out_norm *= float(1.0 / denom_local)
        np.clip(out_norm, 0.0, 1.0, out=out_norm)
        np.nan_to_num(out_norm, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        return float(peak)

    def _base_off_vscale_for_mode(enabled: bool):
        if enabled:
            vmin = float(VMIN_NORM)
            vmax = float(VMAX_NORM)
        else:
            vmin = float(VMIN_RAW)
            vmax = float(VMAX_RAW)
        if vmax <= vmin:
            vmax = vmin + 1.0
        return vmin, vmax

    def _toggle_off_norm(sender=None, app_data=None):
        nonlocal off_ui_pending
        enabled = not bool(off_ui_pending.get("norm_enabled", True))
        off_ui_pending["norm_enabled"] = bool(enabled)
        if dpg.does_item_exist(PROC_BTN_NORM):
            dpg.configure_item(PROC_BTN_NORM, label=_norm_toggle_label(enabled))
        base_vmin, base_vmax = _base_off_vscale_for_mode(enabled)
        if dpg.does_item_exist(PROC_IN_VMIN):
            dpg.set_value(PROC_IN_VMIN, base_vmin)
        if dpg.does_item_exist(PROC_IN_VMAX):
            dpg.set_value(PROC_IN_VMAX, base_vmax)
        _apply_offline_params()

    def _write_offline_zoom_inputs(viewport: DisplayViewport) -> None:
        values = (
            (PROC_IN_ZOOM_XMIN, float(viewport.x_min_m)),
            (PROC_IN_ZOOM_XMAX, float(viewport.x_max_m)),
            (PROC_IN_ZOOM_YMIN, float(viewport.y_min_m)),
            (PROC_IN_ZOOM_YMAX, float(viewport.y_max_m)),
        )
        for tag, value in values:
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, value)

    def _apply_offline_display_zoom(sender=None, app_data=None):
        """Reconstruct the requested offline ROI on the fixed offline grid."""
        nonlocal off_requested_viewport_current, off_ui_dirty, off_ui_dirty_t
        try:
            x0 = float(dpg.get_value(PROC_IN_ZOOM_XMIN))
            x1 = float(dpg.get_value(PROC_IN_ZOOM_XMAX))
            y0 = float(dpg.get_value(PROC_IN_ZOOM_YMIN))
            y1 = float(dpg.get_value(PROC_IN_ZOOM_YMAX))
        except (TypeError, ValueError):
            return
        if not all(np.isfinite(value) for value in (x0, x1, y0, y1)):
            return

        home = off_home_viewport_current
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        x0 = max(float(home.x_min_m), min(float(home.x_max_m), x0))
        x1 = max(float(home.x_min_m), min(float(home.x_max_m), x1))
        y0 = max(float(home.y_min_m), min(float(home.y_max_m), y0))
        y1 = max(float(home.y_min_m), min(float(home.y_max_m), y1))

        min_x_span = max(
            (float(home.x_max_m) - float(home.x_min_m)) / float(max(1, offline_gui_w)),
            1e-6,
        )
        min_y_span = max(float(dr_plot), 1e-6)
        if x1 <= x0:
            x1 = min(float(home.x_max_m), x0 + min_x_span)
            if x1 <= x0:
                x0 = max(float(home.x_min_m), x1 - min_x_span)
        if y1 <= y0:
            y1 = min(float(home.y_max_m), y0 + min_y_span)
            if y1 <= y0:
                y0 = max(float(home.y_min_m), y1 - min_y_span)

        off_requested_viewport_current = build_display_viewport(
            x_min_m=float(x0),
            x_max_m=float(x1),
            y_min_m=float(y0),
            y_max_m=float(y1),
            dr_m=float(dr_plot),
            seq=int(off_requested_viewport_current.seq) + 1,
            home_viewport=home,
        )
        # The offline DSP worker uses this viewport to build its x/y grid, so
        # this is a real back-projection request rather than a plot crop.
        off_ui_pending["reset_view"] = False
        off_ui_pending["reconstruct"] = True
        off_ui_pending["display_refresh"] = False
        off_ui_dirty = True
        off_ui_dirty_t = time.perf_counter()
        _render_offline_summary(state="Offline ROI reconstruction queued")
        _write_offline_zoom_inputs(off_requested_viewport_current)
        if dpg.does_item_exist(PROC_XAXIS_TAG) and dpg.does_item_exist(PROC_YAXIS_TAG):
            dpg.set_axis_limits(PROC_XAXIS_TAG, float(x0), float(x1))
            dpg.set_axis_limits(PROC_YAXIS_TAG, float(y0), float(y1))
        if dpg.does_item_exist(PROC_TXT_ZOOM_STATUS):
            dpg.set_value(
                PROC_TXT_ZOOM_STATUS,
                f"ROI: {dpg.get_item_configuration(PROC_XAXIS_TAG).get('label', 'X')} [{x0:.3f}, {x1:.3f}] m | "
                f"{dpg.get_item_configuration(PROC_YAXIS_TAG).get('label', 'Y')} [{y0:.3f}, {y1:.3f}] m\n"
                f"Offline reconstruction queued on {offline_gui_w}x{offline_gui_h} grid.",
            )

    def _reset_offline_display_zoom(sender=None, app_data=None):
        nonlocal off_requested_viewport_current, off_ui_dirty, off_ui_dirty_t
        off_requested_viewport_current = off_home_viewport_current
        off_ui_pending["reconstruct"] = True
        off_ui_pending["display_refresh"] = False
        off_ui_dirty = True
        off_ui_dirty_t = time.perf_counter()
        _render_offline_summary(state="Offline full-map reconstruction queued")
        _write_offline_zoom_inputs(off_home_viewport_current)
        if dpg.does_item_exist(PROC_XAXIS_TAG) and dpg.does_item_exist(PROC_YAXIS_TAG):
            dpg.set_axis_limits(
                PROC_XAXIS_TAG,
                float(off_home_viewport_current.x_min_m),
                float(off_home_viewport_current.x_max_m),
            )
            dpg.set_axis_limits(
                PROC_YAXIS_TAG,
                float(off_home_viewport_current.y_min_m),
                float(off_home_viewport_current.y_max_m),
            )
        if dpg.does_item_exist(PROC_TXT_ZOOM_STATUS):
            dpg.set_value(
                PROC_TXT_ZOOM_STATUS,
                f"Full-map offline reconstruction queued on {offline_gui_w}x{offline_gui_h} grid.",
            )

    def _apply_offline_params(sender=None, app_data=None):
        nonlocal off_ui_dirty, off_ui_dirty_t, off_ui_pending

        try:
            vmin = float(dpg.get_value(PROC_IN_VMIN))
            vmax = float(dpg.get_value(PROC_IN_VMAX))
        except (TypeError, ValueError):
            return
        try:
            x_start = int(dpg.get_value(PROC_IN_XSTART))
            x_end = int(dpg.get_value(PROC_IN_XEND))
        except (TypeError, ValueError):
            return
        x_start_cl = max(off_pos_min, min(off_pos_max, x_start))
        x_end_cl = max(off_pos_min, min(off_pos_max, x_end))
        if x_end_cl < x_start_cl:
            x_start_cl, x_end_cl = x_end_cl, x_start_cl
        if x_start_cl != x_start:
            dpg.set_value(PROC_IN_XSTART, x_start_cl)
        if x_end_cl != x_end:
            dpg.set_value(PROC_IN_XEND, x_end_cl)
        prev_x_start = int(off_ui_pending.get("x_start", x_start_cl))
        prev_x_end = int(off_ui_pending.get("x_end", x_end_cl))
        reconstruction_changed = bool(
            prev_x_start != int(x_start_cl)
            or prev_x_end != int(x_end_cl)
        )

        if vmax <= vmin:
            vmax = vmin + 1.0

        off_ui_pending["vmin"] = float(vmin)
        off_ui_pending["vmax"] = float(vmax)
        off_ui_pending["x_start"] = int(x_start_cl)
        off_ui_pending["x_end"] = int(x_end_cl)
        off_ui_pending["reconstruct"] = bool(
            off_ui_pending.get("reconstruct", False) or reconstruction_changed
        )
        off_ui_pending["display_refresh"] = True
        if bool(off_ui_pending.get("reconstruct", False)):
            update_label = "Offline reconstruction update pending"
        elif offline_frame_valid:
            update_label = "Offline display updated from cached matrix"
        else:
            update_label = "No valid offline result; load/recalculate first"
        _render_offline_summary(state=update_label)
        off_ui_dirty = True
        off_ui_dirty_t = time.perf_counter()

    # Offline tuning deliberately uses an explicit save/apply action.  Unlike
    # realtime tuning, these settings are consumed while the offline reader is
    # built and cannot be changed safely through its live command queue.
    OFFLINE_TUNING_WINDOWS = ["none", "rectangular", "hanning", "hamming", "blackman"]
    OFFLINE_TUNING_ANGLE_MODES = ["fft", "bartlett", "mvdr"]
    OFFLINE_TUNING_AGGREGATIONS = ["frame_loop", "frame", "loop", "none"]
    OFFLINE_TUNING_ALGORITHMS = ["synthetic_range_angle", "backprojection"]

    OFFLINE_TUNING_FIELD_SPECS = [
        {"section": "Data and scan", "path": "data.input_dir", "label": "Input folder", "kind": "text", "default": "logs", "mode": "shared"},
        {"section": "Data and scan", "path": "data.startup_timeout_s", "label": "Startup timeout (s)", "kind": "int", "default": 300, "min": 1, "max": 3600, "step": 30, "mode": "shared"},
        {"section": "Data and scan", "path": "capture.frames_per_position", "label": "Frames to use per position", "kind": "int", "default": 8, "min": 1, "step": 1, "mode": "shared"},
        {"section": "Data and scan", "path": "scan.x_start", "label": "Start position", "kind": "int", "default": 1, "step": 1, "mode": "linear"},
        {"section": "Data and scan", "path": "scan.x_end", "label": "End position", "kind": "int", "default": 1, "step": 1, "mode": "linear"},
        {"section": "Data and scan", "path": "scan.x_step", "label": "Position step (use every Nth)", "kind": "int", "default": 1, "min": 1, "step": 1, "mode": "linear"},
        {"section": "Data and scan", "path": "scan.x_pitch_m", "label": "X pitch (m)", "kind": "float", "default": 0.01, "min": 0.000001, "step": 0.001, "format": "%.6f", "mode": "linear"},
        {"section": "Reconstruction", "path": "reconstruction.algorithm", "label": "Algorithm", "kind": "combo", "items": OFFLINE_TUNING_ALGORITHMS, "default": "synthetic_range_angle", "mode": "linear"},
        {"section": "Reconstruction", "path": "bp.phase_sign", "label": "Phase sign", "kind": "combo", "items": ["-1", "1"], "default": "-1", "mode": "shared"},
        {"section": "Map reconstruction bounds", "path": "reconstruction.map_bounds.x_min_m", "label": "Map X min (m)", "kind": "float", "default": -25.0, "step": 0.5, "mode": "linear"},
        {"section": "Map reconstruction bounds", "path": "reconstruction.map_bounds.x_max_m", "label": "Map X max (m)", "kind": "float", "default": 25.0, "step": 0.5, "mode": "linear"},
        {"section": "Map reconstruction bounds", "path": "reconstruction.map_bounds.y_min_m", "label": "Map Y (range) min (m)", "kind": "float", "default": 0.0, "min": 0.0, "step": 0.5, "mode": "linear"},
        {"section": "Map reconstruction bounds", "path": "reconstruction.map_bounds.y_max_m", "label": "Map Y (range) max (m)", "kind": "float", "default": 50.0, "min": 0.0, "step": 0.5, "mode": "linear"},
        {"section": "Offline reference background", "path": "offline_background.enabled", "label": "Subtract empty-scene reference", "kind": "bool", "default": False},
        {"section": "Offline reference background", "path": "offline_background.reference_dir", "label": "Empty-scene folder", "kind": "text", "default": ""},
        {"section": "Offline reference background", "path": "offline_background.scale", "label": "Reference scale", "kind": "float", "default": 1.0, "min": 0.0, "step": 0.01},
        {"section": "FFT sizing", "path": "offline_sar_range_angle.nfft_range", "label": "Range FFT bins", "kind": "int", "default": 1024, "min": 1, "step": 64},
        {"section": "FFT sizing", "path": "offline_sar_range_angle.nfft_angle", "label": "Angle FFT bins", "kind": "int", "default": 2048, "min": 1, "step": 64},
        {"section": "Shared windows", "path": "offline_sar_range_angle.use_realtime_filters", "label": "Enable preprocessing", "kind": "bool", "default": True},
        {"section": "Shared windows", "path": "offline_sar_range_angle.window_range", "label": "Range window", "kind": "combo", "items": OFFLINE_TUNING_WINDOWS, "default": "hanning"},
        {"section": "Shared windows", "path": "offline_sar_range_angle.window_doppler", "label": "Doppler window", "kind": "combo", "items": OFFLINE_TUNING_WINDOWS, "default": "hanning"},
        {"section": "Shared windows", "path": "offline_sar_range_angle.window_angle", "label": "Angle / aperture (BP)", "kind": "combo", "items": OFFLINE_TUNING_WINDOWS, "default": "hanning"},
        {"section": "Range-angle filters", "path": "offline_sar_range_angle.zero_after_range_fft_bins", "label": "Zero initial bins", "kind": "int", "default": 0, "min": 0, "step": 1},
        {"section": "Range-angle filters", "path": "offline_sar_range_angle.mean_after_range_fft.enabled", "label": "Mean after Range FFT", "kind": "bool", "default": False},
        {"section": "Angle estimation", "path": "offline_sar_range_angle.angle_processing.mode", "label": "Method", "kind": "combo", "items": OFFLINE_TUNING_ANGLE_MODES, "default": "bartlett"},
        {"section": "Angle estimation", "path": "offline_sar_range_angle.angle_processing.mvdr_diagonal_loading", "label": "MVDR loading", "kind": "float", "default": 0.02, "min": 0.0, "step": 0.005},
        {"section": "Angle estimation", "path": "offline_sar_range_angle.angle_processing.aggregation", "label": "Aggregation", "kind": "combo", "items": OFFLINE_TUNING_AGGREGATIONS, "default": "frame_loop"},
        {"section": "Angle estimation", "path": "offline_sar_range_angle.angle_processing.frame_index", "label": "Frame index", "kind": "int", "default": 0, "min": 0, "step": 1},
        {"section": "Angle estimation", "path": "offline_sar_range_angle.angle_processing.loop_index", "label": "Loop index", "kind": "int", "default": 0, "min": 0, "step": 1},
    ]
    offline_config_path = Path(__file__).with_name("offline_config.yaml")
    offline_tuning_cfg: dict = {}
    offline_reload_lock = threading.Lock()
    offline_reload_state = {"running": False, "runtime": None, "info": None, "error": ""}

    def _offline_tune_tag(path: str) -> str:
        return "offline_tune__" + str(path).replace(".", "__")

    def _offline_cfg_path_get(source: dict, path: str, default=None):
        node = source
        for part in str(path).split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def _offline_cfg_path_set(target: dict, path: str, value) -> None:
        node = target
        parts = str(path).split(".")
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value

    def _set_offline_frame_limit(available: int | None, *, input_dir: str = "") -> None:
        """Reflect the inspected frame capacity in the Data-tab control."""
        tag = _offline_tune_tag("capture.frames_per_position")
        available_i = None if available is None else max(1, int(available))
        offline_frame_limit["available"] = available_i
        offline_frame_limit["input_dir"] = str(input_dir)
        if not dpg.does_item_exist(tag):
            return
        if available_i is None:
            dpg.configure_item(
                tag,
                label="Frames to use per position",
                max_clamped=False,
            )
            return
        current = max(1, int(dpg.get_value(tag)))
        dpg.configure_item(
            tag,
            label=f"Frames to use per position (1-{available_i} available)",
            min_value=1,
            max_value=available_i,
            min_clamped=True,
            max_clamped=True,
        )
        dpg.set_value(tag, min(current, available_i))

    def _set_offline_tuning_status(message: str) -> None:
        if dpg.does_item_exist(TXT_OFFLINE_TUNING_STATUS_TAG):
            dpg.set_value(TXT_OFFLINE_TUNING_STATUS_TAG, str(message))

    def _offline_tuning_snapshot() -> dict:
        base = yaml.safe_load(yaml.safe_dump(offline_tuning_cfg, sort_keys=False)) or {}
        if not isinstance(base, dict):
            base = {}
        missing = object()
        for spec in OFFLINE_TUNING_FIELD_SPECS:
            tag = _offline_tune_tag(spec["path"])
            current = _offline_cfg_path_get(base, spec["path"], missing)
            if dpg.does_item_exist(tag):
                value = _offline_tuning_widget_value(spec)
            else:
                value = spec.get("default") if current is missing else current
            _offline_cfg_path_set(base, spec["path"], value)
        return base

    def _apply_offline_geometry_controls() -> None:
        """Apply the fixed controls used by linear V1 captures."""
        for tag in (
            OFFLINE_TUNING_GRP_LINEAR_SCAN,
            OFFLINE_TUNING_GRP_RECONSTRUCTION,
            OFFLINE_TUNING_GRP_MAP_BOUNDS,
            PROC_GRP_LINEAR_POSITION_SELECTION,
        ):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, show=True)
        if dpg.does_item_exist(PROC_XAXIS_TAG):
            dpg.configure_item(PROC_XAXIS_TAG, label="X (m)")
        if dpg.does_item_exist(PROC_YAXIS_TAG):
            dpg.configure_item(PROC_YAXIS_TAG, label="Y (m)")
        for tag, label in (
            (PROC_IN_ZOOM_XMIN, "X min (m)"),
            (PROC_IN_ZOOM_XMAX, "X max (m)"),
            (PROC_IN_ZOOM_YMIN, "Y min (m)"),
            (PROC_IN_ZOOM_YMAX, "Y max (m)"),
        ):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, label=label)
        if dpg.does_item_exist(TXT_OFFLINE_RUN_FORMAT):
            dpg.set_value(TXT_OFFLINE_RUN_FORMAT, "Format: rt_capture_v1 / linear")

    def _inspect_offline_run(sender=None, app_data=None) -> bool:
        """Inspect the configured linear V1 run without loading its payload."""
        try:
            snapshot = _offline_tuning_snapshot()
            # Inspection must work even when the current control value is
            # larger than this run.  Read the full header/file capacity first,
            # then clamp the GUI control to a valid selectable range.
            inspection_snapshot = yaml.safe_load(yaml.safe_dump(snapshot, sort_keys=False)) or {}
            capture_cfg = inspection_snapshot.get("capture")
            if isinstance(capture_cfg, dict):
                capture_cfg.pop("frames_per_position", None)
            reader_cfg = OfflineSARConfig.from_mapping(
                inspection_snapshot,
                base_dir=offline_config_path.parent,
            )
            layout = SARReader(config=reader_cfg).describe_stream()
            available_frames = int(layout.available_frames_per_position)
            _set_offline_frame_limit(
                available_frames,
                input_dir=str(_offline_cfg_path_get(snapshot, "data.input_dir", "")),
            )
            snapshot = _offline_tuning_snapshot()
            message = (
                f"Detected rt_capture_v1 linear run: {int(layout.positions.size)} positions | "
                f"{available_frames} frames available per position"
            )
            _apply_offline_geometry_controls()
            _refresh_offline_memory_estimate(snapshot)
            _set_offline_tuning_status(message)
            return True
        except Exception as exc:
            _apply_offline_geometry_controls()
            _set_offline_tuning_status(f"Run inspection error: {exc}")
            return False

    def _offline_tuning_locked_by_scan() -> bool:
        return bool(
            _any_scan_active()
            or scan_pending_run["start_position_id"] is not None
        )

    def _refresh_offline_memory_estimate(cfg_source: dict | None = None) -> str:
        nonlocal offline_memory_estimate_text
        source = cfg_source if isinstance(cfg_source, dict) else offline_tuning_cfg
        offline_memory_estimate_text = _format_offline_memory_estimate(source)
        _render_offline_summary()
        if offline_runtime is None and not bool(offline_reload_state.get("running")):
            if dpg.does_item_exist(TXT_OFFLINE_TUNING_INFO_TAG):
                dpg.set_value(
                    TXT_OFFLINE_TUNING_INFO_TAG,
                    "Offline runtime not loaded.\n" + offline_memory_estimate_text,
                )
        return offline_memory_estimate_text

    def _read_offline_tuning_config() -> bool:
        nonlocal offline_tuning_cfg
        try:
            with offline_config_path.open("r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                raise ValueError("la radice YAML deve essere una mappa")
            offline_tuning_cfg = loaded
            _set_offline_frame_limit(None)
            for spec in OFFLINE_TUNING_FIELD_SPECS:
                tag = _offline_tune_tag(spec["path"])
                if not dpg.does_item_exist(tag):
                    continue
                value = _offline_cfg_path_get(offline_tuning_cfg, spec["path"], spec.get("default"))
                if spec["kind"] == "combo":
                    value = str(value)
                    if value not in spec.get("items", []):
                        value = str(spec.get("default"))
                dpg.set_value(tag, value)
            try:
                pitch_mm = float(_offline_cfg_path_get(offline_tuning_cfg, "scan.x_pitch_m")) * 1000.0
                _update_scan_pitch_label(pitch_mm)
            except (TypeError, ValueError):
                _update_scan_pitch_label(error="invalid scan.x_pitch_m")
            _refresh_offline_memory_estimate(offline_tuning_cfg)
            _apply_offline_geometry_controls()
            _set_offline_tuning_status("Offline configuration loaded")
            return True
        except Exception as exc:
            _set_offline_tuning_status(f"Configuration read error: {exc}")
            return False

    def _offline_tuning_widget_value(spec: dict):
        value = dpg.get_value(_offline_tune_tag(spec["path"]))
        kind = str(spec["kind"])
        if kind == "text":
            return str(value).strip()
        if kind == "bool":
            return bool(value)
        if kind == "combo":
            value_s = str(value)
            if value_s not in spec.get("items", []):
                value_s = str(spec.get("default"))
            return int(value_s) if spec["path"] == "bp.phase_sign" else value_s
        if kind == "int":
            value_i = int(value)
            if "min" in spec:
                value_i = max(int(spec["min"]), value_i)
            if "max" in spec:
                value_i = min(int(spec["max"]), value_i)
            return value_i
        value_f = float(value)
        if not np.isfinite(value_f):
            raise ValueError(f"{spec['label']}: valore non finito")
        if "min" in spec:
            value_f = max(float(spec["min"]), value_f)
        if "max" in spec:
            value_f = min(float(spec["max"]), value_f)
        return float(value_f)

    def _collect_offline_tuning_config() -> dict:
        base = _offline_tuning_snapshot()

        if not str(_offline_cfg_path_get(base, "data.input_dir", "")).strip():
            raise ValueError("Input folder is required")
        if int(_offline_cfg_path_get(base, "data.startup_timeout_s", 300)) <= 0:
            raise ValueError("Startup timeout must be greater than zero")
        frames_per_position = int(_offline_cfg_path_get(base, "capture.frames_per_position", 1))
        if frames_per_position <= 0:
            raise ValueError("Frames to use per position must be greater than zero")
        inspected_available = offline_frame_limit.get("available")
        inspected_input = str(offline_frame_limit.get("input_dir", ""))
        selected_input = str(_offline_cfg_path_get(base, "data.input_dir", ""))
        if (
            inspected_available is not None
            and inspected_input == selected_input
            and frames_per_position > int(inspected_available)
        ):
            raise ValueError(
                f"Frames to use per position must be between 1 and {int(inspected_available)} for this run"
            )
        if bool(_offline_cfg_path_get(base, "offline_background.enabled", False)) and not str(
            _offline_cfg_path_get(base, "offline_background.reference_dir", "")
        ).strip():
            raise ValueError("Empty-scene folder is required when offline reference background is enabled")
        if int(_offline_cfg_path_get(base, "scan.x_end")) < int(_offline_cfg_path_get(base, "scan.x_start")):
            raise ValueError("End position must be greater than or equal to start position")
        if float(_offline_cfg_path_get(base, "scan.x_pitch_m")) <= 0.0:
            raise ValueError("Il pitch X deve essere > 0")
        offline_map_bounds_from_yaml_dict(base, cfg)
        return base

    def _save_offline_tuning_config(*, allow_scan_finalize: bool = False) -> dict | None:
        nonlocal offline_tuning_cfg
        if _offline_tuning_locked_by_scan() and not allow_scan_finalize:
            _set_offline_tuning_status("Configuration is locked during SAR scan/finalization")
            return None
        try:
            saved_cfg = _collect_offline_tuning_config()
            missing_value = object()
            tuning_updates = {
                str(spec["path"]): _offline_cfg_path_get(saved_cfg, str(spec["path"]))
                for spec in OFFLINE_TUNING_FIELD_SPECS
                if _offline_cfg_path_get(saved_cfg, str(spec["path"]), missing_value) is not missing_value
            }
            pitch_mm = float(_offline_cfg_path_get(saved_cfg, "scan.x_pitch_m")) * 1000.0
            pitch_comment_value = f"{pitch_mm:.6f}".rstrip("0").rstrip(".")
            _update_existing_yaml_scalar_paths(
                offline_config_path,
                tuning_updates,
                inline_comments={"scan.x_pitch_m": f"{pitch_comment_value} mm pitch"},
            )
            offline_tuning_cfg = saved_cfg
            _refresh_offline_memory_estimate(saved_cfg)
            try:
                _update_scan_pitch_label(pitch_mm)
            except (TypeError, ValueError):
                _update_scan_pitch_label(error="invalid scan.x_pitch_m")
            _set_offline_tuning_status("Configuration saved to offline_config.yaml")
            return saved_cfg
        except Exception as exc:
            _set_offline_tuning_status(f"Configuration error: {exc}")
            return None

    def _on_save_offline_tuning(sender=None, app_data=None):
        _save_offline_tuning_config()

    def _on_reload_offline_tuning(sender=None, app_data=None):
        if _offline_tuning_locked_by_scan():
            _set_offline_tuning_status("Reload is locked during SAR scan/finalization")
            return False
        if not _read_offline_tuning_config():
            return False
        return _inspect_offline_run()

    def _on_apply_offline_tuning(
        sender=None,
        app_data=None,
        *,
        save_config: bool = True,
        allow_scan_finalize: bool = False,
    ) -> bool:
        nonlocal offline_runtime
        if _offline_tuning_locked_by_scan() and not allow_scan_finalize:
            _set_offline_tuning_status("Recalculation is locked during SAR scan/finalization")
            return False
        with offline_reload_lock:
            if bool(offline_reload_state["running"]):
                _set_offline_tuning_status("Offline recalculation already in progress")
                return False
        if save_config and not _inspect_offline_run():
            return False
        if save_config and _save_offline_tuning_config(allow_scan_finalize=allow_scan_finalize) is None:
            return False

        old_runtime = offline_runtime
        offline_runtime = None
        # Keep the previous runtime registered until the worker has stopped it:
        # an application shutdown in this small interval must still be able to
        # clean up its child processes.
        shutdown_resources["offline_runtime"] = old_runtime
        with offline_reload_lock:
            offline_reload_state.update({"running": True, "runtime": None, "info": None, "error": ""})
        _set_offline_tuning_status("Reloading data and recalculating offline...")
        _refresh_offline_memory_estimate(offline_tuning_cfg)
        _render_offline_summary(state="Offline load/calculation in progress")
        if dpg.does_item_exist(PROC_BTN_LOAD_OFFLINE):
            dpg.configure_item(PROC_BTN_LOAD_OFFLINE, label="LOADING OFFLINE...", enabled=False)
        for tag in (PROC_IN_XSTART, PROC_IN_XEND):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, enabled=False)

        def _reload_worker():
            new_runtime = None
            try:
                if old_runtime is not None:
                    old_runtime.stop()
                if shutdown_state["in_progress"] or shutdown_state["done"]:
                    with offline_reload_lock:
                        offline_reload_state.update({"running": False, "runtime": None, "info": None, "error": ""})
                    return
                new_runtime = _create_offline_runtime()
                # Register immediately so application shutdown can stop a
                # runtime that is still reading/transforming offline data.
                shutdown_resources["offline_runtime"] = new_runtime
                startup_timeout_s = float(
                    _offline_cfg_path_get(offline_tuning_cfg, "data.startup_timeout_s", 300)
                )
                info = new_runtime.start(timeout_s=startup_timeout_s)
                if shutdown_state["in_progress"] or shutdown_state["done"]:
                    new_runtime.stop()
                    with offline_reload_lock:
                        offline_reload_state.update({"running": False, "runtime": None, "info": None, "error": ""})
                    return
                with offline_reload_lock:
                    offline_reload_state.update({"running": False, "runtime": new_runtime, "info": info, "error": ""})
            except Exception as exc:
                if new_runtime is not None:
                    try:
                        new_runtime.stop()
                    except Exception:
                        pass
                    if shutdown_resources.get("offline_runtime") is new_runtime:
                        shutdown_resources["offline_runtime"] = None
                with offline_reload_lock:
                    offline_reload_state.update({"running": False, "runtime": None, "info": None, "error": str(exc)})

        try:
            threading.Thread(target=_reload_worker, name="offline-reload", daemon=True).start()
        except Exception as exc:
            offline_runtime = old_runtime
            shutdown_resources["offline_runtime"] = old_runtime
            with offline_reload_lock:
                offline_reload_state.update({"running": False, "runtime": None, "info": None, "error": ""})
            _set_offline_tuning_status(f"Offline recalculation start error: {exc}")
            return False
        return True

    def _on_load_offline(sender=None, app_data=None) -> bool:
        # The Processed Data button always uses the persisted YAML. Unsaved
        # tuning edits remain local until Save/Save and Recalculate is chosen.
        if not _read_offline_tuning_config():
            return False
        if not _inspect_offline_run():
            return False
        return _on_apply_offline_tuning(save_config=False)

    def _add_offline_tuning_widget(spec: dict, width: int = 300) -> None:
        tag = _offline_tune_tag(spec["path"])
        kind = str(spec["kind"])
        default = _offline_cfg_path_get(offline_tuning_cfg, spec["path"], spec.get("default"))
        if kind == "text":
            dpg.add_input_text(label=spec["label"], tag=tag, default_value=str(default), width=width)
        elif kind == "bool":
            dpg.add_checkbox(label=spec["label"], tag=tag, default_value=bool(default))
        elif kind == "combo":
            items = list(spec.get("items", []))
            default_s = str(default)
            if default_s not in items:
                default_s = str(spec.get("default", items[0]))
            dpg.add_combo(items, label=spec["label"], tag=tag, default_value=default_s, width=width)
        elif kind == "int":
            dpg.add_input_int(label=spec["label"], tag=tag, default_value=int(default), step=int(spec.get("step", 1)), width=width, min_value=int(spec.get("min", 0)), min_clamped=("min" in spec))
        else:
            dpg.add_input_float(label=spec["label"], tag=tag, default_value=float(default), step=float(spec.get("step", 0.1)), width=width, min_value=float(spec.get("min", 0.0)), max_value=float(spec.get("max", 0.0)), min_clamped=("min" in spec), max_clamped=("max" in spec), format=spec.get("format", "%.3f"))

    def _add_offline_tuning_section(section: str, *, mode: str | None = None) -> None:
        for spec in OFFLINE_TUNING_FIELD_SPECS:
            if spec["section"] == section and (mode is None or spec.get("mode") == mode):
                _add_offline_tuning_widget(spec)

    def _add_debug_metric_table(rows) -> None:
        """Build a compact label/value table for the fixed-width realtime sidebar."""
        with dpg.table(
            header_row=False,
            policy=dpg.mvTable_SizingStretchProp,
            borders_innerH=True,
            pad_outerX=False,
        ):
            dpg.add_table_column(init_width_or_weight=0.38, width_stretch=True)
            dpg.add_table_column(init_width_or_weight=0.62, width_stretch=True)
            for label, tag in rows:
                with dpg.table_row():
                    with dpg.table_cell():
                        dpg.add_text(label, color=(150, 155, 165))
                    with dpg.table_cell():
                        dpg.add_text("--", tag=tag, color=(225, 230, 235))

    def _cfg_path_get(path: str, default=None):
        node = cfg
        for part in str(path).split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node.get(part)
        return node

    def _tune_tag(path: str) -> str:
        return "tune__" + str(path).replace(".", "__")

    TUNING_THRESHOLD_MODES = ["relative", "absolute", "ca_cfar", "os_cfar"]
    TUNING_WINDOWS = ["none", "rectangular", "hanning", "hamming", "blackman"]
    TUNING_BG_MODES = ["ema", "running_mean", "window_mean", "frozen"]
    TUNING_SLOW_TIME_MODES = ["none", "mean_subtraction", "highpass", "doppler_fft"]
    TUNING_MOVING_SLOW_TIME_MODES = ["none", "mean_subtraction", "highpass"]
    TUNING_ANGLE_MODES = ["fft", "bartlett", "mvdr"]
    TUNING_AGGREGATION_MODES = ["frame_loop", "frame", "loop", "none"]
    TUNING_SPATIAL_FILTER_MODES = ["none", "gaussian_3x3"]

    TUNING_FIELD_SPECS = [
        {"section": "DSP", "group": "Windows", "path": "dsp.window_range", "label": "Range", "kind": "combo", "items": TUNING_WINDOWS, "default": "hanning"},
        {"section": "DSP", "group": "Windows", "path": "dsp.window_doppler", "label": "Doppler", "kind": "combo", "items": TUNING_WINDOWS, "default": "hanning"},
        {"section": "DSP", "group": "Windows", "path": "dsp.window_angle", "label": "Angle", "kind": "combo", "items": TUNING_WINDOWS, "default": "hanning"},
        {"section": "DSP", "group": "Range FFT", "path": "dsp.zero_after_range_fft_bins", "label": "Zero first bins", "kind": "int", "default": 0, "min": 0, "step": 1},
        {"section": "DSP", "group": "Moving heatmap", "path": "dsp.range_angle_moving.relative_power_floor_db", "label": "Rel floor dB", "kind": "float", "default": -22.0, "step": 1.0},
        {"section": "DSP", "group": "Moving heatmap", "path": "dsp.range_angle_moving.min_power_db", "label": "Min power dB", "kind": "float", "default": 0.0, "step": 1.0},
        {"section": "DSP", "group": "Moving heatmap", "path": "dsp.range_angle_moving.min_dominance_ratio", "label": "Dominance", "kind": "float", "default": 0.45, "min": 0.0, "max": 1.0, "step": 0.05},
        {"section": "DSP", "group": "Moving heatmap", "path": "dsp.range_angle_moving.velocity_dead_zone", "label": "Dead zone", "kind": "float", "default": 0.08, "min": 0.0, "max": 0.99, "step": 0.01},
        {"section": "DSP", "group": "Moving heatmap", "path": "dsp.range_angle_moving.min_opacity", "label": "Min opacity", "kind": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
        {"section": "DSP", "group": "Display BG", "path": "dsp.display_filters.background_subtraction.enabled", "label": "Enabled", "kind": "bool", "default": True},
        {"section": "DSP", "group": "Display BG", "path": "dsp.display_filters.background_subtraction.mode", "label": "Mode", "kind": "combo", "items": TUNING_BG_MODES, "default": "frozen"},
        {"section": "DSP", "group": "Display BG", "path": "dsp.display_filters.background_subtraction.alpha", "label": "Alpha", "kind": "float", "default": 0.02, "min": 0.0, "max": 1.0, "step": 0.01},
        {"section": "DSP", "group": "Display BG", "path": "dsp.display_filters.background_subtraction.init_frames", "label": "Init frames", "kind": "int", "default": 40, "min": 0, "step": 1},
        {"section": "DSP", "group": "Display BG", "path": "dsp.display_filters.background_subtraction.window_frames", "label": "Window frames", "kind": "int", "default": 40, "min": 1, "step": 1},
        {"section": "DSP", "group": "Display BG", "path": "dsp.display_filters.background_subtraction.clamp_positive_only", "label": "Clamp positive", "kind": "bool", "default": True},
        {"section": "DSP", "group": "Display slow time", "path": "dsp.display_filters.slow_time.enabled", "label": "Enabled", "kind": "bool", "default": False},
        {"section": "DSP", "group": "Display slow time", "path": "dsp.display_filters.slow_time.mode", "label": "Mode", "kind": "combo", "items": TUNING_SLOW_TIME_MODES, "default": "mean_subtraction"},
        {"section": "DSP", "group": "Display slow time", "path": "dsp.display_filters.slow_time.highpass_beta", "label": "Highpass beta", "kind": "float", "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01},
        {"section": "DSP", "group": "Display mean", "path": "dsp.display_filters.mean_after_range_fft.enabled", "label": "Mean after FFT", "kind": "bool", "default": False},
        {"section": "DSP", "group": "Display mean", "path": "dsp.display_filters.loop_average_after_background.enabled", "label": "Loop avg after BG", "kind": "bool", "default": False},
        {"section": "DSP", "group": "Angle", "path": "dsp.angle_processing.mode", "label": "Mode", "kind": "combo", "items": TUNING_ANGLE_MODES, "default": "bartlett"},
        {"section": "DSP", "group": "Angle", "path": "dsp.angle_processing.mvdr_diagonal_loading", "label": "MVDR loading", "kind": "float", "default": 0.02, "min": 0.0, "step": 0.005},
        {"section": "DSP", "group": "Angle", "path": "dsp.angle_processing.aggregation", "label": "Aggregation", "kind": "combo", "items": TUNING_AGGREGATION_MODES, "default": "frame_loop"},
        {"section": "DSP", "group": "Heatmap EMA", "path": "dsp.heatmap_ema.enabled", "label": "Enabled", "kind": "bool", "default": False},
        {"section": "DSP", "group": "Heatmap EMA", "path": "dsp.heatmap_ema.alpha", "label": "Alpha", "kind": "float", "default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01},
        {"section": "DSP", "group": "Spatial filter", "path": "dsp.heatmap_spatial_filter.enabled", "label": "Enabled", "kind": "bool", "default": False},
        {"section": "DSP", "group": "Spatial filter", "path": "dsp.heatmap_spatial_filter.mode", "label": "Mode", "kind": "combo", "items": TUNING_SPATIAL_FILTER_MODES, "default": "none"},
        {"section": "Tracking", "group": "Lifecycle", "path": "tracking.enabled", "label": "Enabled", "kind": "bool", "default": True},
        {"section": "Tracking", "group": "Lifecycle", "path": "tracking.max_tracks", "label": "Max tracks", "kind": "int", "default": 8, "min": 1, "step": 1},
        {"section": "Tracking", "group": "Lifecycle", "path": "tracking.min_hits_to_confirm", "label": "Min hits", "kind": "int", "default": 3, "min": 1, "step": 1},
        {"section": "Tracking", "group": "Lifecycle", "path": "tracking.max_missed_tentative", "label": "Miss tentative", "kind": "int", "default": 3, "min": 0, "step": 1},
        {"section": "Tracking", "group": "Lifecycle", "path": "tracking.max_missed_confirmed", "label": "Miss confirmed", "kind": "int", "default": 18, "min": 0, "step": 1},
        {"section": "Tracking", "group": "Lifecycle", "path": "tracking.max_track_age", "label": "Max age", "kind": "int", "default": 0, "min": 0, "step": 1},
        {"section": "Tracking", "group": "Association", "path": "tracking.gating_xy_m", "label": "Gate XY m", "kind": "float", "default": 1.0, "min": 0.0, "step": 0.05},
        {"section": "Tracking", "group": "Association", "path": "tracking.gating_doppler_mps", "label": "Gate Doppler", "kind": "float", "default": 0.5, "min": 0.0, "step": 0.05},
        {"section": "Tracking", "group": "Association", "path": "tracking.birth_min_separation_m", "label": "Birth separation", "kind": "float", "default": 0.35, "min": 0.0, "step": 0.05},
        {"section": "Tracking", "group": "Association", "path": "tracking.use_doppler_in_cost", "label": "Use Doppler cost", "kind": "bool", "default": True},
        {"section": "Tracking", "group": "Kalman", "path": "tracking.process_noise_pos", "label": "Noise pos", "kind": "float", "default": 0.3, "min": 0.0001, "step": 0.05},
        {"section": "Tracking", "group": "Kalman", "path": "tracking.process_noise_vel", "label": "Noise vel", "kind": "float", "default": 0.25, "min": 0.0001, "step": 0.05},
        {"section": "Tracking", "group": "Kalman", "path": "tracking.measurement_noise_xy", "label": "Meas noise XY", "kind": "float", "default": 0.10, "min": 0.0001, "step": 0.01},
        {"section": "Tracking", "group": "Motion state", "path": "tracking.moving_speed_threshold_mps", "label": "Moving speed", "kind": "float", "default": 0.22, "min": 0.0, "step": 0.02},
        {"section": "Tracking", "group": "Motion state", "path": "tracking.stopped_speed_threshold_mps", "label": "Stopped speed", "kind": "float", "default": 0.10, "min": 0.0, "step": 0.02},
        {"section": "Tracking", "group": "Motion state", "path": "tracking.doppler_moving_threshold_mps", "label": "Doppler moving", "kind": "float", "default": 0.15, "min": 0.0, "step": 0.02},
        {"section": "Tracking", "group": "Motion state", "path": "tracking.motion_confirm_frames_moving", "label": "Confirm moving", "kind": "int", "default": 2, "min": 1, "step": 1},
        {"section": "Tracking", "group": "Motion state", "path": "tracking.motion_confirm_frames_stopped", "label": "Confirm stopped", "kind": "int", "default": 3, "min": 1, "step": 1},
        {"section": "Tracking", "group": "Stopped memory", "path": "tracking.stopped_memory_s", "label": "Memory s", "kind": "float", "default": 5.0, "min": 0.0, "step": 0.5},
        {"section": "Tracking", "group": "Stopped memory", "path": "tracking.stopped_resume_gate_m", "label": "Resume gate", "kind": "float", "default": 0.8, "min": 0.0, "step": 0.05},
        {"section": "Tracking", "group": "Stopped memory", "path": "tracking.stop_position_alpha", "label": "Stop alpha", "kind": "float", "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05},
        {"section": "Detection static", "group": "Core", "path": "detection_static.enabled", "label": "Enabled", "kind": "bool", "default": True},
        {"section": "Detection static", "group": "Core", "path": "detection_static.threshold_mode", "label": "Mode", "kind": "combo", "items": TUNING_THRESHOLD_MODES, "default": "relative"},
        {"section": "Detection static", "group": "Core", "path": "detection_static.threshold_db", "label": "Threshold dB", "kind": "float", "default": -10.0, "step": 1.0},
        {"section": "Detection static", "group": "Core", "path": "detection_static.min_power_db", "label": "Min power dB", "kind": "float", "default": 6.0, "step": 1.0},
        {"section": "Detection static", "group": "Core", "path": "detection_static.max_detections", "label": "Max detections", "kind": "int", "default": 12, "min": 1, "step": 1},
        {"section": "Detection static", "group": "Local max", "path": "detection_static.localmax_range_bins", "label": "Range bins", "kind": "int", "default": 1, "min": 0, "step": 1},
        {"section": "Detection static", "group": "Local max", "path": "detection_static.localmax_angle_bins", "label": "Angle bins", "kind": "int", "default": 1, "min": 0, "step": 1},
        {"section": "Detection static", "group": "CFAR", "path": "detection_static.cfar_train_range_bins", "label": "Train range", "kind": "int", "default": 8, "min": 0, "step": 1},
        {"section": "Detection static", "group": "CFAR", "path": "detection_static.cfar_guard_range_bins", "label": "Guard range", "kind": "int", "default": 2, "min": 0, "step": 1},
        {"section": "Detection static", "group": "CFAR", "path": "detection_static.cfar_train_col_bins", "label": "Train angle", "kind": "int", "default": 8, "min": 0, "step": 1},
        {"section": "Detection static", "group": "CFAR", "path": "detection_static.cfar_guard_col_bins", "label": "Guard angle", "kind": "int", "default": 2, "min": 0, "step": 1},
        {"section": "Detection static", "group": "CFAR", "path": "detection_static.cfar_threshold_db", "label": "Offset dB", "kind": "float", "default": 10.0, "step": 1.0},
        {"section": "Detection static", "group": "CFAR", "path": "detection_static.os_cfar_rank", "label": "OS rank", "kind": "int", "default": 0, "min": 0, "step": 1},
        {"section": "Detection moving", "group": "Core", "path": "detection_moving.enabled", "label": "Enabled", "kind": "bool", "default": True},
        {"section": "Detection moving", "group": "Core", "path": "detection_moving.threshold_mode", "label": "Mode", "kind": "combo", "items": TUNING_THRESHOLD_MODES, "default": "ca_cfar"},
        {"section": "Detection moving", "group": "Core", "path": "detection_moving.threshold_db", "label": "Threshold dB", "kind": "float", "default": -7.0, "step": 1.0},
        {"section": "Detection moving", "group": "Core", "path": "detection_moving.min_power_db", "label": "Min power dB", "kind": "float", "default": 6.0, "step": 1.0},
        {"section": "Detection moving", "group": "Core", "path": "detection_moving.max_detections", "label": "Max detections", "kind": "int", "default": 12, "min": 1, "step": 1},
        {"section": "Detection moving", "group": "Local max", "path": "detection_moving.localmax_range_bins", "label": "Range bins", "kind": "int", "default": 2, "min": 0, "step": 1},
        {"section": "Detection moving", "group": "Local max", "path": "detection_moving.localmax_doppler_bins", "label": "Doppler bins", "kind": "int", "default": 2, "min": 0, "step": 1},
        {"section": "Detection moving", "group": "Local max", "path": "detection_moving.zero_doppler_exclusion_bins", "label": "Zero excl bins", "kind": "int", "default": 1, "min": 0, "step": 1},
        {"section": "Detection moving", "group": "CFAR", "path": "detection_moving.cfar_train_range_bins", "label": "Train range", "kind": "int", "default": 8, "min": 0, "step": 1},
        {"section": "Detection moving", "group": "CFAR", "path": "detection_moving.cfar_guard_range_bins", "label": "Guard range", "kind": "int", "default": 2, "min": 0, "step": 1},
        {"section": "Detection moving", "group": "CFAR", "path": "detection_moving.cfar_train_col_bins", "label": "Train doppler", "kind": "int", "default": 4, "min": 0, "step": 1},
        {"section": "Detection moving", "group": "CFAR", "path": "detection_moving.cfar_guard_col_bins", "label": "Guard doppler", "kind": "int", "default": 1, "min": 0, "step": 1},
        {"section": "Detection moving", "group": "CFAR", "path": "detection_moving.cfar_threshold_db", "label": "Offset dB", "kind": "float", "default": 10.0, "step": 1.0},
        {"section": "Detection moving", "group": "CFAR", "path": "detection_moving.os_cfar_rank", "label": "OS rank", "kind": "int", "default": 0, "min": 0, "step": 1},
        {"section": "Detection moving", "group": "Pre-Doppler filters", "path": "dsp.detection_moving_pre_doppler_filters.slow_time.enabled", "label": "Slow time", "kind": "bool", "default": True},
        {"section": "Detection moving", "group": "Pre-Doppler filters", "path": "dsp.detection_moving_pre_doppler_filters.slow_time.mode", "label": "Slow mode", "kind": "combo", "items": TUNING_MOVING_SLOW_TIME_MODES, "default": "mean_subtraction"},
        {"section": "Detection moving", "group": "Pre-Doppler filters", "path": "dsp.detection_moving_pre_doppler_filters.slow_time.highpass_beta", "label": "Highpass beta", "kind": "float", "default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01},
        {"section": "Fusion", "group": "Core", "path": "fusion.enabled", "label": "Enabled", "kind": "bool", "default": True},
        {"section": "Fusion", "group": "Merge gates", "path": "fusion.merge_xy_m", "label": "Merge XY m", "kind": "float", "default": 0.50, "min": 0.0, "step": 0.05},
        {"section": "Fusion", "group": "Merge gates", "path": "fusion.merge_range_m", "label": "Merge range m", "kind": "float", "default": 0.40, "min": 0.0, "step": 0.05},
        {"section": "Fusion", "group": "Merge gates", "path": "fusion.merge_angle_deg", "label": "Merge angle deg", "kind": "float", "default": 8.0, "min": 0.0, "step": 0.5},
        {"section": "Fusion", "group": "Policy", "path": "fusion.prefer_moving_when_doppler_valid", "label": "Prefer moving", "kind": "bool", "default": False},
    ]

    def _tune_set_patch_value(out: dict, path: str, value):
        node = out
        parts = str(path).split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def _tune_value_from_widget(spec: dict):
        value = dpg.get_value(_tune_tag(spec["path"]))
        kind = str(spec.get("kind", "float"))
        if kind == "bool":
            return bool(value)
        if kind == "int":
            try:
                value = int(value)
            except (TypeError, ValueError):
                value = int(spec.get("default", 0))
            if "min" in spec:
                value = max(int(spec["min"]), value)
            if "max" in spec:
                value = min(int(spec["max"]), value)
            return value
        if kind == "float":
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = float(spec.get("default", 0.0))
            if not np.isfinite(value):
                value = float(spec.get("default", 0.0))
            if "min" in spec:
                value = max(float(spec["min"]), value)
            if "max" in spec:
                value = min(float(spec["max"]), value)
            return float(value)
        return str(value)

    def _apply_tuning_params(sender=None, app_data=None):
        patch = {}
        for spec in TUNING_FIELD_SPECS:
            tag = _tune_tag(spec["path"])
            if not dpg.does_item_exist(tag):
                continue
            try:
                _tune_set_patch_value(patch, spec["path"], _tune_value_from_widget(spec))
            except Exception:
                continue
        try:
            dsp_cmd_q.put_nowait({"type": "update_runtime_config", "cfg_patch": patch, "reset_runtime_state": True})
            if dpg.does_item_exist(TXT_TUNING_STATUS_TAG):
                dpg.set_value(TXT_TUNING_STATUS_TAG, "Runtime update and DSP soft reset sent")
        except Exception as e:
            if dpg.does_item_exist(TXT_TUNING_STATUS_TAG):
                dpg.set_value(TXT_TUNING_STATUS_TAG, f"ERR tuning: {e}")

    def _add_tuning_widget(spec: dict, width: int = 260):
        path = str(spec["path"])
        tag = _tune_tag(path)
        label = str(spec.get("label", path))
        kind = str(spec.get("kind", "float"))
        default = _cfg_path_get(path, spec.get("default"))
        if kind == "bool":
            dpg.add_checkbox(label=label, tag=tag, default_value=bool(default), callback=_apply_tuning_params)
            return
        if kind == "combo":
            items = list(spec.get("items", []))
            default_s = str(default)
            if items and default_s not in items:
                default_s = str(spec.get("default", items[0]))
            dpg.add_combo(items, label=label, tag=tag, default_value=default_s, width=width, callback=_apply_tuning_params)
            return
        if kind == "int":
            dpg.add_input_int(
                label=label,
                tag=tag,
                default_value=int(default),
                step=int(spec.get("step", 1)),
                width=width,
                min_value=int(spec["min"]) if "min" in spec else 0,
                max_value=int(spec["max"]) if "max" in spec else 0,
                min_clamped=("min" in spec),
                max_clamped=("max" in spec),
                callback=_apply_tuning_params,
                on_enter=False,
            )
            return
        dpg.add_input_float(
            label=label,
            tag=tag,
            default_value=float(default),
            step=float(spec.get("step", 0.1)),
            width=width,
            min_value=float(spec["min"]) if "min" in spec else 0.0,
            max_value=float(spec["max"]) if "max" in spec else 0.0,
            min_clamped=("min" in spec),
            max_clamped=("max" in spec),
            callback=_apply_tuning_params,
            on_enter=False,
        )

    def _add_tuning_section(section: str, width: int = 260):
        current_group = None
        for spec in TUNING_FIELD_SPECS:
            if spec.get("section") != section:
                continue
            group = str(spec.get("group", ""))
            if group != current_group:
                if current_group is not None:
                    dpg.add_spacer(height=8)
                dpg.add_text(group.upper(), color=(255, 200, 0))
                dpg.add_separator()
                current_group = group
            _add_tuning_widget(spec, width=width)

    _read_offline_tuning_config()

    track_annotation_tags: list[str] = []
    supports_plot_annotation = bool(hasattr(dpg, "add_plot_annotation"))

    # 6) Build UI (ONLY TABLE LAYOUT)
    with dpg.window(tag=TAG_MAIN_WINDOW):
        with dpg.tab_bar(tag=TAG_MAIN_TABBAR, callback=_on_main_tab_changed):
            dpg.add_tab(label="Real Time", tag=TAB_REALTIME_TAG)
            dpg.add_tab(label="Processed Data", tag=TAB_PROCESSED_TAG)
            dpg.add_tab(label="Tuning DSP", tag=TAB_TUNING_TAG)
            dpg.add_tab(label="Tuning Offline", tag=TAB_OFFLINE_TUNING_TAG)

        with dpg.table(header_row=False, resizable=True, policy=dpg.mvTable_SizingFixedFit, parent=TAB_REALTIME_TAG):

            # Colonne: sidebar (fissa), plot (stretch), colorbar (fissa), range FFT (fissa)
            dpg.add_table_column(init_width_or_weight=340, width_fixed=True)                                      # sidebar
            dpg.add_table_column(init_width_or_weight=1.0, width_stretch=True, width_fixed=False)                 # plot stretch
            dpg.add_table_column(init_width_or_weight=100, width_fixed=True)                                      # colorbar
            dpg.add_table_column(init_width_or_weight=500, width_fixed=True)                                      # range FFT


            with dpg.table_row():
                CTRL_W = 240
                # --- SIDEBAR ---
                with dpg.table_cell():
                    with dpg.child_window(tag=TAG_SIDEBAR, width=-1, height=-1, border=True):
                        dpg.add_text("DISPLAY PARAMETERS", color=(255, 200, 0))
                        dpg.add_separator()
                        dpg.add_spacer(width=20)
                        dpg.add_button(
                            label=_heatmap_mode_label(vis_heatmap_mode),
                            tag=BTN_HEATMAP_MODE_TAG,
                            callback=_toggle_heatmap_mode,
                            width=-1,
                            height=32,
                        )
                        dpg.add_spacer(height=6)
                        dpg.add_input_float(label=("Vmin (m/s)" if vis_heatmap_mode == 1 else "Vmin (dB)"), tag=IN_VMIN, default_value=vis_vmin, step=1.0, width=CTRL_W,callback=_apply_params, on_enter=False)
                        dpg.add_input_float(label=("Vmax (m/s)" if vis_heatmap_mode == 1 else "Vmax (dB)"), tag=IN_VMAX, default_value=vis_vmax, step=1.0, width=CTRL_W,callback=_apply_params, on_enter=False)
                        dpg.add_input_float(label="Rmax (m)", tag=IN_RMAX, default_value=vis_rmax, step=0.5, width=CTRL_W,
                                          min_value=0.0, max_value=RMAX_HARD_MAX, min_clamped=True, max_clamped=True,
                                          callback=_apply_params, on_enter=False)
                        dpg.add_input_float(label="Xmax (m)", tag=IN_XMAX, default_value=vis_xmax, step=0.5, width=CTRL_W,
                                          min_value=0.0, max_value=HEATMAP_XMAX_HARD_MAX, min_clamped=True, max_clamped=True,
                                          callback=_apply_params, on_enter=False)
                        dpg.add_spacer(height=8)
                        dpg.add_button(
                            label=_heatmap_norm_label(vis_heatmap_mode, _get_norm_enabled()),
                            tag=BTN_NORM_TAG,
                            callback=_toggle_norm,
                            width=-1,
                            height=32,
                            enabled=(vis_heatmap_mode == 0),
                        )
                        dpg.add_spacer(height=14)
                        dpg.add_text("MMWAVE STUDIO CONTROL", color=(255, 200, 0))
                        dpg.add_separator()
                        dpg.add_button(
                            label=_mmwave_connect_label(False),
                            tag=BTN_MMWAVE_CONNECT_TAG,
                            callback=_on_mmwave_connect_toggle,
                            width=-1,
                            height=36,
                        )
                        dpg.add_button(
                            label=_mmwave_stream_label(False),
                            tag=BTN_MMWAVE_STREAM_TAG,
                            callback=_on_mmwave_stream_toggle,
                            width=-1,
                            height=36,
                        )
                        dpg.add_text("mmWave Studio bridge idle", tag=TXT_MMWAVE_STATUS_TAG, wrap=-1)
                        dpg.add_text("", tag=TXT_MMWAVE_LINKS_TAG, wrap=-1)
                        dpg.add_spacer(height=14)
                        dpg.add_text("SAR ACQUISITION", color=(255, 200, 0))
                        dpg.add_separator()
                        dpg.add_text(f"Current run: {out_dir.name}", tag=TXT_RUN_DIR_TAG, wrap=CTRL_W, color=(120, 210, 255))
                        dpg.add_text("Ready for a manual capture in the current run.", tag=TXT_CAPTURE_STATUS_TAG, wrap=CTRL_W)
                        dpg.add_text("Next ID: 1", tag=TXT_POS_TAG, color=(190, 190, 190))
                        dpg.add_button(
                            label="CAPTURE POSITION",
                            tag=BTN_CAPTURE_TAG,
                            callback=_on_capture,
                            width=CTRL_W,
                            height=38,
                        )
                        dpg.add_button(
                            label="NEW SESSION",
                            tag=BTN_NEW_SESSION_TAG,
                            callback=_on_new_sar_session,
                            width=CTRL_W,
                            height=32,
                        )
                        dpg.add_text(
                            f"Each position saves {int(FRAMES_PER_POSITION)} radar frames.",
                            color=(170, 170, 170),
                            wrap=CTRL_W,
                        )
                        dpg.add_spacer(height=12)
                        dpg.add_text("CARRIAGE AND SCAN", color=(255, 200, 0))
                        dpg.add_separator()
                        dpg.add_text("1. Carriage positioning", color=(210, 210, 210))
                        if stepper_controller is None:
                            dpg.add_text(
                                f"Phidget unavailable: {stepper_error or 'backend not loaded'}",
                                tag=TXT_MOTOR_STATUS_TAG,
                                wrap=CTRL_W,
                                color=(255, 150, 120),
                            )
                        else:
                            dpg.add_text("Carriage: disconnected", tag=TXT_MOTOR_STATUS_TAG, wrap=CTRL_W)
                        dpg.add_text("Position: --", tag=TXT_MOTOR_POSITION_TAG, color=(100, 220, 255))
                        dpg.add_button(label="CONNECT", callback=_on_motor_connect, width=CTRL_W)
                        dpg.add_button(label="DISCONNECT", callback=_on_motor_disconnect, width=CTRL_W)
                        dpg.add_button(label="HOME", callback=_on_motor_home, width=CTRL_W)
                        dpg.add_text("Manual step [mm]", color=(210, 210, 210))
                        dpg.add_input_float(
                            label="",
                            tag=IN_MOTOR_JOG_TAG,
                            default_value=float(stepper_controller.config.cycle.step_mm) if stepper_controller is not None else 1.0,
                            min_value=0.001,
                            min_clamped=True,
                            width=CTRL_W,
                        )
                        dpg.add_button(label="MOVE -", callback=lambda: _on_motor_jog(-1), width=CTRL_W)
                        dpg.add_button(label="MOVE +", callback=lambda: _on_motor_jog(+1), width=CTRL_W)
                        dpg.add_button(label="STOP / CANCEL", callback=_on_motor_stop, width=CTRL_W, height=30)
                        dpg.add_text(
                            "Before scanning, run HOME and move the carriage to the start position.",
                            wrap=CTRL_W,
                            color=(190, 190, 190),
                        )
                        dpg.add_separator()
                        dpg.add_text("2. Automatic scan", color=(210, 210, 210))
                        pitch_label = (
                            f"Pitch da offline_config: {scan_pitch_mm_default:.6f} mm"
                            if np.isfinite(scan_pitch_mm_default)
                            else f"offline_config error: {scan_config_error}"
                        )
                        dpg.add_text(pitch_label, tag=TXT_SCAN_PITCH_TAG, wrap=CTRL_W)
                        dpg.add_text("Number of positions", color=(210, 210, 210))
                        dpg.add_input_int(
                            label="",
                            tag=IN_SCAN_POSITIONS_TAG,
                            default_value=int(scan_positions_default),
                            min_value=1,
                            min_clamped=True,
                            width=CTRL_W,
                        )
                        dpg.add_button(label="START SCAN", tag=BTN_SCAN_START_TAG, callback=_on_start_sar_scan, width=CTRL_W, height=38)
                        dpg.add_button(label="CANCEL SCAN", tag=BTN_SCAN_CANCEL_TAG, callback=_on_cancel_sar_scan, width=CTRL_W, height=30)
                        dpg.add_text("Scan status: ready", tag=TXT_SCAN_STATUS_TAG, wrap=CTRL_W, color=(190, 220, 190))
                        with dpg.collapsing_header(label="Carriage diagnostics", default_open=False):
                            dpg.add_text("No motor events.", tag=TXT_MOTOR_LOG_TAG, wrap=CTRL_W, color=(175, 175, 175))
                        _refresh_mmwave_controls()

                        if DEBUG_STATS:
                            dpg.add_spacer(height=14)
                            dpg.add_text("REALTIME PIPELINE DEBUG", color=(255, 200, 0))
                            dpg.add_separator()
                            dpg.add_text("CONFIGURATION", color=(175, 175, 175))
                            dpg.add_text(
                                (
                                    f"Grid RT/OFF: {gui_w}x{gui_h} / {offline_gui_w}x{offline_gui_h}\n"
                                    f"FFT range/angle: {NFFT_RANGE} / {NFFT_ANGLE}"
                                ),
                                tag=TXT_PIPELINE_CONFIG_TAG,
                                wrap=-1,
                                color=(175, 175, 175),
                            )
                            dpg.add_spacer(height=8)
                            dpg.add_text("STREAM", color=(105, 180, 225))
                            _add_debug_metric_table(
                                [
                                    ("UDP gaps / RX", DEBUG_STAT_VALUE_TAGS["udp_rx"]),
                                    ("Frames", DEBUG_STAT_VALUE_TAGS["frames"]),
                                    ("Ring ready", DEBUG_STAT_VALUE_TAGS["ring"]),
                                    ("Ring drops", DEBUG_STAT_VALUE_TAGS["drops"]),
                                ]
                            )
                            dpg.add_spacer(height=6)
                            dpg.add_text("PROCESSING", color=(115, 205, 165))
                            _add_debug_metric_table(
                                [
                                    ("DSP frame", DEBUG_STAT_VALUE_TAGS["dsp_frame"]),
                                    ("DSP stale / image", DEBUG_STAT_VALUE_TAGS["dsp_stale"]),
                                    ("CPU DSP / log", DEBUG_STAT_VALUE_TAGS["cpu"]),
                                    ("Logger write", DEBUG_STAT_VALUE_TAGS["logger"]),
                                    ("RX stalls", DEBUG_STAT_VALUE_TAGS["stalls"]),
                                    ("RX resyncs", DEBUG_STAT_VALUE_TAGS["resyncs"]),
                                ]
                            )
                            dpg.add_spacer(height=8)
                            dpg.add_text("DISPLAY DIAGNOSTICS", color=(175, 175, 175))
                            dpg.add_text("Waiting for display data...", tag=TXT_DISPLAY_DIAG_TAG, wrap=-1)

                # --- PLOT ---
                with dpg.table_cell():
                    with dpg.plot(tag=HEAT_PLOT_TAG, width=-1, height=-1, equal_aspects=True):
                        dpg.add_plot_axis(dpg.mvXAxis, label="X (m)", tag=XAXIS_TAG)
                        dpg.add_plot_axis(dpg.mvYAxis, label="Y (m)", tag=YAXIS_TAG)

                        # bounds in metri fissi (full FOV): zoom/crop gestiti dagli assi
                        dpg.add_image_series(
                            TEX_TAG,
                            bounds_min=(float(rt_applied_meta_current.x_min_m), float(rt_applied_meta_current.y_min_m)),
                            bounds_max=(float(rt_applied_meta_current.x_max_m), float(rt_applied_meta_current.y_max_m)),
                            tag=IMG_SERIES_TAG,
                            parent=YAXIS_TAG,
                        )
                        dpg.add_scatter_series([], [], label="Tracks confirmed", tag=TRACK_SCATTER_CONF_TAG, parent=YAXIS_TAG)
                        dpg.add_scatter_series([], [], label="Tracks tentative", tag=TRACK_SCATTER_UNCONF_TAG, parent=YAXIS_TAG)
                        dpg.add_scatter_series([], [], label="Tracks moving", tag=TRACK_SCATTER_MOVING_TAG, parent=YAXIS_TAG)
                        dpg.add_scatter_series([], [], label="Tracks stopped", tag=TRACK_SCATTER_STOPPED_TAG, parent=YAXIS_TAG)
                        dpg.add_scatter_series([], [], label="Tracks unknown", tag=TRACK_SCATTER_UNKNOWN_TAG, parent=YAXIS_TAG)
                        dpg.add_scatter_series([], [], label="Stop point", tag=TRACK_STOP_MARKER_TAG, parent=YAXIS_TAG)
                        dpg.add_line_series([], [], label="Track velocity", tag=TRACK_VEL_SERIES_TAG, parent=YAXIS_TAG)
                        guide_neg_x, guide_y = _guide_line_points(-1.0, RANGE_MAX_DISPLAY)
                        guide_pos_x, _ = _guide_line_points(+1.0, RANGE_MAX_DISPLAY)
                        dpg.add_line_series(guide_neg_x, guide_y, label="-20 deg", tag=GUIDE_NEG20_TAG, parent=YAXIS_TAG)
                        dpg.add_line_series(guide_pos_x, guide_y, label="+20 deg", tag=GUIDE_POS20_TAG, parent=YAXIS_TAG)
                        dpg.bind_item_theme(TRACK_SCATTER_CONF_TAG, track_conf_theme)
                        dpg.bind_item_theme(TRACK_SCATTER_UNCONF_TAG, track_unconf_theme)
                        dpg.bind_item_theme(TRACK_SCATTER_MOVING_TAG, track_moving_theme)
                        dpg.bind_item_theme(TRACK_SCATTER_STOPPED_TAG, track_stopped_theme)
                        dpg.bind_item_theme(TRACK_SCATTER_UNKNOWN_TAG, track_unknown_theme)
                        dpg.bind_item_theme(TRACK_STOP_MARKER_TAG, track_stop_marker_theme)
                        dpg.bind_item_theme(TRACK_VEL_SERIES_TAG, track_vel_theme)
                        dpg.bind_item_theme(GUIDE_NEG20_TAG, guide_line_theme)
                        dpg.bind_item_theme(GUIDE_POS20_TAG, guide_line_theme)
                        if supports_plot_annotation:
                            add_plot_annotation_fn = getattr(dpg, "add_plot_annotation", None)
                            if callable(add_plot_annotation_fn):
                                for ann_i in range(track_max_shared):
                                    ann_tag = f"{TRACK_ANN_PREFIX}{ann_i}"
                                    try:
                                        add_plot_annotation_fn(
                                            label="",
                                            default_value=(0.0, 0.0),
                                            offset=(6, 6),
                                            color=(255, 255, 255, 220),
                                            clamped=False,
                                            parent=YAXIS_TAG,
                                            tag=ann_tag,
                                        )
                                        dpg.configure_item(ann_tag, show=False)
                                        track_annotation_tags.append(ann_tag)
                                    except Exception:
                                        track_annotation_tags.clear()
                                        supports_plot_annotation = False
                                        break

                # --- COLORBAR ---
                with dpg.table_cell():
                    with dpg.child_window(tag=TAG_CBAR_COL, width=-1, height=-1, border=True):
                        # width=-1 cosÃ¬ si adatta alla colonna; height=-1 cosÃ¬ prende tutta l'altezza
                        dpg.add_colormap_scale(
                            tag=CMAP_SCALE_TAG,
                            min_scale=vis_vmin,
                            max_scale=vis_vmax,
                            format=(CMAP_VELOCITY_NUM_FMT if vis_heatmap_mode == 1 else CMAP_NUM_FMT),
                            width=-1,
                            height=-1,
                            colormap=(
                                CMAP_VELOCITY_TAG
                                if vis_heatmap_mode == 1 and dpg.does_item_exist(CMAP_VELOCITY_TAG)
                                else dpg.mvPlotColormap_Jet
                            ),
                        )
                        if font_mono:
                            dpg.bind_item_font(CMAP_SCALE_TAG, font_mono)

                # --- RANGE FFT PROFILES ---
                with dpg.table_cell():
                    with dpg.child_window(tag=TAG_RANGEFFT_COL, width=-1, height=-1, border=True):
                        dpg.add_text("FFT DIAGNOSTICS", color=(255, 200, 0))
                        dpg.add_separator()
                        with dpg.tab_bar(tag=FFT_DIAG_TABBAR_TAG):
                            with dpg.tab(label="Range"):
                                with dpg.plot(
                                    tag=RANGEFFT_PLOT_TAG,
                                    label=_fft_plot_title(vis_fft_view_full),
                                    width=-1,
                                    height=500,
                                    no_menus=True,
                                    no_box_select=True,
                                    no_mouse_pos=True,
                                ):
                                    dpg.add_plot_axis(
                                        dpg.mvXAxis,
                                        label="Range (m)",
                                        tag=RANGEFFT_XAXIS_TAG,
                                        auto_fit=False,
                                        lock_min=True,
                                        lock_max=True,
                                    )
                                    dpg.add_plot_axis(
                                        dpg.mvYAxis,
                                        label=_fft_axis_label(vis_fft_mode_db),
                                        tag=RANGEFFT_YAXIS_TAG,
                                        auto_fit=False,
                                        lock_min=True,
                                        lock_max=True,
                                    )
                                    dpg.add_plot_legend(location=dpg.mvPlot_Location_SouthEast, outside=True, horizontal=True)
                                    for ant_i in range(RANGE_PROFILE_COUNT):
                                        line_tag = RANGEFFT_LINE_TAGS[ant_i]
                                        dpg.add_line_series(
                                            [0.0],
                                            [0.0],
                                            label=f"A{ant_i + 1}",
                                            tag=line_tag,
                                            parent=RANGEFFT_YAXIS_TAG,
                                        )
                                        if ant_i < len(rangefft_line_themes):
                                            dpg.bind_item_theme(line_tag, rangefft_line_themes[ant_i])
                                _, init_fft_rmax_m = _fft_visible_range_from_view(vis_rmax, vis_fft_view_full)
                                vis_fft_xmin, vis_fft_xmax, _ = _clamp_fft_x_window(vis_fft_xmin, vis_fft_xmax, vis_rmax, vis_fft_view_full)
                                dpg.set_axis_limits(RANGEFFT_XAXIS_TAG, float(vis_fft_xmin), float(vis_fft_xmax))
                                _set_fft_x_ticks_window(vis_fft_xmin, vis_fft_xmax)
                                dpg.set_axis_limits(RANGEFFT_YAXIS_TAG, float(vis_fft_vmin), float(vis_fft_vmax))
                                dpg.add_separator()
                                dpg.add_button(
                                    label=_fft_view_label(fft_view_full),
                                    tag=BTN_FFT_VIEW_TAG,
                                    callback=_toggle_fft_view,
                                    width=-1,
                                    height=30,
                                )
                                dpg.add_button(
                                    label=_fft_mode_label(fft_mode_db),
                                    tag=BTN_FFT_MODE_TAG,
                                    callback=_toggle_fft_mode,
                                    width=-1,
                                    height=30,
                                )
                                dpg.add_input_double(
                                    label="FFT Xmin (m)",
                                    tag=IN_FFT_XMIN,
                                    default_value=float(vis_fft_xmin),
                                    format="%.2f",
                                    step=0.5,
                                    step_fast=2.0,
                                    width=220,
                                    callback=_apply_params,
                                    on_enter=False,
                                )
                                dpg.add_input_double(
                                    label="FFT Xmax (m)",
                                    tag=IN_FFT_XMAX,
                                    default_value=float(vis_fft_xmax),
                                    format="%.2f",
                                    step=0.5,
                                    step_fast=2.0,
                                    width=220,
                                    callback=_apply_params,
                                    on_enter=False,
                                )
                                dpg.add_input_double(
                                    label="FFT Ymin (dB)",
                                    tag=IN_FFT_VMIN,
                                    default_value=float(vis_fft_vmin),
                                    format="%.0f",
                                    step=1.0,
                                    step_fast=10.0,
                                    width=220,
                                    callback=_apply_params,
                                    on_enter=False,
                                )
                                dpg.add_input_double(
                                    label="FFT Ymax (dB)",
                                    tag=IN_FFT_VMAX,
                                    default_value=float(vis_fft_vmax),
                                    format="%.0f",
                                    step=1.0,
                                    step_fast=10.0,
                                    width=220,
                                    callback=_apply_params,
                                    on_enter=False,
                                )

                            with dpg.tab(label="Angle"):
                                with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp):
                                    dpg.add_table_column(init_width_or_weight=1.0, width_stretch=True)
                                    dpg.add_table_column(init_width_or_weight=70, width_fixed=True)
                                    with dpg.table_row():
                                        with dpg.table_cell():
                                            with dpg.plot(
                                                tag=ANGLEFFT_HEAT_PLOT_TAG,
                                                label="Range-Angle FFT",
                                                width=-1,
                                                height=500,
                                                no_menus=True,
                                            ):
                                                dpg.add_plot_axis(dpg.mvXAxis, label="Angle (deg)", tag=ANGLEFFT_XAXIS_TAG)
                                                dpg.add_plot_axis(dpg.mvYAxis, label="Range (m)", tag=ANGLEFFT_YAXIS_TAG)
                                                dpg.add_image_series(
                                                    ANGLEFFT_TEX_TAG,
                                                    bounds_min=(float(angle_axis_min), 0.0),
                                                    bounds_max=(float(angle_axis_max), float(fft_plot_h) * float(dr_plot)),
                                                    tag=ANGLEFFT_IMG_SERIES_TAG,
                                                    parent=ANGLEFFT_YAXIS_TAG,
                                                )
                                            with dpg.plot(
                                                tag=ANGLEFFT_PROFILE_PLOT_TAG,
                                                label="Angle FFT @ range bin",
                                                width=-1,
                                                height=500,
                                                no_menus=True,
                                                show=False,
                                            ):
                                                dpg.add_plot_axis(dpg.mvXAxis, label="Angle (deg)", tag=ANGLEFFT_PROFILE_XAXIS_TAG)
                                                dpg.add_plot_axis(dpg.mvYAxis, label="dB", tag=ANGLEFFT_PROFILE_YAXIS_TAG)
                                                dpg.add_line_series([], [], label="Angle", tag=ANGLEFFT_PROFILE_LINE_TAG, parent=ANGLEFFT_PROFILE_YAXIS_TAG)
                                        with dpg.table_cell():
                                            dpg.add_colormap_scale(
                                                tag=ANGLEFFT_CMAP_SCALE_TAG,
                                                min_scale=float(vis_angle_diag_vmin),
                                                max_scale=float(vis_angle_diag_vmax),
                                                format=CMAP_NUM_FMT,
                                                width=-1,
                                                height=500,
                                                colormap=dpg.mvPlotColormap_Jet,
                                            )
                                with dpg.group():
                                    dpg.add_checkbox(
                                        label="Single range bin",
                                        tag=CHK_ANGLE_SINGLE_BIN,
                                        default_value=False,
                                        callback=_on_angle_single_bin,
                                    )
                                    dpg.add_checkbox(
                                        label="Norm to peak",
                                        tag=CHK_ANGLE_NORM,
                                        default_value=bool(angle_diag_norm_to_peak),
                                    )
                                with dpg.group():
                                    dpg.add_input_int(
                                        label="Range bin",
                                        tag=IN_ANGLE_BIN,
                                        default_value=0,
                                        min_value=0,
                                        max_value=max(0, int(fft_plot_h) - 1),
                                        min_clamped=True,
                                        max_clamped=True,
                                        width=220,
                                    )
                                    dpg.add_input_double(
                                        label="Vmin dB",
                                        tag=IN_ANGLE_VMIN,
                                        default_value=float(vis_angle_diag_vmin),
                                        format="%.0f",
                                        step=1.0,
                                        step_fast=10.0,
                                        width=220,
                                    )
                                    dpg.add_input_double(
                                        label="Vmax dB",
                                        tag=IN_ANGLE_VMAX,
                                        default_value=float(vis_angle_diag_vmax),
                                        format="%.0f",
                                        step=1.0,
                                        step_fast=10.0,
                                        width=220,
                                    )
                                dpg.set_axis_limits(ANGLEFFT_XAXIS_TAG, float(angle_axis_min), float(angle_axis_max))
                                dpg.set_axis_limits(ANGLEFFT_YAXIS_TAG, 0.0, float(vis_fft_xmax))
                                dpg.set_axis_limits(ANGLEFFT_PROFILE_XAXIS_TAG, float(angle_axis_min), float(angle_axis_max))
                                dpg.set_axis_limits(ANGLEFFT_PROFILE_YAXIS_TAG, float(vis_angle_diag_vmin), float(vis_angle_diag_vmax))

                            with dpg.tab(label="Doppler"):
                                with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp):
                                    dpg.add_table_column(init_width_or_weight=1.0, width_stretch=True)
                                    dpg.add_table_column(init_width_or_weight=70, width_fixed=True)
                                    with dpg.table_row():
                                        with dpg.table_cell():
                                            with dpg.plot(
                                                tag=DOPPLERFFT_HEAT_PLOT_TAG,
                                                label="Range-Doppler FFT",
                                                width=-1,
                                                height=500,
                                                no_menus=True,
                                            ):
                                                dpg.add_plot_axis(dpg.mvXAxis, label="Doppler (m/s)", tag=DOPPLERFFT_XAXIS_TAG)
                                                dpg.add_plot_axis(dpg.mvYAxis, label="Range (m)", tag=DOPPLERFFT_YAXIS_TAG)
                                                dpg.add_image_series(
                                                    DOPPLERFFT_TEX_TAG,
                                                    bounds_min=(float(doppler_axis_min), 0.0),
                                                    bounds_max=(float(doppler_axis_max), float(fft_plot_h) * float(dr_plot)),
                                                    tag=DOPPLERFFT_IMG_SERIES_TAG,
                                                    parent=DOPPLERFFT_YAXIS_TAG,
                                                )
                                            with dpg.plot(
                                                tag=DOPPLERFFT_PROFILE_PLOT_TAG,
                                                label="Doppler FFT @ range bin",
                                                width=-1,
                                                height=500,
                                                no_menus=True,
                                                show=False,
                                            ):
                                                dpg.add_plot_axis(dpg.mvXAxis, label="Doppler (m/s)", tag=DOPPLERFFT_PROFILE_XAXIS_TAG)
                                                dpg.add_plot_axis(dpg.mvYAxis, label="dB", tag=DOPPLERFFT_PROFILE_YAXIS_TAG)
                                                dpg.add_line_series([], [], label="Doppler", tag=DOPPLERFFT_PROFILE_LINE_TAG, parent=DOPPLERFFT_PROFILE_YAXIS_TAG)
                                        with dpg.table_cell():
                                            dpg.add_colormap_scale(
                                                tag=DOPPLERFFT_CMAP_SCALE_TAG,
                                                min_scale=float(vis_doppler_diag_vmin),
                                                max_scale=float(vis_doppler_diag_vmax),
                                                format=CMAP_NUM_FMT,
                                                width=-1,
                                                height=500,
                                                colormap=dpg.mvPlotColormap_Jet,
                                            )
                                with dpg.group():
                                    dpg.add_checkbox(
                                        label="Single range bin",
                                        tag=CHK_DOPPLER_SINGLE_BIN,
                                        default_value=False,
                                        callback=_on_doppler_single_bin,
                                    )
                                    dpg.add_checkbox(
                                        label="Norm to peak",
                                        tag=CHK_DOPPLER_NORM,
                                        default_value=bool(doppler_diag_norm_to_peak),
                                    )
                                with dpg.group():
                                    dpg.add_input_int(
                                        label="Range bin",
                                        tag=IN_DOPPLER_BIN,
                                        default_value=0,
                                        min_value=0,
                                        max_value=max(0, int(fft_plot_h) - 1),
                                        min_clamped=True,
                                        max_clamped=True,
                                        width=220,
                                    )
                                    dpg.add_input_double(
                                        label="Vmin dB",
                                        tag=IN_DOPPLER_VMIN,
                                        default_value=float(vis_doppler_diag_vmin),
                                        format="%.0f",
                                        step=1.0,
                                        step_fast=10.0,
                                        width=220,
                                    )
                                    dpg.add_input_double(
                                        label="Vmax dB",
                                        tag=IN_DOPPLER_VMAX,
                                        default_value=float(vis_doppler_diag_vmax),
                                        format="%.0f",
                                        step=1.0,
                                        step_fast=10.0,
                                        width=220,
                                    )
                                dpg.set_axis_limits(DOPPLERFFT_XAXIS_TAG, float(doppler_axis_min), float(doppler_axis_max))
                                dpg.set_axis_limits(DOPPLERFFT_YAXIS_TAG, 0.0, float(vis_fft_xmax))
                                dpg.set_axis_limits(DOPPLERFFT_PROFILE_XAXIS_TAG, float(doppler_axis_min), float(doppler_axis_max))
                                dpg.set_axis_limits(DOPPLERFFT_PROFILE_YAXIS_TAG, float(vis_doppler_diag_vmin), float(vis_doppler_diag_vmax))

        with dpg.table(header_row=False, resizable=True, policy=dpg.mvTable_SizingFixedFit, parent=TAB_PROCESSED_TAG):
            dpg.add_table_column(init_width_or_weight=340, width_fixed=True)
            dpg.add_table_column(init_width_or_weight=1.0, width_stretch=True, width_fixed=False)
            dpg.add_table_column(init_width_or_weight=100, width_fixed=True)

            with dpg.table_row():
                with dpg.table_cell():
                    with dpg.child_window(width=-1, height=-1, border=True):
                        dpg.add_text("OFFLINE PROCESSING", color=(255, 200, 0))
                        dpg.add_separator()
                        dpg.add_button(
                            label="LOAD / CALCULATE OFFLINE",
                            tag=PROC_BTN_LOAD_OFFLINE,
                            callback=_on_load_offline,
                            width=-1,
                            height=38,
                        )
                        dpg.add_text(
                            offline_memory_estimate_text,
                            tag=PROC_TXT_MEMORY_ESTIMATE,
                            wrap=300,
                            color=(255, 190, 120),
                        )
                        dpg.add_spacer(height=12)
                        dpg.add_text("DISPLAY PARAMETERS", color=(255, 200, 0))
                        dpg.add_separator()
                        dpg.add_spacer(width=20)
                        dpg.add_input_float(
                            label="Vmin (dB)",
                            tag=PROC_IN_VMIN,
                            default_value=float(off_vmin),
                            step=1.0,
                            width=220,
                            callback=_apply_offline_params,
                            on_enter=False,
                        )
                        dpg.add_input_float(
                            label="Vmax (dB)",
                            tag=PROC_IN_VMAX,
                            default_value=float(off_vmax),
                            step=1.0,
                            width=220,
                            callback=_apply_offline_params,
                            on_enter=False,
                        )
                        dpg.add_spacer(height=8)
                        dpg.add_button(
                            label=_norm_toggle_label(bool(off_ui_pending.get("norm_enabled", True))),
                            tag=PROC_BTN_NORM,
                            callback=_toggle_off_norm,
                            width=-1,
                            height=32,
                        )
                        dpg.add_spacer(height=14)
                        dpg.add_text("OFFLINE ROI RECONSTRUCTION", color=(255, 200, 0))
                        dpg.add_separator()
                        dpg.add_input_float(
                            label="X min (m)",
                            tag=PROC_IN_ZOOM_XMIN,
                            default_value=float(off_home_viewport_current.x_min_m),
                            step=0.25,
                            width=220,
                            on_enter=False,
                        )
                        dpg.add_input_float(
                            label="X max (m)",
                            tag=PROC_IN_ZOOM_XMAX,
                            default_value=float(off_home_viewport_current.x_max_m),
                            step=0.25,
                            width=220,
                            on_enter=False,
                        )
                        dpg.add_input_float(
                            label="Y min (m)",
                            tag=PROC_IN_ZOOM_YMIN,
                            default_value=float(off_home_viewport_current.y_min_m),
                            step=0.5,
                            width=220,
                            on_enter=False,
                        )
                        dpg.add_input_float(
                            label="Y max (m)",
                            tag=PROC_IN_ZOOM_YMAX,
                            default_value=float(off_home_viewport_current.y_max_m),
                            step=0.5,
                            width=220,
                            on_enter=False,
                        )
                        with dpg.group(horizontal=True):
                            dpg.add_button(
                                label="Apply ROI",
                                callback=_apply_offline_display_zoom,
                                width=140,
                            )
                            dpg.add_button(
                                label="Full map",
                                callback=_reset_offline_display_zoom,
                                width=140,
                            )
                        dpg.add_text(
                            "Full reconstructed map.",
                            tag=PROC_TXT_ZOOM_STATUS,
                            wrap=300,
                            color=(180, 210, 255),
                        )
                        dpg.add_spacer(height=14)
                        with dpg.group(tag=PROC_GRP_LINEAR_POSITION_SELECTION):
                            dpg.add_text("LINEAR SAR POSITION SELECTION", color=(255, 200, 0))
                            dpg.add_separator()
                            dpg.add_input_int(
                                label="x_start",
                                tag=PROC_IN_XSTART,
                                default_value=int(off_x_start),
                                step=1,
                                step_fast=4,
                                width=220,
                                callback=_apply_offline_params,
                                on_enter=False,
                            )
                            dpg.add_input_int(
                                label="x_end",
                                tag=PROC_IN_XEND,
                                default_value=int(off_x_end),
                                step=1,
                                step_fast=4,
                                width=220,
                                callback=_apply_offline_params,
                                on_enter=False,
                            )

                with dpg.table_cell():
                    with dpg.plot(tag=PROC_HEAT_PLOT_TAG, width=-1, height=-1, equal_aspects=True):
                        dpg.add_plot_axis(dpg.mvXAxis, label="X (m)", tag=PROC_XAXIS_TAG)
                        dpg.add_plot_axis(dpg.mvYAxis, label="Y (m)", tag=PROC_YAXIS_TAG)
                        dpg.add_image_series(
                            PROC_TEX_TAG,
                            bounds_min=(float(off_applied_meta_current.x_min_m), float(off_applied_meta_current.y_min_m)),
                            bounds_max=(float(off_applied_meta_current.x_max_m), float(off_applied_meta_current.y_max_m)),
                            tag=PROC_IMG_SERIES_TAG,
                            parent=PROC_YAXIS_TAG,
                        )

                with dpg.table_cell():
                    with dpg.child_window(width=-1, height=-1, border=True):
                        dpg.add_colormap_scale(
                            tag=PROC_CMAP_SCALE_TAG,
                            min_scale=float(off_vmin),
                            max_scale=float(off_vmax),
                            format=PROC_CMAP_NUM_FMT,
                            width=-1,
                            height=-1,
                            colormap=dpg.mvPlotColormap_Jet,
                        )
                        if font_mono:
                            dpg.bind_item_font(PROC_CMAP_SCALE_TAG, font_mono)

        with dpg.child_window(parent=TAB_TUNING_TAG, width=-1, height=-1, border=True):
            dpg.add_text("RUNTIME TUNING", color=(255, 200, 0))
            dpg.add_separator()
            dpg.add_text("Ready", tag=TXT_TUNING_STATUS_TAG, wrap=-1)
            dpg.add_spacer(height=6)
            with dpg.tab_bar(tag="tuning_subtabbar"):
                with dpg.tab(label="DSP"):
                    with dpg.child_window(width=-1, height=-1, border=False):
                        _add_tuning_section("DSP", width=280)
                with dpg.tab(label="Tracking"):
                    with dpg.child_window(width=-1, height=-1, border=False):
                        _add_tuning_section("Tracking", width=280)
                with dpg.tab(label="Detection static"):
                    with dpg.child_window(width=-1, height=-1, border=False):
                        _add_tuning_section("Detection static", width=280)
                with dpg.tab(label="Detection moving"):
                    with dpg.child_window(width=-1, height=-1, border=False):
                        _add_tuning_section("Detection moving", width=280)
                with dpg.tab(label="Fusion"):
                    with dpg.child_window(width=-1, height=-1, border=False):
                        _add_tuning_section("Fusion", width=280)

        with dpg.child_window(parent=TAB_OFFLINE_TUNING_TAG, width=-1, height=-1, border=True):
            dpg.add_text("TUNING OFFLINE", color=(255, 200, 0))
            dpg.add_text(
                "Changes remain local until you choose 'Save' or 'Save and Recalculate Offline'.",
                wrap=-1,
            )
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="Reload from File", callback=_on_reload_offline_tuning, width=160)
                dpg.add_button(label="Inspect run", tag=BTN_OFFLINE_INSPECT_RUN, callback=_inspect_offline_run, width=140)
                dpg.add_button(label="Save Configuration", callback=_on_save_offline_tuning, width=180)
                dpg.add_button(label="Save and Recalculate Offline", callback=_on_apply_offline_tuning, width=230)
            dpg.add_text("Ready", tag=TXT_OFFLINE_TUNING_STATUS_TAG, wrap=-1)
            dpg.add_text("Format: not inspected", tag=TXT_OFFLINE_RUN_FORMAT, wrap=-1, color=(255, 210, 120))
            dpg.add_text(
                "Offline runtime not loaded.\n" + offline_memory_estimate_text,
                tag=TXT_OFFLINE_TUNING_INFO_TAG,
                wrap=-1,
                color=(180, 210, 255),
            )
            dpg.add_spacer(height=4)
            with dpg.tab_bar(tag="offline_tuning_subtabbar"):
                with dpg.tab(label="Data"):
                    with dpg.child_window(width=-1, height=-1, border=False):
                        _add_offline_tuning_section("Data and scan", mode="shared")
                        with dpg.group(tag=OFFLINE_TUNING_GRP_LINEAR_SCAN):
                            _add_offline_tuning_section("Data and scan", mode="linear")
                with dpg.tab(label="Reconstruction"):
                    with dpg.child_window(width=-1, height=-1, border=False):
                        _add_offline_tuning_section("Reconstruction", mode="shared")
                        with dpg.group(tag=OFFLINE_TUNING_GRP_RECONSTRUCTION):
                            _add_offline_tuning_section("Reconstruction", mode="linear")
                with dpg.tab(label="Map"):
                    with dpg.child_window(width=-1, height=-1, border=False):
                        with dpg.group(tag=OFFLINE_TUNING_GRP_MAP_BOUNDS):
                            dpg.add_text(
                                "Defines the full physical rectangle available to offline reconstruction. "
                                "Full map reconstructs this rectangle; Apply ROI reconstructs a contained section on the same fixed image grid.",
                                wrap=-1,
                            )
                            _add_offline_tuning_section("Map reconstruction bounds", mode="linear")
                with dpg.tab(label="FFT"):
                    with dpg.child_window(width=-1, height=-1, border=False):
                        dpg.add_text(
                            "Below input: truncate data. Equal: no padding. Above input: zero-padding. "
                            "Angle FFT is not used by backprojection; with Bartlett/MVDR it is the output grid size.",
                            wrap=-1,
                        )
                        _add_offline_tuning_section("FFT sizing")
                with dpg.tab(label="Filters"):
                    with dpg.child_window(width=-1, height=-1, border=False):
                        dpg.add_text("Windows are used by both offline algorithms; the aperture window tapers SAR positions × antennas in backprojection.", wrap=-1)
                        _add_offline_tuning_section("Shared windows")
                        dpg.add_separator()
                        _add_offline_tuning_section("Range-angle filters")
                with dpg.tab(label="Background"):
                    with dpg.child_window(width=-1, height=-1, border=False):
                        dpg.add_text(
                            "Subtract a separate empty-scene acquisition before reconstruction.",
                            wrap=-1,
                        )
                        _add_offline_tuning_section("Offline reference background")
                with dpg.tab(label="Angle"):
                    with dpg.child_window(width=-1, height=-1, border=False):
                        _add_offline_tuning_section("Angle estimation")

    _update_heatmap_scale_input_labels(vis_heatmap_mode)
    _apply_heatmap_mode_controls(vis_heatmap_mode)
    _apply_heatmap_plot_geometry(vis_heatmap_mode, vis_rmax, vis_xmax, reset_limits=True)
    _update_fft_scale_input_labels(fft_mode_db)
    # Applicazione parametri DOPO creazione items
    _apply_params()
    _apply_offline_geometry_controls()
    if dpg.does_item_exist(PROC_XAXIS_TAG) and dpg.does_item_exist(PROC_YAXIS_TAG):
        dpg.set_axis_limits(
            PROC_XAXIS_TAG,
            float(off_home_viewport_current.x_min_m),
            float(off_home_viewport_current.x_max_m),
        )
        dpg.set_axis_limits(
            PROC_YAXIS_TAG,
            float(off_home_viewport_current.y_min_m),
            float(off_home_viewport_current.y_max_m),
        )
    if dpg.does_item_exist(PROC_CMAP_SCALE_TAG):
        dpg.configure_item(
            PROC_CMAP_SCALE_TAG,
            min_scale=float(off_vmin),
            max_scale=float(off_vmax),
            format=PROC_CMAP_NUM_FMT,
        )

    if offline_runtime is None:
        for tag in (PROC_IN_XSTART, PROC_IN_XEND):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, enabled=False)
    else:
        off_ui_dirty = True
        off_ui_dirty_t = time.perf_counter() - 1.0

    dpg.create_viewport(title="MIMO Radar Real-Time", width=1400, height=900)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window(TAG_MAIN_WINDOW, True)
    dpg.set_viewport_min_width(900)
    dpg.set_viewport_min_height(500)

    # --- LOOP PRINCIPALE ---
    if DEBUG_STATS:
        t_mon = time.perf_counter()
        img_updates = 0
        t_img_start = time.perf_counter()
        lost_prev = 0
        pkts_prev = 0
        frames_ok_prev = 0
        ring_drop_prev = 0
        dsp_skip_prev = 0
        stall_events_prev = 0
        stream_resets_prev = 0
        log_bytes_prev = 0
    gui_last_seq = 0
    tracks_last_seq = 0
    gui_frame = np.zeros((gui_h, gui_w), dtype=np.float32)
    gui_alpha_frame = np.zeros((gui_h, gui_w), dtype=np.float32)
    rangefft_frame = np.full((RANGE_PROFILE_COUNT, fft_plot_h), -120.0, dtype=np.float32)
    anglefft_frame = np.full((fft_plot_h, ANGLEFFT_BINS), -120.0, dtype=np.float32)
    dopplerfft_frame = np.full((fft_plot_h, DOPPLERFFT_BINS), -120.0, dtype=np.float32)
    gui_tracks_xy_view = np.frombuffer(gui_tracks_xy_dbuf, dtype=np.float32, count=track_max_shared * 4)
    gui_tracks_meta_view = np.frombuffer(gui_tracks_meta_dbuf, dtype=np.int32, count=track_max_shared * 4)
    gui_tracks_state_view = np.frombuffer(gui_tracks_state_dbuf, dtype=np.int32, count=track_max_shared * 2)
    gui_tracks_stop_xy_view = np.frombuffer(gui_tracks_stop_xy_dbuf, dtype=np.float32, count=track_max_shared * 2)
    tracks_out: list[dict[str, float | int | bool]] = []
    proc_frame = np.full((proc_tex_h, proc_tex_w), off_vmin, dtype=np.float32)
    x_range_cache = {}
    jet_lut = _build_jet_lut(2048)
    velocity_lut = _build_velocity_lut(2048)
    velocity_nodata_rgba = np.asarray([0.015, 0.015, 0.018, 1.0], dtype=np.float32)
    norm_frame = np.empty((gui_h, gui_w), dtype=np.float32)
    lut_idx = np.empty((gui_h, gui_w), dtype=np.int32)
    rgba_frame = np.empty((gui_h, gui_w, 4), dtype=np.float32)
    fft_lin_frame = np.empty((RANGE_PROFILE_COUNT, fft_plot_h), dtype=np.float32)
    angle_diag_norm_frame = np.empty((fft_plot_h, ANGLEFFT_BINS), dtype=np.float32)
    angle_diag_lut_idx = np.empty((fft_plot_h, ANGLEFFT_BINS), dtype=np.int32)
    angle_diag_rgba_frame = np.empty((fft_plot_h, ANGLEFFT_BINS, 4), dtype=np.float32)
    doppler_diag_norm_frame = np.empty((fft_plot_h, DOPPLERFFT_BINS), dtype=np.float32)
    doppler_diag_lut_idx = np.empty((fft_plot_h, DOPPLERFFT_BINS), dtype=np.int32)
    doppler_diag_rgba_frame = np.empty((fft_plot_h, DOPPLERFFT_BINS, 4), dtype=np.float32)
    proc_norm_frame = np.empty((proc_tex_h, proc_tex_w), dtype=np.float32)
    proc_lut_idx = np.empty((proc_tex_h, proc_tex_w), dtype=np.int32)
    proc_rgba_frame = np.empty((proc_tex_h, proc_tex_w, 4), dtype=np.float32)
    proc_view_frame = np.empty((proc_tex_h, proc_tex_w), dtype=np.float32)
    lut_last = float(jet_lut.shape[0] - 1)

    def _refresh_offline_texture_from_cached_matrix() -> None:
        """Reapply normalization/levels/LUT without invoking offline DSP."""
        if not offline_frame_valid:
            proc_lut_idx.fill(0)
            np.take(jet_lut, proc_lut_idx, axis=0, out=proc_rgba_frame)
            proc_tex_np[:] = proc_rgba_frame.reshape(-1)
            if dpg.does_item_exist(PROC_TEX_TAG):
                dpg.set_value(PROC_TEX_TAG, proc_tex_buf)
            return
        if off_norm_enabled and proc_frame.size > 0:
            finite_values = proc_frame[np.isfinite(proc_frame)]
            peak_db = float(np.max(finite_values)) if finite_values.size > 0 else 0.0
            np.subtract(proc_frame, peak_db, out=proc_view_frame)
        else:
            proc_view_frame[:, :] = proc_frame

        denom = max(float(off_vmax - off_vmin), 1e-6)
        np.subtract(proc_view_frame, float(off_vmin), out=proc_norm_frame)
        np.multiply(proc_norm_frame, float(1.0 / denom), out=proc_norm_frame)
        np.clip(proc_norm_frame, 0.0, 1.0, out=proc_norm_frame)
        np.nan_to_num(proc_norm_frame, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        np.multiply(proc_norm_frame, lut_last, out=proc_norm_frame)
        np.rint(proc_norm_frame, out=proc_norm_frame)
        proc_lut_idx[:, :] = proc_norm_frame
        np.take(jet_lut, proc_lut_idx[::-1, :], axis=0, out=proc_rgba_frame)
        proc_tex_np[:] = proc_rgba_frame.reshape(-1)
        if dpg.does_item_exist(PROC_TEX_TAG):
            dpg.set_value(PROC_TEX_TAG, proc_tex_buf)

    fft_plot_period_s = 0.05
    fft_plot_last_t = 0.0
    last_scan_terminal_state: ScanState | None = None
    if DEBUG_STATS:
        ring_max_sampled = 0
        cpu_state = {}

    try:
        while dpg.is_dearpygui_running():
            now = time.perf_counter()
            refresh_offline_display = bool(off_texture_upload_requested)
            off_texture_upload_requested = False
            # The DSP result is copied before the texture upload.  Defer its
            # human-readable status and timing until that upload succeeds, so
            # the reported calculation always corresponds to the image shown.
            offline_frame_summary_pending = None

            _drain_motor_events()
            scan_state_now = _refresh_sar_scan_ui()
            if scan_state_now not in {ScanState.COMPLETED, ScanState.CANCELLED, ScanState.FAILED}:
                last_scan_terminal_state = None
            elif (
                scan_state_now is ScanState.COMPLETED
                and last_scan_terminal_state is not ScanState.COMPLETED
            ):
                with offline_reload_lock:
                    reload_already_running = bool(offline_reload_state["running"])
                if reload_already_running:
                    # Non consumare lo stato terminale: appena il ricalcolo
                    # precedente termina, questo ramo viene ritentato.
                    _set_scan_status("Scan completed: waiting for the previous offline recalculation...")
                else:
                    start_id = scan_pending_run["start_position_id"]
                    positions = scan_pending_run["positions"]
                    if start_id is None or positions is None:
                        _set_scan_status("SAR scan completed.")
                        last_scan_terminal_state = ScanState.COMPLETED
                    else:
                        try:
                            # Solo ora tutti i file sono chiusi con successo:
                            # conferma directory e intervallo della nuova run.
                            pitch_mm = configure_offline_scan_for_run(
                                offline_scan_config_path,
                                output_dir=out_dir,
                                start_position_id=int(start_id),
                                positions=int(positions),
                                frames_per_position=FRAMES_PER_POSITION,
                            )
                            _update_scan_pitch_label(pitch_mm)
                            if not _read_offline_tuning_config():
                                raise RuntimeError("impossibile rileggere offline_config.yaml")
                            reload_started = _on_apply_offline_tuning(
                                save_config=False,
                                allow_scan_finalize=True,
                            )
                            if reload_started:
                                scan_pending_run.update({"start_position_id": None, "positions": None})
                                _set_scan_status("Scan completed: offline reload started.")
                                last_scan_terminal_state = ScanState.COMPLETED
                        except Exception as exc:
                            # I file restano validi; sblocca il tuning manuale
                            # e comunica chiaramente il solo errore offline.
                            scan_pending_run.update({"start_position_id": None, "positions": None})
                            _set_scan_status(f"Scan completed; offline reload did not start: {exc}")
                            last_scan_terminal_state = ScanState.COMPLETED
            elif scan_state_now in {ScanState.CANCELLED, ScanState.FAILED}:
                # Non puntare l'offline a un insieme di file parziale.
                scan_pending_run.update({"start_position_id": None, "positions": None})
                last_scan_terminal_state = scan_state_now
            else:
                last_scan_terminal_state = scan_state_now

            # A restart is performed in a worker because loading the capture
            # files and building the first image can take seconds.  Adopt its
            # result only from the GUI thread.
            reload_runtime = None
            reload_info = None
            reload_error = ""
            with offline_reload_lock:
                if offline_reload_state.get("runtime") is not None or offline_reload_state.get("error"):
                    reload_runtime = offline_reload_state.get("runtime")
                    reload_info = offline_reload_state.get("info")
                    reload_error = str(offline_reload_state.get("error") or "")
                    offline_reload_state.update({"runtime": None, "info": None, "error": ""})
            if reload_runtime is not None:
                offline_runtime = reload_runtime
                offline_recalculation_completion_pending = offline_runtime
                shutdown_resources["offline_runtime"] = offline_runtime
                offline_info = dict(reload_info or offline_runtime.last_info)
                off_map_bounds_current = offline_runtime.map_bounds
                off_home_viewport_current = build_display_viewport(
                    x_min_m=float(off_map_bounds_current.x_min_m),
                    x_max_m=float(off_map_bounds_current.x_max_m),
                    y_min_m=float(off_map_bounds_current.y_min_m),
                    y_max_m=float(off_map_bounds_current.y_max_m),
                    dr_m=float(dr_plot),
                    seq=int(off_requested_viewport_current.seq) + 1,
                )
                _apply_offline_geometry_controls()
                off_requested_viewport_current = off_home_viewport_current
                off_pos_min = int(offline_info.get("pos_min", 0))
                off_pos_max = int(offline_info.get("pos_max", off_pos_min))
                if off_pos_max < off_pos_min:
                    off_pos_min, off_pos_max = off_pos_max, off_pos_min
                off_x_start = max(off_pos_min, min(off_pos_max, int(offline_info.get("x_start", off_pos_min))))
                off_x_end = max(off_pos_min, min(off_pos_max, int(offline_info.get("x_end", off_pos_max))))
                if off_x_end < off_x_start:
                    off_x_start, off_x_end = off_x_end, off_x_start
                off_ui_pending.update({
                    "x_start": int(off_x_start),
                    "x_end": int(off_x_end),
                    "reset_view": True,
                })
                off_ui_dirty = True
                off_ui_dirty_t = now - 1.0
                if dpg.does_item_exist(PROC_IN_XSTART):
                    dpg.set_value(PROC_IN_XSTART, int(off_x_start))
                    dpg.configure_item(PROC_IN_XSTART, enabled=True)
                if dpg.does_item_exist(PROC_IN_XEND):
                    dpg.set_value(PROC_IN_XEND, int(off_x_end))
                    dpg.configure_item(PROC_IN_XEND, enabled=True)
                if dpg.does_item_exist(PROC_BTN_LOAD_OFFLINE):
                    dpg.configure_item(
                        PROC_BTN_LOAD_OFFLINE,
                        label="RELOAD / RECALCULATE OFFLINE",
                        enabled=True,
                    )
                _set_offline_tuning_status(
                    "Offline runtime ready; waiting for the reconstructed image..."
                )
            elif reload_error:
                offline_runtime = None
                shutdown_resources["offline_runtime"] = None
                for tag in (PROC_IN_XSTART, PROC_IN_XEND):
                    if dpg.does_item_exist(tag):
                        dpg.configure_item(tag, enabled=False)
                if dpg.does_item_exist(PROC_BTN_LOAD_OFFLINE):
                    dpg.configure_item(
                        PROC_BTN_LOAD_OFFLINE,
                        label="RETRY LOAD / CALCULATE OFFLINE",
                        enabled=True,
                    )
                _render_offline_summary(state=f"Offline load/calculation error: {reload_error}")
                _set_offline_tuning_status(f"Offline recalculation error: {reload_error}")

            if dpg.does_item_exist(TXT_OFFLINE_TUNING_INFO_TAG):
                if bool(offline_reload_state.get("running")):
                    if dpg.does_item_exist(PROC_BTN_LOAD_OFFLINE):
                        dpg.configure_item(PROC_BTN_LOAD_OFFLINE, label="LOADING OFFLINE...", enabled=False)
                    loading_runtime = shutdown_resources.get("offline_runtime")
                    try:
                        loading_info = loading_runtime.last_info if loading_runtime is not None else {}
                    except RuntimeError:
                        loading_info = {}
                    loading_phase = str(loading_info.get("phase", "starting offline workers"))
                    loading_text = f"Offline load/calculation in progress: {loading_phase}\n{offline_memory_estimate_text}"
                    _render_offline_summary(state=f"Offline loading: {loading_phase}")
                    dpg.set_value(
                        TXT_OFFLINE_TUNING_INFO_TAG,
                        loading_text,
                    )
                elif offline_runtime is not None:
                    info_now = offline_runtime.last_info
                    filters_now = info_now.get("range_angle_enabled_filters", ())
                    filters_text = ", ".join(str(v) for v in filters_now) or "none"
                    range_input = info_now.get("range_input_samples", "n/a")
                    range_used = info_now.get("range_samples_used", "n/a")
                    range_nfft = info_now.get("nfft_range", "n/a")
                    if str(info_now.get("algorithm", "")) == "backprojection":
                        angle_fft_text = "Angle FFT: n/a for backprojection"
                    else:
                        angle_fft_text = (
                            f"Angle: input {info_now.get('angle_input_elements', 'n/a')} -> "
                            f"used {info_now.get('angle_elements_used', 'n/a')} -> "
                            f"FFT/grid {info_now.get('nfft_angle_effective', info_now.get('range_angle_nfft_angle', 'n/a'))}"
                        )
                    if bool(info_now.get("background_reference_enabled", False)):
                        background_text = (
                            "Reference background: ON "
                            f"({info_now.get('background_reference_frames', 'n/a')} frames mean, "
                            f"scale {info_now.get('background_reference_scale', 'n/a')})"
                        )
                    else:
                        background_text = "Reference background: OFF"
                    dpg.set_value(
                        TXT_OFFLINE_TUNING_INFO_TAG,
                        f"Algorithm: {info_now.get('algorithm', 'n/a')} | "
                        f"Angle: {info_now.get('angle_mode', info_now.get('range_angle_angle_mode_requested', 'n/a'))} | "
                        f"Filters: {filters_text} | {background_text}\n"
                        f"Range: input {range_input} -> used {range_used} -> FFT {range_nfft} | {angle_fft_text}",
                    )


            # --- UI apply (throttled, non-blocking) ---
            if ui_dirty and (
                bool(ui_pending.get("reset_view", False))
                or (now - ui_dirty_t) >= 0.08
            ):  # reset immediato; altrimenti throttle ~12.5 Hz
                ui_dirty = False
                ui_dirty_t = now

                vmin = ui_pending["vmin"]
                vmax = ui_pending["vmax"]
                # HARD clamp anche qui (ridondanza: protegge da valori inseriti via codice)
                rmax = max(0.0, min(ui_pending["rmax"], RMAX_HARD_MAX))
                xmax = max(0.0, min(ui_pending["xmax"], HEATMAP_XMAX_HARD_MAX))
                fft_xmin = float(ui_pending.get("fft_xmin", 0.0))
                fft_xmax = float(ui_pending.get("fft_xmax", float(fft_plot_h) * float(dr_plot)))
                fft_vmin = float(ui_pending["fft_vmin"])
                fft_vmax = float(ui_pending["fft_vmax"])
                fft_mode_db = bool(ui_pending["fft_mode_db"])
                fft_view_full = bool(ui_pending.get("fft_view_full", False))
                heatmap_mode_now = 1 if int(ui_pending.get("heatmap_mode", 0)) == 1 else 0
                reset_realtime_view = bool(ui_pending.get("reset_view", False))
                vmin, vmax = _sanitize_heatmap_scale_inputs(heatmap_mode_now, float(vmin), float(vmax))
                fft_xmin, fft_xmax, _ = _clamp_fft_x_window(fft_xmin, fft_xmax, rmax, fft_view_full)
                fft_eps = 1.0
                if fft_vmax <= fft_vmin:
                    fft_vmax = fft_vmin + fft_eps

                vis_vmin = vmin
                vis_vmax = vmax
                vis_rmax = rmax
                vis_xmax = xmax
                vis_heatmap_mode = int(heatmap_mode_now)
                vis_fft_xmin = fft_xmin
                vis_fft_xmax = fft_xmax
                vis_fft_mode_db = fft_mode_db
                vis_fft_view_full = fft_view_full
                vis_fft_vmin = fft_vmin
                vis_fft_vmax = fft_vmax

                rt_home_viewport_current = build_display_viewport(
                    x_min_m=-float(vis_xmax),
                    x_max_m=float(vis_xmax),
                    y_min_m=0.0,
                    y_max_m=float(vis_rmax),
                    dr_m=float(dr_plot),
                    seq=int(rt_requested_viewport_current.seq),
                )
                _write_realtime_home_viewport(rt_home_viewport_current)
                if reset_realtime_view:
                    rt_requested_viewport_current = rt_home_viewport_current
                    _write_realtime_requested_viewport(rt_requested_viewport_current)
                    # Aggiorna subito i bounds del plot alla home/requested view
                    # mentre aspettiamo il prossimo frame applicato dal DSP.
                    rt_applied_meta_current = applied_viewport_meta_from_viewport(
                        rt_requested_viewport_current,
                        fallback_used=bool(rt_applied_meta_current.fallback_used),
                        frame_seq=int(rt_applied_meta_current.frame_seq),
                    )
                _apply_heatmap_plot_geometry(
                    vis_heatmap_mode,
                    vis_rmax,
                    vis_xmax,
                    reset_limits=bool(reset_realtime_view),
                )
                ui_pending["reset_view"] = False
                if vis_heatmap_mode == 0 and dpg.does_item_exist(GUIDE_NEG20_TAG):
                    guide_neg_x, guide_y = _guide_line_points(-1.0, vis_rmax)
                    dpg.set_value(GUIDE_NEG20_TAG, [guide_neg_x, guide_y])
                if vis_heatmap_mode == 0 and dpg.does_item_exist(GUIDE_POS20_TAG):
                    guide_pos_x, guide_y = _guide_line_points(+1.0, vis_rmax)
                    dpg.set_value(GUIDE_POS20_TAG, [guide_pos_x, guide_y])

                # colorbar
                if dpg.does_item_exist(CMAP_SCALE_TAG):
                    fmt = CMAP_VELOCITY_NUM_FMT if vis_heatmap_mode == 1 else CMAP_NUM_FMT
                    cmap = (
                        CMAP_VELOCITY_TAG
                        if vis_heatmap_mode == 1 and dpg.does_item_exist(CMAP_VELOCITY_TAG)
                        else dpg.mvPlotColormap_Jet
                    )
                    dpg.configure_item(CMAP_SCALE_TAG, min_scale=vis_vmin, max_scale=vis_vmax, format=fmt, colormap=cmap)
                _apply_heatmap_mode_controls(vis_heatmap_mode)
                if vis_heatmap_mode == 1:
                    if dpg.does_item_exist(IN_VMIN):
                        dpg.set_value(IN_VMIN, vis_vmin)
                    if dpg.does_item_exist(IN_VMAX):
                        dpg.set_value(IN_VMAX, vis_vmax)
                if dpg.does_item_exist(BTN_FFT_VIEW_TAG):
                    dpg.configure_item(BTN_FFT_VIEW_TAG, label=_fft_view_label(vis_fft_view_full))
                if dpg.does_item_exist(BTN_FFT_MODE_TAG):
                    dpg.configure_item(BTN_FFT_MODE_TAG, label=_fft_mode_label(vis_fft_mode_db))
                if dpg.does_item_exist(RANGEFFT_PLOT_TAG):
                    dpg.configure_item(RANGEFFT_PLOT_TAG, label=_fft_plot_title(vis_fft_view_full))
                if dpg.does_item_exist(IN_FFT_XMIN):
                    dpg.set_value(IN_FFT_XMIN, float(vis_fft_xmin))
                if dpg.does_item_exist(IN_FFT_XMAX):
                    dpg.set_value(IN_FFT_XMAX, float(vis_fft_xmax))
                _update_fft_scale_input_labels(vis_fft_mode_db)

                if dpg.does_item_exist(RANGEFFT_YAXIS_TAG):
                    dpg.set_axis_limits(RANGEFFT_YAXIS_TAG, float(vis_fft_vmin), float(vis_fft_vmax))
                    dpg.configure_item(RANGEFFT_YAXIS_TAG, label=_fft_axis_label(vis_fft_mode_db))
                if dpg.does_item_exist(RANGEFFT_XAXIS_TAG):
                    dpg.set_axis_limits(RANGEFFT_XAXIS_TAG, float(vis_fft_xmin), float(vis_fft_xmax))
                    _set_fft_x_ticks_window(vis_fft_xmin, vis_fft_xmax)
                if dpg.does_item_exist(ANGLEFFT_YAXIS_TAG):
                    dpg.set_axis_limits(ANGLEFFT_YAXIS_TAG, 0.0, float(vis_fft_xmax))
                if dpg.does_item_exist(DOPPLERFFT_YAXIS_TAG):
                    dpg.set_axis_limits(DOPPLERFFT_YAXIS_TAG, 0.0, float(vis_fft_xmax))
                if dpg.does_item_exist(ANGLEFFT_PROFILE_YAXIS_TAG):
                    angle_vmin_now, angle_vmax_now = _read_diag_scale(IN_ANGLE_VMIN, IN_ANGLE_VMAX, vis_angle_diag_vmin, vis_angle_diag_vmax)
                    dpg.set_axis_limits(ANGLEFFT_PROFILE_YAXIS_TAG, float(angle_vmin_now), float(angle_vmax_now))
                    _update_diag_colormap_scale(ANGLEFFT_CMAP_SCALE_TAG, angle_vmin_now, angle_vmax_now)
                if dpg.does_item_exist(DOPPLERFFT_PROFILE_YAXIS_TAG):
                    doppler_vmin_now, doppler_vmax_now = _read_diag_scale(IN_DOPPLER_VMIN, IN_DOPPLER_VMAX, vis_doppler_diag_vmin, vis_doppler_diag_vmax)
                    dpg.set_axis_limits(DOPPLERFFT_PROFILE_YAXIS_TAG, float(doppler_vmin_now), float(doppler_vmax_now))
                    _update_diag_colormap_scale(DOPPLERFFT_CMAP_SCALE_TAG, doppler_vmin_now, doppler_vmax_now)

            polled_rt_viewport = _poll_requested_viewport(
                x_axis_tag=XAXIS_TAG,
                y_axis_tag=YAXIS_TAG,
                home_viewport=rt_home_viewport_current,
                current_viewport=rt_requested_viewport_current,
            )
            if display_viewport_signature(polled_rt_viewport) != display_viewport_signature(rt_requested_viewport_current):
                rt_requested_viewport_current = polled_rt_viewport
                _write_realtime_requested_viewport(rt_requested_viewport_current)

            # --- OFFLINE UI apply (throttled) ---
            if off_ui_dirty and (
                bool(off_ui_pending.get("reset_view", False))
                or (now - off_ui_dirty_t) >= 0.08
            ):
                off_ui_dirty = False
                off_ui_dirty_t = now
                off_vmin = float(off_ui_pending["vmin"])
                off_vmax = float(off_ui_pending["vmax"])
                off_norm_enabled = bool(off_ui_pending.get("norm_enabled", True))
                reset_offline_view = bool(off_ui_pending.get("reset_view", False))
                reconstruct_offline = bool(off_ui_pending.get("reconstruct", False))
                display_refresh_requested = bool(off_ui_pending.get("display_refresh", False))
                off_ui_pending["reconstruct"] = False
                off_ui_pending["display_refresh"] = False
                if off_vmax <= off_vmin:
                    off_vmax = off_vmin + 1.0
                    if dpg.does_item_exist(PROC_IN_VMAX):
                        dpg.set_value(PROC_IN_VMAX, off_vmax)

                off_ui_pending["vmin"] = float(off_vmin)
                off_ui_pending["vmax"] = float(off_vmax)
                off_ui_pending["norm_enabled"] = bool(off_norm_enabled)
                if reset_offline_view:
                    off_requested_viewport_current = off_home_viewport_current

                if dpg.does_item_exist(PROC_BTN_NORM):
                    dpg.configure_item(PROC_BTN_NORM, label=_norm_toggle_label(off_norm_enabled))
                if reset_offline_view and dpg.does_item_exist(PROC_XAXIS_TAG) and dpg.does_item_exist(PROC_YAXIS_TAG):
                    dpg.set_axis_limits(
                        PROC_XAXIS_TAG,
                        float(off_home_viewport_current.x_min_m),
                        float(off_home_viewport_current.x_max_m),
                    )
                    dpg.set_axis_limits(
                        PROC_YAXIS_TAG,
                        float(off_home_viewport_current.y_min_m),
                        float(off_home_viewport_current.y_max_m),
                    )
                    _write_offline_zoom_inputs(off_requested_viewport_current)
                    if dpg.does_item_exist(PROC_TXT_ZOOM_STATUS):
                        dpg.set_value(
                            PROC_TXT_ZOOM_STATUS,
                            f"Full map: X [{off_home_viewport_current.x_min_m:.3f}, "
                            f"{off_home_viewport_current.x_max_m:.3f}] m | "
                            f"Y [{off_home_viewport_current.y_min_m:.3f}, "
                            f"{off_home_viewport_current.y_max_m:.3f}] m\n"
                            f"Offline reconstruction queued on {offline_gui_w}x{offline_gui_h} grid.",
                        )
                off_ui_pending["reset_view"] = False
                if dpg.does_item_exist(PROC_CMAP_SCALE_TAG):
                    dpg.configure_item(
                        PROC_CMAP_SCALE_TAG,
                        min_scale=float(off_vmin),
                        max_scale=float(off_vmax),
                        format=PROC_CMAP_NUM_FMT,
                    )

                refresh_offline_display = bool(display_refresh_requested and not reconstruct_offline)

                if offline_runtime is not None and reconstruct_offline:
                    try:
                        offline_runtime.update_params(
                            x_start=int(off_ui_pending["x_start"]),
                            x_end=int(off_ui_pending["x_end"]),
                            viewport=off_requested_viewport_current,
                        )
                    except Exception as e:
                        _render_offline_summary(state=f"Offline update error: {e}")

            # The manual offline ROI controls submit a new viewport to the DSP
            # worker.  Direct plot pan/zoom remains a display-only interaction
            # until the user confirms it through "Apply ROI".

            # --- OFFLINE FRAME update (double-buffer latest-wins) ---
            if offline_runtime is not None:
                off_frame = offline_runtime.poll_frame(copy_frame=False)
                if off_frame is not None:
                    frame_db, info = off_frame
                    proc_frame[:, :] = frame_db
                    offline_frame_valid = True
                    refresh_offline_display = True
                    try:
                        off_applied_meta_current = AppliedViewportMeta(
                            x_min_m=float(info.get("applied_viewport_x_min_m", off_applied_meta_current.x_min_m)),
                            x_max_m=float(info.get("applied_viewport_x_max_m", off_applied_meta_current.x_max_m)),
                            y_min_m=float(info.get("applied_viewport_y_min_m", off_applied_meta_current.y_min_m)),
                            y_max_m=float(info.get("applied_viewport_y_max_m", off_applied_meta_current.y_max_m)),
                            range_min_bin_f=float(info.get("applied_viewport_range_min_bin_f", off_applied_meta_current.range_min_bin_f)),
                            range_max_bin_f=float(info.get("applied_viewport_range_max_bin_f", off_applied_meta_current.range_max_bin_f)),
                            angle_min_deg=float(info.get("applied_viewport_angle_min_deg", off_applied_meta_current.angle_min_deg)),
                            angle_max_deg=float(info.get("applied_viewport_angle_max_deg", off_applied_meta_current.angle_max_deg)),
                            zoom_level=float(info.get("applied_viewport_zoom_level", off_applied_meta_current.zoom_level)),
                            seq=int(info.get("applied_viewport_seq", off_applied_meta_current.seq)),
                            fallback_used=bool(info.get("fallback_used", False)),
                            frame_seq=int(info.get("frame_seq", off_applied_meta_current.frame_seq)),
                        )
                        _configure_image_bounds(PROC_IMG_SERIES_TAG, off_applied_meta_current)
                    except Exception:
                        pass
                    algorithm_text = str(info.get("algorithm", "backprojection"))
                    status_text = (
                        f"Offline {algorithm_text}: positions {info.get('x_start')} - "
                        f"{info.get('x_end')} ({info.get('n_pos_used', 'n/a')} used)"
                    )
                    offline_frame_summary_pending = (
                        status_text,
                        float(info.get("elapsed_ms", 0.0)),
                    )
                elif offline_runtime.last_error:
                    _render_offline_summary(state=f"ERROR: {offline_runtime.last_error}")

            if refresh_offline_display:
                _refresh_offline_texture_from_cached_matrix()
                if offline_frame_summary_pending is not None:
                    status_text, calculation_ms = offline_frame_summary_pending
                    _render_offline_summary(
                        state=status_text,
                        calculation_ms=calculation_ms,
                    )
                if offline_recalculation_completion_pending is offline_runtime and offline_frame_valid:
                    _set_offline_tuning_status("Offline recalculation completed")
                    offline_recalculation_completion_pending = None

            
            # 1. STATS UPDATE (1Hz) - GESTIONE DATI STAMPATI
            if DEBUG_STATS and (now - t_mon >= 1.0):
                dt_mon = now - t_mon
                t_mon = now
                
                # Hz aggiornamento immagine
                dt_img = now - t_img_start
                img_hz = img_updates / dt_img if dt_img > 0 else 0.0
                img_updates = 0
                t_img_start = now

                # READY slots currently in ring (cheap: N_SLOTS=64)
                ready_slots = 0
                try:
                    for _s in range(N_SLOTS):
                        if slot_state[_s] == 1:
                            ready_slots += 1
                except Exception:
                    ready_slots = 0
                ring_max_sampled = max(ring_max_sampled, int(ready_slots))

                with rx_put_drops.get_lock(): drop_f = int(rx_put_drops.value)
                drop_delta = max(0, int(drop_f) - int(ring_drop_prev))
                ring_drop_prev = int(drop_f)
                with rx_frames_ok.get_lock(): frames_ok_now = int(rx_frames_ok.value)
                frames_ok_delta = frames_ok_now - frames_ok_prev
                frames_ok_prev = frames_ok_now

                # Packet Stats
                with lost_pkts.get_lock(): lost_now = lost_pkts.value
                lost_delta = lost_now - lost_prev
                lost_prev = lost_now

                with rx_pkts.get_lock(): pkts_now = rx_pkts.value
                pkts_delta = pkts_now - pkts_prev
                pkts_prev = pkts_now
                total = pkts_delta + lost_delta
                loss_pct = (100.0 * lost_delta / total) if total > 0 else 0.0
                pkt_rate = (pkts_delta / dt_mon) if dt_mon > 0 else 0.0
                frames_ok_rate = (frames_ok_delta / dt_mon) if dt_mon > 0 else 0.0
                with dsp_ms_avg.get_lock(): dsp_avg_now = float(dsp_ms_avg.value)
                with dsp_ms_p95.get_lock(): dsp_p95_now = float(dsp_ms_p95.value)
                with dsp_skip.get_lock(): dsp_skip_now = int(dsp_skip.value)
                dsp_skip_delta = max(0, int(dsp_skip_now) - int(dsp_skip_prev))
                dsp_skip_prev = int(dsp_skip_now)
                with log_bytes.get_lock(): log_bytes_now = int(log_bytes.value)
                log_mbps = ((log_bytes_now - log_bytes_prev) / dt_mon) / (1024.0 * 1024.0) if dt_mon > 0 else 0.0
                log_bytes_prev = log_bytes_now
                cpu_dsp = _cpu_percent_pid(int(p_dsp.pid or 0), cpu_state)
                cpu_log = _cpu_percent_pid(int(p_log.pid or 0), cpu_state)
                with rx_stall_events.get_lock(): stall_events_now = int(rx_stall_events.value)
                with rx_stream_resets.get_lock(): stream_resets_now = int(rx_stream_resets.value)
                stall_events_delta = max(0, int(stall_events_now) - int(stall_events_prev))
                stream_resets_delta = max(0, int(stream_resets_now) - int(stream_resets_prev))
                stall_events_prev = int(stall_events_now)
                stream_resets_prev = int(stream_resets_now)
                with stat_raw_min_db.get_lock(): raw_min_now = float(stat_raw_min_db.value)
                with stat_raw_max_db.get_lock(): raw_max_now = float(stat_raw_max_db.value)
                with stat_norm_min_db.get_lock(): norm_min_now = float(stat_norm_min_db.value)
                with stat_norm_max_db.get_lock(): norm_max_now = float(stat_norm_max_db.value)
                raw_minmax_str = f"{raw_min_now:.2f}/{raw_max_now:.2f}" if (np.isfinite(raw_min_now) and np.isfinite(raw_max_now)) else "n/a"
                norm_minmax_str = f"{norm_min_now:.2f}/{norm_max_now:.2f}" if (np.isfinite(norm_min_now) and np.isfinite(norm_max_now)) else "n/a"
                cpu_dsp_str = f"{cpu_dsp:.1f}%" if np.isfinite(cpu_dsp) else "n/a"
                cpu_log_str = f"{cpu_log:.1f}%" if np.isfinite(cpu_log) else "n/a"
                with heatmap_view_mode.get_lock(): heatmap_mode_now = int(heatmap_view_mode.value)
                with norm_to_peak.get_lock(): normalization_requested = bool(norm_to_peak.value)
                with cap_active.get_lock(): capture_active_now = bool(cap_active.value)
                
                stat_values = {
                    "udp_rx": f"{loss_pct:.3f}% | {pkt_rate:.0f}/s",
                    "frames": f"{frames_ok_rate:.1f}/s  ({frames_ok_now})",
                    "ring": f"{ready_slots}/{N_SLOTS} | peak {ring_max_sampled}",
                    "drops": f"{drop_delta / dt_mon:.1f}/s  ({drop_f})",
                    "dsp_frame": f"{dsp_avg_now:.2f} ms | p95 {dsp_p95_now:.2f}",
                    "dsp_stale": f"{dsp_skip_delta / dt_mon:.1f}/s | {img_hz:.1f}/s",
                    "cpu": f"{cpu_dsp_str} / {cpu_log_str}",
                    "logger": f"{log_mbps:.2f} MB/s" if capture_active_now else "Inactive",
                    "stalls": f"{stall_events_delta / dt_mon:.1f}/s  ({stall_events_now})",
                    "resyncs": f"{stream_resets_delta / dt_mon:.1f}/s  ({stream_resets_now})",
                }
                if heatmap_mode_now == 1:
                    display_diag = (
                        "Moving velocity\n"
                        f"Range: {raw_minmax_str} m/s"
                    )
                elif normalization_requested:
                    display_diag = (
                        "Power XY\n"
                        f"Raw: {raw_minmax_str} dB\n"
                        f"Norm: {norm_minmax_str} dB"
                    )
                else:
                    display_diag = (
                        "Power XY\n"
                        f"Raw: {raw_minmax_str} dB\n"
                        "Normalization: off"
                    )

                for stat_name, value in stat_values.items():
                    dpg.set_value(DEBUG_STAT_VALUE_TAGS[stat_name], value)
                dpg.set_value(TXT_DISPLAY_DIAG_TAG, display_diag)
                _refresh_mmwave_controls()
            # 2. GUI TEXTURE UPDATE (double-buffer latest-wins)
            seq_now = int(gui_latest_seq.value)
            if seq_now != gui_last_seq:
                rt_applied_meta_current = _read_realtime_applied_meta()
                if int(rt_applied_meta_current.seq) != int(rt_applied_seq_local):
                    _configure_image_bounds(IMG_SERIES_TAG, rt_applied_meta_current)
                    rt_applied_seq_local = int(rt_applied_meta_current.seq)
                with gui_lock:
                    seq_locked = int(gui_latest_seq.value)
                    idx = int(gui_latest_idx.value)
                    if idx in (0, 1):
                        base = idx * gui_h * gui_w
                        src_flat = np.frombuffer(gui_dbuf, dtype=np.float32, count=gui_h * gui_w, offset=base * 4)
                        gui_frame.reshape(-1)[:] = src_flat
                        alpha_flat = np.frombuffer(
                            gui_alpha_dbuf,
                            dtype=np.float32,
                            count=gui_h * gui_w,
                            offset=base * 4,
                        )
                        gui_alpha_frame.reshape(-1)[:] = alpha_flat
                        prof_base = idx * RANGE_PROFILE_COUNT * fft_plot_h
                        prof_flat = np.frombuffer(
                            gui_prof_dbuf,
                            dtype=np.float32,
                            count=RANGE_PROFILE_COUNT * fft_plot_h,
                            offset=prof_base * 4,
                        )
                        rangefft_frame.reshape(-1)[:] = prof_flat
                        angle_base = idx * fft_plot_h * ANGLEFFT_BINS
                        angle_flat = np.frombuffer(
                            gui_angle_diag_dbuf,
                            dtype=np.float32,
                            count=fft_plot_h * ANGLEFFT_BINS,
                            offset=angle_base * 4,
                        )
                        anglefft_frame.reshape(-1)[:] = angle_flat
                        doppler_base = idx * fft_plot_h * DOPPLERFFT_BINS
                        doppler_flat = np.frombuffer(
                            gui_doppler_diag_dbuf,
                            dtype=np.float32,
                            count=fft_plot_h * DOPPLERFFT_BINS,
                            offset=doppler_base * 4,
                        )
                        dopplerfft_frame.reshape(-1)[:] = doppler_flat
                    gui_last_seq = seq_locked

                denom = (vis_vmax - vis_vmin)
                if denom < 1e-6:
                    denom = 1e-6

                active_lut = velocity_lut if int(vis_heatmap_mode) == 1 else jet_lut
                active_lut_last = float(active_lut.shape[0] - 1)
                if int(vis_heatmap_mode) == 1:
                    valid_heatmap = np.isfinite(gui_frame) & np.isfinite(gui_alpha_frame) & (gui_alpha_frame > np.float32(0.0))
                    np.subtract(gui_frame, float(vis_vmin), out=norm_frame, where=valid_heatmap)
                    norm_frame[~valid_heatmap] = np.float32(0.0)
                    norm_frame *= float(1.0 / denom)
                    np.clip(norm_frame, 0.0, 1.0, out=norm_frame)
                else:
                    np.subtract(gui_frame, float(vis_vmin), out=norm_frame)
                    norm_frame *= float(1.0 / denom)
                    np.clip(norm_frame, 0.0, 1.0, out=norm_frame)
                np.nan_to_num(norm_frame, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
                np.multiply(norm_frame, active_lut_last, out=norm_frame)
                np.rint(norm_frame, out=norm_frame)
                lut_idx[:, :] = norm_frame
                np.take(active_lut, lut_idx[::-1, :], axis=0, out=rgba_frame)
                if int(vis_heatmap_mode) == 1:
                    alpha_display = gui_alpha_frame[::-1, :]
                    np.nan_to_num(alpha_display, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
                    np.clip(alpha_display, 0.0, 1.0, out=alpha_display)
                    valid_alpha = alpha_display > np.float32(0.0)
                    alpha_out = rgba_frame[:, :, 3]
                    alpha_out.fill(np.float32(0.0))
                    min_opacity = np.float32(HEATMAP_VELOCITY_MIN_OPACITY)
                    np.multiply(alpha_display, np.float32(1.0) - min_opacity, out=alpha_out, where=valid_alpha)
                    alpha_out[valid_alpha] += min_opacity
                    rgba_frame[~valid_alpha, :3] = velocity_nodata_rgba[:3]
                tex_np[:] = rgba_frame.reshape(-1)
                dpg.set_value(TEX_TAG, tex_buf)

                if (now - fft_plot_last_t) >= fft_plot_period_s:
                    fft_plot_last_t = now
                    max_r_bin, max_r_m = _fft_visible_range_from_view(vis_rmax, vis_fft_view_full)
                    x_cache_key = (int(max_r_bin), round(float(max_r_m), 6))
                    x_range_m = x_range_cache.get(x_cache_key)
                    if x_range_m is None:
                        if max_r_bin <= 1:
                            x_range_m = [0.0]
                        else:
                            x_range_m = np.linspace(0.0, float(max_r_m), int(max_r_bin), dtype=np.float32).tolist()
                        x_range_cache[x_cache_key] = x_range_m
                    fft_slice = rangefft_frame[:, :max_r_bin]
                    if vis_fft_mode_db:
                        fft_plot_vals = fft_slice
                    else:
                        fft_plot_vals = fft_lin_frame[:, :max_r_bin]
                        np.multiply(fft_slice, np.float32(0.1), out=fft_plot_vals)
                        np.power(np.float32(10.0), fft_plot_vals, out=fft_plot_vals)
                    for ant_i in range(min(RANGE_PROFILE_COUNT, int(fft_plot_vals.shape[0]))):
                        line_tag = RANGEFFT_LINE_TAGS[ant_i]
                        if dpg.does_item_exist(line_tag):
                            y_vals = fft_plot_vals[ant_i, :].tolist()
                            dpg.set_value(line_tag, [x_range_m, y_vals])
                    vis_angle_diag_vmin, vis_angle_diag_vmax = _read_diag_scale(
                        IN_ANGLE_VMIN,
                        IN_ANGLE_VMAX,
                        vis_angle_diag_vmin,
                        vis_angle_diag_vmax,
                    )
                    vis_doppler_diag_vmin, vis_doppler_diag_vmax = _read_diag_scale(
                        IN_DOPPLER_VMIN,
                        IN_DOPPLER_VMAX,
                        vis_doppler_diag_vmin,
                        vis_doppler_diag_vmax,
                    )
                    angle_diag_norm_to_peak = _diag_norm_enabled(CHK_ANGLE_NORM, angle_diag_norm_to_peak)
                    doppler_diag_norm_to_peak = _diag_norm_enabled(CHK_DOPPLER_NORM, doppler_diag_norm_to_peak)
                    _update_diag_colormap_scale(ANGLEFFT_CMAP_SCALE_TAG, vis_angle_diag_vmin, vis_angle_diag_vmax)
                    _update_diag_colormap_scale(DOPPLERFFT_CMAP_SCALE_TAG, vis_doppler_diag_vmin, vis_doppler_diag_vmax)

                    angle_peak_db = _normalize_diag_map_to_lut(
                        anglefft_frame,
                        angle_diag_norm_frame,
                        vmin=vis_angle_diag_vmin,
                        vmax=vis_angle_diag_vmax,
                        norm_to_peak=angle_diag_norm_to_peak,
                    )
                    np.multiply(angle_diag_norm_frame, lut_last, out=angle_diag_norm_frame)
                    np.rint(angle_diag_norm_frame, out=angle_diag_norm_frame)
                    angle_diag_lut_idx[:, :] = angle_diag_norm_frame
                    np.take(jet_lut, angle_diag_lut_idx[::-1, :], axis=0, out=angle_diag_rgba_frame)
                    angle_diag_tex_np[:] = angle_diag_rgba_frame.reshape(-1)
                    if dpg.does_item_exist(ANGLEFFT_TEX_TAG):
                        dpg.set_value(ANGLEFFT_TEX_TAG, angle_diag_tex_buf)

                    doppler_peak_db = _normalize_diag_map_to_lut(
                        dopplerfft_frame,
                        doppler_diag_norm_frame,
                        vmin=vis_doppler_diag_vmin,
                        vmax=vis_doppler_diag_vmax,
                        norm_to_peak=doppler_diag_norm_to_peak,
                    )
                    np.multiply(doppler_diag_norm_frame, lut_last, out=doppler_diag_norm_frame)
                    np.rint(doppler_diag_norm_frame, out=doppler_diag_norm_frame)
                    doppler_diag_lut_idx[:, :] = doppler_diag_norm_frame
                    np.take(jet_lut, doppler_diag_lut_idx[::-1, :], axis=0, out=doppler_diag_rgba_frame)
                    doppler_diag_tex_np[:] = doppler_diag_rgba_frame.reshape(-1)
                    if dpg.does_item_exist(DOPPLERFFT_TEX_TAG):
                        dpg.set_value(DOPPLERFFT_TEX_TAG, doppler_diag_tex_buf)

                    if dpg.does_item_exist(ANGLEFFT_YAXIS_TAG):
                        dpg.set_axis_limits(ANGLEFFT_YAXIS_TAG, 0.0, float(max_r_m))
                    if dpg.does_item_exist(DOPPLERFFT_YAXIS_TAG):
                        dpg.set_axis_limits(DOPPLERFFT_YAXIS_TAG, 0.0, float(max_r_m))
                    if dpg.does_item_exist(ANGLEFFT_PROFILE_YAXIS_TAG):
                        dpg.set_axis_limits(ANGLEFFT_PROFILE_YAXIS_TAG, float(vis_angle_diag_vmin), float(vis_angle_diag_vmax))
                    if dpg.does_item_exist(DOPPLERFFT_PROFILE_YAXIS_TAG):
                        dpg.set_axis_limits(DOPPLERFFT_PROFILE_YAXIS_TAG, float(vis_doppler_diag_vmin), float(vis_doppler_diag_vmax))

                    try:
                        angle_selected_bin = int(dpg.get_value(IN_ANGLE_BIN)) if dpg.does_item_exist(IN_ANGLE_BIN) else int(angle_selected_bin)
                    except Exception:
                        angle_selected_bin = 0
                    angle_selected_bin = max(0, min(int(angle_selected_bin), int(max_r_bin) - 1, int(fft_plot_h) - 1))
                    if dpg.does_item_exist(IN_ANGLE_BIN) and int(dpg.get_value(IN_ANGLE_BIN)) != int(angle_selected_bin):
                        dpg.set_value(IN_ANGLE_BIN, int(angle_selected_bin))
                    try:
                        doppler_selected_bin = int(dpg.get_value(IN_DOPPLER_BIN)) if dpg.does_item_exist(IN_DOPPLER_BIN) else int(doppler_selected_bin)
                    except Exception:
                        doppler_selected_bin = 0
                    doppler_selected_bin = max(0, min(int(doppler_selected_bin), int(max_r_bin) - 1, int(fft_plot_h) - 1))
                    if dpg.does_item_exist(IN_DOPPLER_BIN) and int(dpg.get_value(IN_DOPPLER_BIN)) != int(doppler_selected_bin):
                        dpg.set_value(IN_DOPPLER_BIN, int(doppler_selected_bin))

                    if angle_single_bin and dpg.does_item_exist(ANGLEFFT_PROFILE_LINE_TAG):
                        angle_x_vals = np.nan_to_num(angle_axis_diag, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False).tolist()
                        angle_y_vals = anglefft_frame[int(angle_selected_bin), :]
                        if angle_diag_norm_to_peak:
                            angle_y_vals = angle_y_vals - np.float32(angle_peak_db)
                        dpg.set_value(ANGLEFFT_PROFILE_LINE_TAG, [angle_x_vals, angle_y_vals.astype(np.float32, copy=False).tolist()])
                    if doppler_single_bin and dpg.does_item_exist(DOPPLERFFT_PROFILE_LINE_TAG):
                        doppler_y_vals = dopplerfft_frame[int(doppler_selected_bin), :]
                        if doppler_diag_norm_to_peak:
                            doppler_y_vals = doppler_y_vals - np.float32(doppler_peak_db)
                        dpg.set_value(DOPPLERFFT_PROFILE_LINE_TAG, [doppler_axis_diag.astype(np.float32, copy=False).tolist(), doppler_y_vals.astype(np.float32, copy=False).tolist()])
                if DEBUG_STATS:
                    img_updates += 1

            # 3. TRACK OVERLAY UPDATE (latest-wins shared state)
            tracks_seq_now = int(gui_tracks_seq.value)
            if tracks_seq_now != tracks_last_seq:
                with gui_tracks_lock:
                    tracks_seq_locked = int(gui_tracks_seq.value)
                    n_tracks = max(0, min(int(gui_tracks_count.value), track_max_shared))
                    tracks_out = []
                    for tr_i in range(n_tracks):
                        base = tr_i * 4
                        base2 = tr_i * 2
                        tr_id = int(gui_tracks_meta_view[base + 0])
                        tr_confirmed = bool(int(gui_tracks_meta_view[base + 1]))
                        tr_age = int(gui_tracks_meta_view[base + 2])
                        tr_missed = int(gui_tracks_meta_view[base + 3])
                        tr_x = float(gui_tracks_xy_view[base + 0])
                        tr_y = float(gui_tracks_xy_view[base + 1])
                        tr_vx = float(gui_tracks_xy_view[base + 2])
                        tr_vy = float(gui_tracks_xy_view[base + 3])
                        tr_motion_state_code = int(gui_tracks_state_view[base2 + 0])
                        tr_has_stop = bool(int(gui_tracks_state_view[base2 + 1]))
                        tr_stop_x = float(gui_tracks_stop_xy_view[base2 + 0])
                        tr_stop_y = float(gui_tracks_stop_xy_view[base2 + 1])
                        if not tr_has_stop or not np.isfinite(tr_stop_x) or not np.isfinite(tr_stop_y):
                            tr_has_stop = False
                            tr_stop_x = float("nan")
                            tr_stop_y = float("nan")
                        tracks_out.append(
                            {
                                "id": tr_id,
                                "x": tr_x,
                                "y": tr_y,
                                "vx": tr_vx,
                                "vy": tr_vy,
                                "confirmed": tr_confirmed,
                                "age": tr_age,
                                "missed": tr_missed,
                                "motion_state_code": tr_motion_state_code,
                                "has_stop": tr_has_stop,
                                "stop_x": tr_stop_x,
                                "stop_y": tr_stop_y,
                            }
                        )
                    tracks_last_seq = tracks_seq_locked

                conf_x: list[float] = []
                conf_y: list[float] = []
                unconf_x: list[float] = []
                unconf_y: list[float] = []
                moving_x: list[float] = []
                moving_y: list[float] = []
                stopped_x: list[float] = []
                stopped_y: list[float] = []
                unknown_x: list[float] = []
                unknown_y: list[float] = []
                stop_mark_x: list[float] = []
                stop_mark_y: list[float] = []
                vel_x: list[float] = []
                vel_y: list[float] = []
                for tr in tracks_out:
                    tx = float(tr["x"])
                    ty = float(tr["y"])
                    motion_state_code = int(tr.get("motion_state_code", 0))
                    has_stop = bool(tr.get("has_stop", False))
                    if bool(tr["confirmed"]):
                        conf_x.append(tx)
                        conf_y.append(ty)
                    else:
                        unconf_x.append(tx)
                        unconf_y.append(ty)

                    if motion_state_code == 1:
                        moving_x.append(tx)
                        moving_y.append(ty)
                        vel_x.extend([tx, tx + float(tr["vx"]) * TRACK_VEL_SCALE, float("nan")])
                        vel_y.extend([ty, ty + float(tr["vy"]) * TRACK_VEL_SCALE, float("nan")])
                    elif motion_state_code == 2:
                        stopped_x.append(tx)
                        stopped_y.append(ty)
                    else:
                        unknown_x.append(tx)
                        unknown_y.append(ty)

                    if has_stop:
                        sx = float(tr.get("stop_x", float("nan")))
                        sy = float(tr.get("stop_y", float("nan")))
                        if np.isfinite(sx) and np.isfinite(sy):
                            stop_mark_x.append(sx)
                            stop_mark_y.append(sy)

                if dpg.does_item_exist(TRACK_SCATTER_CONF_TAG):
                    dpg.set_value(TRACK_SCATTER_CONF_TAG, [conf_x, conf_y])
                if dpg.does_item_exist(TRACK_SCATTER_UNCONF_TAG):
                    dpg.set_value(TRACK_SCATTER_UNCONF_TAG, [unconf_x, unconf_y])
                if dpg.does_item_exist(TRACK_SCATTER_MOVING_TAG):
                    dpg.set_value(TRACK_SCATTER_MOVING_TAG, [moving_x, moving_y])
                if dpg.does_item_exist(TRACK_SCATTER_STOPPED_TAG):
                    dpg.set_value(TRACK_SCATTER_STOPPED_TAG, [stopped_x, stopped_y])
                if dpg.does_item_exist(TRACK_SCATTER_UNKNOWN_TAG):
                    dpg.set_value(TRACK_SCATTER_UNKNOWN_TAG, [unknown_x, unknown_y])
                if dpg.does_item_exist(TRACK_STOP_MARKER_TAG):
                    dpg.set_value(TRACK_STOP_MARKER_TAG, [stop_mark_x, stop_mark_y])
                if dpg.does_item_exist(TRACK_VEL_SERIES_TAG):
                    dpg.set_value(TRACK_VEL_SERIES_TAG, [vel_x, vel_y])

                if supports_plot_annotation and track_annotation_tags:
                    n_labels = min(len(track_annotation_tags), len(tracks_out))
                    for ann_i, ann_tag in enumerate(track_annotation_tags):
                        if not dpg.does_item_exist(ann_tag):
                            continue
                        try:
                            if ann_i < n_labels:
                                tr = tracks_out[ann_i]
                                motion_code = int(tr.get("motion_state_code", 0))
                                if motion_code == 1:
                                    suffix = " M"
                                elif motion_code == 2:
                                    suffix = " S"
                                else:
                                    suffix = " U"
                                speed = float(np.hypot(float(tr["vx"]), float(tr["vy"])))
                                label = f"ID {int(tr['id'])}{suffix}"
                                if motion_code == 1:
                                    label = f"{label} {speed:.2f}"
                                dpg.configure_item(
                                    ann_tag,
                                    label=label,
                                    default_value=(float(tr["x"]), float(tr["y"])),
                                    show=True,
                                )
                            else:
                                dpg.configure_item(ann_tag, show=False)
                        except Exception:
                            supports_plot_annotation = False
                            for hide_tag in track_annotation_tags:
                                if dpg.does_item_exist(hide_tag):
                                    dpg.configure_item(hide_tag, show=False)
                            break

            _maybe_rearm_mmwave_stream(now)
            dpg.render_dearpygui_frame()

    except KeyboardInterrupt:
        pass
    finally:
        _shutdown()



        
if __name__ == "__main__":
    mp.freeze_support()
    main()





