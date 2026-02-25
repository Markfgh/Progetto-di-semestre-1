import socket
import time
import queue as pyqueue
from pathlib import Path
from multiprocessing import Process, Queue, Value
from array import array
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray, Synchronized
import yaml
import numpy as np
import dearpygui.dearpygui as dpg
import scipy.fft as fft
# ----------------------------
# RAM usage helper (per-process)
# ----------------------------
import os
import sys

def _rss_bytes_pid(pid: int) -> int:
    """Return resident set size (RSS) in bytes for a given pid. Best-effort."""
    if pid is None or pid <= 0:
        return 0

    # 1) psutil if available (fast + portable)
    try:
        import psutil  # type: ignore
        p = psutil.Process(pid)
        return int(p.memory_info().rss)
    except Exception:
        pass

    # 2) Windows fallback via ctypes (no deps)
    if os.name == "nt":
        try:
            import ctypes
            import ctypes.wintypes as wt

            PROCESS_QUERY_INFORMATION = 0x0400
            PROCESS_VM_READ = 0x0010

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wt.DWORD),
                    ("PageFaultCount", wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            OpenProcess = ctypes.windll.kernel32.OpenProcess
            GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
            CloseHandle = ctypes.windll.kernel32.CloseHandle

            h = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, int(pid))
            if not h:
                return 0
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            ok = GetProcessMemoryInfo(h, ctypes.byref(counters), counters.cb)
            CloseHandle(h)
            if ok:
                return int(counters.WorkingSetSize)
        except Exception:
            return 0

    # 3) Linux fallback via /proc
    try:
        if sys.platform.startswith("linux"):
            with open(f"/proc/{int(pid)}/statm", "r", encoding="utf-8") as f:
                parts = f.read().strip().split()
            if len(parts) >= 2:
                rss_pages = int(parts[1])
                page = getattr(os, 'sysconf', lambda x: 4096)("SC_PAGE_SIZE") if hasattr(os, 'sysconf') else 4096
                return rss_pages * page
    except Exception:
        pass

    return 0


def _cpu_percent_pid(pid: int, cpu_state: dict) -> float:
    """Best-effort process CPU% using psutil, non-blocking."""
    if pid is None or pid <= 0:
        return 0.0
    try:
        import psutil  # type: ignore
    except Exception:
        return 0.0
    p = cpu_state.get(int(pid))
    if p is None:
        try:
            p = psutil.Process(int(pid))
            p.cpu_percent(None)
            cpu_state[int(pid)] = p
            return 0.0
        except Exception:
            return 0.0
    try:
        return float(p.cpu_percent(None))
    except Exception:
        cpu_state.pop(int(pid), None)
        return 0.0


def _normalize_priority_level(level: str) -> str:
    level_s = str(level).strip().lower()
    aliases = {
        "idle": "idle",
        "lowest": "idle",
        "background": "idle",
        "low": "below_normal",
        "below": "below_normal",
        "below_normal": "below_normal",
        "normal": "normal",
        "above": "above_normal",
        "above_normal": "above_normal",
        "high": "high",
        "higher": "high",
        "realtime": "realtime",
        "real_time": "realtime",
        "real-time": "realtime",
        "rt": "realtime",
    }
    return aliases.get(level_s, level_s)


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

from Dsp_processing import selection_from_yaml_dict, build_windows
#import dpnp as dp


# --- CONFIGURAZIONE ---
CFG_PATH = Path(__file__).with_name("Config.yaml")  # <-- nome esatto del tuo file
with CFG_PATH.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


# --- CONFIGURAZIONE FISICA ---
C = float(cfg["radar"]["c"])
FS = float(cfg["radar"]["fs"])
SLOPE = float(cfg["radar"]["slope"])

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
# vmin/vmax: valori di BASE (default) per la scala colori (NON hard limit)
VMIN = float(cfg["display"]["vmin"])
VMAX = float(cfg["display"]["vmax"])

# range_max / crossrange_max: HARD LIMIT (0 .. max config)
RMAX_HARD_MAX = float(cfg["display"]["range_max"])
XMAX_HARD_MAX = float(cfg.get("display", {}).get("crossrange_max", RMAX_HARD_MAX))

# valori iniziali di visualizzazione (partono dai limiti config, ma poi l'utente puÃ² ridurli fino a 0)
RANGE_MAX_DISPLAY = float(cfg["display"]["range_max"])
CROSSRANGE_MAX_DISPLAY = float(cfg.get("display", {}).get("crossrange_max", RANGE_MAX_DISPLAY))
# --- CODE QUEUE ---
DEBUG_STATS = bool(cfg["debug"]["debug_stats"])

# --- WORKERS FFT ---
FFT_WORKERS = int(cfg.get("dsp", {}).get("fft_workers", cfg.get("fft", {}).get("workers", 6)))

# --- PROCESS PRIORITY ---
prio_cfg = cfg.get("process", {}).get("priority", {}) or {}
PRIO_ENABLED = bool(prio_cfg.get("enabled", False))
PRIO_MAIN = str(prio_cfg.get("main", "normal"))
PRIO_RX = str(prio_cfg.get("rx", "normal"))
PRIO_LOG = str(prio_cfg.get("logger", prio_cfg.get("log", "normal")))
PRIO_DSP = str(prio_cfg.get("dsp", "normal"))

# --- SAR capture-only parameters ---
sar_cfg = cfg.get("sar", {}) or {}
# Backward-compat: if not present, reuse old 'logger.settling_delay_s'
SETTLING_DELAY_S = float(sar_cfg.get("settling_delay_s", 0.4))
FRAMES_PER_POSITION = int(sar_cfg.get("frames_per_position", 8))


# ----------------------------
# FUNZIONI DI ELABORAZIONE
# ----------------------------
def radar_rx(
    cmd_queue: Queue,
    free_slots: Queue,
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
    lost_pkts: Synchronized,
    rx_pkts: Synchronized,
    rx_put_drops: Synchronized,
    rx_frames_ok: Synchronized,
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
      - incoerenza byte_count: frame corrotto -> SCARTATO

    Performance:
      - zero-copy interprocess: RX scrive raw frame in shared memory (ring di slot)
      - nessuna Queue per i frame (nÃ© verso DSP nÃ© verso Logger)
      - se non ci sono slot liberi: drop (no block), ma manteniamo l'allineamento consumando i byte

    Capture:
      - trigger via cmd_queue ("CAPTURE", pos_id)
      - dopo settling_delay_s: cap_active=1 e RX tagga slot_pos_id
      - il logger salva X frame validi per pos_id (puÃ² perdere frame, quindi dura piÃ¹ a lungo se necessario)
    """
    PC_IP = "192.168.33.30"
    PORT = 4098
    HEADER_LEN = 10
    RCVBUF_BYTES = 256 * 1024 * 1024

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RCVBUF_BYTES)
    sock.bind((PC_IP, PORT))
    sock.settimeout(0.2)

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
    publish_seq = 0

    # Capture control
    pending = False
    pending_start_t = 0.0
    pending_pos_id = 0
    def _poll_commands(now_perf: float) -> None:
        nonlocal pending, pending_start_t, pending_pos_id
        while True:
            try:
                cmd = cmd_queue.get_nowait()
            except pyqueue.Empty:
                break
            if not cmd:
                continue
            if cmd[0] == "CAPTURE":
                pending_pos_id = int(cmd[1])
                pending_start_t = now_perf + max(0.0, float(settling_delay_s))
                pending = True

                # reset capture counters shared
                with cap_saved.get_lock():
                    cap_saved.value = 0
                with cap_pos_id.get_lock():
                    cap_pos_id.value = pending_pos_id

                # bump cap_id (unique capture session)
                with cap_id.get_lock():
                    cap_id.value = (int(cap_id.value) + 1) & 0xFFFFFFFF

                # not active until settling passes
                with cap_active.get_lock():
                    cap_active.value = 0

    def _maybe_enable_capture(now_perf: float) -> None:
        nonlocal pending
        if pending and (now_perf >= pending_start_t):
            pending = False
            with cap_active.get_lock():
                cap_active.value = 1

    def _free_current_slot():
        nonlocal curr_slot, frame_view, have_slot
        # Drop only local references; do not release published slots here.
        curr_slot = None
        frame_view = None
        have_slot = False

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
                # Prepare metadata first
                m = 1  # DSP always consumes
                if int(cap_active.value) == 1:
                    slot_pos_id[slot] = int(cap_pos_id.value)
                    m |= 2  # LOGGER also consumes during capture
                else:
                    slot_pos_id[slot] = -1

                # --- ATOMIC PUBLISH (coherent READY+slot+seq) ---
                with publish_lock:
                    next_seq = int(publish_seq) + 1
                    slot_ok[slot] = 1
                    slot_usemask[slot] = m
                    slot_pub_seq[slot] = next_seq
                    slot_state[slot] = 1  # READY (set before seq bump)
                    publish_seq = next_seq
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

    while not stop_evt.is_set():
        now_perf = time.perf_counter()
        _poll_commands(now_perf)
        _maybe_enable_capture(now_perf)

        try:
            n_bytes, _ = sock.recvfrom_into(packet_mv)
            pkts_local += 1
            now = time.perf_counter()
            if now - t_flush >= 0.1:
                with rx_pkts.get_lock():
                    rx_pkts.value += pkts_local
                pkts_local = 0
                t_flush = now
        except socket.timeout:
            continue

        if n_bytes <= HEADER_LEN:
            continue

        if payload_len_ref is None:
            payload_len_ref = n_bytes - HEADER_LEN

        # --- parse header ---
        seq = int.from_bytes(packet_view[0:4], "little", signed=False)

        # uint48 little-endian
        bc = int.from_bytes(packet_view[4:10], "little", signed=False) & (MOD48 - 1)

        # --- sanity: out-of-order ---
        if last_seq is not None and seq <= last_seq:
            # ignore old/reordered packet
            continue

        # --- byte_count check (robust alignment) ---
        if last_byte_count is not None and payload_len_ref is not None:
            expected = (last_byte_count + payload_len_ref) % MOD48
            if bc != expected:
                frame_ok = False
        last_byte_count = bc

        # --- GAP handling (consume missing bytes to keep frame alignment) ---
        if last_seq is not None:
            gap = seq - last_seq - 1
            if gap > 0 and payload_len_ref is not None:
                with lost_pkts.get_lock():
                    lost_pkts.value += int(gap)
                frame_ok = False

                bytes_missing = int(gap) * int(payload_len_ref)
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
    log_bytes: Synchronized,
    stop_evt,
    out_dir_s: str,
    frames_per_position: int,
    block_frames: int = 16,
):
    """
    Logger capture-only (reworked for performance):
      - produce 1 file .bin per click: capture_pos{pos_id}.bin
      - scrive a blocchi (block_frames) per ridurre overhead I/O
      - salva SOLO frame validi (slot_ok==1) e SOLO quando cap_active==1
      - deve arrivare a frames_per_position frame completi per la posizione corrente.
        Se perde frame (ring overwrite / race), semplicemente continua finchÃ© non raggiunge X.
    """
    out_dir = Path(out_dir_s)
    out_dir.mkdir(parents=True, exist_ok=True)

    shm_view = memoryview(shm_frames).cast("B")

    last_seen_cap_id = int(cap_id.value)  # avoid auto-start at process boot
    pending_cap_id = None
    pending_pos_id = None

    fbin = None
    buf = None
    buf_used = 0
    saved_local = 0
    pos_local = -1

    # scan pointer to avoid always starting at 0
    scan_i = 0
    n_slots = len(slot_state)
    LOGGER_BIT = 2
    LOGGER_BUSY_BIT = 0x80

    def _close_file():
        nonlocal fbin, buf, buf_used, saved_local, pos_local
        if fbin is not None:
            try:
                if buf_used and buf is not None:
                    fbin.write(buf[:buf_used])
                    if log_bytes is not None:
                        with log_bytes.get_lock():
                            log_bytes.value += int(buf_used)
                fbin.flush()
            except Exception:
                pass
            try:
                fbin.close()
            except Exception:
                pass
        fbin = None
        buf = None
        buf_used = 0
        saved_local = 0
        pos_local = -1

    def _open_file(pos_id: int, capid: int):
        nonlocal fbin, buf, buf_used, saved_local, pos_local
        _close_file()
        pos_local = int(pos_id)
        saved_local = 0
        buf_used = 0
        buf = bytearray(BYTES_PER_FRAME * int(block_frames))
        p = out_dir / f"capture_pos{pos_local}.bin"
        fbin = open(p, "wb", buffering=1024 * 1024)
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

    while not stop_evt.is_set():
        capid = int(cap_id.value)
        posid = int(cap_pos_id.value)

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
            if pending_cap_id is None or int(cap_active.value) != 1:
                time.sleep(0.002)
                continue
            _open_file(int(pending_pos_id), int(pending_cap_id))
            pending_cap_id = None
            pending_pos_id = None

        # If capture was externally stopped, close the file and idle
        if int(cap_active.value) != 1:
            _close_file()
            time.sleep(0.002)
            continue

        # scan a bounded number of slots per tick to keep CPU low
        did_work = False
        while saved_local < int(frames_per_position):
            s, ok = _claim_slot_for_logger(int(posid))
            if s < 0:
                break

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

            # always release logger ownership, even for corrupt slot
            _finalize_logger_slot(int(s))

        # stop condition: reached target
        if saved_local >= int(frames_per_position):
            if buf_used:
                fbin.write(buf[:buf_used])
                if log_bytes is not None:
                    with log_bytes.get_lock():
                        log_bytes.value += int(buf_used)
                buf_used = 0
            try:
                fbin.flush()
            except Exception:
                pass
            _close_file()
            with cap_active.get_lock():
                cap_active.value = 0
            continue


        if not did_work:
            time.sleep(0.002)


# ----------------------------
# WORKER DSP (latest-wins)
# ----------------------------
def dsp_worker(
    free_slots: Queue,
    shm_frames,
    slot_state,
    slot_ok,
    slot_usemask,
    slot_pub_seq,
    publish_lock,
    gui_dbuf,
    gui_h: int,
    gui_w: int,
    gui_latest_idx: Synchronized,
    gui_latest_seq: Synchronized,
    gui_lock,
    dsp_skip: Synchronized,
    dsp_ms_avg: Synchronized,
    dsp_ms_p95: Synchronized,
    stop_evt,
):
    # --- SETUP STATIC ---
    selection = selection_from_yaml_dict(cfg)
    window_range, window_angle = build_windows(selection, samples=SAMPLES, virtual_ant=VIRTUAL_ANT)

    # Costanti
    dr = C * FS / (2.0 * SLOPE * NFFT_RANGE)
    max_bin = int(np.floor(RANGE_MAX_DISPLAY / dr))
    max_bin = max(1, min(max_bin, NFFT_RANGE // 2))

    # Parametri memoria
    i16_per_frame = BYTES_PER_FRAME // 2
    total_samples_needed = X_FRAMES * CHIRPS * SAMPLES * RX

    # --- PRE-ALLOCAZIONE MEMORIA (Fuori dal loop!) ---
    i16_batch = np.zeros((X_FRAMES, i16_per_frame), dtype=np.int16)
    complex_data = np.zeros(total_samples_needed, dtype=np.complex64)

    shm_view = memoryview(shm_frames).cast("B")
    n_slots = len(slot_state)
    heatmap_ema = None
    dsp_ms_samples = []

    if RX != 4:
        raise ValueError("DSP: conversione I/Q attuale assume RX=4 (packing IIIIQQQQ).")

    def _release_slot_dsp(s: int) -> None:
        """Clear DSP bit and free slot only when no other consumer needs it."""
        to_free = False
        try:
            with publish_lock:
                m = int(slot_usemask[s]) & (~1)
                slot_usemask[s] = m
                if m == 0:
                    slot_state[s] = 0
                    to_free = True
        except Exception:
            to_free = False
        if to_free:
            try:
                free_slots.put_nowait(int(s))
            except Exception:
                pass
    while True:
        if stop_evt.is_set():
            break

        # Snapshot coherent dei frame READY destinati al DSP
        ready = []
        with publish_lock:
            for s in range(n_slots):
                if int(slot_state[s]) != 1:
                    continue
                if (int(slot_usemask[s]) & 1) == 0:
                    continue
                ready.append((int(slot_pub_seq[s]), s, int(slot_ok[s])))

        if not ready:
            time.sleep(0.001)
            continue

        # Ordina per seq crescente: i più vecchi prima
        ready.sort(key=lambda t: t[0])

        # Se backlog > X_FRAMES, scarta i più vecchi (latest-wins)
        if len(ready) > X_FRAMES:
            to_drop = ready[:-X_FRAMES]
            ready = ready[-X_FRAMES:]
            if DEBUG_STATS and dsp_skip is not None:
                with dsp_skip.get_lock():
                    dsp_skip.value += len(to_drop)
            for _, s, _ in to_drop:
                _release_slot_dsp(int(s))

        # Togli eventuali corrotti dal batch da processare
        proc_slots = [s for _, s, ok in ready if ok == 1]
        for _, s, ok in ready:
            if ok != 1:
                _release_slot_dsp(int(s))
        if not proc_slots:
            continue

        n_proc = min(len(proc_slots), X_FRAMES)

        # --- 1. COPIA DA SHARED MEMORY (Veloce) ---
        for k, s in enumerate(proc_slots[:n_proc]):
            base = s * BYTES_PER_FRAME
            # Copia diretta nel buffer pre-allocato
            i16_batch[k, :] = np.frombuffer(shm_view[base : base + BYTES_PER_FRAME], dtype=np.int16)
            
        # Restituisci subito gli slot al RX (Pipelining)
        for s in proc_slots[:n_proc]:
            _release_slot_dsp(int(s))

        # --- 2. CONVERSIONE INT16 -> COMPLEX64 (Zero Alloc) ---
        flat_i16 = i16_batch[:n_proc, :].reshape(-1)
        n_blocks = flat_i16.size // 8
        
        # View strutturata (n, 8)
        block_view = flat_i16[:n_blocks*8].reshape(n_blocks, 8)
        
        # Copia nei canali Real e Imag del buffer complesso PRE-ALLOCATO
        # Nota: usiamo [:] per forzare la copia in-place senza riallocare
        n_cplx = n_proc * CHIRPS * SAMPLES * RX
        complex_view = complex_data[:n_cplx]
        complex_view.real[:] = block_view[:, :4].reshape(-1)  # type: ignore
        complex_view.imag[:] = block_view[:, 4:].reshape(-1)  # type: ignore
        # --- 3. PROCESSING ---
        t0_proc = time.perf_counter()
        heatmap_ema = process_buffer(
            complex_view, 
            n_proc,
            window_range, 
            window_angle, 
            heatmap_ema, 
            0.2, # alpha
            gui_dbuf,
            gui_h,
            gui_w,
            gui_latest_idx,
            gui_latest_seq,
            gui_lock,
            max_bin
        )
        t1_proc = time.perf_counter()
        if n_proc > 0 and DEBUG_STATS:
            ms_per_frame = ((t1_proc - t0_proc) * 1000.0) / float(n_proc)
            dsp_ms_samples.extend([ms_per_frame] * int(n_proc))
            if len(dsp_ms_samples) > 256:
                dsp_ms_samples = dsp_ms_samples[-256:]
            if dsp_ms_samples:
                with dsp_ms_avg.get_lock():
                    dsp_ms_avg.value = float(np.mean(dsp_ms_samples))
                with dsp_ms_p95.get_lock():
                    dsp_ms_p95.value = float(np.percentile(dsp_ms_samples, 95))

def process_buffer(
    raw_buffer,
    n_frames: int,
    w_range,
    w_angle,
    heatmap_ema,
    alpha,
    gui_dbuf,
    gui_h,
    gui_w,
    gui_latest_idx,
    gui_latest_seq,
    gui_lock,
    max_bin,
):

    try:
        # A. Reshape SENZA transpose (evita copia grossa)

        # Shape: (F, chirpsPerTx, TX, SAMPLES, RX)  -> samples Ã¨ axis=3

        data = raw_buffer.reshape(n_frames, CHIRPS // TX, TX, SAMPLES, RX)


        # B. DSP IN-PLACE

        # Clutter removal: sottrai la media lungo SAMPLES (axis=3)

        #data -= data.mean(axis=2, keepdims=True, dtype=np.complex64)

        # Finestra Range: w_range deve essere broadcastabile su axis=3 (shape: 1,1,1,SAMPLES,1)

        data *= w_range


        # C. RANGE FFT (samples axis=3)

        range_fft = fft.fft(data, n=NFFT_RANGE, axis=3, workers=FFT_WORKERS, overwrite_x=True)


        # Preparazione Virtual Array senza transpose+ascontiguousarray:

        # range_fft: (F, C, TX, R, RX) con R=NFFT_RANGE

        va = np.moveaxis(range_fft, 4, 3)     # -> (F, C, TX, RX, R)

        va = np.moveaxis(va, 4, 2)           # -> (F, C, R, TX, RX)

        virtual_array = va.reshape(n_frames, CHIRPS // TX, NFFT_RANGE, VIRTUAL_ANT)


        # Limitiamo i range bin al solo intervallo visualizzato prima della angle-FFT.
        # Evita lavoro inutile su bin che verrebbero comunque scartati.
        virtual_array = virtual_array[:, :, :max_bin, :]

        # Finestra Angolo IN-PLACE (w_angle: (1,1,1,VIRTUAL_ANT))
        virtual_array *= w_angle

        # D. ANGLE FFT
        angle_fft = fft.fft(virtual_array, n=NFFT_ANGLE, axis=-1, workers=FFT_WORKERS, overwrite_x=True)
        
        # Modulo quadro e media.
        # Poi applichiamo fftshift sulla heatmap 2D (molto piu piccola) invece che sul tensor 4D.
        re = angle_fft.real
        im = angle_fft.imag
        heatmap = (re * re + im * im).mean(axis=(0, 1))
        heatmap = np.fft.fftshift(heatmap, axes=-1)

        # E. EMA e Logaritmica
        if heatmap_ema is None:
            heatmap_ema = heatmap
        else:
            # EMA ottimizzata
            heatmap_ema *= (1.0 - alpha)
            heatmap_ema += (alpha * heatmap)
        
        # Conversione dB (crea copia, inevitabile ma veloce su array piccolo)
        heatmap_db = 10 * np.log10(heatmap_ema + 1e-12)
        
        # Normalizzazione e taglio
        view_db = heatmap_db
        if view_db.size > 0:
            mx = np.max(view_db)
            view_db -= mx

        # publish verso GUI su double-buffer shared (latest-wins, lock breve)
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
            gui_latest_idx.value = next_idx
            gui_latest_seq.value = int(gui_latest_seq.value) + 1
        return heatmap_ema

    except Exception as e:
        print(f"[DSP ERR] {e}")
        return heatmap_ema




def main():
    # --- SETUP CODE E PROCESSI ---
    free_slots = Queue()

    # Ring slots (user requested 64)
    N_SLOTS = 64
    shm_frames = RawArray("B", N_SLOTS * BYTES_PER_FRAME)
    for i in range(N_SLOTS):
        free_slots.put(i)

    # Slot metadata (shared)
    # slot_state: 0=FREE, 1=READY
    slot_state = RawArray("b", N_SLOTS)
    slot_ok = RawArray("b", N_SLOTS)
    slot_pos_id = RawArray("i", N_SLOTS)
    for i in range(N_SLOTS):
        slot_state[i] = 0
        slot_ok[i] = 0
        slot_pos_id[i] = -1

    # slot_usemask: bit0=DSP, bit1=LOGGER (capture)
    slot_usemask = RawArray("B", N_SLOTS)
    slot_pub_seq = RawArray("Q", N_SLOTS)
    for i in range(N_SLOTS):
        slot_usemask[i] = 0
        slot_pub_seq[i] = 0

    publish_lock = mp.Lock()  # atomic publish lock (seq+slot+state)


    # Capture shared state
    cap_active = Value("i", 0)   # 1 while capturing
    cap_pos_id = Value("i", 0)   # current position id (from GUI)
    cap_id = Value("I", 0)       # increments each CAPTURE command
    cap_saved = Value("i", 0)    # frames saved for current position

    dr_shared = C * FS / (2.0 * SLOPE * NFFT_RANGE)
    gui_h = int(np.floor(RANGE_MAX_DISPLAY / dr_shared))
    gui_h = max(1, min(gui_h, NFFT_RANGE // 2))
    gui_w = int(NFFT_ANGLE)
    gui_dbuf = RawArray("f", 2 * gui_h * gui_w)
    gui_latest_idx = Value("i", -1)
    gui_latest_seq = Value("Q", 0)
    gui_lock = mp.Lock()

    cmd_q = Queue(maxsize=16)

    stop_evt = mp.Event()
    sar_pos_counter = Value("L", 0)  # GUI-only counter (pos id generator)

    # Stats
    lost_pkts = Value("L", 0)
    rx_pkts = Value("L", 0)
    rx_put_drops = Value("L", 0)
    rx_frames_ok = Value("L", 0)
    dsp_skip = Value("L", 0)
    dsp_ms_avg = Value("d", 0.0)
    dsp_ms_p95 = Value("d", 0.0)
    log_bytes = Value("L", 0)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(__file__).with_name("logs")
    out_dir = out_root / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- AVVIO PROCESSI ---
    p_rx = Process(
        target=radar_rx,
        args=(
            cmd_q,
            free_slots,
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
            lost_pkts,
            rx_pkts,
            rx_put_drops,
            rx_frames_ok,
            stop_evt,
            SETTLING_DELAY_S,
        ),
    )

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
            log_bytes,
            stop_evt,
            str(out_dir),
            FRAMES_PER_POSITION,
            16,  # block_frames
        ),
    )

    p_dsp = Process(
        target=dsp_worker,
        args=(
            free_slots,
            shm_frames,
            slot_state,
            slot_ok,
            slot_usemask,
            slot_pub_seq,
            publish_lock,
            gui_dbuf,
            gui_h,
            gui_w,
            gui_latest_idx,
            gui_latest_seq,
            gui_lock,
            dsp_skip,
            dsp_ms_avg,
            dsp_ms_p95,
            stop_evt,
        ),
    )

    p_rx.daemon = True
    p_log.daemon = True
    p_dsp.daemon = True
    p_rx.start()
    p_log.start()
    p_dsp.start()

    _apply_process_priority("MAIN", os.getpid(), PRIO_MAIN, PRIO_ENABLED)
    _apply_process_priority("RX", int(p_rx.pid or 0), PRIO_RX, PRIO_ENABLED)
    _apply_process_priority("LOG", int(p_log.pid or 0), PRIO_LOG, PRIO_ENABLED)
    _apply_process_priority("DSP", int(p_dsp.pid or 0), PRIO_DSP, PRIO_ENABLED)

    print(f"[LOGGER] dir: {out_dir}")

    # =========================================================================
    # GUI SETUP (RESPONSIVE - CLEAN)
    # =========================================================================

    dr_plot = C * FS / (2.0 * SLOPE * NFFT_RANGE)

    # display state (cartesian X-Y only)
    vis_vmin = float(VMIN)
    vis_vmax = float(VMAX)
    vis_rmax = float(RANGE_MAX_DISPLAY)
    vis_xmax = float(CROSSRANGE_MAX_DISPLAY)

    ui_dirty = True
    ui_dirty_t = 0.0
    ui_pending = {"vmin": vis_vmin, "vmax": vis_vmax, "rmax": vis_rmax, "xmax": vis_xmax}

    # 2) DearPyGui Init
    dpg.create_context()

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
    TAG_SIDEBAR     = "sidebar_col"
    TAG_CBAR_COL    = "cbar_col"

    TXT_STATS_TAG = "txt_stats"
    IN_VMIN, IN_VMAX = "in_vmin", "in_vmax"
    IN_RMAX, IN_XMAX = "in_rmax", "in_xmax"
    TXT_POS_TAG = "txt_pos_counter"
    TXT_LOG_TAG = "txt_log"

    TEX_TAG = "heat_tex"
    HEAT_PLOT_TAG = "heat_plot"
    XAXIS_TAG, YAXIS_TAG = "xaxis", "yaxis"
    IMG_SERIES_TAG = "img_series"
    CMAP_SCALE_TAG = "cmap_scale"

    # 4) Texture setup
    max_bin_tex = int(np.floor(RANGE_MAX_DISPLAY / dr_plot))
    max_bin_tex = max(1, min(max_bin_tex, NFFT_RANGE // 2))
    tex_w, tex_h = int(NFFT_ANGLE), int(max_bin_tex)
    tex_buf = array("f", [0.0]) * (tex_w * tex_h * 4)
    tex_np = np.frombuffer(tex_buf, dtype=np.float32)

    with dpg.texture_registry(show=False):
        dpg.add_dynamic_texture(
            width=tex_w,
            height=tex_h,
            default_value=[0.0, 0.0, 0.0, 1.0] * tex_w * tex_h,
            tag=TEX_TAG,
        )

    # 5) Callbacks
    def _shutdown():
        stop_evt.set()
        # allow workers to exit cleanly
        for p in (p_rx, p_log, p_dsp):
            try:
                p.join(timeout=0.2)
            except Exception:
                pass
        # hard-kill if still alive
        for p in (p_rx, p_log, p_dsp):
            try:
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass
        for p in (p_rx, p_log, p_dsp):
            try:
                p.join(timeout=0.2)
            except Exception:
                pass

    dpg.set_exit_callback(_shutdown)

    def _jet_rgba(norm01):
        x = np.clip(norm01, 0.0, 1.0)
        r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
        a = np.ones_like(r)
        return np.stack((r, g, b, a), axis=-1)

    def _apply_params(sender=None, app_data=None):
        nonlocal ui_dirty, ui_pending
        # callback ultraleggera: salva valori e setta flag
        # Nota:
        #  - vmin/vmax sono "base values" (non hard limit). Li lasciamo liberi.
        #  - rmax/xmax sono HARD LIMIT: 0 .. (range_max/crossrange_max da config)
        try:
            vmin = float(dpg.get_value(IN_VMIN))
            vmax = float(dpg.get_value(IN_VMAX))
            rmax = float(dpg.get_value(IN_RMAX))
            xmax = float(dpg.get_value(IN_XMAX))
        except (ValueError, TypeError):
            return  # mentre scrivi '-', '.', '' ecc.

        # HARD clamp su Rmax/Xmax
        rmax_cl = max(0.0, min(rmax, RMAX_HARD_MAX))
        xmax_cl = max(0.0, min(xmax, XMAX_HARD_MAX))

        if rmax_cl != rmax:
            dpg.set_value(IN_RMAX, rmax_cl)
        if xmax_cl != xmax:
            dpg.set_value(IN_XMAX, xmax_cl)

        ui_pending["vmin"] = vmin
        ui_pending["vmax"] = vmax
        ui_pending["rmax"] = rmax_cl
        ui_pending["xmax"] = xmax_cl

        ui_dirty = True


    def _on_capture():
        with sar_pos_counter.get_lock():
            sar_pos_counter.value += 1
            pid = int(sar_pos_counter.value)
        try:
            cmd_q.put_nowait(("CAPTURE", pid))
        except Exception:
            pass
        dpg.set_value(TXT_POS_TAG, f"Pos Counter: {pid}")

    def row(k, v):
        return f"| {k:<16} | {v:<18} |\n"

    # 6) Build UI (ONLY TABLE LAYOUT)
    with dpg.window(tag=TAG_MAIN_WINDOW):

        with dpg.table(header_row=False, resizable=True, policy=dpg.mvTable_SizingStretchProp):

            # Colonne: sidebar (fissa, min 320px), plot (stretch), colorbar (fissa, min 110px)
            dpg.add_table_column(init_width_or_weight=320, width_fixed=True)   # sidebar MIN 320px
            dpg.add_table_column(init_width_or_weight=1.0)                     # plot stretch
            dpg.add_table_column(init_width_or_weight=110, width_fixed=True)   # colorbar MIN 110px


            with dpg.table_row():
                CTRL_W = 240
                # --- SIDEBAR ---
                with dpg.table_cell():
                    with dpg.child_window(tag=TAG_SIDEBAR, width=-1, height=-1, border=True):
                        dpg.add_text("DISPLAY PARAMETERS", color=(255, 200, 0))
                        dpg.add_separator()
                        dpg.add_spacer(width=20)
                        dpg.add_input_float(label="Vmin (dB)", tag=IN_VMIN, default_value=vis_vmin, step=1.0, width=CTRL_W,callback=_apply_params, on_enter=False)
                        dpg.add_input_float(label="Vmax (dB)", tag=IN_VMAX, default_value=vis_vmax, step=1.0, width=CTRL_W,callback=_apply_params, on_enter=False)
                        dpg.add_input_float(label="Rmax (m)", tag=IN_RMAX, default_value=vis_rmax, step=0.5, width=CTRL_W,
                                          min_value=0.0, max_value=RMAX_HARD_MAX, min_clamped=True, max_clamped=True,
                                          callback=_apply_params, on_enter=False)
                        dpg.add_input_float(label="Xmax (m)", tag=IN_XMAX, default_value=vis_xmax, step=0.5, width=CTRL_W,
                                          min_value=0.0, max_value=XMAX_HARD_MAX, min_clamped=True, max_clamped=True,
                                          callback=_apply_params, on_enter=False)
                        dpg.add_spacer(height=14)
                        dpg.add_text("SAR CONTROL", color=(255, 200, 0))
                        dpg.add_separator()
                        dpg.add_button(label="CAPTURE FRAME", callback=_on_capture, width=-1, height=40)
                        dpg.add_text("Pos Counter: 0", tag=TXT_POS_TAG)
                        dpg.add_text("", tag=TXT_LOG_TAG, wrap=-1)

                        dpg.add_spacer(height=14)
                        dpg.add_text("SYSTEM STATS", color=(255, 200, 0))
                        dpg.add_separator()
                        dpg.add_text("Waiting for data...", tag=TXT_STATS_TAG, wrap=-1)
                        if font_mono:
                            dpg.bind_item_font(TXT_STATS_TAG, font_mono)

                # --- PLOT ---
                with dpg.table_cell():
                    with dpg.plot(tag=HEAT_PLOT_TAG, width=-1, height=-1, equal_aspects=True):
                        dpg.add_plot_axis(dpg.mvXAxis, label="X (m)", tag=XAXIS_TAG)
                        dpg.add_plot_axis(dpg.mvYAxis, label="Y (m)", tag=YAXIS_TAG)

                        # bounds in metri (li aggiorneremo quando cambia rmax/ang)
                        dpg.add_image_series(
                            TEX_TAG,
                            bounds_min=(-vis_xmax, 0.0),
                            bounds_max=(+vis_xmax, float(vis_rmax)),
                            tag=IMG_SERIES_TAG,
                            parent=YAXIS_TAG,
                        )

                # --- COLORBAR ---
                with dpg.table_cell():
                    with dpg.child_window(tag=TAG_CBAR_COL, width=-1, height=-1, border=True):
                        # width=-1 cosÃ¬ si adatta alla colonna; height=-1 cosÃ¬ prende tutta l'altezza
                        dpg.add_colormap_scale(
                            tag=CMAP_SCALE_TAG,
                            min_scale=vis_vmin,
                            max_scale=vis_vmax,
                            width=-1,
                            height=-1,
                            colormap=dpg.mvPlotColormap_Jet,
                        )

    # Applicazione parametri DOPO creazione items
    _apply_params()

    dpg.create_viewport(title="MIMO Radar Real-Time", width=1400, height=900)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window(TAG_MAIN_WINDOW, True)
    dpg.set_viewport_min_width(900)
    dpg.set_viewport_min_height(500)

    # --- LOOP PRINCIPALE ---
    t_mon = time.perf_counter()
    img_updates = 0
    t_img_start = time.perf_counter()
    lost_prev = 0
    pkts_prev = 0
    frames_ok_prev = 0
    log_bytes_prev = 0
    gui_last_seq = 0
    gui_frame = np.zeros((gui_h, gui_w), dtype=np.float32)
    ring_hwm = 0
    cpu_state = {}

    try:
        while dpg.is_dearpygui_running():
            now = time.perf_counter()


            # --- UI apply (throttled, non-blocking) ---
            if ui_dirty and (now - ui_dirty_t) >= 0.08:  # ~12.5 Hz, cambia 0.05..0.15 come vuoi
                ui_dirty = False
                ui_dirty_t = now

                vmin = ui_pending["vmin"]
                vmax = ui_pending["vmax"]
                # HARD clamp anche qui (ridondanza: protegge da valori inseriti via codice)
                rmax = max(0.0, min(ui_pending["rmax"], RMAX_HARD_MAX))
                xmax = max(0.0, min(ui_pending["xmax"], XMAX_HARD_MAX))
                if vmax <= vmin:
                    vmax = vmin + 1.0

                vis_vmin = vmin
                vis_vmax = vmax
                vis_rmax = rmax
                vis_xmax = xmax

                # update plot bounds (meters)
                if dpg.does_item_exist(IMG_SERIES_TAG):
                    dpg.configure_item(IMG_SERIES_TAG, bounds_min=(-vis_xmax, 0.0), bounds_max=(+vis_xmax, float(vis_rmax)))
                if dpg.does_item_exist(XAXIS_TAG) and dpg.does_item_exist(YAXIS_TAG):
                    dpg.set_axis_limits(XAXIS_TAG, -vis_xmax, +vis_xmax)
                    dpg.set_axis_limits(YAXIS_TAG, 0.0, float(vis_rmax))

                # colorbar
                if dpg.does_item_exist(CMAP_SCALE_TAG):
                    dpg.configure_item(CMAP_SCALE_TAG, min_scale=vis_vmin, max_scale=vis_vmax)

            
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
                ring_hwm = max(ring_hwm, int(ready_slots))

                with rx_put_drops.get_lock(): drop_f = rx_put_drops.value
                with rx_frames_ok.get_lock(): frames_ok_now = rx_frames_ok.value
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
                with log_bytes.get_lock(): log_bytes_now = int(log_bytes.value)
                log_mbps = ((log_bytes_now - log_bytes_prev) / dt_mon) / (1024.0 * 1024.0) if dt_mon > 0 else 0.0
                log_bytes_prev = log_bytes_now
                cpu_dsp = _cpu_percent_pid(int(p_dsp.pid or 0), cpu_state)
                cpu_log = _cpu_percent_pid(int(p_log.pid or 0), cpu_state)
                
                # --- AGGIORNAMENTO TESTI GUI (FORMATO TABELLA ASCII) ---
                stats_str = (
                    "+------------------+--------------------+\n"
                    + row("LOSS %", f"{loss_pct:.3f}%")
                    + row("RX pkt/s", f"{pkt_rate:.0f}")
                    + row("RX frames_ok", f"{frames_ok_rate:.1f}/s ({int(frames_ok_now)})")
                    + row("RX drops_ring", f"{int(drop_f)}")
                    + row("RING ready/slots", f"{ready_slots}/{N_SLOTS} HWM {ring_hwm}")
                    + row("DSP ms avg/p95", f"{dsp_avg_now:.2f} / {dsp_p95_now:.2f}")
                    + row("DSP skip", f"{dsp_skip_now}")
                    + row("GUI Hz", f"{img_hz:.1f}")
                    + row("CAP active pos", f"{int(cap_active.value)} {int(cap_pos_id.value)}")
                    + row("CAP saved/target", f"{int(cap_saved.value)}/{FRAMES_PER_POSITION}")
                    + row("LOG MB/s", f"{log_mbps:.2f}")
                    + row("CPU dsp%", f"{cpu_dsp:.1f}%")
                    + row("CPU log%", f"{cpu_log:.1f}%")
                    + "+------------------+--------------------+\n"
                )

                dpg.set_value(TXT_STATS_TAG, stats_str)
                
                # Update Log Info
                dpg.set_value(
                    TXT_LOG_TAG,
                    "LOGGER\n"
                    f"{'dir':<12}{out_dir.name}\n"
                    f"{'cap':<12}{int(cap_active.value):>8d}\n"
                    f"{'pos_id':<12}{int(cap_pos_id.value):>8d}\n"
                    f"{'saved':<12}{int(cap_saved.value):>8d}/{int(FRAMES_PER_POSITION)}\n"
                    f"{'sar_pos':<12}{int(sar_pos_counter.value):>8d}"
                )
            # 2. GUI TEXTURE UPDATE (double-buffer latest-wins)
            seq_now = int(gui_latest_seq.value)
            if seq_now != gui_last_seq:
                with gui_lock:
                    seq_locked = int(gui_latest_seq.value)
                    idx = int(gui_latest_idx.value)
                    if idx in (0, 1):
                        base = idx * gui_h * gui_w
                        src_flat = np.frombuffer(gui_dbuf, dtype=np.float32, count=gui_h * gui_w, offset=base * 4)
                        gui_frame.reshape(-1)[:] = src_flat
                    gui_last_seq = seq_locked

                denom = (vis_vmax - vis_vmin)
                if denom < 1e-6:
                    denom = 1e-6

                src = gui_frame
                max_r_bin = int(vis_rmax / dr_plot)
                max_r_bin = max(1, min(max_r_bin, src.shape[0]))

                half_bins = src.shape[1] // 2
                if CROSSRANGE_MAX_DISPLAY > 1e-6:
                    keep = int(half_bins * (abs(vis_xmax) / float(CROSSRANGE_MAX_DISPLAY)))
                else:
                    keep = half_bins
                keep = max(8, min(half_bins, keep))

                x0 = half_bins - keep
                x1 = half_bins + keep
                win = src[:max_r_bin, x0:x1]

                ys = np.linspace(0, win.shape[0] - 1, tex_h).astype(np.int32)
                xs = np.linspace(0, win.shape[1] - 1, tex_w).astype(np.int32)
                img = win[ys[:, None], xs[None, :]]

                norm = (img - vis_vmin) / denom
                norm = np.clip(norm, 0.0, 1.0)
                rgba = _jet_rgba(norm)
                rgba = rgba[::-1, :, :]
                tex_np[:] = rgba.reshape(-1)
                dpg.set_value(TEX_TAG, tex_buf)
                img_updates += 1

            dpg.render_dearpygui_frame()

    except KeyboardInterrupt:
        pass
    finally:
        _shutdown()
        dpg.destroy_context()



        
if __name__ == "__main__":
    mp.freeze_support()
    main()





