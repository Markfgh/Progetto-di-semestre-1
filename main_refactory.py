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
import os

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

def _cpu_percent_pid(pid: int, cpu_state: dict) -> float:
    """Best-effort process CPU% using psutil, non-blocking."""
    if pid is None or pid <= 0:
        return 0.0
    if psutil is None:
        return float("nan")
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
# vmin/vmax: scala di base per modalita NON normalizzata (NON hard limit)
VMIN_RAW = float(cfg["display"]["vmin"])
VMAX_RAW = float(cfg["display"]["vmax"])
# vmin_norm/vmax_norm: scala di base per modalita normalizzata (fallback su raw)
VMIN_NORM = float(cfg.get("display", {}).get("vmin_norm", VMIN_RAW))
VMAX_NORM = float(cfg.get("display", {}).get("vmax_norm", VMAX_RAW))

# range_max / crossrange_max: HARD LIMIT (0 .. max config)
RMAX_HARD_MAX = float(cfg["display"]["range_max"])
XMAX_HARD_MAX = float(cfg.get("display", {}).get("crossrange_max", RMAX_HARD_MAX))

# valori iniziali di visualizzazione (partono dai limiti config, ma poi l'utente puÃ² ridurli fino a 0)
RANGE_MAX_DISPLAY = float(cfg["display"]["range_max"])
CROSSRANGE_MAX_DISPLAY = float(cfg.get("display", {}).get("crossrange_max", RANGE_MAX_DISPLAY))
RANGEFFT_DB_MIN = float(cfg.get("display", {}).get("rangefft_db_min", VMIN_RAW))
RANGEFFT_DB_MAX = float(cfg.get("display", {}).get("rangefft_db_max", VMAX_RAW))
if RANGEFFT_DB_MAX <= RANGEFFT_DB_MIN:
    RANGEFFT_DB_MAX = RANGEFFT_DB_MIN + 1.0
RANGEFFT_LIN_MIN = float(cfg.get("display", {}).get("rangefft_lin_min", 10.0 ** (RANGEFFT_DB_MIN / 10.0)))
RANGEFFT_LIN_MAX = float(cfg.get("display", {}).get("rangefft_lin_max", 10.0 ** (RANGEFFT_DB_MAX / 10.0)))
if RANGEFFT_LIN_MAX <= RANGEFFT_LIN_MIN:
    RANGEFFT_LIN_MAX = RANGEFFT_LIN_MIN + 1e-6
RANGE_PROFILE_COUNT = 8
RANGEFFT_PLOT_COUNT = int(TX)
RANGEFFT_LINES_PER_PLOT = int(RX)
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
    STALL_TIMEOUT_S = float(cfg.get("rx", {}).get("stall_timeout_s", 2.0))
    if STALL_TIMEOUT_S <= 0.0:
        STALL_TIMEOUT_S = 2.0
    SEQ_REORDER_MAX_BACK = int(cfg.get("rx", {}).get("seq_reorder_max_back", 32))
    if SEQ_REORDER_MAX_BACK < 0:
        SEQ_REORDER_MAX_BACK = 0
    U32_MOD = 1 << 32
    U32_HALF = 1 << 31

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
            recv_perf = time.perf_counter()
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

        if payload_len_ref is None:
            payload_len_ref = n_bytes - HEADER_LEN

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
                # backward packet: small back = reorder, large back = stream restart
                back = (int(last_seq) - int(seq)) & (U32_MOD - 1)
                if back <= SEQ_REORDER_MAX_BACK:
                    continue
                _soft_reset_stream()
                payload_len_ref = n_bytes - HEADER_LEN
                seq_gap_pkts = 0

        # --- byte_count check (robust alignment) ---
        if last_byte_count is not None and payload_len_ref is not None:
            expected = (last_byte_count + payload_len_ref) % MOD48
            if bc != expected:
                frame_ok = False
        last_byte_count = bc

        # --- GAP handling (consume missing bytes to keep frame alignment) ---
        if seq_gap_pkts > 0 and payload_len_ref is not None:
            with lost_pkts.get_lock():
                lost_pkts.value += int(seq_gap_pkts)
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

    def _open_file(pos_id: int):
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
            _open_file(int(pending_pos_id))
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
        try:
            with norm_to_peak.get_lock():
                normalize_to_peak = bool(norm_to_peak.value)
        except Exception:
            normalize_to_peak = True
        heatmap_ema = process_buffer(
            complex_view, 
            n_proc,
            window_range, 
            window_angle, 
            heatmap_ema, 
            0.2, # alpha
            gui_dbuf,
            gui_prof_dbuf,
            gui_h,
            gui_w,
            gui_latest_idx,
            gui_latest_seq,
            gui_lock,
            max_bin,
            normalize_to_peak,
            stat_raw_min_db,
            stat_raw_max_db,
            stat_norm_min_db,
            stat_norm_max_db,
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
    gui_prof_dbuf,
    gui_h,
    gui_w,
    gui_latest_idx,
    gui_latest_seq,
    gui_lock,
    max_bin,
    normalize_to_peak: bool,
    stat_raw_min_db: Synchronized,
    stat_raw_max_db: Synchronized,
    stat_norm_min_db: Synchronized,
    stat_norm_max_db: Synchronized,
):

    try:
        # A. Reshape SENZA transpose (evita copia grossa)

        # Shape: (F, chirpsPerTx, TX, SAMPLES, RX)  -> samples Ã¨ axis=3

        data = raw_buffer.reshape(n_frames, CHIRPS // TX, TX, SAMPLES, RX)


        # B. DSP IN-PLACE

        # Clutter removal: sottrai la media lungo SAMPLES (axis=3)

        data -= data.mean(axis=2, keepdims=True, dtype=np.complex64)

        # Finestra Range: w_range deve essere broadcastabile su axis=3 (shape: 1,1,1,SAMPLES,1)

        data *= w_range


        # C. RANGE FFT (samples axis=3)

        range_fft = fft.fft(data, n=NFFT_RANGE, axis=3, workers=FFT_WORKERS, overwrite_x=True)


        # Preparazione Virtual Array senza transpose+ascontiguousarray:
        # range_fft: (F, C, TX, R, RX)
        va = range_fft.transpose(0, 1, 3, 2, 4)   # -> (F, C, R, TX, RX) view
        va = np.ascontiguousarray(va)            # rende il reshape prevedibile (una copia al massimo)
        virtual_array = va.reshape(n_frames, CHIRPS // TX, NFFT_RANGE, VIRTUAL_ANT)

        # Limitiamo i range bin al solo intervallo visualizzato prima della angle-FFT.
        # Evita lavoro inutile su bin che verrebbero comunque scartati.
        virtual_array = virtual_array[:, :, :max_bin, :]

        # Profili range FFT per antenna virtuale: modulo^2 medio su frame/chirp, poi dB.
        prof_re = virtual_array.real
        prof_im = virtual_array.imag
        profiles_pow = (prof_re * prof_re + prof_im * prof_im).mean(axis=(0, 1))
        profiles_db_va = (10.0 * np.log10(profiles_pow + 1e-12)).transpose(1, 0)
        profiles_out = np.full((RANGE_PROFILE_COUNT, max_bin), -120.0, dtype=np.float32)
        copy_rows = min(int(RANGE_PROFILE_COUNT), int(profiles_db_va.shape[0]))
        if copy_rows > 0:
            profiles_out[:copy_rows, :] = profiles_db_va[:copy_rows, :].astype(np.float32, copy=False)

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
            raw_max = float(np.max(view_db))
            if DEBUG_STATS:
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

            prof_base = next_idx * int(RANGE_PROFILE_COUNT) * int(gui_h)
            dst_prof = np.frombuffer(
                gui_prof_dbuf,
                dtype=np.float32,
                count=int(RANGE_PROFILE_COUNT) * int(gui_h),
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
    gui_prof_dbuf = RawArray("f", 2 * RANGE_PROFILE_COUNT * gui_h)
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
    rx_stall_events = Value("L", 0)
    rx_stream_resets = Value("L", 0)
    dsp_skip = Value("L", 0)
    dsp_ms_avg = Value("d", 0.0)
    dsp_ms_p95 = Value("d", 0.0)
    log_bytes = Value("L", 0)
    norm_to_peak = Value("b", 1)
    stat_raw_min_db = Value("d", float("nan"))
    stat_raw_max_db = Value("d", float("nan"))
    stat_norm_min_db = Value("d", float("nan"))
    stat_norm_max_db = Value("d", float("nan"))

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
            rx_stall_events,
            rx_stream_resets,
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
            gui_prof_dbuf,
            gui_h,
            gui_w,
            gui_latest_idx,
            gui_latest_seq,
            gui_lock,
            dsp_skip,
            dsp_ms_avg,
            dsp_ms_p95,
            norm_to_peak,
            stat_raw_min_db,
            stat_raw_max_db,
            stat_norm_min_db,
            stat_norm_max_db,
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
    if DEBUG_STATS and psutil is None:
        print("[STATS] CPU disabled: install 'psutil' in the active Python interpreter.")

    # =========================================================================
    # GUI SETUP (RESPONSIVE - CLEAN)
    # =========================================================================

    # display state (cartesian X-Y only)
    norm_enabled_init = bool(norm_to_peak.value)
    dr_plot = float(dr_shared)
    if norm_enabled_init:
        vis_vmin = float(VMIN_NORM)
        vis_vmax = float(VMAX_NORM)
    else:
        vis_vmin = float(VMIN_RAW)
        vis_vmax = float(VMAX_RAW)
    if vis_vmax <= vis_vmin:
        vis_vmax = vis_vmin + 1.0
    vis_rmax = float(RANGE_MAX_DISPLAY)
    vis_xmax = float(CROSSRANGE_MAX_DISPLAY)
    vis_fft_mode_db = True
    vis_fft_vmin = float(RANGEFFT_DB_MIN)
    vis_fft_vmax = float(RANGEFFT_DB_MAX)
    if vis_fft_vmax <= vis_fft_vmin:
        vis_fft_vmax = vis_fft_vmin + 1.0
    fft_mode_db = bool(vis_fft_mode_db)

    ui_dirty = True
    ui_dirty_t = 0.0
    ui_pending = {
        "vmin": vis_vmin,
        "vmax": vis_vmax,
        "rmax": vis_rmax,
        "xmax": vis_xmax,
        "fft_vmin": vis_fft_vmin,
        "fft_vmax": vis_fft_vmax,
        "fft_mode_db": bool(fft_mode_db),
    }

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
    TAG_RANGEFFT_COL = "rangefft_col"

    TXT_STATS_TAG = "txt_stats"
    IN_VMIN, IN_VMAX = "in_vmin", "in_vmax"
    IN_RMAX, IN_XMAX = "in_rmax", "in_xmax"
    IN_FFT_VMIN, IN_FFT_VMAX = "in_fft_vmin", "in_fft_vmax"
    TXT_POS_TAG = "txt_pos_counter"
    TXT_LOG_TAG = "txt_log"
    BTN_NORM_TAG = "btn_norm_toggle"
    BTN_FFT_MODE_TAG = "btn_fft_mode"

    TEX_TAG = "heat_tex"
    HEAT_PLOT_TAG = "heat_plot"
    XAXIS_TAG, YAXIS_TAG = "xaxis", "yaxis"
    IMG_SERIES_TAG = "img_series"
    CMAP_SCALE_TAG = "cmap_scale"
    CMAP_NUM_FMT = "%+6.1f"
    RANGEFFT_PLOT_TAG = "rangefft_plot"
    RANGEFFT_XAXIS_TAG = "rangefft_xaxis"
    RANGEFFT_YAXIS_TAG = "rangefft_yaxis"
    RANGEFFT_LINE_TAGS = [f"rangefft_line_ant{ant_i}" for ant_i in range(RANGE_PROFILE_COUNT)]
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

    # 4) Texture setup (1:1 con il buffer GUI, senza resampling)
    tex_w, tex_h = int(gui_w), int(gui_h)
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

    def _fft_mode_label(mode_db: bool) -> str:
        return "FFT SCALE: dB" if mode_db else "FFT SCALE: LINEAR"

    def _fft_axis_label(mode_db: bool) -> str:
        return "dB" if mode_db else "Linear"

    def _fft_visible_range_from_rmax(rmax_m: float):
        max_r_bin = int(rmax_m / dr_plot) if dr_plot > 1e-12 else gui_h
        max_r_bin = max(1, min(max_r_bin, gui_h))
        max_r_m = float(max_r_bin) * float(dr_plot)
        return max_r_bin, max_r_m

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
        nonlocal ui_dirty, ui_pending, fft_mode_db
        # callback ultraleggera: salva valori e setta flag
        # Nota:
        #  - vmin/vmax sono "base values" (non hard limit). Li lasciamo liberi.
        #  - rmax/xmax sono HARD LIMIT: 0 .. (range_max/crossrange_max da config)
        try:
            vmin = float(dpg.get_value(IN_VMIN))
            vmax = float(dpg.get_value(IN_VMAX))
            rmax = float(dpg.get_value(IN_RMAX))
            xmax = float(dpg.get_value(IN_XMAX))
            fft_vmin = float(dpg.get_value(IN_FFT_VMIN))
            fft_vmax = float(dpg.get_value(IN_FFT_VMAX))
        except (ValueError, TypeError):
            return  # mentre scrivi '-', '.', '' ecc.

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

        ui_pending["vmin"] = vmin
        ui_pending["vmax"] = vmax
        ui_pending["rmax"] = rmax_cl
        ui_pending["xmax"] = xmax_cl
        ui_pending["fft_vmin"] = fft_vmin
        ui_pending["fft_vmax"] = fft_vmax
        ui_pending["fft_mode_db"] = bool(fft_mode_db)

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

    def _norm_toggle_label(enabled: bool) -> str:
        return "NORM: ON" if enabled else "NORM: OFF"

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
        try:
            with norm_to_peak.get_lock():
                norm_to_peak.value = 0 if int(norm_to_peak.value) else 1
                enabled = bool(norm_to_peak.value)
        except Exception:
            enabled = True
        if dpg.does_item_exist(BTN_NORM_TAG):
            dpg.configure_item(BTN_NORM_TAG, label=_norm_toggle_label(enabled))
        base_vmin, base_vmax = _base_vscale_for_mode(enabled)
        if dpg.does_item_exist(IN_VMIN):
            dpg.set_value(IN_VMIN, base_vmin)
        if dpg.does_item_exist(IN_VMAX):
            dpg.set_value(IN_VMAX, base_vmax)
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

    STATS_KEY_W = 16
    STATS_VAL_W = 13

    def _format_stats_table(rows):
        if not rows:
            return ""
        rows_s = [(str(k), str(v)) for k, v in rows]
        border = f"+{'-' * (STATS_KEY_W + 2)}+{'-' * (STATS_VAL_W + 2)}+\n"
        body = "".join(f"| {k:<{STATS_KEY_W}} | {v:>{STATS_VAL_W}} |\n" for k, v in rows_s)
        return border + body + border

    # 6) Build UI (ONLY TABLE LAYOUT)
    with dpg.window(tag=TAG_MAIN_WINDOW):

        with dpg.table(header_row=False, resizable=True, policy=dpg.mvTable_SizingFixedFit):

            # Colonne: sidebar (fissa), plot (stretch), colorbar (fissa), range FFT (fissa)
            dpg.add_table_column(init_width_or_weight=340, width_fixed=True, no_resize=True)                      # sidebar
            dpg.add_table_column(init_width_or_weight=1.0, width_stretch=True, width_fixed=False)                 # plot stretch
            dpg.add_table_column(init_width_or_weight=100, width_fixed=True, no_resize=True)                      # colorbar
            dpg.add_table_column(init_width_or_weight=500, width_fixed=True, no_resize=True)                      # range FFT


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
                        dpg.add_spacer(height=8)
                        dpg.add_button(
                            label=_norm_toggle_label(_get_norm_enabled()),
                            tag=BTN_NORM_TAG,
                            callback=_toggle_norm,
                            width=-1,
                            height=32,
                        )
                        dpg.add_spacer(height=14)
                        dpg.add_text("SAR CONTROL", color=(255, 200, 0))
                        dpg.add_separator()
                        dpg.add_button(label="CAPTURE FRAME", callback=_on_capture, width=-1, height=40)
                        dpg.add_text("Pos Counter: 0", tag=TXT_POS_TAG)
                        dpg.add_text("", tag=TXT_LOG_TAG, wrap=-1)

                        if DEBUG_STATS:
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

                        # bounds in metri fissi (full FOV): zoom/crop gestiti dagli assi
                        dpg.add_image_series(
                            TEX_TAG,
                            bounds_min=(-float(CROSSRANGE_MAX_DISPLAY), 0.0),
                            bounds_max=(+float(CROSSRANGE_MAX_DISPLAY), float(RANGE_MAX_DISPLAY)),
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
                            format=CMAP_NUM_FMT,
                            width=-1,
                            height=-1,
                            colormap=dpg.mvPlotColormap_Jet,
                        )
                        if font_mono:
                            dpg.bind_item_font(CMAP_SCALE_TAG, font_mono)

                # --- RANGE FFT PROFILES ---
                with dpg.table_cell():
                    with dpg.child_window(tag=TAG_RANGEFFT_COL, width=-1, height=-1, border=True):
                        dpg.add_text("RANGE FFT |TXxRX|", color=(255, 200, 0))
                        dpg.add_separator()
                        with dpg.plot(
                            tag=RANGEFFT_PLOT_TAG,
                            label=f"TX 1-{RANGEFFT_PLOT_COUNT} (RX 1-{RANGEFFT_LINES_PER_PLOT})",
                            width=-1,
                            height=540,
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
                        _, init_fft_rmax_m = _fft_visible_range_from_rmax(vis_rmax)
                        dpg.set_axis_limits(RANGEFFT_XAXIS_TAG, 0.0, float(init_fft_rmax_m))
                        dpg.set_axis_limits(RANGEFFT_YAXIS_TAG, float(vis_fft_vmin), float(vis_fft_vmax))
                        dpg.add_separator()
                        dpg.add_button(
                            label=_fft_mode_label(fft_mode_db),
                            tag=BTN_FFT_MODE_TAG,
                            callback=_toggle_fft_mode,
                            width=-1,
                            height=30,
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

    _update_fft_scale_input_labels(fft_mode_db)
    # Applicazione parametri DOPO creazione items
    _apply_params()

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
        log_bytes_prev = 0
    gui_last_seq = 0
    gui_frame = np.zeros((gui_h, gui_w), dtype=np.float32)
    rangefft_frame = np.full((RANGE_PROFILE_COUNT, gui_h), -120.0, dtype=np.float32)
    range_axis_m = np.arange(gui_h, dtype=np.float32) * np.float32(dr_plot)
    if DEBUG_STATS:
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
                fft_vmin = float(ui_pending["fft_vmin"])
                fft_vmax = float(ui_pending["fft_vmax"])
                fft_mode_db = bool(ui_pending["fft_mode_db"])
                if vmax <= vmin:
                    vmax = vmin + 1.0
                fft_eps = 1.0
                if fft_vmax <= fft_vmin:
                    fft_vmax = fft_vmin + fft_eps

                vis_vmin = vmin
                vis_vmax = vmax
                vis_rmax = rmax
                vis_xmax = xmax
                vis_fft_mode_db = fft_mode_db
                vis_fft_vmin = fft_vmin
                vis_fft_vmax = fft_vmax

                # update axis limits (meters). L'immagine resta su bounds fissi per mantenere texture 1:1.
                if dpg.does_item_exist(XAXIS_TAG) and dpg.does_item_exist(YAXIS_TAG):
                    dpg.set_axis_limits(XAXIS_TAG, -vis_xmax, +vis_xmax)
                    dpg.set_axis_limits(YAXIS_TAG, 0.0, float(vis_rmax))

                # colorbar
                if dpg.does_item_exist(CMAP_SCALE_TAG):
                    dpg.configure_item(CMAP_SCALE_TAG, min_scale=vis_vmin, max_scale=vis_vmax, format=CMAP_NUM_FMT)
                if dpg.does_item_exist(BTN_FFT_MODE_TAG):
                    dpg.configure_item(BTN_FFT_MODE_TAG, label=_fft_mode_label(vis_fft_mode_db))
                _update_fft_scale_input_labels(vis_fft_mode_db)

                max_r_bin, max_r_m = _fft_visible_range_from_rmax(vis_rmax)
                if dpg.does_item_exist(RANGEFFT_YAXIS_TAG):
                    dpg.set_axis_limits(RANGEFFT_YAXIS_TAG, float(vis_fft_vmin), float(vis_fft_vmax))
                    dpg.configure_item(RANGEFFT_YAXIS_TAG, label=_fft_axis_label(vis_fft_mode_db))
                if dpg.does_item_exist(RANGEFFT_XAXIS_TAG):
                    dpg.set_axis_limits(RANGEFFT_XAXIS_TAG, 0.0, float(max_r_m))

            
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
                with rx_stall_events.get_lock(): stall_events_now = int(rx_stall_events.value)
                with rx_stream_resets.get_lock(): stream_resets_now = int(rx_stream_resets.value)
                with stat_raw_min_db.get_lock(): raw_min_now = float(stat_raw_min_db.value)
                with stat_raw_max_db.get_lock(): raw_max_now = float(stat_raw_max_db.value)
                with stat_norm_min_db.get_lock(): norm_min_now = float(stat_norm_min_db.value)
                with stat_norm_max_db.get_lock(): norm_max_now = float(stat_norm_max_db.value)
                raw_minmax_str = f"{raw_min_now:.2f}/{raw_max_now:.2f}" if (np.isfinite(raw_min_now) and np.isfinite(raw_max_now)) else "n/a"
                norm_minmax_str = f"{norm_min_now:.2f}/{norm_max_now:.2f}" if (np.isfinite(norm_min_now) and np.isfinite(norm_max_now)) else "n/a"
                cpu_dsp_str = f"{cpu_dsp:.1f}%" if np.isfinite(cpu_dsp) else "n/a"
                cpu_log_str = f"{cpu_log:.1f}%" if np.isfinite(cpu_log) else "n/a"
                
                # --- AGGIORNAMENTO TESTI GUI (FORMATO TABELLA ASCII) ---
                stats_rows = [
                    ("LOSS %", f"{loss_pct:.3f}%"),
                    ("RX pkt/s", f"{pkt_rate:.0f}"),
                    ("RX frames_ok", f"{frames_ok_rate:.1f}/s ({int(frames_ok_now)})"),
                    ("RX drops_ring", f"{int(drop_f)}"),
                    ("RING ready/slots", f"{ready_slots}/{N_SLOTS} HWM {ring_hwm}"),
                    ("DSP ms avg/p95", f"{dsp_avg_now:.2f}/{dsp_p95_now:.2f}"),
                    ("DSP skip", f"{dsp_skip_now}"),
                    ("GUI Hz", f"{img_hz:.1f}"),
                    ("GUI tex bins", f"{gui_w}x{gui_h}"),
                    ("RAW min/max dB", raw_minmax_str),
                    ("NORM min/max dB", norm_minmax_str),
                    ("LOG MB/s", f"{log_mbps:.2f}"),
                    ("CPU dsp%", cpu_dsp_str),
                    ("CPU log%", cpu_log_str),
                    ("stall_events", f"{stall_events_now}"),
                    ("stream_resets", f"{stream_resets_now}"),
                ]

                stats_str = _format_stats_table(stats_rows)

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
                        prof_base = idx * RANGE_PROFILE_COUNT * gui_h
                        prof_flat = np.frombuffer(
                            gui_prof_dbuf,
                            dtype=np.float32,
                            count=RANGE_PROFILE_COUNT * gui_h,
                            offset=prof_base * 4,
                        )
                        rangefft_frame.reshape(-1)[:] = prof_flat
                    gui_last_seq = seq_locked

                denom = (vis_vmax - vis_vmin)
                if denom < 1e-6:
                    denom = 1e-6

                src = gui_frame
                img = src

                norm = (img - vis_vmin) / denom
                norm = np.clip(norm, 0.0, 1.0)
                rgba = _jet_rgba(norm)
                rgba = rgba[::-1, :, :]
                tex_np[:] = rgba.reshape(-1)
                dpg.set_value(TEX_TAG, tex_buf)

                max_r_bin, max_r_m = _fft_visible_range_from_rmax(vis_rmax)
                x_range_m = range_axis_m[:max_r_bin].tolist()
                fft_slice = rangefft_frame[:, :max_r_bin]
                if vis_fft_mode_db:
                    fft_plot_vals = fft_slice
                else:
                    fft_plot_vals = np.power(10.0, fft_slice * 0.1).astype(np.float32, copy=False)
                if dpg.does_item_exist(RANGEFFT_YAXIS_TAG):
                    dpg.set_axis_limits(RANGEFFT_YAXIS_TAG, float(vis_fft_vmin), float(vis_fft_vmax))
                    dpg.configure_item(RANGEFFT_YAXIS_TAG, label=_fft_axis_label(vis_fft_mode_db))
                if dpg.does_item_exist(RANGEFFT_XAXIS_TAG):
                    dpg.set_axis_limits(RANGEFFT_XAXIS_TAG, 0.0, float(max_r_m))
                for ant_i in range(min(RANGE_PROFILE_COUNT, int(fft_plot_vals.shape[0]))):
                    line_tag = RANGEFFT_LINE_TAGS[ant_i]
                    if dpg.does_item_exist(line_tag):
                        y_vals = fft_plot_vals[ant_i, :].tolist()
                        dpg.set_value(line_tag, [x_range_m, y_vals])
                if DEBUG_STATS:
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





