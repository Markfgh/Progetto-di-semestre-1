import socket
import time
import queue as pyqueue
from pathlib import Path
from multiprocessing import Process, Queue, Value
from array import array
import struct
import json
import psutil

from multiprocessing.sharedctypes import RawArray, Synchronized
import multiprocessing as mp

import yaml
import numpy as np
import dearpygui.dearpygui as dpg
import scipy.fft as fft
import os


# ----------------------------
# COSTANTI FISICHE RADAR
# ----------------------------
C = 3e8

# ----------------------------
# CONFIGURAZIONE DA YAML
# ----------------------------
CFG_PATH = "Config.yaml"

with open(CFG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# --- RADAR / CAPTURE ---
_phys = cfg.get("radar") or cfg.get("physical") or {}
# allow overriding speed of light if provided
C = float(_phys.get("c", C))
FS = float(_phys["fs"])
SLOPE = float(_phys["slope"])
FC = float(_phys["fc"])
LAMBDA = C / FC

SAMPLES = int(cfg["capture"]["samples"])
CHIRPS = int(cfg["capture"]["chirps"])
RX = int(cfg["capture"]["rx"])
TX = int(cfg["capture"]["tx"])
X_FRAMES = int(cfg["capture"]["x_frames"])

VIRTUAL_ANT = RX * TX

# --- SAR / CAPTURE CONTROL ---
SETTLING_DELAY_S = float(cfg["sar"]["settling_delay_s"])
FRAMES_PER_POSITION = int(cfg["sar"]["frames_per_position"])

# --- DSP ---
_dsp = cfg.get("dsp") or {}
FFT_WORKERS = int(_dsp.get("fft_workers", (cfg.get("fft") or {}).get("workers", 1)))
WINDOW_RANGE = str(_dsp.get("window_range", "hann"))
WINDOW_ANGLE = str(_dsp.get("window_angle", "hann"))

# --- UDP ---
BYTES_PER_SAMPLE = 2  # i16
BYTES_PER_COMPLEX = 4  # I/Q i16
BYTES_PER_FRAME = CHIRPS * SAMPLES * RX * 4

# --- FFT ---
NFFT_RANGE = int(cfg["fft"]["nfft_range"])
NFFT_ANGLE = int(cfg["fft"]["nfft_angle"])

# --- DISPLAY ---
VMIN = float(cfg["display"]["vmin"])
VMAX = float(cfg["display"]["vmax"])
RANGE_MAX_DISPLAY = float(cfg["display"]["range_max"])
ANGLE_MAX_DISPLAY = float(cfg["display"].get("angle_max", 90.0))

# --- CODE QUEUE ---
DEBUG_STATS = bool(cfg["debug"]["debug_stats"])

# --- QUEUE SIZE ---
FRAME_Q_MAX = 20  # dimensione coda frame tra RX e DSP (in numero di frame, non byte)
GUI_Q_MAX = 5  # dimensione coda tra DSP e GUI (in numero di heatmap, non byte).
LOG_RING_N = 2048  # numero di frame bufferizzabili per logging (deve essere >= FRAME_Q_MAX + X_FRAMES)


# ----------------------------
# FUNZIONI DI ELABORAZIONE
# ----------------------------

def _make_window(name: str, n: int) -> np.ndarray:
    """Return a 1-D window of length n."""
    if not name:
        return np.ones(n, dtype=np.float32)
    name = str(name).strip().lower()
    if name in ("hann", "hanning"):
        w = np.hanning(n)
    elif name in ("hamming",):
        w = np.hamming(n)
    elif name in ("blackman",):
        w = np.blackman(n)
    elif name in ("rect", "rectangular", "boxcar", "none"):
        w = np.ones(n)
    else:
        raise ValueError(f"Unsupported window: {name!r}")
    return w.astype(np.float32)


def build_windows(samples: int, virtual_ant: int, win_range: str, win_angle: str):
    """Costruisce finestre di Range e Angolo (broadcast-friendly)."""
    w_r = _make_window(win_range, samples)[None, None, None, :, None]      # (1,1,1,S,1)
    w_a = _make_window(win_angle, virtual_ant)[None, None, :, None]       # (1,1,V,1)
    return w_r, w_a


def _rss_bytes_pid(pid: int) -> int:
    """Restituisce RSS in byte per un PID (Linux-only; su Windows restituisce 0)."""
    try:
        process = psutil.Process(pid)
        return process.memory_info().rss
    except:
        return 0


def _fmt_mb(nbytes: int) -> str:
    return f"{(nbytes / (1024 * 1024)):.1f}MB"


# ----------------------------
# WORKER RX
# ----------------------------

def radar_rx(
    cmd_queue: Queue,
    log_meta: RawArray,
    log_head: Synchronized,
    log_tail: Synchronized,
    log_overflow_drops: Synchronized,
    free_slots: Queue,
    shm_frames,
    lost_pkts: Synchronized,
    rx_pkts: Synchronized,
    rx_put_drops: Synchronized,
    frame_queue: Queue,
    frame_put_ok: Synchronized,
    stop_evt,
    settling_delay_s: float,
    frames_per_position: int,
):
    """
    Riceve UDP DCA1000.

    Corruzione (gap / reset sequenza):
      - se viene rilevato un gap di pacchetti, il/i frame coinvolti vengono marcati come "corrotti"
        e azzerati COMPLETAMENTE (frame di soli zeri) per evitare artefatti nel DSP.

    Logging:
      - viene salvato su disco SOLO durante la fase di recording avviata via comando CAPTURE.
      - fuori dal recording, i frame sono inoltrati direttamente al DSP/GUI best-effort.
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
    curr_slot = None
    frame_view = None

    ZERO_FRAME_BLOCK = b"\x00" * BYTES_PER_FRAME

    w = 0
    last_seq = None
    payload_len_ref = None

    frame_ok = True
    in_gap = False  # True mentre consumiamo bytes mancanti (gap) e quindi ogni frame è corrotto

    pkts_local = 0
    t_flush = time.perf_counter()

    # ----------------------------
    # Capture-only recording control (commands from GUI)
    # ----------------------------
    pending = False
    pending_start_t = 0.0  # perf_counter time when recording should start (after settling)
    rec_pos_id = 0
    rec_frames_left = 0
    rec_frame_in_pos = 0
    recording = False

    def _poll_commands(now_perf: float) -> None:
        nonlocal pending, pending_start_t, rec_pos_id, rec_frames_left, rec_frame_in_pos, recording
        while True:
            try:
                cmd = cmd_queue.get_nowait()
            except pyqueue.Empty:
                break
            if not cmd:
                continue
            if cmd[0] == "CAPTURE":
                rec_pos_id = int(cmd[1])
                rec_frames_left = max(0, int(frames_per_position))
                rec_frame_in_pos = 0
                if rec_frames_left <= 0:
                    recording = False
                    pending = False
                    continue
                pending = True
                pending_start_t = now_perf + settling_delay_s
            else:
                continue

        if pending and now_perf >= pending_start_t:
            recording = True
            pending = False

    def _record_decision(now_perf: float):
        nonlocal pending, pending_start_t, rec_pos_id, rec_frames_left, rec_frame_in_pos, recording

        if recording and rec_frames_left > 0:
            do_log = True
            pos_id = rec_pos_id
            frame_in_pos = rec_frame_in_pos

            rec_frame_in_pos += 1
            rec_frames_left -= 1
            if rec_frames_left <= 0:
                recording = False
            return do_log, pos_id, frame_in_pos
        return False, 0, 0

    def ensure_slot() -> bool:
        nonlocal curr_slot, frame_view, w, frame_ok, in_gap
        if w != 0 or frame_view is not None:
            return True
        while not stop_evt.is_set():
            try:
                curr_slot = free_slots.get(timeout=0.5)
                break
            except pyqueue.Empty:
                continue
        if stop_evt.is_set():
            return False

        base = int(curr_slot) * BYTES_PER_FRAME
        frame_view = shm_view[base: base + BYTES_PER_FRAME]
        # Se stiamo "consumando" bytes di un gap, ogni frame attraversato è corrotto.
        frame_ok = (not in_gap)
        return True

    def _push_frame(slot_to_push: int, ok: int, ts_ns: int, do_log: bool, pos_id: int = 0, frame_in_pos: int = 0):
        """Invia il frame:
        - do_log=True: manda al logger (scrittura su disco) che poi farà forward best-effort al DSP.
        - do_log=False: bypass diretto al DSP/GUI best-effort (non bloccare RX).
        """
        if do_log:
            # Ring overflow protection: non bloccare mai il RX
            # Calcola riempimento attuale in modo atomico (snapshot)
            with log_head.get_lock():
                head = log_head.value
            with log_tail.get_lock():
                tail = log_tail.value

            fill = head - tail
            if fill >= LOG_RING_N:
                # Ring pieno: scarta la richiesta di logging ma libera subito lo slot
                if log_overflow_drops is not None:
                    with log_overflow_drops.get_lock():
                        log_overflow_drops.value += 1
                free_slots.put(int(slot_to_push))
                return

            # Scrittura del record nel ring
            with log_head.get_lock():
                idx = log_head.value % LOG_RING_N
                base = idx * 5

                log_meta[base + 0] = int(slot_to_push)
                log_meta[base + 1] = int(ok)
                log_meta[base + 2] = int(pos_id)
                log_meta[base + 3] = int(frame_in_pos)
                log_meta[base + 4] = int(ts_ns)

                log_head.value += 1
            return

        try:
            frame_queue.put_nowait(int(slot_to_push))
            if DEBUG_STATS and frame_put_ok is not None:
                with frame_put_ok.get_lock():
                    frame_put_ok.value += 1
        except pyqueue.Full:
            if rx_put_drops is not None:
                with rx_put_drops.get_lock():
                    rx_put_drops.value += 1
            free_slots.put(int(slot_to_push))

    while not stop_evt.is_set():
        now_perf = time.perf_counter()
        _poll_commands(now_perf)

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
            print("RX TIMEOUT: no packets")
            continue

        if not ensure_slot():
            continue

        if n_bytes <= HEADER_LEN:
            continue

        seq = int.from_bytes(packet_view[0:4], "little", signed=False)

        if payload_len_ref is None:
            payload_len_ref = n_bytes - HEADER_LEN

        if last_seq is not None and seq <= last_seq:
            in_gap = True
            frame_ok = False
            print(f"[RX] SEQ RESET: last={last_seq} now={seq}")
        elif last_seq is not None and seq != last_seq + 1:
            gap = seq - (last_seq + 1)
            print(f"[RX] GAP: expected {last_seq + 1}, got {seq}, gap={gap}")
            if lost_pkts is not None:
                with lost_pkts.get_lock():
                    lost_pkts.value += gap
            in_gap = True
            frame_ok = False

        last_seq = seq

        payload = packet_view[HEADER_LEN:n_bytes]
        payload_len = len(payload)
        if payload_len != payload_len_ref:
            print(f"[RX] PAYLOAD LEN CHANGE: ref={payload_len_ref} now={payload_len}")
            payload_len_ref = payload_len
            in_gap = True
            frame_ok = False

        bytes_missing = payload_len
        offset = 0

        while bytes_missing > 0:
            if w == BYTES_PER_FRAME:
                slot_to_push = curr_slot
                curr_slot = None
                local_ok = 1 if frame_ok else 0

                if not frame_ok and frame_view is not None:
                    frame_view[:] = ZERO_FRAME_BLOCK

                ts_ns = time.time_ns()
                do_log, pos_id, finpos = _record_decision(time.perf_counter())
                _push_frame(slot_to_push, local_ok, ts_ns, do_log, pos_id, finpos)

                w = 0
                frame_view = None
                frame_ok = (not in_gap)

                if not ensure_slot():
                    break

            can_take = min(bytes_missing, BYTES_PER_FRAME - w)
            if frame_view is not None:
                frame_view[w: w + can_take] = payload[offset: offset + can_take]

            w += can_take
            offset += can_take
            bytes_missing -= can_take

        if in_gap:
            while w != 0:
                take = min(payload_len, BYTES_PER_FRAME - w)
                w += take
                if w == BYTES_PER_FRAME:
                    frame_view[:] = ZERO_FRAME_BLOCK

                    slot_to_push = curr_slot
                    ts_ns = time.time_ns()
                    do_log, pos_id, finpos = _record_decision(time.perf_counter())
                    _push_frame(slot_to_push, 0, ts_ns, do_log, pos_id, finpos)

                    w = 0
                    curr_slot = None
                    frame_view = None
                    frame_ok = True
            in_gap = False

    if curr_slot is not None:
        try:
            free_slots.put(curr_slot)
        except Exception:
            pass

    sock.close()


# ----------------------------
# WORKER LOGGER
# ----------------------------

def logger_worker(
    log_meta: RawArray,
    log_head: Synchronized,
    log_tail: Synchronized,
    frame_queue: Queue,
    free_slots: Queue,
    shm_frames,
    stop_evt,
    log_saved: Synchronized,
    dsp_put_drops: Synchronized,
    frame_put_ok: Synchronized,
    data_path_s: str,
    idx_path_s: str,
    header_path_s: str,
    settling_delay_s: float,
    frames_per_position: int,
):
    """
    Salva SOLO i frame registrati (recording) in:
      - data_path_s: file .bin con frame concatenati (BYTES_PER_FRAME fissi)
      - idx_path_s:  file .idx con header fisso + record per frame (32 byte)
      - header_path_s: .json con snapshot config + descrizione formato

    Forward verso DSP: best-effort (se frame_queue è piena, il frame è comunque loggato
    ma NON viene inviato al DSP/GUI per non bloccare la registrazione).
    """
    data_path = Path(data_path_s)
    idx_path = Path(idx_path_s)
    header_path = Path(header_path_s)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    hdr = {
        "version": 2,
        "mode": "capture_only",
        "format": {
            "data_bin": "recorded frames only, concatenated, fixed BYTES_PER_FRAME",
            "idx_bin": "binary records (32B) per frame: frame_id, pos_id, frame_in_pos, ok, ts_ns",
        },
        "capture": {
            "samples": SAMPLES,
            "chirps": CHIRPS,
            "rx": RX,
            "tx": TX,
            "x_frames": X_FRAMES,
            "bytes_per_frame": BYTES_PER_FRAME,
            "frames_per_position": int(frames_per_position),
            "settling_delay_s": float(settling_delay_s),
        },
        "radar": {
            "c": C,
            "fs": FS,
            "slope": SLOPE,
            "fc": FC,
            "lambda": LAMBDA,
        },
        "fft": {"nfft_range": NFFT_RANGE, "nfft_angle": NFFT_ANGLE},
    }
    try:
        header_path.write_text(json.dumps(hdr, indent=2), encoding="utf-8")
    except Exception:
        pass

    magic = 0xAABBCCDD
    version = 2
    REC_SIZE = 32
    idx_header = struct.pack(
        "<11I",
        magic,
        version,
        REC_SIZE,
        int(BYTES_PER_FRAME),
        int(SAMPLES),
        int(CHIRPS),
        int(RX),
        int(TX),
        int(X_FRAMES),
        int(NFFT_RANGE),
        int(NFFT_ANGLE),
    )

    REC_FMT = "<Q I H B B Q I I"

    shm_view = memoryview(shm_frames).cast("B")
    frame_id = 0

    with open(data_path, "wb", buffering=1024 * 1024) as fbin, open(
        idx_path, "wb", buffering=1024 * 1024
    ) as fidx:
        fidx.write(idx_header)

        while True:
            try:
                # Snapshot atomico di head/tail per evitare letture incoerenti
                with log_head.get_lock():
                    head = log_head.value
                with log_tail.get_lock():
                    tail = log_tail.value

                # Se è stato richiesto lo stop e non ci sono più frame pendenti, esci
                if stop_evt.is_set() and tail == head:
                    break

                if tail == head:
                    # Nessun frame da loggare al momento
                    time.sleep(0.001)
                    continue

                idx = tail % LOG_RING_N
                base = idx * 5

                slot = log_meta[base + 0]
                ok = log_meta[base + 1]
                pos_id = log_meta[base + 2]
                frame_in_pos = log_meta[base + 3]
                ts_ns = log_meta[base + 4]

                # Avanza il tail (con lock) dopo aver letto i metadati
                with log_tail.get_lock():
                    log_tail.value += 1

                base = int(slot) * BYTES_PER_FRAME
                fbin.write(shm_view[base: base + BYTES_PER_FRAME])

                rec = struct.pack(
                    REC_FMT,
                    int(frame_id),
                    int(pos_id) & 0xFFFFFFFF,
                    int(frame_in_pos) & 0xFFFF,
                    int(ok) & 0xFF,
                    0,
                    int(ts_ns),
                    0,
                    0,
                )
                fidx.write(rec)

                frame_id += 1
                if log_saved is not None:
                    with log_saved.get_lock():
                        log_saved.value = frame_id

                # Forward best-effort verso DSP
                try:
                    frame_queue.put_nowait(int(slot))
                    if DEBUG_STATS:
                        with frame_put_ok.get_lock():
                            frame_put_ok.value += 1
                except pyqueue.Full:
                    if DEBUG_STATS:
                        with dsp_put_drops.get_lock():
                            dsp_put_drops.value += 1
                    # DSP non lo userà: libera subito lo slot
                    free_slots.put(int(slot))

            except Exception as e:
                print(f"[LOGGER] Error: {e}")
                time.sleep(0.01)


# ----------------------------
# WORKER DSP
# ----------------------------

def dsp_worker(frame_queue, free_slots, shm_frames, gui_queue, gui_put_drops, frame_get_ok, gui_put_ok, stop_evt):
    window_range, window_angle = build_windows(samples=SAMPLES, virtual_ant=VIRTUAL_ANT, win_range=WINDOW_RANGE, win_angle=WINDOW_ANGLE)

    dr = C * FS / (2.0 * SLOPE * NFFT_RANGE)
    max_bin = int(np.floor(RANGE_MAX_DISPLAY / dr))
    max_bin = max(1, min(max_bin, NFFT_RANGE // 2))

    i16_per_frame = BYTES_PER_FRAME // 2
    total_samples_needed = X_FRAMES * CHIRPS * SAMPLES * RX

    i16_batch = np.zeros((X_FRAMES, i16_per_frame), dtype=np.int16)
    complex_data = np.zeros(total_samples_needed, dtype=np.complex64)

    shm_view = memoryview(shm_frames).cast("B")
    batch_slots = []
    heatmap_ema = None

    if RX != 4:
        raise ValueError("DSP: conversione I/Q attuale assume RX=4 (packing IIIIQQQQ).")

    while True:
        try:
            slot = frame_queue.get(timeout=0.5)
            if slot is None:
                break
            if DEBUG_STATS:
                with frame_get_ok.get_lock():
                    frame_get_ok.value += 1
            batch_slots.append(slot)
        except pyqueue.Empty:
            if stop_evt.is_set():
                break
            continue

        terminate = False
        try:
            for _ in range(FRAME_Q_MAX):
                s_old = frame_queue.get_nowait()
                if s_old is None:
                    terminate = True
                    break
                batch_slots.append(s_old)
        except pyqueue.Empty:
            pass

        if terminate:
            for s in batch_slots:
                try:
                    free_slots.put(s)
                except Exception:
                    pass
            break

        if len(batch_slots) > X_FRAMES:
            drop_slots = batch_slots[:-X_FRAMES]
            batch_slots = batch_slots[-X_FRAMES:]
            for s in drop_slots:
                free_slots.put(s)

        if len(batch_slots) < X_FRAMES:
            continue

        for k, s in enumerate(batch_slots):
            base = s * BYTES_PER_FRAME
            i16_batch[k, :] = np.frombuffer(shm_view[base: base + BYTES_PER_FRAME], dtype=np.int16)

        for s in batch_slots:
            free_slots.put(s)
        batch_slots.clear()

        raw_buffer = i16_batch.reshape(-1)

        flat_i16 = raw_buffer
        n_blocks = flat_i16.size // 8

        block_view = flat_i16[: n_blocks * 8].reshape(n_blocks, 8)

        complex_data.real[:] = block_view[:, :4].reshape(-1)
        complex_data.imag[:] = block_view[:, 4:].reshape(-1)

        heatmap_ema = process_buffer(
            complex_data,
            window_range,
            window_angle,
            heatmap_ema,
            0.2,
            gui_queue,
            gui_put_drops,
            gui_put_ok,
            max_bin,
        )


def process_buffer(raw_buffer, w_range, w_angle, heatmap_ema, alpha, gui_queue, gui_put_drops, gui_put_ok, max_bin):
    try:
        # A. Reshape SENZA transpose (evita copia grossa)
        # Shape: (F, chirpsPerTx, TX, SAMPLES, RX)
        data = raw_buffer.reshape(X_FRAMES, CHIRPS // TX, TX, SAMPLES, RX)

        # B. DSP IN-PLACE

        # Rimozione offset DC sul segnale tempo (asse SAMPLES, axis=3)
        data -= data.mean(axis=3, keepdims=True, dtype=np.complex64)

        # Finestra Range: w_range deve essere broadcastabile su axis=3 (shape: 1,1,1,SAMPLES,1)
        data *= w_range

        # C. RANGE FFT (samples axis=3)
        range_fft = fft.fft(data, n=NFFT_RANGE, axis=3, workers=FFT_WORKERS, overwrite_x=True)

        # Rimozione clutter statico: sottrai la media lungo slow-time (chirpsPerTx, axis=1)
        range_fft -= range_fft.mean(axis=1, keepdims=True, dtype=np.complex64)

        # Preparazione Virtual Array senza transpose+ascontiguousarray:
        # range_fft: (F, C, TX, R, RX) con R=NFFT_RANGE
        va = np.moveaxis(range_fft, 4, 3)   # -> (F, C, TX, RX, R)
        va = np.moveaxis(va, 2, 3)          # -> (F, C, RX, TX, R)
        virtual_array = va.reshape(X_FRAMES, CHIRPS // TX, VIRTUAL_ANT, NFFT_RANGE)

        virtual_array *= w_angle  # w_angle: (1,1,V,1) -> apply along antenna axis

        # Angle FFT along antenna axis (axis=2)
        angle_fft = fft.fft(virtual_array, n=NFFT_ANGLE, axis=2, workers=FFT_WORKERS, overwrite_x=True)
        angle_fft = fft.fftshift(angle_fft, axes=2)
        # power, average over frames and chirps -> (NFFT_ANGLE, NFFT_RANGE)
        heatmap = (angle_fft.real**2 + angle_fft.imag**2).mean(axis=(0,1))
        heatmap = heatmap.T

        if heatmap_ema is None:
            heatmap_ema = heatmap
        else:
            heatmap_ema *= (1.0 - alpha)
            heatmap_ema += (alpha * heatmap)

        heatmap_db = 10 * np.log10(heatmap_ema + 1e-12)

        view_db = heatmap_db[:max_bin, :]
        if view_db.size > 0:
            mx = np.max(view_db)
            view_db -= mx

        try:
            gui_queue.put_nowait(view_db)
            if DEBUG_STATS:
                with gui_put_ok.get_lock():
                    gui_put_ok.value += 1
        except pyqueue.Full:
            if DEBUG_STATS:
                with gui_put_drops.get_lock():
                    gui_put_drops.value += 1

    except Exception as e:
        print(f"[DSP] Error: {e}")

    return heatmap_ema


# ----------------------------
# MAIN / GUI
# ----------------------------

def main():
    N_SLOTS = FRAME_Q_MAX + X_FRAMES + 64
    free_slots = Queue()
    shm_frames = RawArray("B", N_SLOTS * BYTES_PER_FRAME)
    for i in range(N_SLOTS):
        free_slots.put(i)

    gui_q = Queue(maxsize=GUI_Q_MAX)

    stop_evt = mp.Event()

    sar_pos_counter = Value("L", 0)
    log_saved = Value("L", 0)
    dsp_put_drops = Value("L", 0)

    cmd_q = Queue(maxsize=16)

    log_meta = RawArray("Q", LOG_RING_N * 5)
    log_head = Value("L", 0)
    log_tail = Value("L", 0)
    log_overflow_drops = Value("L", 0)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).with_name("logs")
    data_path = out_dir / f"run_{run_id}.bin"
    idx_path = out_dir / f"run_{run_id}.idx"
    hdr_path = out_dir / f"run_{run_id}.json"

    lost_pkts = Value("L", 0)
    rx_pkts = Value("L", 0)
    rx_put_drops = Value("L", 0)
    gui_put_drops = Value("L", 0)

    frame_put_ok = Value("L", 0)
    frame_get_ok = Value("L", 0)

    gui_put_ok = Value("L", 0)
    gui_get_ok = Value("L", 0)

    frame_q = Queue(maxsize=FRAME_Q_MAX)

    p_rx = Process(
        target=radar_rx,
        args=(
            cmd_q,
            log_meta,
            log_head,
            log_tail,
            log_overflow_drops,
            free_slots,
            shm_frames,
            lost_pkts,
            rx_pkts,
            rx_put_drops,
            frame_q,
            frame_put_ok,
            stop_evt,
            SETTLING_DELAY_S,
            FRAMES_PER_POSITION,
        ),
    )

    p_log = Process(
        target=logger_worker,
        args=(
            log_meta,
            log_head,
            log_tail,
            frame_q,
            free_slots,
            shm_frames,
            stop_evt,
            log_saved,
            dsp_put_drops,
            frame_put_ok,
            str(data_path),
            str(idx_path),
            str(hdr_path),
            SETTLING_DELAY_S,
            FRAMES_PER_POSITION,
        ),
    )

    p_dsp = Process(
        target=dsp_worker,
        args=(frame_q, free_slots, shm_frames, gui_q, gui_put_drops, frame_get_ok, gui_put_ok, stop_evt),
    )

    p_rx.daemon = True
    p_dsp.daemon = True

    p_rx.start()
    p_log.start()
    p_dsp.start()

    print(f"[LOGGER] data: {data_path}")
    print(f"[LOGGER] idx : {idx_path}")
    print(f"[LOGGER] hdr : {hdr_path}")

    disp = cfg.get("display", {}) or {}
    ANG_MIN_CFG = float(disp.get("angle_min", -90.0))
    ANG_MAX_CFG = float(disp.get("angle_max", 90.0))

    ANG_ABS_MAX = max(abs(ANG_MIN_CFG), abs(ANG_MAX_CFG))

    vis_vmin = float(VMIN)
    vis_vmax = float(VMAX)
    vis_rmax = float(RANGE_MAX_DISPLAY)
    vis_ang_abs = float(ANG_ABS_MAX)

    t_mon = time.perf_counter()
    t_img_start = time.perf_counter()
    img_updates = 0
    lost_prev = 0
    pkts_prev = 0

    dpg.create_context()

    CMAP = "cmap_jet"
    with dpg.colormap_registry():
        # usa la colormap built-in "Jet"
        dpg.add_colormap(dpg.mvPlotColormap_Jet, tag=CMAP)


    with dpg.font_registry():
        font_ui = dpg.add_font(r"C:\Windows\Fonts\segoeui.ttf", 16)
        font_mono = dpg.add_font(r"C:\Windows\Fonts\consola.ttf", 14)

    TEX_TAG = "heatmap_tex"
    CBAR_TEX_TAG = "cbar_tex"
    MAIN_WIN = "MainWindow"
    HEATMAP_TEX_W = 256
    HEATMAP_TEX_H = 128

    IN_VMIN = "input_vmin"
    IN_VMAX = "input_vmax"
    IN_RMAX = "input_rmax"
    IN_AABS = "input_aabs"

    BTN_APPLY = "btn_apply"
    BTN_CAPTURE = "btn_capture"
    TXT_POS_TAG = "txt_pos"
    TXT_LOG_TAG = "txt_log"
    TXT_STATS_TAG = "txt_stats"

    def _shutdown():
        """Termina i processi figli in modo robusto e senza perdere gli ultimi frame nel log."""
        try:
            stop_evt.set()
        except Exception:
            pass

        try:
            p_rx.join(timeout=2.0)
        except Exception:
            pass
        try:
            if p_rx.is_alive():
                p_rx.terminate()
        except Exception:
            pass

        try:
            p_log.join(timeout=3.0)
        except Exception:
            pass
        try:
            if p_log.is_alive():
                p_log.terminate()
        except Exception:
            pass

        try:
            frame_q.put_nowait(None)
        except Exception:
            pass
        try:
            p_dsp.join(timeout=2.0)
        except Exception:
            pass
        try:
            if p_dsp.is_alive():
                p_dsp.terminate()
        except Exception:
            pass

    dpg.set_exit_callback(_shutdown)

    def _jet_rgba(norm01: np.ndarray) -> np.ndarray:
        x = np.clip(norm01, 0.0, 1.0).astype(np.float32)
        r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
        a = np.ones_like(r, dtype=np.float32)
        return np.stack((r, g, b, a), axis=-1)

    def _apply_params(sender=None, app_data=None, user_data=None):
        nonlocal vis_vmin, vis_vmax, vis_rmax, vis_ang_abs

        vmin = float(dpg.get_value(IN_VMIN))
        vmax = float(dpg.get_value(IN_VMAX))
        if vmax <= vmin:
            vmax = vmin + 1e-6

        vis_vmin = vmin
        vis_vmax = vmax

        rmax = float(dpg.get_value(IN_RMAX))
        if rmax <= 0:
            rmax = RANGE_MAX_DISPLAY
        vis_rmax = float(rmax)

        aabs = float(dpg.get_value(IN_AABS))
        if aabs <= 0:
            aabs = ANG_ABS_MAX
        vis_ang_abs = float(aabs)

        _update_colorbar_labels()
        _layout_resize()


    def _on_btn_capture(sender, app_data, user_data):
        with sar_pos_counter.get_lock():
            sar_pos_counter.value += 1
            pos_id = sar_pos_counter.value

        try:
            cmd_q.put_nowait(("CAPTURE", pos_id))
            dpg.set_value(TXT_LOG_TAG, f"CAPTURE pos={pos_id} in pending...\n")
        except pyqueue.Full:
            dpg.set_value(TXT_LOG_TAG, "CAPTURE cmd_q FULL!\n")

        dpg.set_value(TXT_POS_TAG, f"Position counter: {pos_id}")

    heatmap_tex_data = np.zeros((HEATMAP_TEX_H, HEATMAP_TEX_W, 4), dtype=np.float32)

    with dpg.texture_registry(show=True):
        dpg.add_dynamic_texture(
            width=HEATMAP_TEX_W,
            height=HEATMAP_TEX_H,
            default_value=heatmap_tex_data.reshape(-1),
            tag=TEX_TAG,
        )

        cbar_h = 256
        CBAR_TEX_W = 24
        cbar = np.linspace(0, 1, cbar_h, dtype=np.float32)[:, None]
        cbar_rgba = _jet_rgba(cbar)                         # (H,1,4)
        cbar_rgba = np.repeat(cbar_rgba, CBAR_TEX_W, axis=1) # (H,W,4)

        dpg.add_dynamic_texture(
            width=CBAR_TEX_W,
            height=cbar_h,
            default_value=cbar_rgba.reshape(-1),
            tag=CBAR_TEX_TAG,
        )


    PLOT_W = 900
    PLOT_H = 600
    CBAR_W = 80

    dpg.create_viewport(title="Radar Heatmap", width=1400, height=950)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    with dpg.window(label="Radar Heatmap", tag=MAIN_WIN, width=1400, height=950):
        dpg.bind_font(font_ui)

        # Layout: [Controls | Plot | Colorbar]
        with dpg.group(horizontal=True):
            # Left: controls / buttons / stats (unchanged content)
            with dpg.child_window(width=420, height=PLOT_H + 200):
                dpg.add_input_float(label="VMIN [dB]", tag=IN_VMIN, default_value=vis_vmin)
                dpg.add_input_float(label="VMAX [dB]", tag=IN_VMAX, default_value=vis_vmax)
                dpg.add_input_float(label="Range Max [m]", tag=IN_RMAX, default_value=vis_rmax)
                dpg.add_input_float(label="Angle +/- [deg]", tag=IN_AABS, default_value=vis_ang_abs)
                dpg.add_button(label="Apply", tag=BTN_APPLY, callback=_apply_params)
                dpg.add_separator()
                dpg.add_text("SAR Capture")
                dpg.add_text(f"Settling delay: {SETTLING_DELAY_S:.2f} s")
                dpg.add_text(f"Frames per position: {FRAMES_PER_POSITION}")
                dpg.add_button(label="CAPTURE", tag=BTN_CAPTURE, callback=_on_btn_capture, width=360)
                dpg.add_text("Position counter: 0", tag=TXT_POS_TAG)
                dpg.add_text("", tag=TXT_LOG_TAG)
                dpg.add_spacer(height=10)

                if DEBUG_STATS:
                    dpg.add_text("", tag=TXT_STATS_TAG)
                    dpg.bind_item_font(TXT_STATS_TAG, font_mono)

            # Right: plot + colorbar side-by-side

            CBAR_T_TOP  = "cbar_t_top"
            CBAR_T_MID  = "cbar_t_mid"
            CBAR_T_BOT  = "cbar_t_bot"
            CBAR_SP1    = "cbar_sp1"
            CBAR_SP2    = "cbar_sp2"

            CBAR_FIXED_W = 90   # larghezza pannello colorbar (puoi aumentare)
            CBAR_IMG_W   = 24   # larghezza immagine colorbar (texture scalata)


            RIGHT_PANEL = "right_panel"
            PLOT_TAG    = "heat_plot"
            XAX_TAG     = "x_axis"
            YAX_TAG     = "y_axis"
            SER_TAG     = "heat_series"
            CBAR_TAG    = "cbar_scale"

            with dpg.child_window(tag=RIGHT_PANEL, width=-1, height=-1):
                with dpg.group(horizontal=True):

                    # ---- PLOT (assi) ----
                    with dpg.plot(tag=PLOT_TAG, width=-1, height=-1):
                        dpg.add_plot_legend(show=False)
                        dpg.add_plot_axis(dpg.mvXAxis, label="Azimuth (deg)", tag=XAX_TAG)
                        dpg.add_plot_axis(dpg.mvYAxis, label="Range (m)", tag=YAX_TAG)

                        # immagine nel plot: bounds saranno impostati nel loop (in metri e gradi)
                        dpg.add_image_series(
                            TEX_TAG,
                            [ -vis_ang_abs, 0.0 ],      # pmin (x_min, y_min)
                            [  vis_ang_abs, vis_rmax ],  # pmax (x_max, y_max)
                            parent=YAX_TAG,
                            tag=SER_TAG,
                        )

                        # imposta colormap del plot
                        dpg.bind_colormap(PLOT_TAG, CMAP)

                    # ---- COLORBAR GRADUATA ----
                    with dpg.child_window(width=110, height=-1):
                        dpg.add_text("dB")
                        dpg.add_colormap_scale(
                            vmin=vis_vmin,
                            vmax=vis_vmax,
                            colormap=CMAP,
                            height=-1,
                            width=30,
                            label="",      # niente label interna
                            tag=CBAR_TAG,
                        )


    def _update_colorbar_labels():
        vmin = float(vis_vmin)
        vmax = float(vis_vmax)

        ticks = np.linspace(vmax, vmin, 5)

        dpg.set_value(CBAR_T_TOP, f"{ticks[0]:.0f}")
        dpg.set_value(CBAR_T_MID, f"{ticks[2]:.0f}")
        dpg.set_value(CBAR_T_BOT, f"{ticks[-1]:.0f}")


    def _layout_resize():
        rw, rh = dpg.get_item_rect_size(RIGHT_PANEL)
        if rw <= 10 or rh <= 10:
            return

        heat_w = int(rw - CBAR_FIXED_W - 8)
        heat_h = int(rh - 8)

        # Heatmap
        dpg.configure_item(HEAT_IMG, width=heat_w, height=heat_h)

        # Colorbar stessa altezza IDENTICA
        dpg.configure_item(CBAR_IMG, width=CBAR_IMG_W, height=heat_h)

        # posizionamento numeri esatto (no approssimazioni)
        top_margin = 25
        bottom_margin = 25
        usable = max(10, heat_h - top_marginToggleU - bottom_margin)

        dpg.configure_item(CBAR_SP1, height=int(usable * 0.25))
        dpg.configure_item(CBAR_SP2, height=int(usable * 0.25))


    _apply_params()
    try:
        while dpg.is_dearpygui_running():
            now = time.perf_counter()
            if DEBUG_STATS and (now - t_mon >= 2):
                dt_mon = now - t_mon
                t_mon = now

                dt_img = now - t_img_start
                img_hz = img_updates / dt_img if dt_img > 0 else 0.0
                img_updates = 0
                t_img_start = now

                with frame_put_ok.get_lock():
                    put_f = frame_put_ok.value
                with frame_get_ok.get_lock():
                    get_f = frame_get_ok.value
                backlog_f = put_f - get_f

                with gui_put_ok.get_lock():
                    put_g = gui_put_ok.value
                with gui_get_ok.get_lock():
                    get_g = gui_get_ok.value
                backlog_g = put_g - get_g

                with lost_pkts.get_lock():
                    lost_now = lost_pkts.value
                lost_delta = lost_now - lost_prev
                lost_prev = lost_now

                with rx_pkts.get_lock():
                    pkts_now = rx_pkts.value
                pkts_delta = pkts_now - pkts_prev
                pkts_prev = pkts_now

                with rx_put_drops.get_lock():
                    drop_f = rx_put_drops.value
                with gui_put_drops.get_lock():
                    drop_g = gui_put_drops.value

                with log_head.get_lock():
                    h = log_head.value
                with log_tail.get_lock():
                    t = log_tail.value

                fill = h - t
                fill_pct = 100.0 * fill / LOG_RING_N

                with log_overflow_drops.get_lock():
                    ovf = log_overflow_drops.value

                sat_f = 100.0 * backlog_f / FRAME_Q_MAX
                sat_g = 100.0 * backlog_g / GUI_Q_MAX

                total = pkts_delta + lost_delta
                loss_pct = (100.0 * lost_delta / total) if total > 0 else 0.0
                pkt_rate = (pkts_delta / dt_mon) if dt_mon > 0 else 0.0

                dpg.set_value(
                    TXT_STATS_TAG,
                    "+----------------+--------------------------+\n"
                    f"| {'LOSS':<14}| {f'{loss_pct:.3f} %':<26}|\n"
                    f"| {'RX RATE':<14}| {f'{pkt_rate:.0f} pkt/s':<26}|\n"
                    f"| {'GUI RATE':<14}| {f'{img_hz:.1f} Hz':<26}|\n"
                    "+----------------+--------------------------+\n"
                    f"| {'FRAME Q':<14}| {f'{backlog_f}/{FRAME_Q_MAX} drop {drop_f}':<26}|\n"
                    f"| {'GUI Q':<14}| {f'{backlog_g}/{GUI_Q_MAX} drop {drop_g}':<26}|\n"
                    f"| {'LOG RING':<14}| {f'{int(fill)}/{LOG_RING_N} ({fill_pct:.0f}%)':<26}|\n"
                    "+----------------+--------------------------+\n"
                    f"| {'LOG SAVED':<14}| {str(int(log_saved.value)):<26}|\n"
                    f"| {'LOG OVF':<14}| {str(int(ovf)):<26}|\n"
                    f"| {'DSP SKIP':<14}| {str(int(dsp_put_drops.value)):<26}|\n"
                    f"| {'SAR POS':<14}| {str(int(sar_pos_counter.value)):<26}|\n"
                    "+----------------+--------------------------+\n"
                    f"| {'RX PID':<14}| {f'{p_rx.pid} alive={p_rx.is_alive()}':<26}|\n"
                    f"| {'LOG PID':<14}| {f'{p_log.pid} alive={p_log.is_alive()}':<26}|\n"
                    f"| {'DSP PID':<14}| {f'{p_dsp.pid} alive={p_dsp.is_alive()}':<26}|\n"
                    "+----------------+--------------------------+\n"
                    f"| {'RAM MAIN':<14}| {f'{_rss_bytes_pid(os.getpid())/1e6:.1f} MB':<26}|\n"
                    f"| {'RAM RX':<14}| {f'{_rss_bytes_pid(p_rx.pid or 0)/1e6:.1f} MB':<26}|\n"
                    f"| {'RAM LOG':<14}| {f'{_rss_bytes_pid(p_log.pid or 0)/1e6:.1f} MB':<26}|\n"
                    f"| {'RAM DSP':<14}| {f'{_rss_bytes_pid(p_dsp.pid or 0)/1e6:.1f} MB':<26}|\n"
                    "+----------------+--------------------------+"
                )

                dpg.set_value(
                    TXT_LOG_TAG,
                    "LOGGER\n"
                    f"data: {data_path.name}\n"
                    f"idx : {idx_path.name}\n"
                    f"hdr : {hdr_path.name}\n",
                )

            try:
                view_db = gui_q.get_nowait()
                with gui_get_ok.get_lock():
                    gui_get_ok.value += 1
            except pyqueue.Empty:
                view_db = None

            if view_db is not None:
                H, W = view_db.shape
                rmax_plot = int(np.floor(vis_rmax / (C * FS / (2.0 * SLOPE * NFFT_RANGE))))
                rmax_plot = max(1, min(rmax_plot, H))

                aabs = vis_ang_abs
                n_angles = W
                ang_axis = np.linspace(-ANG_ABS_MAX, ANG_ABS_MAX, n_angles)
                idx_min = int(np.floor((( -aabs - (-ANG_ABS_MAX)) / (2 * ANG_ABS_MAX)) * n_angles))
                idx_max = int(np.ceil((( aabs - (-ANG_ABS_MAX)) / (2 * ANG_ABS_MAX)) * n_angles))
                idx_min = max(0, idx_min)
                idx_max = min(n_angles, idx_max)
                if idx_min >= idx_max:
                    idx_min = 0
                    idx_max = n_angles

                sub = view_db[:rmax_plot, idx_min:idx_max]

                vmin = vis_vmin
                vmax = vis_vmax
                sub = np.clip(sub, vmin, vmax)
                norm = (sub - vmin) / (vmax - vmin + 1e-9)
                rgba = _jet_rgba(norm)

                tex_h, tex_w = HEATMAP_TEX_H, HEATMAP_TEX_W
                src_h, src_w = rgba.shape[:2]
                yy = np.linspace(0, src_h - 1, tex_h)
                xx = np.linspace(0, src_w - 1, tex_w)
                xi, yi = np.meshgrid(xx, yy)
                xi_int = np.clip(xi.astype(np.int32), 0, src_w - 1)
                yi_int = np.clip(yi.astype(np.int32), 0, src_h - 1)
                rgba_resampled = rgba[yi_int, xi_int, :]
                rgba_resampled = np.flipud(rgba_resampled)

                tex_buf = rgba_resampled.reshape(-1)
                dpg.set_value(TEX_TAG, tex_buf)

                if DEBUG_STATS:
                    img_updates += 1

            _layout_resize()
            dpg.render_dearpygui_frame()


    except KeyboardInterrupt:
        print("Chiusura...")
    finally:
        _shutdown()
        dpg.destroy_context()


if __name__ == "__main__":
    mp.freeze_support()
    main()
