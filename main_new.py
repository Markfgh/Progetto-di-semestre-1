import socket
import time
import queue as pyqueue
from pathlib import Path
from multiprocessing import Process, Queue, Value
from array import array
import struct
import json
import platform
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


def _fmt_mb(nbytes: int) -> str:
    return f"{(nbytes / (1024 * 1024)):.1f}MB"

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
FC = float(cfg["radar"]["fc"])

LAMBDA = C / FC

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

# valori iniziali di visualizzazione (partono dai limiti config, ma poi l'utente può ridurli fino a 0)
RANGE_MAX_DISPLAY = float(cfg["display"]["range_max"])
CROSSRANGE_MAX_DISPLAY = float(cfg.get("display", {}).get("crossrange_max", RANGE_MAX_DISPLAY))
# --- CODE QUEUE ---
DEBUG_STATS = bool(cfg["debug"]["debug_stats"])

# --- WORKERS FFT ---
FFT_WORKERS = int(cfg.get("dsp", {}).get("fft_workers", 6))

# --- SAR capture-only parameters ---
sar_cfg = cfg.get("sar", {}) or {}
# Backward-compat: if not present, reuse old 'logger.settling_delay_s'
SETTLING_DELAY_S = float(sar_cfg.get("settling_delay_s", 0.4))
FRAMES_PER_POSITION = int(sar_cfg.get("frames_per_position", 8))


# --- QUEUE SIZE ---
FRAME_Q_MAX = 20 # dimensione coda frame tra RX e DSP (in numero di frame, non byte)
GUI_Q_MAX = 5 # dimensione coda tra DSP e GUI (in numero di heatmap, non byte).



# ----------------------------
# FUNZIONI DI ELABORAZIONE
# ----------------------------
def radar_rx(
    cmd_queue: Queue,
    log_queue: Queue,
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
      - se viene rilevato un gap di pacchetti, il/i frame coinvolti vengono marcati come corrotti
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
                pending_start_t = now_perf + max(0.0, float(settling_delay_s))
                pending = True
                recording = False

    def _record_decision(now_perf: float):
        """Return (do_log, pos_id, frame_in_pos) and advance counters if logging."""
        nonlocal pending, recording, rec_frames_left, rec_frame_in_pos
        if pending and (now_perf >= pending_start_t):
            pending = False
            recording = True
        if recording and (rec_frames_left > 0):
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

        assert curr_slot is not None, "curr_slot should not be None after successful slot acquisition"
        base = int(curr_slot) * BYTES_PER_FRAME
        frame_view = shm_view[base : base + BYTES_PER_FRAME]
        # Se stiamo "consumando" bytes di un gap, ogni frame attraversato è corrotto.
        frame_ok = (not in_gap)
        return True

    def _push_frame(slot_to_push: int, ok: int, ts_ns: int, do_log: bool, pos_id: int = 0, frame_in_pos: int = 0):
        """Invia il frame:
        - do_log=True: manda al logger (scrittura su disco) che poi farà forward best-effort al DSP.
        - do_log=False: bypass diretto al DSP/GUI best-effort (non bloccare RX).
        """
        if do_log:
            # In modalità recording NON vogliamo perdere frame: backpressure sul logger.
            log_queue.put((int(slot_to_push), int(ok), int(pos_id), int(frame_in_pos), int(ts_ns)))
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
            continue

        # ---------------- GESTIONE GAP (CORRUZIONE) ----------------
        if last_seq is not None:
            gap = seq - last_seq - 1
            if gap > 0:
                with lost_pkts.get_lock():
                    lost_pkts.value += gap

                frame_ok = False
                in_gap = True

                bytes_missing = gap * payload_len_ref
                while bytes_missing > 0 and (not stop_evt.is_set()):
                    if not ensure_slot():
                        break

                    take = min(bytes_missing, BYTES_PER_FRAME - w)
                    w += take
                    bytes_missing -= take

                    if w == BYTES_PER_FRAME:
                        # frame interamente perso -> azzera e, se in recording, logga
                        assert frame_view is not None, "frame_view should not be None"
                        frame_view[:] = ZERO_FRAME_BLOCK

                        slot_to_push = curr_slot
                        assert slot_to_push is not None, "slot_to_push should not be None"
                        ts_ns = time.time_ns()
                        do_log, pos_id, finpos = _record_decision(time.perf_counter())
                        _push_frame(slot_to_push, 0, ts_ns, do_log, pos_id, finpos)

                        w = 0
                        curr_slot = None
                        frame_view = None

                in_gap = False
                frame_ok = True ## Fine fase gap: riparti pulito (altrimenti rischi di azzerare anche il primo frame reale dopo il gap)
            elif gap < 0:
                frame_ok = False

        last_seq = seq

        # ---------------- COPIA PAYLOAD ----------------
        off = HEADER_LEN
        current_payload_len = n_bytes - off
        payload_cursor = 0

        while payload_cursor < current_payload_len:
            if not ensure_slot():
                break

            assert frame_view is not None, "frame_view should not be None"
            chunk_size = min(current_payload_len - payload_cursor, BYTES_PER_FRAME - w)
            start_src = off + payload_cursor

            frame_view[w : w + chunk_size] = packet_view[start_src : start_src + chunk_size]

            w += chunk_size
            payload_cursor += chunk_size

            if w == BYTES_PER_FRAME:
                if not frame_ok:
                    assert frame_view is not None, "frame_view should not be None"
                    frame_view[:] = ZERO_FRAME_BLOCK

                slot_to_push = curr_slot
                assert slot_to_push is not None, "slot_to_push should not be None"
                ts_ns = time.time_ns()
                do_log, pos_id, finpos = _record_decision(time.perf_counter())
                _push_frame(slot_to_push, 1 if frame_ok else 0, ts_ns, do_log, pos_id, finpos)

                w = 0
                curr_slot = None
                frame_view = None
                frame_ok = True
def logger_worker(
    log_queue: Queue,
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

    # Header "umano" (utile per debug e player)
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

    IDX_MAGIC = b"RLOGIDX2"
    IDX_VERSION = 2
    REC_SIZE = 32

    # idx header (32B)
    IDX_HDR_FMT = "<8s H H I H H B B H I I"  # 32 bytes
    idx_header = struct.pack(
        IDX_HDR_FMT,
        IDX_MAGIC,
        IDX_VERSION,
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

    # record (32B)
    REC_FMT = "<Q I H B B Q I I"  # frame_id, pos_id, frame_in_pos, ok, pad, ts_ns, r0, r1

    shm_view = memoryview(shm_frames).cast("B")
    frame_id = 0

    with open(data_path, "wb", buffering=1024 * 1024) as fbin, open(
        idx_path, "wb", buffering=1024 * 1024
    ) as fidx:
        fidx.write(idx_header)

        while True:
            try:
                item = log_queue.get(timeout=0.5)
            except pyqueue.Empty:
                continue

            if item is None:
                break

            slot, ok, pos_id, frame_in_pos, ts_ns = item

            base = int(slot) * BYTES_PER_FRAME
            fbin.write(shm_view[base : base + BYTES_PER_FRAME])

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

        try:
            fbin.flush()
            fidx.flush()
        except Exception:
            pass

# ----------------------------
# WORKER DSP
# ----------------------------
def dsp_worker(frame_queue, free_slots, shm_frames, gui_queue, gui_put_drops, frame_get_ok, gui_put_ok, stop_evt):
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
    # 1. Buffer per i dati int16 grezzi
    i16_batch = np.zeros((X_FRAMES, i16_per_frame), dtype=np.int16)
    
    # 2. Buffer per i dati complessi (Input del DSP)
    #    Lo allochiamo piatto, poi useremo reshape/view
    complex_data = np.zeros(total_samples_needed, dtype=np.complex64)
    
    shm_view = memoryview(shm_frames).cast("B")
    batch_slots = []
    heatmap_ema = None
    
    if RX != 4:
        raise ValueError("DSP: conversione I/Q attuale assume RX=4 (packing IIIIQQQQ).")


    while True:
        try:
            # Blocca solo sul primo frame
            slot = frame_queue.get(timeout=0.5)
            if slot is None:
                break
            if DEBUG_STATS:
                with frame_get_ok.get_lock(): frame_get_ok.value += 1
            batch_slots.append(slot)
        except pyqueue.Empty:
            if stop_evt.is_set():
                break
            continue        # --- Lag protection (senza qsize): drena opportunisticamente alcuni elementi ---
        # Evita qsize() perché può essere costoso/non affidabile su multiprocessing.
        terminate = False
        try:
            for _ in range(FRAME_Q_MAX):  # limite hard per non stare troppo qui
                s_old = frame_queue.get_nowait()
                if s_old is None:
                    terminate = True
                    break
                if DEBUG_STATS:
                    with frame_get_ok.get_lock():
                        frame_get_ok.value += 1
                batch_slots.append(s_old)
        except pyqueue.Empty:
            pass

        if terminate:
            # libera eventuali slot già presi e termina
            for s in batch_slots:
                try:
                    free_slots.put(s)
                except Exception:
                    pass
            break

        # Tieni solo gli ultimi X_FRAMES necessari
        if len(batch_slots) > X_FRAMES:
            drop_slots = batch_slots[:-X_FRAMES]
            batch_slots = batch_slots[-X_FRAMES:]
            for s in drop_slots:
                free_slots.put(s)


        # Se non ho abbastanza frame dopo il drain, continuo ad aspettare
        if len(batch_slots) < X_FRAMES:
            continue

        # --- 1. COPIA DA SHARED MEMORY (Veloce) ---
        for k, s in enumerate(batch_slots):
            base = s * BYTES_PER_FRAME
            # Copia diretta nel buffer pre-allocato
            i16_batch[k, :] = np.frombuffer(shm_view[base : base + BYTES_PER_FRAME], dtype=np.int16)
            
        # Restituisci subito gli slot al RX (Pipelining)
        for s in batch_slots:
            free_slots.put(s)
        batch_slots.clear()

        # --- 2. CONVERSIONE INT16 -> COMPLEX64 (Zero Alloc) ---
        # reshape view temporanea (costo quasi nullo)
        # Assumiamo struttura: [Re, Re, Re, Re, Im, Im, Im, Im] per blocchi da 8
        flat_i16 = i16_batch.reshape(-1)
        n_blocks = flat_i16.size // 8
        
        # View strutturata (n, 8)
        block_view = flat_i16[:n_blocks*8].reshape(n_blocks, 8)
        
        # Copia nei canali Real e Imag del buffer complesso PRE-ALLOCATO
        # Nota: usiamo [:] per forzare la copia in-place senza riallocare
        complex_data.real[:] = block_view[:, :4].reshape(-1)  # type: ignore
        complex_data.imag[:] = block_view[:, 4:].reshape(-1)  # type: ignore


        # --- 3. PROCESSING ---
        # Passiamo il buffer statico 'complex_data' alla funzione
        heatmap_ema = process_buffer(
            complex_data, 
            window_range, 
            window_angle, 
            heatmap_ema, 
            0.2, # alpha
            gui_queue, 
            gui_put_drops, 
            gui_put_ok, 
            max_bin
        )

def process_buffer(raw_buffer, w_range, w_angle, heatmap_ema, alpha, gui_queue, gui_put_drops, gui_put_ok, max_bin):

    try:
        # A. Reshape SENZA transpose (evita copia grossa)

        # Shape: (F, chirpsPerTx, TX, SAMPLES, RX)  -> samples è axis=3

        data = raw_buffer.reshape(X_FRAMES, CHIRPS // TX, TX, SAMPLES, RX)


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

        virtual_array = va.reshape(X_FRAMES, CHIRPS // TX, NFFT_RANGE, VIRTUAL_ANT)


        # Finestra Angolo IN-PLACE (w_angle: (1,1,1,VIRTUAL_ANT))

        virtual_array *= w_angle


        # D. ANGLE FFT

        angle_fft = fft.fft(virtual_array, n=NFFT_ANGLE, axis=-1, workers=FFT_WORKERS, overwrite_x=True)

        angle_fft = fft.fftshift(angle_fft, axes=-1)
        
        # Modulo quadro e media
        re = angle_fft.real
        im = angle_fft.imag
        heatmap = (re * re + im * im).mean(axis=(0, 1))

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
        view_db = heatmap_db[:max_bin, :]
        if view_db.size > 0:
            mx = np.max(view_db)
            view_db -= mx

        # push verso GUI
        try:
            gui_queue.put_nowait(view_db)
            if DEBUG_STATS:
                with gui_put_ok.get_lock():
                    gui_put_ok.value += 1

        except pyqueue.Full:
            if DEBUG_STATS:
                with gui_put_drops.get_lock():
                    gui_put_drops.value += 1

        return heatmap_ema

    except Exception as e:
        print(f"[DSP ERR] {e}")
        return heatmap_ema


def main():
    # --- SETUP CODE E PROCESSI (INVARIATO) ---
    frame_q = Queue(maxsize=FRAME_Q_MAX)
    free_slots = Queue()

    N_SLOTS = FRAME_Q_MAX + X_FRAMES + 64
    shm_frames = RawArray("B", N_SLOTS * BYTES_PER_FRAME)
    for i in range(N_SLOTS):
        free_slots.put(i)
    
    gui_q = Queue(maxsize=GUI_Q_MAX)

    stop_evt = mp.Event()
    sar_pos_counter = Value("L", 0)
    log_saved = Value("L", 0)
    dsp_put_drops = Value("L", 0)

    log_q = Queue(maxsize=0)
    cmd_q = Queue(maxsize=16)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).with_name("logs")
    out_dir.mkdir(parents=True, exist_ok=True) 
    data_path = out_dir / f"run_{run_id}.bin"
    idx_path  = out_dir / f"run_{run_id}.idx"
    hdr_path  = out_dir / f"run_{run_id}.json"

    lost_pkts = Value("L", 0)
    rx_pkts = Value("L", 0)
    rx_put_drops = Value("L", 0)
    gui_put_drops = Value("L", 0)
    frame_put_ok = Value("L", 0)
    frame_get_ok = Value("L", 0)
    gui_put_ok = Value("L", 0)
    gui_get_ok = Value("L", 0)

    # --- AVVIO PROCESSI ---
    p_rx = Process(
        target=radar_rx,
        args=(cmd_q, log_q, free_slots, shm_frames, lost_pkts, rx_pkts, rx_put_drops, frame_q, frame_put_ok, stop_evt, SETTLING_DELAY_S, FRAMES_PER_POSITION),
    )
    p_log = Process(
        target=logger_worker,
        args=(log_q, frame_q, free_slots, shm_frames, stop_evt, log_saved, dsp_put_drops, frame_put_ok, str(data_path), str(idx_path), str(hdr_path), SETTLING_DELAY_S, FRAMES_PER_POSITION),
    )
    p_dsp = Process(
        target=dsp_worker,
        args=(frame_q, free_slots, shm_frames, gui_q, gui_put_drops, frame_get_ok, gui_put_ok, stop_evt),
    )

    p_rx.daemon = True
    p_log.daemon = True
    p_dsp.daemon = True
    p_rx.start()
    p_log.start()
    p_dsp.start()

    print(f"[LOGGER] data: {data_path}")

    # =========================================================================
    # GUI SETUP (RESPONSIVE - CLEAN)
    # =========================================================================

    # 1) Display params da YAML
    disp = cfg.get("display", {}) or {}

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
        try:
            p_rx.terminate(); p_rx.join(0.1)
        except Exception:
            pass
        try:
            log_q.put_nowait(None); p_log.terminate(); p_log.join(0.1)
        except Exception:
            pass
        try:
            frame_q.put_nowait(None); p_dsp.terminate(); p_dsp.join(0.1)
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
        return f"| {k:<9} | {v:<14} |\n"

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
                        # width=-1 così si adatta alla colonna; height=-1 così prende tutta l'altezza
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

                # Backlogs & Drops
                with frame_put_ok.get_lock(): put_f = frame_put_ok.value
                with frame_get_ok.get_lock(): get_f = frame_get_ok.value
                backlog_f = put_f - get_f

                with gui_put_ok.get_lock(): put_g = gui_put_ok.value
                with gui_get_ok.get_lock(): get_g = gui_get_ok.value
                backlog_g = put_g - get_g

                with rx_put_drops.get_lock(): drop_f = rx_put_drops.value
                with gui_put_drops.get_lock(): drop_g = gui_put_drops.value

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
                
                # --- AGGIORNAMENTO TESTI GUI (FORMATO TABELLA ASCII) ---
                stats_str = (
                    "+-----------+----------------+\n" +
                    row("LOSS", f"{loss_pct:.3f}%") +
                    row("RX RATE", f"{pkt_rate:.0f} pkt/s") +
                    row("GUI RATE", f"{img_hz:.1f} Hz") +
                    "+-----------+----------------+\n" +
                    row("FRAME Q", f"{backlog_f}/{FRAME_Q_MAX} drop{drop_f}") +
                    row("GUI Q", f"{backlog_g}/{GUI_Q_MAX} drop{drop_g}") +
                    "+-----------+----------------+\n" +
                    row("LOG", f"{int(log_saved.value)}") +
                    row("DSP SKIP", f"{int(dsp_put_drops.value)}") +
                    row("SAR POS", f"{int(sar_pos_counter.value)}") +
                    "+-----------+----------------+\n" +
                    row("RX", f"{p_rx.pid} {p_rx.is_alive()}") +
                    row("LOG", f"{p_log.pid} {p_log.is_alive()}") +
                    row("DSP", f"{p_dsp.pid} {p_dsp.is_alive()}") +
                    "+-----------+----------------+\n" +
                    row("RAM MAIN", f"{_rss_bytes_pid(os.getpid())/1e6:.1f}MB") +
                    row("RAM RX", f"{_rss_bytes_pid(p_rx.pid or 0)/1e6:.1f}MB") +
                    row("RAM LOG", f"{_rss_bytes_pid(p_log.pid or 0)/1e6:.1f}MB") +
                    row("RAM DSP", f"{_rss_bytes_pid(p_dsp.pid or 0)/1e6:.1f}MB") +
                    "+-----------+----------------+\n"
                )

                dpg.set_value(TXT_STATS_TAG, stats_str)
                
                # Update Log Info
                dpg.set_value(TXT_LOG_TAG, 
                    "LOGGER\n"
                    f"{'file':<12}{data_path.name}\n"
                    f"{'saved':<12}{int(log_saved.value):>8d}\n"
                    f"{'dsp_skip':<12}{int(dsp_put_drops.value):>8d}\n"
                    f"{'position':<12}{int(sar_pos_counter.value):>8d}"
                )

            # 2. GUI TEXTURE UPDATE
            heatmap = None
            while True:
                try:
                    heatmap = gui_q.get_nowait()
                    if DEBUG_STATS:
                        with gui_get_ok.get_lock(): gui_get_ok.value += 1
                except pyqueue.Empty:
                    break
            
                if heatmap is not None:
                    denom = (vis_vmax - vis_vmin)
                    if denom < 1e-6:
                        denom = 1e-6

                # heatmap: (range_bins, x_bins)
                src = heatmap.astype(np.float32, copy=False)

                # --- range window (Y) in bins ---
                max_r_bin = int(vis_rmax / dr_plot)
                max_r_bin = max(1, min(max_r_bin, src.shape[0]))

                # --- cross-range window (X) in bins (centered) ---
                half_bins = src.shape[1] // 2
                if CROSSRANGE_MAX_DISPLAY > 1e-6:
                    keep = int(half_bins * (abs(vis_xmax) / float(CROSSRANGE_MAX_DISPLAY)))
                else:
                    keep = half_bins
                keep = max(8, min(half_bins, keep))

                x0 = half_bins - keep
                x1 = half_bins + keep

                win = src[:max_r_bin, x0:x1]   # (Ry, Xw)

                # --- RESCALE window -> full texture (tex_h, tex_w) (zoom effect) ---
                # Nearest-neighbor resample (fast, no scipy)
                ys = np.linspace(0, win.shape[0] - 1, tex_h).astype(np.int32)
                xs = np.linspace(0, win.shape[1] - 1, tex_w).astype(np.int32)
                img = win[ys[:, None], xs[None, :]]   # (tex_h, tex_w)

                # Normalize + colormap
                norm = (img - vis_vmin) / denom
                norm = np.clip(norm, 0.0, 1.0)
                rgba = _jet_rgba(norm)
                rgba = rgba[::-1, :, :]  # vertical flip only (Y up)
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
