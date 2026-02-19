import socket
import time
import queue as pyqueue
from pathlib import Path
from multiprocessing import Process, Queue, Value
from array import array
import struct
import json

from multiprocessing.sharedctypes import RawArray, Synchronized
import multiprocessing as mp

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
                    ("PeakWorkingSetSize", wt.SIZE_T),
                    ("WorkingSetSize", wt.SIZE_T),
                    ("QuotaPeakPagedPoolUsage", wt.SIZE_T),
                    ("QuotaPagedPoolUsage", wt.SIZE_T),
                    ("QuotaPeakNonPagedPoolUsage", wt.SIZE_T),
                    ("QuotaNonPagedPoolUsage", wt.SIZE_T),
                    ("PagefileUsage", wt.SIZE_T),
                    ("PeakPagefileUsage", wt.SIZE_T),
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
                page = os.sysconf("SC_PAGE_SIZE")
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
VMIN = float(cfg["display"]["vmin"])
VMAX = float(cfg["display"]["vmax"])
RANGE_MAX_DISPLAY = float(cfg["display"]["range_max"])

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
                        frame_view[:] = ZERO_FRAME_BLOCK

                        slot_to_push = curr_slot
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

            chunk_size = min(current_payload_len - payload_cursor, BYTES_PER_FRAME - w)
            start_src = off + payload_cursor

            frame_view[w : w + chunk_size] = packet_view[start_src : start_src + chunk_size]

            w += chunk_size
            payload_cursor += chunk_size

            if w == BYTES_PER_FRAME:
                if not frame_ok:
                    frame_view[:] = ZERO_FRAME_BLOCK

                slot_to_push = curr_slot
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
        complex_data.real[:] = block_view[:, :4].reshape(-1)
        complex_data.imag[:] = block_view[:, 4:].reshape(-1)

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

        data -= data.mean(axis=3, keepdims=True, dtype=np.complex64)


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




# ----------------------------
# MAIN (GRAFICA)
# ----------------------------


def main():
    # Queue di indici (slot) pronti per DSP
    frame_q = Queue(maxsize=FRAME_Q_MAX)
    #Pool di slot liberi (evita overwrite e copie)
    free_slots = Queue()

    # Shared ring per i frame (byte)
    N_SLOTS = FRAME_Q_MAX + X_FRAMES + 64  # margine extra per logging
    shm_frames = RawArray("B", N_SLOTS * BYTES_PER_FRAME)
    for i in range(N_SLOTS):
        free_slots.put(i)
    
    gui_q = Queue(maxsize=GUI_Q_MAX)


    # ----------------------------
    # Logger (sempre: tutti i frame vengono salvati su disco)
    # ----------------------------
    stop_evt = mp.Event()

    # SAR: contatore posizioni (incrementato ad ogni CAPTURE)
    sar_pos_counter = Value("L", 0)
    log_saved = Value("L", 0)      # quanti frame sono stati scritti su disco
    dsp_put_drops = Value("L", 0)  # quanti frame loggati NON sono stati inviati al DSP (queue piena)

    # coda tra RX e logger (slot + metadati). La dimensione segue N_SLOTS per evitare deadlock.
    # NOTA: ogni elemento occupa pochissimo (solo indici+metadati), i bytes stanno in shm_frames.
    log_q = Queue(maxsize=0)  # 0 -> best effort "unbounded" (in pratica: evita drop)
    cmd_q = Queue(maxsize=16)

    # file di output (run_id unico)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).with_name("logs")
    data_path = out_dir / f"run_{run_id}.bin"
    idx_path  = out_dir / f"run_{run_id}.idx"
    hdr_path  = out_dir / f"run_{run_id}.json"

    # stats condivise
    lost_pkts = Value("L", 0)
    rx_pkts = Value("L", 0)
    rx_put_drops = Value("L", 0)
    gui_put_drops = Value("L", 0)

    frame_put_ok = Value("L", 0)
    frame_get_ok = Value("L", 0)

    gui_put_ok = Value("L", 0)
    gui_get_ok = Value("L", 0)

    p_rx = Process(
        target=radar_rx,
        args=(cmd_q, log_q, free_slots, shm_frames, lost_pkts, rx_pkts, rx_put_drops, frame_q, frame_put_ok, stop_evt, SETTLING_DELAY_S, FRAMES_PER_POSITION),
    )

    p_log = Process(
        target=logger_worker,
        args=(
            log_q,
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
    p_log.daemon = True
    p_dsp.daemon = True
    p_rx.start()
    p_log.start()
    p_dsp.start()

    print(f"[LOGGER] data: {data_path}")
    print(f"[LOGGER] idx : {idx_path}")
    print(f"[LOGGER] hdr : {hdr_path}")

    # ----------------------------
    # GUI (DEAR PYGUI) - Texture (Opzione B)
    # ----------------------------

    # limiti angolo (deg) da YAML (fallback: -90..+90)
    disp = cfg.get("display", {}) or {}
    ANG_MIN_CFG = float(disp.get("angle_min", -90.0))
    ANG_MAX_CFG = float(disp.get("angle_max", 90.0))
    if ANG_MAX_CFG <= ANG_MIN_CFG:
        ANG_MIN_CFG, ANG_MAX_CFG = -90.0, 90.0
    ANG_ABS_MAX = float(min(abs(ANG_MIN_CFG), abs(ANG_MAX_CFG)))

    # risoluzione in range (metri per bin)
    dr_plot = C * FS / (2.0 * SLOPE * NFFT_RANGE)

    # parametri visualizzazione (default)
    vis_vmin = float(VMIN)
    vis_vmax = float(VMAX)
    vis_rmax = float(RANGE_MAX_DISPLAY)   # [0..x] metri
    vis_ang_abs = float(ANG_ABS_MAX)          # [-ang, +ang]
    # monitor STATS (1 Hz)

    t_mon = time.perf_counter()
    t_img_start = time.perf_counter()
    img_updates = 0
    lost_prev = 0
    pkts_prev = 0
    # --- Setup DearPyGui ---
    dpg.create_context()
    with dpg.font_registry():
        font_ui = dpg.add_font(r"C:\Windows\Fonts\segoeui.ttf", 16)
        font_mono = dpg.add_font(r"C:\Windows\Fonts\consola.ttf", 14)  # più piccolo
    dpg.bind_font(font_ui)  # font normale per tutta la UI

    # tags
    TXT_STATS_TAG = "txt_stats"
    IN_VMIN = "in_vmin"
    IN_VMAX = "in_vmax"
    IN_RMAX = "in_rmax"
    IN_AABS = "in_aabs"
    BTN_APPLY = "btn_apply"
    # SAR / Logger tags
    TXT_LOG_TAG = "txt_log"

    TEX_REG = "tex_reg"
    TEX_TAG = "heat_tex"
    IMG_SERIES_TAG = "img_series"
    HEAT_PLOT_TAG = "heat_plot"
    XAXIS_TAG = "xaxis"
    YAXIS_TAG = "yaxis"
    CMAP_SCALE_TAG = "cmap_scale"

    # texture state (texture size fixed: evita ricreazioni runtime che possono crashare)
    # texture size = (NFFT_ANGLE x max_bin_tex)
    max_bin_tex = int(np.floor(RANGE_MAX_DISPLAY / dr_plot))
    max_bin_tex = max(1, min(max_bin_tex, NFFT_RANGE // 2))
    tex_w = int(NFFT_ANGLE)
    tex_h = int(max_bin_tex)
    # ---- TEXTURE BUFFER PERSISTENTE ----
    tex_buf = array('f', [0.0]) * (tex_w * tex_h * 4)
    tex_np  = np.frombuffer(tex_buf, dtype=np.float32)

    def _shutdown():
        """Termina i processi figli in modo robusto e senza perdere gli ultimi frame nel log."""
        try:
            stop_evt.set()
        except Exception:
            pass

        # 1) ferma RX per primo (così nessun frame viene accodato dopo il sentinel)
        try:
            p_rx.join(timeout=2.0)
        except Exception:
            pass
        try:
            if p_rx.is_alive():
                p_rx.terminate()
        except Exception:
            pass

        # 2) chiudi logger DOPO RX: così drena tutta la coda e fa flush
        try:
            log_q.put_nowait(None)
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

        # 3) chiudi DSP/GUI queue
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
        """Colormap Jet (approx) -> RGBA float32 in [0,1]. norm01 shape (H,W)."""
        x = np.clip(norm01.astype(np.float32, copy=False), 0.0, 1.0)
        r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
        a = np.ones_like(r, dtype=np.float32)
        return np.stack((r, g, b, a), axis=-1)  # (H,W,4)

    
    def _apply_params(sender=None, app_data=None, user_data=None):
        """Applica valori input (vmin/vmax, range max, ang abs) e aggiorna limiti/scala."""
        nonlocal vis_vmin, vis_vmax, vis_rmax, vis_ang_abs

        vmin = float(dpg.get_value(IN_VMIN))
        vmax = float(dpg.get_value(IN_VMAX))
        if vmax <= vmin:
            vmax = vmin + 1e-6

        rmax = float(dpg.get_value(IN_RMAX))
        aabs = float(dpg.get_value(IN_AABS))

        vis_vmin, vis_vmax, vis_rmax, vis_ang_abs = vmin, vmax, rmax, aabs

        # aggiorna scala colori
        try:
            dpg.configure_item(CMAP_SCALE_TAG, min_scale=vis_vmin, max_scale=vis_vmax, colormap=dpg.mvPlotColormap_Jet)
            dpg.bind_colormap(CMAP_SCALE_TAG, dpg.mvPlotColormap_Jet)
        except Exception:
            pass

        # aggiorna limiti assi (ZOOM/CROP):
        # - i bounds dell'immagine restano fissi ai limiti fisici da YAML
        # - qui cambiamo SOLO i limiti degli assi (come con ax.set_xlim/ylim in matplotlib)
        ang_abs = float(max(1.0, min(abs(vis_ang_abs), ANG_ABS_MAX)))
        x_min = float(max(ANG_MIN_CFG, -ang_abs))
        x_max = float(min(ANG_MAX_CFG, +ang_abs))
        y_max = float(min(max(0.05, vis_rmax), RANGE_MAX_DISPLAY))

        try:
            dpg.set_axis_limits(XAXIS_TAG, x_min, x_max)
        except Exception:
            pass
        try:
            dpg.set_axis_limits(YAXIS_TAG, 0.0, y_max)
        except Exception:
            pass

    # --- Layout GUI ---
    PLOT_H = 820
    CBAR_W = 90

    # ----------------------------
    # SAR capture-only controls (GUI)
    # ----------------------------
    # Un solo tasto: CAPTURE.
    # La logica è automatica: dopo il click, RX attende SETTLING_DELAY_S e registra FRAMES_PER_POSITION frame.
    TXT_POS_TAG = "txt_pos_counter"

    def _on_btn_capture(sender=None, app_data=None, user_data=None):
        # incrementa contatore posizioni e invia comando al processo RX
        with sar_pos_counter.get_lock():
            sar_pos_counter.value += 1
            pos_id = int(sar_pos_counter.value)
        try:
            cmd_q.put_nowait(("CAPTURE", pos_id))
        except pyqueue.Full:
            # se piena (molto raro), ignora: non bloccare la GUI
            pass
        try:
            dpg.set_value(TXT_POS_TAG, f"Position counter: {pos_id}")
        except Exception:
            pass

    with dpg.window(label="Real-Time MIMO Radar Heatmap", tag="main_win"):
        with dpg.group(horizontal=True):

            # --- pannello sinistro: controlli + stats ---
            with dpg.child_window(width=400, height=-1):
                dpg.add_text("Parametri visualizzazione ")
                dpg.add_input_float(label="Vmin (dB)", tag=IN_VMIN, default_value=vis_vmin, step=1.0, width=220)
                dpg.add_input_float(label="Vmax (dB)", tag=IN_VMAX, default_value=vis_vmax, step=1.0, width=220)
                dpg.add_input_float(label="Range max (m)", tag=IN_RMAX, default_value=vis_rmax, step=0.1, width=220)
                dpg.add_input_float(label="Ang abs (deg)", tag=IN_AABS, default_value=vis_ang_abs, step=1.0, width=220)
                dpg.add_button(label="Apply", tag=BTN_APPLY, callback=_apply_params)
                dpg.add_separator()
                dpg.add_text("SAR Capture")
                dpg.add_text(f"Settling delay: {SETTLING_DELAY_S:.2f} s")
                dpg.add_text(f"Frames per position: {FRAMES_PER_POSITION}")
                dpg.add_button(label="CAPTURE", callback=_on_btn_capture, width=360)
                dpg.add_text("Position counter: 0", tag=TXT_POS_TAG)
                dpg.add_text("", tag=TXT_LOG_TAG)
                dpg.add_spacer(height=10)

                if DEBUG_STATS:
                    dpg.add_text("", tag=TXT_STATS_TAG)
                    dpg.bind_item_font(TXT_STATS_TAG, font_mono)


            # --- centro: plot ---
            with dpg.plot(label="Heatmap", height=PLOT_H, width=-1, tag=HEAT_PLOT_TAG):
                dpg.add_plot_axis(dpg.mvXAxis, label="Azimuth (deg)", tag=XAXIS_TAG)
                dpg.add_plot_axis(dpg.mvYAxis, label="Range (m)", tag=YAXIS_TAG)

                with dpg.texture_registry(tag=TEX_REG, show=False):
                    _default = [0.0, 0.0, 0.0, 1.0] * (tex_w * tex_h)
                    dpg.add_dynamic_texture(width=tex_w, height=tex_h, default_value=_default, tag=TEX_TAG)
                dpg.add_image_series(
                    TEX_TAG,
                    bounds_min=(float(ANG_MIN_CFG), 0.0),
                    bounds_max=(float(ANG_MAX_CFG), float(RANGE_MAX_DISPLAY)),
                    tag=IMG_SERIES_TAG,
                    parent=YAXIS_TAG,
                )

                dpg.set_axis_limits(XAXIS_TAG, float(ANG_MIN_CFG), float(ANG_MAX_CFG))
                dpg.set_axis_limits(YAXIS_TAG, 0.0, float(RANGE_MAX_DISPLAY))


            # --- destra: colorbar della stessa altezza del plot ---
            with dpg.child_window(width=CBAR_W, height=PLOT_H):
                dpg.add_text("Jet")
                dpg.add_colormap_scale(
                    label="dB",
                    colormap=dpg.mvPlotColormap_Jet,
                    min_scale=vis_vmin,
                    max_scale=vis_vmax,
                    tag=CMAP_SCALE_TAG,
                    height=PLOT_H - 30,
                    width=CBAR_W - 10,
                )
                try:
                    dpg.bind_colormap(CMAP_SCALE_TAG, dpg.mvPlotColormap_Jet)
                except Exception:
                    pass

    dpg.create_viewport(title="Radar Heatmap", width=1400, height=950)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    # apply defaults once
    _apply_params()

    try:
        while dpg.is_dearpygui_running():
            now = time.perf_counter()
            # ---------------- STATISTICHE (Ogni 1 sec) ----------------
            if DEBUG_STATS and (now - t_mon >= 1):
                dt_mon = now - t_mon
                t_mon = now

                # Hz aggiornamento immagine
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
                    "+----------------+--------------------------+\n"
                    f"| {'LOG SAVED':<14}| {str(int(log_saved.value)):<26}|\n"
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





                # info breve logger (pannello SAR)
                dpg.set_value(
                    TXT_LOG_TAG,
                    "LOGGER\n"
                    f"{'file':<12}{data_path.name}\n"
                    f"{'saved':<12}{int(log_saved.value):>8d}\n"
                    f"{'dsp_skip':<12}{int(dsp_put_drops.value):>8d}\n"
                    f"{'position':<12}{int(sar_pos_counter.value):>8d}"
                )

                t_mon = now

            # ---------------- AGGIORNAMENTO GUI ----------------
            # Frame skipping: prendo solo l'ultimo heatmap disponibile
            # texture buffer persistente (NON cresce mai)
            heatmap = None
            while True:
                try:
                    heatmap = gui_q.get_nowait()
                    if DEBUG_STATS:
                        with gui_get_ok.get_lock():
                            gui_get_ok.value += 1
                except pyqueue.Empty:
                    break

            if heatmap is not None:
                # heatmap arriva già in dB, shape attesa (tex_h, tex_w)
                H, W = heatmap.shape
                if H != tex_h or W != tex_w:
                    # Se per qualunque motivo cambia shape, evitiamo di ricreare texture a runtime (causa crash).
                    # Aggiorna solo se è compatibile.
                    print(f"[GUI WARN] heatmap shape {H}x{W} != texture {tex_h}x{tex_w} (skip frame)")
                else:
                    # dB -> norm -> RGBA
                    denom = (vis_vmax - vis_vmin)
                    if denom <= 0:
                        denom = 1e-6
                    norm = (heatmap - vis_vmin) / denom

                    # DearPyGui considera la 1a riga della texture come "top".
                    # In matplotlib usavi origin="lower" -> flip verticale per avere range=0 in basso.
                    rgba = _jet_rgba(norm)[::-1, :, :]
                    # copia veloce senza allocazioni
                    tex_np[:] = rgba.reshape(-1)
                    dpg.set_value(TEX_TAG, tex_buf)

                    if DEBUG_STATS:
                        img_updates += 1
            dpg.render_dearpygui_frame()

    except KeyboardInterrupt:
        print("Chiusura...")
    finally:
        _shutdown()
        dpg.destroy_context()

if __name__ == "__main__":
    mp.freeze_support()
    main()
