import socket
import time
import queue as pyqueue
from pathlib import Path
from multiprocessing import Process, Queue, Value
from multiprocessing.sharedctypes import RawArray, Synchronized

import yaml
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft

from Dsp_processing import selection_from_yaml_dict, build_windows
#import dpnp as dp


# --- CONFIGURAZIONE ---
CFG_PATH = Path(__file__).with_name("Config.yaml")  # <-- nome esatto del tuo file
with CFG_PATH.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# --- CONFIGURAZIONE FISICA ---
C = float(cfg["physical"]["c"])
FS = float(cfg["physical"]["fs"])
SLOPE = float(cfg["physical"]["slope"])
FC = float(cfg["physical"]["fc"])

LAMBDA = C / FC
D = LAMBDA / 2

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


FRAME_Q_MAX = 20 # dimensione coda frame tra RX e DSP (in numero di frame, non byte)
GUI_Q_MAX = 5 # dimensione coda tra DSP e GUI (in numero di heatmap, non byte).





# ----------------------------
# FUNZIONI DI ELABORAZIONE
# ----------------------------
def radar_rx(
    frame_queue: Queue,
    free_slots: Queue,
    shm_frames,
    lost_pkts: Synchronized,
    rx_pkts: Synchronized,
    rx_put_drops: Synchronized,
    frame_put_ok: Synchronized,
):
    """
    Riceve UDP DCA1000. 
    MODIFICA: Se viene rilevata una perdita di pacchetti (gap), 
    invece di fare zero-fill parziale, l'INTERO frame viene invalidato (messo a zero)
    prima di essere inviato, per evitare artefatti nel DSP.
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
    packet_mv  = memoryview(packet_buf)
    packet_view = packet_mv.cast("B")
    
    shm_view = memoryview(shm_frames).cast("B")
    curr_slot = None
    frame_view = None  

    # 1. Blocco di zeri grande quanto un intero frame per azzeramento rapido
    ZERO_FRAME_BLOCK = b'\x00' * BYTES_PER_FRAME

    w = 0 
    last_seq = None 
    payload_len_ref = None 
    
    # Flag per indicare se il frame corrente è integro
    frame_ok = True 

    pkts_local = 0
    t_flush = time.perf_counter()

    def ensure_slot():
        nonlocal curr_slot, frame_view, w, frame_ok
        if w != 0 or frame_view is not None:
            return True
        try:
            curr_slot = free_slots.get_nowait()
        except pyqueue.Empty:
            try:
                old = frame_queue.get_nowait()
                free_slots.put(old)
            except pyqueue.Empty:
                pass
            return False
        
        base = curr_slot * BYTES_PER_FRAME
        frame_view = shm_view[base: base + BYTES_PER_FRAME]
        # Appena prendo uno slot nuovo, assumo sia ok (se non siamo in un gap loop)
        frame_ok = True 
        return True

    while True:
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

                # SE C'È UN GAP, IL FRAME CORRENTE È CORROTTO
                frame_ok = False
                
                # Calcoliamo di quanto avanzare 'w' per mantenere l'allineamento
                bytes_missing = gap * payload_len_ref
                
                while bytes_missing > 0: 
                    if not ensure_slot():
                        break
                    
                    # Avanziamo il cursore 'w' fittiziamente senza scrivere nulla (tanto azzereremo tutto)
                    take = min(bytes_missing, BYTES_PER_FRAME - w) 
                    w += take
                    bytes_missing -= take

                    # Se abbiamo completato un frame DURANTE un gap, quel frame è interamente perso.
                    if w == BYTES_PER_FRAME:
                        # 1. Azzera tutto il frame
                        frame_view[:] = ZERO_FRAME_BLOCK
                        
                        # 2. Push
                        slot_to_push = curr_slot
                        try:
                            frame_queue.put_nowait(slot_to_push)
                            if DEBUG_STATS:
                                with frame_put_ok.get_lock(): frame_put_ok.value += 1
                        except pyqueue.Full: 
                            if DEBUG_STATS:
                                with rx_put_drops.get_lock(): rx_put_drops.value += 1
                            free_slots.put(slot_to_push)
                        
                        # Reset per prossimo slot
                        w = 0  
                        curr_slot = None
                        frame_view = None
                        # Nota: il prossimo frame inizia ancora "corrotto" se bytes_missing > 0, 
                        # ma ensure_slot() resetta a True. Quindi lo forziamo a False se siamo ancora nel loop gap.
                        if bytes_missing > 0:
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
            
            # Copiamo i dati. 
            # NOTA: Anche se frame_ok=False, scriviamo i dati per semplicità logica,
            # ma verranno sovrascritti da zeri al momento del push se il flag è False.
            frame_view[w:w + chunk_size] = packet_view[start_src:start_src + chunk_size]
            
            w += chunk_size
            payload_cursor += chunk_size

            # Se abbiamo completato un frame
            if w == BYTES_PER_FRAME: 
                # CONTROLLO INTEGRITÀ: Se c'è stato un gap in questo frame, azzera tutto
                if not frame_ok:
                    frame_view[:] = ZERO_FRAME_BLOCK
                
                slot_to_push = curr_slot
                try:
                    frame_queue.put_nowait(slot_to_push)
                    if DEBUG_STATS:
                        with frame_put_ok.get_lock():
                            frame_put_ok.value += 1

                except pyqueue.Full:
                    if DEBUG_STATS:
                        with rx_put_drops.get_lock():
                            rx_put_drops.value += 1
                    free_slots.put(slot_to_push)
                
                w = 0  
                curr_slot = None
                frame_view = None
                # Il prossimo frame nasce ottimista
                frame_ok = True



# ----------------------------
# WORKER DSP
# ----------------------------
def dsp_worker(frame_queue, free_slots, shm_frames, gui_queue, gui_put_drops, frame_get_ok, gui_put_ok):
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

    while True:
        try:
            # Blocca solo sul primo frame
            slot = frame_queue.get(timeout=0.5)
            if DEBUG_STATS:
                with frame_get_ok.get_lock(): frame_get_ok.value += 1
            batch_slots.append(slot)
        except pyqueue.Empty:
            continue

        # --- Logica svuotamento coda (Lag protection) ---
        qsz = 0
        try: qsz = frame_queue.qsize()
        except NotImplementedError: pass
        
        # Se siamo indietro, svuota tutto e tieni solo gli ultimi
        if qsz >= FRAME_Q_MAX - 2:
            try:
                while True:
                    s_old = frame_queue.get_nowait()
                    if DEBUG_STATS:
                        with frame_get_ok.get_lock(): frame_get_ok.value += 1
                    batch_slots.append(s_old)
            except pyqueue.Empty:
                pass
            
            # Tieni solo gli ultimi X_FRAMES necessari
            if len(batch_slots) > X_FRAMES:
                drop_slots = batch_slots[:-X_FRAMES]
                batch_slots = batch_slots[-X_FRAMES:]
                # Restituisci subito gli slot droppati
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
if __name__ == "__main__":

    # Queue di indici (slot) pronti per DSP
    frame_q = Queue(maxsize=FRAME_Q_MAX)
    #Pool di slot liberi (evita overwrite e copie)
    free_slots = Queue()

    # Shared ring per i frame (byte)
    N_SLOTS = FRAME_Q_MAX + X_FRAMES + 32 # un po' di margine
    shm_frames = RawArray("B", N_SLOTS * BYTES_PER_FRAME)
    for i in range(N_SLOTS):
        free_slots.put(i)
        
    gui_q = Queue(maxsize=GUI_Q_MAX)


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
        args=(frame_q, free_slots, shm_frames, lost_pkts, rx_pkts, rx_put_drops, frame_put_ok),
    )
    p_dsp = Process(target=dsp_worker, args=(frame_q, free_slots, shm_frames, gui_q, gui_put_drops, frame_get_ok, gui_put_ok))

    p_rx.daemon = True
    p_dsp.daemon = True
    p_rx.start()
    p_dsp.start()

    # Avvio grafica

    fig, ax = plt.subplots(figsize=(10, 8))

    # assi
    ang_axis = np.linspace(-90.0, 90.0, NFFT_ANGLE)

    # dummy (dimensione coerente con RANGE_MAX_DISPLAY)
    dr_plot = C * FS / (2.0 * SLOPE * NFFT_RANGE)
    max_bin_plot = int(np.floor(RANGE_MAX_DISPLAY / dr_plot))
    max_bin_plot = max(1, min(max_bin_plot, NFFT_RANGE // 2))
    dummy_data = np.zeros((max_bin_plot, NFFT_ANGLE), dtype=np.float32)

    img = ax.imshow(
        dummy_data,
        extent=[ang_axis[0], ang_axis[-1], 0, RANGE_MAX_DISPLAY],
        aspect="auto",
        cmap="jet",
        vmin=VMIN,
        vmax=VMAX,
        interpolation="nearest",
        origin="lower",
        animated=True,
    )

    plt.colorbar(img, ax=ax, label="Power (dB)")
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Range (m)")
    ax.set_ylim(0, RANGE_MAX_DISPLAY)
    ax.set_title("Real-Time MIMO Radar Heatmap")

    plt.tight_layout()
    plt.show(block=False)

    # blitting setup
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)
    # Loop grafico

    # monitor STATS (1 Hz)
    t_mon = time.perf_counter()
    t_img_start = time.perf_counter()
    img_updates = 0
    lost_prev = 0
    pkts_prev = 0

    try:
        while True:
            # ---------------- STATISTICHE (Ogni 1 sec) ----------------
            if DEBUG_STATS:
                now = time.perf_counter()
                if now - t_mon >= 0.0001:
                    dt_mon = now - t_mon
                    
                    # Calcolo FPS Immagine
                    dt_img = now - t_img_start
                    img_hz = img_updates / dt_img if dt_img > 0 else 0.0
                    
                    # Reset contatori immagine
                    img_updates = 0
                    t_img_start = now

                    # Lettura atomica statistiche code
                    with frame_put_ok.get_lock(): put_f = frame_put_ok.value
                    with frame_get_ok.get_lock(): get_f = frame_get_ok.value
                    backlog_f = put_f - get_f

                    with gui_put_ok.get_lock(): put_g = gui_put_ok.value
                    with gui_get_ok.get_lock(): get_g = gui_get_ok.value
                    backlog_g = put_g - get_g

                    with lost_pkts.get_lock(): 
                        lost_now = lost_pkts.value
                        lost_delta = lost_now - lost_prev
                        lost_prev = lost_now

                    with rx_pkts.get_lock():
                        pkts_now = rx_pkts.value
                        pkts_delta = pkts_now - pkts_prev
                        pkts_prev = pkts_now

                    with rx_put_drops.get_lock(): drop_f = rx_put_drops.value
                    with gui_put_drops.get_lock(): drop_g = gui_put_drops.value

                    sat_f = 100.0 * backlog_f / FRAME_Q_MAX
                    sat_g = 100.0 * backlog_g / GUI_Q_MAX
                    
                    total = pkts_delta + lost_delta
                    loss_pct = (100.0 * lost_delta / total) if total > 0 else 0.0
                    pkt_rate = (pkts_delta / dt_mon) if dt_mon > 0 else 0.0

                    print(f"loss={loss_pct:.3f}% | rx_rate={pkt_rate:.0f} | fps_gui={img_hz:.1f}")
                    print(f"  Q_FRAME: {sat_f:.0f}% (drop {drop_f}) | Q_GUI: {sat_g:.0f}% (drop {drop_g})")
                    
                    t_mon = now

            # ---------------- AGGIORNAMENTO GUI (Sempre) ----------------
            # Nota: Questa parte DEVE essere fuori dall'if dei timer
            
            # Svuota la coda e prendi solo l'ultimo frame disponibile (frame skipping)
            heatmap = None
            while True:
                try:
                    heatmap = gui_q.get_nowait()
                    if DEBUG_STATS:
                        with gui_get_ok.get_lock(): gui_get_ok.value += 1
                except pyqueue.Empty:
                    break
            
            # Se non c'è nulla di nuovo, sleep breve per non fondere la CPU nel loop grafico
            if heatmap is None:
                time.sleep(0.001) # 1ms è sufficiente
                # Esegui comunque flush per eventi GUI (resize, close window)
                fig.canvas.flush_events() 
                continue

            # Blit grafico
            img.set_data(heatmap)
            ax.draw_artist(img)
            fig.canvas.blit(ax.bbox)
            fig.canvas.flush_events()

            if DEBUG_STATS:
                img_updates += 1

    except KeyboardInterrupt:
        print("Chiusura...")
        p_rx.terminate()
        p_dsp.terminate()