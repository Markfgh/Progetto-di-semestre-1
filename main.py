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

from dsp_processing import selection_from_yaml_dict, build_windows
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


FRAME_Q_MAX = 10 # dimensione coda frame tra RX e DSP (in numero di frame, non byte)
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
    Riceve UDP DCA1000 (porta data), ricostruisce frame fissi (BYTES_PER_FRAME).
    Se mancano pacchetti (gap seq), fa zero-fill dei byte mancanti.
    """
    PC_IP = "192.168.33.30"
    PORT = 4098
    HEADER_LEN = 10
    RCVBUF_BYTES = 256 * 1024 * 1024 # 256MB per evitare perdite a livello di socket 
    
    #### 1: BUFFER UDP AUMENTATO ###
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RCVBUF_BYTES)

    # Nota: niente print qui (lo stato lo mostra il monitor nel MAIN)
    ####2. Associa socket a IP e porta ####
    sock.bind((PC_IP, PORT))
    #timeout breve per evitare blocchi infiniti su recvfrom
    sock.settimeout(0.2) 


    ####3. Buffer UDP ####
    packet_buf = bytearray(2048)
    packet_mv  = memoryview(packet_buf)            # buffer per recvfrom_into (NO cast)
    packet_view = packet_mv.cast("B")              # vista bytes per slicing/copy
    
    # Shared-memory ring: buffer globale (n_slots * BYTES_PER_FRAME)
    shm_view = memoryview(shm_frames).cast("B")   # 1D bytes
    curr_slot = None
    frame_view = None  # view sullo slot corrente (BYTES_PER_FRAME)

    ####4. Chunk di zeri per il zero-fill  ####
    ZERO_CHUNK = b"\x00" * 2048  
    ZERO_VIEW = memoryview(ZERO_CHUNK).cast("B")

    w = 0 # write cursor all'interno del frame corrente (0..BYTES_PER_FRAME)
    last_seq = None # ultima sequenza ricevuta (per rilevare gap)
    payload_len_ref = None # lunghezza payload di riferimento (stima dal primo pacchetto valido)

    # Contatore pacchetti RX locale (flush periodico su Value condiviso per minimizzare lock)
    pkts_local = 0
    t_flush = time.perf_counter()
    # stats RX condivise (conteggi monotoni, aggiornati in modo lock-free)


    def ensure_slot():
        """Assicura che curr_slot/frame_view siano validi. Se non ci sono slot, droppa il più vecchio e ritorna False."""
        nonlocal curr_slot, frame_view, w
        if w != 0 or frame_view is not None:
            return True
        try:
            curr_slot = free_slots.get_nowait()
        except pyqueue.Empty:
            # nessuno slot libero: prova a liberarne uno droppando il più vecchio
            try:
                old = frame_queue.get_nowait()
                free_slots.put(old)
            except pyqueue.Empty:
                pass
            return False
        base = curr_slot * BYTES_PER_FRAME
        frame_view = shm_view[base: base + BYTES_PER_FRAME]
        return True


    ####5. Ricezione pacchetti UDP e controllo sequenza ####
    while True:
        try:
            # riceve direttamente nel buffer pre-allocato e ottiene il numero di byte ricevuti
            n_bytes, _ = sock.recvfrom_into(packet_mv)
            pkts_local += 1

            # flush periodico sul contatore condiviso (riduce overhead lock)
            now = time.perf_counter()
            if now - t_flush >= 0.1:
                with rx_pkts.get_lock():
                    rx_pkts.value += pkts_local
                pkts_local = 0
                t_flush = now
        except socket.timeout:
            continue


        #### LOGICA DI ASSEGNAZIONE SLOT E ZERO-FILL ###
        # prima di usare frame_view in questo pacchetto
        if not ensure_slot():
            continue


        # Se il pacchetto è troppo corto per contenere header + payload, scarta
        if n_bytes <= HEADER_LEN:
            continue

        # Estrae numero di sequenza (4 byte, little-endian, unsigned) 
        seq = int.from_bytes(packet_view[0:4], "little", signed=False)

        # Stima lunghezza payload per calcolare gap in byte in caso di pacchetti persi
        if payload_len_ref is None:
            payload_len_ref = n_bytes - HEADER_LEN

        # Se arriva duplicato/out-of-order
        if last_seq is not None and seq <= last_seq:
            continue

    ####6.Gestione pacchetti persi (Zero Fill) ####
        if last_seq is not None: # non è il primo pacchetto
            gap = seq - last_seq - 1 #n. pacchetti persi 
            if gap > 0: 

                # aggiorna contatore pacchetti persi (monotono)
                with lost_pkts.get_lock():
                    lost_pkts.value += gap

                # Calcola quanti byte mancano per colmare il gap e scrivi zeri nel frame
                # Nota: se il gap è molto grande, potremmo dover scrivere più di BYTES_PER_FRAME 
                # di zeri, quindi usiamo un loop per gestire questo caso.
                bytes_missing = gap * payload_len_ref
                while bytes_missing > 0: 
                    if not ensure_slot():
                        break
                    take = min(bytes_missing, BYTES_PER_FRAME - w) # n.byte da scrivere in questo frame
                    remaining = take # byte rimanenti da scrivere in questo frame
                    
                    # Scrive zeri nel frame usando chunk pre-allocati per efficienza
                    while remaining > 0:
                        if not ensure_slot():
                            remaining = 0
                            break
                        t = min(remaining, len(ZERO_CHUNK))
                        frame_view[w:w + t] = ZERO_VIEW[:t]
                        w += t
                        remaining -= t

                        # Se abbiamo completato un frame, pushiamolo nella coda e resettiamo il cursore
                        # se piena droppo il frame 
                        if w == BYTES_PER_FRAME:
                            slot_to_push = curr_slot
                            try:
                                frame_queue.put_nowait(slot_to_push) # manda solo l'indice (Shared-memory ring)
                                if DEBUG_STATS:
                                    with frame_put_ok.get_lock():
                                        frame_put_ok.value += 1

                            except pyqueue.Full: 
                                if DEBUG_STATS:
                                    with rx_put_drops.get_lock():
                                        rx_put_drops.value += 1
                                # slot non consegnato -> restituiscilo
                                free_slots.put(slot_to_push)
                            # prepara slot nuovo
                            w = 0  
                            curr_slot = None
                            frame_view = None
          
                    bytes_missing -= take # aggiorna byte mancanti per il gap

        last_seq = seq # aggiorna ultima sequenza ricevuta


        ####6. Copia Payload nel Frame ####
        off = HEADER_LEN # offset iniziale del payload all'interno del pacchetto
        current_payload_len = n_bytes - off # lunghezza effettiva del payload in questo pacchetto 
        payload_cursor = 0 # cursore per scorrere il payload del pacchetto


        while payload_cursor < current_payload_len:
            if not ensure_slot():
                break

            # Calcola quanti byte possiamo copiare in questo frame (fino a completare BYTES_PER_FRAME)
            chunk_size = min(current_payload_len - payload_cursor, BYTES_PER_FRAME - w)
            # Copia chunk di dati dal pacchetto al frame usando memoryview per evitare copie intermedie
            start_src = off + payload_cursor
            # La seguente assegnazione scrive direttamente nel buffer del frame senza creare copie intermedie
            frame_view[w:w + chunk_size] = packet_view[start_src:start_src + chunk_size]
            # Aggiorna cursori
            w += chunk_size
            payload_cursor += chunk_size

            # Se abbiamo completato un frame, pushiamolo nella coda e resettiamo il cursore
            if w == BYTES_PER_FRAME: 
                slot_to_push = curr_slot
                try:
                    frame_queue.put_nowait(slot_to_push) # manda solo l'indice (Shared-memory ring)
                    if DEBUG_STATS:
                        with frame_put_ok.get_lock():
                            frame_put_ok.value += 1

                except pyqueue.Full:
                    if DEBUG_STATS:
                        with rx_put_drops.get_lock():
                            rx_put_drops.value += 1
                    # slot non consegnato -> restituiscilo
                    free_slots.put(slot_to_push)
                # prepara slot nuovo
                w = 0  
                curr_slot = None
                frame_view = None






# ----------------------------
# WORKER DSP
# ----------------------------
def dsp_worker(frame_queue,free_slots,shm_frames,gui_queue,gui_put_drops,frame_get_ok,gui_put_ok):   
    
    # Costruisci finestre DSP in base alla selezione (con reshape per broadcasting)
    selection = selection_from_yaml_dict(cfg)
    window_range, window_angle = build_windows(selection,samples=SAMPLES,virtual_ant=VIRTUAL_ANT,)

    heatmap_ema = None
    batch_slots = []
    shm_view = memoryview(shm_frames).cast("B")   
 
    # buffer locale contiguo per X_FRAMES (evita join di bytes)
    i16_per_frame = BYTES_PER_FRAME // 2
    i16_batch = np.empty((X_FRAMES, i16_per_frame), dtype=np.int16)

    # DSP worker avviato
    while True:
        try:
            slot = frame_queue.get(timeout=0.5)
            if DEBUG_STATS:  
                with frame_get_ok.get_lock():
                    frame_get_ok.value += 1
        except pyqueue.Empty:
            continue

        # ### FIX 1: LOGICA DI SVUOTAMENTO CONDIZIONALE ###
        # Se X_FRAMES > 1, NON possiamo buttare via i frame vecchi, 
        # perché servono tutti in sequenza per l'elaborazione (es. Doppler).
        # Svuotiamo la coda solo se stiamo lavorando frame-by-frame (X_FRAMES=1).
        if X_FRAMES == 1:
            last = slot
            while True:
                try:
                    old = frame_queue.get_nowait()
                    if DEBUG_STATS:
                        with frame_get_ok.get_lock():
                            frame_get_ok.value += 1
                    
                    # restituisci subito gli slot scartati
                    free_slots.put(old)
                    last = old
                except pyqueue.Empty:
                    break
            slot = last
        # ### FINE FIX 1 ###

        batch_slots.append(slot)
        
        # Se non abbiamo ancora abbastanza frame, continua ad accumulare
        if len(batch_slots) < X_FRAMES:
            continue
        
        # Se ne abbiamo troppi (caso raro/safety), teniamo gli ultimi X
        if len(batch_slots) > X_FRAMES:
            # se teniamo solo gli ultimi, restituiamo quelli scartati
            to_free = batch_slots[:-X_FRAMES]
            for s in to_free:
                free_slots.put(s)
            batch_slots = batch_slots[-X_FRAMES:]

        # Copia una sola volta dal ring condiviso al buffer numpy locale contiguo
        for k, s in enumerate(batch_slots):
            base = s * BYTES_PER_FRAME
            frame_mv = shm_view[base: base + BYTES_PER_FRAME]
            i16_batch[k, :] = np.frombuffer(frame_mv, dtype=np.int16, count=i16_per_frame)
 
        i16 = i16_batch.reshape(-1)  # contiguo

       
        n_blocks = i16.size // 8
        if n_blocks == 0:
            for s in batch_slots:
                free_slots.put(s)
            batch_slots.clear()
            continue

        i16 = i16[:n_blocks * 8].reshape(n_blocks, 8)

        # Conversione più efficiente: una sola allocazione complex64 e riempimento real/imag
        re_i16 = i16[:, :4]
        im_i16 = i16[:, 4:]
        complex_data = np.empty(re_i16.size, dtype=np.complex64)
        complex_data.real = re_i16.reshape(-1)
        complex_data.imag = im_i16.reshape(-1)
      
        total_samples_needed = X_FRAMES * CHIRPS * SAMPLES * RX
        if complex_data.size < total_samples_needed:
            for s in batch_slots:
                free_slots.put(s)
            batch_slots.clear()
            continue

        raw_buffer = complex_data[:total_samples_needed]
        heatmap_ema = process_buffer(
            raw_buffer,
            window_range,
            window_angle,
            heatmap_ema,
            alpha=0.2,
            gui_queue=gui_queue,
            gui_put_drops=gui_put_drops,
            gui_put_ok=gui_put_ok,
        )

        # restituisci slot al RX
        for s in batch_slots:
            free_slots.put(s)
        batch_slots.clear()


def process_buffer(raw_buffer, w_range, w_angle, heatmap_ema, alpha, gui_queue, gui_put_drops, gui_put_ok):
    # Setup parametri scalari
    dr = C * FS / (2.0 * SLOPE * NFFT_RANGE)
    max_bin = int(np.floor(RANGE_MAX_DISPLAY / dr))
    max_bin = max(1, min(max_bin, NFFT_RANGE // 2))

    try:
        # A. Reshape
        # Creiamo una copia esplicita qui per garantire che i dati siano 
        # "C-Contiguous" in memoria. Questo è fondamentale per la velocità delle FFT successive.
        data = raw_buffer.reshape(X_FRAMES, CHIRPS // TX, TX, SAMPLES, RX) \
                        .transpose(0, 1, 2, 4, 3)

        # Se non contiguo, rendilo contiguo (ma solo se serve)
        if not data.flags["C_CONTIGUOUS"]:
            data = np.ascontiguousarray(data)

        # B. DSP IN-PLACE (Risparmia allocazioni)
        # Sottrai media
        data -= data.mean(axis=-1, keepdims=True, dtype=np.complex64)
        # Finestra Range
        data *= w_range
        
        # C. RANGE FFT (Ottimizzata)
        # overwrite_x=True: distrugge 'data' per calcolare la FFT più velocemente
        # workers=-1 usa tutti i core disponibili automaticamente
        range_fft = fft.fft(data, n=NFFT_RANGE, axis=-1, workers=FFT_WORKERS, overwrite_x=True)
        
        # Preparazione Virtual Array
        # Nota: qui creiamo una nuova view/copia per il transpose necessario
        virtual_array = range_fft.transpose(0, 1, 4, 2, 3).reshape(X_FRAMES, CHIRPS//TX, NFFT_RANGE, VIRTUAL_ANT)
        
        # Finestra Angolo In-Place
        # Assicurati che w_angle sia broadcastabile correttamente o usa .copy() se serve, 
        # ma qui dovrebbe andare bene.
        virtual_array *= w_angle
        
        # D. ANGLE FFT (Ottimizzata)
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
    if DEBUG_STATS:
        # monitor UPDATE IMMAGINE (1 Hz) -> QUELLO CHE TI INTERESSA
        img_updates = 0
        t_img_start = time.perf_counter()
        lost_prev = 0
        pkts_prev = 0


    try:
        while True:

            if DEBUG_STATS:
                now = time.perf_counter()
                # ----- stampa stats 1 Hz -----
                if now - t_mon >= 1.0:
                    dt_mon = now - t_mon
               
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

                    print(f"loss={loss_pct:.3f}% rate={pkt_rate:.0f} pkt/s")
                    print(
                        f"frame_q: backlog={backlog_f}/{FRAME_Q_MAX} ({sat_f:.0f}%) drops={drop_f} | "
                        f"gui_q: backlog={backlog_g}/{GUI_Q_MAX} ({sat_g:.0f}%) drops={drop_g}"
                    )



                    # ---- stampa frequenza update immagine (reale) ----
                    dt = now - t_img_start
                    img_hz = (img_updates / dt) if dt > 0 else 0.0
                    print(f"[IMG] update_rate = {img_hz:.1f} Hz")

                    # reset finestra 1 Hz
                    img_updates = 0
                    t_img_start = now
                    t_mon = now

               



            

# ---- prendi ultimo heatmap disponibile (senza empty()) ----
            heatmap = None
            while True:
                try:
                    heatmap = gui_q.get_nowait()
                    if DEBUG_STATS:
                        with gui_get_ok.get_lock():
                            gui_get_ok.value += 1
                except pyqueue.Empty:
                    break

            if heatmap is None:
                time.sleep(0.002)
                continue

            # ---- BLIT ----
            fig.canvas.restore_region(background)
            img.set_data(heatmap)

            if DEBUG_STATS:
                img_updates += 1

            ax.draw_artist(img)
            fig.canvas.blit(ax.bbox)
            fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("Chiusura...")
        p_rx.terminate()
        p_dsp.terminate()