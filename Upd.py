import socket
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue, Value
import time
import queue as pyqueue
import scipy.fft as fft  
#import dpnp as dp

# --- CONFIGURAZIONE FISICA ---
FS = 10e6
SLOPE = 60.012e12
C = 3e8
FC = 77e9
LAMBDA = C / FC
D = LAMBDA / 2

# --- CONFIGURAZIONE ACQUISIZIONE ---
SAMPLES = 256
CHIRPS = 128
RX = 4
TX = 2
VIRTUAL_ANT = TX * RX #antena virtuale totale dopo combinazione TX-RX
X_FRAMES = 1  # Numero di frame da processare insieme (es. per Doppler).
BYTES_PER_FRAME = CHIRPS * SAMPLES * RX * 4 # (4 byte per complesso I+Q int16)

# Zero Padding
NFFT_RANGE = 1024 #>= SAMPLES 
NFFT_ANGLE = 256  #>= SAMPLES 

# PARAMETRI DEBUG / STATS
DEBUG_STATS = False  #(true/false)


# ----------------------------
# PARAMETRI GRAFICA
# ----------------------------
VMIN = -15 # dB, min
VMAX = 0   # dB, max 
RANGE_MAX_DISPLAY = 5.0

# Limiti delle code 
FRAME_Q_MAX = 10 # coda tra RX e DSP (contiene frame interi)
GUI_Q_MAX = 5 # coda tra DSP e GUI 



# ----------------------------
# FUNZIONI DI ELABORAZIONE
# ----------------------------
def radar_rx(frame_queue: Queue,
             lost_pkts: Value,
             rx_put_drops: Value,
             frame_put_ok: Value):
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

    ### VERIFICA BUFFER AUMENTATO se il sistema lo supporta ###
    if DEBUG_STATS:
        actual_buf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        if actual_buf < RCVBUF_BYTES // 2:
            print(f"[RX WARNING] Buffer UDP richiesto: {RCVBUF_BYTES}, ottenuto dal sistema: {actual_buf}")
            print("[RX WARNING] Esegui: sudo sysctl -w net.core.rmem_max=268435456")
        else:
            print(f"[RX] Buffer UDP OK: {actual_buf}")
   

    ####2. Associa socket a IP e porta ####
    sock.bind((PC_IP, PORT))
    #timeout breve per evitare blocchi infiniti su recvfrom
    sock.settimeout(0.2) 


    ####3. Buffer UDP e Frame ####
    packet_buf = bytearray(2048) #buffer udp - size framentato (tipicamente < MTU)
    packet_view = memoryview(packet_buf) # Crea una vista sul buffer senza copiarlo
    
    frame = bytearray(BYTES_PER_FRAME)   # buffer per un frame completo (chirps * samples * rx * 4 byte)
    frame_view = memoryview(frame)       # View per assegnazione veloce
    
    ####4. Chunk di zeri per il zero-fill  ####
    ZERO_CHUNK = bytearray(64 * 1024)  # 64KB di zeri per zero-fill


    w = 0 # write cursor all'interno del frame corrente (0..BYTES_PER_FRAME)
    last_seq = None # ultima sequenza ricevuta (per rilevare gap)
    payload_len_ref = None # lunghezza payload di riferimento (stima dal primo pacchetto valido)

    print("[RX] Avviato.")

    ####5. Ricezione pacchetti UDP ####
    while True:
        try:
            n_bytes, _ = sock.recvfrom_into(packet_view) # riceve direttamente nel buffer pre-allocato
        except socket.timeout:
            continue

        # Se il pacchetto è troppo corto per contenere header + payload, scarta
        if n_bytes <= HEADER_LEN:
            continue

        # seq (little endian)
        seq = int.from_bytes(packet_view[0:4].tobytes(), "little", signed=False)

        if payload_len_ref is None:
            payload_len_ref = n_bytes - HEADER_LEN

        # Se arriva duplicato/out-of-order: scarta SENZA abbassare last_seq
        if last_seq is not None and seq <= last_seq:
            continue
        # ---- Gestione pacchetti persi (Zero Fill) ----
        if last_seq is not None:
            gap = seq - last_seq - 1
            if gap > 0:
                with lost_pkts.get_lock():
                    lost_pkts.value += gap

                bytes_missing = gap * payload_len_ref
                while bytes_missing > 0:
                    take = min(bytes_missing, BYTES_PER_FRAME - w)

                    remaining = take
                    while remaining > 0:
                        t = min(remaining, len(ZERO_CHUNK))
                        frame_view[w:w + t] = ZERO_CHUNK[:t]
                        w += t
                        remaining -= t

                        if w == BYTES_PER_FRAME:
                            try:
                                frame_queue.put_nowait(bytes(frame))
                                with frame_put_ok.get_lock():
                                    frame_put_ok.value += 1
                            except pyqueue.Full:
                                with rx_put_drops.get_lock():
                                    rx_put_drops.value += 1
                            w = 0

                    bytes_missing -= take

        last_seq = seq

        # ---- Copia Payload nel Frame ----
        off = HEADER_LEN
        current_payload_len = n_bytes - off
        
        payload_cursor = 0
        while payload_cursor < current_payload_len:
            chunk_size = min(current_payload_len - payload_cursor, BYTES_PER_FRAME - w)
            
            # Copia memoryview -> memoryview (Estremamente veloce in C)
            start_src = off + payload_cursor
            frame_view[w:w + chunk_size] = packet_view[start_src:start_src + chunk_size]

            w += chunk_size
            payload_cursor += chunk_size

            if w == BYTES_PER_FRAME:
                try:
                    frame_queue.put_nowait(bytes(frame))
                    with frame_put_ok.get_lock():
                        frame_put_ok.value += 1
                except pyqueue.Full:
                    with rx_put_drops.get_lock():
                        rx_put_drops.value += 1
                w = 0



def dsp_worker(frame_queue, gui_queue, gui_put_drops, frame_get_ok, gui_put_ok):
    window_range = np.blackman(SAMPLES).reshape(1, 1, 1, 1, SAMPLES)
    window_angle = np.hanning(VIRTUAL_ANT).astype(np.float32).reshape(1, 1, 1, VIRTUAL_ANT)

    heatmap_ema = None
    batch = []

    print("[DSP] Avviato.")

    while True:
        try:
            fb = frame_queue.get(timeout=0.5)
            with frame_get_ok.get_lock():
                frame_get_ok.value += 1
        except pyqueue.Empty:
            continue

        # ### FIX 1: LOGICA DI SVUOTAMENTO CONDIZIONALE ###
        # Se X_FRAMES > 1, NON possiamo buttare via i frame vecchi, 
        # perché servono tutti in sequenza per l'elaborazione (es. Doppler).
        # Svuotiamo la coda solo se stiamo lavorando frame-by-frame (X_FRAMES=1).
        if X_FRAMES == 1:
            last = fb
            while True:
                try:
                    last = frame_queue.get_nowait()
                    with frame_get_ok.get_lock():
                        frame_get_ok.value += 1
                except pyqueue.Empty:
                    break
            fb = last
        # ### FINE FIX 1 ###

        batch.append(fb)
        
        # Se non abbiamo ancora abbastanza frame, continua ad accumulare
        if len(batch) < X_FRAMES:
            continue
        
        # Se ne abbiamo troppi (caso raro/safety), teniamo gli ultimi X
        if len(batch) > X_FRAMES:
            batch = batch[-X_FRAMES:]

        raw = b"".join(batch)
        i16 = np.frombuffer(raw, dtype=np.int16)

        # ... (Il resto della funzione rimane identico fino alla fine) ...
        n_blocks = i16.size // 8
        if n_blocks == 0:
            batch.clear()
            continue

        i16 = i16[:n_blocks * 8].reshape(n_blocks, 8)

        re = i16[:, :4].astype(np.float32, copy=False)
        im = i16[:, 4:].astype(np.float32, copy=False)
        complex_data = (re + 1j * im).astype(np.complex64, copy=False).reshape(-1)

        total_samples_needed = X_FRAMES * CHIRPS * SAMPLES * RX
        if complex_data.size < total_samples_needed:
            batch.clear()
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

        batch.clear()


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
                         .transpose(0, 1, 2, 4, 3).copy()

        # B. DSP IN-PLACE (Risparmia allocazioni)
        # Sottrai media
        data -= np.mean(data, axis=-1, keepdims=True)
        # Finestra Range
        data *= w_range
        
        # C. RANGE FFT (Ottimizzata)
        # overwrite_x=True: distrugge 'data' per calcolare la FFT più velocemente
        # workers=-1 usa tutti i core disponibili automaticamente
        range_fft = fft.fft(data, n=NFFT_RANGE, axis=-1, workers=-1, overwrite_x=True)
        
        # Preparazione Virtual Array
        # Nota: qui creiamo una nuova view/copia per il transpose necessario
        virtual_array = range_fft.transpose(0, 1, 4, 2, 3).reshape(X_FRAMES, CHIRPS//TX, NFFT_RANGE, VIRTUAL_ANT)
        
        # Finestra Angolo In-Place
        # Assicurati che w_angle sia broadcastabile correttamente o usa .copy() se serve, 
        # ma qui dovrebbe andare bene.
        virtual_array *= w_angle
        
        # D. ANGLE FFT (Ottimizzata)
        angle_fft = fft.fft(virtual_array, n=NFFT_ANGLE, axis=-1, workers=-1, overwrite_x=True)
        angle_fft = fft.fftshift(angle_fft, axes=-1)
        
        # Modulo quadro e media
        heatmap = np.mean(np.abs(angle_fft)**2, axis=(0, 1))
        
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
            with gui_put_ok.get_lock():
                gui_put_ok.value += 1
        except pyqueue.Full:
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
    frame_q = Queue(maxsize=FRAME_Q_MAX)
    gui_q = Queue(maxsize=GUI_Q_MAX)

    # stats condivise
    lost_pkts = Value("L", 0)
    rx_put_drops = Value("L", 0)
    gui_put_drops = Value("L", 0)

    frame_put_ok = Value("L", 0)
    frame_get_ok = Value("L", 0)

    gui_put_ok = Value("L", 0)
    gui_get_ok = Value("L", 0)

    p_rx = Process(target=radar_rx, args=(frame_q, lost_pkts, rx_put_drops, frame_put_ok))
    p_dsp = Process(target=dsp_worker, args=(frame_q, gui_q, gui_put_drops, frame_get_ok, gui_put_ok))

    p_rx.daemon = True
    p_dsp.daemon = True
    p_rx.start()
    p_dsp.start()

    print("[MAIN] Avvio grafica...")

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
    print("[MAIN] Loop grafico.")

    # monitor STATS (1 Hz)
    if DEBUG_STATS:
        t_mon = time.perf_counter()
        # monitor UPDATE IMMAGINE (1 Hz) -> QUELLO CHE TI INTERESSA
        img_updates = 0
        t_img_start = time.perf_counter()

    try:
        while True:

            if DEBUG_STATS:
                now = time.perf_counter()
                # ----- stampa stats 1 Hz -----
                if now - t_mon >= 1.0:
               
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
                        lost = lost_pkts.value
                    with rx_put_drops.get_lock():
                        drop_f = rx_put_drops.value
                    with gui_put_drops.get_lock():
                        drop_g = gui_put_drops.value

                    sat_f = 100.0 * backlog_f / FRAME_Q_MAX
                    sat_g = 100.0 * backlog_g / GUI_Q_MAX

                    print(
                        f"[STATS] lost_pkts={lost} | "
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


            try:
                # prendi l'ultimo disponibile (svuota coda)
                heatmap = gui_q.get_nowait()
                with gui_get_ok.get_lock():
                    gui_get_ok.value += 1

                while not gui_q.empty():
                    heatmap = gui_q.get_nowait()
                    with gui_get_ok.get_lock():
                        gui_get_ok.value += 1

                # --- BLIT veloce ---
                fig.canvas.restore_region(background)
                img.set_data(heatmap)

                if DEBUG_STATS:
                    img_updates += 1  

                ax.draw_artist(img)
                fig.canvas.blit(ax.bbox)
                fig.canvas.flush_events()

            except pyqueue.Empty:
                fig.canvas.blit(ax.bbox)
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("Chiusura...")
        p_rx.terminate()
        p_dsp.terminate()
