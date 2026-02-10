import socket
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
import time
import queue as pyqueue

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
VIRTUAL_ANT = TX * RX
X_FRAMES = 1  # Accumula 1 frame prima di processare (attenzione alla latenza!)
BYTES_PER_FRAME = CHIRPS * SAMPLES * RX * 4 # (4 byte per complesso I+Q int16)

# Zero Padding
NFFT_RANGE = 1024
NFFT_ANGLE = 128

# Parametri Grafica
VMIN = -30
VMAX = 0
RANGE_MAX_DISPLAY = 3




def radar_processing_core(frame_queue):
    PC_IP = "192.168.33.30"
    PORT = 4098
    HEADER_LEN = 10
    RCVBUF_BYTES = 256 * 1024 * 1024
    ZERO_CHUNK = b"\x00" * 65536

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RCVBUF_BYTES)
    sock.bind((PC_IP, PORT))
    sock.settimeout(0.2)

    frame = bytearray(BYTES_PER_FRAME)
    w = 0

    last_seq = None
    payload_len_ref = None


    lost_pkts = 0
    total_pkts = 0
    t0 = time.time()

    frames_out = 0
    t_stat = time.perf_counter()

    print("[RX] Avviato (bytes-only, zero-fill).")

    while True:
        try:
            pkt, _ = sock.recvfrom(2048)
        except socket.timeout:
            continue

        if len(pkt) <= HEADER_LEN:
            continue

        seq = int.from_bytes(pkt[0:4], "little", signed=False)
        payload = memoryview(pkt)[HEADER_LEN:]
        plen = len(payload)
        if plen == 0:
            continue

        if payload_len_ref is None:
            payload_len_ref = plen

        # ---- zero fill per pacchetti persi (in bytes) ----
        if last_seq is not None:
            gap = seq - last_seq - 1
            if gap > 0:
                zbytes = gap * payload_len_ref
                lost_pkts += gap
                while zbytes > 0:
                    take = min(zbytes, BYTES_PER_FRAME - w)
                    frame[w:w+take] = ZERO_CHUNK[:take]     # <-- zero-fill reale
                    w += take
                    zbytes -= take
                    if w == BYTES_PER_FRAME:
                        try:
                            frame_queue.put_nowait(bytes(frame))
                            frames_out += 1
                        except pyqueue.Full:
                            pass
                        #frame[:] = ZERO_CHUNK[:BYTES_PER_FRAME]
                        w = 0

        last_seq = seq
    


        # Aggiorna statistiche ogni 10 secondi
        total_pkts += 1
        if total_pkts % 5000 == 0:
            dt = time.time() - t0
            print(f"[RX] pkts={total_pkts} lost={lost_pkts} ({lost_pkts/max(total_pkts,1)*100:.3f}%)")
        
        now = time.perf_counter()
        if now - t_stat >= 1.0:
            print(f"[RX] frames_out={frames_out}  qsize~={getattr(frame_queue, 'qsize', lambda: -1)()}")
            t_stat = now
            frames_out = 0


        # ---- copia payload nel frame ----
        off = 0
        while off < plen:
            take = min(plen - off, BYTES_PER_FRAME - w)
            frame[w:w+take] = payload[off:off+take]
            w += take
            off += take

            if w == BYTES_PER_FRAME:
                try:
                    frame_queue.put_nowait(bytes(frame))
                    frames_out += 1
                except pyqueue.Full:
                    pass
                #frame[:] = ZERO_CHUNK[:BYTES_PER_FRAME]
                w = 0






def dsp_worker(frame_queue, gui_queue):
    window_range = np.blackman(SAMPLES).reshape(1,1,1,1,SAMPLES)
    window_angle = np.hanning(VIRTUAL_ANT).astype(np.float32).reshape(1, 1, 1, VIRTUAL_ANT)

    heatmap_ema = None
    batch = []

    frames_proc = 0
    t_stat = time.perf_counter()

    print("[DSP] Avviato.")

    while True:
        try:
            fb = frame_queue.get(timeout=0.5)
        except pyqueue.Empty:
            continue
        
        # svuota la queue e tieni SOLO l'ultimo frame (low-latency)
        last = fb
        while True:
            try:
                last = frame_queue.get_nowait()
            except pyqueue.Empty:
                break
        fb = last

        batch.append(fb)
        if len(batch) < X_FRAMES:
            continue

        if len(batch) > X_FRAMES:
            batch = batch[-X_FRAMES:]

        # qui converti UNA VOLTA
        raw = b"".join(batch)
        i16 = np.frombuffer(raw, dtype=np.int16)

        n_blocks = i16.size // 8
        if n_blocks == 0:
            batch.clear()
            continue

        i16 = i16[:n_blocks * 8].reshape(n_blocks, 8)
        raw_chunk = i16  # (N, 8)

        re = raw_chunk[:, :4].astype(np.float32, copy=False)
        im = raw_chunk[:, 4:].astype(np.float32, copy=False)
        complex_data = (re + 1j * im).astype(np.complex64, copy=False).reshape(-1)

        # riempi raw_buffer per process_buffer
        total_samples_needed = X_FRAMES * CHIRPS * SAMPLES * RX
        if complex_data.size < total_samples_needed:
            batch.clear()
            continue

        raw_buffer = complex_data[:total_samples_needed]  # view
        heatmap_ema = process_buffer(raw_buffer, window_range, window_angle, heatmap_ema, alpha=0.2, gui_queue=gui_queue)
        frames_proc += X_FRAMES

        now = time.perf_counter()
        if now - t_stat >= 1.0:
            print(f"[DSP] frames_proc={frames_proc}")
            t_stat = now
            frames_proc = 0

        batch.clear()



def process_buffer(raw_buffer, w_range, w_angle, heatmap_ema, alpha, gui_queue):
    """ Funzione helper per elaborare il buffer quando è pieno (sia di dati che di zeri) """
    # calcolo asse range locale (serve anche in subprocess)
    dr = C * FS / (2.0 * SLOPE * NFFT_RANGE)
    range_axis_m = np.arange(NFFT_RANGE//2) * dr   # solo metà spettro
    max_bin = int(np.floor(RANGE_MAX_DISPLAY / dr))
    max_bin = max(1, min(max_bin, NFFT_RANGE // 2))

    
    try:
        # A. Reshape
        # [Frames, Chirps, Samples, RX]
        data = raw_buffer.reshape(X_FRAMES, CHIRPS, SAMPLES, RX)
        
        # B. TDM-MIMO & Transpose
        # [Frames, Chirps, TX, Samples, RX]
        data = data.reshape(X_FRAMES, CHIRPS // TX, TX, SAMPLES, RX)
        # [Frames, Loops, TX, RX, Samples] -> Ready for Range FFT
        data = data.transpose(0, 1, 2, 4, 3) 
        
        # C. DSP
        data = data - np.mean(data, axis=-1, keepdims=True)
        data = data * w_range
        range_fft = np.fft.fft(data, n=NFFT_RANGE, axis=-1)
        
        virtual_array = range_fft.transpose(0, 1, 4, 2, 3).reshape(X_FRAMES, CHIRPS//TX, NFFT_RANGE, VIRTUAL_ANT)
        
        virtual_array = virtual_array * w_angle
        angle_fft = np.fft.fft(virtual_array, n=NFFT_ANGLE, axis=-1)
        angle_fft = np.fft.fftshift(angle_fft, axes=-1)
        
        heatmap = np.mean(np.abs(angle_fft)**2, axis=(0, 1))
        
        # EMA
        if heatmap_ema is None:
            heatmap_ema = heatmap
        else:
            heatmap_ema = (1.0 - alpha) * heatmap_ema + alpha * heatmap
        
        # Output
        heatmap_db = 10 * np.log10(heatmap_ema + 1e-12)
        heatmap_db -= np.max(heatmap_db[:max_bin, :])
        heatmap_db = heatmap_db[:max_bin, :]

        try:
            gui_queue.put_nowait(heatmap_db)
        except pyqueue.Full:
            pass
            
        return heatmap_ema

    except ValueError as ve:
        print(f"[DSP ERR] {ve}")
        return heatmap_ema
    


# --- MAIN (Gestione Grafica) ---
if __name__ == "__main__":
    frame_q = Queue(maxsize=32)   # frame completi (bytes)
    gui_q   = Queue(maxsize=2)   # heatmap per GUI

    p_rx  = Process(target=radar_processing_core, args=(frame_q,))
    p_dsp = Process(target=dsp_worker, args=(frame_q, gui_q))

    p_rx.daemon = True
    p_dsp.daemon = True
    p_rx.start()
    p_dsp.start()

    # --- SETUP GRAFICA (Main Thread) ---
    print("[MAIN] Avvio grafica...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calcolo assi
    range_res = C / (2 * (FS * SAMPLES / SLOPE)) # Risoluzione teorica
    k_axis = np.arange(NFFT_RANGE)
    range_axis_meters = (np.arange(NFFT_RANGE) * C * FS) / (2 * SLOPE * SAMPLES)
    
    # Angoli (approssimazione small angle)
    ang_axis = np.linspace(-90, 90, NFFT_ANGLE)
    
    # Inizializza immagine vuota
    dr_plot = (C * FS) / (2.0 * SLOPE * NFFT_RANGE)
    max_bin_plot = int(np.floor(RANGE_MAX_DISPLAY / dr_plot))
    max_bin_plot = max(1, min(max_bin_plot, NFFT_RANGE // 2))
    dummy_data = np.zeros((max_bin_plot, NFFT_ANGLE), dtype=np.float32)

    
    img = ax.imshow(dummy_data,
                    extent=[ang_axis[0], ang_axis[-1], 0, RANGE_MAX_DISPLAY],
                    aspect='auto',
                    cmap='jet',
                    vmin=VMIN,
                    vmax=VMAX,
                    interpolation="bilinear",
                    origin='lower')

    
    plt.colorbar(img, ax=ax, label='Power (dB)')
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Range (m)")
    ax.set_ylim(0, RANGE_MAX_DISPLAY)
    ax.set_title("Real-Time MIMO Radar Heatmap")
    
    plt.tight_layout()
    plt.show(block=False) # Non bloccante per poter aggiornare il loop
    
    # Loop grafico principale
    try:
        while True:
            # Controlla se ci sono nuovi dati
            try:
                heatmap = gui_q.get_nowait()    
                img.set_data(heatmap)
                img.set_extent([ang_axis[0], ang_axis[-1], 0, RANGE_MAX_DISPLAY])
                img.set_clim(VMIN, VMAX)
            except pyqueue.Empty:
                pass
            # Ridisegna efficientemente
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            
            # Sleep piccolo per non saturare la CPU del main thread
            time.sleep(0.10) 
            
    except KeyboardInterrupt:
        print("Chiusura...")
        p_rx.terminate()
        p_dsp.terminate()