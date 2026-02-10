import socket
import struct
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt



# --- CONFIGURAZIONE FISICA (Controlla su mmWave Studio) ---
FS = 10e6            # 10 Msps (10000 ksps)
SLOPE = 60.012e12        # 70 MHz/us -> 70e12 Hz/s
C = 3e8              # Velocità luce
FC = 77e9            # 77 GHz
LAMBDA = C / FC
D = LAMBDA / 2       # Spaziatura antenna (standard)

# --- CONFIGURAZIONE ---
SAMPLES = 256
CHIRPS = 128
RX = 4
TX = 2
VIRTUAL_ANT = TX * RX # 8 antenne virtuali per TDM-MIMO
X_FRAMES = 5  # Elabora e visualizza ogni 5 frame
BUF_SIZE = (X_FRAMES, CHIRPS // TX, TX, RX, SAMPLES)

# Zero Padding
NFFT_RANGE = 512
NFFT_ANGLE = 128

# --- PARAMETRI VISUALIZZAZIONE ---
VMIN = -35  # Soglia minima dB (regola per pulire il rumore)
VMAX = 0 # Soglia massima dB
RANGE_MAX_DISPLAY = 15 # Visualizza solo i primi 15 metri




# Code di comunicazione
raw_q = queue.Queue(maxsize=10)       # Raw UDP -> Parser
pre_proc_q = queue.Queue(maxsize=5)   # Parser -> Pre-Proc (ogni X frame)
proc_q = queue.Queue(maxsize=5)       # Pre-Proc -> Proc
post_proc_q = queue.Queue(maxsize=5)  # Proc -> Post-Proc
plot_q = queue.Queue(maxsize=5)       # Post-Proc -> GUI

def thread_receiver():
    """Ricezione UDP standard dalla DCA1000[cite: 1462, 1972]."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("192.168.33.30", 4098))
    while True:
        packet, _ = sock.recvfrom(2048)
        raw_q.put(packet[10:]) # Salta header DCA1000 [cite: 2049]

def thread_parser():
    """Converte raw in complessi e accumula X frame[cite: 452, 610]."""
    accum_buffer = np.zeros(BUF_SIZE, dtype=complex)
    f_idx, c_idx, s_idx = 0, 0, 0
    while True:
        data = raw_q.get()
        num_sets = len(data) // 16 # 4 lane I/Q = 16 byte [cite: 453]
        for i in range(num_sets):
            v = struct.unpack('<8h', data[i*16:(i+1)*16])
            
            # Calcolo indici
            tx_id = (c_idx % TX)
            loop_id = (c_idx // TX)
            
            for r in range(RX):
                accum_buffer[f_idx, loop_id, tx_id, r, s_idx] = complex(v[r], v[r+4])
            
            s_idx += 1
            if s_idx == SAMPLES:
                s_idx = 0
                c_idx += 1
                if c_idx == CHIRPS:
                    c_idx = 0
                    f_idx += 1
                    # Se abbiamo accumulato X frame, invia alla pipeline
                    if f_idx == X_FRAMES:
                        pre_proc_q.put(accum_buffer.copy())
                        f_idx = 0

def thread_pre_processing():
    """Stadio 1: Rimozione DC e Windowing di Hann."""
    
    # Pre-generiamo la finestra di Hann per la dimensione dei campioni (SAMPLES)
    # La portiamo alla stessa forma dei dati per il broadcasting (1, 1, 1, 1, 256)
    window = np.hanning(SAMPLES).reshape(1, 1, 1, 1, SAMPLES)
    
    while True:
        data = pre_proc_q.get()
        
        # 1. Rimozione DC (Sottrazione della media lungo l'asse dei campioni)
        # Questo elimina il picco di "rumore" a distanza zero
        data_dc_removed = data - np.mean(data, axis=-1, keepdims=True)
        
        # 2. Applicazione Finestra di Hann
        # Riduce lo spectral leakage e i lobi secondari nella Range FFT
        processed = data_dc_removed * window
        
        proc_q.put(processed)



def thread_processing():
    """Stadio 2: FFT (Range/Doppler) con Zero Padding."""
    
    # --- FISSA QUI: Inizializzazione fuori dal loop ---
    heatmap_ema = None 
    # --------------------------------------------------

    while True:
        data = proc_q.get()
    
        # 2. Range FFT
        range_fft = np.fft.fft(data, n=NFFT_RANGE, axis=-1)

        # MTI / clutter remove: togli la media lungo i chirp (asse loop/chirp)
        # range_fft shape: (X_FRAMES, loops, TX, RX, NFFT_RANGE) dopo fft?
        # Nel tuo caso è (X_FRAMES, loops, TX, RX, NFFT_RANGE) perché axis=-1
        range_fft = range_fft - np.mean(range_fft, axis=1, keepdims=True)

        # 3. Costruzione Array Virtuale (MIMO)
        virtual_array = range_fft.transpose(0, 1, 4, 2, 3).reshape(X_FRAMES, CHIRPS//TX, NFFT_RANGE, VIRTUAL_ANT)

        # Window sulle antenne (riduce sidelobes in angolo)
        w_ang = np.hanning(VIRTUAL_ANT).astype(np.float32)
        virtual_array = virtual_array * w_ang.reshape(1, 1, 1, VIRTUAL_ANT)

        # 4. Angle FFT
        angle_fft = np.fft.fft(virtual_array, n=NFFT_ANGLE, axis=-1)
        angle_fft = np.fft.fftshift(angle_fft, axes=-1)

        # 5. Generazione Heatmap (Potenza media)
        heatmap = np.mean(np.abs(angle_fft)**2, axis=(0, 1)).astype(np.float32)

        # 6. Media mobile esponenziale (EMA)
        alpha = 0.2
        if heatmap_ema is None or heatmap_ema.shape != heatmap.shape:
            heatmap_ema = heatmap.copy()
        else:
            # Formula EMA corretta: EMA = (1-alpha)*vecchia + alpha*nuova
            heatmap_ema = (1.0 - alpha) * heatmap_ema + alpha * heatmap

        post_proc_q.put(heatmap_ema)



def thread_post_processing():
    """Stadio 3: Conversione dB e Debug Valori."""
    while True:
        data = post_proc_q.get()
        
        # Conversione in dB
        # Nota: usiamo 10*log10 perché 'data' è già la potenza (quadrato della magnitudo)
        mag_db = 10 * np.log10(data + 1e-12) 
        mag_db = mag_db - np.max(mag_db)   # 0 dB = massimo del frame (o batch)

        plot_q.put(mag_db)

def thread_graphics():
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Assi
    # fb_bin = k * Fs/Nfft,  R = c*fb/(2*SLOPE)
    k = np.arange(NFFT_RANGE)
    range_axis = (C * FS * k) / (2.0 * SLOPE * NFFT_RANGE)
    max_range = range_axis[-1]

    # bin FFT shiftati in [-0.5, 0.5)
    u = np.fft.fftshift(np.fft.fftfreq(NFFT_ANGLE, d=1.0))  # cicli/campione
    # per ULA: u = (d/λ) * sin(theta)  -> sin(theta) = u * (λ/d)
    sin_theta = u * (LAMBDA / D)
    angles_deg = np.degrees(np.arcsin(np.clip(sin_theta, -1.0, 1.0)))

    # Inizializzazione plot con vmin e vmax
    dummy_data = np.ones((NFFT_RANGE, NFFT_ANGLE)) * VMIN
    img = ax.imshow(dummy_data, 
                    extent=(angles_deg[0], angles_deg[-1], 0, max_range),
                    aspect='auto', cmap='jet', origin='lower',
                    vmin=VMIN, vmax=VMAX)
    
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label('Intensità (dB)')
    
    ax.set_xlabel("Angolo (Gradi)")
    ax.set_ylabel("Distanza (Metri)")
    ax.set_ylim(0, RANGE_MAX_DISPLAY) # Limita la vista ai metri interessanti
    ax.set_title("MIMO Radar: Range-Angle Heatmap")

    while True:
        heatmap_db = plot_q.get()
        img.set_data(heatmap_db)
        
        # Opzionale: aggiornamento dinamico vmin/vmax se vuoi
        # img.set_clim(vmin=VMIN, vmax=VMAX) 
        
        plt.pause(0.01)



# Start Threads
threads = [
    threading.Thread(target=thread_receiver, daemon=True),
    threading.Thread(target=thread_parser, daemon=True),
    threading.Thread(target=thread_pre_processing, daemon=True),
    threading.Thread(target=thread_processing, daemon=True),
    threading.Thread(target=thread_post_processing, daemon=True),
    threading.Thread(target=thread_graphics, daemon=True)
]

for t in threads: t.start()
input("Premi INVIO per fermare...\n")