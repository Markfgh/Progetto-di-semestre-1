"""main_rt.py
Entry-point real-time: avvia RX + DSP e mostra heatmap in real-time (Matplotlib + blit).
"""

from __future__ import annotations

import time
import queue as pyqueue
from pathlib import Path
from multiprocessing import Process, Queue, Value, freeze_support
from multiprocessing.sharedctypes import RawArray

import yaml
import numpy as np
import matplotlib.pyplot as plt

from radar_utils import radar_rx
from dsp_processing import dsp_worker


def load_cfg(cfg_path: Path) -> dict:
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_params(cfg: dict) -> dict:
    # --- fisica ---
    c = float(cfg["physical"]["c"])
    fs = float(cfg["physical"]["fs"])
    slope = float(cfg["physical"]["slope"])
    fc = float(cfg["physical"]["fc"])  # tenuto per completezza

    # --- capture ---
    samples = int(cfg["capture"]["samples"])
    chirps = int(cfg["capture"]["chirps"])
    rx = int(cfg["capture"]["rx"])
    tx = int(cfg["capture"]["tx"])
    x_frames = int(cfg["capture"]["x_frames"])
    virtual_ant = tx * rx
    bytes_per_frame = chirps * samples * rx * 4  # 4 byte per complesso I+Q int16

    # --- fft ---
    nfft_range = int(cfg["fft"]["nfft_range"])
    nfft_angle = int(cfg["fft"]["nfft_angle"])

    # --- display ---
    vmin = float(cfg["display"]["vmin"])
    vmax = float(cfg["display"]["vmax"])
    range_max_display = float(cfg["display"]["range_max"])

    # --- debug / dsp ---
    debug_stats = bool(cfg["debug"]["debug_stats"])
    fft_workers = int(cfg.get("dsp", {}).get("fft_workers", 6))

    params = dict(
        # udp
        pc_ip="192.168.33.30",
        port=4098,
        header_len=10,
        rcvbuf_bytes=256 * 1024 * 1024,

        # fisica
        c=c,
        fs=fs,
        slope=slope,
        fc=fc,

        # capture
        samples=samples,
        chirps=chirps,
        rx=rx,
        tx=tx,
        x_frames=x_frames,
        virtual_ant=virtual_ant,
        bytes_per_frame=bytes_per_frame,

        # fft
        nfft_range=nfft_range,
        nfft_angle=nfft_angle,
        fft_workers=fft_workers,

        # display
        vmin=vmin,
        vmax=vmax,
        range_max_display=range_max_display,

        # dsp misc
        ema_alpha=0.2,

        # debug
        debug_stats=debug_stats,
    )
    return params


def main() -> None:
    cfg_path = Path(__file__).with_name("Config.yaml")
    cfg = load_cfg(cfg_path)
    params = build_params(cfg)

    # code sizes
    FRAME_Q_MAX = 10
    GUI_Q_MAX = 5

    # Queue di indici (slot) pronti per DSP
    frame_q = Queue(maxsize=FRAME_Q_MAX)
    free_slots = Queue()

    # Shared ring per i frame (byte)
    N_SLOTS = FRAME_Q_MAX + int(params["x_frames"]) + 32
    shm_frames = RawArray("B", N_SLOTS * int(params["bytes_per_frame"]))
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

    # processi
    p_rx = Process(
        target=radar_rx,
        args=(frame_q, free_slots, shm_frames, lost_pkts, rx_pkts, rx_put_drops, frame_put_ok, params),
    )
    p_dsp = Process(
        target=dsp_worker,
        args=(frame_q, free_slots, shm_frames, gui_q, gui_put_drops, frame_get_ok, gui_put_ok, cfg, params),
    )

    p_rx.daemon = True
    p_dsp.daemon = True
    p_rx.start()
    p_dsp.start()

    # ---------------- GUI (Matplotlib) ----------------
    fig, ax = plt.subplots(figsize=(10, 8))

    ang_axis = np.linspace(-50.0, 50.0, int(params["nfft_angle"]))

    dr_plot = float(params["c"]) * float(params["fs"]) / (2.0 * float(params["slope"]) * int(params["nfft_range"]))
    max_bin_plot = int(np.floor(float(params["range_max_display"]) / dr_plot))
    max_bin_plot = max(1, min(max_bin_plot, int(params["nfft_range"]) // 2))
    dummy = np.zeros((max_bin_plot, int(params["nfft_angle"])), dtype=np.float32)

    img = ax.imshow(
        dummy,
        extent=[ang_axis[0], ang_axis[-1], 0, float(params["range_max_display"])],
        aspect="auto",
        cmap="jet",
        vmin=float(params["vmin"]),
        vmax=float(params["vmax"]),
        interpolation="nearest",
        origin="lower",
        animated=True,
    )
    plt.colorbar(img, ax=ax, label="Power (dB)")
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Range (m)")
    ax.set_ylim(0, float(params["range_max_display"]))
    ax.set_title("Real-Time MIMO Radar Heatmap")
    plt.tight_layout()
    plt.show(block=False)

    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)

    debug_stats = bool(params.get("debug_stats", False))
    t_mon = time.perf_counter()
    if debug_stats:
        img_updates = 0
        t_img_start = time.perf_counter()
        lost_prev = 0
        pkts_prev = 0

    try:
        while True:
            if debug_stats:
                now = time.perf_counter()
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

                    dt = now - t_img_start
                    img_hz = (img_updates / dt) if dt > 0 else 0.0
                    print(f"[IMG] update_rate = {img_hz:.1f} Hz")

                    img_updates = 0
                    t_img_start = now
                    t_mon = now

            # prendi ultimo heatmap disponibile
            heatmap = None
            while True:
                try:
                    heatmap = gui_q.get_nowait()
                    if debug_stats:
                        with gui_get_ok.get_lock():
                            gui_get_ok.value += 1
                except pyqueue.Empty:
                    break

            if heatmap is None:
                time.sleep(0.002)
                continue

            fig.canvas.restore_region(background)
            img.set_data(heatmap)

            if debug_stats:
                img_updates += 1

            ax.draw_artist(img)
            fig.canvas.blit(ax.bbox)
            fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("Chiusura...")
        p_rx.terminate()
        p_dsp.terminate()


if __name__ == "__main__":
    freeze_support()
    main()
