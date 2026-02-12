"""radar_utils.py
Common logic: lettura UDP dalla DCA1000 e ricostruzione frame con zero-fill.
"""

from __future__ import annotations

import socket
import time
import queue as pyqueue
from multiprocessing import Queue
from multiprocessing.sharedctypes import Synchronized
from typing import Dict, Any


def radar_rx(
    frame_queue: Queue,
    free_slots: Queue,
    shm_frames,
    lost_pkts: Synchronized,
    rx_pkts: Synchronized,
    rx_put_drops: Synchronized,
    frame_put_ok: Synchronized,
    params: Dict[str, Any],
) -> None:
    """
    Riceve UDP DCA1000 (porta data), ricostruisce frame fissi (bytes_per_frame).
    Se mancano pacchetti (gap seq), fa zero-fill dei byte mancanti.

    Parametri attesi in `params`:
      - pc_ip (str), port (int)
      - header_len (int)   default 10
      - rcvbuf_bytes (int) default 256MB
      - bytes_per_frame (int)
      - debug_stats (bool)
    """
    pc_ip = str(params.get("pc_ip", "192.168.33.30"))
    port = int(params.get("port", 4098))
    header_len = int(params.get("header_len", 10))
    rcvbuf_bytes = int(params.get("rcvbuf_bytes", 256 * 1024 * 1024))
    bytes_per_frame = int(params["bytes_per_frame"])
    debug_stats = bool(params.get("debug_stats", False))

    # 1) socket + buffer grande
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf_bytes)
    sock.bind((pc_ip, port))
    sock.settimeout(0.2)

    # 2) buffer pacchetto
    packet_buf = bytearray(2048)
    packet_mv = memoryview(packet_buf)          # per recvfrom_into (NO cast)
    packet_view = packet_mv.cast("B")           # vista bytes per slicing/copy

    # 3) vista su shared-memory ring (bytes)
    shm_view = memoryview(shm_frames).cast("B")
    curr_slot = None
    frame_view = None  # view sullo slot corrente (bytes_per_frame)

    # 4) chunk di zeri per zero-fill
    ZERO_CHUNK = b"\x00" * 2048
    ZERO_VIEW = memoryview(ZERO_CHUNK).cast("B")

    w = 0
    last_seq = None
    payload_len_ref = None

    # contatore rx locale -> flush su Value condiviso
    pkts_local = 0
    t_flush = time.perf_counter()

    def ensure_slot() -> bool:
        """Assicura che ci sia uno slot pronto quando w==0."""
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

        base = curr_slot * bytes_per_frame
        frame_view = shm_view[base: base + bytes_per_frame]
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

        if n_bytes <= header_len:
            continue

        seq = int.from_bytes(packet_view[0:4], "little", signed=False)

        if payload_len_ref is None:
            payload_len_ref = n_bytes - header_len

        # dup / out-of-order
        if last_seq is not None and seq <= last_seq:
            continue

        # --- zero-fill se gap ---
        if last_seq is not None:
            gap = seq - last_seq - 1
            if gap > 0:
                with lost_pkts.get_lock():
                    lost_pkts.value += gap

                bytes_missing = gap * payload_len_ref
                while bytes_missing > 0:
                    if not ensure_slot():
                        break

                    take = min(bytes_missing, bytes_per_frame - w)
                    remaining = take

                    while remaining > 0:
                        if not ensure_slot():
                            remaining = 0
                            break
                        t = min(remaining, len(ZERO_CHUNK))
                        frame_view[w:w + t] = ZERO_VIEW[:t]
                        w += t
                        remaining -= t

                        if w == bytes_per_frame:
                            slot_to_push = curr_slot
                            try:
                                frame_queue.put_nowait(slot_to_push)
                                if debug_stats:
                                    with frame_put_ok.get_lock():
                                        frame_put_ok.value += 1
                            except pyqueue.Full:
                                if debug_stats:
                                    with rx_put_drops.get_lock():
                                        rx_put_drops.value += 1
                                free_slots.put(slot_to_push)

                            w = 0
                            curr_slot = None
                            frame_view = None

                    bytes_missing -= take

        last_seq = seq

        # --- copia payload nel frame ---
        off = header_len
        current_payload_len = n_bytes - off
        payload_cursor = 0

        while payload_cursor < current_payload_len:
            if not ensure_slot():
                break

            chunk_size = min(current_payload_len - payload_cursor, bytes_per_frame - w)
            start_src = off + payload_cursor
            frame_view[w:w + chunk_size] = packet_view[start_src:start_src + chunk_size]
            w += chunk_size
            payload_cursor += chunk_size

            if w == bytes_per_frame:
                slot_to_push = curr_slot
                try:
                    frame_queue.put_nowait(slot_to_push)
                    if debug_stats:
                        with frame_put_ok.get_lock():
                            frame_put_ok.value += 1
                except pyqueue.Full:
                    if debug_stats:
                        with rx_put_drops.get_lock():
                            rx_put_drops.value += 1
                    free_slots.put(slot_to_push)

                w = 0
                curr_slot = None
                frame_view = None
