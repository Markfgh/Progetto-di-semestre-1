from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path


HEADER_MAGIC = b"RTPBIN1\x00"
HEADER_PREFIX_LEN = len(HEADER_MAGIC) + 4
HEADER_MAX_LEN = 256 * 1024
CAPTURE_NAME_RE = re.compile(r"^capture_pos(-?\d+)\.bin$")


def _fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    unit = units[0]
    for u in units:
        unit = u
        if value < 1024.0 or u == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.2f} {unit}"


def _to_int(x) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def read_capture_header(path: Path) -> tuple[dict | None, int]:
    file_size = path.stat().st_size
    if file_size < HEADER_PREFIX_LEN:
        return None, 0

    with path.open("rb") as f:
        prefix = f.read(HEADER_PREFIX_LEN)
        if len(prefix) != HEADER_PREFIX_LEN:
            return None, 0
        if prefix[: len(HEADER_MAGIC)] != HEADER_MAGIC:
            return None, 0

        header_len = int(struct.unpack("<I", prefix[len(HEADER_MAGIC) :])[0])
        if header_len <= 0 or header_len > HEADER_MAX_LEN:
            raise ValueError(f"header_len non valido: {header_len}")

        data_offset = HEADER_PREFIX_LEN + header_len
        if data_offset >= file_size:
            raise ValueError(
                f"header invalido: data_offset={data_offset}, file_size={file_size}"
            )

        payload = f.read(header_len)
        if len(payload) != header_len:
            raise ValueError(
                f"header tronco: attesi {header_len} byte, letti {len(payload)}"
            )

    try:
        meta = json.loads(payload.decode("utf-8"))
    except Exception as e:
        raise ValueError("header JSON non valido") from e
    if not isinstance(meta, dict):
        raise ValueError("header JSON deve essere un oggetto")
    return meta, data_offset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ispeziona un file capture .bin (con/senza header RTPBIN1)."
    )
    parser.add_argument("bin_path", help="Path del file .bin da ispezionare")
    parser.add_argument("--samples", type=int, default=None, help="capture.samples (fallback)")
    parser.add_argument("--chirps", type=int, default=None, help="capture.chirps (fallback)")
    parser.add_argument("--rx", type=int, default=None, help="capture.rx (fallback)")
    parser.add_argument(
        "--show-header-json",
        action="store_true",
        help="Stampa il JSON header completo (se presente)",
    )
    args = parser.parse_args(argv)

    path = Path(args.bin_path)
    if not path.is_file():
        print(f"[ERR] File non trovato: {path}")
        return 1

    file_size = int(path.stat().st_size)
    try:
        header, data_offset = read_capture_header(path)
    except Exception as e:
        print(f"[ERR] Header non valido: {e}")
        return 2

    payload_size = file_size - data_offset

    print(f"File: {path}")
    print(f"Dimensione: {file_size} ({_fmt_bytes(file_size)})")
    print(f"Header presente: {'si' if header is not None else 'no'}")
    print(f"Data offset: {data_offset} byte")
    print(f"Payload: {payload_size} byte ({_fmt_bytes(payload_size)})")

    pos_from_name = None
    m = CAPTURE_NAME_RE.match(path.name)
    if m is not None:
        pos_from_name = int(m.group(1))
        print(f"Posizione da nome file: {pos_from_name}")

    capture = {}
    if header is not None:
        pos_from_header = _to_int(header.get("position"))
        created_at = header.get("created_at")
        radar = header.get("radar") if isinstance(header.get("radar"), dict) else {}
        capture = header.get("capture") if isinstance(header.get("capture"), dict) else {}

        if pos_from_header is not None:
            print(f"Posizione da header: {pos_from_header}")
            if pos_from_name is not None and pos_from_name != pos_from_header:
                print("[WARN] Posizione nome file != posizione header")
        if created_at is not None:
            print(f"Data/ora: {created_at}")

        if radar:
            print(
                "Radar: "
                f"c={radar.get('c')} fs={radar.get('fs')} "
                f"slope={radar.get('slope')} fc={radar.get('fc')}"
            )
        if capture:
            print(
                "Capture: "
                f"samples={capture.get('samples')} chirps={capture.get('chirps')} "
                f"rx={capture.get('rx')} tx={capture.get('tx')} x_frames={capture.get('x_frames')}"
            )

        if args.show_header_json:
            print("Header JSON:")
            print(json.dumps(header, indent=2, ensure_ascii=False))

    samples = _to_int(capture.get("samples")) if capture else None
    chirps = _to_int(capture.get("chirps")) if capture else None
    rx = _to_int(capture.get("rx")) if capture else None

    if samples is None:
        samples = args.samples
    if chirps is None:
        chirps = args.chirps
    if rx is None:
        rx = args.rx

    if samples is not None and chirps is not None and rx is not None:
        bytes_per_frame = int(chirps) * int(samples) * int(rx) * 4
        print(f"bytes_per_frame: {bytes_per_frame}")
        if payload_size % bytes_per_frame != 0:
            print(
                "[WARN] Payload non multiplo di bytes_per_frame: "
                f"resto={payload_size % bytes_per_frame}"
            )
        else:
            n_frames = payload_size // bytes_per_frame
            print(f"Frame nel payload: {n_frames}")
    else:
        print(
            "[INFO] Per calcolare i frame serve samples/chirps/rx "
            "(da header o da --samples --chirps --rx)."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
