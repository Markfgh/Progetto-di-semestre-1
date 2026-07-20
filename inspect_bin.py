from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path


# Parametri modificabili ------------------------------------------------------
# File da ispezionare se non viene passato come primo argomento dalla riga di comando.
BIN_PATH: str | None = None
# Stampa l'intero header JSON; lascia False per un riepilogo compatto.
SHOW_HEADER_JSON = True

HEADER_MAGIC = b"RTPBIN1\x00"
HEADER_PREFIX_LEN = len(HEADER_MAGIC) + 4
HEADER_MAX_LEN = 256 * 1024
CAPTURE_NAME_RE = re.compile(r"^capture_pos(-?\d+)\.bin$")
SUPPORTED_FORMATS = {"rt_capture_v1", "rt_capture_v2"}


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


def read_capture_header(path: Path) -> tuple[dict, int]:
    file_size = path.stat().st_size
    if file_size < HEADER_PREFIX_LEN:
        raise ValueError(f"file troppo piccolo per header {HEADER_MAGIC!r}")

    with path.open("rb") as f:
        prefix = f.read(HEADER_PREFIX_LEN)
        if len(prefix) != HEADER_PREFIX_LEN:
            raise ValueError("prefisso header incompleto")
        if prefix[: len(HEADER_MAGIC)] != HEADER_MAGIC:
            raise ValueError(f"magic header mancante o non valida (attesa {HEADER_MAGIC!r})")

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
    format_name = meta.get("format")
    if format_name not in SUPPORTED_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        raise ValueError(f"formato header non supportato: {format_name!r} (attesi: {supported})")
    return meta, data_offset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ispeziona un file capture .bin con header rt_capture_v1 o rt_capture_v2."
    )
    parser.add_argument(
        "bin_path",
        nargs="?",
        default=BIN_PATH,
        help="Path del file .bin da ispezionare (oppure imposta BIN_PATH nel file)",
    )
    parser.add_argument(
        "--show-header-json",
        action="store_true",
        default=SHOW_HEADER_JSON,
        help="Stampa il JSON header completo (se presente)",
    )
    args = parser.parse_args(argv)

    if args.bin_path is None:
        parser.error("specifica bin_path oppure imposta BIN_PATH in cima al file")

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
    print(f"Formato: {header['format']}")
    print(f"Data offset: {data_offset} byte")
    print(f"Payload: {payload_size} byte ({_fmt_bytes(payload_size)})")

    pos_from_name = None
    m = CAPTURE_NAME_RE.match(path.name)
    if m is not None:
        pos_from_name = int(m.group(1))
        print(f"Posizione da nome file: {pos_from_name}")

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

    samples = _to_int(capture.get("samples"))
    chirps = _to_int(capture.get("chirps"))
    rx = _to_int(capture.get("rx"))

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
            "[WARN] Header capture incompleto: impossibile calcolare i frame "
            "(servono samples/chirps/rx)."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
