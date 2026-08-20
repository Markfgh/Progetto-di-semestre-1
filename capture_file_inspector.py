"""Utility CLI per verificare struttura e metadati dei file di cattura RTPBIN.

Valida l'header JSON e la geometria completa del payload prima di avviare una
ricostruzione offline più costosa.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any


# Parametri modificabili ------------------------------------------------------
# File da ispezionare se non viene passato come primo argomento dalla riga di comando.
BIN_PATH: str | None = None
# Stampa l'intero header JSON; lascia False per un riepilogo compatto.
SHOW_HEADER_JSON = True

HEADER_MAGIC = b"RTPBIN1\x00"
HEADER_PREFIX_LEN = len(HEADER_MAGIC) + 4
HEADER_MAX_LEN = 256 * 1024
CAPTURE_NAME_RE = re.compile(r"^capture_pos(-?\d+)\.bin$")
SUPPORTED_FORMATS = {"rt_capture_v1"}
CAPTURE_KIND_SAR = "sar"
CAPTURE_KIND_MANUAL_NO_STAGE = "manual_no_stage"
SUPPORTED_CAPTURE_KINDS = {CAPTURE_KIND_SAR, CAPTURE_KIND_MANUAL_NO_STAGE}


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


def _required_object(meta: dict[str, Any], field_name: str) -> dict[str, Any]:
    value = meta.get(field_name)
    if not isinstance(value, dict):
        raise ValueError(f"header senza oggetto obbligatorio {field_name!r}")
    return value


def _required_int(field_name: str, value: Any) -> int:
    if value is None:
        raise ValueError(f"{field_name} mancante")
    if isinstance(value, bool):
        raise ValueError(f"{field_name} non valido: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        raise ValueError(f"{field_name} deve essere un intero, trovato: {value!r}")
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value.strip())
    raise ValueError(f"{field_name} non valido: {value!r}")


def _required_positive_int(field_name: str, value: Any) -> int:
    result = _required_int(field_name, value)
    if result <= 0:
        raise ValueError(f"{field_name} deve essere > 0")
    return result


def _required_finite_float(field_name: str, value: Any) -> float:
    if value is None:
        raise ValueError(f"{field_name} mancante")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} non valido: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} deve essere finito")
    return result


def _required_positive_float(field_name: str, value: Any) -> float:
    result = _required_finite_float(field_name, value)
    if result <= 0.0:
        raise ValueError(f"{field_name} deve essere > 0")
    return result


def validate_capture_metadata(meta: dict[str, Any], path: Path) -> None:
    """Validate capture metadata, including explicitly non-SAR captures."""
    format_name = meta.get("format")
    if format_name not in SUPPORTED_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        raise ValueError(
            f"formato header non supportato: {format_name!r} (attesi: {supported})"
        )

    position = _required_int("header.position", meta.get("position"))
    name_match = CAPTURE_NAME_RE.match(path.name)
    if name_match is not None:
        position_from_name = int(name_match.group(1))
        if position_from_name != position:
            raise ValueError(
                f"{path.name}: posizione incoerente "
                f"(nome={position_from_name}, header={position})"
            )

    radar = _required_object(meta, "radar")
    for field in ("c", "fs", "slope", "fc"):
        _required_positive_float(f"header.radar.{field}", radar.get(field))

    capture = _required_object(meta, "capture")
    _required_positive_int("header.capture.samples", capture.get("samples"))
    chirps = _required_positive_int("header.capture.chirps", capture.get("chirps"))
    _required_positive_int("header.capture.rx", capture.get("rx"))
    tx = _required_positive_int("header.capture.tx", capture.get("tx"))
    if chirps % tx != 0:
        raise ValueError("header.capture.chirps deve essere multiplo di tx")

    # Sono conteggi facoltativi per compatibilita' tra versioni del logger,
    # ma quando presenti devono comunque descrivere batch non vuoti.
    for field in ("x_frames", "frames_per_position"):
        if capture.get(field) is not None:
            _required_positive_int(f"header.capture.{field}", capture.get(field))

    capture_kind = str(capture.get("kind", CAPTURE_KIND_SAR)).strip().lower()
    if capture_kind not in SUPPORTED_CAPTURE_KINDS:
        allowed = ", ".join(sorted(SUPPORTED_CAPTURE_KINDS))
        raise ValueError(f"header.capture.kind non valido: {capture_kind!r} (attesi: {allowed})")
    for field in ("capture_id", "acquisition_index"):
        if capture.get(field) is not None:
            value = _required_int(f"header.capture.{field}", capture.get(field))
            if value < 0:
                raise ValueError(f"header.capture.{field} deve essere >= 0")

    # L'asse stage puo' legittimamente avere coordinate <= 0.
    stage = meta.get("stage")
    if capture_kind == CAPTURE_KIND_SAR:
        stage = _required_object(meta, "stage")
        _required_finite_float("header.stage.position_mm", stage.get("position_mm"))
    elif stage is not None:
        if not isinstance(stage, dict):
            raise ValueError("header.stage deve essere un oggetto quando presente")
        if stage.get("position_mm") is not None:
            _required_finite_float("header.stage.position_mm", stage.get("position_mm"))


def _validate_payload_layout(path: Path, meta: dict[str, Any], data_offset: int) -> int:
    """Return the complete frame count or reject a truncated/incoherent payload."""
    capture = _required_object(meta, "capture")
    samples = _required_positive_int("header.capture.samples", capture.get("samples"))
    chirps = _required_positive_int("header.capture.chirps", capture.get("chirps"))
    rx = _required_positive_int("header.capture.rx", capture.get("rx"))
    bytes_per_frame = int(chirps) * int(samples) * int(rx) * 4
    payload_size = int(path.stat().st_size) - int(data_offset)
    remainder = int(payload_size % bytes_per_frame)
    if remainder != 0:
        raise ValueError(
            f"payload non multiplo di bytes_per_frame={bytes_per_frame}: resto={remainder}"
        )
    actual_frames = int(payload_size // bytes_per_frame)
    if actual_frames <= 0:
        raise ValueError("payload senza frame completi")
    if capture.get("frames_per_position") is not None:
        expected_frames = _required_positive_int(
            "header.capture.frames_per_position",
            capture.get("frames_per_position"),
        )
        if actual_frames != expected_frames:
            raise ValueError(
                "numero frame incoerente: "
                f"header.frames_per_position={expected_frames}, payload={actual_frames}"
            )
    return actual_frames


def read_capture_header(path: Path) -> tuple[dict, int]:
    """Valida l'header RTPBIN v1 e restituisce metadata più offset del payload."""
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
    validate_capture_metadata(meta, path)
    _validate_payload_layout(path, meta, data_offset)
    return meta, data_offset


def main(argv: list[str] | None = None) -> int:
    """Espone il controllo dell'header come comando a riga di comando."""
    parser = argparse.ArgumentParser(
        description="Ispeziona un file capture .bin con header rt_capture_v1."
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

    # Questi campi sono gia' stati validati da ``read_capture_header``.
    pos_from_header = _required_int("header.position", header.get("position"))
    created_at = header.get("created_at")
    radar = _required_object(header, "radar")
    capture = _required_object(header, "capture")

    if pos_from_header is not None:
        print(f"Posizione da header: {pos_from_header}")
    if created_at is not None:
        print(f"Data/ora: {created_at}")

    if radar:
        print(
            "Radar: "
            f"c={radar.get('c')} fs={radar.get('fs')} "
            f"slope={radar.get('slope')} fc={radar.get('fc')}"
        )
    if capture:
        capture_kind = str(capture.get("kind", CAPTURE_KIND_SAR)).strip().lower()
        print(
            "Capture: "
            f"samples={capture.get('samples')} chirps={capture.get('chirps')} "
            f"rx={capture.get('rx')} tx={capture.get('tx')} x_frames={capture.get('x_frames')} "
            f"kind={capture_kind}"
        )
        if capture_kind == CAPTURE_KIND_MANUAL_NO_STAGE:
            print("[WARN] Cattura manuale senza stage: non utilizzabile per ricostruzione SAR offline.")

    if args.show_header_json:
        print("Header JSON:")
        print(json.dumps(header, indent=2, ensure_ascii=False))

    samples = _required_positive_int("header.capture.samples", capture.get("samples"))
    chirps = _required_positive_int("header.capture.chirps", capture.get("chirps"))
    rx = _required_positive_int("header.capture.rx", capture.get("rx"))

    bytes_per_frame = chirps * samples * rx * 4
    print(f"bytes_per_frame: {bytes_per_frame}")
    n_frames = payload_size // bytes_per_frame
    print(f"Frame nel payload: {n_frames}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
