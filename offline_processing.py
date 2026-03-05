from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import json
import struct
import time
from typing import Any, Callable, Literal
import multiprocessing as mp
import queue as pyqueue
from multiprocessing import Process, Queue
from multiprocessing.sharedctypes import Synchronized

import numpy as np
import yaml

_CAPTURE_FILE_RE = re.compile(r"^capture_pos(-?\d+)\.bin$")
_CAPTURE_HEADER_MAGIC = b"RTPBIN1\x00"
_CAPTURE_HEADER_PREFIX_LEN = len(_CAPTURE_HEADER_MAGIC) + 4
_CAPTURE_HEADER_MAX_LEN = 256 * 1024


@dataclass(frozen=True)
class OfflineSARConfig:
    input_dir: Path
    x_start: int
    x_end: int
    x_step: int
    frames_per_position: int | None = None

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path = "offline_config.yaml",
        fallback_capture_cfg: str | Path = "Config.yaml",
    ) -> "OfflineSARConfig":
        cfg_path = Path(config_path)
        cfg = _load_yaml_file(cfg_path)
        cap_fallback_cfg = _load_yaml_file(Path(fallback_capture_cfg))

        data_cfg = cfg.get("data", {}) or {}
        scan_cfg = cfg.get("scan", {}) or {}
        cap_cfg = cfg.get("capture", {}) or {}
        sar_fallback = cap_fallback_cfg.get("sar", {}) or {}

        input_dir = _pick(data_cfg.get("input_dir"), data_cfg.get("folder"), data_cfg.get("path"))
        if input_dir is None:
            raise ValueError("offline_config.yaml: manca data.input_dir")
        input_dir_path = Path(str(input_dir))
        if not input_dir_path.is_absolute():
            input_dir_path = (cfg_path.parent / input_dir_path).resolve()

        x_start = _to_int("scan.x_start", _pick(scan_cfg.get("x_start"), scan_cfg.get("x_min")))
        x_end = _to_int("scan.x_end", _pick(scan_cfg.get("x_end"), scan_cfg.get("x_max")))
        x_step = _to_int("scan.x_step", _pick(scan_cfg.get("x_step"), scan_cfg.get("step"), 1))

        frames_per_position_raw = _pick(
            cap_cfg.get("frames_per_position"),
            cfg.get("frames_per_position"),
            sar_fallback.get("frames_per_position"),
        )
        frames_per_position = None if frames_per_position_raw is None else _to_int(
            "capture.frames_per_position",
            frames_per_position_raw,
        )

        if x_step <= 0:
            raise ValueError("scan.x_step deve essere > 0")
        if x_end < x_start:
            raise ValueError("scan.x_end deve essere >= scan.x_start")
        if frames_per_position is not None and frames_per_position <= 0:
            raise ValueError("capture.frames_per_position deve essere > 0")

        return cls(
            input_dir=input_dir_path,
            x_start=x_start,
            x_end=x_end,
            x_step=x_step,
            frames_per_position=frames_per_position,
        )


@dataclass
class SARData:
    source_dir: Path
    positions: np.ndarray
    iq_cube: np.ndarray
    raw_frames: np.ndarray | None
    n_frames_per_position: int
    bytes_per_frame: int
    samples: int
    chirps: int
    rx: int
    tx: int


class SARReader:
    """
    Legge i bin SAR e costruisce:
    iq_cube shape = [pos, frame, loop, antenna, sample]
    """

    def __init__(
        self,
        offline_config_path: str | Path = "offline_config.yaml",
        fallback_capture_cfg: str | Path = "Config.yaml",
    ) -> None:
        self.config = OfflineSARConfig.from_yaml(offline_config_path, fallback_capture_cfg)

    def load(self, keep_raw: bool = True) -> SARData:
        source_dir = self._resolve_source_dir(self.config.input_dir)
        pos_files = self._scan_position_files(source_dir)
        expected_positions = list(range(self.config.x_start, self.config.x_end + 1, self.config.x_step))
        self._validate_positions(pos_files, expected_positions)
        samples, chirps, rx, tx, frames_per_pos_hdr = self._derive_capture_layout(pos_files)

        n_pos = len(expected_positions)
        loops = chirps // tx
        n_ant = tx * rx
        bytes_per_frame = chirps * samples * rx * 4
        i16_per_frame = bytes_per_frame // 2
        frames_per_pos_cfg = self.config.frames_per_position
        if frames_per_pos_cfg is None:
            frames_per_pos_cfg = frames_per_pos_hdr

        pos_to_file = {pos: path for pos, path in pos_files}

        iq_cube: np.ndarray | None = None
        raw_by_pos: list[np.ndarray] = []
        n_frames_ref: int | None = None

        for pos_idx, pos in enumerate(expected_positions):
            raw_frames, n_frames = self.read_position(
                pos_to_file[pos],
                bytes_per_frame=bytes_per_frame,
                i16_per_frame=i16_per_frame,
            )

            if frames_per_pos_cfg is not None and n_frames != frames_per_pos_cfg:
                raise ValueError(
                    f"Posizione {pos}: n_frames={n_frames}, atteso frames_per_position={frames_per_pos_cfg}"
                )

            if n_frames_ref is None:
                n_frames_ref = n_frames
                iq_cube = np.empty((n_pos, n_frames_ref, loops, n_ant, samples), dtype=np.complex64)
            elif n_frames != n_frames_ref:
                raise ValueError(
                    f"Posizione {pos}: n_frames={n_frames}, atteso {n_frames_ref} "
                    "(tutti i file devono avere uguale numero di frame)"
                )

            iq_frames = self._raw_to_iq(
                raw_frames,
                n_frames,
                samples=samples,
                chirps=chirps,
                rx=rx,
                tx=tx,
            )
            assert iq_cube is not None
            iq_cube[pos_idx, ...] = iq_frames

            if keep_raw:
                raw_by_pos.append(raw_frames)

        if iq_cube is None or n_frames_ref is None:
            raise RuntimeError("Nessun dato caricato")

        raw_out: np.ndarray | None = None
        if keep_raw:
            raw_out = np.stack(raw_by_pos, axis=0)

        return SARData(
            source_dir=source_dir,
            positions=np.asarray(expected_positions, dtype=np.int32),
            iq_cube=iq_cube,
            raw_frames=raw_out,
            n_frames_per_position=n_frames_ref,
            bytes_per_frame=bytes_per_frame,
            samples=samples,
            chirps=chirps,
            rx=rx,
            tx=tx,
        )

    def read_position(
        self,
        file_path: str | Path,
        *,
        bytes_per_frame: int,
        i16_per_frame: int,
    ) -> tuple[np.ndarray, int]:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File non trovato: {path}")

        file_size = path.stat().st_size
        data_offset = self._detect_capture_data_offset(path, file_size)
        payload_size = int(file_size - data_offset)
        if payload_size % bytes_per_frame != 0:
            raise ValueError(
                f"{path.name}: payload_size={payload_size} (offset={data_offset}) "
                f"non multiplo di bytes_per_frame={bytes_per_frame}"
            )

        n_frames = payload_size // bytes_per_frame
        if n_frames <= 0:
            raise ValueError(f"{path.name}: nessun frame nel payload (offset={data_offset})")

        raw = np.fromfile(path, dtype=np.int16, offset=int(data_offset))
        expected_i16 = n_frames * i16_per_frame
        if raw.size != expected_i16:
            raise ValueError(
                f"{path.name}: int16 letti={raw.size}, attesi={expected_i16}"
            )

        raw_frames = raw.reshape(n_frames, i16_per_frame)
        return raw_frames, int(n_frames)

    def _detect_capture_data_offset(self, path: Path, file_size: int) -> int:
        if file_size < _CAPTURE_HEADER_PREFIX_LEN:
            raise ValueError(
                f"{path.name}: file troppo piccolo per header {_CAPTURE_HEADER_MAGIC!r}"
            )

        with path.open("rb") as f:
            prefix = f.read(_CAPTURE_HEADER_PREFIX_LEN)

        if len(prefix) != _CAPTURE_HEADER_PREFIX_LEN:
            raise ValueError(
                f"{path.name}: prefisso header incompleto (attesi {_CAPTURE_HEADER_PREFIX_LEN} byte)"
            )
        if prefix[: len(_CAPTURE_HEADER_MAGIC)] != _CAPTURE_HEADER_MAGIC:
            raise ValueError(
                f"{path.name}: magic header mancante o non valida (attesa {_CAPTURE_HEADER_MAGIC!r})"
            )

        header_len = int(struct.unpack("<I", prefix[len(_CAPTURE_HEADER_MAGIC) :])[0])
        if header_len <= 0 or header_len > _CAPTURE_HEADER_MAX_LEN:
            raise ValueError(
                f"{path.name}: header_len non valido ({header_len}), magic riconosciuta"
            )

        data_offset = _CAPTURE_HEADER_PREFIX_LEN + header_len
        if data_offset >= file_size:
            raise ValueError(
                f"{path.name}: header invalido (offset={data_offset}, file_size={file_size})"
            )
        return int(data_offset)

    def _extract_position_from_header(self, path: Path) -> int:
        meta = self._read_capture_header_metadata(path)
        if "position" not in meta:
            raise ValueError(f"{path.name}: header senza campo obbligatorio 'position'")
        try:
            return int(meta["position"])
        except Exception as e:
            raise ValueError(f"{path.name}: campo header 'position' non valido ({meta['position']!r})") from e

    def _read_capture_header_metadata(self, path: Path) -> dict:
        file_size = path.stat().st_size
        if file_size < _CAPTURE_HEADER_PREFIX_LEN:
            raise ValueError(
                f"{path.name}: file troppo piccolo per header {_CAPTURE_HEADER_MAGIC!r}"
            )

        with path.open("rb") as f:
            prefix = f.read(_CAPTURE_HEADER_PREFIX_LEN)
            if len(prefix) != _CAPTURE_HEADER_PREFIX_LEN:
                raise ValueError(
                    f"{path.name}: prefisso header incompleto (attesi {_CAPTURE_HEADER_PREFIX_LEN} byte)"
                )
            if prefix[: len(_CAPTURE_HEADER_MAGIC)] != _CAPTURE_HEADER_MAGIC:
                raise ValueError(
                    f"{path.name}: magic header mancante o non valida (attesa {_CAPTURE_HEADER_MAGIC!r})"
                )

            header_len = int(struct.unpack("<I", prefix[len(_CAPTURE_HEADER_MAGIC) :])[0])
            if header_len <= 0 or header_len > _CAPTURE_HEADER_MAX_LEN:
                raise ValueError(
                    f"{path.name}: header_len non valido ({header_len}), magic riconosciuta"
                )

            data_offset = _CAPTURE_HEADER_PREFIX_LEN + header_len
            if data_offset >= file_size:
                raise ValueError(
                    f"{path.name}: header invalido (offset={data_offset}, file_size={file_size})"
                )

            payload = f.read(header_len)
            if len(payload) != header_len:
                raise ValueError(
                    f"{path.name}: header tronco (attesi {header_len} byte, letti {len(payload)})"
                )

        try:
            meta = json.loads(payload.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"{path.name}: header JSON non valido") from e
        if not isinstance(meta, dict):
            raise ValueError(f"{path.name}: header JSON deve essere un oggetto")
        fmt = str(meta.get("format", ""))
        if fmt != "rt_capture_v1":
            raise ValueError(
                f"{path.name}: header format non supportato ({fmt!r}), atteso 'rt_capture_v1'"
            )
        if "position" not in meta:
            raise ValueError(f"{path.name}: header senza campo obbligatorio 'position'")
        return meta

    def _raw_to_iq(
        self,
        raw_frames: np.ndarray,
        n_frames: int,
        *,
        samples: int,
        chirps: int,
        rx: int,
        tx: int,
    ) -> np.ndarray:
        flat_i16 = raw_frames.reshape(-1)
        iq_block = 2 * rx
        if flat_i16.size % iq_block != 0:
            raise ValueError(
                f"Stream int16 non allineato a blocchi IQ da {iq_block} canali (I...Q...)"
            )

        block_view = flat_i16.reshape(-1, iq_block)
        complex_flat = np.empty(block_view.shape[0] * rx, dtype=np.complex64)
        complex_flat.real = block_view[:, :rx].reshape(-1)
        complex_flat.imag = block_view[:, rx:iq_block].reshape(-1)

        expected_complex = n_frames * chirps * samples * rx
        if complex_flat.size != expected_complex:
            raise ValueError(
                f"Campioni complessi={complex_flat.size}, attesi={expected_complex}"
            )

        loops = chirps // tx
        virtual_ant = tx * rx
        data_5d = complex_flat.reshape(n_frames, loops, tx, samples, rx)
        iq_frames = data_5d.transpose(0, 1, 2, 4, 3).reshape(
            n_frames,
            loops,
            virtual_ant,
            samples,
        )
        return iq_frames

    def _derive_capture_layout(
        self,
        pos_files: list[tuple[int, Path]],
    ) -> tuple[int, int, int, int, int | None]:
        samples_ref: int | None = None
        chirps_ref: int | None = None
        rx_ref: int | None = None
        tx_ref: int | None = None
        frames_per_pos_ref: int | None = None

        for _, path in pos_files:
            meta = self._read_capture_header_metadata(path)
            capture = meta.get("capture")
            if not isinstance(capture, dict):
                raise ValueError(f"{path.name}: header senza oggetto 'capture'")

            samples = _to_int("header.capture.samples", capture.get("samples"))
            chirps = _to_int("header.capture.chirps", capture.get("chirps"))
            rx = _to_int("header.capture.rx", capture.get("rx"))
            tx = _to_int("header.capture.tx", capture.get("tx"))
            frames = _pick(capture.get("x_frames"), capture.get("frames_per_position"))
            frames_per_pos = None if frames is None else _to_int("header.capture.x_frames", frames)

            if samples <= 0 or chirps <= 0 or rx <= 0 or tx <= 0:
                raise ValueError(f"{path.name}: header.capture contiene valori <= 0")
            if chirps % tx != 0:
                raise ValueError(f"{path.name}: header.capture.chirps deve essere multiplo di tx")
            if frames_per_pos is not None and frames_per_pos <= 0:
                raise ValueError(f"{path.name}: header.capture.x_frames deve essere > 0")

            if samples_ref is None:
                samples_ref, chirps_ref, rx_ref, tx_ref = samples, chirps, rx, tx
                frames_per_pos_ref = frames_per_pos
                continue

            if (
                samples != samples_ref
                or chirps != chirps_ref
                or rx != rx_ref
                or tx != tx_ref
            ):
                raise ValueError(
                    f"{path.name}: capture incoerente tra file "
                    f"(samples/chirps/rx/tx={samples}/{chirps}/{rx}/{tx}, "
                    f"attesi {samples_ref}/{chirps_ref}/{rx_ref}/{tx_ref})"
                )
            if frames_per_pos_ref is not None and frames_per_pos is not None and frames_per_pos != frames_per_pos_ref:
                raise ValueError(
                    f"{path.name}: x_frames incoerente ({frames_per_pos}, atteso {frames_per_pos_ref})"
                )

        if samples_ref is None or chirps_ref is None or rx_ref is None or tx_ref is None:
            raise RuntimeError("Impossibile derivare capture layout dai file .bin")
        return samples_ref, chirps_ref, rx_ref, tx_ref, frames_per_pos_ref

    def _scan_position_files(self, source_dir: Path) -> list[tuple[int, Path]]:
        pos_files: list[tuple[int, Path]] = []
        for p in source_dir.glob("*.bin"):
            match = _CAPTURE_FILE_RE.match(p.name)
            pos_from_name = int(match.group(1)) if match is not None else None
            try:
                pos_from_header = self._extract_position_from_header(p)
            except ValueError:
                # Skip files not in current capture format.
                continue

            if pos_from_name is not None and pos_from_header is not None and pos_from_name != pos_from_header:
                raise ValueError(
                    f"{p.name}: posizione incoerente (nome={pos_from_name}, header={pos_from_header})"
                )

            pos = pos_from_header
            pos_files.append((int(pos), p))

        if not pos_files:
            raise FileNotFoundError(
                f"Nessun file di capture valido trovato in {source_dir} "
                f"(richiesto header {_CAPTURE_HEADER_MAGIC!r} con format='rt_capture_v1')"
            )

        pos_files.sort(key=lambda t: t[0])
        pos_ids = [pos for pos, _ in pos_files]
        if len(pos_ids) != len(set(pos_ids)):
            raise ValueError(f"Posizioni duplicate trovate in {source_dir}: {pos_ids}")

        return pos_files

    def _validate_positions(
        self,
        pos_files: list[tuple[int, Path]],
        expected_positions: list[int],
    ) -> None:
        found_positions = [pos for pos, _ in pos_files]
        found_set = set(found_positions)
        expected_set = set(expected_positions)

        missing = [pos for pos in expected_positions if pos not in found_set]
        extra = sorted([pos for pos in found_positions if pos not in expected_set])

        if missing or extra:
            raise ValueError(
                "Posizioni non valide. "
                f"Missing={missing if missing else '[]'} | Extra={extra if extra else '[]'}"
            )

    @staticmethod
    def _resolve_source_dir(input_dir: Path) -> Path:
        if not input_dir.exists():
            raise FileNotFoundError(f"Cartella input non trovata: {input_dir}")

        direct = list(input_dir.glob("*.bin"))
        if direct:
            return input_dir

        run_dirs = sorted([p for p in input_dir.glob("run_*") if p.is_dir()], key=lambda p: p.name)
        for run_dir in reversed(run_dirs):
            if any(run_dir.glob("*.bin")):
                return run_dir

        raise FileNotFoundError(
            f"Nessun file .bin trovato in {input_dir} (neanche dentro run_*)"
        )


class SARProcessor:
    def __init__(self, reader: SARReader) -> None:
        self.reader = reader

    def read(self, keep_raw: bool = False) -> SARData:
        return self.reader.load(keep_raw=keep_raw)

    def process(self, data: SARData | None = None) -> SARData:
        if data is None:
            data = self.read(keep_raw=False)
        return data

    @staticmethod
    def range_fft(iq_cube: np.ndarray, nfft: int | None = None) -> np.ndarray:
        fft_len = int(nfft) if nfft is not None else int(iq_cube.shape[-1])
        return np.fft.fft(iq_cube, n=fft_len, axis=-1)


class PostProcManager:
    def __init__(
        self,
        reader: SARReader | None = None,
        processor: SARProcessor | None = None,
    ) -> None:
        self.reader = reader or SARReader()
        self.processor = processor or SARProcessor(self.reader)
        self._steps: list[Callable[[SARData], SARData]] = []

    def add_step(self, step: Callable[[SARData], SARData]) -> None:
        self._steps.append(step)

    def run(self, keep_raw: bool = False) -> SARData:
        data = self.processor.process(self.processor.read(keep_raw=keep_raw))
        for step in self._steps:
            data = step(data)
        return data


def _load_yaml_file(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded or {}


def _pick(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _to_int(field_name: str, value) -> int:
    if value is None:
        raise ValueError(f"{field_name} mancante")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} non valido: {value!r}") from exc


def _queue_put_latest(q: Queue, msg: dict[str, Any]) -> None:
    """
    Inserisce l'ultimo messaggio disponibile senza bloccare.
    Se la queue e' piena, scarta elementi vecchi per fare spazio.
    """
    try:
        q.put_nowait(msg)
        return
    except pyqueue.Full:
        pass
    except Exception:
        return

    try:
        while True:
            q.get_nowait()
    except pyqueue.Empty:
        pass
    except Exception:
        return

    try:
        q.put_nowait(msg)
    except Exception:
        pass


AvgMode = Literal["none", "loop", "frame", "both"]
_AVG_MODES = {"none", "loop", "frame", "both"}


def _avg_mode_normalize(mode: str | None) -> AvgMode:
    mode_s = str(mode or "both").strip().lower()
    if mode_s not in _AVG_MODES:
        return "both"
    return mode_s  # type: ignore[return-value]


def _phase_sign_normalize(value: Any, *, field_name: str = "phase_sign") -> int:
    try:
        phase_sign_i = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} non valido: {value!r}") from exc
    if phase_sign_i not in (-1, 1):
        raise ValueError(f"{field_name} deve essere +1 o -1, trovato: {value!r}")
    return int(phase_sign_i)


def _read_x_pitch_m(offline_config_path: str | Path) -> float:
    cfg = _load_yaml_file(Path(offline_config_path))
    scan_cfg = cfg.get("scan", {}) or {}
    raw = _pick(scan_cfg.get("x_pitch_m"), scan_cfg.get("pitch_m"), scan_cfg.get("step_m"), 0.01)
    try:
        x_pitch_m = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"offline_config: scan.x_pitch_m non valido: {raw!r}") from exc
    if x_pitch_m <= 0.0:
        raise ValueError("offline_config: scan.x_pitch_m deve essere > 0")
    return float(x_pitch_m)


def _read_phase_sign(offline_config_path: str | Path) -> int:
    cfg = _load_yaml_file(Path(offline_config_path))
    bp_cfg = cfg.get("bp", {}) or {}
    raw = _pick(bp_cfg.get("phase_sign"), -1)
    return _phase_sign_normalize(raw, field_name="offline_config: bp.phase_sign")


def _reduce_avg_mode(range_fft_4d: np.ndarray, avg_mode: AvgMode) -> np.ndarray:
    # Input shape: [pos, frame, loop, range_bin]
    if range_fft_4d.ndim != 4:
        raise ValueError(f"range_fft_4d shape non valido: {range_fft_4d.shape!r}")

    if avg_mode == "both":
        out = range_fft_4d.mean(axis=(1, 2), dtype=np.complex64)  # [pos, range_bin]
        return out[:, np.newaxis, :].astype(np.complex64, copy=False)  # [pos, 1, range_bin]
    if avg_mode == "frame":
        out = range_fft_4d.mean(axis=1, dtype=np.complex64)
        return out.astype(np.complex64, copy=False)
    if avg_mode == "loop":
        out = range_fft_4d.mean(axis=2, dtype=np.complex64)
        return out.astype(np.complex64, copy=False)

    n_pos, n_frames, n_loops, n_bins = range_fft_4d.shape
    return range_fft_4d.reshape(n_pos, n_frames * n_loops, n_bins)


def _back_projection_image(
    range_fft_sel: np.ndarray,
    x_pos_m: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    dr_m: float,
    fc_hz: float,
    c_m_s: float,
    max_bin: int,
    phase_sign: int = -1,
    chunk_size: int = 16384,
) -> np.ndarray:
    # range_fft_sel shape: [pos, pulses, range_bin]
    if range_fft_sel.ndim != 3:
        raise ValueError(f"range_fft_sel shape non valido: {range_fft_sel.shape!r}")
    if x_pos_m.ndim != 1:
        raise ValueError(f"x_pos_m shape non valido: {x_pos_m.shape!r}")
    if x_grid.shape != y_grid.shape:
        raise ValueError("x_grid e y_grid devono avere stessa shape")
    if float(dr_m) <= 0.0:
        raise ValueError("dr_m deve essere > 0")
    phase_sign_i = _phase_sign_normalize(phase_sign, field_name="phase_sign")

    n_pos = int(range_fft_sel.shape[0])
    if n_pos <= 0:
        raise ValueError("Nessuna posizione selezionata per back projection")
    if x_pos_m.size != n_pos:
        raise ValueError("x_pos_m size != numero posizioni range_fft_sel")

    x_flat = x_grid.reshape(-1).astype(np.float32, copy=False)
    y_flat = y_grid.reshape(-1).astype(np.float32, copy=False)
    y_sq = (y_flat * y_flat).astype(np.float32, copy=False)
    img_flat = np.zeros(x_flat.shape[0], dtype=np.complex64)
    k = np.float32((4.0 * np.pi * float(fc_hz)) / float(c_m_s))
    phase_scale = np.float32(float(phase_sign_i)) * k
    inv_dr = np.float32(1.0 / float(dr_m))
    n_bins_avail = int(range_fft_sel.shape[2])
    max_bin_eff = max(0, min(int(max_bin), n_bins_avail))
    if max_bin_eff < 2:
        img_mag = np.abs(img_flat).reshape(x_grid.shape)
        return (20.0 * np.log10(img_mag + 1e-6)).astype(np.float32, copy=False)
    chunk_n = max(1, int(chunk_size))

    for pos_i in range(n_pos):
        dx = x_flat - np.float32(x_pos_m[pos_i])
        rr = np.sqrt(dx * dx + y_sq).astype(np.float32, copy=False)
        b = rr * inv_dr
        b0 = np.floor(b).astype(np.int32, copy=False)
        valid = np.isfinite(b) & (b0 >= 0) & (b0 < (max_bin_eff - 1))
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            continue

        # La media sui pulse preserva la logica avg_mode ed evita allocazioni [pulse, n_pixel].
        spec = range_fft_sel[pos_i]
        if spec.ndim != 2 or spec.shape[0] <= 0:
            continue
        s_mean = spec.mean(axis=0, dtype=np.complex64)
        if s_mean.size < max_bin_eff:
            max_bin_local = int(s_mean.size)
            if max_bin_local < 2:
                continue
        else:
            max_bin_local = max_bin_eff

        for start in range(0, int(valid_idx.size), chunk_n):
            idx = valid_idx[start : start + chunk_n]
            if idx.size == 0:
                continue
            b_chunk = b[idx]
            b0_chunk = np.floor(b_chunk).astype(np.int32, copy=False)
            np.clip(b0_chunk, 0, max_bin_local - 2, out=b0_chunk)
            frac = (b_chunk - b0_chunk.astype(np.float32, copy=False)).astype(np.float32, copy=False)

            s0 = s_mean[b0_chunk]
            s1 = s_mean[b0_chunk + 1]
            s_interp = s0 + (s1 - s0) * frac

            phase = np.exp(1j * (phase_scale * rr[idx])).astype(np.complex64, copy=False)
            img_flat[idx] += s_interp * phase

    img_mag = np.abs(img_flat).reshape(x_grid.shape)
    img_db = (20.0 * np.log10(img_mag + 1e-6)).astype(np.float32, copy=False)
    return img_db


def _offline_reader_worker(
    offline_config_path: str,
    fallback_capture_cfg: str,
    nfft_range: int,
    reader_to_dsp_q: Queue,
    status_q: Queue,
    stop_evt,
) -> None:
    try:
        reader = SARReader(offline_config_path=offline_config_path, fallback_capture_cfg=fallback_capture_cfg)
        data = reader.load(keep_raw=False)

        # Riduzione preliminare: media sulle antenne virtuali per ridurre payload e costo DSP.
        sig = data.iq_cube.mean(axis=3, dtype=np.complex64)  # [pos, frame, loop, sample]
        range_fft = np.fft.fft(sig, n=int(nfft_range), axis=-1).astype(np.complex64, copy=False)

        msg = {
            "type": "data",
            "range_fft": range_fft,
            "positions": data.positions.astype(np.int32, copy=False),
            "x_start_cfg": int(reader.config.x_start),
            "x_end_cfg": int(reader.config.x_end),
        }
        _queue_put_latest(reader_to_dsp_q, msg)
        _queue_put_latest(
            status_q,
            {
                "type": "reader_ready",
                "positions": int(data.positions.size),
                "frames_per_pos": int(data.n_frames_per_position),
            },
        )
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        try:
            _queue_put_latest(reader_to_dsp_q, {"type": "error", "error": err})
        except Exception:
            pass
        try:
            _queue_put_latest(status_q, {"type": "error", "error": f"[offline_reader] {err}"})
        except Exception:
            pass
    finally:
        stop_evt.wait(0.01)


def _offline_dsp_worker(
    reader_to_dsp_q: Queue,
    cmd_q: Queue,
    status_q: Queue,
    gui_dbuf,
    gui_h: int,
    gui_w: int,
    gui_latest_idx: Synchronized,
    gui_latest_seq: Synchronized,
    gui_lock,
    stop_evt,
    *,
    c_m_s: float,
    fs_hz: float,
    slope_hz_s: float,
    fc_hz: float,
    nfft_range: int,
    range_max_m: float,
    crossrange_max_m: float,
    x_pitch_m: float,
    default_avg_mode: str,
    phase_sign: int,
) -> None:
    try:
        phase_sign_i = _phase_sign_normalize(phase_sign, field_name="phase_sign")
        init_msg = None
        while not stop_evt.is_set():
            try:
                init_msg = reader_to_dsp_q.get(timeout=0.1)
                break
            except pyqueue.Empty:
                continue

        if init_msg is None:
            _queue_put_latest(status_q, {"type": "error", "error": "[offline_dsp] timeout attesa dati dal reader"})
            return

        if str(init_msg.get("type")) == "error":
            _queue_put_latest(status_q, {"type": "error", "error": str(init_msg.get("error", "errore reader"))})
            return

        range_fft_4d = np.asarray(init_msg["range_fft"], dtype=np.complex64)
        positions = np.asarray(init_msg["positions"], dtype=np.int32)
        if range_fft_4d.ndim != 4:
            raise ValueError(f"range_fft_4d shape non valido: {range_fft_4d.shape!r}")
        if positions.ndim != 1 or positions.size != range_fft_4d.shape[0]:
            raise ValueError("positions non coerente con range_fft_4d")

        n_pos, _, _, n_bins_total = range_fft_4d.shape
        dr_m = float(c_m_s) * float(fs_hz) / (2.0 * float(slope_hz_s) * float(nfft_range))
        max_bin = int(np.floor(float(range_max_m) / dr_m))
        max_bin = max(1, min(max_bin, int(n_bins_total)))

        x_axis = np.linspace(
            -float(crossrange_max_m),
            +float(crossrange_max_m),
            int(gui_w),
            dtype=np.float32,
        )
        y_axis = np.linspace(
            0.0,
            float(range_max_m),
            int(gui_h),
            dtype=np.float32,
        )
        x_grid, y_grid = np.meshgrid(x_axis, y_axis)

        pos_f = positions.astype(np.float32, copy=False)
        pos_center = float(np.mean(pos_f)) if pos_f.size > 0 else 0.0
        x_pos_m_full = (pos_f - np.float32(pos_center)) * np.float32(float(x_pitch_m))

        pos_min = int(np.min(positions))
        pos_max = int(np.max(positions))
        x_start = int(_pick(init_msg.get("x_start_cfg"), pos_min))
        x_end = int(_pick(init_msg.get("x_end_cfg"), pos_max))
        x_start = max(pos_min, min(pos_max, x_start))
        x_end = max(pos_min, min(pos_max, x_end))
        if x_end < x_start:
            x_start, x_end = x_end, x_start
        avg_mode: AvgMode = _avg_mode_normalize(default_avg_mode)

        _queue_put_latest(
            status_q,
            {
                "type": "ready",
                "pos_min": pos_min,
                "pos_max": pos_max,
                "x_start": x_start,
                "x_end": x_end,
                "avg_mode": avg_mode,
                "phase_sign": int(phase_sign_i),
                "img_h": int(gui_h),
                "img_w": int(gui_w),
                "dr_m": float(dr_m),
            },
        )

        last_job_key = None
        dirty = True

        while not stop_evt.is_set():
            got_cmd = False
            while True:
                try:
                    cmd = cmd_q.get_nowait()
                except pyqueue.Empty:
                    break

                got_cmd = True
                if not isinstance(cmd, dict):
                    continue
                cmd_type = str(cmd.get("type", "")).strip().lower()
                if cmd_type == "stop":
                    stop_evt.set()
                    break
                if cmd_type != "update":
                    continue

                x_start_new = _pick(cmd.get("x_start"), x_start)
                x_end_new = _pick(cmd.get("x_end"), x_end)
                try:
                    x_start = max(pos_min, min(pos_max, int(x_start_new)))
                    x_end = max(pos_min, min(pos_max, int(x_end_new)))
                except Exception:
                    pass
                if x_end < x_start:
                    x_start, x_end = x_end, x_start
                avg_mode_new = cmd.get("avg_mode", None)
                if avg_mode_new is not None:
                    avg_mode = _avg_mode_normalize(str(avg_mode_new))
                dirty = True

            if stop_evt.is_set():
                break

            job_key = (int(x_start), int(x_end), str(avg_mode))
            if dirty or job_key != last_job_key or got_cmd:
                t0 = time.perf_counter()

                sel_mask = (positions >= int(x_start)) & (positions <= int(x_end))
                if not np.any(sel_mask):
                    sel_mask[:] = True
                sel_idx = np.where(sel_mask)[0]
                range_fft_sel = range_fft_4d[sel_idx, :, :, :max_bin]
                x_pos_m_sel = x_pos_m_full[sel_idx]

                reduced = _reduce_avg_mode(range_fft_sel, avg_mode)
                img_db = _back_projection_image(
                    reduced,
                    x_pos_m_sel,
                    x_grid,
                    y_grid,
                    dr_m=dr_m,
                    fc_hz=fc_hz,
                    c_m_s=c_m_s,
                    max_bin=max_bin,
                    phase_sign=phase_sign_i,
                    chunk_size=16384,
                )

                with gui_lock:
                    prev_idx = int(gui_latest_idx.value)
                    next_idx = 1 if prev_idx == 0 else 0
                    base = next_idx * int(gui_h) * int(gui_w)
                    dst = np.frombuffer(
                        gui_dbuf,
                        dtype=np.float32,
                        count=int(gui_h) * int(gui_w),
                        offset=base * 4,
                    )
                    dst[:] = img_db.reshape(-1)
                    gui_latest_idx.value = int(next_idx)
                    gui_latest_seq.value = int(gui_latest_seq.value) + 1

                t1 = time.perf_counter()
                _queue_put_latest(
                    status_q,
                    {
                        "type": "frame",
                        "x_start": int(x_start),
                        "x_end": int(x_end),
                        "avg_mode": str(avg_mode),
                        "n_pos_used": int(sel_idx.size),
                        "elapsed_ms": float((t1 - t0) * 1000.0),
                    },
                )

                last_job_key = job_key
                dirty = False
            else:
                time.sleep(0.01)
    except Exception as exc:
        try:
            _queue_put_latest(status_q, {"type": "error", "error": f"[offline_dsp] {type(exc).__name__}: {exc}"})
        except Exception:
            pass


class OfflineBPRuntime:
    """
    Runtime offline a due processi:
    - reader process: carica bin e prepara range-FFT 4D [pos, frame, loop, range]
    - dsp process: applica media runtime + back projection e pubblica frame su double-buffer
    """

    def __init__(
        self,
        *,
        offline_config_path: str | Path = "offline_config.yaml",
        fallback_capture_cfg: str | Path = "Config.yaml",
        c_m_s: float,
        fs_hz: float,
        slope_hz_s: float,
        fc_hz: float,
        nfft_range: int,
        range_max_m: float,
        crossrange_max_m: float,
        image_h: int,
        image_w: int,
        x_pitch_m: float | None = None,
        default_avg_mode: str = "both",
        phase_sign: int | None = None,
    ) -> None:
        self.offline_config_path = str(offline_config_path)
        self.fallback_capture_cfg = str(fallback_capture_cfg)
        self.c_m_s = float(c_m_s)
        self.fs_hz = float(fs_hz)
        self.slope_hz_s = float(slope_hz_s)
        self.fc_hz = float(fc_hz)
        self.nfft_range = int(nfft_range)
        self.range_max_m = float(range_max_m)
        self.crossrange_max_m = float(crossrange_max_m)
        self.image_h = int(image_h)
        self.image_w = int(image_w)
        self.x_pitch_m = float(x_pitch_m) if x_pitch_m is not None else _read_x_pitch_m(self.offline_config_path)
        self.default_avg_mode = _avg_mode_normalize(default_avg_mode)
        self.phase_sign = _phase_sign_normalize(
            _pick(phase_sign, _read_phase_sign(self.offline_config_path)),
            field_name="phase_sign",
        )

        self._started = False
        self._ready = False
        self._last_seq = 0
        self._last_error: str | None = None
        self._last_info: dict[str, Any] = {}
        self._frame_cache = np.zeros((self.image_h, self.image_w), dtype=np.float32)

        self._reader_to_dsp_q: Queue | None = None
        self._cmd_q: Queue | None = None
        self._status_q: Queue | None = None
        self._stop_evt = None
        self._reader_p: Process | None = None
        self._dsp_p: Process | None = None

        self.gui_dbuf = mp.RawArray("f", 2 * self.image_h * self.image_w)
        self.gui_latest_idx = mp.Value("i", -1)
        self.gui_latest_seq = mp.Value("Q", 0)
        self.gui_lock = mp.Lock()

    @property
    def ready(self) -> bool:
        return bool(self._ready)

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def last_info(self) -> dict[str, Any]:
        return dict(self._last_info)

    def start(self, timeout_s: float = 30.0) -> dict[str, Any]:
        if self._started:
            self._drain_status()
            return self.last_info

        # Reset stato locale prima di un nuovo avvio.
        self._ready = False
        self._last_seq = 0
        self._last_error = None
        self._last_info = {}
        self._frame_cache.fill(0.0)
        with self.gui_lock:
            self.gui_latest_idx.value = -1
            self.gui_latest_seq.value = 0

        self._reader_to_dsp_q = mp.Queue(maxsize=1)
        self._cmd_q = mp.Queue(maxsize=4)
        self._status_q = mp.Queue(maxsize=64)
        self._stop_evt = mp.Event()

        self._reader_p = Process(
            target=_offline_reader_worker,
            args=(
                self.offline_config_path,
                self.fallback_capture_cfg,
                int(self.nfft_range),
                self._reader_to_dsp_q,
                self._status_q,
                self._stop_evt,
            ),
        )
        self._dsp_p = Process(
            target=_offline_dsp_worker,
            args=(
                self._reader_to_dsp_q,
                self._cmd_q,
                self._status_q,
                self.gui_dbuf,
                int(self.image_h),
                int(self.image_w),
                self.gui_latest_idx,
                self.gui_latest_seq,
                self.gui_lock,
                self._stop_evt,
            ),
            kwargs={
                "c_m_s": float(self.c_m_s),
                "fs_hz": float(self.fs_hz),
                "slope_hz_s": float(self.slope_hz_s),
                "fc_hz": float(self.fc_hz),
                "nfft_range": int(self.nfft_range),
                "range_max_m": float(self.range_max_m),
                "crossrange_max_m": float(self.crossrange_max_m),
                "x_pitch_m": float(self.x_pitch_m),
                "default_avg_mode": str(self.default_avg_mode),
                "phase_sign": int(self.phase_sign),
            },
        )
        self._reader_p.daemon = True
        self._dsp_p.daemon = True
        self._reader_p.start()
        self._dsp_p.start()
        self._started = True

        t_deadline = time.perf_counter() + float(timeout_s)
        while time.perf_counter() < t_deadline:
            self._drain_status()
            if self._ready:
                return self.last_info
            if self._last_error is not None:
                raise RuntimeError(self._last_error)
            if self._dsp_p is not None and (not self._dsp_p.is_alive()):
                break
            time.sleep(0.02)

        self._drain_status()
        err = self._last_error or "timeout start offline runtime"
        raise RuntimeError(err)

    def stop(self) -> None:
        if not self._started:
            return

        try:
            if self._cmd_q is not None:
                self._put_latest_cmd({"type": "stop"})
        except Exception:
            pass
        try:
            if self._stop_evt is not None:
                self._stop_evt.set()
        except Exception:
            pass

        for proc in (self._reader_p, self._dsp_p):
            if proc is None:
                continue
            try:
                proc.join(timeout=0.4)
            except Exception:
                pass
        for proc in (self._reader_p, self._dsp_p):
            if proc is None:
                continue
            try:
                if proc.is_alive():
                    proc.terminate()
            except Exception:
                pass
        for proc in (self._reader_p, self._dsp_p):
            if proc is None:
                continue
            try:
                proc.join(timeout=0.2)
            except Exception:
                pass

        self._started = False
        self._ready = False
        self._reader_p = None
        self._dsp_p = None
        self._reader_to_dsp_q = None
        self._cmd_q = None
        self._status_q = None
        self._stop_evt = None

    def update_params(
        self,
        *,
        x_start: int | None = None,
        x_end: int | None = None,
        avg_mode: str | None = None,
    ) -> None:
        if not self._started or self._cmd_q is None:
            return
        cmd = {
            "type": "update",
            "x_start": x_start,
            "x_end": x_end,
            "avg_mode": None if avg_mode is None else _avg_mode_normalize(avg_mode),
        }
        self._put_latest_cmd(cmd)

    def poll_frame(self) -> tuple[np.ndarray, dict[str, Any]] | None:
        self._drain_status()
        if self._last_error is not None:
            return None

        seq_now = int(self.gui_latest_seq.value)
        if seq_now == int(self._last_seq):
            return None

        with self.gui_lock:
            seq_locked = int(self.gui_latest_seq.value)
            idx = int(self.gui_latest_idx.value)
            if idx not in (0, 1):
                return None
            base = idx * int(self.image_h) * int(self.image_w)
            src = np.frombuffer(
                self.gui_dbuf,
                dtype=np.float32,
                count=int(self.image_h) * int(self.image_w),
                offset=base * 4,
            )
            self._frame_cache.reshape(-1)[:] = src
            self._last_seq = int(seq_locked)

        return self._frame_cache.copy(), self.last_info

    def _put_latest_cmd(self, cmd: dict[str, Any]) -> None:
        if self._cmd_q is None:
            return
        try:
            while True:
                self._cmd_q.get_nowait()
        except pyqueue.Empty:
            pass
        try:
            self._cmd_q.put_nowait(cmd)
        except pyqueue.Full:
            pass

    def _drain_status(self) -> None:
        if self._status_q is None:
            return
        while True:
            try:
                msg = self._status_q.get_nowait()
            except pyqueue.Empty:
                break
            if not isinstance(msg, dict):
                continue
            kind = str(msg.get("type", "")).strip().lower()
            if kind == "error":
                self._last_error = str(msg.get("error", "errore offline runtime"))
                continue
            # Any valid status/frame message clears previous transient errors.
            self._last_error = None
            if kind == "ready":
                self._ready = True
            self._last_info.update(msg)
