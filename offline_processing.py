from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re
import json
import struct
import time
from typing import Any
import multiprocessing as mp
import queue as pyqueue
from multiprocessing import Process, Queue
from multiprocessing import shared_memory
from multiprocessing.sharedctypes import Synchronized

import numpy as np
import scipy.fft as fft
import yaml

from realtime_dsp import (
    AngleProcessingConfig,
    BackgroundSubtractionConfig,
    BackgroundSubtractionState,
    DisplayViewport,
    DisplayProjectionConfig,
    PostRangeFftFilterConfig,
    RealtimeDSPConfig,
    VirtualArrayGeometry,
    angle_processing_from_yaml_dict,
    apply_background_subtraction,
    apply_slow_time_filter,
    build_display_viewport,
    build_angle_axis_deg,
    build_angle_steering_matrix,
    build_display_projection_lut,
    compute_angle_heatmap,
    clamp_display_viewport,
    display_post_range_fft_filters_from_yaml_dict,
    display_viewport_signature,
    display_projection_from_yaml_dict,
    project_heatmap_for_display,
    sanitize_display_post_range_fft_filters,
    subtract_selected_mean,
    window_type_normalize,
)
from offline_dsp import (
    AvgMode,
    BpMode,
    MotionMode,
    avg_mode_normalize as _avg_mode_normalize,
    back_projection_image as _back_projection_image,
    back_projection_image_mimo as _back_projection_image_mimo,
    bp_mode_normalize as _bp_mode_normalize,
    build_mimo_back_projection_plan as _build_mimo_back_projection_plan,
    build_mimo_geometry as _build_mimo_geometry,
    motion_mode_normalize as _motion_mode_normalize,
    phase_sign_normalize as _phase_sign_normalize,
    power_image_to_db as _power_image_to_db,
    prepare_mimo_snapshots as _prepare_mimo_snapshots,
    prepare_synthetic_aperture_data as _prepare_synthetic_aperture_data,
    reduce_avg_mode as _reduce_avg_mode,
    synthetic_aperture_uniform_spacing_lambda as _synthetic_aperture_uniform_spacing_lambda,
)
from shutdown_utils import cleanup_processes, close_queues

_CAPTURE_FILE_RE = re.compile(r"^capture_pos(-?\d+)\.bin$")
_CAPTURE_HEADER_MAGIC = b"RTPBIN1\x00"
_CAPTURE_HEADER_PREFIX_LEN = len(_CAPTURE_HEADER_MAGIC) + 4
_CAPTURE_HEADER_MAX_LEN = 256 * 1024
_RECONSTRUCTION_ALGORITHMS = {"backprojection", "synthetic_range_angle"}


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


@dataclass(frozen=True)
class OfflineSyntheticRangeAngleConfig:
    use_realtime_filters: bool
    window_range: str
    window_doppler: str
    window_angle: str
    zero_after_range_fft_bins: int
    post_range_fft_filters: PostRangeFftFilterConfig
    angle_processing: AngleProcessingConfig
    nfft_angle: int
    projection: DisplayProjectionConfig
    filter_warnings: tuple[str, ...] = ()


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


def _reconstruction_algorithm_normalize(value: Any) -> str:
    algorithm = str(value or "backprojection").strip().lower()
    if algorithm not in _RECONSTRUCTION_ALGORITHMS:
        raise ValueError(
            "reconstruction.algorithm non valido: "
            f"{value!r}. Valori supportati: 'backprojection', 'synthetic_range_angle'."
        )
    return algorithm


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


def _viewport_status_fields(viewport: DisplayViewport, *, fallback_used: bool) -> dict[str, Any]:
    return {
        "applied_viewport_x_min_m": float(viewport.x_min_m),
        "applied_viewport_x_max_m": float(viewport.x_max_m),
        "applied_viewport_y_min_m": float(viewport.y_min_m),
        "applied_viewport_y_max_m": float(viewport.y_max_m),
        "applied_viewport_range_min_bin_f": float(viewport.range_min_bin_f),
        "applied_viewport_range_max_bin_f": float(viewport.range_max_bin_f),
        "applied_viewport_angle_min_deg": float(viewport.angle_min_deg),
        "applied_viewport_angle_max_deg": float(viewport.angle_max_deg),
        "applied_viewport_zoom_level": float(viewport.zoom_level),
        "applied_viewport_seq": int(viewport.seq),
        "fallback_used": bool(fallback_used),
    }


def _viewport_to_cmd_payload(viewport: DisplayViewport | None) -> dict[str, Any] | None:
    if viewport is None:
        return None
    return {
        "x_min_m": float(viewport.x_min_m),
        "x_max_m": float(viewport.x_max_m),
        "y_min_m": float(viewport.y_min_m),
        "y_max_m": float(viewport.y_max_m),
        "seq": int(viewport.seq),
    }


def _viewport_from_cmd_payload(
    payload: Any,
    *,
    home_viewport: DisplayViewport,
    output_width: int,
    output_height: int,
    dr_m: float,
) -> DisplayViewport | None:
    if payload is None:
        return None
    block = payload if isinstance(payload, dict) else {}
    if not block:
        return None
    try:
        x_min_m = float(block.get("x_min_m"))
        x_max_m = float(block.get("x_max_m"))
        y_min_m = float(block.get("y_min_m"))
        y_max_m = float(block.get("y_max_m"))
    except Exception:
        return None
    seq = int(block.get("seq", 0))
    return clamp_display_viewport(
        x_min_m=float(x_min_m),
        x_max_m=float(x_max_m),
        y_min_m=float(y_min_m),
        y_max_m=float(y_max_m),
        home_viewport=home_viewport,
        output_width=int(output_width),
        output_height=int(output_height),
        dr_m=float(dr_m),
        seq=int(seq),
    )


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


def _read_default_avg_mode(offline_config_path: str | Path) -> AvgMode:
    cfg = _load_yaml_file(Path(offline_config_path))
    bp_cfg = cfg.get("bp", {}) or {}
    raw = _pick(bp_cfg.get("avg_mode"), "both")
    return _avg_mode_normalize(None if raw is None else str(raw))


def _read_default_motion_mode(offline_config_path: str | Path) -> MotionMode:
    cfg = _load_yaml_file(Path(offline_config_path))
    bp_cfg = cfg.get("bp", {}) or {}
    raw_motion = bp_cfg.get("motion_mode", None)
    if raw_motion is not None:
        return _motion_mode_normalize(str(raw_motion))
    return "static_zero_doppler"


def _read_fft_workers(fallback_capture_cfg: str | Path) -> int:
    cfg = _load_yaml_file(Path(fallback_capture_cfg))
    fft_cfg = cfg.get("fft", {}) or {}
    raw = _pick(fft_cfg.get("workers"), 1)
    try:
        workers = int(raw)
    except (TypeError, ValueError):
        workers = 1
    return max(1, workers)


def _build_window_1d(window_type: str, size: int) -> np.ndarray:
    window = str(window_type).strip().lower()
    size_i = max(1, int(size))
    if window in {"none", "rectangular"}:
        return np.ones(size_i, dtype=np.float32)
    if window == "hanning":
        return np.hanning(size_i).astype(np.float32, copy=False)
    if window == "hamming":
        return np.hamming(size_i).astype(np.float32, copy=False)
    if window == "blackman":
        return np.blackman(size_i).astype(np.float32, copy=False)
    raise ValueError(f"window type non valido: {window_type!r}")


def _offline_sar_range_angle_filters_enabled(cfg: OfflineSyntheticRangeAngleConfig) -> tuple[str, ...]:
    enabled: list[str] = []
    if cfg.use_realtime_filters:
        enabled.append("range_window")
        if int(cfg.zero_after_range_fft_bins) > 0:
            enabled.append("zero_after_range_fft_bins")
        if cfg.post_range_fft_filters.mean_after_range_fft.enabled:
            enabled.append("mean_after_range_fft")
        if cfg.post_range_fft_filters.slow_time.enabled and cfg.post_range_fft_filters.slow_time.mode != "none":
            enabled.append(f"slow_time:{cfg.post_range_fft_filters.slow_time.mode}")
        if cfg.post_range_fft_filters.background_subtraction.enabled:
            enabled.append(f"background:{cfg.post_range_fft_filters.background_subtraction.mode}")
        if cfg.post_range_fft_filters.loop_average_after_background.enabled:
            enabled.append("loop_average_after_background")
        if str(cfg.window_doppler) not in {"none", "rectangular"}:
            enabled.append("window_doppler")
        if str(cfg.window_angle) not in {"none", "rectangular"}:
            enabled.append("window_angle")
    return tuple(enabled)


def _to_bool(field_name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, str):
        value_s = value.strip().lower()
        if value_s in {"1", "true", "yes", "y", "on"}:
            return True
        if value_s in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"{field_name} non valido: {value!r}")


def _to_float(field_name: str, value: Any, default: float | None = None) -> float:
    if value is None:
        if default is None:
            raise ValueError(f"{field_name} mancante")
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} non valido: {value!r}") from exc


def _parse_float_array(
    field_name: str,
    value: Any,
    *,
    expected_len: int,
) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(list(value), dtype=np.float32).reshape(-1)
    except Exception as exc:
        raise ValueError(f"{field_name} non valido: {value!r}") from exc
    if arr.size != int(expected_len):
        raise ValueError(f"{field_name} size={arr.size}, atteso {expected_len}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{field_name} contiene valori non finiti")
    return arr.astype(np.float32, copy=False)


def _resolve_wavelength_m(cfg: dict[str, Any]) -> float:
    radar = cfg.get("radar", {}) or {}
    c_m_s = _to_float("radar.c", _pick(radar.get("c"), 3e8), 3e8)
    fc_hz = _to_float("radar.fc", radar.get("fc"))
    if not np.isfinite(c_m_s) or c_m_s <= 0.0:
        raise ValueError("radar.c non valido")
    if not np.isfinite(fc_hz) or fc_hz <= 0.0:
        raise ValueError("radar.fc non valido")
    wavelength_m = float(c_m_s) / float(fc_hz)
    if not np.isfinite(wavelength_m) or wavelength_m <= 0.0:
        raise ValueError("wavelength_m non valida")
    return float(wavelength_m)


def _resolve_bp_offsets_m(
    bp_cfg: dict[str, Any],
    *,
    key_root: str,
    expected_len: int,
    wavelength_m: float,
) -> np.ndarray | None:
    offsets_m = _parse_float_array(
        f"offline_config: bp.{key_root}_m",
        bp_cfg.get(f"{key_root}_m", None),
        expected_len=int(expected_len),
    )
    if offsets_m is not None:
        return offsets_m.astype(np.float32, copy=False)

    offsets_lambda = _parse_float_array(
        f"offline_config: bp.{key_root}_lambda",
        bp_cfg.get(f"{key_root}_lambda", None),
        expected_len=int(expected_len),
    )
    if offsets_lambda is None:
        return None
    return (offsets_lambda * np.float32(float(wavelength_m))).astype(np.float32, copy=False)


def _append_deprecated_mimo_geometry_warnings(
    *,
    warnings: list[str],
    source_label: str,
    cfg: dict[str, Any],
) -> None:
    antenna_block = cfg.get("antenna", {}) or cfg.get("virtual_array", {}) or {}
    for key in (
        "virtual_array_order",
        "order",
        "virtual_array_phase_centers_m",
        "phase_centers_m",
        "virtual_array_phase_centers_lambda",
        "phase_centers_lambda",
    ):
        if key in antenna_block:
            warnings.append(
                f"[OFFLINE WARN] {source_label}: antenna.{key} is deprecated/ignored in mimo_sar; "
                "offline MIMO-SAR uses physical bistatic TX/RX geometry."
            )


def _read_offline_sar_range_angle_cfg(
    cfg: dict[str, Any],
    fallback_cfg: dict[str, Any],
) -> OfflineSyntheticRangeAngleConfig:
    branch = cfg.get("offline_sar_range_angle", {}) or {}
    fallback_dsp = fallback_cfg.get("dsp", {}) or {}
    fallback_display_filters = fallback_dsp.get("display_filters", {}) or {}
    fallback_display = fallback_cfg.get("display", {}) or {}

    use_realtime_filters = bool(branch.get("use_realtime_filters", True))
    window_range = window_type_normalize(
        _pick(branch.get("window_range"), fallback_dsp.get("window_range"), "blackman"),
        "blackman",
    )
    window_doppler = window_type_normalize(
        _pick(branch.get("window_doppler"), fallback_dsp.get("window_doppler"), "hanning"),
        "hanning",
    )
    window_angle = window_type_normalize(
        _pick(branch.get("window_angle"), fallback_dsp.get("window_angle"), "hanning"),
        "hanning",
    )
    zero_after_raw = _pick(
        branch.get("zero_after_range_fft_bins"),
        fallback_dsp.get("zero_after_range_fft_bins"),
        0,
    )
    try:
        zero_after_range_fft_bins = int(zero_after_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"offline_config: offline_sar_range_angle.zero_after_range_fft_bins non valido: {zero_after_raw!r}"
        ) from exc
    zero_after_range_fft_bins = max(0, zero_after_range_fft_bins)

    filters_cfg = display_post_range_fft_filters_from_yaml_dict(
        {
            "dsp": {
                "display_filters": {
                    "mean_after_range_fft": _pick(
                        branch.get("mean_after_range_fft"),
                        fallback_display_filters.get("mean_after_range_fft"),
                        fallback_dsp.get("mean_after_range_fft", {}),
                    )
                    or {},
                    "slow_time": _pick(
                        branch.get("slow_time"),
                        fallback_display_filters.get("slow_time"),
                        fallback_dsp.get("slow_time", {}),
                    )
                    or {},
                    "background_subtraction": _pick(
                        branch.get("background_subtraction"),
                        fallback_display_filters.get("background_subtraction"),
                        fallback_dsp.get("background_subtraction", {}),
                    )
                    or {},
                    "loop_average_after_background": _pick(
                        branch.get("loop_average_after_background"),
                        fallback_display_filters.get("loop_average_after_background"),
                        fallback_dsp.get("loop_average_after_background", {}),
                    )
                    or {},
                }
            }
        }
    )
    filters_cfg, filter_warnings = sanitize_display_post_range_fft_filters(filters_cfg)
    if filters_cfg.slow_time.enabled and filters_cfg.slow_time.mode == "doppler_fft":
        filter_warnings.append(
            "offline_sar_range_angle.slow_time.mode=doppler_fft conflicts with the dedicated zero-Doppler "
            "TDM-MIMO reconstruction step; forcing slow_time off."
        )
        filters_cfg = replace(
            filters_cfg,
            slow_time=replace(filters_cfg.slow_time, enabled=False, mode="none", doppler_zero_notch=False),
        )

    angle_processing = angle_processing_from_yaml_dict(
        {
            "dsp": {
                "angle_processing": _pick(
                    branch.get("angle_processing"),
                    fallback_dsp.get("angle_processing", {}),
                )
                or {}
            }
        }
    )
    # Keep the global setting as a fallback, but allow offline tuning without
    # changing the realtime angular FFT resolution.
    nfft_angle_raw = _pick(branch.get("nfft_angle"), (fallback_cfg.get("fft", {}) or {}).get("nfft_angle"), 256)
    try:
        nfft_angle = max(1, int(nfft_angle_raw))
    except (TypeError, ValueError):
        nfft_angle = 256
    projection = display_projection_from_yaml_dict(
        {
            "display": {
                **fallback_display,
                **(cfg.get("display", {}) or {}),
            }
        }
    )
    return OfflineSyntheticRangeAngleConfig(
        use_realtime_filters=bool(use_realtime_filters),
        window_range=str(window_range),
        window_doppler=str(window_doppler),
        window_angle=str(window_angle),
        zero_after_range_fft_bins=int(zero_after_range_fft_bins),
        post_range_fft_filters=filters_cfg,
        angle_processing=angle_processing,
        nfft_angle=int(nfft_angle),
        projection=projection,
        filter_warnings=tuple(str(w) for w in filter_warnings),
    )


def _read_bp_runtime_cfg(offline_config_path: str | Path, fallback_capture_cfg: str | Path) -> dict[str, Any]:
    cfg = _load_yaml_file(Path(offline_config_path))
    fallback_cfg = _load_yaml_file(Path(fallback_capture_cfg))
    reconstruction_cfg = cfg.get("reconstruction", {}) or {}
    bp_cfg = cfg.get("bp", {}) or {}
    warnings: list[str] = []
    algorithm = _reconstruction_algorithm_normalize(_pick(reconstruction_cfg.get("algorithm"), "backprojection"))

    mode: BpMode = _bp_mode_normalize(_pick(bp_cfg.get("mode"), "sar_only"))
    if algorithm == "synthetic_range_angle" and mode != "mimo_sar":
        raise ValueError("reconstruction.algorithm=synthetic_range_angle richiede bp.mode=mimo_sar")
    if mode == "mimo_sar":
        _append_deprecated_mimo_geometry_warnings(
            warnings=warnings,
            source_label="offline_config",
            cfg=cfg,
        )
        _append_deprecated_mimo_geometry_warnings(
            warnings=warnings,
            source_label="capture_config",
            cfg=fallback_cfg,
        )
    coherent_sum = _to_bool(
        "offline_config: bp.coherent_sum",
        _pick(bp_cfg.get("coherent_sum"), True),
    )
    avg_mode: AvgMode | None = None
    motion_mode: MotionMode = "static_zero_doppler"
    if mode == "mimo_sar":
        motion_mode = _motion_mode_normalize(_pick(bp_cfg.get("motion_mode"), "static_zero_doppler"))
    else:
        avg_mode = _avg_mode_normalize(_pick(bp_cfg.get("avg_mode"), "both"))

    wavelength_m = _resolve_wavelength_m(fallback_cfg)
    tx_offsets_m = _resolve_bp_offsets_m(
        bp_cfg,
        key_root="tx_offsets",
        expected_len=max(1, int(_pick(cfg.get("capture", {}).get("tx", None), fallback_cfg.get("capture", {}).get("tx", 2)))),
        wavelength_m=float(wavelength_m),
    )
    rx_offsets_m = _resolve_bp_offsets_m(
        bp_cfg,
        key_root="rx_offsets",
        expected_len=max(1, int(_pick(cfg.get("capture", {}).get("rx", None), fallback_cfg.get("capture", {}).get("rx", 4)))),
        wavelength_m=float(wavelength_m),
    )

    runtime_cfg = {
        "mode": mode,
        "motion_mode": motion_mode,
        "coherent_sum": bool(coherent_sum),
        "tx_offsets_m": None if tx_offsets_m is None else tx_offsets_m.astype(np.float32, copy=False),
        "rx_offsets_m": None if rx_offsets_m is None else rx_offsets_m.astype(np.float32, copy=False),
        "warnings": warnings,
    }
    if avg_mode is not None:
        runtime_cfg["avg_mode"] = avg_mode
    runtime_cfg["algorithm"] = str(algorithm)
    runtime_cfg["range_angle"] = _read_offline_sar_range_angle_cfg(cfg, fallback_cfg)
    return runtime_cfg


def _resolve_effective_avg_mode(
    requested_avg_mode: AvgMode,
    *,
    bp_mode: BpMode,
) -> tuple[AvgMode, str | None]:
    if bp_mode == "sar_only":
        if requested_avg_mode != "both":
            return (
                "both",
                f"[OFFLINE WARN] avg_mode={requested_avg_mode} ignored because bp.mode=sar_only collapses snapshots inside BP; forcing avg_mode=both to match current output.",
            )
        return "both", None
    return requested_avg_mode, None


def _resolve_offline_mimo_geometry(
    fallback_capture_cfg: str | Path,
    *,
    tx_i: int,
    rx_i: int,
    tx_offsets_override_m: np.ndarray | None,
    rx_offsets_override_m: np.ndarray | None,
) -> dict[str, Any]:
    if tx_i <= 0 or rx_i <= 0:
        raise ValueError(f"tx/rx non validi per geometria mimo: tx={tx_i}, rx={rx_i}")

    fallback_cfg = _load_yaml_file(Path(fallback_capture_cfg))
    radar_cfg = fallback_cfg.get("radar", {}) or {}
    c_m_s = _to_float("radar.c", _pick(radar_cfg.get("c"), 3e8), 3e8)
    fc_hz = _to_float("radar.fc", radar_cfg.get("fc"))
    x_tx_by_ant, x_rx_by_ant = _build_mimo_geometry(
        int(tx_i),
        int(rx_i),
        fc_hz=float(fc_hz),
        c_m_s=float(c_m_s),
        tx_offsets_m=None if tx_offsets_override_m is None else np.asarray(tx_offsets_override_m, dtype=np.float32).reshape(-1),
        rx_offsets_m=None if rx_offsets_override_m is None else np.asarray(rx_offsets_override_m, dtype=np.float32).reshape(-1),
    )
    geometry_source = "config_offsets" if (tx_offsets_override_m is not None or rx_offsets_override_m is not None) else "default_iwr1443boost_azimuth"

    return {
        "x_tx_ant_m": x_tx_by_ant.astype(np.float32, copy=False),
        "x_rx_ant_m": x_rx_by_ant.astype(np.float32, copy=False),
        "geometry_source": geometry_source,
    }


def _apply_offline_sar_range_angle_pre_filters(
    raw_mimo: np.ndarray,
    *,
    filters_cfg: PostRangeFftFilterConfig,
    fft_workers: int,
) -> np.ndarray:
    if raw_mimo.ndim != 6:
        raise ValueError(f"raw_mimo shape non valido per pre-filters: {raw_mimo.shape!r}")
    if (
        (not filters_cfg.mean_after_range_fft.enabled or not filters_cfg.mean_after_range_fft.axes)
        and (not filters_cfg.slow_time.enabled or filters_cfg.slow_time.mode == "none")
    ):
        return np.asarray(raw_mimo, dtype=np.complex64, copy=False)

    out = np.array(raw_mimo, dtype=np.complex64, copy=True)
    n_pos = int(out.shape[0])
    for pos_i in range(n_pos):
        # Reuse the realtime helper axis convention: [frame, loop, tx, range_bin, rx].
        view = np.transpose(out[pos_i], (0, 1, 2, 4, 3)).astype(np.complex64, copy=False)
        view = apply_slow_time_filter(view, filters_cfg.slow_time, fft_workers=int(fft_workers))
        view = subtract_selected_mean(view, filters_cfg.mean_after_range_fft)
        out[pos_i] = np.transpose(view, (0, 1, 2, 4, 3)).astype(np.complex64, copy=False)
    return out.astype(np.complex64, copy=False)


def _apply_offline_sar_range_angle_background(
    snapshot_cube: np.ndarray,
    *,
    bg_cfg: BackgroundSubtractionConfig,
) -> np.ndarray:
    cube = np.asarray(snapshot_cube, dtype=np.complex64)
    if not bg_cfg.enabled or cube.ndim != 4 or int(cube.shape[0]) <= 0:
        return cube

    if str(bg_cfg.mode) != "frozen":
        state = BackgroundSubtractionState()
        return apply_background_subtraction(np.array(cube, dtype=np.complex64, copy=True), bg_cfg, state)

    warmup = min(max(1, int(bg_cfg.init_frames)), int(cube.shape[0]))
    model = cube[:warmup].mean(axis=0, dtype=np.complex64).astype(np.complex64, copy=False)
    if int(cube.shape[0]) <= warmup:
        return np.empty((0,) + tuple(int(v) for v in cube.shape[1:]), dtype=np.complex64)

    state = BackgroundSubtractionState(model=np.array(model, dtype=np.complex64, copy=True))
    return apply_background_subtraction(
        np.array(cube[warmup:], dtype=np.complex64, copy=True),
        bg_cfg,
        state,
    )


def _build_synthetic_virtual_array_geometry(
    x_element_m: np.ndarray,
    *,
    wavelength_m: float,
) -> tuple[VirtualArrayGeometry, bool]:
    x_element = np.asarray(x_element_m, dtype=np.float32).reshape(-1)
    if x_element.size <= 0:
        raise ValueError("x_element_m vuoto per synthetic aperture")
    wavelength = float(wavelength_m)
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError(f"wavelength_m non valida: {wavelength_m!r}")

    phase_centers_lambda = (x_element / np.float32(wavelength)).astype(np.float32, copy=False)
    diffs = np.diff(phase_centers_lambda.astype(np.float64, copy=False))
    finite_diffs = np.abs(diffs[np.isfinite(diffs)])
    base_spacing = float(np.min(finite_diffs)) if finite_diffs.size > 0 else 0.25
    if not np.isfinite(base_spacing) or base_spacing <= 1e-8:
        base_spacing = 0.25
    uniform_spacing = _synthetic_aperture_uniform_spacing_lambda(
        x_element,
        wavelength_m=float(wavelength),
    )
    fft_uniform = uniform_spacing is not None
    geometry = VirtualArrayGeometry(
        order_flat=np.arange(int(phase_centers_lambda.size), dtype=np.int32),
        phase_centers_lambda=phase_centers_lambda.astype(np.float32, copy=False),
        identity_order=True,
        uniform_half_lambda=bool(
            fft_uniform and np.isclose(abs(float(uniform_spacing)), 0.5, rtol=0.0, atol=1e-6)
        ),
        # Keep the actual element coordinates for steering and use the finest observed spacing
        # to define the steering/u grid even when the aperture is sparse.
        uniform_spacing_lambda=float(abs(float(uniform_spacing))) if fft_uniform else float(base_spacing),
        angle_axis_sign=1.0,
        angle_u_to_sin_scale=2.0,
    )
    return geometry, bool(fft_uniform)


def _resolve_synthetic_angle_processing(
    angle_cfg: AngleProcessingConfig,
    *,
    fft_uniform: bool,
) -> tuple[AngleProcessingConfig, str | None]:
    if angle_cfg.mode != "fft" or fft_uniform:
        return angle_cfg, None
    warning = (
        "[OFFLINE WARN] synthetic_range_angle requested angle_processing.mode=fft on a non-uniform synthetic "
        "aperture; falling back to bartlett."
    )
    return replace(angle_cfg, mode="bartlett"), warning


def _build_synthetic_angle_dsp_cfg(
    *,
    c_m_s: float,
    fs_hz: float,
    slope_hz_s: float,
    nfft_range: int,
    nfft_angle: int,
    range_max_m: float,
    synthetic_ant: int,
    fft_workers: int,
    frames_like: int,
) -> RealtimeDSPConfig:
    return RealtimeDSPConfig(
        c=float(c_m_s),
        fs=float(fs_hz),
        slope=float(slope_hz_s),
        samples=1,
        chirps=1,
        rx=max(1, int(synthetic_ant)),
        tx=1,
        x_frames=max(1, int(frames_like)),
        bytes_per_frame=max(1, int(synthetic_ant) * 8),
        nfft_range=max(1, int(nfft_range)),
        nfft_angle=max(1, int(nfft_angle)),
        range_max_display=float(range_max_m),
        range_profile_count=max(1, int(nfft_range)),
        virtual_ant=max(1, int(synthetic_ant)),
        fft_workers=max(1, int(fft_workers)),
        debug_stats=False,
    )


def _compute_synthetic_range_angle_image(
    range_fft_sel: np.ndarray,
    *,
    selected_positions: np.ndarray,
    x_pitch_m: float,
    tx_i: int,
    rx_i: int,
    x_tx_ant_m: np.ndarray,
    x_rx_ant_m: np.ndarray,
    range_angle_cfg: OfflineSyntheticRangeAngleConfig,
    c_m_s: float,
    fs_hz: float,
    slope_hz_s: float,
    fc_hz: float,
    fft_workers: int,
    nfft_range: int,
    gui_h: int,
    gui_w: int,
    viewport: DisplayViewport,
    projection_lut: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if range_fft_sel.ndim != 5:
        raise ValueError(
            "range_fft_sel shape non valido per synthetic_range_angle: "
            f"{range_fft_sel.shape!r}; atteso [pos, frame, loop, ant, bin]."
        )

    n_pos_sel, n_frames_sel, n_loops_sel, n_ant_sel, n_bins_sel = range_fft_sel.shape
    if x_tx_ant_m.size != int(n_ant_sel) or x_rx_ant_m.size != int(n_ant_sel):
        raise ValueError("x_tx_ant_m/x_rx_ant_m size != asse antenna synthetic_range_angle")
    if int(n_ant_sel) <= 0 or int(n_bins_sel) <= 0:
        return np.zeros((int(gui_h), int(gui_w)), dtype=np.float32), {
            "synthetic_antennas": 0,
            "angle_mode_requested": str(range_angle_cfg.angle_processing.mode),
            "angle_mode": str(range_angle_cfg.angle_processing.mode),
            "fft_uniform_geometry": False,
            "enabled_filters": _offline_sar_range_angle_filters_enabled(range_angle_cfg),
        }
    if int(tx_i) <= 0 or int(rx_i) <= 0 or int(tx_i) * int(rx_i) != int(n_ant_sel):
        raise ValueError(
            f"tx/rx non coerente con asse antenna synthetic_range_angle: tx={tx_i}, rx={rx_i}, ant={n_ant_sel}"
        )

    raw_mimo = range_fft_sel.reshape(
        int(n_pos_sel),
        int(n_frames_sel),
        int(n_loops_sel),
        int(tx_i),
        int(rx_i),
        int(n_bins_sel),
    ).astype(np.complex64, copy=False)
    if range_angle_cfg.use_realtime_filters:
        filtered_mimo = _apply_offline_sar_range_angle_pre_filters(
            raw_mimo,
            filters_cfg=range_angle_cfg.post_range_fft_filters,
            fft_workers=int(fft_workers),
        )
        doppler_window = _build_window_1d(str(range_angle_cfg.window_doppler), int(n_loops_sel))
    else:
        filtered_mimo = raw_mimo
        doppler_window = None

    zero_doppler = _prepare_mimo_snapshots(
        filtered_mimo,
        n_tx=int(tx_i),
        motion_mode="static_zero_doppler",
        window_doppler=doppler_window,
    )
    synthetic = _prepare_synthetic_aperture_data(
        zero_doppler,
        selected_positions=np.asarray(selected_positions, dtype=np.int32).reshape(-1),
        x_pitch_m=float(x_pitch_m),
        x_tx_ant_m=x_tx_ant_m,
        x_rx_ant_m=x_rx_ant_m,
    )
    geometry, fft_uniform = _build_synthetic_virtual_array_geometry(
        synthetic.x_element_m,
        wavelength_m=float(c_m_s) / float(fc_hz),
    )
    angle_cfg_eff, angle_warning = _resolve_synthetic_angle_processing(
        range_angle_cfg.angle_processing,
        fft_uniform=fft_uniform,
    )
    if angle_warning:
        print(angle_warning)

    synthetic_cube = np.array(synthetic.snapshot_cube, dtype=np.complex64, copy=True)
    if range_angle_cfg.use_realtime_filters:
        synthetic_cube = _apply_offline_sar_range_angle_background(
            synthetic_cube,
            bg_cfg=range_angle_cfg.post_range_fft_filters.background_subtraction,
        )
        if range_angle_cfg.post_range_fft_filters.loop_average_after_background.enabled and synthetic_cube.ndim == 4:
            synthetic_cube = synthetic_cube.mean(axis=1, keepdims=True, dtype=np.complex64)
        if synthetic_cube.size > 0:
            w_angle = _build_window_1d(str(range_angle_cfg.window_angle), int(synthetic_cube.shape[-1]))
            synthetic_cube *= w_angle.reshape(1, 1, 1, int(synthetic_cube.shape[-1])).astype(
                np.complex64,
                copy=False,
            )

    synthetic_ant = int(synthetic.x_element_m.size)
    dsp_cfg = _build_synthetic_angle_dsp_cfg(
        c_m_s=float(c_m_s),
        fs_hz=float(fs_hz),
        slope_hz_s=float(slope_hz_s),
        nfft_range=int(nfft_range),
        nfft_angle=int(range_angle_cfg.nfft_angle),
        range_max_m=float(viewport.y_max_m),
        synthetic_ant=int(synthetic_ant),
        fft_workers=int(fft_workers),
        frames_like=int(max(1, synthetic_cube.shape[0] if synthetic_cube.ndim == 4 else 1)),
    )
    angle_axis = build_angle_axis_deg(int(range_angle_cfg.nfft_angle), geometry=geometry)
    angle_steering = build_angle_steering_matrix(
        int(synthetic_ant),
        int(range_angle_cfg.nfft_angle),
        geometry=geometry,
    )

    if synthetic_cube.ndim != 4 or int(synthetic_cube.shape[0]) <= 0:
        heatmap_lin = np.zeros((int(n_bins_sel), int(range_angle_cfg.nfft_angle)), dtype=np.float32)
    else:
        heatmap_lin = compute_angle_heatmap(
            synthetic_cube,
            angle_cfg=angle_cfg_eff,
            dsp_cfg=dsp_cfg,
            angle_steering=angle_steering,
            geometry=geometry,
            ant_spacing=None,
        ).astype(np.float32, copy=False)
        if heatmap_lin.ndim != 2:
            heatmap_lin = np.zeros((int(n_bins_sel), int(range_angle_cfg.nfft_angle)), dtype=np.float32)

    projection_mode = "cartesian"
    projection_interp = str(range_angle_cfg.projection.projection_interp)
    lut = projection_lut
    if lut is None:
        lut = build_display_projection_lut(
            gui_h=int(gui_h),
            gui_w=int(gui_w),
            x_max_m=float(viewport.x_max_m),
            y_max_m=float(viewport.y_max_m),
            dr_m=float(c_m_s) * float(fs_hz) / (2.0 * float(slope_hz_s) * float(nfft_range)),
            angle_axis_deg=angle_axis,
            projection_mode=projection_mode,
            projection_interp=projection_interp,
            x_min_m=float(viewport.x_min_m),
            y_min_m=float(viewport.y_min_m),
        )
    projected_lin = project_heatmap_for_display(
        heatmap_lin,
        angle_axis_deg=angle_axis,
        dr_m=float(c_m_s) * float(fs_hz) / (2.0 * float(slope_hz_s) * float(nfft_range)),
        gui_h=int(gui_h),
        gui_w=int(gui_w),
        y_max_m=float(viewport.y_max_m),
        x_max_m=float(viewport.x_max_m),
        projection_mode=projection_mode,
        projection_interp=projection_interp,
        precomputed_lut=lut,
        x_min_m=float(viewport.x_min_m),
        y_min_m=float(viewport.y_min_m),
    )
    img_db = _power_image_to_db(projected_lin)
    meta = {
        "synthetic_antennas": int(synthetic_ant),
        "angle_mode_requested": str(range_angle_cfg.angle_processing.mode),
        "angle_mode": str(angle_cfg_eff.mode),
        "fft_uniform_geometry": bool(fft_uniform),
        "enabled_filters": _offline_sar_range_angle_filters_enabled(range_angle_cfg),
        "selected_positions": tuple(int(v) for v in np.asarray(selected_positions, dtype=np.int32).reshape(-1).tolist()),
        "projection_lut": lut,
    }
    return img_db.astype(np.float32, copy=False), meta


# ---------------------------------------------------------------------
# Offline Multiprocess Pipeline
# - reader process: load + range-FFT prep
# - dsp process: avg/back-projection + publish frame
# ---------------------------------------------------------------------
def _offline_reader_worker(
    offline_config_path: str,
    fallback_capture_cfg: str,
    nfft_range: int,
    reader_to_dsp_q: Queue,
    status_q: Queue,
    stop_evt,
) -> None:
    shm_range_fft = None
    shm_cleanup_transferred = False
    try:
        reader = SARReader(offline_config_path=offline_config_path, fallback_capture_cfg=fallback_capture_cfg)
        bp_runtime_cfg = _read_bp_runtime_cfg(offline_config_path, fallback_capture_cfg)
        fft_workers = _read_fft_workers(fallback_capture_cfg)
        algorithm = str(bp_runtime_cfg.get("algorithm", "backprojection"))
        bp_mode: BpMode = bp_runtime_cfg["mode"]
        range_angle_cfg: OfflineSyntheticRangeAngleConfig = bp_runtime_cfg["range_angle"]
        data = reader.load(keep_raw=False)

        mimo_geometry = None
        if bp_mode == "mimo_sar":
            mimo_geometry = _resolve_offline_mimo_geometry(
                fallback_capture_cfg,
                tx_i=int(data.tx),
                rx_i=int(data.rx),
                tx_offsets_override_m=bp_runtime_cfg.get("tx_offsets_m"),
                rx_offsets_override_m=bp_runtime_cfg.get("rx_offsets_m"),
            )
            warning = mimo_geometry.get("warning", None)
            if warning:
                print(str(warning))

        for warning in bp_runtime_cfg.get("warnings", []):
            print(str(warning))
        for warning in range_angle_cfg.filter_warnings:
            print(f"[OFFLINE WARN] {warning}")

        if bp_mode == "mimo_sar":
            # MIMO-SAR: preserva asse antenna [pos, frame, loop, ant, sample].
            sig = data.iq_cube
            if algorithm == "synthetic_range_angle" and range_angle_cfg.use_realtime_filters:
                sig = np.array(sig, dtype=np.complex64, copy=True)
                w_range = _build_window_1d(str(range_angle_cfg.window_range), int(data.samples))
                sig *= w_range.reshape(1, 1, 1, 1, int(data.samples)).astype(np.complex64, copy=False)
            range_fft = fft.fft(sig, n=int(nfft_range), axis=-1, workers=fft_workers).astype(np.complex64, copy=False)
        else:
            # Legacy SAR-only: media preliminare sulle antenne virtuali.
            sig = data.iq_cube.mean(axis=3, dtype=np.complex64)  # [pos, frame, loop, sample]
            range_fft = fft.fft(sig, n=int(nfft_range), axis=-1, workers=fft_workers).astype(np.complex64, copy=False)
        if algorithm == "synthetic_range_angle" and range_angle_cfg.use_realtime_filters:
            zero_bins = min(int(range_angle_cfg.zero_after_range_fft_bins), int(range_fft.shape[-1]))
            if zero_bins > 0:
                range_fft[..., :zero_bins] = np.complex64(0.0)
        range_fft = np.ascontiguousarray(range_fft, dtype=np.complex64)
        shm_range_fft = shared_memory.SharedMemory(create=True, size=int(range_fft.nbytes))
        shm_arr = np.ndarray(range_fft.shape, dtype=np.complex64, buffer=shm_range_fft.buf)
        shm_arr[:] = range_fft

        msg = {
            "type": "data",
            "range_fft_shm_name": str(shm_range_fft.name),
            "range_fft_shape": tuple(int(x) for x in range_fft.shape),
            "range_fft_dtype": "complex64",
            "positions": data.positions.astype(np.int32, copy=False),
            "x_start_cfg": int(reader.config.x_start),
            "x_end_cfg": int(reader.config.x_end),
            "algorithm": str(algorithm),
            "bp_mode": str(bp_mode),
            "bp_motion_mode": str(bp_runtime_cfg["motion_mode"]),
            "bp_coherent_sum": bool(bp_runtime_cfg["coherent_sum"]),
            "range_angle_cfg": range_angle_cfg,
            "tx": int(data.tx),
            "rx": int(data.rx),
        }
        if mimo_geometry is not None:
            msg["bp_x_tx_ant_m"] = np.asarray(mimo_geometry["x_tx_ant_m"], dtype=np.float32)
            msg["bp_x_rx_ant_m"] = np.asarray(mimo_geometry["x_rx_ant_m"], dtype=np.float32)
            msg["bp_geometry_source"] = str(mimo_geometry["geometry_source"])
        _queue_put_latest(reader_to_dsp_q, msg)
        shm_cleanup_transferred = True
        _queue_put_latest(
            status_q,
            {
                "type": "reader_ready",
                "positions": int(data.positions.size),
                "frames_per_pos": int(data.n_frames_per_position),
                "algorithm": str(algorithm),
                "bp_mode": str(bp_mode),
                "motion_mode": str(bp_runtime_cfg["motion_mode"]),
                "range_angle_enabled_filters": _offline_sar_range_angle_filters_enabled(range_angle_cfg),
                "range_angle_use_realtime_filters": bool(range_angle_cfg.use_realtime_filters),
                "range_angle_angle_mode": str(range_angle_cfg.angle_processing.mode),
                "range_angle_nfft_angle": int(range_angle_cfg.nfft_angle),
                "geometry_source": None if mimo_geometry is None else str(mimo_geometry["geometry_source"]),
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
        if shm_range_fft is not None:
            if not shm_cleanup_transferred:
                try:
                    shm_range_fft.unlink()
                except Exception:
                    pass
            try:
                shm_range_fft.close()
            except Exception:
                pass
        stop_evt.wait(0.01)


def _range_fft_from_init_msg(init_msg: dict[str, Any]) -> tuple[np.ndarray, shared_memory.SharedMemory | None]:
    shm_name = init_msg.get("range_fft_shm_name")
    if shm_name is not None:
        shape_raw = init_msg.get("range_fft_shape")
        if not isinstance(shape_raw, (tuple, list)):
            raise ValueError("range_fft_shape mancante o non valido nel messaggio init")
        shape = tuple(int(x) for x in shape_raw)
        if len(shape) not in (4, 5):
            raise ValueError(f"range_fft_shape non valido: {shape!r}")
        dtype_s = str(init_msg.get("range_fft_dtype", "complex64")).strip().lower()
        if dtype_s != "complex64":
            raise ValueError(f"range_fft_dtype non supportato: {dtype_s!r}")
        shm_obj = shared_memory.SharedMemory(name=str(shm_name))
        arr = np.ndarray(shape, dtype=np.complex64, buffer=shm_obj.buf)
        return arr, shm_obj

    # Backward-compatible fallback for legacy queue payloads.
    arr = np.asarray(init_msg["range_fft"], dtype=np.complex64)
    if arr.ndim not in (4, 5):
        raise ValueError(f"range_fft shape non valido: {arr.shape!r}")
    return arr, None


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
    fft_workers: int,
    nfft_range: int,
    range_max_m: float,
    crossrange_max_m: float,
    x_pitch_m: float,
    default_avg_mode: str,
    default_motion_mode: str,
    phase_sign: int,
) -> None:
    shm_range_fft = None
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

        range_fft_data, shm_range_fft = _range_fft_from_init_msg(init_msg)
        positions = np.asarray(init_msg["positions"], dtype=np.int32)
        algorithm = _reconstruction_algorithm_normalize(_pick(init_msg.get("algorithm"), "backprojection"))
        bp_mode: BpMode = _bp_mode_normalize(_pick(init_msg.get("bp_mode"), "sar_only"))
        if ("bp_mode" not in init_msg) and range_fft_data.ndim == 5:
            bp_mode = "mimo_sar"
        range_angle_cfg_raw = init_msg.get("range_angle_cfg")
        if isinstance(range_angle_cfg_raw, OfflineSyntheticRangeAngleConfig):
            range_angle_cfg = range_angle_cfg_raw
        else:
            range_angle_cfg = OfflineSyntheticRangeAngleConfig(
                use_realtime_filters=False,
                window_range="blackman",
                window_doppler="hanning",
                window_angle="hanning",
                zero_after_range_fft_bins=0,
                post_range_fft_filters=PostRangeFftFilterConfig(),
                angle_processing=AngleProcessingConfig(),
                nfft_angle=256,
                projection=DisplayProjectionConfig(),
                filter_warnings=(),
            )
        if range_fft_data.ndim not in (4, 5):
            raise ValueError(f"range_fft shape non valido: {range_fft_data.shape!r}")
        if bp_mode == "sar_only" and range_fft_data.ndim != 4:
            raise ValueError(f"range_fft shape non coerente con bp.mode=sar_only: {range_fft_data.shape!r}")
        if bp_mode == "mimo_sar" and range_fft_data.ndim != 5:
            raise ValueError(f"range_fft shape non coerente con bp.mode=mimo_sar: {range_fft_data.shape!r}")
        if positions.ndim != 1 or positions.size != range_fft_data.shape[0]:
            raise ValueError("positions non coerente con range_fft")
        if algorithm == "synthetic_range_angle" and bp_mode != "mimo_sar":
            raise ValueError("synthetic_range_angle richiede range_fft MIMO-SAR a 5 dimensioni")

        coherent_sum = _to_bool(
            "init_msg: bp_coherent_sum",
            _pick(init_msg.get("bp_coherent_sum"), True),
        )
        motion_mode_requested: MotionMode = _motion_mode_normalize(default_motion_mode)

        tx_i = max(1, int(_pick(init_msg.get("tx"), 2)))
        rx_i = max(1, int(_pick(init_msg.get("rx"), 4)))
        x_tx_ant_m = np.zeros(1, dtype=np.float32)
        x_rx_ant_m = np.zeros(1, dtype=np.float32)
        geometry_source = "sar_only"
        n_ant_used = 1
        if bp_mode == "mimo_sar":
            x_tx_ant_m = np.asarray(init_msg.get("bp_x_tx_ant_m"), dtype=np.float32).reshape(-1)
            x_rx_ant_m = np.asarray(init_msg.get("bp_x_rx_ant_m"), dtype=np.float32).reshape(-1)
            geometry_source = str(_pick(init_msg.get("bp_geometry_source"), "unknown"))
            n_ant_data = int(range_fft_data.shape[3])
            if x_tx_ant_m.size != n_ant_data or x_rx_ant_m.size != n_ant_data:
                raise ValueError("bp_x_tx_ant_m/bp_x_rx_ant_m size != asse antenna range_fft")
            n_ant_used = int(n_ant_data)
            motion_mode_requested = _motion_mode_normalize(_pick(init_msg.get("bp_motion_mode"), default_motion_mode))

        n_bins_total = int(range_fft_data.shape[-1])
        dr_m = float(c_m_s) * float(fs_hz) / (2.0 * float(slope_hz_s) * float(nfft_range))
        max_bin = int(np.floor(float(range_max_m) / dr_m))
        max_bin = max(1, min(max_bin, int(n_bins_total)))
        home_viewport = build_display_viewport(
            x_min_m=-float(crossrange_max_m),
            x_max_m=float(crossrange_max_m),
            y_min_m=0.0,
            y_max_m=float(range_max_m),
            dr_m=float(dr_m),
            seq=0,
        )
        applied_viewport = home_viewport

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
        avg_mode_requested: AvgMode = _avg_mode_normalize(default_avg_mode)
        avg_mode, avg_mode_warning = _resolve_effective_avg_mode(avg_mode_requested, bp_mode=bp_mode)
        if avg_mode_warning:
            print(avg_mode_warning)
        motion_mode = motion_mode_requested

        _queue_put_latest(
            status_q,
            {
                "type": "ready",
                "algorithm": str(algorithm),
                "pos_min": pos_min,
                "pos_max": pos_max,
                "x_start": x_start,
                "x_end": x_end,
                "avg_mode": avg_mode,
                "avg_mode_requested": str(avg_mode_requested),
                "motion_mode": str(motion_mode),
                "phase_sign": int(phase_sign_i),
                "img_h": int(gui_h),
                "img_w": int(gui_w),
                "dr_m": float(dr_m),
                "bp_mode": str(bp_mode),
                "virtual_antennas": int(n_ant_used),
                "geometry_source": str(geometry_source),
                "doppler_bins_used": 1,
                "range_angle_use_realtime_filters": bool(range_angle_cfg.use_realtime_filters),
                "range_angle_enabled_filters": _offline_sar_range_angle_filters_enabled(range_angle_cfg),
                "range_angle_angle_mode_requested": str(range_angle_cfg.angle_processing.mode),
                "range_angle_nfft_angle": int(range_angle_cfg.nfft_angle),
                **_viewport_status_fields(applied_viewport, fallback_used=False),
            },
        )
        print(
            f"[OFFLINE INFO] algorithm={algorithm} bp_mode={bp_mode} "
            f"default_positions={x_start}:{x_end} angle_mode={range_angle_cfg.angle_processing.mode} "
            f"filters={_offline_sar_range_angle_filters_enabled(range_angle_cfg)}"
        )

        last_job_key = None
        dirty = True
        prepared_cache_key = None
        prepared_cache = None
        bp_plan_cache_key = None
        bp_plan_cache = None
        grid_cache_key = None
        grid_cache = None
        synthetic_projection_lut_cache_key = None
        synthetic_projection_lut_cache = None

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
                motion_mode_new = cmd.get("motion_mode", None)
                avg_mode_new = cmd.get("avg_mode", None)
                if motion_mode_new is not None:
                    motion_mode_requested = _motion_mode_normalize(str(motion_mode_new))
                avg_mode_new = cmd.get("avg_mode", None)
                if avg_mode_new is not None and bp_mode == "sar_only":
                    avg_mode_requested = _avg_mode_normalize(str(avg_mode_new))
                    avg_mode, avg_mode_warning = _resolve_effective_avg_mode(avg_mode_requested, bp_mode=bp_mode)
                    if avg_mode_warning:
                        print(avg_mode_warning)
                viewport_new = _viewport_from_cmd_payload(
                    cmd.get("viewport"),
                    home_viewport=home_viewport,
                    output_width=int(gui_w),
                    output_height=int(gui_h),
                    dr_m=float(dr_m),
                )
                if viewport_new is not None:
                    applied_viewport = viewport_new
                motion_mode = motion_mode_requested
                dirty = True

            if stop_evt.is_set():
                break

            job_key = (
                str(algorithm),
                int(x_start),
                int(x_end),
                str(avg_mode) if bp_mode == "sar_only" else str(motion_mode),
                bool(coherent_sum),
                display_viewport_signature(applied_viewport),
            )
            if dirty or job_key != last_job_key or got_cmd:
                t0 = time.perf_counter()
                sel_mask = (positions >= int(x_start)) & (positions <= int(x_end))
                if not np.any(sel_mask):
                    sel_mask[:] = True
                sel_idx = np.where(sel_mask)[0]
                sel_key = tuple(int(v) for v in sel_idx.tolist())
                frame_meta: dict[str, Any] = {}
                if algorithm == "synthetic_range_angle":
                    selected_positions = positions[sel_idx].astype(np.int32, copy=False)
                    range_fft_sel = range_fft_data[sel_idx, :, :, :, :max_bin]
                    n_pos_sel, _n_frames_sel, _n_loops_sel, n_ant_sel, _n_bins_sel = range_fft_sel.shape
                    if int(n_ant_sel) != int(tx_i) * int(rx_i):
                        raise ValueError(
                            f"asse antenna synthetic={n_ant_sel} non coerente con tx/rx={tx_i}/{rx_i}"
                        )
                    synthetic_projection_lut_key = (
                        str(algorithm),
                        sel_key,
                        int(range_angle_cfg.nfft_angle),
                        str(range_angle_cfg.projection.projection_interp),
                        display_viewport_signature(applied_viewport),
                    )
                    if (
                        synthetic_projection_lut_cache_key == synthetic_projection_lut_key
                        and synthetic_projection_lut_cache is not None
                    ):
                        projection_lut = synthetic_projection_lut_cache
                    else:
                        projection_lut = None
                    img_db, frame_meta = _compute_synthetic_range_angle_image(
                        range_fft_sel,
                        selected_positions=selected_positions,
                        x_pitch_m=float(x_pitch_m),
                        tx_i=int(tx_i),
                        rx_i=int(rx_i),
                        x_tx_ant_m=x_tx_ant_m,
                        x_rx_ant_m=x_rx_ant_m,
                        range_angle_cfg=range_angle_cfg,
                        c_m_s=float(c_m_s),
                        fs_hz=float(fs_hz),
                        slope_hz_s=float(slope_hz_s),
                        fc_hz=float(fc_hz),
                        fft_workers=int(fft_workers),
                        nfft_range=int(nfft_range),
                        gui_h=int(gui_h),
                        gui_w=int(gui_w),
                        viewport=applied_viewport,
                        projection_lut=projection_lut,
                    )
                    synthetic_projection_lut_cache = frame_meta.pop("projection_lut", None)
                    synthetic_projection_lut_cache_key = synthetic_projection_lut_key
                    doppler_bins_used = 1
                else:
                    grid_key = display_viewport_signature(applied_viewport)
                    if grid_cache_key == grid_key and grid_cache is not None:
                        x_grid, y_grid = grid_cache
                    else:
                        x_axis = np.linspace(
                            float(applied_viewport.x_min_m),
                            float(applied_viewport.x_max_m),
                            int(gui_w),
                            dtype=np.float32,
                        )
                        y_axis = np.linspace(
                            float(applied_viewport.y_min_m),
                            float(applied_viewport.y_max_m),
                            int(gui_h),
                            dtype=np.float32,
                        )
                        x_grid, y_grid = np.meshgrid(x_axis, y_axis)
                        grid_cache_key = grid_key
                        grid_cache = (x_grid, y_grid)

                    x_pos_m_sel = x_pos_m_full[sel_idx]
                    if bp_mode == "mimo_sar":
                        range_fft_sel = range_fft_data[sel_idx, :, :, :, :max_bin]
                        n_pos_sel, n_frames_sel, n_loops_sel, n_ant_sel, n_bins_sel = range_fft_sel.shape
                        if int(n_ant_sel) != int(tx_i) * int(rx_i):
                            raise ValueError(
                                f"asse antenna mimo={n_ant_sel} non coerente con tx/rx={tx_i}/{rx_i}"
                            )
                        prepared_key = (
                            sel_key,
                            str(motion_mode),
                            int(max_bin),
                            int(tx_i),
                            int(rx_i),
                        )
                        if prepared_cache_key == prepared_key and prepared_cache is not None:
                            prepared = prepared_cache
                        else:
                            raw_mimo = range_fft_sel.reshape(
                                int(n_pos_sel),
                                int(n_frames_sel),
                                int(n_loops_sel),
                                int(tx_i),
                                int(rx_i),
                                int(n_bins_sel),
                            ).astype(np.complex64, copy=False)
                            prepared = _prepare_mimo_snapshots(
                                raw_mimo,
                                n_tx=int(tx_i),
                                motion_mode=motion_mode,
                                window_doppler=None,
                            )
                            prepared_cache_key = prepared_key
                            prepared_cache = prepared

                        bp_plan_key = (
                            sel_key,
                            int(gui_h),
                            int(gui_w),
                            int(max_bin),
                            int(phase_sign_i),
                            float(dr_m),
                            float(fc_hz),
                            float(c_m_s),
                            display_viewport_signature(applied_viewport),
                        )
                        if bp_plan_cache_key == bp_plan_key and bp_plan_cache is not None:
                            bp_plan = bp_plan_cache
                        else:
                            bp_plan = _build_mimo_back_projection_plan(
                                x_pos_m_sel,
                                x_tx_ant_m,
                                x_rx_ant_m,
                                x_grid,
                                y_grid,
                                dr_m=dr_m,
                                fc_hz=fc_hz,
                                c_m_s=c_m_s,
                                max_bin=max_bin,
                                phase_sign=phase_sign_i,
                            )
                            bp_plan_cache_key = bp_plan_key
                            bp_plan_cache = bp_plan

                        img_db = _back_projection_image_mimo(
                            prepared,
                            x_pos_m_sel,
                            x_tx_ant_m,
                            x_rx_ant_m,
                            x_grid,
                            y_grid,
                            dr_m=dr_m,
                            fc_hz=fc_hz,
                            c_m_s=c_m_s,
                            max_bin=max_bin,
                            motion_mode=motion_mode,
                            phase_sign=phase_sign_i,
                            chunk_size=16384,
                            coherent_sum=bool(coherent_sum),
                            bp_plan=bp_plan,
                        )
                        doppler_bins_used = 1
                    else:
                        range_fft_sel = range_fft_data[sel_idx, :, :, :max_bin]
                        reduced = _reduce_avg_mode(range_fft_sel, avg_mode)
                        if reduced.ndim != 3:
                            raise ValueError(f"reduce_avg_mode sar_only shape non valido: {reduced.shape!r}")
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
                        doppler_bins_used = 1

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
                print(
                    f"[OFFLINE INFO] frame algorithm={algorithm} positions={sel_key} "
                    f"synthetic_ant={frame_meta.get('synthetic_antennas', n_ant_used)} "
                    f"angle_mode={frame_meta.get('angle_mode', range_angle_cfg.angle_processing.mode)} "
                    f"filters={frame_meta.get('enabled_filters', _offline_sar_range_angle_filters_enabled(range_angle_cfg))} "
                    f"fft_uniform={frame_meta.get('fft_uniform_geometry', 'n/a')}"
                )
                _queue_put_latest(
                    status_q,
                    {
                        "type": "frame",
                        "algorithm": str(algorithm),
                        "x_start": int(x_start),
                        "x_end": int(x_end),
                        "avg_mode": str(avg_mode),
                        "avg_mode_requested": str(avg_mode_requested),
                        "motion_mode": str(motion_mode),
                        "n_pos_used": int(sel_idx.size),
                        "bp_mode": str(bp_mode),
                        "geometry_source": str(geometry_source),
                        "doppler_bins_used": int(doppler_bins_used),
                        "synthetic_antennas": int(frame_meta.get("synthetic_antennas", 0)),
                        "angle_mode_requested": str(
                            frame_meta.get("angle_mode_requested", range_angle_cfg.angle_processing.mode)
                        ),
                        "angle_mode": str(frame_meta.get("angle_mode", range_angle_cfg.angle_processing.mode)),
                        "fft_uniform_geometry": bool(frame_meta.get("fft_uniform_geometry", False)),
                        "range_angle_enabled_filters": frame_meta.get(
                            "enabled_filters",
                            _offline_sar_range_angle_filters_enabled(range_angle_cfg),
                        ),
                        "elapsed_ms": float((t1 - t0) * 1000.0),
                        **_viewport_status_fields(applied_viewport, fallback_used=False),
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
    finally:
        if shm_range_fft is not None:
            try:
                shm_range_fft.close()
            except Exception:
                pass
            try:
                shm_range_fft.unlink()
            except Exception:
                pass


class OfflineBPRuntime:
    """
    Runtime/controller offline a due processi:
    - reader process: carica bin e prepara range-FFT 4D/5D
      [pos, frame, loop, range] oppure [pos, frame, loop, ant, range]
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
        default_avg_mode: str | None = None,
        default_motion_mode: str | None = None,
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
        offline_cfg_dict = _load_yaml_file(Path(self.offline_config_path))
        fallback_cfg_dict = _load_yaml_file(Path(self.fallback_capture_cfg))
        reconstruction_cfg = offline_cfg_dict.get("reconstruction", {}) or {}
        self.reconstruction_algorithm = _reconstruction_algorithm_normalize(
            _pick(reconstruction_cfg.get("algorithm"), "backprojection")
        )
        self.range_angle_cfg = _read_offline_sar_range_angle_cfg(offline_cfg_dict, fallback_cfg_dict)
        self.fft_workers = int(_read_fft_workers(self.fallback_capture_cfg))
        self.x_pitch_m = float(x_pitch_m) if x_pitch_m is not None else _read_x_pitch_m(self.offline_config_path)
        self.default_avg_mode = _avg_mode_normalize(
            _pick(default_avg_mode, _read_default_avg_mode(self.offline_config_path))
        )
        self.default_motion_mode = _motion_mode_normalize(
            _pick(default_motion_mode, _read_default_motion_mode(self.offline_config_path))
        )
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

        try:
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
                    "fft_workers": int(self.fft_workers),
                    "nfft_range": int(self.nfft_range),
                    "range_max_m": float(self.range_max_m),
                    "crossrange_max_m": float(self.crossrange_max_m),
                    "x_pitch_m": float(self.x_pitch_m),
                    "default_avg_mode": str(self.default_avg_mode),
                    "default_motion_mode": str(self.default_motion_mode),
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
        except Exception as exc:
            err = str(exc)
            try:
                self.stop()
            finally:
                self._last_error = err
            raise

    def stop(self) -> None:
        if (
            not self._started
            and self._reader_p is None
            and self._dsp_p is None
            and self._reader_to_dsp_q is None
            and self._cmd_q is None
            and self._status_q is None
            and self._stop_evt is None
        ):
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

        cleanup_processes(
            (self._reader_p, self._dsp_p),
            graceful_timeout_s=0.4,
            terminate_timeout_s=0.2,
            close_handles=True,
        )
        self._cleanup_queued_shared_memory()
        close_queues((self._reader_to_dsp_q, self._cmd_q, self._status_q))

        self._started = False
        self._ready = False
        self._reader_p = None
        self._dsp_p = None
        self._reader_to_dsp_q = None
        self._cmd_q = None
        self._status_q = None
        self._stop_evt = None

    def _cleanup_queued_shared_memory(self) -> None:
        if self._reader_to_dsp_q is None:
            return
        while True:
            try:
                msg = self._reader_to_dsp_q.get_nowait()
            except pyqueue.Empty:
                break
            except Exception:
                break
            if not isinstance(msg, dict):
                continue
            shm_name = msg.get("range_fft_shm_name")
            if shm_name is None:
                continue
            shm_obj = None
            try:
                shm_obj = shared_memory.SharedMemory(name=str(shm_name))
                try:
                    shm_obj.unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    pass
            except FileNotFoundError:
                pass
            except Exception:
                pass
            finally:
                if shm_obj is not None:
                    try:
                        shm_obj.close()
                    except Exception:
                        pass

    def update_params(
        self,
        *,
        x_start: int | None = None,
        x_end: int | None = None,
        motion_mode: str | None = None,
        avg_mode: str | None = None,
        viewport: DisplayViewport | None = None,
    ) -> None:
        if not self._started or self._cmd_q is None:
            return
        cmd = {
            "type": "update",
            "x_start": x_start,
            "x_end": x_end,
            "motion_mode": None if motion_mode is None else _motion_mode_normalize(motion_mode),
            "avg_mode": None if avg_mode is None else _avg_mode_normalize(avg_mode),
            "viewport": _viewport_to_cmd_payload(viewport),
        }
        self._put_latest_cmd(cmd)

    def poll_frame(self, copy_frame: bool = True) -> tuple[np.ndarray, dict[str, Any]] | None:
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

        if copy_frame:
            return self._frame_cache.copy(), self.last_info
        return self._frame_cache, self.last_info

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
