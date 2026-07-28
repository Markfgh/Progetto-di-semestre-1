"""Lettura delle acquisizioni SAR e orchestrazione della ricostruzione offline.

Il reader valida il layout dei file e produce campioni IQ; i worker eseguono
la pipeline in processi separati; ``OfflineBPRuntime`` è l'API usata dalla GUI
per avviare, aggiornare e arrestare l'elaborazione.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re
import json
import struct
import time
from typing import Any, Mapping
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
    DisplayViewport,
    DisplayProjectionConfig,
    PostRangeFftFilterConfig,
    RealtimeDSPConfig,
    VirtualArrayGeometry,
    angle_processing_from_yaml_dict,
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
    back_projection_image_mimo as _back_projection_image_mimo,
    build_mimo_geometry as _build_mimo_geometry,
    phase_sign_normalize as _phase_sign_normalize,
    power_image_to_db as _power_image_to_db,
    prepare_mimo_snapshots as _prepare_mimo_snapshots,
    residual_video_phase_sign_normalize as _residual_video_phase_sign_normalize,
    prepare_synthetic_aperture_data as _prepare_synthetic_aperture_data,
    synthetic_aperture_uniform_spacing_lambda as _synthetic_aperture_uniform_spacing_lambda,
)
from process_cleanup import cleanup_processes, close_queues

_CAPTURE_FILE_RE = re.compile(r"^capture_pos(-?\d+)\.bin$")
_CAPTURE_HEADER_MAGIC = b"RTPBIN1\x00"
_CAPTURE_HEADER_PREFIX_LEN = len(_CAPTURE_HEADER_MAGIC) + 4
_CAPTURE_HEADER_MAX_LEN = 256 * 1024
_RECONSTRUCTION_ALGORITHMS = {"backprojection", "synthetic_range_angle"}


@dataclass(frozen=True)
class OfflineSARConfig:
    """Selezione dei file e delle posizioni da ricostruire offline."""

    input_dir: Path
    x_start: int
    x_end: int
    x_step: int
    frames_per_position: int | None = None

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path = "offline_config.yaml",
    ) -> "OfflineSARConfig":
        cfg_path = Path(config_path)
        cfg = _load_yaml_file(cfg_path)
        return cls.from_mapping(cfg, base_dir=cfg_path.parent)

    @classmethod
    def from_mapping(
        cls,
        cfg: Mapping[str, Any],
        *,
        base_dir: str | Path = ".",
    ) -> "OfflineSARConfig":
        """Build a config from in-memory GUI values without a temporary YAML."""
        if not isinstance(cfg, Mapping):
            raise ValueError("offline_config: la radice YAML deve essere una mappa")
        base_path = Path(base_dir)

        data_cfg = cfg.get("data", {}) or {}
        scan_cfg = cfg.get("scan", {}) or {}
        cap_cfg = cfg.get("capture", {}) or {}

        input_dir = data_cfg.get("input_dir")
        if input_dir is None:
            raise ValueError("offline_config.yaml: manca data.input_dir")
        input_dir_path = Path(str(input_dir))
        if not input_dir_path.is_absolute():
            input_dir_path = (base_path / input_dir_path).resolve()

        x_start = _to_int("scan.x_start", scan_cfg.get("x_start"))
        x_end = _to_int("scan.x_end", scan_cfg.get("x_end"))
        x_step = _to_int("scan.x_step", _pick(scan_cfg.get("x_step"), 1))

        frames_per_position_raw = cap_cfg.get("frames_per_position")
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


@dataclass(frozen=True)
class SARStreamLayout:
    """Layout validato e omogeneo dell'intera acquisizione SAR su disco."""

    source_dir: Path
    positions: np.ndarray
    files: tuple[Path, ...]
    n_frames_per_position: int
    # Frames physically available in every capture file.  The processing
    # count above can intentionally be smaller when the user selects just
    # the first N frames of a run.
    available_frames_per_position: int
    bytes_per_frame: int
    i16_per_frame: int
    samples: int
    chirps: int
    rx: int
    tx: int
    # Each v1 header carries the measured carriage coordinate.  It is the
    # only valid physical X geometry for linear backprojection; ``positions``
    # remains solely an acquisition/selection identifier.
    stage_positions_m: np.ndarray


@dataclass(frozen=True)
class _CaptureFileRecord:
    """Validated v1 capture-file identity used only by :class:`SARReader`."""

    path: Path
    position_legacy: int
    stage_position_m: float


@dataclass(frozen=True)
class OfflineSyntheticRangeAngleConfig:
    """Configurazione della ricostruzione alternativa range-angolo sintetica."""

    use_realtime_filters: bool
    window_range: str
    window_doppler: str
    window_angle: str
    zero_after_range_fft_bins: int
    post_range_fft_filters: PostRangeFftFilterConfig
    angle_processing: AngleProcessingConfig
    nfft_range: int
    nfft_angle: int
    projection: DisplayProjectionConfig
    filter_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class OfflineReferenceBackgroundConfig:
    """A separate empty-scene acquisition used for complex offline subtraction."""

    enabled: bool = False
    reference_dir: Path | None = None
    scale: float = 1.0


@dataclass(frozen=True)
class OfflineMapBounds:
    """Physical rectangle available to the offline reconstruction and ROI."""

    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float


class SARReader:
    """Validate and stream SAR capture files one position at a time."""

    def __init__(
        self,
        offline_config_path: str | Path = "offline_config.yaml",
        *,
        config: OfflineSARConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else OfflineSARConfig.from_yaml(offline_config_path)
        )

    def describe_stream(self) -> SARStreamLayout:
        """Validate the configured run without loading capture payloads."""
        source_dir = self._resolve_source_dir(self.config.input_dir)
        records = self._scan_capture_records(source_dir)
        expected_positions = list(
            range(
                int(self.config.x_start),
                int(self.config.x_end) + 1,
                int(self.config.x_step),
            )
        )
        pos_files = [(record.position_legacy, record.path) for record in records]
        self._validate_positions(pos_files, expected_positions)
        record_by_position = {record.position_legacy: record for record in records}
        ordered_records = [record_by_position[int(pos)] for pos in expected_positions]
        positions = np.asarray(expected_positions, dtype=np.int32)
        stage_positions_m = np.asarray(
            [float(record.stage_position_m) for record in ordered_records],
            dtype=np.float32,
        )
        if stage_positions_m.shape != positions.shape or not np.all(np.isfinite(stage_positions_m)):
            raise ValueError("Coordinate stage non valide per la scansione lineare")

        samples, chirps, rx, tx, frames_per_pos_hdr = self._derive_capture_layout(
            [(record.position_legacy, record.path) for record in ordered_records]
        )

        bytes_per_frame = int(chirps) * int(samples) * int(rx) * 4
        i16_per_frame = bytes_per_frame // 2
        frames_requested = self.config.frames_per_position
        if frames_requested is None:
            frames_requested = frames_per_pos_hdr

        ordered_files: list[Path] = []
        actual_frames_ref: int | None = None
        for record in ordered_records:
            path = record.path
            capture_label = f"Posizione {record.position_legacy}"
            file_size = int(path.stat().st_size)
            data_offset = int(self._detect_capture_data_offset(path, file_size))
            payload_size = int(file_size - data_offset)
            if payload_size % int(bytes_per_frame) != 0:
                raise ValueError(
                    f"{path.name}: payload_size={payload_size} (offset={data_offset}) "
                    f"non multiplo di bytes_per_frame={bytes_per_frame}"
                )
            n_frames = int(payload_size // int(bytes_per_frame))
            if n_frames <= 0:
                raise ValueError(f"{path.name}: nessun frame nel payload (offset={data_offset})")
            if frames_requested is not None and n_frames < int(frames_requested):
                raise ValueError(
                    f"{capture_label}: n_frames={n_frames}, "
                    f"ma frames_per_position richiesto={int(frames_requested)}"
                )
            if actual_frames_ref is None:
                actual_frames_ref = int(n_frames)
            elif int(n_frames) != int(actual_frames_ref):
                raise ValueError(
                    f"{capture_label}: n_frames={n_frames}, atteso {actual_frames_ref} "
                    "(tutti i file devono avere uguale numero di frame)"
                )
            ordered_files.append(path)

        if actual_frames_ref is None:
            raise RuntimeError("Nessun dato configurato")
        selected_frames = int(actual_frames_ref if frames_requested is None else frames_requested)
        return SARStreamLayout(
            source_dir=source_dir,
            positions=positions,
            files=tuple(ordered_files),
            n_frames_per_position=selected_frames,
            available_frames_per_position=int(actual_frames_ref),
            bytes_per_frame=int(bytes_per_frame),
            i16_per_frame=int(i16_per_frame),
            samples=int(samples),
            chirps=int(chirps),
            rx=int(rx),
            tx=int(tx),
            stage_positions_m=stage_positions_m,
        )

    def iter_iq_positions(self, layout: SARStreamLayout):
        """Yield one decoded IQ position at a time."""
        for pos, path in zip(layout.positions.tolist(), layout.files):
            raw_frames, n_frames = self.read_position(
                path,
                bytes_per_frame=int(layout.bytes_per_frame),
                i16_per_frame=int(layout.i16_per_frame),
                max_frames=int(layout.n_frames_per_position),
            )
            if int(n_frames) != int(layout.n_frames_per_position):
                raise ValueError(
                    f"Posizione {pos}: n_frames={n_frames}, "
                    f"atteso {layout.n_frames_per_position}"
                )
            iq_frames = self._raw_to_iq(
                raw_frames,
                int(n_frames),
                samples=int(layout.samples),
                chirps=int(layout.chirps),
                rx=int(layout.rx),
                tx=int(layout.tx),
            )
            yield int(pos), iq_frames

    def read_position(
        self,
        file_path: str | Path,
        *,
        bytes_per_frame: int,
        i16_per_frame: int,
        max_frames: int | None = None,
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

        frames_to_read = int(n_frames)
        if max_frames is not None:
            if int(max_frames) <= 0:
                raise ValueError("max_frames deve essere > 0")
            if int(n_frames) < int(max_frames):
                raise ValueError(
                    f"{path.name}: n_frames={n_frames}, richiesti max_frames={int(max_frames)}"
                )
            frames_to_read = int(max_frames)

        raw = np.fromfile(
            path,
            dtype=np.int16,
            count=int(frames_to_read) * int(i16_per_frame),
            offset=int(data_offset),
        )
        expected_i16 = frames_to_read * i16_per_frame
        if raw.size != expected_i16:
            raise ValueError(
                f"{path.name}: int16 letti={raw.size}, attesi={expected_i16}"
            )

        raw_frames = raw.reshape(frames_to_read, i16_per_frame)
        return raw_frames, int(frames_to_read)

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
                f"{path.name}: header format non supportato ({fmt!r}); "
                "il reader offline supporta solo 'rt_capture_v1' lineare"
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

        # Il payload DCA è impacchettato in blocchi [I_rx0…I_rxN, Q_rx0…Q_rxN].
        # Il transpose finale espone l'antenna virtuale in ordine TX-major/RX-minor,
        # identico a quello richiesto dalle funzioni DSP e dalla geometria MIMO.
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
            # ``x_frames`` è il batch realtime; per le catture SAR recenti il
            # numero realmente scritto dal logger è esplicito e prioritario.
            frames = _pick(capture.get("frames_per_position"), capture.get("x_frames"))
            frames_per_pos = None if frames is None else _to_int("header.capture.frames_per_position", frames)

            if samples <= 0 or chirps <= 0 or rx <= 0 or tx <= 0:
                raise ValueError(f"{path.name}: header.capture contiene valori <= 0")
            if chirps % tx != 0:
                raise ValueError(f"{path.name}: header.capture.chirps deve essere multiplo di tx")
            if frames_per_pos is not None and frames_per_pos <= 0:
                raise ValueError(f"{path.name}: header.capture.frames_per_position deve essere > 0")

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
                    f"{path.name}: frames_per_position incoerente ({frames_per_pos}, atteso {frames_per_pos_ref})"
                )

        if samples_ref is None or chirps_ref is None or rx_ref is None or tx_ref is None:
            raise RuntimeError("Impossibile derivare capture layout dai file .bin")
        return samples_ref, chirps_ref, rx_ref, tx_ref, frames_per_pos_ref

    @staticmethod
    def _position_legacy_from_metadata(path: Path, meta: Mapping[str, Any]) -> int:
        """Read the linear capture position from a v1 header."""
        if "position" not in meta:
            raise ValueError(f"{path.name}: header senza campo obbligatorio 'position'")
        try:
            return int(meta["position"])
        except Exception as exc:
            raise ValueError(
                f"{path.name}: campo header 'position' non valido ({meta['position']!r})"
            ) from exc

    @staticmethod
    def _stage_position_m_from_metadata(path: Path, meta: Mapping[str, Any]) -> float:
        """Read the measured carriage coordinate required by linear BP."""
        stage = meta.get("stage")
        if not isinstance(stage, Mapping):
            raise ValueError(f"{path.name}: header senza oggetto obbligatorio 'stage'")
        if "position_mm" not in stage:
            raise ValueError(f"{path.name}: header.stage senza campo obbligatorio 'position_mm'")
        try:
            position_mm = float(stage["position_mm"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path.name}: header.stage.position_mm non valido ({stage['position_mm']!r})"
            ) from exc
        if not np.isfinite(position_mm):
            raise ValueError(f"{path.name}: header.stage.position_mm deve essere finito")
        return float(position_mm * 1e-3)

    def _capture_record_from_header(self, path: Path, meta: Mapping[str, Any]) -> _CaptureFileRecord:
        """Turn one accepted v1 header into an immutable reader record."""
        position_legacy = self._position_legacy_from_metadata(path, meta)
        return _CaptureFileRecord(
            path=path,
            position_legacy=position_legacy,
            stage_position_m=self._stage_position_m_from_metadata(path, meta),
        )

    def _scan_capture_records(self, source_dir: Path) -> list[_CaptureFileRecord]:
        """Discover valid linear RTP v1 capture files."""
        records: list[_CaptureFileRecord] = []
        for path in sorted(source_dir.glob("*.bin"), key=lambda item: item.name):
            try:
                metadata = self._read_capture_header_metadata(path)
                record = self._capture_record_from_header(path, metadata)
            except ValueError:
                # The capture directory historically may contain unrelated .bin
                # files.  Keep the v1 behaviour of ignoring non-capture files,
                # but never hide a malformed file that claims the capture
                # filename convention.
                if _CAPTURE_FILE_RE.match(path.name) is not None:
                    raise
                continue

            name_match = _CAPTURE_FILE_RE.match(path.name)
            if name_match is not None:
                file_id = int(name_match.group(1))
                if file_id != record.position_legacy:
                    raise ValueError(
                        f"{path.name}: posizione incoerente "
                        f"(nome={file_id}, header={record.position_legacy})"
                    )
            records.append(record)

        if not records:
            raise FileNotFoundError(
                f"Nessun file di capture valido trovato in {source_dir} "
                f"(richiesto header {_CAPTURE_HEADER_MAGIC!r} con format='rt_capture_v1')"
            )
        return records

    def _validate_positions(
        self,
        pos_files: list[tuple[int, Path]],
        expected_positions: list[int],
    ) -> None:
        found_positions = [pos for pos, _ in pos_files]
        found_set = set(found_positions)

        missing = [pos for pos in expected_positions if pos not in found_set]
        duplicate_positions = sorted(
            pos for pos in found_set if found_positions.count(pos) > 1
        )

        # A run may contain more positions than the reconstruction currently
        # needs.  ``x_start/x_end/x_step`` deliberately select a subset (for
        # example every second capture), therefore non-selected files are not
        # an error.  Every requested position must still exist exactly once.
        if missing or duplicate_positions:
            raise ValueError(
                "Posizioni richieste non valide. "
                f"Missing={missing if missing else '[]'} | "
                f"Duplicate={duplicate_positions if duplicate_positions else '[]'}"
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
    """Inserisce un comando senza bloccare, scartando quello più vecchio se serve."""
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
        # A BP grid can sample arbitrary physical coordinates.  Unlike a
        # realtime raster viewport, preserve the explicitly requested ROI
        # instead of snapping it to a display-pixel quantum.
        quantize=False,
    )


def _backprojection_viewport_max_bin(
    viewport: DisplayViewport,
    *,
    x_pos_m: np.ndarray,
    x_tx_ant_m: np.ndarray,
    x_rx_ant_m: np.ndarray,
    dr_m: float,
    available_bins: int,
) -> int:
    """Return the highest FMCW bin that can be addressed by an active BP ROI.

    The range is evaluated against the actual selected SAR and MIMO geometry,
    not just against the Y-axis label.  This keeps an ROI reconstruction from
    reading a 32k range FFT when only a short physical section is requested.
    """
    available = max(1, int(available_bins))
    if not np.isfinite(float(dr_m)) or float(dr_m) <= 0.0:
        return available

    x_positions = np.asarray(x_pos_m, dtype=np.float64).reshape(-1)
    x_tx_offsets = np.asarray(x_tx_ant_m, dtype=np.float64).reshape(-1)
    x_rx_offsets = np.asarray(x_rx_ant_m, dtype=np.float64).reshape(-1)
    if x_positions.size == 0 or x_tx_offsets.size == 0 or x_rx_offsets.size == 0:
        return available
    if x_tx_offsets.size != x_rx_offsets.size:
        raise ValueError("geometria TX/RX non coerente nel calcolo dei bin BP")

    x_pixels = np.asarray((viewport.x_min_m, viewport.x_max_m), dtype=np.float64)
    y_far = max(0.0, float(viewport.y_max_m))
    sensor_x = x_positions[:, None]
    x_tx = sensor_x + x_tx_offsets[None, :]
    x_rx = sensor_x + x_rx_offsets[None, :]
    dx_tx = x_pixels[:, None, None] - x_tx[None, :, :]
    dx_rx = x_pixels[:, None, None] - x_rx[None, :, :]
    r_total = np.hypot(dx_tx, y_far) + np.hypot(dx_rx, y_far)
    max_one_way_m = 0.5 * float(np.max(r_total))
    if not np.isfinite(max_one_way_m):
        return available
    # Two guard samples cover the cubic complex interpolation used by BP.
    needed = int(np.ceil(max_one_way_m / float(dr_m))) + 2
    return max(1, min(needed, available))
def _read_x_pitch_m(offline_config_path: str | Path) -> float:
    cfg = _load_yaml_file(Path(offline_config_path))
    scan_cfg = cfg.get("scan", {}) or {}
    raw = _pick(scan_cfg.get("x_pitch_m"), 0.01)
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


def _read_residual_video_phase(offline_config_path: str | Path) -> int:
    cfg = _load_yaml_file(Path(offline_config_path))
    bp_cfg = cfg.get("bp", {}) or {}
    raw = _pick(bp_cfg.get("residual_video_phase"), "off")
    return _residual_video_phase_sign_normalize(
        raw,
        field_name="offline_config: bp.residual_video_phase",
    )


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
        if str(cfg.window_doppler) not in {"none", "rectangular"}:
            enabled.append("window_doppler")
        if str(cfg.window_angle) not in {"none", "rectangular"}:
            enabled.append("window_angle")
    return tuple(enabled)


def _apply_offline_backprojection_range_window(
    signal: np.ndarray,
    *,
    window_type: str,
    enabled: bool,
) -> np.ndarray:
    """Apodize fast-time before the Range FFT for offline BP.

    ``signal`` is MIMO ``[..., frame, loop, ant, sample]``; the last axis is
    always the ADC fast-time axis.
    """
    src = np.asarray(signal, dtype=np.complex64)
    if not enabled or src.ndim < 1:
        return src

    window = _build_window_1d(str(window_type), int(src.shape[-1]))
    if np.allclose(window, 1.0):
        return src

    out = np.array(src, dtype=np.complex64, copy=True)
    out *= window.reshape((1,) * (out.ndim - 1) + (int(window.size),)).astype(
        np.complex64,
        copy=False,
    )
    return out.astype(np.complex64, copy=False)


def _select_offline_range_fft_input(signal: np.ndarray, *, nfft_range: int) -> np.ndarray:
    """Select fast-time samples before windowing and the Range FFT.

    ``nfft_range`` smaller than the capture uses the first N ADC samples;
    a larger value keeps every captured sample and the FFT performs zero-padding.
    """
    src = np.asarray(signal, dtype=np.complex64)
    if src.ndim < 1:
        raise ValueError("signal must have a fast-time axis")
    nfft_i = max(1, int(nfft_range))
    samples_used = min(int(src.shape[-1]), nfft_i)
    return src[..., :samples_used]


def _apply_offline_backprojection_aperture_window(
    snapshots: np.ndarray,
    *,
    window_type: str,
    enabled: bool,
) -> np.ndarray:
    """Taper the complete position×antenna aperture before coherent BP.

    ``snapshots`` has shape ``[pos, frame, ant, range_bin]``.  Flattening
    position-major / antenna-minor matches the MIMO-SAR aperture order used
    by the reader, so a Hamming/Hanning/Blackman window lowers spatial
    sidelobes without changing range or phase geometry.
    """
    src = np.asarray(snapshots, dtype=np.complex64)
    if not enabled or src.ndim != 4:
        return src

    n_pos, _n_frames, n_ant, _n_bins = src.shape
    window = _build_window_1d(str(window_type), int(n_pos) * int(n_ant))
    if np.allclose(window, 1.0):
        return src

    out = np.array(src, dtype=np.complex64, copy=True)
    out *= window.reshape(int(n_pos), 1, int(n_ant), 1).astype(np.complex64, copy=False)
    return out.astype(np.complex64, copy=False)


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
                    # Offline reconstruction is static-only: never inherit
                    # realtime slow-time filters, which suppress Doppler zero.
                    "slow_time": {"enabled": False, "mode": "none"},
                }
            }
        }
    )
    filters_cfg, filter_warnings = sanitize_display_post_range_fft_filters(filters_cfg)

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
    # Keep the global settings as fallbacks, but allow offline tuning without
    # changing the realtime FFT sizes.
    fallback_fft = fallback_cfg.get("fft", {}) or {}
    fallback_capture = fallback_cfg.get("capture", {}) or {}
    nfft_range_raw = _pick(
        branch.get("nfft_range"),
        fallback_fft.get("nfft_range"),
        fallback_capture.get("samples"),
        256,
    )
    try:
        nfft_range = max(1, int(nfft_range_raw))
    except (TypeError, ValueError):
        nfft_range = 256
    nfft_angle_raw = _pick(branch.get("nfft_angle"), fallback_fft.get("nfft_angle"), 256)
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
        nfft_range=int(nfft_range),
        nfft_angle=int(nfft_angle),
        projection=projection,
        filter_warnings=tuple(str(w) for w in filter_warnings),
    )


def _read_offline_background_reference_cfg(
    cfg: dict[str, Any],
    *,
    config_path: Path,
) -> OfflineReferenceBackgroundConfig:
    """Read the empty-scene acquisition subtracted before reconstruction."""
    block = cfg.get("offline_background", {}) or {}
    if not isinstance(block, dict):
        raise ValueError("offline_config: offline_background deve essere una mappa")

    enabled = _to_bool(
        "offline_config: offline_background.enabled",
        _pick(block.get("enabled"), False),
    )
    reference_raw = block.get("reference_dir")
    reference_dir: Path | None = None
    if reference_raw is not None and str(reference_raw).strip():
        reference_dir = Path(str(reference_raw).strip())
        if not reference_dir.is_absolute():
            reference_dir = (config_path.parent / reference_dir).resolve()

    if enabled and reference_dir is None:
        raise ValueError(
            "offline_config: offline_background.reference_dir e' obbligatorio quando il background e' attivo"
        )

    scale = _to_float(
        "offline_config: offline_background.scale",
        _pick(block.get("scale"), 1.0),
        1.0,
    )
    if not np.isfinite(scale) or float(scale) < 0.0:
        raise ValueError("offline_config: offline_background.scale deve essere finito e >= 0")

    return OfflineReferenceBackgroundConfig(
        enabled=bool(enabled),
        reference_dir=reference_dir,
        scale=float(scale),
    )


def offline_map_bounds_from_yaml_dict(
    cfg: dict[str, Any],
    fallback_cfg: dict[str, Any],
) -> OfflineMapBounds:
    """Read the full offline reconstruction rectangle from configuration.

    This rectangle is the only parent domain for the offline ROI.  It is
    deliberately independent of the realtime display settings and of either
    FFT size.  Missing values use the current capture configuration solely as
    conservative defaults for a hand-written minimal offline YAML.
    """
    reconstruction_cfg = cfg.get("reconstruction", {}) or {}
    bounds_cfg = reconstruction_cfg.get("map_bounds", {}) or {}
    if not isinstance(bounds_cfg, dict):
        raise ValueError("offline_config: reconstruction.map_bounds deve essere una mappa")

    fallback_display = fallback_cfg.get("display", {}) or {}
    fallback_processing = fallback_cfg.get("processing", {}) or {}
    fallback_y_raw = _pick(
        fallback_processing.get("range_max_m"),
        fallback_display.get("range_max"),
        50.0,
    )
    fallback_x_raw = _pick(
        fallback_display.get("crossrange_max_m"),
        fallback_display.get("crossrange_max"),
        25.0,
    )
    try:
        fallback_y_max = _to_float("fallback processing/display range", fallback_y_raw, 50.0)
    except ValueError:
        fallback_y_max = 50.0
    try:
        fallback_x_max = _to_float("fallback display crossrange", fallback_x_raw, 25.0)
    except ValueError:
        fallback_x_max = 25.0

    x_min_m = _to_float(
        "offline_config: reconstruction.map_bounds.x_min_m",
        _pick(bounds_cfg.get("x_min_m"), -float(fallback_x_max)),
        -float(fallback_x_max),
    )
    x_max_m = _to_float(
        "offline_config: reconstruction.map_bounds.x_max_m",
        _pick(bounds_cfg.get("x_max_m"), float(fallback_x_max)),
        float(fallback_x_max),
    )
    y_min_m = _to_float(
        "offline_config: reconstruction.map_bounds.y_min_m",
        _pick(bounds_cfg.get("y_min_m"), 0.0),
        0.0,
    )
    y_max_m = _to_float(
        "offline_config: reconstruction.map_bounds.y_max_m",
        _pick(bounds_cfg.get("y_max_m"), fallback_y_max),
        float(fallback_y_max),
    )
    if not all(np.isfinite(value) for value in (x_min_m, x_max_m, y_min_m, y_max_m)):
        raise ValueError("offline_config: reconstruction.map_bounds deve contenere valori finiti")
    if x_max_m <= x_min_m:
        raise ValueError("offline_config: map_bounds.x_max_m deve essere > x_min_m")
    if y_min_m < 0.0:
        raise ValueError("offline_config: map_bounds.y_min_m deve essere >= 0")
    if y_max_m <= y_min_m:
        raise ValueError("offline_config: map_bounds.y_max_m deve essere > y_min_m")
    return OfflineMapBounds(
        x_min_m=float(x_min_m),
        x_max_m=float(x_max_m),
        y_min_m=float(y_min_m),
        y_max_m=float(y_max_m),
    )


def _read_bp_runtime_cfg(offline_config_path: str | Path, fallback_capture_cfg: str | Path) -> dict[str, Any]:
    """Compone i parametri di ricostruzione da YAML offline e config di cattura."""
    cfg = _load_yaml_file(Path(offline_config_path))
    fallback_cfg = _load_yaml_file(Path(fallback_capture_cfg))
    reconstruction_cfg = cfg.get("reconstruction", {}) or {}
    bp_cfg = cfg.get("bp", {}) or {}
    algorithm = _reconstruction_algorithm_normalize(_pick(reconstruction_cfg.get("algorithm"), "backprojection"))
    background_reference = _read_offline_background_reference_cfg(
        cfg,
        config_path=Path(offline_config_path),
    )

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

    return {
        "tx_offsets_m": None if tx_offsets_m is None else tx_offsets_m.astype(np.float32, copy=False),
        "rx_offsets_m": None if rx_offsets_m is None else rx_offsets_m.astype(np.float32, copy=False),
        "algorithm": str(algorithm),
        "residual_video_phase": _read_residual_video_phase(offline_config_path),
        "map_bounds": offline_map_bounds_from_yaml_dict(cfg, fallback_cfg),
        "range_angle": _read_offline_sar_range_angle_cfg(cfg, fallback_cfg),
        "background_reference": background_reference,
    }


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
) -> np.ndarray:
    if raw_mimo.ndim != 6:
        raise ValueError(f"raw_mimo shape non valido per pre-filters: {raw_mimo.shape!r}")
    if not filters_cfg.mean_after_range_fft.enabled or not filters_cfg.mean_after_range_fft.axes:
        return np.asarray(raw_mimo, dtype=np.complex64, copy=False)

    out = np.array(raw_mimo, dtype=np.complex64, copy=True)
    n_pos = int(out.shape[0])
    for pos_i in range(n_pos):
        # Reuse the realtime helper axis convention: [frame, loop, tx, range_bin, rx].
        view = np.transpose(out[pos_i], (0, 1, 2, 4, 3)).astype(np.complex64, copy=False)
        view = subtract_selected_mean(view, filters_cfg.mean_after_range_fft)
        out[pos_i] = np.transpose(view, (0, 1, 2, 4, 3)).astype(np.complex64, copy=False)
    return out.astype(np.complex64, copy=False)


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
    if range_fft_sel.ndim != 4:
        raise ValueError(
            "range_fft_sel shape non valido per synthetic_range_angle: "
            f"{range_fft_sel.shape!r}; atteso [pos, frame, ant, bin]."
        )

    _n_pos_sel, _n_frames_sel, n_ant_sel, n_bins_sel = range_fft_sel.shape
    if x_tx_ant_m.size != int(n_ant_sel) or x_rx_ant_m.size != int(n_ant_sel):
        raise ValueError("x_tx_ant_m/x_rx_ant_m size != asse antenna synthetic_range_angle")
    if int(n_ant_sel) <= 0 or int(n_bins_sel) <= 0:
        return np.zeros((int(gui_h), int(gui_w)), dtype=np.float32), {
            "synthetic_antennas": 0,
            "angle_input_elements": 0,
            "angle_elements_used": 0,
            "angle_mode_requested": str(range_angle_cfg.angle_processing.mode),
            "angle_mode": str(range_angle_cfg.angle_processing.mode),
            "nfft_angle_requested": int(range_angle_cfg.nfft_angle),
            "nfft_angle_effective": int(range_angle_cfg.nfft_angle),
            "fft_uniform_geometry": False,
            "enabled_filters": _offline_sar_range_angle_filters_enabled(range_angle_cfg),
        }
    if int(tx_i) <= 0 or int(rx_i) <= 0 or int(tx_i) * int(rx_i) != int(n_ant_sel):
        raise ValueError(
            f"tx/rx non coerente con asse antenna synthetic_range_angle: tx={tx_i}, rx={rx_i}, ant={n_ant_sel}"
        )

    zero_doppler = np.asarray(range_fft_sel, dtype=np.complex64)
    synthetic = _prepare_synthetic_aperture_data(
        zero_doppler,
        selected_positions=np.asarray(selected_positions, dtype=np.int32).reshape(-1),
        x_pitch_m=float(x_pitch_m),
        x_tx_ant_m=x_tx_ant_m,
        x_rx_ant_m=x_rx_ant_m,
    )
    _full_geometry, fft_uniform = _build_synthetic_virtual_array_geometry(
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

    synthetic_ant_input = int(synthetic.x_element_m.size)
    requested_nfft_angle = max(1, int(range_angle_cfg.nfft_angle))
    effective_nfft_angle = int(requested_nfft_angle)
    angle_elements_used = int(synthetic_ant_input)
    x_element_used = np.asarray(synthetic.x_element_m, dtype=np.float32).reshape(-1)
    if angle_cfg_eff.mode == "fft" and int(requested_nfft_angle) < int(synthetic_ant_input):
        # Explicitly use a centered sub-aperture.  Slicing before apodization
        # ensures the selected elements receive a complete symmetric window.
        angle_elements_used = int(requested_nfft_angle)
        first = (int(synthetic_ant_input) - int(angle_elements_used)) // 2
        last = first + int(angle_elements_used)
        synthetic_cube = synthetic_cube[..., first:last]
        x_element_used = x_element_used[first:last]

    geometry, _used_fft_uniform = _build_synthetic_virtual_array_geometry(
        x_element_used,
        wavelength_m=float(c_m_s) / float(fc_hz),
    )
    if range_angle_cfg.use_realtime_filters and synthetic_cube.size > 0:
        w_angle = _build_window_1d(str(range_angle_cfg.window_angle), int(synthetic_cube.shape[-1]))
        synthetic_cube *= w_angle.reshape(1, 1, 1, int(synthetic_cube.shape[-1])).astype(
            np.complex64,
            copy=False,
        )

    dsp_cfg = _build_synthetic_angle_dsp_cfg(
        c_m_s=float(c_m_s),
        fs_hz=float(fs_hz),
        slope_hz_s=float(slope_hz_s),
        nfft_range=int(nfft_range),
        nfft_angle=int(effective_nfft_angle),
        range_max_m=float(viewport.y_max_m),
        synthetic_ant=int(angle_elements_used),
        fft_workers=int(fft_workers),
        frames_like=int(max(1, synthetic_cube.shape[0])),
    )
    angle_axis = build_angle_axis_deg(int(effective_nfft_angle), geometry=geometry)
    angle_steering = (
        np.empty((0, 0), dtype=np.complex64)
        if angle_cfg_eff.mode == "fft"
        else build_angle_steering_matrix(
            int(angle_elements_used),
            int(effective_nfft_angle),
            geometry=geometry,
        )
    )

    if int(synthetic_cube.shape[0]) <= 0:
        heatmap_lin = np.zeros((int(n_bins_sel), int(effective_nfft_angle)), dtype=np.float32)
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
            heatmap_lin = np.zeros((int(n_bins_sel), int(effective_nfft_angle)), dtype=np.float32)

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
        "synthetic_antennas": int(synthetic_ant_input),
        "angle_input_elements": int(synthetic_ant_input),
        "angle_elements_used": int(angle_elements_used),
        "angle_mode_requested": str(range_angle_cfg.angle_processing.mode),
        "angle_mode": str(angle_cfg_eff.mode),
        "nfft_angle_requested": int(requested_nfft_angle),
        "nfft_angle_effective": int(effective_nfft_angle),
        "fft_uniform_geometry": bool(fft_uniform),
        "enabled_filters": _offline_sar_range_angle_filters_enabled(range_angle_cfg),
        "selected_positions": tuple(int(v) for v in np.asarray(selected_positions, dtype=np.int32).reshape(-1).tolist()),
        "projection_lut": lut,
    }
    return img_db.astype(np.float32, copy=False), meta


def _validate_background_reference_layout(
    target: SARStreamLayout,
    reference: SARStreamLayout,
) -> None:
    """Ensure that an empty-scene acquisition can be subtracted sample-wise.

    The number of frames is deliberately *not* compared: each acquisition is
    averaged independently over its own frames before complex subtraction.
    """
    if not np.array_equal(target.positions, reference.positions):
        raise ValueError(
            "Scansione background non compatibile: posizioni diverse dalla scansione target"
        )
    if not np.allclose(
        target.stage_positions_m,
        reference.stage_positions_m,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Scansione background non compatibile: coordinate stage diverse dalla scansione target"
        )

    mismatches: list[str] = []
    for field in ("samples", "chirps", "rx", "tx"):
        target_value = int(getattr(target, field))
        reference_value = int(getattr(reference, field))
        if target_value != reference_value:
            mismatches.append(f"{field} target={target_value}, riferimento={reference_value}")
    if mismatches:
        raise ValueError(
            "Scansione background non compatibile: " + "; ".join(mismatches)
        )


def _prepare_offline_zero_doppler_position(
    iq_position: np.ndarray,
    *,
    nfft_range: int,
    chirps: int,
    tx: int,
    rx: int,
    algorithm: str,
    range_angle_cfg: OfflineSyntheticRangeAngleConfig,
    fft_workers: int,
    doppler_window: np.ndarray | None,
) -> np.ndarray:
    """Prepare one capture position as ``[frame, antenna, range_bin]``."""
    iq = np.asarray(iq_position, dtype=np.complex64)
    if iq.ndim != 4:
        raise ValueError(
            f"iq_position shape non valida: {iq.shape!r}; atteso [frame, loop, ant, sample]"
        )
    n_frames = int(iq.shape[0])
    n_loops = int(chirps) // int(tx)
    n_ant = int(tx) * int(rx)
    if int(iq.shape[1]) != n_loops or int(iq.shape[2]) != n_ant:
        raise ValueError(
            "iq_position non coerente con chirps/tx/rx: "
            f"shape={iq.shape!r}, atteso loop={n_loops}, ant={n_ant}"
        )

    sig = _select_offline_range_fft_input(iq, nfft_range=int(nfft_range))
    sig = _apply_offline_backprojection_range_window(
        sig,
        window_type=str(range_angle_cfg.window_range),
        enabled=bool(range_angle_cfg.use_realtime_filters),
    )
    range_fft_pos = fft.fft(
        sig,
        n=int(nfft_range),
        axis=-1,
        workers=int(fft_workers),
    ).astype(np.complex64, copy=False)
    raw_mimo = range_fft_pos.reshape(
        1,
        n_frames,
        n_loops,
        int(tx),
        int(rx),
        int(nfft_range),
    )
    if algorithm == "synthetic_range_angle" and range_angle_cfg.use_realtime_filters:
        raw_mimo = _apply_offline_sar_range_angle_pre_filters(
            raw_mimo,
            filters_cfg=range_angle_cfg.post_range_fft_filters,
        )
        zero_bins = min(int(range_angle_cfg.zero_after_range_fft_bins), int(raw_mimo.shape[-1]))
        if zero_bins > 0:
            raw_mimo[..., :zero_bins] = np.complex64(0.0)

    prepared = _prepare_mimo_snapshots(
        raw_mimo,
        n_tx=int(tx),
        window_doppler=doppler_window,
        log_info=False,
    )
    return np.asarray(prepared[0], dtype=np.complex64)


def _subtract_reference_background(
    target_snapshots: np.ndarray,
    reference_mean: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    """Subtract an empty-scene complex mean without mutating the target data.

    ``target_snapshots`` is ``[frame, antenna, range_bin]`` and
    ``reference_mean`` is ``[antenna, range_bin]``.  The operation is done
    before BP/power/dB so phase is preserved.
    """
    target = np.asarray(target_snapshots, dtype=np.complex64)
    reference = np.asarray(reference_mean, dtype=np.complex64)
    if target.ndim != 3:
        raise ValueError(
            f"target_snapshots shape non valida: {target.shape!r}; atteso [frame, antenna, bin]"
        )
    if reference.ndim != 2:
        raise ValueError(
            f"reference_mean shape non valida: {reference.shape!r}; atteso [antenna, bin]"
        )
    if tuple(target.shape[1:]) != tuple(reference.shape):
        raise ValueError(
            "background reference non coerente: "
            f"target antenna/bin={tuple(target.shape[1:])}, riferimento={tuple(reference.shape)}"
        )
    scale_f = float(scale)
    if not np.isfinite(scale_f) or scale_f < 0.0:
        raise ValueError("background reference scale deve essere finito e >= 0")

    out = np.array(target, dtype=np.complex64, copy=True)
    out -= np.complex64(scale_f) * reference[None, :, :]
    return out.astype(np.complex64, copy=False)


# ---------------------------------------------------------------------
# Offline Multiprocess Pipeline
# - reader process: load + range-FFT prep
# - dsp process: back-projection + publish frame
# ---------------------------------------------------------------------
def _offline_reader_worker(
    offline_config_path: str,
    fallback_capture_cfg: str,
    nfft_range: int,
    reader_to_dsp_q: Queue,
    status_q: Queue,
    stop_evt,
) -> None:
    """Worker produttore: legge file SAR e passa snapshot compatti al DSP.

    Mantiene il payload in memoria condivisa, così il processo di ricostruzione
    non deve rileggere né duplicare l'intero set di file.
    """
    shm_range_fft = None
    shm_cleanup_transferred = False
    try:
        reader = SARReader(offline_config_path=offline_config_path)
        bp_runtime_cfg = _read_bp_runtime_cfg(offline_config_path, fallback_capture_cfg)
        fft_workers = _read_fft_workers(fallback_capture_cfg)
        algorithm = str(bp_runtime_cfg.get("algorithm", "backprojection"))
        range_angle_cfg: OfflineSyntheticRangeAngleConfig = bp_runtime_cfg["range_angle"]
        background_reference: OfflineReferenceBackgroundConfig = bp_runtime_cfg["background_reference"]
        nfft_range = max(1, int(nfft_range))
        if int(range_angle_cfg.nfft_range) != int(nfft_range):
            range_angle_cfg = replace(range_angle_cfg, nfft_range=int(nfft_range))
        validation_phase = (
            "validating target and background capture files"
            if background_reference.enabled
            else "validating capture files"
        )
        _queue_put_latest(status_q, {"type": "progress", "phase": validation_phase})
        stream_layout = reader.describe_stream()
        reference_reader: SARReader | None = None
        reference_layout: SARStreamLayout | None = None
        if background_reference.enabled:
            assert background_reference.reference_dir is not None
            reference_reader = SARReader(
                config=replace(
                    reader.config,
                    input_dir=background_reference.reference_dir,
                    # The empty-scene run may contain a different number of
                    # frames.  Its own header/layout is still validated by
                    # SARReader for every reference position.
                    frames_per_position=None,
                ),
            )
            reference_layout = reference_reader.describe_stream()
            if stream_layout.source_dir.resolve() == reference_layout.source_dir.resolve():
                raise ValueError(
                    "offline_background.reference_dir deve indicare una scansione diversa da data.input_dir"
                )
            _validate_background_reference_layout(stream_layout, reference_layout)
        range_input_samples = int(stream_layout.samples)
        range_samples_used = min(int(range_input_samples), int(nfft_range))
        _queue_put_latest(
            status_q,
            {
                "type": "progress",
                "phase": "computing Range FFT",
                "range_input_samples": int(range_input_samples),
                "range_samples_used": int(range_samples_used),
                "nfft_range": int(nfft_range),
                "background_reference_enabled": bool(background_reference.enabled),
            },
        )

        mimo_geometry = _resolve_offline_mimo_geometry(
            fallback_capture_cfg,
            tx_i=int(stream_layout.tx),
            rx_i=int(stream_layout.rx),
            tx_offsets_override_m=bp_runtime_cfg.get("tx_offsets_m"),
            rx_offsets_override_m=bp_runtime_cfg.get("rx_offsets_m"),
        )
        geometry_source = str(mimo_geometry["geometry_source"])

        for warning in range_angle_cfg.filter_warnings:
            print(f"[OFFLINE WARN] {warning}")

        n_pos = int(stream_layout.positions.size)
        n_frames = int(stream_layout.n_frames_per_position)
        n_ant = int(stream_layout.tx) * int(stream_layout.rx)
        prepared_shape = (n_pos, n_frames, n_ant, int(nfft_range))
        shm_range_fft = shared_memory.SharedMemory(
            create=True,
            size=int(np.prod(prepared_shape, dtype=np.int64)) * np.dtype(np.complex64).itemsize,
        )
        # La shared memory contiene soltanto gli snapshot Doppler-zero compatti,
        # non l'intero IQ cube: il processo BP può ricostruire senza rileggere
        # i file e senza duplicare tutte le acquisizioni in RAM.
        shm_arr = np.ndarray(prepared_shape, dtype=np.complex64, buffer=shm_range_fft.buf)
        doppler_window = None
        if bool(range_angle_cfg.use_realtime_filters):
            doppler_window = _build_window_1d(
                str(range_angle_cfg.window_doppler),
                int(stream_layout.chirps) // int(stream_layout.tx),
            )

        reference_iter = (
            None
            if reference_reader is None or reference_layout is None
            else iter(reference_reader.iter_iq_positions(reference_layout))
        )
        for pos_idx, (position_id, iq_position) in enumerate(reader.iter_iq_positions(stream_layout)):
            if stop_evt.is_set():
                raise RuntimeError("offline reader stopped")
            phase_prefix = "streaming target + reference" if reference_iter is not None else "streaming"
            _queue_put_latest(
                status_q,
                {
                    "type": "progress",
                    "phase": f"{phase_prefix} Range FFT + Doppler zero {pos_idx + 1}/{n_pos}",
                    "position": int(position_id),
                    "completed_positions": int(pos_idx),
                    "total_positions": int(n_pos),
                },
            )
            prepared_pos = _prepare_offline_zero_doppler_position(
                iq_position,
                nfft_range=int(nfft_range),
                chirps=int(stream_layout.chirps),
                tx=int(stream_layout.tx),
                rx=int(stream_layout.rx),
                algorithm=algorithm,
                range_angle_cfg=range_angle_cfg,
                fft_workers=int(fft_workers),
                doppler_window=doppler_window,
            )
            if reference_iter is not None:
                try:
                    reference_position_id, reference_iq_position = next(reference_iter)
                except StopIteration as exc:
                    raise ValueError(
                        "Scansione background incompleta rispetto alla scansione target"
                    ) from exc
                if int(reference_position_id) != int(position_id):
                    # Il riferimento va sottratto solo dalla stessa posizione
                    # meccanica; accoppiare per ordine senza questo controllo
                    # introdurrebbe una fase/geometria di background errata.
                    raise ValueError(
                        "Scansione background non compatibile: ordine posizioni diverso "
                        f"({reference_position_id} != {position_id})"
                    )
                reference_snapshots = _prepare_offline_zero_doppler_position(
                    reference_iq_position,
                    nfft_range=int(nfft_range),
                    chirps=int(stream_layout.chirps),
                    tx=int(stream_layout.tx),
                    rx=int(stream_layout.rx),
                    algorithm=algorithm,
                    range_angle_cfg=range_angle_cfg,
                    fft_workers=int(fft_workers),
                    doppler_window=doppler_window,
                )
                reference_mean = reference_snapshots.mean(axis=0, dtype=np.complex64)
                prepared_pos = _subtract_reference_background(
                    prepared_pos,
                    reference_mean,
                    scale=float(background_reference.scale),
                )
            # Scrittura streaming: dopo ogni posizione il payload IQ temporaneo
            # può essere rilasciato, mentre il DSP riceverà il cube completo SHM.
            shm_arr[pos_idx] = prepared_pos
            del prepared_pos, iq_position

        if reference_iter is not None:
            try:
                extra_reference_position, _extra_reference_iq = next(reference_iter)
            except StopIteration:
                pass
            else:
                raise ValueError(
                    "Scansione background non compatibile: posizioni aggiuntive "
                    f"a partire da {extra_reference_position}"
                )

        range_fft = shm_arr
        _queue_put_latest(
            status_q,
            {
                "type": "progress",
                "phase": "zero-Doppler snapshots ready",
                "completed_positions": int(n_pos),
                "total_positions": int(n_pos),
            },
        )

        # Passa al DSP il nome della mappa, non l'array serializzato. Il reader
        # mantiene il proprio handle aperto fino allo stop per la semantica SHM
        # di Windows; il consumatore diventa responsabile dell'unlink finale.
        msg = {
            "type": "data",
            "range_fft_shm_name": str(shm_range_fft.name),
            "range_fft_shape": tuple(int(x) for x in range_fft.shape),
            "range_fft_dtype": "complex64",
            "positions": stream_layout.positions.astype(np.int32, copy=False),
            "x_start_cfg": int(reader.config.x_start),
            "x_end_cfg": int(reader.config.x_end),
            "algorithm": str(algorithm),
            "range_angle_cfg": range_angle_cfg,
            "nfft_range": int(nfft_range),
            "range_input_samples": int(range_input_samples),
            "range_samples_used": int(range_samples_used),
            "tx": int(stream_layout.tx),
            "rx": int(stream_layout.rx),
            "background_reference_enabled": bool(background_reference.enabled),
            "background_reference_dir": (
                None if reference_layout is None else str(reference_layout.source_dir)
            ),
            "background_reference_frames": (
                0 if reference_layout is None else int(reference_layout.n_frames_per_position)
            ),
            "background_reference_scale": float(background_reference.scale),
            "bp_x_tx_ant_m": np.asarray(mimo_geometry["x_tx_ant_m"], dtype=np.float32),
            "bp_x_rx_ant_m": np.asarray(mimo_geometry["x_rx_ant_m"], dtype=np.float32),
            "bp_x_pos_m": np.asarray(stream_layout.stage_positions_m, dtype=np.float32),
            "bp_geometry_source": geometry_source,
        }
        _queue_put_latest(reader_to_dsp_q, msg)
        shm_cleanup_transferred = True
        _queue_put_latest(
            status_q,
            {
                "type": "reader_ready",
                "phase": "zero-Doppler snapshots ready",
                "positions": int(stream_layout.positions.size),
                "frames_per_pos": int(stream_layout.n_frames_per_position),
                "algorithm": str(algorithm),
                "range_angle_enabled_filters": _offline_sar_range_angle_filters_enabled(range_angle_cfg),
                "range_angle_use_realtime_filters": bool(range_angle_cfg.use_realtime_filters),
                "range_angle_angle_mode": str(range_angle_cfg.angle_processing.mode),
                "nfft_range": int(nfft_range),
                "range_input_samples": int(range_input_samples),
                "range_samples_used": int(range_samples_used),
                "range_angle_nfft_angle": int(range_angle_cfg.nfft_angle),
                "geometry_source": geometry_source,
                "background_reference_enabled": bool(background_reference.enabled),
                "background_reference_dir": (
                    None if reference_layout is None else str(reference_layout.source_dir)
                ),
                "background_reference_frames": (
                    0 if reference_layout is None else int(reference_layout.n_frames_per_position)
                ),
                "background_reference_scale": float(background_reference.scale),
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
        # On Windows a named mapping disappears when the last open handle is
        # closed.  Keep the producer handle alive for the runtime lifetime so
        # the DSP process can always attach, even if process scheduling delays
        # it after the queue message has been delivered.
        if shm_range_fft is not None and shm_cleanup_transferred:
            stop_evt.wait()
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


def _range_fft_from_init_msg(init_msg: dict[str, Any]) -> tuple[np.ndarray, shared_memory.SharedMemory | None]:
    shm_name = init_msg.get("range_fft_shm_name")
    if shm_name is None:
        raise ValueError("range_fft_shm_name mancante nel messaggio init")
    shape_raw = init_msg.get("range_fft_shape")
    if not isinstance(shape_raw, (tuple, list)):
        raise ValueError("range_fft_shape mancante o non valido nel messaggio init")
    shape = tuple(int(x) for x in shape_raw)
    if len(shape) != 4:
        raise ValueError(f"snapshot MIMO shape non valida: {shape!r}")
    dtype_s = str(init_msg.get("range_fft_dtype", "complex64")).strip().lower()
    if dtype_s != "complex64":
        raise ValueError(f"range_fft_dtype non supportato: {dtype_s!r}")
    shm_obj = shared_memory.SharedMemory(name=str(shm_name))
    arr = np.ndarray(shape, dtype=np.complex64, buffer=shm_obj.buf)
    return arr, shm_obj


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
    map_bounds: OfflineMapBounds,
    x_pitch_m: float,
    phase_sign: int,
    residual_video_phase: int,
) -> None:
    """Worker consumatore: ricostruisce e pubblica immagini nel double buffer."""
    shm_range_fft = None
    try:
        phase_sign_i = _phase_sign_normalize(phase_sign, field_name="phase_sign")
        residual_video_phase_i = _residual_video_phase_sign_normalize(
            residual_video_phase,
            field_name="residual_video_phase",
        )
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
                nfft_range=max(1, int(nfft_range)),
                nfft_angle=256,
                projection=DisplayProjectionConfig(),
                filter_warnings=(),
            )
        if positions.ndim != 1 or positions.size != range_fft_data.shape[0]:
            raise ValueError("positions non coerente con range_fft")

        nfft_range = max(1, int(_pick(init_msg.get("nfft_range"), range_angle_cfg.nfft_range, nfft_range)))
        range_input_samples = max(1, int(_pick(init_msg.get("range_input_samples"), nfft_range)))
        range_samples_used = max(1, int(_pick(init_msg.get("range_samples_used"), min(range_input_samples, nfft_range))))
        background_reference_enabled = bool(init_msg.get("background_reference_enabled", False))
        background_reference_dir = init_msg.get("background_reference_dir", None)
        background_reference_frames = max(0, int(_pick(init_msg.get("background_reference_frames"), 0)))
        background_reference_scale = float(_pick(init_msg.get("background_reference_scale"), 1.0))

        # A BP aperture is a Cartesian product of poses and physical MIMO
        # channels.  Do not apply the historical flattened position×antenna
        # window here: the generalized kernel owns separate pose/channel
        # weights (uniform by default).  Range and zero-Doppler processing are
        # intentionally unchanged.
        bp_window_stages: tuple[str, ...] = ()

        tx_i = max(1, int(_pick(init_msg.get("tx"), 2)))
        rx_i = max(1, int(_pick(init_msg.get("rx"), 4)))
        geometry_source = str(_pick(init_msg.get("bp_geometry_source"), "unknown"))
        n_ant_data = int(range_fft_data.shape[2])
        x_tx_ant_m = np.asarray(init_msg.get("bp_x_tx_ant_m"), dtype=np.float32).reshape(-1)
        x_rx_ant_m = np.asarray(init_msg.get("bp_x_rx_ant_m"), dtype=np.float32).reshape(-1)
        if x_tx_ant_m.size != n_ant_data or x_rx_ant_m.size != n_ant_data:
            raise ValueError("bp_x_tx_ant_m/bp_x_rx_ant_m size != asse antenna range_fft")
        x_pos_m_full = np.asarray(init_msg.get("bp_x_pos_m"), dtype=np.float32).reshape(-1)
        if x_pos_m_full.size != int(positions.size) or not np.all(np.isfinite(x_pos_m_full)):
            raise ValueError("bp_x_pos_m non coerente con le posizioni della scansione lineare")
        # BP is invariant to a common X translation.  Center only the
        # measured trajectory, preserving its real pitch and any carriage
        # positioning error captured in the headers.
        x_pos_m_full = (
            x_pos_m_full - np.mean(x_pos_m_full, dtype=np.float32)
        ).astype(np.float32, copy=False)
        n_ant_used = int(n_ant_data)
        n_bins_total = int(range_fft_data.shape[-1])
        dr_m = float(c_m_s) * float(fs_hz) / (2.0 * float(slope_hz_s) * float(nfft_range))
        home_viewport = build_display_viewport(
            x_min_m=float(map_bounds.x_min_m),
            x_max_m=float(map_bounds.x_max_m),
            y_min_m=float(map_bounds.y_min_m),
            y_max_m=float(map_bounds.y_max_m),
            dr_m=float(dr_m),
            seq=0,
        )
        applied_viewport = home_viewport

        pos_min = int(np.min(positions))
        pos_max = int(np.max(positions))
        x_start = int(_pick(init_msg.get("x_start_cfg"), pos_min))
        x_end = int(_pick(init_msg.get("x_end_cfg"), pos_max))
        x_start = max(pos_min, min(pos_max, x_start))
        x_end = max(pos_min, min(pos_max, x_end))
        if x_end < x_start:
            x_start, x_end = x_end, x_start
        _queue_put_latest(
            status_q,
            {
                "type": "ready",
                "algorithm": str(algorithm),
                "pos_min": pos_min,
                "pos_max": pos_max,
                "x_start": x_start,
                "x_end": x_end,
                "phase_sign": int(phase_sign_i),
                "residual_video_phase": int(residual_video_phase_i),
                "img_h": int(gui_h),
                "img_w": int(gui_w),
                "dr_m": float(dr_m),
                "nfft_range": int(nfft_range),
                "range_input_samples": int(range_input_samples),
                "range_samples_used": int(range_samples_used),
                "virtual_antennas": int(n_ant_used),
                "geometry_source": str(geometry_source),
                "doppler_bins_used": 1,
                "range_angle_use_realtime_filters": bool(range_angle_cfg.use_realtime_filters),
                "range_angle_enabled_filters": _offline_sar_range_angle_filters_enabled(range_angle_cfg),
                "range_angle_angle_mode_requested": str(range_angle_cfg.angle_processing.mode),
                "range_angle_nfft_angle": int(range_angle_cfg.nfft_angle),
                "bp_window_stages": bp_window_stages,
                "background_reference_enabled": bool(background_reference_enabled),
                "background_reference_dir": background_reference_dir,
                "background_reference_frames": int(background_reference_frames),
                "background_reference_scale": float(background_reference_scale),
                **_viewport_status_fields(applied_viewport, fallback_used=False),
            },
        )
        print(
            f"[OFFLINE INFO] algorithm={algorithm} "
            f"default_positions={x_start}:{x_end} angle_mode={range_angle_cfg.angle_processing.mode} "
            f"range={range_input_samples}->{nfft_range} used={range_samples_used} "
            f"filters={_offline_sar_range_angle_filters_enabled(range_angle_cfg)} "
            f"bp_windows={bp_window_stages} "
            f"reference_background={'on' if background_reference_enabled else 'off'}"
        )

        # Le cache descrivono artefatti costosi ma puri. Ogni chiave include i
        # parametri fisici che possono cambiarne il risultato (ROI, posizioni
        # o viewport), così un aggiornamento GUI non riusa dati incompatibili.
        last_job_key = None
        dirty = True
        prepared_cache_key = None
        prepared_cache = None
        grid_cache_key = None
        grid_cache = None
        synthetic_projection_lut_cache_key = None
        synthetic_projection_lut_cache = None

        while not stop_evt.is_set():
            got_cmd = False
            # Drena tutti i comandi prima di ricalcolare: una raffica di slider
            # genera una sola ricostruzione con l'ultima ROI effettiva.
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
                viewport_new = _viewport_from_cmd_payload(
                    cmd.get("viewport"),
                    home_viewport=home_viewport,
                    output_width=int(gui_w),
                    output_height=int(gui_h),
                    dr_m=float(dr_m),
                )
                if viewport_new is not None:
                    applied_viewport = viewport_new
                dirty = True

            if stop_evt.is_set():
                break

            job_key = (
                str(algorithm),
                int(x_start),
                int(x_end),
                display_viewport_signature(applied_viewport),
            )
            if dirty or job_key != last_job_key or got_cmd:
                t0 = time.perf_counter()
                frame_meta: dict[str, Any] = {}
                sel_mask = (positions >= int(x_start)) & (positions <= int(x_end))
                if not np.any(sel_mask):
                    sel_mask[:] = True
                sel_idx = np.where(sel_mask)[0]
                sel_key = tuple(int(v) for v in sel_idx.tolist())
                x_pos_m_sel = x_pos_m_full[sel_idx]
                # This bounds both reconstruction paths to the active ROI.
                # For BP it includes the selected SAR/MIMO sensor positions;
                # for synthetic range-angle it is a conservative range slice.
                viewport_max_bin = _backprojection_viewport_max_bin(
                    applied_viewport,
                    x_pos_m=x_pos_m_sel,
                    x_tx_ant_m=x_tx_ant_m,
                    x_rx_ant_m=x_rx_ant_m,
                    dr_m=float(dr_m),
                    available_bins=int(n_bins_total),
                )
                if algorithm == "synthetic_range_angle":
                    selected_positions = positions[sel_idx].astype(np.int32, copy=False)
                    range_fft_sel = range_fft_data[sel_idx, :, :, :viewport_max_bin]
                    n_pos_sel, n_frames_sel, n_ant_sel, _n_bins_sel = range_fft_sel.shape
                    if int(n_ant_sel) != int(tx_i) * int(rx_i):
                        raise ValueError(
                            f"asse antenna synthetic={n_ant_sel} non coerente con tx/rx={tx_i}/{rx_i}"
                        )
                    mvdr_elements = int(n_pos_sel) * int(n_ant_sel)
                    if (
                        str(range_angle_cfg.angle_processing.mode) == "mvdr"
                        and int(n_frames_sel) < int(mvdr_elements)
                    ):
                        raise ValueError(
                            "MVDR offline sottodeterminato: "
                            f"snapshot={n_frames_sel}, elementi={mvdr_elements}. "
                            "Seleziona meno posizioni (con 8 frame usare una sola posizione), "
                            "aumenta frames_per_position oppure usa Bartlett/backprojection."
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

                    range_fft_sel = range_fft_data[sel_idx, :, :, :viewport_max_bin]
                    _n_pos_sel, _n_frames_sel, n_ant_sel, _n_bins_sel = range_fft_sel.shape
                    if int(n_ant_sel) != int(tx_i) * int(rx_i):
                        raise ValueError(f"asse antenna mimo={n_ant_sel} non coerente con tx/rx={tx_i}/{rx_i}")
                    prepared_key = (
                        sel_key,
                        int(viewport_max_bin),
                        int(tx_i),
                        int(rx_i),
                    )
                    if prepared_cache_key == prepared_key and prepared_cache is not None:
                        prepared = prepared_cache
                    else:
                        prepared = np.asarray(range_fft_sel, dtype=np.complex64)
                        prepared = _apply_offline_backprojection_aperture_window(
                            prepared,
                            window_type=str(range_angle_cfg.window_angle),
                            enabled=bool(range_angle_cfg.use_realtime_filters),
                        )
                        prepared_cache_key = prepared_key
                        prepared_cache = prepared

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
                        max_bin=viewport_max_bin,
                        phase_sign=phase_sign_i,
                        residual_video_phase=residual_video_phase_i,
                        slope_hz_s=float(slope_hz_s),
                        chunk_size=16384,
                    )
                    doppler_bins_used = 1

                # Il writer riempie il buffer inattivo e pubblica indice+seq
                # solo alla fine; la GUI può quindi leggere un frame completo.
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
                positions_text = (
                    f"{int(positions[sel_idx[0]])}..{int(positions[sel_idx[-1]])} "
                    f"count={int(sel_idx.size)}"
                    if int(sel_idx.size) > 0
                    else "none"
                )
                print(
                    f"[OFFLINE INFO] frame algorithm={algorithm} positions={positions_text} "
                    f"synthetic_ant={frame_meta.get('synthetic_antennas', n_ant_used)} "
                    f"angle_used={frame_meta.get('angle_elements_used', 'n/a')} "
                    f"angle_mode={frame_meta.get('angle_mode', range_angle_cfg.angle_processing.mode)} "
                    f"nfft_range={nfft_range} "
                    f"nfft_angle={frame_meta.get('nfft_angle_effective', range_angle_cfg.nfft_angle)} "
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
                        "n_pos_used": int(sel_idx.size),
                        "geometry_source": str(geometry_source),
                        "doppler_bins_used": int(doppler_bins_used),
                        "bp_range_bins_used": int(viewport_max_bin),
                        "nfft_range": int(nfft_range),
                        "range_input_samples": int(range_input_samples),
                        "range_samples_used": int(range_samples_used),
                        "synthetic_antennas": int(frame_meta.get("synthetic_antennas", 0)),
                        "angle_input_elements": int(frame_meta.get("angle_input_elements", 0)),
                        "angle_elements_used": int(frame_meta.get("angle_elements_used", 0)),
                        "angle_mode_requested": str(
                            frame_meta.get("angle_mode_requested", range_angle_cfg.angle_processing.mode)
                        ),
                        "angle_mode": str(frame_meta.get("angle_mode", range_angle_cfg.angle_processing.mode)),
                        "nfft_angle_requested": int(
                            frame_meta.get("nfft_angle_requested", range_angle_cfg.nfft_angle)
                        ),
                        "nfft_angle_effective": int(
                            frame_meta.get("nfft_angle_effective", range_angle_cfg.nfft_angle)
                        ),
                        "fft_uniform_geometry": bool(frame_meta.get("fft_uniform_geometry", False)),
                        "range_angle_enabled_filters": frame_meta.get(
                            "enabled_filters",
                            _offline_sar_range_angle_filters_enabled(range_angle_cfg),
                        ),
                        "bp_window_stages": bp_window_stages,
                        "background_reference_enabled": bool(background_reference_enabled),
                        "background_reference_dir": background_reference_dir,
                        "background_reference_frames": int(background_reference_frames),
                        "background_reference_scale": float(background_reference_scale),
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
    - reader process: elabora una posizione alla volta e pubblica snapshot
      Doppler-zero MIMO ``[pos, frame, ant, range]``;
    - dsp process: applica la ricostruzione richiesta e pubblica frame su
      double-buffer senza conservare un piano geometrico globale.
    """

    def __init__(
        self,
        *,
        offline_config_path: str | Path = "offline_config.yaml",
        fallback_capture_cfg: str | Path = "realtime_config.yaml",
        c_m_s: float,
        fs_hz: float,
        slope_hz_s: float,
        fc_hz: float,
        nfft_range: int,
        image_h: int,
        image_w: int,
        x_pitch_m: float | None = None,
        phase_sign: int | None = None,
        residual_video_phase: int | str | None = None,
    ) -> None:
        self.offline_config_path = str(offline_config_path)
        self.fallback_capture_cfg = str(fallback_capture_cfg)
        self.c_m_s = float(c_m_s)
        self.fs_hz = float(fs_hz)
        self.slope_hz_s = float(slope_hz_s)
        self.fc_hz = float(fc_hz)
        self.image_h = int(image_h)
        self.image_w = int(image_w)
        offline_cfg_dict = _load_yaml_file(Path(self.offline_config_path))
        fallback_cfg_dict = _load_yaml_file(Path(self.fallback_capture_cfg))
        self.map_bounds = offline_map_bounds_from_yaml_dict(offline_cfg_dict, fallback_cfg_dict)
        self.range_angle_cfg = _read_offline_sar_range_angle_cfg(offline_cfg_dict, fallback_cfg_dict)
        offline_fft_branch = offline_cfg_dict.get("offline_sar_range_angle", {}) or {}
        self.nfft_range = int(
            self.range_angle_cfg.nfft_range
            if "nfft_range" in offline_fft_branch
            else max(1, int(nfft_range))
        )
        if int(self.range_angle_cfg.nfft_range) != int(self.nfft_range):
            self.range_angle_cfg = replace(self.range_angle_cfg, nfft_range=int(self.nfft_range))
        self.fft_workers = int(_read_fft_workers(self.fallback_capture_cfg))
        self.x_pitch_m = float(x_pitch_m) if x_pitch_m is not None else _read_x_pitch_m(self.offline_config_path)
        self.phase_sign = _phase_sign_normalize(
            _pick(phase_sign, _read_phase_sign(self.offline_config_path)),
            field_name="phase_sign",
        )
        self.residual_video_phase = _residual_video_phase_sign_normalize(
            _pick(residual_video_phase, _read_residual_video_phase(self.offline_config_path)),
            field_name="residual_video_phase",
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
        """Crea risorse IPC, avvia reader/DSP e attende il loro handshake iniziale."""
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
                    "map_bounds": self.map_bounds,
                    "x_pitch_m": float(self.x_pitch_m),
                    "phase_sign": int(self.phase_sign),
                    "residual_video_phase": int(self.residual_video_phase),
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
        """Segnala lo stop e chiude con cura processi, code e shared memory."""
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
        viewport: DisplayViewport | None = None,
    ) -> None:
        """Invia l'ultimo ROI/intervallo richiesto senza riavviare i processi."""
        if not self._started or self._cmd_q is None:
            return
        cmd = {
            "type": "update",
            "x_start": x_start,
            "x_end": x_end,
            "viewport": _viewport_to_cmd_payload(viewport),
        }
        self._put_latest_cmd(cmd)

    def poll_frame(self, copy_frame: bool = True) -> tuple[np.ndarray, dict[str, Any]] | None:
        """Legge un frame nuovo dal double buffer, oppure ``None`` se invariato."""
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
