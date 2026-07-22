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
    back_projection_power_mimo_geometry as _back_projection_power_mimo_geometry,
    back_projection_image_mimo as _back_projection_image_mimo,
    build_mimo_geometry as _build_mimo_geometry,
    phase_sign_normalize as _phase_sign_normalize,
    power_image_to_db as _power_image_to_db,
    prepare_mimo_snapshots as _prepare_mimo_snapshots,
    residual_video_phase_sign_normalize as _residual_video_phase_sign_normalize,
    prepare_synthetic_aperture_data as _prepare_synthetic_aperture_data,
    synthetic_aperture_uniform_spacing_lambda as _synthetic_aperture_uniform_spacing_lambda,
)
from sar_geometry import (
    CylindricalCapture,
    default_iwr1443_2tx4rx_geometry,
    transform_element_coordinates,
    xy_plane_voxel_grid,
    xz_plane_voxel_grid,
    yz_plane_voxel_grid,
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
    # ``scan.*`` describes only the legacy linear run.  A v2 cylindrical
    # acquisition is completely ordered by its header metadata and may omit
    # this block from offline_config.yaml.
    x_start: int | None
    x_end: int | None
    x_step: int | None
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

        x_start_raw = scan_cfg.get("x_start")
        x_end_raw = scan_cfg.get("x_end")
        x_step_raw = scan_cfg.get("x_step")
        has_linear_scan = x_start_raw is not None or x_end_raw is not None or x_step_raw is not None
        if has_linear_scan:
            if x_start_raw is None or x_end_raw is None:
                raise ValueError(
                    "scan.x_start e scan.x_end sono entrambi obbligatori "
                    "quando è presente la configurazione scan lineare"
                )
            x_start = _to_int("scan.x_start", x_start_raw)
            x_end = _to_int("scan.x_end", x_end_raw)
            x_step = _to_int("scan.x_step", _pick(x_step_raw, 1))
        else:
            x_start = None
            x_end = None
            x_step = None

        frames_per_position_raw = cap_cfg.get("frames_per_position")
        frames_per_position = None if frames_per_position_raw is None else _to_int(
            "capture.frames_per_position",
            frames_per_position_raw,
        )

        if x_step is not None and x_step <= 0:
            raise ValueError("scan.x_step deve essere > 0")
        if x_start is not None and x_end is not None and x_end < x_start:
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
    # ``positions`` remains the legacy public field.  It contains the linear
    # position IDs for v1 and the capture IDs for v2.  New code must use the
    # explicit fields below instead of assigning a geometric meaning to it.
    geometry_mode: str = "legacy_linear"
    capture_ids: np.ndarray | None = None
    acquisition_indices: np.ndarray | None = None
    cylindrical_captures: tuple[CylindricalCapture, ...] = ()
    # Linear runs carry the measured carriage coordinate in every v1 header.
    # It is the only valid physical X geometry for legacy BP; ``positions``
    # remains solely an acquisition/selection identifier.
    stage_positions_m: np.ndarray | None = None


@dataclass(frozen=True)
class _CaptureFileRecord:
    """Validated capture-file identity used only by :class:`SARReader`."""

    path: Path
    format: str
    position_legacy: int
    capture_id: int
    acquisition_index: int | None
    cylindrical: CylindricalCapture | None
    stage_position_m: float | None


def cylindrical_capture_world_coordinates(
    layout: SARStreamLayout,
    *,
    fc_hz: float,
    c_m_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical TX/RX world coordinates for a regular v2 cylinder.

    The IWR1443 2 TX × 4 RX local geometry is already calibrated and fixed.
    This helper performs only the rigid body-to-world transform specified by
    the capture headers; it never estimates or applies a calibration.
    """
    if str(layout.geometry_mode) != "cylindrical_regular":
        raise ValueError("layout non cilindrico: richiesto geometry_mode='cylindrical_regular'")
    if int(layout.tx) != 2 or int(layout.rx) != 4:
        raise ValueError(
            "La geometria cilindrica di questa fase richiede la ULA fisica 2 TX × 4 RX"
        )
    captures = tuple(layout.cylindrical_captures)
    if len(captures) != int(layout.positions.size):
        raise ValueError("metadata cylindrical non coerente con il numero di catture")
    array_geometry = default_iwr1443_2tx4rx_geometry(
        fc_hz=float(fc_hz),
        c_m_s=float(c_m_s),
    )
    tx_global, rx_global = transform_element_coordinates(captures, array_geometry)
    return (
        np.asarray(tx_global, dtype=np.float32),
        np.asarray(rx_global, dtype=np.float32),
    )


@dataclass(frozen=True)
class OfflineSyntheticRangeAngleConfig:
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


@dataclass(frozen=True)
class CylindricalPlane:
    """Fixed-height world XY plane used by the circular offline preview.

    It is deliberately separate from ``OfflineMapBounds``: the latter keeps
    the historical forward-looking linear-SAR convention (``y >= 0``), while
    a circular world plane legitimately spans both signs of Y.
    """

    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    z_m: float


_CYLINDRICAL_SECTION_PLANES = {"xy", "xz", "yz"}


@dataclass(frozen=True)
class CylindricalViewBounds:
    """World bounds available to a regular cylindrical-SAR section view.

    X/Y bounds are always required.  Z bounds are optional only for a
    single-height circular capture, where vertical sections are forbidden.
    """

    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    z_min_m: float | None = None
    z_max_m: float | None = None

    def __post_init__(self) -> None:
        values = {
            "x_min_m": self.x_min_m,
            "x_max_m": self.x_max_m,
            "y_min_m": self.y_min_m,
            "y_max_m": self.y_max_m,
        }
        normalized: dict[str, float] = {}
        for name, value in values.items():
            normalized[name] = _to_float(f"cylindrical view bounds.{name}", value)
        if normalized["x_max_m"] <= normalized["x_min_m"]:
            raise ValueError("cylindrical view bounds.x_max_m deve essere > x_min_m")
        if normalized["y_max_m"] <= normalized["y_min_m"]:
            raise ValueError("cylindrical view bounds.y_max_m deve essere > y_min_m")

        has_z_min = self.z_min_m is not None
        has_z_max = self.z_max_m is not None
        if has_z_min != has_z_max:
            raise ValueError("cylindrical view bounds richiede entrambi z_min_m e z_max_m")
        if has_z_min:
            z_min = _to_float("cylindrical view bounds.z_min_m", self.z_min_m)
            z_max = _to_float("cylindrical view bounds.z_max_m", self.z_max_m)
            if z_max <= z_min:
                raise ValueError("cylindrical view bounds.z_max_m deve essere > z_min_m")
        else:
            z_min = None
            z_max = None

        for name, value in normalized.items():
            object.__setattr__(self, name, float(value))
        object.__setattr__(self, "z_min_m", z_min)
        object.__setattr__(self, "z_max_m", z_max)

    @property
    def has_z_bounds(self) -> bool:
        return self.z_min_m is not None and self.z_max_m is not None


@dataclass(frozen=True)
class CylindricalSection:
    """One oriented, fixed-coordinate 2-D world section."""

    plane: str = "xy"
    coordinate_m: float = 0.0

    def __post_init__(self) -> None:
        plane = str(self.plane).strip().lower()
        if plane not in _CYLINDRICAL_SECTION_PLANES:
            raise ValueError("cylindrical section.plane deve essere 'xy', 'xz' o 'yz'")
        coordinate = _to_float("cylindrical section.coordinate_m", self.coordinate_m)
        object.__setattr__(self, "plane", plane)
        object.__setattr__(self, "coordinate_m", float(coordinate))


@dataclass(frozen=True)
class CylindricalView:
    """Configured bounds and active plane for a v2 circular/cylindrical run."""

    bounds: CylindricalViewBounds
    section: CylindricalSection

    def __post_init__(self) -> None:
        if not isinstance(self.bounds, CylindricalViewBounds):
            raise TypeError("cylindrical view.bounds deve essere CylindricalViewBounds")
        if not isinstance(self.section, CylindricalSection):
            raise TypeError("cylindrical view.section deve essere CylindricalSection")


@dataclass(frozen=True)
class CylindricalRunSummary:
    """Header-derived topology and geometry exposed to the offline UI."""

    kind: str
    capture_count: int
    angle_count: int
    height_count: int
    radius_m: float
    scene_center_m: tuple[float, float, float]
    height_m_values: tuple[float, ...]
    world_z_values_m: tuple[float, ...]

    @property
    def has_vertical_resolution(self) -> bool:
        return self.height_count >= 2


def cylindrical_run_summary(layout: SARStreamLayout) -> CylindricalRunSummary:
    """Summarize a validated regular v2 run without reading its IQ payload."""
    if str(layout.geometry_mode) != "cylindrical_regular":
        raise ValueError("richiesta sintesi cilindrica per una run non v2")
    captures = tuple(layout.cylindrical_captures)
    if not captures:
        raise ValueError("nessuna cattura cilindrica nella run v2")
    first = captures[0]
    by_height: dict[int, CylindricalCapture] = {}
    for capture in captures:
        by_height.setdefault(int(capture.height_index), capture)
    ordered_heights = tuple(by_height[index] for index in sorted(by_height))
    height_values = tuple(float(capture.height_m) for capture in ordered_heights)
    center = tuple(float(value) for value in first.scene_center_m.tolist())
    world_z_values = tuple(float(center[2] + height) for height in height_values)
    height_count = len(ordered_heights)
    return CylindricalRunSummary(
        kind="cylindrical" if height_count >= 2 else "circular",
        capture_count=int(len(captures)),
        angle_count=int(first.angle_count),
        height_count=int(height_count),
        radius_m=float(first.radius_m),
        scene_center_m=center,
        height_m_values=height_values,
        world_z_values_m=world_z_values,
    )


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
        formats = {record.format for record in records}
        if len(formats) != 1:
            raise ValueError(
                "La stessa directory non può mescolare header rt_capture_v1 e rt_capture_v2"
            )

        format_name = next(iter(formats))
        if format_name == "rt_capture_v1":
            if (
                self.config.x_start is None
                or self.config.x_end is None
                or self.config.x_step is None
            ):
                raise ValueError(
                    "scan.x_start, scan.x_end e scan.x_step sono obbligatori "
                    "per le catture lineari rt_capture_v1"
                )
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
            capture_ids = positions.copy()
            acquisition_indices = np.arange(positions.size, dtype=np.int32)
            geometry_mode = "legacy_linear"
            cylindrical_captures: tuple[CylindricalCapture, ...] = ()
            stage_positions_m = np.asarray(
                [float(record.stage_position_m) for record in ordered_records],
                dtype=np.float32,
            )
            if (
                stage_positions_m.shape != positions.shape
                or not np.all(np.isfinite(stage_positions_m))
            ):
                raise ValueError("Coordinate stage non valide per la scansione lineare")
        elif format_name == "rt_capture_v2":
            ordered_records = self._validate_regular_cylindrical_records(records)
            positions = np.asarray(
                [record.capture_id for record in ordered_records],
                dtype=np.int32,
            )
            capture_ids = positions.copy()
            acquisition_indices = np.asarray(
                [int(record.acquisition_index) for record in ordered_records],
                dtype=np.int32,
            )
            geometry_mode = "cylindrical_regular"
            stage_positions_m = None
            cylindrical_captures = tuple(
                record.cylindrical for record in ordered_records if record.cylindrical is not None
            )
            if len(cylindrical_captures) != len(ordered_records):
                raise RuntimeError("Header v2 senza metadata cylindrical validato")
        else:  # Defensive: _read_capture_header_metadata already restricts this.
            raise RuntimeError(f"Formato cattura inatteso: {format_name!r}")

        samples, chirps, rx, tx, frames_per_pos_hdr = self._derive_capture_layout(
            [(record.capture_id, record.path) for record in ordered_records]
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
            capture_label = (
                f"Cattura {record.capture_id}"
                if geometry_mode == "cylindrical_regular"
                else f"Posizione {record.position_legacy}"
            )
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
            geometry_mode=geometry_mode,
            capture_ids=capture_ids,
            acquisition_indices=acquisition_indices,
            cylindrical_captures=cylindrical_captures,
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

    def _extract_position_from_header(self, path: Path) -> int:
        meta = self._read_capture_header_metadata(path)
        return self._position_legacy_from_metadata(path, meta)

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
        if fmt not in {"rt_capture_v1", "rt_capture_v2"}:
            raise ValueError(
                f"{path.name}: header format non supportato ({fmt!r}), "
                "attesi 'rt_capture_v1' o 'rt_capture_v2'"
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
        """Read the legacy ``position`` field without assigning it v2 geometry."""
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
        """Turn one accepted v1/v2 header into an immutable reader record."""
        format_name = str(meta.get("format", ""))
        position_legacy = self._position_legacy_from_metadata(path, meta)
        if format_name == "rt_capture_v1":
            return _CaptureFileRecord(
                path=path,
                format=format_name,
                position_legacy=position_legacy,
                capture_id=position_legacy,
                acquisition_index=None,
                cylindrical=None,
                stage_position_m=self._stage_position_m_from_metadata(path, meta),
            )

        if format_name != "rt_capture_v2":
            raise ValueError(f"{path.name}: formato header non gestito ({format_name!r})")
        cylindrical_raw = meta.get("cylindrical")
        if not isinstance(cylindrical_raw, Mapping):
            raise ValueError(f"{path.name}: header v2 senza oggetto obbligatorio 'cylindrical'")
        if "capture_id" not in meta or "acquisition_index" not in meta:
            raise ValueError(
                f"{path.name}: header v2 richiede 'capture_id' e 'acquisition_index'"
            )

        cylindrical_metadata = dict(cylindrical_raw)
        for key in ("capture_id", "acquisition_index"):
            if key in cylindrical_metadata and cylindrical_metadata[key] != meta[key]:
                raise ValueError(
                    f"{path.name}: {key} incoerente tra header v2 e blocco cylindrical"
                )
            cylindrical_metadata[key] = meta[key]
        try:
            cylindrical = CylindricalCapture.from_dict(cylindrical_metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path.name}: metadata cylindrical non valido: {exc}") from exc

        return _CaptureFileRecord(
            path=path,
            format=format_name,
            position_legacy=position_legacy,
            capture_id=cylindrical.capture_id,
            acquisition_index=cylindrical.acquisition_index,
            cylindrical=cylindrical,
            stage_position_m=None,
        )

    def _scan_capture_records(self, source_dir: Path) -> list[_CaptureFileRecord]:
        """Discover valid RTP capture files while retaining their header version."""
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
                header_id = (
                    record.capture_id
                    if record.format == "rt_capture_v2"
                    else record.position_legacy
                )
                if file_id != header_id:
                    field_name = "capture_id" if record.format == "rt_capture_v2" else "posizione"
                    raise ValueError(
                        f"{path.name}: {field_name} incoerente "
                        f"(nome={file_id}, header={header_id})"
                    )
            records.append(record)

        if not records:
            raise FileNotFoundError(
                f"Nessun file di capture valido trovato in {source_dir} "
                f"(richiesto header {_CAPTURE_HEADER_MAGIC!r} con format='rt_capture_v1' o 'rt_capture_v2')"
            )
        return records

    def _scan_position_files(self, source_dir: Path) -> list[tuple[int, Path]]:
        """Compatibility helper for callers that still inspect v1 positions."""
        pos_files = [
            (record.position_legacy, record.path)
            for record in self._scan_capture_records(source_dir)
            if record.format == "rt_capture_v1"
        ]
        pos_files.sort(key=lambda item: item[0])
        pos_ids = [pos for pos, _ in pos_files]
        if len(pos_ids) != len(set(pos_ids)):
            raise ValueError(f"Posizioni duplicate trovate in {source_dir}: {pos_ids}")
        return pos_files

    @staticmethod
    def _validate_regular_cylindrical_records(
        records: list[_CaptureFileRecord],
    ) -> list[_CaptureFileRecord]:
        """Validate the single-turn-per-height cylindrical acquisition contract.

        A v2 run has a contiguous temporal sequence.  For a fixed
        ``angle_count``, temporal item ``i`` must be exactly
        ``height_index=i//angle_count, angle_index=i%angle_count``.  This
        excludes arbitrary paths, multiple turns at one height, incomplete
        turns, and unwrapped angular metadata by construction.
        """
        if not records:
            raise RuntimeError("Nessuna cattura cilindrica da validare")
        captures: list[CylindricalCapture] = []
        for record in records:
            if record.format != "rt_capture_v2" or record.cylindrical is None:
                raise ValueError("La scansione cilindrica richiede solo header rt_capture_v2")
            captures.append(record.cylindrical)

        capture_ids = [capture.capture_id for capture in captures]
        if len(capture_ids) != len(set(capture_ids)):
            raise ValueError(f"capture_id duplicati nella scansione v2: {capture_ids}")
        acquisition_indices = [capture.acquisition_index for capture in captures]
        if len(acquisition_indices) != len(set(acquisition_indices)):
            raise ValueError(
                f"acquisition_index duplicati nella scansione v2: {acquisition_indices}"
            )

        ordered = sorted(records, key=lambda record: int(record.acquisition_index))
        ordered_captures = [record.cylindrical for record in ordered]
        if any(capture is None for capture in ordered_captures):
            raise RuntimeError("Cattura cilindrica mancante dopo la validazione")
        captures_ordered = [capture for capture in ordered_captures if capture is not None]
        expected_indices = list(range(len(captures_ordered)))
        actual_indices = [capture.acquisition_index for capture in captures_ordered]
        if actual_indices != expected_indices:
            raise ValueError(
                "acquisition_index v2 deve essere contiguo e iniziare da zero "
                f"(trovato {actual_indices})"
            )

        angle_counts = {capture.angle_count for capture in captures_ordered}
        if len(angle_counts) != 1:
            raise ValueError("angle_count deve essere costante nella scansione cilindrica")
        angle_count = next(iter(angle_counts))
        if len(captures_ordered) % angle_count != 0:
            raise ValueError(
                "La scansione cilindrica contiene un giro incompleto: "
                f"{len(captures_ordered)} catture per angle_count={angle_count}"
            )

        reference_radius = captures_ordered[0].radius_m
        reference_center = captures_ordered[0].scene_center_m
        azimuth_by_angle: dict[int, float] = {}
        height_by_index: dict[int, float] = {}
        for acquisition_index, capture in enumerate(captures_ordered):
            expected_height_index = acquisition_index // angle_count
            expected_angle_index = acquisition_index % angle_count
            if (
                capture.height_index != expected_height_index
                or capture.angle_index != expected_angle_index
            ):
                raise ValueError(
                    "Sequenza cilindrica non regolare: "
                    f"acquisition_index={capture.acquisition_index} richiede "
                    f"height_index={expected_height_index}, angle_index={expected_angle_index}; "
                    f"trovati height_index={capture.height_index}, angle_index={capture.angle_index}"
                )
            if not np.isclose(capture.radius_m, reference_radius, rtol=0.0, atol=1e-12):
                raise ValueError("radius_m deve rimanere costante nella scansione cilindrica")
            if not np.allclose(capture.scene_center_m, reference_center, rtol=0.0, atol=1e-12):
                raise ValueError("scene_center_m deve rimanere costante nella scansione cilindrica")

            previous_azimuth = azimuth_by_angle.setdefault(capture.angle_index, capture.azimuth_rad)
            if not np.isclose(capture.azimuth_rad, previous_azimuth, rtol=0.0, atol=1e-12):
                raise ValueError(
                    "azimuth_rad deve essere uguale per lo stesso angle_index a ogni quota"
                )
            previous_height = height_by_index.setdefault(capture.height_index, capture.height_m)
            if not np.isclose(capture.height_m, previous_height, rtol=0.0, atol=1e-12):
                raise ValueError(
                    "height_m deve rimanere costante durante ogni giro completo"
                )

        # The mechanical path is a single regular 360-degree turn.  A common
        # starting azimuth is allowed, but every angle index must advance by
        # one fixed signed step of 2*pi/angle_count and the wrapped values
        # must repeat at each height.  This intentionally rejects arbitrary
        # or unwrapped paths while allowing either mechanical direction.
        azimuth_zero = azimuth_by_angle[0]
        angular_step = (2.0 * np.pi) / float(angle_count)
        if angle_count <= 1:
            direction = 1.0
        else:
            observed_step = float(
                ((azimuth_by_angle[1] - azimuth_zero + np.pi) % (2.0 * np.pi)) - np.pi
            )
            if np.isclose(observed_step, angular_step, rtol=0.0, atol=1e-10):
                direction = 1.0
            elif np.isclose(observed_step, -angular_step, rtol=0.0, atol=1e-10):
                direction = -1.0
            else:
                raise ValueError(
                    "azimuth_rad non descrive passi regolari di 2*pi/angle_count"
                )
        for angle_index in range(angle_count):
            expected = float(
                (azimuth_zero + direction * float(angle_index) * angular_step) % (2.0 * np.pi)
            )
            observed = float(azimuth_by_angle[angle_index])
            wrapped_error = float(((observed - expected + np.pi) % (2.0 * np.pi)) - np.pi)
            if not np.isclose(wrapped_error, 0.0, rtol=0.0, atol=1e-10):
                raise ValueError(
                    "azimuth_rad non descrive passi regolari di 2*pi/angle_count"
                )

        declared_height_counts = {capture.height_count for capture in captures_ordered}
        if len(declared_height_counts) > 1:
            raise ValueError("height_count deve essere coerente nella scansione cilindrica")
        declared_height_count = next(iter(declared_height_counts))
        n_heights = len(captures_ordered) // angle_count
        if declared_height_count is not None and declared_height_count != n_heights:
            raise ValueError(
                f"height_count={declared_height_count} non coerente con le {n_heights} quote presenti"
            )
        return ordered

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


def _build_cylindrical_section_viewport(
    *,
    x_min_m: float,
    x_max_m: float,
    y_min_m: float,
    y_max_m: float,
    seq: int = 0,
    home_viewport: DisplayViewport | None = None,
) -> DisplayViewport:
    """Build a generic signed-world viewport for v2 section reconstruction.

    ``DisplayViewport`` is shared with the legacy range/angle display, whose
    builder correctly clamps range to ``y >= 0``.  Cylindrical world Y and Z
    coordinates are signed, so v2 must construct its metadata without that
    linear-SAR assumption.
    """
    values = (x_min_m, x_max_m, y_min_m, y_max_m)
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("viewport cilindrico contiene coordinate non finite")
    x0, x1 = sorted((float(x_min_m), float(x_max_m)))
    y0, y1 = sorted((float(y_min_m), float(y_max_m)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("viewport cilindrico richiede estensioni positive su entrambi gli assi")
    if home_viewport is None:
        zoom_level = 1.0
    else:
        home_width = max(1e-12, float(home_viewport.x_max_m - home_viewport.x_min_m))
        home_height = max(1e-12, float(home_viewport.y_max_m - home_viewport.y_min_m))
        zoom_level = max(home_width / (x1 - x0), home_height / (y1 - y0), 1.0)
    return DisplayViewport(
        x_min_m=x0,
        x_max_m=x1,
        y_min_m=y0,
        y_max_m=y1,
        # These fields are diagnostic-only for a v2 Cartesian section.
        range_min_bin_f=0.0,
        range_max_bin_f=0.0,
        angle_min_deg=0.0,
        angle_max_deg=0.0,
        zoom_level=float(zoom_level),
        seq=int(seq),
    )


def _cylindrical_viewport_from_cmd_payload(
    payload: Any,
    *,
    home_viewport: DisplayViewport,
    output_width: int,
    output_height: int,
) -> DisplayViewport | None:
    """Clamp a v2 ROI to signed world bounds without range-axis semantics."""
    if not isinstance(payload, Mapping):
        return None
    try:
        x0 = float(payload.get("x_min_m"))
        x1 = float(payload.get("x_max_m"))
        y0 = float(payload.get("y_min_m"))
        y1 = float(payload.get("y_max_m"))
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(value) for value in (x0, x1, y0, y1)):
        return None
    home_x0, home_x1 = float(home_viewport.x_min_m), float(home_viewport.x_max_m)
    home_y0, home_y1 = float(home_viewport.y_min_m), float(home_viewport.y_max_m)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0 = max(home_x0, min(home_x1, x0))
    x1 = max(home_x0, min(home_x1, x1))
    y0 = max(home_y0, min(home_y1, y0))
    y1 = max(home_y0, min(home_y1, y1))
    min_x = max((home_x1 - home_x0) / float(max(1, int(output_width))), 1e-9)
    min_y = max((home_y1 - home_y0) / float(max(1, int(output_height))), 1e-9)
    if x1 <= x0:
        x1 = min(home_x1, x0 + min_x)
        if x1 <= x0:
            x0 = max(home_x0, x1 - min_x)
    if y1 <= y0:
        y1 = min(home_y1, y0 + min_y)
        if y1 <= y0:
            y0 = max(home_y0, y1 - min_y)
    try:
        seq = int(payload.get("seq", int(home_viewport.seq) + 1))
    except (TypeError, ValueError):
        seq = int(home_viewport.seq) + 1
    return _build_cylindrical_section_viewport(
        x_min_m=x0,
        x_max_m=x1,
        y_min_m=y0,
        y_max_m=y1,
        seq=seq,
        home_viewport=home_viewport,
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


def _offline_backprojection_windows_enabled(cfg: OfflineSyntheticRangeAngleConfig) -> tuple[str, ...]:
    """Return the window stages shared by the offline BP pipeline.

    The settings under ``offline_sar_range_angle`` are useful preprocessing
    controls for both reconstruction algorithms. Unlike mean subtraction, a window
    does not remove static scene content and is therefore safe for the
    zero-Doppler backprojection path.
    """
    if not cfg.use_realtime_filters:
        return ()

    enabled: list[str] = []
    for stage, window_type in (
        ("range_window", cfg.window_range),
        ("doppler_window", cfg.window_doppler),
        ("aperture_window", cfg.window_angle),
    ):
        if str(window_type).strip().lower() not in {"none", "rectangular"}:
            enabled.append(f"{stage}:{window_type}")
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


def cylindrical_plane_from_yaml_dict(cfg: Mapping[str, Any]) -> CylindricalPlane | None:
    """Read the optional circular-SAR world plane from ``offline_config``.

    YAML shape::

        reconstruction:
          cylindrical_plane:
            x_min_m: -1.0
            x_max_m:  1.0
            y_min_m: -1.0
            y_max_m:  1.0
            z_m: 0.0

    If the block is absent, the reader derives a conservative square from the
    capture's fixed radius and scene center.  No 3-D renderer is configured
    here; volumes are constructed explicitly with ``xyz_volume_voxel_grid``.
    """
    reconstruction = cfg.get("reconstruction", {}) or {}
    if not isinstance(reconstruction, Mapping):
        raise ValueError("offline_config: reconstruction deve essere un oggetto")
    raw = reconstruction.get("cylindrical_plane")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("offline_config: reconstruction.cylindrical_plane deve essere un oggetto")
    required = ("x_min_m", "x_max_m", "y_min_m", "y_max_m", "z_m")
    missing = [key for key in required if raw.get(key) is None]
    if missing:
        raise ValueError(
            "offline_config: reconstruction.cylindrical_plane senza campi: "
            + ", ".join(missing)
        )
    values = {
        key: _to_float(f"offline_config: reconstruction.cylindrical_plane.{key}", raw[key])
        for key in required
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("offline_config: reconstruction.cylindrical_plane deve contenere valori finiti")
    if values["x_max_m"] <= values["x_min_m"]:
        raise ValueError("offline_config: cylindrical_plane.x_max_m deve essere > x_min_m")
    if values["y_max_m"] <= values["y_min_m"]:
        raise ValueError("offline_config: cylindrical_plane.y_max_m deve essere > y_min_m")
    return CylindricalPlane(**values)


def cylindrical_view_from_yaml_dict(cfg: Mapping[str, Any]) -> CylindricalView | None:
    """Read the extended v2 section-view configuration from ``offline_config``.

    ``bounds.z_*`` is intentionally optional here: a legacy circular run can
    display XY only, while a regular multi-height run gets header-derived
    bounds until the user saves an explicit ``cylindrical_view`` block.
    """
    reconstruction = cfg.get("reconstruction", {}) or {}
    if not isinstance(reconstruction, Mapping):
        raise ValueError("offline_config: reconstruction deve essere un oggetto")
    raw = reconstruction.get("cylindrical_view")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("offline_config: reconstruction.cylindrical_view deve essere un oggetto")
    raw_bounds = raw.get("bounds")
    if not isinstance(raw_bounds, Mapping):
        raise ValueError("offline_config: reconstruction.cylindrical_view.bounds deve essere un oggetto")
    required_xy = ("x_min_m", "x_max_m", "y_min_m", "y_max_m")
    missing = [key for key in required_xy if raw_bounds.get(key) is None]
    if missing:
        raise ValueError(
            "offline_config: reconstruction.cylindrical_view.bounds senza campi: "
            + ", ".join(missing)
        )
    z_min_raw = raw_bounds.get("z_min_m")
    z_max_raw = raw_bounds.get("z_max_m")
    bounds = CylindricalViewBounds(
        x_min_m=_to_float(
            "offline_config: reconstruction.cylindrical_view.bounds.x_min_m",
            raw_bounds["x_min_m"],
        ),
        x_max_m=_to_float(
            "offline_config: reconstruction.cylindrical_view.bounds.x_max_m",
            raw_bounds["x_max_m"],
        ),
        y_min_m=_to_float(
            "offline_config: reconstruction.cylindrical_view.bounds.y_min_m",
            raw_bounds["y_min_m"],
        ),
        y_max_m=_to_float(
            "offline_config: reconstruction.cylindrical_view.bounds.y_max_m",
            raw_bounds["y_max_m"],
        ),
        z_min_m=(
            None
            if z_min_raw is None
            else _to_float("offline_config: reconstruction.cylindrical_view.bounds.z_min_m", z_min_raw)
        ),
        z_max_m=(
            None
            if z_max_raw is None
            else _to_float("offline_config: reconstruction.cylindrical_view.bounds.z_max_m", z_max_raw)
        ),
    )
    raw_section = raw.get("section", {}) or {}
    if not isinstance(raw_section, Mapping):
        raise ValueError("offline_config: reconstruction.cylindrical_view.section deve essere un oggetto")
    plane = str(_pick(raw_section.get("plane"), "xy"))
    coordinate_raw = _pick(raw_section.get("coordinate_m"), 0.0)
    section = CylindricalSection(
        plane=plane,
        coordinate_m=_to_float(
            "offline_config: reconstruction.cylindrical_view.section.coordinate_m",
            coordinate_raw,
        ),
    )
    if section.plane != "xy" and not bounds.has_z_bounds:
        raise ValueError(
            "offline_config: reconstruction.cylindrical_view.bounds.z_min_m/z_max_m "
            "sono obbligatori per una sezione verticale"
        )
    return CylindricalView(bounds=bounds, section=section)


def _default_cylindrical_plane(captures: tuple[CylindricalCapture, ...]) -> CylindricalPlane:
    """Derive a neutral XY plane centred on a regular cylindrical scan."""
    if not captures:
        raise ValueError("impossibile derivare il piano: nessuna cattura cilindrica")
    first = captures[0]
    center = first.scene_center_m
    radius = float(first.radius_m)
    return CylindricalPlane(
        x_min_m=float(center[0] - radius),
        x_max_m=float(center[0] + radius),
        y_min_m=float(center[1] - radius),
        y_max_m=float(center[1] + radius),
        z_m=float(center[2]),
    )


def _default_cylindrical_view(captures: tuple[CylindricalCapture, ...]) -> CylindricalView:
    """Derive safe world bounds and a horizontal section from v2 metadata."""
    if not captures:
        raise ValueError("impossibile derivare la vista: nessuna cattura cilindrica")
    first = captures[0]
    center = first.scene_center_m
    radius = float(first.radius_m)
    by_height: dict[int, CylindricalCapture] = {}
    for capture in captures:
        by_height.setdefault(int(capture.height_index), capture)
    ordered = tuple(by_height[index] for index in sorted(by_height))
    world_z = tuple(float(first.scene_center_m[2] + capture.height_m) for capture in ordered)
    has_vertical_resolution = len(world_z) >= 2
    if has_vertical_resolution:
        z_min_m = float(min(world_z))
        z_max_m = float(max(world_z))
        section_z = float(0.5 * (z_min_m + z_max_m))
    else:
        z_min_m = None
        z_max_m = None
        section_z = float(world_z[0])
    return CylindricalView(
        bounds=CylindricalViewBounds(
            x_min_m=float(center[0] - radius),
            x_max_m=float(center[0] + radius),
            y_min_m=float(center[1] - radius),
            y_max_m=float(center[1] + radius),
            z_min_m=z_min_m,
            z_max_m=z_max_m,
        ),
        section=CylindricalSection(plane="xy", coordinate_m=section_z),
    )


def resolve_cylindrical_view(
    *,
    configured_view: CylindricalView | None,
    legacy_plane: CylindricalPlane | None,
    captures: tuple[CylindricalCapture, ...],
) -> CylindricalView:
    """Resolve v2 configuration while retaining old XY-only plane files."""
    default_view = _default_cylindrical_view(captures)
    height_count = len({int(capture.height_index) for capture in captures})
    if configured_view is not None:
        bounds = configured_view.bounds
        if not bounds.has_z_bounds and default_view.bounds.has_z_bounds:
            bounds = replace(
                bounds,
                z_min_m=default_view.bounds.z_min_m,
                z_max_m=default_view.bounds.z_max_m,
            )
        view = CylindricalView(bounds=bounds, section=configured_view.section)
    elif legacy_plane is not None:
        bounds = CylindricalViewBounds(
            x_min_m=float(legacy_plane.x_min_m),
            x_max_m=float(legacy_plane.x_max_m),
            y_min_m=float(legacy_plane.y_min_m),
            y_max_m=float(legacy_plane.y_max_m),
            z_min_m=default_view.bounds.z_min_m,
            z_max_m=default_view.bounds.z_max_m,
        )
        view = CylindricalView(
            bounds=bounds,
            section=CylindricalSection(plane="xy", coordinate_m=float(legacy_plane.z_m)),
        )
    else:
        view = default_view

    if view.section.plane != "xy" and height_count < 2:
        raise ValueError("le sezioni XZ/YZ richiedono almeno due quote della scansione cilindrica")
    if view.section.plane != "xy" and not view.bounds.has_z_bounds:
        raise ValueError("le sezioni XZ/YZ richiedono bounds Z nella configurazione cilindrica")
    return view


def cylindrical_section_axis_labels(section: CylindricalSection) -> tuple[str, str]:
    """Return horizontal and vertical world-axis labels for a section."""
    if section.plane == "xy":
        return ("X", "Y")
    if section.plane == "xz":
        return ("X", "Z")
    if section.plane == "yz":
        return ("Y", "Z")
    raise ValueError(f"sezione cilindrica non supportata: {section.plane!r}")


def cylindrical_section_bounds(
    view: CylindricalView,
    *,
    section: CylindricalSection | None = None,
) -> tuple[float, float, float, float]:
    """Return horizontal/vertical viewport bounds for the active 2-D section."""
    active = view.section if section is None else section
    bounds = view.bounds
    if active.plane == "xy":
        return (bounds.x_min_m, bounds.x_max_m, bounds.y_min_m, bounds.y_max_m)
    if not bounds.has_z_bounds:
        raise ValueError("sezione verticale richiesta senza bounds Z")
    assert bounds.z_min_m is not None and bounds.z_max_m is not None
    if active.plane == "xz":
        return (bounds.x_min_m, bounds.x_max_m, bounds.z_min_m, bounds.z_max_m)
    if active.plane == "yz":
        return (bounds.y_min_m, bounds.y_max_m, bounds.z_min_m, bounds.z_max_m)
    raise ValueError(f"sezione cilindrica non supportata: {active.plane!r}")


def validate_cylindrical_section(
    view: CylindricalView,
    summary: CylindricalRunSummary,
    section: CylindricalSection,
) -> CylindricalSection:
    """Validate that a requested section is supported by the acquired run."""
    if section.plane != "xy" and not summary.has_vertical_resolution:
        raise ValueError("le sezioni XZ/YZ richiedono almeno due quote della scansione cilindrica")
    bounds = view.bounds
    if section.plane == "xy":
        if bounds.has_z_bounds:
            assert bounds.z_min_m is not None and bounds.z_max_m is not None
            if not bounds.z_min_m <= section.coordinate_m <= bounds.z_max_m:
                raise ValueError("la quota Z della sezione XY deve rientrare nei bounds cilindrici")
    elif section.plane == "xz":
        if not bounds.y_min_m <= section.coordinate_m <= bounds.y_max_m:
            raise ValueError("la coordinata Y della sezione XZ deve rientrare nei bounds cilindrici")
    elif section.plane == "yz":
        if not bounds.x_min_m <= section.coordinate_m <= bounds.x_max_m:
            raise ValueError("la coordinata X della sezione YZ deve rientrare nei bounds cilindrici")
    return section


def cylindrical_view_to_dict(view: CylindricalView) -> dict[str, Any]:
    """Return JSON/process-message-safe v2 view metadata."""
    return {
        "bounds": {
            "x_min_m": float(view.bounds.x_min_m),
            "x_max_m": float(view.bounds.x_max_m),
            "y_min_m": float(view.bounds.y_min_m),
            "y_max_m": float(view.bounds.y_max_m),
            "z_min_m": None if view.bounds.z_min_m is None else float(view.bounds.z_min_m),
            "z_max_m": None if view.bounds.z_max_m is None else float(view.bounds.z_max_m),
        },
        "section": {
            "plane": str(view.section.plane),
            "coordinate_m": float(view.section.coordinate_m),
        },
        "axes": list(cylindrical_section_axis_labels(view.section)),
    }


def cylindrical_xy_plane_if_active(view: CylindricalView) -> CylindricalPlane | None:
    """Expose the active section through the pre-v2-view XY compatibility type."""
    if view.section.plane != "xy":
        return None
    return CylindricalPlane(
        x_min_m=float(view.bounds.x_min_m),
        x_max_m=float(view.bounds.x_max_m),
        y_min_m=float(view.bounds.y_min_m),
        y_max_m=float(view.bounds.y_max_m),
        z_m=float(view.section.coordinate_m),
    )


def cylindrical_run_summary_to_dict(summary: CylindricalRunSummary) -> dict[str, Any]:
    """Return process-message-safe metadata for the offline GUI."""
    return {
        "kind": str(summary.kind),
        "capture_count": int(summary.capture_count),
        "angle_count": int(summary.angle_count),
        "height_count": int(summary.height_count),
        "radius_m": float(summary.radius_m),
        "scene_center_m": [float(value) for value in summary.scene_center_m],
        "height_m_values": [float(value) for value in summary.height_m_values],
        "world_z_values_m": [float(value) for value in summary.world_z_values_m],
        "has_vertical_resolution": bool(summary.has_vertical_resolution),
    }


def _read_bp_runtime_cfg(offline_config_path: str | Path, fallback_capture_cfg: str | Path) -> dict[str, Any]:
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
        "cylindrical_plane": cylindrical_plane_from_yaml_dict(cfg),
        "cylindrical_view": cylindrical_view_from_yaml_dict(cfg),
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
    if str(target.geometry_mode) != str(reference.geometry_mode):
        raise ValueError(
            "Scansione background non compatibile: geometria v1/v2 diversa dalla scansione target"
        )
    if str(target.geometry_mode) == "cylindrical_regular":
        if len(target.cylindrical_captures) != len(reference.cylindrical_captures):
            raise ValueError(
                "Scansione background non compatibile: numero di pose cilindriche diverso"
            )
        for index, (target_capture, reference_capture) in enumerate(
            zip(target.cylindrical_captures, reference.cylindrical_captures)
        ):
            target_values = target_capture.to_dict()
            reference_values = reference_capture.to_dict()
            if (
                target_values["capture_id"] != reference_values["capture_id"]
                or target_values["acquisition_index"] != reference_values["acquisition_index"]
                or target_values["angle_index"] != reference_values["angle_index"]
                or target_values["height_index"] != reference_values["height_index"]
                or target_values["angle_count"] != reference_values["angle_count"]
                or not np.isclose(target_values["azimuth_rad"], reference_values["azimuth_rad"], rtol=0.0, atol=1e-12)
                or not np.isclose(target_values["height_m"], reference_values["height_m"], rtol=0.0, atol=1e-12)
                or not np.isclose(target_values["radius_m"], reference_values["radius_m"], rtol=0.0, atol=1e-12)
                or not np.allclose(
                    target_values["scene_center_m"],
                    reference_values["scene_center_m"],
                    rtol=0.0,
                    atol=1e-12,
                )
            ):
                raise ValueError(
                    "Scansione background non compatibile: posa cilindrica diversa "
                    f"alla cattura {index}"
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

        geometry_mode = str(stream_layout.geometry_mode)
        bp_tx_global_m: np.ndarray | None = None
        bp_rx_global_m: np.ndarray | None = None
        cylindrical_plane: CylindricalPlane | None = None
        cylindrical_view: CylindricalView | None = None
        cylindrical_summary: CylindricalRunSummary | None = None
        mimo_geometry: dict[str, Any] | None = None
        if geometry_mode == "cylindrical_regular":
            if algorithm != "backprojection":
                print(
                    "[OFFLINE WARN] rt_capture_v2 richiede backprojection: "
                    f"forzo reconstruction.algorithm='backprojection' (config era {algorithm!r})"
                )
                algorithm = "backprojection"
            fallback_cfg = _load_yaml_file(Path(fallback_capture_cfg))
            radar_cfg = fallback_cfg.get("radar", {}) or {}
            geometry_c_m_s = _to_float("radar.c", _pick(radar_cfg.get("c"), 3e8), 3e8)
            geometry_fc_hz = _to_float("radar.fc", radar_cfg.get("fc"))
            bp_tx_global_m, bp_rx_global_m = cylindrical_capture_world_coordinates(
                stream_layout,
                fc_hz=float(geometry_fc_hz),
                c_m_s=float(geometry_c_m_s),
            )
            captures = tuple(stream_layout.cylindrical_captures)
            cylindrical_summary = cylindrical_run_summary(stream_layout)
            cylindrical_view = resolve_cylindrical_view(
                configured_view=bp_runtime_cfg.get("cylindrical_view"),
                legacy_plane=bp_runtime_cfg.get("cylindrical_plane"),
                captures=captures,
            )
            # Preserve the legacy status field while new consumers use the
            # complete section view.  It always denotes an XY plane.
            legacy_plane = bp_runtime_cfg.get("cylindrical_plane")
            if isinstance(legacy_plane, CylindricalPlane):
                cylindrical_plane = legacy_plane
            else:
                default_xy = _default_cylindrical_plane(captures)
                cylindrical_plane = CylindricalPlane(
                    x_min_m=float(cylindrical_view.bounds.x_min_m),
                    x_max_m=float(cylindrical_view.bounds.x_max_m),
                    y_min_m=float(cylindrical_view.bounds.y_min_m),
                    y_max_m=float(cylindrical_view.bounds.y_max_m),
                    z_m=float(
                        cylindrical_view.section.coordinate_m
                        if cylindrical_view.section.plane == "xy"
                        else default_xy.z_m
                    ),
                )
            geometry_source = "fixed_precalibrated_iwr1443_2tx4rx_world"
        elif geometry_mode == "legacy_linear":
            mimo_geometry = _resolve_offline_mimo_geometry(
                fallback_capture_cfg,
                tx_i=int(stream_layout.tx),
                rx_i=int(stream_layout.rx),
                tx_offsets_override_m=bp_runtime_cfg.get("tx_offsets_m"),
                rx_offsets_override_m=bp_runtime_cfg.get("rx_offsets_m"),
            )
            warning = mimo_geometry.get("warning", None)
            if warning:
                print(str(warning))
            geometry_source = str(mimo_geometry["geometry_source"])
        else:
            raise ValueError(f"geometry_mode offline non supportato: {geometry_mode!r}")

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

        msg = {
            "type": "data",
            "range_fft_shm_name": str(shm_range_fft.name),
            "range_fft_shape": tuple(int(x) for x in range_fft.shape),
            "range_fft_dtype": "complex64",
            "positions": stream_layout.positions.astype(np.int32, copy=False),
            "x_start_cfg": (
                None if reader.config.x_start is None else int(reader.config.x_start)
            ),
            "x_end_cfg": (
                None if reader.config.x_end is None else int(reader.config.x_end)
            ),
            "algorithm": str(algorithm),
            "geometry_mode": geometry_mode,
            "capture_ids": (
                stream_layout.positions.astype(np.int32, copy=False)
                if stream_layout.capture_ids is None
                else stream_layout.capture_ids.astype(np.int32, copy=False)
            ),
            "acquisition_indices": (
                None
                if stream_layout.acquisition_indices is None
                else stream_layout.acquisition_indices.astype(np.int32, copy=False)
            ),
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
        }
        if geometry_mode == "cylindrical_regular":
            assert bp_tx_global_m is not None and bp_rx_global_m is not None
            assert cylindrical_plane is not None
            assert cylindrical_view is not None and cylindrical_summary is not None
            msg["bp_tx_global_m"] = bp_tx_global_m
            msg["bp_rx_global_m"] = bp_rx_global_m
            msg["cylindrical_plane"] = cylindrical_plane
            msg["cylindrical_view"] = cylindrical_view
            msg["cylindrical_summary"] = cylindrical_summary
        else:
            assert mimo_geometry is not None
            if stream_layout.stage_positions_m is None:
                raise ValueError("Coordinate stage mancanti per la BP lineare")
            msg["bp_x_tx_ant_m"] = np.asarray(mimo_geometry["x_tx_ant_m"], dtype=np.float32)
            msg["bp_x_rx_ant_m"] = np.asarray(mimo_geometry["x_rx_ant_m"], dtype=np.float32)
            msg["bp_x_pos_m"] = np.asarray(stream_layout.stage_positions_m, dtype=np.float32)
        msg["bp_geometry_source"] = geometry_source
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
                "geometry_mode": geometry_mode,
                "geometry_source": geometry_source,
                "background_reference_enabled": bool(background_reference.enabled),
                "background_reference_dir": (
                    None if reference_layout is None else str(reference_layout.source_dir)
                ),
                "background_reference_frames": (
                    0 if reference_layout is None else int(reference_layout.n_frames_per_position)
                ),
                "background_reference_scale": float(background_reference.scale),
                "cylindrical_view": (
                    None
                    if cylindrical_view is None
                    else cylindrical_view_to_dict(cylindrical_view)
                ),
                "cylindrical_summary": (
                    None
                    if cylindrical_summary is None
                    else cylindrical_run_summary_to_dict(cylindrical_summary)
                ),
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
        geometry_mode = str(_pick(init_msg.get("geometry_mode"), "legacy_linear"))
        geometry_source = str(_pick(init_msg.get("bp_geometry_source"), "unknown"))
        n_ant_data = int(range_fft_data.shape[2])
        cylindrical_plane: CylindricalPlane | None = None
        cylindrical_view: CylindricalView | None = None
        cylindrical_summary: CylindricalRunSummary | None = None
        tx_global_m: np.ndarray | None = None
        rx_global_m: np.ndarray | None = None
        if geometry_mode == "cylindrical_regular":
            if algorithm != "backprojection":
                raise ValueError(
                    "La ricostruzione cilindrica supporta solo l'algoritmo backprojection"
                )
            tx_global_m = np.asarray(init_msg.get("bp_tx_global_m"), dtype=np.float32)
            rx_global_m = np.asarray(init_msg.get("bp_rx_global_m"), dtype=np.float32)
            expected_geometry_shape = (int(positions.size), int(n_ant_data), 3)
            if tx_global_m.shape != expected_geometry_shape or rx_global_m.shape != expected_geometry_shape:
                raise ValueError(
                    "bp_tx_global_m/bp_rx_global_m non coerenti con gli snapshot: "
                    f"atteso {expected_geometry_shape}, trovate {tx_global_m.shape}/{rx_global_m.shape}"
                )
            plane_raw = init_msg.get("cylindrical_plane")
            if not isinstance(plane_raw, CylindricalPlane):
                raise ValueError("cylindrical_plane mancante o non valida per la ricostruzione circular SAR")
            cylindrical_plane = plane_raw
            view_raw = init_msg.get("cylindrical_view")
            summary_raw = init_msg.get("cylindrical_summary")
            if not isinstance(view_raw, CylindricalView):
                raise ValueError("cylindrical_view mancante o non valida per la ricostruzione v2")
            if not isinstance(summary_raw, CylindricalRunSummary):
                raise ValueError("cylindrical_summary mancante o non valida per la ricostruzione v2")
            cylindrical_view = view_raw
            cylindrical_summary = summary_raw
            x_tx_ant_m = np.empty(0, dtype=np.float32)
            x_rx_ant_m = np.empty(0, dtype=np.float32)
        elif geometry_mode == "legacy_linear":
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
        else:
            raise ValueError(f"geometry_mode non supportato nel DSP offline: {geometry_mode!r}")
        n_ant_used = int(n_ant_data)
        n_bins_total = int(range_fft_data.shape[-1])
        dr_m = float(c_m_s) * float(fs_hz) / (2.0 * float(slope_hz_s) * float(nfft_range))
        if geometry_mode == "cylindrical_regular":
            assert cylindrical_view is not None
            section_x_min, section_x_max, section_y_min, section_y_max = cylindrical_section_bounds(
                cylindrical_view
            )
            home_viewport = _build_cylindrical_section_viewport(
                x_min_m=float(section_x_min),
                x_max_m=float(section_x_max),
                y_min_m=float(section_y_min),
                y_max_m=float(section_y_max),
                seq=0,
            )
        else:
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
                "geometry_mode": geometry_mode,
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
                "cylindrical_plane": (
                    None
                    if cylindrical_plane is None
                    else {
                        "x_min_m": float(cylindrical_plane.x_min_m),
                        "x_max_m": float(cylindrical_plane.x_max_m),
                        "y_min_m": float(cylindrical_plane.y_min_m),
                        "y_max_m": float(cylindrical_plane.y_max_m),
                        "z_m": float(cylindrical_plane.z_m),
                    }
                ),
                "cylindrical_view": (
                    None
                    if cylindrical_view is None
                    else cylindrical_view_to_dict(cylindrical_view)
                ),
                "cylindrical_summary": (
                    None
                    if cylindrical_summary is None
                    else cylindrical_run_summary_to_dict(cylindrical_summary)
                ),
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

                if geometry_mode == "cylindrical_regular":
                    assert cylindrical_view is not None and cylindrical_summary is not None
                    section_raw = cmd.get("cylindrical_section")
                    if section_raw is not None:
                        if not isinstance(section_raw, CylindricalSection):
                            raise ValueError("cylindrical_section non valida nel comando runtime")
                        validate_cylindrical_section(cylindrical_view, cylindrical_summary, section_raw)
                        cylindrical_view = replace(cylindrical_view, section=section_raw)
                        section_x_min, section_x_max, section_y_min, section_y_max = cylindrical_section_bounds(
                            cylindrical_view
                        )
                        home_viewport = _build_cylindrical_section_viewport(
                            x_min_m=float(section_x_min),
                            x_max_m=float(section_x_max),
                            y_min_m=float(section_y_min),
                            y_max_m=float(section_y_max),
                            seq=int(applied_viewport.seq) + 1,
                        )
                        # A changed orientation has a different physical
                        # domain; default to its full configured section.
                        applied_viewport = home_viewport
                        cylindrical_plane = cylindrical_xy_plane_if_active(cylindrical_view)
                else:
                    x_start_new = _pick(cmd.get("x_start"), x_start)
                    x_end_new = _pick(cmd.get("x_end"), x_end)
                    try:
                        x_start = max(pos_min, min(pos_max, int(x_start_new)))
                        x_end = max(pos_min, min(pos_max, int(x_end_new)))
                    except Exception:
                        pass
                    if x_end < x_start:
                        x_start, x_end = x_end, x_start
                if geometry_mode == "cylindrical_regular":
                    viewport_new = _cylindrical_viewport_from_cmd_payload(
                        cmd.get("viewport"),
                        home_viewport=home_viewport,
                        output_width=int(gui_w),
                        output_height=int(gui_h),
                    )
                else:
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
                (
                    None
                    if cylindrical_view is None
                    else (str(cylindrical_view.section.plane), float(cylindrical_view.section.coordinate_m))
                ),
                display_viewport_signature(applied_viewport),
            )
            if dirty or job_key != last_job_key or got_cmd:
                t0 = time.perf_counter()
                frame_meta: dict[str, Any] = {}
                if geometry_mode == "cylindrical_regular":
                    # A valid v2 run is exactly one full turn per height.
                    # Selecting arbitrary capture-ID subsets would violate
                    # that acquisition contract, so circular BP always uses
                    # every pose in acquisition order.
                    sel_idx = np.arange(int(positions.size), dtype=np.intp)
                    sel_key = tuple(int(value) for value in sel_idx.tolist())
                    assert cylindrical_view is not None
                    assert tx_global_m is not None and rx_global_m is not None
                    grid_key = (
                        "cylindrical_section",
                        str(cylindrical_view.section.plane),
                        float(cylindrical_view.section.coordinate_m),
                        display_viewport_signature(applied_viewport),
                        int(gui_h),
                        int(gui_w),
                    )
                    if grid_cache_key == grid_key and grid_cache is not None:
                        voxel_xyz = grid_cache
                    else:
                        horizontal_axis = np.linspace(
                            float(applied_viewport.x_min_m),
                            float(applied_viewport.x_max_m),
                            int(gui_w),
                            dtype=np.float32,
                        )
                        vertical_axis = np.linspace(
                            float(applied_viewport.y_min_m),
                            float(applied_viewport.y_max_m),
                            int(gui_h),
                            dtype=np.float32,
                        )
                        if cylindrical_view.section.plane == "xy":
                            voxel_xyz = xy_plane_voxel_grid(
                                horizontal_axis,
                                vertical_axis,
                                z_m=float(cylindrical_view.section.coordinate_m),
                            )
                        elif cylindrical_view.section.plane == "xz":
                            voxel_xyz = xz_plane_voxel_grid(
                                horizontal_axis,
                                vertical_axis,
                                y_m=float(cylindrical_view.section.coordinate_m),
                            )
                        else:
                            voxel_xyz = yz_plane_voxel_grid(
                                horizontal_axis,
                                vertical_axis,
                                x_m=float(cylindrical_view.section.coordinate_m),
                            )
                        voxel_xyz = voxel_xyz.astype(np.float32, copy=False)
                        grid_cache_key = grid_key
                        grid_cache = voxel_xyz

                    viewport_max_bin = int(n_bins_total)
                    range_fft_sel = np.asarray(range_fft_data[sel_idx, :, :, :viewport_max_bin], dtype=np.complex64)
                    power = _back_projection_power_mimo_geometry(
                        range_fft_sel,
                        tx_global_m[sel_idx],
                        rx_global_m[sel_idx],
                        voxel_xyz,
                        dr_m=float(dr_m),
                        fc_hz=float(fc_hz),
                        c_m_s=float(c_m_s),
                        max_bin=int(viewport_max_bin),
                        phase_sign=int(phase_sign_i),
                        residual_video_phase=int(residual_video_phase_i),
                        slope_hz_s=float(slope_hz_s),
                        chunk_size=16384,
                    )
                    img_db = _power_image_to_db(power)
                    frame_meta["cylindrical_view"] = cylindrical_view
                    doppler_bins_used = 1
                else:
                    sel_mask = (positions >= int(x_start)) & (positions <= int(x_end))
                    if not np.any(sel_mask):
                        sel_mask[:] = True
                    sel_idx = np.where(sel_mask)[0]
                    sel_key = tuple(int(v) for v in sel_idx.tolist())
                    x_pos_m_sel = x_pos_m_full[sel_idx]
                    # This bounds both legacy reconstruction paths to the active ROI.
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
                        "geometry_mode": geometry_mode,
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
                        "cylindrical_plane": (
                            None
                            if cylindrical_plane is None
                            else {
                                "x_min_m": float(cylindrical_plane.x_min_m),
                                "x_max_m": float(cylindrical_plane.x_max_m),
                                "y_min_m": float(cylindrical_plane.y_min_m),
                                "y_max_m": float(cylindrical_plane.y_max_m),
                                "z_m": float(cylindrical_plane.z_m),
                            }
                        ),
                        "cylindrical_view": (
                            None
                            if cylindrical_view is None
                            else cylindrical_view_to_dict(cylindrical_view)
                        ),
                        "cylindrical_summary": (
                            None
                            if cylindrical_summary is None
                            else cylindrical_run_summary_to_dict(cylindrical_summary)
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
    - reader process: elabora una posizione alla volta e pubblica snapshot
      Doppler-zero MIMO ``[pos, frame, ant, range]``;
    - dsp process: applica la ricostruzione richiesta e pubblica frame su
      double-buffer senza conservare un piano geometrico globale.
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
        cylindrical_section: CylindricalSection | None = None,
    ) -> None:
        if not self._started or self._cmd_q is None:
            return
        cmd = {
            "type": "update",
            "x_start": x_start,
            "x_end": x_end,
            "viewport": _viewport_to_cmd_payload(viewport),
            "cylindrical_section": cylindrical_section,
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
