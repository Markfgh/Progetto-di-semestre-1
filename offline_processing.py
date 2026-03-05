from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable

import numpy as np
import yaml

_CAPTURE_FILE_RE = re.compile(r"^capture_pos(-?\d+)\.bin$")


@dataclass(frozen=True)
class OfflineSARConfig:
    input_dir: Path
    x_start: int
    x_end: int
    x_step: int
    samples: int
    chirps: int
    rx: int
    tx: int
    frames_per_position: int | None = None

    @property
    def loops(self) -> int:
        return self.chirps // self.tx

    @property
    def virtual_ant(self) -> int:
        return self.tx * self.rx

    @property
    def bytes_per_frame(self) -> int:
        return self.chirps * self.samples * self.rx * 4

    @property
    def i16_per_frame(self) -> int:
        return self.bytes_per_frame // 2

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

        cap_fallback = cap_fallback_cfg.get("capture", {}) or {}
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

        samples = _to_int("capture.samples", _pick(cap_cfg.get("samples"), cap_fallback.get("samples")))
        chirps = _to_int("capture.chirps", _pick(cap_cfg.get("chirps"), cap_fallback.get("chirps")))
        rx = _to_int("capture.rx", _pick(cap_cfg.get("rx"), cap_fallback.get("rx")))
        tx = _to_int("capture.tx", _pick(cap_cfg.get("tx"), cap_fallback.get("tx")))

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
        if tx <= 0 or rx <= 0 or chirps <= 0 or samples <= 0:
            raise ValueError("capture: samples/chirps/rx/tx devono essere > 0")
        if chirps % tx != 0:
            raise ValueError("capture.chirps deve essere multiplo di capture.tx")
        if frames_per_position is not None and frames_per_position <= 0:
            raise ValueError("capture.frames_per_position deve essere > 0")

        return cls(
            input_dir=input_dir_path,
            x_start=x_start,
            x_end=x_end,
            x_step=x_step,
            samples=samples,
            chirps=chirps,
            rx=rx,
            tx=tx,
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

        n_pos = len(expected_positions)
        loops = self.config.loops
        n_ant = self.config.virtual_ant
        samples = self.config.samples
        frames_per_pos_cfg = self.config.frames_per_position

        pos_to_file = {pos: path for pos, path in pos_files}

        iq_cube: np.ndarray | None = None
        raw_by_pos: list[np.ndarray] = []
        n_frames_ref: int | None = None

        for pos_idx, pos in enumerate(expected_positions):
            raw_frames, n_frames = self.read_position(pos_to_file[pos])

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

            iq_frames = self._raw_to_iq(raw_frames, n_frames)
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
            bytes_per_frame=self.config.bytes_per_frame,
        )

    def read_position(self, file_path: str | Path) -> tuple[np.ndarray, int]:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File non trovato: {path}")

        file_size = path.stat().st_size
        bytes_per_frame = self.config.bytes_per_frame
        if file_size % bytes_per_frame != 0:
            raise ValueError(
                f"{path.name}: file_size={file_size} non multiplo di bytes_per_frame={bytes_per_frame}"
            )

        n_frames = file_size // bytes_per_frame
        if n_frames <= 0:
            raise ValueError(f"{path.name}: file vuoto")

        raw = np.fromfile(path, dtype=np.int16)
        expected_i16 = n_frames * self.config.i16_per_frame
        if raw.size != expected_i16:
            raise ValueError(
                f"{path.name}: int16 letti={raw.size}, attesi={expected_i16}"
            )

        raw_frames = raw.reshape(n_frames, self.config.i16_per_frame)
        return raw_frames, int(n_frames)

    def _raw_to_iq(self, raw_frames: np.ndarray, n_frames: int) -> np.ndarray:
        cfg = self.config

        flat_i16 = raw_frames.reshape(-1)
        iq_block = 2 * cfg.rx
        if flat_i16.size % iq_block != 0:
            raise ValueError(
                f"Stream int16 non allineato a blocchi IQ da {iq_block} canali (I...Q...)"
            )

        block_view = flat_i16.reshape(-1, iq_block)
        complex_flat = np.empty(block_view.shape[0] * cfg.rx, dtype=np.complex64)
        complex_flat.real = block_view[:, : cfg.rx].reshape(-1)
        complex_flat.imag = block_view[:, cfg.rx : iq_block].reshape(-1)

        expected_complex = n_frames * cfg.chirps * cfg.samples * cfg.rx
        if complex_flat.size != expected_complex:
            raise ValueError(
                f"Campioni complessi={complex_flat.size}, attesi={expected_complex}"
            )

        data_5d = complex_flat.reshape(n_frames, cfg.loops, cfg.tx, cfg.samples, cfg.rx)
        iq_frames = data_5d.transpose(0, 1, 2, 4, 3).reshape(
            n_frames,
            cfg.loops,
            cfg.virtual_ant,
            cfg.samples,
        )
        return iq_frames

    def _scan_position_files(self, source_dir: Path) -> list[tuple[int, Path]]:
        pos_files: list[tuple[int, Path]] = []
        for p in source_dir.glob("*.bin"):
            match = _CAPTURE_FILE_RE.match(p.name)
            if match is None:
                continue
            pos_files.append((int(match.group(1)), p))

        if not pos_files:
            raise FileNotFoundError(f"Nessun file capture_pos*.bin trovato in {source_dir}")

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

        direct = list(input_dir.glob("capture_pos*.bin"))
        if direct:
            return input_dir

        run_dirs = sorted([p for p in input_dir.glob("run_*") if p.is_dir()], key=lambda p: p.name)
        for run_dir in reversed(run_dirs):
            if any(run_dir.glob("capture_pos*.bin")):
                return run_dir

        raise FileNotFoundError(
            f"Nessun capture_pos*.bin trovato in {input_dir} (neanche dentro run_*)"
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
