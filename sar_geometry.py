"""Pure geometry primitives for cylindrical 2-TX x 4-RX MIMO-SAR captures.

All coordinates are expressed in metres in a right-handed world frame.  The
radar body frame is fixed as follows:

* ``+X``: along the physical ULA;
* ``+Y``: boresight, directed towards the scene centre for a circular scan;
* ``+Z``: upwards.

``CylindricalCapture.rotation_world_from_body`` is an active rotation whose
columns are the body-frame unit axes expressed in the world frame.  Hence a
local element coordinate is transformed with ``p_world = p_radar + R @
p_body``.

This module deliberately contains no hardware, file, DSP, quaternion or
trajectory-provider logic.  It is safe to use from capture/header code as well
as offline reconstruction code.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np


_TWO_PI = float(2.0 * np.pi)


def _readonly_array(value: Any, *, shape: tuple[int, ...], field_name: str) -> np.ndarray:
    """Return an immutable finite float64 array with an exact expected shape."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} non valido") from exc
    if array.shape != shape:
        raise ValueError(f"{field_name} deve avere shape {shape}, trovata {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} contiene valori non finiti")
    out = np.array(array, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out


def _readonly_index_array(value: Any, *, shape: tuple[int, ...], field_name: str) -> np.ndarray:
    """Return an immutable integer array, rejecting lossy float conversion."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} non valido") from exc
    if raw.shape != shape:
        raise ValueError(f"{field_name} deve avere shape {shape}, trovata {raw.shape}")
    if raw.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{field_name} deve contenere indici interi")
    out = np.array(raw, dtype=np.int64, copy=True)
    out.setflags(write=False)
    return out


def _nonnegative_index(value: Any, *, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{field_name} deve essere un intero non negativo")
    out = int(value)
    if out < 0:
        raise ValueError(f"{field_name} deve essere un intero non negativo")
    return out


def _positive_index(value: Any, *, field_name: str) -> int:
    out = _nonnegative_index(value, field_name=field_name)
    if out <= 0:
        raise ValueError(f"{field_name} deve essere maggiore di zero")
    return out


def _finite_scalar(value: Any, *, field_name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} non valido: {value!r}") from exc
    if not np.isfinite(out):
        raise ValueError(f"{field_name} deve essere finito")
    return out


def _canonical_tx_major_rx_minor_pairing(n_tx: int, n_rx: int) -> np.ndarray:
    pairing = np.column_stack(
        (
            np.repeat(np.arange(n_tx, dtype=np.int64), n_rx),
            np.tile(np.arange(n_rx, dtype=np.int64), n_tx),
        )
    )
    pairing.setflags(write=False)
    return pairing


@dataclass(frozen=True)
class CylindricalCapture:
    """Metadata for one capture acquired on a fixed-radius cylindrical path.

    ``height_m`` is the vertical offset from ``scene_center_m[2]``.  Therefore
    the radar reference-point position is::

        scene_center_m + [radius_m*cos(azimuth),
                          radius_m*sin(azimuth),
                          height_m]

    ``capture_id`` identifies the stored capture, while ``acquisition_index``
    is its temporal order.  They intentionally need not have the same value.
    """

    capture_id: int
    acquisition_index: int
    angle_index: int
    height_index: int
    azimuth_rad: float
    height_m: float
    radius_m: float
    scene_center_m: np.ndarray
    angle_count: int
    height_count: int | None = None

    def __post_init__(self) -> None:
        capture_id = _nonnegative_index(self.capture_id, field_name="capture_id")
        acquisition_index = _nonnegative_index(
            self.acquisition_index,
            field_name="acquisition_index",
        )
        angle_count = _positive_index(self.angle_count, field_name="angle_count")
        angle_index = _nonnegative_index(self.angle_index, field_name="angle_index")
        if angle_index >= angle_count:
            raise ValueError("angle_index deve essere minore di angle_count")
        height_index = _nonnegative_index(self.height_index, field_name="height_index")

        height_count: int | None
        if self.height_count is None:
            height_count = None
        else:
            height_count = _positive_index(self.height_count, field_name="height_count")
            if height_index >= height_count:
                raise ValueError("height_index deve essere minore di height_count")

        azimuth_rad = _finite_scalar(self.azimuth_rad, field_name="azimuth_rad")
        if not 0.0 <= azimuth_rad < _TWO_PI:
            raise ValueError("azimuth_rad deve appartenere all'intervallo [0, 2*pi)")
        height_m = _finite_scalar(self.height_m, field_name="height_m")
        radius_m = _finite_scalar(self.radius_m, field_name="radius_m")
        if radius_m <= 0.0:
            raise ValueError("radius_m deve essere maggiore di zero")

        object.__setattr__(self, "capture_id", capture_id)
        object.__setattr__(self, "acquisition_index", acquisition_index)
        object.__setattr__(self, "angle_index", angle_index)
        object.__setattr__(self, "height_index", height_index)
        object.__setattr__(self, "angle_count", angle_count)
        object.__setattr__(self, "height_count", height_count)
        object.__setattr__(self, "azimuth_rad", azimuth_rad)
        object.__setattr__(self, "height_m", height_m)
        object.__setattr__(self, "radius_m", radius_m)
        object.__setattr__(
            self,
            "scene_center_m",
            _readonly_array(self.scene_center_m, shape=(3,), field_name="scene_center_m"),
        )

    @property
    def position_world_m(self) -> np.ndarray:
        """Radar reference-point position in the world frame, shape ``[3]``."""
        azimuth = self.azimuth_rad
        out = np.array(
            (
                self.scene_center_m[0] + self.radius_m * np.cos(azimuth),
                self.scene_center_m[1] + self.radius_m * np.sin(azimuth),
                self.scene_center_m[2] + self.height_m,
            ),
            dtype=np.float64,
        )
        out.setflags(write=False)
        return out

    @property
    def rotation_world_from_body(self) -> np.ndarray:
        """Right-handed ``R_world_from_body`` matrix, shape ``[3, 3]``.

        Its columns are respectively the ULA tangent, boresight toward the
        scene centre, and upward axes.  At azimuth zero body ``+X`` maps to
        world ``+Y`` and body ``+Y`` maps to world ``-X``.
        """
        azimuth = self.azimuth_rad
        tangent = (-np.sin(azimuth), np.cos(azimuth), 0.0)
        inward = (-np.cos(azimuth), -np.sin(azimuth), 0.0)
        out = np.asarray(
            (
                (tangent[0], inward[0], 0.0),
                (tangent[1], inward[1], 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        out.setflags(write=False)
        return out

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible source metadata, excluding derived values."""
        return {
            "capture_id": self.capture_id,
            "acquisition_index": self.acquisition_index,
            "angle_index": self.angle_index,
            "height_index": self.height_index,
            "azimuth_rad": self.azimuth_rad,
            "height_m": self.height_m,
            "radius_m": self.radius_m,
            "scene_center_m": self.scene_center_m.tolist(),
            "angle_count": self.angle_count,
            "height_count": self.height_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CylindricalCapture":
        """Create a capture from the JSON-compatible output of :meth:`to_dict`."""
        if not isinstance(value, Mapping):
            raise ValueError("metadata cylindrical capture non valido")
        required = (
            "capture_id",
            "acquisition_index",
            "angle_index",
            "height_index",
            "azimuth_rad",
            "height_m",
            "radius_m",
            "scene_center_m",
            "angle_count",
        )
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"metadata cylindrical capture senza campi: {', '.join(missing)}")
        return cls(
            capture_id=value["capture_id"],
            acquisition_index=value["acquisition_index"],
            angle_index=value["angle_index"],
            height_index=value["height_index"],
            azimuth_rad=value["azimuth_rad"],
            height_m=value["height_m"],
            radius_m=value["radius_m"],
            scene_center_m=value["scene_center_m"],
            angle_count=value["angle_count"],
            height_count=value.get("height_count"),
        )


@dataclass(frozen=True)
class ArrayGeometry3D:
    """Physical local TX/RX element geometry and TDM-MIMO channel pairing.

    The only accepted channel order is explicit TX-major/RX-minor pairing:
    ``(tx0,rx0), ..., (tx0,rxN), (tx1,rx0), ...``.  This is the order of the
    current 2 TX x 4 RX TDM-MIMO flattening and deliberately refers to physical
    TX/RX coordinates, not to virtual phase centres.
    """

    tx_local_m: np.ndarray
    rx_local_m: np.ndarray
    pairing_tx_rx: np.ndarray | None = None
    array_id: str = "array_geometry_3d"

    def __post_init__(self) -> None:
        tx_raw = np.asarray(self.tx_local_m)
        rx_raw = np.asarray(self.rx_local_m)
        if tx_raw.ndim != 2 or tx_raw.shape[0] <= 0 or tx_raw.shape[1:] != (3,):
            raise ValueError("tx_local_m deve avere shape [n_tx, 3] con n_tx > 0")
        if rx_raw.ndim != 2 or rx_raw.shape[0] <= 0 or rx_raw.shape[1:] != (3,):
            raise ValueError("rx_local_m deve avere shape [n_rx, 3] con n_rx > 0")
        tx_local_m = _readonly_array(
            self.tx_local_m,
            shape=(int(tx_raw.shape[0]), 3),
            field_name="tx_local_m",
        )
        rx_local_m = _readonly_array(
            self.rx_local_m,
            shape=(int(rx_raw.shape[0]), 3),
            field_name="rx_local_m",
        )
        expected_pairing = _canonical_tx_major_rx_minor_pairing(
            int(tx_local_m.shape[0]),
            int(rx_local_m.shape[0]),
        )
        if self.pairing_tx_rx is None:
            pairing_tx_rx = expected_pairing
        else:
            pairing_tx_rx = _readonly_index_array(
                self.pairing_tx_rx,
                shape=expected_pairing.shape,
                field_name="pairing_tx_rx",
            )
            if not np.array_equal(pairing_tx_rx, expected_pairing):
                raise ValueError("pairing_tx_rx deve usare l'ordine TX-major/RX-minor")

        if not isinstance(self.array_id, str) or not self.array_id.strip():
            raise ValueError("array_id deve essere una stringa non vuota")
        object.__setattr__(self, "tx_local_m", tx_local_m)
        object.__setattr__(self, "rx_local_m", rx_local_m)
        object.__setattr__(self, "pairing_tx_rx", pairing_tx_rx)
        object.__setattr__(self, "array_id", self.array_id.strip())

    @property
    def n_tx(self) -> int:
        return int(self.tx_local_m.shape[0])

    @property
    def n_rx(self) -> int:
        return int(self.rx_local_m.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.pairing_tx_rx.shape[0])

    @property
    def paired_tx_local_m(self) -> np.ndarray:
        """Physical TX coordinates expanded in TDM-MIMO channel order."""
        out = np.array(self.tx_local_m[self.pairing_tx_rx[:, 0]], dtype=np.float64, copy=True)
        out.setflags(write=False)
        return out

    @property
    def paired_rx_local_m(self) -> np.ndarray:
        """Physical RX coordinates expanded in TDM-MIMO channel order."""
        out = np.array(self.rx_local_m[self.pairing_tx_rx[:, 1]], dtype=np.float64, copy=True)
        out.setflags(write=False)
        return out

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible local geometry and the explicit pairing."""
        return {
            "array_id": self.array_id,
            "tx_local_m": self.tx_local_m.tolist(),
            "rx_local_m": self.rx_local_m.tolist(),
            "pairing_tx_rx": self.pairing_tx_rx.tolist(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArrayGeometry3D":
        """Create a geometry from the JSON-compatible output of :meth:`to_dict`."""
        if not isinstance(value, Mapping):
            raise ValueError("metadata array geometry non valido")
        required = ("tx_local_m", "rx_local_m", "pairing_tx_rx")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"metadata array geometry senza campi: {', '.join(missing)}")
        return cls(
            tx_local_m=value["tx_local_m"],
            rx_local_m=value["rx_local_m"],
            pairing_tx_rx=value["pairing_tx_rx"],
            array_id=value.get("array_id", "array_geometry_3d"),
        )


def default_iwr1443_2tx4rx_geometry(
    *,
    fc_hz: float | None = None,
    c_m_s: float | None = None,
    wavelength_m: float | None = None,
) -> ArrayGeometry3D:
    """Build the physical 2-TX x 4-RX ULA geometry used by the legacy BP.

    Exactly one wavelength source must be supplied: either ``wavelength_m`` or
    the pair ``fc_hz``/``c_m_s``.  The legacy implementation expresses the
    default layout as TX ``[0, 2] lambda`` and RX ``[0, .5, 1, 1.5] lambda``,
    then centres both physical arrays at the mean bistatic phase centre.  The
    same float32 arithmetic is intentionally retained here so that the local
    X offsets match the existing linear backprojection defaults.
    """
    has_wavelength = wavelength_m is not None
    has_rf_pair = fc_hz is not None or c_m_s is not None
    if has_wavelength and has_rf_pair:
        raise ValueError("usa wavelength_m oppure fc_hz/c_m_s, non entrambi")
    if has_wavelength:
        wavelength = _finite_scalar(wavelength_m, field_name="wavelength_m")
    else:
        if fc_hz is None or c_m_s is None:
            raise ValueError("specifica wavelength_m oppure entrambi fc_hz e c_m_s")
        fc = _finite_scalar(fc_hz, field_name="fc_hz")
        c = _finite_scalar(c_m_s, field_name="c_m_s")
        if fc <= 0.0 or c <= 0.0:
            raise ValueError("fc_hz e c_m_s devono essere maggiori di zero")
        wavelength = c / fc
    if wavelength <= 0.0:
        raise ValueError("wavelength_m deve essere maggiore di zero")

    wavelength32 = np.float32(wavelength)
    tx_base = np.asarray((0.0, 2.0), dtype=np.float32) * wavelength32
    rx_base = np.asarray((0.0, 0.5, 1.0, 1.5), dtype=np.float32) * wavelength32
    pairing = _canonical_tx_major_rx_minor_pairing(2, 4)
    paired_tx = tx_base[pairing[:, 0]]
    paired_rx = rx_base[pairing[:, 1]]
    centre = np.float32(
        np.mean((paired_tx + paired_rx).astype(np.float32, copy=False), dtype=np.float32)
        * np.float32(0.5)
    )
    tx_x = (tx_base - centre).astype(np.float32, copy=False)
    rx_x = (rx_base - centre).astype(np.float32, copy=False)
    tx_local_m = np.column_stack((tx_x, np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)))
    rx_local_m = np.column_stack((rx_x, np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32)))
    return ArrayGeometry3D(
        tx_local_m=tx_local_m,
        rx_local_m=rx_local_m,
        pairing_tx_rx=pairing,
        array_id="iwr1443_2tx4rx_azimuth_v1",
    )


def transform_element_coordinates(
    captures: Sequence[CylindricalCapture],
    array_geometry: ArrayGeometry3D,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform local physical TX/RX coordinates into world coordinates.

    The returned arrays have shape ``[P, A, 3]``, where ``P`` is the input
    capture count and ``A`` is the explicit TDM-MIMO channel count.  Input
    order is preserved; callers should order captures by ``acquisition_index``
    when temporal order is required.
    """
    if not isinstance(array_geometry, ArrayGeometry3D):
        raise TypeError("array_geometry deve essere un ArrayGeometry3D")
    captures_tuple = tuple(captures)
    for capture in captures_tuple:
        if not isinstance(capture, CylindricalCapture):
            raise TypeError("captures deve contenere solo CylindricalCapture")

    n_capture = len(captures_tuple)
    n_channel = array_geometry.n_channels
    tx_global = np.empty((n_capture, n_channel, 3), dtype=np.float64)
    rx_global = np.empty((n_capture, n_channel, 3), dtype=np.float64)
    tx_local = array_geometry.paired_tx_local_m
    rx_local = array_geometry.paired_rx_local_m
    for capture_index, capture in enumerate(captures_tuple):
        rotation = capture.rotation_world_from_body
        position = capture.position_world_m
        tx_global[capture_index] = (rotation @ tx_local.T).T + position
        rx_global[capture_index] = (rotation @ rx_local.T).T + position
    return tx_global, rx_global


def _axis_1d(value: Any, *, field_name: str) -> np.ndarray:
    """Validate a finite non-empty world-coordinate axis."""
    try:
        axis = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} non valido") from exc
    if axis.ndim != 1 or axis.size <= 0:
        raise ValueError(f"{field_name} deve essere un vettore 1D non vuoto")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{field_name} contiene valori non finiti")
    return axis


def xy_plane_voxel_grid(
    x_axis_m: Any,
    y_axis_m: Any,
    *,
    z_m: float,
) -> np.ndarray:
    """Build a fixed-height XY reconstruction grid with shape ``[Y, X, 3]``.

    This is the regular circular-SAR image plane.  It only constructs world
    coordinates; :func:`offline_dsp.back_projection_power_mimo_geometry`
    performs the reconstruction from physical TX/RX locations.
    """
    x_axis = _axis_1d(x_axis_m, field_name="x_axis_m")
    y_axis = _axis_1d(y_axis_m, field_name="y_axis_m")
    z_value = _finite_scalar(z_m, field_name="z_m")
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="xy")
    out = np.empty(x_grid.shape + (3,), dtype=np.float64)
    out[..., 0] = x_grid
    out[..., 1] = y_grid
    out[..., 2] = z_value
    return out


def xz_plane_voxel_grid(
    x_axis_m: Any,
    z_axis_m: Any,
    *,
    y_m: float,
) -> np.ndarray:
    """Build a fixed-Y XZ reconstruction grid with shape ``[Z, X, 3]``.

    Rows follow the vertical world ``Z`` axis and columns follow world ``X``.
    The function only creates voxel coordinates for the generalized bistatic
    backprojection kernel; it never creates a volumetric reconstruction.
    """
    x_axis = _axis_1d(x_axis_m, field_name="x_axis_m")
    z_axis = _axis_1d(z_axis_m, field_name="z_axis_m")
    y_value = _finite_scalar(y_m, field_name="y_m")
    x_grid, z_grid = np.meshgrid(x_axis, z_axis, indexing="xy")
    out = np.empty(x_grid.shape + (3,), dtype=np.float64)
    out[..., 0] = x_grid
    out[..., 1] = y_value
    out[..., 2] = z_grid
    return out


def yz_plane_voxel_grid(
    y_axis_m: Any,
    z_axis_m: Any,
    *,
    x_m: float,
) -> np.ndarray:
    """Build a fixed-X YZ reconstruction grid with shape ``[Z, Y, 3]``.

    Rows follow the vertical world ``Z`` axis and columns follow world ``Y``.
    """
    y_axis = _axis_1d(y_axis_m, field_name="y_axis_m")
    z_axis = _axis_1d(z_axis_m, field_name="z_axis_m")
    x_value = _finite_scalar(x_m, field_name="x_m")
    y_grid, z_grid = np.meshgrid(y_axis, z_axis, indexing="xy")
    out = np.empty(y_grid.shape + (3,), dtype=np.float64)
    out[..., 0] = x_value
    out[..., 1] = y_grid
    out[..., 2] = z_grid
    return out


def xyz_volume_voxel_grid(
    x_axis_m: Any,
    y_axis_m: Any,
    z_axis_m: Any,
) -> np.ndarray:
    """Build a cylindrical-SAR volume grid with shape ``[Z, Y, X, 3]``.

    The explicit axis order prevents an accidental reuse of the historical
    two-dimensional linear-SAR grid.  The function is deliberately pure and
    does not allocate a renderer, writer, or GUI resource.
    """
    x_axis = _axis_1d(x_axis_m, field_name="x_axis_m")
    y_axis = _axis_1d(y_axis_m, field_name="y_axis_m")
    z_axis = _axis_1d(z_axis_m, field_name="z_axis_m")
    z_grid, y_grid, x_grid = np.meshgrid(z_axis, y_axis, x_axis, indexing="ij")
    out = np.empty(x_grid.shape + (3,), dtype=np.float64)
    out[..., 0] = x_grid
    out[..., 1] = y_grid
    out[..., 2] = z_grid
    return out


__all__ = [
    "ArrayGeometry3D",
    "CylindricalCapture",
    "default_iwr1443_2tx4rx_geometry",
    "transform_element_coordinates",
    "xy_plane_voxel_grid",
    "xz_plane_voxel_grid",
    "yz_plane_voxel_grid",
    "xyz_volume_voxel_grid",
]
