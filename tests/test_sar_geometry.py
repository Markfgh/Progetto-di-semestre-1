from __future__ import annotations

import numpy as np
import pytest

from offline_dsp import build_mimo_geometry
from sar_geometry import (
    ArrayGeometry3D,
    CylindricalCapture,
    default_iwr1443_2tx4rx_geometry,
    transform_element_coordinates,
    xy_plane_voxel_grid,
    xz_plane_voxel_grid,
    yz_plane_voxel_grid,
    xyz_volume_voxel_grid,
)


def _capture(*, azimuth_rad: float = 0.0, scene_center_m: tuple[float, float, float] = (1.0, 2.0, 3.0)) -> CylindricalCapture:
    return CylindricalCapture(
        capture_id=42,
        acquisition_index=7,
        angle_index=0,
        height_index=1,
        azimuth_rad=azimuth_rad,
        height_m=0.4,
        radius_m=2.5,
        scene_center_m=scene_center_m,
        angle_count=8,
        height_count=3,
    )


def test_cylindrical_capture_position_and_right_handed_inward_frame() -> None:
    capture = _capture(azimuth_rad=0.0)

    np.testing.assert_allclose(capture.position_world_m, [3.5, 2.0, 3.4])
    rotation = capture.rotation_world_from_body
    np.testing.assert_allclose(rotation[:, 0], [0.0, 1.0, 0.0])  # body +X tangent
    np.testing.assert_allclose(rotation[:, 1], [-1.0, 0.0, 0.0])  # body +Y inward
    np.testing.assert_allclose(rotation[:, 2], [0.0, 0.0, 1.0])  # body +Z up
    np.testing.assert_allclose(np.cross(rotation[:, 0], rotation[:, 1]), rotation[:, 2])
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_cylindrical_capture_quarter_turn_moves_and_rotates_inward() -> None:
    capture = _capture(azimuth_rad=np.pi / 2.0, scene_center_m=(0.0, 0.0, 0.0))

    np.testing.assert_allclose(capture.position_world_m, [0.0, 2.5, 0.4], atol=1e-14)
    rotation = capture.rotation_world_from_body
    np.testing.assert_allclose(rotation[:, 0], [-1.0, 0.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(rotation[:, 1], [0.0, -1.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(rotation[:, 2], [0.0, 0.0, 1.0], atol=1e-14)


def test_default_iwr1443_geometry_matches_legacy_bistatic_offsets() -> None:
    fc_hz = 77.0e9
    c_m_s = 3.0e8
    geometry = default_iwr1443_2tx4rx_geometry(fc_hz=fc_hz, c_m_s=c_m_s)
    old_tx, old_rx = build_mimo_geometry(n_tx=2, n_rx=4, fc_hz=fc_hz, c_m_s=c_m_s)

    np.testing.assert_array_equal(geometry.pairing_tx_rx, np.asarray(
        [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [1, 3]],
        dtype=np.int64,
    ))
    np.testing.assert_array_equal(geometry.paired_tx_local_m[:, 0].astype(np.float32), old_tx)
    np.testing.assert_array_equal(geometry.paired_rx_local_m[:, 0].astype(np.float32), old_rx)
    np.testing.assert_array_equal(geometry.paired_tx_local_m[:, 1:], np.zeros((8, 2)))
    np.testing.assert_array_equal(geometry.paired_rx_local_m[:, 1:], np.zeros((8, 2)))


def test_transform_element_coordinates_keeps_tx_major_rx_minor_order() -> None:
    geometry = ArrayGeometry3D(
        tx_local_m=np.asarray([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
        rx_local_m=np.asarray([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 4.0, 0.0]]),
    )
    # At 3*pi/2 the cylindrical body frame is exactly aligned with world XYZ.
    capture = CylindricalCapture(
        capture_id=9,
        acquisition_index=2,
        angle_index=6,
        height_index=0,
        azimuth_rad=3.0 * np.pi / 2.0,
        height_m=1.0,
        radius_m=2.0,
        scene_center_m=[5.0, -2.0, 3.0],
        angle_count=8,
    )

    tx_global, rx_global = transform_element_coordinates([capture], geometry)

    assert tx_global.shape == (1, 8, 3)
    assert rx_global.shape == (1, 8, 3)
    position = capture.position_world_m
    np.testing.assert_allclose(tx_global[0] - position, geometry.paired_tx_local_m, atol=1e-14)
    np.testing.assert_allclose(rx_global[0] - position, geometry.paired_rx_local_m, atol=1e-14)
    np.testing.assert_array_equal(
        geometry.pairing_tx_rx,
        [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [1, 3]],
    )


def test_metadata_round_trip_and_strict_cylindrical_bounds() -> None:
    capture = _capture(azimuth_rad=0.25)
    geometry = default_iwr1443_2tx4rx_geometry(wavelength_m=0.004)

    restored_capture = CylindricalCapture.from_dict(capture.to_dict())
    restored_geometry = ArrayGeometry3D.from_dict(geometry.to_dict())
    np.testing.assert_allclose(restored_capture.position_world_m, capture.position_world_m)
    np.testing.assert_array_equal(restored_geometry.pairing_tx_rx, geometry.pairing_tx_rx)
    with pytest.raises(ValueError, match=r"\[0, 2\*pi\)"):
        _capture(azimuth_rad=2.0 * np.pi)


def test_fixed_xy_xz_yz_and_xyz_grids_have_explicit_world_axis_order() -> None:
    plane = xy_plane_voxel_grid([-1.0, 2.0], [3.0, 5.0, 7.0], z_m=0.25)
    xz_plane = xz_plane_voxel_grid([-1.0, 2.0], [-0.5, 0.25], y_m=7.0)
    yz_plane = yz_plane_voxel_grid([3.0, 5.0, 7.0], [-0.5, 0.25], x_m=-1.0)
    volume = xyz_volume_voxel_grid([-1.0, 2.0], [3.0, 5.0, 7.0], [-0.5, 0.25])

    assert plane.shape == (3, 2, 3)
    np.testing.assert_allclose(plane[2, 1], [2.0, 7.0, 0.25])
    assert xz_plane.shape == (2, 2, 3)
    np.testing.assert_allclose(xz_plane[1, 1], [2.0, 7.0, 0.25])
    assert yz_plane.shape == (2, 3, 3)
    np.testing.assert_allclose(yz_plane[1, 2], [-1.0, 7.0, 0.25])
    assert volume.shape == (2, 3, 2, 3)
    np.testing.assert_allclose(volume[1, 2, 1], [2.0, 7.0, 0.25])
