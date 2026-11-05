"""PointCloud2 ↔ numpy conversion, voxel downsample, height filter, ground removal."""

from __future__ import annotations

import numpy as np
import struct
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


# PointField datatype constants → (struct format char, byte size)
_PFTYPE_TO_STRUCT = {
    PointField.INT8: ("b", 1),
    PointField.UINT8: ("B", 1),
    PointField.INT16: ("h", 2),
    PointField.UINT16: ("H", 2),
    PointField.INT32: ("i", 4),
    PointField.UINT32: ("I", 4),
    PointField.FLOAT32: ("f", 4),
    PointField.FLOAT64: ("d", 8),
}


def pointcloud2_to_xyz(msg: PointCloud2) -> np.ndarray:
    """Convert PointCloud2 to (N, 3) float32 array [x, y, z].

    Assumes x/y/z are the first three FLOAT32 fields (Ouster convention).
    Falls back to generic field parsing if layout differs.
    """
    point_step = msg.point_step
    n_points = msg.width * msg.height
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(n_points, point_step)

    # Fast path: x at 0, y at 4, z at 8 as float32
    x_off = y_off = z_off = None
    for f in msg.fields:
        if f.name == "x":
            x_off = f.offset
        elif f.name == "y":
            y_off = f.offset
        elif f.name == "z":
            z_off = f.offset
    if x_off is None or y_off is None or z_off is None:
        raise ValueError("PointCloud2 missing x/y/z fields")

    x = np.ascontiguousarray(data[:, x_off : x_off + 4]).view(np.float32).reshape(-1)
    y = np.ascontiguousarray(data[:, y_off : y_off + 4]).view(np.float32).reshape(-1)
    z = np.ascontiguousarray(data[:, z_off : z_off + 4]).view(np.float32).reshape(-1)
    pts = np.column_stack([x, y, z])

    # Remove NaN / inf
    valid = np.isfinite(pts).all(axis=1)
    return pts[valid].astype(np.float32)


def xyz_to_pointcloud2(
    points: np.ndarray, header: Header
) -> PointCloud2:
    """Convert (N, 3) float32 array to a minimal PointCloud2."""
    pts = np.ascontiguousarray(points, dtype=np.float32)
    n = pts.shape[0]

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * n
    msg.data = pts.tobytes()
    msg.is_dense = True
    return msg


def height_filter(points: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    """Keep points within [z_min, z_max]."""
    mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    return points[mask]


def voxel_downsample_range_split(
    points: np.ndarray,
    range_split_m: float,
    voxel_near: float,
    voxel_far: float,
    skip_far_downsample: bool = False,
) -> np.ndarray:
    """Near/far split: finer voxels close in, keep more points far away."""
    if len(points) == 0 or range_split_m <= 0:
        return voxel_downsample(points, voxel_near)
    r = np.linalg.norm(points[:, :2], axis=1)
    near = points[r < range_split_m]
    far = points[r >= range_split_m]
    parts = []
    if len(near):
        parts.append(voxel_downsample(near, voxel_near))
    if len(far):
        if skip_far_downsample:
            parts.append(far)
        else:
            parts.append(voxel_downsample(far, voxel_far))
    if not parts:
        return points
    return np.vstack(parts).astype(np.float32)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Simple voxel grid downsample — one point per voxel (mean)."""
    if voxel_size <= 0 or len(points) == 0:
        return points
    quantized = np.floor(points[:, :3] / voxel_size).astype(np.int32)
    _, unique_idx, inverse = np.unique(
        quantized, axis=0, return_index=True, return_inverse=True
    )
    # One representative per voxel — use mean for accuracy
    n_voxels = len(unique_idx)
    sums = np.zeros((n_voxels, 3), dtype=np.float64)
    counts = np.zeros(n_voxels, dtype=np.int32)
    np.add.at(sums, inverse, points[:, :3])
    np.add.at(counts, inverse, 1)
    return (sums / counts[:, None]).astype(np.float32)


def remove_ground_grid(
    points: np.ndarray,
    cell: float = 1.0,
    delta: float = 0.25,
    z_global_max: float | None = None,
) -> np.ndarray:
    """Local-min ground removal.

    For each cell (cell x cell, XY plane), estimate ground z as the cell's
    z-min. Points with z > cell_zmin + delta are kept as non-ground. Handles
    sloped/uneven roads where a single global height threshold fails.

    Parameters
    ----------
    cell : XY cell size in meters.
    delta : a point is non-ground if its z exceeds cell ground by this much (m).
    z_global_max : optional hard cap; points with z > z_global_max are dropped
                   (filters tall vegetation / overhangs).
    """
    if len(points) == 0:
        return points
    qx = np.floor(points[:, 0] / cell).astype(np.int64)
    qy = np.floor(points[:, 1] / cell).astype(np.int64)
    # 32-bit packing with offset to allow negatives
    keys = (qx + (1 << 20)) * (1 << 22) + (qy + (1 << 20))
    order = np.argsort(keys, kind="stable")
    keys_s = keys[order]
    z_s = points[order, 2]
    uniq_keys, start = np.unique(keys_s, return_index=True)
    cell_zmin = np.minimum.reduceat(z_s, start)
    # Map every original point's key index back to its cell's zmin
    inv = np.searchsorted(uniq_keys, keys)
    is_nonground = points[:, 2] > (cell_zmin[inv] + delta)
    if z_global_max is not None:
        is_nonground &= points[:, 2] <= z_global_max
    return points[is_nonground]


def remove_ground(
    points: np.ndarray, method: str = "height_threshold", **kwargs
) -> np.ndarray:
    """Remove ground points.

    Methods:
      - height_threshold: global z <= ground_z_max (fast, fails on slopes).
      - grid_local_min:   per-cell local-min ground (robust to road slope).
    """
    if method == "height_threshold":
        ground_z_max = kwargs.get("ground_z_max", -1.4)
        return points[points[:, 2] > ground_z_max]
    if method == "grid_local_min":
        return remove_ground_grid(
            points,
            cell=kwargs.get("cell", 1.0),
            delta=kwargs.get("delta", 0.25),
            z_global_max=kwargs.get("z_global_max"),
        )
    raise ValueError(f"unknown ground removal method: {method}")
