"""ROI polygon utilities — 2D point-in-polygon crop on the XY plane."""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path


def make_polygon_path(vertices: list[list[float]]) -> Path:
    """Create a matplotlib Path from a list of [x, y] vertices."""
    return Path(vertices, closed=True)


def shrink_polygon_inward(
    vertices: list[list[float]], margin_m: float
) -> list[list[float]]:
    """Approximate inward ROI offset by shifting each vertex toward the centroid.

    Reduces edge vegetation / tree-shadow false positives without extra deps.
    """
    if margin_m <= 0 or len(vertices) < 3:
        return vertices
    pts = np.asarray(vertices, dtype=np.float64)
    center = pts.mean(axis=0)
    out: list[list[float]] = []
    for p in pts:
        d = center - p
        dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            out.append([float(p[0]), float(p[1])])
            continue
        step = min(float(margin_m), dist * 0.45)
        q = p + (d / dist) * step
        out.append([float(q[0]), float(q[1])])
    return out


def crop_points_to_roi(
    points: np.ndarray, polygon_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Return points inside the 2D polygon (XY only).

    Parameters
    ----------
    points : (N, 3+) float array — at minimum columns x, y, z.
    polygon_path : matplotlib Path for the ROI.

    Returns
    -------
    roi_points : subset of points inside the polygon.
    mask : boolean mask of shape (N,).
    """
    xy = points[:, :2]
    mask = polygon_path.contains_points(xy)
    return points[mask], mask
