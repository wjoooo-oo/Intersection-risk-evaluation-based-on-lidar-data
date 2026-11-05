"""Background / static object filter.

Point-level: remove voxels that reappear in the same grid cells across many frames.
Detection-level: drop clusters whose centroids sit in persistently occupied cells.
"""

from __future__ import annotations

import numpy as np
from .clustering_detector import Detection
from typing import List


class _OccupancyWindow:
    """Sliding-window XY grid occupancy shared by point and detection filters."""

    def __init__(
        self,
        resolution: float,
        history_frames: int,
        warmup_frames: int,
    ):
        self.resolution = resolution
        self.history_frames = history_frames
        self.warmup_frames = warmup_frames
        self._occupancy_count: dict[tuple[int, int], int] = {}
        self._history: list[set[tuple[int, int]]] = []
        self._frame_idx = 0

    def _xy_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(np.floor(x / self.resolution)),
            int(np.floor(y / self.resolution)),
        )

    def record_cells(self, cells: set[tuple[int, int]]) -> None:
        self._history.append(cells)
        for cell in cells:
            self._occupancy_count[cell] = self._occupancy_count.get(cell, 0) + 1
        if len(self._history) > self.history_frames:
            oldest = self._history.pop(0)
            for cell in oldest:
                self._occupancy_count[cell] -= 1
                if self._occupancy_count[cell] <= 0:
                    del self._occupancy_count[cell]
        self._frame_idx += 1

    @property
    def ready(self) -> bool:
        return self._frame_idx >= self.warmup_frames

    def static_ratio(self, x: float, y: float) -> float:
        """Max occupancy fraction in 3x3 neighborhood (0..1)."""
        window = min(self._frame_idx, self.history_frames)
        if window == 0:
            return 0.0
        cx, cy = self._xy_to_cell(x, y)
        best = 0.0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                count = self._occupancy_count.get((cx + dx, cy + dy), 0)
                best = max(best, count / window)
        return best


class StaticPointFilter:
    """Drop LiDAR points in cells occupied on most frames (trees, poles, curbs)."""

    def __init__(
        self,
        resolution: float = 0.5,
        history_frames: int = 60,
        static_threshold: float = 0.65,
        warmup_frames: int = 20,
    ):
        self.static_threshold = static_threshold
        self._grid = _OccupancyWindow(resolution, history_frames, warmup_frames)

    def filter_points(self, points: np.ndarray) -> tuple[np.ndarray, int]:
        """Update occupancy from this frame, return non-static subset."""
        if len(points) == 0:
            return points, 0

        cells: set[tuple[int, int]] = set()
        xy = points[:, :2]
        for x, y in xy:
            cells.add(self._grid._xy_to_cell(float(x), float(y)))
        self._grid.record_cells(cells)

        if not self._grid.ready:
            return points, 0

        keep = np.ones(len(points), dtype=bool)
        for i, (x, y) in enumerate(xy):
            if self._grid.static_ratio(float(x), float(y)) >= self.static_threshold:
                keep[i] = False
        n_removed = int((~keep).sum())
        return points[keep], n_removed


class BackgroundFilter:
    """Drop detections whose centroids sit in persistently occupied cells."""

    def __init__(
        self,
        resolution: float = 0.5,
        history_frames: int = 50,
        static_threshold: float = 0.8,
        warmup_frames: int = 20,
        exempt_classes: list[str] | None = None,
    ):
        self.static_threshold = static_threshold
        self.exempt_classes = set(exempt_classes or [])
        self._grid = _OccupancyWindow(resolution, history_frames, warmup_frames)

    def update(self, detections: List[Detection]) -> None:
        cells: set[tuple[int, int]] = set()
        for det in detections:
            cells.add(
                self._grid._xy_to_cell(float(det.centroid[0]), float(det.centroid[1]))
            )
        self._grid.record_cells(cells)

    def is_static(self, det: Detection) -> bool:
        if not self._grid.ready:
            return False
        return (
            self._grid.static_ratio(float(det.centroid[0]), float(det.centroid[1]))
            >= self.static_threshold
        )

    def filter_detections(self, detections: List[Detection]) -> List[Detection]:
        self.update(detections)
        if not self._grid.ready:
            return detections
        return [
            d
            for d in detections
            if d.class_name in self.exempt_classes or not self.is_static(d)
        ]
