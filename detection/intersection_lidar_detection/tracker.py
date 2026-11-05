"""Simple multi-object tracker: Kalman filter (constant velocity) + Hungarian assignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from .clustering_detector import Detection


@dataclass
class Track:
    track_id: int
    class_name: str
    # State: [x, y, z, vx, vy, vz]
    state: np.ndarray
    covariance: np.ndarray
    age: int = 0              # total frames since creation
    hits: int = 1             # frames with matched detection
    misses: int = 0           # consecutive frames without match
    detection: Optional[Detection] = field(default=None, repr=False)
    trajectory: list = field(default_factory=list)  # history of centroids
    # EMA-smoothed OBB so RViz boxes don't shrink/expand when a single frame
    # has a degraded cluster (partial occlusion, edge of beam pattern, etc.).
    smoothed_length: float = 0.0
    smoothed_width: float = 0.0
    smoothed_height: float = 0.0
    smoothed_yaw: float = 0.0

    @property
    def centroid(self) -> np.ndarray:
        return self.state[:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:6]

    def recent_speed(self, window: int = 5, dt: float = 0.1) -> float:
        """Median XY speed (m/s) over the last `window` trajectory points.

        Robust to single-frame jitter (Kalman vx/vy can spike on outliers).
        Returns 0.0 if there is not enough history.
        """
        if len(self.trajectory) < 2:
            return 0.0
        pts = np.asarray(self.trajectory[-(window + 1):])  # (W+1, 3)
        if len(pts) < 2:
            return 0.0
        diffs = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
        return float(np.median(diffs) / max(dt, 1e-6))


def obb_bev_iou(
    cx1: float, cy1: float, l1: float, w1: float, yaw1: float,
    cx2: float, cy2: float, l2: float, w2: float, yaw2: float,
) -> float:
    """BEV IoU of two oriented rectangles (XY plane).

    Uses cv2.rotatedRectangleIntersection (degrees, OpenCV convention). Returns
    0.0 if cv2 is unavailable, the boxes don't intersect, or the intersection
    polygon is degenerate.
    """
    try:
        import cv2  # noqa: WPS433 (lazy import: cv2 is optional)
    except ImportError:
        return 0.0
    if l1 <= 0 or w1 <= 0 or l2 <= 0 or w2 <= 0:
        return 0.0
    r1 = ((float(cx1), float(cy1)), (float(l1), float(w1)), float(np.degrees(yaw1)))
    r2 = ((float(cx2), float(cy2)), (float(l2), float(w2)), float(np.degrees(yaw2)))
    ret, inter = cv2.rotatedRectangleIntersection(r1, r2)
    if ret == 0 or inter is None or len(inter) < 3:
        return 0.0
    inter_area = float(cv2.contourArea(inter))
    if inter_area <= 0:
        return 0.0
    a1 = l1 * w1
    a2 = l2 * w2
    union = a1 + a2 - inter_area
    return inter_area / max(union, 1e-9)


def _smooth_yaw(prev: float, new: float, alpha: float) -> float:
    """EMA yaw smoothing, accounting for OBB's pi sign-ambiguity.

    The major axis of an OBB has no canonical direction, so yaw and yaw+pi
    represent the same orientation. Wrap `new` into the half closest to `prev`
    before averaging, so a 180-deg PCA flip doesn't ruin the smoothing.
    """
    diff = new - prev
    if diff > np.pi / 2:
        new -= np.pi
    elif diff < -np.pi / 2:
        new += np.pi
    smoothed = (1.0 - alpha) * prev + alpha * new
    # Renormalize back to (-pi/2, pi/2]
    while smoothed > np.pi / 2:
        smoothed -= np.pi
    while smoothed <= -np.pi / 2:
        smoothed += np.pi
    return smoothed


class MultiObjectTracker:
    """Kalman + Hungarian tracker for intersection detections."""

    def __init__(
        self,
        max_age: int = 10,
        min_hits: int = 3,
        min_track_age: int = 0,
        distance_threshold: float = 3.0,
        process_noise_pos: float = 0.5,
        process_noise_vel: float = 1.0,
        measurement_noise: float = 0.3,
        max_trajectory_length: int = 100,
        size_ema_alpha: float = 0.25,
        size_ema_min_points: int = 12,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        # Persistence: only publish tracks that have been alive >= this many
        # frames overall (including brief misses). Filters one-frame phantoms.
        self.min_track_age = min_track_age
        self.distance_threshold = distance_threshold
        self.process_noise_pos = process_noise_pos
        self.process_noise_vel = process_noise_vel
        self.measurement_noise = measurement_noise
        self.max_trajectory_length = max_trajectory_length
        # EMA for OBB size/yaw. Effective alpha is scaled by detection point
        # count so degraded clusters (few points) barely move the smoothed
        # values, while dense clusters update them quickly.
        self.size_ema_alpha = size_ema_alpha
        self.size_ema_min_points = size_ema_min_points

        self._next_id = 1
        self.tracks: List[Track] = []

    def _init_state(self, det: Detection) -> tuple[np.ndarray, np.ndarray]:
        state = np.zeros(6, dtype=np.float64)
        state[:3] = det.centroid
        P = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])
        return state, P

    def _predict(self, track: Track, dt: float = 0.1) -> None:
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        track.state = F @ track.state
        Q = np.diag([
            self.process_noise_pos * dt,
            self.process_noise_pos * dt,
            self.process_noise_pos * dt,
            self.process_noise_vel * dt,
            self.process_noise_vel * dt,
            self.process_noise_vel * dt,
        ])
        track.covariance = F @ track.covariance @ F.T + Q

    def _update(self, track: Track, det: Detection) -> None:
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        R = np.eye(3) * self.measurement_noise
        z = det.centroid.astype(np.float64)

        y = z - H @ track.state
        S = H @ track.covariance @ H.T + R
        K = track.covariance @ H.T @ np.linalg.inv(S)

        track.state = track.state + K @ y
        track.covariance = (np.eye(6) - K @ H) @ track.covariance
        track.detection = det
        track.class_name = det.class_name
        track.hits += 1
        track.misses = 0
        track.trajectory.append(track.centroid.copy())
        if len(track.trajectory) > self.max_trajectory_length:
            track.trajectory.pop(0)

        # --- Smoothed OBB update (size + yaw EMA, trust scales with point count) ---
        if track.smoothed_length <= 0.0:
            track.smoothed_length = float(det.length)
            track.smoothed_width = float(det.width)
            track.smoothed_height = float(det.height)
            track.smoothed_yaw = float(det.yaw)
        else:
            # Confidence in this detection's measured size: full at >= min_points,
            # 0 at 0 points, linear between.
            conf = min(1.0, det.num_points / max(self.size_ema_min_points, 1))
            alpha = self.size_ema_alpha * conf
            track.smoothed_length = (
                (1.0 - alpha) * track.smoothed_length + alpha * float(det.length)
            )
            track.smoothed_width = (
                (1.0 - alpha) * track.smoothed_width + alpha * float(det.width)
            )
            track.smoothed_height = (
                (1.0 - alpha) * track.smoothed_height + alpha * float(det.height)
            )
            track.smoothed_yaw = _smooth_yaw(
                track.smoothed_yaw, float(det.yaw), alpha
            )

    def _cost_matrix(self, detections: List[Detection]) -> np.ndarray:
        n_tracks = len(self.tracks)
        n_dets = len(detections)
        cost = np.full((n_tracks, n_dets), 1e6, dtype=np.float64)
        for i, trk in enumerate(self.tracks):
            for j, det in enumerate(detections):
                dist = np.linalg.norm(trk.centroid - det.centroid)
                cost[i, j] = dist
        return cost

    def step(self, detections: List[Detection], dt: float = 0.1) -> List[Track]:
        """Run one tracking cycle. Returns confirmed tracks."""
        # Predict all existing tracks
        for trk in self.tracks:
            self._predict(trk, dt)
            trk.age += 1

        # Associate detections to tracks (Hungarian)
        matched_trk_idx = set()
        matched_det_idx = set()

        if self.tracks and detections:
            cost = self._cost_matrix(detections)
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < self.distance_threshold:
                    matched_trk_idx.add(r)
                    matched_det_idx.add(c)
                    self._update(self.tracks[r], detections[c])

        # Unmatched tracks: increment misses
        for i, trk in enumerate(self.tracks):
            if i not in matched_trk_idx:
                trk.misses += 1
                trk.detection = None

        # Unmatched detections: create new tracks
        for j, det in enumerate(detections):
            if j not in matched_det_idx:
                state, P = self._init_state(det)
                new_track = Track(
                    track_id=self._next_id,
                    class_name=det.class_name,
                    state=state,
                    covariance=P,
                    detection=det,
                    trajectory=[det.centroid.copy()],
                    smoothed_length=float(det.length),
                    smoothed_width=float(det.width),
                    smoothed_height=float(det.height),
                    smoothed_yaw=float(det.yaw),
                )
                self._next_id += 1
                self.tracks.append(new_track)

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]

        # Return tracks that have both:
        #   - hits >= min_hits        (matched detection enough times) and
        #   - age  >= min_track_age   (existed across enough frames overall).
        return [
            t for t in self.tracks
            if t.hits >= self.min_hits and t.age >= self.min_track_age
        ]

    def nms(self, tracks: List["Track"], iou_threshold: float = 0.3) -> List["Track"]:
        """Suppress overlapping tracks in BEV.

        When two tracks share BEV IoU above the threshold (e.g. one car split
        into two clusters by DBSCAN, or a re-id failure spawned a duplicate),
        keep the more mature one (more hits, longer age, fewer misses) and
        DELETE the loser from self.tracks so it doesn't keep getting matched.
        """
        if len(tracks) <= 1 or iou_threshold <= 0:
            return tracks
        # Maturity score: higher = better
        ordered = sorted(
            tracks,
            key=lambda t: (t.hits, t.age, -t.misses),
            reverse=True,
        )
        suppressed_ids: set[int] = set()
        for i, ti in enumerate(ordered):
            if ti.track_id in suppressed_ids:
                continue
            for tj in ordered[i + 1:]:
                if tj.track_id in suppressed_ids:
                    continue
                iou = obb_bev_iou(
                    ti.centroid[0], ti.centroid[1],
                    ti.smoothed_length, ti.smoothed_width, ti.smoothed_yaw,
                    tj.centroid[0], tj.centroid[1],
                    tj.smoothed_length, tj.smoothed_width, tj.smoothed_yaw,
                )
                if iou >= iou_threshold:
                    suppressed_ids.add(tj.track_id)
        if suppressed_ids:
            # Drop the duplicates entirely so they don't keep stealing detections
            self.tracks = [t for t in self.tracks if t.track_id not in suppressed_ids]
        return [t for t in tracks if t.track_id not in suppressed_ids]
