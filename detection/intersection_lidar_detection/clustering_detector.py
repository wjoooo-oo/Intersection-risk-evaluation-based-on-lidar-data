"""DBSCAN clustering and rule-based vehicle/pedestrian classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from sklearn.cluster import DBSCAN


@dataclass
class Detection:
    class_name: str  # "vehicle", "pedestrian", "unknown"
    centroid: np.ndarray  # [x, y, z] of oriented box center
    bbox_min: np.ndarray  # axis-aligned [x_min, y_min, z_min] (legacy/fallback)
    bbox_max: np.ndarray  # axis-aligned [x_max, y_max, z_max] (legacy/fallback)
    length: float         # OBB extent along major axis (>= width)
    width: float          # OBB extent along minor axis
    height: float         # z extent
    num_points: int
    points: np.ndarray = field(repr=False)  # (M, 3) cluster points
    yaw: float = 0.0      # rotation of major axis around +Z (radians)
    corners_xy: np.ndarray | None = field(default=None, repr=False)  # (4, 2)

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "centroid": self.centroid.tolist(),
            "bbox_min": self.bbox_min.tolist(),
            "bbox_max": self.bbox_max.tolist(),
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "yaw": float(self.yaw),
            "corners_xy": (
                self.corners_xy.tolist() if self.corners_xy is not None else None
            ),
            "num_points": self.num_points,
        }


def cluster_points(
    points: np.ndarray,
    eps: float = 0.7,
    min_samples: int = 8,
    min_cluster_points: int = 15,
    max_cluster_points: int = 5000,
) -> List[np.ndarray]:
    """Run DBSCAN and return list of point arrays per cluster."""
    if len(points) < min_samples:
        return []

    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(points[:, :3])

    clusters = []
    unique_labels = set(labels)
    unique_labels.discard(-1)  # noise

    for lbl in unique_labels:
        cluster_pts = points[labels == lbl]
        n = len(cluster_pts)
        if min_cluster_points <= n <= max_cluster_points:
            clusters.append(cluster_pts)

    return clusters


def cluster_points_range_adaptive(
    points: np.ndarray, clustering_cfg: dict
) -> List[np.ndarray]:
    """Near/far DBSCAN with larger eps and lower min size thresholds far from sensor."""
    split = float(clustering_cfg.get("range_split_m", 0))
    if split <= 0 or len(points) == 0:
        return cluster_points(
            points,
            eps=clustering_cfg.get("eps", 0.7),
            min_samples=clustering_cfg.get("min_samples", 8),
            min_cluster_points=clustering_cfg.get("min_cluster_points", 15),
            max_cluster_points=clustering_cfg.get("max_cluster_points", 5000),
        )

    r = np.linalg.norm(points[:, :2], axis=1)
    near = points[r < split]
    far = points[r >= split]
    clusters: List[np.ndarray] = []
    if len(near):
        clusters.extend(
            cluster_points(
                near,
                eps=float(clustering_cfg.get("eps_near", clustering_cfg.get("eps", 0.7))),
                min_samples=int(clustering_cfg.get("min_samples_near", clustering_cfg.get("min_samples", 8))),
                min_cluster_points=int(
                    clustering_cfg.get("min_cluster_near", clustering_cfg.get("min_cluster_points", 15))
                ),
                max_cluster_points=int(clustering_cfg.get("max_cluster_points", 5000)),
            )
        )
    if len(far):
        clusters.extend(
            cluster_points(
                far,
                eps=float(clustering_cfg.get("eps_far", clustering_cfg.get("eps", 0.7) * 1.5)),
                min_samples=int(clustering_cfg.get("min_samples_far", 2)),
                min_cluster_points=int(clustering_cfg.get("min_cluster_far", 5)),
                max_cluster_points=int(clustering_cfg.get("max_cluster_points", 5000)),
            )
        )
    return clusters


def compute_bbox(cluster: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Axis-aligned bounding box. Returns: centroid, bbox_min, bbox_max, length, width, height."""
    bbox_min = cluster.min(axis=0)
    bbox_max = cluster.max(axis=0)
    centroid = (bbox_min + bbox_max) / 2.0
    dims = bbox_max - bbox_min
    return centroid, bbox_min, bbox_max, float(dims[0]), float(dims[1]), float(dims[2])


def compute_obb_pca(cluster: np.ndarray) -> dict:
    """Oriented bounding box via PCA on XY (axis-aligned in z).

    Returns a dict with keys:
        center  (3,) center of the OBB (x, y, z_center)
        length  major-axis extent (>= width)
        width   minor-axis extent
        height  z extent
        yaw     rotation of the major axis around +Z, radians, in (-pi/2, pi/2]
        corners (4, 2) XY corners of the OBB, CCW order
        bbox_min, bbox_max axis-aligned (kept for backward-compat)
    """
    xy = cluster[:, :2].astype(np.float64)
    mu = xy.mean(axis=0)
    centered = xy - mu
    C = np.cov(centered.T) if len(xy) > 1 else np.eye(2)
    evals, evecs = np.linalg.eigh(C)  # ascending
    # major axis = eigenvector with larger eigenvalue
    R = np.column_stack([evecs[:, 1], evecs[:, 0]])  # cols: major, minor
    if np.linalg.det(R) < 0:
        R[:, 1] *= -1
    local = centered @ R  # rotate to OBB frame
    mn = local.min(axis=0)
    mx = local.max(axis=0)
    extent = mx - mn  # [length, width]
    center_local = (mn + mx) / 2.0
    center_xy = mu + R @ center_local
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    # normalize to (-pi/2, pi/2] since major axis direction is sign-free
    if yaw > np.pi / 2:
        yaw -= np.pi
    elif yaw <= -np.pi / 2:
        yaw += np.pi
    z = cluster[:, 2]
    z_min = float(z.min())
    z_max = float(z.max())
    height = z_max - z_min
    z_center = (z_min + z_max) / 2.0

    length, width = float(extent[0]), float(extent[1])
    if width > length:  # enforce length >= width
        length, width = width, length
        yaw = yaw + np.pi / 2
        if yaw > np.pi / 2:
            yaw -= np.pi

    half = np.array(
        [
            [length / 2.0, width / 2.0],
            [length / 2.0, -width / 2.0],
            [-length / 2.0, -width / 2.0],
            [-length / 2.0, width / 2.0],
        ]
    )
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rfinal = np.array([[cy, -sy], [sy, cy]])
    corners = half @ Rfinal.T + np.array([center_xy[0], center_xy[1]])

    bbox_min = np.array([cluster[:, 0].min(), cluster[:, 1].min(), z_min])
    bbox_max = np.array([cluster[:, 0].max(), cluster[:, 1].max(), z_max])

    return {
        "center": np.array([center_xy[0], center_xy[1], z_center]),
        "length": length,
        "width": width,
        "height": height,
        "yaw": yaw,
        "corners": corners.astype(np.float32),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
    }


def _vehicle_class_ok(
    length: float,
    width: float,
    height: float,
    num_points: int,
    veh: dict,
) -> bool:
    if not (
        veh.get("min_length", 0) <= length <= veh.get("max_length", 999)
        and veh.get("min_width", 0) <= width <= veh.get("max_width", 999)
        and veh.get("min_height", 0) <= height <= veh.get("max_height", 999)
        and num_points >= veh.get("min_points", 0)
    ):
        return False
    min_h2l = float(veh.get("min_height_to_length", 0.0))
    if min_h2l > 0 and length > 1e-3 and height / length < min_h2l:
        return False
    return True


def classify_object(
    length: float,
    width: float,
    height: float,
    num_points: int,
    config: dict,
    range_m: float = 0.0,
) -> str:
    """Rule-based classification using size thresholds from config."""
    veh = dict(config.get("vehicle", {}))
    ped = config.get("pedestrian", {})
    far = config.get("vehicle_far", {})
    if far and range_m >= float(far.get("range_min_m", 1e9)):
        for key, val in far.items():
            if key != "range_min_m":
                veh[key] = val

    if _vehicle_class_ok(length, width, height, num_points, veh):
        return "vehicle"

    # Check pedestrian
    if (
        ped.get("min_length", 0) <= length <= ped.get("max_length", 999)
        and ped.get("min_width", 0) <= width <= ped.get("max_width", 999)
        and ped.get("min_height", 0) <= height <= ped.get("max_height", 999)
        and num_points >= ped.get("min_points", 0)
    ):
        return "pedestrian"

    return "unknown"


def _detection_nms(
    detections: List[Detection], iou_threshold: float
) -> List[Detection]:
    """Suppress duplicate detections via BEV IoU (keep the one with more points)."""
    if len(detections) <= 1 or iou_threshold <= 0:
        return detections
    try:
        import cv2  # noqa: WPS433
    except ImportError:
        return detections

    def _iou(d1: Detection, d2: Detection) -> float:
        l1, w1 = d1.length, d1.width
        l2, w2 = d2.length, d2.width
        if l1 <= 0 or w1 <= 0 or l2 <= 0 or w2 <= 0:
            return 0.0
        r1 = (
            (float(d1.centroid[0]), float(d1.centroid[1])),
            (float(l1), float(w1)),
            float(np.degrees(d1.yaw)),
        )
        r2 = (
            (float(d2.centroid[0]), float(d2.centroid[1])),
            (float(l2), float(w2)),
            float(np.degrees(d2.yaw)),
        )
        ret, inter = cv2.rotatedRectangleIntersection(r1, r2)
        if ret == 0 or inter is None or len(inter) < 3:
            return 0.0
        inter_area = float(cv2.contourArea(inter))
        if inter_area <= 0:
            return 0.0
        return inter_area / max(l1 * w1 + l2 * w2 - inter_area, 1e-9)

    # Order by num_points desc → keep denser cluster as canonical
    ordered = sorted(detections, key=lambda d: -d.num_points)
    kept: List[Detection] = []
    for cand in ordered:
        if any(_iou(cand, k) >= iou_threshold for k in kept):
            continue
        kept.append(cand)
    return kept


def detect_objects(
    points: np.ndarray,
    clustering_cfg: dict,
    classification_cfg: dict,
) -> List[Detection]:
    """Full detection pipeline: cluster -> oriented bbox -> classify.

    Set clustering_cfg['bbox_method'] to 'aabb' to keep the legacy axis-aligned
    bbox; default is 'obb' (PCA-based oriented box on XY).
    """
    if float(clustering_cfg.get("range_split_m", 0)) > 0:
        clusters = cluster_points_range_adaptive(points, clustering_cfg)
    else:
        clusters = cluster_points(
            points,
            eps=clustering_cfg.get("eps", 0.7),
            min_samples=clustering_cfg.get("min_samples", 8),
            min_cluster_points=clustering_cfg.get("min_cluster_points", 15),
            max_cluster_points=clustering_cfg.get("max_cluster_points", 5000),
        )
    bbox_method = clustering_cfg.get("bbox_method", "obb")

    detections: List[Detection] = []
    for cluster in clusters:
        num_points = len(cluster)
        if bbox_method == "aabb":
            centroid, bbox_min, bbox_max, length, width, height = compute_bbox(cluster)
            yaw = 0.0
            corners = None
        else:
            obb = compute_obb_pca(cluster)
            centroid = obb["center"]
            bbox_min = obb["bbox_min"]
            bbox_max = obb["bbox_max"]
            length = obb["length"]
            width = obb["width"]
            height = obb["height"]
            yaw = obb["yaw"]
            corners = obb["corners"]

        range_m = float(np.linalg.norm(centroid[:2]))
        class_name = classify_object(
            length, width, height, num_points, classification_cfg, range_m=range_m
        )
        if (
            class_name == "unknown"
            and classification_cfg.get("promote_vehicle_like_unknown")
            and range_m >= float(classification_cfg.get("promote_min_range_m", 25.0))
            and height >= float(classification_cfg.get("promote_min_height_m", 0.5))
        ):
            relaxed = dict(classification_cfg.get("vehicle", {}))
            relaxed.update(classification_cfg.get("vehicle_far", {}))
            relaxed.pop("range_min_m", None)
            relaxed["min_height_to_length"] = min(
                float(relaxed.get("min_height_to_length", 0.15)), 0.12
            )
            if _vehicle_class_ok(length, width, height, num_points, relaxed):
                class_name = "vehicle"
        detections.append(
            Detection(
                class_name=class_name,
                centroid=centroid,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                length=length,
                width=width,
                height=height,
                num_points=num_points,
                points=cluster,
                yaw=yaw,
                corners_xy=corners,
            )
        )

    nms_iou = float(clustering_cfg.get("nms_iou", 0.0))
    if nms_iou > 0:
        detections = _detection_nms(detections, nms_iou)
    return detections
