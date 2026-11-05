#!/usr/bin/env python3
"""Clustering-based 3D object detection for roadside LiDAR.

Uses DBSCAN to find point clusters, fits oriented bounding boxes in BEV,
and classifies objects by size heuristics. Works model-free with any
point density — no pretrained weights needed.

Usage:
    python tools/cluster_detect.py \
        --data_path lidar_data/session_a \
        --save_path lidar_data/results_cluster_session_a/0000 \
        --ground_z -1.3
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN


def fit_oriented_bbox_2d(points_xy: np.ndarray):
    """Fit minimum-area oriented bounding box in BEV via PCA."""
    if len(points_xy) < 3:
        cx, cy = points_xy.mean(axis=0)
        dx = (points_xy[:, 0].max() - points_xy[:, 0].min()) + 0.3
        dy = (points_xy[:, 1].max() - points_xy[:, 1].min()) + 0.3
        return cx, cy, max(dx, 0.4), max(dy, 0.4), 0.0

    cov = np.cov(points_xy, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, np.argmax(eigenvalues)]
    heading = np.arctan2(principal[1], principal[0])

    cos_h, sin_h = np.cos(-heading), np.sin(-heading)
    centered = points_xy - points_xy.mean(axis=0)
    rotated = np.column_stack([
        centered[:, 0] * cos_h - centered[:, 1] * sin_h,
        centered[:, 0] * sin_h + centered[:, 1] * cos_h,
    ])
    dx = (rotated[:, 0].max() - rotated[:, 0].min()) + 0.2
    dy = (rotated[:, 1].max() - rotated[:, 1].min()) + 0.2

    cx, cy = points_xy.mean(axis=0)
    return cx, cy, max(dx, 0.3), max(dy, 0.3), heading


def classify_by_size(dx, dy, dz, n_points):
    """Heuristic classification based on bounding box dimensions.

    Returns (class_id, class_name) or (0, None) to reject.
    """
    length = max(dx, dy)
    width = min(dx, dy)
    aspect = length / max(width, 0.01)

    # Reject very flat clusters (ground remnants, road markings)
    if dz < 0.35:
        return 0, None
    # Reject very elongated + flat (curbs, guardrails)
    if aspect > 6 and dz < 0.8:
        return 0, None
    # Reject oversized clusters (merged structures)
    if length > 8:
        return 0, None

    if 2.5 <= length <= 6.5 and 1.0 <= width <= 3.0 and n_points >= 12:
        return 1, "Car"
    elif length < 1.5 and width < 1.2 and dz > 0.5:
        return 2, "Pedestrian"
    elif 1.2 <= length <= 3.0 and width < 1.5 and dz > 0.5:
        return 3, "Cyclist"
    elif 1.5 <= length <= 7 and width >= 1.0 and dz >= 0.4:
        return 1, "Car"
    elif dz > 0.5 and n_points >= 8:
        return 2, "Pedestrian"
    else:
        return 0, None


def detect_frame(points: np.ndarray, ground_z: float = -1.3,
                 eps: float = 0.8, min_samples: int = 4,
                 min_cluster_pts: int = 5):
    """Run clustering detection on a single frame.

    Args:
        points: (N, 4) float32 [x, y, z, intensity]
        ground_z: approximate ground plane z; points near/below are filtered
        eps: DBSCAN neighborhood radius (meters)
        min_samples: DBSCAN minimum points to form a core point
        min_cluster_pts: minimum points in a cluster to count as detection

    Returns:
        dict with 'bbox', 'score', 'label' lists (same format as model output)
    """
    above_ground = points[:, 2] > (ground_z + 0.15)
    pts = points[above_ground]

    if len(pts) < min_cluster_pts:
        return {"bbox": [], "score": [], "label": []}

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pts[:, :3])
    labels = clustering.labels_
    unique_labels = set(labels) - {-1}

    bboxes, scores, det_labels = [], [], []
    for lbl in unique_labels:
        mask = labels == lbl
        cluster_pts = pts[mask]
        n = len(cluster_pts)
        if n < min_cluster_pts:
            continue

        xy = cluster_pts[:, :2]
        z_vals = cluster_pts[:, 2]

        cx, cy, dx, dy, heading = fit_oriented_bbox_2d(xy)
        z_min, z_max = z_vals.min(), z_vals.max()
        dz = max(z_max - z_min, 0.3)
        cz = (z_min + z_max) / 2.0

        cls_id, cls_name = classify_by_size(dx, dy, dz, n)
        if cls_id == 0:
            continue

        score = min(1.0, n / 50.0)

        bboxes.append([float(cx), float(cy), float(cz),
                       float(dx), float(dy), float(dz), float(heading)])
        scores.append(float(score))
        det_labels.append(int(cls_id))

    return {"bbox": bboxes, "score": scores, "label": det_labels}


def main():
    parser = argparse.ArgumentParser(
        description="Clustering-based 3D detection for roadside LiDAR")
    parser.add_argument("--data_path", required=True,
                        help="Directory with .bin point cloud files")
    parser.add_argument("--save_path", required=True,
                        help="Output directory for *_pred.json files")
    parser.add_argument("--ext", default=".bin")
    parser.add_argument("--ground_z", type=float, default=-1.3,
                        help="Ground plane z value (default: -1.3)")
    parser.add_argument("--eps", type=float, default=0.8,
                        help="DBSCAN eps radius in meters (default: 0.8)")
    parser.add_argument("--min_samples", type=int, default=4,
                        help="DBSCAN min_samples (default: 4)")
    parser.add_argument("--min_cluster_pts", type=int, default=5,
                        help="Minimum points per cluster (default: 5)")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(data_path / f"*{args.ext}")))
    if not files:
        print(f"No {args.ext} files found in {data_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(files)} frames → {save_path}")

    total_dets = 0
    for i, fpath in enumerate(files):
        pts = np.fromfile(fpath, dtype=np.float32).reshape(-1, 4)
        result = detect_frame(pts, ground_z=args.ground_z,
                              eps=args.eps, min_samples=args.min_samples,
                              min_cluster_pts=args.min_cluster_pts)

        stem = Path(fpath).stem
        out_file = save_path / f"{stem}_pred.json"
        with open(out_file, "w") as f:
            json.dump(result, f)

        n = len(result["bbox"])
        total_dets += n
        if i % 200 == 0:
            print(f"  [{i+1}/{len(files)}] {stem}: {n} detections")

    avg = total_dets / max(len(files), 1)
    print(f"Done. {total_dets} total detections, {avg:.1f} avg/frame")


if __name__ == "__main__":
    main()
