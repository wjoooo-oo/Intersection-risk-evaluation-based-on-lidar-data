#!/usr/bin/env python3
"""Run the full detection + tracking pipeline on a rosbag2 offline and dump
per-frame track records to a JSONL file for downstream TTC / PET analysis.

Output one JSON object per frame, e.g.:
    {
      "frame": 12,
      "stamp_sec": 1778183647.05,
      "tracks": [
        {"track_id": 7, "class": "vehicle",
         "x": 12.3, "y": -4.5, "z": 0.6,
         "length": 4.2, "width": 1.9, "height": 1.6,
         "yaw": -1.21, "speed": 5.4,
         "num_points": 256, "hits": 24, "age": 31}
      ]
    }

This is the same pipeline (ROI -> grid-ground -> DBSCAN -> NMS -> bg filter ->
Kalman tracker -> EMA smoothing -> persistence/speed/NMS) that runs in
detection_node.py, but offline so the dump is deterministic and reproducible.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "detection"
sys.path.insert(0, str(PKG))

from intersection_lidar_detection.background_filter import (  # noqa: E402
    BackgroundFilter,
    StaticPointFilter,
)
from intersection_lidar_detection.clustering_detector import detect_objects  # noqa: E402
from intersection_lidar_detection.clustering_detector import (  # noqa: E402
    classify_object,
    _vehicle_class_ok,
)
from intersection_lidar_detection.pointcloud_utils import (  # noqa: E402
    height_filter,
    remove_ground_grid,
    voxel_downsample,
    voxel_downsample_range_split,
)
from intersection_lidar_detection.roi_utils import (  # noqa: E402
    crop_points_to_roi,
    make_polygon_path,
    shrink_polygon_inward,
)
from intersection_lidar_detection.tracker import MultiObjectTracker, Track  # noqa: E402

# PC2 parsing from offline helper (same as bag player topic layout)
# Offline PointCloud2 parser lives with ROI / bag helper tools
# 离线 PointCloud2 解析工具与 ROI / bag 辅助脚本放在同一目录
sys.path.insert(0, str(REPO / "tools" / "roi"))
from run_dbscan_offline import parse_pointcloud2_xyz  # noqa: E402


def _promote_track_class(track: Track, class_cfg: dict) -> None:
    if track.class_name != "unknown":
        return
    det = track.detection
    length = float(track.smoothed_length or (det.length if det else 0))
    width = float(track.smoothed_width or (det.width if det else 0))
    height = float(track.smoothed_height or (det.height if det else 0))
    n_pts = int(det.num_points) if det else 0
    range_m = float(np.linalg.norm(track.centroid[:2]))
    if classify_object(length, width, height, n_pts, class_cfg, range_m=range_m) == "vehicle":
        track.class_name = "vehicle"
        return
    relaxed = dict(class_cfg.get("vehicle", {}))
    relaxed.update(class_cfg.get("vehicle_far", {}))
    relaxed.pop("range_min_m", None)
    relaxed["min_height_to_length"] = 0.08
    relaxed["min_points"] = 3
    relaxed["min_height"] = 0.35
    if _vehicle_class_ok(length, width, height, max(n_pts, 3), relaxed):
        track.class_name = "vehicle"


def _preprocess_points(roi_pts: np.ndarray, pp: dict, gr: dict) -> np.ndarray:
    roi_pts = height_filter(roi_pts, float(pp["z_min"]), float(pp["z_max"]))
    range_split = float(pp.get("range_split_m", 0))
    if range_split > 0:
        pts = voxel_downsample_range_split(
            roi_pts,
            range_split,
            float(pp.get("voxel_near_m", pp.get("voxel_size", 0.15))),
            float(pp.get("voxel_far_m", pp.get("voxel_size", 0.15))),
            bool(pp.get("skip_far_downsample", False)),
        )
    else:
        pts = voxel_downsample(roi_pts, float(pp.get("voxel_size", 0.15)))
    return remove_ground_grid(
        pts,
        cell=float(gr.get("cell", 1.0)),
        delta=float(gr.get("delta", 0.25)),
        z_global_max=gr.get("z_global_max"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bag",
        type=Path,
        default=Path("<BAG_DIR>/session_a"),
    )
    parser.add_argument("--topic", default="/lidar/points")
    parser.add_argument("--roi-yaml", type=Path, default=REPO / "config" / "roi.yaml")
    parser.add_argument(
        "--detection-config",
        type=Path,
        default=PKG / "config" / "detection_config.yaml",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "conflict_analysis" / "output" / "tracks.jsonl",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=-1,
        help="frames to process (-1 = all)",
    )
    parser.add_argument(
        "--include-class",
        nargs="*",
        default=["vehicle"],
        help="class names to dump (default: vehicle only for TTC/PET)",
    )
    args = parser.parse_args()

    with open(args.detection_config) as f:
        det_cfg = yaml.safe_load(f)

    roi_cfg = det_cfg["roi"]
    roi_pts = list(roi_cfg["points"])
    erode_m = float(roi_cfg.get("erode_margin_m", 0.0))
    if erode_m > 0:
        roi_pts = shrink_polygon_inward(roi_pts, erode_m)
        print(f"ROI eroded inward by {erode_m:.1f} m")
    roi_path = make_polygon_path(roi_pts)

    pp = det_cfg["preprocess"]
    gr = pp["ground_removal"]
    cluster_cfg = {**det_cfg["clustering"]}
    cluster_cfg.setdefault("bbox_method", "obb")
    class_cfg = det_cfg["classification"]
    bg_cfg = det_cfg.get("background_filter", {})
    trk_cfg = det_cfg.get("tracking", {})
    min_speed = float(trk_cfg.get("min_track_speed", 0.0))
    speed_window = int(trk_cfg.get("speed_window", 5))
    track_nms_iou = float(trk_cfg.get("nms_iou", 0.0))

    sp_cfg = pp.get("static_point_filter", {})
    static_pts = None
    if sp_cfg.get("enabled", False):
        static_pts = StaticPointFilter(
            resolution=sp_cfg.get("resolution", 0.5),
            history_frames=sp_cfg.get("history_frames", 60),
            static_threshold=sp_cfg.get("static_threshold", 0.65),
            warmup_frames=sp_cfg.get("warmup_frames", 20),
        )

    bg = BackgroundFilter(
        resolution=bg_cfg.get("resolution", 1.0),
        history_frames=bg_cfg.get("history_frames", 100),
        static_threshold=bg_cfg.get("static_threshold", 0.55),
        warmup_frames=bg_cfg.get("warmup_frames", 25),
        exempt_classes=bg_cfg.get("exempt_classes"),
    )
    tracker = MultiObjectTracker(
        max_age=trk_cfg.get("max_age", 10),
        min_hits=trk_cfg.get("min_hits", 4),
        min_track_age=trk_cfg.get("min_track_age", 6),
        distance_threshold=trk_cfg.get("distance_threshold", 3.0),
        process_noise_pos=trk_cfg.get("process_noise_pos", 0.5),
        process_noise_vel=trk_cfg.get("process_noise_vel", 1.0),
        measurement_noise=trk_cfg.get("measurement_noise", 0.3),
        max_trajectory_length=trk_cfg.get("max_trajectory_length", 100),
        size_ema_alpha=trk_cfg.get("size_ema_alpha", 0.25),
        size_ema_min_points=trk_cfg.get("size_ema_min_points", 15),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    include = set(args.include_class)
    n_total = args.n_frames if args.n_frames > 0 else -1
    start = time.time()
    n_records = 0

    with Reader(args.bag) as reader, open(args.out, "w") as out_f:
        conns = [c for c in reader.connections if c.topic == args.topic]
        if not conns:
            raise SystemExit(f"topic {args.topic!r} not in bag")
        frame_idx = 0
        last_ts = None
        for conn, ts, raw in reader.messages(connections=conns):
            if n_total > 0 and frame_idx >= n_total:
                break
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            dt = 0.1 if last_ts is None else max(stamp_sec - last_ts, 0.01)
            last_ts = stamp_sec

            xyz = parse_pointcloud2_xyz(msg)
            roi_raw, _ = crop_points_to_roi(xyz, roi_path)
            pts = _preprocess_points(roi_raw, pp, gr)
            if static_pts is not None:
                pts, _ = static_pts.filter_points(pts)
                if len(pts) == 0:
                    frame_idx += 1
                    continue

            detections = detect_objects(pts, cluster_cfg, class_cfg)
            dynamic = bg.filter_detections(detections)
            confirmed = tracker.step(dynamic, dt=dt)

            if track_nms_iou > 0:
                confirmed = tracker.nms(confirmed, iou_threshold=track_nms_iou)

            for trk in confirmed:
                _promote_track_class(trk, class_cfg)

            if min_speed > 0:
                confirmed = [
                    t for t in confirmed
                    if len(t.trajectory) <= speed_window
                    or t.recent_speed(window=speed_window, dt=dt) >= min_speed
                ]

            tracks_out = []
            for t in confirmed:
                if t.class_name not in include:
                    continue
                speed = t.recent_speed(window=speed_window, dt=dt)
                pts_used = t.detection.num_points if t.detection is not None else 0
                tracks_out.append(
                    {
                        "track_id": int(t.track_id),
                        "class": t.class_name,
                        "x": float(t.centroid[0]),
                        "y": float(t.centroid[1]),
                        "z": float(t.centroid[2]),
                        "length": float(t.smoothed_length),
                        "width": float(t.smoothed_width),
                        "height": float(t.smoothed_height),
                        "yaw": float(t.smoothed_yaw),
                        "speed": float(speed),
                        "vx_kf": float(t.velocity[0]),
                        "vy_kf": float(t.velocity[1]),
                        "num_points": int(pts_used),
                        "hits": int(t.hits),
                        "age": int(t.age),
                    }
                )

            rec = {
                "frame": frame_idx,
                "stamp_sec": stamp_sec,
                "dt": dt,
                "n_tracks": len(tracks_out),
                "tracks": tracks_out,
            }
            out_f.write(json.dumps(rec) + "\n")
            n_records += 1

            if frame_idx % 100 == 0:
                elapsed = time.time() - start
                rate = (frame_idx + 1) / max(elapsed, 1e-6)
                eta = ((n_total - frame_idx - 1) / rate) if n_total > 0 else float("nan")
                eta_str = f"{eta:.0f}s" if not np.isnan(eta) else "?"
                print(
                    f"  frame {frame_idx:5d}  tracks={len(tracks_out):2d}  "
                    f"({rate:.1f} fps, ETA {eta_str})"
                )
            frame_idx += 1

    elapsed = time.time() - start
    print(f"\nWrote {n_records} frames to {args.out}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
