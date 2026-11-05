#!/usr/bin/env python3
"""Compare raw detection length vs smoothed track length over time, per track ID.

Quick sanity-check that the EMA size smoothing actually reduces frame-to-frame
bbox jitter for tracked vehicles.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "detection"
sys.path.insert(0, str(PKG))

from intersection_lidar_detection.background_filter import BackgroundFilter  # noqa: E402
from intersection_lidar_detection.clustering_detector import detect_objects  # noqa: E402
from intersection_lidar_detection.roi_utils import (  # noqa: E402
    crop_points_to_roi,
    make_polygon_path,
)
from intersection_lidar_detection.tracker import MultiObjectTracker  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_dbscan_offline import (  # noqa: E402
    parse_pointcloud2_xyz,
    voxel_downsample,
    height_filter,
    remove_ground_grid,
)


def main():
    bag = Path("<BAG_DIR>/session_a")
    with open(REPO / "config" / "roi.yaml") as f:
        roi_cfg = yaml.safe_load(f)
    with open(PKG / "config" / "detection_config.yaml") as f:
        det_cfg = yaml.safe_load(f)

    polygon = roi_cfg["polygon_xy"]
    roi_path = make_polygon_path(polygon)
    pp = det_cfg["preprocess"]
    gr = pp["ground_removal"]
    cl = det_cfg["clustering"]
    cls = det_cfg["classification"]
    bg_cfg = det_cfg["background_filter"]
    trk_cfg = det_cfg["tracking"]

    bg = BackgroundFilter(
        resolution=bg_cfg["resolution"],
        history_frames=bg_cfg["history_frames"],
        static_threshold=bg_cfg["static_threshold"],
        warmup_frames=bg_cfg["warmup_frames"],
    )
    trk = MultiObjectTracker(
        max_age=trk_cfg["max_age"],
        min_hits=trk_cfg["min_hits"],
        min_track_age=trk_cfg.get("min_track_age", 0),
        distance_threshold=trk_cfg["distance_threshold"],
        size_ema_alpha=trk_cfg.get("size_ema_alpha", 0.25),
        size_ema_min_points=trk_cfg.get("size_ema_min_points", 12),
    )

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    # Map track_id -> list of (frame, det.length, smoothed_length, det.num_points)
    log: dict[int, list[tuple]] = {}

    n_frames = 300
    with Reader(bag) as reader:
        conns = [c for c in reader.connections if c.topic == "/lidar/points"]
        frame_idx = 0
        for conn, _ts, raw in reader.messages(connections=conns):
            if frame_idx >= n_frames:
                break
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            xyz = parse_pointcloud2_xyz(msg)
            pts, _ = crop_points_to_roi(xyz, roi_path)
            pts = height_filter(pts, pp["z_min"], pp["z_max"])
            pts = voxel_downsample(pts, pp["voxel_size"])
            pts = remove_ground_grid(
                pts, cell=gr["cell"], delta=gr["delta"], z_global_max=gr.get("z_global_max")
            )
            dets = detect_objects(pts, cl, cls)
            dynamic = bg.filter_detections(dets)
            confirmed = trk.step(dynamic, dt=0.1)
            for t in confirmed:
                if t.detection is None:
                    continue
                log.setdefault(t.track_id, []).append(
                    (frame_idx, t.detection.length, t.smoothed_length, t.detection.num_points)
                )
            frame_idx += 1

    # Pick top 3 longest-living tracks
    top = sorted(log.items(), key=lambda kv: -len(kv[1]))[:3]
    print(f"\nTop {len(top)} longest-living tracks (raw vs smoothed length):\n")
    for tid, rows in top:
        raw = np.array([r[1] for r in rows])
        sm = np.array([r[2] for r in rows])
        pts_cnt = np.array([r[3] for r in rows])
        print(f"--- track {tid}  frames {rows[0][0]}-{rows[-1][0]}  ({len(rows)} obs) ---")
        print(f"  raw      length:  mean={raw.mean():.2f}  std={raw.std():.2f}  min={raw.min():.2f}  max={raw.max():.2f}")
        print(f"  smoothed length:  mean={sm.mean():.2f}  std={sm.std():.2f}  min={sm.min():.2f}  max={sm.max():.2f}")
        # Frame-to-frame jitter
        raw_jitter = np.abs(np.diff(raw)).mean() if len(raw) > 1 else 0.0
        sm_jitter = np.abs(np.diff(sm)).mean() if len(sm) > 1 else 0.0
        print(f"  mean |Δlength| per frame:  raw={raw_jitter:.3f}  smoothed={sm_jitter:.3f}  (reduction {(1-sm_jitter/max(raw_jitter,1e-6))*100:.0f}%)")
        # Show first 10 rows for sanity
        print(f"  first 10:  raw->smoothed  (pts)")
        for r in rows[:10]:
            print(f"    f{r[0]:4d}: {r[1]:5.2f} -> {r[2]:5.2f}   (pts={r[3]})")


if __name__ == "__main__":
    main()
