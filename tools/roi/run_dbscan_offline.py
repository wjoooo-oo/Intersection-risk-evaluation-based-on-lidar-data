#!/usr/bin/env python3
"""Offline DBSCAN detection on a rosbag2 — reuse repo pipeline, no ROS runtime.

Pipeline (mirrors detection_node._lidar_callback):
  PointCloud2 → numpy xyz → ROI crop → height filter → voxel downsample
   → ground removal → DBSCAN → bbox + classify → background filter
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "detection"
sys.path.insert(0, str(PKG))

from intersection_lidar_detection.background_filter import BackgroundFilter  # noqa: E402
from intersection_lidar_detection.clustering_detector import (  # noqa: E402
    Detection,
    detect_objects,
)
from intersection_lidar_detection.roi_utils import (  # noqa: E402
    crop_points_to_roi,
    make_polygon_path,
)
from intersection_lidar_detection.tracker import MultiObjectTracker  # noqa: E402

CLASS_COLOR = {
    "vehicle": (0.0, 1.0, 0.0),
    "pedestrian": (1.0, 1.0, 0.0),
    "unknown": (0.5, 0.5, 0.5),
}


def parse_pointcloud2_xyz(msg) -> np.ndarray:
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
    x = data[:, 0:4].view(np.float32).reshape(-1)
    y = data[:, 4:8].view(np.float32).reshape(-1)
    z = data[:, 8:12].view(np.float32).reshape(-1)
    pts = np.column_stack([x, y, z])
    valid = np.isfinite(pts).all(axis=1)
    return pts[valid]


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0 or len(points) == 0:
        return points
    q = np.floor(points[:, :3] / voxel_size).astype(np.int32)
    _, idx, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    sums = np.zeros((len(idx), 3), dtype=np.float64)
    counts = np.zeros(len(idx), dtype=np.int32)
    np.add.at(sums, inv, points[:, :3])
    np.add.at(counts, inv, 1)
    return (sums / counts[:, None]).astype(np.float32)


def height_filter(points: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    m = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    return points[m]


def remove_ground(points: np.ndarray, ground_z_max: float) -> np.ndarray:
    return points[points[:, 2] > ground_z_max]


def remove_ground_grid(
    points: np.ndarray,
    cell: float = 1.0,
    delta: float = 0.25,
    z_global_max: float | None = None,
) -> np.ndarray:
    """Mirror of pointcloud_utils.remove_ground_grid (inlined to avoid ROS import)."""
    if len(points) == 0:
        return points
    qx = np.floor(points[:, 0] / cell).astype(np.int64)
    qy = np.floor(points[:, 1] / cell).astype(np.int64)
    keys = (qx + (1 << 20)) * (1 << 22) + (qy + (1 << 20))
    order = np.argsort(keys, kind="stable")
    keys_s = keys[order]
    z_s = points[order, 2]
    uniq_keys, start = np.unique(keys_s, return_index=True)
    cell_zmin = np.minimum.reduceat(z_s, start)
    inv = np.searchsorted(uniq_keys, keys)
    is_nonground = points[:, 2] > (cell_zmin[inv] + delta)
    if z_global_max is not None:
        is_nonground &= points[:, 2] <= z_global_max
    return points[is_nonground]


def render_frame(
    ax,
    raw_xy: np.ndarray,
    roi_xy: np.ndarray,
    detections: list[Detection],
    dynamic: list[Detection],
    polygon_xy: list[list[float]],
    extent: tuple[float, float, float, float],
    title: str,
) -> None:
    x_min, x_max, y_min, y_max = extent

    bg_mask = (
        (raw_xy[:, 0] >= x_min)
        & (raw_xy[:, 0] <= x_max)
        & (raw_xy[:, 1] >= y_min)
        & (raw_xy[:, 1] <= y_max)
    )
    if bg_mask.any():
        ax.scatter(
            raw_xy[bg_mask, 0],
            raw_xy[bg_mask, 1],
            s=0.2,
            c="0.55",
            alpha=0.35,
            linewidths=0,
        )
    if len(roi_xy):
        ax.scatter(
            roi_xy[:, 0], roi_xy[:, 1], s=0.6, c="white", alpha=0.85, linewidths=0
        )

    ax.add_patch(
        MplPolygon(
            polygon_xy,
            closed=True,
            fill=False,
            edgecolor="cyan",
            linewidth=1.5,
            linestyle="--",
        )
    )

    dyn_ids = {id(d) for d in dynamic}
    for det in detections:
        col = CLASS_COLOR.get(det.class_name, (0.7, 0.7, 0.7))
        is_dyn = id(det) in dyn_ids
        lw = 2.0 if is_dyn else 1.0
        ls = "-" if is_dyn else ":"

        if det.corners_xy is not None:
            poly = MplPolygon(
                det.corners_xy,
                closed=True,
                fill=False,
                edgecolor=col,
                linewidth=lw,
                linestyle=ls,
            )
            ax.add_patch(poly)
            head = det.corners_xy[:2].mean(axis=0)
            ax.plot(
                [det.centroid[0], head[0]],
                [det.centroid[1], head[1]],
                color=col,
                linewidth=lw * 0.7,
                linestyle=ls,
            )
        else:
            rect = Rectangle(
                (det.bbox_min[0], det.bbox_min[1]),
                det.length,
                det.width,
                fill=False,
                edgecolor=col,
                linewidth=lw,
                linestyle=ls,
            )
            ax.add_patch(rect)
        ax.text(
            det.centroid[0],
            det.centroid[1],
            f"{det.class_name[:3]}",
            color=col,
            fontsize=7,
            ha="center",
            va="center",
        )

    ax.plot(0, 0, "r+", markersize=10, markeredgewidth=1.5)
    ax.set_facecolor("black")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.15, color="white")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bag",
        type=Path,
        default=Path("<BAG_DIR>/session_a"),
    )
    parser.add_argument("--topic", default="/lidar/points")
    parser.add_argument(
        "--roi-yaml", type=Path, default=REPO / "config" / "roi.yaml"
    )
    parser.add_argument(
        "--detection-config",
        type=Path,
        default=PKG / "config" / "detection_config.yaml",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=200,
        help="frames to process (>= warmup_frames so bg filter activates)",
    )
    parser.add_argument(
        "--vis-indices",
        type=int,
        nargs="*",
        default=[40, 80, 120, 160, 199],
        help="frame indices to render (after bg filter warmup)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "output" / "dbscan_demo",
    )
    parser.add_argument(
        "--variant",
        choices=["new", "old", "both"],
        default="new",
        help="'new'=grid ground + OBB, 'old'=height-threshold + AABB, 'both'=side-by-side",
    )
    args = parser.parse_args()

    with open(args.roi_yaml) as f:
        roi_cfg = yaml.safe_load(f)
    with open(args.detection_config) as f:
        det_cfg = yaml.safe_load(f)

    polygon_xy = roi_cfg["polygon_xy"]
    bev = roi_cfg["bev"]
    extent = (
        float(bev["x_min"]),
        float(bev["x_max"]),
        float(bev["y_min"]),
        float(bev["y_max"]),
    )
    roi_path = make_polygon_path(polygon_xy)

    preproc = det_cfg["preprocess"]
    z_min = float(preproc["z_min"])
    z_max = float(preproc["z_max"])
    voxel_size = float(preproc["voxel_size"])
    gr = preproc["ground_removal"]
    cluster_cfg = det_cfg["clustering"]
    class_cfg = det_cfg["classification"]
    bg_cfg = det_cfg.get("background_filter", {})
    trk_cfg = det_cfg.get("tracking", {})
    min_track_speed = float(trk_cfg.get("min_track_speed", 0.0))
    speed_window = int(trk_cfg.get("speed_window", 5))
    dt_default = 0.1

    variants_to_run: list[str]
    if args.variant == "both":
        variants_to_run = ["old", "new"]
    else:
        variants_to_run = [args.variant]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    def make_variant_cfg(name: str) -> dict:
        """Return per-variant settings used by the pipeline."""
        if name == "old":
            return {
                "ground_fn": lambda pts: remove_ground(pts, ground_z_max=-1.4),
                "cluster_cfg": {**cluster_cfg, "eps": 0.7, "min_samples": 8,
                                "min_cluster_points": 15, "bbox_method": "aabb"},
            }
        return {
            "ground_fn": lambda pts: remove_ground_grid(
                pts,
                cell=float(gr.get("cell", 1.0)),
                delta=float(gr.get("delta", 0.25)),
                z_global_max=gr.get("z_global_max"),
            ),
            "cluster_cfg": {**cluster_cfg, "bbox_method": "obb"},
        }

    results: dict[str, dict] = {}

    for variant in variants_to_run:
        print(f"\n=== variant: {variant} ===")
        vcfg = make_variant_cfg(variant)
        bg_filter = BackgroundFilter(
            resolution=bg_cfg.get("resolution", 0.5),
            history_frames=bg_cfg.get("history_frames", 50),
            static_threshold=bg_cfg.get("static_threshold", 0.8),
            warmup_frames=bg_cfg.get("warmup_frames", 20),
        )
        tracker = MultiObjectTracker(
            max_age=trk_cfg.get("max_age", 10),
            min_hits=trk_cfg.get("min_hits", 3),
            min_track_age=trk_cfg.get("min_track_age", 0),
            distance_threshold=trk_cfg.get("distance_threshold", 3.0),
            process_noise_pos=trk_cfg.get("process_noise_pos", 0.5),
            process_noise_vel=trk_cfg.get("process_noise_vel", 1.0),
            measurement_noise=trk_cfg.get("measurement_noise", 0.3),
            max_trajectory_length=trk_cfg.get("max_trajectory_length", 100),
            size_ema_alpha=trk_cfg.get("size_ema_alpha", 0.25),
            size_ema_min_points=trk_cfg.get("size_ema_min_points", 12),
        ) if variant == "new" else None

        per_frame_stats: list[dict] = []
        rendered: list[dict] = []
        target_idx = set(args.vis_indices)

        with Reader(args.bag) as reader:
            conns = [c for c in reader.connections if c.topic == args.topic]
            if not conns:
                raise SystemExit(f"topic {args.topic!r} not in bag")
            frame_idx = 0
            for conn, _ts, raw in reader.messages(connections=conns):
                if frame_idx >= args.n_frames:
                    break
                msg = typestore.deserialize_cdr(raw, conn.msgtype)
                xyz = parse_pointcloud2_xyz(msg)

                roi_pts, _ = crop_points_to_roi(xyz, roi_path)
                roi_pts = height_filter(roi_pts, z_min, z_max)
                ds = voxel_downsample(roi_pts, voxel_size)
                ng = vcfg["ground_fn"](ds)

                detections = detect_objects(ng, vcfg["cluster_cfg"], class_cfg)
                dynamic = bg_filter.filter_detections(detections)

                # Speed-based static suppression (only "new" variant)
                if tracker is not None and min_track_speed > 0:
                    confirmed = tracker.step(dynamic, dt=dt_default)
                    # Determine which tracks are slow (=> static noise)
                    drop_centers = []
                    for t in tracker.tracks:
                        if len(t.trajectory) > speed_window and \
                                t.recent_speed(window=speed_window, dt=dt_default) < min_track_speed:
                            drop_centers.append(t.centroid[:2])
                    if drop_centers:
                        drop_arr = np.array(drop_centers)
                        dynamic = [
                            d for d in dynamic
                            if np.linalg.norm(drop_arr - d.centroid[:2], axis=1).min() > 0.8
                        ]

                counts = {"vehicle": 0, "pedestrian": 0, "unknown": 0}
                for d in dynamic:
                    counts[d.class_name] = counts.get(d.class_name, 0) + 1

                per_frame_stats.append(
                    {
                        "frame": frame_idx,
                        "raw_points": int(len(xyz)),
                        "roi_points": int(len(roi_pts)),
                        "nonground_points": int(len(ng)),
                        "detections_raw": len(detections),
                        "detections_dynamic": len(dynamic),
                        **{f"dyn_{k}": v for k, v in counts.items()},
                    }
                )

                if frame_idx in target_idx:
                    rendered.append(
                        {
                            "frame": frame_idx,
                            "raw_xyz": xyz,
                            "roi_xyz": ng,
                            "detections": detections,
                            "dynamic": dynamic,
                        }
                    )

                if frame_idx % 40 == 0 or frame_idx == args.n_frames - 1:
                    print(
                        f"  [{variant}] frame {frame_idx:4d}: raw={len(xyz):6d} "
                        f"roi={len(roi_pts):5d} ng={len(ng):5d} "
                        f"det={len(detections):2d} dyn={len(dynamic):2d} "
                        f"({counts['vehicle']}v/{counts['pedestrian']}p/{counts['unknown']}u)"
                    )
                frame_idx += 1

        summary = {
            "variant": variant,
            "bag": str(args.bag),
            "topic": args.topic,
            "n_frames_processed": len(per_frame_stats),
            "totals": {
                "detections_raw": sum(s["detections_raw"] for s in per_frame_stats),
                "detections_dynamic": sum(s["detections_dynamic"] for s in per_frame_stats),
                "dyn_vehicle": sum(s["dyn_vehicle"] for s in per_frame_stats),
                "dyn_pedestrian": sum(s["dyn_pedestrian"] for s in per_frame_stats),
                "dyn_unknown": sum(s["dyn_unknown"] for s in per_frame_stats),
            },
            "averages_per_frame": {
                "raw_det": float(np.mean([s["detections_raw"] for s in per_frame_stats])),
                "dyn_det": float(np.mean([s["detections_dynamic"] for s in per_frame_stats])),
                "veh": float(np.mean([s["dyn_vehicle"] for s in per_frame_stats])),
                "ped": float(np.mean([s["dyn_pedestrian"] for s in per_frame_stats])),
                "unk": float(np.mean([s["dyn_unknown"] for s in per_frame_stats])),
                "nonground_pts": float(np.mean([s["nonground_points"] for s in per_frame_stats])),
            },
        }
        print(
            f"  TOTAL dyn={summary['totals']['detections_dynamic']} "
            f"(veh={summary['totals']['dyn_vehicle']} "
            f"ped={summary['totals']['dyn_pedestrian']} "
            f"unk={summary['totals']['dyn_unknown']})  "
            f"avg/frame dyn={summary['averages_per_frame']['dyn_det']:.2f} "
            f"unk%={summary['averages_per_frame']['unk']/max(summary['averages_per_frame']['dyn_det'],1e-9)*100:.0f}"
        )

        (args.output_dir / f"summary_{variant}.json").write_text(json.dumps(summary, indent=2))

        for r in rendered:
            fig, ax = plt.subplots(figsize=(12, 12))
            title = (
                f"[{variant}] {args.bag.name}  frame {r['frame']}  "
                f"raw_det={len(r['detections'])} dynamic={len(r['dynamic'])}\n"
                f"green=vehicle  yellow=pedestrian  gray=unknown  "
                f"solid=dynamic  dotted=static-filtered"
            )
            render_frame(
                ax,
                r["raw_xyz"][:, :2],
                r["roi_xyz"][:, :2],
                r["detections"],
                r["dynamic"],
                polygon_xy,
                extent,
                title,
            )
            out = args.output_dir / f"{variant}_frame_{r['frame']:04d}.png"
            fig.tight_layout()
            fig.savefig(out, dpi=150, facecolor="black")
            plt.close(fig)

        results[variant] = {"summary": summary, "rendered": rendered}

    # Side-by-side comparison montage
    if "old" in results and "new" in results:
        old = results["old"]["rendered"]
        new = results["new"]["rendered"]
        n = min(len(old), len(new))
        if n > 0:
            fig, axes = plt.subplots(n, 2, figsize=(14, 7 * n))
            if n == 1:
                axes = np.array([axes])
            for i in range(n):
                ro, rn = old[i], new[i]
                render_frame(
                    axes[i, 0],
                    ro["raw_xyz"][:, :2],
                    ro["roi_xyz"][:, :2],
                    ro["detections"],
                    ro["dynamic"],
                    polygon_xy,
                    extent,
                    f"OLD  frame {ro['frame']}  dyn={len(ro['dynamic'])}",
                )
                render_frame(
                    axes[i, 1],
                    rn["raw_xyz"][:, :2],
                    rn["roi_xyz"][:, :2],
                    rn["detections"],
                    rn["dynamic"],
                    polygon_xy,
                    extent,
                    f"NEW  frame {rn['frame']}  dyn={len(rn['dynamic'])}",
                )
            fig.suptitle(
                "LEFT: height-threshold ground + AABB  |  "
                "RIGHT: grid-local-min ground + PCA OBB",
                fontsize=13,
                color="white",
            )
            out = args.output_dir / "compare_old_vs_new.png"
            fig.tight_layout()
            fig.savefig(out, dpi=130, facecolor="black")
            plt.close(fig)
            print(f"\nsaved comparison → {out}")

    # Single-variant montage (when not 'both')
    for variant in variants_to_run:
        if "old" in results and "new" in results:
            break
        rendered = results[variant]["rendered"]
        n = len(rendered)
        if n >= 2:
            cols = min(3, n)
            rows = int(np.ceil(n / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
            axes = np.atleast_1d(axes).flatten()
            for ax, r in zip(axes, rendered):
                render_frame(
                    ax,
                    r["raw_xyz"][:, :2],
                    r["roi_xyz"][:, :2],
                    r["detections"],
                    r["dynamic"],
                    polygon_xy,
                    extent,
                    f"frame {r['frame']}  dyn={len(r['dynamic'])}",
                )
            for ax in axes[len(rendered):]:
                ax.axis("off")
            out = args.output_dir / f"montage_{variant}.png"
            fig.suptitle(
                f"[{variant}] DBSCAN detections — {args.bag.name}",
                fontsize=12,
                color="white",
            )
            fig.tight_layout()
            fig.savefig(out, dpi=130, facecolor="black")
            plt.close(fig)
            print(f"saved {out}")


if __name__ == "__main__":
    main()
