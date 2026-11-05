#!/usr/bin/env python3
"""Accumulate lidar frames from a rosbag2, show BEV, pick a 2D polygon, write roi.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backend_bases import KeyEvent
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore


def _parse_pointcloud2(msg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
    x = data[:, 0:4].view(np.float32).reshape(-1)
    y = data[:, 4:8].view(np.float32).reshape(-1)
    z = data[:, 8:12].view(np.float32).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return x[valid], y[valid], z[valid]


def load_accumulated_points(
    bag_uri: Path,
    topic: str,
    n_frames: int,
    z_min: float,
    z_max: float,
    frame_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    frame_id = "lidar"

    with Reader(bag_uri) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            topics = sorted({c.topic for c in reader.connections})
            raise SystemExit(f"topic {topic!r} not in bag. Topics: {topics}")

        got = 0
        scanned = 0
        stride = max(1, int(frame_stride))
        for conn, _timestamp, rawdata in reader.messages(connections=connections):
            scanned += 1
            if (scanned - 1) % stride != 0:
                continue
            msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
            frame_id = msg.header.frame_id or frame_id
            x, y, z = _parse_pointcloud2(msg)
            keep = (z >= z_min) & (z <= z_max)
            xs.append(x[keep])
            ys.append(y[keep])
            zs.append(z[keep])
            got += 1
            print(
                f"  frame {got}/{n_frames} (scan {scanned}, stride {stride}): "
                f"{int(keep.sum())} points (z in [{z_min}, {z_max}])"
            )
            if got >= n_frames:
                break

    if got < n_frames:
        print(f"warning: only {got} point-cloud messages (requested {n_frames})", file=sys.stderr)

    return np.concatenate(xs), np.concatenate(ys), np.concatenate(zs), frame_id


def bev_extent(
    x: np.ndarray,
    y: np.ndarray,
    margin: float,
    p_low: float,
    p_high: float,
) -> tuple[float, float, float, float]:
    x0, x1 = np.percentile(x, [p_low, p_high])
    y0, y1 = np.percentile(y, [p_low, p_high])
    return x0 - margin, x1 + margin, y0 - margin, y1 + margin


def make_bev_image(
    x: np.ndarray,
    y: np.ndarray,
    extent: tuple[float, float, float, float],
    resolution: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max, y_min, y_max = extent
    x_edges = np.arange(x_min, x_max + resolution, resolution)
    y_edges = np.arange(y_min, y_max + resolution, resolution)
    if len(x_edges) < 2 or len(y_edges) < 2:
        raise SystemExit("BEV extent too small; adjust --margin or percentiles.")
    hist, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    return hist.T, x_edges, y_edges


def pick_polygon(
    bev: np.ndarray,
    extent: tuple[float, float, float, float],
    title: str,
) -> list[list[float]]:
    x_min, x_max, y_min, y_max = extent
    vertices: list[list[float]] = []
    fig, ax = plt.subplots(figsize=(24, 24))
    ax.imshow(
        np.log1p(bev),
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap="gray",
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xlabel("X (m) — lidar forward")
    ax.set_ylabel("Y (m) — lidar left")
    ax.set_title(title)
    (line,) = ax.plot([], [], "c-", linewidth=1.5)
    (markers,) = ax.plot([], [], "co", markersize=6)

    def refresh() -> None:
        if not vertices:
            line.set_data([], [])
            markers.set_data([], [])
        else:
            xs = [v[0] for v in vertices] + ([vertices[0][0]] if len(vertices) >= 3 else [])
            ys = [v[1] for v in vertices] + ([vertices[0][1]] if len(vertices) >= 3 else [])
            line.set_data(xs, ys)
            markers.set_data([v[0] for v in vertices], [v[1] for v in vertices])
        fig.canvas.draw_idle()

    def onclick(event) -> None:
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            vertices.append([float(event.xdata), float(event.ydata)])
            print(f"  vertex {len(vertices)}: ({event.xdata:.3f}, {event.ydata:.3f})")
            refresh()
        elif event.button == 3 and vertices:
            removed = vertices.pop()
            print(f"  undo -> removed ({removed[0]:.3f}, {removed[1]:.3f})")
            refresh()

    def onkey(event: KeyEvent) -> None:
        if event.key in ("enter", "return"):
            if len(vertices) < 3:
                print(f"need at least 3 vertices (have {len(vertices)})")
                return
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)
    instructions = (
        "Left-click: add vertex | Right-click: undo | Enter: finish (min 3 vertices)"
    )
    fig.text(0.5, 0.01, instructions, ha="center", fontsize=9, color="0.85")
    print(instructions)
    plt.tight_layout()
    plt.show()
    return vertices


def write_roi_yaml(
    path: Path,
    *,
    bag_uri: Path,
    topic: str,
    frame_id: str,
    n_frames: int,
    z_min: float,
    z_max: float,
    frame_stride: int,
    resolution: float,
    extent: tuple[float, float, float, float],
    polygon_xy: list[list[float]],
) -> None:
    x_min, x_max, y_min, y_max = extent
    doc = {
        "source_bag": str(bag_uri),
        "topic": topic,
        "frame_id": frame_id,
        "accumulation": {
            "n_frames": n_frames,
            "frame_stride": frame_stride,
            "z_min": z_min,
            "z_max": z_max,
        },
        "bev": {
            "resolution_m": resolution,
            "x_min": float(x_min),
            "x_max": float(x_max),
            "y_min": float(y_min),
            "y_max": float(y_max),
        },
        "polygon_xy": [[float(px), float(py)] for px, py in polygon_xy],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
    print(f"wrote {path} ({len(polygon_xy)} vertices)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bag",
        type=Path,
        default=Path("<BAG_DIR>/session_a"),
        help="rosbag2 directory (metadata.yaml + .db3)",
    )
    parser.add_argument("--topic", default="/lidar/points")
    parser.add_argument("-n", "--n-frames", type=int, default=30, help="point-cloud frames to merge")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="use every Nth scan (1=consecutive); widens time coverage when n_frames is fixed",
    )
    parser.add_argument("--z-min", type=float, default=-1.5)
    parser.add_argument("--z-max", type=float, default=2.5)
    parser.add_argument("--resolution", type=float, default=0.1, help="BEV cell size (m)")
    parser.add_argument("--margin", type=float, default=2.0, help="padding around auto extent (m)")
    parser.add_argument("--percentile-low", type=float, default=2.0)
    parser.add_argument("--percentile-high", type=float, default=98.0)
    parser.add_argument("--x-min", type=float, default=None, help="override BEV x_min (m)")
    parser.add_argument("--x-max", type=float, default=None, help="override BEV x_max (m)")
    parser.add_argument("--y-min", type=float, default=None, help="override BEV y_min (m)")
    parser.add_argument("--y-max", type=float, default=None, help="override BEV y_max (m)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("config/roi.yaml"),
        help="output roi.yaml path",
    )
    args = parser.parse_args()

    if not args.bag.is_dir():
        raise SystemExit(f"bag not found: {args.bag}")

    print(f"reading {args.bag} topic {args.topic} …")
    x, y, z, frame_id = load_accumulated_points(
        args.bag,
        args.topic,
        args.n_frames,
        args.z_min,
        args.z_max,
        args.frame_stride,
    )
    print(f"merged {len(x)} points, frame_id={frame_id!r}")

    extent = bev_extent(x, y, args.margin, args.percentile_low, args.percentile_high)
    ex_min, ex_max, ey_min, ey_max = extent
    if args.x_min is not None:
        ex_min = args.x_min
    if args.x_max is not None:
        ex_max = args.x_max
    if args.y_min is not None:
        ey_min = args.y_min
    if args.y_max is not None:
        ey_max = args.y_max
    extent = (ex_min, ex_max, ey_min, ey_max)
    bev, _, _ = make_bev_image(x, y, extent, args.resolution)
    title = f"BEV — {args.bag.name} ({args.n_frames} frames, z∈[{args.z_min},{args.z_max}])"

    polygon = pick_polygon(bev, extent, title)
    if len(polygon) < 3:
        print("no polygon saved (need ≥3 vertices and Enter)", file=sys.stderr)
        return 1

    write_roi_yaml(
        args.output,
        bag_uri=args.bag.resolve(),
        topic=args.topic,
        frame_id=frame_id,
        n_frames=args.n_frames,
        z_min=args.z_min,
        z_max=args.z_max,
        frame_stride=args.frame_stride,
        resolution=args.resolution,
        extent=extent,
        polygon_xy=polygon,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
