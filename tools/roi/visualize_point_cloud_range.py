#!/usr/bin/env python3
"""BEV visualization of POINT_CLOUD_RANGE box (+ optional ROI polygon) from roi.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pick_roi_polygon import load_accumulated_points, make_bev_image  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi", type=Path, default=root / "config" / "roi.yaml")
    parser.add_argument("--n-frames", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "output" / "point_cloud_range_bev.png",
    )
    args = parser.parse_args()

    with open(args.roi) as f:
        cfg = yaml.safe_load(f)

    bag = Path(cfg["source_bag"])
    topic = cfg.get("topic", "/lidar/points")
    acc = cfg["accumulation"]
    bev = cfg["bev"]
    poly = cfg.get("polygon_xy", [])

    x_min, x_max = float(bev["x_min"]), float(bev["x_max"])
    y_min, y_max = float(bev["y_min"]), float(bev["y_max"])
    z_min, z_max = float(acc["z_min"]), float(acc["z_max"])
    res = float(bev.get("resolution_m", 0.1))

    print(f"Loading {args.n_frames} frames from {bag} ...")
    x, y, _z, frame_id = load_accumulated_points(
        bag, topic, args.n_frames, z_min, z_max
    )
    print(f"merged {len(x)} points, frame_id={frame_id!r}")

    extent = (x_min, x_max, y_min, y_max)
    bev_img, _, _ = make_bev_image(x, y, extent, res)

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(
        np.log1p(bev_img),
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap="gray",
        aspect="equal",
        interpolation="nearest",
    )

    rect = Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        fill=False,
        edgecolor="lime",
        linewidth=2.5,
        label=f"range x[{x_min},{x_max}] y[{y_min:.1f},{y_max:.1f}]",
    )
    ax.add_patch(rect)

    if poly:
        patch = MplPolygon(
            poly,
            closed=True,
            fill=False,
            edgecolor="cyan",
            linewidth=2,
            linestyle="--",
            label="roi polygon_xy",
        )
        ax.add_patch(patch)

    ax.plot(0, 0, "r+", markersize=14, markeredgewidth=2, label="lidar origin")
    ax.arrow(
        0,
        0,
        8,
        0,
        head_width=1.5,
        head_length=1.5,
        fc="red",
        ec="red",
        length_includes_head=True,
        label="+X (lidar forward)",
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X (m) — lidar forward")
    ax.set_ylabel("Y (m) — lidar left")
    ax.set_title(
        f"POINT_CLOUD_RANGE check  |  z in [{z_min}, {z_max}]  |  {args.n_frames} frames\n"
        f"{bag.name}"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
