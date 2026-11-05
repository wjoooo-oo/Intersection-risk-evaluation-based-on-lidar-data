#!/usr/bin/env python3
"""Pick intersection ROI polygon in RViz via Publish Point -> /clicked_point."""

from __future__ import annotations

import argparse
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import Point, PointStamped
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray


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

FRAME_ID = "lidar"
DEFAULT_Z = 0.05
LINE_WIDTH = 0.35
CYAN = (0.0, 1.0, 1.0, 0.95)


class RoiDrawer(Node):
    def __init__(
        self,
        *,
        output_yaml: Path,
        bag_uri: Path,
        topic: str,
        n_frames: int,
        frame_stride: int,
        z_min: float,
        z_max: float,
        resolution: float,
        margin: float,
    ) -> None:
        super().__init__("roi_drawer")
        self.output_yaml = output_yaml
        self.bag_uri = bag_uri
        self.topic = topic
        self.n_frames = n_frames
        self.frame_stride = frame_stride
        self.z_min = z_min
        self.z_max = z_max
        self.resolution = resolution
        self.margin = margin
        self.frame_id = FRAME_ID
        self.vertices: list[list[float]] = []

        self.marker_pub = self.create_publisher(MarkerArray, "/roi_drawer/markers", 10)
        self.create_subscription(PointStamped, "/clicked_point", self._on_click, 10)
        self.create_subscription(String, "/roi_drawer_cmd", self._on_cmd, 10)

        if self.output_yaml.is_file():
            self._load_yaml(self.output_yaml)

        self.get_logger().info(f"ROI drawer -> {self.output_yaml}")
        self.get_logger().info(
            "RViz: Publish Point on point cloud. Commands: "
            "bash scripts/roi_drawer_cmd.sh <undo|clear|save|reload>"
        )
        self._publish_markers()

    def _load_yaml(self, path: Path) -> None:
        with path.open(encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        self.vertices = [[float(p[0]), float(p[1])] for p in doc.get("polygon_xy", [])]
        self.frame_id = doc.get("frame_id", FRAME_ID)
        acc = doc.get("accumulation", {})
        self.n_frames = int(acc.get("n_frames", self.n_frames))
        self.frame_stride = int(acc.get("frame_stride", self.frame_stride))
        self.z_min = float(acc.get("z_min", self.z_min))
        self.z_max = float(acc.get("z_max", self.z_max))
        bev = doc.get("bev", {})
        self.resolution = float(bev.get("resolution_m", self.resolution))
        self.get_logger().info(f"Loaded {len(self.vertices)} vertices from {path}")

    def _on_click(self, msg: PointStamped) -> None:
        x, y = float(msg.point.x), float(msg.point.y)
        self.vertices.append([x, y])
        n = len(self.vertices)
        self.get_logger().info(f"vertex {n}: ({x:.3f}, {y:.3f})")
        self._publish_markers()

    def _on_cmd(self, msg: String) -> None:
        cmd = msg.data.strip().lower()
        aliases = {"u": "undo", "c": "clear", "s": "save", "r": "reload"}
        cmd = aliases.get(cmd, cmd)
        handlers = {
            "undo": self._cmd_undo,
            "clear": self._cmd_clear,
            "save": self._cmd_save,
            "reload": self._cmd_reload,
        }
        handler = handlers.get(cmd)
        if handler is None:
            self.get_logger().warn(f"unknown command: {msg.data!r}")
            return
        handler()

    def _cmd_undo(self) -> None:
        if not self.vertices:
            self.get_logger().warn("no vertices to undo")
            return
        removed = self.vertices.pop()
        self.get_logger().info(f"undo -> removed ({removed[0]:.3f}, {removed[1]:.3f})")
        self._publish_markers()

    def _cmd_clear(self) -> None:
        self.vertices.clear()
        self.get_logger().info("cleared polygon")
        self._publish_markers()

    def _cmd_reload(self) -> None:
        if not self.output_yaml.is_file():
            self.get_logger().error(f"yaml not found: {self.output_yaml}")
            return
        self._load_yaml(self.output_yaml)
        self._publish_markers()

    def _bev_extent(self) -> tuple[float, float, float, float]:
        if not self.vertices:
            return (-50.0, 50.0, -50.0, 50.0)
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        m = self.margin
        return min(xs) - m, max(xs) + m, min(ys) - m, max(ys) + m

    def _cmd_save(self) -> None:
        if len(self.vertices) < 3:
            self.get_logger().error(f"need >= 3 vertices (have {len(self.vertices)})")
            return
        extent = self._bev_extent()
        write_roi_yaml(
            self.output_yaml,
            bag_uri=self.bag_uri.resolve(),
            topic=self.topic,
            frame_id=self.frame_id,
            n_frames=self.n_frames,
            z_min=self.z_min,
            z_max=self.z_max,
            frame_stride=self.frame_stride,
            resolution=self.resolution,
            extent=extent,
            polygon_xy=self.vertices,
        )
        self.get_logger().info(f"saved {len(self.vertices)} vertices -> {self.output_yaml}")

    def _publish_markers(self) -> None:
        arr = MarkerArray()
        delete_all = Marker()
        delete_all.header.frame_id = self.frame_id
        delete_all.action = Marker.DELETEALL
        arr.markers.append(delete_all)

        if self.vertices:
            spheres = Marker()
            spheres.header.frame_id = self.frame_id
            spheres.header.stamp = self.get_clock().now().to_msg()
            spheres.ns = "roi_vertices"
            spheres.id = 0
            spheres.type = Marker.SPHERE_LIST
            spheres.action = Marker.ADD
            spheres.scale.x = spheres.scale.y = spheres.scale.z = 0.6
            spheres.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
            spheres.pose.orientation.w = 1.0
            for vx, vy in self.vertices:
                p = Point(x=vx, y=vy, z=DEFAULT_Z)
                spheres.points.append(p)
            arr.markers.append(spheres)

        if len(self.vertices) >= 2:
            line = Marker()
            line.header.frame_id = self.frame_id
            line.header.stamp = self.get_clock().now().to_msg()
            line.ns = "roi_polygon"
            line.id = 1
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = LINE_WIDTH
            line.color = ColorRGBA(r=CYAN[0], g=CYAN[1], b=CYAN[2], a=CYAN[3])
            line.pose.orientation.w = 1.0
            for vx, vy in self.vertices:
                line.points.append(Point(x=vx, y=vy, z=DEFAULT_Z))
            if len(self.vertices) >= 3:
                vx, vy = self.vertices[0]
                line.points.append(Point(x=vx, y=vy, z=DEFAULT_Z))
            arr.markers.append(line)

        self.marker_pub.publish(arr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--topic", default="/lidar/points")
    parser.add_argument("-n", "--n-frames", type=int, default=80)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--z-min", type=float, default=-2.0)
    parser.add_argument("--z-max", type=float, default=3.0)
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=8.0, help="BEV padding around polygon (m)")
    args = parser.parse_args()

    rclpy.init()
    node = RoiDrawer(
        output_yaml=args.output,
        bag_uri=args.bag,
        topic=args.topic,
        n_frames=args.n_frames,
        frame_stride=args.frame_stride,
        z_min=args.z_min,
        z_max=args.z_max,
        resolution=args.resolution,
        margin=args.margin,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
