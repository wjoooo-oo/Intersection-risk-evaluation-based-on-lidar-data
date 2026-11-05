"""RViz2 visualization: Markers for ROI polygon, bounding boxes, centroids, labels."""

from __future__ import annotations

import math
from typing import List

import numpy as np
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Quaternion
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

from .clustering_detector import Detection
from .tracker import Track


def _yaw_to_quat(yaw: float) -> Quaternion:
    """Quaternion for rotation around +Z by yaw (radians)."""
    half = 0.5 * float(yaw)
    return Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))


def _obb_line_strip_points(det: Detection) -> list[Point]:
    """8 corners forming the wire-frame OBB (bottom rectangle + top + 4 risers)."""
    if det.corners_xy is None:
        return []
    corners = np.asarray(det.corners_xy, dtype=np.float64)
    z_bot = float(det.bbox_min[2])
    z_top = float(det.bbox_max[2])
    bot = [(float(c[0]), float(c[1]), z_bot) for c in corners]
    top = [(float(c[0]), float(c[1]), z_top) for c in corners]

    # LINE_LIST: list of vertex pairs. 4 bottom + 4 top + 4 risers = 12 segments.
    segs: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    for i in range(4):
        j = (i + 1) % 4
        segs.append((bot[i], bot[j]))
        segs.append((top[i], top[j]))
        segs.append((bot[i], top[i]))

    pts: list[Point] = []
    for a, b in segs:
        pts.append(Point(x=a[0], y=a[1], z=a[2]))
        pts.append(Point(x=b[0], y=b[1], z=b[2]))
    return pts


def _rgba(color_list: list[float]) -> ColorRGBA:
    return ColorRGBA(r=color_list[0], g=color_list[1], b=color_list[2], a=color_list[3])


def _lifetime(sec: float = 0.3) -> Duration:
    s = int(sec)
    ns = int((sec - s) * 1e9)
    return Duration(sec=s, nanosec=ns)


def make_roi_marker(
    polygon_points: list[list[float]],
    header: Header,
    color: list[float],
    z_base: float = -2.0,
    height: float = 3.0,
) -> Marker:
    """LINE_STRIP marker for the ROI polygon boundary (extruded vertically)."""
    marker = Marker()
    marker.header = header
    marker.ns = "roi_polygon"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.12
    marker.color = _rgba(color)
    marker.lifetime = _lifetime(0.0)  # persistent

    # Bottom ring
    for pt in polygon_points:
        marker.points.append(Point(x=pt[0], y=pt[1], z=z_base))
    # Close the polygon
    marker.points.append(Point(x=polygon_points[0][0], y=polygon_points[0][1], z=z_base))

    return marker


def make_delete_all_marker(header: Header) -> MarkerArray:
    """MarkerArray with DELETEALL to clear previous frame's markers."""
    m = Marker()
    m.header = header
    m.action = Marker.DELETEALL
    return MarkerArray(markers=[m])


def color_for_class(class_name: str, vis_cfg: dict) -> list[float]:
    if class_name == "vehicle":
        return vis_cfg.get("vehicle_color", [0.0, 1.0, 0.0, 0.6])
    elif class_name == "pedestrian":
        return vis_cfg.get("pedestrian_color", [1.0, 1.0, 0.0, 0.6])
    else:
        return vis_cfg.get("unknown_color", [0.5, 0.5, 0.5, 0.4])


def detections_to_marker_array(
    detections: List[Detection],
    header: Header,
    vis_cfg: dict,
) -> MarkerArray:
    """Build a MarkerArray with bbox CUBE, centroid SPHERE, and TEXT for each detection."""
    markers: list[Marker] = []

    # DELETEALL first to avoid stale markers
    delete_m = Marker()
    delete_m.header = header
    delete_m.ns = "detections"
    delete_m.action = Marker.DELETEALL
    markers.append(delete_m)

    for i, det in enumerate(detections):
        color = color_for_class(det.class_name, vis_cfg)
        base_id = i * 4
        quat = _yaw_to_quat(det.yaw)

        cube = Marker()
        cube.header = header
        cube.ns = "det_bbox"
        cube.id = base_id
        cube.type = Marker.CUBE
        cube.action = Marker.ADD
        cube.pose.position.x = float(det.centroid[0])
        cube.pose.position.y = float(det.centroid[1])
        cube.pose.position.z = float(det.centroid[2])
        cube.pose.orientation = quat
        cube.scale.x = max(det.length, 0.1)
        cube.scale.y = max(det.width, 0.1)
        cube.scale.z = max(det.height, 0.1)
        cube.color = _rgba(color)
        cube.lifetime = _lifetime(0.3)
        markers.append(cube)

        wire_pts = _obb_line_strip_points(det)
        if wire_pts:
            wire = Marker()
            wire.header = header
            wire.ns = "det_wire"
            wire.id = base_id + 1
            wire.type = Marker.LINE_LIST
            wire.action = Marker.ADD
            wire.pose.orientation.w = 1.0
            wire.scale.x = 0.05
            wc = color.copy()
            wc[3] = 1.0
            wire.color = _rgba(wc)
            wire.lifetime = _lifetime(0.3)
            wire.points = wire_pts
            markers.append(wire)

        sphere = Marker()
        sphere.header = header
        sphere.ns = "det_centroid"
        sphere.id = base_id + 2
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = float(det.centroid[0])
        sphere.pose.position.y = float(det.centroid[1])
        sphere.pose.position.z = float(det.centroid[2])
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = 0.3
        sphere.scale.y = 0.3
        sphere.scale.z = 0.3
        c = color.copy()
        c[3] = 1.0
        sphere.color = _rgba(c)
        sphere.lifetime = _lifetime(0.3)
        markers.append(sphere)

        text = Marker()
        text.header = header
        text.ns = "det_text"
        text.id = base_id + 3
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(det.centroid[0])
        text.pose.position.y = float(det.centroid[1])
        text.pose.position.z = float(det.bbox_max[2]) + 0.5
        text.pose.orientation.w = 1.0
        text.scale.z = 0.4
        text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        yaw_deg = math.degrees(det.yaw)
        text.text = (
            f"{det.class_name}\n"
            f"{det.length:.1f}x{det.width:.1f}x{det.height:.1f}m  "
            f"yaw={yaw_deg:+.0f}\n"
            f"pts:{det.num_points}"
        )
        text.lifetime = _lifetime(0.3)
        markers.append(text)

    return MarkerArray(markers=markers)


def tracks_to_marker_array(
    tracks: List[Track],
    header: Header,
    vis_cfg: dict,
) -> MarkerArray:
    """Build MarkerArray for confirmed tracks: bbox, ID text, trajectory line."""
    markers: list[Marker] = []

    # DELETEALL
    delete_m = Marker()
    delete_m.header = header
    delete_m.ns = "tracking"
    delete_m.action = Marker.DELETEALL
    markers.append(delete_m)

    for i, trk in enumerate(tracks):
        color = color_for_class(trk.class_name, vis_cfg)
        base_id = i * 4
        yaw = float(trk.smoothed_yaw) if trk.smoothed_length > 0 else (
            trk.detection.yaw if trk.detection is not None else 0.0
        )
        length = max(float(trk.smoothed_length), 0.1)
        width = max(float(trk.smoothed_width), 0.1)
        height = max(float(trk.smoothed_height), 0.1)

        cube = Marker()
        cube.header = header
        cube.ns = "track_bbox"
        cube.id = base_id
        cube.type = Marker.CUBE
        cube.action = Marker.ADD
        cube.pose.position.x = float(trk.centroid[0])
        cube.pose.position.y = float(trk.centroid[1])
        cube.pose.position.z = float(trk.centroid[2])
        cube.pose.orientation = _yaw_to_quat(yaw)
        cube.scale.x = length
        cube.scale.y = width
        cube.scale.z = height
        c = color.copy()
        c[3] = 0.5
        cube.color = _rgba(c)
        cube.lifetime = _lifetime(0.3)
        markers.append(cube)

        # Wire-frame OBB built from smoothed pose (more stable than detection's)
        cy_, sy_ = math.cos(yaw), math.sin(yaw)
        half = np.array([
            [length / 2,  width / 2],
            [length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2,  width / 2],
        ])
        Rmat = np.array([[cy_, -sy_], [sy_, cy_]])
        center_xy = np.array([float(trk.centroid[0]), float(trk.centroid[1])])
        corners = half @ Rmat.T + center_xy
        z_bot = float(trk.centroid[2]) - height / 2.0
        z_top = float(trk.centroid[2]) + height / 2.0

        wire = Marker()
        wire.header = header
        wire.ns = "track_wire"
        wire.id = base_id + 3
        wire.type = Marker.LINE_LIST
        wire.action = Marker.ADD
        wire.pose.orientation.w = 1.0
        wire.scale.x = 0.06
        wc = color.copy()
        wc[3] = 1.0
        wire.color = _rgba(wc)
        wire.lifetime = _lifetime(0.3)
        for k in range(4):
            j = (k + 1) % 4
            wire.points.append(Point(x=float(corners[k, 0]), y=float(corners[k, 1]), z=z_bot))
            wire.points.append(Point(x=float(corners[j, 0]), y=float(corners[j, 1]), z=z_bot))
            wire.points.append(Point(x=float(corners[k, 0]), y=float(corners[k, 1]), z=z_top))
            wire.points.append(Point(x=float(corners[j, 0]), y=float(corners[j, 1]), z=z_top))
            wire.points.append(Point(x=float(corners[k, 0]), y=float(corners[k, 1]), z=z_bot))
            wire.points.append(Point(x=float(corners[k, 0]), y=float(corners[k, 1]), z=z_top))
        markers.append(wire)

        # --- TEXT: track ID + class + velocity ---
        text = Marker()
        text.header = header
        text.ns = "track_id"
        text.id = base_id + 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(trk.centroid[0])
        text.pose.position.y = float(trk.centroid[1])
        text.pose.position.z = float(trk.centroid[2]) + 2.0
        text.pose.orientation.w = 1.0
        text.scale.z = 0.5
        text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        speed = trk.recent_speed(window=5, dt=0.1)
        text.text = f"ID:{trk.track_id} {trk.class_name}\n{speed:.1f} m/s"
        text.lifetime = _lifetime(0.3)
        markers.append(text)

        # --- LINE_STRIP: trajectory ---
        if len(trk.trajectory) >= 2:
            line = Marker()
            line.header = header
            line.ns = "track_trajectory"
            line.id = base_id + 2
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.08
            tc = color.copy()
            tc[3] = 0.9
            line.color = _rgba(tc)
            line.lifetime = _lifetime(0.3)
            for pt in trk.trajectory:
                line.points.append(Point(x=float(pt[0]), y=float(pt[1]), z=float(pt[2])))
            markers.append(line)

    return MarkerArray(markers=markers)
