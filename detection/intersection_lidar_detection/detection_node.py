"""Main ROS2 node: subscribe to LiDAR, run detection + tracking pipeline, publish visualization."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from .roi_utils import make_polygon_path, crop_points_to_roi, shrink_polygon_inward
from .pointcloud_utils import (
    pointcloud2_to_xyz,
    xyz_to_pointcloud2,
    height_filter,
    voxel_downsample,
    voxel_downsample_range_split,
    remove_ground,
)
from .clustering_detector import detect_objects, classify_object, _vehicle_class_ok
from .tracker import Track
from .background_filter import BackgroundFilter, StaticPointFilter
from .tracker import MultiObjectTracker
from .visualization_utils import (
    make_roi_marker,
    detections_to_marker_array,
    tracks_to_marker_array,
    make_delete_all_marker,
)


class DetectionNode(Node):
    def __init__(self):
        super().__init__("intersection_detection_node")

        # Declare parameter for config path
        self.declare_parameter(
            "config_path",
            os.path.join(
                os.path.dirname(__file__), "..", "config", "detection_config.yaml"
            ),
        )
        config_path = self.get_parameter("config_path").get_parameter_value().string_value
        self.cfg = self._load_config(config_path)
        self.get_logger().info(f"loaded config from {config_path}")

        # Build ROI polygon (optional inward margin drops edge tree/shadow clutter)
        roi_cfg = self.cfg["roi"]
        roi_pts = list(roi_cfg["points"])
        erode_m = float(roi_cfg.get("erode_margin_m", 0.0))
        if erode_m > 0:
            roi_pts = shrink_polygon_inward(roi_pts, erode_m)
            self.get_logger().info(f"ROI eroded inward by {erode_m:.1f} m ({len(roi_pts)} vertices)")
        self.roi_path = make_polygon_path(roi_pts)
        self.roi_points = roi_pts

        # Static point filter (multi-frame fixed voxels) + detection-level filter
        sp_cfg = self.cfg.get("preprocess", {}).get("static_point_filter", {})
        self.static_point_filter = None
        if sp_cfg.get("enabled", False):
            self.static_point_filter = StaticPointFilter(
                resolution=sp_cfg.get("resolution", 0.5),
                history_frames=sp_cfg.get("history_frames", 60),
                static_threshold=sp_cfg.get("static_threshold", 0.65),
                warmup_frames=sp_cfg.get("warmup_frames", 20),
            )
            self.get_logger().info(
                "static point filter enabled "
                f"(cell={sp_cfg.get('resolution', 0.5)}m, "
                f"thresh={sp_cfg.get('static_threshold', 0.65)})"
            )

        bg_cfg = self.cfg.get("background_filter", {})
        self.bg_filter = BackgroundFilter(
            resolution=bg_cfg.get("resolution", 0.5),
            history_frames=bg_cfg.get("history_frames", 50),
            static_threshold=bg_cfg.get("static_threshold", 0.8),
            warmup_frames=bg_cfg.get("warmup_frames", 20),
            exempt_classes=bg_cfg.get("exempt_classes"),
        )

        # Tracker
        trk_cfg = self.cfg.get("tracking", {})
        self.tracker = MultiObjectTracker(
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
        )

        # QoS — compatible with rosbag2 --clock
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # Subscriber
        lidar_topic = self.cfg["lidar_topic"]
        self.sub = self.create_subscription(
            PointCloud2, lidar_topic, self._lidar_callback, qos
        )
        self.get_logger().info(f"subscribed to {lidar_topic}")

        # Publishers
        self.pub_raw = self.create_publisher(PointCloud2, "/intersection/raw_cloud", 5)
        self.pub_roi_marker = self.create_publisher(
            Marker, "/intersection/roi/marker", 1
        )
        self.pub_det_markers = self.create_publisher(
            MarkerArray, "/intersection/detections/markers", 5
        )
        self.pub_track_markers = self.create_publisher(
            MarkerArray, "/intersection/tracking/markers", 5
        )

        # Publish ROI marker at 1 Hz
        self.roi_timer = self.create_timer(1.0, self._publish_roi_marker)

        self._frame_count = 0
        self._last_stamp = None

    def _load_config(self, path: str) -> dict:
        p = Path(path).resolve()
        if not p.exists():
            self.get_logger().error(f"config not found: {p}")
            raise FileNotFoundError(p)
        with open(p, "r") as f:
            return yaml.safe_load(f)

    def _publish_roi_marker(self):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.cfg["lidar_frame"]
        vis_cfg = self.cfg.get("visualization", {})
        color = vis_cfg.get("roi_marker_color", [0.0, 1.0, 1.0, 0.8])
        marker = make_roi_marker(self.roi_points, header, color)
        self.pub_roi_marker.publish(marker)

    def _promote_track_class(self, track: Track) -> None:
        """Re-label sparse far clusters that were stuck as unknown."""
        if track.class_name != "unknown":
            return
        det = track.detection
        length = float(track.smoothed_length or (det.length if det else 0))
        width = float(track.smoothed_width or (det.width if det else 0))
        height = float(track.smoothed_height or (det.height if det else 0))
        n_pts = int(det.num_points) if det else 0
        range_m = float(np.linalg.norm(track.centroid[:2]))
        class_cfg = self.cfg.get("classification", {})
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

    def _compute_dt(self, header: Header) -> float:
        stamp = header.stamp.sec + header.stamp.nanosec * 1e-9
        if self._last_stamp is None:
            dt = 0.1  # default for first frame
        else:
            dt = max(stamp - self._last_stamp, 0.01)
        self._last_stamp = stamp
        return dt

    def _lidar_callback(self, msg: PointCloud2):
        self._frame_count += 1
        header = msg.header
        vis_cfg = self.cfg.get("visualization", {})
        preproc = self.cfg.get("preprocess", {})
        cluster_cfg = self.cfg.get("clustering", {})
        class_cfg = self.cfg.get("classification", {})
        dt = self._compute_dt(header)

        # 1. PointCloud2 → numpy
        points = pointcloud2_to_xyz(msg)

        # Publish raw cloud
        if vis_cfg.get("publish_raw_cloud", True):
            self.pub_raw.publish(msg)

        # 2. ROI crop (XY only)
        roi_points, _ = crop_points_to_roi(points, self.roi_path)
        if len(roi_points) == 0:
            return

        # 3. Height filter
        z_min = preproc.get("z_min", -3.0)
        z_max = preproc.get("z_max", 4.0)
        roi_points = height_filter(roi_points, z_min, z_max)
        if len(roi_points) == 0:
            return

        # 4. Voxel downsample (optional near/far split keeps sparse far points)
        range_split = float(preproc.get("range_split_m", 0))
        if range_split > 0:
            downsampled = voxel_downsample_range_split(
                roi_points,
                range_split,
                float(preproc.get("voxel_near_m", preproc.get("voxel_size", 0.15))),
                float(preproc.get("voxel_far_m", preproc.get("voxel_size", 0.15))),
                bool(preproc.get("skip_far_downsample", False)),
            )
        else:
            downsampled = voxel_downsample(roi_points, float(preproc.get("voxel_size", 0.15)))

        # 5. Ground removal
        gr_cfg = preproc.get("ground_removal", {})
        nonground = remove_ground(
            downsampled,
            method=gr_cfg.get("method", "height_threshold"),
            ground_z_max=gr_cfg.get("ground_z_max", -1.4),
        )
        if len(nonground) == 0:
            return

        # 5b. Multi-frame static point removal (trees / fixed clutter)
        n_static_pts = 0
        if self.static_point_filter is not None:
            nonground, n_static_pts = self.static_point_filter.filter_points(nonground)
            if len(nonground) == 0:
                return

        # 6. DBSCAN clustering + classification
        detections = detect_objects(nonground, cluster_cfg, class_cfg)

        # 7. Background filter — remove static objects (trees, poles)
        dynamic_detections = self.bg_filter.filter_detections(detections)

        # 9. Tracking (detection markers published below, after speed filter)
        confirmed_tracks = self.tracker.step(dynamic_detections, dt=dt)

        # 9a. Track-level NMS: drop overlapping tracks (split clusters,
        #     re-id failures). Keeps the more mature track.
        nms_iou = float(self.cfg.get("tracking", {}).get("nms_iou", 0.0))
        if nms_iou > 0:
            confirmed_tracks = self.tracker.nms(confirmed_tracks, iou_threshold=nms_iou)

        for trk in confirmed_tracks:
            self._promote_track_class(trk)

        # 9b. Speed-based static filter: drop confirmed tracks that have been
        #     barely moving over the last few frames (true noise stays put).
        trk_cfg = self.cfg.get("tracking", {})
        min_speed = float(trk_cfg.get("min_track_speed", 0.0))  # m/s
        speed_window = int(trk_cfg.get("speed_window", 5))
        if min_speed > 0:
            moving_tracks = []
            for t in confirmed_tracks:
                # only enforce after the track has enough trajectory points
                if len(t.trajectory) <= speed_window:
                    moving_tracks.append(t)
                    continue
                if t.recent_speed(window=speed_window, dt=dt) >= min_speed:
                    moving_tracks.append(t)
            n_static_drop = len(confirmed_tracks) - len(moving_tracks)
            confirmed_tracks = moving_tracks
        else:
            n_static_drop = 0

        # Also suppress detection markers that lie under a dropped (static) cell:
        #   skip detections whose centroid is < 0.5 m from any current Kalman
        #   track that fails the speed test. This avoids flickering noise boxes.
        if min_speed > 0:
            drop_centers = []
            for t in self.tracker.tracks:
                if len(t.trajectory) > speed_window and \
                        t.recent_speed(window=speed_window, dt=dt) < min_speed:
                    drop_centers.append(t.centroid[:2])
            if drop_centers:
                drop_arr = np.array(drop_centers)
                kept = []
                for det in dynamic_detections:
                    dists = np.linalg.norm(drop_arr - det.centroid[:2], axis=1)
                    if dists.min() > 0.8:  # not near any static track
                        kept.append(det)
                dynamic_detections = kept

        # 8b. (Re-)publish detection markers using the speed-filtered set.
        #     Disable via visualization.publish_detection_markers: false to
        #     avoid visual overlap between per-frame detection boxes and the
        #     smoothed track boxes.
        if vis_cfg.get("publish_detection_markers", False):
            det_markers = detections_to_marker_array(dynamic_detections, header, vis_cfg)
            self.pub_det_markers.publish(det_markers)
        else:
            self.pub_det_markers.publish(make_delete_all_marker(header))

        # 10. Publish tracking markers
        track_markers = tracks_to_marker_array(confirmed_tracks, header, vis_cfg)
        self.pub_track_markers.publish(track_markers)

        if self._frame_count % 10 == 0:
            n_veh = sum(1 for t in confirmed_tracks if t.class_name == "vehicle")
            n_ped = sum(1 for t in confirmed_tracks if t.class_name == "pedestrian")
            n_unk = sum(1 for t in confirmed_tracks if t.class_name == "unknown")
            n_veh_det = sum(1 for d in dynamic_detections if d.class_name == "vehicle")
            self.get_logger().info(
                f"frame {self._frame_count}: "
                f"det={len(dynamic_detections)} (veh_det={n_veh_det}) "
                f"tracks={len(confirmed_tracks)} (veh={n_veh} ped={n_ped} unk={n_unk}) "
                f"static_pts={n_static_pts} "
                f"bg_filtered={len(detections) - len(dynamic_detections)} "
                f"speed_filtered={n_static_drop}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
