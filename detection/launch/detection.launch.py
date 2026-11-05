"""Launch intersection LiDAR detection + optional RViz2.

Starts:
  - detection_node — subscribes to lidar_topic from config/detection_config.yaml,
    runs ROI → preprocess → DBSCAN detection → background filter → tracking,
    publishes /intersection/* markers and clouds.
  - rviz2 (optional) — detection_view.rviz, use_sim_time for bag playback.

Args:
  config_path   — YAML path (default: share/.../detection_config.yaml)
  use_sim_time  — true when playing rosbag with --clock
  launch_rviz   — start RViz alongside the node
  rviz_config   — RViz layout file
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("intersection_lidar_detection")
    default_config = os.path.join(pkg_dir, "config", "detection_config.yaml")
    default_rviz = os.path.join(pkg_dir, "rviz", "detection_view.rviz")

    return LaunchDescription([
        DeclareLaunchArgument("config_path", default_value=default_config),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("launch_rviz", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value=default_rviz),

        Node(
            package="intersection_lidar_detection",
            executable="detection_node",
            name="intersection_detection_node",
            output="screen",
            parameters=[{
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "config_path": LaunchConfiguration("config_path"),
            }],
        ),

        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="log",
            arguments=["-d", LaunchConfiguration("rviz_config")],
            parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
            condition=IfCondition(LaunchConfiguration("launch_rviz")),
        ),
    ])
