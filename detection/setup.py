from setuptools import setup, find_packages
import os
from glob import glob

package_name = "intersection_lidar_detection"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools", "numpy", "scikit-learn", "PyYAML"],
    zip_safe=True,
    maintainer="contributors",
    maintainer_email="user@todo.todo",
    description="Clustering-based vehicle/pedestrian detection for roadside LiDAR",
    license="MIT",
    entry_points={
        "console_scripts": [
            "detection_node = intersection_lidar_detection.detection_node:main",
            "inspect_bag = intersection_lidar_detection.inspect_bag:main",
        ],
    },
)
