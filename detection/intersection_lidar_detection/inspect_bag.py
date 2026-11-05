"""Inspect a ROS2 bag: print topics, types, message counts, and estimated frequency."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Inspect a ROS2 bag (metadata.yaml)")
    parser.add_argument("bag_path", type=str, help="Path to rosbag2 directory")
    args = parser.parse_args()

    bag_dir = Path(args.bag_path)
    meta = bag_dir / "metadata.yaml"
    if not meta.exists():
        print(f"ERROR: {meta} not found", file=sys.stderr)
        sys.exit(1)

    with open(meta) as f:
        info = yaml.safe_load(f)

    bag_info = info.get("rosbag2_bagfile_information", info)
    duration_ns = bag_info.get("duration", {}).get("nanoseconds", 0)
    duration_s = duration_ns / 1e9
    total_msgs = bag_info.get("message_count", 0)

    print(f"Bag: {bag_dir}")
    print(f"Duration: {duration_s:.2f} s ({duration_s/60:.1f} min)")
    print(f"Total messages: {total_msgs}")
    print(f"{'─'*80}")
    print(f"{'Topic':<40} {'Type':<40} {'Count':>8} {'Hz':>8}")
    print(f"{'─'*80}")

    topics = bag_info.get("topics_with_message_count", [])
    for entry in topics:
        tm = entry.get("topic_metadata", {})
        name = tm.get("name", "?")
        msg_type = tm.get("type", "?")
        count = entry.get("message_count", 0)
        hz = count / duration_s if duration_s > 0 else 0.0
        print(f"{name:<40} {msg_type:<40} {count:>8} {hz:>8.1f}")

    print(f"{'─'*80}")


if __name__ == "__main__":
    main()
