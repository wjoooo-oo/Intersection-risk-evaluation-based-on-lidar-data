#!/usr/bin/env python3
"""Copy polygon_xy from config/roi_*.yaml into detection_config_*.yaml roi.points."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roi-yaml",
        type=Path,
        default=REPO / "config" / "roi_session_c.yaml",
    )
    parser.add_argument(
        "--detection-config",
        type=Path,
        default=REPO / "detection/config/detection_config_session_c.yaml",
    )
    args = parser.parse_args()

    roi = yaml.safe_load(args.roi_yaml.read_text())
    poly = roi.get("polygon_xy") or roi.get("polygon")
    if not poly:
        raise SystemExit(f"No polygon_xy in {args.roi_yaml}")

    det = yaml.safe_load(args.detection_config.read_text())
    det.setdefault("roi", {})
    det["roi"]["type"] = "polygon"
    det["roi"]["frame"] = det["roi"].get("frame", "lidar")
    det["roi"]["points"] = [[float(p[0]), float(p[1])] for p in poly]
    if "erode_margin_m" not in det["roi"]:
        det["roi"]["erode_margin_m"] = 0.0

    args.detection_config.write_text(yaml.dump(det, default_flow_style=False, sort_keys=False))
    print(f"Synced {len(poly)} vertices:")
    print(f"  {args.roi_yaml}")
    print(f"  -> {args.detection_config}")


if __name__ == "__main__":
    main()
