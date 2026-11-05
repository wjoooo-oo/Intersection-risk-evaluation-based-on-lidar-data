#!/usr/bin/env python3
"""Report track quality issues that hurt PET (short turns, fragments, low speed gaps)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def load_frames(path: Path) -> list[dict]:
    frames = []
    with open(path) as f:
        for line in f:
            if line.strip():
                frames.append(json.loads(line))
    return sorted(frames, key=lambda r: r["frame"])


def build_by_id(frames: list[dict], cls: str) -> dict[int, list]:
    by_id: dict[int, list] = {}
    for rec in frames:
        ts = rec["stamp_sec"]
        for t in rec["tracks"]:
            if t["class"] != cls:
                continue
            by_id.setdefault(t["track_id"], []).append(
                (ts, t["x"], t["y"], float(t["yaw"]), float(t.get("speed", 0)))
            )
    for tid in by_id:
        by_id[tid] = sorted(by_id[tid], key=lambda r: r[0])
    return by_id


def path_len(pts: list) -> float:
    if len(pts) < 2:
        return 0.0
    arr = np.array([(p[1], p[2]) for p in pts])
    return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--min-len-m", type=float, default=4.0)
    parser.add_argument("--min-dur-s", type=float, default=3.0)
    args = parser.parse_args()

    frames = load_frames(args.tracks)
    by_id = build_by_id(frames, "vehicle")
    lens, durs, turns = [], [], []
    for pts in by_id.values():
        lens.append(path_len(pts))
        durs.append(pts[-1][0] - pts[0][0])
        y = np.array([p[3] for p in pts])
        net = (y[-1] - y[0] + np.pi) % (2 * np.pi) - np.pi
        turns.append(abs(net))

    lens = np.array(lens)
    durs = np.array(durs)
    turns = np.array(turns)

    print(f"Bag frames: {len(frames)}  vehicle track IDs: {len(by_id)}")
    print(f"  path length >= {args.min_len_m}m: {(lens >= args.min_len_m).sum()}")
    print(f"  duration >= {args.min_dur_s}s: {(durs >= args.min_dur_s).sum()}")
    print(f"  |yaw change| >= 30°: {(turns >= 0.52).sum()}")
    short_turn = sum(
        1 for i in range(len(by_id)) if turns[i] >= 0.8 and lens[i] < args.min_len_m
    )
    turn_short_dur = sum(
        1 for i in range(len(by_id)) if turns[i] >= 0.52 and durs[i] < args.min_dur_s
    )
    print(f"  turning but path < {args.min_len_m}m (often lost in intersection): {short_turn}")
    print(f"  turning but duration < {args.min_dur_s}s (PET overlap filter drops): {turn_short_dur}")

    frag = 0
    starts, ends = [], []
    for tid, pts in by_id.items():
        starts.append((pts[0][0], pts[0][1], pts[0][2], tid))
        ends.append((pts[-1][0], pts[-1][1], pts[-1][2], tid))
    for t1, x1, y1, id1 in ends:
        for t0, x0, y0, id0 in starts:
            if id0 == id1:
                continue
            if 0 < t0 - t1 < 2.5 and np.hypot(x0 - x1, y0 - y1) < 6:
                frag += 1
                break
    print(f"  likely re-id fragments (next track starts <6m, <2.5s): {frag}")
    print("\nIf short_turn / turn_short_dur are large, re-run record_tracks with")
    print("  min_track_speed: 0  and  max_age: 25  (see detection_config_session_c.yaml)")


if __name__ == "__main__":
    main()
