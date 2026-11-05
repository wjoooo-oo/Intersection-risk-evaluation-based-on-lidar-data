#!/usr/bin/env python3
"""TTA — Time To Accident / contact for ANY converging vehicle pair (not 前后车).

Uses the same bumper-circle contact model as TTC, but without car-following
geometry filters.  Suitable for intersection crossing / merging conflicts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conflict_metrics import compute_time_to_contact, resolve_velocity

REPO = Path(__file__).resolve().parents[2]


def load_tracks_jsonl(path: Path) -> list[dict]:
    frames: list[dict] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                frames.append(json.loads(line))
    frames.sort(key=lambda r: r["frame"])
    return frames


def build_trajectories(frames: list[dict]) -> dict[int, list]:
    traj: dict[int, list] = defaultdict(list)
    for rec in frames:
        ts = rec["stamp_sec"]
        for t in rec["tracks"]:
            traj[t["track_id"]].append((ts, t["x"], t["y"]))
    for tid in traj:
        traj[tid].sort()
    return traj


def stats_summary(events: list[dict], key: str, danger_s: float) -> dict:
    if not events:
        return {"n_events": 0, "n_unique_pairs": 0, "n_pairs_danger": 0}
    vals = np.array([e[key] for e in events])
    pair_min: dict[tuple[int, int], float] = {}
    for e in events:
        k = (min(e["a_id"], e["b_id"]), max(e["a_id"], e["b_id"]))
        pair_min[k] = min(pair_min.get(k, float("inf")), float(e[key]))
    min_arr = np.array(list(pair_min.values()))
    return {
        "n_events": len(events),
        "n_unique_pairs": len(pair_min),
        "n_pairs_danger": int((min_arr < danger_s).sum()),
        "frac_pairs_danger": float((min_arr < danger_s).mean()),
        "frac_events_danger": float((vals < danger_s).mean()),
        "range_s": [float(vals.min()), float(vals.max())],
        "mean_s": float(vals.mean()),
        "median_s": float(np.median(vals)),
        "pair_min_range_s": [float(min_arr.min()), float(min_arr.max())],
        "pair_min_median_s": float(np.median(min_arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracks",
        type=Path,
        default=REPO / "conflict_analysis" / "output" / "tracks.jsonl",
    )
    parser.add_argument(
        "--out-events",
        type=Path,
        default=REPO / "conflict_analysis" / "output" / "tta_events.jsonl",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=REPO / "conflict_analysis" / "output" / "tta_summary.json",
    )
    parser.add_argument("--pair-classes", nargs="*", default=["vehicle"])
    parser.add_argument("--max-pair-distance", type=float, default=50.0)
    parser.add_argument("--min-vehicle-speed", type=float, default=0.5)
    parser.add_argument("--min-closing-speed", type=float, default=0.3)
    parser.add_argument("--dcpa-margin", type=float, default=3.5)
    parser.add_argument("--tta-cap", type=float, default=15.0)
    parser.add_argument("--danger-tta", type=float, default=2.0)
    parser.add_argument("--velocity-window", type=float, default=0.7)
    args = parser.parse_args()

    frames = load_tracks_jsonl(args.tracks)
    trajectories = build_trajectories(frames)
    allowed = set(args.pair_classes)
    args.out_events.parent.mkdir(parents=True, exist_ok=True)

    events = []
    with open(args.out_events, "w") as ef:
        for rec in frames:
            ts, frame_idx = rec["stamp_sec"], rec["frame"]
            tracks = [t for t in rec["tracks"] if t["class"] in allowed]
            n = len(tracks)
            for i in range(n):
                a = tracks[i]
                a_vel = resolve_velocity(
                    a, trajectories[a["track_id"]], ts, args.velocity_window
                )
                for j in range(i + 1, n):
                    b = tracks[j]
                    if (b["x"] - a["x"]) ** 2 + (b["y"] - a["y"]) ** 2 > args.max_pair_distance ** 2:
                        continue
                    b_vel = resolve_velocity(
                        b, trajectories[b["track_id"]], ts, args.velocity_window
                    )
                    res = compute_time_to_contact(
                        a, b, a_vel, b_vel,
                        dcpa_margin=args.dcpa_margin,
                        min_closing_speed=args.min_closing_speed,
                        min_vehicle_speed=args.min_vehicle_speed,
                        time_cap=args.tta_cap,
                    )
                    if res is None:
                        continue
                    tta = res["time_s"]
                    ev = {
                        "frame": frame_idx,
                        "stamp_sec": ts,
                        "a_id": a["track_id"],
                        "b_id": b["track_id"],
                        "tta": tta,
                        "danger": tta <= args.danger_tta,
                        **{k: float(v) for k, v in res.items() if k != "time_s"},
                    }
                    ef.write(json.dumps(ev) + "\n")
                    events.append(ev)

    summ = stats_summary(events, "tta", args.danger_tta)
    summ["metric"] = "TTA"
    summ["definition"] = "any vehicle pair with predicted bumper contact"
    summ["config"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    args.out_summary.write_text(json.dumps(summ, indent=2))

    print(f"TTA events: {summ['n_events']}, unique pairs: {summ['n_unique_pairs']}")
    print(
        f"  TTA < {args.danger_tta}s: {summ['n_pairs_danger']}/{summ['n_unique_pairs']} pairs "
        f"({summ['frac_pairs_danger']*100:.1f}%), "
        f"{summ['frac_events_danger']*100:.1f}% of frame-records"
    )
    print(f"  TTA range (events): [{summ['range_s'][0]:.3f}, {summ['range_s'][1]:.3f}] s")
    print(f"  mean={summ['mean_s']:.3f}s median={summ['median_s']:.3f}s")
    print(f"Wrote {args.out_events}")
    print(f"Wrote {args.out_summary}")


if __name__ == "__main__":
    main()
