#!/usr/bin/env python3
"""TTC — Time To Collision for car-following (前后车) pairs only.

Same bumper-circle contact model as TTA, but only emitted when the pair passes
lead-follow geometry checks (aligned heading, lead ahead of rear, lateral offset
within lane width, longitudinal gap in range).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conflict_metrics import (
    assign_lead_follow,
    compute_time_to_contact,
    resolve_velocity,
)

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


def stats_summary(events: list[dict], danger_s: float) -> dict:
    if not events:
        return {"n_events": 0, "n_unique_pairs": 0, "n_pairs_danger": 0}
    vals = np.array([e["ttc"] for e in events])
    pair_min: dict[tuple[int, int], float] = {}
    for e in events:
        k = (min(e["rear_id"], e["lead_id"]), max(e["rear_id"], e["lead_id"]))
        pair_min[k] = min(pair_min.get(k, float("inf")), float(e["ttc"]))
    min_arr = np.array(list(pair_min.values()))
    return {
        "n_events": len(events),
        "n_events_negative": int((vals < 0).sum()),
        "n_unique_pairs": len(pair_min),
        "n_pairs_negative": int((min_arr < 0).sum()),
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
        default=REPO / "conflict_analysis" / "output" / "ttc_events.jsonl",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=REPO / "conflict_analysis" / "output" / "ttc_summary.json",
    )
    parser.add_argument("--pair-classes", nargs="*", default=["vehicle"])
    parser.add_argument("--max-pair-distance", type=float, default=50.0)
    parser.add_argument("--min-vehicle-speed", type=float, default=0.5)
    parser.add_argument("--min-closing-speed", type=float, default=0.3)
    parser.add_argument("--dcpa-margin", type=float, default=3.5)
    parser.add_argument("--ttc-cap", type=float, default=15.0)
    parser.add_argument("--danger-ttc", type=float, default=2.0)
    parser.add_argument("--velocity-window", type=float, default=0.7)
    parser.add_argument(
        "--signed-ttc",
        action="store_true",
        help="include signed TTC for pairs whose first contact may be in the past (ttc<0). "
        "When set, we drop the 'approaching now' constraint so negative TTC values appear.",
    )
    parser.add_argument("--max-heading-diff-deg", type=float, default=35.0)
    parser.add_argument("--max-lateral-m", type=float, default=4.0)
    parser.add_argument("--min-longitudinal-m", type=float, default=3.0)
    parser.add_argument("--max-longitudinal-m", type=float, default=80.0)
    args = parser.parse_args()

    frames = load_tracks_jsonl(args.tracks)
    trajectories = build_trajectories(frames)
    allowed = set(args.pair_classes)
    args.out_events.parent.mkdir(parents=True, exist_ok=True)

    events = []
    lf_checked = 0
    with open(args.out_events, "w") as ef:
        for rec in frames:
            ts, frame_idx = rec["stamp_sec"], rec["frame"]
            tracks = [t for t in rec["tracks"] if t["class"] in allowed]
            n = len(tracks)
            # Iterate all C(n,2) pairs first, then apply lead-follow gating.
            # So TTC does NOT keep all pairs: only pairs passing assign_lead_follow()
            # are considered "adjacent / car-following" and continue to TTC solve.
            for i in range(n):
                a = tracks[i]
                a_vel = resolve_velocity(
                    a, trajectories[a["track_id"]], ts, args.velocity_window
                )
                for j in range(i + 1, n):
                    b = tracks[j]
                    # Fast distance culling before geometry checks.
                    if (b["x"] - a["x"]) ** 2 + (b["y"] - a["y"]) ** 2 > args.max_pair_distance ** 2:
                        continue
                    b_vel = resolve_velocity(
                        b, trajectories[b["track_id"]], ts, args.velocity_window
                    )
                    lf = assign_lead_follow(
                        a, b, a_vel, b_vel,
                        max_heading_diff_deg=args.max_heading_diff_deg,
                        max_lateral_m=args.max_lateral_m,
                        min_longitudinal_m=args.min_longitudinal_m,
                        max_longitudinal_m=args.max_longitudinal_m,
                        min_speed=args.min_vehicle_speed,
                    )
                    if lf is None:
                        continue
                    # Count of pairs that satisfy car-following geometry constraints.
                    lf_checked += 1
                    rear, lead, v_rear, v_lead = lf
                    res = compute_time_to_contact(
                        rear, lead, v_rear, v_lead,
                        dcpa_margin=args.dcpa_margin,
                        min_closing_speed=args.min_closing_speed,
                        min_vehicle_speed=args.min_vehicle_speed,
                        time_cap=args.ttc_cap,
                        # default: require_closing=True -> mostly TTC>0 "approaching now"
                        # signed mode: require_closing=False -> keep TTC<0 normal cases too
                        require_closing=not args.signed_ttc,
                    )
                    if res is None:
                        continue
                    ttc = res["time_s"]
                    ev = {
                        "frame": frame_idx,
                        "stamp_sec": ts,
                        "rear_id": rear["track_id"],
                        "lead_id": lead["track_id"],
                        "rear_xy": [rear["x"], rear["y"]],
                        "lead_xy": [lead["x"], lead["y"]],
                        "ttc": ttc,
                        # Negative TTC means the first-touch solution lies in the past
                        # (often normal/non-risk at current frame), so do not mark danger.
                        "danger": (ttc >= 0.0) and (ttc <= args.danger_ttc),
                        **{k: float(v) for k, v in res.items() if k != "time_s"},
                    }
                    ef.write(json.dumps(ev) + "\n")
                    events.append(ev)

    summ = stats_summary(events, args.danger_ttc)
    summ["metric"] = "TTC"
    summ["definition"] = "car-following (rear/lead) pairs with predicted bumper contact"
    summ["n_lead_follow_checks"] = lf_checked
    summ["config"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    args.out_summary.write_text(json.dumps(summ, indent=2))

    print(f"TTC (lead-follow) events: {summ['n_events']}, unique pairs: {summ['n_unique_pairs']}")
    if summ["n_unique_pairs"]:
        print(
            f"  TTC < {args.danger_ttc}s: {summ['n_pairs_danger']}/{summ['n_unique_pairs']} pairs "
            f"({summ['frac_pairs_danger']*100:.1f}%)"
        )
        print(f"  TTC range: [{summ['range_s'][0]:.3f}, {summ['range_s'][1]:.3f}] s")
        print(
            f"  TTC < 0: {summ['n_events_negative']}/{summ['n_events']} events, "
            f"{summ['n_pairs_negative']}/{summ['n_unique_pairs']} pairs"
        )
        print(f"  mean={summ['mean_s']:.3f}s median={summ['median_s']:.3f}s")
    print(f"Wrote {args.out_events}")
    print(f"Wrote {args.out_summary}")


if __name__ == "__main__":
    main()
