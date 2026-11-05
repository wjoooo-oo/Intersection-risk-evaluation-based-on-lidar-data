#!/usr/bin/env python3
"""PET from trajectory conflict point + bbox front-bumper passage time.

Definition
----------
1. Conflict point P = polyline segment intersection (preferred), or closest
   approach between segments if gap <= max_conflict_gap and crossing angle
   is large enough (excludes parallel lane pairs).
2. Passage time = timestamp when the vehicle **front bumper** (centroid +
   half length along yaw) is closest to P.
3. PET = |t_A - t_B|, with temporal overlap filter on track lifetimes.

Outputs: pet_events.jsonl (one row per pair), pet_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

from track_postprocess import (  # noqa: E402
    prepare_trajectories_from_frames,
    stitch_fragmented_trajectories,
)


def load_tracks_jsonl(path: Path) -> list[dict]:
    frames: list[dict] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                frames.append(json.loads(line))
    frames.sort(key=lambda r: r["frame"])
    return frames


def build_trajectories(frames: list[dict], allowed_classes: set[str]) -> dict[int, dict]:
    """track_id -> pts (N,6): t, x, y, yaw, length, width."""
    by_id: dict[int, dict] = {}
    for rec in frames:
        ts = rec["stamp_sec"]
        for t in rec["tracks"]:
            if t["class"] not in allowed_classes:
                continue
            tid = t["track_id"]
            if tid not in by_id:
                by_id[tid] = {"class": t["class"], "pts": []}
            by_id[tid]["pts"].append(
                (ts, t["x"], t["y"], float(t["yaw"]), float(t["length"]), float(t["width"]))
            )
    for tid, data in by_id.items():
        data["pts"] = np.array(sorted(data["pts"]), dtype=float)
    return by_id


def downsample_polyline(pts: np.ndarray, step: float) -> np.ndarray:
    if len(pts) <= 2:
        return pts
    kept = [pts[0]]
    last = pts[0]
    for p in pts[1:]:
        if np.hypot(p[1] - last[1], p[2] - last[2]) >= step:
            kept.append(p)
            last = p
    if not np.array_equal(kept[-1], pts[-1]):
        kept.append(pts[-1])
    return np.array(kept)


def path_length_xy(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts[:, 1:3], axis=0), axis=1).sum())


def trajectory_co_overlap_fraction(
    traj_a: np.ndarray,
    traj_b: np.ndarray,
    close_m: float = 2.5,
) -> tuple[float, float, float]:
    """Fraction of each path within close_m of the other + median cross-track separation."""
    a_xy = traj_a[:, 1:3]
    b_xy = traj_b[:, 1:3]
    if len(a_xy) < 2 or len(b_xy) < 2:
        return 1.0, 1.0, 0.0
    dists_a = [float(np.min(np.linalg.norm(b_xy - p, axis=1))) for p in a_xy]
    dists_b = [float(np.min(np.linalg.norm(a_xy - p, axis=1))) for p in b_xy]
    frac_a = sum(d < close_m for d in dists_a) / len(dists_a)
    frac_b = sum(d < close_m for d in dists_b) / len(dists_b)
    return frac_a, frac_b, float(np.median(dists_a))


def approach_angle_at_conflict_deg(event: dict) -> float:
    """Angle between vectors from each passage bumper toward conflict point P."""
    cx, cy = float(event["x"]), float(event["y"])
    pa = np.array(event["passage_a_xy"], dtype=float)
    pb = np.array(event["passage_b_xy"], dtype=float)
    va = np.array([cx, cy]) - pa
    vb = np.array([cx, cy]) - pb
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na < 0.5 or nb < 0.5:
        return 0.0
    cosang = float(np.clip(np.dot(va / na, vb / nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def is_true_pet_conflict_point(
    traj_a: np.ndarray,
    traj_b: np.ndarray,
    event: dict,
    *,
    max_co_overlap_frac: float = 0.40,
    min_median_sep_m: float = 2.5,
    min_approach_angle_deg: float = 45.0,
) -> bool:
    """True intersection conflict: paths cross at P, not co-traveling in the same corridor.

    PET measures time gap between two vehicles passing the **same conflict point**
    from meaningfully different directions.  Pairs that mostly overlap (same lane /
    queue / follow-through) are excluded even if one track is labelled turning.
    """
    frac_a, frac_b, median_sep = trajectory_co_overlap_fraction(traj_a, traj_b)
    if max(frac_a, frac_b) > max_co_overlap_frac:
        return False
    if median_sep < min_median_sep_m:
        return False
    if approach_angle_at_conflict_deg(event) < min_approach_angle_deg:
        return False
    return True


def front_bumper_xy(row: np.ndarray) -> np.ndarray:
    """row: t,x,y,yaw,length,width -> front centre."""
    yaw = row[3]
    hl = 0.5 * row[4]
    return np.array([row[1] + hl * np.cos(yaw), row[2] + hl * np.sin(yaw)])


def segment_intersection(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> np.ndarray | None:
    """Return intersection point of segments in R^2, or None."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    def on_seg(pxx: float, pyy: float, ax: float, ay: float, bx: float, by: float) -> bool:
        return (
            min(ax, bx) - 1e-3 <= pxx <= max(ax, bx) + 1e-3
            and min(ay, by) - 1e-3 <= pyy <= max(ay, by) + 1e-3
        )
    if on_seg(px, py, x1, y1, x2, y2) and on_seg(px, py, x3, y3, x4, y4):
        return np.array([px, py])
    return None


def segment_closest_points(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    u = p2 - p1
    v = p4 - p3
    w = p1 - p3
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denom = a * c - b * b
    if denom < 1e-12:
        s = 0.0
        t = float(np.clip(np.dot(p1 - p3, v) / max(c, 1e-12), 0.0, 1.0))
    else:
        s = float(np.clip((b * e - c * d) / denom, 0.0, 1.0))
        t = float(np.clip((a * e - b * d) / denom, 0.0, 1.0))

    def dist_at(sv: float, tv: float) -> float:
        return float(np.linalg.norm((p1 + sv * u) - (p3 + tv * v)))

    best_s, best_t, best_d = s, t, dist_at(s, t)
    for sv in (0.0, 1.0):
        tv = float(np.clip(np.dot(p1 + sv * u - p3, v) / max(c, 1e-12), 0.0, 1.0))
        d = dist_at(sv, tv)
        if d < best_d:
            best_s, best_t, best_d = sv, tv, d
    for tv in (0.0, 1.0):
        sv = float(np.clip(np.dot(p3 + tv * v - p1, u) / max(a, 1e-12), 0.0, 1.0))
        d = dist_at(sv, tv)
        if d < best_d:
            best_s, best_t, best_d = sv, tv, d
    q1 = p1 + best_s * u
    q2 = p3 + best_t * v
    return best_s, best_t, best_d, q1, q2


def heading_at_point(traj: np.ndarray, xy: np.ndarray) -> float:
    idx = int(np.argmin(np.linalg.norm(traj[:, 1:3] - xy, axis=1)))
    return float(traj[idx, 3])


def turn_amount_near_conflict(
    traj: np.ndarray, conflict_xy: np.ndarray, window_m: float = 10.0
) -> float:
    """Net yaw change (rad) along path near P — not sum of wiggles (avoids false 'turning')."""
    xy = traj[:, 1:3]
    dists = np.linalg.norm(xy - conflict_xy, axis=1)
    idx0 = int(np.argmin(dists))
    yaws = traj[:, 3]
    indices = [idx0]
    acc = 0.0
    for i in range(idx0 - 1, -1, -1):
        acc += float(np.linalg.norm(xy[i + 1] - xy[i]))
        indices.insert(0, i)
        if acc >= window_m:
            break
    acc = 0.0
    for i in range(idx0, len(traj) - 1):
        acc += float(np.linalg.norm(xy[i + 1] - xy[i]))
        indices.append(i + 1)
        if acc >= window_m:
            break
    idxs = sorted(set(indices))
    if len(idxs) < 2:
        return 0.0
    net = yaws[idxs[-1]] - yaws[idxs[0]]
    net = (net + np.pi) % (2 * np.pi) - np.pi
    return float(abs(net))


def classify_maneuver(turn_rad: float, thresh_rad: float = 0.52) -> str:
    """~30 deg net change near conflict counts as turning."""
    return "turning" if turn_rad >= thresh_rad else "straight"


def is_straight_vs_turn_encounter(ev: dict, min_crossing_angle_deg: float) -> bool:
    """Keep only straight-vs-turn conflicts; drop parallel straight and turn-turn."""
    if ev["encounter_type"] not in ("straight_vs_turn", "turn_vs_straight"):
        return False
    if float(ev["crossing_angle_deg"]) < min_crossing_angle_deg:
        return False
    return True


def encounter_label(man_a: str, man_b: str) -> str:
    if man_a == "turning" and man_b == "straight":
        return "turn_vs_straight"
    if man_b == "turning" and man_a == "straight":
        return "straight_vs_turn"
    if man_a == "turning" and man_b == "turning":
        return "turn_vs_turn"
    return "straight_vs_straight"


def passage_time_bbox(traj: np.ndarray, conflict_xy: np.ndarray) -> tuple[float, float, float, int]:
    """t, front-bumper distance to P, sample index at minimum distance."""
    times = traj[:, 0]
    dists = np.array([np.linalg.norm(front_bumper_xy(traj[i]) - conflict_xy) for i in range(len(traj))])
    idx = int(np.argmin(dists))
    return float(times[idx]), float(dists[idx]), idx


def find_conflict_points(
    traj_a: np.ndarray,
    traj_b: np.ndarray,
    max_conflict_gap: float,
    min_crossing_angle_deg: float,
    point_dedupe_m: float,
) -> list[tuple[np.ndarray, float, str]]:
    """All distinct conflict points P along two polylines."""
    a_xy = traj_a[:, 1:3]
    b_xy = traj_b[:, 1:3]
    min_angle = np.radians(min_crossing_angle_deg)
    out: list[tuple[np.ndarray, float, str]] = []

    for i in range(len(a_xy) - 1):
        for j in range(len(b_xy) - 1):
            hit = segment_intersection(a_xy[i], a_xy[i + 1], b_xy[j], b_xy[j + 1])
            if hit is not None:
                ha = heading_at_point(traj_a, hit)
                hb = heading_at_point(traj_b, hit)
                ang = abs(np.arctan2(np.sin(ha - hb), np.cos(ha - hb)))
                if ang >= min_angle:
                    out.append((hit, 0.0, "intersection"))
                continue
            _, _, sep, q1, q2 = segment_closest_points(
                a_xy[i], a_xy[i + 1], b_xy[j], b_xy[j + 1]
            )
            if sep > max_conflict_gap:
                continue
            p = 0.5 * (q1 + q2)
            ha = heading_at_point(traj_a, p)
            hb = heading_at_point(traj_b, p)
            ang = abs(np.arctan2(np.sin(ha - hb), np.cos(ha - hb)))
            if ang >= min_angle:
                out.append((p, float(sep), "closest_cross"))

    merged: list[tuple[np.ndarray, float, str]] = []
    for p, sep, meth in out:
        if any(np.linalg.norm(p - m[0]) < point_dedupe_m for m in merged):
            continue
        merged.append((p, sep, meth))
    return merged


def _event_at_point(
    traj_a: np.ndarray,
    traj_b: np.ndarray,
    conflict_xy: np.ndarray,
    sep_dist: float,
    method: str,
    min_time_gap: float,
    pet_cap: float,
) -> dict | None:
    t_a, d_a, idx_a = passage_time_bbox(traj_a, conflict_xy)
    t_b, d_b, idx_b = passage_time_bbox(traj_b, conflict_xy)
    pet = abs(t_a - t_b)
    if pet < min_time_gap or pet > pet_cap:
        return None
    turn_a = turn_amount_near_conflict(traj_a, conflict_xy)
    turn_b = turn_amount_near_conflict(traj_b, conflict_xy)
    man_a = classify_maneuver(turn_a)
    man_b = classify_maneuver(turn_b)
    fb_a = front_bumper_xy(traj_a[idx_a])
    fb_b = front_bumper_xy(traj_b[idx_b])
    ha = float(traj_a[idx_a, 3])
    hb = float(traj_b[idx_b, 3])
    cross_angle = float(np.degrees(abs(np.arctan2(np.sin(ha - hb), np.cos(ha - hb)))))
    return {
        "x": float(conflict_xy[0]),
        "y": float(conflict_xy[1]),
        "t_first": float(min(t_a, t_b)),
        "t_second": float(max(t_a, t_b)),
        "t_a": t_a,
        "t_b": t_b,
        "first_is": "a" if t_a < t_b else "b",
        "pet": float(pet),
        "sep_dist": float(sep_dist),
        "d_a_at_passage": d_a,
        "d_b_at_passage": d_b,
        "method": method,
        "a_maneuver": man_a,
        "b_maneuver": man_b,
        "encounter_type": encounter_label(man_a, man_b),
        "crossing_angle_deg": cross_angle,
        "turn_a_rad": turn_a,
        "turn_b_rad": turn_b,
        "passage_a_xy": [float(fb_a[0]), float(fb_a[1])],
        "passage_b_xy": [float(fb_b[0]), float(fb_b[1])],
    }


def pet_events_at_conflict_points(
    traj_a: np.ndarray,
    traj_b: np.ndarray,
    max_conflict_gap: float,
    min_crossing_angle_deg: float,
    point_dedupe_m: float,
    min_time_gap: float,
    pet_cap: float,
    *,
    one_per_pair: bool = False,
    max_points_per_pair: int = 12,
) -> list[dict]:
    points = find_conflict_points(
        traj_a, traj_b, max_conflict_gap, min_crossing_angle_deg, point_dedupe_m
    )
    if not points:
        return []

    events: list[dict] = []
    for conflict_xy, sep_dist, method in points:
        ev = _event_at_point(
            traj_a, traj_b, conflict_xy, sep_dist, method, min_time_gap, pet_cap
        )
        if ev is not None:
            events.append(ev)
    if not events:
        return []
    if one_per_pair:
        return [min(events, key=lambda e: e["pet"])]
    events.sort(key=lambda e: e["pet"])
    return events[:max_points_per_pair]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path,
                        default=REPO / "conflict_analysis" / "output" / "tracks.jsonl")
    parser.add_argument("--out-events", type=Path,
                        default=REPO / "conflict_analysis" / "output" / "pet_events.jsonl")
    parser.add_argument("--out-summary", type=Path,
                        default=REPO / "conflict_analysis" / "output" / "pet_summary.json")
    parser.add_argument("--pair-classes", nargs="*", default=["vehicle"])
    parser.add_argument("--min-trajectory-length", type=float, default=3.0)
    parser.add_argument("--downsample-step", type=float, default=0.5)
    parser.add_argument("--max-conflict-gap", type=float, default=5.0,
                        help="m; max segment gap for virtual conflict point (not a disk zone)")
    parser.add_argument("--min-crossing-angle-deg", type=float, default=35.0,
                        help="min heading diff at conflict (filters parallel lanes)")
    parser.add_argument(
        "--straight-turn-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep only straight_vs_turn / turn_vs_straight encounters",
    )
    parser.add_argument("--point-dedupe-m", type=float, default=2.0,
                        help="merge conflict points closer than this")
    parser.add_argument("--max-points-per-pair", type=int, default=12,
                        help="cap conflict points emitted per track pair")
    parser.add_argument("--one-per-pair", action="store_true",
                        help="legacy: only min-PET point per pair")
    parser.add_argument("--pet-cap", type=float, default=10.0)
    parser.add_argument("--min-time-gap", type=float, default=0.15)
    parser.add_argument("--danger-pet", type=float, default=3.0)
    parser.add_argument("--min-time-overlap", type=float, default=1.5,
                        help="s; both tracks must overlap at least this long (lower helps short turn tracks)")
    parser.add_argument(
        "--stitch-fragments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="merge track IDs when one ends and another starts nearby (intersection re-id)",
    )
    parser.add_argument("--stitch-max-gap-s", type=float, default=3.0)
    parser.add_argument("--stitch-max-jump-m", type=float, default=8.0)
    parser.add_argument(
        "--complete-trajectories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stitch fragments and linearly fill time gaps (detection dropout)",
    )
    parser.add_argument("--max-interp-gap-s", type=float, default=3.0,
                        help="max time gap (s) to fill with interpolated points")
    parser.add_argument("--interp-dt", type=float, default=0.1,
                        help="time step (s) for gap fill")
    parser.add_argument("--max-speed-mps", type=float, default=18.0,
                        help="reject gap fill if implied speed exceeds this")
    parser.add_argument("--stitch-passes", type=int, default=2,
                        help="repeat fragment stitching passes")
    parser.add_argument(
        "--max-co-overlap-frac",
        type=float,
        default=0.40,
        help="drop pairs whose paths are within 2.5m for more than this fraction (co-travel)",
    )
    parser.add_argument(
        "--min-median-sep-m",
        type=float,
        default=2.5,
        help="median A-to-B path separation must exceed this (m)",
    )
    parser.add_argument(
        "--min-approach-angle-deg",
        type=float,
        default=45.0,
        help="min angle between approach vectors toward conflict point P",
    )
    parser.add_argument(
        "--include-cotravel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write co-traveling pairs too (tagged true_conflict_point=false); "
        "summary counts only true conflict points unless --count-all",
    )
    parser.add_argument(
        "--count-all",
        action="store_true",
        help="summary stats include all written events, not only true_conflict_point",
    )
    args = parser.parse_args()

    print(f"Loading {args.tracks} ...")
    frames = load_tracks_jsonl(args.tracks)
    allowed = set(args.pair_classes)
    n_trajectories_raw = 0
    if args.complete_trajectories:
        n_trajectories_raw = len(build_trajectories(frames, allowed))
        filt_traj, prep_stats = prepare_trajectories_from_frames(
            frames,
            allowed,
            stitch=args.stitch_fragments,
            complete_gaps=True,
            max_stitch_gap_s=args.stitch_max_gap_s,
            max_stitch_jump_m=args.stitch_max_jump_m,
            max_interp_gap_s=args.max_interp_gap_s,
            interp_dt=args.interp_dt,
            max_speed_mps=args.max_speed_mps,
            min_length_m=args.min_trajectory_length,
            stitch_passes=args.stitch_passes,
        )
        print(
            f"  prepared trips: {prep_stats['raw_track_ids']} raw IDs -> "
            f"{prep_stats['trips_after_prepare']} trips, "
            f"stitched {prep_stats['fragments_stitched']} fragments, "
            f"+{prep_stats['points_interpolated']} interpolated points"
        )
    else:
        trajectories = build_trajectories(frames, allowed)
        n_trajectories_raw = len(trajectories)
        print(f"  {len(trajectories)} trajectories")
        if args.stitch_fragments:
            trajectories, n_stitch = stitch_fragmented_trajectories(
                trajectories,
                max_gap_s=args.stitch_max_gap_s,
                max_jump_m=args.stitch_max_jump_m,
            )
            print(f"  stitched {n_stitch} fragments -> {len(trajectories)} trajectories")
        filt_traj = {}
        for tid, data in trajectories.items():
            pts = data["pts"]
            if len(pts) < 3:
                continue
            seg_len = path_length_xy(pts)
            if seg_len < args.min_trajectory_length:
                continue
            filt_traj[tid] = data

    for tid, data in filt_traj.items():
        pts = data["pts"]
        data["length_m"] = path_length_xy(pts)
        data["pts"] = downsample_polyline(pts, args.downsample_step)
        data["t_start"] = float(data["pts"][0, 0])
        data["t_end"] = float(data["pts"][-1, 0])
    print(f"  {len(filt_traj)} trajectories after filter/downsample")
    mode = "min PET per pair" if args.one_per_pair else f"all P (max {args.max_points_per_pair}/pair)"
    print(
        f"  PET: conflict point + bbox ({mode}), "
        f"gap<={args.max_conflict_gap}m, angle>={args.min_crossing_angle_deg}deg (closest_cross)"
    )

    args.out_events.parent.mkdir(parents=True, exist_ok=True)
    events = []
    n_skipped_cotravel = 0
    ids = list(filt_traj.keys())
    n_pairs = 0
    danger_count = 0
    with open(args.out_events, "w") as ef:
        for i in range(len(ids)):
            tid_a = ids[i]
            a = filt_traj[tid_a]
            for j in range(i + 1, len(ids)):
                tid_b = ids[j]
                b = filt_traj[tid_b]
                n_pairs += 1
                overlap = min(a["t_end"], b["t_end"]) - max(a["t_start"], b["t_start"])
                if overlap < args.min_time_overlap:
                    continue
                crossings = pet_events_at_conflict_points(
                    a["pts"], b["pts"],
                    args.max_conflict_gap,
                    args.min_crossing_angle_deg,
                    args.point_dedupe_m,
                    args.min_time_gap,
                    args.pet_cap,
                    one_per_pair=args.one_per_pair,
                    max_points_per_pair=args.max_points_per_pair,
                )
                for crossing in crossings:
                    if args.straight_turn_only and not is_straight_vs_turn_encounter(
                        crossing, args.min_crossing_angle_deg
                    ):
                        continue
                    frac_a, frac_b, med_sep = trajectory_co_overlap_fraction(
                        a["pts"], b["pts"]
                    )
                    crossing["co_overlap_frac_a"] = float(frac_a)
                    crossing["co_overlap_frac_b"] = float(frac_b)
                    crossing["median_sep_m"] = float(med_sep)
                    crossing["approach_angle_deg"] = approach_angle_at_conflict_deg(
                        crossing
                    )
                    is_true = is_true_pet_conflict_point(
                        a["pts"],
                        b["pts"],
                        crossing,
                        max_co_overlap_frac=args.max_co_overlap_frac,
                        min_median_sep_m=args.min_median_sep_m,
                        min_approach_angle_deg=args.min_approach_angle_deg,
                    )
                    crossing["true_conflict_point"] = is_true
                    if not args.include_cotravel and not is_true:
                        n_skipped_cotravel += 1
                        continue
                    crossing["a_id"] = tid_a
                    crossing["b_id"] = tid_b
                    crossing["a_class"] = a["class"]
                    crossing["b_class"] = b["class"]
                    crossing["danger"] = crossing["pet"] <= args.danger_pet
                    if crossing["danger"]:
                        danger_count += 1
                    ef.write(json.dumps(crossing) + "\n")
                    events.append(crossing)

    true_events = [e for e in events if e.get("true_conflict_point")]
    stats_events = events if args.count_all else true_events
    pets = np.array([e["pet"] for e in stats_events]) if stats_events else np.array([])
    top = sorted(true_events, key=lambda e: e["pet"])[:20]
    by_enc: dict[str, int] = {}
    for e in events:
        by_enc[e["encounter_type"]] = by_enc.get(e["encounter_type"], 0) + 1

    pair_keys = {
        (min(e["a_id"], e["b_id"]), max(e["a_id"], e["b_id"])) for e in stats_events
    }
    print(f"\nPET events written: {len(events)}  (true conflict points: {len(true_events)})")
    print(f"  Unique pairs in summary: {len(pair_keys)}")
    if n_skipped_cotravel:
        print(f"  Skipped co-travel (not written): {n_skipped_cotravel}")
    print(f"  Dangerous (PET <= {args.danger_pet}s): {danger_count}")
    if len(pets):
        print(f"  Range: [{pets.min():.3f}, {pets.max():.3f}] s")
        print(f"  PET < 2s: {(pets < 2).sum()}/{len(pets)} ({100*(pets < 2).mean():.1f}%)")
        print(f"  By encounter_type: {by_enc}")
        print(f"  By method: {dict(zip(*np.unique([e.get('method','') for e in events], return_counts=True)))}")

    summary = {
        "method": "conflict_point_bbox",
        "n_frames": len(frames),
        "n_trajectories": n_trajectories_raw,
        "n_trajectories_after_filter": len(filt_traj),
        "n_pairs_checked": n_pairs,
        "n_crossings": len(stats_events),
        "n_crossings_written": len(events),
        "n_true_conflict_points": len(true_events),
        "n_unique_pairs": len(pair_keys),
        "n_danger_crossings": danger_count,
        "n_skipped_cotravel": n_skipped_cotravel,
        "encounter_type_counts": by_enc,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "pet_stats": {
            "min": float(pets.min()) if len(pets) else None,
            "median": float(np.median(pets)) if len(pets) else None,
            "max": float(pets.max()) if len(pets) else None,
            "mean": float(pets.mean()) if len(pets) else None,
            "range_s": [float(pets.min()), float(pets.max())] if len(pets) else None,
            "n_below_2s": int((pets < 2.0).sum()) if len(pets) else 0,
            "frac_below_2s": float((pets < 2.0).mean()) if len(pets) else None,
            "n_danger": int((pets <= args.danger_pet).sum()) if len(pets) else 0,
            "frac_danger": float((pets <= args.danger_pet).mean()) if len(pets) else None,
        },
        "top_dangerous_crossings": top,
    }
    args.out_summary.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.out_events}")
    print(f"Wrote {args.out_summary}")


if __name__ == "__main__":
    main()
