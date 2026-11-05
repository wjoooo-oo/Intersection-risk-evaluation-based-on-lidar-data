"""Stitch track fragments, interpolate detection gaps, optional ROI checks.

Completion standards (defaults tuned for ~10 Hz tracks):
- Stitch: next fragment starts within ``max_stitch_gap_s`` and ``max_stitch_jump_m``,
  heading change below ``max_stitch_heading_rad``.
- Interpolate: fill time gaps up to ``max_interp_gap_s`` when implied speed
  <= ``max_speed_mps``; sample every ``interp_dt`` s (linear XY, unwrap yaw).
"""

from __future__ import annotations

import numpy as np


def path_length_xy(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts[:, 1:3], axis=0), axis=1).sum())


def _unwrap_yaw_delta(y0: float, y1: float) -> float:
    d = y1 - y0
    return (d + np.pi) % (2 * np.pi) - np.pi


def interpolate_gaps_in_trajectory(
    pts: np.ndarray,
    *,
    max_interp_gap_s: float = 3.0,
    interp_dt: float = 0.1,
    max_speed_mps: float = 18.0,
) -> tuple[np.ndarray, int]:
    """Insert rows into time gaps. Returns (completed_pts, n_inserted)."""
    if len(pts) < 2:
        return pts, 0
    pts = np.asarray(pts, dtype=float)
    out: list[np.ndarray] = [pts[0]]
    n_inserted = 0
    for i in range(1, len(pts)):
        prev, cur = pts[i - 1], pts[i]
        dt = float(cur[0] - prev[0])
        if dt <= interp_dt * 1.05:
            out.append(cur)
            continue
        if dt > max_interp_gap_s:
            out.append(cur)
            continue
        dist = float(np.linalg.norm(cur[1:3] - prev[1:3]))
        if dist / max(dt, 1e-6) > max_speed_mps:
            out.append(cur)
            continue
        n_seg = max(1, int(round(dt / interp_dt)) - 1)
        dyaw = _unwrap_yaw_delta(float(prev[3]), float(cur[3]))
        for k in range(1, n_seg + 1):
            alpha = k / (n_seg + 1)
            row = prev.copy()
            row[0] = prev[0] + alpha * dt
            row[1:3] = prev[1:3] + alpha * (cur[1:3] - prev[1:3])
            row[3] = prev[3] + alpha * dyaw
            if row.shape[0] >= 6:
                row[4:6] = prev[4:6] + alpha * (cur[4:6] - prev[4:6])
            out.append(row)
            n_inserted += 1
        out.append(cur)
    return np.array(out), n_inserted


def stitch_fragmented_trajectories(
    trajectories: dict[int, dict],
    *,
    max_gap_s: float = 3.0,
    max_jump_m: float = 8.0,
    max_heading_diff_rad: float = 1.6,
) -> tuple[dict[int, dict], int]:
    """Merge track fragments when one ends and another starts nearby."""
    chains: list[dict] = []
    for tid, data in trajectories.items():
        pts = np.asarray(data["pts"], dtype=float)
        if len(pts) < 2:
            continue
        chains.append(
            {
                "id": tid,
                "class": data.get("class", "vehicle"),
                "pts": pts,
                "t_start": float(pts[0, 0]),
                "t_end": float(pts[-1, 0]),
                "end_xy": pts[-1, 1:3].copy(),
                "start_xy": pts[0, 1:3].copy(),
                "end_yaw": float(pts[-1, 3]),
                "start_yaw": float(pts[0, 3]),
                "merged_into": None,
            }
        )
    chains.sort(key=lambda c: c["t_start"])
    n_merged = 0
    for head in chains:
        if head["merged_into"] is not None:
            continue
        current = head
        while True:
            best = None
            best_gap = float("inf")
            for cand in chains:
                if cand["merged_into"] is not None or cand is current:
                    continue
                gap = cand["t_start"] - current["t_end"]
                if gap < -0.05 or gap > max_gap_s:
                    continue
                dist = float(np.linalg.norm(cand["start_xy"] - current["end_xy"]))
                if dist > max_jump_m:
                    continue
                dyaw = _unwrap_yaw_delta(current["end_yaw"], cand["start_yaw"])
                if abs(dyaw) > max_heading_diff_rad:
                    continue
                if gap < best_gap:
                    best_gap = gap
                    best = cand
            if best is None:
                break
            merged_pts = np.vstack([current["pts"], best["pts"]])
            order = np.argsort(merged_pts[:, 0])
            merged_pts = merged_pts[order]
            keep = [0]
            for j in range(1, len(merged_pts)):
                if merged_pts[j, 0] - merged_pts[keep[-1], 0] > 1e-4:
                    keep.append(j)
            current["pts"] = merged_pts[np.array(keep)]
            current["t_end"] = float(current["pts"][-1, 0])
            current["end_xy"] = current["pts"][-1, 1:3].copy()
            current["end_yaw"] = float(current["pts"][-1, 3])
            best["merged_into"] = head
            n_merged += 1
    out: dict[int, dict] = {}
    for head in chains:
        if head["merged_into"] is not None:
            continue
        out[int(head["id"])] = {"class": head["class"], "pts": head["pts"]}
    return out, n_merged


def prepare_trajectories_from_frames(
    frames: list[dict],
    allowed_classes: set[str],
    *,
    stitch: bool = True,
    complete_gaps: bool = True,
    max_stitch_gap_s: float = 3.0,
    max_stitch_jump_m: float = 8.0,
    max_stitch_heading_rad: float = 1.6,
    max_interp_gap_s: float = 3.0,
    interp_dt: float = 0.1,
    max_speed_mps: float = 18.0,
    min_length_m: float = 2.0,
    stitch_passes: int = 2,
) -> tuple[dict[int, dict], dict[str, int | float]]:
    """Build trajectories with optional stitch + gap interpolation."""
    from evaluate_pet import build_trajectories

    raw = build_trajectories(frames, allowed_classes)
    traj = raw
    n_stitch = 0
    if stitch:
        for _ in range(max(1, stitch_passes)):
            traj, n = stitch_fragmented_trajectories(
                traj,
                max_gap_s=max_stitch_gap_s,
                max_jump_m=max_stitch_jump_m,
                max_heading_diff_rad=max_stitch_heading_rad,
            )
            n_stitch += n

    n_inserted = 0
    out: dict[int, dict] = {}
    trip_id = 1
    for _tid, data in sorted(traj.items(), key=lambda kv: float(kv[1]["pts"][0, 0])):
        pts = np.asarray(data["pts"], dtype=float)
        n_add = 0
        if complete_gaps:
            pts, n_add = interpolate_gaps_in_trajectory(
                pts,
                max_interp_gap_s=max_interp_gap_s,
                interp_dt=interp_dt,
                max_speed_mps=max_speed_mps,
            )
            n_inserted += n_add
        if path_length_xy(pts) < min_length_m:
            continue
        out[trip_id] = {
            "class": data.get("class", "vehicle"),
            "pts": pts,
            "n_interp": n_add,
            "source_ids": [int(_tid)],
        }
        trip_id += 1

    stats: dict[str, int | float] = {
        "raw_track_ids": len(raw),
        "trips_after_prepare": len(out),
        "fragments_stitched": n_stitch,
        "points_interpolated": n_inserted,
    }
    return out, stats


def dist_point_to_polygon_boundary(xy: np.ndarray, vertices: np.ndarray) -> float:
    p = np.asarray(xy, dtype=float).reshape(2)
    verts = np.asarray(vertices, dtype=float)
    n = len(verts)
    best = float("inf")
    for i in range(n):
        a = verts[i]
        b = verts[(i + 1) % n]
        ab = b - a
        t = float(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12))
        t = max(0.0, min(1.0, t))
        q = a + t * ab
        best = min(best, float(np.linalg.norm(p - q)))
    return best


def track_boundary_status(
    pts: np.ndarray,
    roi_vertices: np.ndarray,
    *,
    margin_m: float = 4.0,
) -> tuple[bool, bool, float, float]:
    if len(pts) < 2:
        return False, False, float("inf"), float("inf")
    d0 = dist_point_to_polygon_boundary(pts[0, 1:3], roi_vertices)
    d1 = dist_point_to_polygon_boundary(pts[-1, 1:3], roi_vertices)
    return d0 <= margin_m, d1 <= margin_m, d0, d1


def is_through_track(
    pts: np.ndarray,
    roi_vertices: np.ndarray,
    *,
    margin_m: float = 4.0,
    min_length_m: float = 6.0,
) -> bool:
    if path_length_xy(pts) < min_length_m:
        return False
    start_ok, end_ok, _, _ = track_boundary_status(pts, roi_vertices, margin_m=margin_m)
    return start_ok and end_ok


def build_completed_lines_from_frames(
    frames: list[dict],
    roi_vertices: np.ndarray | None = None,
    *,
    allowed_classes: set[str] | None = None,
    min_length_m: float = 4.0,
    **prepare_kw,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, int | float]]:
    """Return (primary_lines, low_quality_lines, stats) after stitch+interpolate."""
    allowed = allowed_classes or {"vehicle"}
    trips, stats = prepare_trajectories_from_frames(
        frames, allowed, min_length_m=min_length_m, **prepare_kw
    )
    primary: list[np.ndarray] = []
    secondary: list[np.ndarray] = []
    for data in trips.values():
        pts = data["pts"]
        if len(pts) < 2:
            continue
        line = pts[:, 1:3]
        if roi_vertices is not None and not is_through_track(
            pts, roi_vertices, min_length_m=min_length_m
        ):
            secondary.append(line)
        else:
            primary.append(line)
    stats = dict(stats)
    stats["lines_primary"] = len(primary)
    stats["lines_secondary"] = len(secondary)
    return primary, secondary, stats
