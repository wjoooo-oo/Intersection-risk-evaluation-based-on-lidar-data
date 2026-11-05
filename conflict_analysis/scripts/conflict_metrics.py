"""Shared geometry for TTA (any collision risk) and TTC (car-following only)."""

from __future__ import annotations

import numpy as np


def velocity_from_track(track: dict) -> np.ndarray | None:
    if "vx_kf" in track and "vy_kf" in track:
        v = np.array([float(track["vx_kf"]), float(track["vy_kf"])])
        if float(np.linalg.norm(v)) >= 0.05:
            return v
    return None


def velocity_from_traj(
    traj: list[tuple[float, float, float]], ts: float, window_sec: float = 0.5
) -> np.ndarray:
    if len(traj) < 2:
        return np.zeros(2)
    times = np.array([r[0] for r in traj])
    idx = int(np.argmin(np.abs(times - ts)))
    lo = hi = idx
    while lo > 0 and (times[idx] - times[lo - 1]) <= window_sec:
        lo -= 1
    while hi < len(traj) - 1 and (times[hi + 1] - times[idx]) <= window_sec:
        hi += 1
    if hi == lo:
        return np.zeros(2)
    dt = times[hi] - times[lo]
    if dt < 1e-6:
        return np.zeros(2)
    return np.array(
        [(traj[hi][1] - traj[lo][1]) / dt, (traj[hi][2] - traj[lo][2]) / dt]
    )


def resolve_velocity(
    track: dict,
    traj: list[tuple[float, float, float]],
    ts: float,
    window_sec: float = 0.5,
) -> np.ndarray:
    v = velocity_from_track(track)
    if v is not None:
        return v
    return velocity_from_traj(traj, ts, window_sec)


def compute_time_to_contact(
    a: dict,
    b: dict,
    a_vel: np.ndarray,
    b_vel: np.ndarray,
    *,
    dcpa_margin: float,
    min_closing_speed: float,
    min_vehicle_speed: float,
    time_cap: float,
    require_closing: bool = True,
) -> dict | None:
    """Time until bumper circles first touch (constant velocity). Returns metrics dict."""
    a_pos = np.array([a["x"], a["y"]])
    b_pos = np.array([b["x"], b["y"]])
    rel_pos = b_pos - a_pos
    rel_vel = b_vel - a_vel
    dist = float(np.linalg.norm(rel_pos))
    if dist < 1e-6:
        return None

    if float(np.linalg.norm(a_vel)) < min_vehicle_speed:
        return None
    if float(np.linalg.norm(b_vel)) < min_vehicle_speed:
        return None

    rel_speed_sq = float(np.dot(rel_vel, rel_vel))
    if rel_speed_sq < 1e-6:
        return None

    ndir = rel_pos / dist
    closing_speed = float(np.dot(a_vel - b_vel, ndir))
    tcpa = float(-np.dot(rel_pos, rel_vel) / rel_speed_sq)
    dcpa = float(np.linalg.norm(rel_pos + rel_vel * tcpa))

    if require_closing:
        # Current behavior: only compute TTC for pairs approaching each other now.
        if closing_speed < min_closing_speed:
            return None
        if tcpa <= 0.0:
            return None

    a_radius = 0.5 * float(np.hypot(a["length"], a["width"]))
    b_radius = 0.5 * float(np.hypot(b["length"], b["width"]))
    safe_radius = a_radius + b_radius + dcpa_margin
    if dcpa > safe_radius:
        return None

    bumper_radius = a_radius + b_radius
    A_q = rel_speed_sq
    B_q = 2.0 * float(np.dot(rel_pos, rel_vel))
    C_q = float(np.dot(rel_pos, rel_pos)) - bumper_radius ** 2
    disc = B_q ** 2 - 4.0 * A_q * C_q
    if disc < 0:
        return None
    t_touch = (-B_q - np.sqrt(disc)) / (2.0 * A_q)
    t_contact = float(t_touch)
    if t_contact > time_cap:
        return None

    gap = max(0.0, dist - bumper_radius)
    return {
        "time_s": t_contact,
        "tcpa": tcpa,
        "dcpa": dcpa,
        "gap": gap,
        "dist": dist,
        "closing_speed": closing_speed,
        "bumper_radius": bumper_radius,
    }


def assign_lead_follow(
    a: dict,
    b: dict,
    a_vel: np.ndarray,
    b_vel: np.ndarray,
    *,
    max_heading_diff_deg: float = 35.0,
    max_lateral_m: float = 4.0,
    min_longitudinal_m: float = 3.0,
    max_longitudinal_m: float = 80.0,
    min_speed: float = 0.5,
) -> tuple[dict, dict, np.ndarray, np.ndarray] | None:
    """If car-following, return (rear, lead, v_rear, v_lead). Else None.

    NOTE (for TTC statistics):
    This function is the key gate that makes TTC focus on "adjacent/leader-follower"
    pairs instead of all vehicle pairs.

    A pair is kept only when all these hold:
    1) Both vehicles are moving.
    2) Their headings are roughly aligned (same general direction).
    3) One vehicle is in front of the other along the travel direction
       (longitudinal distance in [min_longitudinal_m, max_longitudinal_m]).
    4) Lateral offset is small (within max_lateral_m), i.e., near same lane/corridor.

    If none of the two directional checks passes, return None and this pair is not
    counted by TTC.
    """
    na = float(np.linalg.norm(a_vel))
    nb = float(np.linalg.norm(b_vel))
    if na < min_speed or nb < min_speed:
        return None
    cos_align = float(np.dot(a_vel, b_vel) / (na * nb))
    if cos_align < float(np.cos(np.radians(max_heading_diff_deg))):
        return None

    rel = np.array([b["x"] - a["x"], b["y"] - a["y"]])
    ua = a_vel / na
    ub = b_vel / nb
    long_a = float(np.dot(rel, ua))
    lat_a = float(abs(rel[0] * ua[1] - rel[1] * ua[0]))
    long_b = float(np.dot(-rel, ub))
    lat_b = float(abs(rel[0] * ub[1] - rel[1] * ub[0]))

    if (
        min_longitudinal_m <= long_a <= max_longitudinal_m
        and lat_a <= max_lateral_m
    ):
        return a, b, a_vel, b_vel
    if (
        min_longitudinal_m <= long_b <= max_longitudinal_m
        and lat_b <= max_lateral_m
    ):
        return b, a, b_vel, a_vel
    return None
