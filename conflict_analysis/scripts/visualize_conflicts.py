#!/usr/bin/env python3
"""Visualize TTC and PET results computed from a tracks JSONL.

Four artefact groups
--------------------
1) `summary/*.png`         - global histograms / cdf / counts of TTC and PET.
2) `overview/conflicts_bev.png`
                          - one BEV chart of the full session: every
                            trajectory drawn faintly, dangerous TTC events
                            as red lines, PET crossings as yellow markers.
3) `ttc_frames/frame_*.png`
                          - one BEV per frame that has at least one dangerous
                            TTC event (capped by --max-bev-frames).
4) `pet_events/event_*.png`
                          - one chart per top-K dangerous PET crossing,
                            showing the two full trajectories colored by
                            time, the crossing point, and the time gap.

Designed to be the "show me what's happening" companion to evaluate_ttc.py
and evaluate_pet.py.  Adapted from
lidar_object_detection_and_tracking/scripts/visualize_ttc_evaluation.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "conflict_analysis" / "scripts"))
from evaluate_pet import (  # noqa: E402
    build_trajectories as build_pet_trajectories,
    downsample_polyline,
    is_true_pet_conflict_point,
)
from track_postprocess import build_completed_lines_from_frames  # noqa: E402
CLASS_COLORS = {
    "vehicle": "#1F77B4",
    "pedestrian": "#D62728",
    "unknown": "#9E9E9E",
}

# PET histogram / CDF display range (matches pet_cap and LiDAR quantization floor)
PET_XMIN = 0.2
PET_XMAX = 10.0

# Default BEV extent (overview / TTC frames / PET events)
BEV_XLIM = (-60.0, 60.0)
BEV_YLIM = (-40.0, 40.0)
BEV_TITLE_FONTSIZE = 16
BEV_ANNOTATION_FONTSIZE = 14
BEV_LEGEND_FONTSIZE = 13
BEV_AXIS_LABEL_FONTSIZE = 13
BEV_TICK_FONTSIZE = 12


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def ttc_pair_key(e: dict) -> tuple[int, int]:
    """rear/lead (car-following) or legacy ego/lead."""
    rear = e["rear_id"] if "rear_id" in e else e["ego_id"]
    lead = e["lead_id"]
    return (min(rear, lead), max(rear, lead))


def ttc_rear_xy(e: dict) -> list[float]:
    return e["rear_xy"] if "rear_xy" in e else e["ego_xy"]


def pet_pair_key(e: dict) -> tuple[int, int]:
    return (min(e["a_id"], e["b_id"]), max(e["a_id"], e["b_id"]))


def pet_direction_crossing_ok(
    event: dict,
    *,
    min_crossing_angle_deg: float,
    min_approach_angle_deg: float,
    max_co_overlap_frac: float,
) -> bool:
    """Keep PET events where trajectories meet at clearly different directions."""
    cross = float(event.get("crossing_angle_deg", 0.0))
    approach = float(event.get("approach_angle_deg", 0.0))
    ov = max(
        float(event.get("co_overlap_frac_a", 1.0)),
        float(event.get("co_overlap_frac_b", 1.0)),
    )
    return (
        cross >= min_crossing_angle_deg
        and approach >= min_approach_angle_deg
        and ov <= max_co_overlap_frac
    )


def _filter_pet_for_viz(pet_events: list[dict], args: argparse.Namespace) -> list[dict]:
    """Select PET BEVs: diverging directions at P, optional straight/turn-only."""
    straight_turn = {"straight_vs_turn", "turn_vs_straight"}
    kept: list[dict] = []
    for e in pet_events:
        if args.pet_viz_straight_turn_only and e.get("encounter_type") not in straight_turn:
            continue
        if not args.pet_conflict_point_only and not pet_direction_crossing_ok(
            e,
            min_crossing_angle_deg=args.pet_viz_min_crossing_angle,
            min_approach_angle_deg=args.pet_viz_min_approach_angle,
            max_co_overlap_frac=args.pet_viz_max_co_overlap,
        ):
            continue
        kept.append(e)
    if args.pet_conflict_point_only:
        return [e for e in kept if e.get("true_conflict_point")]
    return kept


def pet_conflict_priority(event: dict) -> float:
    """Higher = diverging-direction crossing at P (not smaller PET)."""
    score = float(event.get("crossing_angle_deg", 0.0)) * 1.2
    score += float(event.get("approach_angle_deg", 0.0)) * 1.5
    score += float(event.get("median_sep_m", 0.0)) * 2.0
    ov = max(
        float(event.get("co_overlap_frac_a", 1.0)),
        float(event.get("co_overlap_frac_b", 1.0)),
    )
    score -= ov * 100.0
    if event.get("method") == "intersection":
        score += 20.0
    if event.get("true_conflict_point"):
        score += 30.0
    return score


# ----------------- summary plots -----------------

def get_bbox_corners(x: float, y: float, length: float, width: float, yaw: float) -> np.ndarray:
    half_l, half_w = length / 2.0, width / 2.0
    corners = np.array(
        [[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]]
    )
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return corners @ R.T + np.array([x, y])


def plot_ttc_histogram(
    ax,
    values: np.ndarray,
    *,
    color: str,
    alpha: float = 0.85,
    subzero_color: str = "#C62828",
    subzero_bar_width: float = 0.35,
    subzero_bar_center: float = -0.55,
    n_pos_bins: int = 50,
    pos_upper_cap: float = 16.0,
) -> int:
    """Draw TTC histogram for values >= 0 plus one narrow aggregated bar for TTC<0."""
    n_neg = int((values < 0).sum())
    nonneg = values[values >= 0]
    upper = (
        min(float(nonneg.max()) + 0.5, pos_upper_cap)
        if len(nonneg)
        else pos_upper_cap
    )
    pos_bins = np.linspace(0, upper, n_pos_bins)
    if len(nonneg):
        ax.hist(nonneg, bins=pos_bins, color=color, alpha=alpha, edgecolor="black")
    if n_neg:
        ax.bar(
            subzero_bar_center,
            n_neg,
            width=subzero_bar_width,
            color=subzero_color,
            alpha=alpha,
            edgecolor="black",
            align="center",
            zorder=5,
        )
        x_right = pos_bins[-1] if len(nonneg) else upper
        ax.set_xlim(subzero_bar_center - subzero_bar_width * 0.75, x_right)
        xticks = [t for t in ax.get_xticks() if t >= -1e-9]
        xlabels = [f"{t:g}" for t in xticks]
        xticks = [subzero_bar_center] + xticks
        xlabels = ["<0"] + xlabels
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
    return n_neg


def plot_ttc_distribution(
    ttc_events: list[dict],
    nearmiss_ttc: float,
    out_dir: Path,
) -> None:
    """TTC count distributions (per-frame and per unique pair).

    This plot is intended for quick sanity-checking. It shows:
    - all per-frame TTC rows (pair × time)
    - per-pair minimum TTC (one value per rear/lead pair)

    Bars for 0 < TTC < nearmiss_ttc are highlighted in red. TTC < 0 is kept
    (common "normal" cases where closest approach is in the past).
    """
    if not ttc_events:
        return
    ttcs = np.array([e["ttc"] for e in ttc_events])
    n_neg = int((ttcs < 0).sum())
    n_danger = int(((ttcs > 0) & (ttcs < nearmiss_ttc)).sum())
    pct_neg_ev = 100.0 * n_neg / len(ttcs)
    pct_danger_ev = 100.0 * n_danger / len(ttcs)
    mean_ev = float(ttcs.mean()) if len(ttcs) else float("nan")
    range_ev = (float(ttcs.min()), float(ttcs.max())) if len(ttcs) else (float("nan"), float("nan"))

    pair_min: dict[tuple[int, int], float] = {}
    for e in ttc_events:
        key = ttc_pair_key(e)
        t = float(e["ttc"])
        pair_min[key] = min(pair_min.get(key, float("inf")), t)
    min_per_pair = np.array(list(pair_min.values()))
    n_neg_pairs = int((min_per_pair < 0).sum())
    n_danger_pairs = int(((min_per_pair > 0) & (min_per_pair < nearmiss_ttc)).sum())
    pct_neg_pair = 100.0 * n_neg_pairs / len(min_per_pair)
    pct_danger_pair = 100.0 * n_danger_pairs / len(min_per_pair)
    mean_pair = float(min_per_pair.mean()) if len(min_per_pair) else float("nan")
    range_pair = (
        (float(min_per_pair.min()), float(min_per_pair.max())) if len(min_per_pair) else (float("nan"), float("nan"))
    )

    def _nice_limits(arr: np.ndarray) -> tuple[float, float]:
        if len(arr) == 0:
            return -5.0, 15.0
        lo = float(np.floor(arr.min() * 2.0) / 2.0)
        hi = float(np.ceil(arr.max() * 2.0) / 2.0)
        # keep a stable view comparable across runs
        lo = max(lo, -15.0)
        hi = min(hi, 20.0)
        if hi - lo < 6.0:
            c = 0.5 * (lo + hi)
            lo, hi = c - 3.0, c + 3.0
        return lo, hi

    x0, x1 = _nice_limits(ttcs)
    bins = np.linspace(x0, x1, 61)  # 60 bins

    # Styling: keep this figure presentation-only (no title/info boxes).
    # Requested: enlarge font size ~2×.
    label_fs = 22
    tick_fs = 18

    def plot_count_hist(ax, vals: np.ndarray, *, xlabel: str, ylabel: str) -> None:
        # np.histogram drops values outside bin edges; for TTC we want the
        # visible histogram to account for *all* samples within the chosen
        # x-range (extreme negatives/positives are collapsed into edge bins).
        vals_plot = np.clip(vals, x0, x1 - 1e-9)
        counts, edges = np.histogram(vals_plot, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)
        base = "#90A4AE"  # blue-grey
        hi_color = "#C62828"
        ax.bar(centers, counts, width=widths, color=base, edgecolor="black", alpha=0.85)
        mask = (centers > 0) & (centers < nearmiss_ttc)
        if mask.any():
            ax.bar(
                centers[mask],
                counts[mask],
                width=widths[mask],
                color=hi_color,
                edgecolor="black",
                alpha=0.9,
                zorder=5,
            )
        ax.set_xlim(x0, x1)
        ax.set_xlabel(xlabel, fontsize=label_fs)
        ax.set_ylabel(ylabel, fontsize=label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.grid(alpha=0.25)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    ax = axes[0]
    plot_count_hist(
        ax,
        ttcs,
        xlabel="TTC (s)",
        ylabel="Number of observations",
    )
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor="#C62828", edgecolor="black", alpha=0.9)
        ],
        labels=[
            f"0<TTC<{nearmiss_ttc:g}s: {n_danger}/{len(ttcs)} ({pct_danger_ev:.2f}%)"
        ],
        loc="upper right",
        frameon=False,
        fontsize=14,
    )

    ax = axes[1]
    plot_count_hist(
        ax,
        min_per_pair,
        xlabel="min TTC per rear/lead pair (s)",
        ylabel="Number of observations",
    )
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor="#C62828", edgecolor="black", alpha=0.9)
        ],
        labels=[
            f"0<min TTC<{nearmiss_ttc:g}s: {n_danger_pairs}/{len(min_per_pair)} ({pct_danger_pair:.2f}%)"
        ],
        loc="upper right",
        frameon=False,
        fontsize=14,
    )

    fig.tight_layout()
    out = out_dir / "ttc_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out}")
    # Console summary (requested format)
    print(
        f"all pair (frame×pair): "
        f"ttc<0: {n_neg}/{len(ttcs)} ({pct_neg_ev:.2f}%), "
        f"0<ttc<{nearmiss_ttc:g}s: {n_danger}/{len(ttcs)} ({pct_danger_ev:.2f}%)."
    )
    print(f"ttc range:[{range_ev[0]:.3f}, {range_ev[1]:.3f}], mean={mean_ev:.3f} s")
    print(
        f"unique pair (min per pair): "
        f"ttc<0: {n_neg_pairs}/{len(min_per_pair)} ({pct_neg_pair:.2f}%), "
        f"0<ttc<{nearmiss_ttc:g}s: {n_danger_pairs}/{len(min_per_pair)} ({pct_danger_pair:.2f}%)."
    )
    print(f"ttc range:[{range_pair[0]:.3f}, {range_pair[1]:.3f}], mean={mean_pair:.3f} s")


def plot_ttc_summary(
    ttc_events: list[dict], danger_ttc: float, out_dir: Path, *, title_suffix: str = ""
) -> None:
    ttcs = np.array([e["ttc"] for e in ttc_events])
    gaps = np.array([e["gap"] for e in ttc_events])
    closing = np.array([e["closing_speed"] for e in ttc_events])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    if len(ttcs):
        n_neg = plot_ttc_histogram(ax, ttcs, color="#1F77B4", alpha=0.8)
        ax.axvline(danger_ttc, color="red", linestyle="--",
                   label=f"threshold = {danger_ttc}s")
        if n_neg:
            ax.plot([], [], color="#C62828", linewidth=8, label=f"TTC<0: {n_neg}")
        ax.set_title(f"TTC distribution (n={len(ttcs)}){title_suffix}")
        ax.set_xlabel("TTC (s)")
        ax.set_ylabel("count")
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.set_title("No TTC events")

    ax = axes[0, 1]
    if len(ttcs):
        sorted_ttcs = np.sort(ttcs)
        cdf = np.arange(1, len(sorted_ttcs) + 1) / len(sorted_ttcs)
        ax.plot(sorted_ttcs, cdf, color="#1F77B4")
        ax.axvline(danger_ttc, color="red", linestyle="--")
        ax.set_title("TTC CDF")
        ax.set_xlabel("TTC (s)")
        ax.set_ylabel("fraction <= x")
        ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if len(gaps):
        ax.scatter(gaps, ttcs, c=closing, cmap="viridis",
                   s=10, alpha=0.6, edgecolors="none")
        ax.set_xlabel("gap distance (m)")
        ax.set_ylabel("TTC (s)")
        ax.set_title("gap vs TTC (color = closing speed m/s)")
        ax.grid(alpha=0.3)
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label("closing speed (m/s)")

    ax = axes[1, 1]
    if len(closing):
        ax.hist(closing, bins=40, color="#FF7F0E", alpha=0.8, edgecolor="black")
        ax.set_title("closing-speed distribution")
        ax.set_xlabel("closing speed (m/s)")
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)

    n_pairs = len({ttc_pair_key(e) for e in ttc_events})
    fig.suptitle(
        f"TTC summary{title_suffix}\n"
        f"{len(ttc_events)} frame-records, {n_pairs} unique rear/lead pairs "
        f"(car-following only)",
        fontsize=12,
    )
    fig.tight_layout()
    out = out_dir / "ttc_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out}")


def plot_pet_distribution(
    pet_events: list[dict],
    danger_pet: float,
    out_dir: Path,
) -> None:
    """PET distribution — one record per unique track pair (min PET over full bag)."""
    if not pet_events:
        return
    pets = np.array([e["pet"] for e in pet_events])
    seps = np.array([e.get("sep_dist", float("nan")) for e in pet_events])
    n_pairs = len({pet_pair_key(e) for e in pet_events})
    nm_danger = pets[pets <= danger_pet]
    pct_danger = 100.0 * len(nm_danger) / len(pets)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    bins = np.linspace(PET_XMIN, PET_XMAX, 40)
    ax.hist(pets, bins=bins, range=(PET_XMIN, PET_XMAX), color="#FFA000", alpha=0.85, edgecolor="black")
    ax.set_xlim(PET_XMIN, PET_XMAX)
    ax.axvline(danger_pet, color="red", linestyle="--", linewidth=2,
               label=f"PET<={danger_pet}s: {len(nm_danger)}/{len(pets)} ({pct_danger:.1f}%)")
    ax.set_xlabel("PET (s)")
    ax.set_ylabel("unique track pairs")
    ax.set_title(f"Per unique track pair (n={n_pairs})\nmin PET over whole bag")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    sorted_pets = np.sort(pets)
    cdf = np.arange(1, len(sorted_pets) + 1) / len(sorted_pets)
    ax.plot(sorted_pets, cdf, color="#FFA000", linewidth=2)
    ax.axvline(danger_pet, color="red", linestyle="--",
               label=f"P(PET<={danger_pet}s)={pct_danger/100:.3f}")
    ax.set_xlim(PET_XMIN, PET_XMAX)
    ax.set_xlabel("PET (s)")
    ax.set_ylabel("fraction <= x")
    ax.set_title("CDF (one encounter per pair)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    valid = np.isfinite(seps)
    if valid.any():
        ax.scatter(seps[valid], pets[valid], c=pets[valid], cmap="autumn_r",
                   s=12, alpha=0.5, edgecolors="none")
        ax.axhline(danger_pet, color="red", linestyle="--", alpha=0.7)
        ax.set_xlabel("closest sample separation at conflict (m)")
        ax.set_ylabel("PET (s)")
        ax.set_title("sep_dist vs PET (color = PET)")
        ax.grid(alpha=0.3)
        plt.colorbar(ax.collections[0], ax=ax, label="PET (s)")

    ax = axes[1, 1]
    ax.axis("off")
    n_danger = int((pets <= danger_pet).sum())
    txt = (
        "Counting note\n"
        "─────────────\n"
        f"• PET = conflict point (path cross) + bbox front-bumper passage.\n"
        f"  pairs need >=5s time overlap; crossing angle >=25 deg.\n"
        f"• {len(pets)} events at distinct conflict points P.\n"
        f"  (multiple P per track pair allowed.)\n\n"
        f"PET (n={len(pets)} pairs)\n"
        f"  range [{pets.min():.2f}, {pets.max():.2f}] s\n"
        f"  mean {pets.mean():.2f} s, median {np.median(pets):.2f} s\n"
        f"  PET <= {danger_pet}s (near-miss): {n_danger}/{len(pets)} "
        f"({100*n_danger/len(pets):.1f}%)\n\n"
        f"sep_dist: mean {np.nanmean(seps):.2f} m, "
        f"median {np.nanmedian(seps):.2f} m"
    )
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=10,
            va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round", facecolor="#FFF8E1", edgecolor="#999"))

    fig.suptitle(
        "PET distribution — full bag (conflict point + bbox, one encounter per pair)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out = out_dir / "pet_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def plot_pet_summary(
    pet_events: list[dict],
    danger_pet: float,
    out_dir: Path,
    *,
    title_suffix: str = "",
) -> None:
    pets = np.array([e["pet"] for e in pet_events])
    seps = np.array([e.get("sep_dist", float("nan")) for e in pet_events])
    n_pairs = len({pet_pair_key(e) for e in pet_events})

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    if len(pets):
        ax.hist(pets, bins=np.linspace(PET_XMIN, PET_XMAX, 35),
                range=(PET_XMIN, PET_XMAX), color="#FFA000", alpha=0.85, edgecolor="black")
        ax.set_xlim(PET_XMIN, PET_XMAX)
        ax.axvline(danger_pet, color="red", linestyle="--",
                   label=f"threshold = {danger_pet}s")
        ax.set_title(f"PET distribution (n={len(pets)}){title_suffix}")
        ax.set_xlabel("PET (s)")
        ax.set_ylabel("count")
        ax.legend()
        ax.grid(alpha=0.3)

    ax = axes[0, 1]
    if len(pets):
        sorted_pets = np.sort(pets)
        cdf = np.arange(1, len(sorted_pets) + 1) / len(sorted_pets)
        ax.plot(sorted_pets, cdf, color="#FFA000", linewidth=2)
        ax.axvline(danger_pet, color="red", linestyle="--")
        ax.set_xlim(PET_XMIN, PET_XMAX)
        ax.set_title("PET CDF")
        ax.set_xlabel("PET (s)")
        ax.set_ylabel("fraction <= x")
        ax.grid(alpha=0.3)

    ax = axes[1, 0]
    valid = np.isfinite(seps) & len(pets) > 0
    if valid.any():
        ax.scatter(seps[valid], pets[valid], c=pets[valid], cmap="autumn_r",
                   s=14, alpha=0.6, edgecolors="none")
        ax.set_xlabel("sep_dist at conflict (m)")
        ax.set_ylabel("PET (s)")
        ax.set_title("sep_dist vs PET")
        ax.grid(alpha=0.3)
        plt.colorbar(ax.collections[0], ax=ax, label="PET (s)")

    ax = axes[1, 1]
    if len(pets):
        ax.hist(seps[np.isfinite(seps)], bins=30, color="#FF7F0E", alpha=0.85, edgecolor="black")
        ax.set_title("sep_dist distribution")
        ax.set_xlabel("separation (m)")
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"PET summary{title_suffix}\n"
        f"{len(pet_events)} encounters, {n_pairs} unique track pairs "
        f"(proximity conflict zone)",
        fontsize=12,
    )
    fig.tight_layout()
    out = out_dir / "pet_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ----------------- BEV plots -----------------

def setup_bev_axes(
    ax,
    roi_cfg,
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    transparent: bool = False,
) -> None:
    polygon = np.array(roi_cfg["polygon_xy"])
    ax.add_patch(
        MplPolygon(
            polygon,
            closed=True,
            fill=False,
            edgecolor="black",
            linewidth=1.4 if transparent else 1.2,
            linestyle="-",
            alpha=1.0 if transparent else 0.85,
        )
    )
    x0, x1 = xlim if xlim is not None else BEV_XLIM
    y0, y1 = ylim if ylim is not None else BEV_YLIM
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", fontsize=BEV_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Y (m)", fontsize=BEV_AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=BEV_TICK_FONTSIZE)
    if transparent:
        ax.grid(False)
    else:
        ax.grid(alpha=0.2)


def _savefig_kwargs(transparent: bool) -> dict:
    kw = dict(dpi=150, bbox_inches="tight")
    if transparent:
        kw.update(transparent=True, facecolor="none", edgecolor="none")
    return kw


def plot_overview_bev(
    frames: list[dict],
    pet_events: list[dict],
    roi_cfg: dict,
    out_path: Path,
    *,
    top_k_pet: int = 5,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    stitch_tracks: bool = True,
    include_filled_trips: bool = True,
    raw_tracking_only: bool = False,
    show_title: bool = True,
) -> None:
    """Near-miss overview BEV: trajectories + top-K PET centres (orange)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    setup_bev_axes(ax, roi_cfg, xlim=xlim, ylim=ylim)

    roi_vertices = np.array(roi_cfg["polygon_xy"], dtype=float)
    traj_color = CLASS_COLORS["vehicle"]
    traj_note = ""

    if raw_tracking_only:
        traj: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for rec in frames:
            for t in rec["tracks"]:
                if t["class"] != "vehicle":
                    continue
                traj[t["track_id"]].append((t["x"], t["y"]))
        n_vehicles = 0
        for pts in traj.values():
            if len(pts) < 3:
                continue
            n_vehicles += 1
            arr = np.array(pts)
            ax.plot(arr[:, 0], arr[:, 1], color=traj_color,
                    linewidth=0.85, alpha=0.5, zorder=4)
        traj_note = f"{n_vehicles} vehicle tracks (detection output)"
    elif stitch_tracks:
        primary, secondary, stats = build_completed_lines_from_frames(
            frames,
            roi_vertices,
            min_length_m=4.0,
        )
        if include_filled_trips:
            for line in secondary:
                ax.plot(line[:, 0], line[:, 1], color=traj_color,
                        linewidth=0.85, alpha=0.5, zorder=4)
        for line in primary:
            ax.plot(line[:, 0], line[:, 1], color=traj_color,
                    linewidth=0.85, alpha=0.5, zorder=4)
        if not include_filled_trips:
            traj_note = f"{len(primary)} stitched through-trips (ROI in/out)"
        else:
            traj_note = (
                f"{stats['trips_after_prepare']} completed trips "
                f"(+{stats['points_interpolated']} interp pts), "
                f"{stats['lines_primary']} ROI in/out"
            )
    else:
        traj = defaultdict(list)
        n_vehicles = 0
        for rec in frames:
            for t in rec["tracks"]:
                if t["class"] != "vehicle":
                    continue
                traj[t["track_id"]].append((t["x"], t["y"]))
        for pts in traj.values():
            if len(pts) < 3:
                continue
            n_vehicles += 1
            arr = np.array(pts)
            ax.plot(arr[:, 0], arr[:, 1], color=traj_color,
                    linewidth=0.6, alpha=0.35)
        traj_note = f"{n_vehicles} raw track IDs"

    # Overview stars: skip co-travel-like types (yellow/purple), show top crossings in orange.
    _overview_skip_enc = {"straight_vs_straight", "turn_vs_straight"}
    pet_ranked = sorted(pet_events, key=pet_conflict_priority, reverse=True)
    pet_top = [e for e in pet_ranked if e.get("encounter_type") not in _overview_skip_enc][
        :top_k_pet
    ]
    star_orange = "#FF9800"
    if pet_top:
        ax.scatter(
            [e["x"] for e in pet_top],
            [e["y"] for e in pet_top],
            marker="*",
            s=220,
            c=star_orange,
            edgecolors="black",
            linewidths=0.8,
            zorder=6,
        )

    if show_title:
        ax.set_title(
            f"Conflict overview — {traj_note}, "
            f"{len(pet_top)} PET centres (of {len(pet_events)} encounters)",
            fontsize=BEV_TITLE_FONTSIZE,
        )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def bev_limits_from_roi(roi_cfg: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    """Read optional `bev:` block from roi yaml; fall back to module defaults."""
    bev = roi_cfg.get("bev") or {}
    xlim = (float(bev.get("x_min", BEV_XLIM[0])), float(bev.get("x_max", BEV_XLIM[1])))
    ylim = (float(bev.get("y_min", BEV_YLIM[0])), float(bev.get("y_max", BEV_YLIM[1])))
    return xlim, ylim


def compute_ttc_frame_limits(
    rec: dict,
    ttc_events_in_frame: list[dict],
    *,
    margin_m: float = 12.0,
    min_span_m: float = 18.0,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Tight crop around vehicles and TTC endpoints so per-frame BEVs stay readable."""
    xs: list[float] = []
    ys: list[float] = []
    for t in rec["tracks"]:
        corners = get_bbox_corners(t["x"], t["y"], t["length"], t["width"], t["yaw"])
        xs.extend(corners[:, 0].tolist())
        ys.extend(corners[:, 1].tolist())
    for e in ttc_events_in_frame:
        rxy, lxy = ttc_rear_xy(e), e["lead_xy"]
        xs.extend([rxy[0], lxy[0]])
        ys.extend([rxy[1], lxy[1]])
    if not xs:
        return None

    x0, x1 = min(xs) - margin_m, max(xs) + margin_m
    y0, y1 = min(ys) - margin_m, max(ys) + margin_m
    if x1 - x0 < min_span_m:
        cx = 0.5 * (x0 + x1)
        x0, x1 = cx - 0.5 * min_span_m, cx + 0.5 * min_span_m
    if y1 - y0 < min_span_m:
        cy = 0.5 * (y0 + y1)
        y0, y1 = cy - 0.5 * min_span_m, cy + 0.5 * min_span_m
    return (x0, x1), (y0, y1)


def _ttc_event_label(e: dict) -> str:
    rear = e["rear_id"] if "rear_id" in e else e["ego_id"]
    lead = e["lead_id"]
    parts = [f"#{rear}→#{lead}", f"TTC {e['ttc']:.2f} s"]
    if "dcpa" in e:
        parts.append(f"DCPA {e['dcpa']:.1f} m")
    if "gap" in e:
        parts.append(f"gap {e['gap']:.1f} m")
    return "  ".join(parts)


def _draw_velocity_arrow(
    ax,
    track: dict,
    *,
    color: str,
    scale: float = 1.5,
) -> None:
    vx = track.get("vx_kf")
    vy = track.get("vy_kf")
    if vx is None or vy is None:
        return
    speed = float(np.hypot(vx, vy))
    if speed < 0.3:
        return
    x, y = track["x"], track["y"]
    ax.annotate(
        "",
        xy=(x + vx * scale, y + vy * scale),
        xytext=(x, y),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0, mutation_scale=14),
        zorder=7,
    )


def plot_ttc_frame_bev(
    rec: dict,
    ttc_events_in_frame: list[dict],
    roi_cfg: dict,
    danger_ttc: float,
    ttc_cap: float,
    out_path: Path,
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    auto_zoom: bool = True,
    zoom_margin_m: float = 12.0,
) -> None:
    if auto_zoom:
        local = compute_ttc_frame_limits(
            rec, ttc_events_in_frame, margin_m=zoom_margin_m,
        )
        if local is not None:
            xlim, ylim = local

    track_by_id = {t["track_id"]: t for t in rec["tracks"]}
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")
    setup_bev_axes(ax, roi_cfg, xlim=xlim, ylim=ylim)

    for t in rec["tracks"]:
        corners = get_bbox_corners(t["x"], t["y"], t["length"], t["width"], t["yaw"])
        color = CLASS_COLORS.get(t["class"], "#888")
        ax.add_patch(MplPolygon(corners, closed=True, fill=True,
                                facecolor=color, edgecolor="black",
                                linewidth=0.6, alpha=0.5))

    cmap = plt.get_cmap("autumn_r")
    for e in ttc_events_in_frame:
        ttc = e["ttc"]
        frac = max(0.0, min(1.0, 1.0 - ttc / max(ttc_cap, 1e-6)))
        color = cmap(frac)
        lw = 2.5 if ttc <= danger_ttc else 1.2
        rxy, lxy = ttc_rear_xy(e), e["lead_xy"]
        ax.plot([rxy[0], lxy[0]], [rxy[1], lxy[1]],
                color=color, linewidth=lw, alpha=0.85, zorder=5)

        rear_id = e["rear_id"] if "rear_id" in e else e["ego_id"]
        lead_id = e["lead_id"]
        rear = track_by_id.get(rear_id)
        lead = track_by_id.get(lead_id)
        if rear is not None:
            _draw_velocity_arrow(ax, rear, color="#1565C0")
        if lead is not None:
            _draw_velocity_arrow(ax, lead, color="#C62828")

    if ttc_events_in_frame:
        lines = [
            _ttc_event_label(e)
            for e in sorted(ttc_events_in_frame, key=lambda x: x["ttc"])
        ]
        ax.text(
            0.02,
            0.98,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=BEV_ANNOTATION_FONTSIZE,
            ha="left",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#FFCDD2",
                edgecolor="#C62828",
                alpha=0.95,
                linewidth=1.2,
            ),
            zorder=10,
        )

    ax.set_title(f"Frame {rec['frame']:5d}", fontsize=BEV_TITLE_FONTSIZE)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


ENCOUNTER_LABEL = {
    "straight_vs_turn": "straight vs turn",
    "turn_vs_straight": "turn vs straight",
    "turn_vs_turn": "turn vs turn",
    "straight_vs_straight": "straight vs straight",
}


def _row_at_time(pts: np.ndarray, t_pass: float) -> np.ndarray:
    idx = int(np.argmin(np.abs(pts[:, 0] - t_pass)))
    return pts[idx]


def pet_event_legend_handles(a_id: int, b_id: int) -> list[Line2D]:
    """Legend entries for front bumper @ passage and conflict point P (session_b style)."""
    return [
        Line2D(
            [0], [0], marker="^", linestyle="None",
            markerfacecolor="#2196F3", markeredgecolor="black",
            markersize=12, label=f"#{a_id} front bumper",
        ),
        Line2D(
            [0], [0], marker="^", linestyle="None",
            markerfacecolor="#F44336", markeredgecolor="black",
            markersize=12, label=f"#{b_id} front bumper",
        ),
        Line2D(
            [0], [0], marker="X", linestyle="None",
            markerfacecolor="#FFC107", markeredgecolor="black",
            markeredgewidth=1.5, markersize=14, label="conflict point P",
        ),
    ]


def plot_pet_event(
    event: dict,
    trajectories: dict[int, dict],
    roi_cfg: dict,
    out_path: Path,
    t0: float,
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    transparent: bool = False,
) -> None:
    """Per-PET-event chart: both full trajectories colored by time, crossing
    point and time gap clearly annotated.  This is the visualization the
    overview/per-frame plots cannot give, because PET is fundamentally about
    two different moments in time.

    Timestamps are displayed relative to `t0` (seconds since the first frame
    of the bag) so the reader can mentally verify `PET = t_second - t_first`.
    """
    a_id = event["a_id"]
    b_id = event["b_id"]
    a = trajectories.get(a_id)
    b = trajectories.get(b_id)
    if a is None or b is None:
        return

    # pts shape (N, 3) with rows (t, x, y)
    a_pts = np.asarray(a["pts"], dtype=float)
    b_pts = np.asarray(b["pts"], dtype=float)
    if a_pts.shape[0] < 2 or b_pts.shape[0] < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    if transparent:
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
        ax.patch.set_alpha(0.0)
    setup_bev_axes(ax, roi_cfg, xlim=xlim, ylim=ylim, transparent=transparent)

    cx, cy = event["x"], event["y"]
    man_a = event.get("a_maneuver", "straight")
    man_b = event.get("b_maneuver", "straight")

    def draw_traj(pts: np.ndarray, cmap_name: str, *, lw_scale: float = 1.0):
        n = len(pts)
        cmap = plt.get_cmap(cmap_name)
        for k in range(n - 1):
            color = cmap(k / max(n - 1, 1))
            ax.plot(pts[k:k + 2, 1], pts[k:k + 2, 2], color=color,
                    linewidth=2.4 * lw_scale, alpha=0.95, zorder=4)
        ax.scatter([pts[0, 1]], [pts[0, 2]], c="white", edgecolors="black",
                   s=40, zorder=5)
        ax.scatter([pts[-1, 1]], [pts[-1, 2]], color=cmap(1.0),
                   edgecolors="black", s=70, marker=">", zorder=5)

    # # Annotate start/end timestamps on each trajectory (disabled — clutters BEV).
    # def annotate_endpoints(pts: np.ndarray, color: str):
    #     t_start_rel = pts[0, 0] - t0
    #     t_end_rel = pts[-1, 0] - t0
    #     ax.annotate(
    #         f"start  t={t_start_rel:.2f}s",
    #         xy=(pts[0, 1], pts[0, 2]),
    #         xytext=(pts[0, 1] - 1.0, pts[0, 2] - 1.0),
    #         fontsize=7, color=color,
    #         bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
    #                   edgecolor=color, alpha=0.9, linewidth=0.6),
    #     )
    #     ax.annotate(
    #         f"end  t={t_end_rel:.2f}s",
    #         xy=(pts[-1, 1], pts[-1, 2]),
    #         xytext=(pts[-1, 1] + 0.5, pts[-1, 2] - 1.0),
    #         fontsize=7, color=color,
    #         bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
    #                   edgecolor=color, alpha=0.9, linewidth=0.6),
    #     )

    draw_traj(a_pts, "Blues", lw_scale=1.35 if man_a == "turning" else 1.0)
    draw_traj(b_pts, "Reds", lw_scale=1.35 if man_b == "turning" else 1.0)
    # annotate_endpoints(a_pts, plt.get_cmap("Blues")(0.85))
    # annotate_endpoints(b_pts, plt.get_cmap("Reds")(0.85))

    t_a = event.get("t_a", event["t_first"] if event["first_is"] == "a" else event["t_second"])
    t_b = event.get("t_b", event["t_second"] if event["first_is"] == "a" else event["t_first"])
    pet = event["pet"]

    pa = event.get("passage_a_xy")
    pb = event.get("passage_b_xy")
    if pa is None:
        row_a = _row_at_time(a_pts, t_a)
        pa = [float(row_a[1]), float(row_a[2])]
    if pb is None:
        row_b = _row_at_time(b_pts, t_b)
        pb = [float(row_b[1]), float(row_b[2])]

    row_a = _row_at_time(a_pts, t_a)
    row_b = _row_at_time(b_pts, t_b)
    if row_a.shape[0] >= 6:
        corners_a = get_bbox_corners(row_a[1], row_a[2], row_a[4], row_a[5], row_a[3])
        ax.add_patch(MplPolygon(corners_a, closed=True, fill=False,
                                edgecolor="#1565C0", linewidth=1.8, linestyle="-", zorder=6))
    if row_b.shape[0] >= 6:
        corners_b = get_bbox_corners(row_b[1], row_b[2], row_b[4], row_b[5], row_b[3])
        ax.add_patch(MplPolygon(corners_b, closed=True, fill=False,
                                edgecolor="#C62828", linewidth=1.8, linestyle="-", zorder=6))

    ax.plot([pa[0], cx], [pa[1], cy], color="#1565C0", linewidth=1.2, linestyle=":", zorder=5)
    ax.plot([pb[0], cx], [pb[1], cy], color="#C62828", linewidth=1.2, linestyle=":", zorder=5)
    ax.scatter([pa[0]], [pa[1]], marker="^", s=120, c="#2196F3",
               edgecolors="black", zorder=7)
    ax.scatter([pb[0]], [pb[1]], marker="^", s=120, c="#F44336",
               edgecolors="black", zorder=7)

    ax.scatter([cx], [cy], marker="X", s=320,
               c="#FFC107", edgecolors="black", linewidths=1.5, zorder=8)

    pet_box_alpha = 0.75 if transparent else 0.92
    ax.text(
        0.02,
        0.98,
        f"PET={pet:.2f}s",
        transform=ax.transAxes,
        fontsize=BEV_ANNOTATION_FONTSIZE,
        ha="left",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="#888888",
            alpha=pet_box_alpha,
            linewidth=0.8,
        ),
        zorder=10,
    )

    if not transparent:
        ax.set_title(f"#{a_id} vs #{b_id}", fontsize=BEV_TITLE_FONTSIZE)
    leg = ax.legend(
        handles=pet_event_legend_handles(a_id, b_id),
        loc="lower right",
        fontsize=BEV_LEGEND_FONTSIZE,
        framealpha=0.85 if transparent else 0.92,
    )
    if transparent:
        leg.get_frame().set_facecolor((1.0, 1.0, 1.0, 0.72))
        leg.get_frame().set_edgecolor("#666666")
    fig.savefig(out_path, **_savefig_kwargs(transparent))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracks",
        type=Path,
        default=REPO / "conflict_analysis" / "output" / "tracks.jsonl",
    )
    parser.add_argument(
        "--ttc-events",
        type=Path,
        default=REPO / "conflict_analysis" / "output" / "ttc_events.jsonl",
    )
    parser.add_argument(
        "--pet-events",
        type=Path,
        default=REPO / "conflict_analysis" / "output" / "pet_events.jsonl",
    )
    parser.add_argument(
        "--ttc-summary",
        type=Path,
        default=REPO / "conflict_analysis" / "output" / "ttc_summary.json",
    )
    parser.add_argument(
        "--pet-summary",
        type=Path,
        default=REPO / "conflict_analysis" / "output" / "pet_summary.json",
    )
    parser.add_argument(
        "--roi-yaml",
        type=Path,
        default=REPO / "config" / "roi.yaml",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "conflict_analysis" / "output" / "viz",
    )
    parser.add_argument("--max-bev-frames", type=int, default=40,
                        help="cap the number of per-frame TTC BEV plots")
    parser.add_argument("--top-k-pet", type=int, default=0,
                        help="max PET BEV plots (0 = all after filters); ranked by crossing quality")
    parser.add_argument(
        "--pet-render-all",
        action="store_true",
        help="render all PET events in pet_events.jsonl (ignore viz-pet-max filter)",
    )
    parser.add_argument(
        "--pet-conflict-point-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="strict: only true geometric conflict points (few plots); "
        "default off: straight/turn crossings ranked by trajectory divergence",
    )
    parser.add_argument(
        "--pet-viz-max-co-overlap",
        type=float,
        default=0.90,
        help="drop pairs with path co-overlap above this (co-travel / same corridor)",
    )
    parser.add_argument(
        "--pet-viz-min-crossing-angle",
        type=float,
        default=40.0,
        help="min heading difference at conflict P for PET BEV selection",
    )
    parser.add_argument(
        "--pet-viz-min-approach-angle",
        type=float,
        default=40.0,
        help="min approach angle toward P (filters same-direction follow-through)",
    )
    parser.add_argument(
        "--pet-viz-straight-turn-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="if set, only straight_vs_turn / turn_vs_straight (legacy filter)",
    )
    parser.add_argument("--pet-max-co-overlap", type=float, default=0.40)
    parser.add_argument("--pet-min-median-sep", type=float, default=2.5)
    parser.add_argument("--pet-min-approach-angle", type=float, default=45.0)
    parser.add_argument(
        "--pet-one-per-pair",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="one BEV per track pair (best crossing quality, not smallest PET)",
    )
    parser.add_argument("--overview-top-pet", type=int, default=3,
                        help="on overview BEV, top-N conflict points (excl. straight/straight, turn/straight)")
    parser.add_argument(
        "--overview-out",
        type=Path,
        default=None,
        help="overview PNG path (default: out-dir/overview/conflicts_bev_nearmiss.png)",
    )
    parser.add_argument(
        "--overview-raw-tracking",
        action="store_true",
        help="overview: raw track IDs only (no stitch / filled trips)",
    )
    parser.add_argument(
        "--overview-no-filled-trips",
        action="store_true",
        help="overview: stitched through-trips only, omit filled (non in/out) trips",
    )
    parser.add_argument(
        "--overview-no-title",
        action="store_true",
        help="overview: omit figure title",
    )
    parser.add_argument("--pet-min-traj-length", type=float, default=2.0,
                        help="meters; same filter as evaluate_pet.py (must match)")
    parser.add_argument("--pet-downsample-step", type=float, default=0.5,
                        help="meters; same as evaluate_pet.py (must match)")
    parser.add_argument("--pet-pair-classes", nargs="*", default=["vehicle"],
                        help="classes used when re-building trajectories for PET plots")
    parser.add_argument(
        "--viz-ttc-max",
        type=float,
        default=2.0,
        help="only visualize TTC events with ttc < this (seconds); 0 = no filter",
    )
    parser.add_argument(
        "--viz-pet-max",
        type=float,
        default=0.0,
        help="optional cap: only PET < this (seconds); 0 = no PET-size filter",
    )
    parser.add_argument(
        "--danger-pet",
        type=float,
        default=None,
        help="red threshold on PET plots (default: pet_summary config or 2.0s)",
    )
    parser.add_argument("--bev-xmin", type=float, default=None)
    parser.add_argument("--bev-xmax", type=float, default=None)
    parser.add_argument("--bev-ymin", type=float, default=None)
    parser.add_argument("--bev-ymax", type=float, default=None)
    parser.add_argument(
        "--ttc-frame-auto-zoom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="per-frame TTC BEV: crop to vehicles in frame (default on)",
    )
    parser.add_argument(
        "--ttc-frame-zoom-margin",
        type=float,
        default=12.0,
        help="metres of padding around vehicles for per-frame TTC BEV crop",
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="PET BEV PNGs: transparent figure background (for road-map overlay)",
    )
    parser.add_argument(
        "--pet-plots-only",
        action="store_true",
        help="only render per-PET-event BEVs (skip summary / overview / TTC frames)",
    )
    args = parser.parse_args()

    with open(args.roi_yaml) as f:
        roi_cfg = yaml.safe_load(f)
    roi_xlim, roi_ylim = bev_limits_from_roi(roi_cfg)
    bev_xlim = (
        args.bev_xmin if args.bev_xmin is not None else roi_xlim[0],
        args.bev_xmax if args.bev_xmax is not None else roi_xlim[1],
    )
    bev_ylim = (
        args.bev_ymin if args.bev_ymin is not None else roi_ylim[0],
        args.bev_ymax if args.bev_ymax is not None else roi_ylim[1],
    )

    ttc_summary = {}
    if args.ttc_summary.exists():
        ttc_summary = json.loads(args.ttc_summary.read_text())
    pet_summary = {}
    if args.pet_summary.exists():
        pet_summary = json.loads(args.pet_summary.read_text())
    danger_ttc = ttc_summary.get("config", {}).get("danger_ttc", 3.0)
    ttc_cap = ttc_summary.get("config", {}).get("ttc_cap", 15.0)
    danger_pet = (
        args.danger_pet
        if args.danger_pet is not None
        else pet_summary.get("config", {}).get("danger_pet", 2.0)
    )

    frames = []
    if args.tracks.exists():
        frames = load_jsonl(args.tracks)
    print(f"Loaded {len(frames)} frames of tracks")

    ttc_all = load_jsonl(args.ttc_events) if args.ttc_events.exists() else []
    pet_all = load_jsonl(args.pet_events) if args.pet_events.exists() else []
    print(f"Loaded {len(ttc_all)} TTC events, {len(pet_all)} PET crossings")

    viz_ttc_max = args.viz_ttc_max if args.viz_ttc_max > 0 else float("inf")
    viz_pet_max = args.viz_pet_max if args.viz_pet_max > 0 else float("inf")
    ttc_events = [e for e in ttc_all if e["ttc"] < viz_ttc_max]
    pet_events = [e for e in pet_all if e["pet"] < viz_pet_max]
    print(
        f"Viz filter: TTC < {viz_ttc_max}s -> {len(ttc_events)} events, "
        f"PET < {viz_pet_max}s -> {len(pet_events)} events"
    )

    summary_dir = args.out_dir / "summary"
    overview_dir = args.out_dir / "overview"
    ttc_frame_dir = args.out_dir / "ttc_frames"
    pet_event_dir = args.out_dir / (
        "pet_conflicts" if args.pet_conflict_point_only else "pet_events"
    )
    for d in (summary_dir, overview_dir, ttc_frame_dir, pet_event_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not args.pet_plots_only:
        print("TTC distribution (full bag, lead-follow) ...")
        if ttc_all:
            plot_ttc_distribution(ttc_all, nearmiss_ttc=danger_ttc, out_dir=summary_dir)

        print("Summary plots (near-miss subset for BEV stats) ...")
        if ttc_events:
            plot_ttc_summary(
                ttc_events, viz_ttc_max, summary_dir,
                title_suffix=f"  [viz: TTC<{viz_ttc_max}s]",
            )
        print("PET distribution (full bag) ...")
        if pet_all:
            plot_pet_distribution(pet_all, danger_pet=danger_pet, out_dir=summary_dir)

        if pet_events:
            plot_pet_summary(
                pet_events, danger_pet, summary_dir,
                title_suffix=f"  [viz: PET<{viz_pet_max}s]",
            )

        if pet_all:
            print("Overview BEV (top crossing-quality conflict points) ...")
            st_pet = _filter_pet_for_viz(pet_all, args)
            st_pet = sorted(st_pet, key=pet_conflict_priority, reverse=True)
            overview_path = args.overview_out or (
                overview_dir / "conflicts_bev_nearmiss.png"
            )
            plot_overview_bev(
                frames, st_pet, roi_cfg,
                overview_path,
                top_k_pet=args.overview_top_pet,
                xlim=bev_xlim,
                ylim=bev_ylim,
                include_filled_trips=not args.overview_no_filled_trips,
                raw_tracking_only=args.overview_raw_tracking,
                show_title=not args.overview_no_title,
            )

        print(f"Per-frame TTC BEV (TTC < {viz_ttc_max}s only) ...")
        events_by_frame: dict[int, list[dict]] = defaultdict(list)
        for e in ttc_events:
            events_by_frame[e["frame"]].append(e)
        dangerous_frames = sorted(events_by_frame.keys())[: args.max_bev_frames]
        frame_index = {r["frame"]: r for r in frames}
        for fidx in dangerous_frames:
            rec = frame_index.get(fidx)
            if rec is None:
                continue
            out = ttc_frame_dir / f"frame_{fidx:05d}.png"
            plot_ttc_frame_bev(
                rec, events_by_frame[fidx], roi_cfg,
                viz_ttc_max, ttc_cap, out,
                xlim=bev_xlim, ylim=bev_ylim,
                auto_zoom=args.ttc_frame_auto_zoom,
                zoom_margin_m=args.ttc_frame_zoom_margin,
            )
        print(f"  wrote {len(dangerous_frames)} per-frame BEVs -> {ttc_frame_dir}")

    pet_to_plot = pet_all if args.pet_render_all else pet_events
    if pet_to_plot:
        label = "all candidates" if args.pet_render_all else (
            f"PET < {viz_pet_max}s" if args.viz_pet_max > 0 else "all PET (no size cap)"
        )
        if args.pet_conflict_point_only:
            label += ", strict true conflict point"
        elif args.pet_viz_straight_turn_only:
            label += ", straight/turn only"
        else:
            label += (
                f", crossing>={args.pet_viz_min_crossing_angle}° "
                f"approach>={args.pet_viz_min_approach_angle}°"
            )
        print(f"Per-PET-event BEV ({label}) ...")
        trajectories = build_pet_trajectories(frames, set(args.pet_pair_classes))
        for tid, data in list(trajectories.items()):
            pts = data["pts"]
            if len(pts) < 3:
                trajectories.pop(tid)
                continue
            seg_len = float(np.linalg.norm(np.diff(pts[:, 1:3], axis=0), axis=1).sum())
            if seg_len < args.pet_min_traj_length:
                trajectories.pop(tid)
                continue
            data["length_m"] = seg_len
            data["pts"] = downsample_polyline(pts, args.pet_downsample_step)
        if args.pet_conflict_point_only:
            kept: list[dict] = []
            for e in pet_to_plot:
                a_id, b_id = e["a_id"], e["b_id"]
                if a_id not in trajectories or b_id not in trajectories:
                    continue
                if e.get("true_conflict_point") is True:
                    kept.append(e)
                    continue
                if is_true_pet_conflict_point(
                    trajectories[a_id]["pts"],
                    trajectories[b_id]["pts"],
                    e,
                    max_co_overlap_frac=args.pet_max_co_overlap,
                    min_median_sep_m=args.pet_min_median_sep,
                    min_approach_angle_deg=args.pet_min_approach_angle,
                ):
                    kept.append(e)
            pet_to_plot = kept
            print(f"  {len(pet_to_plot)} strict true conflict-point events")
        else:
            pet_to_plot = _filter_pet_for_viz(pet_to_plot, args)
            print(f"  {len(pet_to_plot)} diverging-direction crossing events")
        if args.pet_one_per_pair:
            best_by_pair: dict[tuple[int, int], dict] = {}
            for e in pet_to_plot:
                k = pet_pair_key(e)
                if k not in best_by_pair or pet_conflict_priority(e) > pet_conflict_priority(
                    best_by_pair[k]
                ):
                    best_by_pair[k] = e
            pet_to_plot = list(best_by_pair.values())
            print(f"  {len(pet_to_plot)} unique track pairs (best crossing per pair)")
        sorted_events = sorted(pet_to_plot, key=pet_conflict_priority, reverse=True)
        if args.top_k_pet > 0:
            limit = args.top_k_pet
        elif args.pet_render_all:
            limit = len(sorted_events)
        else:
            limit = len(sorted_events)
        t0 = frames[0]["stamp_sec"] if frames else 0.0
        rendered = 0
        for rank, ev in enumerate(sorted_events[:limit]):
            tag = ev.get("encounter_type", "pet")
            pri = int(pet_conflict_priority(ev))
            out = pet_event_dir / (
                f"conflict_{rank:03d}_{tag}_"
                f"pair{min(ev['a_id'], ev['b_id'])}_{max(ev['a_id'], ev['b_id'])}_"
                f"pet_{ev['pet']:.2f}s_pri{pri}.png"
            )
            plot_pet_event(
                ev, trajectories, roi_cfg, out, t0=t0,
                xlim=bev_xlim, ylim=bev_ylim,
                transparent=args.transparent,
            )
            rendered += 1
        print(f"  wrote {rendered} PET-event plots -> {pet_event_dir}")


if __name__ == "__main__":
    main()
