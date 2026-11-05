#!/usr/bin/env python3
"""BEV heatmap of PET conflict points — smaller PET → darker (higher risk weight)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath

REPO = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def filter_events_in_roi(events: list[dict], roi_path: MplPath) -> list[dict]:
    """Keep only conflict points whose centre lies inside the ROI polygon."""
    inside = roi_path.contains_points(
        np.array([[e["x"], e["y"]] for e in events], dtype=float)
    )
    kept = [e for e, ok in zip(events, inside) if ok]
    n_drop = len(events) - len(kept)
    if n_drop:
        print(f"  ROI filter: {len(kept)} in / {len(events)} events ({n_drop} outside skipped)")
    return kept


def cell_center_mask(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    roi_path: MplPath,
) -> np.ndarray:
    """True where cell centre is inside ROI (strict polygon mask)."""
    cx = 0.5 * (x_edges[:-1] + x_edges[1:])
    cy = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(cx, cy)
    inside = roi_path.contains_points(np.column_stack([xx.ravel(), yy.ravel()]))
    return inside.reshape(xx.shape)


def smooth_grid(
    grid: np.ndarray,
    mask: np.ndarray,
    sigma_cells: float,
) -> np.ndarray:
    """Gaussian blur using only in-ROI cells; hard-clip to ROI after blur (no exterior bleed)."""
    if sigma_cells <= 0:
        return np.where(mask, grid, np.nan)
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        return np.where(mask, grid, np.nan)

    m = mask.astype(np.float64)
    filled = np.where(mask, np.nan_to_num(grid, nan=0.0), 0.0)
    num = gaussian_filter(filled * m, sigma=sigma_cells, mode="constant", cval=0.0)
    den = gaussian_filter(m, sigma=sigma_cells, mode="constant", cval=0.0)
    blurred = np.full_like(filled, np.nan, dtype=float)
    valid = (den > 1e-8) & mask
    blurred[valid] = num[valid] / den[valid]
    return np.where(mask, blurred, np.nan)


def roi_interior_fade(
    mask: np.ndarray,
    cell_size_m: float,
    fade_m: float,
) -> np.ndarray:
    """Weight 0→1 from ROI boundary inward over fade_m (soft edge, no hard clip)."""
    if fade_m <= 0:
        return np.where(mask, 1.0, 0.0)
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return np.where(mask, 1.0, 0.0)
    dist_m = distance_transform_edt(mask) * cell_size_m
    w = np.clip(dist_m / fade_m, 0.0, 1.0)
    return np.where(mask, w, 0.0)


def pet_weight(pet: float, *, mode: str, pet_ref: float) -> float:
    """Larger weight for smaller PET (more dangerous encounter)."""
    pet = max(float(pet), 0.05)
    if mode == "inverse":
        return 1.0 / pet
    if mode == "inverse_square":
        return 1.0 / (pet * pet)
    # linear: strongest when pet -> 0
    return max(0.0, pet_ref - pet) / max(pet_ref, 1e-6)


def plot_pet_heatmap(
    events: list[dict],
    roi_cfg: dict,
    out_path: Path,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    resolution: float = 1.0,
    blur_sigma_m: float = 3.0,
    edge_fade_m: float | None = None,
    weight_mode: str = "inverse",
    pet_ref: float = 10.0,
    cmap: str = "hot_r",
    title: str = "",
    transparent: bool = False,
    white_cut_frac: float = 0.08,
) -> None:
    polygon = np.array(roi_cfg["polygon_xy"])
    roi_path = MplPath(polygon, closed=True)

    if not events:
        print(f"skip heatmap (no events): {out_path}")
        return

    events_in = filter_events_in_roi(events, roi_path)
    if not events_in:
        print(f"skip heatmap (no in-ROI events): {out_path}")
        return

    xs = np.array([e["x"] for e in events_in], dtype=float)
    ys = np.array([e["y"] for e in events_in], dtype=float)
    ws = np.array([pet_weight(e["pet"], mode=weight_mode, pet_ref=pet_ref) for e in events_in])

    x_min, x_max = xlim
    y_min, y_max = ylim
    x_edges = np.arange(x_min, x_max + resolution, resolution)
    y_edges = np.arange(y_min, y_max + resolution, resolution)
    if len(x_edges) < 2 or len(y_edges) < 2:
        raise SystemExit("BEV extent too small for heatmap resolution.")

    hist, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges], weights=ws)
    grid = hist.T

    mask = cell_center_mask(x_edges, y_edges, roi_path)
    grid = np.where(mask, grid, np.nan)
    sigma_cells = float(blur_sigma_m) / max(resolution, 1e-6)
    grid = smooth_grid(grid, mask, sigma_cells)

    fade_m = edge_fade_m if edge_fade_m is not None else max(2.0, min(blur_sigma_m * 0.75, 5.0))
    fade_w = roi_interior_fade(mask, resolution, fade_m)
    grid = np.where(mask, grid * fade_w, np.nan)

    display_grid = grid
    if transparent:
        inside = mask & np.isfinite(grid)
        if np.any(inside):
            vmax = float(np.nanmax(grid[inside]))
            cut = max(vmax * white_cut_frac, 1e-9)
            display_grid = grid.copy()
            display_grid[inside & (display_grid < cut)] = np.nan

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap_obj = plt.get_cmap(cmap).copy()
    if transparent:
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
        ax.patch.set_alpha(0.0)
        cmap_obj.set_bad((0.0, 0.0, 0.0, 0.0))
        ax.grid(False)
    else:
        ax.set_facecolor("white")
        cmap_obj.set_bad(color="white")
        ax.grid(alpha=0.15)

    im = ax.imshow(
        display_grid,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap=cmap_obj,
        aspect="equal",
        interpolation="bilinear",
    )

    legend_scale = 1.25
    cbar = fig.colorbar(
        im, ax=ax, fraction=0.046 * legend_scale, pad=0.04,
    )
    cbar.ax.tick_params(labelsize=10 * legend_scale)
    cbar.set_label("conflict intensity", fontsize=11 * legend_scale)
    if transparent:
        cbar.ax.set_facecolor("none")

    ax.add_patch(
        MplPolygon(
            polygon,
            closed=True,
            fill=False,
            edgecolor="#333",
            linewidth=1.2,
            linestyle="-",
            alpha=0.9,
            zorder=6,
        )
    )

    if len(ws):
        ax.scatter(
            xs, ys,
            c="white",
            s=18 + 40 * (ws / max(ws.max(), 1e-6)),
            edgecolors="#333",
            linewidths=0.3,
            alpha=0.45,
            zorder=7,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    if title and not transparent:
        ax.set_title(title, fontsize=11)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kw = dict(dpi=150, bbox_inches="tight")
    if transparent:
        save_kw.update(transparent=True, facecolor="none", edgecolor="none")
    fig.savefig(out_path, **save_kw)
    plt.close(fig)
    print(f"  -> {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pet-events", type=Path, required=True)
    parser.add_argument("--roi-yaml", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="", help="title suffix e.g. session_a")
    parser.add_argument("--bev-xmin", type=float, default=None)
    parser.add_argument("--bev-xmax", type=float, default=None)
    parser.add_argument("--bev-ymin", type=float, default=None)
    parser.add_argument("--bev-ymax", type=float, default=None)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument(
        "--blur-sigma-m",
        type=float,
        default=3.0,
        help="Gaussian blur radius in metres (diffusion / 晕染)",
    )
    parser.add_argument(
        "--edge-fade-m",
        type=float,
        default=None,
        help="ROI boundary fade width in metres (default ~0.75× blur, 2–5 m)",
    )
    parser.add_argument("--weight-mode", choices=["inverse", "inverse_square", "linear"],
                        default="inverse")
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="PNG with transparent background; no title; ROI外/低强度区透明",
    )
    parser.add_argument(
        "--white-cut-frac",
        type=float,
        default=0.08,
        help="transparent mode: hide values below this fraction of peak (removes white halo)",
    )
    args = parser.parse_args()

    with open(args.roi_yaml) as f:
        roi_cfg = yaml.safe_load(f)
    bev = roi_cfg.get("bev", {})
    poly = np.array(roi_cfg["polygon_xy"])

    xlim = (
        args.bev_xmin if args.bev_xmin is not None else float(bev.get("x_min", poly[:, 0].min()) - 2),
        args.bev_xmax if args.bev_xmax is not None else float(bev.get("x_max", poly[:, 0].max()) + 2),
    )
    ylim = (
        args.bev_ymin if args.bev_ymin is not None else float(bev.get("y_min", poly[:, 1].min()) - 2),
        args.bev_ymax if args.bev_ymax is not None else float(bev.get("y_max", poly[:, 1].max()) + 2),
    )

    events = load_jsonl(args.pet_events)
    title = (
        ""
        if args.transparent
        else (f"PET conflict heatmap — {args.label}" if args.label else "")
    )
    plot_pet_heatmap(
        events,
        roi_cfg,
        args.out,
        xlim=xlim,
        ylim=ylim,
        resolution=args.resolution,
        blur_sigma_m=args.blur_sigma_m,
        edge_fade_m=args.edge_fade_m,
        weight_mode=args.weight_mode,
        title=title,
        transparent=args.transparent,
        white_cut_frac=args.white_cut_frac,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
