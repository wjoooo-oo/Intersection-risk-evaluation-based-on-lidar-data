#!/usr/bin/env python3
"""Combined TTC vs TTA distribution summary PNG + console stats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]


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


def load_jsonl(path: Path, key: str) -> list[float]:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(float(json.loads(line)[key]))
    return out


def pair_mins(events_path: Path, id_keys: tuple[str, str], val_key: str) -> np.ndarray:
    pair_min: dict[tuple[int, int], float] = {}
    with open(events_path) as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            k = (min(e[id_keys[0]], e[id_keys[1]]), max(e[id_keys[0]], e[id_keys[1]]))
            v = float(e[val_key])
            pair_min[k] = min(pair_min.get(k, v), v)
    return np.array(list(pair_min.values())) if pair_min else np.array([])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "conflict_analysis" / "output" / "viz" / "summary" / "ttc_tta_comparison.png",
    )
    parser.add_argument(
        "--ttc-events",
        type=Path,
        default=REPO / "conflict_analysis" / "output" / "ttc_events.jsonl",
    )
    parser.add_argument(
        "--tta-events",
        type=Path,
        default=REPO / "conflict_analysis" / "output" / "tta_events.jsonl",
    )
    args = parser.parse_args()

    ttc_path = args.ttc_events
    tta_path = args.tta_events

    ttc_ev = load_jsonl(ttc_path, "ttc") if ttc_path.exists() else []
    tta_ev = load_jsonl(tta_path, "tta") if tta_path.exists() else []
    ttc_pair = pair_mins(ttc_path, ("rear_id", "lead_id"), "ttc") if ttc_path.exists() else np.array([])
    tta_pair = pair_mins(tta_path, ("a_id", "b_id"), "tta") if tta_path.exists() else np.array([])

    th = args.threshold
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, vals, name, color in (
        (axes[0, 0], np.array(ttc_ev), "TTC (lead-follow)", "#1565C0"),
        (axes[0, 1], np.array(tta_ev), "TTA (any pair)", "#E65100"),
    ):
        if len(vals):
            if "TTC" in name:
                n_neg = plot_ttc_histogram(ax, vals, color=color)
            else:
                n_neg = 0
                ax.hist(vals, bins=40, color=color, alpha=0.85, edgecolor="black")
            ax.axvline(th, color="red", linestyle="--", linewidth=2)
            n_d = int((vals < th).sum())
            title_extra = f", <0: {n_neg}/{len(vals)}" if n_neg else ""
            ax.set_title(f"{name} per-frame\nn={len(vals)}, <{th}s: {n_d}/{len(vals)}{title_extra}")
        else:
            ax.set_title(f"{name}: no events")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)

    for ax, vals, name, color in (
        (axes[1, 0], ttc_pair, "TTC unique pairs", "#2E7D32"),
        (axes[1, 1], tta_pair, "TTA unique pairs", "#6D4C41"),
    ):
        if len(vals):
            if "TTC" in name:
                n_neg = plot_ttc_histogram(ax, vals, color=color)
            else:
                n_neg = 0
                ax.hist(vals, bins=30, color=color, alpha=0.85, edgecolor="black")
            ax.axvline(th, color="red", linestyle="--", linewidth=2)
            n_d = int((vals < th).sum())
            title_extra = f", <0: {n_neg}/{len(vals)}" if n_neg else ""
            ax.set_title(f"{name} (min per pair)\nn={len(vals)}, <{th}s: {n_d}/{len(vals)}{title_extra}")
        else:
            ax.set_title(f"{name}: no pairs")
        ax.set_xlabel("min time (s)")
        ax.set_ylabel("pairs")
        ax.grid(alpha=0.3)

    def line(name, ev, pr):
        if len(ev) == 0:
            return f"{name}: (none)\n"
        neg_line = (
            f"  TTC < 0: {int((ev < 0).sum())}/{len(ev)}\n"
            if "TTC" in name
            else ""
        )
        return (
            f"{name}\n"
            f"  events: {len(ev)},  <{th}s: {int((ev<th).sum())}/{len(ev)}\n"
            f"  range: [{ev.min():.3f}, {ev.max():.3f}] s\n"
            f"{neg_line}"
            f"  mean {ev.mean():.3f} s, median {np.median(ev):.3f} s\n"
            f"  unique pairs: {len(pr)},  <{th}s: {int((pr<th).sum())}/{len(pr)}\n"
            f"  pair min range: [{pr.min():.3f}, {pr.max():.3f}] s\n"
        )

    txt = line("TTC (lead-follow)", np.array(ttc_ev), ttc_pair) + line("TTA (any pair)", np.array(tta_ev), tta_pair)
    fig.text(0.5, 0.02, txt, ha="center", va="bottom", fontsize=10, family="monospace",
             bbox=dict(boxstyle="round", facecolor="#F5F5F5", edgecolor="#999"))

    fig.suptitle("TTC (car-following) vs TTA (any converging pair)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(txt)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
