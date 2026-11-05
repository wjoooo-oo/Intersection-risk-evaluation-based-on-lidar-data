#!/usr/bin/env python3
"""Export canonical PET tables for session_b and session_c.

session_b (55 encounters)
  - straight_vs_turn / turn_vs_straight only
  - one row per track pair (best crossing quality)
  - top 55 by crossing rank (matches pipeline --top-k-pet 55)

session_c (91 encounters)
  - true_conflict_point only
  - one row per unique track pair
  - representative event = PET closest to that pair's median PET
    (mean ~4.01 s for the published 91-pair summary)
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys_path = REPO / "conflict_analysis" / "scripts"
import sys

sys.path.insert(0, str(sys_path))
from visualize_conflicts import pet_conflict_priority, pet_pair_key  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_jsonl(events: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def write_csv(events: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a_id", "b_id", "pet"])
        for e in events:
            w.writerow([e["a_id"], e["b_id"], e["pet"]])


def select_pet_session_b(pet_events: list[dict]) -> list[dict]:
    straight_turn = {
        "straight_vs_turn",
        "turn_vs_straight",
    }
    pool = [e for e in pet_events if e.get("encounter_type") in straight_turn]
    best: dict[tuple[int, int], dict] = {}
    for e in pool:
        k = pet_pair_key(e)
        if k not in best or pet_conflict_priority(e) > pet_conflict_priority(best[k]):
            best[k] = e
    ranked = sorted(best.values(), key=pet_conflict_priority, reverse=True)
    return ranked[:55]


def select_pet_session_c(pet_events: list[dict]) -> list[dict]:
    """One event per true-conflict pair (91 pairs in current bags)."""
    true = [e for e in pet_events if e.get("true_conflict_point")]
    by_pair: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for e in true:
        by_pair[pet_pair_key(e)].append(e)

    selected: list[dict] = []
    for evs in by_pair.values():
        pets = [float(e["pet"]) for e in evs]
        med = float(np.median(pets))
        rep = min(evs, key=lambda e: abs(float(e["pet"]) - med))
        selected.append(rep)
    selected.sort(key=lambda e: (e["a_id"], e["b_id"]))
    return selected


def load_pet_csv(path: Path) -> np.ndarray:
    pets: list[float] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pets.append(float(row["pet"]))
    return np.array(pets, dtype=float)


def plot_pet_distribution(
    pets: np.ndarray,
    out_path: Path,
    *,
    nearmiss_pet: float = 2.0,
    xlim: tuple[float, float] | None = None,
) -> None:
    """Single-panel PET count histogram (matches TTC distribution style)."""
    if len(pets) == 0:
        return
    n_danger = int(((pets > 0) & (pets < nearmiss_pet)).sum())
    pct_danger = 100.0 * n_danger / len(pets)

    if xlim is None:
        x0 = 0.0
        x1 = float(min(10.0, np.ceil(pets.max() * 2.0) / 2.0))
        if x1 - x0 < 4.0:
            x1 = x0 + 4.0
    else:
        x0, x1 = xlim

    bins = np.linspace(x0, x1, 41)
    vals_plot = np.clip(pets, x0, x1 - 1e-9)
    counts, edges = np.histogram(vals_plot, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    label_fs, tick_fs = 22, 18
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.bar(
        centers,
        counts,
        width=widths,
        color="#90A4AE",
        edgecolor="black",
        alpha=0.85,
    )
    mask = (centers > 0) & (centers < nearmiss_pet)
    if mask.any():
        ax.bar(
            centers[mask],
            counts[mask],
            width=widths[mask],
            color="#C62828",
            edgecolor="black",
            alpha=0.9,
            zorder=5,
        )
    ax.set_xlim(x0, x1)
    ax.set_xlabel("PET (s)", fontsize=label_fs)
    ax.set_ylabel("Number of observations", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.grid(alpha=0.25)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor="#C62828", edgecolor="black", alpha=0.9)
        ],
        labels=[f"0<PET<{nearmiss_pet:g}s: {n_danger}/{len(pets)} ({pct_danger:.2f}%)"],
        loc="upper right",
        frameon=False,
        fontsize=14,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(
        f"  -> {out_path}  "
        f"range=[{pets.min():.3f},{pets.max():.3f}] mean={pets.mean():.3f} "
        f"pet<2s={n_danger}/{len(pets)}"
    )


def print_stats(events: list[dict], label: str) -> None:
    pets = np.array([float(e["pet"]) for e in events])
    n = len(pets)
    lt2 = int((pets < 2.0).sum())
    print(
        f"{label}: n={n}  "
        f"range=[{pets.min():.3f}, {pets.max():.3f}] s  "
        f"mean={pets.mean():.3f} s  "
        f"pet<2s: {lt2}/{n} ({100.0 * lt2 / n:.2f}%)"
    )


def export_scene(scene_dir: Path, selector, *, source_name: str = "pet_events.jsonl") -> None:
    pet_in = scene_dir / source_name
    if not pet_in.exists():
        raise FileNotFoundError(pet_in)
    all_events = load_jsonl(pet_in)
    canonical = selector(all_events)

    # Backup full bag once
    backup = scene_dir / "pet_events_all.jsonl"
    if not backup.exists() and pet_in.exists():
        backup.write_text(pet_in.read_text())

    write_jsonl(canonical, scene_dir / "pet_events.jsonl")
    write_csv(canonical, scene_dir / "pet_pairs.csv")
    print_stats(canonical, scene_dir.name)
    plot_pet_distribution(
        np.array([float(e["pet"]) for e in canonical]),
        scene_dir / "pet_distribution.png",
    )


def plot_from_csv(scene_dir: Path, *, xlim: tuple[float, float] | None = None) -> None:
    csv_path = scene_dir / "pet_pairs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    pets = load_pet_csv(csv_path)
    plot_pet_distribution(pets, scene_dir / "pet_distribution.png", xlim=xlim)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session_b",
        action="store_true",
        help="export conflict_analysis/output/session_b",
    )
    parser.add_argument(
        "--session_c",
        action="store_true",
        help="export conflict_analysis/output/session_c",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="only draw pet_distribution.png from existing pet_pairs.csv",
    )
    parser.add_argument("--pet-xmin", type=float, default=0.0)
    parser.add_argument("--pet-xmax", type=float, default=10.0)
    args = parser.parse_args()
    if not args.__dict__["session_b"] and not args.__dict__["session_c"]:
        args.__dict__["session_b"] = True
        args.__dict__["session_c"] = True

    out = REPO / "conflict_analysis" / "output"
    xlim = (args.pet_xmin, args.pet_xmax)

    if args.plot_only:
        if args.__dict__["session_b"]:
            plot_from_csv(out / "session_b", xlim=xlim)
        if args.__dict__["session_c"]:
            plot_from_csv(out / "session_c", xlim=xlim)
        return
    if args.__dict__["session_b"]:
        d164 = out / "session_b"
        src = (
            "pet_events_all.jsonl"
            if (d164 / "pet_events_all.jsonl").exists()
            else "pet_events.jsonl"
        )
        export_scene(d164, select_pet_session_b, source_name=src)
    if args.__dict__["session_c"]:
        d173 = out / "session_c"
        if (d173 / "pet_events_true.jsonl").exists():
            src = "pet_events_true.jsonl"
        elif (d173 / "pet_events_all.jsonl").exists():
            src = "pet_events_all.jsonl"
        else:
            src = "pet_events.jsonl"
        export_scene(d173, select_pet_session_c, source_name=src)


if __name__ == "__main__":
    main()
