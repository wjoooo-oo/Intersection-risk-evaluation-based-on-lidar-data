#!/usr/bin/env python3
"""Export TTC / PET events to minimal CSV: pair IDs + metric value only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

SCENES = [
    ("session_a", REPO / "conflict_analysis" / "output"),
    ("session_b", REPO / "conflict_analysis" / "output" / "session_b"),
    ("session_c", REPO / "conflict_analysis" / "output" / "session_c"),
]


def export_ttc(ttc_jsonl: Path, out_csv: Path) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(ttc_jsonl) as fin, open(out_csv, "w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["rear_id", "lead_id", "ttc"])
        for line in fin:
            if not line.strip():
                continue
            e = json.loads(line)
            w.writerow([e["rear_id"], e["lead_id"], e["ttc"]])
            n += 1
    return n


def export_pet(pet_jsonl: Path, out_csv: Path) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(pet_jsonl) as fin, open(out_csv, "w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["a_id", "b_id", "pet"])
        for line in fin:
            if not line.strip():
                continue
            e = json.loads(line)
            w.writerow([e["a_id"], e["b_id"], e["pet"]])
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-dir",
        type=Path,
        action="append",
        help="scene output dir containing ttc_events.jsonl and pet_events.jsonl (repeatable)",
    )
    parser.add_argument(
        "--all-default-scenes",
        action="store_true",
        help="export output/, output/session_b/, output/session_c/",
    )
    args = parser.parse_args()

    dirs: list[Path] = []
    if args.all_default_scenes:
        dirs = [p for _, p in SCENES]
    if args.scene_dir:
        dirs.extend(args.scene_dir)

    if not dirs:
        parser.error("Specify --all-default-scenes or --scene-dir")

    for d in dirs:
        ttc_in = d / "ttc_events.jsonl"
        pet_in = d / "pet_events.jsonl"
        if ttc_in.exists():
            ttc_out = d / "ttc_pairs.csv"
            n = export_ttc(ttc_in, ttc_out)
            print(f"Wrote {n} rows -> {ttc_out}")
        else:
            print(f"skip (missing): {ttc_in}")
        if pet_in.exists():
            pet_out = d / "pet_pairs.csv"
            n = export_pet(pet_in, pet_out)
            print(f"Wrote {n} rows -> {pet_out}")
        else:
            print(f"skip (missing): {pet_in}")


if __name__ == "__main__":
    main()
