#!/usr/bin/env python3
"""Merge TTC and PET outputs into a single near-miss report for intersection safety.

Near-miss criteria (defaults tuned for intersection conflict screening):
  - TTC event with ttc <= danger_ttc  (default 2.0 s)
  - PET encounter with pet <= danger_pet (default 3.0 s)

Outputs
-------
nearmiss_report.json   counts, top events, unique risky track pairs
nearmiss_ttc.jsonl     filtered TTC rows
nearmiss_pet.jsonl     filtered PET rows
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--out-dir",
        type=Path,
        default=REPO / "conflict_analysis" / "output",
    )
    parser.add_argument("--danger-ttc", type=float, default=2.0)
    parser.add_argument("--danger-pet", type=float, default=3.0)
    args = parser.parse_args()

    ttc_all = load_jsonl(args.ttc_events)
    pet_all = load_jsonl(args.pet_events)

    ttc_nm = [e for e in ttc_all if e.get("ttc", 999) <= args.danger_ttc]
    pet_nm = [e for e in pet_all if e.get("pet", 999) <= args.danger_pet]

    ttc_nm.sort(key=lambda e: e["ttc"])
    pet_nm.sort(key=lambda e: e["pet"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ttc_out = args.out_dir / "nearmiss_ttc.jsonl"
    pet_out = args.out_dir / "nearmiss_pet.jsonl"
    report_out = args.out_dir / "nearmiss_report.json"

    with open(ttc_out, "w") as f:
        for e in ttc_nm:
            f.write(json.dumps(e) + "\n")
    with open(pet_out, "w") as f:
        for e in pet_nm:
            f.write(json.dumps(e) + "\n")

    def _ttc_pair_key(e: dict) -> tuple[int, int]:
        a = int(e.get("rear_id", e.get("ego_id")))
        b = int(e.get("lead_id", e.get("target_id", 0)))
        return (min(a, b), max(a, b))

    ttc_pairs: dict[tuple[int, int], list] = defaultdict(list)
    for e in ttc_nm:
        ttc_pairs[_ttc_pair_key(e)].append(e["ttc"])

    pet_pairs = {}
    for e in pet_nm:
        key = (min(e["a_id"], e["b_id"]), max(e["a_id"], e["b_id"]))
        pet_pairs[key] = min(pet_pairs.get(key, 999), e["pet"])

    all_risky_pairs = set(ttc_pairs.keys()) | set(pet_pairs.keys())

    report = {
        "criteria": {
            "danger_ttc_s": args.danger_ttc,
            "danger_pet_s": args.danger_pet,
        },
        "counts": {
            "ttc_events_total": len(ttc_all),
            "ttc_nearmiss": len(ttc_nm),
            "pet_events_total": len(pet_all),
            "pet_nearmiss": len(pet_nm),
            "unique_risky_track_pairs": len(all_risky_pairs),
        },
        "top_ttc_nearmiss": ttc_nm[:30],
        "top_pet_nearmiss": pet_nm[:30],
        "ttc_by_pair": [
            {
                "pair": list(k),
                "n_frames": len(v),
                "min_ttc": min(v),
                "median_ttc": float(sorted(v)[len(v) // 2]),
            }
            for k, v in sorted(ttc_pairs.items(), key=lambda kv: min(kv[1]))
        ][:30],
        "pet_by_pair": [
            {"pair": list(k), "min_pet": v}
            for k, v in sorted(pet_pairs.items(), key=lambda kv: kv[1])
        ][:30],
    }
    report_out.write_text(json.dumps(report, indent=2))

    print("=" * 60)
    print("NEAR-MISS REPORT (intersection safety)")
    print("=" * 60)
    print(f"  TTC <= {args.danger_ttc}s : {len(ttc_nm)} / {len(ttc_all)} events")
    print(f"  PET <= {args.danger_pet}s : {len(pet_nm)} / {len(pet_all)} encounters")
    print(f"  Unique risky track pairs : {len(all_risky_pairs)}")
    if ttc_nm:
        e0 = ttc_nm[0]
        id_a = e0.get("rear_id", e0.get("ego_id"))
        id_b = e0.get("lead_id", e0.get("target_id"))
        print(f"  Min TTC: {e0['ttc']:.2f}s  (frame {e0['frame']}, tracks #{id_a} vs #{id_b})")
    if pet_nm:
        print(f"  Min PET: {pet_nm[0]['pet']:.2f}s  (tracks "
              f"#{pet_nm[0]['a_id']} vs #{pet_nm[0]['b_id']})")
    print(f"\nWrote {ttc_out}")
    print(f"Wrote {pet_out}")
    print(f"Wrote {report_out}")


if __name__ == "__main__":
    main()
