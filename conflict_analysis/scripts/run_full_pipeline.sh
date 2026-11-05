#!/usr/bin/env bash
# Full-bag intersection safety pipeline: tracks -> TTC -> PET -> near-miss -> viz
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
OUT="$REPO/conflict_analysis/output"
mkdir -p "$OUT"

BAG="${BAG:-<BAG_DIR>/session_a}"
DANGER_TTC="${DANGER_TTC:-2.0}"
DANGER_PET="${DANGER_PET:-3.0}"

echo "=== 1/5 record_tracks (full bag) ==="
python3 conflict_analysis/scripts/record_tracks.py \
  --bag "$BAG" \
  --n-frames -1 \
  --out "$OUT/tracks.jsonl"

echo "=== 2/7 evaluate_tta (any converging pair) ==="
python3 conflict_analysis/scripts/evaluate_tta.py \
  --tracks "$OUT/tracks.jsonl" \
  --danger-tta "$DANGER_TTC"

echo "=== 3/7 evaluate_ttc (car-following / lead-follow only) ==="
python3 conflict_analysis/scripts/evaluate_ttc.py \
  --tracks "$OUT/tracks.jsonl" \
  --danger-ttc "$DANGER_TTC"

echo "=== 4/7 evaluate_pet (proximity conflict zone) ==="
python3 conflict_analysis/scripts/evaluate_pet.py \
  --tracks "$OUT/tracks.jsonl" \
  --danger-pet "$DANGER_PET"

echo "=== 5/7 summarize_nearmiss ==="
python3 conflict_analysis/scripts/summarize_nearmiss.py \
  --danger-ttc "$DANGER_TTC" \
  --danger-pet "$DANGER_PET"

echo "=== 6/7 plot_ttc_tta_summary ==="
python3 conflict_analysis/scripts/plot_ttc_tta_summary.py --threshold "$DANGER_TTC"

echo "=== 7/7 visualize_conflicts ==="
python3 conflict_analysis/scripts/visualize_conflicts.py \
  --tracks "$OUT/tracks.jsonl" \
  --max-bev-frames 60 \
  --top-k-pet 30

echo "Done. See $OUT/nearmiss_report.json and $OUT/viz/summary/ttc_tta_comparison.png"
