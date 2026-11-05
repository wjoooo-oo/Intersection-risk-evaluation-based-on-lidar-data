#!/usr/bin/env bash
# Full TTC / PET pipeline for session_c (outputs separate from session_a / session_b).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

OUT="${OUT:-$REPO/conflict_analysis/output/session_c}"
BAG="${BAG:-<BAG_DIR>/session_c}"
ROI_YAML="${ROI_YAML:-$REPO/detection/config/roi_session_c.yaml}"
DET_CFG="${DET_CFG:-$REPO/detection/config/detection_config_session_c.yaml}"
DANGER_TTC="${DANGER_TTC:-2.0}"
DANGER_PET="${DANGER_PET:-3.0}"
# Tight BEV crop for session_c (from config/roi_session_c.yaml bev + margin)
BEV_XMIN="${BEV_XMIN:--60}"
BEV_XMAX="${BEV_XMAX:-60}"
BEV_YMIN="${BEV_YMIN:--40}"
BEV_YMAX="${BEV_YMAX:-40}"
LOG_DIR="$OUT/logs"
mkdir -p "$OUT" "$OUT/viz" "$LOG_DIR"
LOG="$LOG_DIR/pipeline.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== session_c pipeline started $(date -Iseconds) ==="
echo "OUT=$OUT  BAG=$BAG"

echo "=== 1/7 record_tracks (full bag) ==="
python3 conflict_analysis/scripts/record_tracks.py \
  --bag "$BAG" \
  --detection-config "$DET_CFG" \
  --roi-yaml "$ROI_YAML" \
  --n-frames -1 \
  --include-class vehicle \
  --out "$OUT/tracks.jsonl"

echo "=== 2/7 evaluate_tta ==="
python3 conflict_analysis/scripts/evaluate_tta.py \
  --tracks "$OUT/tracks.jsonl" \
  --out-events "$OUT/tta_events.jsonl" \
  --out-summary "$OUT/tta_summary.json" \
  --danger-tta "$DANGER_TTC"

echo "=== 3/7 evaluate_ttc ==="
python3 conflict_analysis/scripts/evaluate_ttc.py \
  --tracks "$OUT/tracks.jsonl" \
  --out-events "$OUT/ttc_events.jsonl" \
  --out-summary "$OUT/ttc_summary.json" \
  --danger-ttc "$DANGER_TTC"

echo "=== 4/7 evaluate_pet ==="
python3 conflict_analysis/scripts/evaluate_pet.py \
  --tracks "$OUT/tracks.jsonl" \
  --out-events "$OUT/pet_events.jsonl" \
  --out-summary "$OUT/pet_summary.json" \
  --danger-pet "$DANGER_PET" \
  --no-straight-turn-only \
  --min-trajectory-length 2.0 \
  --min-time-overlap 1.5 \
  --stitch-fragments \
  --complete-trajectories \
  --max-interp-gap-s 3.0

echo "=== 5/7 summarize_nearmiss ==="
python3 conflict_analysis/scripts/summarize_nearmiss.py \
  --ttc-events "$OUT/ttc_events.jsonl" \
  --pet-events "$OUT/pet_events.jsonl" \
  --out-dir "$OUT" \
  --danger-ttc "$DANGER_TTC" \
  --danger-pet "$DANGER_PET"

echo "=== 6/7 plot_ttc_tta_summary ==="
python3 conflict_analysis/scripts/plot_ttc_tta_summary.py \
  --threshold "$DANGER_TTC" \
  --out "$OUT/viz/summary/ttc_tta_comparison.png" \
  --ttc-events "$OUT/ttc_events.jsonl" \
  --tta-events "$OUT/tta_events.jsonl"

echo "=== 7/7 visualize_conflicts ==="
# PET BEVs: pet_events/ — straight vs turn, ranked by crossing (not smallest PET)
python3 conflict_analysis/scripts/visualize_conflicts.py \
  --tracks "$OUT/tracks.jsonl" \
  --ttc-events "$OUT/ttc_events.jsonl" \
  --pet-events "$OUT/pet_events.jsonl" \
  --ttc-summary "$OUT/ttc_summary.json" \
  --pet-summary "$OUT/pet_summary.json" \
  --roi-yaml "$ROI_YAML" \
  --out-dir "$OUT/viz" \
  --max-bev-frames 60 \
  --viz-pet-max 0 \
  --pet-render-all \
  --danger-pet 3.0 \
  --bev-xmin "$BEV_XMIN" \
  --bev-xmax "$BEV_XMAX" \
  --bev-ymin "$BEV_YMIN" \
  --bev-ymax "$BEV_YMAX"

echo "Done. Report: $OUT/nearmiss_report.json"
echo "Viz: $OUT/viz/"
