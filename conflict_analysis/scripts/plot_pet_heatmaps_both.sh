#!/usr/bin/env bash
# PET conflict heatmaps for session_a and session_b (separate output paths).
set -eo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

python3 conflict_analysis/scripts/plot_pet_heatmap.py \
  --pet-events conflict_analysis/output/pet_events.jsonl \
  --roi-yaml detection/config/roi_session_a.yaml \
  --out conflict_analysis/output/viz/pet_conflict_heatmap_session_a.png \
  --label "session_a" \
  --bev-xmin -20 --bev-xmax 45 --bev-ymin -30 --bev-ymax 40 \
  --resolution 0.5 \
  --blur-sigma-m 4.0 \
  --transparent

python3 conflict_analysis/scripts/plot_pet_heatmap.py \
  --pet-events conflict_analysis/output/session_b/pet_events.jsonl \
  --roi-yaml detection/config/roi_session_b.yaml \
  --out conflict_analysis/output/session_b/viz/pet_conflict_heatmap_session_b.png \
  --label "session_b" \
  --bev-xmin -150 --bev-xmax 100 --bev-ymin -50 --bev-ymax 100 \
  --resolution 2.0 \
  --blur-sigma-m 8.0 \
  --transparent

echo "Done."
