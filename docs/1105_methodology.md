# 1105 Methodology Notes

## Goal / 目标

Use roadside LiDAR tracks to extract surrogate safety measures (TTC, PET) and
locate near-miss hotspots without waiting for crash records.

利用路侧激光雷达轨迹提取替代安全指标（TTC、PET），在不等待事故记录的情况下定位近碰撞热点。

## Perception stack / 感知流程

1. ROI polygon crop in the LiDAR frame  
2. Height filter + voxel downsample  
3. Grid local-min ground removal  
4. DBSCAN clustering → PCA OBB → size-based vehicle / pedestrian labels  
5. Background / static filter  
6. Constant-velocity Kalman tracker + Hungarian association + EMA box smoothing  

## Conflict metrics / 冲突指标

- **TTC**: TCPA/DCPA circle-contact model; critical if `TTC ≤ 2 s`  
- **PET**: proximity conflict zone (default radius ~6 m); critical if `PET ≤ 2 s`  
- **TTA**: looser converging-pair time metric (optional companion to TTC)

## Sites / 站点（已脱敏）

| Code in repo | Role |
|--------------|------|
| `session_a` / Site A | Signalized intersection (~30 min) |
| `session_b` / Site B | Signalized intersection (~30 min) |
| `session_c` / Site C | Railroad-crossing corridor (~30 min) |

Street names and county identifiers are omitted from docs and figure labels.
文档与图注中已隐去具体路名与县名。
