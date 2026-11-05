#!/usr/bin/env python3
"""
TTC Evaluation Visualization:
1. Summary plots (scatter, confusion matrix, per-class bars, error histogram)
2. Per-frame BEV images comparing GT TTC vs predicted TTC for each pair

Color convention for pair connections:
  TP (both have TTC): green
  TN (both null TTC): gray
  FP (pred has TTC but GT null): orange
  FN (GT has TTC but pred null): red
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from collections import defaultdict
import argparse
import warnings
warnings.filterwarnings('ignore')


CLASS_COLORS = {
    'CAR': '#2196F3',
    'TRUCK': '#4CAF50',
    'BUS': '#FF9800',
    'VAN': '#9C27B0',
    'TRAILER': '#00BCD4',
    'PEDESTRIAN': '#F44336',
    'BICYCLE': '#FFEB3B',
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--detail_json', default=None,
                   help='Path to ttc_evaluation_detail.json (auto-generated from evaluate_ttc.py)')
    p.add_argument('--gt_pairs',
                   default='<DATASET_ROOT>/...')
    p.add_argument('--label_dir',
                   default='<DATASET_ROOT>/...')
    p.add_argument('--tracking_dir',
                   default='<DATASET_ROOT>/...')
    p.add_argument('--map_path',
                   default='<DATASET_ROOT>/...')
    p.add_argument('--output_dir',
                   default='<DATASET_ROOT>/...')
    p.add_argument('--range_limit', type=float, default=100.0)
    p.add_argument('--match_thresh', type=float, default=3.0)
    p.add_argument('--velocity_window', type=int, default=3)
    p.add_argument('--skip_bev', action='store_true', help='Skip per-frame BEV images')
    return p.parse_args()


def load_map(map_path):
    with open(map_path) as f:
        data = json.load(f)
    center = np.array(data['metadata']['center'])
    lanes = [np.array(lane['points']) for lane in data['lanes']]
    return center, lanes


def load_original_labels(label_dir, filename):
    filepath = os.path.join(label_dir, filename)
    with open(filepath) as f:
        data = json.load(f)
    frame = next(iter(data['openlabel']['frames'].values()))
    objects = {}
    for obj_id, obj_data in frame['objects'].items():
        obj = obj_data['object_data']
        cuboid = obj['cuboid']['val']
        x, y, z = cuboid[0], cuboid[1], cuboid[2]
        qx, qy, qz, qw = cuboid[3], cuboid[4], cuboid[5], cuboid[6]
        length, width, height = cuboid[7], cuboid[8], cuboid[9]
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        heading = np.arctan2(siny_cosp, cosy_cosp)
        objects[obj_id] = {
            'x': x, 'y': y, 'z': z, 'heading': heading,
            'length': length, 'width': width, 'height': height,
            'type': obj['type']
        }
    return objects


def load_tracking_frame(tracking_dir, timestamp_part):
    filename = f'{timestamp_part}_s110_lidar_ouster_south.txt'
    filepath = os.path.join(tracking_dir, filename)
    if not os.path.exists(filepath):
        return []
    tracked = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 17:
                continue
            tracked.append({
                'track_id': int(parts[16]),
                'class': parts[0],
                'x': float(parts[11]), 'y': float(parts[12]),
                'length': float(parts[10]), 'width': float(parts[9]),
                'heading': float(parts[14]),
                'score': float(parts[15]),
            })
    return tracked


def match_gt_to_tracking(gt_objects, tracked_objects, match_thresh=3.0):
    if not gt_objects or not tracked_objects:
        return {}
    uuid_list = list(gt_objects.keys())
    gt_positions = np.array([[gt_objects[uid]['x'], gt_objects[uid]['y']] for uid in uuid_list])
    gt_types = [gt_objects[uid]['type'] for uid in uuid_list]
    track_positions = np.array([[t['x'], t['y']] for t in tracked_objects])
    track_types = [t['class'] for t in tracked_objects]
    track_ids = [t['track_id'] for t in tracked_objects]

    dist_matrix = np.sqrt(
        (gt_positions[:, 0:1] - track_positions[:, 0:1].T) ** 2 +
        (gt_positions[:, 1:2] - track_positions[:, 1:2].T) ** 2
    )
    mapping = {}
    for _ in range(min(len(uuid_list), len(tracked_objects))):
        min_dist = dist_matrix.min()
        if min_dist > match_thresh:
            break
        gi, ti = np.unravel_index(dist_matrix.argmin(), dist_matrix.shape)
        if gt_types[gi] == track_types[ti]:
            mapping[uuid_list[gi]] = track_ids[ti]
            dist_matrix[gi, :] = 1e9
            dist_matrix[:, ti] = 1e9
        else:
            dist_matrix[gi, ti] = 1e9
    return mapping


def compute_velocity_from_trajectory(trajectory, frame_idx, window=3):
    if len(trajectory) < 2:
        return np.array([0.0, 0.0])
    current_pos = None
    for i, (fi, x, y, ts) in enumerate(trajectory):
        if fi == frame_idx:
            current_pos = i
            break
    if current_pos is None:
        return np.array([0.0, 0.0])
    start_idx = max(0, current_pos - window)
    end_idx = min(len(trajectory) - 1, current_pos + window)
    if start_idx == end_idx:
        return np.array([0.0, 0.0])
    _, x0, y0, ts0 = trajectory[start_idx]
    _, x1, y1, ts1 = trajectory[end_idx]
    dt = ts1 - ts0
    if dt < 1e-6:
        return np.array([0.0, 0.0])
    return np.array([(x1 - x0) / dt, (y1 - y0) / dt])


def compute_ttc(ego_pos, lead_pos, ego_vel, lead_vel, ego_length, lead_length):
    direction = lead_pos - ego_pos
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        return None
    direction_norm = direction / dist
    closing_speed = np.dot(ego_vel, direction_norm) - np.dot(lead_vel, direction_norm)
    gap = max(0.0, dist - ego_length / 2.0 - lead_length / 2.0)
    if closing_speed > 0.1:
        ttc = gap / closing_speed
        return ttc if 0 <= ttc <= 100 else None
    return None


def get_bbox_corners(x, y, length, width, heading):
    half_l, half_w = length / 2.0, width / 2.0
    corners = np.array([[half_l, half_w], [half_l, -half_w],
                        [-half_l, -half_w], [-half_l, half_w]])
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    return corners @ R.T + np.array([x, y])


# ---- Summary Plots ----

def plot_summary(detail_data, output_dir):
    results = detail_data['results']
    summary = detail_data['summary']

    summary_dir = os.path.join(output_dir, 'summary')
    os.makedirs(summary_dir, exist_ok=True)

    # 1. Scatter: GT TTC vs Pred TTC
    fig, ax = plt.subplots(figsize=(10, 10))
    ego_types = sorted(set(r['ego_type'] for r in results if r['gt_ttc'] is not None and r['pred_ttc'] is not None))

    for etype in ego_types:
        gt_vals = [r['gt_ttc'] for r in results if r['gt_ttc'] is not None and r['pred_ttc'] is not None and r['ego_type'] == etype]
        pred_vals = [r['pred_ttc'] for r in results if r['gt_ttc'] is not None and r['pred_ttc'] is not None and r['ego_type'] == etype]
        color = CLASS_COLORS.get(etype, '#888888')
        ax.scatter(gt_vals, pred_vals, c=color, label=f'{etype} ({len(gt_vals)})',
                   alpha=0.6, s=40, edgecolors='black', linewidth=0.3)

    max_val = max(
        max((r['gt_ttc'] for r in results if r['gt_ttc'] is not None and r['pred_ttc'] is not None), default=1),
        max((r['pred_ttc'] for r in results if r['gt_ttc'] is not None and r['pred_ttc'] is not None), default=1)
    )
    ax.plot([0, max_val * 1.1], [0, max_val * 1.1], 'k--', alpha=0.4, label='Perfect prediction')
    ax.set_xlabel('Ground Truth TTC (s)', fontsize=12)
    ax.set_ylabel('Predicted TTC (s)', fontsize=12)
    ax.set_title('TTC: Ground Truth vs Predicted (TP cases)', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_val * 1.1)
    ax.set_ylim(0, max_val * 1.1)
    plt.savefig(os.path.join(summary_dir, 'scatter_gt_vs_pred.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    overall = summary.get('OVERALL', {})
    binary = overall.get('binary', {})
    cm_data = np.array([[binary.get('tp', 0), binary.get('fn', 0)],
                        [binary.get('fp', 0), binary.get('tn', 0)]])

    ax = axes[0]
    im = ax.imshow(cm_data, cmap='Blues', aspect='auto')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_data[i, j]), ha='center', va='center',
                    fontsize=18, fontweight='bold',
                    color='white' if cm_data[i, j] > cm_data.max() / 2 else 'black')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred: Exists', 'Pred: Null'], fontsize=11)
    ax.set_yticklabels(['GT: Exists', 'GT: Null'], fontsize=11)
    ax.set_title(f'Binary Classification (Overall)\n'
                 f'Acc={binary.get("accuracy", 0):.3f}  F1={binary.get("f1", 0):.3f}', fontsize=12)
    plt.colorbar(im, ax=ax)

    # Per-class accuracy + F1 bar chart
    ax = axes[1]
    class_labels = [k for k in summary.keys() if k.startswith('Ego=')]
    class_names = [k.replace('Ego=', '') for k in class_labels]
    accs = [summary[k]['binary']['accuracy'] for k in class_labels]
    f1s = [summary[k]['binary']['f1'] for k in class_labels]

    x_pos = np.arange(len(class_names))
    width = 0.35
    bars1 = ax.bar(x_pos - width / 2, accs, width, label='Accuracy',
                   color=[CLASS_COLORS.get(c, '#888') for c in class_names], alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x_pos + width / 2, f1s, width, label='F1 Score',
                   color=[CLASS_COLORS.get(c, '#888') for c in class_names], alpha=0.4, edgecolor='black', hatch='//')

    for bar, val in zip(bars1, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Binary Classification: Per Ego Class', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'confusion_and_classification.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. RMSE per class
    fig, ax = plt.subplots(figsize=(10, 6))
    rmse_classes = []
    rmse_vals = []
    rmse_n = []
    for k in class_labels:
        reg = summary[k]['regression']
        if reg['rmse'] is not None:
            rmse_classes.append(k.replace('Ego=', ''))
            rmse_vals.append(reg['rmse'])
            rmse_n.append(reg['n'])

    if rmse_classes:
        bars = ax.bar(rmse_classes, rmse_vals,
                      color=[CLASS_COLORS.get(c, '#888') for c in rmse_classes],
                      alpha=0.8, edgecolor='black')
        for bar, val, n in zip(bars, rmse_vals, rmse_n):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f'{val:.2f}\n(n={n})', ha='center', va='bottom', fontsize=9)

        overall_rmse = summary.get('OVERALL', {}).get('regression', {}).get('rmse')
        if overall_rmse is not None:
            ax.axhline(y=overall_rmse, color='red', linestyle='--', alpha=0.6,
                       label=f'Overall RMSE = {overall_rmse:.2f}')
            ax.legend(fontsize=10)

    ax.set_ylabel('RMSE (seconds)', fontsize=11)
    ax.set_title('TTC RMSE by Ego Vehicle Class (TP cases only)', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(summary_dir, 'rmse_per_class.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Error distribution histogram
    gt_vals = [r['gt_ttc'] for r in results if r['gt_ttc'] is not None and r['pred_ttc'] is not None]
    pred_vals = [r['pred_ttc'] for r in results if r['gt_ttc'] is not None and r['pred_ttc'] is not None]
    if gt_vals:
        errors = np.array(pred_vals) - np.array(gt_vals)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        ax = axes[0]
        ax.hist(errors, bins=40, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.6)
        ax.axvline(x=np.mean(errors), color='orange', linestyle='-', alpha=0.8,
                   label=f'Mean = {np.mean(errors):.2f}s')
        ax.set_xlabel('Prediction Error (Pred - GT) [seconds]', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('TTC Error Distribution', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        ax = axes[1]
        abs_errors = np.abs(errors)
        ax.hist(abs_errors, bins=40, color='#FF7043', alpha=0.7, edgecolor='black')
        percentiles = [50, 75, 90]
        for pct in percentiles:
            val = np.percentile(abs_errors, pct)
            ax.axvline(x=val, linestyle='--', alpha=0.6,
                       label=f'P{pct} = {val:.2f}s')
        ax.set_xlabel('Absolute Error [seconds]', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('TTC Absolute Error Distribution', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, 'error_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # 5. Coverage bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    coverage_classes = []
    coverage_vals = []
    coverage_total = []
    for k in class_labels:
        etype = k.replace('Ego=', '')
        group = [r for r in results if r['ego_type'] == etype]
        tracked = sum(1 for r in group if r['ego_tracked'] and r['lead_tracked'])
        coverage_classes.append(etype)
        coverage_vals.append(tracked / max(len(group), 1) * 100)
        coverage_total.append(len(group))

    bars = ax.bar(coverage_classes, coverage_vals,
                  color=[CLASS_COLORS.get(c, '#888') for c in coverage_classes],
                  alpha=0.8, edgecolor='black')
    for bar, val, n in zip(bars, coverage_vals, coverage_total):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Tracking Coverage (%)', fontsize=11)
    ax.set_title('Pair Tracking Coverage by Ego Class', fontsize=13)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(summary_dir, 'tracking_coverage.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Summary plots saved to: {summary_dir}")


# ---- Per-Frame BEV Visualization ----

def visualize_frame_bev(frame_data, gt_objects, gt_pairs, pair_results,
                        center, lanes, range_limit, frame_idx, output_path):
    """
    BEV visualization for one frame showing GT TTC vs predicted TTC.
    """
    fig, ax = plt.subplots(figsize=(16, 16))

    for lane in lanes:
        ax.plot(lane[:, 0], lane[:, 1], 'steelblue', linewidth=0.8, alpha=0.5)

    pair_uids = set()
    for pair in gt_pairs:
        pair_uids.add(pair['ego_id'])
        pair_uids.add(pair['lead_id'])

    for obj_id, obj in gt_objects.items():
        x, y = obj['x'], obj['y']
        heading = obj['heading']
        length, width = obj['length'], obj['width']

        is_in_pair = obj_id in pair_uids
        color = CLASS_COLORS.get(obj['type'], '#888888') if is_in_pair else 'lightgray'
        alpha = 0.7 if is_in_pair else 0.25
        lw = 1.5 if is_in_pair else 0.5

        corners = get_bbox_corners(x, y, length, width, heading)
        rect = Polygon(corners, fill=True, facecolor=color,
                       edgecolor='black', alpha=alpha, linewidth=lw)
        ax.add_patch(rect)

        arrow_len = length * 0.4
        dx, dy = arrow_len * np.cos(heading), arrow_len * np.sin(heading)
        ax.arrow(x, y, dx, dy, head_width=0.4, head_length=0.2,
                 fc='black', ec='black', alpha=0.3)

        if is_in_pair:
            ax.text(x, y + width / 2 + 0.6, obj['type'],
                    fontsize=4.5, ha='center', va='bottom', color='black', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor=color, alpha=0.7,
                              edgecolor='black', linewidth=0.4))

    tp_count = fp_count = tn_count = fn_count = 0

    for i, pair in enumerate(gt_pairs):
        ego_id, lead_id = pair['ego_id'], pair['lead_id']
        ego_obj = gt_objects.get(ego_id)
        lead_obj = gt_objects.get(lead_id)

        if not ego_obj or not lead_obj:
            continue

        gt_ttc = pair.get('ttc')

        pr = pair_results[i] if i < len(pair_results) else None
        pred_ttc = pr['pred_ttc'] if pr else None

        gt_exists = gt_ttc is not None
        pred_exists = pred_ttc is not None

        if gt_exists and pred_exists:
            line_color = '#4CAF50'
            status = 'TP'
            tp_count += 1
        elif not gt_exists and not pred_exists:
            line_color = '#BDBDBD'
            status = 'TN'
            tn_count += 1
        elif pred_exists and not gt_exists:
            line_color = '#FF9800'
            status = 'FP'
            fp_count += 1
        else:
            line_color = '#F44336'
            status = 'FN'
            fn_count += 1

        ax.plot([ego_obj['x'], lead_obj['x']],
                [ego_obj['y'], lead_obj['y']],
                color=line_color, linewidth=1.5 if status != 'TN' else 0.8,
                alpha=0.7 if status != 'TN' else 0.3, linestyle='--')

        mid_x = (ego_obj['x'] + lead_obj['x']) / 2
        mid_y = (ego_obj['y'] + lead_obj['y']) / 2

        gt_str = f'{gt_ttc:.1f}s' if gt_ttc is not None else 'null'
        pred_str = f'{pred_ttc:.1f}s' if pred_ttc is not None else 'null'
        gap_str = f'{pair["gap_distance"]:.1f}m'

        label_text = f'GT:{gt_str}\nPred:{pred_str}\n{gap_str}'

        if status == 'TP' and gt_ttc is not None and pred_ttc is not None:
            err = abs(pred_ttc - gt_ttc)
            if err < 2.0:
                bbox_color = '#C8E6C9'
                text_color = 'black'
            elif err < 5.0:
                bbox_color = '#FFF9C4'
                text_color = 'black'
            else:
                bbox_color = '#FFCDD2'
                text_color = 'black'
        elif status == 'FP':
            bbox_color = '#FFE0B2'
            text_color = 'black'
        elif status == 'FN':
            bbox_color = '#FFCDD2'
            text_color = 'black'
        else:
            bbox_color = '#F5F5F5'
            text_color = '#666666'

        if status != 'TN':
            ax.text(mid_x + 0.5, mid_y + 0.5, label_text, fontsize=3.5,
                    ha='center', va='center', color=text_color,
                    bbox=dict(boxstyle='round,pad=0.1', facecolor=bbox_color,
                              alpha=0.85, edgecolor=line_color, linewidth=0.5))

    legend_elements = [
        Line2D([0], [0], color='#4CAF50', linewidth=2, linestyle='--', label=f'TP: both have TTC ({tp_count})'),
        Line2D([0], [0], color='#BDBDBD', linewidth=2, linestyle='--', label=f'TN: both null ({tn_count})'),
        Line2D([0], [0], color='#FF9800', linewidth=2, linestyle='--', label=f'FP: pred only ({fp_count})'),
        Line2D([0], [0], color='#F44336', linewidth=2, linestyle='--', label=f'FN: GT only ({fn_count})'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=7, framealpha=0.9)

    cx, cy = center
    ax.set_xlim(cx - range_limit, cx + range_limit)
    ax.set_ylim(cy - range_limit, cy + range_limit)
    ax.set_xlabel('X (m)', fontsize=9)
    ax.set_ylabel('Y (m)', fontsize=9)
    ax.set_title(
        f'Frame {frame_idx:04d} | TTC Evaluation\n'
        f'TP={tp_count} TN={tn_count} FP={fp_count} FN={fn_count} | '
        f'Total {len(gt_pairs)} pairs',
        fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def run_full_pipeline(args):
    """Run TTC computation + evaluation + visualization in one pass."""
    from datetime import datetime

    center, lanes = load_map(args.map_path)
    print(f"Map loaded: center=({center[0]:.1f}, {center[1]:.1f}), {len(lanes)} lanes")

    with open(args.gt_pairs) as f:
        gt_data = json.load(f)
    print(f"Loaded GT: {len(gt_data)} frames, {sum(len(f['pairs']) for f in gt_data)} pairs")

    print("\nPhase 1: Loading labels and tracking data, matching objects...")
    frame_gt_objects = {}
    frame_tracking = {}
    frame_uuid_to_trackid = {}
    frame_timestamps = {}
    frame_uuid_to_type = {}

    for frame in gt_data:
        frame_idx = frame['frame_idx']
        filename = frame['filename']
        ts_part = '_'.join(filename.split('_')[:2])

        gt_objects = load_original_labels(args.label_dir, filename)
        tracked_objects = load_tracking_frame(args.tracking_dir, ts_part)
        mapping = match_gt_to_tracking(gt_objects, tracked_objects, args.match_thresh)

        frame_gt_objects[frame_idx] = gt_objects
        frame_tracking[frame_idx] = {t['track_id']: t for t in tracked_objects}
        frame_uuid_to_trackid[frame_idx] = mapping

        fp = os.path.join(args.label_dir, filename)
        with open(fp) as f:
            d = json.load(f)
        fv = next(iter(d['openlabel']['frames'].values()))
        ts = fv['frame_properties']['timestamp']
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                frame_timestamps[frame_idx] = dt.timestamp()
            except Exception:
                frame_timestamps[frame_idx] = 0.0
        else:
            frame_timestamps[frame_idx] = float(ts)

        for uid, obj in gt_objects.items():
            frame_uuid_to_type[uid] = obj['type']

    print("Phase 2: Building tracking trajectories...")
    track_trajectories = defaultdict(list)
    for frame_idx in sorted(frame_timestamps.keys()):
        ts = frame_timestamps[frame_idx]
        for track_id, obj in frame_tracking[frame_idx].items():
            track_trajectories[track_id].append((frame_idx, obj['x'], obj['y'], ts))
    for tid in track_trajectories:
        track_trajectories[tid].sort(key=lambda x: x[0])
    print(f"  Built {len(track_trajectories)} trajectories")

    print("Phase 3: Computing TTC + visualization...")
    bev_dir = os.path.join(args.output_dir, 'bev')
    os.makedirs(bev_dir, exist_ok=True)

    all_results = []
    match_stats = defaultdict(int)

    for fi, frame in enumerate(gt_data):
        frame_idx = frame['frame_idx']
        mapping = frame_uuid_to_trackid[frame_idx]
        tracking = frame_tracking[frame_idx]
        gt_objects = frame_gt_objects[frame_idx]

        pair_results = []
        for pair in frame['pairs']:
            ego_uid = pair['ego_id']
            lead_uid = pair['lead_id']
            gt_ttc = pair['ttc']
            ego_type = frame_uuid_to_type.get(ego_uid, 'UNKNOWN')
            lead_type = frame_uuid_to_type.get(lead_uid, 'UNKNOWN')
            ego_tid = mapping.get(ego_uid)
            lead_tid = mapping.get(lead_uid)

            pred_ttc = None
            if ego_tid is not None and lead_tid is not None:
                match_stats['both_tracked'] += 1
                ego_vel = compute_velocity_from_trajectory(
                    track_trajectories[ego_tid], frame_idx, args.velocity_window)
                lead_vel = compute_velocity_from_trajectory(
                    track_trajectories[lead_tid], frame_idx, args.velocity_window)
                ego_obj = tracking[ego_tid]
                lead_obj = tracking[lead_tid]
                pred_ttc = compute_ttc(
                    np.array([ego_obj['x'], ego_obj['y']]),
                    np.array([lead_obj['x'], lead_obj['y']]),
                    ego_vel, lead_vel, ego_obj['length'], lead_obj['length'])
            else:
                if ego_tid is None and lead_tid is None:
                    match_stats['both_missing'] += 1
                elif ego_tid is None:
                    match_stats['ego_missing'] += 1
                else:
                    match_stats['lead_missing'] += 1

            pr = {
                'frame_idx': frame_idx,
                'ego_type': ego_type, 'lead_type': lead_type,
                'gt_ttc': gt_ttc, 'pred_ttc': pred_ttc,
                'ego_tracked': ego_tid is not None, 'lead_tracked': lead_tid is not None,
            }
            pair_results.append(pr)
            all_results.append(pr)

        if not args.skip_bev:
            ts_part = '_'.join(frame['filename'].split('_')[:2])
            out_path = os.path.join(bev_dir, f'frame_{fi:04d}_{ts_part}.png')
            visualize_frame_bev(frame, gt_objects, frame['pairs'], pair_results,
                                center, lanes, args.range_limit, fi, out_path)

        if (fi + 1) % 20 == 0 or fi == 0:
            print(f"  [{fi + 1}/{len(gt_data)}] frame_idx={frame_idx}")

    if not args.skip_bev:
        print(f"\nBEV images saved to: {bev_dir}")

    print(f"\nMatch stats: {dict(match_stats)}")

    ego_types_in_data = sorted(set(r['ego_type'] for r in all_results))

    def evaluate_group(group, label):
        tp = fp = tn = fn = 0
        gt_v, pred_v = [], []
        for r in group:
            ge = r['gt_ttc'] is not None
            pe = r['pred_ttc'] is not None
            if ge and pe:
                tp += 1; gt_v.append(r['gt_ttc']); pred_v.append(r['pred_ttc'])
            elif not ge and not pe:
                tn += 1
            elif pe and not ge:
                fp += 1
            else:
                fn += 1
        total = tp + fp + tn + fn
        acc = (tp + tn) / max(total, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        rmse = float(np.sqrt(np.mean((np.array(gt_v) - np.array(pred_v)) ** 2))) if gt_v else None
        mae = float(np.mean(np.abs(np.array(gt_v) - np.array(pred_v)))) if gt_v else None
        return {'label': label, 'total': total, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
                'rmse': rmse, 'mae': mae, 'n_valid_ttc': len(gt_v)}

    summary_evals = {'OVERALL': evaluate_group(all_results, 'OVERALL')}
    for et in ego_types_in_data:
        summary_evals[f'Ego={et}'] = evaluate_group(
            [r for r in all_results if r['ego_type'] == et], f'Ego={et}')

    pair_type_groups = defaultdict(list)
    for r in all_results:
        pair_type_groups[f"{r['ego_type']}->{r['lead_type']}"].append(r)
    pair_evals = {}
    for pt in sorted(pair_type_groups.keys(), key=lambda k: -len(pair_type_groups[k])):
        pair_evals[pt] = evaluate_group(pair_type_groups[pt], pt)

    detail_data = {
        'summary': {k: {
            'total': v['total'],
            'binary': {'tp': v['tp'], 'fp': v['fp'], 'tn': v['tn'], 'fn': v['fn'],
                       'accuracy': v['accuracy'], 'precision': v['precision'],
                       'recall': v['recall'], 'f1': v['f1']},
            'regression': {'n': v['n_valid_ttc'], 'rmse': v['rmse'], 'mae': v['mae']},
        } for k, v in summary_evals.items()},
        'pair_types': {k: {
            'total': v['total'],
            'binary': {'tp': v['tp'], 'fp': v['fp'], 'tn': v['tn'], 'fn': v['fn'],
                       'accuracy': v['accuracy']},
            'regression': {'n': v['n_valid_ttc'], 'rmse': v['rmse'], 'mae': v['mae']},
        } for k, v in pair_evals.items()},
        'results': [{
            'frame_idx': r['frame_idx'],
            'ego_type': r['ego_type'], 'lead_type': r['lead_type'],
            'gt_ttc': round(r['gt_ttc'], 4) if r['gt_ttc'] is not None else None,
            'pred_ttc': round(r['pred_ttc'], 4) if r['pred_ttc'] is not None else None,
            'ego_tracked': r['ego_tracked'], 'lead_tracked': r['lead_tracked'],
        } for r in all_results],
    }

    plot_summary(detail_data, args.output_dir)
    print("Done!")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.detail_json and os.path.exists(args.detail_json):
        with open(args.detail_json) as f:
            detail_data = json.load(f)
        print("Loaded existing detail JSON, generating summary plots only...")
        plot_summary(detail_data, args.output_dir)
        if not args.skip_bev:
            print("Note: BEV images require full pipeline. Running full pipeline...")
            run_full_pipeline(args)
    else:
        run_full_pipeline(args)


if __name__ == '__main__':
    main()
