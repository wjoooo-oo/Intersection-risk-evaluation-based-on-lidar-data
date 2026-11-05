#!/usr/bin/env python3
"""
TTC Evaluation: Compare tracking-based TTC with ground truth.

Pipeline:
1. Load GT lead-follow pairs (with UUIDs and GT TTC)
2. Load original JSON labels to get UUID → position/type mapping
3. Load tracking results (per-frame) → track_id → position/type
4. Match GT UUIDs to tracking track_ids by spatial proximity
5. Build tracking trajectories (positions across frames)
6. Compute TTC from tracking data for each GT pair
7. Evaluate: binary classification (TTC exists?) + RMSE (TTC accuracy)
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate TTC from tracking results')
    parser.add_argument('--gt_pairs', type=str,
                        default='<DATASET_ROOT>/...')
    parser.add_argument('--label_dir', type=str,
                        default='<DATASET_ROOT>/...')
    parser.add_argument('--tracking_dir', type=str,
                        default='<DATASET_ROOT>/...')
    parser.add_argument('--output_dir', type=str,
                        default='<DATASET_ROOT>/...')
    parser.add_argument('--match_thresh', type=float, default=3.0,
                        help='Max BEV distance (meters) to match GT object to tracked object')
    parser.add_argument('--velocity_window', type=int, default=3,
                        help='Number of frames before/after for velocity estimation')
    return parser.parse_args()


def load_original_labels(label_dir, filename):
    """Load original TUMTraf JSON label file → UUID → object info."""
    filepath = os.path.join(label_dir, filename)
    with open(filepath) as f:
        data = json.load(f)

    frame = next(iter(data['openlabel']['frames'].values()))
    timestamp = frame['frame_properties']['timestamp']
    if isinstance(timestamp, str):
        from datetime import datetime
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            ts_sec = dt.timestamp()
        except Exception:
            ts_sec = 0.0
    else:
        ts_sec = float(timestamp)

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
            'x': x, 'y': y, 'z': z,
            'heading': heading,
            'length': length, 'width': width, 'height': height,
            'type': obj['type']
        }

    return objects, ts_sec


def load_tracking_frame(tracking_dir, timestamp_part):
    """Load tracking per-frame result file."""
    filename = f'{timestamp_part}_s110_lidar_ouster_south.txt'
    filepath = os.path.join(tracking_dir, filename)
    if not os.path.exists(filepath):
        return []

    tracked_objects = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 17:
                continue
            cls = parts[0]
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
            ry = float(parts[14])
            score = float(parts[15])
            track_id = int(parts[16])

            tracked_objects.append({
                'track_id': track_id,
                'class': cls,
                'x': x, 'y': y, 'z': z,
                'length': l, 'width': w, 'height': h,
                'heading': ry,
                'score': score,
            })
    return tracked_objects


def match_gt_to_tracking(gt_objects, tracked_objects, match_thresh=3.0):
    """
    Match GT UUIDs to tracking track_ids by BEV distance.
    Also considers type consistency.
    Returns: {uuid: track_id} mapping
    """
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
    used_tracks = set()

    for _ in range(min(len(uuid_list), len(tracked_objects))):
        min_dist = dist_matrix.min()
        if min_dist > match_thresh:
            break
        gi, ti = np.unravel_index(dist_matrix.argmin(), dist_matrix.shape)
        if gt_types[gi] == track_types[ti]:
            mapping[uuid_list[gi]] = track_ids[ti]
            used_tracks.add(ti)
            dist_matrix[gi, :] = 1e9
            dist_matrix[:, ti] = 1e9
        else:
            dist_matrix[gi, ti] = 1e9

    return mapping


def compute_velocity_from_trajectory(trajectory, frame_idx, window=3):
    """
    Compute velocity from tracking trajectory using ±window frames.
    trajectory: list of (frame_idx, x, y, timestamp)
    Returns: (speed, velocity_vector [vx, vy])
    """
    if len(trajectory) < 2:
        return 0.0, np.array([0.0, 0.0])

    current_pos = None
    for i, (fi, x, y, ts) in enumerate(trajectory):
        if fi == frame_idx:
            current_pos = i
            break

    if current_pos is None:
        return 0.0, np.array([0.0, 0.0])

    start_idx = max(0, current_pos - window)
    end_idx = min(len(trajectory) - 1, current_pos + window)

    if start_idx == end_idx:
        return 0.0, np.array([0.0, 0.0])

    _, x0, y0, ts0 = trajectory[start_idx]
    _, x1, y1, ts1 = trajectory[end_idx]

    dt = ts1 - ts0
    if dt < 1e-6:
        return 0.0, np.array([0.0, 0.0])

    vx = (x1 - x0) / dt
    vy = (y1 - y0) / dt
    speed = np.sqrt(vx ** 2 + vy ** 2)

    return speed, np.array([vx, vy])


def compute_ttc(ego_pos, lead_pos, ego_vel, lead_vel, ego_length, lead_length):
    """
    Compute TTC using the same formula as GT.
    TTC = gap_distance / closing_speed (if closing_speed > 0.1)
    """
    direction = np.array([lead_pos[0] - ego_pos[0], lead_pos[1] - ego_pos[1]])
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        return None, 0.0

    direction_norm = direction / dist

    ego_speed_long = np.dot(ego_vel, direction_norm)
    lead_speed_long = np.dot(lead_vel, direction_norm)
    closing_speed = ego_speed_long - lead_speed_long

    gap_distance = dist - (ego_length / 2.0) - (lead_length / 2.0)
    if gap_distance < 0:
        gap_distance = 0.0

    if closing_speed > 0.1:
        ttc = gap_distance / closing_speed
        if ttc < 0 or ttc > 100:
            ttc = None
    else:
        ttc = None

    return ttc, gap_distance


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("TTC Evaluation: Tracking vs Ground Truth")
    print("=" * 60)

    with open(args.gt_pairs) as f:
        gt_data = json.load(f)
    print(f"Loaded GT: {len(gt_data)} frames")

    # Phase 1: Load all frames, match UUIDs to track_ids, build trajectories
    print("\nPhase 1: Loading data and matching objects...")

    frame_uuid_to_type = {}
    frame_uuid_to_info = {}
    frame_tracking = {}
    frame_uuid_to_trackid = {}
    frame_timestamps = {}

    for frame in gt_data:
        frame_idx = frame['frame_idx']
        filename = frame['filename']
        ts_part = '_'.join(filename.split('_')[:2])

        gt_objects, ts_sec = load_original_labels(args.label_dir, filename)
        tracked_objects = load_tracking_frame(args.tracking_dir, ts_part)
        mapping = match_gt_to_tracking(gt_objects, tracked_objects, args.match_thresh)

        frame_uuid_to_info[frame_idx] = gt_objects
        frame_tracking[frame_idx] = {t['track_id']: t for t in tracked_objects}
        frame_uuid_to_trackid[frame_idx] = mapping
        frame_timestamps[frame_idx] = ts_sec

        for uid, obj in gt_objects.items():
            frame_uuid_to_type[uid] = obj['type']

    # Phase 2: Build tracking trajectories
    print("Phase 2: Building tracking trajectories...")

    track_trajectories = defaultdict(list)
    sorted_frame_idxs = sorted(frame_timestamps.keys())

    for frame_idx in sorted_frame_idxs:
        ts = frame_timestamps[frame_idx]
        for track_id, obj in frame_tracking[frame_idx].items():
            track_trajectories[track_id].append((frame_idx, obj['x'], obj['y'], ts))

    for tid in track_trajectories:
        track_trajectories[tid].sort(key=lambda x: x[0])

    print(f"  Built {len(track_trajectories)} tracking trajectories")

    # Phase 3: Compute TTC for each GT pair
    print("Phase 3: Computing TTC from tracking...")

    results = []
    match_stats = {'both_tracked': 0, 'ego_missing': 0, 'lead_missing': 0, 'both_missing': 0}

    for frame in gt_data:
        frame_idx = frame['frame_idx']
        mapping = frame_uuid_to_trackid[frame_idx]
        tracking = frame_tracking[frame_idx]

        for pair in frame['pairs']:
            ego_uid = pair['ego_id']
            lead_uid = pair['lead_id']
            gt_ttc = pair['ttc']

            ego_type = frame_uuid_to_type.get(ego_uid, 'UNKNOWN')
            lead_type = frame_uuid_to_type.get(lead_uid, 'UNKNOWN')

            ego_tid = mapping.get(ego_uid)
            lead_tid = mapping.get(lead_uid)

            pred_ttc = None
            pred_gap_distance = None

            if ego_tid is not None and lead_tid is not None:
                match_stats['both_tracked'] += 1

                ego_speed, ego_vel = compute_velocity_from_trajectory(
                    track_trajectories[ego_tid], frame_idx, args.velocity_window)
                lead_speed, lead_vel = compute_velocity_from_trajectory(
                    track_trajectories[lead_tid], frame_idx, args.velocity_window)

                ego_obj = tracking[ego_tid]
                lead_obj = tracking[lead_tid]

                ego_pos = np.array([ego_obj['x'], ego_obj['y']])
                lead_pos = np.array([lead_obj['x'], lead_obj['y']])

                pred_ttc, pred_gap_distance = compute_ttc(
                    ego_pos, lead_pos,
                    ego_vel, lead_vel,
                    ego_obj['length'], lead_obj['length']
                )
            else:
                ego_missing = ego_tid is None
                lead_missing = lead_tid is None
                if ego_missing and lead_missing:
                    match_stats['both_missing'] += 1
                elif ego_missing:
                    match_stats['ego_missing'] += 1
                else:
                    match_stats['lead_missing'] += 1

            results.append({
                'frame_idx': frame_idx,
                'ego_uid': ego_uid,
                'lead_uid': lead_uid,
                'ego_type': ego_type,
                'lead_type': lead_type,
                'gt_ttc': gt_ttc,
                'pred_ttc': pred_ttc,
                'gt_gap_distance': pair['gap_distance'],
                'pred_gap_distance': pred_gap_distance,
                'ego_tracked': ego_tid is not None,
                'lead_tracked': lead_tid is not None,
            })

    print(f"\n  Matching stats:")
    print(f"    Both tracked: {match_stats['both_tracked']}")
    print(f"    Ego missing:  {match_stats['ego_missing']}")
    print(f"    Lead missing: {match_stats['lead_missing']}")
    print(f"    Both missing: {match_stats['both_missing']}")
    total_pairs = len(results)
    print(f"    Total pairs:  {total_pairs}")

    # Phase 4: Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    ego_types_in_data = sorted(set(r['ego_type'] for r in results))

    def evaluate_group(group_results, label):
        """Evaluate binary classification and RMSE for a group of results."""
        tp = fp = tn = fn = 0
        gt_values = []
        pred_values = []

        for r in group_results:
            gt_exists = r['gt_ttc'] is not None
            pred_exists = r['pred_ttc'] is not None

            if gt_exists and pred_exists:
                tp += 1
                gt_values.append(r['gt_ttc'])
                pred_values.append(r['pred_ttc'])
            elif not gt_exists and not pred_exists:
                tn += 1
            elif pred_exists and not gt_exists:
                fp += 1
            else:
                fn += 1

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / max(total, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        rmse = None
        mae = None
        if gt_values:
            gt_arr = np.array(gt_values)
            pred_arr = np.array(pred_values)
            rmse = np.sqrt(np.mean((gt_arr - pred_arr) ** 2))
            mae = np.mean(np.abs(gt_arr - pred_arr))

        return {
            'label': label,
            'total': total,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'rmse': rmse,
            'mae': mae,
            'n_valid_ttc': len(gt_values),
        }

    all_evals = []

    # Overall
    overall = evaluate_group(results, 'OVERALL')
    all_evals.append(overall)

    # Per ego type
    for etype in ego_types_in_data:
        group = [r for r in results if r['ego_type'] == etype]
        if group:
            ev = evaluate_group(group, f'Ego={etype}')
            all_evals.append(ev)

    # Print results
    output_lines = []

    def log(line):
        print(line)
        output_lines.append(line)

    log("\n1. BINARY CLASSIFICATION: Does TTC exist?")
    log("   (TP=both exist, TN=both null, FP=pred exists but GT null, FN=pred null but GT exists)")
    log("")
    header = f"{'Group':<20} {'Total':>6} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}"
    log(header)
    log("-" * len(header))
    for ev in all_evals:
        line = (f"{ev['label']:<20} {ev['total']:>6} {ev['tp']:>5} {ev['fp']:>5} "
                f"{ev['tn']:>5} {ev['fn']:>5} {ev['accuracy']:>7.4f} "
                f"{ev['precision']:>7.4f} {ev['recall']:>7.4f} {ev['f1']:>7.4f}")
        log(line)

    log("\n2. TTC VALUE ACCURACY (when both GT and Pred TTC exist)")
    log("")
    header2 = f"{'Group':<20} {'N':>6} {'RMSE':>10} {'MAE':>10}"
    log(header2)
    log("-" * len(header2))
    for ev in all_evals:
        rmse_str = f"{ev['rmse']:.4f}" if ev['rmse'] is not None else "N/A"
        mae_str = f"{ev['mae']:.4f}" if ev['mae'] is not None else "N/A"
        line = f"{ev['label']:<20} {ev['n_valid_ttc']:>6} {rmse_str:>10} {mae_str:>10}"
        log(line)

    # Detailed per-pair-type analysis
    log("\n3. PER PAIR TYPE (Ego→Lead)")
    log("")
    pair_type_groups = defaultdict(list)
    for r in results:
        key = f"{r['ego_type']}->{r['lead_type']}"
        pair_type_groups[key].append(r)

    pair_evals = []
    for ptype in sorted(pair_type_groups.keys(), key=lambda k: -len(pair_type_groups[k])):
        group = pair_type_groups[ptype]
        ev = evaluate_group(group, ptype)
        pair_evals.append(ev)

    header3 = f"{'Pair Type':<20} {'N':>5} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} {'Acc':>7} {'RMSE':>10}"
    log(header3)
    log("-" * len(header3))
    for ev in pair_evals:
        rmse_str = f"{ev['rmse']:.4f}" if ev['rmse'] is not None else "N/A"
        line = (f"{ev['label']:<20} {ev['total']:>5} {ev['tp']:>4} {ev['fp']:>4} "
                f"{ev['tn']:>4} {ev['fn']:>4} {ev['accuracy']:>7.4f} {rmse_str:>10}")
        log(line)

    # Phase 5: Coverage analysis
    log("\n4. TRACKING COVERAGE")
    log("")
    both_tracked = sum(1 for r in results if r['ego_tracked'] and r['lead_tracked'])
    coverage = both_tracked / max(total_pairs, 1)
    log(f"   Pairs with both objects tracked: {both_tracked}/{total_pairs} ({coverage:.1%})")
    for etype in ego_types_in_data:
        group = [r for r in results if r['ego_type'] == etype]
        tracked = sum(1 for r in group if r['ego_tracked'] and r['lead_tracked'])
        cov = tracked / max(len(group), 1)
        log(f"   {etype}: {tracked}/{len(group)} ({cov:.1%})")

    # Save evaluation
    eval_file = os.path.join(args.output_dir, 'ttc_evaluation.txt')
    with open(eval_file, 'w') as f:
        f.write('\n'.join(output_lines) + '\n')
    print(f"\nEvaluation saved to: {eval_file}")

    # Save detailed results as JSON
    detail_file = os.path.join(args.output_dir, 'ttc_evaluation_detail.json')
    json_results = []
    for r in results:
        jr = {k: v for k, v in r.items()}
        jr['gt_ttc'] = round(r['gt_ttc'], 4) if r['gt_ttc'] is not None else None
        jr['pred_ttc'] = round(r['pred_ttc'], 4) if r['pred_ttc'] is not None else None
        jr['pred_gap_distance'] = round(r['pred_gap_distance'], 4) if r['pred_gap_distance'] is not None else None
        json_results.append(jr)

    with open(detail_file, 'w') as f:
        json.dump({
            'summary': {ev['label']: {
                'total': ev['total'],
                'binary': {'tp': ev['tp'], 'fp': ev['fp'], 'tn': ev['tn'], 'fn': ev['fn'],
                           'accuracy': ev['accuracy'], 'precision': ev['precision'],
                           'recall': ev['recall'], 'f1': ev['f1']},
                'regression': {'n': ev['n_valid_ttc'], 'rmse': ev['rmse'], 'mae': ev['mae']},
            } for ev in all_evals},
            'pair_types': {ev['label']: {
                'total': ev['total'],
                'binary': {'tp': ev['tp'], 'fp': ev['fp'], 'tn': ev['tn'], 'fn': ev['fn'],
                           'accuracy': ev['accuracy']},
                'regression': {'n': ev['n_valid_ttc'], 'rmse': ev['rmse'], 'mae': ev['mae']},
            } for ev in pair_evals},
            'results': json_results,
        }, f, indent=2)
    print(f"Detailed results saved to: {detail_file}")


if __name__ == '__main__':
    main()
