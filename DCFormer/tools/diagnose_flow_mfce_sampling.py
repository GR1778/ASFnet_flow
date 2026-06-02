import argparse
import json
import math
import pickle
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ACTION_NAMES = [
    "Directions-1",
    "Directions-2",
    "Discussion-1",
    "Discussion-2",
    "Eating-1",
    "Eating-2",
    "Greeting-1",
    "Greeting-2",
    "Phoning-1",
    "Phoning-2",
    "Posing-1",
    "Posing-2",
    "Purchases-1",
    "Purchases-2",
    "Sitting-1",
    "Sitting-2",
    "SittingDown-1",
    "SittingDown-2",
    "Smoking-1",
    "Smoking-2",
    "TakingPhoto-1",
    "TakingPhoto-2",
    "Waiting-1",
    "Waiting-2",
    "Walking-1",
    "Walking-2",
    "WalkingDog-1",
    "WalkingDog-2",
    "WalkingTogether-1",
    "WalkingTogether-2",
]

JOINT_NAMES = [
    "Pelvis",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "Spine",
    "Thorax",
    "Neck",
    "Head",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RShoulder",
    "RElbow",
    "RWrist",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose why DCE/AMS-style multi-sampling may hurt optical-flow tokens. "
            "It compares center flow sampling with simulated DCE initial sampling "
            "against previous-frame GT joint displacement in current-crop coordinates."
        )
    )
    parser.add_argument("--labels", default="data/h36m_validation.pkl")
    parser.add_argument("--flow-dir", default="../H36M-Toolbox/flow_images_float")
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument(
        "--num-samples-list",
        default="4,5",
        help="Comma-separated sample counts to compare, e.g. 4,5.",
    )
    parser.add_argument("--max-frames", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--patch-radius", type=int, default=3)
    parser.add_argument("--out", default="debug_vis/flow_mfce_sampling_diagnosis.json")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a small synthetic motion-boundary diagnostic and exit.",
    )
    return parser.parse_args()


def seq_name(shot):
    return "s_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}".format(
        int(shot["subject"]),
        int(shot["action"]),
        int(shot["subaction"]),
        int(shot["camera_id"]) + 1,
    )


def image_stem(seq, frame_id):
    return "{}_{:06d}".format(seq, int(frame_id))


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center, scale, output_size, inv=False):
    center = np.array(center, dtype=np.float32)
    scale = np.array(scale, dtype=np.float32)
    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w, dst_h = output_size

    src_dir = np.array([0, (src_w - 1) * -0.5], dtype=np.float32)
    dst_dir = np.array([0, (dst_w - 1) * -0.5], dtype=np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    src[2, :] = get_3rd_point(src[0, :], src[1, :])
    dst[0, :] = [(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]
    dst[1, :] = dst[0, :] + dst_dir
    dst[2, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def affine_apply(transform, xy):
    xy = np.asarray(xy, dtype=np.float32)
    ones = np.ones((xy.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([xy, ones], axis=1)
    return homo @ transform.T


def load_labels(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def dce_initial_offsets_px(num_heads, num_samples, width, height, apply_tanh=True):
    """Return the DCE/AMS initial offset pattern in pixel units.

    This mirrors DeformableBlock._reset_parameters: heads point in different
    circular directions, samples grow radially, and offsets are later passed
    through tanh before being added to normalized grid coordinates.
    """
    thetas = np.arange(num_heads, dtype=np.float32) * (2.0 * math.pi / num_heads)
    grid = np.stack([np.cos(thetas), np.sin(thetas)], axis=-1)
    grid = 0.01 * grid / np.max(np.abs(grid), axis=-1, keepdims=True)
    grid = np.repeat(grid[:, None, :], num_samples, axis=1)
    for sample_idx in range(num_samples):
        grid[:, sample_idx, :] *= sample_idx + 1
    if apply_tanh:
        grid = np.tanh(grid)

    offsets = grid.copy()
    offsets[..., 0] *= (width - 1) / 2.0
    offsets[..., 1] *= (height - 1) / 2.0
    return offsets.astype(np.float32)


def bilinear_sample_flow(flow, xy):
    flow = np.asarray(flow, dtype=np.float32)
    coords = np.asarray(xy, dtype=np.float32)
    original_shape = coords.shape[:-1]
    flat = coords.reshape(-1, 2)
    h, w = flow.shape[:2]

    x = np.clip(flat[:, 0], 0.0, w - 1.0)
    y = np.clip(flat[:, 1], 0.0, h - 1.0)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # If x or y lies exactly on the image border, the bilinear weights above
    # can collapse to zero because x0 == x1 or y0 == y1. Handle this by using
    # the clipped border value for those degenerate coordinates.
    sampled = (
        flow[y0, x0] * wa[:, None]
        + flow[y1, x0] * wb[:, None]
        + flow[y0, x1] * wc[:, None]
        + flow[y1, x1] * wd[:, None]
    )
    border_mask = (x0 == x1) | (y0 == y1)
    if np.any(border_mask):
        sampled[border_mask] = flow[np.rint(y[border_mask]).astype(np.int32), np.rint(x[border_mask]).astype(np.int32)]
    return sampled.reshape(*original_shape, 2)


def patch_stats(flow, xy, radius):
    h, w = flow.shape[:2]
    mag = np.sqrt(np.sum(flow.astype(np.float32) ** 2, axis=-1))
    grad_x = np.zeros_like(mag)
    grad_y = np.zeros_like(mag)
    grad_x[:, 1:] = np.abs(mag[:, 1:] - mag[:, :-1])
    grad_y[1:, :] = np.abs(mag[1:, :] - mag[:-1, :])
    edge = grad_x + grad_y

    stats = []
    for coord in np.asarray(xy, dtype=np.float32):
        cx = int(np.clip(round(float(coord[0])), 0, w - 1))
        cy = int(np.clip(round(float(coord[1])), 0, h - 1))
        x0 = max(0, cx - radius)
        x1 = min(w, cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(h, cy + radius + 1)
        patch = flow[y0:y1, x0:x1].astype(np.float32)
        patch_mag = mag[y0:y1, x0:x1]
        stats.append(
            [
                float(patch_mag.mean()),
                float(patch_mag.std()),
                float(np.var(patch[..., 0]) + np.var(patch[..., 1])),
                float(edge[cy, cx]),
            ]
        )
    return np.asarray(stats, dtype=np.float32)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": None, "p50": None, "p90": None, "p95": None}
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
    }


def corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    x = x[mask]
    y = y[mask]
    if x.std() < 1e-12 or y.std() < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def action_name(action):
    idx = int(action) - 2
    if 0 <= idx < len(ACTION_NAMES):
        return ACTION_NAMES[idx]
    return str(action)


def joint_name(joint):
    if 0 <= int(joint) < len(JOINT_NAMES):
        return JOINT_NAMES[int(joint)]
    return str(joint)


def row_summary(rows, key):
    return summarize([row[key] for row in rows])


def grouped_summary(rows, group_key, metric_key, top_k=10, reverse=True):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[group_key]].append(row[metric_key])

    out = []
    for group, values in grouped.items():
        item = {group_key: group, **summarize(values)}
        if group_key == "joint":
            item["name"] = joint_name(group)
        elif group_key == "action":
            item["name"] = action_name(group)
        out.append(item)
    return sorted(out, key=lambda item: (-float("inf") if item["mean"] is None else item["mean"]), reverse=reverse)[:top_k]


def diagnose_for_offsets(flow, cur_cpn, gt_target, stats, offsets_px, meta, margin):
    center = bilinear_sample_flow(flow, cur_cpn)
    sample_xy = cur_cpn[:, None, None, :] + offsets_px[None, :, :, :]
    sampled = bilinear_sample_flow(flow, sample_xy)
    flat_sampled = sampled.reshape(sampled.shape[0], -1, 2)

    center_error = np.linalg.norm(center - gt_target, axis=1)
    point_error = np.linalg.norm(flat_sampled - gt_target[:, None, :], axis=-1)
    multi_mean = flat_sampled.mean(axis=1)
    multi_mean_error = np.linalg.norm(multi_mean - gt_target, axis=1)
    best_error = point_error.min(axis=1)
    mean_point_error = point_error.mean(axis=1)
    flow_delta_from_center = np.linalg.norm(flat_sampled - center[:, None, :], axis=-1)

    rows = []
    for joint_idx in range(cur_cpn.shape[0]):
        rows.append(
            {
                **meta,
                "joint": joint_idx,
                "center_error": float(center_error[joint_idx]),
                "multi_mean_error": float(multi_mean_error[joint_idx]),
                "best_error": float(best_error[joint_idx]),
                "mean_point_error": float(mean_point_error[joint_idx]),
                "multi_minus_center": float(multi_mean_error[joint_idx] - center_error[joint_idx]),
                "best_minus_center": float(best_error[joint_idx] - center_error[joint_idx]),
                "frac_points_better_than_center": float(np.mean(point_error[joint_idx] + margin < center_error[joint_idx])),
                "frac_points_worse_than_center": float(np.mean(point_error[joint_idx] > center_error[joint_idx] + margin)),
                "frac_flow_delta_gt_1px": float(np.mean(flow_delta_from_center[joint_idx] > 1.0)),
                "frac_flow_delta_gt_2px": float(np.mean(flow_delta_from_center[joint_idx] > 2.0)),
                "frac_flow_delta_gt_5px": float(np.mean(flow_delta_from_center[joint_idx] > 5.0)),
                "mean_flow_delta_from_center": float(flow_delta_from_center[joint_idx].mean()),
                "patch_mag_mean": float(stats[joint_idx, 0]),
                "patch_mag_std": float(stats[joint_idx, 1]),
                "patch_flow_var": float(stats[joint_idx, 2]),
                "flow_edge": float(stats[joint_idx, 3]),
            }
        )
    return rows


def summarize_diagnosis(rows, margin):
    multi_delta = [row["multi_minus_center"] for row in rows]
    best_delta = [row["best_minus_center"] for row in rows]
    return {
        "center_error": row_summary(rows, "center_error"),
        "multi_mean_error": row_summary(rows, "multi_mean_error"),
        "best_error": row_summary(rows, "best_error"),
        "mean_point_error": row_summary(rows, "mean_point_error"),
        "multi_minus_center": summarize(multi_delta),
        "best_minus_center": summarize(best_delta),
        "frac_multi_mean_worse_than_center": float(np.mean(np.asarray(multi_delta) > margin)) if rows else None,
        "frac_oracle_sample_better_than_center": float(np.mean(np.asarray(best_delta) < -margin)) if rows else None,
        "frac_points_better_than_center": row_summary(rows, "frac_points_better_than_center"),
        "frac_points_worse_than_center": row_summary(rows, "frac_points_worse_than_center"),
        "frac_flow_delta_gt_1px": row_summary(rows, "frac_flow_delta_gt_1px"),
        "frac_flow_delta_gt_2px": row_summary(rows, "frac_flow_delta_gt_2px"),
        "frac_flow_delta_gt_5px": row_summary(rows, "frac_flow_delta_gt_5px"),
        "mean_flow_delta_from_center": row_summary(rows, "mean_flow_delta_from_center"),
        "correlations_with_multi_degradation": {
            "center_error": corr([row["center_error"] for row in rows], multi_delta),
            "patch_mag_std": corr([row["patch_mag_std"] for row in rows], multi_delta),
            "patch_flow_var": corr([row["patch_flow_var"] for row in rows], multi_delta),
            "flow_edge": corr([row["flow_edge"] for row in rows], multi_delta),
        },
        "worst_joints_by_multi_degradation": grouped_summary(rows, "joint", "multi_minus_center"),
        "worst_actions_by_multi_degradation": grouped_summary(rows, "action", "multi_minus_center"),
        "best_joints_by_oracle_opportunity": grouped_summary(rows, "joint", "best_minus_center", reverse=False),
        "top_degradation_examples": sorted(rows, key=lambda row: row["multi_minus_center"], reverse=True)[:20],
    }


def run_self_test():
    flow = np.zeros((64, 64, 2), dtype=np.float32)
    flow[:, :32, 0] = 1.0
    flow[:, 32:, 0] = -1.0
    cur_cpn = np.array([[31.2, 32.0]], dtype=np.float32)
    gt_target = np.array([[1.0, 0.0]], dtype=np.float32)
    stats = np.zeros((1, 4), dtype=np.float32)
    offsets = dce_initial_offsets_px(num_heads=4, num_samples=5, width=64, height=64)
    row = diagnose_for_offsets(
        flow=flow,
        cur_cpn=cur_cpn,
        gt_target=gt_target,
        stats=stats,
        offsets_px=offsets,
        meta={"subject": 1, "action": 2, "action_name": "Directions-1", "subaction": 1, "camera_id": 0, "frame_id": 100},
        margin=0.05,
    )[0]
    print(json.dumps(row, indent=2))
    if not (row["center_error"] < 0.5 and row["multi_mean_error"] > row["center_error"]):
        raise SystemExit("Synthetic self-test failed: expected multi-sampling to be worse near a motion boundary.")
    print("Synthetic self-test passed.")


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    labels_path = Path(args.labels).expanduser().resolve()
    flow_dir = Path(args.flow_dir).expanduser().resolve()
    output_size = (args.width, args.height)
    sample_counts = [int(item.strip()) for item in args.num_samples_list.split(",") if item.strip()]

    labels = load_labels(labels_path)
    by_key = {}
    for shot in labels:
        key = (
            int(shot["subject"]),
            int(shot["action"]),
            int(shot["subaction"]),
            int(shot["camera_id"]),
            int(shot["image_id"]),
        )
        by_key[key] = shot

    keys = list(by_key)
    random.Random(args.seed).shuffle(keys)

    rows_by_samples = {sample_count: [] for sample_count in sample_counts}
    offsets_by_samples = {
        sample_count: dce_initial_offsets_px(args.num_heads, sample_count, args.width, args.height)
        for sample_count in sample_counts
    }
    missing_prev = 0
    missing_flow = 0
    frame_count = 0
    for key in keys:
        if frame_count >= args.max_frames:
            break
        subject, action, subaction, camera_id, frame_id = key
        prev_key = (subject, action, subaction, camera_id, frame_id - args.frame_gap)
        prev = by_key.get(prev_key)
        if prev is None:
            missing_prev += 1
            continue

        cur = by_key[key]
        seq = seq_name(cur)
        flow_path = flow_dir / seq / (image_stem(seq, frame_id) + ".npy")
        if not flow_path.exists():
            missing_flow += 1
            continue

        flow = np.load(flow_path).astype(np.float32)
        if flow.ndim != 3 or flow.shape[-1] != 2:
            raise ValueError("Expected flow [H,W,2], got {} at {}".format(flow.shape, flow_path))

        cur_cpn = np.asarray(cur["joints_2d_cpn_crop"], dtype=np.float32)
        cur_gt = np.asarray(cur["joints_2d_gt_crop"], dtype=np.float32)
        prev_gt = np.asarray(prev["joints_2d_gt_crop"], dtype=np.float32)

        prev_to_raw = get_affine_transform(prev["center"], prev["scale"], output_size, inv=True)
        raw_to_cur = get_affine_transform(cur["center"], cur["scale"], output_size, inv=False)
        prev_gt_same = affine_apply(raw_to_cur, affine_apply(prev_to_raw, prev_gt))
        gt_target = prev_gt_same - cur_gt
        stats = patch_stats(flow, cur_cpn, args.patch_radius)
        frame_count += 1

        meta = {
            "subject": subject,
            "action": action,
            "action_name": action_name(action),
            "subaction": subaction,
            "camera_id": camera_id,
            "frame_id": frame_id,
        }
        for sample_count, offsets_px in offsets_by_samples.items():
            rows_by_samples[sample_count].extend(
                diagnose_for_offsets(flow, cur_cpn, gt_target, stats, offsets_px, meta, args.margin)
            )

    out = {
        "meta": {
            "labels": str(labels_path),
            "flow_dir": str(flow_dir),
            "frame_gap": args.frame_gap,
            "width": args.width,
            "height": args.height,
            "num_heads": args.num_heads,
            "num_samples_list": sample_counts,
            "max_frames": args.max_frames,
            "num_frame_samples": frame_count,
            "missing_prev": missing_prev,
            "missing_flow": missing_flow,
            "margin": args.margin,
            "patch_radius": args.patch_radius,
        },
        "offset_geometry_px": {
            str(sample_count): {
                "max_abs_dx": float(np.max(np.abs(offsets[..., 0]))),
                "max_abs_dy": float(np.max(np.abs(offsets[..., 1]))),
                "offsets": offsets.reshape(-1, 2).round(4).tolist(),
            }
            for sample_count, offsets in offsets_by_samples.items()
        },
        "by_num_samples": {
            str(sample_count): summarize_diagnosis(rows, args.margin)
            for sample_count, rows in rows_by_samples.items()
        },
    }

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(out, file, indent=2)
    print("Wrote {}".format(out_path))
    print(json.dumps(out["meta"], indent=2))
    for sample_count in sample_counts:
        summary = out["by_num_samples"][str(sample_count)]
        print("num_samples={}".format(sample_count))
        print(json.dumps({k: summary[k] for k in ["center_error", "multi_mean_error", "multi_minus_center", "frac_multi_mean_worse_than_center", "frac_oracle_sample_better_than_center"]}, indent=2))


if __name__ == "__main__":
    main()
