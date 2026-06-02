import argparse
import json
import math
import pickle
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze joint-level optical-flow reliability for same-current-crop H36M flow. "
            "The script compares sampled flow at current CPN joints with the previous-frame "
            "joint displacement expressed in the current crop coordinates."
        )
    )
    parser.add_argument("--labels", default="data/h36m_validation.pkl")
    parser.add_argument("--flow-dir", default="../H36M-Toolbox/flow_images_float")
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-radius", type=int, default=3)
    parser.add_argument("--out", default="debug_vis/flow_reliability_probe.json")
    return parser.parse_args()


def seq_name(shot):
    return (
        "s_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}".format(
            int(shot["subject"]),
            int(shot["action"]),
            int(shot["subaction"]),
            int(shot["camera_id"]) + 1,
        )
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


def sample_flow(flow, xy):
    h, w = flow.shape[:2]
    coords = np.asarray(xy, dtype=np.float32)
    x = np.clip(np.rint(coords[:, 0]).astype(np.int32), 0, w - 1)
    y = np.clip(np.rint(coords[:, 1]).astype(np.int32), 0, h - 1)
    return flow[y, x]


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


def load_labels(path):
    with open(path, "rb") as file:
        labels = pickle.load(file)
    return labels


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


def main():
    args = parse_args()
    labels_path = Path(args.labels).expanduser().resolve()
    flow_dir = Path(args.flow_dir).expanduser().resolve()
    output_size = (args.width, args.height)

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

    rows = []
    missing_flow = 0
    missing_prev = 0
    for key in keys:
        if len(rows) >= args.max_samples:
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
        prev_cpn = np.asarray(prev["joints_2d_cpn_crop"], dtype=np.float32)
        prev_gt = np.asarray(prev["joints_2d_gt_crop"], dtype=np.float32)

        prev_to_raw = get_affine_transform(prev["center"], prev["scale"], output_size, inv=True)
        raw_to_cur = get_affine_transform(cur["center"], cur["scale"], output_size, inv=False)
        prev_cpn_same = affine_apply(raw_to_cur, affine_apply(prev_to_raw, prev_cpn))
        prev_gt_same = affine_apply(raw_to_cur, affine_apply(prev_to_raw, prev_gt))

        sampled = sample_flow(flow, cur_cpn)
        gt_target = prev_gt_same - cur_gt
        cpn_target = prev_cpn_same - cur_cpn
        stats = patch_stats(flow, cur_cpn, args.patch_radius)

        gt_error = np.linalg.norm(sampled - gt_target, axis=1)
        cpn_error = np.linalg.norm(sampled - cpn_target, axis=1)
        flow_mag = np.linalg.norm(sampled, axis=1)
        gt_motion = np.linalg.norm(gt_target, axis=1)
        cpn_motion = np.linalg.norm(cpn_target, axis=1)

        for joint_idx in range(cur_cpn.shape[0]):
            rows.append(
                {
                    "subject": subject,
                    "action": action,
                    "subaction": subaction,
                    "camera_id": camera_id,
                    "frame_id": frame_id,
                    "joint": joint_idx,
                    "gt_error": float(gt_error[joint_idx]),
                    "cpn_error": float(cpn_error[joint_idx]),
                    "flow_mag": float(flow_mag[joint_idx]),
                    "gt_motion": float(gt_motion[joint_idx]),
                    "cpn_motion": float(cpn_motion[joint_idx]),
                    "patch_mag_mean": float(stats[joint_idx, 0]),
                    "patch_mag_std": float(stats[joint_idx, 1]),
                    "patch_flow_var": float(stats[joint_idx, 2]),
                    "flow_edge": float(stats[joint_idx, 3]),
                }
            )

    grouped_by_joint = defaultdict(list)
    grouped_by_action = defaultdict(list)
    for row in rows:
        grouped_by_joint[row["joint"]].append(row["gt_error"])
        grouped_by_action[row["action"]].append(row["gt_error"])

    gt_error = [row["gt_error"] for row in rows]
    cpn_error = [row["cpn_error"] for row in rows]
    out = {
        "meta": {
            "labels": str(labels_path),
            "flow_dir": str(flow_dir),
            "frame_gap": args.frame_gap,
            "max_samples": args.max_samples,
            "num_joint_rows": len(rows),
            "num_frame_samples": len({(r["subject"], r["action"], r["subaction"], r["camera_id"], r["frame_id"]) for r in rows}),
            "missing_prev": missing_prev,
            "missing_flow": missing_flow,
            "patch_radius": args.patch_radius,
        },
        "error": {
            "gt_target": summarize(gt_error),
            "cpn_target": summarize(cpn_error),
        },
        "signals": {
            "flow_mag": summarize([row["flow_mag"] for row in rows]),
            "gt_motion": summarize([row["gt_motion"] for row in rows]),
            "patch_flow_var": summarize([row["patch_flow_var"] for row in rows]),
            "flow_edge": summarize([row["flow_edge"] for row in rows]),
        },
        "correlations_with_gt_error": {
            "flow_mag": corr([row["flow_mag"] for row in rows], gt_error),
            "gt_motion": corr([row["gt_motion"] for row in rows], gt_error),
            "patch_mag_std": corr([row["patch_mag_std"] for row in rows], gt_error),
            "patch_flow_var": corr([row["patch_flow_var"] for row in rows], gt_error),
            "flow_edge": corr([row["flow_edge"] for row in rows], gt_error),
        },
        "worst_joints_by_gt_error": sorted(
            [{"joint": joint, **summarize(values)} for joint, values in grouped_by_joint.items()],
            key=lambda item: item["mean"],
            reverse=True,
        ),
        "worst_actions_by_gt_error": sorted(
            [{"action": action, **summarize(values)} for action, values in grouped_by_action.items()],
            key=lambda item: item["mean"],
            reverse=True,
        ),
    }

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(out, file, indent=2)
    print("Wrote {}".format(out_path))
    print(json.dumps(out["meta"], indent=2))
    print(json.dumps(out["error"], indent=2))
    print(json.dumps(out["correlations_with_gt_error"], indent=2))


if __name__ == "__main__":
    main()
