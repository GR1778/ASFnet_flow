import argparse
import json
import math
import pickle
import random
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


JOINT_NAMES = [
    "Hip",
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

PARENTS = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare static DCE offsets and motion-oriented offsets on joint-level optical flow. "
            "The test measures which candidate set samples flow vectors closer to the GT joint displacement."
        )
    )
    parser.add_argument("--labels", default="data/h36m_validation.pkl")
    parser.add_argument("--flow-dir", default="../H36M-Toolbox/flow_images_float")
    parser.add_argument("--image-root", default="", help="Optional root for cropped RGB images used in visualizations.")
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--min-cpn-gt-error", type=float, default=4.0)
    parser.add_argument("--out", default="debug_vis/flow_offset_strategy_compare.json")
    parser.add_argument("--vis-dir", default="debug_vis/flow_offset_strategy_vis")
    parser.add_argument("--no-vis", action="store_true")
    return parser.parse_args()


def load_labels(path):
    with open(path, "rb") as file:
        return pickle.load(file)


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

    sampled = (
        flow[y0, x0] * wa[:, None]
        + flow[y1, x0] * wb[:, None]
        + flow[y0, x1] * wc[:, None]
        + flow[y1, x1] * wd[:, None]
    )
    border_mask = (x0 == x1) | (y0 == y1)
    if np.any(border_mask):
        sampled[border_mask] = flow[
            np.rint(y[border_mask]).astype(np.int32),
            np.rint(x[border_mask]).astype(np.int32),
        ]
    return sampled.reshape(*original_shape, 2)


def dce_offsets_px(num_heads, num_samples, width, height):
    thetas = np.arange(num_heads, dtype=np.float32) * (2.0 * math.pi / num_heads)
    unit = np.stack([np.cos(thetas), np.sin(thetas)], axis=-1)
    unit = unit / np.maximum(np.max(np.abs(unit), axis=-1, keepdims=True), 1e-6)
    offsets = []
    for head in range(num_heads):
        for sample_idx in range(num_samples):
            norm = 0.01 * float(sample_idx + 1) * unit[head]
            offsets.append([norm[0] * (width - 1) / 2.0, norm[1] * (height - 1) / 2.0])
    return np.asarray(offsets, dtype=np.float32)


def rotate_offsets_by_flow(static_offsets, center_flow):
    center_flow = np.asarray(center_flow, dtype=np.float32)
    out = np.zeros((center_flow.shape[0], static_offsets.shape[0], 2), dtype=np.float32)
    for joint_idx, vec in enumerate(center_flow):
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            out[joint_idx] = static_offsets
            continue
        e_parallel = vec / norm
        e_perp = np.array([-e_parallel[1], e_parallel[0]], dtype=np.float32)
        basis = np.stack([e_parallel, e_perp], axis=1)
        out[joint_idx] = static_offsets @ basis.T
    return out


def motion_strip_offsets(center_flow, num_samples):
    """An anisotropic local kernel: longer along motion, narrow across motion."""
    center_flow = np.asarray(center_flow, dtype=np.float32)
    forward = np.linspace(-num_samples, num_samples, 2 * num_samples + 1, dtype=np.float32)
    lateral = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    template = np.asarray([[a, b] for a in forward for b in lateral], dtype=np.float32)
    out = np.zeros((center_flow.shape[0], template.shape[0], 2), dtype=np.float32)
    for joint_idx, vec in enumerate(center_flow):
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            e_parallel = np.array([1.0, 0.0], dtype=np.float32)
        else:
            e_parallel = vec / norm
        e_perp = np.array([-e_parallel[1], e_parallel[0]], dtype=np.float32)
        basis = np.stack([e_parallel, e_perp], axis=1)
        out[joint_idx] = template @ basis.T
    return out


def flow_to_rgb(flow):
    flow = np.asarray(flow, dtype=np.float32)
    fx, fy = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)
    hsv[..., 1] = np.clip(mag / (np.percentile(mag, 99) + 1e-6) * 255, 0, 255).astype(np.uint8)
    hsv[..., 2] = 255
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    low = mag < max(0.05, np.percentile(mag, 30))
    rgb[low] = 255
    return rgb


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "p50": None, "p90": None, "p95": None}
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
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


def strategy_errors(flow, cur_cpn, gt_target, offsets_by_joint):
    xy = cur_cpn[:, None, :] + offsets_by_joint
    sampled = bilinear_sample_flow(flow, xy)
    err = np.linalg.norm(sampled - gt_target[:, None, :], axis=-1)
    return sampled, err


def read_optional_image(image_root, rel_path):
    if not image_root:
        return None
    path = Path(image_root) / rel_path
    if not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_skeleton(ax, joints, color, linewidth=1.0, alpha=0.9):
    for joint, parent in enumerate(PARENTS):
        if parent < 0:
            continue
        ax.plot(
            [joints[joint, 0], joints[parent, 0]],
            [joints[joint, 1], joints[parent, 1]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )


def visualize_example(row, cur, flow, static_xy, motion_xy, strip_xy, out_path, image_root):
    joint = int(row["joint"])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), dpi=180)
    rgb = read_optional_image(image_root, cur.get("image", ""))
    if rgb is None:
        rgb = np.ones((flow.shape[0], flow.shape[1], 3), dtype=np.uint8) * 255
    flow_rgb = flow_to_rgb(flow)

    cur_gt = np.asarray(cur["joints_2d_gt_crop"], dtype=np.float32)
    cur_cpn = np.asarray(cur["joints_2d_cpn_crop"], dtype=np.float32)

    axes[0].imshow(rgb)
    draw_skeleton(axes[0], cur_gt, "#ef4444", linewidth=1.0, alpha=0.75)
    draw_skeleton(axes[0], cur_cpn, "#6366f1", linewidth=1.0, alpha=0.75)
    axes[0].scatter(cur_gt[joint, 0], cur_gt[joint, 1], marker="D", s=28, color="#ef4444", label="GT")
    axes[0].scatter(cur_cpn[joint, 0], cur_cpn[joint, 1], marker="D", s=28, color="#818cf8", label="CPN")
    axes[0].legend(loc="lower right", fontsize=6, frameon=False)
    axes[0].set_axis_off()

    axes[1].imshow(flow_rgb)
    axes[1].scatter(static_xy[:, 0], static_xy[:, 1], s=12, color="#f97316", alpha=0.65, label="static")
    axes[1].scatter(motion_xy[:, 0], motion_xy[:, 1], s=12, color="#22c55e", alpha=0.65, label="motion-cross")
    axes[1].scatter(strip_xy[:, 0], strip_xy[:, 1], s=8, color="#06b6d4", alpha=0.45, label="motion-strip")
    axes[1].scatter(cur_gt[joint, 0], cur_gt[joint, 1], marker="D", s=28, color="#ef4444")
    axes[1].scatter(cur_cpn[joint, 0], cur_cpn[joint, 1], marker="D", s=28, color="#818cf8")
    axes[1].legend(loc="lower right", fontsize=6, frameon=False)
    axes[1].set_axis_off()

    fig.suptitle(
        "{} f{} {}: CPN-GT {:.1f}px, center {:.3f}, static-best {:.3f}, motion-best {:.3f}, strip-best {:.3f}".format(
            row["seq"],
            row["frame_id"],
            JOINT_NAMES[joint],
            row["cpn_gt_error"],
            row["center_error"],
            row["static_best_error"],
            row["motion_best_error"],
            row["strip_best_error"],
        ),
        fontsize=8,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def grouped_summary(rows, group_key, metric_key, top_k=12, reverse=True):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[group_key]].append(row[metric_key])
    items = []
    for key, values in grouped.items():
        item = {group_key: key, **summarize(values)}
        if group_key == "joint":
            item["name"] = JOINT_NAMES[int(key)]
        items.append(item)
    return sorted(items, key=lambda item: item["mean"] if item["mean"] is not None else -1e9, reverse=reverse)[:top_k]


def main():
    args = parse_args()
    if not Path(args.flow_dir).exists():
        raise FileNotFoundError(
            "Flow directory not found: {}. Run this script in DCFormer with --flow-dir pointing to "
            "../H36M-Toolbox/flow_images_float or ../H36M-Toolbox/flow_images_float_clip10.".format(args.flow_dir)
        )
    labels = load_labels(Path(args.labels))
    flow_dir = Path(args.flow_dir)
    static_offsets = dce_offsets_px(args.num_heads, args.num_samples, args.width, args.height)

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
    examples = []
    missing_prev = 0
    missing_flow = 0
    frame_count = 0
    output_size = (args.width, args.height)

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
        cur_cpn = np.asarray(cur["joints_2d_cpn_crop"], dtype=np.float32)
        cur_gt = np.asarray(cur["joints_2d_gt_crop"], dtype=np.float32)
        prev_gt = np.asarray(prev["joints_2d_gt_crop"], dtype=np.float32)
        prev_to_raw = get_affine_transform(prev["center"], prev["scale"], output_size, inv=True)
        raw_to_cur = get_affine_transform(cur["center"], cur["scale"], output_size, inv=False)
        prev_gt_same = affine_apply(raw_to_cur, affine_apply(prev_to_raw, prev_gt))
        gt_target = prev_gt_same - cur_gt

        center_flow = bilinear_sample_flow(flow, cur_cpn)
        center_error = np.linalg.norm(center_flow - gt_target, axis=1)
        cpn_gt_error = np.linalg.norm(cur_cpn - cur_gt, axis=1)

        static_offsets_by_joint = np.repeat(static_offsets[None, :, :], cur_cpn.shape[0], axis=0)
        motion_offsets_by_joint = rotate_offsets_by_flow(static_offsets, center_flow)
        strip_offsets_by_joint = motion_strip_offsets(center_flow, args.num_samples)

        _, static_err = strategy_errors(flow, cur_cpn, gt_target, static_offsets_by_joint)
        _, motion_err = strategy_errors(flow, cur_cpn, gt_target, motion_offsets_by_joint)
        _, strip_err = strategy_errors(flow, cur_cpn, gt_target, strip_offsets_by_joint)

        for joint_idx in range(cur_cpn.shape[0]):
            row = {
                "seq": seq,
                "subject": subject,
                "action": action,
                "subaction": subaction,
                "camera_id": camera_id,
                "frame_id": frame_id,
                "joint": joint_idx,
                "joint_name": JOINT_NAMES[joint_idx],
                "cpn_gt_error": float(cpn_gt_error[joint_idx]),
                "center_error": float(center_error[joint_idx]),
                "static_mean_error": float(static_err[joint_idx].mean()),
                "static_best_error": float(static_err[joint_idx].min()),
                "motion_mean_error": float(motion_err[joint_idx].mean()),
                "motion_best_error": float(motion_err[joint_idx].min()),
                "strip_mean_error": float(strip_err[joint_idx].mean()),
                "strip_best_error": float(strip_err[joint_idx].min()),
            }
            row["motion_best_minus_static_best"] = row["motion_best_error"] - row["static_best_error"]
            row["strip_best_minus_static_best"] = row["strip_best_error"] - row["static_best_error"]
            row["static_best_minus_center"] = row["static_best_error"] - row["center_error"]
            row["motion_best_minus_center"] = row["motion_best_error"] - row["center_error"]
            row["strip_best_minus_center"] = row["strip_best_error"] - row["center_error"]
            rows.append(row)

            if row["cpn_gt_error"] >= args.min_cpn_gt_error:
                examples.append(
                    (
                        row,
                        cur,
                        flow,
                        cur_cpn[joint_idx] + static_offsets_by_joint[joint_idx],
                        cur_cpn[joint_idx] + motion_offsets_by_joint[joint_idx],
                        cur_cpn[joint_idx] + strip_offsets_by_joint[joint_idx],
                    )
                )

        frame_count += 1

    def metric(name):
        return summarize([row[name] for row in rows])

    out = {
        "meta": {
            "labels": str(Path(args.labels).resolve()),
            "flow_dir": str(flow_dir.resolve()),
            "frame_gap": args.frame_gap,
            "num_heads": args.num_heads,
            "num_samples": args.num_samples,
            "max_frames": args.max_frames,
            "num_frame_samples": frame_count,
            "num_joint_rows": len(rows),
            "missing_prev": missing_prev,
            "missing_flow": missing_flow,
            "min_cpn_gt_error": args.min_cpn_gt_error,
        },
        "summary": {
            "cpn_gt_error": metric("cpn_gt_error"),
            "center_error": metric("center_error"),
            "static_best_error": metric("static_best_error"),
            "motion_best_error": metric("motion_best_error"),
            "strip_best_error": metric("strip_best_error"),
            "motion_best_minus_static_best": metric("motion_best_minus_static_best"),
            "strip_best_minus_static_best": metric("strip_best_minus_static_best"),
            "static_best_minus_center": metric("static_best_minus_center"),
            "motion_best_minus_center": metric("motion_best_minus_center"),
            "strip_best_minus_center": metric("strip_best_minus_center"),
            "frac_motion_best_better_static": float(np.mean([row["motion_best_error"] < row["static_best_error"] for row in rows])) if rows else None,
            "frac_strip_best_better_static": float(np.mean([row["strip_best_error"] < row["static_best_error"] for row in rows])) if rows else None,
            "corr_cpn_gt_with_static_gain": corr([row["cpn_gt_error"] for row in rows], [-row["static_best_minus_center"] for row in rows]),
            "corr_cpn_gt_with_motion_vs_static": corr([row["cpn_gt_error"] for row in rows], [-row["motion_best_minus_static_best"] for row in rows]),
            "corr_cpn_gt_with_strip_vs_static": corr([row["cpn_gt_error"] for row in rows], [-row["strip_best_minus_static_best"] for row in rows]),
            "by_joint_motion_vs_static": grouped_summary(rows, "joint", "motion_best_minus_static_best", reverse=False),
            "by_joint_strip_vs_static": grouped_summary(rows, "joint", "strip_best_minus_static_best", reverse=False),
            "large_2d_error_count": len(examples),
        },
        "top_large_2d_error_examples": sorted(
            [item[0] for item in examples],
            key=lambda row: (row["cpn_gt_error"], -row["motion_best_minus_static_best"]),
            reverse=True,
        )[: args.top_k],
        "top_motion_better_examples": sorted(rows, key=lambda row: row["motion_best_minus_static_best"])[: args.top_k],
        "top_strip_better_examples": sorted(rows, key=lambda row: row["strip_best_minus_static_best"])[: args.top_k],
        "top_static_better_examples": sorted(rows, key=lambda row: row["motion_best_minus_static_best"], reverse=True)[: args.top_k],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    if not args.no_vis and examples:
        vis_dir = Path(args.vis_dir)
        ranked = sorted(
            examples,
            key=lambda item: (item[0]["cpn_gt_error"], -item[0]["motion_best_minus_static_best"]),
            reverse=True,
        )[: args.top_k]
        for idx, (row, cur, flow, static_xy, motion_xy, strip_xy) in enumerate(ranked):
            out_file = vis_dir / "{:02d}_{}_f{}_{}_joint{}.png".format(
                idx,
                row["seq"],
                row["frame_id"],
                row["joint_name"],
                row["joint"],
            )
            visualize_example(row, cur, flow, static_xy, motion_xy, strip_xy, out_file, args.image_root)

    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
