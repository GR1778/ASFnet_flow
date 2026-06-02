import argparse
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from diagnose_flow_feature_sampling import (
    CONV_KEY_CANDIDATES,
    action_name,
    bilinear_sample_feature,
    corr,
    joint_name,
    load_flow_conv,
    summarize,
)
from diagnose_flow_mfce_sampling import (
    affine_apply,
    bilinear_sample_flow,
    dce_initial_offsets_px,
    get_affine_transform,
    image_stem,
    patch_stats,
    seq_name,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose how much the trained Conv2d(2->128, 3x3) flow encoder depends on "
            "neighboring pixels. The script compares the full 3x3 output with a center-only "
            "version that keeps only the learned central kernel, approximating a 1x1 projection."
        )
    )
    parser.add_argument("--labels", default="data/h36m_validation.pkl")
    parser.add_argument("--flow-dir", default="../H36M-Toolbox/flow_images_float")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument(
        "--conv-prefix",
        default="auto",
        help=(
            "Prefix of flow Conv2d weights in checkpoint. Use auto to search common keys: "
            + ", ".join(CONV_KEY_CANDIDATES)
        ),
    )
    parser.add_argument("--flow-clip", type=float, default=5.0)
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-radius", type=int, default=3)
    parser.add_argument("--low-mag-threshold", type=float, default=0.04)
    parser.add_argument("--out", default="debug_vis/flow_conv_kernel_effect.json")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def load_labels(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def encode_flow(flow, weight, bias):
    try:
        import torch
        import torch.nn.functional as F

        x = torch.from_numpy(flow.transpose(2, 0, 1)).unsqueeze(0).float()
        w = torch.from_numpy(weight).float()
        b = torch.from_numpy(bias).float()
        with torch.no_grad():
            y = F.conv2d(x, w, b, padding=1)
        return y.squeeze(0).permute(1, 2, 0).contiguous().numpy()
    except ImportError:
        h, w_in = flow.shape[:2]
        out_channels = weight.shape[0]
        out = np.zeros((h, w_in, out_channels), dtype=np.float32)
        for out_idx in range(out_channels):
            acc = np.full((h, w_in), bias[out_idx], dtype=np.float32)
            for in_idx in range(weight.shape[1]):
                acc += cv2.filter2D(
                    flow[:, :, in_idx],
                    ddepth=-1,
                    kernel=weight[out_idx, in_idx],
                    borderType=cv2.BORDER_CONSTANT,
                )
            out[:, :, out_idx] = acc
        return out


def center_only_weight(weight):
    center = np.zeros_like(weight)
    kh, kw = weight.shape[-2:]
    center[:, :, kh // 2, kw // 2] = weight[:, :, kh // 2, kw // 2]
    return center


def cosine_similarity(a, b, eps=1e-8):
    numerator = np.sum(a * b, axis=-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + eps
    return numerator / denom


def grouped_summary(rows, group_key, metric_key, top_k=10, reverse=True):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[group_key]].append(row[metric_key])
    items = []
    for group, values in grouped.items():
        item = {group_key: group, **summarize(values)}
        if group_key == "joint":
            item["name"] = joint_name(group)
        elif group_key == "action":
            item["name"] = action_name(group)
        items.append(item)
    return sorted(
        items,
        key=lambda item: -float("inf") if item["mean"] is None else item["mean"],
        reverse=reverse,
    )[:top_k]


def subgroup_summary(rows, selector):
    selected = [row for row in rows if selector(row)]
    return {
        "count": len(selected),
        "full_vs_center_l2": summarize([row["full_vs_center_l2"] for row in selected]),
        "full_vs_center_rel": summarize([row["full_vs_center_rel"] for row in selected]),
        "full_vs_center_cos": summarize([row["full_vs_center_cos"] for row in selected]),
        "neighbor_to_full_signal_ratio": summarize([row["neighbor_to_full_signal_ratio"] for row in selected]),
        "full_spread_minus_center_spread": summarize([row["full_spread_minus_center_spread"] for row in selected]),
    }


def summarize_rows(rows, low_mag_threshold):
    patch_vars = np.asarray([row["patch_flow_var"] for row in rows], dtype=np.float64)
    flow_edges = np.asarray([row["flow_edge"] for row in rows], dtype=np.float64)
    patch_var_p90 = float(np.percentile(patch_vars, 90)) if len(rows) else None
    edge_p90 = float(np.percentile(flow_edges, 90)) if len(rows) else None

    out = {
        "full_vs_center_l2": summarize([row["full_vs_center_l2"] for row in rows]),
        "full_vs_center_rel": summarize([row["full_vs_center_rel"] for row in rows]),
        "full_vs_center_cos": summarize([row["full_vs_center_cos"] for row in rows]),
        "neighbor_to_center_signal_ratio": summarize([row["neighbor_to_center_signal_ratio"] for row in rows]),
        "neighbor_to_full_signal_ratio": summarize([row["neighbor_to_full_signal_ratio"] for row in rows]),
        "full_local_spread_l2": summarize([row["full_local_spread_l2"] for row in rows]),
        "center_only_local_spread_l2": summarize([row["center_only_local_spread_l2"] for row in rows]),
        "full_spread_minus_center_spread": summarize([row["full_spread_minus_center_spread"] for row in rows]),
        "raw_center_error": summarize([row["raw_center_error"] for row in rows]),
        "raw_flow_mag": summarize([row["raw_flow_mag"] for row in rows]),
        "patch_flow_var": summarize([row["patch_flow_var"] for row in rows]),
        "flow_edge": summarize([row["flow_edge"] for row in rows]),
        "correlations": {
            "kernel_delta_vs_patch_flow_var": corr(
                [row["full_vs_center_l2"] for row in rows],
                [row["patch_flow_var"] for row in rows],
            ),
            "kernel_delta_vs_flow_edge": corr(
                [row["full_vs_center_l2"] for row in rows],
                [row["flow_edge"] for row in rows],
            ),
            "kernel_delta_vs_raw_center_error": corr(
                [row["full_vs_center_l2"] for row in rows],
                [row["raw_center_error"] for row in rows],
            ),
            "full_minus_center_spread_vs_patch_flow_var": corr(
                [row["full_spread_minus_center_spread"] for row in rows],
                [row["patch_flow_var"] for row in rows],
            ),
            "full_minus_center_spread_vs_flow_edge": corr(
                [row["full_spread_minus_center_spread"] for row in rows],
                [row["flow_edge"] for row in rows],
            ),
        },
        "thresholds": {
            "patch_flow_var_p90": patch_var_p90,
            "flow_edge_p90": edge_p90,
            "low_mag_threshold": low_mag_threshold,
        },
        "subgroups": {
            "high_patch_flow_var_top10": subgroup_summary(
                rows, lambda row: patch_var_p90 is not None and row["patch_flow_var"] >= patch_var_p90
            ),
            "high_flow_edge_top10": subgroup_summary(
                rows, lambda row: edge_p90 is not None and row["flow_edge"] >= edge_p90
            ),
            "low_raw_flow_magnitude": subgroup_summary(
                rows, lambda row: row["raw_flow_mag"] <= low_mag_threshold
            ),
        },
        "worst_joints_by_kernel_delta": grouped_summary(rows, "joint", "full_vs_center_l2"),
        "worst_actions_by_kernel_delta": grouped_summary(rows, "action", "full_vs_center_l2"),
        "worst_joints_by_full_spread_increase": grouped_summary(rows, "joint", "full_spread_minus_center_spread"),
        "top_kernel_delta_examples": sorted(rows, key=lambda row: row["full_vs_center_l2"], reverse=True)[:20],
    }
    return out


def process_frame(flow, full_feature, center_feature, cur, prev, output_size, offsets_px, args, meta):
    cur_cpn = np.asarray(cur["joints_2d_cpn_crop"], dtype=np.float32)
    cur_gt = np.asarray(cur["joints_2d_gt_crop"], dtype=np.float32)
    prev_gt = np.asarray(prev["joints_2d_gt_crop"], dtype=np.float32)

    prev_to_raw = get_affine_transform(prev["center"], prev["scale"], output_size, inv=True)
    raw_to_cur = get_affine_transform(cur["center"], cur["scale"], output_size, inv=False)
    prev_gt_same = affine_apply(raw_to_cur, affine_apply(prev_to_raw, prev_gt))
    gt_target = prev_gt_same - cur_gt
    if args.flow_clip > 0:
        gt_target = np.clip(gt_target, -args.flow_clip, args.flow_clip) / args.flow_clip

    raw_center = bilinear_sample_flow(flow, cur_cpn)
    raw_center_error = np.linalg.norm(raw_center - gt_target, axis=1)
    raw_flow_mag = np.linalg.norm(raw_center, axis=1)
    stats = patch_stats(flow, cur_cpn, args.patch_radius)

    full_center = bilinear_sample_feature(full_feature, cur_cpn)
    center_center = bilinear_sample_feature(center_feature, cur_cpn)
    diff = full_center - center_center

    full_signal = full_center - full_feature[0, 0][None, :]
    center_signal = center_center - center_feature[0, 0][None, :]

    feature_xy = cur_cpn[:, None, None, :] + offsets_px[None, :, :, :]
    full_sampled = bilinear_sample_feature(full_feature, feature_xy).reshape(cur_cpn.shape[0], -1, full_feature.shape[-1])
    center_sampled = bilinear_sample_feature(center_feature, feature_xy).reshape(
        cur_cpn.shape[0], -1, center_feature.shape[-1]
    )
    full_spread = np.linalg.norm(full_sampled - full_center[:, None, :], axis=-1).mean(axis=1)
    center_spread = np.linalg.norm(center_sampled - center_center[:, None, :], axis=-1).mean(axis=1)

    rows = []
    for joint_idx in range(cur_cpn.shape[0]):
        full_norm = float(np.linalg.norm(full_center[joint_idx]))
        center_signal_norm = float(np.linalg.norm(center_signal[joint_idx]))
        full_signal_norm = float(np.linalg.norm(full_signal[joint_idx]))
        diff_norm = float(np.linalg.norm(diff[joint_idx]))
        rows.append(
            {
                **meta,
                "joint": joint_idx,
                "joint_name": joint_name(joint_idx),
                "raw_center_error": float(raw_center_error[joint_idx]),
                "raw_flow_mag": float(raw_flow_mag[joint_idx]),
                "patch_mag_mean": float(stats[joint_idx, 0]),
                "patch_mag_std": float(stats[joint_idx, 1]),
                "patch_flow_var": float(stats[joint_idx, 2]),
                "flow_edge": float(stats[joint_idx, 3]),
                "full_feature_norm": full_norm,
                "center_only_feature_norm": float(np.linalg.norm(center_center[joint_idx])),
                "full_vs_center_l2": diff_norm,
                "full_vs_center_rel": diff_norm / (full_norm + 1e-8),
                "full_vs_center_cos": float(cosine_similarity(full_center[joint_idx], center_center[joint_idx])),
                "neighbor_to_center_signal_ratio": diff_norm / (center_signal_norm + 1e-8),
                "neighbor_to_full_signal_ratio": diff_norm / (full_signal_norm + 1e-8),
                "full_local_spread_l2": float(full_spread[joint_idx]),
                "center_only_local_spread_l2": float(center_spread[joint_idx]),
                "full_spread_minus_center_spread": float(full_spread[joint_idx] - center_spread[joint_idx]),
            }
        )
    return rows


def run_self_test():
    flow = np.zeros((8, 8, 2), dtype=np.float32)
    flow[:, 4:, 0] = 1.0
    weight = np.zeros((4, 2, 3, 3), dtype=np.float32)
    weight[:, 0, 1, 1] = 1.0
    weight[:, 0, 1, 0] = 0.5
    bias = np.zeros((4,), dtype=np.float32)
    full = encode_flow(flow, weight, bias)
    center = encode_flow(flow, center_only_weight(weight), bias)
    delta = np.linalg.norm(full - center, axis=-1)
    if delta[:, 4:].max() <= 0:
        raise SystemExit("Self-test failed: neighbor contribution is zero.")
    print("Self-test passed.")


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --self-test is used.")

    conv_prefix, weight, bias = load_flow_conv(args.checkpoint, args.conv_prefix)
    center_weight = center_only_weight(weight)
    labels_path = Path(args.labels).expanduser().resolve()
    flow_dir = Path(args.flow_dir).expanduser().resolve()
    output_size = (args.width, args.height)
    offsets_px = dce_initial_offsets_px(args.num_heads, args.num_samples, args.width, args.height)

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
    frame_count = 0
    missing_prev = 0
    missing_flow = 0
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
        if args.flow_clip > 0:
            flow = np.clip(flow, -args.flow_clip, args.flow_clip) / args.flow_clip

        full_feature = encode_flow(flow, weight, bias)
        center_feature = encode_flow(flow, center_weight, bias)
        rows.extend(
            process_frame(
                flow=flow,
                full_feature=full_feature,
                center_feature=center_feature,
                cur=cur,
                prev=prev,
                output_size=output_size,
                offsets_px=offsets_px,
                args=args,
                meta={
                    "subject": subject,
                    "action": action,
                    "action_name": action_name(action),
                    "subaction": subaction,
                    "camera_id": camera_id,
                    "frame_id": frame_id,
                },
            )
        )
        frame_count += 1

    out = {
        "meta": {
            "labels": str(labels_path),
            "flow_dir": str(flow_dir),
            "checkpoint": str(Path(args.checkpoint).expanduser()),
            "conv_prefix": conv_prefix,
            "flow_clip": args.flow_clip,
            "frame_gap": args.frame_gap,
            "width": args.width,
            "height": args.height,
            "num_heads": args.num_heads,
            "num_samples": args.num_samples,
            "max_frames": args.max_frames,
            "num_frame_samples": frame_count,
            "num_joint_rows": len(rows),
            "missing_prev": missing_prev,
            "missing_flow": missing_flow,
            "patch_radius": args.patch_radius,
        },
        "conv_weight_energy": {
            "center_kernel_l2": float(np.linalg.norm(weight[:, :, weight.shape[-2] // 2, weight.shape[-1] // 2])),
            "neighbor_kernel_l2": float(np.linalg.norm(weight - center_weight)),
            "neighbor_over_center_kernel_l2": float(
                np.linalg.norm(weight - center_weight)
                / (np.linalg.norm(weight[:, :, weight.shape[-2] // 2, weight.shape[-1] // 2]) + 1e-8)
            ),
        },
        "summary": summarize_rows(rows, args.low_mag_threshold),
    }
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(out, file, indent=2)
    print("Wrote {}".format(out_path))
    print(json.dumps(out["meta"], indent=2))
    print(json.dumps(out["conv_weight_energy"], indent=2))
    print(json.dumps(out["summary"]["full_vs_center_rel"], indent=2))
    print(json.dumps(out["summary"]["correlations"], indent=2))


if __name__ == "__main__":
    main()
