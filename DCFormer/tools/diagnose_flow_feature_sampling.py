import argparse
import json
import math
import pickle
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from diagnose_flow_mfce_sampling import (
    ACTION_NAMES,
    JOINT_NAMES,
    affine_apply,
    bilinear_sample_flow,
    corr,
    dce_initial_offsets_px,
    get_affine_transform,
    patch_stats,
    seq_name,
    image_stem,
    summarize,
)


CONV_KEY_CANDIDATES = [
    "Lifting_net.motion_field_embed",
    "module.Lifting_net.motion_field_embed",
    "Lifting_net.flow_embed",
    "module.Lifting_net.flow_embed",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Feature-level diagnostic for optical-flow adaptive sampling. "
            "Unlike raw-flow diagnostics, this script samples the 128D motion "
            "feature map after the trained Conv2d(2->128) flow encoder."
        )
    )
    parser.add_argument("--labels", default="data/h36m_validation.pkl")
    parser.add_argument("--flow-dir", default="../H36M-Toolbox/flow_images_float")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument(
        "--conv-prefix",
        default="auto",
        help=(
            "Prefix of the Conv2d flow encoder in checkpoint, e.g. "
            "Lifting_net.motion_field_embed or Lifting_net.flow_embed. "
            "Use 'auto' to search common keys."
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
    parser.add_argument("--out", default="debug_vis/flow_feature_sampling_diagnosis.json")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def load_labels(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def action_name(action):
    idx = int(action) - 2
    if 0 <= idx < len(ACTION_NAMES):
        return ACTION_NAMES[idx]
    return str(action)


def joint_name(joint):
    if 0 <= int(joint) < len(JOINT_NAMES):
        return JOINT_NAMES[int(joint)]
    return str(joint)


def bilinear_sample_feature(feature_map, xy):
    """Sample HxWxC feature map at crop-space xy coordinates."""
    feature_map = np.asarray(feature_map, dtype=np.float32)
    coords = np.asarray(xy, dtype=np.float32)
    original_shape = coords.shape[:-1]
    flat = coords.reshape(-1, 2)
    h, w, c = feature_map.shape

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
        feature_map[y0, x0] * wa[:, None]
        + feature_map[y1, x0] * wb[:, None]
        + feature_map[y0, x1] * wc[:, None]
        + feature_map[y1, x1] * wd[:, None]
    )
    border_mask = (x0 == x1) | (y0 == y1)
    if np.any(border_mask):
        sampled[border_mask] = feature_map[
            np.rint(y[border_mask]).astype(np.int32),
            np.rint(x[border_mask]).astype(np.int32),
        ]
    return sampled.reshape(*original_shape, c)


def cosine_similarity(a, b, eps=1e-8):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
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
    return sorted(items, key=lambda item: (-float("inf") if item["mean"] is None else item["mean"]), reverse=reverse)[:top_k]


def summarize_rows(rows):
    raw_degradation = [row["raw_multi_minus_center"] for row in rows]
    feature_delta = [row["feature_multi_delta_l2"] for row in rows]
    feature_spread = [row["feature_sample_spread_l2"] for row in rows]
    return {
        "raw_center_error": summarize([row["raw_center_error"] for row in rows]),
        "raw_multi_error": summarize([row["raw_multi_error"] for row in rows]),
        "raw_multi_minus_center": summarize(raw_degradation),
        "feature_multi_delta_l2": summarize(feature_delta),
        "feature_sample_spread_l2": summarize(feature_spread),
        "feature_min_cos_to_center": summarize([row["feature_min_cos_to_center"] for row in rows]),
        "feature_mean_cos_to_center": summarize([row["feature_mean_cos_to_center"] for row in rows]),
        "frac_feature_cos_below_0_5": summarize([row["frac_feature_cos_below_0_5"] for row in rows]),
        "frac_feature_cos_below_0": summarize([row["frac_feature_cos_below_0"] for row in rows]),
        "correlations": {
            "feature_delta_vs_raw_degradation": corr(feature_delta, raw_degradation),
            "feature_spread_vs_raw_degradation": corr(feature_spread, raw_degradation),
            "feature_delta_vs_patch_flow_var": corr(feature_delta, [row["patch_flow_var"] for row in rows]),
            "feature_delta_vs_flow_edge": corr(feature_delta, [row["flow_edge"] for row in rows]),
            "feature_spread_vs_patch_flow_var": corr(feature_spread, [row["patch_flow_var"] for row in rows]),
            "feature_spread_vs_flow_edge": corr(feature_spread, [row["flow_edge"] for row in rows]),
        },
        "worst_joints_by_feature_delta": grouped_summary(rows, "joint", "feature_multi_delta_l2"),
        "worst_actions_by_feature_delta": grouped_summary(rows, "action", "feature_multi_delta_l2"),
        "worst_joints_by_raw_degradation": grouped_summary(rows, "joint", "raw_multi_minus_center"),
        "worst_actions_by_raw_degradation": grouped_summary(rows, "action", "raw_multi_minus_center"),
        "top_feature_delta_examples": sorted(rows, key=lambda row: row["feature_multi_delta_l2"], reverse=True)[:20],
    }


def feature_rows_for_frame(flow, feature_map, cur_cpn, gt_target, stats, offsets_px, meta):
    raw_center = bilinear_sample_flow(flow, cur_cpn)
    raw_xy = cur_cpn[:, None, None, :] + offsets_px[None, :, :, :]
    raw_sampled = bilinear_sample_flow(flow, raw_xy).reshape(cur_cpn.shape[0], -1, 2)
    raw_multi = raw_sampled.mean(axis=1)
    raw_center_error = np.linalg.norm(raw_center - gt_target, axis=1)
    raw_multi_error = np.linalg.norm(raw_multi - gt_target, axis=1)

    center = bilinear_sample_feature(feature_map, cur_cpn)
    feature_xy = cur_cpn[:, None, None, :] + offsets_px[None, :, :, :]
    sampled = bilinear_sample_feature(feature_map, feature_xy).reshape(cur_cpn.shape[0], -1, feature_map.shape[-1])
    multi = sampled.mean(axis=1)

    feature_delta = np.linalg.norm(multi - center, axis=1)
    spread = np.linalg.norm(sampled - center[:, None, :], axis=-1)
    cos_to_center = cosine_similarity(sampled, center[:, None, :])

    rows = []
    for joint_idx in range(cur_cpn.shape[0]):
        rows.append(
            {
                **meta,
                "joint": joint_idx,
                "joint_name": joint_name(joint_idx),
                "raw_center_error": float(raw_center_error[joint_idx]),
                "raw_multi_error": float(raw_multi_error[joint_idx]),
                "raw_multi_minus_center": float(raw_multi_error[joint_idx] - raw_center_error[joint_idx]),
                "feature_center_norm": float(np.linalg.norm(center[joint_idx])),
                "feature_multi_norm": float(np.linalg.norm(multi[joint_idx])),
                "feature_multi_delta_l2": float(feature_delta[joint_idx]),
                "feature_sample_spread_l2": float(spread[joint_idx].mean()),
                "feature_sample_spread_l2_p90": float(np.percentile(spread[joint_idx], 90)),
                "feature_min_cos_to_center": float(cos_to_center[joint_idx].min()),
                "feature_mean_cos_to_center": float(cos_to_center[joint_idx].mean()),
                "frac_feature_cos_below_0_5": float(np.mean(cos_to_center[joint_idx] < 0.5)),
                "frac_feature_cos_below_0": float(np.mean(cos_to_center[joint_idx] < 0.0)),
                "patch_mag_mean": float(stats[joint_idx, 0]),
                "patch_mag_std": float(stats[joint_idx, 1]),
                "patch_flow_var": float(stats[joint_idx, 2]),
                "flow_edge": float(stats[joint_idx, 3]),
            }
        )
    return rows


def load_flow_conv(checkpoint_path, conv_prefix):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to load checkpoint weights.") from exc

    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    raw = torch.load(str(checkpoint_path), map_location="cpu")
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    prefixes = CONV_KEY_CANDIDATES if conv_prefix == "auto" else [conv_prefix]
    for prefix in prefixes:
        weight_key = prefix + ".weight"
        bias_key = prefix + ".bias"
        if weight_key in state and bias_key in state:
            weight = state[weight_key].detach().cpu().float().numpy()
            bias = state[bias_key].detach().cpu().float().numpy()
            return prefix, weight, bias
    available = [key for key in state if key.endswith(".weight") and ("flow_embed" in key or "motion_field_embed" in key)]
    raise KeyError("Could not find flow Conv2d keys. Available candidates: {}".format(available[:20]))


def resolve_checkpoint_path(checkpoint_path):
    path = Path(checkpoint_path).expanduser()
    if path.exists():
        return path

    alternatives = []
    text = str(path)
    if "\uf03a" in text:
        alternatives.append(Path(text.replace("\uf03a", ":")))
    if ":" in text:
        alternatives.append(Path(text.replace(":", "\uf03a")))

    for alt in alternatives:
        if alt.exists():
            return alt

    parent = path.parent
    search_root = parent
    for _ in range(3):
        if search_root.exists():
            matches = sorted(search_root.glob("**/best_epoch.bin"))
            if matches:
                raise FileNotFoundError(
                    "Checkpoint not found: {}. Did you mean one of these?\n{}".format(
                        checkpoint_path,
                        "\n".join(str(match) for match in matches[:20]),
                    )
                )
        search_root = search_root.parent

    raise FileNotFoundError(
        "Checkpoint not found: {}. Tip: run `find logs_mfce_unified -path '*/checkpoints/best_epoch.bin' -print` "
        "and pass the printed path to --checkpoint.".format(checkpoint_path)
    )


def encode_flow_with_torch(flow, weight, bias):
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(flow.transpose(2, 0, 1)).unsqueeze(0).float()
    w = torch.from_numpy(weight).float()
    b = torch.from_numpy(bias).float()
    with torch.no_grad():
        y = F.conv2d(x, w, b, padding=1)
    return y.squeeze(0).permute(1, 2, 0).contiguous().numpy()


def synthetic_feature_map(flow):
    """Create a deterministic 128D feature map from a 2D flow map for self-test."""
    channels = []
    fx = flow[..., 0]
    fy = flow[..., 1]
    mag = np.sqrt(fx**2 + fy**2)
    for idx in range(32):
        scale = (idx + 1) / 32.0
        channels.extend([fx * scale, fy * scale, mag * scale, np.tanh(fx * scale)])
    return np.stack(channels, axis=-1).astype(np.float32)


def run_self_test():
    flow = np.zeros((64, 64, 2), dtype=np.float32)
    flow[:, :32, 0] = 1.0
    flow[:, 32:, 0] = -1.0
    feature_map = synthetic_feature_map(flow)
    cur_cpn = np.array([[31.2, 32.0]], dtype=np.float32)
    gt_target = np.array([[1.0, 0.0]], dtype=np.float32)
    stats = np.zeros((1, 4), dtype=np.float32)
    offsets = dce_initial_offsets_px(num_heads=4, num_samples=5, width=64, height=64)
    rows = feature_rows_for_frame(
        flow=flow,
        feature_map=feature_map,
        cur_cpn=cur_cpn,
        gt_target=gt_target,
        stats=stats,
        offsets_px=offsets,
        meta={"subject": 1, "action": 2, "action_name": "Directions-1", "subaction": 1, "camera_id": 0, "frame_id": 100},
    )
    print(json.dumps(rows[0], indent=2))
    if not (rows[0]["feature_multi_delta_l2"] > 0.1 and rows[0]["raw_multi_error"] > rows[0]["raw_center_error"]):
        raise SystemExit("Synthetic self-test failed.")
    print("Synthetic feature-level self-test passed.")


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required for real feature-level diagnosis.")

    conv_prefix, weight, bias = load_flow_conv(args.checkpoint, args.conv_prefix)
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

        flow_raw = np.load(flow_path).astype(np.float32)
        if flow_raw.ndim != 3 or flow_raw.shape[-1] != 2:
            raise ValueError("Expected flow [H,W,2], got {} at {}".format(flow_raw.shape, flow_path))

        flow = flow_raw.copy()
        if args.flow_clip > 0:
            flow = np.clip(flow, -args.flow_clip, args.flow_clip) / args.flow_clip

        cur_cpn = np.asarray(cur["joints_2d_cpn_crop"], dtype=np.float32)
        cur_gt = np.asarray(cur["joints_2d_gt_crop"], dtype=np.float32)
        prev_gt = np.asarray(prev["joints_2d_gt_crop"], dtype=np.float32)
        prev_to_raw = get_affine_transform(prev["center"], prev["scale"], output_size, inv=True)
        raw_to_cur = get_affine_transform(cur["center"], cur["scale"], output_size, inv=False)
        prev_gt_same = affine_apply(raw_to_cur, affine_apply(prev_to_raw, prev_gt))
        gt_target = prev_gt_same - cur_gt
        if args.flow_clip > 0:
            gt_target = np.clip(gt_target, -args.flow_clip, args.flow_clip) / args.flow_clip

        feature_map = encode_flow_with_torch(flow, weight, bias)
        stats = patch_stats(flow, cur_cpn, args.patch_radius)
        rows.extend(
            feature_rows_for_frame(
                flow=flow,
                feature_map=feature_map,
                cur_cpn=cur_cpn,
                gt_target=gt_target,
                stats=stats,
                offsets_px=offsets_px,
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
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
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
        "offset_geometry_px": {
            "max_abs_dx": float(np.max(np.abs(offsets_px[..., 0]))),
            "max_abs_dy": float(np.max(np.abs(offsets_px[..., 1]))),
            "offsets": offsets_px.reshape(-1, 2).round(4).tolist(),
        },
        "summary": summarize_rows(rows),
    }
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(out, file, indent=2)
    print("Wrote {}".format(out_path))
    print(json.dumps(out["meta"], indent=2))
    print(json.dumps(out["summary"]["correlations"], indent=2))
    print(json.dumps(out["summary"]["feature_multi_delta_l2"], indent=2))


if __name__ == "__main__":
    main()
