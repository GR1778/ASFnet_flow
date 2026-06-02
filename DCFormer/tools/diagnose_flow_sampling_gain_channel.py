#!/usr/bin/env python3
"""Check whether optical-flow sampling can plausibly become P1 gain.

This is an internal upper-bound diagnostic, not a train/eval entry point. It
intervenes at the CAPF flow-token slot and compares:

1. normal: the original +Flow feature token,
2. oracle_ring: the best local flow sample chosen by GT 2D motion,
3. gt_motion: a token built from GT short-term 2D motion.

The question is deliberately narrow: if even oracle or GT motion barely moves
P1, a learned sampling module has little downstream room to help in the
current fusion path. If oracle or GT motion helps in specific buckets, those
buckets tell us what a sampling module must target.
"""

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from diagnose_flow_mfce_sampling import ACTION_NAMES, JOINT_NAMES  # noqa: E402
from diagnose_flow_token_p1_upper import (  # noqa: E402
    IndexedDataset,
    build_label_map,
    build_val_dataset,
    forward_with_mode,
    get_flow_scale,
    global_scores,
    load_model,
    make_ring_offsets,
    normalize_ref,
    parse_float_list,
    preprocess_batch,
    sample_map,
    target_motion_px,
)
from mvn.datasets import utils as dataset_utils  # noqa: E402
from mvn.utils.cfg import config, update_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upper-bound and bucket analysis for flow-sampling P1 gain."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=512, help="0 means full validation set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--radii", default="2,4,6,8,12,16")
    parser.add_argument("--num-directions", type=int, default=16)
    parser.add_argument("--meaningful-p1-gain", type=float, default=0.2)
    parser.add_argument("--out", default="debug_vis/flow_sampling_gain_channel.json")
    return parser.parse_args()


def safe_name(names, index, prefix):
    index = int(index)
    if 0 <= index < len(names):
        return names[index]
    return "{}_{}".format(prefix, index)


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "p50": None, "p90": None, "count": 0}
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "count": int(arr.size),
    }


def safe_corr(x, y):
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


def bucket_value(value, edges, unit):
    value = float(value)
    lower = "-inf"
    for edge in edges:
        if value < edge:
            return "{}_to_{}{}".format(lower, edge, unit)
        lower = str(edge)
    return "{}_plus{}".format(lower, unit)


def label_meta(dataset, indices):
    items = [dataset.labels[int(idx)] for idx in indices.detach().cpu().numpy().tolist()]
    gt_crop = np.stack([np.asarray(item["joints_2d_gt_crop"], dtype=np.float32) for item in items], axis=0)
    actions = np.asarray([int(item["action"]) for item in items], dtype=np.int64)
    subjects = np.asarray([int(item["subject"]) for item in items], dtype=np.int64)
    return gt_crop, actions, subjects


def compute_flow_oracle_stats(flow_images, keypoints_2d_crop, target_norm, valid, offsets_px, flow_norm):
    device = flow_images.device
    ref = normalize_ref(keypoints_2d_crop, device)
    flow_map = flow_images.permute(0, 3, 1, 2).contiguous()
    _, _, h, w = flow_map.shape
    scale = torch.tensor([2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1)], device=device, dtype=ref.dtype)
    offsets = offsets_px.to(device=device, dtype=ref.dtype) * scale
    grid = ref.unsqueeze(2) + offsets.view(1, 1, -1, 2)

    samples = sample_map(flow_map, grid)
    errors = torch.linalg.norm(samples - target_norm.unsqueeze(2), dim=-1)
    center_error = errors[:, :, 0]
    oracle_error, best_idx = errors.min(dim=-1)
    gain = center_error - oracle_error

    multiplier = float(flow_norm) if flow_norm is not None and flow_norm > 0 else 1.0
    valid_joint = valid.view(-1, 1).expand_as(center_error)
    center_error = torch.where(valid_joint, center_error * multiplier, torch.full_like(center_error, float("nan")))
    oracle_error = torch.where(valid_joint, oracle_error * multiplier, torch.full_like(oracle_error, float("nan")))
    gain = torch.where(valid_joint, gain * multiplier, torch.full_like(gain, float("nan")))

    return {
        "center_flow_error": center_error.detach().cpu().numpy(),
        "oracle_flow_error": oracle_error.detach().cpu().numpy(),
        "oracle_flow_gain": gain.detach().cpu().numpy(),
        "oracle_best_idx": best_idx.detach().cpu().numpy(),
    }


def per_joint_error_mm(pred, gt):
    pred = pred.squeeze(1)
    gt = gt.squeeze(1)
    return torch.linalg.norm(pred - gt, dim=-1).detach().cpu().numpy() * 1000.0


def group_rows(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return grouped


def summarize_group(items):
    normal = [row["normal_err_mm"] for row in items]
    oracle = [row["oracle_ring_err_mm"] for row in items]
    gt_motion = [row["gt_motion_err_mm"] for row in items]
    oracle_gain_3d = [row["normal_minus_oracle_ring_mm"] for row in items]
    gt_gain_3d = [row["normal_minus_gt_motion_mm"] for row in items]
    flow_gain = [row["oracle_flow_gain_px"] for row in items]
    return {
        "count": len(items),
        "normal_err_mm": summarize(normal),
        "oracle_ring_err_mm": summarize(oracle),
        "gt_motion_err_mm": summarize(gt_motion),
        "normal_minus_oracle_ring_mm": summarize(oracle_gain_3d),
        "normal_minus_gt_motion_mm": summarize(gt_gain_3d),
        "oracle_flow_gain_px": summarize(flow_gain),
        "flow_gain_vs_oracle_3d_gain_corr": safe_corr(flow_gain, oracle_gain_3d),
    }


def summarize_by(rows, key):
    grouped = group_rows(rows, key)
    return {str(name): summarize_group(items) for name, items in sorted(grouped.items(), key=lambda kv: str(kv[0]))}


def verdict(summary, threshold):
    normal = summary["modes"]["normal"]["P1"]
    oracle = summary["modes"]["oracle_ring"]["P1"]
    gt_motion = summary["modes"]["gt_motion"]["P1"]
    oracle_gain = normal - oracle
    gt_gain = normal - gt_motion

    if oracle_gain >= threshold:
        label = "sampling_channel_exists"
        reason = "oracle sampled flow improves P1 by at least the threshold."
    elif gt_gain >= threshold:
        label = "candidate_or_selector_bottleneck"
        reason = "GT motion improves P1, but oracle local sampling does not."
    else:
        label = "weak_downstream_flow_channel"
        reason = "even GT motion has little P1 effect in this fusion path."
    return {
        "label": label,
        "threshold_mm": float(threshold),
        "normal_minus_oracle_ring_P1_mm": float(oracle_gain),
        "normal_minus_gt_motion_P1_mm": float(gt_gain),
        "reason": reason,
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    update_config(args.config)
    config.val.flip_test = False
    config.val.batch_size = args.batch_size
    config.val.num_workers = args.num_workers

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = build_val_dataset()
    label_map = build_label_map(config.dataset.val_labels_path)
    loader = DataLoader(
        IndexedDataset(dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=dataset_utils.worker_init_fn,
    )
    model = load_model(args.checkpoint, device)
    offsets_px = make_ring_offsets(parse_float_list(args.radii), args.num_directions)
    flow_clip, flow_norm = get_flow_scale()
    output_size = (args.width, args.height)
    modes = ["normal", "oracle_ring", "gt_motion"]

    all_gt = []
    all_pred = {mode: [] for mode in modes}
    rows = []
    valid_count = 0
    total_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break

            images, gt_3d, keypoints_2d, keypoints_2d_crop, flow_images, indices = preprocess_batch(batch, device)
            target_px_np, valid_np = target_motion_px(dataset, label_map, indices, args.frame_gap, output_size)
            target_px = torch.from_numpy(target_px_np).to(device=device, dtype=flow_images.dtype)
            valid = torch.from_numpy(valid_np).to(device=device)

            target_norm = target_px.clamp(-flow_clip, flow_clip) if flow_clip is not None and flow_clip > 0 else target_px
            if flow_norm is not None and flow_norm > 0:
                target_norm = target_norm / float(flow_norm)

            gt_crop_np, action_np, subject_np = label_meta(dataset, indices)
            cpn_crop_np = keypoints_2d_crop.detach().cpu().numpy()
            anchor_error = np.linalg.norm(cpn_crop_np - gt_crop_np, axis=-1)
            target_motion_mag = np.linalg.norm(target_px_np, axis=-1)

            flow_stats = compute_flow_oracle_stats(
                flow_images,
                keypoints_2d_crop,
                target_norm,
                valid,
                offsets_px,
                flow_norm,
            )

            batch_pred = {}
            for mode in modes:
                pred = forward_with_mode(
                    model,
                    images,
                    keypoints_2d,
                    keypoints_2d_crop,
                    flow_images,
                    mode,
                    target_norm,
                    valid,
                    offsets_px,
                )
                batch_pred[mode] = pred
                all_pred[mode].append(pred.detach().cpu())

            all_gt.append(gt_3d.detach().cpu())
            total_count += int(valid.numel())
            valid_count += int(valid.sum().item())

            err = {mode: per_joint_error_mm(batch_pred[mode], gt_3d) for mode in modes}
            batch_size, num_joints = err["normal"].shape
            for b in range(batch_size):
                action_idx = int(action_np[b]) - 2
                action_name = safe_name(ACTION_NAMES, action_idx, "action")
                for joint in range(num_joints):
                    row = {
                        "subject": int(subject_np[b]),
                        "action": int(action_np[b]),
                        "action_name": action_name,
                        "joint": int(joint),
                        "joint_name": safe_name(JOINT_NAMES, joint, "joint"),
                        "valid_motion_target": bool(valid_np[b]),
                        "anchor_error_px": float(anchor_error[b, joint]),
                        "target_motion_mag_px": float(target_motion_mag[b, joint]),
                        "center_flow_error_px": float(flow_stats["center_flow_error"][b, joint]),
                        "oracle_flow_error_px": float(flow_stats["oracle_flow_error"][b, joint]),
                        "oracle_flow_gain_px": float(flow_stats["oracle_flow_gain"][b, joint]),
                        "oracle_best_idx": int(flow_stats["oracle_best_idx"][b, joint]),
                        "normal_err_mm": float(err["normal"][b, joint]),
                        "oracle_ring_err_mm": float(err["oracle_ring"][b, joint]),
                        "gt_motion_err_mm": float(err["gt_motion"][b, joint]),
                    }
                    row["normal_minus_oracle_ring_mm"] = row["normal_err_mm"] - row["oracle_ring_err_mm"]
                    row["normal_minus_gt_motion_mm"] = row["normal_err_mm"] - row["gt_motion_err_mm"]
                    row["anchor_error_bucket"] = bucket_value(row["anchor_error_px"], [1, 2, 4, 8], "px")
                    row["target_motion_bucket"] = bucket_value(row["target_motion_mag_px"], [0.5, 1, 2, 4, 8], "px")
                    row["center_flow_error_bucket"] = bucket_value(row["center_flow_error_px"], [0.5, 1, 2, 4, 8], "px")
                    row["oracle_flow_gain_bucket"] = bucket_value(row["oracle_flow_gain_px"], [0.1, 0.5, 1, 2, 4], "px")
                    rows.append(row)

            print(
                "processed batch {}/{}".format(
                    batch_idx + 1,
                    "full" if not args.max_batches else args.max_batches,
                ),
                flush=True,
            )

    gt = torch.cat(all_gt, dim=0)
    pred_all = {mode: torch.cat(chunks, dim=0) for mode, chunks in all_pred.items()}

    summary = {
        "meta": {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "samples": int(gt.shape[0]),
            "joint_rows": len(rows),
            "target_valid_samples": int(valid_count),
            "target_total_samples": int(total_count),
            "radii": parse_float_list(args.radii),
            "num_directions": args.num_directions,
            "flow_clip": None if flow_clip is None else float(flow_clip),
            "flow_norm": None if flow_norm is None else float(flow_norm),
            "max_batches": int(args.max_batches),
        },
        "modes": {},
    }
    for mode in modes:
        p1, p2 = global_scores(gt, pred_all[mode])
        summary["modes"][mode] = {"P1": p1, "P2": p2}

    summary["verdict"] = verdict(summary, args.meaningful_p1_gain)
    summary["correlations"] = {
        "flow_oracle_gain_vs_oracle_3d_gain": safe_corr(
            [row["oracle_flow_gain_px"] for row in rows],
            [row["normal_minus_oracle_ring_mm"] for row in rows],
        ),
        "flow_oracle_gain_vs_gt_motion_3d_gain": safe_corr(
            [row["oracle_flow_gain_px"] for row in rows],
            [row["normal_minus_gt_motion_mm"] for row in rows],
        ),
        "center_flow_error_vs_normal_3d_error": safe_corr(
            [row["center_flow_error_px"] for row in rows],
            [row["normal_err_mm"] for row in rows],
        ),
    }
    summary["overall_joint_rows"] = summarize_group(rows)
    summary["by_joint"] = summarize_by(rows, "joint_name")
    summary["by_action"] = summarize_by(rows, "action_name")
    summary["by_anchor_error_bucket"] = summarize_by(rows, "anchor_error_bucket")
    summary["by_target_motion_bucket"] = summarize_by(rows, "target_motion_bucket")
    summary["by_center_flow_error_bucket"] = summarize_by(rows, "center_flow_error_bucket")
    summary["by_oracle_flow_gain_bucket"] = summarize_by(rows, "oracle_flow_gain_bucket")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"modes": summary["modes"], "verdict": summary["verdict"]}, indent=2))
    print("wrote {}".format(output_path))


if __name__ == "__main__":
    main()
