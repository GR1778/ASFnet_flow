import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from mvn import datasets
from mvn.datasets import utils as dataset_utils
from mvn.utils.cfg import config, update_config


H36M_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num_batches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--strip_samples", type=int, default=9)
    parser.add_argument("--strip_width_px", type=float, default=4.0)
    parser.add_argument("--crossing_threshold_px", type=float, default=8.0)
    parser.add_argument("--output_dir", default="module_feasibility_analysis")
    return parser.parse_args()


def safe_corr(x, y, fn):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return np.nan
    return float(fn(x, y)[0])


def normalize_crop_coords(kp_crop, height, width):
    coords = kp_crop[..., :2].float()
    if coords.detach().abs().max() <= 2.5:
        return coords
    scale = torch.tensor([width / 2.0, height / 2.0], device=coords.device, dtype=coords.dtype)
    return coords / scale - 1.0


def sample_depth(depth, coords_norm):
    grid = coords_norm.view(coords_norm.shape[0], -1, 1, 2)
    sampled = F.grid_sample(depth.unsqueeze(1), grid, align_corners=True, padding_mode="border")
    return sampled.squeeze(1).squeeze(-1).view(*coords_norm.shape[:-1])


def build_bone_strip_coords(joints_norm, height, width, samples, strip_width_px):
    b = joints_norm.shape[0]
    t = torch.linspace(0.0, 1.0, samples, device=joints_norm.device, dtype=joints_norm.dtype)
    offsets = torch.tensor([-strip_width_px, 0.0, strip_width_px], device=joints_norm.device, dtype=joints_norm.dtype)
    all_coords = []
    for parent, child in H36M_EDGES:
        p0 = joints_norm[:, parent]
        p1 = joints_norm[:, child]
        line = p0[:, None, :] * (1.0 - t[None, :, None]) + p1[:, None, :] * t[None, :, None]
        direction = p1 - p0
        perp = torch.stack([-direction[:, 1], direction[:, 0]], dim=-1)
        perp = perp / perp.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        px_to_norm = torch.tensor([2.0 / width, 2.0 / height], device=joints_norm.device, dtype=joints_norm.dtype)
        strip = line[:, :, None, :] + perp[:, None, None, :] * offsets[None, None, :, None] * px_to_norm
        all_coords.append(strip.reshape(b, samples * len(offsets), 2))
    return torch.stack(all_coords, dim=1)


def pairwise_sign_metrics(pred, target):
    pred = np.asarray(pred)
    target = np.asarray(target)
    n, j = pred.shape
    mask = ~np.eye(j, dtype=bool)
    pred_delta = pred[:, :, None] - pred[:, None, :]
    target_delta = target[:, :, None] - target[:, None, :]
    valid = mask[None] & (np.abs(target_delta) > 1e-8)
    direct = (np.sign(pred_delta[valid]) == np.sign(target_delta[valid])).mean()
    inverse = (np.sign(-pred_delta[valid]) == np.sign(target_delta[valid])).mean()
    return float(direct), float(inverse), float(max(direct, inverse))


def segment_distance(a0, a1, b0, b1):
    def point_segment_distance(p, s0, s1):
        v = s1 - s0
        denom = float(np.dot(v, v))
        if denom < 1e-8:
            return float(np.linalg.norm(p - s0))
        t = np.clip(float(np.dot(p - s0, v) / denom), 0.0, 1.0)
        return float(np.linalg.norm(p - (s0 + t * v)))

    return min(
        point_segment_distance(a0, b0, b1),
        point_segment_distance(a1, b0, b1),
        point_segment_distance(b0, a0, a1),
        point_segment_distance(b1, a0, a1),
    )


def collect_batches(args):
    update_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    val_dataset = eval("datasets." + config.dataset.val_dataset)(
        root=config.dataset.root,
        pred_results_path=config.val.pred_results_path,
        depth_image_path=config.dataset.depth_image_path,
        train=False,
        test=True,
        image_shape=config.model.image_shape,
        labels_path=config.dataset.val_labels_path,
        with_damaged_actions=config.val.with_damaged_actions,
        retain_every_n_frames_in_test=config.val.retain_every_n_frames_in_test,
        scale_bbox=config.val.scale_bbox,
        kind=config.kind,
        undistort_images=config.val.undistort_images,
        ignore_cameras=config.val.ignore_cameras,
        crop=config.val.crop,
        erase=config.val.erase,
        rank=None,
        world_size=None,
        data_format=config.dataset.data_format,
        frame=1,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=True,
    )

    depth_list, kp_crop_list, gt_list = [], [], []
    prefetcher = dataset_utils.data_prefetcher(val_loader, device, is_train=False, flip_test=False)
    batch = prefetcher.next()
    batch_idx = 0
    while batch is not None and batch_idx < args.num_batches:
        _images, kp3d_gt, _kp2d, kp2d_crop, depth = batch
        depth_list.append(depth.detach().float().cpu())
        kp_crop_list.append(kp2d_crop.detach().float().cpu())
        gt_list.append(kp3d_gt[:, 0].detach().float().cpu())
        batch_idx += 1
        print(f"[collect] batch {batch_idx}/{args.num_batches}")
        batch = prefetcher.next()

    return (
        torch.cat(depth_list, dim=0).to(device),
        torch.cat(kp_crop_list, dim=0).to(device),
        torch.cat(gt_list, dim=0).to(device),
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    depth, kp_crop, gt3d = collect_batches(args)
    n, height, width = depth.shape
    joints_norm = normalize_crop_coords(kp_crop, height, width).clamp(-1.2, 1.2)
    joints_px = (joints_norm + 1.0) * torch.tensor([width / 2.0, height / 2.0], device=depth.device)
    gt_z = gt3d[..., 2]
    gt_z_norm = (gt_z - gt_z.mean(dim=1, keepdim=True)) / gt_z.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)

    joint_depth = sample_depth(depth, joints_norm).detach().cpu().numpy()
    gt_z_np = gt_z_norm.detach().cpu().numpy()
    joint_direct, joint_inverse, joint_best = pairwise_sign_metrics(joint_depth, gt_z_np)
    joint_spearman = np.nanmean([safe_corr(joint_depth[i], gt_z_np[i], spearmanr) for i in range(n)])
    joint_pearson_abs = np.nanmean([abs(safe_corr(joint_depth[i], gt_z_np[i], pearsonr)) for i in range(n)])

    strip_coords = build_bone_strip_coords(
        joints_norm, height, width, args.strip_samples, args.strip_width_px
    ).clamp(-1.2, 1.2)
    strip_depth = sample_depth(depth, strip_coords.reshape(n, -1, 2)).view(n, len(H36M_EDGES), -1)
    strip_np = strip_depth.detach().cpu().numpy()
    bone_depth_mean = strip_np.mean(axis=-1)
    bone_depth_range = strip_np.max(axis=-1) - strip_np.min(axis=-1)
    bone_depth_std = strip_np.std(axis=-1)

    gt_z_bone = []
    gt_z_span = []
    for parent, child in H36M_EDGES:
        z0 = gt_z_norm[:, parent]
        z1 = gt_z_norm[:, child]
        gt_z_bone.append(((z0 + z1) * 0.5).detach().cpu().numpy())
        gt_z_span.append((z1 - z0).abs().detach().cpu().numpy())
    gt_z_bone = np.stack(gt_z_bone, axis=1)
    gt_z_span = np.stack(gt_z_span, axis=1)
    bone_direct, bone_inverse, bone_best = pairwise_sign_metrics(bone_depth_mean, gt_z_bone)
    bone_spearman = np.nanmean([safe_corr(bone_depth_mean[i], gt_z_bone[i], spearmanr) for i in range(n)])
    continuity_span_corr = safe_corr(bone_depth_range.reshape(-1), gt_z_span.reshape(-1), spearmanr)
    continuity_std_corr = safe_corr(bone_depth_std.reshape(-1), gt_z_span.reshape(-1), spearmanr)

    crossing_pred_delta, crossing_gt_delta, crossing_ranges = [], [], []
    pair_depth_delta, pair_gt_delta, pair_dist_px = [], [], []
    joints_px_np = joints_px.detach().cpu().numpy()
    joint_depth_np = joint_depth
    gt_z_np = gt_z_np
    for sample_idx in range(n):
        for i in range(17):
            for j in range(i + 1, 17):
                pair_depth_delta.append(joint_depth_np[sample_idx, i] - joint_depth_np[sample_idx, j])
                pair_gt_delta.append(gt_z_np[sample_idx, i] - gt_z_np[sample_idx, j])
                pair_dist_px.append(float(np.linalg.norm(joints_px_np[sample_idx, i] - joints_px_np[sample_idx, j])))

    for sample_idx in range(n):
        for e1, (a0, a1) in enumerate(H36M_EDGES):
            for e2, (b0, b1) in enumerate(H36M_EDGES):
                if e2 <= e1 or len({a0, a1, b0, b1}) < 4:
                    continue
                dist = segment_distance(
                    joints_px_np[sample_idx, a0], joints_px_np[sample_idx, a1],
                    joints_px_np[sample_idx, b0], joints_px_np[sample_idx, b1],
                )
                if dist <= args.crossing_threshold_px:
                    crossing_pred_delta.append(bone_depth_mean[sample_idx, e1] - bone_depth_mean[sample_idx, e2])
                    crossing_gt_delta.append(gt_z_bone[sample_idx, e1] - gt_z_bone[sample_idx, e2])
                    crossing_ranges.append(0.5 * (bone_depth_range[sample_idx, e1] + bone_depth_range[sample_idx, e2]))

    crossing_pred_delta = np.asarray(crossing_pred_delta)
    crossing_gt_delta = np.asarray(crossing_gt_delta)
    pair_depth_delta = np.asarray(pair_depth_delta)
    pair_gt_delta = np.asarray(pair_gt_delta)
    pair_dist_px = np.asarray(pair_dist_px)
    proximity_stats = {}
    for threshold in [16.0, 24.0, 32.0, 48.0, 64.0]:
        near = pair_dist_px <= threshold
        if near.sum() == 0:
            continue
        direct = (np.sign(pair_depth_delta[near]) == np.sign(pair_gt_delta[near])).mean()
        inverse = (np.sign(-pair_depth_delta[near]) == np.sign(pair_gt_delta[near])).mean()
        proximity_stats[str(int(threshold)) + "px"] = {
            "num_pairs": int(near.sum()),
            "best_order_acc": float(max(direct, inverse)),
            "delta_spearman_abs": float(abs(safe_corr(pair_depth_delta[near], pair_gt_delta[near], spearmanr))),
        }

    if len(crossing_pred_delta) > 0:
        crossing_direct = (np.sign(crossing_pred_delta) == np.sign(crossing_gt_delta)).mean()
        crossing_inverse = (np.sign(-crossing_pred_delta) == np.sign(crossing_gt_delta)).mean()
        crossing_best = float(max(crossing_direct, crossing_inverse))
        crossing_corr = abs(safe_corr(crossing_pred_delta, crossing_gt_delta, spearmanr))
    else:
        crossing_direct = crossing_inverse = crossing_best = crossing_corr = np.nan

    summary = {
        "num_samples": int(n),
        "joint_depth_signal": {
            "pairwise_sign_acc_direct": joint_direct,
            "pairwise_sign_acc_inverse": joint_inverse,
            "pairwise_sign_acc_best_orientation": joint_best,
            "within_sample_spearman_mean": float(joint_spearman),
            "within_sample_abs_pearson_mean": float(joint_pearson_abs),
            "verdict": "strong" if joint_best >= 0.68 else "moderate" if joint_best >= 0.60 else "weak",
        },
        "bone_surface_signal": {
            "pairwise_bone_order_acc_direct": bone_direct,
            "pairwise_bone_order_acc_inverse": bone_inverse,
            "pairwise_bone_order_acc_best_orientation": bone_best,
            "within_sample_bone_spearman_mean": float(bone_spearman),
            "depth_range_vs_gt_bone_z_span_spearman": float(continuity_span_corr),
            "depth_std_vs_gt_bone_z_span_spearman": float(continuity_std_corr),
            "verdict": "strong" if bone_best >= 0.68 else "moderate" if bone_best >= 0.60 else "weak",
        },
        "occlusion_crossing_signal": {
            "num_candidate_crossings": int(len(crossing_pred_delta)),
            "pairwise_order_acc_direct": float(crossing_direct),
            "pairwise_order_acc_inverse": float(crossing_inverse),
            "pairwise_order_acc_best_orientation": float(crossing_best),
            "delta_spearman_abs": float(crossing_corr),
            "mean_depth_range_at_crossings": float(np.mean(crossing_ranges)) if len(crossing_ranges) else np.nan,
            "verdict": "strong" if crossing_best >= 0.68 else "moderate" if crossing_best >= 0.60 else "weak",
        },
        "near_joint_pair_signal": proximity_stats,
        "scheme_recommendation": {
            "skeleton_depth_surface_reasoner": "high" if bone_best >= 0.60 else "medium",
            "depth_pose_cycle_consistency": "high" if joint_best >= 0.60 else "medium",
            "occlusion_aware_relative_depth_graph": "high" if len(crossing_pred_delta) >= 50 and crossing_best >= 0.60 else "medium" if len(crossing_pred_delta) >= 20 else "low",
        },
    }

    out_json = os.path.join(args.output_dir, "summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[done] wrote {out_json}")


if __name__ == "__main__":
    main()
