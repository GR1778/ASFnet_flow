#!/usr/bin/env python3
"""Probe whether depth discontinuities explain ASFNet depth-branch failures.

This diagnostic is intentionally narrow. It does not ask whether depth, AMS, or
UDE are useful; ASFNet already establishes that. It asks whether a local
geometric signal that RGB cannot directly provide--depth discontinuity--is a
plausible missing input for depth-branch evidence selection.

The script hooks the official ASFNet DGLifting AMS blocks and measures:
- joint-level discontinuity sampled at the 2D joint reference point;
- candidate-level discontinuity sampled at each AMS depth candidate;
- whether discontinuity correlates with depth errors and AMS oracle gaps;
- whether discontinuity is non-redundant with AMS weights and UDE uncertainty;
- whether adding candidate discontinuity to AMS weights improves oracle-candidate
  ranking in a simple diagnostic logistic probe.
"""

import argparse
import json
import os
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvn import datasets
from mvn.models.DGPose import DepthGuidedPose
from mvn.utils.cfg import config, update_config, update_dir


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/human36m/human36m_single.yaml")
    parser.add_argument("--checkpoint", default="checkpoint/h36m_v2b.bin")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional hard cap on dataset items.")
    parser.add_argument("--depth-image-path", default="", help="Override config.dataset.depth_image_path.")
    parser.add_argument("--depth-format", default="", choices=("", "image", "npy"))
    parser.add_argument("--output", default="debug_vis/depth_discontinuity_evidence_probe.json")
    parser.add_argument("--save-arrays", default="", help="Optional .npz path for raw per-joint/per-candidate arrays.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logistic-steps", type=int, default=200)
    parser.add_argument("--logistic-lr", type=float, default=0.2)
    parser.add_argument("--max-logistic-points", type=int, default=250000)
    return parser.parse_args()


class CaptureState:
    def __init__(self):
        self.depth_images = None
        self.depth_edges = None
        self.layers = []

    def reset(self, depth_images, depth_edges):
        self.depth_images = depth_images
        self.depth_edges = depth_edges
        self.layers = []


def patch_ams_blocks(lifting_net, capture):
    for layer_idx, block in enumerate(lifting_net.RGBD_Extraction):
        original_forward = block.forward

        def forward_with_capture(self, x, ref, features_list, _layer_idx=layer_idx, _original_forward=original_forward):
            x_0, x_levels = x[:, :1], x[:, 1:]
            b, levels, joints, dim = x_levels.shape
            residual = x_levels
            normed = self.norm1(x_levels + x_0)

            weights = self.attention_weights(normed).view(
                b, levels, joints, self.num_heads, self.num_samples
            )
            weights = F.softmax(weights, dim=-1).unsqueeze(-1)
            offsets = self.sampling_offsets(normed).reshape(
                b, levels, joints, self.num_heads * self.num_samples, 2
            ).tanh()
            pos = offsets + ref.view(b, 1, joints, 1, -1)

            depth_level = levels - 1
            if capture.depth_images is not None:
                depth_grid = pos[:, depth_level]
                sampled_depth = F.grid_sample(
                    capture.depth_images.unsqueeze(1),
                    depth_grid,
                    padding_mode="border",
                    align_corners=True,
                ).squeeze(1)
                sampled_edge = F.grid_sample(
                    capture.depth_edges.unsqueeze(1),
                    depth_grid,
                    padding_mode="border",
                    align_corners=True,
                ).squeeze(1)
                capture.layers.append(
                    {
                        "layer": _layer_idx,
                        "pos_depth": depth_grid.detach(),
                        "weights_depth": weights[:, depth_level, ..., 0].detach(),
                        "sampled_depth": sampled_depth.detach(),
                        "sampled_edge": sampled_edge.detach(),
                    }
                )

            features_sampled = [
                F.grid_sample(features, pos[:, idx], padding_mode="border", align_corners=True)
                .permute(0, 2, 3, 1)
                .contiguous()
                for idx, features in enumerate(features_list)
            ]
            features_sampled = [embed(features_sampled[idx]) for idx, embed in enumerate(self.embed_proj)]
            features_sampled = torch.stack(features_sampled, dim=1)
            features_sampled = (
                weights * features_sampled.view(b, levels, joints, self.num_heads, self.num_samples, -1)
            ).sum(dim=-2).view(b, levels, joints, -1)

            out = residual + self.drop_path(features_sampled)
            out = out + self.drop_path(self.mlp(self.norm2(out)))
            return torch.cat([x_0, out], dim=1)

        block.forward = types.MethodType(forward_with_capture, block)


def strip_module_prefix(state_dict):
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def load_model(args, device):
    model = DepthGuidedPose(config, device)
    raw = torch.load(args.checkpoint, map_location="cpu")
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    state = strip_module_prefix(state)
    ret = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Missing keys: {len(ret.missing_keys)} | Unexpected keys: {len(ret.unexpected_keys)}")
    model.to(device).eval()
    return model


def build_dataset(args):
    if args.depth_image_path:
        config.dataset.depth_image_path = args.depth_image_path
    if args.depth_format:
        config.dataset.depth_format = args.depth_format

    common = dict(
        root=config.dataset.root,
        pred_results_path=(config.train.pred_results_path if args.split == "train" else config.val.pred_results_path),
        depth_image_path=config.dataset.depth_image_path,
        image_shape=config.model.image_shape,
        labels_path=(config.dataset.train_labels_path if args.split == "train" else config.dataset.val_labels_path),
        with_damaged_actions=(config.train.with_damaged_actions if args.split == "train" else config.val.with_damaged_actions),
        scale_bbox=(config.train.scale_bbox if args.split == "train" else config.val.scale_bbox),
        kind=config.kind,
        undistort_images=(config.train.undistort_images if args.split == "train" else config.val.undistort_images),
        ignore_cameras=(config.train.ignore_cameras if args.split == "train" else config.val.ignore_cameras),
        crop=(config.train.crop if args.split == "train" else config.val.crop),
        erase=(config.train.erase if args.split == "train" else config.val.erase),
        data_format=config.dataset.data_format,
        depth_format=config.dataset.depth_format,
    )
    if args.split == "train":
        dataset = datasets.multiview_human36m(train=True, test=False, **common)
    else:
        dataset = datasets.human36m(
            train=False,
            test=True,
            retain_every_n_frames_in_test=config.val.retain_every_n_frames_in_test,
            **common,
        )

    if args.max_samples and args.max_samples < len(dataset):
        dataset.labels = dataset.labels[: args.max_samples]
        if hasattr(dataset, "labels_action_idx"):
            dataset.labels_action_idx = dataset.labels_action_idx[: args.max_samples]
        if hasattr(dataset, "labels_subject_idx"):
            dataset.labels_subject_idx = dataset.labels_subject_idx[: args.max_samples]
        if hasattr(dataset, "video_idx"):
            dataset.video_idx = dataset.video_idx[: args.max_samples]
    return dataset


def preprocess_batch(batch, device):
    images, gt_3d, keypoints_2d, keypoints_2d_crop, depth_images = batch
    images = images.to(device, non_blocking=True).float()
    gt_3d = gt_3d.to(device, non_blocking=True).float()
    keypoints_2d = keypoints_2d.to(device, non_blocking=True).float()
    keypoints_2d_crop = keypoints_2d_crop.to(device, non_blocking=True).float()
    depth_images = depth_images.to(device, non_blocking=True)

    if torch.is_floating_point(depth_images):
        depth_images = depth_images.float()
    else:
        depth_images = depth_images.float() / 255.0

    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)
    images = torch.flip(images, [-1])
    images = (images / 255.0 - mean) / std

    gt_3d[:, :, 1:] -= gt_3d[:, :, :1]
    gt_3d[:, :, 0] = 0
    return images, gt_3d, keypoints_2d, keypoints_2d_crop, depth_images


def compute_depth_edges(depth_images, eps=1e-6):
    kernel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=depth_images.device,
        dtype=depth_images.dtype,
    ).unsqueeze(0) / 8.0
    kernel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=depth_images.device,
        dtype=depth_images.dtype,
    ).unsqueeze(0) / 8.0
    depth = depth_images.unsqueeze(1)
    grad_x = F.conv2d(depth, kernel_x, padding=1)
    grad_y = F.conv2d(depth, kernel_y, padding=1)
    edge = torch.sqrt(grad_x.square() + grad_y.square() + eps).squeeze(1)

    flat = edge.flatten(1)
    median = flat.median(dim=1).values.view(-1, 1, 1)
    mad = (flat - median.flatten(1)).abs().median(dim=1).values.view(-1, 1, 1)
    robust_edge = (edge - median) / mad.clamp_min(eps)
    return edge, robust_edge.clamp_min(0.0)


def make_ref_grid(keypoints_2d_crop, device):
    ref = keypoints_2d_crop.clone()
    ref[..., :2] /= torch.tensor([192 // 2, 256 // 2], device=device)
    ref[..., :2] -= torch.tensor([1, 1], device=device)
    return ref


def fit_affine(raw_ref, gt_z, eps=1e-6):
    raw_mean = raw_ref.mean(dim=1, keepdim=True)
    gt_mean = gt_z.mean(dim=1, keepdim=True)
    raw_centered = raw_ref - raw_mean
    gt_centered = gt_z - gt_mean
    scale = (raw_centered * gt_centered).sum(dim=1, keepdim=True) / (
        raw_centered.square().sum(dim=1, keepdim=True).clamp_min(eps)
    )
    bias = gt_mean - scale * raw_mean
    pred_ref = scale * raw_ref + bias
    ss_res = (gt_z - pred_ref).square().sum(dim=1)
    ss_tot = (gt_z - gt_mean).square().sum(dim=1).clamp_min(eps)
    r2 = 1.0 - ss_res / ss_tot
    return scale, bias, r2


def pearson_corr(x, y, eps=1e-8):
    x = x.flatten()
    y = y.flatten()
    mask = torch.isfinite(x) & torch.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.numel() < 2:
        return torch.tensor(float("nan"), device=x.device)
    x = x - x.mean()
    y = y - y.mean()
    return (x * y).mean() / (x.std(unbiased=False).clamp_min(eps) * y.std(unbiased=False).clamp_min(eps))


def spearman_corr(x, y):
    x = x.flatten()
    y = y.flatten()
    mask = torch.isfinite(x) & torch.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.numel() < 2:
        return torch.tensor(float("nan"), device=x.device)
    x_rank = torch.argsort(torch.argsort(x)).float()
    y_rank = torch.argsort(torch.argsort(y)).float()
    return pearson_corr(x_rank, y_rank)


def update_meter(meter, key, value):
    if isinstance(value, torch.Tensor):
        value = value.detach().float().mean().item()
    meter[key].append(float(value))


def append_array(arrays, key, value):
    arrays[key].append(value.detach().float().cpu())


def edge_bucket_metrics(meter, prefix, edge, weighted_err, oracle_gap):
    flat_edge = edge.flatten()
    flat_weighted = weighted_err.flatten()
    flat_gap = oracle_gap.flatten()
    if flat_edge.numel() < 8:
        return
    q25, q75 = torch.quantile(flat_edge, torch.tensor([0.25, 0.75], device=edge.device, dtype=edge.dtype))
    low = flat_edge <= q25
    high = flat_edge >= q75
    if low.any() and high.any():
        update_meter(meter, f"{prefix}/high_edge_weighted_err", flat_weighted[high])
        update_meter(meter, f"{prefix}/low_edge_weighted_err", flat_weighted[low])
        update_meter(meter, f"{prefix}/high_minus_low_weighted_err", flat_weighted[high].mean() - flat_weighted[low].mean())
        update_meter(meter, f"{prefix}/high_edge_oracle_gap", flat_gap[high])
        update_meter(meter, f"{prefix}/low_edge_oracle_gap", flat_gap[low])
        update_meter(meter, f"{prefix}/high_minus_low_oracle_gap", flat_gap[high].mean() - flat_gap[low].mean())


def compute_layer_metrics(meter, arrays, captured, ref_raw, ref_edge, gt_z, pred_z=None, ude_uncer=None):
    scale, bias, r2 = fit_affine(ref_raw, gt_z)
    ref_calibrated = scale * ref_raw + bias
    ref_err = (ref_calibrated - gt_z).abs()

    if pred_z is not None:
        pose_z_err = (pred_z - gt_z).abs()
        update_meter(meter, "joint/ref_edge_vs_pose_z_err_corr", pearson_corr(ref_edge, pose_z_err))
        update_meter(meter, "joint/ref_edge_vs_pose_z_err_spearman", spearman_corr(ref_edge, pose_z_err))
        append_array(arrays, "joint/ref_edge", ref_edge)
        append_array(arrays, "joint/pose_z_err", pose_z_err)

    update_meter(meter, "joint/ref_edge_vs_ref_depth_err_corr", pearson_corr(ref_edge, ref_err))
    update_meter(meter, "joint/ref_edge_vs_ref_depth_err_spearman", spearman_corr(ref_edge, ref_err))
    append_array(arrays, "joint/ref_depth_err", ref_err)

    if ude_uncer is not None:
        update_meter(meter, "joint/ref_edge_vs_ude_uncer_corr", pearson_corr(ref_edge, ude_uncer))
        update_meter(meter, "joint/ude_uncer_vs_ref_depth_err_corr", pearson_corr(ude_uncer, ref_err))
        append_array(arrays, "joint/ude_uncer", ude_uncer)

    for item in captured:
        layer = item["layer"]
        raw_samples = item["sampled_depth"]
        edge_samples = item["sampled_edge"]
        weights = item["weights_depth"]
        b, joints, heads, samples = weights.shape
        flat_weights = weights.reshape(b, joints, heads * samples)
        flat_weights = flat_weights / flat_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        flat_edges = edge_samples.reshape(b, joints, heads * samples)

        calibrated = scale[:, None, :] * raw_samples + bias[:, None, :]
        errors = (calibrated - gt_z.unsqueeze(-1)).abs()
        weighted_z = (flat_weights * calibrated).sum(dim=-1)
        weighted_err = (weighted_z - gt_z).abs()
        oracle_err = errors.min(dim=-1).values
        oracle_idx = errors.argmin(dim=-1)
        top_idx = flat_weights.argmax(dim=-1)
        top_edge = flat_edges.gather(-1, top_idx.unsqueeze(-1)).squeeze(-1)
        oracle_edge = flat_edges.gather(-1, oracle_idx.unsqueeze(-1)).squeeze(-1)
        weighted_edge = (flat_weights * flat_edges).sum(dim=-1)
        oracle_gap = weighted_err - oracle_err
        edge_spread = flat_edges.std(dim=-1, unbiased=False)

        prefix = f"layer{layer}"
        update_meter(meter, f"{prefix}/affine_r2", r2)
        update_meter(meter, f"{prefix}/ref_edge", ref_edge)
        update_meter(meter, f"{prefix}/weighted_edge", weighted_edge)
        update_meter(meter, f"{prefix}/edge_spread", edge_spread)
        update_meter(meter, f"{prefix}/weighted_err", weighted_err)
        update_meter(meter, f"{prefix}/oracle_err", oracle_err)
        update_meter(meter, f"{prefix}/oracle_gain_over_weighted", oracle_gap)
        update_meter(meter, f"{prefix}/weighted_edge_vs_weighted_err_corr", pearson_corr(weighted_edge, weighted_err))
        update_meter(meter, f"{prefix}/weighted_edge_vs_oracle_gap_corr", pearson_corr(weighted_edge, oracle_gap))
        update_meter(meter, f"{prefix}/ref_edge_vs_weighted_err_corr", pearson_corr(ref_edge, weighted_err))
        update_meter(meter, f"{prefix}/ref_edge_vs_oracle_gap_corr", pearson_corr(ref_edge, oracle_gap))
        update_meter(meter, f"{prefix}/edge_spread_vs_oracle_gap_corr", pearson_corr(edge_spread, oracle_gap))
        update_meter(meter, f"{prefix}/candidate_edge_vs_error_corr", pearson_corr(flat_edges, errors))
        update_meter(meter, f"{prefix}/candidate_edge_vs_negative_error_corr", pearson_corr(flat_edges, -errors))
        update_meter(meter, f"{prefix}/candidate_weight_vs_edge_corr", pearson_corr(flat_weights, flat_edges))
        update_meter(meter, f"{prefix}/candidate_weight_vs_negative_error_corr", pearson_corr(flat_weights, -errors))
        update_meter(meter, f"{prefix}/top_edge_minus_oracle_edge", top_edge - oracle_edge)
        update_meter(meter, f"{prefix}/top_has_lower_edge_than_oracle_frac", (top_edge < oracle_edge).float())
        edge_bucket_metrics(meter, prefix, weighted_edge, weighted_err, oracle_gap)

        if ude_uncer is not None:
            update_meter(meter, f"{prefix}/ude_uncer_vs_weighted_edge_corr", pearson_corr(ude_uncer, weighted_edge))
            update_meter(meter, f"{prefix}/ude_uncer_vs_oracle_gap_corr", pearson_corr(ude_uncer, oracle_gap))

        append_array(arrays, f"{prefix}/candidate_weight", flat_weights)
        append_array(arrays, f"{prefix}/candidate_edge", flat_edges)
        append_array(arrays, f"{prefix}/candidate_error", errors)
        append_array(arrays, f"{prefix}/oracle_idx", oracle_idx)
        append_array(arrays, f"{prefix}/weighted_edge", weighted_edge)
        append_array(arrays, f"{prefix}/weighted_err", weighted_err)
        append_array(arrays, f"{prefix}/oracle_gap", oracle_gap)


def summarize(meter):
    return {key: float(np.mean(values)) for key, values in sorted(meter.items()) if values}


def cat_arrays(arrays):
    out = {}
    for key, values in arrays.items():
        if values:
            out[key] = torch.cat([v.reshape(-1, *v.shape[2:]) if v.ndim >= 3 and key.endswith("oracle_idx") else v.reshape(-1, *v.shape[2:]) for v in values], dim=0)
    return out


def flatten_arrays(arrays):
    out = {}
    for key, values in arrays.items():
        if not values:
            continue
        out[key] = torch.cat([v.reshape(-1, *v.shape[2:]) if v.ndim >= 3 else v.reshape(-1) for v in values], dim=0)
    return out


def standardize(x, eps=1e-6):
    return (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)


def binary_auc(scores, labels):
    scores = scores.detach().flatten().cpu()
    labels = labels.detach().flatten().cpu().bool()
    pos = labels.sum().item()
    neg = labels.numel() - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32)
    pos_rank_sum = ranks[labels].sum().item()
    return float((pos_rank_sum - pos * (pos + 1) / 2) / (pos * neg))


def fit_candidate_logistic(summary, arrays, steps, lr, max_points, device):
    rng = torch.Generator(device="cpu")
    rng.manual_seed(123)
    for key in sorted(arrays):
        if not key.endswith("/candidate_weight"):
            continue
        prefix = key.rsplit("/", 1)[0]
        candidate_count = arrays[f"{prefix}/candidate_weight"].shape[-1]
        weights = arrays[f"{prefix}/candidate_weight"].reshape(-1, candidate_count).float()
        edges = arrays[f"{prefix}/candidate_edge"].reshape(-1, candidate_count).float()
        errors = arrays[f"{prefix}/candidate_error"].reshape(-1, candidate_count).float()
        oracle_idx = arrays[f"{prefix}/oracle_idx"].reshape(-1).long()
        n, k = weights.shape
        labels = F.one_hot(oracle_idx, num_classes=k).float().reshape(-1)
        weight_feat = weights.reshape(-1, 1)
        edge_feat = edges.reshape(-1, 1)
        error_feat = errors.reshape(-1, 1)
        valid = torch.isfinite(weight_feat[:, 0]) & torch.isfinite(edge_feat[:, 0]) & torch.isfinite(labels)
        weight_feat = weight_feat[valid]
        edge_feat = edge_feat[valid]
        error_feat = error_feat[valid]
        labels = labels[valid]

        if labels.numel() > max_points:
            perm = torch.randperm(labels.numel(), generator=rng)[:max_points]
            weight_feat = weight_feat[perm]
            edge_feat = edge_feat[perm]
            error_feat = error_feat[perm]
            labels = labels[perm]

        summary[f"{prefix}/diagnostic_auc_weight"] = binary_auc(weight_feat[:, 0], labels)
        summary[f"{prefix}/diagnostic_auc_neg_edge"] = binary_auc(-edge_feat[:, 0], labels)
        summary[f"{prefix}/diagnostic_auc_edge"] = binary_auc(edge_feat[:, 0], labels)
        summary[f"{prefix}/diagnostic_auc_neg_error_oracle_ceiling"] = binary_auc(-error_feat[:, 0], labels)

        x_weight = standardize(weight_feat)
        x_weight_edge = standardize(torch.cat([weight_feat, edge_feat], dim=1))
        y = labels.to(device)
        pos_weight = ((y.numel() - y.sum()) / y.sum().clamp_min(1.0)).clamp_max(100.0)

        def train_auc(x_cpu):
            x = x_cpu.to(device)
            linear = torch.nn.Linear(x.shape[1], 1, device=device)
            optimizer = torch.optim.AdamW(linear.parameters(), lr=lr, weight_decay=1e-3)
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                logits = linear(x).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                logits = linear(x).squeeze(-1).detach().cpu()
            return binary_auc(logits, labels)

        summary[f"{prefix}/diagnostic_logreg_auc_weight"] = train_auc(x_weight)
        summary[f"{prefix}/diagnostic_logreg_auc_weight_plus_edge"] = train_auc(x_weight_edge)
        summary[f"{prefix}/diagnostic_logreg_auc_gain_from_edge"] = (
            summary[f"{prefix}/diagnostic_logreg_auc_weight_plus_edge"]
            - summary[f"{prefix}/diagnostic_logreg_auc_weight"]
        )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    update_config(args.config)
    update_dir("", "logs")
    config.model.name = "DepthGuidedPose"
    if not hasattr(config.dataset, "depth_format"):
        config.dataset.depth_format = "image"

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = load_model(args, device)
    capture = CaptureState()
    patch_ams_blocks(model.Lifting_net, capture)

    meter = defaultdict(list)
    arrays = defaultdict(list)
    num_batches = 0
    num_examples = 0

    print(f"Dataset: split={args.split} size={len(dataset)} depth_format={config.dataset.depth_format}")
    print(f"Depth path: {config.dataset.depth_image_path}")
    print(f"Device: {device} batch_size={args.batch_size} max_batches={args.max_batches}")

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.max_batches:
                break
            images, gt_3d, keypoints_2d, keypoints_2d_crop, depth_images = preprocess_batch(batch, device)
            _, robust_edges = compute_depth_edges(depth_images)
            capture.reset(depth_images, robust_edges)
            output = model(
                images,
                keypoints_2d,
                keypoints_2d_crop.clone(),
                depth_images,
            )
            pred, coarse_depth, uncer = output

            ref = make_ref_grid(keypoints_2d_crop, device)
            ref_raw = F.grid_sample(
                depth_images.unsqueeze(1),
                ref.unsqueeze(-2),
                padding_mode="border",
                align_corners=True,
            ).squeeze(1).squeeze(-1)
            ref_edge = F.grid_sample(
                robust_edges.unsqueeze(1),
                ref.unsqueeze(-2),
                padding_mode="border",
                align_corners=True,
            ).squeeze(1).squeeze(-1)
            gt_z = gt_3d.squeeze(1)[..., 2]
            pred_z = pred.squeeze(1)[..., 2] if pred is not None else None
            ude_uncer = uncer.squeeze(-1).detach() if uncer is not None else None

            compute_layer_metrics(
                meter,
                arrays,
                capture.layers,
                ref_raw,
                ref_edge,
                gt_z,
                pred_z=pred_z,
                ude_uncer=ude_uncer,
            )
            num_batches += 1
            num_examples += images.shape[0]
            print(f"processed batch {num_batches} examples={num_examples}", flush=True)

    summary = summarize(meter)
    flat_arrays = flatten_arrays(arrays)
    fit_candidate_logistic(
        summary,
        flat_arrays,
        steps=args.logistic_steps,
        lr=args.logistic_lr,
        max_points=args.max_logistic_points,
        device=device,
    )

    summary["meta/num_batches"] = num_batches
    summary["meta/num_examples"] = num_examples
    summary["meta/checkpoint"] = args.checkpoint
    summary["meta/config"] = args.config
    summary["meta/depth_format"] = config.dataset.depth_format
    summary["meta/depth_image_path"] = config.dataset.depth_image_path
    summary["meta/logistic_steps"] = args.logistic_steps

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    if args.save_arrays:
        array_path = Path(args.save_arrays)
        array_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(array_path, **{key.replace("/", "__"): value.numpy() for key, value in flat_arrays.items()})
        print(f"Saved arrays: {array_path}")

    print("\n=== Depth discontinuity evidence probe summary ===")
    for key, value in summary.items():
        if key.startswith("meta/"):
            continue
        print(f"{key}: {value:.6f}")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
