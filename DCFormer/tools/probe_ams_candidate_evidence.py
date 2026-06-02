#!/usr/bin/env python
"""Probe whether ASFNet AMS depth candidates are collapsed too early.

This script loads the official ASFNet-style DepthGuidedPose model, hooks the
RGBD_Extraction/AMS blocks, and evaluates the depth branch samples before they
are weighted-summed into a single joint token.

The diagnostic is deliberately sample-level:
- sample each AMS depth candidate's raw depth value;
- affine-calibrate raw monocular depths to GT joint z per person using initial
  joint samples, so scale/sign ambiguity is handled;
- compare AMS weighted aggregation against an oracle candidate selector;
- measure whether AMS weights correlate with candidate correctness;
- measure whether UDE token uncertainty tracks candidate contamination.
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
    parser.add_argument("--output", default="debug_vis/ams_candidate_evidence_probe.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class CaptureState:
    def __init__(self):
        self.depth_images = None
        self.layers = []

    def reset(self, depth_images):
        self.depth_images = depth_images
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
                capture.layers.append(
                    {
                        "layer": _layer_idx,
                        "pos_depth": depth_grid.detach(),
                        "weights_depth": weights[:, depth_level, ..., 0].detach(),
                        "sampled_depth": sampled_depth.detach(),
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


def update_meter(meter, key, value):
    if isinstance(value, torch.Tensor):
        value = value.detach().float().mean().item()
    meter[key].append(float(value))


def compute_layer_metrics(meter, captured, ref_raw, gt_z, ude_uncer=None):
    scale, bias, r2 = fit_affine(ref_raw, gt_z)
    ref_err = (scale * ref_raw + bias - gt_z).abs()

    for item in captured:
        layer = item["layer"]
        raw_samples = item["sampled_depth"]
        weights = item["weights_depth"]
        b, joints, heads, samples = weights.shape
        flat_weights = weights.reshape(b, joints, heads * samples)
        flat_weights = flat_weights / flat_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        calibrated = scale[:, None, :] * raw_samples + bias[:, None, :]
        errors = (calibrated - gt_z.unsqueeze(-1)).abs()
        weighted_z = (flat_weights * calibrated).sum(dim=-1)
        weighted_err = (weighted_z - gt_z).abs()
        oracle_err = errors.min(dim=-1).values
        mean_candidate_err = errors.mean(dim=-1)
        top_idx = flat_weights.argmax(dim=-1, keepdim=True)
        top_err = errors.gather(-1, top_idx).squeeze(-1)
        oracle_idx = errors.argmin(dim=-1)
        top_is_oracle = (top_idx.squeeze(-1) == oracle_idx).float()

        worse_than_ref_mass = (flat_weights * (errors > ref_err.unsqueeze(-1)).float()).sum(dim=-1)
        oracle_gap = weighted_err - oracle_err
        top_gap = top_err - oracle_err
        candidate_std = calibrated.std(dim=-1, unbiased=False)

        prefix = f"layer{layer}"
        update_meter(meter, f"{prefix}/affine_r2", r2)
        update_meter(meter, f"{prefix}/ref_err", ref_err)
        update_meter(meter, f"{prefix}/weighted_err", weighted_err)
        update_meter(meter, f"{prefix}/oracle_err", oracle_err)
        update_meter(meter, f"{prefix}/mean_candidate_err", mean_candidate_err)
        update_meter(meter, f"{prefix}/top_weight_err", top_err)
        update_meter(meter, f"{prefix}/oracle_gain_over_weighted", oracle_gap)
        update_meter(meter, f"{prefix}/top_gap_to_oracle", top_gap)
        update_meter(meter, f"{prefix}/top_is_oracle", top_is_oracle)
        update_meter(meter, f"{prefix}/worse_than_ref_weight_mass", worse_than_ref_mass)
        update_meter(meter, f"{prefix}/candidate_depth_std", candidate_std)
        update_meter(meter, f"{prefix}/weight_vs_negative_error_corr", pearson_corr(flat_weights, -errors))
        update_meter(meter, f"{prefix}/weight_vs_error_corr", pearson_corr(flat_weights, errors))
        update_meter(meter, f"{prefix}/oracle_beats_weighted_frac", (oracle_gap > 1e-4).float())
        update_meter(meter, f"{prefix}/weighted_beats_ref_frac", (weighted_err < ref_err).float())

        if ude_uncer is not None:
            update_meter(meter, f"{prefix}/ude_uncer_vs_weighted_err_corr", pearson_corr(ude_uncer, weighted_err))
            update_meter(meter, f"{prefix}/ude_uncer_vs_oracle_gap_corr", pearson_corr(ude_uncer, oracle_gap))


def summarize(meter):
    return {key: float(np.mean(values)) for key, values in sorted(meter.items()) if values}


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
            capture.reset(depth_images)
            pred, coarse_depth, uncer = model(
                images,
                keypoints_2d,
                keypoints_2d_crop.clone(),
                depth_images,
            )

            ref = keypoints_2d_crop.clone()
            ref[..., :2] /= torch.tensor([192 // 2, 256 // 2], device=device)
            ref[..., :2] -= torch.tensor([1, 1], device=device)
            ref_raw = F.grid_sample(
                depth_images.unsqueeze(1),
                ref.unsqueeze(-2),
                padding_mode="border",
                align_corners=True,
            ).squeeze(1).squeeze(-1)
            gt_z = gt_3d.squeeze(1)[..., 2]
            ude_uncer = uncer.squeeze(-1).detach() if uncer is not None else None

            compute_layer_metrics(meter, capture.layers, ref_raw, gt_z, ude_uncer=ude_uncer)
            num_batches += 1
            num_examples += images.shape[0]
            print(f"processed batch {num_batches} examples={num_examples}", flush=True)

    summary = summarize(meter)
    summary["meta/num_batches"] = num_batches
    summary["meta/num_examples"] = num_examples
    summary["meta/checkpoint"] = args.checkpoint
    summary["meta/config"] = args.config
    summary["meta/depth_format"] = config.dataset.depth_format
    summary["meta/depth_image_path"] = config.dataset.depth_image_path

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print("\n=== AMS candidate evidence probe summary ===")
    for key, value in summary.items():
        if key.startswith("meta/"):
            continue
        print(f"{key}: {value:.6f}")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
