#!/usr/bin/env python3
"""
RSDR motivation diagnostics.

This script tests whether AMS depth tokens contain unevenly useful channel
subspaces, which is the key premise behind Reliable Subspace Depth Restoration.
It does not train RSDR. It only analyzes an existing DLST checkpoint.
"""

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mvn import datasets
from mvn.datasets import utils as dataset_utils
from mvn.models.DGPose_dlst import DepthGuidedPoseDLST
from mvn.utils.cfg import config, update_config, update_dir


JOINT_NAMES = [
    "pelvis",
    "r_hip",
    "r_knee",
    "r_ankle",
    "l_hip",
    "l_knee",
    "l_ankle",
    "spine",
    "thorax",
    "neck",
    "head",
    "l_shoulder",
    "l_elbow",
    "l_wrist",
    "r_shoulder",
    "r_elbow",
    "r_wrist",
]

BONES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

AMBIGUOUS_PAIRS = [
    (13, 16),
    (12, 15),
    (3, 6),
    (2, 5),
    (13, 7),
    (16, 7),
    (13, 8),
    (16, 8),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze group-wise depth-token subspace usefulness for RSDR.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="rsdr_motivation/summary.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", default="hrnet_32", choices=["hrnet_32", "hrnet_48"])
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-groups", type=int, default=8)
    parser.add_argument("--probe-steps", type=int, default=250)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--ablate-groups", action="store_true")
    parser.add_argument("--ablation-max-batches", type=int, default=20)
    parser.add_argument("--replacement", default="zero", choices=["zero", "mean", "shuffle"])
    parser.add_argument("--random-subset", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return safe_corr(rankdata_average(x), rankdata_average(y))


def pair_indices(num_joints: int, pair_set: str) -> List[Tuple[int, int]]:
    if pair_set == "bones":
        return BONES
    if pair_set == "ambiguous":
        return AMBIGUOUS_PAIRS
    return list(combinations(range(num_joints), 2))


def build_val_dataset(args: argparse.Namespace):
    return datasets.human36m(
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
        data_format=config.dataset.data_format,
        frame=1,
    )


def select_indices(args: argparse.Namespace, dataset) -> np.ndarray:
    n = min(args.num_samples, len(dataset))
    if args.random_subset:
        rng = np.random.default_rng(args.seed)
        return np.sort(rng.choice(len(dataset), size=n, replace=False))
    return np.arange(n)


def build_loader(args: argparse.Namespace):
    dataset = build_val_dataset(args)
    indices = select_indices(args, dataset)
    subset = Subset(dataset, indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=True,
    )
    return dataset, loader, indices


def load_model(args: argparse.Namespace, device: torch.device) -> DepthGuidedPoseDLST:
    model = DepthGuidedPoseDLST(config, device=device)
    raw = torch.load(args.checkpoint, map_location="cpu")
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    state = {key.replace("module.", ""): value for key, value in state.items()}
    msg = model.load_state_dict(state, strict=False)
    print(f"[model] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    model.to(device)
    model.eval()
    return model


def capture_ams_tokens(args: argparse.Namespace, model: DepthGuidedPoseDLST, loader: DataLoader, device: torch.device):
    fd_chunks = []
    gt_chunks = []

    def hook(_module, _inputs, output):
        fd_chunks.append(output[:, -1].detach().float().cpu())

    handle = model.Lifting_net.RGBD_Extraction[-1].register_forward_hook(hook)
    prefetcher = dataset_utils.data_prefetcher(loader, device, is_train=False, flip_test=False)
    batch = prefetcher.next()
    seen = 0
    with torch.no_grad():
        while batch is not None and seen < args.num_samples:
            images, gt_3d, kp2d, kp2d_crop, depth_images = batch
            _pred, _rel_depth, _layer_assign = model(images, kp2d, kp2d_crop.clone(), depth_images)
            if gt_3d.dim() == 4:
                gt_3d = gt_3d.squeeze(1)
            gt_chunks.append(gt_3d.detach().float().cpu())
            seen += gt_3d.shape[0]
            print(f"[capture] {min(seen, args.num_samples)}/{args.num_samples}", flush=True)
            batch = prefetcher.next()
    handle.remove()
    fd = torch.cat(fd_chunks, dim=0)[: args.num_samples].numpy()
    gt = torch.cat(gt_chunks, dim=0)[: args.num_samples].numpy()
    return fd, gt


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray):
    mu = x_train.mean(axis=0, keepdims=True)
    std = np.maximum(x_train.std(axis=0, keepdims=True), 1e-6)
    return (x_train - mu) / std, (x_test - mu) / std


def ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, ridge: float) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1, 1)
    x_test = np.asarray(x_test, dtype=np.float64)
    x_train, x_test = standardize_train_test(x_train, x_test)
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    x_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)
    eye = np.eye(x_train.shape[1], dtype=np.float64)
    eye[-1, -1] = 0.0
    weights = np.linalg.solve(x_train.T @ x_train + ridge * eye, x_train.T @ y_train)
    return (x_test @ weights).reshape(-1)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    denom = np.sum((y_true - y_true.mean()) ** 2)
    if denom < 1e-12:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def build_pairwise_dataset(fd: np.ndarray, gt_z: np.ndarray, indices: np.ndarray, pairs: List[Tuple[int, int]]):
    x_parts = []
    y_parts = []
    for left, right in pairs:
        x_parts.append(fd[indices, right] - fd[indices, left])
        y_parts.append(gt_z[indices, right] - gt_z[indices, left])
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def group_slices(dim: int, num_groups: int):
    if dim % num_groups != 0:
        raise ValueError(f"Channel dim {dim} must be divisible by num_groups={num_groups}.")
    group_size = dim // num_groups
    return [slice(i * group_size, (i + 1) * group_size) for i in range(num_groups)]


def probe_group(fd: np.ndarray, gt_z: np.ndarray, channel_slice: slice, train_idx: np.ndarray, test_idx: np.ndarray, ridge: float):
    fd_group = fd[:, :, channel_slice]
    x_train = fd_group[train_idx].reshape(-1, fd_group.shape[-1])
    y_train = gt_z[train_idx].reshape(-1)
    x_test = fd_group[test_idx].reshape(-1, fd_group.shape[-1])
    y_test = gt_z[test_idx].reshape(-1)
    abs_pred = ridge_fit_predict(x_train, y_train, x_test, ridge)

    all_pairs = pair_indices(gt_z.shape[1], "all")
    bone_pairs = pair_indices(gt_z.shape[1], "bones")
    ambiguous_pairs = pair_indices(gt_z.shape[1], "ambiguous")

    def pair_probe(pairs):
        xp_train, yp_train = build_pairwise_dataset(fd_group, gt_z, train_idx, pairs)
        xp_test, yp_test = build_pairwise_dataset(fd_group, gt_z, test_idx, pairs)
        pred = ridge_fit_predict(xp_train, yp_train, xp_test, ridge)
        valid = np.abs(yp_test) > 1e-8
        significant = np.abs(yp_test) > 0.05
        return {
            "r2": r2_score(yp_test, pred),
            "pearson": safe_corr(yp_test, pred),
            "spearman": safe_spearman(yp_test, pred),
            "ordinal_acc": float((np.sign(pred[valid]) == np.sign(yp_test[valid])).mean()) if valid.any() else float("nan"),
            "ordinal_acc_delta": float((np.sign(pred[significant]) == np.sign(yp_test[significant])).mean()) if significant.any() else float("nan"),
            "mae": float(np.mean(np.abs(yp_test - pred))),
        }

    return {
        "absolute_z": {
            "r2": r2_score(y_test, abs_pred),
            "pearson": safe_corr(y_test, abs_pred),
            "spearman": safe_spearman(y_test, abs_pred),
            "mae": float(np.mean(np.abs(y_test - abs_pred))),
            "mae_over_std": float(np.mean(np.abs(y_test - abs_pred)) / (np.std(y_test) + 1e-12)),
        },
        "pairwise_all": pair_probe(all_pairs),
        "pairwise_bones": pair_probe(bone_pairs),
        "pairwise_ambiguous": pair_probe(ambiguous_pairs),
    }


def summarize_group_spread(group_results: List[Dict[str, object]]) -> Dict[str, float]:
    all_acc = np.asarray([g["pairwise_all"]["ordinal_acc_delta"] for g in group_results], dtype=np.float64)
    abs_r2 = np.asarray([g["absolute_z"]["r2"] for g in group_results], dtype=np.float64)
    return {
        "pairwise_all_acc_delta_min": float(np.nanmin(all_acc)),
        "pairwise_all_acc_delta_max": float(np.nanmax(all_acc)),
        "pairwise_all_acc_delta_range": float(np.nanmax(all_acc) - np.nanmin(all_acc)),
        "absolute_z_r2_min": float(np.nanmin(abs_r2)),
        "absolute_z_r2_max": float(np.nanmax(abs_r2)),
        "absolute_z_r2_range": float(np.nanmax(abs_r2) - np.nanmin(abs_r2)),
    }


def preprocess_batch(batch, device: torch.device):
    images, keypoints_3d_gt, keypoints_2d_cpn, keypoints_2d_crop, depth_images = batch
    images = images.to(device, non_blocking=True).float()
    keypoints_3d_gt = keypoints_3d_gt.to(device, non_blocking=True).float()
    keypoints_2d_cpn = keypoints_2d_cpn.to(device, non_blocking=True).float()
    keypoints_2d_crop = keypoints_2d_crop.to(device, non_blocking=True).float()
    depth_images = depth_images.to(device, non_blocking=True).float()

    depth_images = depth_images / 255.0
    images = torch.flip(images, [-1])
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)
    images = (images / 255.0 - mean) / std

    keypoints_3d_gt[:, :, 1:] -= keypoints_3d_gt[:, :, :1]
    keypoints_3d_gt[:, :, 0] = 0

    return images, keypoints_3d_gt, keypoints_2d_cpn, keypoints_2d_crop, depth_images


def root_aligned_mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> float:
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if gt.dim() == 4:
        gt = gt.squeeze(1)
    pred_rel = pred - pred[:, 0:1]
    gt_rel = gt - gt[:, 0:1]
    return torch.linalg.norm(pred_rel - gt_rel, dim=-1).mean().item() * 1000.0


def install_group_ablation(model: DepthGuidedPoseDLST, channel_slice: slice, replacement: str):
    target = model.Lifting_net.RGBD_Extraction[-1]

    def hook(_module, _inputs, output):
        out = output.clone()
        depth_tokens = out[:, -1]
        if replacement == "zero":
            repl = torch.zeros_like(depth_tokens[:, :, channel_slice])
        elif replacement == "mean":
            repl = depth_tokens[:, :, channel_slice].mean(dim=1, keepdim=True).expand_as(depth_tokens[:, :, channel_slice])
        elif replacement == "shuffle":
            perm = torch.randperm(depth_tokens.shape[1], device=depth_tokens.device)
            repl = depth_tokens[:, perm, channel_slice]
        else:
            raise ValueError(replacement)
        depth_tokens = depth_tokens.clone()
        depth_tokens[:, :, channel_slice] = repl
        out[:, -1] = depth_tokens
        return out

    return target.register_forward_hook(hook)


def evaluate_mpjpe(model: DepthGuidedPoseDLST, loader: DataLoader, device: torch.device, max_batches: int = None):
    total = 0
    weighted = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images, gt, kp2d, crop, depth = preprocess_batch(batch, device)
            pred, _rel_depth, _layer_assign = model(images, kp2d, crop.clone(), depth)
            batch_mpjpe = root_aligned_mpjpe(pred, gt)
            batch_size = gt.shape[0]
            weighted += batch_mpjpe * batch_size
            total += batch_size
            print(f"[eval] batch={batch_idx:04d} mpjpe={batch_mpjpe:.3f}", flush=True)
    return weighted / max(total, 1), total


def ablate_groups(args: argparse.Namespace, model: DepthGuidedPoseDLST, loader: DataLoader, device: torch.device, slices):
    print("[ablation] baseline")
    baseline, num_samples = evaluate_mpjpe(model, loader, device, args.ablation_max_batches)
    out = {
        "baseline_mpjpe": baseline,
        "num_samples": int(num_samples),
        "groups": [],
    }
    for group_idx, channel_slice in enumerate(slices):
        print(f"[ablation] group={group_idx} channels={channel_slice.start}:{channel_slice.stop}")
        handle = install_group_ablation(model, channel_slice, args.replacement)
        mpjpe, _num = evaluate_mpjpe(model, loader, device, args.ablation_max_batches)
        handle.remove()
        out["groups"].append({
            "group": group_idx,
            "channels": [int(channel_slice.start), int(channel_slice.stop)],
            "mpjpe": mpjpe,
            "delta_mpjpe": mpjpe - baseline,
        })
    return out


def write_report(output_path: Path, result: Dict[str, object]):
    summary = result["summary"]
    lines = [
        "# RSDR Motivation Diagnostics",
        "",
        f"- config: `{result['config']}`",
        f"- checkpoint: `{result['checkpoint']}`",
        f"- samples: {result['num_samples']}",
        f"- groups: {result['num_groups']}",
        "",
        "## Group Probe Spread",
        "",
        f"- pairwise all ordinal acc range: {summary['pairwise_all_acc_delta_range']:.4f}",
        f"- absolute z R2 range: {summary['absolute_z_r2_range']:.4f}",
    ]
    if result.get("ablation") is not None:
        deltas = [g["delta_mpjpe"] for g in result["ablation"]["groups"]]
        lines.extend([
            "",
            "## Channel Group Ablation",
            "",
            f"- baseline MPJPE: {result['ablation']['baseline_mpjpe']:.3f} mm",
            f"- max delta MPJPE: {max(deltas):+.3f} mm",
            f"- min delta MPJPE: {min(deltas):+.3f} mm",
        ])
    output_path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    update_config(args.config)
    update_dir("", "logs/")
    config.model.backbone.type = args.backbone
    if args.backbone == "hrnet_32":
        config.model.poseformer.base_dim = 32
    else:
        config.model.poseformer.base_dim = 48

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset, loader, indices = build_loader(args)
    model = load_model(args, device)

    fd, gt = capture_ams_tokens(args, model, loader, device)
    gt_z = gt[..., 2]
    dim = fd.shape[-1]
    slices = group_slices(dim, args.num_groups)

    num_frames = fd.shape[0]
    split = max(1, int(0.7 * num_frames))
    train_idx = np.arange(split)
    test_idx = np.arange(split, num_frames)
    if len(test_idx) < 2:
        raise RuntimeError("Need at least a few test samples. Increase --num-samples.")

    group_results = []
    for group_idx, channel_slice in enumerate(slices):
        print(f"[probe] group={group_idx} channels={channel_slice.start}:{channel_slice.stop}", flush=True)
        metrics = probe_group(fd, gt_z, channel_slice, train_idx, test_idx, args.ridge)
        metrics["group"] = group_idx
        metrics["channels"] = [int(channel_slice.start), int(channel_slice.stop)]
        group_results.append(metrics)

    full_metrics = probe_group(fd, gt_z, slice(0, dim), train_idx, test_idx, args.ridge)
    ablation = ablate_groups(args, model, loader, device, slices) if args.ablate_groups else None

    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "num_samples": int(fd.shape[0]),
        "selected_indices_start": int(indices[0]) if len(indices) else None,
        "selected_indices_end": int(indices[-1]) if len(indices) else None,
        "num_groups": args.num_groups,
        "group_size": int(dim // args.num_groups),
        "probe_split": {"train_frames": int(len(train_idx)), "test_frames": int(len(test_idx))},
        "full_token_probe": full_metrics,
        "group_probes": group_results,
        "summary": summarize_group_spread(group_results),
        "ablation": ablation,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(output_path, result)
    print(f"Wrote {output_path}")
    print(f"Wrote {output_path.with_suffix('.md')}")


if __name__ == "__main__":
    main()
