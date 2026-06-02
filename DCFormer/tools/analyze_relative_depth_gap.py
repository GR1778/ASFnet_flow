#!/usr/bin/env python3
"""
Diagnose what AMS still misses after sampling, without using UDE as the
motivation.

Core question:
    Does AMS output F_d encode explicit joint-to-joint relative depth layout,
    or is it mainly a pose/global-depth feature that still needs a relation
    transform module?

The script captures F_d after the last AMS block, then measures:
1) scalar depth head ordinal/affine quality from F_d,
2) unsupervised correlation between pairwise F_d distances and GT depth gaps,
3) linear-probe decodability of per-joint absolute Z from F_d,
4) linear-probe decodability of pairwise relative Z from F_i - F_j.

If global/absolute depth is easier than pairwise layout, a reasonable module
direction is a Relative Depth Relation Transformer after AMS.
"""

import argparse
import json
import os
import sys
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mvn import datasets
from mvn.datasets import utils as dataset_utils
from mvn.models.DGPose import DepthGuidedPose
from mvn.utils.cfg import config, update_config


BONES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

AMBIGUOUS_PAIRS = [
    (13, 16),  # wrists
    (12, 15),  # elbows
    (3, 6),    # ankles
    (2, 5),    # knees
    (13, 7), (16, 7),  # wrists vs pelvis/root chain
    (13, 8), (16, 8),  # wrists vs thorax
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_samples", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="relative_depth_gap_analysis")
    p.add_argument("--retain_every_n_frames", type=int, default=None)
    p.add_argument("--ridge", type=float, default=1e-2)
    return p.parse_args()


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
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
    x, y = x[mask], y[mask]
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return safe_corr(rankdata_average(x), rankdata_average(y))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def ordinal_accuracy(score: np.ndarray, gt_z: np.ndarray, pairs: List[Tuple[int, int]]) -> float:
    correct = []
    for i, j in pairs:
        pred = score[:, i] - score[:, j]
        gt = gt_z[:, i] - gt_z[:, j]
        valid = np.abs(gt) > 1e-8
        if valid.any():
            correct.append((np.sign(pred[valid]) == np.sign(gt[valid])).astype(np.float64))
    if not correct:
        return float("nan")
    return float(np.concatenate(correct).mean())


def framewise_affine_metrics(pred_z: np.ndarray, gt_z: np.ndarray) -> Dict[str, float]:
    r2s, maes, alphas, betas = [], [], [], []
    for p, g in zip(pred_z, gt_z):
        x = np.stack([p, np.ones_like(p)], axis=1)
        try:
            alpha, beta = np.linalg.lstsq(x, g, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        aligned = alpha * p + beta
        r2s.append(r2_score(g, aligned))
        maes.append(float(np.mean(np.abs(aligned - g))))
        alphas.append(float(alpha))
        betas.append(float(beta))
    return {
        "affine_r2_mean": float(np.nanmean(r2s)),
        "affine_mae_mean": float(np.nanmean(maes)),
        "alpha_mean": float(np.nanmean(alphas)),
        "alpha_std": float(np.nanstd(alphas)),
        "beta_mean": float(np.nanmean(betas)),
        "beta_std": float(np.nanstd(betas)),
    }


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return (x_train - mu) / std, (x_test - mu) / std


def ridge_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    ridge: float,
) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1, 1)
    x_test = np.asarray(x_test, dtype=np.float64)
    x_train, x_test = standardize_train_test(x_train, x_test)
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    x_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)
    reg = ridge * np.eye(x_train.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    w = np.linalg.solve(x_train.T @ x_train + reg, x_train.T @ y_train)
    return (x_test @ w).reshape(-1)


def build_loader(args: argparse.Namespace) -> DataLoader:
    retain = (
        args.retain_every_n_frames
        if args.retain_every_n_frames is not None
        else config.val.retain_every_n_frames_in_test
    )
    val_dataset = eval("datasets." + config.dataset.val_dataset)(
        root=config.dataset.root,
        pred_results_path=config.val.pred_results_path,
        depth_image_path=config.dataset.depth_image_path,
        train=False,
        test=True,
        image_shape=config.model.image_shape,
        labels_path=config.dataset.val_labels_path,
        with_damaged_actions=config.val.with_damaged_actions,
        retain_every_n_frames_in_test=retain,
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
    n = min(args.num_samples, len(val_dataset))
    subset = Subset(val_dataset, list(range(n)))
    return DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=True,
    )


def load_model(args: argparse.Namespace, device: torch.device) -> DepthGuidedPose:
    model = DepthGuidedPose(config, device).to(device)
    raw = torch.load(args.checkpoint, map_location=device)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    state = {k.replace("module.", ""): v for k, v in state.items()}
    msg = model.load_state_dict(state, strict=False)
    print(f"[model] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    model.eval()
    return model


def capture_ams_outputs(args: argparse.Namespace, model: DepthGuidedPose, loader: DataLoader, device: torch.device):
    fd_chunks, mu_chunks, gt_chunks = [], [], []

    def hook(_module, _inputs, output):
        fd_chunks.append(output[:, -1].detach().float().cpu())

    handle = model.Lifting_net.RGBD_Extraction[-1].register_forward_hook(hook)
    prefetcher = dataset_utils.data_prefetcher(loader, device, is_train=False, flip_test=False)
    batch = prefetcher.next()
    seen = 0
    with torch.no_grad():
        while batch is not None and seen < args.num_samples:
            images, gt_3d, kp2d, kp2d_crop, depth_images = batch
            pred, coarse_depth, _s = model(images, kp2d, kp2d_crop.clone(), depth_images)
            del pred
            mu_chunks.append(coarse_depth.detach().squeeze(-1).float().cpu())
            if gt_3d.dim() == 4:
                gt_3d = gt_3d.squeeze(1)
            gt_chunks.append(gt_3d.detach().float().cpu())
            seen += gt_3d.shape[0]
            print(f"[capture] {min(seen, args.num_samples)}/{args.num_samples}", flush=True)
            batch = prefetcher.next()
    handle.remove()

    fd = torch.cat(fd_chunks, dim=0)[: args.num_samples].numpy()
    mu = torch.cat(mu_chunks, dim=0)[: args.num_samples].numpy()
    gt = torch.cat(gt_chunks, dim=0)[: args.num_samples].numpy()
    return fd, mu, gt


def build_pairwise_dataset(
    fd: np.ndarray,
    gt_z: np.ndarray,
    frame_indices: np.ndarray,
    pairs: List[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for n in frame_indices:
        for i, j in pairs:
            xs.append(fd[n, i] - fd[n, j])
            ys.append(gt_z[n, i] - gt_z[n, j])
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def probe_absolute_depth(fd: np.ndarray, gt_z: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, ridge: float):
    x_train = fd[train_idx].reshape(-1, fd.shape[-1])
    y_train = gt_z[train_idx].reshape(-1)
    x_test = fd[test_idx].reshape(-1, fd.shape[-1])
    y_test = gt_z[test_idx].reshape(-1)
    pred = ridge_fit_predict(x_train, y_train, x_test, ridge)
    return {
        "r2": r2_score(y_test, pred),
        "pearson": safe_corr(y_test, pred),
        "mae": float(np.mean(np.abs(y_test - pred))),
        "target_std": float(np.std(y_test)),
        "mae_over_std": float(np.mean(np.abs(y_test - pred)) / (np.std(y_test) + 1e-12)),
    }


def probe_pairwise_depth(
    fd: np.ndarray,
    gt_z: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    pairs: List[Tuple[int, int]],
    ridge: float,
):
    x_train, y_train = build_pairwise_dataset(fd, gt_z, train_idx, pairs)
    x_test, y_test = build_pairwise_dataset(fd, gt_z, test_idx, pairs)
    pred = ridge_fit_predict(x_train, y_train, x_test, ridge)
    valid = np.abs(y_test) > 1e-8
    return {
        "num_pairs": int(len(pairs)),
        "r2": r2_score(y_test, pred),
        "pearson": safe_corr(y_test, pred),
        "spearman": safe_spearman(y_test, pred),
        "ordinal_acc": float((np.sign(pred[valid]) == np.sign(y_test[valid])).mean()),
        "mae": float(np.mean(np.abs(y_test - pred))),
        "target_std": float(np.std(y_test)),
        "mae_over_std": float(np.mean(np.abs(y_test - pred)) / (np.std(y_test) + 1e-12)),
    }


def unsupervised_pairwise_geometry(fd: np.ndarray, gt_z: np.ndarray, pairs: List[Tuple[int, int]]):
    feat_dist, depth_gap = [], []
    for i, j in pairs:
        feat_dist.append(np.linalg.norm(fd[:, i] - fd[:, j], axis=-1))
        depth_gap.append(np.abs(gt_z[:, i] - gt_z[:, j]))
    feat_dist = np.stack(feat_dist, axis=1)
    depth_gap = np.stack(depth_gap, axis=1)
    return {
        "spearman_feature_distance_vs_abs_depth_gap": safe_spearman(feat_dist, depth_gap),
        "pearson_feature_distance_vs_abs_depth_gap": safe_corr(feat_dist, depth_gap),
        "feature_distance_mean": float(np.mean(feat_dist)),
        "abs_depth_gap_mean": float(np.mean(depth_gap)),
    }


def body_mean_stats(fd: np.ndarray):
    fd_norm = fd / (np.linalg.norm(fd, axis=-1, keepdims=True) + 1e-8)
    cos = np.einsum("nic,njc->nij", fd_norm, fd_norm)
    j = fd.shape[1]
    off = ~np.eye(j, dtype=bool)
    body = fd.mean(axis=1, keepdims=True)
    residual = fd - body
    body_energy = np.sum(body ** 2, axis=(1, 2))
    residual_energy = np.sum(residual ** 2, axis=(1, 2)) / j
    return {
        "offdiag_cosine_mean": float(cos[:, off].mean()),
        "offdiag_cosine_std": float(cos[:, off].std()),
        "body_mean_energy_fraction": float((body_energy / (body_energy + residual_energy + 1e-12)).mean()),
    }


def make_verdict(summary: Dict[str, object]) -> Dict[str, object]:
    abs_r2 = summary["linear_probe_absolute_z"]["r2"]
    pair_r2 = summary["linear_probe_pairwise_all"]["r2"]
    pair_ord = summary["linear_probe_pairwise_all"]["ordinal_acc"]
    geom_s = summary["unsupervised_pairwise_all"]["spearman_feature_distance_vs_abs_depth_gap"]
    mu_ord = summary["coarse_mu"]["ordinal_acc_all_pairs"]
    body_frac = summary["fd_body_mean_stats"]["body_mean_energy_fraction"]

    reasons = []
    if body_frac > 0.5:
        reasons.append("AMS F_d is strongly frame/global-component dominated.")
    if abs_r2 - pair_r2 > 0.15:
        reasons.append("Absolute per-joint Z is more decodable than pairwise relative Z.")
    if pair_ord < 0.80:
        reasons.append("Pairwise ordinal depth accuracy leaves clear headroom.")
    if np.isfinite(geom_s) and geom_s < 0.35:
        reasons.append("Raw F_d pairwise distances weakly track true depth gaps.")
    if pair_r2 > 0.65 and np.isfinite(geom_s) and geom_s < 0.35:
        reasons.append(
            "Relative depth is linearly decodable from F_d differences, but not explicitly organized in the raw feature geometry."
        )
    if mu_ord < 0.85:
        reasons.append("The scalar depth head does not reliably preserve joint depth order.")

    if len(reasons) >= 2:
        level = "strong"
    elif reasons:
        level = "moderate"
    else:
        level = "weak"

    return {
        "relative_depth_relation_module_motivation": level,
        "hypothesis": (
            "AMS captures depth-layout information implicitly, but the representation "
            "does not expose it as a structured root-relative or pairwise relation signal."
        ),
        "recommended_module_direction": (
            "Add an AMS-after module that converts sampled depth features into "
            "root-relative and pairwise joint-depth relations, then feeds relation-aware "
            "joint tokens to the multimodal fusion transformer. This is not denoising; "
            "it is depth-layout/ordinal-geometry modeling."
        ),
        "candidate_names": [
            "Relative Depth Relation Transformer (RDRT)",
            "Depth Layout Relation Module (DLRM)",
            "Pairwise Ordinal Depth Adapter (PODA)",
        ],
        "reasons": reasons,
    }


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    update_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}")
    print(f"[setup] config={args.config}")
    print(f"[setup] checkpoint={args.checkpoint}")

    model = load_model(args, device)
    loader = build_loader(args)
    fd, mu, gt = capture_ams_outputs(args, model, loader, device)
    gt_z = gt[..., 2]
    all_pairs = list(combinations(range(fd.shape[1]), 2))

    n = fd.shape[0]
    split = max(1, int(0.7 * n))
    train_idx = np.arange(split)
    test_idx = np.arange(split, n)
    if len(test_idx) < 2:
        raise RuntimeError("Need at least a few test samples. Increase --num_samples.")

    summary = {
        "num_samples": int(n),
        "fd_shape": list(fd.shape),
        "coarse_mu": {
            **framewise_affine_metrics(mu, gt_z),
            "ordinal_acc_all_pairs": ordinal_accuracy(mu, gt_z, all_pairs),
            "ordinal_acc_bones": ordinal_accuracy(mu, gt_z, BONES),
            "ordinal_acc_ambiguous_pairs": ordinal_accuracy(mu, gt_z, AMBIGUOUS_PAIRS),
        },
        "fd_body_mean_stats": body_mean_stats(fd),
        "unsupervised_pairwise_all": unsupervised_pairwise_geometry(fd, gt_z, all_pairs),
        "unsupervised_pairwise_bones": unsupervised_pairwise_geometry(fd, gt_z, BONES),
        "unsupervised_pairwise_ambiguous": unsupervised_pairwise_geometry(fd, gt_z, AMBIGUOUS_PAIRS),
        "linear_probe_absolute_z": probe_absolute_depth(fd, gt_z, train_idx, test_idx, args.ridge),
        "linear_probe_pairwise_all": probe_pairwise_depth(fd, gt_z, train_idx, test_idx, all_pairs, args.ridge),
        "linear_probe_pairwise_bones": probe_pairwise_depth(fd, gt_z, train_idx, test_idx, BONES, args.ridge),
        "linear_probe_pairwise_ambiguous": probe_pairwise_depth(fd, gt_z, train_idx, test_idx, AMBIGUOUS_PAIRS, args.ridge),
    }
    summary["verdict"] = make_verdict(summary)

    out_json = os.path.join(args.output_dir, "summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[done] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
