#!/usr/bin/env python3
"""
Diagnose whether ASFnet's UDE scalar Gaussian representation is an information
bottleneck for AMS depth tokens.

This script does not claim that UDE is wrong. It tests increasingly strong
claims:

1) Scalar bottleneck gap:
   Can a small probe using the full AMS depth token F_d predict joint depth
   better than a probe using only UDE's scalar outputs (mu, s)?

2) Residual recoverability:
   After the best scalar probe from (mu, s), can F_d still predict the residual
   depth error? If yes, the final token contains depth evidence not preserved by
   the scalar UDE representation.

3) Optional distribution probe:
   A simple depth-bin classifier over F_d estimates whether a distributional
   representation has useful top-k alternatives. This is evidence for a
   distributional direction, not a proof of true multi-modality.
"""

import argparse
import json
import os
import sys
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mvn import datasets
from mvn.models.DGPose import DepthGuidedPose
from mvn.utils.cfg import config, update_config


JOINT_NAMES = [
    "Pelvis", "R_Hip", "R_Knee", "R_Ankle", "L_Hip", "L_Knee", "L_Ankle",
    "Torso", "Neck", "Nose", "Head", "L_Shoulder", "L_Elbow", "L_Wrist",
    "R_Shoulder", "R_Elbow", "R_Wrist",
]

AMBIGUOUS_PAIRS = [
    (13, 16), (12, 15), (3, 6), (2, 5),
    (13, 7), (16, 7), (13, 8), (16, 8),
]


def safe_corr(x, y, rank=False):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    if rank:
        x = rankdata_average(x)
        y = rankdata_average(y)
    return float(np.corrcoef(x, y)[0, 1])


def rankdata_average(x):
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


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    denom = np.sum((y_true - y_true.mean()) ** 2)
    if denom < 1e-12:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return {
        "r2": r2_score(y_true, y_pred),
        "pearson": safe_corr(y_true, y_pred, rank=False),
        "spearman": safe_corr(y_true, y_pred, rank=True),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "target_std": float(np.std(y_true)),
        "mae_over_std": float(np.mean(np.abs(y_true - y_pred)) / (np.std(y_true) + 1e-12)),
    }


def standardize_train_test(x_train, x_test):
    mu = x_train.mean(axis=0, keepdims=True)
    std = np.maximum(x_train.std(axis=0, keepdims=True), 1e-6)
    return (x_train - mu) / std, (x_test - mu) / std


def ridge_fit_predict(x_train, y_train, x_test, ridge):
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1, 1)
    x_test = np.asarray(x_test, dtype=np.float64)
    x_train, x_test = standardize_train_test(x_train, x_test)
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    x_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)
    reg = ridge * np.eye(x_train.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x_train.T @ x_train + reg, x_train.T @ y_train)
    return (x_test @ weights).reshape(-1), weights


def ridge_predict_with_weights(x_train_ref, x, weights):
    # Refit-free prediction is only used where train/test standardization is
    # not needed; keep the main path through ridge_fit_predict.
    raise NotImplementedError


def make_loader(num_samples, batch_size, seed, num_workers):
    val_dataset = eval("datasets." + config.dataset.val_dataset)(
        root=config.dataset.root,
        pred_results_path=config.val.pred_results_path,
        depth_image_path=config.dataset.depth_image_path,
        train=False,
        test=True,
        image_shape=config.model.image_shape,
        labels_path=config.dataset.val_labels_path,
        with_damaged_actions=config.val.with_damaged_actions,
        retain_every_n_frames_in_test=100,
        scale_bbox=config.val.scale_bbox,
        kind=config.kind,
        undistort_images=config.val.undistort_images,
        data_format=config.dataset.data_format,
        frame=1,
    )
    rng = np.random.default_rng(seed)
    n = min(num_samples, len(val_dataset))
    idx = rng.choice(len(val_dataset), n, replace=False)
    subset = Subset(val_dataset, idx.tolist())
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, n


def build_model(device):
    model = DepthGuidedPose(config, device)
    model.to(device)
    model.eval()
    return model


def load_ckpt(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    msg = model.load_state_dict(ckpt, strict=False)
    print(f"[model] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")


@torch.no_grad()
def collect(model, loader, device, limit):
    fd_chunks = []
    mu_chunks = []
    s_chunks = []
    gt_z_chunks = []

    def hook(_module, _inputs, output):
        fd_chunks.append(output[:, -1].detach().float().cpu())

    handle = model.Lifting_net.RGBD_Extraction[-1].register_forward_hook(hook)

    seen = 0
    for batch in loader:
        if len(batch) != 5:
            continue
        images, keypoints_3d_gt, keypoints_2d, keypoints_2d_crop, depth_images = batch
        images = images.float().to(device) / 255.0
        keypoints_3d_gt = keypoints_3d_gt.float().to(device)
        keypoints_2d = keypoints_2d.float().to(device)
        keypoints_2d_crop = keypoints_2d_crop.float().to(device)
        depth_images = depth_images.float().to(device)

        _pred, coarse_depth, uncer = model(
            images, keypoints_2d, keypoints_2d_crop.clone(), depth_images
        )
        if keypoints_3d_gt.dim() == 4:
            gt_z = keypoints_3d_gt[:, 0, :, 2]
        else:
            gt_z = keypoints_3d_gt[..., 2]
        mu_chunks.append(coarse_depth.detach().squeeze(-1).float().cpu())
        s_chunks.append(uncer.detach().squeeze(-1).float().cpu())
        gt_z_chunks.append(gt_z.detach().float().cpu())
        seen += gt_z.shape[0]
        print(f"[capture] {min(seen, limit)}/{limit}", flush=True)

    handle.remove()
    fd = torch.cat(fd_chunks, dim=0)[:limit].numpy()
    mu = torch.cat(mu_chunks, dim=0)[:limit].numpy()
    s = torch.cat(s_chunks, dim=0)[:limit].numpy()
    gt_z = torch.cat(gt_z_chunks, dim=0)[:limit].numpy()
    return fd, mu, s, gt_z


def one_hot_joint(num_frames, num_joints):
    eye = np.eye(num_joints, dtype=np.float64)
    return np.tile(eye[None, :, :], (num_frames, 1, 1))


def flatten_joint_features(fd, mu, s, gt_z, frame_idx):
    jhot = one_hot_joint(fd.shape[0], fd.shape[1])
    y = gt_z[frame_idx].reshape(-1)
    scalar = np.concatenate(
        [
            mu[frame_idx, :, None],
            s[frame_idx, :, None],
            jhot[frame_idx],
        ],
        axis=-1,
    ).reshape(-1, 2 + fd.shape[1])
    mu_only = np.concatenate(
        [mu[frame_idx, :, None], jhot[frame_idx]], axis=-1
    ).reshape(-1, 1 + fd.shape[1])
    fd_feat = np.concatenate([fd[frame_idx], jhot[frame_idx]], axis=-1).reshape(
        -1, fd.shape[-1] + fd.shape[1]
    )
    fd_scalar = np.concatenate(
        [fd[frame_idx], mu[frame_idx, :, None], s[frame_idx, :, None], jhot[frame_idx]],
        axis=-1,
    ).reshape(-1, fd.shape[-1] + 2 + fd.shape[1])
    return {
        "y": y,
        "mu_only": mu_only,
        "scalar": scalar,
        "fd": fd_feat,
        "fd_scalar": fd_scalar,
    }


def scalar_bottleneck_probe(fd, mu, s, gt_z, train_idx, test_idx, ridge):
    train = flatten_joint_features(fd, mu, s, gt_z, train_idx)
    test = flatten_joint_features(fd, mu, s, gt_z, test_idx)
    y_train = train["y"]
    y_test = test["y"]

    out = {}
    preds_train = {}
    preds_test = {}
    for name in ["mu_only", "scalar", "fd", "fd_scalar"]:
        pred_test, _weights = ridge_fit_predict(train[name], y_train, test[name], ridge)
        pred_train, _ = ridge_fit_predict(train[name], y_train, train[name], ridge)
        out[name] = regression_metrics(y_test, pred_test)
        preds_train[name] = pred_train
        preds_test[name] = pred_test

    scalar_resid_train = y_train - preds_train["scalar"]
    scalar_resid_test = y_test - preds_test["scalar"]
    fd_resid_pred, _ = ridge_fit_predict(train["fd"], scalar_resid_train, test["fd"], ridge)
    scalar_plus_resid = preds_test["scalar"] + fd_resid_pred
    out["fd_predicts_scalar_residual"] = regression_metrics(scalar_resid_test, fd_resid_pred)
    out["scalar_plus_fd_residual"] = regression_metrics(y_test, scalar_plus_resid)
    out["residual_mae_reduction_over_scalar_percent"] = float(
        100.0 * (out["scalar"]["mae"] - out["scalar_plus_fd_residual"]["mae"])
        / (out["scalar"]["mae"] + 1e-12)
    )
    return out


def build_pairwise_features(fd, mu, s, gt_z, frame_idx, pairs):
    scalar_parts = []
    fd_parts = []
    y_parts = []
    for i, j in pairs:
        scalar_parts.append(
            np.stack(
                [
                    mu[frame_idx, i] - mu[frame_idx, j],
                    s[frame_idx, i],
                    s[frame_idx, j],
                ],
                axis=-1,
            )
        )
        fd_parts.append(fd[frame_idx, i] - fd[frame_idx, j])
        y_parts.append(gt_z[frame_idx, i] - gt_z[frame_idx, j])
    return (
        np.concatenate(scalar_parts, axis=0),
        np.concatenate(fd_parts, axis=0),
        np.concatenate(y_parts, axis=0),
    )


def ordinal_metrics(delta_true, delta_pred, margin):
    delta_true = np.asarray(delta_true).reshape(-1)
    delta_pred = np.asarray(delta_pred).reshape(-1)
    valid = np.abs(delta_true) > 1e-8
    significant = np.abs(delta_true) > margin
    return {
        "r2": r2_score(delta_true, delta_pred),
        "spearman": safe_corr(delta_true, delta_pred, rank=True),
        "ordinal_acc": float((np.sign(delta_pred[valid]) == np.sign(delta_true[valid])).mean()),
        "ordinal_acc_margin": float(
            (np.sign(delta_pred[significant]) == np.sign(delta_true[significant])).mean()
        ) if significant.any() else float("nan"),
    }


def pairwise_probe(fd, mu, s, gt_z, train_idx, test_idx, ridge, margin):
    all_pairs = list(combinations(range(fd.shape[1]), 2))
    out = {}
    for pair_name, pairs in [("all", all_pairs), ("ambiguous", AMBIGUOUS_PAIRS)]:
        scalar_train, fd_train, y_train = build_pairwise_features(fd, mu, s, gt_z, train_idx, pairs)
        scalar_test, fd_test, y_test = build_pairwise_features(fd, mu, s, gt_z, test_idx, pairs)
        scalar_pred, _ = ridge_fit_predict(scalar_train, y_train, scalar_test, ridge)
        fd_pred, _ = ridge_fit_predict(fd_train, y_train, fd_test, ridge)
        out[pair_name] = {
            "scalar_mu_s": ordinal_metrics(y_test, scalar_pred, margin),
            "fd": ordinal_metrics(y_test, fd_pred, margin),
        }
    return out


class FlatJointDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def make_depth_bins(y_train, bin_count):
    # Quantile bins keep classes reasonably balanced for a light diagnostic.
    qs = np.linspace(0.0, 1.0, bin_count + 1)
    edges = np.quantile(y_train, qs)
    edges = np.unique(edges)
    if len(edges) < 4:
        raise RuntimeError("Too few unique depth-bin edges. Increase samples or reduce --bin_count.")
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def assign_bins(y, edges):
    return np.clip(np.searchsorted(edges[1:-1], y, side="right"), 0, len(edges) - 2)


def count_distribution_peaks(prob, threshold):
    peaks = np.zeros(prob.shape[0], dtype=np.int64)
    for row_idx, p in enumerate(prob):
        count = 0
        for k in range(len(p)):
            left_ok = k == 0 or p[k] >= p[k - 1]
            right_ok = k == len(p) - 1 or p[k] >= p[k + 1]
            if left_ok and right_ok and p[k] >= threshold:
                count += 1
        peaks[row_idx] = count
    return peaks


def distribution_probe(fd, mu, s, gt_z, train_idx, test_idx, args, device):
    train = flatten_joint_features(fd, mu, s, gt_z, train_idx)
    test = flatten_joint_features(fd, mu, s, gt_z, test_idx)
    y_train = train["y"]
    y_test = test["y"]
    edges, centers = make_depth_bins(y_train, args.bin_count)
    yb_train = assign_bins(y_train, edges)
    yb_test = assign_bins(y_test, edges)

    x_train = train["fd"]
    x_test = test["fd"]
    x_train, x_test = standardize_train_test(x_train, x_test)
    train_ds = FlatJointDataset(x_train, yb_train)
    train_loader = DataLoader(train_ds, batch_size=args.probe_batch_size, shuffle=True)

    model = nn.Sequential(
        nn.Linear(x_train.shape[1], args.probe_hidden),
        nn.GELU(),
        nn.Linear(args.probe_hidden, len(centers)),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.probe_lr, weight_decay=1e-4)

    model.train()
    for epoch in range(args.probe_epochs):
        total_loss = 0.0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.shape[0]
            total += xb.shape[0]
        if (epoch + 1) % max(1, args.probe_epochs // 5) == 0:
            print(f"[dist-probe] epoch={epoch + 1}/{args.probe_epochs} loss={total_loss / max(total, 1):.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_test).float().to(device)).cpu().numpy()
    prob = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob = prob / np.maximum(prob.sum(axis=1, keepdims=True), 1e-12)
    centers_np = centers.astype(np.float64)
    expected = prob @ centers_np
    top = np.argsort(-prob, axis=1)
    top1 = top[:, 0]
    top2 = top[:, :2]
    top3 = top[:, :3]
    entropy = -np.sum(prob * np.log(np.maximum(prob, 1e-12)), axis=1)
    peaks = count_distribution_peaks(prob, args.peak_threshold)

    # Scalar baseline: train-calibrated scalar probe converted to bins.
    scalar_pred, _ = ridge_fit_predict(train["scalar"], y_train, test["scalar"], args.ridge)
    scalar_bin = assign_bins(scalar_pred, edges)

    in_top2 = np.any(top2 == yb_test[:, None], axis=1)
    in_top3 = np.any(top3 == yb_test[:, None], axis=1)
    return {
        "bin_count": int(len(centers)),
        "bin_edges_minmax": [float(edges[0]), float(edges[-1])],
        "expected_depth_metrics": regression_metrics(y_test, expected),
        "scalar_probe_bin_top1_acc": float((scalar_bin == yb_test).mean()),
        "fd_distribution_top1_acc": float((top1 == yb_test).mean()),
        "fd_distribution_top2_acc": float(in_top2.mean()),
        "fd_distribution_top3_acc": float(in_top3.mean()),
        "top2_gain_over_top1": float(in_top2.mean() - (top1 == yb_test).mean()),
        "entropy_error_spearman": safe_corr(entropy, np.abs(expected - y_test), rank=True),
        "multi_peak_rate": float((peaks >= 2).mean()),
        "mean_num_peaks": float(peaks.mean()),
        "note": (
            "Top-k and peak statistics only suggest whether a distributional "
            "depth representation is useful; they do not by themselves prove "
            "the true posterior is multi-modal."
        ),
    }


def interpret_report(report):
    scalar = report["scalar_bottleneck_probe"]["scalar"]
    fd = report["scalar_bottleneck_probe"]["fd"]
    combined = report["scalar_bottleneck_probe"]["scalar_plus_fd_residual"]
    resid_gain = report["scalar_bottleneck_probe"]["residual_mae_reduction_over_scalar_percent"]
    fd_ord = report["pairwise_depth_probe"]["all"]["fd"]["ordinal_acc_margin"]
    scalar_ord = report["pairwise_depth_probe"]["all"]["scalar_mu_s"]["ordinal_acc_margin"]

    claims = []
    if fd["mae_over_std"] + 1e-6 < scalar["mae_over_std"] and resid_gain > 2.0:
        claims.append(
            "Supported: F_d contains depth information not preserved by the scalar UDE outputs (mu, s)."
        )
    else:
        claims.append(
            "Not supported yet: F_d did not clearly improve over the scalar UDE outputs in this run."
        )
    if fd_ord > scalar_ord + 0.05:
        claims.append(
            "Supported: F_d preserves pairwise depth/order evidence beyond scalar UDE outputs."
        )
    else:
        claims.append(
            "Weak: pairwise depth/order evidence is not clearly stronger than scalar UDE outputs."
        )
    if "distribution_probe" in report:
        dist = report["distribution_probe"]
        if dist["top2_gain_over_top1"] > 0.10:
            claims.append(
                "Suggestive only: distributional top-k alternatives may be useful for ambiguous depth cases."
            )
        else:
            claims.append(
                "No strong distributional evidence from the simple depth-bin probe."
            )
    claims.append(
        "Important: these diagnostics can prove a scalar bottleneck, but true multi-modality requires stronger evidence or a trained distributional module ablation."
    )
    return claims


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_samples", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--ordinal_margin", type=float, default=0.05)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_distribution_probe", action="store_true")
    p.add_argument("--bin_count", type=int, default=17)
    p.add_argument("--probe_hidden", type=int, default=128)
    p.add_argument("--probe_epochs", type=int, default=25)
    p.add_argument("--probe_batch_size", type=int, default=2048)
    p.add_argument("--probe_lr", type=float, default=2e-3)
    p.add_argument("--peak_threshold", type=float, default=0.10)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    update_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}")

    model = build_model(device)
    load_ckpt(model, args.checkpoint, device)
    loader, used_n = make_loader(args.num_samples, args.batch_size, args.seed, args.num_workers)
    fd, mu, s, gt_z = collect(model, loader, device, used_n)

    num_frames = fd.shape[0]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(num_frames)
    split = max(2, int(0.7 * num_frames))
    train_idx = np.sort(order[:split])
    test_idx = np.sort(order[split:])

    report = {
        "meta": {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "num_samples_requested": args.num_samples,
            "num_samples_used": int(num_frames),
            "train_frames": int(len(train_idx)),
            "test_frames": int(len(test_idx)),
            "fd_shape": list(fd.shape),
        },
        "scalar_bottleneck_probe": scalar_bottleneck_probe(
            fd, mu, s, gt_z, train_idx, test_idx, args.ridge
        ),
        "pairwise_depth_probe": pairwise_probe(
            fd, mu, s, gt_z, train_idx, test_idx, args.ridge, args.ordinal_margin
        ),
        "ude_raw_statistics": {
            "mu_mean": float(np.mean(mu)),
            "mu_std": float(np.std(mu)),
            "s_mean": float(np.mean(s)),
            "s_std": float(np.std(s)),
            "s_min": float(np.min(s)),
            "s_max": float(np.max(s)),
            "abs_mu_error_spearman_with_s": safe_corr(np.abs(mu - gt_z), s, rank=True),
        },
    }
    if args.run_distribution_probe:
        report["distribution_probe"] = distribution_probe(
            fd, mu, s, gt_z, train_idx, test_idx, args, device
        )
    report["interpretation"] = interpret_report(report)

    out_path = os.path.join(args.output_dir, "ude_distributional_gap_diagnostics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "scalar_bottleneck_probe": report["scalar_bottleneck_probe"],
        "pairwise_depth_probe": report["pairwise_depth_probe"],
        "interpretation": report["interpretation"],
    }, indent=2, ensure_ascii=False))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
