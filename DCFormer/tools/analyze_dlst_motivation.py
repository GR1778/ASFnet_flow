#!/usr/bin/env python3
"""
DLST motivation diagnostics.

This script tests whether the "Depth-Layer Sorting Transformer" premise is
supported by data and current AMS features:

1) Can GT joint depth order be approximated by a small number of ordered
   depth layers?
2) Can AMS output F_d predict those depth-layer assignments?
3) Does a layer-derived relation matrix R=AΩA^T provide a cleaner depth-order
   signal than the current scalar coarse-depth head?

It does not train DLST. It only verifies that the proposed intermediate
structure is plausible and measurable.
"""

import argparse
import json
import math
import os
import sys
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mvn import datasets
from mvn.datasets.human36m import retval
from mvn.datasets import utils as dataset_utils
from mvn.models.DGPose import DepthGuidedPose
from mvn.utils.cfg import config, update_config


AMBIGUOUS_PAIRS = [
    (13, 16),  # wrists
    (12, 15),  # elbows
    (3, 6),    # ankles
    (2, 5),    # knees
    (13, 7), (16, 7),
    (13, 8), (16, 8),
]


BONES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_samples", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="dlst_motivation_analysis")
    p.add_argument("--retain_every_n_frames", type=int, default=None)
    p.add_argument("--k_values", default="3,4,5,6")
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=0.05, help="Significant depth gap threshold in dataset units. 0.05 ~= 50mm if units are meters.")
    p.add_argument("--probe_steps", type=int, default=250)
    p.add_argument("--random_subset", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--per_action", action="store_true")
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


def build_dataset(args: argparse.Namespace):
    retain = (
        args.retain_every_n_frames
        if args.retain_every_n_frames is not None
        else config.val.retain_every_n_frames_in_test
    )
    return eval("datasets." + config.dataset.val_dataset)(
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


def select_indices(args: argparse.Namespace, dataset) -> np.ndarray:
    n = min(args.num_samples, len(dataset))
    if args.random_subset:
        rng = np.random.default_rng(args.seed)
        return np.sort(rng.choice(len(dataset), size=n, replace=False))
    return np.arange(n)


def build_loader(args: argparse.Namespace):
    val_dataset = build_dataset(args)
    indices = select_indices(args, val_dataset)
    subset = Subset(val_dataset, indices.tolist())
    return DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=True,
    ), val_dataset, indices


def load_model(args: argparse.Namespace, device: torch.device) -> DepthGuidedPose:
    model = DepthGuidedPose(config, device).to(device)
    raw = torch.load(args.checkpoint, map_location=device)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    state = {k.replace("module.", ""): v for k, v in state.items()}
    msg = model.load_state_dict(state, strict=False)
    print(f"[model] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    model.eval()
    return model


def capture_ams(args: argparse.Namespace, model: DepthGuidedPose, loader: DataLoader, device: torch.device):
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


def make_omega(k: int, tau: float) -> np.ndarray:
    idx = np.arange(k, dtype=np.float64)
    return np.tanh((idx[None, :] - idx[:, None]) / tau)


def rank_layer_labels(z: np.ndarray, k: int) -> np.ndarray:
    """Assign each frame's joints to K ordered layers by GT depth rank."""
    n, j = z.shape
    labels = np.zeros((n, j), dtype=np.int64)
    for bi in range(n):
        order = np.argsort(z[bi])
        ranks = np.empty(j, dtype=np.int64)
        ranks[order] = np.arange(j)
        labels[bi] = np.floor(ranks * k / j).astype(np.int64)
        labels[bi] = np.clip(labels[bi], 0, k - 1)
    return labels


def labels_to_rel(labels: np.ndarray, k: int, tau: float) -> np.ndarray:
    omega = make_omega(k, tau)
    return omega[labels[:, :, None], labels[:, None, :]]


def pair_indices(j: int, pair_set: str = "all") -> List[Tuple[int, int]]:
    if pair_set == "bones":
        return BONES
    if pair_set == "ambiguous":
        return AMBIGUOUS_PAIRS
    return list(combinations(range(j), 2))


def ordinal_metrics_from_rel(
    rel: np.ndarray,
    gt_z: np.ndarray,
    delta: float,
    pairs: List[Tuple[int, int]],
) -> Dict[str, float]:
    y_all, p_all, abs_rel = [], [], []
    for i, j in pairs:
        y = gt_z[:, j] - gt_z[:, i]
        p = rel[:, i, j]
        mask = np.abs(y) > delta
        if mask.any():
            y_all.append(np.sign(y[mask]))
            p_all.append(np.sign(p[mask]))
            abs_rel.append(np.abs(p[mask]))
    if not y_all:
        return {
            "num_eval_pairs": 0,
            "resolved_rate": float("nan"),
            "ordinal_acc_zero_wrong": float("nan"),
            "ordinal_acc_resolved_only": float("nan"),
            "mean_abs_relation": float("nan"),
        }
    yv = np.concatenate(y_all)
    pv = np.concatenate(p_all)
    av = np.concatenate(abs_rel)
    resolved = pv != 0
    zero_wrong = (pv == yv) & resolved
    return {
        "num_eval_pairs": int(yv.size),
        "resolved_rate": float(resolved.mean()),
        "ordinal_acc_zero_wrong": float(zero_wrong.mean()),
        "ordinal_acc_resolved_only": float((pv[resolved] == yv[resolved]).mean()) if resolved.any() else float("nan"),
        "mean_abs_relation": float(av.mean()),
    }


def coarse_mu_metrics(mu: np.ndarray, gt_z: np.ndarray, delta: float, pairs: List[Tuple[int, int]]) -> Dict[str, float]:
    direct, inverse = [], []
    for i, j in pairs:
        y = gt_z[:, j] - gt_z[:, i]
        p = mu[:, j] - mu[:, i]
        mask = np.abs(y) > delta
        if mask.any():
            ys = np.sign(y[mask])
            ps = np.sign(p[mask])
            direct.append((ps == ys).astype(np.float64))
            inverse.append((-ps == ys).astype(np.float64))
    if not direct:
        return {"direct": float("nan"), "inverse": float("nan"), "best": float("nan")}
    direct_v = np.concatenate(direct)
    inverse_v = np.concatenate(inverse)
    return {
        "direct": float(direct_v.mean()),
        "inverse": float(inverse_v.mean()),
        "best": float(max(direct_v.mean(), inverse_v.mean())),
    }


def fd_geometry_metrics(fd: np.ndarray, gt_z: np.ndarray, pairs: List[Tuple[int, int]]) -> Dict[str, float]:
    feat_dist, depth_gap = [], []
    for i, j in pairs:
        feat_dist.append(np.linalg.norm(fd[:, i] - fd[:, j], axis=-1))
        depth_gap.append(np.abs(gt_z[:, i] - gt_z[:, j]))
    feat_dist = np.stack(feat_dist, axis=1)
    depth_gap = np.stack(depth_gap, axis=1)
    return {
        "spearman_fd_distance_vs_abs_depth_gap": safe_spearman(feat_dist, depth_gap),
        "pearson_fd_distance_vs_abs_depth_gap": safe_corr(feat_dist, depth_gap),
    }


def train_layer_probe(
    fd: np.ndarray,
    labels: np.ndarray,
    k: int,
    steps: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    n, j, c = fd.shape
    split = max(1, int(0.7 * n))
    x_train = fd[:split].reshape(-1, c)
    y_train = labels[:split].reshape(-1)
    x_test = fd[split:].reshape(-1, c)
    y_test = labels[split:].reshape(-1)

    mu = x_train.mean(axis=0, keepdims=True)
    std = np.maximum(x_train.std(axis=0, keepdims=True), 1e-6)
    x_train = (x_train - mu) / std
    x_test = (x_test - mu) / std

    xt = torch.from_numpy(x_train).float().to(device)
    yt = torch.from_numpy(y_train).long().to(device)
    xv = torch.from_numpy(x_test).float().to(device)
    model = nn.Linear(c, k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=1e-4)

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(xt), yt)
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = model(xv)
        prob = torch.softmax(logits, dim=-1).cpu().numpy()
        pred = prob.argmax(axis=-1)
    return {
        "pred_labels": pred.reshape(n - split, j),
        "prob": prob.reshape(n - split, j, k),
        "true_labels": y_test.reshape(n - split, j),
        "test_frame_start": np.asarray(split),
    }


def soft_assign_to_rel(assign: np.ndarray, k: int, tau: float) -> np.ndarray:
    omega = make_omega(k, tau)
    return assign @ omega @ np.swapaxes(assign, -1, -2)


def occupancy_stats(labels: np.ndarray, k: int) -> Dict[str, float]:
    hist = np.zeros(k, dtype=np.float64)
    for kk in range(k):
        hist[kk] = np.mean(labels == kk)
    return {
        "min_layer_occupancy": float(hist.min()),
        "max_layer_occupancy": float(hist.max()),
        "layer_occupancy": [float(x) for x in hist],
    }


def analyze_k(
    fd: np.ndarray,
    mu: np.ndarray,
    gt_z: np.ndarray,
    k: int,
    tau: float,
    delta: float,
    steps: int,
    device: torch.device,
) -> Dict[str, object]:
    labels = rank_layer_labels(gt_z, k)
    oracle_rel = labels_to_rel(labels, k, tau)

    all_pairs = pair_indices(gt_z.shape[1], "all")
    bones = pair_indices(gt_z.shape[1], "bones")
    ambiguous = pair_indices(gt_z.shape[1], "ambiguous")

    probe = train_layer_probe(fd, labels, k, steps, device)
    split = int(probe["test_frame_start"])
    test_gt_z = gt_z[split:]
    hard_rel = labels_to_rel(probe["pred_labels"], k, tau)
    soft_rel = soft_assign_to_rel(probe["prob"], k, tau)

    true = probe["true_labels"].reshape(-1)
    pred = probe["pred_labels"].reshape(-1)
    layer_acc = float((pred == true).mean())
    adjacent_acc = float((np.abs(pred - true) <= 1).mean())

    return {
        "k": k,
        "gt_layer_occupancy": occupancy_stats(labels, k),
        "oracle_layer_R": {
            "all": ordinal_metrics_from_rel(oracle_rel, gt_z, delta, all_pairs),
            "bones": ordinal_metrics_from_rel(oracle_rel, gt_z, delta, bones),
            "ambiguous": ordinal_metrics_from_rel(oracle_rel, gt_z, delta, ambiguous),
        },
        "fd_linear_layer_probe": {
            "layer_acc": layer_acc,
            "adjacent_layer_acc": adjacent_acc,
            "hard_R_all": ordinal_metrics_from_rel(hard_rel, test_gt_z, delta, all_pairs),
            "soft_R_all": ordinal_metrics_from_rel(soft_rel, test_gt_z, delta, all_pairs),
            "soft_R_bones": ordinal_metrics_from_rel(soft_rel, test_gt_z, delta, bones),
            "soft_R_ambiguous": ordinal_metrics_from_rel(soft_rel, test_gt_z, delta, ambiguous),
        },
        "current_coarse_mu": {
            "all": coarse_mu_metrics(mu, gt_z, delta, all_pairs),
            "bones": coarse_mu_metrics(mu, gt_z, delta, bones),
            "ambiguous": coarse_mu_metrics(mu, gt_z, delta, ambiguous),
        },
        "raw_fd_geometry": {
            "all": fd_geometry_metrics(fd, gt_z, all_pairs),
            "bones": fd_geometry_metrics(fd, gt_z, bones),
            "ambiguous": fd_geometry_metrics(fd, gt_z, ambiguous),
        },
    }


def action_names_for_indices(dataset, indices: np.ndarray) -> List[str]:
    action_idx = getattr(dataset, "labels_action_idx", None)
    if action_idx is None:
        return ["unknown"] * len(indices)
    names = retval["action_names"]
    return [names[int(action_idx[int(i)])] for i in indices]


def combine_trial_action(name: str) -> str:
    if name.endswith("-1") or name.endswith("-2"):
        return name[:-2]
    return name


def per_action_summary_for_k(
    fd: np.ndarray,
    mu: np.ndarray,
    gt_z: np.ndarray,
    action_names: List[str],
    k: int,
    tau: float,
    delta: float,
) -> Dict[str, Dict[str, float]]:
    labels = rank_layer_labels(gt_z, k)
    oracle_rel = labels_to_rel(labels, k, tau)
    pairs = pair_indices(gt_z.shape[1], "all")
    ambiguous = pair_indices(gt_z.shape[1], "ambiguous")
    grouped: Dict[str, List[int]] = {}
    for idx, name in enumerate(action_names):
        grouped.setdefault(combine_trial_action(name), []).append(idx)

    out: Dict[str, Dict[str, float]] = {}
    for action, idxs in sorted(grouped.items()):
        if len(idxs) < 16:
            continue
        idx = np.asarray(idxs, dtype=np.int64)
        oracle_all = ordinal_metrics_from_rel(oracle_rel[idx], gt_z[idx], delta, pairs)
        oracle_amb = ordinal_metrics_from_rel(oracle_rel[idx], gt_z[idx], delta, ambiguous)
        mu_all = coarse_mu_metrics(mu[idx], gt_z[idx], delta, pairs)
        geom = fd_geometry_metrics(fd[idx], gt_z[idx], pairs)
        out[action] = {
            "num_samples": int(len(idx)),
            "oracle_k_layer_all_acc": oracle_all["ordinal_acc_zero_wrong"],
            "oracle_k_layer_ambiguous_acc": oracle_amb["ordinal_acc_zero_wrong"],
            "coarse_mu_all_best_acc": mu_all["best"],
            "raw_fd_spearman": geom["spearman_fd_distance_vs_abs_depth_gap"],
        }
    return out


def make_verdict(k_results: List[Dict[str, object]]) -> Dict[str, object]:
    k4 = None
    for item in k_results:
        if item["k"] == 4:
            k4 = item
            break
    ref = k4 if k4 is not None else k_results[0]
    oracle_acc = ref["oracle_layer_R"]["all"]["ordinal_acc_zero_wrong"]
    probe_acc = ref["fd_linear_layer_probe"]["soft_R_all"]["ordinal_acc_zero_wrong"]
    adjacent = ref["fd_linear_layer_probe"]["adjacent_layer_acc"]
    raw_s = ref["raw_fd_geometry"]["all"]["spearman_fd_distance_vs_abs_depth_gap"]
    mu_best = ref["current_coarse_mu"]["all"]["best"]
    reasons = []
    if oracle_acc > 0.85:
        reasons.append("少量有序深度层可以近似 GT 前后排序。")
    if adjacent > 0.85:
        reasons.append("AMS F_d 可以较可靠地预测关节属于哪个相邻深度层。")
    if probe_acc > mu_best:
        reasons.append("由层分配推导的 R 比当前 coarse depth head 更适合表达前后顺序。")
    if np.isfinite(raw_s) and raw_s < 0.2:
        reasons.append("原始 F_d 几何本身并未显式组织真实深度层次。")

    level = "strong" if len(reasons) >= 3 else "moderate" if len(reasons) >= 2 else "weak"
    return {
        "dlst_motivation": level,
        "reference_k": int(ref["k"]),
        "reasons": reasons,
        "interpretation": (
            "DLST 的关键前提不是直接预测任意 pairwise depth，而是用少量有序层作为全局一致的中间结构。"
        ),
    }


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    update_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]

    print(f"[setup] device={device}")
    print(f"[setup] k_values={k_values} delta={args.delta} tau={args.tau}")
    model = load_model(args, device)
    loader, dataset, selected_indices = build_loader(args)
    fd, mu, gt = capture_ams(args, model, loader, device)
    gt_z = gt[..., 2]
    action_names = action_names_for_indices(dataset, selected_indices[: fd.shape[0]])

    results = []
    for k in k_values:
        print(f"[analyze] K={k}", flush=True)
        results.append(analyze_k(fd, mu, gt_z, k, args.tau, args.delta, args.probe_steps, device))

    summary = {
        "num_samples": int(fd.shape[0]),
        "fd_shape": list(fd.shape),
        "delta": args.delta,
        "tau": args.tau,
        "subset": {
            "random_subset": bool(args.random_subset),
            "seed": int(args.seed),
            "selected_index_min": int(selected_indices[: fd.shape[0]].min()) if len(selected_indices) else 0,
            "selected_index_max": int(selected_indices[: fd.shape[0]].max()) if len(selected_indices) else 0,
        },
        "k_results": results,
    }
    if args.per_action:
        ref_k = 4 if 4 in k_values else k_values[0]
        summary["per_action_k{}".format(ref_k)] = per_action_summary_for_k(
            fd, mu, gt_z, action_names, ref_k, args.tau, args.delta
        )
    summary["verdict"] = make_verdict(results)

    out_json = os.path.join(args.output_dir, "summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[done] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
