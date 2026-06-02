#!/usr/bin/env python3
"""
AMS sampled-evidence diagnostics.

This script does not question whether AMS is useful. Instead, it tests whether
the sampled candidates inside AMS still contain residual reliability gaps after
AMS has learned offsets and weights.

The main question is:
    Does AMS already weight the best depth evidence, or is there an oracle gap
    that motivates a sampling-level reliability module before UDE/fusion?
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mvn import datasets
from mvn.datasets import utils as dataset_utils
from mvn.models.DGPose_dlst import DepthGuidedPoseDLST
from mvn.utils.cfg import config, update_config, update_dir


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
    parser = argparse.ArgumentParser(description="Analyze residual reliability gaps inside AMS sampled evidence.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="ams_evidence_reliability/summary.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", default="hrnet_32", choices=["hrnet_32", "hrnet_48"])
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--random-subset", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--block-index", type=int, default=-1, help="Which AMS block to inspect; -1 means the last block.")
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


def build_pairwise_dataset(features: np.ndarray, gt_z: np.ndarray, frame_idx: np.ndarray, pairs: List[Tuple[int, int]]):
    x_parts = []
    y_parts = []
    for left, right in pairs:
        x_parts.append(features[frame_idx, right] - features[frame_idx, left])
        y_parts.append(gt_z[frame_idx, right] - gt_z[frame_idx, left])
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def pairwise_metrics(features: np.ndarray, gt_z: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, pairs, ridge, delta):
    x_train, y_train = build_pairwise_dataset(features, gt_z, train_idx, pairs)
    x_test, y_test = build_pairwise_dataset(features, gt_z, test_idx, pairs)
    pred = ridge_fit_predict(x_train, y_train, x_test, ridge)
    valid = np.abs(y_test) > 1e-8
    significant = np.abs(y_test) > delta
    return {
        "pearson": safe_corr(y_test, pred),
        "spearman": safe_spearman(y_test, pred),
        "ordinal_acc": float((np.sign(pred[valid]) == np.sign(y_test[valid])).mean()) if valid.any() else float("nan"),
        "ordinal_acc_delta": float((np.sign(pred[significant]) == np.sign(y_test[significant])).mean()) if significant.any() else float("nan"),
        "mae": float(np.mean(np.abs(y_test - pred))),
    }


def score_candidates(pair_features: np.ndarray, gt_z: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, ridge: float):
    # pair_features: [N,J,K,C]
    num_candidates = pair_features.shape[2]
    x_train = pair_features[train_idx].reshape(-1, pair_features.shape[-1])
    y_train = np.repeat(gt_z[train_idx].reshape(-1), num_candidates)
    x_test = pair_features[test_idx].reshape(-1, pair_features.shape[-1])
    pred = ridge_fit_predict(x_train, y_train, x_test, ridge)
    return pred.reshape(len(test_idx), pair_features.shape[1], num_candidates)


def candidate_oracle_metrics(candidate_scores: np.ndarray, weights: np.ndarray, gt_z: np.ndarray, test_idx: np.ndarray, pairs, delta):
    # candidate_scores: [Ntest,J,K], weights: [Ntest,J,K]
    gt_test = gt_z[test_idx]
    metrics = {}
    agg = (candidate_scores * weights).sum(axis=-1)
    uniform = candidate_scores.mean(axis=-1)

    def ordinal_acc_for(score):
        y_all = []
        p_all = []
        for left, right in pairs:
            y = gt_test[:, right] - gt_test[:, left]
            pred = score[:, right] - score[:, left]
            mask = np.abs(y) > delta
            if mask.any():
                y_all.append(np.sign(y[mask]))
                p_all.append(np.sign(pred[mask]))
        yv = np.concatenate(y_all)
        pv = np.concatenate(p_all)
        return float((yv == pv).mean())

    oracle = []
    for left, right in pairs:
        y = gt_test[:, right] - gt_test[:, left]
        pred = candidate_scores[:, right, :, None] - candidate_scores[:, left, None, :]
        abs_err = np.abs(pred - y[:, None, None])
        best_pred = pred.reshape(pred.shape[0], -1)[np.arange(pred.shape[0]), abs_err.reshape(abs_err.shape[0], -1).argmin(axis=1)]
        mask = np.abs(y) > delta
        if mask.any():
            oracle.append((np.sign(best_pred[mask]) == np.sign(y[mask])).astype(np.float64))
    oracle_acc = float(np.concatenate(oracle).mean()) if oracle else float("nan")

    gt_expanded = np.repeat(gt_test[:, :, None], candidate_scores.shape[-1], axis=-1)
    abs_err = np.abs(candidate_scores - gt_expanded)
    best_idx = abs_err.argmin(axis=-1)
    top_idx = weights.argmax(axis=-1)
    rank = np.argsort(np.argsort(abs_err, axis=-1), axis=-1)
    top_rank = np.take_along_axis(rank, top_idx[..., None], axis=-1).squeeze(-1)

    metrics["ams_weighted_ordinal_acc_delta"] = ordinal_acc_for(agg)
    metrics["uniform_ordinal_acc_delta"] = ordinal_acc_for(uniform)
    metrics["oracle_pair_ordinal_acc_delta"] = oracle_acc
    metrics["top_weight_is_best_candidate_rate"] = float((top_idx == best_idx).mean())
    metrics["top_weight_candidate_mean_rank"] = float(top_rank.mean())
    metrics["best_candidate_abs_error"] = float(abs_err.min(axis=-1).mean())
    metrics["top_weight_candidate_abs_error"] = float(np.take_along_axis(abs_err, top_idx[..., None], axis=-1).mean())
    metrics["weighted_abs_error"] = float(np.abs(agg - gt_test).mean())
    metrics["uniform_abs_error"] = float(np.abs(uniform - gt_test).mean())
    metrics["weight_error_corr"] = safe_corr(weights, -abs_err)
    return metrics


def build_val_loader(args: argparse.Namespace):
    val_dataset = datasets.human36m(
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
    n = min(args.num_samples, len(val_dataset))
    if args.random_subset:
        rng = np.random.default_rng(args.seed)
        indices = np.sort(rng.choice(len(val_dataset), size=n, replace=False))
    else:
        indices = np.arange(n)
    loader = DataLoader(
        Subset(val_dataset, indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=True,
    )
    return loader, indices


def load_model(args: argparse.Namespace, device: torch.device):
    model = DepthGuidedPoseDLST(config, device=device)
    raw = torch.load(args.checkpoint, map_location="cpu")
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    state = {key.replace("module.", ""): value for key, value in state.items()}
    msg = model.load_state_dict(state, strict=False)
    print(f"[model] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    model.to(device)
    model.eval()
    return model


def capture_ams_evidence(args: argparse.Namespace, model, loader, device: torch.device):
    block = model.Lifting_net.RGBD_Extraction[args.block_index]
    captures = {"candidate": [], "weighted": [], "uniform": [], "weights": [], "gt": []}
    original_forward = block.forward

    def wrapped_forward(x, ref, features_list):
        x_0, x_body = x[:, :1], x[:, 1:]
        b, levels, joints, channels = x_body.shape
        residual = x_body
        x_norm = block.norm1(x_body + x_0)

        weights_raw = block.attention_weights(x_norm).view(b, levels, joints, block.num_heads, block.num_samples)
        weights = F.softmax(weights_raw, dim=-1).unsqueeze(-1)
        offsets = block.sampling_offsets(x_norm).reshape(b, levels, joints, block.num_heads * block.num_samples, 2).tanh()
        pos = offsets + ref.view(b, 1, joints, 1, -1)

        features_sampled = [
            F.grid_sample(features, pos[:, idx], padding_mode="border", align_corners=True)
            .permute(0, 2, 3, 1)
            .contiguous()
            for idx, features in enumerate(features_list)
        ]
        features_sampled = [embed(features_sampled[idx]) for idx, embed in enumerate(block.embed_proj)]
        features_sampled = torch.stack(features_sampled, dim=1)
        sampled = features_sampled.view(b, levels, joints, block.num_heads, block.num_samples, -1)
        weighted = (weights * sampled).sum(dim=-2).view(b, levels, joints, -1)
        uniform = sampled.mean(dim=-2).view(b, levels, joints, -1)

        captures["candidate"].append(sampled[:, -1].reshape(b, joints, block.num_heads * block.num_samples, -1).detach().float().cpu())
        captures["weighted"].append(weighted[:, -1].detach().float().cpu())
        captures["uniform"].append(uniform[:, -1].detach().float().cpu())
        captures["weights"].append(weights[:, -1].squeeze(-1).reshape(b, joints, block.num_heads * block.num_samples).detach().float().cpu())

        x_out = residual + block.drop_path(weighted)
        x_out = x_out + block.drop_path(block.mlp(block.norm2(x_out)))
        return torch.cat([x_0, x_out], dim=1)

    block.forward = wrapped_forward
    prefetcher = dataset_utils.data_prefetcher(loader, device, is_train=False, flip_test=False)
    batch = prefetcher.next()
    seen = 0
    with torch.no_grad():
        while batch is not None and seen < args.num_samples:
            images, gt_3d, kp2d, kp2d_crop, depth_images = batch
            _pred, _rel_depth, _layer_assign = model(images, kp2d, kp2d_crop.clone(), depth_images)
            if gt_3d.dim() == 4:
                gt_3d = gt_3d.squeeze(1)
            captures["gt"].append(gt_3d.detach().float().cpu())
            seen += gt_3d.shape[0]
            print(f"[capture] {min(seen, args.num_samples)}/{args.num_samples}", flush=True)
            batch = prefetcher.next()
    block.forward = original_forward

    return {key: torch.cat(value, dim=0)[: args.num_samples].numpy() for key, value in captures.items()}


def write_report(path: Path, result: Dict[str, object]):
    all_metrics = result["candidate_oracle"]["all"]
    bones_metrics = result["candidate_oracle"]["bones"]
    lines = [
        "# AMS Evidence Reliability Diagnostics",
        "",
        f"- config: `{result['config']}`",
        f"- checkpoint: `{result['checkpoint']}`",
        f"- samples: {result['num_samples']}",
        f"- block index: {result['block_index']}",
        f"- candidates per joint: {result['num_candidates']}",
        "",
        "## Candidate Gap",
        "",
        f"- AMS weighted all acc: {all_metrics['ams_weighted_ordinal_acc_delta']:.4f}",
        f"- uniform all acc: {all_metrics['uniform_ordinal_acc_delta']:.4f}",
        f"- oracle all acc: {all_metrics['oracle_pair_ordinal_acc_delta']:.4f}",
        f"- top-weight best-candidate rate: {all_metrics['top_weight_is_best_candidate_rate']:.4f}",
        f"- weight/error corr: {all_metrics['weight_error_corr']:.4f}",
        "",
        "## Bone Pairs",
        "",
        f"- AMS weighted bones acc: {bones_metrics['ams_weighted_ordinal_acc_delta']:.4f}",
        f"- oracle bones acc: {bones_metrics['oracle_pair_ordinal_acc_delta']:.4f}",
    ]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    loader, indices = build_val_loader(args)
    model = load_model(args, device)
    captured = capture_ams_evidence(args, model, loader, device)

    gt_z = captured["gt"][..., 2]
    num_frames = gt_z.shape[0]
    split = max(1, int(0.7 * num_frames))
    train_idx = np.arange(split)
    test_idx = np.arange(split, num_frames)
    if len(test_idx) < 2:
        raise RuntimeError("Need more samples for a train/test split.")

    pair_sets = {
        "all": pair_indices(gt_z.shape[1], "all"),
        "bones": pair_indices(gt_z.shape[1], "bones"),
        "ambiguous": pair_indices(gt_z.shape[1], "ambiguous"),
    }

    aggregate_probes = {}
    for name, pairs in pair_sets.items():
        aggregate_probes[name] = {
            "ams_weighted": pairwise_metrics(captured["weighted"], gt_z, train_idx, test_idx, pairs, args.ridge, args.delta),
            "uniform": pairwise_metrics(captured["uniform"], gt_z, train_idx, test_idx, pairs, args.ridge, args.delta),
        }

    candidate_scores = score_candidates(captured["candidate"], gt_z, train_idx, test_idx, args.ridge)
    weights_test = captured["weights"][test_idx]
    candidate_oracle = {
        name: candidate_oracle_metrics(candidate_scores, weights_test, gt_z, test_idx, pairs, args.delta)
        for name, pairs in pair_sets.items()
    }

    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "num_samples": int(num_frames),
        "selected_indices_start": int(indices[0]) if len(indices) else None,
        "selected_indices_end": int(indices[-1]) if len(indices) else None,
        "block_index": args.block_index,
        "num_candidates": int(captured["candidate"].shape[2]),
        "candidate_dim": int(captured["candidate"].shape[3]),
        "probe_split": {"train_frames": int(len(train_idx)), "test_frames": int(len(test_idx))},
        "aggregate_probes": aggregate_probes,
        "candidate_oracle": candidate_oracle,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(output, result)
    print(f"Wrote {output}")
    print(f"Wrote {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
