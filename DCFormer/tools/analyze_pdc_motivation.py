#!/usr/bin/env python3
"""
PDC motivation diagnostics.

This script tests whether the trajectory of depth tokens across stacked AMS
blocks carries a useful reliability signal. It does not train PDC.

Core question:
    Does larger stage disagreement across progressive AMS refinement correlate
    with larger joint pose/depth errors or DLST depth-order mistakes?
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mvn import datasets
from mvn.datasets import utils as dataset_utils
from mvn.models.DGPose_dlst import DepthGuidedPoseDLST
from mvn.utils.cfg import config, update_config, update_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze whether AMS stage disagreement predicts depth-token reliability.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="pdc_motivation/summary.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", default="hrnet_32", choices=["hrnet_32", "hrnet_48"])
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--random-subset", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--margin", type=float, default=0.05)
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


def summarize_corr(signal: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    signal = np.asarray(signal, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    flat_signal = signal.reshape(-1)
    flat_target = target.reshape(-1)
    q25, q75 = np.nanpercentile(flat_signal, [25, 75])
    low = flat_signal <= q25
    high = flat_signal >= q75
    return {
        "pearson": safe_corr(flat_signal, flat_target),
        "spearman": safe_spearman(flat_signal, flat_target),
        "low_signal_target_mean": float(np.nanmean(flat_target[low])),
        "high_signal_target_mean": float(np.nanmean(flat_target[high])),
        "high_minus_low": float(np.nanmean(flat_target[high]) - np.nanmean(flat_target[low])),
    }


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


def capture_stage_track(args: argparse.Namespace, model, loader, device: torch.device):
    stage_chunks = []
    pred_chunks = []
    gt_chunks = []
    rel_chunks = []

    active_track = []
    original_lifting_forward = model.Lifting_net.forward

    def wrapped_lifting_forward(keypoints_2d, ref, depth_images, features_list_hr):
        lifting = model.Lifting_net
        b, _p, _ = keypoints_2d.shape

        x_pose = lifting.coord_embed(keypoints_2d)
        depth_features = lifting.depth_embed(depth_images.unsqueeze(1))
        features_list_hr = list(features_list_hr) + [depth_features]

        features_ref_list = [
            torch.nn.functional.grid_sample(features, ref.unsqueeze(-2), align_corners=True)
            .squeeze(-1)
            .permute(0, 2, 1)
            .contiguous()
            for features in features_list_hr
        ]
        features_ref_list = [embed(features_ref_list[idx]) for idx, embed in enumerate(lifting.feat_embed)]

        x = torch.stack([x_pose, *features_ref_list], dim=1)
        x = lifting.pos_drop(x + lifting.Spatial_pos_embed)
        active_track.append(x[:, -1].detach().float().cpu())

        for block in lifting.RGBD_Extraction:
            x = block(x, ref, features_list_hr)
            active_track.append(x[:, -1].detach().float().cpu())

        depth_joint_tokens = x[:, -1]
        depth_joint_tokens, rel_depth, layer_assign = lifting.dlst(depth_joint_tokens)
        x = torch.cat((x[:, :-1], depth_joint_tokens.unsqueeze(1)), dim=1)

        from einops import rearrange

        x = rearrange(x, "b l p c -> (b p) l c")
        for block in lifting.Features_Fusion:
            x = block(x)
        x = rearrange(x, "(b p) l c -> b p (l c)", b=b)

        for block in lifting.Spatial_Transformer:
            x = block(x)
        x = lifting.head(x).view(b, 1, _p, -1)
        return x, rel_depth, layer_assign

    model.Lifting_net.forward = wrapped_lifting_forward

    prefetcher = dataset_utils.data_prefetcher(loader, device, is_train=False, flip_test=False)
    batch = prefetcher.next()
    seen = 0
    with torch.no_grad():
        while batch is not None and seen < args.num_samples:
            images, gt_3d, kp2d, kp2d_crop, depth_images = batch
            active_track.clear()
            pred, rel_depth, _layer_assign = model(images, kp2d, kp2d_crop.clone(), depth_images)
            if gt_3d.dim() == 4:
                gt_3d = gt_3d.squeeze(1)
            if pred.dim() == 4:
                pred = pred.squeeze(1)
            if len(active_track) == 0:
                raise RuntimeError("No AMS stages were captured.")
            stage_chunks.append(torch.stack(active_track, dim=1))
            pred_chunks.append(pred.detach().float().cpu())
            gt_chunks.append(gt_3d.detach().float().cpu())
            rel_chunks.append(rel_depth.detach().float().cpu())
            seen += gt_3d.shape[0]
            print(f"[capture] {min(seen, args.num_samples)}/{args.num_samples}", flush=True)
            batch = prefetcher.next()

    model.Lifting_net.forward = original_lifting_forward

    return {
        "stage_track": torch.cat(stage_chunks, dim=0)[: args.num_samples].numpy(),
        "pred": torch.cat(pred_chunks, dim=0)[: args.num_samples].numpy(),
        "gt": torch.cat(gt_chunks, dim=0)[: args.num_samples].numpy(),
        "rel_depth": torch.cat(rel_chunks, dim=0)[: args.num_samples].numpy(),
    }


def valid_margin(z: np.ndarray, margin: float) -> float:
    if np.nanmedian(np.abs(z)) > 10.0 and margin < 1.0:
        return margin * 1000.0
    return margin


def order_error_per_joint(rel_depth: np.ndarray, gt_z: np.ndarray, margin: float) -> np.ndarray:
    diff = gt_z[:, None, :] - gt_z[:, :, None]
    target = np.sign(diff)
    pred = np.sign(rel_depth)
    eye = np.eye(gt_z.shape[1], dtype=bool)[None]
    m = valid_margin(gt_z, margin)
    mask = (np.abs(diff) > m) & (~eye)
    wrong = (pred != target) & mask
    denom = mask.sum(axis=-1)
    numer = wrong.sum(axis=-1)
    return numer / np.maximum(denom, 1)


def compute_metrics(captured: Dict[str, np.ndarray], margin: float):
    track = captured["stage_track"]
    pred = captured["pred"]
    gt = captured["gt"]
    rel_depth = captured["rel_depth"]

    pred_rel = pred - pred[:, :1]
    gt_rel = gt - gt[:, :1]
    joint_mpjpe = np.linalg.norm(pred_rel - gt_rel, axis=-1) * 1000.0
    joint_depth_error = np.abs(pred_rel[..., 2] - gt_rel[..., 2]) * 1000.0
    order_error = order_error_per_joint(rel_depth, gt_rel[..., 2], margin)

    mean_token = track.mean(axis=1, keepdims=True)
    stage_disagreement = np.linalg.norm(track - mean_token, axis=-1).mean(axis=1)
    first_last_shift = np.linalg.norm(track[:, -1] - track[:, 0], axis=-1)
    late_shift = np.linalg.norm(track[:, -1] - track[:, -2], axis=-1) if track.shape[1] > 1 else np.zeros_like(stage_disagreement)
    final_to_consensus = np.linalg.norm(track[:, -1] - mean_token[:, 0], axis=-1)

    stage_step = np.linalg.norm(track[:, 1:] - track[:, :-1], axis=-1) if track.shape[1] > 1 else np.zeros((*track.shape[:2], 0))
    step_mean = stage_step.mean(axis=(0, 2)).tolist() if stage_step.size else []
    stage_norm_mean = np.linalg.norm(track, axis=-1).mean(axis=(0, 2)).tolist()

    signals = {
        "stage_disagreement": stage_disagreement,
        "first_last_shift": first_last_shift,
        "late_shift": late_shift,
        "final_to_consensus": final_to_consensus,
    }
    targets = {
        "joint_mpjpe_mm": joint_mpjpe,
        "joint_depth_error_mm": joint_depth_error,
        "dlst_order_error_rate": order_error,
    }

    correlations = {
        signal_name: {
            target_name: summarize_corr(signal_value, target_value)
            for target_name, target_value in targets.items()
        }
        for signal_name, signal_value in signals.items()
    }

    return {
        "num_stages": int(track.shape[1]),
        "stage_step_mean": [float(x) for x in step_mean],
        "stage_norm_mean": [float(x) for x in stage_norm_mean],
        "signal_means": {name: float(value.mean()) for name, value in signals.items()},
        "target_means": {name: float(value.mean()) for name, value in targets.items()},
        "correlations": correlations,
    }


def write_report(path: Path, result: Dict[str, object]):
    corr = result["metrics"]["correlations"]
    sd = corr["stage_disagreement"]
    fl = corr["first_last_shift"]
    lines = [
        "# PDC Motivation Diagnostics",
        "",
        f"- config: `{result['config']}`",
        f"- checkpoint: `{result['checkpoint']}`",
        f"- samples: {result['num_samples']}",
        f"- stages: {result['metrics']['num_stages']}",
        "",
        "## Stage Disagreement",
        "",
        f"- vs joint MPJPE spearman: {sd['joint_mpjpe_mm']['spearman']:.4f}",
        f"- vs joint depth error spearman: {sd['joint_depth_error_mm']['spearman']:.4f}",
        f"- vs DLST order error spearman: {sd['dlst_order_error_rate']['spearman']:.4f}",
        f"- high-low joint MPJPE: {sd['joint_mpjpe_mm']['high_minus_low']:+.4f} mm",
        "",
        "## First-Last Shift",
        "",
        f"- vs joint MPJPE spearman: {fl['joint_mpjpe_mm']['spearman']:.4f}",
        f"- high-low joint MPJPE: {fl['joint_mpjpe_mm']['high_minus_low']:+.4f} mm",
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
    captured = capture_stage_track(args, model, loader, device)
    metrics = compute_metrics(captured, args.margin)

    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "num_samples": int(captured["gt"].shape[0]),
        "selected_indices_start": int(indices[0]) if len(indices) else None,
        "selected_indices_end": int(indices[-1]) if len(indices) else None,
        "metrics": metrics,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(output, result)
    print(f"Wrote {output}")
    print(f"Wrote {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
