#!/usr/bin/env python3
"""Diagnose where the direct optical-flow P1 gain comes from.

This script does not train or edit a model. It intervenes at the existing
RGBFlowPoseCAPF flow-token slot and asks which part of the learned direct-flow
token is actually useful for P1:

- zero/shuffle modes estimate whether the trained model uses the flow token;
- sign/x/y/magnitude/direction modes probe which raw-flow factor matters;
- oracle_ring/gt_motion reuse the local-sampling upper-bound probes.

The output is intentionally bucketed by joint, action, target-motion magnitude,
and center-flow error. That is the evidence needed before designing another
sampling module.
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
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, Subset

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


DEFAULT_MODES = (
    "normal,zero_token,shuffle_batch,shuffle_joint,sign_flip,"
    "x_only,y_only,magnitude_only,direction_only,oracle_ring,gt_motion"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Counterfactual source analysis for direct optical-flow P1 gain."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=512, help="0 means full validation set.")
    parser.add_argument(
        "--subset",
        default="random",
        choices=["random", "sequential"],
        help="Use a random validation subset when max-batches is nonzero to avoid action-order bias.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", default=DEFAULT_MODES)
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--radii", default="2,4,6,8,12,16")
    parser.add_argument("--num-directions", type=int, default=16)
    parser.add_argument("--out", default="debug_vis/direct_flow_gain_source_clip5_512b.json")
    return parser.parse_args()


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
    if int(mask.sum()) < 3:
        return None
    x = x[mask]
    y = y[mask]
    if x.std() < 1.0e-12 or y.std() < 1.0e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def bucket_value(value, edges, unit):
    value = float(value)
    if not np.isfinite(value):
        return "invalid"
    lower = "-inf"
    for edge in edges:
        if value < edge:
            return "{}_to_{}{}".format(lower, edge, unit)
        lower = str(edge)
    return "{}_plus{}".format(lower, unit)


def action_name(label):
    idx = (int(label["action"]) - 2) * 2 + (int(label["subaction"]) - 1)
    if 0 <= idx < len(ACTION_NAMES):
        return ACTION_NAMES[idx]
    return "action_{}_{}".format(label["action"], label["subaction"])


def batch_action_names(dataset, indices):
    names = []
    for idx in indices.detach().cpu().numpy().tolist():
        names.append(action_name(dataset.labels[int(idx)]))
    return np.asarray(names, dtype=object)


def embed_center_flow(lifting, flow_map_bhwc, ref):
    flow_features = lifting.flow_embed(flow_map_bhwc.permute(0, 3, 1, 2).contiguous())
    flow_embed = (
        F.grid_sample(flow_features, ref.unsqueeze(-2), align_corners=True)
        .squeeze(-1)
        .permute(0, 2, 1)
        .contiguous()
    )
    return lifting.flow_feat_embed(flow_embed)


def constant_flow_embed(conv, flow_vectors):
    weight = conv.weight.sum(dim=(2, 3))
    token = torch.einsum("bjc,oc->bjo", flow_vectors, weight)
    if conv.bias is not None:
        token = token + conv.bias.view(1, 1, -1)
    return token


def modified_flow_map(flow_images, mode):
    if mode == "sign_flip":
        return -flow_images
    if mode == "x_only":
        zeros = torch.zeros_like(flow_images[..., 1:2])
        return torch.cat([flow_images[..., 0:1], zeros], dim=-1)
    if mode == "y_only":
        zeros = torch.zeros_like(flow_images[..., 0:1])
        return torch.cat([zeros, flow_images[..., 1:2]], dim=-1)
    if mode == "magnitude_only":
        mag = torch.linalg.norm(flow_images, dim=-1, keepdim=True)
        zeros = torch.zeros_like(mag)
        return torch.cat([mag, zeros], dim=-1)
    if mode == "direction_only":
        mag = torch.linalg.norm(flow_images, dim=-1, keepdim=True)
        unit = flow_images / mag.clamp_min(1.0e-6)
        mean_mag = mag.mean(dim=(1, 2, 3), keepdim=True)
        return unit * mean_mag
    raise ValueError("Unknown modified flow mode: {}".format(mode))


def capf_forward_modes(
    model,
    images,
    keypoints_2d,
    keypoints_2d_crop,
    flow_images,
    modes,
    target_motion_norm,
    valid_target,
    offsets_px,
):
    if config.model.name != "RGBFlowPoseCAPF":
        raise ValueError("This diagnostic currently targets RGBFlowPoseCAPF, got {}".format(config.model.name))

    device = images.device
    lifting = model.Lifting_net
    b, p, _ = keypoints_2d.shape
    ref = normalize_ref(keypoints_2d_crop, device)

    features_list_hr = model.backbone(images.permute(0, 3, 1, 2).contiguous())
    pose_token = lifting.coord_embed(keypoints_2d)
    features_ref_list = [
        F.grid_sample(features, ref.unsqueeze(-2), align_corners=True)
        .squeeze(-1)
        .permute(0, 2, 1)
        .contiguous()
        for features in features_list_hr
    ]
    features_ref_list = [embed(features_ref_list[idx]) for idx, embed in enumerate(lifting.feat_embed)]

    rgb_tokens = torch.stack([pose_token, *features_ref_list], dim=1)
    rgb_tokens = lifting.pos_drop(rgb_tokens + lifting.RGB_pos_embed)
    for block in lifting.RGB_Extraction:
        rgb_tokens = block(rgb_tokens, ref, features_list_hr)

    flow_features = lifting.flow_embed(flow_images.permute(0, 3, 1, 2).contiguous())
    normal_flow_embed = (
        F.grid_sample(flow_features, ref.unsqueeze(-2), align_corners=True)
        .squeeze(-1)
        .permute(0, 2, 1)
        .contiguous()
    )
    normal_flow_token = lifting.flow_feat_embed(normal_flow_embed)

    token_cache = {"normal": normal_flow_token}
    outputs = {}

    for mode in modes:
        if mode in token_cache:
            flow_token = token_cache[mode]
        elif mode == "zero_token":
            flow_token = torch.zeros_like(normal_flow_token)
        elif mode == "shuffle_batch":
            if b > 1:
                flow_token = normal_flow_token[torch.randperm(b, device=device)]
            else:
                flow_token = normal_flow_token
        elif mode == "shuffle_joint":
            flow_token = normal_flow_token[:, torch.randperm(p, device=device)]
        elif mode in {"sign_flip", "x_only", "y_only", "magnitude_only", "direction_only"}:
            flow_token = embed_center_flow(lifting, modified_flow_map(flow_images, mode), ref)
        elif mode == "gt_motion":
            gt_embed = constant_flow_embed(lifting.flow_embed, target_motion_norm)
            gt_token = lifting.flow_feat_embed(gt_embed)
            flow_token = torch.where(valid_target.view(b, 1, 1), gt_token, normal_flow_token)
        elif mode == "oracle_ring":
            _, _, h, w = flow_features.shape
            scale = torch.tensor([2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1)], device=device, dtype=ref.dtype)
            offsets = offsets_px.to(device=device, dtype=ref.dtype) * scale
            grid = ref.unsqueeze(2) + offsets.view(1, 1, -1, 2)

            raw_samples = sample_map(flow_images.permute(0, 3, 1, 2).contiguous(), grid)
            errors = torch.linalg.norm(raw_samples - target_motion_norm.unsqueeze(2), dim=-1)
            best_idx = errors.argmin(dim=-1)

            feat_samples = sample_map(flow_features, grid)
            gather_idx = best_idx.view(b, p, 1, 1).expand(-1, -1, 1, feat_samples.shape[-1])
            best_embed = feat_samples.gather(dim=2, index=gather_idx).squeeze(2)
            oracle_token = lifting.flow_feat_embed(best_embed)
            flow_token = torch.where(valid_target.view(b, 1, 1), oracle_token, normal_flow_token)
        else:
            raise ValueError("Unknown mode: {}".format(mode))

        flow_token = lifting.pos_drop(flow_token.unsqueeze(1) + lifting.Flow_pos_embed)
        x = torch.cat([rgb_tokens, flow_token], dim=1)

        x = rearrange(x, "b l p c -> (b p) l c")
        for block in lifting.Features_Fusion:
            x = block(x)

        x = rearrange(x, "(b p) l c -> b p (l c)", b=b)
        for block in lifting.Spatial_Transformer:
            x = block(x)

        outputs[mode] = lifting.head(x).view(b, 1, p, -1)

    return outputs


def center_flow_error_px(flow_images, keypoints_2d_crop, target_norm, valid, flow_norm):
    device = flow_images.device
    ref = normalize_ref(keypoints_2d_crop, device)
    center = (
        F.grid_sample(flow_images.permute(0, 3, 1, 2).contiguous(), ref.unsqueeze(-2), align_corners=True)
        .squeeze(-1)
        .permute(0, 2, 1)
        .contiguous()
    )
    error = torch.linalg.norm(center - target_norm, dim=-1)
    multiplier = float(flow_norm) if flow_norm is not None and flow_norm > 0 else 1.0
    error = error * multiplier
    error = torch.where(valid.view(-1, 1), error, torch.full_like(error, float("nan")))
    return error.detach().cpu().numpy()


def per_joint_error_mm(pred, gt):
    pred = pred.squeeze(1)
    gt = gt.squeeze(1)
    return torch.linalg.norm(pred - gt, dim=-1).detach().cpu().numpy() * 1000.0


def build_row_arrays(gt, preds, action_names, target_motion_mag, center_flow_error):
    gt_errors = {}
    for mode, pred in preds.items():
        gt_errors[mode] = per_joint_error_mm(pred, gt)
    return {
        "errors": gt_errors,
        "actions": action_names,
        "target_motion_mag": target_motion_mag,
        "center_flow_error": center_flow_error,
    }


def group_summary(mask, errors, normal_err, modes):
    out = {
        "count": int(mask.sum()),
        "normal_err_mm": summarize(normal_err[mask]),
        "modes": {},
    }
    for mode in modes:
        mode_err = errors[mode]
        delta = mode_err - normal_err
        out["modes"][mode] = {
            "err_mm": summarize(mode_err[mask]),
            "mode_minus_normal_mm": summarize(delta[mask]),
        }
    return out


def summarize_groups(group_keys, errors, modes):
    normal_err = errors["normal"]
    flat_keys = np.asarray(group_keys, dtype=object).reshape(-1)
    flat_errors = {mode: value.reshape(-1) for mode, value in errors.items()}
    flat_normal = normal_err.reshape(-1)
    summary = {}
    for key in sorted(set(flat_keys.tolist()), key=str):
        mask = flat_keys == key
        summary[str(key)] = group_summary(mask, flat_errors, flat_normal, modes)
    return summary


def summarize_overall(errors, modes):
    normal_err = errors["normal"].reshape(-1)
    flat_errors = {mode: value.reshape(-1) for mode, value in errors.items()}
    mask = np.isfinite(normal_err)
    return group_summary(mask, flat_errors, normal_err, modes)


def summarize_metrics(gt, preds, dataset, max_batches):
    modes = list(preds.keys())
    summary = {}
    for mode in modes:
        pred = preds[mode]
        if max_batches:
            p1, p2 = global_scores(gt, pred)
            summary[mode] = {"P1": p1, "P2": p2}
        else:
            result = dataset.evaluate(gt, pred, None, config)
            p1 = [value["MPJPE"] * 1000.0 for value in result.values()]
            p2 = [value["P_MPJPE"] * 1000.0 for value in result.values()]
            summary[mode] = {
                "P1": round(float(np.mean(p1)), 3),
                "P2": round(float(np.mean(p2)), 3),
                "actions": {
                    name: {
                        "P1": float(value["MPJPE"] * 1000.0),
                        "P2": float(value["P_MPJPE"] * 1000.0),
                    }
                    for name, value in result.items()
                },
            }
    normal_p1 = summary["normal"]["P1"]
    for mode in modes:
        summary[mode]["delta_P1_vs_normal"] = round(float(summary[mode]["P1"] - normal_p1), 3)
    return summary


def make_bucket_arrays(target_motion_mag, center_flow_error, normal_err):
    motion_edges = [0.5, 1, 2, 4, 8]
    flow_error_edges = [0.5, 1, 2, 4, 8]
    normal_error_edges = [25, 40, 60, 80, 120]

    motion_bucket = np.empty_like(target_motion_mag, dtype=object)
    center_error_bucket = np.empty_like(center_flow_error, dtype=object)
    normal_error_bucket = np.empty_like(normal_err, dtype=object)
    for idx in np.ndindex(target_motion_mag.shape):
        motion_bucket[idx] = bucket_value(target_motion_mag[idx], motion_edges, "px")
        center_error_bucket[idx] = bucket_value(center_flow_error[idx], flow_error_edges, "px")
        normal_error_bucket[idx] = bucket_value(normal_err[idx], normal_error_edges, "mm")
    return motion_bucket, center_error_bucket, normal_error_bucket


def build_verdict(metric_summary):
    normal = metric_summary["normal"]["P1"]
    verdict = {}
    for mode, values in metric_summary.items():
        if mode == "normal":
            continue
        delta = values["P1"] - normal
        if mode in {"zero_token", "shuffle_batch", "shuffle_joint", "sign_flip"}:
            meaning = "positive means the original flow token carries useful information"
        elif mode in {"oracle_ring", "gt_motion"}:
            meaning = "negative means this replacement improves the current flow channel"
        else:
            meaning = "compare with normal to infer which raw-flow factor is retained"
        verdict[mode] = {
            "delta_P1_vs_normal": round(float(delta), 3),
            "meaning": meaning,
        }
    return verdict


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    update_config(args.config)
    config.val.flip_test = False
    config.val.batch_size = args.batch_size
    config.val.num_workers = args.num_workers

    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    if "normal" not in modes:
        modes = ["normal"] + modes

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = build_val_dataset()
    label_map = build_label_map(config.dataset.val_labels_path)
    indexed_dataset = IndexedDataset(dataset)
    loader_dataset = indexed_dataset
    max_batches = args.max_batches
    if args.max_batches and args.subset == "random":
        max_samples = min(len(indexed_dataset), args.max_batches * args.batch_size)
        rng = np.random.default_rng(args.seed)
        subset_indices = rng.choice(len(indexed_dataset), size=max_samples, replace=False).tolist()
        loader_dataset = Subset(indexed_dataset, subset_indices)
        max_batches = 0

    loader = DataLoader(
        loader_dataset,
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

    all_gt = []
    all_pred = {mode: [] for mode in modes}
    all_actions = []
    all_target_motion_mag = []
    all_center_flow_error = []
    valid_count = 0
    total_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break

            images, gt_3d, keypoints_2d, keypoints_2d_crop, flow_images, indices = preprocess_batch(batch, device)
            target_px_np, valid_np = target_motion_px(dataset, label_map, indices, args.frame_gap, output_size)
            target_px = torch.from_numpy(target_px_np).to(device=device, dtype=flow_images.dtype)
            valid = torch.from_numpy(valid_np).to(device=device)
            if flow_clip is not None and flow_clip > 0:
                target_norm = target_px.clamp(-flow_clip, flow_clip)
            else:
                target_norm = target_px
            if flow_norm is not None and flow_norm > 0:
                target_norm = target_norm / float(flow_norm)

            preds = capf_forward_modes(
                model,
                images,
                keypoints_2d,
                keypoints_2d_crop,
                flow_images,
                modes,
                target_norm,
                valid,
                offsets_px,
            )

            all_gt.append(gt_3d.detach().cpu())
            for mode in modes:
                all_pred[mode].append(preds[mode].detach().cpu())
            all_actions.append(batch_action_names(dataset, indices))
            all_target_motion_mag.append(np.linalg.norm(target_px_np, axis=-1))
            all_center_flow_error.append(center_flow_error_px(flow_images, keypoints_2d_crop, target_norm, valid, flow_norm))
            valid_count += int(valid.sum().item())
            total_count += int(valid.numel())

            print(
                "processed batch {}/{}".format(batch_idx + 1, "full" if not args.max_batches else args.max_batches),
                flush=True,
            )

    gt = torch.cat(all_gt, dim=0)
    preds = {mode: torch.cat(chunks, dim=0) for mode, chunks in all_pred.items()}
    actions = np.concatenate(all_actions, axis=0)
    target_motion_mag = np.concatenate(all_target_motion_mag, axis=0)
    center_flow_error = np.concatenate(all_center_flow_error, axis=0)

    row_arrays = build_row_arrays(gt, preds, actions, target_motion_mag, center_flow_error)
    errors = row_arrays["errors"]
    joint_keys = np.tile(np.asarray(JOINT_NAMES, dtype=object).reshape(1, -1), (gt.shape[0], 1))
    action_keys = np.tile(actions.reshape(-1, 1), (1, len(JOINT_NAMES)))
    motion_bucket, center_error_bucket, normal_error_bucket = make_bucket_arrays(
        target_motion_mag, center_flow_error, errors["normal"]
    )

    metric_summary = summarize_metrics(gt, preds, dataset, args.max_batches)
    summary = {
        "meta": {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "samples": int(gt.shape[0]),
            "joint_rows": int(gt.shape[0] * len(JOINT_NAMES)),
            "target_valid_samples": int(valid_count),
            "target_total_samples": int(total_count),
            "flow_clip": None if flow_clip is None else float(flow_clip),
            "flow_norm": None if flow_norm is None else float(flow_norm),
            "modes": modes,
                "subset": args.subset if args.max_batches else "full",
                "metric_note": "global_subset_P1_P2" if args.max_batches else "full_action_average_P1_P2",
        },
        "metrics": metric_summary,
        "verdict": build_verdict(metric_summary),
        "overall_joint_rows": summarize_overall(errors, modes),
        "by_joint": summarize_groups(joint_keys, errors, modes),
        "by_action": summarize_groups(action_keys, errors, modes),
        "by_target_motion_bucket": summarize_groups(motion_bucket, errors, modes),
        "by_center_flow_error_bucket": summarize_groups(center_error_bucket, errors, modes),
        "by_normal_error_bucket": summarize_groups(normal_error_bucket, errors, modes),
        "correlations": {
            "target_motion_mag_vs_normal_err": safe_corr(target_motion_mag.reshape(-1), errors["normal"].reshape(-1)),
            "center_flow_error_vs_normal_err": safe_corr(center_flow_error.reshape(-1), errors["normal"].reshape(-1)),
        },
    }

    print(json.dumps(summary["metrics"], indent=2))
    print(json.dumps(summary["verdict"], indent=2))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("wrote {}".format(out_path))


if __name__ == "__main__":
    main()
