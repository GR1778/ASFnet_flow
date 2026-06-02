#!/usr/bin/env python3
"""Ablate and replace the flow token to test whether it can move P1.

This diagnostic is intentionally about the downstream effect on 3D MPJPE, not
about raw flow error. It reuses a trained RGBFlowPoseSingle checkpoint and
intervenes at the single flow-token slot before multimodal fusion.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from mvn import datasets  # noqa: E402
from mvn.datasets import utils as dataset_utils  # noqa: E402
from mvn.models.DGPose_rgbflow_capf import RGBFlowPoseCAPF  # noqa: E402
from mvn.models.DGPose_rgbflow_single import RGBFlowPoseSingle  # noqa: E402
from mvn.models.loss import MPJPE, P_MPJPE  # noqa: E402
from mvn.utils.cfg import config, update_config  # noqa: E402
from diagnose_flow_mfce_sampling import (  # noqa: E402
    affine_apply,
    get_affine_transform,
    load_labels,
)


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (*self.dataset[idx], idx)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means full validation set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modes",
        default="normal,zero_token,shuffle_batch,shuffle_joint,oracle_ring,gt_motion",
        help="Comma-separated modes to evaluate.",
    )
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--radii", default="2,4,6,8,12,16")
    parser.add_argument("--num-directions", type=int, default=16)
    parser.add_argument("--out", default="")
    parser.add_argument("--skip-eval", action="store_true", help="Only test forward passes; useful for smoke runs.")
    return parser.parse_args()


def parse_float_list(text):
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def make_ring_offsets(radii, num_directions):
    offsets = [torch.zeros(1, 2, dtype=torch.float32)]
    theta = torch.arange(num_directions, dtype=torch.float32) * (2.0 * math.pi / float(num_directions))
    unit = torch.stack([theta.cos(), theta.sin()], dim=-1)
    for radius in radii:
        if radius > 0:
            offsets.append(unit * float(radius))
    return torch.cat(offsets, dim=0)


def build_val_dataset():
    return datasets.human36m(
        root=config.dataset.root,
        pred_results_path=config.val.pred_results_path,
        depth_image_path=config.dataset.flow_image_path,
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
        depth_format=config.dataset.flow_format,
        frame=1,
    )


def load_model(checkpoint, device):
    model_map = {
        "RGBFlowPoseSingle": RGBFlowPoseSingle,
        "RGBFlowPoseCAPF": RGBFlowPoseCAPF,
    }
    if config.model.name not in model_map:
        raise ValueError("Only RGBFlowPoseSingle/RGBFlowPoseCAPF are supported, got {}".format(config.model.name))
    model = model_map[config.model.name](config, device=str(device))
    raw = torch.load(checkpoint, map_location="cpu")
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    state = {key.replace("module.", ""): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("Missing keys:", missing[:10], "... total", len(missing))
    if unexpected:
        print("Unexpected keys:", unexpected[:10], "... total", len(unexpected))
    model.to(device)
    model.eval()
    return model


def get_flow_scale():
    flow_clip = getattr(config.dataset, "flow_clip", 20.0)
    flow_norm = getattr(config.dataset, "flow_norm", None)
    flow_norm = flow_clip if flow_norm is None else flow_norm
    return flow_clip, flow_norm


def normalize_ref(keypoints_2d_crop, device):
    image_shape = getattr(config.model, "image_shape", [192, 256])
    width, height = int(image_shape[0]), int(image_shape[1])
    ref = keypoints_2d_crop.clone()
    ref[..., :2] /= torch.tensor([width // 2, height // 2], device=device)
    ref[..., :2] -= torch.tensor([1.0, 1.0], device=device)
    return ref


def build_label_map(labels_path):
    labels = load_labels(labels_path)
    mapping = {}
    for shot in labels:
        key = (
            int(shot["subject"]),
            int(shot["action"]),
            int(shot["subaction"]),
            int(shot["camera_id"]),
            int(shot["image_id"]),
        )
        mapping[key] = shot
    return mapping


def target_motion_px(dataset, label_map, indices, frame_gap, output_size):
    motions = []
    valid = []
    for item in indices.detach().cpu().numpy().tolist():
        cur = dataset.labels[int(item)]
        key = (
            int(cur["subject"]),
            int(cur["action"]),
            int(cur["subaction"]),
            int(cur["camera_id"]),
            int(cur["image_id"]) - int(frame_gap),
        )
        prev = label_map.get(key)
        if prev is None:
            motions.append(np.zeros((17, 2), dtype=np.float32))
            valid.append(False)
            continue

        cur_gt = np.asarray(cur["joints_2d_gt_crop"], dtype=np.float32)
        prev_gt = np.asarray(prev["joints_2d_gt_crop"], dtype=np.float32)
        prev_to_raw = get_affine_transform(prev["center"], prev["scale"], output_size, inv=True)
        raw_to_cur = get_affine_transform(cur["center"], cur["scale"], output_size, inv=False)
        prev_gt_same = affine_apply(raw_to_cur, affine_apply(prev_to_raw, prev_gt))
        motions.append((prev_gt_same - cur_gt).astype(np.float32))
        valid.append(True)
    return np.stack(motions, axis=0), np.asarray(valid, dtype=np.bool_)


def preprocess_batch(batch, device):
    images, keypoints_3d, keypoints_2d, keypoints_2d_crop, flow_images, indices = batch
    images = images.to(device, non_blocking=True).float()
    keypoints_3d = keypoints_3d.to(device, non_blocking=True).float()
    keypoints_2d = keypoints_2d.to(device, non_blocking=True).float()
    keypoints_2d_crop = keypoints_2d_crop.to(device, non_blocking=True).float()
    flow_images = flow_images.to(device, non_blocking=True).float()
    indices = indices.to(device, non_blocking=True).long()

    flow_clip, flow_norm = get_flow_scale()
    if flow_clip is not None and flow_clip > 0:
        flow_images = flow_images.clamp(-flow_clip, flow_clip)
    if flow_norm is not None and flow_norm > 0:
        flow_images = flow_images / flow_norm

    images = torch.flip(images, [-1])
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)
    images = (images / 255.0 - mean) / std

    keypoints_3d[:, :, 1:] -= keypoints_3d[:, :, :1]
    keypoints_3d[:, :, 0] = 0
    return images, keypoints_3d, keypoints_2d, keypoints_2d_crop, flow_images, indices


def constant_flow_embed(conv, flow_vectors):
    weight = conv.weight.sum(dim=(2, 3))
    token = torch.einsum("bjc,oc->bjo", flow_vectors, weight)
    if conv.bias is not None:
        token = token + conv.bias.view(1, 1, -1)
    return token


def sample_map(feature_map, grid):
    b, _, h, w = feature_map.shape
    samples = F.grid_sample(feature_map, grid, padding_mode="border", align_corners=True)
    return samples.permute(0, 2, 3, 1).contiguous()


def single_forward_with_mode(
    model,
    images,
    keypoints_2d,
    keypoints_2d_crop,
    flow_images,
    mode,
    target_motion_norm,
    valid_target,
    offsets_px,
):
    device = images.device
    lifting = model.Lifting_net
    b, p, _ = keypoints_2d.shape

    ref = normalize_ref(keypoints_2d_crop, device)

    features_list = model.backbone(images.permute(0, 3, 1, 2).contiguous())
    x_pose = lifting.coord_embed(keypoints_2d)
    flow_features = lifting.flow_embed(flow_images.permute(0, 3, 1, 2).contiguous())
    features_list = list(features_list) + [flow_features]

    features_ref_list = [
        F.grid_sample(features, ref.unsqueeze(-2), align_corners=True).squeeze(-1).permute(0, 2, 1).contiguous()
        for features in features_list
    ]
    features_ref_list = [embed(features_ref_list[idx]) for idx, embed in enumerate(lifting.feat_embed)]

    normal_flow_token = features_ref_list[-1]
    flow_token = normal_flow_token

    if mode == "zero_token":
        flow_token = torch.zeros_like(normal_flow_token)
    elif mode == "shuffle_batch":
        if b > 1:
            flow_token = normal_flow_token[torch.randperm(b, device=device)]
    elif mode == "shuffle_joint":
        flow_token = normal_flow_token[:, torch.randperm(p, device=device)]
    elif mode == "gt_motion":
        gt_embed = constant_flow_embed(lifting.flow_embed, target_motion_norm)
        gt_token = lifting.feat_embed[-1](gt_embed)
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
        best_feat = feat_samples.gather(dim=2, index=gather_idx).squeeze(2)
        oracle_token = lifting.feat_embed[-1](best_feat)
        flow_token = torch.where(valid_target.view(b, 1, 1), oracle_token, normal_flow_token)
    elif mode != "normal":
        raise ValueError("Unknown mode: {}".format(mode))

    features_ref_list[-1] = flow_token
    x = torch.stack([x_pose, *features_ref_list], dim=1)
    x = x + lifting.Spatial_pos_embed
    x = lifting.pos_drop(x)

    x = rearrange(x, "b l p c -> (b p) l c")
    for block in lifting.res_blocks:
        x = block(x)
    x = rearrange(x, "(b p) l c -> b p (l c)", b=b, p=p)

    for block in lifting.joint_blocks:
        x = block(x)

    return lifting.head(x).view(b, 1, p, -1)


def capf_forward_with_mode(
    model,
    images,
    keypoints_2d,
    keypoints_2d_crop,
    flow_images,
    mode,
    target_motion_norm,
    valid_target,
    offsets_px,
):
    device = images.device
    lifting = model.Lifting_net
    b, p, _ = keypoints_2d.shape

    ref = normalize_ref(keypoints_2d_crop, device)
    features_list_hr = model.backbone(images.permute(0, 3, 1, 2).contiguous())

    x = lifting.coord_embed(keypoints_2d)
    features_ref_list = [
        F.grid_sample(features, ref.unsqueeze(-2), align_corners=True).squeeze(-1).permute(0, 2, 1).contiguous()
        for features in features_list_hr
    ]
    features_ref_list = [embed(features_ref_list[idx]) for idx, embed in enumerate(lifting.feat_embed)]

    x = torch.stack([x, *features_ref_list], dim=1)
    x = x + lifting.RGB_pos_embed
    x = lifting.pos_drop(x)

    for block in lifting.RGB_Extraction:
        x = block(x, ref, features_list_hr)

    flow_features = lifting.flow_embed(flow_images.permute(0, 3, 1, 2).contiguous())
    normal_flow_embed = F.grid_sample(flow_features, ref.unsqueeze(-2), align_corners=True).squeeze(-1).permute(0, 2, 1).contiguous()
    normal_flow_token = lifting.flow_feat_embed(normal_flow_embed)
    flow_token = normal_flow_token

    if mode == "zero_token":
        flow_token = torch.zeros_like(normal_flow_token)
    elif mode == "shuffle_batch":
        if b > 1:
            flow_token = normal_flow_token[torch.randperm(b, device=device)]
    elif mode == "shuffle_joint":
        flow_token = normal_flow_token[:, torch.randperm(p, device=device)]
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
    elif mode != "normal":
        raise ValueError("Unknown mode: {}".format(mode))

    flow_token = lifting.pos_drop(flow_token.unsqueeze(1) + lifting.Flow_pos_embed)
    x = torch.cat([x, flow_token], dim=1)

    x = rearrange(x, "b l p c -> (b p) l c")
    for block in lifting.Features_Fusion:
        x = block(x)

    x = rearrange(x, "(b p) l c -> b p (l c)", b=b)
    for block in lifting.Spatial_Transformer:
        x = block(x)

    return lifting.head(x).view(b, 1, p, -1)


def forward_with_mode(*args, **kwargs):
    model = args[0]
    if config.model.name == "RGBFlowPoseSingle":
        return single_forward_with_mode(*args, **kwargs)
    if config.model.name == "RGBFlowPoseCAPF":
        return capf_forward_with_mode(*args, **kwargs)
    raise ValueError("Unsupported model: {}".format(config.model.name))


def average_action_scores(result):
    p1 = [value["MPJPE"] * 1000.0 for value in result.values()]
    p2 = [value["P_MPJPE"] * 1000.0 for value in result.values()]
    return round(float(np.mean(p1)), 3), round(float(np.mean(p2)), 3)


def global_scores(gt, pred):
    p1 = MPJPE()(pred, gt).item() * 1000.0
    p2 = P_MPJPE()(pred.squeeze(1).numpy(), gt.squeeze(1).numpy()) * 1000.0
    return round(float(p1), 3), round(float(p2), 3)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    update_config(args.config)
    config.val.flip_test = False
    config.val.batch_size = args.batch_size
    config.val.num_workers = args.num_workers

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    base_dataset = build_val_dataset()
    label_map = build_label_map(config.dataset.val_labels_path)
    loader = DataLoader(
        IndexedDataset(base_dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=dataset_utils.worker_init_fn,
    )
    model = load_model(args.checkpoint, device)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    offsets_px = make_ring_offsets(parse_float_list(args.radii), args.num_directions)

    flow_clip, flow_norm = get_flow_scale()
    output_size = (args.width, args.height)

    all_pred = {mode: [] for mode in modes}
    all_gt = []
    valid_count = 0
    total_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            images, keypoints_3d, keypoints_2d, keypoints_2d_crop, flow_images, indices = preprocess_batch(batch, device)

            target_px_np, valid_np = target_motion_px(base_dataset, label_map, indices, args.frame_gap, output_size)
            target_px = torch.from_numpy(target_px_np).to(device=device, dtype=flow_images.dtype)
            valid = torch.from_numpy(valid_np).to(device=device)
            if flow_clip is not None and flow_clip > 0:
                target_norm = target_px.clamp(-flow_clip, flow_clip)
            else:
                target_norm = target_px
            if flow_norm is not None and flow_norm > 0:
                target_norm = target_norm / float(flow_norm)

            total_count += int(valid.numel())
            valid_count += int(valid.sum().item())
            all_gt.append(keypoints_3d.detach().cpu())

            for mode in modes:
                pred = forward_with_mode(
                    model,
                    images,
                    keypoints_2d,
                    keypoints_2d_crop,
                    flow_images,
                    mode,
                    target_norm,
                    valid,
                    offsets_px,
                )
                all_pred[mode].append(pred.detach().cpu())

            print("processed batch {}/{}".format(batch_idx + 1, "full" if not args.max_batches else args.max_batches), flush=True)

    gt = torch.cat(all_gt, dim=0)
    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "samples": int(gt.shape[0]),
        "target_valid_samples": int(valid_count),
        "target_total_samples": int(total_count),
        "metric_note": "global_subset_P1_P2" if args.max_batches else "full_action_average_P1_P2",
        "modes": {},
    }
    if args.skip_eval:
        for mode in modes:
            summary["modes"][mode] = {"pred_shape": list(torch.cat(all_pred[mode], dim=0).shape)}
        print(json.dumps(summary, indent=2))
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return

    for mode in modes:
        pred = torch.cat(all_pred[mode], dim=0)
        if args.max_batches:
            p1, p2 = global_scores(gt, pred)
            summary["modes"][mode] = {"P1": p1, "P2": p2}
        else:
            result = base_dataset.evaluate(gt, pred, None, config)
            p1, p2 = average_action_scores(result)
            summary["modes"][mode] = {
                "P1": p1,
                "P2": p2,
                "actions": {
                    name: {
                        "P1": float(value["MPJPE"] * 1000.0),
                        "P2": float(value["P_MPJPE"] * 1000.0),
                    }
                    for name, value in result.items()
                },
            }

    print(json.dumps(summary, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
