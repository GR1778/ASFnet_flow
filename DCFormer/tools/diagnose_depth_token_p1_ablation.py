#!/usr/bin/env python3
"""Ablate ASFNet depth tokens in a trained DepthGuidedPose checkpoint.

The test mirrors diagnose_flow_token_p1_upper.py, but for the official depth
branch. It can replace the depth token either after UDE calibration (directly
before feature fusion) or before UDE to see where depth information is used.
"""

import argparse
import json
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

from mvn import datasets  # noqa: E402
from mvn.datasets import utils as dataset_utils  # noqa: E402
from mvn.models.DGPose import DepthGuidedPose  # noqa: E402
from mvn.models.loss import MPJPE, P_MPJPE  # noqa: E402
from mvn.utils.cfg import config, update_config  # noqa: E402


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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=512, help="0 means full validation set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modes",
        default="normal,zero_token,shuffle_batch,shuffle_joint,zero_pre_ude,shuffle_batch_pre_ude,shuffle_joint_pre_ude",
        help="Comma-separated modes to evaluate.",
    )
    parser.add_argument("--out", default="")
    parser.add_argument("--skip-eval", action="store_true", help="Only test forward passes; useful for smoke runs.")
    return parser.parse_args()


def build_val_dataset():
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
        depth_format=getattr(config.dataset, "depth_format", "image"),
        frame=1,
    )


def load_model(checkpoint, device):
    if config.model.name != "DepthGuidedPose":
        raise ValueError("This diagnostic supports official DepthGuidedPose only, got {}".format(config.model.name))
    model = DepthGuidedPose(config, device=str(device))
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


def normalize_ref(keypoints_2d_crop, device):
    image_shape = getattr(config.model, "image_shape", [192, 256])
    width, height = int(image_shape[0]), int(image_shape[1])
    ref = keypoints_2d_crop.clone()
    ref[..., :2] /= torch.tensor([width // 2, height // 2], device=device)
    ref[..., :2] -= torch.tensor([1.0, 1.0], device=device)
    return ref


def preprocess_batch(batch, device):
    images, keypoints_3d, keypoints_2d, keypoints_2d_crop, depth_images, indices = batch
    images = images.to(device, non_blocking=True).float()
    keypoints_3d = keypoints_3d.to(device, non_blocking=True).float()
    keypoints_2d = keypoints_2d.to(device, non_blocking=True).float()
    keypoints_2d_crop = keypoints_2d_crop.to(device, non_blocking=True).float()
    depth_images = depth_images.to(device, non_blocking=True)
    indices = indices.to(device, non_blocking=True).long()

    if torch.is_floating_point(depth_images):
        depth_images = depth_images.float()
    else:
        depth_images = depth_images.float() / 255.0

    images = torch.flip(images, [-1])
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)
    images = (images / 255.0 - mean) / std

    keypoints_3d[:, :, 1:] -= keypoints_3d[:, :, :1]
    keypoints_3d[:, :, 0] = 0
    return images, keypoints_3d, keypoints_2d, keypoints_2d_crop, depth_images, indices


def replace_token(token, mode):
    b, p, _ = token.shape
    if mode.startswith("zero"):
        return torch.zeros_like(token)
    if mode.startswith("shuffle_batch"):
        if b > 1:
            return token[torch.randperm(b, device=token.device)]
        return token
    if mode.startswith("shuffle_joint"):
        return token[:, torch.randperm(p, device=token.device)]
    return token


def depth_forward_with_mode(model, images, keypoints_2d, keypoints_2d_crop, depth_images, mode):
    device = images.device
    lifting = model.Lifting_net
    b, p, _ = keypoints_2d.shape

    ref = normalize_ref(keypoints_2d_crop, device)
    features_list_hr = model.backbone(images.permute(0, 3, 1, 2).contiguous())

    x_pose = lifting.coord_embed(keypoints_2d)
    depth_features = lifting.depth_embed(depth_images.unsqueeze(1))
    features_list = list(features_list_hr) + [depth_features]

    features_ref_list = [
        F.grid_sample(features, ref.unsqueeze(-2), align_corners=True).squeeze(-1).permute(0, 2, 1).contiguous()
        for features in features_list
    ]
    features_ref_list = [embed(features_ref_list[idx]) for idx, embed in enumerate(lifting.feat_embed)]

    x = torch.stack([x_pose, *features_ref_list], dim=1)
    x = x + lifting.Spatial_pos_embed
    x = lifting.pos_drop(x)

    for block in lifting.RGBD_Extraction:
        x = block(x, ref, features_list)

    depth_before_ude = x[:, -1]
    if mode.endswith("_pre_ude"):
        depth_before_ude = replace_token(depth_before_ude, mode)

    coarse_depth, uncer = lifting.depth_uncer(depth_before_ude)
    z_value = lifting.z_embed(coarse_depth) + lifting.Spatial_pos_embed2
    joint_uncer = F.softmax(lifting.attn_fc(uncer), dim=1)
    depth_token = torch.cat([joint_uncer, z_value, depth_before_ude], dim=-1)
    depth_token = lifting.attn_depth(depth_token)

    if mode in {"zero_token", "shuffle_batch", "shuffle_joint"}:
        depth_token = replace_token(depth_token, mode)

    x = torch.cat((x[:, :-1], depth_token.unsqueeze(1)), dim=1)
    x = rearrange(x, "b l p c -> (b p) l c")
    for block in lifting.Features_Fusion:
        x = block(x)

    x = rearrange(x, "(b p) l c -> b p (l c)", b=b)
    for block in lifting.Spatial_Transformer:
        x = block(x)

    return lifting.head(x).view(b, 1, p, -1)


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

    all_pred = {mode: [] for mode in modes}
    all_gt = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            images, keypoints_3d, keypoints_2d, keypoints_2d_crop, depth_images, _ = preprocess_batch(batch, device)
            all_gt.append(keypoints_3d.detach().cpu())

            for mode in modes:
                pred = depth_forward_with_mode(
                    model,
                    images,
                    keypoints_2d,
                    keypoints_2d_crop,
                    depth_images,
                    mode,
                )
                all_pred[mode].append(pred.detach().cpu())

            print("processed batch {}/{}".format(batch_idx + 1, "full" if not args.max_batches else args.max_batches), flush=True)

    gt = torch.cat(all_gt, dim=0)
    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "samples": int(gt.shape[0]),
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
