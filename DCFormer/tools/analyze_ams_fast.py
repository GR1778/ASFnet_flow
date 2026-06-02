import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from mvn import datasets
from mvn.datasets import utils as dataset_utils
from mvn.models.DGPose import DepthGuidedPose
from mvn.utils.cfg import config, update_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="depth_analysis_ams_fast")
    parser.add_argument("--retain_every_n_frames", type=int, default=100)
    parser.add_argument("--skip_ablation", action="store_true")
    return parser.parse_args()


def to_numpy(x):
    return x.detach().float().cpu().numpy()


def corrcoef_safe(x, y):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def mpjpe_root(pred, gt):
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if gt.dim() == 4:
        gt = gt.squeeze(1)
    pred = pred - pred[:, :1]
    gt = gt - gt[:, :1]
    return torch.norm(pred - gt, dim=-1).mean()


def feature_depth_corr(features, gt_3d):
    if gt_3d.dim() == 4:
        gt_3d = gt_3d.squeeze(1)
    gt_z = gt_3d[..., 2]
    b, j, c = features.shape
    corrs = []
    features_np = to_numpy(features)
    gt_np = to_numpy(gt_z)
    for bi in range(b):
        for dim in range(c):
            r = corrcoef_safe(features_np[bi, :, dim], gt_np[bi])
            if np.isfinite(r):
                corrs.append(r)
    return float(np.nanmean(corrs)) if corrs else np.nan


def build_loader(args):
    val_dataset = eval("datasets." + config.dataset.val_dataset)(
        root=config.dataset.root,
        pred_results_path=config.val.pred_results_path,
        depth_image_path=config.dataset.depth_image_path,
        train=False,
        test=True,
        image_shape=config.model.image_shape,
        labels_path=config.dataset.val_labels_path,
        with_damaged_actions=config.val.with_damaged_actions,
        retain_every_n_frames_in_test=args.retain_every_n_frames,
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


def load_model(args, device):
    model = DepthGuidedPose(config, device).to(device)
    raw = torch.load(args.checkpoint, map_location=device)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("[stage] update config", flush=True)
    update_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[stage] load model on {device}", flush=True)
    model = load_model(args, device)
    print("[stage] build loader", flush=True)
    loader = build_loader(args)
    print("[stage] start batches", flush=True)

    captured = {}
    last_ams = model.Lifting_net.RGBD_Extraction[-1]

    def hook(_module, inputs, output):
        captured["last_input"] = inputs[0].detach()
        captured["last_ref"] = inputs[1].detach()
        captured["last_output"] = output.detach()

    handle = last_ams.register_forward_hook(hook)
    stats = defaultdict(list)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images, gt_3d, keypoints_2d, keypoints_2d_crop, depth_images = batch
            images = images.to(device, non_blocking=True)
            gt_3d = gt_3d.to(device, non_blocking=True)
            keypoints_2d = keypoints_2d.to(device, non_blocking=True)
            keypoints_2d_crop = keypoints_2d_crop.to(device, non_blocking=True)
            depth_images = depth_images.to(device, non_blocking=True)

            pred_real, coarse_depth, uncer = model(images, keypoints_2d, keypoints_2d_crop.clone(), depth_images)
            ams_input = captured["last_input"]
            ams_output = captured["last_output"]
            x0, x_rest = ams_input[:, :1], ams_input[:, 1:]
            x_norm = last_ams.norm1(x_rest + x0)
            b, levels, joints = x_rest.shape[:3]
            offsets = last_ams.sampling_offsets(x_norm).reshape(
                b, levels, joints, last_ams.num_heads * last_ams.num_samples, 2
            ).tanh()
            offsets_mag = torch.norm(offsets, dim=-1)

            depth_token_in = ams_input[:, -1]
            depth_token_out = ams_output[:, -1]
            stats["ams_offsets_mean"].append(offsets_mag.mean().item())
            stats["ams_offsets_std"].append(offsets_mag.std(unbiased=False).item())
            stats["ams_offsets_p90"].append(torch.quantile(offsets_mag.flatten(), 0.9).item())
            stats["ams_offsets_max"].append(offsets_mag.max().item())
            stats["ams_depth_token_delta_norm"].append(torch.norm(depth_token_out - depth_token_in, dim=-1).mean().item())
            stats["ams_depth_token_in_gtz_corr"].append(feature_depth_corr(depth_token_in, gt_3d))
            stats["ams_depth_token_out_gtz_corr"].append(feature_depth_corr(depth_token_out, gt_3d))
            stats["coarse_depth_gtz_corr"].append(corrcoef_safe(to_numpy(coarse_depth), to_numpy(gt_3d[..., 2:3])))

            real = mpjpe_root(pred_real, gt_3d).item() * 1000.0
            stats["mpjpe_real_depth"].append(real)

            if not args.skip_ablation:
                random_depth = torch.randn_like(depth_images) * depth_images.std() + depth_images.mean()
                zero_depth = torch.zeros_like(depth_images)
                shuffled_depth = depth_images[torch.randperm(depth_images.shape[0], device=device)]
                pred_random, _, _ = model(images, keypoints_2d, keypoints_2d_crop.clone(), random_depth)
                pred_zero, _, _ = model(images, keypoints_2d, keypoints_2d_crop.clone(), zero_depth)
                pred_shuffled, _, _ = model(images, keypoints_2d, keypoints_2d_crop.clone(), shuffled_depth)
                random = mpjpe_root(pred_random, gt_3d).item() * 1000.0
                zero = mpjpe_root(pred_zero, gt_3d).item() * 1000.0
                shuffled = mpjpe_root(pred_shuffled, gt_3d).item() * 1000.0
                stats["mpjpe_random_depth"].append(random)
                stats["mpjpe_zero_depth"].append(zero)
                stats["mpjpe_shuffled_depth"].append(shuffled)
                stats["random_minus_real"].append(random - real)
                stats["zero_minus_real"].append(zero - real)
                stats["shuffled_minus_real"].append(shuffled - real)
            else:
                random = zero = shuffled = float("nan")

            print(
                f"[batch {batch_idx + 1}] real={real:.2f} random_gap={random-real:.2f} "
                f"zero_gap={zero-real:.2f} off_mean={stats['ams_offsets_mean'][-1]:.4f}"
            , flush=True)

    handle.remove()

    summary = {}
    for key, values in stats.items():
        arr = np.asarray(values, dtype=np.float64)
        summary[key] = {
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
        }

    summary["verdict"] = {
        "ams_adaptive": "weak" if summary["ams_offsets_mean"]["mean"] < 0.05 else "moderate",
        "ams_depth_alignment": "weak" if abs(summary["ams_depth_token_out_gtz_corr"]["mean"]) < 0.1 else "visible",
    }
    if "random_minus_real" in summary:
        summary["verdict"]["depth_semantic_use"] = (
            "weak" if summary["random_minus_real"]["mean"] < 3.0 else "meaningful"
        )
        summary["verdict"]["any_depth_benefit"] = (
            "weak" if summary["zero_minus_real"]["mean"] < 3.0 else "meaningful"
        )

    out_path = os.path.join(args.output_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
