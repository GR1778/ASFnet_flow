import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvn import datasets
from mvn.models.DGPose_dlst import DepthGuidedPoseDLST
from mvn.utils.cfg import config, update_config, update_dir

human36m_dataset = importlib.import_module("mvn.datasets.human36m")


JOINT_NAMES = [
    "pelvis", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle",
    "spine", "thorax", "neck", "head", "l_shoulder", "l_elbow", "l_wrist",
    "r_shoulder", "r_elbow", "r_wrist",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe natural depth-map gaps for ASFnet/DLST without changing the model."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="output/depth_input_gap_probe.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", default="hrnet_32", choices=["hrnet_32", "hrnet_48"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=4096)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--no-flip-test", action="store_true")
    parser.add_argument(
        "--conditions",
        default="real,zero,random,batch_shuffle,invert,blur",
        help=(
            "Comma-separated from real,zero,random,batch_shuffle,invert,blur,"
            "pose_jitter_4,pose_jitter_8,pose_jitter_16,"
            "pose_jitter_8_depth_shuffle,pose_jitter_16_depth_shuffle"
        ),
    )
    parser.add_argument("--jitter-seed", type=int, default=123)
    return parser.parse_args()


def build_val_loader(args):
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
    subset_indices = None
    dataset_for_loader = val_dataset
    if args.sample_size is not None and args.sample_size < len(val_dataset):
        rng = np.random.default_rng(args.sample_seed)
        subset_indices = np.sort(rng.choice(len(val_dataset), size=args.sample_size, replace=False))
        dataset_for_loader = Subset(val_dataset, subset_indices.tolist())
    loader = DataLoader(
        dataset_for_loader,
        batch_size=args.batch_size or config.val.batch_size,
        shuffle=False,
        num_workers=args.num_workers if args.num_workers is not None else config.val.num_workers,
        pin_memory=True,
    )
    return val_dataset, loader, subset_indices


def load_model(args, device):
    model = DepthGuidedPoseDLST(config, device=device)
    raw = torch.load(args.checkpoint, map_location="cpu")
    checkpoint = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    checkpoint = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    print(model.load_state_dict(checkpoint, strict=False))
    model.to(device)
    model.eval()
    return model


def preprocess_batch(batch, device, flip_test):
    images, keypoints_3d_gt, keypoints_2d_cpn, keypoints_2d_crop, depth_images = batch
    images = images.to(device, non_blocking=True).float()
    keypoints_3d_gt = keypoints_3d_gt.to(device, non_blocking=True).float()
    keypoints_2d_cpn = keypoints_2d_cpn.to(device, non_blocking=True).float()
    keypoints_2d_crop = keypoints_2d_crop.to(device, non_blocking=True).float()
    depth_images = depth_images.to(device, non_blocking=True).float() / 255.0

    images = torch.flip(images, [-1])
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)
    images = (images / 255.0 - mean) / std

    keypoints_3d_gt[:, :, 1:] -= keypoints_3d_gt[:, :, :1]
    keypoints_3d_gt[:, :, 0] = 0

    if flip_test:
        joints_left = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]
        images_flip = torch.flip(images, [2])
        depth_flip = torch.flip(depth_images, [2])

        k2d_flip = keypoints_2d_cpn.clone()
        k2d_flip[..., 0] *= -1
        k2d_flip[..., joints_left + joints_right, :] = k2d_flip[..., joints_right + joints_left, :]

        crop_flip = keypoints_2d_crop.clone()
        crop_flip[:, :, 0] = 192 - crop_flip[:, :, 0] - 1
        crop_flip[:, joints_left + joints_right] = crop_flip[:, joints_right + joints_left]

        images = torch.stack([images, images_flip], dim=1)
        depth_images = torch.stack([depth_images, depth_flip], dim=1)
        keypoints_2d_cpn = torch.stack([keypoints_2d_cpn, k2d_flip], dim=1)
        keypoints_2d_crop = torch.stack([keypoints_2d_crop, crop_flip], dim=1)
    return images, keypoints_3d_gt, keypoints_2d_cpn, keypoints_2d_crop, depth_images


def transform_depth(depth_images, condition):
    if condition == "real":
        return depth_images
    if condition == "zero":
        return torch.zeros_like(depth_images)
    if condition == "random":
        return torch.rand_like(depth_images)
    if condition == "batch_shuffle":
        if depth_images.shape[0] <= 1:
            return depth_images.flip(0)
        perm = torch.arange(depth_images.shape[0], device=depth_images.device).roll(1)
        return depth_images[perm]
    if condition == "invert":
        return 1.0 - depth_images
    if condition == "blur":
        if depth_images.dim() == 3:
            d = depth_images.unsqueeze(1)
            out = F.avg_pool2d(d, kernel_size=15, stride=1, padding=7)
            return out.squeeze(1)
        d = depth_images.flatten(0, 1).unsqueeze(1)
        out = F.avg_pool2d(d, kernel_size=15, stride=1, padding=7).squeeze(1)
        return out.view_as(depth_images)
    raise ValueError(condition)


def _condition_jitter_pixels(condition):
    if condition.startswith("pose_jitter_"):
        parts = condition.split("_")
        for part in parts:
            if part.isdigit():
                return float(part)
    return 0.0


def transform_pose_and_depth(keypoints_2d_cpn, keypoints_2d_crop, depth_images, condition, generator):
    k2d = keypoints_2d_cpn
    crop = keypoints_2d_crop
    depth = depth_images

    jitter_px = _condition_jitter_pixels(condition)
    if jitter_px > 0:
        noise = torch.randn(crop.shape, device=crop.device, dtype=crop.dtype, generator=generator) * jitter_px
        crop = crop + noise
        cpn_noise = noise.clone()
        cpn_noise[..., 0] = cpn_noise[..., 0] / (192 / 2)
        cpn_noise[..., 1] = cpn_noise[..., 1] / (256 / 2)
        k2d = k2d + cpn_noise

    if "depth_shuffle" in condition:
        depth = transform_depth(depth, "batch_shuffle")

    return k2d, crop, depth


def forward_model(model, images, keypoints_2d_cpn, keypoints_2d_crop, depth_images, flip_test):
    if flip_test:
        pred, _, _ = model(images[:, 0], keypoints_2d_cpn[:, 0], keypoints_2d_crop[:, 0].clone(), depth_images[:, 0])
        pred_flip, _, _ = model(
            images[:, 1], keypoints_2d_cpn[:, 1], keypoints_2d_crop[:, 1].clone(), depth_images[:, 1]
        )
        joints_left = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]
        pred_flip[:, :, :, 0] *= -1
        pred_flip[:, :, :, joints_left + joints_right] = pred_flip[:, :, :, joints_right + joints_left]
        pred = torch.mean(torch.cat((pred, pred_flip), dim=1), dim=1, keepdim=True)
    else:
        pred, _, _ = model(images, keypoints_2d_cpn, keypoints_2d_crop, depth_images)
    return pred


def sample_joint_depth(depth_images, keypoints_2d_crop):
    if depth_images.dim() == 4:
        depth_images = depth_images[:, 0]
        keypoints_2d_crop = keypoints_2d_crop[:, 0]
    ref = keypoints_2d_crop[..., :2].clone()
    ref[..., 0] = ref[..., 0] / (192 / 2) - 1
    ref[..., 1] = ref[..., 1] / (256 / 2) - 1
    sampled = F.grid_sample(depth_images.unsqueeze(1), ref.unsqueeze(-2), align_corners=True)
    return sampled.squeeze(1).squeeze(-1)


def safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def affine_r2_per_frame(sampled_depth, gt_z):
    r2s = []
    corrs = []
    for d, z in zip(sampled_depth, gt_z):
        d = d.astype(np.float64)
        z = z.astype(np.float64)
        corrs.append(safe_corr(d, z))
        a = np.stack([d, np.ones_like(d)], axis=1)
        coef = np.linalg.lstsq(a, z, rcond=None)[0]
        pred = a @ coef
        ss_res = ((z - pred) ** 2).sum()
        ss_tot = ((z - z.mean()) ** 2).sum()
        r2s.append(1.0 - ss_res / (ss_tot + 1e-12))
    return float(np.nanmean(corrs)), float(np.nanmean(r2s))


def summarize(pred, gt, action_idx):
    if gt.dim() == 4:
        gt = gt.squeeze(1)
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    per_joint = torch.linalg.norm(pred - gt, dim=-1).numpy() * 1000.0
    per_pose = per_joint.mean(axis=1)
    action_names = human36m_dataset.retval["action_names"]
    action_scores = {}
    for base_name in sorted({name[:-2] if name.endswith(("-1", "-2")) else name for name in action_names}):
        raw_indices = [
            i for i, name in enumerate(action_names)
            if (name[:-2] if name.endswith(("-1", "-2")) else name) == base_name
        ]
        mask = np.isin(action_idx, raw_indices)
        if mask.any():
            action_scores[base_name] = float(per_pose[mask].mean())
    return {
        "mpjpe": float(per_pose.mean()),
        "per_joint": {name: float(per_joint[:, idx].mean()) for idx, name in enumerate(JOINT_NAMES)},
        "actions": action_scores,
    }


def diff_summary(current, baseline):
    action_delta = {
        name: current["actions"][name] - baseline["actions"][name]
        for name in current["actions"] if name in baseline["actions"]
    }
    joint_delta = {
        name: current["per_joint"][name] - baseline["per_joint"][name]
        for name in current["per_joint"]
    }
    return {
        "delta_mpjpe": current["mpjpe"] - baseline["mpjpe"],
        "top_action_delta": sorted(action_delta.items(), key=lambda item: item[1], reverse=True)[:8],
        "top_joint_delta": sorted(joint_delta.items(), key=lambda item: item[1], reverse=True)[:8],
    }


def main():
    args = parse_args()
    update_config(args.config)
    update_dir("", "logs/")
    config.model.backbone.type = args.backbone
    config.model.poseformer.base_dim = 32 if args.backbone == "hrnet_32" else 48
    if args.no_flip_test:
        config.val.flip_test = False

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    val_dataset, loader, subset_indices = build_val_loader(args)
    model = load_model(args, device)
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    action_idx = val_dataset.labels_action_idx[subset_indices] if subset_indices is not None else val_dataset.labels_action_idx

    outputs = {}
    depth_samples = []
    gt_z_samples = []
    for condition in conditions:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.jitter_seed)
        preds = []
        gts = []
        print("Running condition={}".format(condition), flush=True)
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                images, gt, k2d, crop, depth = preprocess_batch(batch, device, config.val.flip_test)
                if condition == "real":
                    depth_samples.append(sample_joint_depth(depth, crop).detach().cpu())
                    gt_z_samples.append(gt.squeeze(1)[..., 2].detach().cpu())
                if condition.startswith("pose_jitter_"):
                    k2d_cond, crop_cond, depth_cond = transform_pose_and_depth(k2d, crop, depth, condition, generator)
                else:
                    k2d_cond, crop_cond = k2d, crop
                    depth_cond = transform_depth(depth, condition)
                pred = forward_model(model, images, k2d_cond, crop_cond, depth_cond, config.val.flip_test)
                preds.append(pred.detach().cpu())
                gts.append(gt.detach().cpu())
                print("batch {:04d}".format(batch_idx), flush=True)
        outputs[condition] = summarize(torch.cat(preds), torch.cat(gts), action_idx)
        if condition != "real":
            outputs[condition]["vs_real"] = diff_summary(outputs[condition], outputs["real"])
        print("{} MPJPE {:.3f}".format(condition, outputs[condition]["mpjpe"]), flush=True)

    sampled_depth = torch.cat(depth_samples).numpy()
    gt_z = torch.cat(gt_z_samples).numpy()
    depth_corr, depth_affine_r2 = affine_r2_per_frame(sampled_depth, gt_z)

    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "flip_test": bool(config.val.flip_test),
        "sample_size": args.sample_size,
        "sample_seed": args.sample_seed,
        "joint_depth_vs_gt_z": {
            "framewise_corr_mean": depth_corr,
            "framewise_affine_r2_mean": depth_affine_r2,
        },
        "results": outputs,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print("Wrote {}".format(args.output))


if __name__ == "__main__":
    main()
