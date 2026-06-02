import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvn import datasets
from mvn.models.DGPose_dlst import DepthGuidedPoseDLST
from mvn.utils.cfg import config, update_config, update_dir

human36m_dataset = importlib.import_module("mvn.datasets.human36m")


JOINT_GROUPS = {
    "all": list(range(17)),
    "torso": [0, 7, 8, 9, 10],
    "hips": [1, 4],
    "shoulders": [11, 14],
    "elbows": [12, 15],
    "wrists": [13, 16],
    "knees": [2, 5],
    "ankles": [3, 6],
    "arms": [11, 12, 13, 14, 15, 16],
    "legs": [1, 2, 3, 4, 5, 6],
    "limb_ends": [3, 6, 13, 16],
}


JOINT_NAMES = [
    "pelvis",
    "r_hip",
    "r_knee",
    "r_ankle",
    "l_hip",
    "l_knee",
    "l_ankle",
    "spine",
    "thorax",
    "neck",
    "head",
    "l_shoulder",
    "l_elbow",
    "l_wrist",
    "r_shoulder",
    "r_elbow",
    "r_wrist",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate whether DLST depth-token contribution is joint/action dependent."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="depth_contribution_results.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", default="hrnet_32", choices=["hrnet_32", "hrnet_48"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None, help="Random validation subset size.")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--no-flip-test", action="store_true")
    parser.add_argument(
        "--groups",
        default="all,wrists,elbows,knees,ankles,torso,arms,legs,limb_ends",
        help="Comma-separated joint groups from: {}".format(",".join(sorted(JOINT_GROUPS))),
    )
    parser.add_argument(
        "--replacement",
        default="zero",
        choices=["zero", "mean", "raw"],
        help="How to replace ablated depth tokens after DLST. raw keeps AMS depth token and removes only DLST update.",
    )
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
    dataset_for_loader = val_dataset
    subset_indices = None
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
    ret = model.load_state_dict(checkpoint, strict=False)
    print(ret)
    model.to(device)
    model.eval()
    return model


def preprocess_batch(batch, device, flip_test):
    images, keypoints_3d_gt, keypoints_2d_cpn, keypoints_2d_crop, depth_images = batch
    images = images.to(device, non_blocking=True).float()
    keypoints_3d_gt = keypoints_3d_gt.to(device, non_blocking=True).float()
    keypoints_2d_cpn = keypoints_2d_cpn.to(device, non_blocking=True).float()
    keypoints_2d_crop = keypoints_2d_crop.to(device, non_blocking=True).float()
    depth_images = depth_images.to(device, non_blocking=True).float()

    depth_images = depth_images / 255.0
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


def set_depth_ablation(model, joint_indices, replacement):
    dlst = model.Lifting_net.dlst
    if hasattr(dlst, "_depth_contribution_orig_forward"):
        dlst.forward = dlst._depth_contribution_orig_forward
    dlst._depth_contribution_orig_forward = dlst.forward

    if joint_indices is None:
        return

    joint_indices = torch.tensor(joint_indices, dtype=torch.long)
    orig_forward = dlst._depth_contribution_orig_forward

    def wrapped_forward(joint_tokens):
        out, rel_depth, layer_assign = orig_forward(joint_tokens)
        idx = joint_indices.to(out.device)
        out = out.clone()
        if replacement == "zero":
            repl = torch.zeros_like(out[:, idx])
        elif replacement == "mean":
            repl = out.mean(dim=1, keepdim=True).expand(-1, idx.numel(), -1)
        elif replacement == "raw":
            repl = joint_tokens[:, idx]
        else:
            raise ValueError(replacement)
        out[:, idx] = repl
        return out, rel_depth, layer_assign

    dlst.forward = wrapped_forward


def restore_depth_ablation(model):
    dlst = model.Lifting_net.dlst
    if hasattr(dlst, "_depth_contribution_orig_forward"):
        dlst.forward = dlst._depth_contribution_orig_forward
        delattr(dlst, "_depth_contribution_orig_forward")


def forward_model(model, batch, flip_test):
    images, keypoints_3d_gt, keypoints_2d_cpn, keypoints_2d_crop, depth_images = batch
    if flip_test:
        pred, _, _ = model(images[:, 0], keypoints_2d_cpn[:, 0], keypoints_2d_crop[:, 0].clone(), depth_images[:, 0])
        pred_flip, _, _ = model(images[:, 1], keypoints_2d_cpn[:, 1], keypoints_2d_crop[:, 1].clone(), depth_images[:, 1])
        joints_left = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]
        pred_flip[:, :, :, 0] *= -1
        pred_flip[:, :, :, joints_left + joints_right] = pred_flip[:, :, :, joints_right + joints_left]
        pred = torch.mean(torch.cat((pred, pred_flip), dim=1), dim=1, keepdim=True)
    else:
        pred, _, _ = model(images, keypoints_2d_cpn, keypoints_2d_crop, depth_images)
    return pred, keypoints_3d_gt


def collect_predictions(model, loader, device, flip_test, max_batches):
    preds = []
    gts = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = preprocess_batch(batch, device, flip_test)
            pred, gt = forward_model(model, batch, flip_test)
            preds.append(pred.detach().cpu())
            gts.append(gt.detach().cpu())
            print("batch {:04d}".format(batch_idx), flush=True)
    return torch.cat(preds, dim=0), torch.cat(gts, dim=0)


def action_base_name(action_name):
    if action_name.endswith("-1") or action_name.endswith("-2"):
        return action_name[:-2]
    return action_name


def summarize(pred, gt, action_idx):
    if gt.dim() == 4:
        gt = gt.squeeze(1)
    if pred.dim() == 4:
        pred = pred.squeeze(1)

    per_joint = torch.linalg.norm(pred - gt, dim=-1).numpy() * 1000.0
    per_pose = per_joint.mean(axis=1)

    action_names = human36m_dataset.retval["action_names"]
    action_scores = {}
    for base_name in sorted({action_base_name(name) for name in action_names}):
        raw_indices = [i for i, name in enumerate(action_names) if action_base_name(name) == base_name]
        mask = np.isin(action_idx, raw_indices)
        if mask.any():
            action_scores[base_name] = float(per_pose[mask].mean())

    return {
        "mpjpe": float(per_pose.mean()),
        "per_joint": {name: float(per_joint[:, idx].mean()) for idx, name in enumerate(JOINT_NAMES)},
        "actions": action_scores,
        "num_samples": int(per_pose.shape[0]),
    }


def diff_summary(current, baseline):
    action_delta = {
        name: current["actions"][name] - baseline["actions"][name]
        for name in current["actions"]
        if name in baseline["actions"]
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
    if args.backbone == "hrnet_32":
        config.model.poseformer.base_dim = 32
    else:
        config.model.poseformer.base_dim = 48
    if args.no_flip_test:
        config.val.flip_test = False

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    val_dataset, loader, subset_indices = build_val_loader(args)
    model = load_model(args, device)

    groups = [item.strip() for item in args.groups.split(",") if item.strip()]
    for group in groups:
        if group not in JOINT_GROUPS:
            raise ValueError("Unknown group '{}'. Available: {}".format(group, sorted(JOINT_GROUPS)))

    if subset_indices is not None:
        action_idx = val_dataset.labels_action_idx[subset_indices]
        if args.max_batches is not None:
            action_idx = action_idx[: args.max_batches * loader.batch_size]
    else:
        max_samples = None if args.max_batches is None else min(len(val_dataset), args.max_batches * loader.batch_size)
        action_idx = val_dataset.labels_action_idx[:max_samples]

    results = {}
    print("Running baseline normal depth-token eval")
    restore_depth_ablation(model)
    pred, gt = collect_predictions(model, loader, device, config.val.flip_test, args.max_batches)
    results["normal"] = summarize(pred, gt, action_idx)
    print("normal MPJPE {:.3f} mm".format(results["normal"]["mpjpe"]))

    for group in groups:
        print("Running ablation group={} replacement={}".format(group, args.replacement))
        set_depth_ablation(model, JOINT_GROUPS[group], args.replacement)
        pred, gt = collect_predictions(model, loader, device, config.val.flip_test, args.max_batches)
        results[group] = summarize(pred, gt, action_idx)
        results[group]["vs_normal"] = diff_summary(results[group], results["normal"])
        print(
            "{} MPJPE {:.3f} mm delta {:+.3f} mm".format(
                group, results[group]["mpjpe"], results[group]["vs_normal"]["delta_mpjpe"]
            )
        )

    restore_depth_ablation(model)
    output = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "replacement": args.replacement,
        "flip_test": bool(config.val.flip_test),
        "max_batches": args.max_batches,
        "sample_size": args.sample_size,
        "sample_seed": args.sample_seed,
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print("Wrote {}".format(args.output))


if __name__ == "__main__":
    main()
