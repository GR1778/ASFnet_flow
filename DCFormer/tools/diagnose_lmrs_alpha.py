import argparse
import json
import math
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from mvn import datasets
from mvn.datasets import utils as dataset_utils
from mvn.models.DGPose_rgbflow_lmrs import RGBFlowPoseLMRS
from mvn.utils.cfg import config, update_config


JOINT_NAMES = [
    "Hip", "RHip", "RKnee", "RFoot", "LHip", "LKnee", "LFoot", "Spine",
    "Thorax", "Neck/Nose", "Head", "LShoulder", "LElbow", "LWrist",
    "RShoulder", "RElbow", "RWrist",
]


def build_val_loader(batch_size, num_workers):
    dataset = datasets.human36m(
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
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=dataset_utils.worker_init_fn,
    )


def load_model(checkpoint, device):
    model = RGBFlowPoseLMRS(config, device=str(device))
    raw = torch.load(checkpoint, map_location="cpu")
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    state = {key.replace("module.", ""): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print("Unexpected keys:", unexpected[:10], "... total", len(unexpected))
    if missing:
        print("Missing keys:", missing[:10], "... total", len(missing))
    model.to(device)
    model.eval()
    return model


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    update_config(args.config)
    config.val.flip_test = False
    config.val.batch_size = args.batch_size
    config.val.num_workers = args.num_workers

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    loader = build_val_loader(args.batch_size, args.num_workers)
    prefetcher = dataset_utils.data_prefetcher(
        loader,
        device,
        is_train=False,
        flip_test=False,
        aux_mode="flow",
        flow_clip=getattr(config.dataset, "flow_clip", 20.0),
        flow_norm=getattr(config.dataset, "flow_norm", None),
    )

    selector = model.Lifting_net.Flow_Sampling
    offsets = selector.candidate_offsets_px.detach().cpu().numpy()
    radii = np.sqrt((offsets ** 2).sum(axis=1))
    rounded_radii = np.round(radii, 4)
    radius_labels = [float(x) for x in rounded_radii]
    radius_order = sorted(set(radius_labels))

    all_alpha = []
    batch = prefetcher.next()
    seen = 0
    with torch.no_grad():
        while batch is not None and seen < args.max_batches:
            images, _, k2d, k2d_crop, flow = batch
            model(images, k2d, k2d_crop.clone(), flow)
            alpha = selector.latest_alpha.detach().float().cpu()
            all_alpha.append(alpha)
            seen += 1
            batch = prefetcher.next()

    if not all_alpha:
        raise RuntimeError("No batches were processed")

    alpha = torch.cat(all_alpha, dim=0)  # [B, J, K]
    b, j, k = alpha.shape
    eps = 1e-12
    entropy = -(alpha.clamp_min(eps) * alpha.clamp_min(eps).log()).sum(dim=-1)
    entropy_norm = entropy / math.log(k)
    eff_k = entropy.exp()
    max_w, top_idx = alpha.max(dim=-1)
    center_w = alpha[..., 0]
    top3_mass = alpha.topk(min(3, k), dim=-1).values.sum(dim=-1)
    top8_mass = alpha.topk(min(8, k), dim=-1).values.sum(dim=-1)

    top_idx_np = top_idx.numpy().reshape(-1)
    radius_np = rounded_radii[top_idx_np]
    radius_count = Counter(float(x) for x in radius_np)
    radius_frac = {str(r): radius_count.get(r, 0) / float(top_idx_np.size) for r in radius_order}

    joint_rows = []
    for ji in range(j):
        joint_top = top_idx[:, ji].numpy()
        joint_radius = rounded_radii[joint_top]
        counts = Counter(float(x) for x in joint_radius)
        row = {
            "joint": ji,
            "name": JOINT_NAMES[ji] if ji < len(JOINT_NAMES) else str(ji),
            "center_weight_mean": float(center_w[:, ji].mean()),
            "top1_center_frac": float((top_idx[:, ji] == 0).float().mean()),
            "entropy_norm_mean": float(entropy_norm[:, ji].mean()),
            "max_weight_mean": float(max_w[:, ji].mean()),
            "top1_radius_mode": counts.most_common(1)[0][0],
            "top1_radius_mode_frac": counts.most_common(1)[0][1] / float(len(joint_top)),
        }
        joint_rows.append(row)

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "samples": int(b),
        "joints": int(j),
        "candidates": int(k),
        "alpha_sum": summarize(alpha.sum(dim=-1).reshape(-1).numpy()),
        "center_weight": summarize(center_w.reshape(-1).numpy()),
        "max_weight": summarize(max_w.reshape(-1).numpy()),
        "top3_mass": summarize(top3_mass.reshape(-1).numpy()),
        "top8_mass": summarize(top8_mass.reshape(-1).numpy()),
        "entropy_norm": summarize(entropy_norm.reshape(-1).numpy()),
        "effective_k": summarize(eff_k.reshape(-1).numpy()),
        "top1_center_frac": float((top_idx == 0).float().mean()),
        "top1_radius_frac": radius_frac,
        "joint_summary": joint_rows,
    }

    text = json.dumps(summary, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
