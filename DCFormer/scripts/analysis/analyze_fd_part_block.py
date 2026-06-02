"""
Sanity check: is the body-part block structure in F_d real or noise?

Computes within-part vs between-part cosine similarity using H36M parts.
If story holds: within-part >> between-part, with consistent positive gap.
"""
import os, sys, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from mvn.models.DGPose import DepthGuidedPose
from mvn.utils.cfg import config, update_config
from mvn import datasets
from mvn.datasets import utils as dataset_utils

PARTS_3 = {
    "legs"  : [0, 1, 2, 3, 4, 5, 6],
    "torso" : [7, 8, 9, 10],
    "arms"  : [11, 12, 13, 14, 15, 16],
}
PARTS_5 = {
    "rleg" : [1, 2, 3],
    "lleg" : [4, 5, 6],
    "torso": [7, 8, 9, 10],
    "rarm" : [11, 12, 13],
    "larm" : [14, 15, 16],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default="checkpoints/best_epoch.bin")
    p.add_argument("--num_batches", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="depth_analysis_fd_parts")
    return p.parse_args()


def part_block_stats(F_d, parts):
    F_norm = F_d / (np.linalg.norm(F_d, axis=-1, keepdims=True) + 1e-8)
    cos = np.einsum("nic,njc->nij", F_norm, F_norm)
    J = F_d.shape[1]
    within_mask = np.zeros((J, J), dtype=bool)
    for _, idx in parts.items():
        idx = np.array(idx)
        within_mask[np.ix_(idx, idx)] = True
    np.fill_diagonal(within_mask, False)
    between_mask = ~within_mask
    np.fill_diagonal(between_mask, False)
    within_per = cos[:, within_mask].mean(axis=1)
    between_per = cos[:, between_mask].mean(axis=1)
    return within_per, between_per


def main():
    args = parse_args()
    update_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("[model] building + loading checkpoint ...")
    model = DepthGuidedPose(config, device).to(device)
    raw = torch.load(args.checkpoint, map_location=device)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    print("[data] building val loader ...")
    val_dataset = eval("datasets." + config.dataset.val_dataset)(
        root=config.dataset.root, pred_results_path=config.val.pred_results_path,
        depth_image_path=config.dataset.depth_image_path, train=False, test=True,
        image_shape=config.model.image_shape, labels_path=config.dataset.val_labels_path,
        with_damaged_actions=config.val.with_damaged_actions,
        retain_every_n_frames_in_test=config.val.retain_every_n_frames_in_test,
        scale_bbox=config.val.scale_bbox, kind=config.kind,
        undistort_images=config.val.undistort_images, ignore_cameras=config.val.ignore_cameras,
        crop=config.val.crop, erase=config.val.erase, rank=None, world_size=None,
        data_format=config.dataset.data_format, frame=1,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, worker_init_fn=dataset_utils.worker_init_fn, pin_memory=True,
    )

    fd_buffer = []
    handle = model.Lifting_net.RGBD_Extraction[-1].register_forward_hook(
        lambda _m, _i, o: fd_buffer.append(o[:, -1].detach().cpu().float())
    )
    prefetcher = dataset_utils.data_prefetcher(val_loader, device, is_train=False, flip_test=False)
    batch = prefetcher.next()
    bi = 0
    print(f"[run] capturing F_d for {args.num_batches} batches ...")
    with torch.no_grad():
        while batch is not None and bi < args.num_batches:
            images, _gt, kp2d, kp2d_crop, depth = batch
            _ = model(images, kp2d, kp2d_crop.clone(), depth)
            bi += 1
            print(f"  batch {bi}/{args.num_batches}  F_d={tuple(fd_buffer[-1].shape)}")
            batch = prefetcher.next()
    handle.remove()

    F_d = torch.cat(fd_buffer, dim=0).numpy()
    N = F_d.shape[0]
    print(f"[run] N = {N} samples")

    w3, b3 = part_block_stats(F_d, PARTS_3)
    w5, b5 = part_block_stats(F_d, PARTS_5)

    summary = {
        "num_samples": int(N),
        "coarse_3parts": {
            "within_mean": float(w3.mean()), "within_std": float(w3.std()),
            "between_mean": float(b3.mean()), "between_std": float(b3.std()),
            "gap_mean": float((w3 - b3).mean()),
            "gap_std": float((w3 - b3).std()),
            "gap_pct_positive": float(((w3 - b3) > 0).mean()),
        },
        "fine_5parts": {
            "within_mean": float(w5.mean()), "within_std": float(w5.std()),
            "between_mean": float(b5.mean()), "between_std": float(b5.std()),
            "gap_mean": float((w5 - b5).mean()),
            "gap_std": float((w5 - b5).std()),
            "gap_pct_positive": float(((w5 - b5) > 0).mean()),
        },
    }

    print("\n=== PART-BLOCK STATS ===")
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                if isinstance(vv, float):
                    print(f"    {kk:25s} = {vv:.4f}")
                else:
                    print(f"    {kk:25s} = {vv}")
        else:
            print(f"  {k}: {v}")

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, w, b, label in [(axes[0], w3, b3, "3 parts (legs/torso/arms)"),
                              (axes[1], w5, b5, "5 parts (rleg/lleg/torso/rarm/larm)")]:
        ax.hist(w, bins=30, alpha=0.6, label=f"within-part μ={w.mean():.3f}", color="C0")
        ax.hist(b, bins=30, alpha=0.6, label=f"between-part μ={b.mean():.3f}", color="C3")
        ax.axvline(w.mean(), color="C0", linestyle="--")
        ax.axvline(b.mean(), color="C3", linestyle="--")
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Sample count")
        ax.set_title(f"{label}\ngap = {(w-b).mean():.3f} (positive in {((w-b)>0).mean()*100:.0f}% of samples)")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(args.output_dir, "fd_part_block.png")
    plt.savefig(out_png, dpi=150)
    print(f"\n[done] figure -> {out_png}")


if __name__ == "__main__":
    main()
