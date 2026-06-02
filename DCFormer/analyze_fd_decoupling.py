"""
Empirical motivation test for DSD (Depth Signal Decoupling).

Hypothesis: AMS-output F_d in [B, 17, 128] is dominated by a frame-shared
"body-scale" component, with per-joint deviations carrying only a small
fraction of variance. If true, vanilla self-attention on F_d (UDE-style)
behaves close to averaging because tokens are too similar to discriminate.
"""
import os
import sys
import json
import argparse
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default="checkpoints/best_epoch.bin")
    p.add_argument("--num_batches", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="depth_analysis_fd")
    return p.parse_args()


def build_val_loader(args):
    val_dataset = eval("datasets." + config.dataset.val_dataset)(
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
        rank=None,
        world_size=None,
        data_format=config.dataset.data_format,
        frame=1,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=True,
    )
    return val_loader


def main():
    args = parse_args()
    update_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[setup] device = {device}")
    print(f"[setup] config = {args.config}")

    print("[model] building DepthGuidedPose ...")
    model = DepthGuidedPose(config, device).to(device)

    print(f"[model] loading checkpoint = {args.checkpoint}")
    raw = torch.load(args.checkpoint, map_location=device)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    state = {k.replace("module.", ""): v for k, v in state.items()}
    msg = model.load_state_dict(state, strict=False)
    print(f"[model] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    model.eval()

    print("[data] building val loader ...")
    val_loader = build_val_loader(args)

    fd_buffer = []

    def hook(_m, _inp, out):
        fd_buffer.append(out[:, -1].detach().cpu().float())

    handle = model.Lifting_net.RGBD_Extraction[-1].register_forward_hook(hook)

    print(f"[run] capturing F_d for up to {args.num_batches} batches ...")
    prefetcher = dataset_utils.data_prefetcher(val_loader, device, is_train=False, flip_test=False)
    batch = prefetcher.next()
    bi = 0
    with torch.no_grad():
        while batch is not None and bi < args.num_batches:
            images_batch, _kp3d_gt, kp2d_cpn, kp2d_cpn_crop, depth_images = batch
            _ = model(images_batch, kp2d_cpn, kp2d_cpn_crop.clone(), depth_images)
            print(f"  batch {bi+1}/{args.num_batches}  F_d={tuple(fd_buffer[-1].shape)}")
            bi += 1
            batch = prefetcher.next()

    handle.remove()
    if not fd_buffer:
        raise RuntimeError("No F_d captured.")

    F_d = torch.cat(fd_buffer, dim=0).numpy()
    N, J, C = F_d.shape
    print(f"[run] captured F_d: shape={F_d.shape}")

    F_norm = F_d / (np.linalg.norm(F_d, axis=-1, keepdims=True) + 1e-8)
    cos_per = np.einsum("nic,njc->nij", F_norm, F_norm)
    cos_mean = cos_per.mean(axis=0)
    off_diag_mean = (cos_mean.sum() - np.trace(cos_mean)) / (J * J - J)
    off_diag_min = float(np.min(cos_mean + np.eye(J) * 2))

    body_mean_per = F_d.mean(axis=1, keepdims=True)
    residual_per = F_d - body_mean_per

    sv_ratios = []
    for n in range(N):
        s = np.linalg.svd(residual_per[n], compute_uv=False)
        var = s ** 2
        sv_ratios.append(var / (var.sum() + 1e-12))
    sv_ratios = np.stack(sv_ratios, axis=0)
    sv_mean = sv_ratios.mean(axis=0)

    body_mean_energy = np.sum(body_mean_per ** 2, axis=(1, 2))
    residual_energy = np.sum(residual_per ** 2, axis=(1, 2)) / J
    body_mean_fraction = body_mean_energy / (body_mean_energy + residual_energy + 1e-12)

    summary = {
        "num_samples": int(N),
        "F_d_shape": list(F_d.shape),
        "cosine_offdiag_mean": float(off_diag_mean),
        "cosine_offdiag_min": float(off_diag_min),
        "pca_pc1_var": float(sv_mean[0]),
        "pca_top3_var": float(sv_mean[:3].sum()),
        "body_mean_energy_fraction_mean": float(body_mean_fraction.mean()),
        "body_mean_energy_fraction_std": float(body_mean_fraction.std()),
    }
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:35s} = {v:.4f}")
        else:
            print(f"  {k:35s} = {v}")

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im = axes[0].imshow(cos_mean, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0].set_title(
        f"Per-joint F_d cosine similarity (avg over {N} samples)\n"
        f"off-diagonal mean = {off_diag_mean:.3f}   min = {off_diag_min:.3f}"
    )
    axes[0].set_xlabel("joint j")
    axes[0].set_ylabel("joint i")
    plt.colorbar(im, ax=axes[0])

    pcs = np.arange(1, J + 1)
    axes[1].bar(pcs, sv_mean * 100, alpha=0.7, label="per PC")
    axes[1].plot(pcs, np.cumsum(sv_mean) * 100, "k-o", label="cumulative", linewidth=2)
    axes[1].set_title(
        f"PCA on per-frame F_d residuals (body-mean subtracted)\n"
        f"PC1={sv_mean[0]*100:.1f}%   top-3={sv_mean[:3].sum()*100:.1f}%   "
        f"body-mean energy={body_mean_fraction.mean()*100:.1f}%"
    )
    axes[1].set_xlabel("Principal component")
    axes[1].set_ylabel("Variance ratio (%)")
    axes[1].set_ylim([0, 105])
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(args.output_dir, "fd_decoupling_motivation.png")
    plt.savefig(out_png, dpi=150)
    print(f"\n[done] figure -> {out_png}")
    print(f"[done] summary -> {os.path.join(args.output_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
