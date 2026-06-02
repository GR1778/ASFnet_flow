
import os, sys, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist

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
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="depth_analysis_fd_pose")
    return p.parse_args()


def main():
    args = parse_args()
    update_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("[model] loading checkpoint")
    model = DepthGuidedPose(config, device).to(device)
    raw = torch.load(args.checkpoint, map_location=device)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

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

    fd_buffer, pose_buffer = [], []
    handle = model.Lifting_net.RGBD_Extraction[-1].register_forward_hook(
        lambda _m, _i, o: fd_buffer.append(o[:, -1].detach().cpu().float())
    )
    prefetcher = dataset_utils.data_prefetcher(val_loader, device, is_train=False, flip_test=False)
    batch = prefetcher.next()
    bi = 0
    with torch.no_grad():
        while batch is not None and bi < args.num_batches:
            images, kp3d_gt, kp2d, kp2d_crop, depth = batch
            _ = model(images, kp2d, kp2d_crop.clone(), depth)
            pose_buffer.append(kp3d_gt[:, 0].detach().cpu().float())
            bi += 1
            print("  batch", bi, "/", args.num_batches)
            batch = prefetcher.next()
    handle.remove()

    F_d = torch.cat(fd_buffer, dim=0).numpy()
    pose = torch.cat(pose_buffer, dim=0).numpy()
    N = F_d.shape[0]
    print("[run] N =", N)

    actions = None
    try:
        actions = np.array([int(val_dataset.labels[i]["action"]) for i in range(N)])
        print("[run] actions unique:", np.unique(actions))
    except Exception as e:
        print("[run] no actions:", e)

    F_d_flat = F_d.reshape(N, -1)
    pose_flat = pose.reshape(N, -1)
    fd_dist = pdist(F_d_flat, metric="euclidean")
    pose_dist = pdist(pose_flat, metric="euclidean")
    sr, _ = spearmanr(fd_dist, pose_dist)
    pr, _ = pearsonr(fd_dist, pose_dist)
    print("Spearman r =", round(float(sr),4), "  Pearson r =", round(float(pr),4))

    pj = []
    for j in range(17):
        r, _ = spearmanr(pdist(F_d[:, j], "euclidean"), pdist(pose[:, j], "euclidean"))
        pj.append(float(r))
    pj = np.array(pj)
    print("per-joint r mean=", round(float(pj.mean()),4), "min=", round(float(pj.min()),4), "max=", round(float(pj.max()),4))

    action_stats = None
    within_arr = between_arr = None
    if actions is not None and len(np.unique(actions)) > 1:
        wl, bl = [], []
        for i in range(N):
            for j in range(i+1, N):
                d = float(np.linalg.norm(F_d_flat[i] - F_d_flat[j]))
                (wl if actions[i] == actions[j] else bl).append(d)
        within_arr = np.array(wl); between_arr = np.array(bl)
        action_stats = {
            "within_action_mean": float(within_arr.mean()),
            "between_action_mean": float(between_arr.mean()),
            "gap_normalized": float((between_arr.mean() - within_arr.mean()) / (between_arr.mean() + 1e-8)),
        }
        print("action stats:", action_stats)

    summary = {
        "num_samples": int(N),
        "spearman_r": float(sr), "pearson_r": float(pr),
        "per_joint_r_mean": float(pj.mean()),
        "per_joint_r_min": float(pj.min()),
        "per_joint_r_max": float(pj.max()),
        "action_stats": action_stats,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    n_plot = min(20000, len(fd_dist))
    idx = np.random.choice(len(fd_dist), n_plot, replace=False)
    axes[0].scatter(pose_dist[idx], fd_dist[idx], s=1, alpha=0.2)
    axes[0].set_xlabel("Pairwise GT 3D pose distance (mm)")
    axes[0].set_ylabel("Pairwise F_d distance")
    axes[0].set_title("F_d dist vs pose dist  Spearman r=" + str(round(float(sr),3)))
    axes[0].grid(alpha=0.3)
    axes[1].bar(np.arange(17), pj, color="C2", alpha=0.8)
    axes[1].axhline(pj.mean(), color="k", linestyle="--", label="mean=" + str(round(float(pj.mean()),3)))
    axes[1].set_xlabel("Joint"); axes[1].set_ylabel("Spearman r per joint")
    axes[1].set_title("Per-joint F_d vs pose correlation")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    if action_stats is not None:
        axes[2].hist(within_arr, bins=40, alpha=0.6, label="within mu=" + str(round(action_stats["within_action_mean"],1)), color="C0", density=True)
        axes[2].hist(between_arr, bins=40, alpha=0.6, label="between mu=" + str(round(action_stats["between_action_mean"],1)), color="C3", density=True)
        axes[2].set_xlabel("F_d pairwise distance"); axes[2].set_ylabel("Density")
        axes[2].set_title("Action separation gap=" + str(round(action_stats["gap_normalized"],3)))
        axes[2].legend(); axes[2].grid(alpha=0.3)
    else:
        axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "fd_pose_alignment.png"), dpi=150)
    print("done")

    if sr >= 0.6:
        print("VERDICT: F_d already pose-aware, DBA WEAK")
    elif sr >= 0.3:
        print("VERDICT: partially aligned, DBA moderate")
    else:
        print("VERDICT: F_d NOT pose-aware, DBA has clear room")


if __name__ == "__main__":
    main()
