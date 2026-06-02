#!/usr/bin/env python3
"""
Visualize a trained DLST checkpoint.

The script saves per-sample panels containing:
  - 2D skeleton colored by predicted depth layer
  - joint-to-layer assignment A
  - predicted relative depth matrix R
  - GT relative depth sign matrix
  - sign-error matrix
  - optional averaged depth-biased attention map
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mvn import datasets
from mvn.datasets import utils as dataset_utils
from mvn.models.DGPose_dlst import DepthGuidedPoseDLST
from mvn.utils.cfg import config, update_config


BONES = [
    (0, 1), (1, 2), (2, 6),
    (5, 4), (4, 3), (3, 6),
    (6, 7), (7, 8), (8, 16), (9, 16),
    (8, 12), (11, 12), (10, 11),
    (8, 13), (13, 14), (14, 15),
]

JOINT_NAMES = [
    "rank", "rkne", "rhip", "lhip", "lkne", "lank", "root", "belly", "neck",
    "head", "rwri", "relb", "rsho", "lsho", "lelb", "lwri", "nose",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", default="dlst_visualizations")
    p.add_argument("--num_samples", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--random_subset", action="store_true")
    p.add_argument("--retain_every_n_frames", type=int, default=None)
    p.add_argument("--margin", type=float, default=0.05)
    return p.parse_args()


def build_dataset(args: argparse.Namespace):
    retain = (
        args.retain_every_n_frames
        if args.retain_every_n_frames is not None
        else config.val.retain_every_n_frames_in_test
    )
    return eval("datasets." + config.dataset.val_dataset)(
        root=config.dataset.root,
        pred_results_path=config.val.pred_results_path,
        depth_image_path=config.dataset.depth_image_path,
        train=False,
        test=True,
        image_shape=config.model.image_shape,
        labels_path=config.dataset.val_labels_path,
        with_damaged_actions=config.val.with_damaged_actions,
        retain_every_n_frames_in_test=retain,
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


def select_indices(args: argparse.Namespace, dataset) -> np.ndarray:
    n = min(args.num_samples, len(dataset))
    if args.random_subset:
        rng = np.random.default_rng(args.seed)
        return np.sort(rng.choice(len(dataset), size=n, replace=False))
    return np.arange(n)


def build_loader(args: argparse.Namespace):
    dataset = build_dataset(args)
    indices = select_indices(args, dataset)
    subset = Subset(dataset, indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=True,
    )
    return loader, dataset, indices


def load_model(args: argparse.Namespace, device: torch.device):
    model = DepthGuidedPoseDLST(config, device).to(device)
    raw = torch.load(args.checkpoint, map_location=device)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    state = {k.replace("module.", ""): v for k, v in state.items()}
    msg = model.load_state_dict(state, strict=False)
    print(f"[model] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    model.eval()
    return model


def action_name(label: Dict[str, object]) -> str:
    return "S{}_A{:02d}_SA{:02d}_C{:02d}_F{:06d}".format(
        label["subject"],
        label["action"],
        label["subaction"],
        label["camera_id"] + 1,
        label["image_id"],
    )


def valid_margin(z: np.ndarray, margin: float) -> float:
    if np.median(np.abs(z)) > 10.0 and margin < 1.0:
        return margin * 1000.0
    return margin


def gt_rel_from_z(z: np.ndarray, margin: float) -> Tuple[np.ndarray, np.ndarray]:
    # Match train.py / DepthOrderingLoss:
    # torch version uses z.unsqueeze(1) - z.unsqueeze(2), so entry [i,j] is z_j - z_i.
    diff = z[None, :] - z[:, None]
    eye = np.eye(len(z), dtype=bool)
    m = valid_margin(z, margin)
    mask = (np.abs(diff) > m) & (~eye)
    return np.sign(diff), mask


def sign_accuracy(rel: np.ndarray, z: np.ndarray, margin: float) -> Tuple[float, float]:
    target, mask = gt_rel_from_z(z, margin)
    pred = np.sign(rel)
    if not np.any(mask):
        return float("nan"), 0.0
    return float(np.mean(pred[mask] == target[mask])), float(np.mean(pred[mask] != 0))


def normalize_image_for_plot(img: np.ndarray) -> np.ndarray:
    # data_prefetcher normalizes RGB after BGR->RGB flip. Undo that for visualization.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    out = img.astype(np.float32) * std + mean
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def put_text(img: np.ndarray, text: str, org: Tuple[int, int], scale=0.48, color=(20, 20, 20), thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def layer_color(layer: int, k: int) -> Tuple[int, int, int]:
    palette = [
        (60, 120, 230),
        (80, 180, 120),
        (230, 170, 60),
        (200, 90, 190),
        (80, 180, 220),
        (180, 180, 80),
    ]
    return palette[layer % len(palette)]


def draw_skeleton_panel(image: np.ndarray, joints: np.ndarray, layer_ids: np.ndarray, title: str) -> np.ndarray:
    canvas = np.full((360, 300, 3), 255, dtype=np.uint8)
    img = cv2.resize(image, (192, 256))
    canvas[62:318, 54:246] = img
    scale = np.array([192.0 / image.shape[1], 256.0 / image.shape[0]], dtype=np.float32)
    pts = joints * scale + np.array([54.0, 62.0], dtype=np.float32)
    k = int(layer_ids.max()) + 1 if layer_ids.size else 4
    for i, j in BONES:
        pi = tuple(np.round(pts[i]).astype(int))
        pj = tuple(np.round(pts[j]).astype(int))
        cv2.line(canvas, pi, pj, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(canvas, pi, pj, (30, 30, 30), 1, cv2.LINE_AA)
    for idx, (x, y) in enumerate(joints):
        p = tuple(np.round(pts[idx]).astype(int))
        cv2.circle(canvas, p, 5, layer_color(int(layer_ids[idx]), k), -1, cv2.LINE_AA)
        cv2.circle(canvas, p, 5, (20, 20, 20), 1, cv2.LINE_AA)
        put_text(canvas, str(idx), (p[0] + 5, p[1] + 4), scale=0.32, color=(0, 0, 0))
    put_text(canvas, title[:42], (10, 22), scale=0.45, color=(0, 0, 0), thickness=1)
    put_text(canvas, "2D skeleton colored by predicted layer", (10, 42), scale=0.38, color=(50, 50, 50))
    return canvas


def colorize_matrix(mat: np.ndarray, vmin: float, vmax: float, cmap: str) -> np.ndarray:
    x = np.asarray(mat, dtype=np.float32)
    x = (x - vmin) / max(vmax - vmin, 1e-8)
    x = np.clip(x, 0.0, 1.0)
    if cmap == "coolwarm":
        # blue -> white -> red, RGB
        rgb = np.zeros((*x.shape, 3), dtype=np.float32)
        low = x < 0.5
        t = np.where(low, x / 0.5, (x - 0.5) / 0.5)
        rgb[..., 0] = np.where(low, 70 + 185 * t, 255)
        rgb[..., 1] = np.where(low, 120 + 135 * t, 255 - 190 * t)
        rgb[..., 2] = np.where(low, 220 + 35 * t, 255 - 210 * t)
    elif cmap == "reds":
        rgb = np.stack([255 * np.ones_like(x), 255 * (1 - x), 255 * (1 - x)], axis=-1)
    elif cmap == "magma":
        rgb = np.stack([255 * x, 80 * x, 40 + 150 * (1 - x)], axis=-1)
    else:
        rgb = np.stack([255 * np.ones_like(x), 230 - 180 * x, 160 * (1 - x)], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def heatmap_panel(mat: np.ndarray, title: str, vmin: float, vmax: float, cmap="coolwarm", cell=18) -> np.ndarray:
    h, w = mat.shape
    top, left, right, bottom = 46, 76, 18, 28
    canvas = np.full((top + h * cell + bottom, left + w * cell + right, 3), 255, dtype=np.uint8)
    put_text(canvas, title, (8, 22), scale=0.45, color=(0, 0, 0), thickness=1)
    colors = colorize_matrix(mat, vmin, vmax, cmap)
    for r in range(h):
        for c in range(w):
            y0 = top + r * cell
            x0 = left + c * cell
            cv2.rectangle(canvas, (x0, y0), (x0 + cell, y0 + cell), tuple(int(v) for v in colors[r, c]), -1)
            cv2.rectangle(canvas, (x0, y0), (x0 + cell, y0 + cell), (220, 220, 220), 1)
    if h == len(JOINT_NAMES):
        for r, name in enumerate(JOINT_NAMES):
            put_text(canvas, name, (4, top + r * cell + 13), scale=0.28, color=(60, 60, 60))
    for c in range(w):
        label = str(c)
        put_text(canvas, label, (left + c * cell + 4, top + h * cell + 17), scale=0.28, color=(60, 60, 60))
    return canvas


def resize_to(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def compose_grid(panels: List[np.ndarray], cols: int = 3, cell_size: Tuple[int, int] = (360, 360)) -> np.ndarray:
    resized = [resize_to(p, cell_size) for p in panels]
    rows = int(np.ceil(len(resized) / cols))
    canvas = np.full((rows * cell_size[1], cols * cell_size[0], 3), 255, dtype=np.uint8)
    for idx, panel in enumerate(resized):
        r, c = divmod(idx, cols)
        y0, x0 = r * cell_size[1], c * cell_size[0]
        canvas[y0:y0 + cell_size[1], x0:x0 + cell_size[0]] = panel
    return canvas


def save_sample_panel(
    output_path: str,
    image: np.ndarray,
    joints_2d: np.ndarray,
    assign: np.ndarray,
    rel: np.ndarray,
    z: np.ndarray,
    attn: np.ndarray,
    title: str,
    margin: float,
):
    layer_ids = np.argmax(assign, axis=-1)
    gt_rel, valid_mask = gt_rel_from_z(z, margin)
    pred_sign = np.sign(rel)
    errors = np.zeros_like(rel)
    errors[valid_mask] = (pred_sign[valid_mask] != gt_rel[valid_mask]).astype(np.float32)
    acc, resolved = sign_accuracy(rel, z, margin)

    panels = []
    panels.append(draw_skeleton_panel(
        image,
        joints_2d,
        layer_ids,
        f"{title}\nlayer-colored 2D skeleton | sign_acc={acc:.3f} resolved={resolved:.3f}",
    ))
    panels.append(heatmap_panel(assign, "Layer assignment A [J x K]", 0, 1, cmap="YlOrBr", cell=34))
    panels.append(heatmap_panel(rel, "Predicted R = AΩA^T", -1, 1, cmap="coolwarm", cell=18))
    panels.append(heatmap_panel(gt_rel, "GT depth-order sign", -1, 1, cmap="coolwarm", cell=18))
    panels.append(heatmap_panel(errors, "Sign errors on valid pairs", 0, 1, cmap="reds", cell=18))
    if attn is not None:
        panels.append(heatmap_panel(attn, "Mean depth-biased attention", 0, max(float(attn.max()), 1e-6), cmap="magma", cell=18))
    grid = compose_grid(panels, cols=3, cell_size=(360, 360))
    cv2.imwrite(output_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def get_last_dlst_attention(model) -> np.ndarray:
    try:
        blocks = model.Lifting_net.dlst.depth_blocks
        if not blocks:
            return None
        attn = blocks[-1].attn.last_attn
        if attn is None:
            return None
        # [B,H,J,J] -> [B,J,J]
        return attn.detach().float().mean(dim=1).cpu().numpy()
    except AttributeError:
        return None


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    update_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    loader, full_dataset, selected_indices = build_loader(args)

    summary: List[Dict[str, object]] = []
    prefetcher = dataset_utils.data_prefetcher(loader, device, is_train=False, flip_test=False)
    batch = prefetcher.next()
    seen = 0

    with torch.no_grad():
        while batch is not None and seen < args.num_samples:
            images, gt_3d, kp2d, kp2d_crop, depth_images = batch
            pred, rel_depth, assign = model(images, kp2d, kp2d_crop.clone(), depth_images)
            del pred
            attn = get_last_dlst_attention(model)

            if gt_3d.dim() == 4:
                gt_3d = gt_3d.squeeze(1)
            bsz = gt_3d.shape[0]

            images_np = images.detach().cpu().numpy()
            kp2d_crop_np = kp2d_crop.detach().cpu().numpy()
            z_np = gt_3d[..., 2].detach().cpu().numpy()
            rel_np = rel_depth.detach().cpu().numpy()
            assign_np = assign.detach().cpu().numpy()

            for bi in range(bsz):
                if seen >= args.num_samples:
                    break
                original_idx = int(selected_indices[seen])
                label = full_dataset.labels[original_idx]
                title = action_name(label)
                acc, resolved = sign_accuracy(rel_np[bi], z_np[bi], args.margin)
                out_name = f"{seen:03d}_{title}.png"
                out_path = os.path.join(args.output_dir, out_name)
                save_sample_panel(
                    out_path,
                    normalize_image_for_plot(images_np[bi]),
                    kp2d_crop_np[bi],
                    assign_np[bi],
                    rel_np[bi],
                    z_np[bi],
                    attn[bi] if attn is not None else None,
                    title,
                    args.margin,
                )
                summary.append({
                    "sample": seen,
                    "dataset_index": original_idx,
                    "title": title,
                    "file": out_name,
                    "sign_acc": acc,
                    "resolved_rate": resolved,
                    "layer_usage": assign_np[bi].mean(axis=0).tolist(),
                    "assignment_entropy": float(-(assign_np[bi] * np.log(np.clip(assign_np[bi], 1e-8, 1.0))).sum(axis=-1).mean() / np.log(assign_np[bi].shape[-1])),
                })
                print(f"[save] {out_path} sign_acc={acc:.3f}", flush=True)
                seen += 1

            batch = prefetcher.next()

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": args.config,
            "checkpoint": args.checkpoint,
            "num_samples": len(summary),
            "mean_sign_acc": float(np.nanmean([x["sign_acc"] for x in summary])) if summary else float("nan"),
            "samples": summary,
        }, f, indent=2, ensure_ascii=False)
    print(f"[write] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
