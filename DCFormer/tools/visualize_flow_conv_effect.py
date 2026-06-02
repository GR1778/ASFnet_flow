import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from diagnose_flow_conv_kernel_effect import center_only_weight, encode_flow
from diagnose_flow_feature_sampling import load_flow_conv
from diagnose_flow_mfce_sampling import image_stem


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize raw optical flow before the Conv2d encoder and the 128D feature map after "
            "the trained 3x3 flow Conv2d. The output also compares the full 3x3 encoder with a "
            "center-only kernel that approximates a 1x1 projection."
        )
    )
    parser.add_argument("--flow", default="", help="Direct path to one .npy flow file.")
    parser.add_argument("--flow-dir", default="../H36M-Toolbox/flow_images_float")
    parser.add_argument("--seq", default="s_09_act_09_subact_01_ca_02")
    parser.add_argument("--frame-id", type=int, default=1202)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--conv-prefix", default="auto")
    parser.add_argument("--flow-clip", type=float, default=5.0)
    parser.add_argument("--out", default="debug_vis/flow_conv_effect_visualization.png")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def resolve_flow_path(args):
    if args.flow:
        return Path(args.flow).expanduser()
    return Path(args.flow_dir).expanduser() / args.seq / (image_stem(args.seq, args.frame_id) + ".npy")


def flow_to_rgb(flow):
    flow = np.asarray(flow, dtype=np.float32)
    fx, fy = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)
    denom = np.percentile(mag, 99) + 1e-6
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)
    hsv[..., 1] = np.clip(mag / denom * 255, 0, 255).astype(np.uint8)
    hsv[..., 2] = 255
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    low = mag < max(0.05, np.percentile(mag, 30))
    rgb[low] = 255
    return rgb


def normalize01(x, robust=True):
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.float32)
    if robust:
        lo = np.percentile(x[finite], 1)
        hi = np.percentile(x[finite], 99)
    else:
        lo = np.min(x[finite])
        hi = np.max(x[finite])
    if hi <= lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def pca_rgb(feature_map, mask=None):
    h, w, c = feature_map.shape
    flat = feature_map.reshape(-1, c).astype(np.float32)
    if mask is not None and np.any(mask.reshape(-1)):
        fit = flat[mask.reshape(-1)]
    else:
        fit = flat
    fit = fit - fit.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(fit, full_matrices=False)
    proj = (flat - flat.mean(axis=0, keepdims=True)) @ vh[:3].T
    rgb = np.stack([normalize01(proj[:, i]) for i in range(3)], axis=-1)
    return rgb.reshape(h, w, 3)


def make_support_mask(flow):
    mag = np.linalg.norm(flow, axis=-1)
    return mag > max(0.05, np.percentile(mag, 35))


def main():
    args = parse_args()
    flow_path = resolve_flow_path(args)
    if not flow_path.exists():
        raise FileNotFoundError("Flow file not found: {}".format(flow_path))

    conv_prefix, weight, bias = load_flow_conv(args.checkpoint, args.conv_prefix)
    raw_flow = np.load(flow_path).astype(np.float32)
    if raw_flow.ndim != 3 or raw_flow.shape[-1] != 2:
        raise ValueError("Expected flow [H,W,2], got {} at {}".format(raw_flow.shape, flow_path))

    flow = raw_flow.copy()
    if args.flow_clip > 0:
        flow = np.clip(flow, -args.flow_clip, args.flow_clip) / args.flow_clip

    full_feature = encode_flow(flow, weight, bias)
    center_feature = encode_flow(flow, center_only_weight(weight), bias)
    neighbor_feature = full_feature - center_feature

    raw_rgb = flow_to_rgb(raw_flow)
    raw_mag = np.linalg.norm(raw_flow, axis=-1)
    support = make_support_mask(flow)
    full_pca = pca_rgb(full_feature, support)
    center_pca = pca_rgb(center_feature, support)
    neighbor_norm = np.linalg.norm(neighbor_feature, axis=-1)
    delta_norm = np.linalg.norm(full_feature - center_feature, axis=-1)
    full_norm = np.linalg.norm(full_feature, axis=-1)
    rel_delta = delta_norm / (full_norm + 1e-8)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    axes = axes.reshape(2, 3)

    axes[0, 0].imshow(raw_rgb)
    axes[0, 0].set_title("Raw flow color wheel")

    im = axes[0, 1].imshow(normalize01(raw_mag), cmap="magma")
    axes[0, 1].set_title("Raw flow magnitude")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.02)

    axes[0, 2].imshow(full_pca)
    axes[0, 2].set_title("After 3x3 conv: 128D PCA")

    axes[1, 0].imshow(center_pca)
    axes[1, 0].set_title("Center-only kernel: PCA")

    im = axes[1, 1].imshow(normalize01(neighbor_norm), cmap="viridis")
    axes[1, 1].set_title("Neighbor contribution ||3x3 - center||")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.02)

    im = axes[1, 2].imshow(normalize01(rel_delta), cmap="inferno")
    axes[1, 2].set_title("Relative change from 3x3")
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.02)

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle("{} frame {} | conv: {}".format(args.seq, args.frame_id, conv_prefix), fontsize=12)
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print("Wrote {}".format(out_path))


if __name__ == "__main__":
    main()
