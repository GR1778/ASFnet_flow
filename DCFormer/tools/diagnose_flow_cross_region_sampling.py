import argparse
import json
import math
import pickle
import random
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnose_flow_mfce_sampling import (
    ACTION_NAMES,
    JOINT_NAMES,
    affine_apply,
    bilinear_sample_flow,
    get_affine_transform,
    image_stem,
    patch_stats,
    seq_name,
    summarize,
)


PARENTS = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Clean diagnostic for optical-flow sampling correction. It ignores old JSON "
            "outputs and tests only cases where CPN and GT joints lie in different "
            "local flow-motion regions."
        )
    )
    parser.add_argument("--labels", default="data/h36m_validation.pkl")
    parser.add_argument("--flow-dir", default="../H36M-Toolbox/flow_images_float")
    parser.add_argument("--image-root", default="../H36M-Toolbox/images_crop")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--flow-clip", type=float, default=5.0)
    parser.add_argument("--flow-norm", type=float, default=5.0)
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-radius", type=int, default=3)
    parser.add_argument("--min-2d-gap", type=float, default=4.0)
    parser.add_argument("--min-flow-gap", type=float, default=1.0)
    parser.add_argument("--min-center-badness", type=float, default=0.5)
    parser.add_argument("--good-margin", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="debug_vis/flow_cross_region_sampling_clean.json")
    parser.add_argument("--vis-dir", default="debug_vis/flow_cross_region_sampling_clean_vis")
    parser.add_argument("--no-vis", action="store_true")
    return parser.parse_args()


def load_labels(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def action_name(action):
    idx = int(action) - 2
    if 0 <= idx < len(ACTION_NAMES):
        return ACTION_NAMES[idx]
    return str(action)


def joint_name(joint):
    if 0 <= int(joint) < len(JOINT_NAMES):
        return JOINT_NAMES[int(joint)]
    return str(joint)


def resolve_checkpoint_path(checkpoint_path):
    path = Path(checkpoint_path).expanduser()
    if path.exists():
        return path
    text = str(path)
    candidates = []
    if "\uf03a" in text:
        candidates.append(Path(text.replace("\uf03a", ":")))
    if ":" in text:
        candidates.append(Path(text.replace(":", "\uf03a")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))


def strip_lifting_prefix(state):
    out = {}
    for key, value in state.items():
        for prefix in ("module.Lifting_net.", "Lifting_net."):
            if key.startswith(prefix):
                out[key[len(prefix) :]] = value
                break
    return out


def load_flow_probe(checkpoint_path, dim, depth, num_heads, num_samples):
    import torch
    import torch.nn as nn

    from mvn.models.DGLifting_rgbflow_mfce_separate import FlowMFCE

    class FlowProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.coord_embed = nn.Linear(2, dim)
            self.motion_field_embed = nn.Conv2d(2, dim, kernel_size=3, padding=1)
            self.flow_feat_embed = nn.Linear(dim, dim)
            self.RGB_pos_embed = nn.Parameter(torch.zeros(1, 5, 17, dim))
            self.Flow_pos_embed = nn.Parameter(torch.zeros(1, 1, 17, dim))
            self.Flow_Extraction = FlowMFCE(
                dim=dim,
                depth=depth,
                num_heads=num_heads,
                num_samples=num_samples,
                qkv_bias=True,
                drop_path=[0.0] * depth,
            )

    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    raw = torch.load(str(checkpoint_path), map_location="cpu")
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    state = strip_lifting_prefix(state)
    probe = FlowProbe()
    missing, unexpected = probe.load_state_dict(state, strict=False)
    critical = (
        "coord_embed",
        "motion_field_embed",
        "flow_feat_embed",
        "RGB_pos_embed",
        "Flow_pos_embed",
        "Flow_Extraction",
    )
    critical_missing = [key for key in missing if key.startswith(critical)]
    if critical_missing:
        raise KeyError("Missing critical flow-probe weights: {}".format(critical_missing[:20]))
    probe.eval()
    return probe, str(checkpoint_path), len(unexpected)


def normalize_flow_for_model(flow_raw, flow_clip, flow_norm):
    flow = flow_raw.astype(np.float32, copy=True)
    if flow_clip is not None and flow_clip > 0:
        flow = np.clip(flow, -flow_clip, flow_clip)
    if flow_norm is not None and flow_norm > 0:
        flow = flow / flow_norm
    return flow


def ref_from_crop_xy(xy, width, height):
    ref = np.asarray(xy, dtype=np.float32).copy()
    ref[..., 0] = ref[..., 0] / (width // 2) - 1.0
    ref[..., 1] = ref[..., 1] / (height // 2) - 1.0
    return ref


def norm_to_xy(pos, width, height):
    xy = np.asarray(pos, dtype=np.float32).copy()
    xy[..., 0] = (xy[..., 0] + 1.0) * (width - 1) / 2.0
    xy[..., 1] = (xy[..., 1] + 1.0) * (height - 1) / 2.0
    return xy


def normalized_entropy(weights, eps=1e-8):
    w = np.asarray(weights, dtype=np.float64)
    return -np.sum(w * np.log(w + eps), axis=-1) / max(math.log(w.shape[-1]), eps)


def compute_learned_sampling(probe, keypoints_2d, ref, flow_model, device):
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        keypoints = torch.from_numpy(keypoints_2d[None]).float().to(device)
        ref_t = torch.from_numpy(ref[None]).float().to(device)
        flow_t = torch.from_numpy(flow_model[None]).float().to(device)

        pose_token = probe.coord_embed(keypoints)
        motion_field = probe.motion_field_embed(flow_t.permute(0, 3, 1, 2).contiguous())
        flow_token = F.grid_sample(motion_field, ref_t.unsqueeze(-2), align_corners=True)
        flow_token = flow_token.squeeze(-1).permute(0, 2, 1).contiguous()
        flow_token = probe.flow_feat_embed(flow_token)
        x = torch.stack(
            [
                pose_token + probe.RGB_pos_embed[:, 0],
                flow_token + probe.Flow_pos_embed[:, 0],
            ],
            dim=1,
        )

        block_outputs = []
        for block_idx, block in enumerate(probe.Flow_Extraction.blocks):
            x_0, x_rest = x[:, :1], x[:, 1:]
            residual = x_rest
            normed = block.norm1(x_rest + x_0)
            weights = block.attention_weights(normed).view(
                1,
                1,
                17,
                block.num_heads,
                block.num_samples,
            )
            weights = F.softmax(weights, dim=-1)
            offsets = block.sampling_offsets(normed).reshape(
                1,
                1,
                17,
                block.num_heads * block.num_samples,
                2,
            ).tanh()
            pos = offsets + ref_t.view(1, 1, 17, 1, 2)

            sampled = F.grid_sample(probe.motion_field_embed(flow_t.permute(0, 3, 1, 2).contiguous()), pos[:, 0], padding_mode="border", align_corners=True)
            sampled = sampled.permute(0, 2, 3, 1).contiguous()
            sampled = block.embed_proj[0](sampled)
            sampled = sampled.view(1, 17, block.num_heads, block.num_samples, -1)
            features_sampled = (weights[:, 0].unsqueeze(-1) * sampled).sum(dim=-2).view(1, 1, 17, -1)

            x_rest = residual + block.drop_path(features_sampled)
            x_rest = x_rest + block.drop_path(block.mlp(block.norm2(x_rest)))
            x = torch.cat([x_0, x_rest], dim=1)

            block_outputs.append(
                {
                    "block": block_idx,
                    "pos": pos[0, 0].detach().cpu().numpy(),
                    "weights": weights[0, 0].detach().cpu().numpy(),
                }
            )
    return block_outputs


def flow_to_rgb(flow):
    flow = np.asarray(flow, dtype=np.float32)
    fx, fy = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)
    hsv[..., 1] = np.clip(mag / (np.percentile(mag, 99) + 1e-6) * 255, 0, 255).astype(np.uint8)
    hsv[..., 2] = 255
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb[mag < max(0.05, np.percentile(mag, 25))] = 255
    return rgb


def draw_skeleton(ax, joints, color, linewidth=1.0, alpha=0.85):
    for joint, parent in enumerate(PARENTS):
        if parent < 0:
            continue
        ax.plot(
            [joints[parent, 0], joints[joint, 0]],
            [joints[parent, 1], joints[joint, 1]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )


def image_rel_path(row):
    seq = "s_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}".format(
        int(row["subject"]),
        int(row["action"]),
        int(row["subaction"]),
        int(row["camera_id"]) + 1,
    )
    return str(Path(seq) / "{}_{:06d}.jpg".format(seq, int(row["frame_id"])))


def read_image(image_root, row):
    path = Path(image_root) / image_rel_path(row)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def add_flow_arrow(ax, xy, vec, color, label, scale=4.0):
    ax.arrow(
        xy[0],
        xy[1],
        vec[0] * scale,
        vec[1] * scale,
        color=color,
        width=0.35,
        head_width=3.0,
        length_includes_head=True,
        alpha=0.95,
        label=label,
    )


def visualize_case(case, output_path, image_root, width, height):
    row = case["row"]
    cur = case["cur"]
    flow = case["flow"]
    xy = case["xy"]
    weights = case["weights"]
    sampled_err = case["sampled_err"]
    joint_idx = row["joint"]
    cpn = np.asarray(cur["joints_2d_cpn_crop"], dtype=np.float32)
    gt = np.asarray(cur["joints_2d_gt_crop"], dtype=np.float32)

    image = read_image(image_root, row)
    flow_rgb = flow_to_rgb(flow)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    if image is not None:
        plot_cpn = cpn.copy()
        plot_gt = gt.copy()
        image_h, image_w = image.shape[:2]
        if (image_w, image_h) != (width, height):
            scale = np.array([image_w / float(width), image_h / float(height)], dtype=np.float32)
            plot_cpn *= scale
            plot_gt *= scale
        axes[0].imshow(image)
        draw_skeleton(axes[0], plot_cpn, "#4f6cff", linewidth=1.1)
        draw_skeleton(axes[0], plot_gt, "#ff3030", linewidth=1.1)
        axes[0].scatter(plot_cpn[joint_idx, 0], plot_cpn[joint_idx, 1], marker="D", c="#4f6cff", s=55, label="CPN")
        axes[0].scatter(plot_gt[joint_idx, 0], plot_gt[joint_idx, 1], marker="D", c="#ff3030", s=55, label="GT")
        axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_title("{} {} f{}".format(row["joint_name"], row["action_name"], row["frame_id"]))
    axes[0].axis("off")

    flat_xy = xy.reshape(-1, 2)
    flat_w = weights.reshape(-1)
    flat_err = sampled_err.reshape(-1)
    size = 20 + 360 * (flat_w / (flat_w.max() + 1e-8))

    axes[1].imshow(flow_rgb)
    scatter = axes[1].scatter(
        flat_xy[:, 0],
        flat_xy[:, 1],
        s=size,
        c=flat_err,
        cmap="viridis_r",
        edgecolor="white",
        linewidth=0.55,
        alpha=0.88,
        label="samples",
    )
    axes[1].scatter(cpn[joint_idx, 0], cpn[joint_idx, 1], marker="D", c="#4f6cff", s=70, label="CPN")
    axes[1].scatter(gt[joint_idx, 0], gt[joint_idx, 1], marker="D", c="#ff3030", s=70, label="GT")
    axes[1].scatter(row["weighted_xy_x"], row["weighted_xy_y"], marker="*", c="#ffd400", s=110, label="weighted")
    add_flow_arrow(axes[1], cpn[joint_idx], np.array([row["center_flow_x"], row["center_flow_y"]]), "#4f6cff", "center flow")
    add_flow_arrow(axes[1], gt[joint_idx], np.array([row["gt_flow_x"], row["gt_flow_y"]]), "#ff3030", "GT-site flow")
    axes[1].set_xlim(0, width - 1)
    axes[1].set_ylim(height - 1, 0)
    axes[1].legend(loc="lower right", fontsize=7)
    axes[1].set_title(
        "flow err c/w/b {:.2f}/{:.2f}/{:.2f}".format(
            row["center_flow_error_px"],
            row["weighted_flow_error_px"],
            row["best_sample_flow_error_px"],
        )
    )
    axes[1].axis("off")
    fig.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.02, label="sample flow error")

    zoom_pad = 35
    xmin = max(0, min(cpn[joint_idx, 0], gt[joint_idx, 0], flat_xy[:, 0].min()) - zoom_pad)
    xmax = min(width - 1, max(cpn[joint_idx, 0], gt[joint_idx, 0], flat_xy[:, 0].max()) + zoom_pad)
    ymin = max(0, min(cpn[joint_idx, 1], gt[joint_idx, 1], flat_xy[:, 1].min()) - zoom_pad)
    ymax = min(height - 1, max(cpn[joint_idx, 1], gt[joint_idx, 1], flat_xy[:, 1].max()) + zoom_pad)
    axes[2].imshow(flow_rgb)
    axes[2].scatter(flat_xy[:, 0], flat_xy[:, 1], s=size, c=flat_err, cmap="viridis_r", edgecolor="white", linewidth=0.55, alpha=0.88)
    axes[2].scatter(cpn[joint_idx, 0], cpn[joint_idx, 1], marker="D", c="#4f6cff", s=80)
    axes[2].scatter(gt[joint_idx, 0], gt[joint_idx, 1], marker="D", c="#ff3030", s=80)
    axes[2].scatter(row["weighted_xy_x"], row["weighted_xy_y"], marker="*", c="#ffd400", s=130)
    axes[2].plot([cpn[joint_idx, 0], gt[joint_idx, 0]], [cpn[joint_idx, 1], gt[joint_idx, 1]], color="white", linewidth=1.2)
    axes[2].set_xlim(xmin, xmax)
    axes[2].set_ylim(ymax, ymin)
    axes[2].set_title(
        "2D gap {:.1f}, flow gap {:.1f}, gain {:.2f}".format(
            row["cpn_gt_pos_gap_px"],
            row["cpn_gt_flow_gap_px"],
            row["weighted_flow_gain_px"],
        )
    )
    axes[2].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def fraction(rows, predicate):
    if not rows:
        return None
    return float(np.mean([bool(predicate(row)) for row in rows]))


def metric(rows, key):
    return summarize([row[key] for row in rows])


def summarize_rows(rows):
    by_block = {}
    for block_idx in sorted({row["block"] for row in rows}):
        block_rows = [row for row in rows if row["block"] == block_idx]
        by_block[str(block_idx)] = {
            "num_cases": len(block_rows),
            "weighted_flow_gain_px": metric(block_rows, "weighted_flow_gain_px"),
            "best_sample_gain_px": metric(block_rows, "best_sample_gain_px"),
            "weighted_pos_gain_px": metric(block_rows, "weighted_pos_gain_px"),
            "toward_gt_ratio": metric(block_rows, "toward_gt_ratio"),
            "gt_region_mass": metric(block_rows, "gt_region_mass"),
            "center_region_mass": metric(block_rows, "center_region_mass"),
            "top1_gt_region_frac": metric(block_rows, "top1_gt_region_frac"),
            "weight_entropy": metric(block_rows, "weight_entropy"),
            "frac_weighted_flow_improves": fraction(block_rows, lambda row: row["weighted_flow_gain_px"] > 0),
            "frac_weighted_pos_improves": fraction(block_rows, lambda row: row["weighted_pos_gain_px"] > 0),
            "frac_gt_mass_exceeds_center_mass": fraction(block_rows, lambda row: row["gt_region_mass"] > row["center_region_mass"]),
            "frac_top1_in_gt_region": fraction(block_rows, lambda row: row["top1_gt_region_frac"] > 0.5),
        }
    return {
        "by_block": by_block,
        "top_success": sorted(rows, key=lambda row: row["weighted_flow_gain_px"], reverse=True)[:40],
        "top_failure": sorted(rows, key=lambda row: row["weighted_flow_gain_px"])[:40],
        "top_oracle_opportunity": sorted(rows, key=lambda row: row["best_sample_gain_px"], reverse=True)[:40],
    }


def main():
    args = parse_args()
    labels_path = Path(args.labels).expanduser().resolve()
    flow_dir = Path(args.flow_dir).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    if not flow_dir.exists():
        raise FileNotFoundError("Flow directory not found: {}".format(flow_dir))

    import torch

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    probe, checkpoint_path, unexpected_count = load_flow_probe(
        args.checkpoint,
        args.dim,
        args.depth,
        args.num_heads,
        args.num_samples,
    )
    probe.to(device)

    labels = load_labels(labels_path)
    by_key = {}
    for shot in labels:
        key = (
            int(shot["subject"]),
            int(shot["action"]),
            int(shot["subaction"]),
            int(shot["camera_id"]),
            int(shot["image_id"]),
        )
        by_key[key] = shot

    keys = list(by_key)
    random.Random(args.seed).shuffle(keys)
    output_size = (args.width, args.height)
    rows = []
    selected_frame_count = 0
    scanned_frame_count = 0
    missing_prev = 0
    missing_flow = 0
    candidate_joint_count = 0

    for key in keys:
        if scanned_frame_count >= args.max_frames:
            break
        subject, action, subaction, camera_id, frame_id = key
        prev_key = (subject, action, subaction, camera_id, frame_id - args.frame_gap)
        prev = by_key.get(prev_key)
        if prev is None:
            missing_prev += 1
            continue

        cur = by_key[key]
        seq = seq_name(cur)
        flow_path = flow_dir / seq / (image_stem(seq, frame_id) + ".npy")
        if not flow_path.exists():
            missing_flow += 1
            continue
        scanned_frame_count += 1

        flow_raw = np.load(flow_path).astype(np.float32)
        flow_model = normalize_flow_for_model(flow_raw, args.flow_clip, args.flow_norm)
        cur_cpn = np.asarray(cur["joints_2d_cpn_crop"], dtype=np.float32)
        cur_gt = np.asarray(cur["joints_2d_gt_crop"], dtype=np.float32)
        prev_gt = np.asarray(prev["joints_2d_gt_crop"], dtype=np.float32)
        keypoints_2d = np.asarray(cur["joints_2d_cpn"], dtype=np.float32)
        ref = ref_from_crop_xy(cur_cpn, args.width, args.height)

        prev_to_raw = get_affine_transform(prev["center"], prev["scale"], output_size, inv=True)
        raw_to_cur = get_affine_transform(cur["center"], cur["scale"], output_size, inv=False)
        prev_gt_same = affine_apply(raw_to_cur, affine_apply(prev_to_raw, prev_gt))
        gt_target = prev_gt_same - cur_gt

        flow_cpn = bilinear_sample_flow(flow_raw, cur_cpn)
        flow_gt = bilinear_sample_flow(flow_raw, cur_gt)
        center_error = np.linalg.norm(flow_cpn - gt_target, axis=-1)
        gt_site_error = np.linalg.norm(flow_gt - gt_target, axis=-1)
        pos_gap = np.linalg.norm(cur_cpn - cur_gt, axis=-1)
        flow_gap = np.linalg.norm(flow_cpn - flow_gt, axis=-1)
        stats = patch_stats(flow_raw, cur_cpn, args.patch_radius)

        candidate_mask = (
            (pos_gap >= args.min_2d_gap)
            & (flow_gap >= args.min_flow_gap)
            & ((center_error - gt_site_error) >= args.min_center_badness)
        )
        if not np.any(candidate_mask):
            continue
        candidate_joint_count += int(candidate_mask.sum())

        learned = compute_learned_sampling(probe, keypoints_2d, ref, flow_model, device)
        selected_frame_count += 1

        for block in learned:
            pos_norm = block["pos"].reshape(17, args.num_heads, args.num_samples, 2)
            xy = norm_to_xy(pos_norm, args.width, args.height)
            weights = block["weights"]
            sampled_flow = bilinear_sample_flow(flow_raw, xy)
            sampled_error = np.linalg.norm(sampled_flow - gt_target[:, None, None, :], axis=-1)
            sampled_dist_to_gt_flow = np.linalg.norm(sampled_flow - flow_gt[:, None, None, :], axis=-1)
            sampled_dist_to_cpn_flow = np.linalg.norm(sampled_flow - flow_cpn[:, None, None, :], axis=-1)

            weighted_flow_per_head = (weights[..., None] * sampled_flow).sum(axis=2)
            weighted_flow = weighted_flow_per_head.mean(axis=1)
            weighted_error = np.linalg.norm(weighted_flow - gt_target, axis=-1)
            weighted_xy_per_head = (weights[..., None] * xy).sum(axis=2)
            weighted_xy = weighted_xy_per_head.mean(axis=1)
            weighted_pos_error = np.linalg.norm(weighted_xy - cur_gt, axis=-1)

            cpn_to_gt = cur_gt - cur_cpn
            unit = cpn_to_gt / np.maximum(np.linalg.norm(cpn_to_gt, axis=-1, keepdims=True), 1e-6)
            toward_gt = ((weighted_xy - cur_cpn) * unit).sum(axis=-1) / np.maximum(pos_gap, 1e-6)

            best_error = sampled_error.reshape(17, -1).min(axis=1)
            best_pos_error = np.linalg.norm(xy - cur_gt[:, None, None, :], axis=-1).reshape(17, -1).min(axis=1)
            entropy = normalized_entropy(weights).mean(axis=1)

            gt_region = (sampled_dist_to_gt_flow + args.good_margin) < sampled_dist_to_cpn_flow
            center_region = (sampled_dist_to_cpn_flow + args.good_margin) < sampled_dist_to_gt_flow
            gt_region_mass = (weights * gt_region.astype(np.float32)).sum(axis=2).mean(axis=1)
            center_region_mass = (weights * center_region.astype(np.float32)).sum(axis=2).mean(axis=1)
            top_idx = weights.argmax(axis=2)
            top1_gt = np.take_along_axis(gt_region, top_idx[..., None], axis=2).squeeze(-1).mean(axis=1)

            for joint_idx in np.where(candidate_mask)[0]:
                row = {
                    "seq": seq,
                    "subject": subject,
                    "action": action,
                    "action_name": action_name(action),
                    "subaction": subaction,
                    "camera_id": camera_id,
                    "frame_id": frame_id,
                    "block": int(block["block"]),
                    "joint": int(joint_idx),
                    "joint_name": joint_name(joint_idx),
                    "cpn_gt_pos_gap_px": float(pos_gap[joint_idx]),
                    "cpn_gt_flow_gap_px": float(flow_gap[joint_idx]),
                    "center_flow_error_px": float(center_error[joint_idx]),
                    "gt_site_flow_error_px": float(gt_site_error[joint_idx]),
                    "weighted_flow_error_px": float(weighted_error[joint_idx]),
                    "best_sample_flow_error_px": float(best_error[joint_idx]),
                    "weighted_flow_gain_px": float(center_error[joint_idx] - weighted_error[joint_idx]),
                    "best_sample_gain_px": float(center_error[joint_idx] - best_error[joint_idx]),
                    "weighted_pos_error_px": float(weighted_pos_error[joint_idx]),
                    "best_pos_error_px": float(best_pos_error[joint_idx]),
                    "weighted_pos_gain_px": float(pos_gap[joint_idx] - weighted_pos_error[joint_idx]),
                    "toward_gt_ratio": float(toward_gt[joint_idx]),
                    "gt_region_mass": float(gt_region_mass[joint_idx]),
                    "center_region_mass": float(center_region_mass[joint_idx]),
                    "top1_gt_region_frac": float(top1_gt[joint_idx]),
                    "weight_entropy": float(entropy[joint_idx]),
                    "patch_flow_var": float(stats[joint_idx, 2]),
                    "flow_edge": float(stats[joint_idx, 3]),
                    "center_flow_x": float(flow_cpn[joint_idx, 0]),
                    "center_flow_y": float(flow_cpn[joint_idx, 1]),
                    "gt_flow_x": float(flow_gt[joint_idx, 0]),
                    "gt_flow_y": float(flow_gt[joint_idx, 1]),
                    "target_flow_x": float(gt_target[joint_idx, 0]),
                    "target_flow_y": float(gt_target[joint_idx, 1]),
                    "weighted_xy_x": float(weighted_xy[joint_idx, 0]),
                    "weighted_xy_y": float(weighted_xy[joint_idx, 1]),
                }
                rows.append(row)

    summary = summarize_rows(rows) if rows else {"by_block": {}, "top_success": [], "top_failure": [], "top_oracle_opportunity": []}
    out = {
        "meta": {
            "mode": "clean_cross_motion_region_weight_test",
            "labels": str(labels_path),
            "flow_dir": str(flow_dir),
            "image_root": str(image_root),
            "checkpoint": checkpoint_path,
            "flow_clip": args.flow_clip,
            "flow_norm": args.flow_norm,
            "frame_gap": args.frame_gap,
            "width": args.width,
            "height": args.height,
            "max_frames": args.max_frames,
            "scanned_frame_count": scanned_frame_count,
            "selected_frame_count": selected_frame_count,
            "candidate_joint_count_before_blocks": candidate_joint_count,
            "num_rows_after_blocks": len(rows),
            "missing_prev": missing_prev,
            "missing_flow": missing_flow,
            "unexpected_loaded_keys_ignored": unexpected_count,
            "filters": {
                "min_2d_gap": args.min_2d_gap,
                "min_flow_gap": args.min_flow_gap,
                "min_center_badness": args.min_center_badness,
                "good_margin": args.good_margin,
            },
        },
        "summary": summary,
    }

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(out, file, indent=2)

    if rows and not args.no_vis:
        by_key = {
            (
                int(shot["subject"]),
                int(shot["action"]),
                int(shot["subaction"]),
                int(shot["camera_id"]),
                int(shot["image_id"]),
            ): shot
            for shot in labels
        }
        vis_dir = Path(args.vis_dir).expanduser().resolve()
        selected = []
        selected.extend(("success", row) for row in summary["top_success"][: args.top_k])
        selected.extend(("failure", row) for row in summary["top_failure"][: args.top_k])
        selected.extend(("oracle", row) for row in summary["top_oracle_opportunity"][: args.top_k])
        seen = set()
        visual_cases = []
        for kind, row in selected:
            case_key = (
                row["subject"],
                row["action"],
                row["subaction"],
                row["camera_id"],
                row["frame_id"],
                row["block"],
                row["joint"],
                kind,
            )
            if case_key in seen:
                continue
            seen.add(case_key)
            cur = by_key[(row["subject"], row["action"], row["subaction"], row["camera_id"], row["frame_id"])]
            flow_path = flow_dir / row["seq"] / (image_stem(row["seq"], row["frame_id"]) + ".npy")
            flow_raw = np.load(flow_path).astype(np.float32)
            flow_model = normalize_flow_for_model(flow_raw, args.flow_clip, args.flow_norm)
            cur_cpn = np.asarray(cur["joints_2d_cpn_crop"], dtype=np.float32)
            keypoints_2d = np.asarray(cur["joints_2d_cpn"], dtype=np.float32)
            ref = ref_from_crop_xy(cur_cpn, args.width, args.height)
            learned = compute_learned_sampling(probe, keypoints_2d, ref, flow_model, device)
            block = learned[row["block"]]
            xy = norm_to_xy(block["pos"].reshape(17, args.num_heads, args.num_samples, 2), args.width, args.height)[row["joint"]]
            weights = block["weights"][row["joint"]]
            sampled_flow = bilinear_sample_flow(flow_raw, xy)
            target = np.array([row["target_flow_x"], row["target_flow_y"]], dtype=np.float32)
            sampled_err = np.linalg.norm(sampled_flow - target[None, None, :], axis=-1)
            filename = "{:02d}_{}_b{}_s{:02d}_a{:02d}_sub{}_ca{}_f{}_j{}_{}.png".format(
                len(visual_cases),
                kind,
                row["block"],
                row["subject"],
                row["action"],
                row["subaction"],
                row["camera_id"] + 1,
                row["frame_id"],
                row["joint"],
                row["joint_name"],
            )
            visualize_case(
                {"row": row, "cur": cur, "flow": flow_raw, "xy": xy, "weights": weights, "sampled_err": sampled_err},
                vis_dir / filename,
                image_root,
                args.width,
                args.height,
            )
            item = dict(row)
            item["kind"] = kind
            item["file"] = str(vis_dir / filename)
            visual_cases.append(item)
        out["meta"]["vis_dir"] = str(vis_dir)
        out["visual_cases"] = visual_cases
        with open(out_path, "w", encoding="utf-8") as file:
            json.dump(out, file, indent=2)

    print("Wrote {}".format(out_path))
    print(json.dumps(out["meta"], indent=2))
    for block, values in out["summary"]["by_block"].items():
        print("block {}".format(block))
        print(
            json.dumps(
                {
                    "num_cases": values["num_cases"],
                    "weighted_flow_gain_px": values["weighted_flow_gain_px"],
                    "best_sample_gain_px": values["best_sample_gain_px"],
                    "gt_region_mass": values["gt_region_mass"],
                    "center_region_mass": values["center_region_mass"],
                    "frac_weighted_flow_improves": values["frac_weighted_flow_improves"],
                    "frac_gt_mass_exceeds_center_mass": values["frac_gt_mass_exceeds_center_mass"],
                    "weight_entropy": values["weight_entropy"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
