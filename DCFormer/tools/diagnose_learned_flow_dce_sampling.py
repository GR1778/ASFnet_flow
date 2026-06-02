import argparse
import json
import math
import pickle
import random
import sys
from collections import defaultdict
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
    corr,
    get_affine_transform,
    image_stem,
    patch_stats,
    seq_name,
    summarize,
)


PARENTS = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])
LEFT_JOINTS = {4, 5, 6, 11, 12, 13}
RIGHT_JOINTS = {1, 2, 3, 14, 15, 16}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Probe the learned sampling points and weights of the trained "
            "Flow-DCE/MFCE branch. The script measures whether learned samples "
            "move from noisy 2D detections toward the GT joint motion region."
        )
    )
    parser.add_argument("--labels", default="data/h36m_validation.pkl")
    parser.add_argument("--flow-dir", default="../H36M-Toolbox/flow_images_float")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-root", default="", help="Optional cropped RGB root for hard-case visualization.")
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
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--min-cpn-gt-error", type=float, default=6.0)
    parser.add_argument("--out", default="debug_vis/learned_flow_dce_sampling.json")
    parser.add_argument("--vis-dir", default="debug_vis/learned_flow_dce_sampling_vis")
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


def strip_prefix(state):
    prefixes = ["module.Lifting_net.", "Lifting_net."]
    out = {}
    for key, value in state.items():
        for prefix in prefixes:
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
    state = strip_prefix(state)
    probe = FlowProbe()
    missing, unexpected = probe.load_state_dict(state, strict=False)
    critical_missing = [
        key
        for key in missing
        if key.startswith(("coord_embed", "motion_field_embed", "flow_feat_embed", "RGB_pos_embed", "Flow_pos_embed", "Flow_Extraction"))
    ]
    if critical_missing:
        raise KeyError("Missing flow probe keys: {}".format(critical_missing[:20]))
    probe.eval()
    return probe, str(checkpoint_path), unexpected


def normalize_flow_for_model(flow_raw, flow_clip, flow_norm):
    flow = flow_raw.astype(np.float32, copy=True)
    if flow_clip is not None and flow_clip > 0:
        flow = np.clip(flow, -flow_clip, flow_clip)
    if flow_norm is not None and flow_norm > 0:
        flow = flow / flow_norm
    return flow


def normalize_target_for_model(target, flow_clip, flow_norm):
    target = target.astype(np.float32, copy=True)
    if flow_clip is not None and flow_clip > 0:
        target = np.clip(target, -flow_clip, flow_clip)
    if flow_norm is not None and flow_norm > 0:
        target = target / flow_norm
    return target


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


def entropy(weights, eps=1e-8):
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
            b, l, p, c = x_rest.shape
            residual = x_rest
            normed = block.norm1(x_rest + x_0)
            weights = block.attention_weights(normed).view(b, l, p, block.num_heads, block.num_samples)
            weights = F.softmax(weights, dim=-1)
            offsets = block.sampling_offsets(normed).reshape(b, l, p, block.num_heads * block.num_samples, 2).tanh()
            pos = offsets + ref_t.view(b, 1, p, 1, -1)

            sampled = F.grid_sample(motion_field, pos[:, 0], padding_mode="border", align_corners=True)
            sampled = sampled.permute(0, 2, 3, 1).contiguous()
            sampled = block.embed_proj[0](sampled)
            sampled = sampled.view(b, p, block.num_heads, block.num_samples, -1)
            features_sampled = (weights[:, 0].unsqueeze(-1) * sampled).sum(dim=-2).view(b, 1, p, -1)

            x_rest = residual + block.drop_path(features_sampled)
            x_rest = x_rest + block.drop_path(block.mlp(block.norm2(x_rest)))
            x = torch.cat([x_0, x_rest], dim=1)

            block_outputs.append(
                {
                    "block": block_idx,
                    "pos": pos[0, 0].detach().cpu().numpy(),
                    "offsets": offsets[0, 0].detach().cpu().numpy(),
                    "weights": weights[0, 0].detach().cpu().numpy(),
                }
            )

    return block_outputs


def summarize_rows(rows):
    def metric(name):
        return summarize([row[name] for row in rows])

    by_block = {}
    for block_idx in sorted({row["block"] for row in rows}):
        block_rows = [row for row in rows if row["block"] == block_idx]
        by_block[str(block_idx)] = {
            "offset_mag_px": metric_from(block_rows, "offset_mag_px"),
            "weight_entropy": metric_from(block_rows, "weight_entropy"),
            "center_flow_error_px": metric_from(block_rows, "center_flow_error_px"),
            "weighted_flow_error_px": metric_from(block_rows, "weighted_flow_error_px"),
            "best_sample_flow_error_px": metric_from(block_rows, "best_sample_flow_error_px"),
            "weighted_flow_minus_center_px": metric_from(block_rows, "weighted_flow_minus_center_px"),
            "best_sample_minus_center_px": metric_from(block_rows, "best_sample_minus_center_px"),
            "center_pos_error_px": metric_from(block_rows, "center_pos_error_px"),
            "weighted_pos_error_px": metric_from(block_rows, "weighted_pos_error_px"),
            "best_pos_error_px": metric_from(block_rows, "best_pos_error_px"),
            "weighted_pos_minus_center_px": metric_from(block_rows, "weighted_pos_minus_center_px"),
            "best_pos_minus_center_px": metric_from(block_rows, "best_pos_minus_center_px"),
            "frac_weighted_flow_better_center": fraction(block_rows, lambda row: row["weighted_flow_error_px"] < row["center_flow_error_px"]),
            "frac_best_flow_better_center": fraction(block_rows, lambda row: row["best_sample_flow_error_px"] < row["center_flow_error_px"]),
            "frac_weighted_pos_closer_to_gt": fraction(block_rows, lambda row: row["weighted_pos_error_px"] < row["center_pos_error_px"]),
            "frac_best_pos_closer_to_gt": fraction(block_rows, lambda row: row["best_pos_error_px"] < row["center_pos_error_px"]),
            "corr_cpn_gt_with_weighted_flow_gain": corr(
                [row["center_pos_error_px"] for row in block_rows],
                [-row["weighted_flow_minus_center_px"] for row in block_rows],
            ),
            "corr_patch_var_with_weighted_flow_delta": corr(
                [row["patch_flow_var"] for row in block_rows],
                [row["weighted_flow_minus_center_px"] for row in block_rows],
            ),
            "worst_joints_by_weighted_flow_delta": grouped_summary(block_rows, "joint", "weighted_flow_minus_center_px"),
            "best_joints_by_best_flow_opportunity": grouped_summary(block_rows, "joint", "best_sample_minus_center_px", reverse=False),
        }

    return {
        "by_block": by_block,
        "top_opportunity_not_used": sorted(
            rows,
            key=lambda row: row["weighted_flow_minus_center_px"] - row["best_sample_minus_center_px"],
            reverse=True,
        )[:30],
        "top_weighted_flow_degradation": sorted(rows, key=lambda row: row["weighted_flow_minus_center_px"], reverse=True)[:30],
        "top_position_moves_away": sorted(rows, key=lambda row: row["weighted_pos_minus_center_px"], reverse=True)[:30],
    }


def metric_from(rows, key):
    return summarize([row[key] for row in rows])


def fraction(rows, fn):
    if not rows:
        return None
    return float(np.mean([bool(fn(row)) for row in rows]))


def grouped_summary(rows, group_key, metric_key, reverse=True, top_k=12):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[group_key]].append(row[metric_key])
    items = []
    for group, values in grouped.items():
        item = {group_key: group, **summarize(values)}
        if group_key == "joint":
            item["name"] = joint_name(group)
        elif group_key == "action":
            item["name"] = action_name(group)
        items.append(item)
    return sorted(items, key=lambda item: float("-inf") if item["mean"] is None else item["mean"], reverse=reverse)[:top_k]


def flow_to_rgb(flow):
    flow = np.asarray(flow, dtype=np.float32)
    fx, fy = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)
    hsv[..., 1] = np.clip(mag / (np.percentile(mag, 99) + 1e-6) * 255, 0, 255).astype(np.uint8)
    hsv[..., 2] = 255
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb[mag < max(0.05, np.percentile(mag, 30))] = 255
    return rgb


def image_rel_path(row):
    seq = "s_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}".format(
        int(row["subject"]),
        int(row["action"]),
        int(row["subaction"]),
        int(row["camera_id"]) + 1,
    )
    name = "{}_{:06d}.jpg".format(seq, int(row["frame_id"]))
    return str(Path(seq) / name)


def read_optional_image(image_root, rel_path):
    if not image_root:
        return None, None
    path = Path(image_root) / rel_path
    if not path.exists():
        return None, path
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None, path
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), path


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


def visualize_case(case, output_path, image_root, width, height):
    row = case["row"]
    cur = case["cur"]
    flow = case["flow"]
    xy = case["xy"]
    weights = case["weights"]
    joint_idx = row["joint"]

    image, image_path = read_optional_image(image_root, image_rel_path(row))
    flow_rgb = flow_to_rgb(flow)
    cur_cpn = np.asarray(cur["joints_2d_cpn_crop"], dtype=np.float32)
    cur_gt = np.asarray(cur["joints_2d_gt_crop"], dtype=np.float32)

    fig, axes = plt.subplots(1, 2 if image is not None else 1, figsize=(12 if image is not None else 6, 6))
    if image is None:
        axes = [axes]
    else:
        plot_cpn = cur_cpn.copy()
        plot_gt = cur_gt.copy()
        image_h, image_w = image.shape[:2]
        shape_note = ""
        if (image_w, image_h) != (width, height):
            scale = np.array([image_w / float(width), image_h / float(height)], dtype=np.float32)
            plot_cpn = plot_cpn * scale
            plot_gt = plot_gt * scale
            shape_note = " | scaled {}x{}".format(image_w, image_h)
        axes[0].imshow(image)
        draw_skeleton(axes[0], plot_cpn, "#5976ff", linewidth=1.2)
        draw_skeleton(axes[0], plot_gt, "#ff4040", linewidth=1.2)
        axes[0].scatter(plot_gt[joint_idx, 0], plot_gt[joint_idx, 1], marker="D", c="#ff2020", s=55, label="GT")
        axes[0].scatter(plot_cpn[joint_idx, 0], plot_cpn[joint_idx, 1], marker="D", c="#5976ff", s=55, label="CPN")
        axes[0].set_title("RGB: {} {} frame {}{}".format(row["joint_name"], row["action_name"], row["frame_id"], shape_note))
        axes[0].legend(loc="lower right", fontsize=8)
        axes[0].axis("off")

    ax = axes[-1]
    ax.imshow(flow_rgb)
    flat_xy = xy.reshape(-1, 2)
    flat_w = weights.reshape(-1)
    size = 30 + 240 * (flat_w / (flat_w.max() + 1e-8))
    ax.scatter(flat_xy[:, 0], flat_xy[:, 1], s=size, c="#ff8c35", edgecolor="white", linewidth=0.6, alpha=0.85, label="learned samples")
    ax.scatter(cur_gt[joint_idx, 0], cur_gt[joint_idx, 1], marker="D", c="#ff2020", s=65, label="GT")
    ax.scatter(cur_cpn[joint_idx, 0], cur_cpn[joint_idx, 1], marker="D", c="#5976ff", s=65, label="CPN")
    ax.set_xlim(0, width - 1)
    ax.set_ylim(height - 1, 0)
    joint_gap = float(np.linalg.norm(cur_cpn[joint_idx] - cur_gt[joint_idx]))
    ax.set_title(
        "block {} | 2D gap {:.2f}px | flow err c/w/b {:.2f}/{:.2f}/{:.2f}px".format(
            row["block"],
            joint_gap,
            row["center_flow_error_px"],
            row["weighted_flow_error_px"],
            row["best_sample_flow_error_px"],
        )
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    labels_path = Path(args.labels).expanduser().resolve()
    flow_dir = Path(args.flow_dir).expanduser().resolve()
    if not flow_dir.exists():
        raise FileNotFoundError("Flow directory not found: {}".format(flow_dir))

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe, checkpoint_path, unexpected = load_flow_probe(
        args.checkpoint,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_samples=args.num_samples,
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
    cases = []
    frame_count = 0
    missing_prev = 0
    missing_flow = 0

    for key in keys:
        if frame_count >= args.max_frames:
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

        flow_raw = np.load(flow_path).astype(np.float32)
        if flow_raw.ndim != 3 or flow_raw.shape[-1] != 2:
            raise ValueError("Expected flow [H,W,2], got {} at {}".format(flow_raw.shape, flow_path))

        flow_model = normalize_flow_for_model(flow_raw, args.flow_clip, args.flow_norm)
        cur_cpn = np.asarray(cur["joints_2d_cpn_crop"], dtype=np.float32)
        cur_gt = np.asarray(cur["joints_2d_gt_crop"], dtype=np.float32)
        prev_gt = np.asarray(prev["joints_2d_gt_crop"], dtype=np.float32)
        keypoints_2d = np.asarray(cur["joints_2d_cpn"], dtype=np.float32)
        ref = ref_from_crop_xy(cur_cpn, args.width, args.height)

        prev_to_raw = get_affine_transform(prev["center"], prev["scale"], output_size, inv=True)
        raw_to_cur = get_affine_transform(cur["center"], cur["scale"], output_size, inv=False)
        prev_gt_same = affine_apply(raw_to_cur, affine_apply(prev_to_raw, prev_gt))
        gt_target_px = prev_gt_same - cur_gt
        gt_target_model = normalize_target_for_model(gt_target_px, args.flow_clip, args.flow_norm)
        center_flow_px = bilinear_sample_flow(flow_raw, cur_cpn)
        center_flow_model = bilinear_sample_flow(flow_model, cur_cpn)
        center_flow_error_px = np.linalg.norm(center_flow_px - gt_target_px, axis=-1)
        center_flow_error_model = np.linalg.norm(center_flow_model - gt_target_model, axis=-1)
        center_pos_error = np.linalg.norm(cur_cpn - cur_gt, axis=-1)
        stats = patch_stats(flow_raw, cur_cpn, args.patch_radius)

        learned = compute_learned_sampling(probe, keypoints_2d, ref, flow_model, device)
        for block in learned:
            pos_norm = block["pos"].reshape(17, args.num_heads, args.num_samples, 2)
            xy = norm_to_xy(pos_norm, args.width, args.height)
            weights = block["weights"]
            sampled_flow_px = bilinear_sample_flow(flow_raw, xy)
            sampled_flow_model = bilinear_sample_flow(flow_model, xy)
            flow_err_px = np.linalg.norm(sampled_flow_px - gt_target_px[:, None, None, :], axis=-1)
            flow_err_model = np.linalg.norm(sampled_flow_model - gt_target_model[:, None, None, :], axis=-1)
            pos_err = np.linalg.norm(xy - cur_gt[:, None, None, :], axis=-1)

            weighted_flow_px = (weights[..., None] * sampled_flow_px).sum(axis=2).mean(axis=1)
            weighted_flow_model = (weights[..., None] * sampled_flow_model).sum(axis=2).mean(axis=1)
            weighted_xy = (weights[..., None] * xy).sum(axis=2).mean(axis=1)

            weighted_flow_error_px = np.linalg.norm(weighted_flow_px - gt_target_px, axis=-1)
            weighted_flow_error_model = np.linalg.norm(weighted_flow_model - gt_target_model, axis=-1)
            weighted_pos_error = np.linalg.norm(weighted_xy - cur_gt, axis=-1)
            best_flow_error_px = flow_err_px.reshape(17, -1).min(axis=1)
            best_flow_error_model = flow_err_model.reshape(17, -1).min(axis=1)
            best_pos_error = pos_err.reshape(17, -1).min(axis=1)
            offset_px = xy - cur_cpn[:, None, None, :]
            offset_mag = np.linalg.norm(offset_px, axis=-1)
            weight_entropy = entropy(weights)

            for joint_idx in range(17):
                row = {
                    "seq": seq,
                    "subject": subject,
                    "action": action,
                    "action_name": action_name(action),
                    "subaction": subaction,
                    "camera_id": camera_id,
                    "frame_id": frame_id,
                    "block": int(block["block"]),
                    "joint": joint_idx,
                    "joint_name": joint_name(joint_idx),
                    "center_pos_error_px": float(center_pos_error[joint_idx]),
                    "weighted_pos_error_px": float(weighted_pos_error[joint_idx]),
                    "best_pos_error_px": float(best_pos_error[joint_idx]),
                    "weighted_pos_minus_center_px": float(weighted_pos_error[joint_idx] - center_pos_error[joint_idx]),
                    "best_pos_minus_center_px": float(best_pos_error[joint_idx] - center_pos_error[joint_idx]),
                    "center_flow_error_px": float(center_flow_error_px[joint_idx]),
                    "weighted_flow_error_px": float(weighted_flow_error_px[joint_idx]),
                    "best_sample_flow_error_px": float(best_flow_error_px[joint_idx]),
                    "weighted_flow_minus_center_px": float(weighted_flow_error_px[joint_idx] - center_flow_error_px[joint_idx]),
                    "best_sample_minus_center_px": float(best_flow_error_px[joint_idx] - center_flow_error_px[joint_idx]),
                    "center_flow_error_model": float(center_flow_error_model[joint_idx]),
                    "weighted_flow_error_model": float(weighted_flow_error_model[joint_idx]),
                    "best_sample_flow_error_model": float(best_flow_error_model[joint_idx]),
                    "offset_mag_px": float(offset_mag[joint_idx].mean()),
                    "offset_mag_p90_px": float(np.percentile(offset_mag[joint_idx], 90)),
                    "weight_entropy": float(weight_entropy[joint_idx].mean()),
                    "patch_mag_mean": float(stats[joint_idx, 0]),
                    "patch_mag_std": float(stats[joint_idx, 1]),
                    "patch_flow_var": float(stats[joint_idx, 2]),
                    "flow_edge": float(stats[joint_idx, 3]),
                }
                rows.append(row)
                if (
                    len(cases) < args.top_k * 20
                    and row["center_pos_error_px"] >= args.min_cpn_gt_error
                    and row["best_sample_flow_error_px"] < row["center_flow_error_px"]
                ):
                    cases.append({"row": row, "cur": cur, "flow": flow_raw, "xy": xy[joint_idx], "weights": weights[joint_idx]})

        frame_count += 1

    summary = summarize_rows(rows)
    out = {
        "meta": {
            "labels": str(labels_path),
            "flow_dir": str(flow_dir),
            "checkpoint": checkpoint_path,
            "flow_clip": args.flow_clip,
            "flow_norm": args.flow_norm,
            "frame_gap": args.frame_gap,
            "width": args.width,
            "height": args.height,
            "dim": args.dim,
            "depth": args.depth,
            "num_heads": args.num_heads,
            "num_samples": args.num_samples,
            "max_frames": args.max_frames,
            "num_frame_samples": frame_count,
            "num_joint_block_rows": len(rows),
            "missing_prev": missing_prev,
            "missing_flow": missing_flow,
            "unexpected_loaded_keys_ignored": len(unexpected),
        },
        "summary": summary,
    }

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(out, file, indent=2)

    if not args.no_vis and cases:
        vis_dir = Path(args.vis_dir).expanduser().resolve()
        selected = sorted(
            cases,
            key=lambda item: item["row"]["weighted_flow_minus_center_px"] - item["row"]["best_sample_minus_center_px"],
            reverse=True,
        )[: args.top_k]
        for idx, case in enumerate(selected):
            row = case["row"]
            name = "{:02d}_b{}_s{:02d}_a{:02d}_sub{}_ca{}_f{}_j{}_{}.png".format(
                idx,
                row["block"],
                row["subject"],
                row["action"],
                row["subaction"],
                row["camera_id"] + 1,
                row["frame_id"],
                row["joint"],
                row["joint_name"],
            )
            visualize_case(case, vis_dir / name, args.image_root, args.width, args.height)
        out["meta"]["vis_dir"] = str(vis_dir)
        out["visual_cases"] = [case["row"] for case in selected]
        with open(out_path, "w", encoding="utf-8") as file:
            json.dump(out, file, indent=2)

    print("Wrote {}".format(out_path))
    print(json.dumps(out["meta"], indent=2))
    for block_idx, block_summary in out["summary"]["by_block"].items():
        print("block {}".format(block_idx))
        print(
            json.dumps(
                {
                    "weighted_flow_minus_center_px": block_summary["weighted_flow_minus_center_px"],
                    "best_sample_minus_center_px": block_summary["best_sample_minus_center_px"],
                    "weighted_pos_minus_center_px": block_summary["weighted_pos_minus_center_px"],
                    "best_pos_minus_center_px": block_summary["best_pos_minus_center_px"],
                    "frac_weighted_flow_better_center": block_summary["frac_weighted_flow_better_center"],
                    "frac_best_flow_better_center": block_summary["frac_best_flow_better_center"],
                    "weight_entropy": block_summary["weight_entropy"],
                    "offset_mag_px": block_summary["offset_mag_px"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
