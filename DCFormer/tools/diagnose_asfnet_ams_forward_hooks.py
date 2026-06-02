import argparse
import json
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

from diagnose_flow_mfce_sampling import image_stem, seq_name, summarize
from diagnose_learned_flow_dce_sampling import (
    PARENTS,
    action_name,
    draw_skeleton,
    entropy,
    joint_name,
    norm_to_xy,
    resolve_checkpoint_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Hard-validate ASFNet/CAPF-style AMS sampling by running the full "
            "DepthGuidedPose forward and collecting actual RGBD_Extraction pos/weights."
        )
    )
    parser.add_argument("--config", default="experiments/human36m/human36m_single.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--labels", default="")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--depth-root", default="")
    parser.add_argument("--depth-format", default="")
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-level", type=int, default=4, help="AMS level to summarize; ASFNet depth level is 4.")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--min-cpn-gt-error", type=float, default=6.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-mismatch", action="store_true")
    parser.add_argument("--out", default="debug_vis/asfnet_ams_forward_hook_sampling.json")
    parser.add_argument("--vis-dir", default="debug_vis/asfnet_ams_forward_hook_sampling_vis")
    parser.add_argument("--no-vis", action="store_true")
    return parser.parse_args()


def load_labels(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def frame_id(row):
    if "frame_id" in row:
        return int(row["frame_id"])
    return int(row["image_id"])


def image_rel_path(row):
    seq = seq_name(row)
    return str(Path(seq) / (image_stem(seq, frame_id(row)) + ".jpg"))


def make_key(row):
    return (
        int(row["subject"]),
        int(row["action"]),
        int(row["subaction"]),
        int(row["camera_id"]),
        frame_id(row),
    )


def cfg_get(node, key, default=None):
    try:
        if key in node:
            return node[key]
    except TypeError:
        pass
    return getattr(node, key, default)


def resolve_device(device_arg):
    import torch

    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested {}, but CUDA is not available.".format(device_arg))
    return device


def build_model(config_path, checkpoint_path, device, allow_mismatch=False):
    import torch

    from mvn.models.DGPose import DepthGuidedPose
    from mvn.utils.cfg import config, update_config

    update_config(config_path)
    if config.model.name != "DepthGuidedPose":
        raise ValueError("Expected DepthGuidedPose config, got {}".format(config.model.name))
    model = DepthGuidedPose(config, device)

    if cfg_get(config.model.backbone, "init_weights", False):
        ret = model.backbone.load_state_dict(
            torch.load(config.model.backbone.checkpoint, map_location="cpu"),
            strict=False,
        )
        print("Loaded backbone {}: {}".format(config.model.backbone.checkpoint, ret))

    ckpt_path = resolve_checkpoint_path(checkpoint_path)
    raw = torch.load(str(ckpt_path), map_location="cpu")
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    state = {key.replace("module.", ""): value for key, value in state.items()}
    ret = model.load_state_dict(state, strict=False)
    print("Loaded checkpoint {}: {}".format(ckpt_path, ret))
    if not allow_mismatch and (ret.missing_keys or ret.unexpected_keys):
        raise RuntimeError(
            "Checkpoint is not compatible with the current ASFNet model. "
            "Missing keys: {} Unexpected keys: {}. "
            "Use a matching checkpoint or pass --allow-mismatch only for debugging.".format(
                ret.missing_keys[:20],
                ret.unexpected_keys[:20],
            )
        )
    model.to(device)
    model.eval()
    return model, config, str(ckpt_path)


def register_ams_hooks(model, captures):
    import torch
    import torch.nn.functional as F

    handles = []
    blocks = model.Lifting_net.RGBD_Extraction

    def make_hook(block_idx):
        def hook(module, inputs, _output):
            x, ref, _features_list = inputs
            with torch.no_grad():
                x_0, x_rest = x[:, :1], x[:, 1:]
                b, l, p, _c = x_rest.shape
                normed = module.norm1(x_rest + x_0)
                weights = module.attention_weights(normed).view(
                    b, l, p, module.num_heads, module.num_samples
                )
                weights = F.softmax(weights, dim=-1)
                offsets = module.sampling_offsets(normed).reshape(
                    b, l, p, module.num_heads * module.num_samples, 2
                ).tanh()
                pos = offsets + ref.view(b, 1, p, 1, 2)
                captures[block_idx] = {
                    "pos": pos.detach().cpu().numpy(),
                    "weights": weights.detach().cpu().numpy(),
                    "offsets": offsets.detach().cpu().numpy(),
                    "num_heads": int(module.num_heads),
                    "num_samples": int(module.num_samples),
                    "num_levels": int(l),
                }

        return hook

    for idx, block in enumerate(blocks):
        handles.append(block.register_forward_hook(make_hook(idx)))
    return handles


def read_rgb(image_root, row):
    path = Path(image_root) / image_rel_path(row)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise FileNotFoundError("Failed to read image: {}".format(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (image - mean) / std


def read_depth(depth_root, row, depth_format):
    rel_path = image_rel_path(row)
    if depth_format == "npy":
        path = Path(depth_root) / Path(rel_path).with_suffix(".npy")
        depth = np.load(path).astype(np.float32, copy=False)
        return depth

    path = Path(depth_root) / rel_path
    depth = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
    if depth is None:
        raise FileNotFoundError("Failed to read depth: {}".format(path))
    return depth


def prepare_records(labels, image_root, depth_root, depth_format, max_frames, seed):
    keys = list(range(len(labels)))
    rng = random.Random(seed)
    rng.shuffle(keys)

    records = []
    missing_image = 0
    missing_depth = 0
    for idx in keys:
        if len(records) >= max_frames:
            break
        row = labels[idx]
        image_path = Path(image_root) / image_rel_path(row)
        if not image_path.exists():
            missing_image += 1
            continue
        if depth_format == "npy":
            depth_path = Path(depth_root) / Path(image_rel_path(row)).with_suffix(".npy")
        else:
            depth_path = Path(depth_root) / image_rel_path(row)
        if not depth_path.exists():
            missing_depth += 1
            continue
        records.append(row)

    return records, {"missing_image": missing_image, "missing_depth": missing_depth}


def batch_iter(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def summarize_rows(rows):
    def metric_from(block_rows, key):
        return summarize([row[key] for row in block_rows])

    def fraction(block_rows, fn):
        if not block_rows:
            return None
        return float(np.mean([bool(fn(row)) for row in block_rows]))

    by_block = {}
    for block_idx in sorted({row["block"] for row in rows}):
        block_rows = [row for row in rows if row["block"] == block_idx]
        by_block[str(block_idx)] = {
            "offset_mag_px": metric_from(block_rows, "offset_mag_px"),
            "weight_entropy": metric_from(block_rows, "weight_entropy"),
            "center_pos_error_px": metric_from(block_rows, "center_pos_error_px"),
            "weighted_pos_error_px": metric_from(block_rows, "weighted_pos_error_px"),
            "best_pos_error_px": metric_from(block_rows, "best_pos_error_px"),
            "weighted_pos_minus_center_px": metric_from(block_rows, "weighted_pos_minus_center_px"),
            "best_pos_minus_center_px": metric_from(block_rows, "best_pos_minus_center_px"),
            "frac_weighted_pos_closer_to_gt": fraction(
                block_rows, lambda row: row["weighted_pos_error_px"] < row["center_pos_error_px"]
            ),
            "frac_best_pos_closer_to_gt": fraction(
                block_rows, lambda row: row["best_pos_error_px"] < row["center_pos_error_px"]
            ),
        }

    return {
        "by_block": by_block,
        "top_position_moves_closer": sorted(rows, key=lambda row: row["weighted_pos_minus_center_px"])[:30],
        "top_position_moves_away": sorted(rows, key=lambda row: row["weighted_pos_minus_center_px"], reverse=True)[:30],
        "top_best_position_opportunity": sorted(rows, key=lambda row: row["best_pos_minus_center_px"])[:30],
    }


def visualize_depth_case(case, output_path, image_root, depth_root, depth_format, width, height):
    row = case["row"]
    cur = case["cur"]
    xy = case["xy"]
    weights = case["weights"]
    joint_idx = row["joint"]
    rgb = read_rgb(image_root, cur)
    rgb = np.clip(rgb * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
    depth = read_depth(depth_root, cur, depth_format).astype(np.float32)
    cur_cpn = np.asarray(cur["joints_2d_cpn_crop"], dtype=np.float32)
    cur_gt = np.asarray(cur["joints_2d_gt_crop"], dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(rgb)
    draw_skeleton(axes[0], cur_cpn, "#5976ff", linewidth=1.2)
    draw_skeleton(axes[0], cur_gt, "#ff4040", linewidth=1.2)
    axes[0].scatter(cur_gt[joint_idx, 0], cur_gt[joint_idx, 1], marker="D", c="#ff2020", s=55, label="GT")
    axes[0].scatter(cur_cpn[joint_idx, 0], cur_cpn[joint_idx, 1], marker="D", c="#5976ff", s=55, label="CPN")
    axes[0].set_title("RGB: {} {} frame {}".format(row["joint_name"], row["action_name"], row["frame_id"]))
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].axis("off")

    ax = axes[1]
    ax.imshow(depth, cmap="magma")
    flat_xy = xy.reshape(-1, 2)
    flat_w = weights.reshape(-1)
    size = 30 + 240 * (flat_w / (flat_w.max() + 1e-8))
    ax.scatter(flat_xy[:, 0], flat_xy[:, 1], s=size, c="#64d86b", edgecolor="white", linewidth=0.6, alpha=0.85, label="learned samples")
    ax.scatter(cur_gt[joint_idx, 0], cur_gt[joint_idx, 1], marker="D", c="#ff2020", s=65, label="GT")
    ax.scatter(cur_cpn[joint_idx, 0], cur_cpn[joint_idx, 1], marker="D", c="#5976ff", s=65, label="CPN")
    ax.set_xlim(0, width - 1)
    ax.set_ylim(height - 1, 0)
    ax.set_title(
        "block {} level {} | 2D gap {:.2f}px | weighted/best pos delta {:.2f}/{:.2f}px".format(
            row["block"],
            row["level"],
            row["center_pos_error_px"],
            row["weighted_pos_minus_center_px"],
            row["best_pos_minus_center_px"],
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
    import torch

    device = resolve_device(args.device)
    model, config, checkpoint_path = build_model(args.config, args.checkpoint, device, args.allow_mismatch)
    labels_path = Path(args.labels or config.dataset.val_labels_path).expanduser()
    image_root = Path(args.image_root or config.dataset.root).expanduser()
    depth_root = Path(args.depth_root or config.dataset.depth_image_path).expanduser()
    depth_format = args.depth_format or cfg_get(config.dataset, "depth_format", "image")

    labels = load_labels(labels_path)
    records, missing = prepare_records(labels, image_root, depth_root, depth_format, args.max_frames, args.seed)
    if not records:
        raise RuntimeError("No valid records found.")

    captures = {}
    handles = register_ams_hooks(model, captures)
    rows = []
    cases = []

    try:
        with torch.no_grad():
            for batch in batch_iter(records, args.batch_size):
                captures.clear()
                images = []
                keypoints_2d = []
                keypoints_2d_crop = []
                depth_images = []
                raw_depths = []

                for cur in batch:
                    depth = read_depth(depth_root, cur, depth_format)
                    raw_depths.append(depth)
                    if np.issubdtype(depth.dtype, np.floating):
                        depth_model = depth.astype(np.float32, copy=False)
                    else:
                        depth_model = depth.astype(np.float32) / 255.0
                    images.append(read_rgb(image_root, cur))
                    keypoints_2d.append(np.asarray(cur["joints_2d_cpn"], dtype=np.float32))
                    keypoints_2d_crop.append(np.asarray(cur["joints_2d_cpn_crop"], dtype=np.float32))
                    depth_images.append(depth_model)

                images_t = torch.from_numpy(np.stack(images)).float().to(device)
                keypoints_t = torch.from_numpy(np.stack(keypoints_2d)).float().to(device)
                keypoints_crop_t = torch.from_numpy(np.stack(keypoints_2d_crop)).float().to(device)
                depth_t = torch.from_numpy(np.stack(depth_images)).float().to(device)

                _pred = model(images_t, keypoints_t, keypoints_crop_t.clone(), depth_t)

                if not captures:
                    raise RuntimeError("Forward hooks did not capture RGBD_Extraction blocks.")

                for sample_idx, cur in enumerate(batch):
                    cur_cpn = np.asarray(cur["joints_2d_cpn_crop"], dtype=np.float32)
                    cur_gt = np.asarray(cur["joints_2d_gt_crop"], dtype=np.float32)
                    center_pos_error = np.linalg.norm(cur_cpn - cur_gt, axis=-1)

                    for block_idx, block in captures.items():
                        level = min(args.target_level, block["num_levels"] - 1)
                        num_heads = block["num_heads"]
                        num_samples = block["num_samples"]
                        pos_norm = block["pos"][sample_idx, level].reshape(17, num_heads, num_samples, 2)
                        xy = norm_to_xy(pos_norm, args.width, args.height)
                        weights = block["weights"][sample_idx, level]
                        pos_err = np.linalg.norm(xy - cur_gt[:, None, None, :], axis=-1)
                        weighted_xy = (weights[..., None] * xy).sum(axis=2).mean(axis=1)
                        weighted_pos_error = np.linalg.norm(weighted_xy - cur_gt, axis=-1)
                        best_pos_error = pos_err.reshape(17, -1).min(axis=1)
                        offset_px = xy - cur_cpn[:, None, None, :]
                        offset_mag = np.linalg.norm(offset_px, axis=-1)
                        weight_entropy = entropy(weights)

                        for joint_idx in range(17):
                            row = {
                                "seq": seq_name(cur),
                                "subject": int(cur["subject"]),
                                "action": int(cur["action"]),
                                "action_name": action_name(cur["action"]),
                                "subaction": int(cur["subaction"]),
                                "camera_id": int(cur["camera_id"]),
                                "frame_id": frame_id(cur),
                                "block": int(block_idx),
                                "level": int(level),
                                "joint": int(joint_idx),
                                "joint_name": joint_name(joint_idx),
                                "center_pos_error_px": float(center_pos_error[joint_idx]),
                                "weighted_pos_error_px": float(weighted_pos_error[joint_idx]),
                                "best_pos_error_px": float(best_pos_error[joint_idx]),
                                "weighted_pos_minus_center_px": float(weighted_pos_error[joint_idx] - center_pos_error[joint_idx]),
                                "best_pos_minus_center_px": float(best_pos_error[joint_idx] - center_pos_error[joint_idx]),
                                "offset_mag_px": float(offset_mag[joint_idx].mean()),
                                "offset_mag_p90_px": float(np.percentile(offset_mag[joint_idx], 90)),
                                "weight_entropy": float(weight_entropy[joint_idx].mean()),
                            }
                            rows.append(row)
                            if (
                                len(cases) < args.top_k * 20
                                and row["center_pos_error_px"] >= args.min_cpn_gt_error
                                and row["best_pos_error_px"] < row["center_pos_error_px"]
                            ):
                                cases.append({"row": row, "cur": cur, "xy": xy[joint_idx], "weights": weights[joint_idx]})
    finally:
        for handle in handles:
            handle.remove()

    out = {
        "meta": {
            "mode": "asfnet_full_forward_hooks",
            "config": str(Path(args.config).expanduser()),
            "labels": str(labels_path),
            "image_root": str(image_root),
            "depth_root": str(depth_root),
            "depth_format": depth_format,
            "checkpoint": checkpoint_path,
            "width": args.width,
            "height": args.height,
            "target_level": args.target_level,
            "max_frames": args.max_frames,
            "num_frame_samples": len(records),
            "num_joint_block_rows": len(rows),
            **missing,
        },
        "summary": summarize_rows(rows),
    }

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.no_vis and cases:
        vis_dir = Path(args.vis_dir).expanduser().resolve()
        selected = sorted(cases, key=lambda item: item["row"]["weighted_pos_minus_center_px"])[: args.top_k]
        out["visual_cases"] = [case["row"] for case in selected]
        out["meta"]["vis_dir"] = str(vis_dir)
        for idx, case in enumerate(selected):
            row = case["row"]
            name = "{:02d}_b{}_lvl{}_s{:02d}_a{:02d}_sub{}_ca{}_f{}_j{}_{}.png".format(
                idx,
                row["block"],
                row["level"],
                row["subject"],
                row["action"],
                row["subaction"],
                row["camera_id"] + 1,
                row["frame_id"],
                row["joint"],
                row["joint_name"],
            )
            visualize_depth_case(case, vis_dir / name, image_root, depth_root, depth_format, args.width, args.height)

    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(out, file, indent=2)

    print("Wrote {}".format(out_path))
    print(json.dumps(out["meta"], indent=2))
    for block_idx, block_summary in out["summary"]["by_block"].items():
        print("block {}".format(block_idx))
        print(
            json.dumps(
                {
                    "offset_mag_px": block_summary["offset_mag_px"],
                    "weight_entropy": block_summary["weight_entropy"],
                    "weighted_pos_minus_center_px": block_summary["weighted_pos_minus_center_px"],
                    "best_pos_minus_center_px": block_summary["best_pos_minus_center_px"],
                    "frac_weighted_pos_closer_to_gt": block_summary["frac_weighted_pos_closer_to_gt"],
                    "frac_best_pos_closer_to_gt": block_summary["frac_best_pos_closer_to_gt"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
