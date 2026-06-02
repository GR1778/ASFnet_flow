import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnose_flow_mfce_sampling import (
    affine_apply,
    bilinear_sample_flow,
    get_affine_transform,
    image_stem,
    patch_stats,
    seq_name,
    summarize,
)
from diagnose_learned_flow_dce_sampling import (
    action_name,
    entropy,
    joint_name,
    norm_to_xy,
    normalize_flow_for_model,
    resolve_checkpoint_path,
    visualize_case,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Hard-validate Flow-DCE/MFCE learned sampling by running the full "
            "model forward and collecting actual block inputs with forward hooks."
        )
    )
    parser.add_argument("--config", default="experiments/human36m/human36m_single_rgbflow_mfce_separate.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--labels", default="")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--flow-dir", default="")
    parser.add_argument("--flow-clip", type=float, default=None)
    parser.add_argument("--flow-norm", type=float, default=None)
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-radius", type=int, default=3)
    parser.add_argument("--min-cpn-gt-error", type=float, default=6.0)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="debug_vis/flow_mfce_forward_hook_sampling.json")
    parser.add_argument("--vis-dir", default="debug_vis/flow_mfce_forward_hook_sampling_vis")
    parser.add_argument("--no-vis", action="store_true")
    return parser.parse_args()


def load_labels(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def resolve_device(device_arg):
    import torch

    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested {}, but CUDA is not available.".format(device_arg))
    return device


def cfg_get(node, key, default=None):
    try:
        if key in node:
            return node[key]
    except TypeError:
        pass
    return getattr(node, key, default)


def build_model(config_path, checkpoint_path, device):
    import torch

    from mvn.models.DGPose_rgbflow_mfce import RGBFlowPoseMFCE
    from mvn.models.DGPose_rgbflow_mfce_separate import RGBFlowPoseMFCESeparate
    from mvn.utils.cfg import config, update_config

    update_config(config_path)
    model_map = {
        "RGBFlowPoseMFCE": RGBFlowPoseMFCE,
        "RGBFlowPoseMFCESeparate": RGBFlowPoseMFCESeparate,
    }
    if config.model.name not in model_map:
        raise ValueError(
            "This hook diagnostic supports RGBFlowPoseMFCE/RGBFlowPoseMFCESeparate, got {}".format(config.model.name)
        )
    model = model_map[config.model.name](config, device)

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
    model.to(device)
    model.eval()
    return model, config, str(ckpt_path)


def register_flow_hooks(model, captures):
    import torch
    import torch.nn.functional as F

    handles = []
    blocks = model.Lifting_net.Flow_Extraction.blocks

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
                    "pos": pos[:, 0].detach().cpu().numpy(),
                    "weights": weights[:, 0].detach().cpu().numpy(),
                    "offsets": offsets[:, 0].detach().cpu().numpy(),
                    "num_heads": int(module.num_heads),
                    "num_samples": int(module.num_samples),
                }

        return hook

    for idx, block in enumerate(blocks):
        handles.append(block.register_forward_hook(make_hook(idx)))
    return handles


def image_rel_path(row):
    seq = seq_name(row)
    return str(Path(seq) / (image_stem(seq, frame_id(row)) + ".jpg"))


def frame_id(row):
    if "frame_id" in row:
        return int(row["frame_id"])
    return int(row["image_id"])


def read_rgb(image_root, row):
    path = Path(image_root) / image_rel_path(row)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise FileNotFoundError("Failed to read image: {}".format(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (image - mean) / std


def make_key(row):
    return (
        int(row["subject"]),
        int(row["action"]),
        int(row["subaction"]),
        int(row["camera_id"]),
        frame_id(row),
    )


def prepare_records(labels, flow_dir, image_root, frame_gap, max_frames, seed):
    by_key = {make_key(row): row for row in labels}
    keys = sorted(by_key.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    records = []
    missing_prev = 0
    missing_flow = 0
    missing_image = 0
    for key in keys:
        if len(records) >= max_frames:
            break
        subject, action, subaction, camera_id, current_frame_id = key
        prev_key = (subject, action, subaction, camera_id, current_frame_id - frame_gap)
        prev = by_key.get(prev_key)
        if prev is None:
            missing_prev += 1
            continue
        cur = by_key[key]
        seq = seq_name(cur)
        flow_path = Path(flow_dir) / seq / (image_stem(seq, current_frame_id) + ".npy")
        if not flow_path.exists():
            missing_flow += 1
            continue
        image_path = Path(image_root) / image_rel_path(cur)
        if not image_path.exists():
            missing_image += 1
            continue
        records.append({"cur": cur, "prev": prev, "flow_path": flow_path})

    return records, {
        "missing_prev": missing_prev,
        "missing_flow": missing_flow,
        "missing_image": missing_image,
    }


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
            "frac_weighted_flow_better_center": fraction(
                block_rows, lambda row: row["weighted_flow_error_px"] < row["center_flow_error_px"]
            ),
            "frac_best_flow_better_center": fraction(
                block_rows, lambda row: row["best_sample_flow_error_px"] < row["center_flow_error_px"]
            ),
            "frac_weighted_pos_closer_to_gt": fraction(
                block_rows, lambda row: row["weighted_pos_error_px"] < row["center_pos_error_px"]
            ),
            "frac_best_pos_closer_to_gt": fraction(
                block_rows, lambda row: row["best_pos_error_px"] < row["center_pos_error_px"]
            ),
        }

    return {
        "by_block": by_block,
        "top_weighted_flow_degradation": sorted(
            rows, key=lambda row: row["weighted_flow_minus_center_px"], reverse=True
        )[:30],
        "top_opportunity_not_used": sorted(
            rows,
            key=lambda row: row["center_flow_error_px"] - row["best_sample_flow_error_px"],
            reverse=True,
        )[:30],
        "top_position_moves_away": sorted(
            rows, key=lambda row: row["weighted_pos_minus_center_px"], reverse=True
        )[:30],
    }


def main():
    args = parse_args()
    import torch

    device = resolve_device(args.device)
    model, config, checkpoint_path = build_model(args.config, args.checkpoint, device)

    labels_path = Path(args.labels or config.dataset.val_labels_path).expanduser()
    image_root = Path(args.image_root or config.dataset.root).expanduser()
    flow_dir = Path(args.flow_dir or config.dataset.flow_image_path).expanduser()
    flow_clip = args.flow_clip
    if flow_clip is None:
        flow_clip = float(cfg_get(config.dataset, "flow_clip", 20.0))
    flow_norm = args.flow_norm
    if flow_norm is None:
        flow_norm = float(cfg_get(config.dataset, "flow_norm", flow_clip))

    labels = load_labels(labels_path)
    records, missing = prepare_records(
        labels,
        flow_dir,
        image_root,
        args.frame_gap,
        args.max_frames,
        args.seed,
    )
    if not records:
        raise RuntimeError("No valid records found.")

    captures = {}
    handles = register_flow_hooks(model, captures)
    rows = []
    cases = []
    output_size = (args.width, args.height)

    try:
        with torch.no_grad():
            for batch in batch_iter(records, args.batch_size):
                captures.clear()
                images = []
                keypoints_2d = []
                keypoints_2d_crop = []
                flows_model = []
                flows_raw = []

                for record in batch:
                    cur = record["cur"]
                    flow_raw = np.load(record["flow_path"]).astype(np.float32)
                    if flow_raw.ndim != 3 or flow_raw.shape[-1] != 2:
                        raise ValueError("Expected flow [H,W,2], got {} at {}".format(flow_raw.shape, record["flow_path"]))

                    images.append(read_rgb(image_root, cur))
                    keypoints_2d.append(np.asarray(cur["joints_2d_cpn"], dtype=np.float32))
                    keypoints_2d_crop.append(np.asarray(cur["joints_2d_cpn_crop"], dtype=np.float32))
                    flows_raw.append(flow_raw)
                    flows_model.append(normalize_flow_for_model(flow_raw, flow_clip, flow_norm))

                images_t = torch.from_numpy(np.stack(images)).float().to(device)
                keypoints_t = torch.from_numpy(np.stack(keypoints_2d)).float().to(device)
                keypoints_crop_t = torch.from_numpy(np.stack(keypoints_2d_crop)).float().to(device)
                flows_t = torch.from_numpy(np.stack(flows_model)).float().to(device)

                _pred, _joint_depth, _s = model(images_t, keypoints_t, keypoints_crop_t.clone(), flows_t)

                if not captures:
                    raise RuntimeError("Forward hooks did not capture Flow_Extraction blocks.")

                for sample_idx, record in enumerate(batch):
                    cur = record["cur"]
                    prev = record["prev"]
                    flow_raw = flows_raw[sample_idx]
                    cur_cpn = np.asarray(cur["joints_2d_cpn_crop"], dtype=np.float32)
                    cur_gt = np.asarray(cur["joints_2d_gt_crop"], dtype=np.float32)
                    prev_gt = np.asarray(prev["joints_2d_gt_crop"], dtype=np.float32)
                    prev_to_raw = get_affine_transform(prev["center"], prev["scale"], output_size, inv=True)
                    raw_to_cur = get_affine_transform(cur["center"], cur["scale"], output_size, inv=False)
                    prev_gt_same = affine_apply(raw_to_cur, affine_apply(prev_to_raw, prev_gt))
                    gt_target_px = prev_gt_same - cur_gt
                    center_flow_px = bilinear_sample_flow(flow_raw, cur_cpn)
                    center_flow_error_px = np.linalg.norm(center_flow_px - gt_target_px, axis=-1)
                    center_pos_error = np.linalg.norm(cur_cpn - cur_gt, axis=-1)
                    stats = patch_stats(flow_raw, cur_cpn, args.patch_radius)

                    for block_idx, block in captures.items():
                        num_heads = block["num_heads"]
                        num_samples = block["num_samples"]
                        pos_norm = block["pos"][sample_idx].reshape(17, num_heads, num_samples, 2)
                        xy = norm_to_xy(pos_norm, args.width, args.height)
                        weights = block["weights"][sample_idx]
                        sampled_flow_px = bilinear_sample_flow(flow_raw, xy)
                        flow_err_px = np.linalg.norm(sampled_flow_px - gt_target_px[:, None, None, :], axis=-1)
                        pos_err = np.linalg.norm(xy - cur_gt[:, None, None, :], axis=-1)

                        weighted_flow_px = (weights[..., None] * sampled_flow_px).sum(axis=2).mean(axis=1)
                        weighted_xy = (weights[..., None] * xy).sum(axis=2).mean(axis=1)
                        weighted_flow_error_px = np.linalg.norm(weighted_flow_px - gt_target_px, axis=-1)
                        weighted_pos_error = np.linalg.norm(weighted_xy - cur_gt, axis=-1)
                        best_flow_error_px = flow_err_px.reshape(17, -1).min(axis=1)
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
                                "joint": int(joint_idx),
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
    finally:
        for handle in handles:
            handle.remove()

    out = {
        "meta": {
            "mode": "full_forward_hooks",
            "config": str(Path(args.config).expanduser()),
            "labels": str(labels_path),
            "flow_dir": str(flow_dir),
            "image_root": str(image_root),
            "checkpoint": checkpoint_path,
            "flow_clip": flow_clip,
            "flow_norm": flow_norm,
            "frame_gap": args.frame_gap,
            "width": args.width,
            "height": args.height,
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
        selected = sorted(
            cases,
            key=lambda item: item["row"]["weighted_flow_minus_center_px"] - item["row"]["best_sample_minus_center_px"],
            reverse=True,
        )[: args.top_k]
        out["visual_cases"] = [case["row"] for case in selected]
        out["meta"]["vis_dir"] = str(vis_dir)
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
            visualize_case(case, vis_dir / name, image_root, args.width, args.height)

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
                    "weighted_flow_minus_center_px": block_summary["weighted_flow_minus_center_px"],
                    "best_sample_minus_center_px": block_summary["best_sample_minus_center_px"],
                    "frac_weighted_flow_better_center": block_summary["frac_weighted_flow_better_center"],
                    "frac_best_flow_better_center": block_summary["frac_best_flow_better_center"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
