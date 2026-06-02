#!/usr/bin/env python3
"""Analyze how noisy 2D anchors interact with raw optical flow evidence.

This script does not use model weights. It classifies CPN anchors by comparing
their sampled flow against the GT joint's short-term 2D motion in the current
crop coordinate system.
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from diagnose_flow_mfce_sampling import (
    ACTION_NAMES,
    JOINT_NAMES,
    affine_apply,
    bilinear_sample_flow,
    get_affine_transform,
    image_stem,
    load_labels,
    patch_stats,
    seq_name,
    summarize,
)


LIMB_JOINTS = {2, 3, 5, 6, 12, 13, 15, 16}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Classify raw-flow joint anchors into measurable motion-evidence "
            "cases using CPN/GT 2D coordinates and GT short-term joint motion."
        )
    )
    parser.add_argument("--labels", default="data/h36m_validation.pkl")
    parser.add_argument("--flow-dir", default="../H36M-Toolbox/flow_images_float")
    parser.add_argument("--out", default="debug_vis/flow_anchor_motion_cases.json")
    parser.add_argument("--csv-out", default="debug_vis/flow_anchor_motion_cases.csv")
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=4096)
    parser.add_argument("--patch-radius", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ordered", action="store_true", help="Scan frames in dataset order instead of shuffling.")

    parser.add_argument("--small-2d-gap", type=float, default=2.0)
    parser.add_argument("--large-2d-gap", type=float, default=4.0)
    parser.add_argument("--good-motion-error", type=float, default=1.0)
    parser.add_argument("--bad-motion-error", type=float, default=2.0)
    parser.add_argument("--same-flow-delta", type=float, default=1.0)
    parser.add_argument("--zero-flow-mag", type=float, default=0.5)
    parser.add_argument("--moving-target-mag", type=float, default=1.0)
    parser.add_argument("--gt-better-margin", type=float, default=0.5)
    return parser.parse_args()


def action_name(action):
    idx = int(action) - 2
    if 0 <= idx < len(ACTION_NAMES):
        return ACTION_NAMES[idx]
    return "action_{}".format(action)


def joint_name(joint):
    if 0 <= int(joint) < len(JOINT_NAMES):
        return JOINT_NAMES[int(joint)]
    return "joint_{}".format(joint)


def local_flow_stat(flow, xy, radius=2, reducer="median"):
    h, w = flow.shape[:2]
    values = []
    for coord in np.asarray(xy, dtype=np.float32):
        cx = int(np.clip(round(float(coord[0])), 0, w - 1))
        cy = int(np.clip(round(float(coord[1])), 0, h - 1))
        x0 = max(0, cx - radius)
        x1 = min(w, cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(h, cy + radius + 1)
        patch = flow[y0:y1, x0:x1].reshape(-1, 2).astype(np.float32)
        if reducer == "mean":
            values.append(patch.mean(axis=0))
        else:
            values.append(np.median(patch, axis=0))
    return np.stack(values, axis=0).astype(np.float32)


def percentile(values, qs):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {str(q): None for q in qs}
    return {str(q): float(np.percentile(arr, q)) for q in qs}


def classify(row, args):
    d_xy = row["d_xy"]
    e_c = row["e_center"]
    e_g = row["e_gt_site"]
    delta = row["flow_delta_cpn_gt"]
    mag_c = row["center_flow_mag"]
    mag_t = row["target_motion_mag"]

    if d_xy <= args.small_2d_gap and e_c <= args.good_motion_error:
        return "A_2d_ok_motion_ok"
    if d_xy >= args.large_2d_gap and e_c <= args.good_motion_error:
        return "B_2d_off_same_motion"
    if e_c >= args.bad_motion_error and e_g >= args.bad_motion_error and delta <= args.same_flow_delta:
        return "G_flow_bad_both_sites"
    if (
        d_xy >= args.large_2d_gap
        and e_c >= args.bad_motion_error
        and mag_c <= args.zero_flow_mag
        and mag_t >= args.moving_target_mag
        and (e_c - e_g) >= args.gt_better_margin
    ):
        return "D_2d_off_zero_background"
    if (
        d_xy >= args.large_2d_gap
        and e_c >= args.bad_motion_error
        and delta >= args.same_flow_delta
        and (e_c - e_g) >= args.gt_better_margin
    ):
        return "C_2d_off_wrong_motion"
    if d_xy <= args.small_2d_gap and e_c >= args.bad_motion_error:
        return "E_2d_ok_flow_bad"
    return "F_ambiguous"


def summarize_by(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    out = {}
    for group, items in grouped.items():
        out[str(group)] = {
            "count": len(items),
            "d_xy": summarize([x["d_xy"] for x in items]),
            "e_center": summarize([x["e_center"] for x in items]),
            "e_gt_site": summarize([x["e_gt_site"] for x in items]),
            "flow_delta_cpn_gt": summarize([x["flow_delta_cpn_gt"] for x in items]),
            "target_motion_mag": summarize([x["target_motion_mag"] for x in items]),
            "center_flow_mag": summarize([x["center_flow_mag"] for x in items]),
            "center_patch_var": summarize([x["center_patch_var"] for x in items]),
        }
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def top_rows(rows, key, k, reverse=True):
    keep_keys = [
        "seq",
        "subject",
        "action_name",
        "frame_id",
        "joint",
        "joint_name",
        "case",
        "d_xy",
        "e_center",
        "e_gt_site",
        "flow_delta_cpn_gt",
        "target_motion_mag",
        "center_flow_mag",
        "center_patch_var",
        "center_flow_x",
        "center_flow_y",
        "gt_site_flow_x",
        "gt_site_flow_y",
        "target_motion_x",
        "target_motion_y",
    ]
    selected = sorted(rows, key=lambda row: row[key], reverse=reverse)[:k]
    return [{k2: row[k2] for k2 in keep_keys} for row in selected]


def main():
    args = parse_args()
    labels_path = Path(args.labels).expanduser().resolve()
    flow_dir = Path(args.flow_dir).expanduser().resolve()
    output_size = (args.width, args.height)

    labels = load_labels(labels_path)
    labels.sort(key=lambda item: (item["subject"], item["action"], item["subaction"], item["camera_id"], item["image_id"]))
    by_key = {
        (int(item["subject"]), int(item["action"]), int(item["subaction"]), int(item["camera_id"]), int(item["image_id"])): item
        for item in labels
    }

    rows = []
    scanned_frames = 0
    missing_prev = 0
    missing_flow = 0

    keys = list(by_key)
    if args.ordered:
        keys.sort()
    else:
        random.Random(args.seed).shuffle(keys)

    for key in keys:
        if scanned_frames >= args.max_frames:
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
        scanned_frames += 1

        flow = np.load(flow_path).astype(np.float32)
        cur_cpn = np.asarray(cur["joints_2d_cpn_crop"], dtype=np.float32)
        cur_gt = np.asarray(cur["joints_2d_gt_crop"], dtype=np.float32)
        prev_gt = np.asarray(prev["joints_2d_gt_crop"], dtype=np.float32)

        prev_to_raw = get_affine_transform(prev["center"], prev["scale"], output_size, inv=True)
        raw_to_cur = get_affine_transform(cur["center"], cur["scale"], output_size, inv=False)
        prev_gt_same = affine_apply(raw_to_cur, affine_apply(prev_to_raw, prev_gt))
        target_motion = prev_gt_same - cur_gt

        center_flow = local_flow_stat(flow, cur_cpn, args.patch_radius, reducer="median")
        gt_site_flow = local_flow_stat(flow, cur_gt, args.patch_radius, reducer="median")
        center_flow_point = bilinear_sample_flow(flow, cur_cpn)
        stats = patch_stats(flow, cur_cpn, args.patch_radius)

        d_xy = np.linalg.norm(cur_cpn - cur_gt, axis=-1)
        e_center = np.linalg.norm(center_flow - target_motion, axis=-1)
        e_gt_site = np.linalg.norm(gt_site_flow - target_motion, axis=-1)
        delta = np.linalg.norm(center_flow - gt_site_flow, axis=-1)
        mag_c = np.linalg.norm(center_flow, axis=-1)
        mag_t = np.linalg.norm(target_motion, axis=-1)
        point_vs_median = np.linalg.norm(center_flow_point - center_flow, axis=-1)

        for joint in range(cur_cpn.shape[0]):
            row = {
                "seq": seq,
                "subject": int(subject),
                "action": int(action),
                "action_name": action_name(action),
                "subaction": int(subaction),
                "camera_id": int(camera_id),
                "frame_id": int(frame_id),
                "joint": int(joint),
                "joint_name": joint_name(joint),
                "joint_group": "limb" if joint in LIMB_JOINTS else "body",
                "d_xy": float(d_xy[joint]),
                "e_center": float(e_center[joint]),
                "e_gt_site": float(e_gt_site[joint]),
                "flow_delta_cpn_gt": float(delta[joint]),
                "target_motion_mag": float(mag_t[joint]),
                "center_flow_mag": float(mag_c[joint]),
                "center_patch_mag_mean": float(stats[joint, 0]),
                "center_patch_mag_std": float(stats[joint, 1]),
                "center_patch_var": float(stats[joint, 2]),
                "center_flow_edge": float(stats[joint, 3]),
                "point_vs_median": float(point_vs_median[joint]),
                "center_flow_x": float(center_flow[joint, 0]),
                "center_flow_y": float(center_flow[joint, 1]),
                "gt_site_flow_x": float(gt_site_flow[joint, 0]),
                "gt_site_flow_y": float(gt_site_flow[joint, 1]),
                "target_motion_x": float(target_motion[joint, 0]),
                "target_motion_y": float(target_motion[joint, 1]),
            }
            row["case"] = classify(row, args)
            rows.append(row)

    case_counts = Counter(row["case"] for row in rows)
    total = len(rows)
    values = {name: [row[name] for row in rows] for name in ["d_xy", "e_center", "e_gt_site", "flow_delta_cpn_gt", "target_motion_mag", "center_flow_mag", "center_patch_var"]}

    out = {
        "meta": {
            "labels": str(labels_path),
            "flow_dir": str(flow_dir),
            "frame_gap": args.frame_gap,
            "width": args.width,
            "height": args.height,
            "max_frames": args.max_frames,
            "scanned_frames": scanned_frames,
            "num_joint_rows": total,
            "missing_prev": missing_prev,
            "missing_flow": missing_flow,
            "patch_radius": args.patch_radius,
            "seed": args.seed,
            "ordered": args.ordered,
            "thresholds": {
                "small_2d_gap": args.small_2d_gap,
                "large_2d_gap": args.large_2d_gap,
                "good_motion_error": args.good_motion_error,
                "bad_motion_error": args.bad_motion_error,
                "same_flow_delta": args.same_flow_delta,
                "zero_flow_mag": args.zero_flow_mag,
                "moving_target_mag": args.moving_target_mag,
                "gt_better_margin": args.gt_better_margin,
            },
        },
        "case_counts": dict(case_counts),
        "case_fractions": {case: count / max(total, 1) for case, count in case_counts.items()},
        "global_distribution": {name: percentile(vals, [5, 10, 25, 50, 75, 90, 95]) for name, vals in values.items()},
        "by_case": summarize_by(rows, "case"),
        "by_joint": summarize_by(rows, "joint_name"),
        "by_joint_group": summarize_by(rows, "joint_group"),
        "top_wrong_motion": top_rows([r for r in rows if r["case"] == "C_2d_off_wrong_motion"], "e_center", args.top_k),
        "top_zero_background": top_rows([r for r in rows if r["case"] == "D_2d_off_zero_background"], "e_center", args.top_k),
        "top_2d_ok_flow_bad": top_rows([r for r in rows if r["case"] == "E_2d_ok_flow_bad"], "e_center", args.top_k),
        "top_2d_off_same_motion": top_rows([r for r in rows if r["case"] == "B_2d_off_same_motion"], "d_xy", args.top_k),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(out, file, indent=2, ensure_ascii=False)

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps({
        "out": str(out_path),
        "csv_out": str(csv_path),
        "scanned_frames": scanned_frames,
        "num_joint_rows": total,
        "case_counts": dict(case_counts),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
