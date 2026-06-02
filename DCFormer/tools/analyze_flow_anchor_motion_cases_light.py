#!/usr/bin/env python3
"""Lightweight large-sample optical-flow anchor diagnosis.

This variant only writes aggregate JSON statistics. It avoids the per-joint CSV
and expensive patch diagnostics so larger random samples can be checked quickly.
"""

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from analyze_flow_anchor_motion_cases import action_name, classify, joint_name, local_flow_stat  # noqa: E402
from diagnose_flow_mfce_sampling import (  # noqa: E402
    affine_apply,
    bilinear_sample_flow,
    get_affine_transform,
    image_stem,
    load_labels,
    seq_name,
)


METRICS = [
    "d_xy",
    "e_center",
    "e_gt_site",
    "gain",
    "flow_delta_cpn_gt",
    "target_motion_mag",
    "center_flow_mag",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Large-sample flow-anchor case statistics without writing per-joint CSV."
    )
    parser.add_argument("--labels", default="data/h36m_validation.pkl")
    parser.add_argument("--flow-dir", default="../H36M-Toolbox/flow_images_float_clip10")
    parser.add_argument("--out", default="debug_vis/flow_anchor_motion_cases_clip10_light32768.json")
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--ordered", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--stat",
        choices=["point", "median", "mean"],
        default="point",
        help="How to sample local flow at CPN/GT sites. point is true single-point bilinear sampling.",
    )
    parser.add_argument("--patch-radius", type=int, default=2)

    parser.add_argument("--small-2d-gap", type=float, default=2.0)
    parser.add_argument("--large-2d-gap", type=float, default=4.0)
    parser.add_argument("--good-motion-error", type=float, default=1.0)
    parser.add_argument("--bad-motion-error", type=float, default=2.0)
    parser.add_argument("--same-flow-delta", type=float, default=1.0)
    parser.add_argument("--zero-flow-mag", type=float, default=0.5)
    parser.add_argument("--moving-target-mag", type=float, default=1.0)
    parser.add_argument("--gt-better-margin", type=float, default=0.5)
    return parser.parse_args()


def summarize(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def new_metric_store():
    return {name: [] for name in METRICS}


def add_metrics(store, row):
    for name in METRICS:
        store[name].append(float(row[name]))


def summarize_store(store):
    return {name: summarize(values) for name, values in store.items()}


def derived_group_names(row):
    gain = row["gain"]
    d_xy = row["d_xy"]
    e_center = row["e_center"]
    delta = row["flow_delta_cpn_gt"]
    target_mag = row["target_motion_mag"]

    if row["case"] in {"C_2d_off_wrong_motion", "D_2d_off_zero_background"}:
        yield "strict_C_plus_D"
    if d_xy >= 6.0 and target_mag >= 1.0 and e_center >= 2.0 and gain >= 1.0 and delta >= 1.0:
        yield "recoverable_off_gain_1"
    if d_xy >= 6.0 and target_mag >= 1.0 and e_center >= 2.0 and gain >= 0.5 and delta >= 0.5:
        yield "recoverable_off_gain_0_5"
    if d_xy >= 6.0 and target_mag >= 1.0 and e_center >= 2.0:
        yield "off_center_bad"
    if target_mag >= 1.0 and e_center >= 2.0:
        yield "center_bad_any_2d"
    if target_mag >= 1.0 and e_center >= 2.0 and gain >= 1.0:
        yield "center_bad_gt_better_any_2d"


def flow_at_sites(flow, cpn, gt, args):
    if args.stat == "point":
        return bilinear_sample_flow(flow, cpn), bilinear_sample_flow(flow, gt)
    return (
        local_flow_stat(flow, cpn, args.patch_radius, reducer=args.stat),
        local_flow_stat(flow, gt, args.patch_radius, reducer=args.stat),
    )


def main():
    args = parse_args()
    start = time.time()
    labels_path = Path(args.labels)
    flow_dir = Path(args.flow_dir)
    output_size = (args.width, args.height)

    labels = load_labels(labels_path)
    labels.sort(key=lambda item: (item["subject"], item["action"], item["subaction"], item["camera_id"], item["image_id"]))
    by_key = {
        (int(item["subject"]), int(item["action"]), int(item["subaction"]), int(item["camera_id"]), int(item["image_id"])): item
        for item in labels
    }

    keys = list(by_key)
    if args.ordered:
        keys.sort()
    else:
        random.Random(args.seed).shuffle(keys)

    case_counts = Counter()
    case_joint_counts = defaultdict(Counter)
    case_action_counts = defaultdict(Counter)
    case_metrics = defaultdict(new_metric_store)
    derived_counts = Counter()
    derived_metrics = defaultdict(new_metric_store)
    global_metrics = new_metric_store()

    scanned_frames = 0
    missing_prev = 0
    missing_flow = 0
    num_joint_rows = 0

    for key in keys:
        if args.max_frames > 0 and scanned_frames >= args.max_frames:
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

        flow = np.load(flow_path).astype(np.float32)
        cur_cpn = np.asarray(cur["joints_2d_cpn_crop"], dtype=np.float32)
        cur_gt = np.asarray(cur["joints_2d_gt_crop"], dtype=np.float32)
        prev_gt = np.asarray(prev["joints_2d_gt_crop"], dtype=np.float32)

        prev_to_raw = get_affine_transform(prev["center"], prev["scale"], output_size, inv=True)
        raw_to_cur = get_affine_transform(cur["center"], cur["scale"], output_size, inv=False)
        prev_gt_same = affine_apply(raw_to_cur, affine_apply(prev_to_raw, prev_gt))
        target_motion = prev_gt_same - cur_gt

        center_flow, gt_site_flow = flow_at_sites(flow, cur_cpn, cur_gt, args)
        d_xy = np.linalg.norm(cur_cpn - cur_gt, axis=-1)
        e_center = np.linalg.norm(center_flow - target_motion, axis=-1)
        e_gt_site = np.linalg.norm(gt_site_flow - target_motion, axis=-1)
        delta = np.linalg.norm(center_flow - gt_site_flow, axis=-1)
        mag_c = np.linalg.norm(center_flow, axis=-1)
        mag_t = np.linalg.norm(target_motion, axis=-1)

        scanned_frames += 1
        if args.progress_every > 0 and scanned_frames % args.progress_every == 0:
            elapsed = time.time() - start
            print(
                "progress frames={} joints={} elapsed={:.1f}s missing_prev={} missing_flow={}".format(
                    scanned_frames, num_joint_rows, elapsed, missing_prev, missing_flow
                ),
                flush=True,
            )

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
                "d_xy": float(d_xy[joint]),
                "e_center": float(e_center[joint]),
                "e_gt_site": float(e_gt_site[joint]),
                "gain": float(e_center[joint] - e_gt_site[joint]),
                "flow_delta_cpn_gt": float(delta[joint]),
                "target_motion_mag": float(mag_t[joint]),
                "center_flow_mag": float(mag_c[joint]),
            }
            row["case"] = classify(row, args)
            case = row["case"]
            case_counts[case] += 1
            case_joint_counts[case][row["joint_name"]] += 1
            case_action_counts[case][row["action_name"]] += 1
            add_metrics(case_metrics[case], row)
            add_metrics(global_metrics, row)
            for group_name in derived_group_names(row):
                derived_counts[group_name] += 1
                add_metrics(derived_metrics[group_name], row)
            num_joint_rows += 1

    total = max(num_joint_rows, 1)
    out = {
        "meta": {
            "labels": str(labels_path),
            "flow_dir": str(flow_dir),
            "frame_gap": args.frame_gap,
            "width": args.width,
            "height": args.height,
            "max_frames": args.max_frames,
            "scanned_frames": scanned_frames,
            "num_joint_rows": num_joint_rows,
            "missing_prev": missing_prev,
            "missing_flow": missing_flow,
            "seed": args.seed,
            "ordered": args.ordered,
            "stat": args.stat,
            "patch_radius": args.patch_radius,
            "elapsed_sec": time.time() - start,
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
        "case_fractions": {name: count / total for name, count in case_counts.items()},
        "case_metrics": {name: summarize_store(store) for name, store in case_metrics.items()},
        "case_top_joints": {name: counts.most_common(8) for name, counts in case_joint_counts.items()},
        "case_top_actions": {name: counts.most_common(8) for name, counts in case_action_counts.items()},
        "derived_group_counts": dict(derived_counts),
        "derived_group_fractions": {name: count / total for name, count in derived_counts.items()},
        "derived_group_metrics": {name: summarize_store(store) for name, store in derived_metrics.items()},
        "global_metrics": summarize_store(global_metrics),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(out, file, indent=2)

    print(json.dumps({"out": str(out_path), "scanned_frames": scanned_frames, "num_joint_rows": num_joint_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
