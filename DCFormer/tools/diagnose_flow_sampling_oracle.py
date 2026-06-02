#!/usr/bin/env python3
"""Oracle analysis for joint-level optical-flow sampling.

This diagnostic does not train a model. It asks a simpler question first:
given a hand-built candidate set around each 2D joint, is there a flow sample
that better matches the GT short-term joint motion than the center sample?

The output is meant to guide the next flow sampling module:
- whether sampling has enough oracle headroom;
- which joints benefit;
- whether the best samples look like re-centering, limb-support sampling, or
  far/off-body coincidences.
"""

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnose_flow_mfce_sampling import (  # noqa: E402
    ACTION_NAMES,
    JOINT_NAMES,
    affine_apply,
    bilinear_sample_flow,
    get_affine_transform,
    image_stem,
    load_labels,
    seq_name,
    summarize,
)


PARENTS = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])
LIMB_JOINTS = {2, 3, 5, 6, 12, 13, 15, 16}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure the oracle upper bound of interpretable optical-flow sampling "
            "candidates around each detected 2D joint."
        )
    )
    parser.add_argument("--labels", default="data/h36m_validation.pkl")
    parser.add_argument("--flow-dir", default="../H36M-Toolbox/flow_images_float_clip10")
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ordered", action="store_true")

    parser.add_argument("--radii", default="2,4,6,8,12,16")
    parser.add_argument("--num-directions", type=int, default=16)
    parser.add_argument("--dense-grid", action="store_true")
    parser.add_argument("--grid-step", type=float, default=4.0)
    parser.add_argument("--bone-fracs", default="0.25,0.5,0.75")
    parser.add_argument("--bone-band-px", default="-3,0,3")
    parser.add_argument("--gain-margin", type=float, default=0.5)
    parser.add_argument("--center-bad-thr", type=float, default=2.0)
    parser.add_argument("--near-gt-thr", type=float, default=4.0)
    parser.add_argument("--top-k", type=int, default=30)

    parser.add_argument("--out", default="debug_vis/flow_sampling_oracle.json")
    parser.add_argument("--csv-out", default="debug_vis/flow_sampling_oracle.csv")
    return parser.parse_args()


def parse_float_list(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def action_name(action):
    idx = int(action) - 2
    if 0 <= idx < len(ACTION_NAMES):
        return ACTION_NAMES[idx]
    return "action_{}".format(action)


def joint_name(joint):
    if 0 <= int(joint) < len(JOINT_NAMES):
        return JOINT_NAMES[int(joint)]
    return "joint_{}".format(joint)


def children_map():
    children = defaultdict(list)
    for joint, parent in enumerate(PARENTS):
        if parent >= 0:
            children[int(parent)].append(int(joint))
    return children


CHILDREN = children_map()


def local_offsets(radii, num_directions, dense_grid=False, grid_step=4.0):
    offsets = []
    seen = set()

    def add(dx, dy, scheme, radius):
        key = (round(float(dx), 4), round(float(dy), 4), scheme)
        if key in seen:
            return
        seen.add(key)
        offsets.append(
            {
                "offset": np.array([dx, dy], dtype=np.float32),
                "scheme": scheme,
                "radius": float(radius),
                "neighbor": -1,
                "frac": 0.0,
                "band": 0.0,
            }
        )

    for radius in radii:
        if radius <= 0:
            continue
        for idx in range(num_directions):
            theta = 2.0 * math.pi * float(idx) / float(num_directions)
            add(radius * math.cos(theta), radius * math.sin(theta), "local_ring", radius)

    if dense_grid and radii:
        max_radius = max(radii)
        values = np.arange(-max_radius, max_radius + 1e-6, grid_step, dtype=np.float32)
        for dy in values:
            for dx in values:
                if abs(float(dx)) < 1e-6 and abs(float(dy)) < 1e-6:
                    continue
                radius = math.sqrt(float(dx) ** 2 + float(dy) ** 2)
                if radius <= max_radius + 1e-6:
                    add(float(dx), float(dy), "local_grid", radius)
    return offsets


def add_bone_candidates(candidates, cur_cpn, joint, bone_fracs, bone_band_px):
    neighbors = []
    parent = int(PARENTS[joint])
    if parent >= 0:
        neighbors.append(parent)
    neighbors.extend(CHILDREN.get(int(joint), []))

    origin = cur_cpn[joint].astype(np.float32)
    for neighbor in neighbors:
        vec = cur_cpn[neighbor].astype(np.float32) - origin
        length = float(np.linalg.norm(vec))
        if length < 1e-6:
            continue
        unit = vec / length
        perp = np.array([-unit[1], unit[0]], dtype=np.float32)
        for frac in bone_fracs:
            base = origin + unit * (length * float(frac))
            for band in bone_band_px:
                pos = base + perp * float(band)
                candidates.append(
                    {
                        "xy": pos.astype(np.float32),
                        "scheme": "bone_band" if abs(float(band)) > 1e-6 else "bone_line",
                        "radius": float(np.linalg.norm(pos - origin)),
                        "neighbor": int(neighbor),
                        "frac": float(frac),
                        "band": float(band),
                    }
                )


def build_candidates(cur_cpn, joint, offsets, bone_fracs, bone_band_px):
    origin = cur_cpn[joint].astype(np.float32)
    candidates = [
        {
            "xy": origin,
            "scheme": "center",
            "radius": 0.0,
            "neighbor": -1,
            "frac": 0.0,
            "band": 0.0,
        }
    ]
    for item in offsets:
        candidates.append(
            {
                "xy": origin + item["offset"],
                "scheme": item["scheme"],
                "radius": item["radius"],
                "neighbor": item["neighbor"],
                "frac": item["frac"],
                "band": item["band"],
            }
        )
    add_bone_candidates(candidates, cur_cpn, joint, bone_fracs, bone_band_px)
    return candidates


def top_rows(rows, key, count, reverse=True):
    keep = [
        "seq",
        "subject",
        "action_name",
        "frame_id",
        "joint",
        "joint_name",
        "joint_group",
        "d_xy",
        "target_motion_mag",
        "center_error",
        "gt_site_error",
        "best_error",
        "oracle_gain",
        "best_scheme",
        "best_radius_px",
        "best_anchor_dist_px",
        "best_gt_dist_px",
        "best_pos_closer_to_gt",
        "candidate_count",
    ]
    selected = sorted(rows, key=lambda row: row[key], reverse=reverse)[:count]
    return [{name: row[name] for name in keep} for row in selected]


def frac(values, predicate):
    values = list(values)
    if not values:
        return None
    return float(sum(1 for value in values if predicate(value)) / len(values))


def group_summary(rows, gain_margin, center_bad_thr, near_gt_thr):
    if not rows:
        return {}
    scheme_counter = Counter(row["best_scheme"] for row in rows)
    return {
        "count": len(rows),
        "center_error": summarize([row["center_error"] for row in rows]),
        "gt_site_error": summarize([row["gt_site_error"] for row in rows]),
        "best_error": summarize([row["best_error"] for row in rows]),
        "oracle_gain": summarize([row["oracle_gain"] for row in rows]),
        "best_anchor_dist_px": summarize([row["best_anchor_dist_px"] for row in rows]),
        "best_gt_dist_px": summarize([row["best_gt_dist_px"] for row in rows]),
        "frac_best_better_center": frac(rows, lambda row: row["oracle_gain"] > 0.0),
        "frac_gain_ge_margin": frac(rows, lambda row: row["oracle_gain"] >= gain_margin),
        "frac_center_bad": frac(rows, lambda row: row["center_error"] >= center_bad_thr),
        "frac_best_pos_closer_to_gt": frac(rows, lambda row: row["best_pos_closer_to_gt"]),
        "frac_best_near_gt": frac(rows, lambda row: row["best_gt_dist_px"] <= near_gt_thr),
        "best_scheme_counts": dict(scheme_counter.most_common()),
    }


def summarize_by(rows, key, gain_margin, center_bad_thr, near_gt_thr):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {
        name: group_summary(items, gain_margin, center_bad_thr, near_gt_thr)
        for name, items in sorted(grouped.items(), key=lambda item: item[0])
    }


def scheme_min_errors(candidates, errors):
    best = {}
    for idx, item in enumerate(candidates):
        scheme = item["scheme"]
        value = float(errors[idx])
        if scheme not in best or value < best[scheme]:
            best[scheme] = value
    return best


def write_csv(path, rows):
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seq",
        "subject",
        "action",
        "action_name",
        "subaction",
        "camera_id",
        "frame_id",
        "joint",
        "joint_name",
        "joint_group",
        "d_xy",
        "target_motion_mag",
        "center_error",
        "gt_site_error",
        "best_error",
        "oracle_gain",
        "best_scheme",
        "best_radius_px",
        "best_neighbor",
        "best_frac",
        "best_band",
        "best_anchor_dist_px",
        "best_gt_dist_px",
        "best_pos_closer_to_gt",
        "candidate_count",
        "local_ring_best_error",
        "local_grid_best_error",
        "bone_line_best_error",
        "bone_band_best_error",
    ]
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main():
    args = parse_args()
    labels_path = Path(args.labels).expanduser().resolve()
    flow_dir = Path(args.flow_dir).expanduser().resolve()
    output_size = (args.width, args.height)

    if not labels_path.exists():
        raise FileNotFoundError("labels file not found: {}".format(labels_path))
    if not flow_dir.exists():
        raise FileNotFoundError("flow directory not found: {}".format(flow_dir))

    radii = parse_float_list(args.radii)
    bone_fracs = parse_float_list(args.bone_fracs)
    bone_band_px = parse_float_list(args.bone_band_px)
    offsets = local_offsets(radii, args.num_directions, args.dense_grid, args.grid_step)

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

    rows = []
    scanned_frames = 0
    missing_prev = 0
    missing_flow = 0

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

        center_flow = bilinear_sample_flow(flow, cur_cpn)
        gt_site_flow = bilinear_sample_flow(flow, cur_gt)
        center_error = np.linalg.norm(center_flow - target_motion, axis=-1)
        gt_site_error = np.linalg.norm(gt_site_flow - target_motion, axis=-1)
        d_xy = np.linalg.norm(cur_cpn - cur_gt, axis=-1)
        target_motion_mag = np.linalg.norm(target_motion, axis=-1)

        for joint in range(cur_cpn.shape[0]):
            candidates = build_candidates(cur_cpn, joint, offsets, bone_fracs, bone_band_px)
            candidate_xy = np.stack([item["xy"] for item in candidates], axis=0).astype(np.float32)
            sampled = bilinear_sample_flow(flow, candidate_xy)
            errors = np.linalg.norm(sampled - target_motion[joint], axis=-1)
            best_idx = int(np.argmin(errors))
            best = candidates[best_idx]
            scheme_errors = scheme_min_errors(candidates, errors)

            best_anchor_dist = float(np.linalg.norm(best["xy"] - cur_cpn[joint]))
            best_gt_dist = float(np.linalg.norm(best["xy"] - cur_gt[joint]))
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
                "target_motion_mag": float(target_motion_mag[joint]),
                "center_error": float(center_error[joint]),
                "gt_site_error": float(gt_site_error[joint]),
                "best_error": float(errors[best_idx]),
                "oracle_gain": float(center_error[joint] - errors[best_idx]),
                "best_scheme": best["scheme"],
                "best_radius_px": float(best["radius"]),
                "best_neighbor": int(best["neighbor"]),
                "best_frac": float(best["frac"]),
                "best_band": float(best["band"]),
                "best_anchor_dist_px": best_anchor_dist,
                "best_gt_dist_px": best_gt_dist,
                "best_pos_closer_to_gt": bool(best_gt_dist < float(d_xy[joint])),
                "candidate_count": len(candidates),
                "local_ring_best_error": scheme_errors.get("local_ring"),
                "local_grid_best_error": scheme_errors.get("local_grid"),
                "bone_line_best_error": scheme_errors.get("bone_line"),
                "bone_band_best_error": scheme_errors.get("bone_band"),
            }
            rows.append(row)

    top_k = int(args.top_k)
    output = {
        "meta": {
            "labels": str(labels_path),
            "flow_dir": str(flow_dir),
            "frame_gap": args.frame_gap,
            "width": args.width,
            "height": args.height,
            "max_frames": args.max_frames,
            "scanned_frames": scanned_frames,
            "num_joint_rows": len(rows),
            "missing_prev": missing_prev,
            "missing_flow": missing_flow,
            "seed": args.seed,
            "ordered": args.ordered,
            "radii": radii,
            "num_directions": args.num_directions,
            "dense_grid": args.dense_grid,
            "grid_step": args.grid_step,
            "bone_fracs": bone_fracs,
            "bone_band_px": bone_band_px,
            "gain_margin": args.gain_margin,
            "center_bad_thr": args.center_bad_thr,
            "near_gt_thr": args.near_gt_thr,
        },
        "summary": {
            "overall": group_summary(rows, args.gain_margin, args.center_bad_thr, args.near_gt_thr),
            "by_joint": summarize_by(rows, "joint_name", args.gain_margin, args.center_bad_thr, args.near_gt_thr),
            "by_group": summarize_by(rows, "joint_group", args.gain_margin, args.center_bad_thr, args.near_gt_thr),
            "by_best_scheme": summarize_by(rows, "best_scheme", args.gain_margin, args.center_bad_thr, args.near_gt_thr),
            "top_oracle_gain": top_rows(rows, "oracle_gain", top_k, reverse=True),
            "top_center_bad": top_rows(rows, "center_error", top_k, reverse=True),
            "top_far_best": top_rows(
                [row for row in rows if row["oracle_gain"] >= args.gain_margin],
                "best_anchor_dist_px",
                top_k,
                reverse=True,
            ),
        },
    }

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_csv(args.csv_out, rows)

    print(
        "wrote {} rows from {} frames to {}".format(
            len(rows),
            scanned_frames,
            output_path,
        )
    )
    if args.csv_out:
        print("wrote row csv to {}".format(args.csv_out))


if __name__ == "__main__":
    main()
