import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from diagnose_flow_mfce_sampling import (  # noqa: E402
    affine_apply,
    bilinear_sample_flow,
    get_affine_transform,
    image_stem,
    load_labels,
    seq_name,
)


PARENTS = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize single-point optical-flow sampling failures caused by "
            "biased 2D CPN anchors. No learned sampling weights are used."
        )
    )
    parser.add_argument("--csv", default="debug_vis/flow_anchor_motion_cases_clip10_random4096.csv")
    parser.add_argument("--labels", default="data/h36m_validation.pkl")
    parser.add_argument("--flow-dir", default="../H36M-Toolbox/flow_images_float_clip10")
    parser.add_argument("--image-root", default="../H36M-Toolbox/images_crop")
    parser.add_argument("--out-dir", default="debug_vis/single_point_flow_failure_clip10_vis")
    parser.add_argument("--case", default="C_2d_off_wrong_motion")
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-candidates", type=int, default=8000)
    parser.add_argument("--min-2d-gap", type=float, default=6.0)
    parser.add_argument("--min-center-error", type=float, default=2.0)
    parser.add_argument("--min-gt-gain", type=float, default=1.0)
    parser.add_argument("--min-flow-delta", type=float, default=1.0)
    parser.add_argument("--min-target-mag", type=float, default=1.0)
    parser.add_argument(
        "--allow-gt-bad",
        action="store_true",
        help="Keep failures even when the GT-site flow is not better than the CPN-site flow.",
    )
    return parser.parse_args()


def read_csv_rows(path, case_name, limit):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if case_name and row.get("case") != case_name:
                continue
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def labels_by_key(labels):
    return {
        (
            int(item["subject"]),
            int(item["action"]),
            int(item["subaction"]),
            int(item["camera_id"]),
            int(item["image_id"]),
        ): item
        for item in labels
    }


def load_rgb(image_root, row):
    path = Path(image_root) / row["seq"] / "{}_{:06d}.jpg".format(row["seq"], int(row["frame_id"]))
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


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


def draw_skeleton(ax, joints, color, linewidth=1.1, alpha=0.9):
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


def arrow(ax, xy, vec, color, label, scale=4.0, width=0.35, head_width=3.0):
    ax.arrow(
        float(xy[0]),
        float(xy[1]),
        float(vec[0]) * scale,
        float(vec[1]) * scale,
        color=color,
        width=width,
        head_width=head_width,
        length_includes_head=True,
        alpha=0.96,
        label=label,
    )


def target_motion_for(cur, prev, output_size):
    cur_gt = np.asarray(cur["joints_2d_gt_crop"], dtype=np.float32)
    prev_gt = np.asarray(prev["joints_2d_gt_crop"], dtype=np.float32)
    prev_to_raw = get_affine_transform(prev["center"], prev["scale"], output_size, inv=True)
    raw_to_cur = get_affine_transform(cur["center"], cur["scale"], output_size, inv=False)
    prev_gt_same = affine_apply(raw_to_cur, affine_apply(prev_to_raw, prev_gt))
    return prev_gt_same - cur_gt


def enrich_row(row, by_key, flow_dir, output_size, args):
    key = (
        int(row["subject"]),
        int(row["action"]),
        int(row["subaction"]),
        int(row["camera_id"]),
        int(row["frame_id"]),
    )
    prev_key = key[:-1] + (key[-1] - args.frame_gap,)
    cur = by_key.get(key)
    prev = by_key.get(prev_key)
    if cur is None or prev is None:
        return None

    seq = seq_name(cur)
    flow_path = Path(flow_dir) / seq / (image_stem(seq, key[-1]) + ".npy")
    if not flow_path.exists():
        return None

    flow = np.load(flow_path).astype(np.float32)
    cpn = np.asarray(cur["joints_2d_cpn_crop"], dtype=np.float32)
    gt = np.asarray(cur["joints_2d_gt_crop"], dtype=np.float32)
    joint = int(row["joint"])

    center_flow = bilinear_sample_flow(flow, cpn[[joint]])[0]
    gt_flow = bilinear_sample_flow(flow, gt[[joint]])[0]
    target = target_motion_for(cur, prev, output_size)[joint]

    d_xy = float(np.linalg.norm(cpn[joint] - gt[joint]))
    e_center = float(np.linalg.norm(center_flow - target))
    e_gt = float(np.linalg.norm(gt_flow - target))
    gain = e_center - e_gt
    flow_delta = float(np.linalg.norm(center_flow - gt_flow))
    target_mag = float(np.linalg.norm(target))
    center_mag = float(np.linalg.norm(center_flow))

    if d_xy < args.min_2d_gap:
        return None
    if e_center < args.min_center_error:
        return None
    if flow_delta < args.min_flow_delta:
        return None
    if target_mag < args.min_target_mag:
        return None
    if not args.allow_gt_bad and gain < args.min_gt_gain:
        return None

    return {
        "row": row,
        "cur": cur,
        "flow": flow,
        "cpn": cpn,
        "gt": gt,
        "joint": joint,
        "center_flow": center_flow,
        "gt_flow": gt_flow,
        "target": target,
        "d_xy": d_xy,
        "e_center": e_center,
        "e_gt": e_gt,
        "gain": gain,
        "flow_delta": flow_delta,
        "target_mag": target_mag,
        "center_mag": center_mag,
        "score": gain * 2.0 + flow_delta + min(d_xy, 30.0) * 0.05,
    }


def set_image_axes(ax, width, height):
    ax.set_xlim(0, width - 1)
    ax.set_ylim(height - 1, 0)
    ax.axis("off")


def visualize(case, image_root, output_path, width, height):
    row = case["row"]
    joint = case["joint"]
    cpn = case["cpn"]
    gt = case["gt"]
    flow = case["flow"]
    flow_rgb = flow_to_rgb(flow)
    image = load_rgb(image_root, row)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    fig.patch.set_facecolor("white")

    if image is not None:
        plot_cpn = cpn.copy()
        plot_gt = gt.copy()
        image_h, image_w = image.shape[:2]
        if (image_w, image_h) != (width, height):
            scale = np.array([image_w / float(width), image_h / float(height)], dtype=np.float32)
            plot_cpn *= scale
            plot_gt *= scale
        axes[0].imshow(image)
        draw_skeleton(axes[0], plot_cpn, "#4169e1", linewidth=1.15, alpha=0.95)
        draw_skeleton(axes[0], plot_gt, "#ff3030", linewidth=1.15, alpha=0.95)
        axes[0].scatter(plot_cpn[joint, 0], plot_cpn[joint, 1], marker="D", c="#4169e1", s=70, label="CPN")
        axes[0].scatter(plot_gt[joint, 0], plot_gt[joint, 1], marker="D", c="#ff3030", s=70, label="GT")
        axes[0].legend(loc="lower right", fontsize=8, framealpha=0.9)
        axes[0].axis("off")
    axes[0].set_title("{} {} f{}".format(row["joint_name"], row["action_name"], row["frame_id"]), fontsize=11)

    axes[1].imshow(flow_rgb)
    axes[1].scatter(cpn[joint, 0], cpn[joint, 1], marker="D", c="#4169e1", s=82, label="CPN point")
    axes[1].scatter(gt[joint, 0], gt[joint, 1], marker="D", c="#ff3030", s=82, label="GT point")
    axes[1].plot([cpn[joint, 0], gt[joint, 0]], [cpn[joint, 1], gt[joint, 1]], color="white", linewidth=1.2)
    arrow(axes[1], cpn[joint], case["center_flow"], "#4169e1", "CPN sampled flow")
    arrow(axes[1], gt[joint], case["gt_flow"], "#ff3030", "GT-site flow")
    arrow(axes[1], gt[joint], case["target"], "#111111", "GT motion target", scale=4.0, width=0.25, head_width=2.5)
    set_image_axes(axes[1], width, height)
    axes[1].legend(loc="lower right", fontsize=7, framealpha=0.92)
    axes[1].set_title(
        "single point err CPN/GT {:.2f}/{:.2f}, gain {:.2f}".format(
            case["e_center"], case["e_gt"], case["gain"]
        ),
        fontsize=10,
    )

    pad = 28.0
    xmin = max(0.0, min(float(cpn[joint, 0]), float(gt[joint, 0])) - pad)
    xmax = min(float(width - 1), max(float(cpn[joint, 0]), float(gt[joint, 0])) + pad)
    ymin = max(0.0, min(float(cpn[joint, 1]), float(gt[joint, 1])) - pad)
    ymax = min(float(height - 1), max(float(cpn[joint, 1]), float(gt[joint, 1])) + pad)
    if xmax - xmin < 56:
        cx = 0.5 * (xmin + xmax)
        xmin = max(0.0, cx - 28)
        xmax = min(float(width - 1), cx + 28)
    if ymax - ymin < 56:
        cy = 0.5 * (ymin + ymax)
        ymin = max(0.0, cy - 28)
        ymax = min(float(height - 1), cy + 28)

    axes[2].imshow(flow_rgb)
    axes[2].scatter(cpn[joint, 0], cpn[joint, 1], marker="D", c="#4169e1", s=105)
    axes[2].scatter(gt[joint, 0], gt[joint, 1], marker="D", c="#ff3030", s=105)
    axes[2].plot([cpn[joint, 0], gt[joint, 0]], [cpn[joint, 1], gt[joint, 1]], color="white", linewidth=1.4)
    arrow(axes[2], cpn[joint], case["center_flow"], "#4169e1", "CPN sampled flow", scale=4.0, width=0.25, head_width=2.5)
    arrow(axes[2], gt[joint], case["gt_flow"], "#ff3030", "GT-site flow", scale=4.0, width=0.25, head_width=2.5)
    arrow(axes[2], gt[joint], case["target"], "#111111", "GT motion target", scale=4.0, width=0.2, head_width=2.2)
    axes[2].set_xlim(xmin, xmax)
    axes[2].set_ylim(ymax, ymin)
    axes[2].axis("off")
    axes[2].set_title(
        "2D gap {:.1f}px, flow gap {:.1f}px, |target| {:.1f}px".format(
            case["d_xy"], case["flow_delta"], case["target_mag"]
        ),
        fontsize=10,
    )

    fig.tight_layout(pad=0.9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_contact_sheet(paths, output_path, thumb_width=720):
    images = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = thumb_width / float(w)
        img = cv2.resize(img, (thumb_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
        images.append(img)
    if not images:
        return
    rows = []
    for idx in range(0, len(images), 2):
        row_imgs = images[idx : idx + 2]
        if len(row_imgs) == 1:
            blank = np.full_like(row_imgs[0], 255)
            row_imgs.append(blank)
        rows.append(np.concatenate(row_imgs, axis=1))
    sheet = np.concatenate(rows, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def main():
    args = parse_args()
    labels = load_labels(args.labels)
    by_key = labels_by_key(labels)
    output_size = (args.width, args.height)

    csv_rows = read_csv_rows(args.csv, args.case, args.max_candidates)
    cases = []
    for row in csv_rows:
        enriched = enrich_row(row, by_key, args.flow_dir, output_size, args)
        if enriched is not None:
            cases.append(enriched)

    cases.sort(key=lambda item: item["score"], reverse=True)
    selected = cases[: args.top_k]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    summary = []
    for idx, case in enumerate(selected, 1):
        row = case["row"]
        name = "{:02d}_{}_{}_f{}_{}.png".format(
            idx,
            row["action_name"].replace(" ", ""),
            row["joint_name"],
            row["frame_id"],
            row["case"].split("_", 1)[0],
        )
        out_path = out_dir / name
        visualize(case, args.image_root, out_path, args.width, args.height)
        written.append(out_path)
        summary.append(
            {
                "file": str(out_path),
                "seq": row["seq"],
                "action": row["action_name"],
                "frame_id": int(row["frame_id"]),
                "joint": row["joint_name"],
                "case": row["case"],
                "d_xy": case["d_xy"],
                "single_point_center_error": case["e_center"],
                "gt_site_error": case["e_gt"],
                "gt_site_gain": case["gain"],
                "flow_delta_cpn_gt": case["flow_delta"],
                "target_motion_mag": case["target_mag"],
                "center_flow_mag": case["center_mag"],
            }
        )

    make_contact_sheet(written[: min(len(written), 8)], out_dir / "contact_sheet_top8.png")

    summary_path = out_dir / "single_point_failure_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "csv": str(args.csv),
                "case_filter": args.case,
                "num_csv_candidates": len(csv_rows),
                "num_after_point_filter": len(cases),
                "num_written": len(written),
                "filters": {
                    "min_2d_gap": args.min_2d_gap,
                    "min_center_error": args.min_center_error,
                    "min_gt_gain": args.min_gt_gain,
                    "min_flow_delta": args.min_flow_delta,
                    "min_target_mag": args.min_target_mag,
                    "allow_gt_bad": args.allow_gt_bad,
                },
                "items": summary,
            },
            file,
            indent=2,
        )

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "summary": str(summary_path),
                "contact_sheet": str(out_dir / "contact_sheet_top8.png"),
                "num_csv_candidates": len(csv_rows),
                "num_after_point_filter": len(cases),
                "num_written": len(written),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
