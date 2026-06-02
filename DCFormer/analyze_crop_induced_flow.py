import argparse
import csv
import math
import pickle
import re
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_LABELS = SCRIPT_DIR / "data" / "h36m_validation.pkl"
DEFAULT_FLOW_DIR = REPO_ROOT / "H36M-Toolbox" / "flow_raft_bwd_k4_fp16"
DEFAULT_IMAGES_DIR = REPO_ROOT / "H36M-Toolbox" / "images"
DEFAULT_IMAGES_CROP_DIR = REPO_ROOT / "H36M-Toolbox" / "images_crop"
DEFAULT_OUT_DIR = REPO_ROOT / "H36M-Toolbox" / "flow_crop_artifact_analysis"


def parse_seq(seq):
    match = re.fullmatch(r"s_(\d+)_act_(\d+)_subact_(\d+)_ca_(\d+)", seq)
    if not match:
        raise ValueError(f"Bad sequence name: {seq}")
    subject, action, subaction, camera = map(int, match.groups())
    return subject, action, subaction, camera - 1


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center, scale, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    center = np.array(center)
    scale = np.array(scale)
    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w, dst_h = output_size

    src_dir = np.array([0, (src_w - 1) * -0.5], np.float32)
    dst_dir = np.array([0, (dst_w - 1) * -0.5], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]
    dst[1, :] = np.array([(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]) + dst_dir
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def affine_apply(trans, xy):
    ones = np.ones((xy.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([xy.astype(np.float32), ones], axis=1)
    return homo @ trans.T


def mag_stats(mag):
    return {
        "mean": float(mag.mean()),
        "p50": float(np.percentile(mag, 50)),
        "p95": float(np.percentile(mag, 95)),
        "p99": float(np.percentile(mag, 99)),
        "max": float(mag.max()),
    }


def image_name(seq, frame_id):
    return f"{seq}_{frame_id:06d}.jpg"


def crop_raw_image(path, center, scale, output_size):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise OSError(f"Failed to read image: {path}")
    trans = get_affine_transform(center, scale, output_size)
    return cv2.warpAffine(image, trans, output_size, flags=cv2.INTER_LINEAR)


def make_preview(args, seq, rows, label_by_frame):
    out_dir = Path(args.out_dir).expanduser().resolve() / seq
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images_dir).expanduser().resolve() / seq
    images_crop_dir = Path(args.images_crop_dir).expanduser().resolve() / seq
    output_size = (args.width, args.height)

    selected = rows[: args.preview_count]
    tiles = []
    for row in selected:
        frame = row["frame"]
        prev_frame = row["prev_frame"]
        cur_shot = label_by_frame[frame]
        prev_name = image_name(seq, prev_frame)
        cur_name = image_name(seq, frame)

        prev_ind = cv2.imread(str(images_crop_dir / prev_name), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        cur = cv2.imread(str(images_crop_dir / cur_name), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        prev_same = crop_raw_image(images_dir / prev_name, cur_shot["center"], cur_shot["scale"], output_size)
        if prev_ind is None or cur is None:
            continue

        diff_ind = cv2.convertScaleAbs(cv2.absdiff(cur, prev_ind), alpha=3.0)
        diff_same = cv2.convertScaleAbs(cv2.absdiff(cur, prev_same), alpha=3.0)

        panels = [
            label(prev_ind, f"prev independent {prev_frame:06d}"),
            label(cur, f"current {frame:06d}"),
            label(prev_same, f"prev using current crop"),
            label(diff_ind, "diff independent x3"),
            label(diff_same, "diff same-crop x3"),
        ]
        tiles.append(np.concatenate(panels, axis=1))

    if not tiles:
        return None
    sheet = np.concatenate(tiles, axis=0)
    out = out_dir / f"{seq}_gap{args.frame_gap}_crop_artifact_preview.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out


def label(image, text):
    image = image.copy()
    h, w = image.shape[:2]
    cv2.rectangle(image, (0, 0), (w, 24), (0, 0, 0), -1)
    cv2.putText(image, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser(description="Analyze crop-induced pseudo flow from per-frame H36M crop affines.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--seq", default="s_09_act_02_subact_01_ca_01")
    parser.add_argument("--frame-start", type=int, default=5)
    parser.add_argument("--frame-end", type=int, default=120)
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--grid-step", type=int, default=8)
    parser.add_argument("--flow-dir", default=str(DEFAULT_FLOW_DIR))
    parser.add_argument("--images-dir", default=str(DEFAULT_IMAGES_DIR))
    parser.add_argument("--images-crop-dir", default=str(DEFAULT_IMAGES_CROP_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--csv-name", default=None)
    parser.add_argument("--preview-count", type=int, default=12)
    args = parser.parse_args()

    seq_key = parse_seq(args.seq)
    labels_path = Path(args.labels).expanduser().resolve()
    flow_dir = Path(args.flow_dir).expanduser().resolve() / args.seq
    out_dir = Path(args.out_dir).expanduser().resolve() / args.seq
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading labels: {labels_path}")
    with open(labels_path, "rb") as file:
        labels = pickle.load(file)
    print(f"Loaded labels: {len(labels)}")

    label_by_frame = {}
    for shot in labels:
        key = (shot["subject"], shot["action"], shot["subaction"], shot["camera_id"])
        if key == seq_key:
            label_by_frame[int(shot["image_id"])] = shot
    print(f"Sequence labels: {len(label_by_frame)} for {args.seq}")

    xs = np.arange(args.grid_step / 2, args.width, args.grid_step, dtype=np.float32)
    ys = np.arange(args.grid_step / 2, args.height, args.grid_step, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    output_size = (args.width, args.height)

    rows = []
    for frame in range(args.frame_start, args.frame_end + 1):
        prev_frame = frame - args.frame_gap
        if frame not in label_by_frame or prev_frame not in label_by_frame:
            continue
        cur = label_by_frame[frame]
        prev = label_by_frame[prev_frame]
        t_cur_inv = get_affine_transform(cur["center"], cur["scale"], output_size, inv=1)
        t_prev = get_affine_transform(prev["center"], prev["scale"], output_size)
        raw_xy = affine_apply(t_cur_inv, grid)
        prev_crop_xy = affine_apply(t_prev, raw_xy)
        induced = prev_crop_xy - grid
        induced_mag = np.sqrt((induced ** 2).sum(axis=1))
        induced_stats = mag_stats(induced_mag)

        flow_path = flow_dir / f"{args.seq}_{frame:06d}.npy"
        flow_stats = None
        if flow_path.exists():
            flow = np.load(flow_path).astype(np.float32)
            flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            flow_stats = mag_stats(flow_mag)

        center_delta = np.array(cur["center"], dtype=np.float32) - np.array(prev["center"], dtype=np.float32)
        scale_ratio = np.array(cur["scale"], dtype=np.float32) / np.array(prev["scale"], dtype=np.float32)
        row = {
            "frame": frame,
            "prev_frame": prev_frame,
            "center_dx": float(center_delta[0]),
            "center_dy": float(center_delta[1]),
            "scale_rx": float(scale_ratio[0]),
            "scale_ry": float(scale_ratio[1]),
            **{f"crop_{k}": v for k, v in induced_stats.items()},
        }
        if flow_stats is not None:
            row.update({f"raft_{k}": v for k, v in flow_stats.items()})
            row["crop_p95_over_raft_p95"] = induced_stats["p95"] / (flow_stats["p95"] + 1e-6)
        rows.append(row)

    if not rows:
        print("No rows matched.")
        return

    csv_name = args.csv_name or f"{args.seq}_gap{args.frame_gap}_crop_artifact.csv"
    csv_path = out_dir / csv_name
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote CSV: {csv_path}")
    print("\nFirst rows:")
    for row in rows[:15]:
        print(
            f"{row['frame']:06d}: crop_p95={row['crop_p95']:.3f} "
            f"raft_p95={row.get('raft_p95', float('nan')):.3f} "
            f"ratio={row.get('crop_p95_over_raft_p95', float('nan')):.2f} "
            f"center_delta=({row['center_dx']:.2f},{row['center_dy']:.2f})"
        )

    print("\nTop crop-induced p95:")
    for row in sorted(rows, key=lambda item: item["crop_p95"], reverse=True)[:15]:
        print(
            f"{row['frame']:06d}: crop_p95={row['crop_p95']:.3f} "
            f"raft_p95={row.get('raft_p95', float('nan')):.3f} "
            f"ratio={row.get('crop_p95_over_raft_p95', float('nan')):.2f} "
            f"scale=({row['scale_rx']:.4f},{row['scale_ry']:.4f})"
        )

    preview = make_preview(args, args.seq, rows, label_by_frame)
    if preview:
        print(f"Preview: {preview}")


if __name__ == "__main__":
    main()
