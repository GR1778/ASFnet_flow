import argparse
import re
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate dense optical flow for one cropped image sequence and "
            "save flow visualization PNGs."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="/home/SIMON/26HPE/ASFnet/H36M-Toolbox/images_crop/s_01_act_02_subact_01_ca_01",
        help="Directory that contains one ordered image sequence.",
    )
    parser.add_argument(
        "--flow-dir",
        default="/home/SIMON/26HPE/ASFnet/H36M-Toolbox/flow_farneback/s_01_act_02_subact_01_ca_01",
        help="Directory to save .npy flow files (H, W, 2).",
    )
    parser.add_argument(
        "--vis-dir",
        default="/home/SIMON/26HPE/ASFnet/H36M-Toolbox/flow_farneback_vis/s_01_act_02_subact_01_ca_01",
        help="Directory to save visualization PNGs.",
    )
    parser.add_argument(
        "--direction",
        choices=("backward", "forward"),
        default="backward",
        help=(
            "Flow direction. backward: flow(t <- t-1) via Farneback(curr, prev). "
            "forward: flow(t-1 -> t) via Farneback(prev, curr)."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Only process the first N frames in sequence.",
    )
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=99.0,
        help=(
            "Percentile for magnitude clipping in visualization (robust against "
            "a few extreme vectors)."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip frame if both .npy and .png outputs already exist.",
    )
    return parser.parse_args()


def parse_frame_id(path):
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def image_sort_key(path):
    frame_id = parse_frame_id(path)
    return frame_id if frame_id is not None else path.name


def collect_images(input_dir):
    images = [path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(images, key=image_sort_key)


def flow_to_vis(flow_hw2, clip_percentile):
    u = flow_hw2[..., 0]
    v = flow_hw2[..., 1]
    mag, ang = cv2.cartToPolar(u, v, angleInDegrees=False)

    clip_value = float(np.percentile(mag, clip_percentile))
    clip_value = max(clip_value, 1e-6)
    mag_norm = np.clip(mag / clip_value, 0.0, 1.0)

    hsv = np.zeros((flow_hw2.shape[0], flow_hw2.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = np.uint8(ang * 180.0 / np.pi / 2.0)
    hsv[..., 1] = 255
    hsv[..., 2] = np.uint8(mag_norm * 255.0)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def compute_farneback(gray_prev, gray_cur, direction):
    if direction == "backward":
        src0, src1 = gray_cur, gray_prev
    else:
        src0, src1 = gray_prev, gray_cur

    return cv2.calcOpticalFlowFarneback(
        src0,
        src1,
        None,
        pyr_scale=0.5,
        levels=4,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def main():
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    flow_dir = Path(args.flow_dir).expanduser().resolve()
    vis_dir = Path(args.vis_dir).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir does not exist: {input_dir}")

    images = collect_images(input_dir)
    if args.max_frames is not None:
        images = images[: args.max_frames]
    if len(images) < 2:
        raise RuntimeError(f"Need at least 2 images, found {len(images)}")

    flow_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input dir: {input_dir}")
    print(f"Flow dir:  {flow_dir}")
    print(f"Vis dir:   {vis_dir}")
    print(f"Frames:    {len(images)}")
    print(f"Direction: {args.direction}")
    print(f"Clip pctl: {args.clip_percentile}")

    processed = 0
    skipped = 0

    for idx in range(1, len(images)):
        cur = images[idx]
        prev = images[idx - 1]

        frame_id = parse_frame_id(cur)
        out_stem = cur.stem if frame_id is None else f"{cur.stem.rsplit('_', 1)[0]}_{frame_id:06d}"
        flow_path = flow_dir / f"{out_stem}.npy"
        vis_path = vis_dir / f"{out_stem}.png"

        if args.skip_existing and flow_path.exists() and vis_path.exists():
            skipped += 1
            continue

        prev_img = cv2.imread(str(prev), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
        cur_img = cv2.imread(str(cur), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
        if prev_img is None or cur_img is None:
            print(f"Warning: read failed for pair: {prev.name} -> {cur.name}")
            continue

        flow = compute_farneback(prev_img, cur_img, args.direction).astype(np.float32)
        vis = flow_to_vis(flow, args.clip_percentile)

        np.save(flow_path, flow)
        cv2.imwrite(str(vis_path), vis)
        processed += 1

        if processed % 100 == 0:
            print(f"processed={processed} skipped={skipped}")

    print(f"Done. processed={processed} skipped={skipped}")


if __name__ == "__main__":
    main()
