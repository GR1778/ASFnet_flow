import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np


DCFORMER_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DCFORMER_ROOT.parent


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    center = np.array(center)
    scale = np.array(scale)

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

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
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def crop_image(image, center, scale, output_size):
    trans = get_affine_transform(center, scale, 0, output_size)
    image = cv2.warpAffine(image, trans, output_size, flags=cv2.INTER_LINEAR)
    return image


def parse_args():
    parser = argparse.ArgumentParser(description="Crop Human3.6M validation images only for ASFNet evaluation.")
    parser.add_argument(
        "--labels",
        default=str(DCFORMER_ROOT / "data" / "h36m_validation.pkl"),
        help="Path to h36m_validation.pkl",
    )
    parser.add_argument(
        "--input-dir",
        default=str(REPO_ROOT / "H36M-Toolbox" / "images"),
        help="Root directory of extracted Human3.6M RGB frames",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "H36M-Toolbox" / "images_crop"),
        help="Root directory to write cropped validation images",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=192,
        help="Output crop width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Output crop height",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cropped files instead of skipping them",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    labels_path = Path(args.labels).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(labels_path, "rb") as file:
        labels = pickle.load(file)

    processed = 0
    skipped = 0
    failed = 0
    output_size = (args.width, args.height)

    for idx, shot in enumerate(labels):
        subdir = f"s_{shot['subject']:02d}_act_{shot['action']:02d}_subact_{shot['subaction']:02d}_ca_{shot['camera_id'] + 1:02d}"
        imagename = (
            f"s_{shot['subject']:02d}_act_{shot['action']:02d}_subact_{shot['subaction']:02d}_"
            f"ca_{shot['camera_id'] + 1:02d}_{shot['image_id']:06d}.jpg"
        )

        input_file = input_dir / subdir / imagename
        output_subdir = output_dir / subdir
        output_file = output_subdir / imagename
        output_subdir.mkdir(parents=True, exist_ok=True)

        if output_file.exists() and not args.overwrite:
            skipped += 1
            if (idx + 1) % 1000 == 0:
                print(f"[{idx + 1}/{len(labels)}] processed={processed} skipped={skipped} failed={failed}")
            continue

        image = cv2.imread(str(input_file), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            failed += 1
            print(f"Warning: failed to read {input_file}")
            if (idx + 1) % 1000 == 0:
                print(f"[{idx + 1}/{len(labels)}] processed={processed} skipped={skipped} failed={failed}")
            continue

        image = crop_image(image, shot["center"], shot["scale"], output_size)

        if not cv2.imwrite(str(output_file), image):
            failed += 1
            print(f"Warning: failed to write {output_file}")
        else:
            processed += 1

        if (idx + 1) % 1000 == 0:
            print(f"[{idx + 1}/{len(labels)}] processed={processed} skipped={skipped} failed={failed}")

    print(
        f"Done. total={len(labels)} processed={processed} skipped={skipped} failed={failed} output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
