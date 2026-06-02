import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


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
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))

    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def crop_image(image, center, scale, output_size):
    trans = get_affine_transform(center, scale, 0, output_size)
    return cv2.warpAffine(image, trans, output_size, flags=cv2.INTER_LINEAR)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ASFNet Human3.6M cropped images for train/val splits.")
    parser.add_argument(
        "--train-labels",
        default=str(DCFORMER_ROOT / "data" / "h36m_train.pkl"),
        help="Path to h36m_train.pkl",
    )
    parser.add_argument(
        "--val-labels",
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
        help="Root directory to write cropped images",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val"],
        default=["train", "val"],
        help="Which splits to process.",
    )
    parser.add_argument("--width", type=int, default=192, help="Output crop width")
    parser.add_argument("--height", type=int, default=256, help="Output crop height")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cropped files instead of skipping them.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Only process the first N images of each selected split.",
    )
    return parser.parse_args()


def load_labels(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def iter_progress(items, desc):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, unit="img")


def process_labels(labels, input_dir, output_dir, output_size, overwrite=False, max_images=None, desc="crop"):
    total = len(labels) if max_images is None else min(len(labels), max_images)
    processed = 0
    skipped = 0
    failed = 0

    iterator = iter_progress(range(total), desc=desc)
    for idx in iterator:
        shot = labels[idx]
        subdir = f"s_{shot['subject']:02d}_act_{shot['action']:02d}_subact_{shot['subaction']:02d}_ca_{shot['camera_id'] + 1:02d}"
        imagename = (
            f"s_{shot['subject']:02d}_act_{shot['action']:02d}_subact_{shot['subaction']:02d}_"
            f"ca_{shot['camera_id'] + 1:02d}_{shot['image_id']:06d}.jpg"
        )

        input_file = input_dir / subdir / imagename
        output_subdir = output_dir / subdir
        output_file = output_subdir / imagename
        output_subdir.mkdir(parents=True, exist_ok=True)

        if output_file.exists() and not overwrite:
            skipped += 1
            continue

        image = cv2.imread(str(input_file), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            failed += 1
            print(f"Warning: failed to read {input_file}")
            continue

        image = crop_image(image, shot["center"], shot["scale"], output_size)
        if not cv2.imwrite(str(output_file), image):
            failed += 1
            print(f"Warning: failed to write {output_file}")
            continue

        processed += 1

    return processed, skipped, failed, total


def main():
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_size = (args.width, args.height)

    split_to_path = {
        "train": Path(args.train_labels).expanduser().resolve(),
        "val": Path(args.val_labels).expanduser().resolve(),
    }

    grand_processed = 0
    grand_skipped = 0
    grand_failed = 0

    for split in args.splits:
        labels_path = split_to_path[split]
        labels = load_labels(labels_path)
        print(f"Loaded {split} labels from {labels_path}: {len(labels)} samples")
        processed, skipped, failed, total = process_labels(
            labels=labels,
            input_dir=input_dir,
            output_dir=output_dir,
            output_size=output_size,
            overwrite=args.overwrite,
            max_images=args.max_images,
            desc=f"crop-{split}",
        )
        grand_processed += processed
        grand_skipped += skipped
        grand_failed += failed
        print(
            f"{split}: total={total} processed={processed} skipped={skipped} failed={failed} output_dir={output_dir}"
        )

    print(
        f"Done. processed={grand_processed} skipped={grand_skipped} failed={grand_failed} output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
