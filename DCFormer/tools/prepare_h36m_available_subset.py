#!/usr/bin/env python3

import argparse
import json
import math
import os
import pickle
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a Human3.6M subset aligned to the videos currently available "
            "under H36M-Toolbox/images/S*/Videos, then optionally extract frames "
            "and generate cropped images for ASFnet."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="ASFnet repository root",
    )
    parser.add_argument(
        "--train-labels",
        default="h36m_train.pkl",
        help="Input training labels under DCFormer/data",
    )
    parser.add_argument(
        "--val-labels",
        default="h36m_validation.pkl",
        help="Input validation labels under DCFormer/data",
    )
    parser.add_argument(
        "--output-prefix",
        default="h36m_available",
        help="Output prefix for filtered subset labels and manifests",
    )
    parser.add_argument(
        "--subjects",
        default="",
        help="Optional comma-separated subject filter, e.g. S1,S5,S7,S8,S11",
    )
    parser.add_argument(
        "--write-subset",
        action="store_true",
        help="Write filtered subset labels and summary manifests",
    )
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="Extract JPG frames for the available subset videos",
    )
    parser.add_argument(
        "--crop-images",
        action="store_true",
        help="Generate 256x192 cropped RGB images for the subset labels",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip frame/crop outputs that already exist",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel ffmpeg jobs for frame extraction",
    )
    parser.add_argument(
        "--limit-combos",
        type=int,
        default=None,
        help="Process only the first N video combinations (useful for testing)",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="ffmpeg executable",
    )
    parser.add_argument(
        "--ffmpeg-loglevel",
        default="error",
        help="ffmpeg log level",
    )

    args = parser.parse_args()
    args.subjects = [
        subject.strip().upper()
        for subject in args.subjects.split(",")
        if subject.strip()
    ]
    if not any((args.write_subset, args.extract_frames, args.crop_images)):
        args.write_subset = True
    return args


def load_metadata(toolbox_dir: Path):
    metadata_path = toolbox_dir / "metadata.xml"
    import xml.etree.ElementTree as ET

    class Metadata:
        def __init__(self, metadata_file: Path):
            self.subjects = []
            self.sequence_mappings = {}
            self.action_names = {}
            self.camera_ids = []

            tree = ET.parse(metadata_file)
            root = tree.getroot()

            for i, tr in enumerate(root.find("mapping")):
                if i == 0:
                    _, _, *self.subjects = [td.text for td in tr]
                    self.sequence_mappings = {subject: {} for subject in self.subjects}
                elif i < 33:
                    action_id, subaction_id, *prefixes = [td.text for td in tr]
                    for subject, prefix in zip(self.subjects, prefixes):
                        self.sequence_mappings[subject][(action_id, subaction_id)] = prefix

            for i, elem in enumerate(root.find("actionnames")):
                self.action_names[str(i + 1)] = elem.text

            self.camera_ids = [elem.text for elem in root.find("dbcameras/index2id")]

        def get_base_filename(self, subject, action, subaction, camera):
            return f"{self.sequence_mappings[subject][(action, subaction)]}.{camera}"

    return Metadata(metadata_path)


def combo_key(label):
    return (
        int(label["subject"]),
        int(label["action"]),
        int(label["subaction"]),
        int(label["camera_id"]) + 1,
    )


def combo_dirname(combo):
    subject, action, subaction, camera = combo
    return f"s_{subject:02d}_act_{action:02d}_subact_{subaction:02d}_ca_{camera:02d}"


def combo_to_video_rel(metadata, combo):
    subject, action, subaction, camera = combo
    basename = metadata.get_base_filename(
        f"S{subject}", str(action), str(subaction), metadata.camera_ids[camera - 1]
    )
    return Path(f"S{subject}") / "Videos" / f"{basename}.mp4"


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def dump_pickle(path: Path, data):
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=4)


def filter_labels_by_subjects(labels, subjects):
    if not subjects:
        return labels

    subject_ids = set()
    for subject in subjects:
        if not subject.startswith("S"):
            raise ValueError(f"Invalid subject '{subject}', expected values like S1 or S11")
        subject_ids.add(int(subject[1:]))

    return [label for label in labels if int(label["subject"]) in subject_ids]


def build_subset(labels, metadata, video_root: Path):
    combo_counts = Counter()
    combo_max_image = defaultdict(int)
    for label in labels:
        key = combo_key(label)
        combo_counts[key] += 1
        combo_max_image[key] = max(combo_max_image[key], int(label["image_id"]))

    available = {}
    missing = {}
    for key in sorted(combo_counts):
        video_rel = combo_to_video_rel(metadata, key)
        video_path = video_root / video_rel
        if video_path.exists():
            available[key] = {
                "video_rel": str(video_rel),
                "video_path": str(video_path),
                "required_frames": combo_counts[key],
                "max_image_id": combo_max_image[key],
            }
        else:
            missing[key] = {
                "video_rel": str(video_rel),
                "required_frames": combo_counts[key],
                "max_image_id": combo_max_image[key],
            }

    subset_labels = [label for label in labels if combo_key(label) in available]
    return subset_labels, available, missing


def build_summary(split_name, original_labels, subset_labels, available, missing):
    original_combos = Counter(combo_key(label) for label in original_labels)
    subset_combos = Counter(combo_key(label) for label in subset_labels)
    subject_summary = {}
    for subject in [1, 5, 6, 7, 8, 9, 11]:
        original_subject_combos = sum(1 for combo in original_combos if combo[0] == subject)
        subset_subject_combos = sum(1 for combo in subset_combos if combo[0] == subject)
        original_subject_samples = sum(count for combo, count in original_combos.items() if combo[0] == subject)
        subset_subject_samples = sum(count for combo, count in subset_combos.items() if combo[0] == subject)
        subject_summary[f"S{subject}"] = {
            "original_combos": original_subject_combos,
            "subset_combos": subset_subject_combos,
            "original_samples": original_subject_samples,
            "subset_samples": subset_subject_samples,
        }

    return {
        "split": split_name,
        "original_samples": len(original_labels),
        "subset_samples": len(subset_labels),
        "original_combos": len(original_combos),
        "subset_combos": len(subset_combos),
        "missing_combos": len(missing),
        "subject_summary": subject_summary,
    }


def write_text_manifest(path: Path, split_name: str, available, missing):
    with path.open("w") as f:
        f.write(f"[{split_name}] available={len(available)} missing={len(missing)}\n")
        f.write("\n[available]\n")
        for combo, info in available.items():
            f.write(
                f"{combo_dirname(combo)} -> {info['video_rel']} "
                f"(frames={info['required_frames']}, max_image_id={info['max_image_id']})\n"
            )
        f.write("\n[missing]\n")
        for combo, info in missing.items():
            f.write(
                f"{combo_dirname(combo)} -> {info['video_rel']} "
                f"(frames={info['required_frames']}, max_image_id={info['max_image_id']})\n"
            )


def get_affine_transform(center, scale, output_size):
    center = np.array(center, dtype=np.float32)
    scale = np.array(scale, dtype=np.float32)

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = float(output_size[0])
    dst_h = float(output_size[1])

    src_dir = np.array([0, (src_w - 1) * -0.5], dtype=np.float32)
    dst_dir = np.array([0, (dst_w - 1) * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]
    dst[1, :] = np.array([(dst_w - 1) * 0.5, (dst_h - 1) * 0.5], dtype=np.float32) + dst_dir

    src[2:, :] = src[1, :] + np.array([-(src[0, 1] - src[1, 1]), src[0, 0] - src[1, 0]], dtype=np.float32)
    dst[2:, :] = dst[1, :] + np.array([-(dst[0, 1] - dst[1, 1]), dst[0, 0] - dst[1, 0]], dtype=np.float32)

    return cv2.getAffineTransform(src, dst)


def crop_image(image, center, scale, output_size):
    trans = get_affine_transform(center, scale, output_size)
    return cv2.warpAffine(image, trans, output_size, flags=cv2.INTER_LINEAR)


def unique_combo_jobs(available_map, limit=None):
    combos = sorted(available_map)
    if limit is not None:
        combos = combos[:limit]
    return combos


def extract_one_combo(ffmpeg_bin, ffmpeg_loglevel, video_root, image_root, combo, info, skip_existing):
    target_dir = image_root / combo_dirname(combo)
    target_dir.mkdir(parents=True, exist_ok=True)
    required_last_frame = target_dir / f"{combo_dirname(combo)}_{info['max_image_id']:06d}.jpg"
    if skip_existing and required_last_frame.exists():
        return combo, "skipped"

    output_pattern = target_dir / f"{combo_dirname(combo)}_%06d.jpg"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-nostats",
        "-loglevel",
        ffmpeg_loglevel,
        "-i",
        str(video_root / info["video_rel"]),
        "-qscale:v",
        "3",
        str(output_pattern),
    ]
    subprocess.run(cmd, check=True)
    return combo, "processed"


def extract_frames(args, available_by_split, image_root: Path):
    all_available = {}
    for split_available in available_by_split.values():
        all_available.update(split_available)

    combos = unique_combo_jobs(all_available, args.limit_combos)
    if not combos:
        return {"processed": 0, "skipped": 0}

    processed = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [
            executor.submit(
                extract_one_combo,
                args.ffmpeg,
                args.ffmpeg_loglevel,
                image_root,
                image_root,
                combo,
                all_available[combo],
                args.skip_existing,
            )
            for combo in combos
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting frames"):
            _, status = future.result()
            if status == "processed":
                processed += 1
            else:
                skipped += 1
    return {"processed": processed, "skipped": skipped}


def crop_subset_images(labels, image_root: Path, image_crop_root: Path, skip_existing: bool):
    written = 0
    skipped = 0
    for label in tqdm(labels, desc="Cropping images"):
        rel = Path(label["image"])
        src = image_root / rel
        dst = image_crop_root / rel
        if skip_existing and dst.exists():
            skipped += 1
            continue

        image = cv2.imread(str(src), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise FileNotFoundError(f"Missing extracted frame: {src}")

        dst.parent.mkdir(parents=True, exist_ok=True)
        cropped = crop_image(image, label["center"], label["scale"], (192, 256))
        if not cv2.imwrite(str(dst), cropped):
            raise RuntimeError(f"Failed to write crop: {dst}")
        written += 1

    return {"processed": written, "skipped": skipped}


def main():
    args = parse_args()

    repo_root = args.repo_root.resolve()
    dcformer_dir = repo_root / "DCFormer"
    data_dir = dcformer_dir / "data"
    toolbox_dir = repo_root / "H36M-Toolbox"
    image_root = toolbox_dir / "images"
    image_crop_root = toolbox_dir / "images_crop"

    metadata = load_metadata(toolbox_dir)

    splits = {
        "train": data_dir / args.train_labels,
        "val": data_dir / args.val_labels,
    }

    labels_by_split = {}
    subset_by_split = {}
    available_by_split = {}
    missing_by_split = {}
    summary = {}

    for split_name, label_path in splits.items():
        labels = load_pickle(label_path)
        labels = filter_labels_by_subjects(labels, args.subjects)
        subset_labels, available, missing = build_subset(labels, metadata, image_root)

        labels_by_split[split_name] = labels
        subset_by_split[split_name] = subset_labels
        available_by_split[split_name] = available
        missing_by_split[split_name] = missing
        summary[split_name] = build_summary(split_name, labels, subset_labels, available, missing)

    output_train = data_dir / f"{args.output_prefix}_train.pkl"
    output_val = data_dir / f"{args.output_prefix}_validation.pkl"
    summary_json = data_dir / f"{args.output_prefix}_summary.json"
    train_manifest = data_dir / f"{args.output_prefix}_train_manifest.txt"
    val_manifest = data_dir / f"{args.output_prefix}_validation_manifest.txt"

    if args.write_subset:
        dump_pickle(output_train, subset_by_split["train"])
        dump_pickle(output_val, subset_by_split["val"])
        write_text_manifest(train_manifest, "train", available_by_split["train"], missing_by_split["train"])
        write_text_manifest(val_manifest, "val", available_by_split["val"], missing_by_split["val"])
        with summary_json.open("w") as f:
            json.dump(summary, f, indent=2)

    extraction_stats = None
    if args.extract_frames:
        extraction_stats = extract_frames(args, available_by_split, image_root)

    crop_stats = None
    if args.crop_images:
        crop_labels = subset_by_split["train"] + subset_by_split["val"]
        if args.limit_combos is not None:
            keep_combos = set(unique_combo_jobs(
                {**available_by_split["train"], **available_by_split["val"]},
                args.limit_combos,
            ))
            crop_labels = [label for label in crop_labels if combo_key(label) in keep_combos]
        crop_stats = crop_subset_images(crop_labels, image_root, image_crop_root, args.skip_existing)

    print("Subset summary:")
    print(json.dumps(summary, indent=2))
    print(f"train_subset={output_train}")
    print(f"val_subset={output_val}")
    print(f"summary_json={summary_json}")
    if extraction_stats is not None:
        print(f"frame_extraction={extraction_stats}")
    if crop_stats is not None:
        print(f"crop_generation={crop_stats}")


if __name__ == "__main__":
    main()
