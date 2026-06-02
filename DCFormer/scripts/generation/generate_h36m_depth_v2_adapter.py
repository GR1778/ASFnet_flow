import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


if not hasattr(torch.backends, "mps"):
    class _DummyMPSBackend:
        @staticmethod
        def is_available():
            return False

    torch.backends.mps = _DummyMPSBackend()


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPTH_ANYTHING_ROOT = REPO_ROOT / "Depth-Anything-V2"
DEFAULT_INPUT_DIR = REPO_ROOT / "H36M-Toolbox" / "images_crop"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "H36M-Toolbox" / "depth_images"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
OUTPUT_FORMATS = ("jpg", "npy", "both")
NPY_NORMALIZATIONS = ("minmax", "raw")

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

AUTO_BATCH_SIZE = {
    "vits": 16,
    "vitb": 8,
    "vitl": 4,
    "vitg": 2,
}


cv2.setNumThreads(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ASFnet Human3.6M depth maps using a local Depth-Anything-V2 checkout."
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Cropped Human3.6M image directory.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for depth maps.")
    parser.add_argument(
        "--output-format",
        default="jpg",
        choices=OUTPUT_FORMATS,
        help="Depth output format. 'jpg' keeps the original uint8 behavior; 'npy' writes float32 .npy files.",
    )
    parser.add_argument(
        "--npy-normalization",
        default="minmax",
        choices=NPY_NORMALIZATIONS,
        help="Normalization used before saving .npy depth. 'minmax' stores float32 in [0, 1]; 'raw' stores raw DAV2 output.",
    )
    parser.add_argument(
        "--depth-anything-root",
        default=str(DEFAULT_DEPTH_ANYTHING_ROOT),
        help="Path to the local Depth-Anything-V2 repository.",
    )
    parser.add_argument("--encoder", default="vitb", choices=sorted(MODEL_CONFIGS.keys()))
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path. Defaults to <depth-anything-root>/checkpoints/depth_anything_v2_<encoder>.pth",
    )
    parser.add_argument("--input-size", type=int, default=518, help="Depth Anything V2 inference size.")
    parser.add_argument("--max-images", type=int, default=None, help="Only process the first N images.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip depth maps that already exist.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Images per forward pass. Use 0 to choose a CUDA-friendly default automatically.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of data loading workers. Defaults to 0 on CPU and up to 8 on GPU.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision on CUDA for faster inference with lower memory usage.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        default=None,
        help="Optional Human3.6M subject ids to include, e.g. --subjects 9 11",
    )
    parser.add_argument(
        "--top-level-dirs",
        nargs="+",
        default=None,
        help="Optional top-level sequence directories to include, e.g. s_08_act_14_subact_01_ca_01",
    )
    return parser.parse_args()


def normalize_depth(depth):
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    if depth_max <= depth_min:
        return np.zeros(depth.shape, dtype=np.uint8)

    depth = (depth - depth_min) / (depth_max - depth_min)
    return (depth * 255.0).astype(np.uint8)


def normalize_depth_float(depth):
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    if depth_max <= depth_min:
        return np.zeros(depth.shape, dtype=np.float32)

    depth = (depth - depth_min) / (depth_max - depth_min)
    return depth.astype(np.float32, copy=False)


def collect_images(input_dir, top_level_dirs=None):
    if top_level_dirs:
        image_paths = []
        for top_level_dir in top_level_dirs:
            directory = input_dir / top_level_dir
            if not directory.is_dir():
                continue
            image_paths.extend(
                path for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        return sorted(image_paths)

    return sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def filter_images_by_subjects(image_paths, input_dir, subjects):
    if not subjects:
        return image_paths

    allowed_subjects = {int(subject) for subject in subjects}
    filtered_paths = []

    for image_path in image_paths:
        relative_path = image_path.relative_to(input_dir)
        first_part = relative_path.parts[0] if relative_path.parts else ""
        match = re.match(r"s_(\d+)_", first_part)
        if match is None:
            continue

        subject = int(match.group(1))
        if subject in allowed_subjects:
            filtered_paths.append(image_path)

    return filtered_paths


def filter_images_by_top_level_dirs(image_paths, input_dir, top_level_dirs):
    if not top_level_dirs:
        return image_paths

    allowed_dirs = set(top_level_dirs)
    filtered_paths = []

    for image_path in image_paths:
        relative_path = image_path.relative_to(input_dir)
        first_part = relative_path.parts[0] if relative_path.parts else ""
        if first_part in allowed_dirs:
            filtered_paths.append(image_path)

    return filtered_paths


def get_progress(total):
    if tqdm is None:
        return None
    return tqdm(total=total, desc="Generating depth", unit="img")


def build_transform(input_size):
    from torchvision.transforms import Compose
    from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize

    return Compose([
        Resize(
            width=input_size,
            height=input_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])


def prepare_image(raw_image, transform):
    original_hw = raw_image.shape[:2]
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    image = transform({"image": image})["image"]
    image = torch.from_numpy(image)
    return image, original_hw


def get_output_paths(output_path, output_format):
    if output_format == "jpg":
        return [output_path]
    if output_format == "npy":
        return [output_path.with_suffix(".npy")]
    return [output_path, output_path.with_suffix(".npy")]


class DepthImageDataset(Dataset):
    def __init__(self, image_paths, input_dir, output_dir, transform, output_format="jpg", skip_existing=False):
        self.transform = transform
        self.samples = []
        self.skipped_existing = 0

        for image_path in image_paths:
            relative_path = image_path.relative_to(input_dir)
            output_path = output_dir / relative_path

            if skip_existing and all(path.exists() for path in get_output_paths(output_path, output_format)):
                self.skipped_existing += 1
                continue

            self.samples.append((image_path, output_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, output_path = self.samples[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if image is None:
            return {
                "image_path": image_path,
                "output_path": output_path,
                "error": f"failed to read image: {image_path}",
            }

        tensor, original_hw = prepare_image(image, self.transform)
        return {
            "image_path": image_path,
            "output_path": output_path,
            "original_hw": original_hw,
            "tensor": tensor,
        }


def collate_samples(samples):
    return samples


def get_default_num_workers(device):
    if device != "cuda":
        return 0

    cpu_count = os.cpu_count() or 1
    return min(8, cpu_count)


def resolve_batch_size(requested_batch_size, encoder, device):
    if requested_batch_size > 0:
        return requested_batch_size
    if device != "cuda":
        return 1
    return AUTO_BATCH_SIZE[encoder]


def is_cuda_oom_error(error):
    error_types = [
        getattr(torch, "OutOfMemoryError", None),
        getattr(torch.cuda, "OutOfMemoryError", None),
    ]
    error_types = tuple(error_type for error_type in error_types if error_type is not None)

    if error_types and isinstance(error, error_types):
        return True

    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def run_model_on_samples(model, samples, device, use_amp):
    if not samples:
        return []

    batch_tensor = torch.stack([sample["tensor"] for sample in samples], dim=0)
    batch_tensor = batch_tensor.to(device, non_blocking=(device == "cuda"))

    try:
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                depth_batch = model(batch_tensor)
        depth_batch = depth_batch.float()
    except Exception as error:
        if device != "cuda" or len(samples) == 1 or not is_cuda_oom_error(error):
            raise
        torch.cuda.empty_cache()
        mid = len(samples) // 2
        return run_model_on_samples(model, samples[:mid], device, use_amp) + run_model_on_samples(
            model, samples[mid:], device, use_amp
        )

    outputs = []
    for depth, sample in zip(depth_batch, samples):
        h, w = sample["original_hw"]
        depth = F.interpolate(depth[None, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        outputs.append((sample, depth.cpu().numpy()))

    return outputs


def main():
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    depth_anything_root = Path(args.depth_anything_root).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not depth_anything_root.exists():
        raise FileNotFoundError(f"Depth-Anything-V2 directory does not exist: {depth_anything_root}")

    sys.path.insert(0, str(depth_anything_root))
    from depth_anything_v2.dpt import DepthAnythingV2

    if args.checkpoint is None:
        checkpoint_path = depth_anything_root / "checkpoints" / f"depth_anything_v2_{args.encoder}.pth"
    else:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    image_paths = collect_images(input_dir, args.top_level_dirs)
    image_paths = filter_images_by_subjects(image_paths, input_dir, args.subjects)
    image_paths = filter_images_by_top_level_dirs(image_paths, input_dir, args.top_level_dirs)
    if args.max_images is not None:
        image_paths = image_paths[:args.max_images]

    if not image_paths:
        print(f"No images found under {input_dir}")
        return

    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    device = "cuda" if torch.cuda.is_available() else "mps" if has_mps else "cpu"
    use_amp = args.amp and device == "cuda"
    batch_size = resolve_batch_size(args.batch_size, args.encoder, device)
    num_workers = get_default_num_workers(device) if args.num_workers is None else args.num_workers

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(device).eval()

    transform = build_transform(args.input_size)
    dataset = DepthImageDataset(
        image_paths=image_paths,
        input_dir=input_dir,
        output_dir=output_dir,
        transform=transform,
        output_format=args.output_format,
        skip_existing=args.skip_existing,
    )

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Depth-Anything-V2: {depth_anything_root}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Images to process: {len(image_paths)}")
    print(f"Skipped existing before load: {dataset.skipped_existing}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"AMP enabled: {use_amp}")
    print(f"Output format: {args.output_format}")
    print(f"NPY normalization: {args.npy_normalization}")
    print(f"Subjects filter: {args.subjects if args.subjects else 'all'}")
    print(f"Top-level dir filter: {args.top_level_dirs if args.top_level_dirs else 'all'}")

    processed = 0
    skipped = dataset.skipped_existing
    failed = 0
    progress = get_progress(len(dataset))

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device == "cuda",
        "collate_fn": collate_samples,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    dataloader = DataLoader(**loader_kwargs)

    for batch in dataloader:
        valid_groups = defaultdict(list)

        for sample in batch:
            if "error" in sample:
                print(f"Warning: {sample['error']}")
                failed += 1
                if progress is not None:
                    progress.update(1)
                continue

            valid_groups[tuple(sample["tensor"].shape)].append(sample)

        for samples_with_same_shape in valid_groups.values():
            outputs = run_model_on_samples(model, samples_with_same_shape, device, use_amp)

            for sample, depth in outputs:
                output_path = sample["output_path"]
                output_path.parent.mkdir(parents=True, exist_ok=True)

                wrote_all = True
                if args.output_format in ("jpg", "both"):
                    depth_u8 = normalize_depth(depth)
                    if not cv2.imwrite(str(output_path), depth_u8):
                        print(f"Warning: failed to write depth image: {output_path}")
                        wrote_all = False

                if args.output_format in ("npy", "both"):
                    npy_path = output_path.with_suffix(".npy")
                    depth_float = depth.astype(np.float32, copy=False)
                    if args.npy_normalization == "minmax":
                        depth_float = normalize_depth_float(depth_float)
                    try:
                        np.save(npy_path, depth_float)
                    except OSError as error:
                        print(f"Warning: failed to write depth npy: {npy_path} ({error})")
                        wrote_all = False

                if wrote_all:
                    processed += 1
                else:
                    failed += 1

                if progress is not None:
                    progress.update(1)

    if progress is not None:
        progress.close()

    print(f"Done. processed={processed} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
