import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MODELS = ("raft_large", "raft_small")
OUTPUT_FORMATS = ("npy", "npz")
DTYPES = ("float16", "float32")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "H36M-Toolbox" / "images_crop"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "H36M-Toolbox" / "flow_raft_bwd_fp16"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate offline backward optical flow for ASFnet/CA-PF style "
            "Human3.6M cropped frames. For frame t, flow is RAFT(I_t, I_{t-k}), "
            "so the flow field is aligned to the current frame coordinates."
        )
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Cropped image root.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output root for flow files.")
    parser.add_argument("--vis-dir", default=None, help="Optional root for flow visualization PNGs.")
    parser.add_argument("--model", default="raft_large", choices=MODELS)
    parser.add_argument("--weights", default="default", help="'default', 'none', or a local state_dict path.")
    parser.add_argument("--output-format", default="npy", choices=OUTPUT_FORMATS)
    parser.add_argument("--dtype", default="float16", choices=DTYPES)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--subjects", nargs="+", type=int, default=None)
    parser.add_argument("--top-level-dirs", nargs="+", default=None)
    parser.add_argument("--frame-gap", type=int, default=1, help="Use frame t-k as the previous frame.")
    parser.add_argument("--allow-nonconsecutive", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--vis-stride", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.frame_gap < 1:
        raise ValueError("--frame-gap must be >= 1")


def parse_frame_id(path):
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def image_sort_key(path):
    frame_id = parse_frame_id(path)
    return (path.parent.as_posix(), frame_id if frame_id is not None else path.name)


def is_consecutive(prev_path, cur_path):
    prev_id = parse_frame_id(prev_path)
    cur_id = parse_frame_id(cur_path)
    if prev_id is None or cur_id is None:
        return True
    return cur_id == prev_id + 1


def collect_sequence_dirs(input_dir):
    image_paths = [
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    grouped = defaultdict(list)
    for image_path in image_paths:
        grouped[image_path.parent].append(image_path)
    return [(seq_dir, sorted(paths, key=image_sort_key)) for seq_dir, paths in sorted(grouped.items())]


def filter_sequences(sequences, input_dir, subjects, top_level_dirs):
    allowed_subjects = {int(subject) for subject in subjects} if subjects else None
    allowed_top = set(top_level_dirs) if top_level_dirs else None
    filtered = []
    for seq_dir, frames in sequences:
        rel = seq_dir.relative_to(input_dir)
        top = rel.parts[0] if rel.parts else seq_dir.name
        if allowed_top is not None and top not in allowed_top:
            continue
        if allowed_subjects is not None:
            match = re.match(r"s_(\d+)_", top)
            if match is None or int(match.group(1)) not in allowed_subjects:
                continue
        filtered.append((seq_dir, frames))
    return filtered


def output_path_for(output_dir, input_dir, image_path, output_format):
    rel = image_path.relative_to(input_dir)
    return (output_dir / rel).with_suffix("." + output_format)


def vis_path_for(vis_dir, input_dir, image_path):
    if vis_dir is None:
        return None
    rel = image_path.relative_to(input_dir)
    return (vis_dir / rel).with_suffix(".png")


def build_tasks(args):
    validate_args(args)
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    vis_dir = Path(args.vis_dir).expanduser().resolve() if args.vis_dir else None

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    sequences = collect_sequence_dirs(input_dir)
    sequences = filter_sequences(sequences, input_dir, args.subjects, args.top_level_dirs)

    tasks = []
    for _seq_dir, frames in sequences:
        for idx, cur_path in enumerate(frames):
            out_path = output_path_for(output_dir, input_dir, cur_path, args.output_format)
            if args.skip_existing and out_path.exists():
                continue

            prev_path = frames[idx - args.frame_gap] if idx >= args.frame_gap else None
            valid = prev_path is not None and (
                args.allow_nonconsecutive
                or parse_frame_id(cur_path) is None
                or parse_frame_id(prev_path) is None
                or parse_frame_id(cur_path) == parse_frame_id(prev_path) + args.frame_gap
            )
            if not valid:
                prev_path = None

            tasks.append(
                {
                    "cur_path": cur_path,
                    "prev_path": prev_path,
                    "out_path": out_path,
                    "vis_path": vis_path_for(vis_dir, input_dir, cur_path),
                    "valid": valid,
                }
            )
            if args.max_images is not None and len(tasks) >= args.max_images:
                sharded = tasks[args.shard_index :: args.num_shards]
                return input_dir, output_dir, vis_dir, sharded, sequences

    sharded = tasks[args.shard_index :: args.num_shards]
    return input_dir, output_dir, vis_dir, sharded, sequences


def legacy_raft_transforms(image1, image2):
    image1 = image1.float()
    image2 = image2.float()
    if image1.max() > 1:
        image1 = image1 / 255.0
    if image2.max() > 1:
        image2 = image2 / 255.0
    return image1 * 2.0 - 1.0, image2 * 2.0 - 1.0


def load_torchvision_raft(model_name, weights_arg, device):
    import torch

    try:
        from torchvision.models.optical_flow import (
            Raft_Large_Weights,
            Raft_Small_Weights,
            raft_large,
            raft_small,
        )
        has_weight_enums = True
    except ImportError:
        from torchvision.models.optical_flow import raft_large, raft_small

        Raft_Large_Weights = None
        Raft_Small_Weights = None
        has_weight_enums = False

    if model_name == "raft_large":
        model_fn = raft_large
        weights_enum = Raft_Large_Weights if has_weight_enums else None
    else:
        model_fn = raft_small
        weights_enum = Raft_Small_Weights if has_weight_enums else None

    if has_weight_enums and weights_arg == "default":
        weights = weights_enum.DEFAULT
        model = model_fn(weights=weights, progress=True)
        transforms = weights.transforms()
    elif has_weight_enums and weights_arg == "none":
        model = model_fn(weights=None, progress=True)
        transforms = weights_enum.DEFAULT.transforms()
    elif has_weight_enums:
        model = model_fn(weights=None, progress=True)
        state_dict = torch.load(Path(weights_arg).expanduser().resolve(), map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
        transforms = weights_enum.DEFAULT.transforms()
    else:
        if weights_arg == "default":
            model = model_fn(pretrained=True, progress=True)
        elif weights_arg == "none":
            model = model_fn(pretrained=False, progress=True)
        else:
            model = model_fn(pretrained=False, progress=True)
            state_dict = torch.load(Path(weights_arg).expanduser().resolve(), map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict)
        transforms = legacy_raft_transforms

    return model.to(device).eval(), transforms


def resolve_device(device_arg):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_arg}, but CUDA is not available.")
    return device


def read_image_as_chw(path):
    import cv2
    import torch

    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise OSError(f"Failed to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image).permute(2, 0, 1).contiguous()


def pad_to_multiple(tensor, multiple=8):
    import torch.nn.functional as F

    _b, _c, h, w = tensor.shape
    pad_h = int(math.ceil(h / multiple) * multiple - h)
    pad_w = int(math.ceil(w / multiple) * multiple - w)
    if pad_h == 0 and pad_w == 0:
        return tensor, (h, w)
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate"), (h, w)


def run_raft_batch(model, transforms, batch, device, use_amp):
    import torch

    cur = torch.stack([read_image_as_chw(sample["cur_path"]) for sample in batch], dim=0)
    prev = torch.stack([read_image_as_chw(sample["prev_path"]) for sample in batch], dim=0)
    cur, prev = transforms(cur, prev)
    cur = cur.to(device, non_blocking=device.type == "cuda")
    prev = prev.to(device, non_blocking=device.type == "cuda")
    cur, original_hw = pad_to_multiple(cur)
    prev, _ = pad_to_multiple(prev)
    h, w = original_hw

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp and device.type == "cuda"):
            predictions = model(cur, prev)

    return predictions[-1][:, :, :h, :w].detach().cpu()


def flow_to_hwc_numpy(flow_chw):
    return flow_chw.detach().cpu().permute(1, 2, 0).numpy()


def save_flow(sample, flow_chw, dtype_name, output_format):
    import numpy as np

    flow = flow_to_hwc_numpy(flow_chw)
    flow = flow.astype(np.float16 if dtype_name == "float16" else np.float32)
    sample["out_path"].parent.mkdir(parents=True, exist_ok=True)
    if output_format == "npy":
        np.save(sample["out_path"], flow)
    else:
        np.savez_compressed(sample["out_path"], flow=flow, valid=np.uint8(1))


def save_zero_flow(sample, dtype_name, output_format):
    import cv2
    import numpy as np

    image = cv2.imread(str(sample["cur_path"]), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise OSError(f"Failed to read image for zero flow shape: {sample['cur_path']}")
    h, w = image.shape[:2]
    flow = np.zeros((h, w, 2), dtype=np.float16 if dtype_name == "float16" else np.float32)
    sample["out_path"].parent.mkdir(parents=True, exist_ok=True)
    if output_format == "npy":
        np.save(sample["out_path"], flow)
    else:
        np.savez_compressed(sample["out_path"], flow=flow, valid=np.uint8(0))


def save_flow_vis(sample, flow_chw):
    if sample["vis_path"] is None:
        return
    import cv2
    from torchvision.utils import flow_to_image

    sample["vis_path"].parent.mkdir(parents=True, exist_ok=True)
    image = flow_to_image(flow_chw).permute(1, 2, 0).numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(sample["vis_path"]), image)


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def should_write_vis(args, processed_index):
    if args.vis_dir is None:
        return False
    return args.vis_stride == 0 or processed_index % args.vis_stride == 0


def main():
    args = parse_args()
    input_dir, output_dir, vis_dir, tasks, sequences = build_tasks(args)
    valid_tasks = [task for task in tasks if task["valid"]]
    zero_tasks = [task for task in tasks if not task["valid"]]

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Visualization directory: {vis_dir if vis_dir else 'disabled'}")
    print(f"Sequences found: {len(sequences)}")
    print(f"Frames/tasks planned: {len(tasks)}")
    print(f"Valid RAFT pairs: {len(valid_tasks)}")
    print(f"Zero/invalid boundary flows: {len(zero_tasks)}")
    print(f"Output format: {args.output_format}")
    print(f"Saved dtype: {args.dtype}")
    print(f"Model: {args.model}")
    print(f"Shard: {args.shard_index}/{args.num_shards}")
    print(f"Requested device: {args.device}")
    print(f"Subjects filter: {args.subjects if args.subjects else 'all'}")
    print(f"Top-level dir filter: {args.top_level_dirs if args.top_level_dirs else 'all'}")

    if args.dry_run:
        return
    if not tasks:
        print("No tasks to process.")
        return

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    progress = tqdm(total=len(tasks), desc="Generating flow", unit="frame") if tqdm else None
    processed = 0
    failed = 0

    for sample in zero_tasks:
        try:
            save_zero_flow(sample, args.dtype, args.output_format)
            processed += 1
        except Exception as error:
            failed += 1
            print(f"Warning: failed zero flow for {sample['cur_path']}: {error}")
        if progress:
            progress.update(1)

    if valid_tasks:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("Generating non-boundary flow requires PyTorch and TorchVision.") from error

        device = resolve_device(args.device)
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        print(f"Device: {device}")

        model, transforms = load_torchvision_raft(args.model, args.weights, device)
        for batch in batched(valid_tasks, args.batch_size):
            try:
                flow_batch = run_raft_batch(model, transforms, batch, device, args.amp)
            except Exception as error:
                failed += len(batch)
                print(f"Warning: failed RAFT batch beginning at {batch[0]['cur_path']}: {error}")
                if progress:
                    progress.update(len(batch))
                continue

            for sample, flow_chw in zip(batch, flow_batch):
                try:
                    vis_sample = sample
                    if not should_write_vis(args, processed):
                        vis_sample = dict(sample)
                        vis_sample["vis_path"] = None
                    save_flow(sample, flow_chw, args.dtype, args.output_format)
                    save_flow_vis(vis_sample, flow_chw)
                    processed += 1
                except Exception as error:
                    failed += 1
                    print(f"Warning: failed writing flow for {sample['cur_path']}: {error}")
                if progress:
                    progress.update(1)

    if progress:
        progress.close()
    print(f"Done. processed={processed} failed={failed}")


if __name__ == "__main__":
    main()
