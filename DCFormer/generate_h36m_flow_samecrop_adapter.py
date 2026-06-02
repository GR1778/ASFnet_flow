import argparse
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


MODELS = ("raft_large", "raft_small")
BACKENDS = ("torchvision", "searaft")
OUTPUT_FORMATS = ("npy", "npz")
DTYPES = ("float16", "float32")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_LABELS = SCRIPT_DIR / "data" / "h36m_validation.pkl"
DEFAULT_INPUT_DIR = REPO_ROOT / "H36M-Toolbox" / "images"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "H36M-Toolbox" / "flow_images_float"
DEFAULT_SEARAFT_DIR = REPO_ROOT / "third_party" / "SEA-RAFT"
DEFAULT_SEARAFT_CFG = DEFAULT_SEARAFT_DIR / "config" / "eval" / "spring-M.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate backward optical flow on same-current-crop pairs. For target frame t, "
            "both raw frames I_t and I_{t-k} are cropped with frame t center/scale before "
            "RAFT(I_t_crop_by_t, I_{t-k}_crop_by_t)."
        )
    )
    parser.add_argument("--labels", nargs="+", default=[str(DEFAULT_LABELS)], help="One or more H36M label pkl files.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Raw extracted image root.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output root for flow files.")
    parser.add_argument("--vis-dir", default=None, help="Optional root for flow visualization PNGs.")
    parser.add_argument("--backend", default="torchvision", choices=BACKENDS, help="Optical flow estimator backend.")
    parser.add_argument("--model", default="raft_large", choices=MODELS)
    parser.add_argument("--weights", default="default", help="'default', 'none', or a local state_dict path.")
    parser.add_argument("--sea-raft-dir", default=str(DEFAULT_SEARAFT_DIR), help="SEA-RAFT repository root.")
    parser.add_argument("--sea-raft-cfg", default=str(DEFAULT_SEARAFT_CFG), help="SEA-RAFT JSON config.")
    parser.add_argument("--sea-raft-ckpt", default=None, help="Local SEA-RAFT checkpoint path.")
    parser.add_argument("--sea-raft-url", default=None, help="HuggingFace model id for SEA-RAFT.")
    parser.add_argument("--output-format", default="npy", choices=OUTPUT_FORMATS)
    parser.add_argument("--dtype", default="float16", choices=DTYPES)
    parser.add_argument(
        "--suppress-small-flow",
        action="store_true",
        help="Set small-magnitude flow vectors to zero before saving.",
    )
    parser.add_argument(
        "--suppress-threshold",
        type=float,
        default=0.2,
        help="Pixel threshold used by --suppress-small-flow.",
    )
    parser.add_argument(
        "--clip-flow",
        type=float,
        default=None,
        help="Optional pixel magnitude clip. Direction is preserved.",
    )
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--frame-gap", type=int, default=4, help="Use frame t-k as the previous frame.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--subjects", nargs="+", type=int, default=None)
    parser.add_argument("--top-level-dirs", nargs="+", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--vis-stride", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.frame_gap < 1:
        raise ValueError("--frame-gap must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.suppress_threshold < 0:
        raise ValueError("--suppress-threshold must be >= 0")
    if args.clip_flow is not None and args.clip_flow <= 0:
        raise ValueError("--clip-flow must be > 0")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    if args.backend == "searaft" and not args.sea_raft_ckpt and not args.sea_raft_url:
        raise ValueError("SEA-RAFT backend requires --sea-raft-ckpt or --sea-raft-url")


def seq_name(shot):
    return (
        f"s_{shot['subject']:02d}_act_{shot['action']:02d}_"
        f"subact_{shot['subaction']:02d}_ca_{shot['camera_id'] + 1:02d}"
    )


def image_name(seq, frame_id):
    return f"{seq}_{frame_id:06d}.jpg"


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


def crop_raw_image(path, center, scale, output_size):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise OSError(f"Failed to read image: {path}")
    trans = get_affine_transform(center, scale, output_size)
    image = cv2.warpAffine(image, trans, output_size, flags=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def output_path_for(output_dir, seq, frame_id, output_format):
    return output_dir / seq / f"{seq}_{frame_id:06d}.{output_format}"


def vis_path_for(vis_dir, seq, frame_id):
    if vis_dir is None:
        return None
    return vis_dir / seq / f"{seq}_{frame_id:06d}.png"


def load_labels(paths):
    labels = []
    for path in paths:
        label_path = Path(path).expanduser().resolve()
        print(f"Loading labels: {label_path}")
        with open(label_path, "rb") as file:
            part = pickle.load(file)
        print(f"Loaded {len(part)} labels")
        labels.extend(part)
    return labels


def build_tasks(args):
    validate_args(args)
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    vis_dir = Path(args.vis_dir).expanduser().resolve() if args.vis_dir else None
    output_size = (args.width, args.height)
    allowed_subjects = {int(subject) for subject in args.subjects} if args.subjects else None
    allowed_top = set(args.top_level_dirs) if args.top_level_dirs else None

    labels = load_labels(args.labels)
    labels_by_seq = defaultdict(list)
    for shot in labels:
        if allowed_subjects is not None and int(shot["subject"]) not in allowed_subjects:
            continue
        seq = seq_name(shot)
        if allowed_top is not None and seq not in allowed_top:
            continue
        labels_by_seq[seq].append(shot)

    for seq in labels_by_seq:
        labels_by_seq[seq].sort(key=lambda item: int(item["image_id"]))

    tasks = []
    for seq in sorted(labels_by_seq):
        frame_to_shot = {int(shot["image_id"]): shot for shot in labels_by_seq[seq]}
        for shot in labels_by_seq[seq]:
            frame_id = int(shot["image_id"])
            out_path = output_path_for(output_dir, seq, frame_id, args.output_format)
            if args.skip_existing and out_path.exists():
                continue

            prev_frame_id = frame_id - args.frame_gap
            cur_path = input_dir / seq / image_name(seq, frame_id)
            prev_path = input_dir / seq / image_name(seq, prev_frame_id)
            valid = prev_frame_id in frame_to_shot and cur_path.exists() and prev_path.exists()

            tasks.append(
                {
                    "seq": seq,
                    "frame_id": frame_id,
                    "prev_frame_id": prev_frame_id,
                    "cur_path": cur_path,
                    "prev_path": prev_path if valid else None,
                    "center": shot["center"],
                    "scale": shot["scale"],
                    "output_size": output_size,
                    "out_path": out_path,
                    "vis_path": vis_path_for(vis_dir, seq, frame_id),
                    "valid": valid,
                }
            )
            if args.max_images is not None and len(tasks) >= args.max_images:
                sharded = tasks[args.shard_index :: args.num_shards]
                return input_dir, output_dir, vis_dir, sharded, len(labels_by_seq)

    sharded = tasks[args.shard_index :: args.num_shards]
    return input_dir, output_dir, vis_dir, sharded, len(labels_by_seq)


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


def load_searaft(args, device):
    sea_raft_dir = Path(args.sea_raft_dir).expanduser().resolve()
    cfg_path = Path(args.sea_raft_cfg).expanduser().resolve()
    core_dir = sea_raft_dir / "core"
    if not sea_raft_dir.exists():
        raise FileNotFoundError(f"SEA-RAFT directory not found: {sea_raft_dir}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"SEA-RAFT config not found: {cfg_path}")

    for path in (str(sea_raft_dir), str(core_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from config.parser import json_to_args
    from raft import RAFT
    from utils.utils import load_ckpt

    sea_args = json_to_args(str(cfg_path))
    if args.sea_raft_ckpt:
        sea_args.skip_imagenet_pretrain = True
    # Official Spring configs may set scale=-1 for large benchmark frames.
    # Our H36M crops are only 192x256, so extra downscaling can collapse
    # internal feature maps to width 0. Keep the crop resolution unchanged.
    if getattr(sea_args, "scale", 0) < 0:
        sea_args.scale = 0
    if args.sea_raft_ckpt:
        model = RAFT(sea_args)
        load_ckpt(model, str(Path(args.sea_raft_ckpt).expanduser().resolve()))
    else:
        model = RAFT.from_pretrained(args.sea_raft_url, args=sea_args)

    return model.to(device).eval(), sea_args


def resolve_device(device_arg):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_arg}, but CUDA is not available.")
    return device


def read_cropped_pair_as_chw(sample):
    import torch

    cur = crop_raw_image(sample["cur_path"], sample["center"], sample["scale"], sample["output_size"])
    prev = crop_raw_image(sample["prev_path"], sample["center"], sample["scale"], sample["output_size"])
    cur = torch.from_numpy(cur).permute(2, 0, 1).contiguous()
    prev = torch.from_numpy(prev).permute(2, 0, 1).contiguous()
    return cur, prev


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

    pairs = [read_cropped_pair_as_chw(sample) for sample in batch]
    cur = torch.stack([item[0] for item in pairs], dim=0)
    prev = torch.stack([item[1] for item in pairs], dim=0)
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


def run_searaft_batch(model, sea_args, batch, device, use_amp):
    import torch
    import torch.nn.functional as F

    pairs = [read_cropped_pair_as_chw(sample) for sample in batch]
    cur = torch.stack([item[0] for item in pairs], dim=0).float().to(device, non_blocking=device.type == "cuda")
    prev = torch.stack([item[1] for item in pairs], dim=0).float().to(device, non_blocking=device.type == "cuda")

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp and device.type == "cuda"):
            if getattr(sea_args, "scale", 0) != 0:
                scale_factor = 2 ** sea_args.scale
                cur_in = F.interpolate(cur, scale_factor=scale_factor, mode="bilinear", align_corners=False)
                prev_in = F.interpolate(prev, scale_factor=scale_factor, mode="bilinear", align_corners=False)
            else:
                scale_factor = 1.0
                cur_in = cur
                prev_in = prev

            output = model(cur_in, prev_in, iters=sea_args.iters, test_mode=True)
            flow = output["flow"][-1]
            if scale_factor != 1.0:
                flow = F.interpolate(flow, size=cur.shape[-2:], mode="bilinear", align_corners=False) / scale_factor

    return flow.detach().cpu()


def postprocess_flow(flow, suppress_small=False, suppress_threshold=0.2, clip_flow=None):
    if not suppress_small and clip_flow is None:
        return flow
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    if suppress_small:
        flow = flow.copy()
        flow[mag < suppress_threshold] = 0.0
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    if clip_flow is not None:
        scale = np.minimum(1.0, clip_flow / (mag + 1e-6))
        flow = flow * scale[..., None]
    return flow


def save_flow(sample, flow_chw, args, valid=1):
    flow = flow_chw.detach().cpu().permute(1, 2, 0).numpy()
    flow = postprocess_flow(
        flow,
        suppress_small=args.suppress_small_flow,
        suppress_threshold=args.suppress_threshold,
        clip_flow=args.clip_flow,
    )
    flow = flow.astype(np.float16 if args.dtype == "float16" else np.float32)
    sample["out_path"].parent.mkdir(parents=True, exist_ok=True)
    if args.output_format == "npy":
        np.save(sample["out_path"], flow)
    else:
        np.savez_compressed(sample["out_path"], flow=flow, valid=np.uint8(valid))


def save_zero_flow(sample, args):
    import torch

    w, h = sample["output_size"]
    flow = torch.zeros(2, h, w)
    save_flow(sample, flow, args, valid=0)


def save_flow_vis(sample, flow_chw):
    if sample["vis_path"] is None:
        return
    from torchvision.utils import flow_to_image

    image = flow_to_image(flow_chw).permute(1, 2, 0).numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    sample["vis_path"].parent.mkdir(parents=True, exist_ok=True)
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
    input_dir, output_dir, vis_dir, tasks, num_sequences = build_tasks(args)
    valid_tasks = [task for task in tasks if task["valid"]]
    zero_tasks = [task for task in tasks if not task["valid"]]

    print(f"Input raw image directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Visualization directory: {vis_dir if vis_dir else 'disabled'}")
    print(f"Sequences found: {num_sequences}")
    print(f"Frames/tasks planned: {len(tasks)}")
    print(f"Valid RAFT pairs: {len(valid_tasks)}")
    print(f"Zero/invalid boundary flows: {len(zero_tasks)}")
    print(f"Frame gap: {args.frame_gap}")
    print(f"Output size: {args.width}x{args.height}")
    print(f"Output format: {args.output_format}")
    print(f"Saved dtype: {args.dtype}")
    print(f"Suppress small flow: {args.suppress_small_flow}")
    print(f"Suppress threshold: {args.suppress_threshold}")
    print(f"Clip flow: {args.clip_flow if args.clip_flow is not None else 'disabled'}")
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model if args.backend == 'torchvision' else 'SEA-RAFT'}")
    if args.backend == "searaft":
        print(f"SEA-RAFT directory: {Path(args.sea_raft_dir).expanduser().resolve()}")
        print(f"SEA-RAFT config: {Path(args.sea_raft_cfg).expanduser().resolve()}")
        print(f"SEA-RAFT checkpoint: {args.sea_raft_ckpt if args.sea_raft_ckpt else 'disabled'}")
        print(f"SEA-RAFT HuggingFace url: {args.sea_raft_url if args.sea_raft_url else 'disabled'}")
    print(f"Shard: {args.shard_index}/{args.num_shards}")
    print(f"Requested device: {args.device}")
    print(f"Subjects filter: {args.subjects if args.subjects else 'all'}")
    print(f"Top-level dir filter: {args.top_level_dirs if args.top_level_dirs else 'all'}")

    if args.dry_run:
        return
    if not tasks:
        print("No tasks to process.")
        return

    progress = tqdm(total=len(tasks), desc="Generating same-crop flow", unit="frame") if tqdm else None
    processed = 0
    failed = 0

    for sample in zero_tasks:
        try:
            save_zero_flow(sample, args)
            processed += 1
        except Exception as error:
            failed += 1
            print(f"Warning: failed zero flow for {sample['seq']} frame {sample['frame_id']}: {error}")
        if progress:
            progress.update(1)

    if valid_tasks:
        import torch

        device = resolve_device(args.device)
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        print(f"Device: {device}")

        if args.backend == "searaft":
            model, model_args = load_searaft(args, device)
        else:
            model, transforms = load_torchvision_raft(args.model, args.weights, device)
            model_args = transforms

        for batch in batched(valid_tasks, args.batch_size):
            try:
                if args.backend == "searaft":
                    flow_batch = run_searaft_batch(model, model_args, batch, device, args.amp)
                else:
                    flow_batch = run_raft_batch(model, model_args, batch, device, args.amp)
            except Exception as error:
                failed += len(batch)
                print(
                    f"Warning: failed {args.backend} batch beginning at "
                    f"{batch[0]['seq']} {batch[0]['frame_id']}: {error}"
                )
                if progress:
                    progress.update(len(batch))
                continue

            for sample, flow_chw in zip(batch, flow_batch):
                try:
                    vis_sample = sample
                    if not should_write_vis(args, processed):
                        vis_sample = dict(sample)
                        vis_sample["vis_path"] = None
                    save_flow(sample, flow_chw, args, valid=1)
                    save_flow_vis(vis_sample, flow_chw)
                    processed += 1
                except Exception as error:
                    failed += 1
                    print(f"Warning: failed writing flow for {sample['seq']} frame {sample['frame_id']}: {error}")
                if progress:
                    progress.update(1)

    if progress:
        progress.close()
    print(f"Done. processed={processed} failed={failed}")


if __name__ == "__main__":
    main()
