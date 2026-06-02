import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnose_learned_flow_dce_sampling import resolve_checkpoint_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find checkpoints compatible with the current DepthGuidedPose/ASFNet model."
    )
    parser.add_argument("--config", default="experiments/human36m/human36m_single.yaml")
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["logs/*/checkpoints/best_epoch.bin", "checkpoint/*.bin"],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="debug_vis/asfnet_checkpoint_compatibility.json")
    return parser.parse_args()


def build_model(config_path, device):
    import torch

    from mvn.models.DGPose import DepthGuidedPose
    from mvn.utils.cfg import config, update_config

    update_config(config_path)
    model = DepthGuidedPose(config, device)
    model.to(torch.device(device))
    model.eval()
    return model, config


def inspect_checkpoint(model, path):
    import torch

    path = resolve_checkpoint_path(path)
    raw = torch.load(str(path), map_location="cpu")
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    state = {key.replace("module.", ""): value for key, value in state.items()}
    ret = model.load_state_dict(state, strict=False)
    missing = list(ret.missing_keys)
    unexpected = list(ret.unexpected_keys)
    return {
        "path": str(path),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "matched": len(missing) == 0 and len(unexpected) == 0,
        "missing_head": missing[:30],
        "unexpected_head": unexpected[:30],
    }


def main():
    args = parse_args()
    model, config = build_model(args.config, args.device)
    candidates = []
    for pattern in args.patterns:
        candidates.extend(glob.glob(pattern))
    candidates = sorted(set(candidates))
    if not candidates:
        raise RuntimeError("No checkpoints found for patterns: {}".format(args.patterns))

    rows = []
    for path in candidates:
        try:
            row = inspect_checkpoint(model, path)
        except Exception as error:
            row = {
                "path": path,
                "error": repr(error),
                "missing_count": None,
                "unexpected_count": None,
                "matched": False,
            }
        rows.append(row)
        label = "MATCH" if row.get("matched") else "mismatch"
        print(
            "{} missing={} unexpected={} {}".format(
                label,
                row.get("missing_count"),
                row.get("unexpected_count"),
                row["path"],
            )
        )
        if not row.get("matched"):
            if row.get("missing_head"):
                print("  missing:", row["missing_head"][:5])
            if row.get("unexpected_head"):
                print("  unexpected:", row["unexpected_head"][:5])
            if row.get("error"):
                print("  error:", row["error"])

    out = {
        "config": args.config,
        "model_name": config.model.name,
        "candidates": rows,
        "matches": [row for row in rows if row.get("matched")],
    }
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(out, file, indent=2)
    print("Wrote {}".format(out_path))


if __name__ == "__main__":
    main()
