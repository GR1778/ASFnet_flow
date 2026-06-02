#!/usr/bin/env python3
"""
Analyze DLST training signals from an in-progress out.txt log.

This script is intentionally lightweight: it does not load a model checkpoint
or dataset. It parses the epoch MPJPE lines and DLST diagnostic lines emitted by
train.py, then reports whether the DLST ordering branch is learning a useful
signal before training has finished.
"""

import argparse
import json
import math
import os
import re
from glob import glob
from typing import Dict, List, Optional


EPOCH_RE = re.compile(
    r"^\[(?P<epoch>\d+)\]\s+time\s+(?P<time>[-+0-9.eE]+)\s+lr\s+(?P<lr>[-+0-9.eE]+)\s+"
    r"3d_train\s+(?P<train>[-+0-9.eE]+)\s+3d_test_p1\s+(?P<p1>[-+0-9.eE]+)\s+"
    r"3d_test_p2\s+(?P<p2>[-+0-9.eE]+)"
)
DIAG_RE = re.compile(r"^\[(?P<split>train|val) diagnostics\]\s+(?P<body>.*)$")
KV_RE = re.compile(r"(?P<key>[A-Za-z0-9_]+)\s+(?P<value>[-+0-9.eE]+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--log", default=None, help="Path to out.txt. If omitted, use latest logs/ConPose@*/out.txt.")
    p.add_argument("--logs_root", default="logs")
    p.add_argument("--output_dir", default="dlst_training_signal")
    p.add_argument("--target_mpjpe", type=float, default=38.8)
    p.add_argument("--min_good_sign_acc", type=float, default=0.70)
    p.add_argument("--min_early_sign_acc", type=float, default=0.60)
    p.add_argument("--max_entropy_collapse", type=float, default=0.08)
    p.add_argument("--min_usage_floor", type=float, default=0.03)
    return p.parse_args()


def latest_log(logs_root: str) -> str:
    candidates = glob(os.path.join(logs_root, "ConPose@*", "out.txt"))
    if not candidates:
        raise FileNotFoundError(f"No out.txt found under {logs_root}/ConPose@*/")
    return max(candidates, key=os.path.getmtime)


def parse_log(path: str) -> Dict[str, List[Dict[str, float]]]:
    epochs: List[Dict[str, float]] = []
    diagnostics: Dict[str, List[Dict[str, float]]] = {"train": [], "val": []}

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            m = EPOCH_RE.match(line)
            if m:
                epochs.append({
                    "epoch": int(m.group("epoch")),
                    "time_min": float(m.group("time")),
                    "lr": float(m.group("lr")),
                    "train_mpjpe": float(m.group("train")),
                    "p1": float(m.group("p1")),
                    "p2": float(m.group("p2")),
                })
                continue

            m = DIAG_RE.match(line)
            if m:
                item = {kv.group("key"): float(kv.group("value")) for kv in KV_RE.finditer(m.group("body"))}
                # Diagnostics are printed once for train and once for val per epoch.
                item["step"] = len(diagnostics[m.group("split")]) + 1
                diagnostics[m.group("split")].append(item)

    return {"epochs": epochs, "diagnostics": diagnostics}


def trend(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    x = list(range(len(values)))
    xm = sum(x) / len(x)
    ym = sum(values) / len(values)
    denom = sum((v - xm) ** 2 for v in x)
    if denom <= 0:
        return None
    return sum((xi - xm) * (yi - ym) for xi, yi in zip(x, values)) / denom


def finite_mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def tail_values(items: List[Dict[str, float]], key: str, n: int = 3) -> List[float]:
    return [x[key] for x in items[-n:] if key in x and math.isfinite(x[key])]


def summarize(parsed: Dict[str, object], args: argparse.Namespace, log_path: str) -> Dict[str, object]:
    epochs: List[Dict[str, float]] = parsed["epochs"]
    diagnostics: Dict[str, List[Dict[str, float]]] = parsed["diagnostics"]
    val_diag = diagnostics["val"]
    train_diag = diagnostics["train"]

    best = min(epochs, key=lambda x: x["p1"]) if epochs else None
    last = epochs[-1] if epochs else None
    val_sign = [x["dlst_rel_sign_acc"] for x in val_diag if "dlst_rel_sign_acc" in x]
    val_entropy = [x["dlst_assign_entropy"] for x in val_diag if "dlst_assign_entropy" in x]
    val_usage_min = [x["dlst_layer_usage_min"] for x in val_diag if "dlst_layer_usage_min" in x]
    val_usage_max = [x["dlst_layer_usage_max"] for x in val_diag if "dlst_layer_usage_max" in x]
    val_order = [x["order_loss"] for x in val_diag if "order_loss" in x]

    tail_sign = finite_mean(tail_values(val_diag, "dlst_rel_sign_acc"))
    tail_entropy = finite_mean(tail_values(val_diag, "dlst_assign_entropy"))
    tail_usage_min = finite_mean(tail_values(val_diag, "dlst_layer_usage_min"))
    tail_usage_max = finite_mean(tail_values(val_diag, "dlst_layer_usage_max"))

    findings: List[Dict[str, str]] = []
    if not epochs:
        findings.append({
            "severity": "high",
            "issue": "no_epoch_metrics",
            "detail": "No completed epoch line was found. Wait for at least one validation pass.",
        })
    if not val_diag:
        findings.append({
            "severity": "high",
            "issue": "no_val_dlst_diagnostics",
            "detail": "No '[val diagnostics]' line was found. DLST learning cannot be judged from this log yet.",
        })

    if best is not None and best["p1"] <= args.target_mpjpe:
        findings.append({
            "severity": "info",
            "issue": "target_reached",
            "detail": f"Best validation MPJPE is {best['p1']:.1f}, already <= target {args.target_mpjpe:.1f}.",
        })
    elif best is not None:
        findings.append({
            "severity": "info",
            "issue": "target_not_reached_yet",
            "detail": f"Best validation MPJPE is {best['p1']:.1f}; target is {args.target_mpjpe:.1f}.",
        })

    if tail_sign is not None:
        if tail_sign >= args.min_good_sign_acc:
            findings.append({
                "severity": "info",
                "issue": "ordering_signal_good",
                "detail": f"Recent val dlst_rel_sign_acc averages {tail_sign:.3f}, which is a useful ordering signal.",
            })
        elif tail_sign >= args.min_early_sign_acc:
            findings.append({
                "severity": "medium",
                "issue": "ordering_signal_moderate",
                "detail": f"Recent val dlst_rel_sign_acc averages {tail_sign:.3f}; this is plausible early training but not strong yet.",
            })
        else:
            findings.append({
                "severity": "high",
                "issue": "ordering_signal_weak",
                "detail": f"Recent val dlst_rel_sign_acc averages {tail_sign:.3f}, below {args.min_early_sign_acc:.2f}.",
            })

    if tail_entropy is not None:
        if tail_entropy <= args.max_entropy_collapse:
            findings.append({
                "severity": "high",
                "issue": "assignment_entropy_too_low",
                "detail": f"Recent assignment entropy is {tail_entropy:.3f}; layer assignment may be collapsing.",
            })
        elif tail_entropy > 0.95 and tail_sign is not None and tail_sign < args.min_good_sign_acc:
            findings.append({
                "severity": "medium",
                "issue": "assignment_too_uniform",
                "detail": f"Recent assignment entropy is {tail_entropy:.3f}; assignments may be too diffuse to create sharp ordering.",
            })

    if tail_usage_min is not None and tail_usage_min < args.min_usage_floor:
        findings.append({
            "severity": "high",
            "issue": "unused_depth_layer",
            "detail": f"Recent minimum layer usage is {tail_usage_min:.3f}; at least one depth layer is nearly unused.",
        })

    p1_values = [x["p1"] for x in epochs]
    summary = {
        "log_path": log_path,
        "num_epochs": len(epochs),
        "num_train_diagnostics": len(train_diag),
        "num_val_diagnostics": len(val_diag),
        "target_mpjpe": args.target_mpjpe,
        "latest_epoch": last,
        "best_epoch": best,
        "recent_val": {
            "dlst_rel_sign_acc": tail_sign,
            "dlst_assign_entropy": tail_entropy,
            "dlst_layer_usage_min": tail_usage_min,
            "dlst_layer_usage_max": tail_usage_max,
        },
        "trends": {
            "p1_slope_per_epoch": trend(p1_values),
            "val_dlst_rel_sign_acc_slope": trend(val_sign),
            "val_order_loss_slope": trend(val_order),
            "val_assign_entropy_slope": trend(val_entropy),
            "val_layer_usage_min_slope": trend(val_usage_min),
            "val_layer_usage_max_slope": trend(val_usage_max),
        },
        "findings": findings,
        "raw": parsed,
    }
    return summary


def write_report(summary: Dict[str, object], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "summary.json")
    md_path = os.path.join(output_dir, "report.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    latest = summary.get("latest_epoch") or {}
    best = summary.get("best_epoch") or {}
    recent = summary.get("recent_val") or {}
    trends = summary.get("trends") or {}
    findings = summary.get("findings") or []

    lines = [
        "# DLST 训练信号诊断",
        "",
        f"- log: `{summary['log_path']}`",
        f"- completed epochs: {summary['num_epochs']}",
        f"- target MPJPE: {summary['target_mpjpe']:.1f}",
    ]
    if latest:
        lines.append(f"- latest: epoch {latest['epoch']}, p1={latest['p1']:.1f}, p2={latest['p2']:.1f}")
    if best:
        lines.append(f"- best: epoch {best['epoch']}, p1={best['p1']:.1f}, p2={best['p2']:.1f}")

    lines += [
        "",
        "## Recent Val Diagnostics",
    ]
    for key, value in recent.items():
        if value is not None:
            lines.append(f"- {key}: {value:.6f}")

    lines += [
        "",
        "## Trends",
    ]
    for key, value in trends.items():
        if value is not None:
            lines.append(f"- {key}: {value:.6f}")

    lines += [
        "",
        "## Findings",
    ]
    if findings:
        for item in findings:
            lines.append(f"- [{item['severity']}] {item['issue']}: {item['detail']}")
    else:
        lines.append("- No findings yet.")

    lines += [
        "",
        "## Reading Guide",
        "- `dlst_rel_sign_acc` 上升表示 R 的前后排序正在贴近 GT。",
        "- `dlst_assign_entropy` 过低通常是层塌缩，过高且 sign acc 不涨通常是分配太散。",
        "- `layer_usage_min/max` 用来判断 K 个 depth layer 是否都在被使用。",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[write] {json_path}")
    print(f"[write] {md_path}")


def main() -> int:
    args = parse_args()
    log_path = args.log or latest_log(args.logs_root)
    parsed = parse_log(log_path)
    summary = summarize(parsed, args, log_path)
    write_report(summary, args.output_dir)

    best = summary.get("best_epoch")
    recent = summary.get("recent_val", {})
    if best:
        print(f"[summary] best p1={best['p1']:.1f} at epoch {best['epoch']}")
    if recent.get("dlst_rel_sign_acc") is not None:
        print(f"[summary] recent val dlst_rel_sign_acc={recent['dlst_rel_sign_acc']:.3f}")
    for item in summary["findings"]:
        print(f"[{item['severity']}] {item['issue']}: {item['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
