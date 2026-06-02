#!/usr/bin/env python3
from __future__ import annotations
"""
Non-UDE ASFnet design diagnostics.

This script reads the ASFnet paper assumptions against the local DGLifting
implementation and runs small synthetic probes. It deliberately avoids judging
or validating UDE; the focus is AMS, feature alignment, coordinate handling,
depth-path causality, and forward-pass side effects.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    F = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@dataclass
class Finding:
    severity: str
    category: str
    issue: str
    evidence: str
    why_it_matters: str
    suggested_test_or_fix: str


def read_text(rel_path: str) -> str:
    with open(os.path.join(PROJECT_ROOT, rel_path), "r", encoding="utf-8") as f:
        return f.read()


def add_finding(
    findings: List[Finding],
    severity: str,
    category: str,
    issue: str,
    evidence: str,
    why_it_matters: str,
    suggested_test_or_fix: str,
) -> None:
    findings.append(
        Finding(
            severity=severity,
            category=category,
            issue=issue,
            evidence=evidence,
            why_it_matters=why_it_matters,
            suggested_test_or_fix=suggested_test_or_fix,
        )
    )


def static_paper_code_checks() -> List[Finding]:
    findings: List[Finding] = []
    dgl = read_text("mvn/models/DGLifting.py")
    dgpose = read_text("mvn/models/DGPose.py")
    loss = read_text("mvn/models/loss.py")

    if "features_list_hr.append(depth_images)" in dgl:
        add_finding(
            findings,
            "high",
            "forward_side_effect",
            "DGLifting.forward mutates features_list_hr in-place",
            "mvn/models/DGLifting.py appends depth_images directly to features_list_hr.",
            "A reused feature list silently keeps stale depth tokens. This can make repeated probes or ablations compare the wrong depth map.",
            "Use features_list_hr = list(features_list_hr) + [depth_images], then add a regression test that list length stays unchanged after forward.",
        )

    if "padding_mode='border'" in dgl or 'padding_mode="border"' in dgl:
        add_finding(
            findings,
            "high",
            "ams_sampling",
            "AMS uses border padding for refined sampling",
            "DGLifting.py DeformableBlock calls F.grid_sample(..., padding_mode='border', align_corners=True).",
            "The paper says coordinates outside [-1, 1] yield zero-valued features. Border padding instead copies edge features, so large offsets can harvest border artifacts rather than being penalized.",
            "Change refined AMS sampling to padding_mode='zeros' or explicitly justify border behavior with an out-of-bound sampling test.",
        )

    if "keypoints_2d_cpn_crop[..., :2] /=" in dgpose and "keypoints_2d_cpn_crop[..., :2] -=" in dgpose:
        add_finding(
            findings,
            "high",
            "coordinate_handling",
            "DepthGuidedPose.forward normalizes crop coordinates in-place",
            "mvn/models/DGPose.py uses /= and -= on keypoints_2d_cpn_crop.",
            "The training loop often passes clone(), but the model contract itself is unsafe. Reusing the same tensor across calls double-normalizes coordinates and breaks grid_sample alignment.",
            "Clone inside DGPose.forward before normalization and add a test that caller input is unchanged.",
        )

    if "embed_dim_ratio = 128" in dgl and "base_dim = 32" in dgl and "depth = 4" in dgl:
        add_finding(
            findings,
            "medium",
            "config_contract",
            "Core AMS/fusion dimensions ignore config",
            "DGLifting.__init__ hard-codes embed_dim_ratio=128, base_dim=32, depth=4.",
            "The paper reports Cfuse=128 and 4 AMS iterations, but hard-coding makes ablations and replacement modules harder to compare fairly across configs.",
            "Read these values from config with paper defaults as fallbacks, then test a small non-default config can instantiate.",
        )

    if "torch.clamp" not in loss and ".clamp(" not in loss:
        add_finding(
            findings,
            "medium",
            "training_objective",
            "Depth log-variance is not clipped as described in the paper",
            "mvn/models/loss.py has no clamp for s before exp(-s).",
            "Although this is tied to the auxiliary depth head, it affects training stability for any replacement depth module that still emits supervised depth.",
            "Clamp s to [-9, 9] in the depth auxiliary loss and add an extreme-s test that loss stays finite.",
        )

    if "depth_images = self.depth_embed(depth_images.unsqueeze(1))" in dgl:
        add_finding(
            findings,
            "medium",
            "depth_representation",
            "Depth branch uses a shallow conv on scalar depth maps",
            "DGLifting.py maps [B,H,W] depth to Cfuse with one 3x3 Conv2d.",
            "The paper denotes D as a depth feature representation. A single conv may under-represent local discontinuities that AMS is supposed to exploit.",
            "Use a depth-edge synthetic probe: perturb a narrow limb-like depth discontinuity and verify depth tokens change more near affected joints than far joints.",
        )

    return findings


def make_synthetic_inputs(batch: int, joints: int, device) -> Dict[str, object]:
    torch.manual_seed(7)
    keypoints_2d = torch.randn(batch, joints, 2, device=device)
    ref = torch.empty(batch, joints, 2, device=device).uniform_(-0.75, 0.75)
    depth = torch.rand(batch, 256, 192, device=device)
    features = [
        torch.randn(batch, 32, 64, 48, device=device),
        torch.randn(batch, 64, 32, 24, device=device),
        torch.randn(batch, 128, 16, 12, device=device),
        torch.randn(batch, 256, 8, 6, device=device),
    ]
    return {
        "keypoints_2d": keypoints_2d,
        "ref": ref,
        "depth": depth,
        "features": features,
    }


def _fresh_features(features: List[torch.Tensor]) -> List[torch.Tensor]:
    return [f.clone() for f in features]


def synthetic_forward_probes(device) -> List[Finding]:
    if torch is None:
        return [
            Finding(
                severity="medium",
                category="environment",
                issue="Synthetic forward probes skipped because PyTorch is unavailable",
                evidence="python3 cannot import torch in the current shell.",
                why_it_matters="Static checks still run, but causal depth-path probes need the same Python environment used for training.",
                suggested_test_or_fix="Activate the training environment with torch installed, then rerun without --skip_synthetic.",
            )
        ]

    from mvn.models.DGLifting import DGLifting

    findings: List[Finding] = []
    model = DGLifting().to(device)
    model.eval()
    data = make_synthetic_inputs(batch=2, joints=17, device=device)
    keypoints_2d = data["keypoints_2d"]
    ref = data["ref"]
    depth = data["depth"]
    features = data["features"]

    with torch.no_grad():
        reusable_features = _fresh_features(features)
        before_len = len(reusable_features)
        model(keypoints_2d, ref, depth, reusable_features)
        after_len = len(reusable_features)
        if after_len != before_len:
            add_finding(
                findings,
                "high",
                "forward_side_effect",
                "Synthetic probe confirmed feature-list mutation",
                f"features_list length changed from {before_len} to {after_len} after one DGLifting.forward call.",
                "A second call with the same list will contain old embedded depth and can ignore the new appended depth in parts of AMS.",
                "Make the list append functional, then rerun this script and expect unchanged length.",
            )

        depth_a = depth
        depth_b = torch.flip(depth, dims=[2])
        fresh_a = _fresh_features(features)
        fresh_b = _fresh_features(features)
        out_a, _, _ = model(keypoints_2d, ref, depth_a, fresh_a)
        out_b_fresh, _, _ = model(keypoints_2d, ref, depth_b, fresh_b)

        reused = _fresh_features(features)
        model(keypoints_2d, ref, depth_a, reused)
        fresh_delta = torch.norm(out_a - out_b_fresh).mean().item()
        try:
            out_b_reused, _, _ = model(keypoints_2d, ref, depth_b, reused)
            reuse_delta: Optional[float] = torch.norm(out_a - out_b_reused).mean().item()
            stale_gap: Optional[float] = abs(fresh_delta - reuse_delta)
        except Exception as exc:
            reuse_delta = None
            stale_gap = None
            add_finding(
                findings,
                "high",
                "forward_side_effect",
                "Reusing feature lists can crash DGLifting.forward",
                f"Second forward with the same feature list raised {type(exc).__name__}: {exc}",
                "This confirms the feature-list append is not just a measurement artifact; it can break repeated calls and any wrapper that caches/reuses backbone features.",
                "Make DGLifting.forward copy the list before appending depth, then rerun this script and expect no exception.",
            )

        if stale_gap is not None and reuse_delta is not None and stale_gap > 1e-5:
            add_finding(
                findings,
                "high",
                "depth_path_causality",
                "Reusing feature lists changes the measured depth sensitivity",
                f"fresh depth-change delta={fresh_delta:.6f}, reused-list delta={reuse_delta:.6f}, gap={stale_gap:.6f}.",
                "Ablations that compare real/random/zero depth can be contaminated by stale depth features if they reuse feature lists.",
                "Ensure every forward receives a fresh list or make DGLifting.forward copy the list internally.",
            )

    return findings


def boundary_sampling_probe() -> Dict[str, float]:
    if torch is None or F is None:
        return {
            "status": "skipped_no_torch",
        }
    feature = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
    grid = torch.tensor(
        [[[[1.25, 1.25], [2.0, 2.0], [0.0, 0.0]]]],
        dtype=torch.float32,
    )
    border = F.grid_sample(feature, grid, padding_mode="border", align_corners=True)
    zeros = F.grid_sample(feature, grid, padding_mode="zeros", align_corners=True)
    return {
        "slightly_out_border_value": float(border[0, 0, 0, 0].item()),
        "slightly_out_zeros_value": float(zeros[0, 0, 0, 0].item()),
        "far_out_border_value": float(border[0, 0, 0, 1].item()),
        "far_out_zeros_value": float(zeros[0, 0, 0, 1].item()),
        "in_bounds_border_value": float(border[0, 0, 0, 2].item()),
        "in_bounds_zeros_value": float(zeros[0, 0, 0, 2].item()),
        "note": (
            "With bilinear interpolation, zeros padding can still return a partial "
            "edge contribution when the sample is only slightly outside; far-out "
            "samples become exactly zero. Border padding keeps copying edge values."
        ),
    }


def summarize(findings: List[Finding], boundary_probe: Dict[str, float]) -> Dict[str, object]:
    counts: Dict[str, int] = {}
    for finding in findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
    return {
        "scope": "Non-UDE diagnostics for ASFnet/DGLifting",
        "paper_assumptions_checked": [
            "AMS should iteratively refine sampling around 2D pose guided locations.",
            "Out-of-range grid_sample coordinates should produce zero-valued features.",
            "Depth and image features should be aligned to pose without stale forward state.",
            "Forward calls should not mutate caller tensors or feature lists.",
        ],
        "severity_counts": counts,
        "boundary_sampling_probe": boundary_probe,
        "findings": [asdict(f) for f in findings],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip_synthetic", action="store_true")
    parser.add_argument("--output", default="non_ude_design_gaps_report.json")
    parser.add_argument("--fail_on_high", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if torch is None:
        device = "cpu"
    else:
        device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    findings = static_paper_code_checks()
    if not args.skip_synthetic:
        findings.extend(synthetic_forward_probes(device))

    boundary_probe = boundary_sampling_probe()
    report = summarize(findings, boundary_probe)
    out_path = os.path.abspath(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[done] wrote {out_path}")

    if args.fail_on_high and any(f.severity == "high" for f in findings):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
