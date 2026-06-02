#!/usr/bin/env python3
"""
Summarize evidence for an AMS-after explicit relative-depth module.

Inputs are JSON summaries produced by:
  - tools/analyze_relative_depth_gap.py
  - analyze_fd_decoupling.py
  - analyze_fd_part_block.py
  - analyze_fd_pose_alignment.py
"""

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional


def load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "N/A"
    if isinstance(x, float):
        if math.isnan(x):
            return "NaN"
        return f"{x:.{digits}f}"
    return str(x)


def add_evidence(rows: List[Dict[str, str]], dimension: str, metric: str, value: Any, interpretation: str):
    rows.append({
        "dimension": dimension,
        "metric": metric,
        "value": fmt(value),
        "interpretation": interpretation,
    })


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--relative", default="relative_depth_gap_analysis_512/summary.json")
    p.add_argument("--decoupling", default="depth_analysis_fd_smoke/summary.json")
    p.add_argument("--parts", default="depth_analysis_fd_parts/summary.json")
    p.add_argument("--pose", default="depth_analysis_fd_pose/summary.json")
    p.add_argument("--output", default="depth_relation_motivation_report.md")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rel = load_json(args.relative)
    dec = load_json(args.decoupling)
    parts = load_json(args.parts)
    pose = load_json(args.pose)
    rows: List[Dict[str, str]] = []

    if rel:
        add_evidence(
            rows,
            "Implicit vs explicit relation",
            "raw F_d distance vs |ΔZ| Spearman",
            rel["unsupervised_pairwise_all"]["spearman_feature_distance_vs_abs_depth_gap"],
            "Raw AMS feature geometry does not organize joints by true relative depth.",
        )
        add_evidence(
            rows,
            "Implicit vs explicit relation",
            "linear probe R2 for ΔZ from F_i-F_j",
            rel["linear_probe_pairwise_all"]["r2"],
            "Relative depth is present but latent; a lightweight relation transform can expose it.",
        )
        add_evidence(
            rows,
            "Ordinal depth",
            "coarse μ all-pair ordinal acc",
            rel["coarse_mu"]["ordinal_acc_all_pairs"],
            "The scalar depth head alone does not preserve joint ordering reliably.",
        )
        add_evidence(
            rows,
            "Ambiguous limb pairs",
            "linear probe ordinal acc on wrists/elbows/legs",
            rel["linear_probe_pairwise_ambiguous"]["ordinal_acc"],
            "Hard self-occlusion-related pairs remain weaker than all-pair average.",
        )
        add_evidence(
            rows,
            "Global component",
            "F_d body-mean energy fraction",
            rel["fd_body_mean_stats"]["body_mean_energy_fraction"],
            "A large shared component suggests relation-specific residuals are not isolated.",
        )

    if dec:
        add_evidence(
            rows,
            "Feature decoupling",
            "off-diagonal joint-token cosine",
            dec.get("cosine_offdiag_mean"),
            "High inter-joint similarity means vanilla token fusion may blur depth-layout distinctions.",
        )
        add_evidence(
            rows,
            "Feature decoupling",
            "top-3 residual PCA variance",
            dec.get("pca_top3_var"),
            "Depth residual structure is low-dimensional, suitable for explicit relation/layout modeling.",
        )

    if parts:
        add_evidence(
            rows,
            "Body-part structure",
            "3-part within-between cosine gap",
            parts["coarse_3parts"]["gap_mean"],
            "AMS F_d already has body-part grouping; a module can exploit skeleton part priors.",
        )
        add_evidence(
            rows,
            "Body-part structure",
            "5-part positive gap ratio",
            parts["fine_5parts"]["gap_pct_positive"],
            "Fine limb grouping is present but not perfect, leaving room for structured refinement.",
        )

    if pose:
        add_evidence(
            rows,
            "Pose awareness",
            "F_d distance vs 3D pose distance Spearman",
            pose.get("spearman_r"),
            "AMS output is pose-aware, so the missing piece is not generic pose information.",
        )

    lines = []
    lines.append("# Explicit Relative Depth Modeling Motivation")
    lines.append("")
    lines.append("## Claim")
    lines.append("")
    lines.append(
        "AMS already captures useful depth-layout information, but it remains implicit in feature differences "
        "and is not organized as explicit root-relative or pairwise joint-depth relations. Therefore an AMS-after "
        "relation module is motivated independently of UDE denoising."
    )
    lines.append("")
    lines.append("## Evidence")
    lines.append("")
    lines.append("| Dimension | Metric | Value | Interpretation |")
    lines.append("|---|---:|---:|---|")
    for row in rows:
        lines.append(
            f"| {row['dimension']} | {row['metric']} | {row['value']} | {row['interpretation']} |"
        )
    lines.append("")
    lines.append("## Module Direction")
    lines.append("")
    lines.append("- Add the module after AMS and before multimodal token fusion.")
    lines.append("- Convert `F_d` into root-relative depth tokens and pairwise/part-wise depth-layout tokens.")
    lines.append("- Supervise or regularize with ordinal/pairwise depth relations from GT Z during training.")
    lines.append("- This module addresses representation conversion and skeletal depth geometry, not noise suppression.")
    lines.append("")
    lines.append("## Candidate Tests For A New Module")
    lines.append("")
    lines.append("- Pairwise ordinal depth accuracy on all joint pairs, bones, and ambiguous limb pairs.")
    lines.append("- Raw relation-token geometry correlation with `|Z_i-Z_j|` after the module.")
    lines.append("- MPJPE change under self-occlusion-like pairs or large wrist/elbow depth gaps.")
    lines.append("- Whether relation tokens reduce body-mean dominance while preserving pose-awareness.")

    out = os.path.abspath(args.output)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[done] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
