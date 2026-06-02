"""
UDE normal-inference diagnostic analyzer.

This version only uses regular inference diagnostics and does NOT use
random-depth / zero-depth stress tests.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple


DEFAULT_WEIGHTS = {
    "uncertainty_calibration": 0.35,
    "ordinal_metric_consistency": 0.30,
    "affine_stability": 0.20,
    "joint_depth_structure": 0.15,
}


def classify_rationality_level(final_score: float) -> str:
    if final_score >= 80:
        return "合理"
    if final_score >= 60:
        return "部分合理"
    return "不合理"


def compute_weighted_score(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    total = 0.0
    for key, w in weights.items():
        total += float(scores.get(key, 0.0)) * float(w)
    return total


def _extract_first_float(text: str) -> float:
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if m is None:
        raise ValueError(f"Cannot parse float from: {text}")
    return float(m.group(0))


def parse_rigorous_metrics(stdout: str) -> Dict[str, float]:
    """
    Parse metrics printed by tools/analyze_depth_rigorous.py.
    Expected section:
      metric_name:
        mean: ... | std: ... | range: [...]
    """
    metrics: Dict[str, float] = {}
    lines = stdout.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.endswith(":") and i + 1 < len(lines):
            key = line[:-1]
            nxt = lines[i + 1].strip()
            if nxt.startswith("mean:"):
                mean_str = nxt.split("|")[0].replace("mean:", "").strip()
                try:
                    metrics[key] = _extract_first_float(mean_str)
                except ValueError:
                    pass
                i += 2
                continue
        i += 1
    return metrics


def analyze_code_risks(project_root: str, code_paths: List[str]) -> List[Dict[str, str]]:
    findings: List[Dict[str, str]] = []

    texts: Dict[str, str] = {}
    for rel_path in code_paths:
        abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(project_root, rel_path)
        if not os.path.exists(abs_path):
            findings.append({
                "severity": "medium",
                "path": rel_path,
                "issue": "file_missing",
                "detail": "File not found, cannot run consistency checks.",
            })
            continue
        with open(abs_path, "r", encoding="utf-8") as f:
            texts[rel_path] = f.read()

    dgl = texts.get("mvn/models/DGLifting.py", "")
    loss_py = texts.get("mvn/models/loss.py", "")
    util_py = texts.get("tools/analyze_depth_utilization.py", "")

    # Risk 1: UMap(s) softmax dimension choice
    if "joint_uncer = F.softmax(self.attn_fc(uncer), dim=1)" in dgl:
        findings.append({
            "severity": "high",
            "path": "mvn/models/DGLifting.py",
            "issue": "umap_softmax_dim_over_joints",
            "detail": "UMap uses softmax over joint dimension (dim=1). Paper-style UMap usually normalizes feature channel dimension.",
        })

    # Risk 2: Missing log-variance clamp mentioned in paper
    if "torch.clamp" not in loss_py and "clamp(" not in loss_py:
        findings.append({
            "severity": "high",
            "path": "mvn/models/loss.py",
            "issue": "missing_log_variance_clamp",
            "detail": "No explicit clamp for log-variance s was found in depth uncertainty loss.",
        })

    # Risk 3: Diagnostic mismatch for sigma conversion
    if "uncertainties = torch.exp(uncer)" in util_py:
        findings.append({
            "severity": "medium",
            "path": "tools/analyze_depth_utilization.py",
            "issue": "sigma_conversion_mismatch",
            "detail": "Diagnostic computes sigma as exp(s), while common heteroscedastic form uses sigma=exp(s/2).",
        })

    # Risk 4: Hard-coded depth loss weight in train loop
    train_path = os.path.join(project_root, "train.py")
    if os.path.exists(train_path):
        with open(train_path, "r", encoding="utf-8") as f:
            train_text = f.read()
        if "loss_d*0.00001" in train_text:
            findings.append({
                "severity": "medium",
                "path": "train.py",
                "issue": "hardcoded_depth_loss_weight",
                "detail": "Depth uncertainty loss weight is hard-coded (1e-5) in training loop.",
            })

    return findings


def _score_uncertainty_calibration(metrics: Dict[str, float]) -> Tuple[float, str]:
    corr = metrics.get("uncertainty_error_correlation", 0.0)
    high_err = metrics.get("error_high_uncertainty", 0.0)
    low_err = metrics.get("error_low_uncertainty", 0.0)

    score = 0.0
    if corr >= 0.35:
        score += 70.0
    elif corr >= 0.2:
        score += 55.0
    elif corr >= 0.1:
        score += 40.0
    elif corr > 0:
        score += 22.0

    if high_err > low_err:
        ratio = high_err / max(low_err, 1e-8)
        if ratio >= 1.35:
            score += 30.0
        elif ratio >= 1.2:
            score += 22.0
        elif ratio >= 1.1:
            score += 14.0
        else:
            score += 8.0

    score = min(score, 100.0)
    comment = (
        f"corr={corr:.3f}, high_unc_err={high_err:.4f}, "
        f"low_unc_err={low_err:.4f}"
    )
    return score, comment


def _score_ordinal_metric_consistency(metrics: Dict[str, float]) -> Tuple[float, str]:
    ordinal_acc = metrics.get("ordinal_accuracy", 0.0)
    metric_bone_error = metrics.get("metric_bone_depth_error_mm", 1e9)
    affine_r2 = metrics.get("affine_r2_mean", 0.0)

    score = 0.0
    if ordinal_acc >= 0.9:
        score += 45.0
    elif ordinal_acc >= 0.8:
        score += 35.0
    elif ordinal_acc >= 0.7:
        score += 25.0
    elif ordinal_acc >= 0.6:
        score += 15.0

    if affine_r2 >= 0.75:
        score += 35.0
    elif affine_r2 >= 0.55:
        score += 26.0
    elif affine_r2 >= 0.35:
        score += 16.0
    elif affine_r2 > 0:
        score += 8.0

    if metric_bone_error <= 0.02:
        score += 20.0
    elif metric_bone_error <= 0.04:
        score += 14.0
    elif metric_bone_error <= 0.06:
        score += 8.0
    elif metric_bone_error <= 0.1:
        score += 4.0

    score = min(score, 100.0)
    comment = (
        f"ordinal_acc={ordinal_acc:.3f}, affine_r2={affine_r2:.3f}, "
        f"metric_bone_depth_error={metric_bone_error:.4f}"
    )
    return score, comment


def _score_affine_stability(metrics: Dict[str, float]) -> Tuple[float, str]:
    alpha_mean = metrics.get("alpha_mean", 0.0)
    alpha_std = metrics.get("alpha_std", 1e9)
    beta_mean = metrics.get("beta_mean", 0.0)
    beta_std = metrics.get("beta_std", 1e9)

    score = 0.0
    alpha_dist = abs(alpha_mean - 1.0)
    if alpha_dist <= 0.1:
        score += 35.0
    elif alpha_dist <= 0.2:
        score += 26.0
    elif alpha_dist <= 0.35:
        score += 15.0
    elif alpha_dist <= 0.5:
        score += 8.0

    if alpha_std <= 0.15:
        score += 30.0
    elif alpha_std <= 0.25:
        score += 22.0
    elif alpha_std <= 0.4:
        score += 14.0
    elif alpha_std <= 0.6:
        score += 6.0

    if beta_std <= 0.02:
        score += 20.0
    elif beta_std <= 0.05:
        score += 14.0
    elif beta_std <= 0.1:
        score += 8.0
    else:
        score += 2.0

    if abs(beta_mean) <= 0.02:
        score += 15.0
    elif abs(beta_mean) <= 0.05:
        score += 10.0
    elif abs(beta_mean) <= 0.1:
        score += 6.0
    else:
        score += 2.0

    score = min(score, 100.0)
    comment = (
        f"alpha={alpha_mean:.3f}±{alpha_std:.3f}, "
        f"beta={beta_mean:.3f}±{beta_std:.3f}"
    )
    return score, comment


def _score_joint_depth_structure(metrics: Dict[str, float]) -> Tuple[float, str]:
    grad_mean = metrics.get("grad_mag_mean", 0.0)
    grad_std = metrics.get("grad_mag_std", 0.0)
    joint_depth_var = metrics.get("joint_depth_variance", 0.0)
    depth_joint_mean = metrics.get("depth_at_joint_mean", 0.0)

    score = 0.0
    if grad_mean >= 0.02:
        score += 40.0
    elif grad_mean >= 0.01:
        score += 30.0
    elif grad_mean >= 0.005:
        score += 20.0
    elif grad_mean > 0:
        score += 10.0

    if joint_depth_var >= 0.01:
        score += 35.0
    elif joint_depth_var >= 0.005:
        score += 24.0
    elif joint_depth_var >= 0.001:
        score += 14.0
    else:
        score += 6.0

    if grad_std <= max(grad_mean * 2.5, 1e-6):
        score += 15.0
    else:
        score += 8.0

    if depth_joint_mean > 0:
        score += 10.0

    score = min(score, 100.0)
    comment = (
        f"grad_mean={grad_mean:.4f}, grad_std={grad_std:.4f}, "
        f"joint_depth_variance={joint_depth_var:.4f}, depth_at_joint_mean={depth_joint_mean:.4f}"
    )
    return score, comment


def score_from_rigorous_metrics(metrics: Dict[str, float]) -> Dict:
    dim_scores = {}
    dim_comments = {}

    dim_scores["uncertainty_calibration"], dim_comments["uncertainty_calibration"] = _score_uncertainty_calibration(metrics)
    dim_scores["ordinal_metric_consistency"], dim_comments["ordinal_metric_consistency"] = _score_ordinal_metric_consistency(metrics)
    dim_scores["affine_stability"], dim_comments["affine_stability"] = _score_affine_stability(metrics)
    dim_scores["joint_depth_structure"], dim_comments["joint_depth_structure"] = _score_joint_depth_structure(metrics)

    final_score = compute_weighted_score(dim_scores, DEFAULT_WEIGHTS)
    level = classify_rationality_level(final_score)
    return {
        "dimension_scores": dim_scores,
        "dimension_comments": dim_comments,
        "weights": DEFAULT_WEIGHTS,
        "final_score": final_score,
        "level": level,
    }


def build_rigorous_command(
    python_exec: str,
    config_path: str,
    checkpoint_path: str,
    num_samples: int,
    batch_size: int,
    device: str,
    conda_env: str,
) -> List[str]:
    cmd = [
        python_exec,
        "tools/analyze_depth_utilization.py",
        "--config",
        config_path,
        "--checkpoint",
        checkpoint_path,
        "--num_samples",
        str(num_samples),
        "--batch_size",
        str(batch_size),
        "--device",
        device,
    ]
    if conda_env and shutil.which("conda") is not None:
        return ["conda", "run", "-n", conda_env] + cmd
    return cmd


def run_rigorous_analysis(
    project_root: str,
    python_exec: str,
    config_path: str,
    checkpoint_path: str,
    num_samples: int,
    batch_size: int,
    device: str,
    conda_env: str,
) -> Tuple[str, int, List[str]]:
    cmd = build_rigorous_command(
        python_exec=python_exec,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        num_samples=num_samples,
        batch_size=batch_size,
        device=device,
        conda_env=conda_env,
    )
    proc = subprocess.run(
        cmd,
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return combined, proc.returncode, cmd


def render_markdown_report(result: Dict) -> str:
    lines = []
    lines.append("# UDE 正常推理诊断报告")
    lines.append("")
    lines.append(f"- 结论: **{result['scoring']['level']}**")
    lines.append(f"- 综合分数: **{result['scoring']['final_score']:.2f}/100**")
    lines.append(f"- 时间: `{result['created_at']}`")
    lines.append("")
    lines.append("## 执行信息")
    lines.append(f"- 命令: `{result['command']}`")
    lines.append(f"- 返回码: `{result['return_code']}`")
    lines.append("")
    lines.append("## 关键原始指标")
    for k in sorted(result["rigorous_metrics"].keys()):
        lines.append(f"- `{k}`: {result['rigorous_metrics'][k]:.6f}")
    lines.append("")
    lines.append("## 维度评分")
    for k, v in result["scoring"]["dimension_scores"].items():
        lines.append(f"- `{k}`: {v:.2f}")
        lines.append(f"  - {result['scoring']['dimension_comments'][k]}")
    lines.append("")
    lines.append("## 代码一致性风险")
    if not result.get("code_risks"):
        lines.append("- 未发现显著规则命中风险")
    else:
        for item in result["code_risks"]:
            lines.append(
                f"- [{item['severity']}] `{item['path']}` `{item['issue']}`: {item['detail']}"
            )
    lines.append("")
    lines.append("## 判定建议")
    lines.append("- 本报告仅基于真实输入下的正常推理，不包含随机/零深度测试。")
    lines.append("- 若 `uncertainty_error_correlation <= 0`，说明 UDE 不确定性分支未校准。")
    lines.append("- 若 `affine_r2_mean` 偏低且 `alpha_std` 偏高，说明深度映射稳定性不足。")
    return "\n".join(lines) + "\n"


def run_pipeline(args: argparse.Namespace) -> Dict:
    project_root = os.path.abspath(args.project_root)
    warning_messages: List[str] = []
    if args.conda_env and shutil.which("conda") is None:
        warning_messages.append(
            f"conda not found in PATH, fallback to direct `{args.python_exec}` execution."
        )

    stdout, return_code, cmd = run_rigorous_analysis(
        project_root=project_root,
        python_exec=args.python_exec,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device,
        conda_env=args.conda_env,
    )

    if return_code != 0:
        raise RuntimeError(
            "Normal-inference analysis failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Output:\n{stdout}"
        )

    rigorous_metrics = parse_rigorous_metrics(stdout)
    if not rigorous_metrics:
        raise RuntimeError(
            "No metrics parsed. Please check analyze_depth_utilization.py output format."
        )

    scoring = score_from_rigorous_metrics(rigorous_metrics)
    code_risks = analyze_code_risks(
        project_root=project_root,
        code_paths=args.code_paths,
    )
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(cmd),
        "return_code": return_code,
        "warnings": warning_messages,
        "rigorous_metrics": rigorous_metrics,
        "scoring": scoring,
        "code_risks": code_risks,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Normal-inference UDE diagnostic analyzer")
    parser.add_argument("--project_root", type=str, default=".", help="Project root")
    parser.add_argument("--python_exec", type=str, default="python3", help="Python executable")
    parser.add_argument("--conda_env", type=str, default="asfnet", help="Conda env name; empty to disable")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/human36m/human36m_single.yaml",
        help="Config for analyze_depth_utilization.py",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint/h36m_v2b.bin",
        help="Checkpoint for analyze_depth_utilization.py",
    )
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="ude_rationality_analysis")
    parser.add_argument(
        "--code_paths",
        nargs="+",
        default=[
            "mvn/models/DGLifting.py",
            "mvn/models/loss.py",
            "tools/analyze_depth_utilization.py",
        ],
        help="Code files for static consistency checks",
    )
    args = parser.parse_args()

    result = run_pipeline(args)
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "ude_rationality_data_report.json")
    md_path = os.path.join(args.output_dir, "ude_rationality_data_report.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown_report(result))

    print(f"[DONE] level={result['scoring']['level']}, final_score={result['scoring']['final_score']:.2f}")
    print(f"[DONE] json={json_path}")
    print(f"[DONE] markdown={md_path}")


if __name__ == "__main__":
    main()
