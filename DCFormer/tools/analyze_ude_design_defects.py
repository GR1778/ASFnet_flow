"""
UDE design-defect analyzer (normal inference only).

Goal:
1) Test whether uncertainty s is causally used by UDE.
2) Test whether uncertainty-to-gating relation is monotonic as expected.
3) Compare mu / s contribution to final 3D performance.

This script does NOT use random-depth/zero-depth stress tests.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, ".")

from mvn.models.DGPose import DepthGuidedPose
from mvn.utils.cfg import config, update_config
from mvn import datasets


def root_aligned_mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred_rel = pred - pred[:, 0:1, :]
    gt_rel = gt - gt[:, 0:1, :]
    return torch.norm(pred_rel - gt_rel, dim=-1).mean().item()


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return 0.0
    aa = a[mask]
    bb = b[mask]
    if np.std(aa) < 1e-8 or np.std(bb) < 1e-8:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


class UDEDesignAnalyzer:
    def __init__(self, model: DepthGuidedPose, device: torch.device):
        self.model = model
        self.device = device

    def _forward_u_decomposed(
        self,
        images: torch.Tensor,
        keypoints_2d: torch.Tensor,
        keypoints_2d_crop: torch.Tensor,
        depth_images: torch.Tensor,
        intervention: str = "none",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward with optional intervention on mu/s while keeping inputs normal.
        intervention: none | s_shuffle | s_zero | mu_shuffle
        """
        lifting = self.model.Lifting_net
        b, j = keypoints_2d.shape[:2]

        x = lifting.coord_embed(keypoints_2d)
        depth_embedded = lifting.depth_embed(depth_images.unsqueeze(1))

        images_chw = images.permute(0, 3, 1, 2).contiguous()
        features_list_hr = self.model.backbone(images_chw)
        features_list_hr.append(depth_embedded)

        ref = keypoints_2d_crop[..., :2].clone()
        ref[..., 0] = ref[..., 0] / (192 / 2) - 1
        ref[..., 1] = ref[..., 1] / (256 / 2) - 1

        features_ref_list = [
            F.grid_sample(f, ref.unsqueeze(-2), align_corners=True).squeeze(-1).permute(0, 2, 1).contiguous()
            for f in features_list_hr
        ]
        features_ref_list = [embed(f) for embed, f in zip(lifting.feat_embed, features_ref_list)]

        x = torch.stack([x, *features_ref_list], dim=1)
        x = x + lifting.Spatial_pos_embed

        for blk in lifting.RGBD_Extraction:
            x = blk(x, ref, features_list_hr)

        x_depth_raw = x[:, -1]
        mu, s = lifting.depth_uncer(x_depth_raw)

        if intervention == "s_shuffle":
            perm = torch.randperm(j, device=s.device)
            s_used = s[:, perm]
            mu_used = mu
        elif intervention == "s_zero":
            s_used = torch.zeros_like(s)
            mu_used = mu
        elif intervention == "mu_shuffle":
            perm = torch.randperm(j, device=mu.device)
            mu_used = mu[:, perm]
            s_used = s
        else:
            mu_used = mu
            s_used = s

        z_value = lifting.z_embed(mu_used) + lifting.Spatial_pos_embed2
        joint_uncer = F.softmax(lifting.attn_fc(s_used), dim=1)
        joint_importance = joint_uncer.mean(dim=-1)  # [B, J]

        fcat = torch.cat([joint_uncer, z_value, x_depth_raw], dim=-1)

        b_attn, n_attn, c_attn = fcat.shape
        qkv = lifting.attn_depth.qkv(fcat).reshape(
            b_attn, n_attn, 3, lifting.attn_depth.num_heads, c_attn // lifting.attn_depth.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv[0], qkv[1], qkv[2]
        attn_logits = (q @ k.transpose(-2, -1)) * lifting.attn_depth.scale
        attn_weights = attn_logits.softmax(dim=-1)  # [B, H, J, J]

        x_depth_enhanced = lifting.attn_depth(fcat)
        x_fused = torch.cat((x[:, :-1], x_depth_enhanced.unsqueeze(1)), dim=1)

        x_fused = rearrange(x_fused, "b l p c -> (b p) l c")
        for blk in lifting.Features_Fusion:
            x_fused = blk(x_fused)

        x_fused = rearrange(x_fused, "(b p) l c -> b p (l c)", b=b)
        for blk in lifting.Spatial_Transformer:
            x_fused = blk(x_fused)

        pred_3d = lifting.head(x_fused).view(b, 1, j, 3).squeeze(1)
        return {
            "pred_3d": pred_3d,
            "mu": mu.squeeze(-1),
            "s": s.squeeze(-1),
            "joint_importance": joint_importance,
            "attn_weights": attn_weights,
        }

    @torch.no_grad()
    def analyze_batch(
        self,
        images: torch.Tensor,
        keypoints_2d: torch.Tensor,
        keypoints_2d_crop: torch.Tensor,
        depth_images: torch.Tensor,
        gt_3d: torch.Tensor,
    ) -> Dict[str, float]:
        if gt_3d.dim() == 4:
            gt_3d = gt_3d.squeeze(1)

        base = self._forward_u_decomposed(images, keypoints_2d, keypoints_2d_crop, depth_images, "none")
        s_shuffle = self._forward_u_decomposed(images, keypoints_2d, keypoints_2d_crop, depth_images, "s_shuffle")
        s_zero = self._forward_u_decomposed(images, keypoints_2d, keypoints_2d_crop, depth_images, "s_zero")
        mu_shuffle = self._forward_u_decomposed(images, keypoints_2d, keypoints_2d_crop, depth_images, "mu_shuffle")

        mpjpe_base = root_aligned_mpjpe(base["pred_3d"], gt_3d)
        mpjpe_s_shuffle = root_aligned_mpjpe(s_shuffle["pred_3d"], gt_3d)
        mpjpe_s_zero = root_aligned_mpjpe(s_zero["pred_3d"], gt_3d)
        mpjpe_mu_shuffle = root_aligned_mpjpe(mu_shuffle["pred_3d"], gt_3d)

        s_np = base["s"].detach().cpu().numpy().reshape(-1)
        imp_np = base["joint_importance"].detach().cpu().numpy().reshape(-1)
        s_gate_corr = safe_corr(s_np, imp_np)

        pred_err = torch.norm((base["pred_3d"] - base["pred_3d"][:, 0:1]) - (gt_3d - gt_3d[:, 0:1]), dim=-1)
        err_np = pred_err.detach().cpu().numpy().reshape(-1)
        s_err_corr = safe_corr(s_np, err_np)

        attn = base["attn_weights"].detach().cpu().numpy()  # [B,H,J,J]
        mean_attn = attn.mean(axis=(0, 1))
        diag = np.eye(mean_attn.shape[0], dtype=bool)
        diag_w = float(mean_attn[diag].mean())
        offdiag_w = float(mean_attn[~diag].mean())

        return {
            "mpjpe_base_mm": mpjpe_base * 1000.0,
            "mpjpe_s_shuffle_mm": mpjpe_s_shuffle * 1000.0,
            "mpjpe_s_zero_mm": mpjpe_s_zero * 1000.0,
            "mpjpe_mu_shuffle_mm": mpjpe_mu_shuffle * 1000.0,
            "delta_s_shuffle_mm": (mpjpe_s_shuffle - mpjpe_base) * 1000.0,
            "delta_s_zero_mm": (mpjpe_s_zero - mpjpe_base) * 1000.0,
            "delta_mu_shuffle_mm": (mpjpe_mu_shuffle - mpjpe_base) * 1000.0,
            "s_gate_corr": s_gate_corr,
            "s_error_corr": s_err_corr,
            "attn_diag_weight": diag_w,
            "attn_offdiag_weight": offdiag_w,
            "attn_diag_ratio": diag_w / (offdiag_w + 1e-8),
        }


def aggregate(all_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = all_metrics[0].keys()
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        arr = np.array([m[k] for m in all_metrics], dtype=np.float64)
        out[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return out


def summarize_design_defects(agg: Dict[str, Dict[str, float]]) -> List[Dict[str, str]]:
    findings: List[Dict[str, str]] = []

    d_s = agg["delta_s_shuffle_mm"]["mean"]
    d_sz = agg["delta_s_zero_mm"]["mean"]
    d_mu = agg["delta_mu_shuffle_mm"]["mean"]
    s_gate_corr = agg["s_gate_corr"]["mean"]
    s_err_corr = agg["s_error_corr"]["mean"]

    if d_s < 0.2 and d_sz < 0.2:
        findings.append({
            "severity": "高",
            "issue": "s因果性不足",
            "detail": f"扰动 s 后 MPJPE 仅变化 shuffle={d_s:.3f}mm, zero={d_sz:.3f}mm。",
        })
    if d_mu > d_s * 2.0 and d_mu > 0.5:
        findings.append({
            "severity": "高",
            "issue": "mu主导而s弱化",
            "detail": f"mu 扰动影响 ({d_mu:.3f}mm) 明显大于 s 扰动 ({d_s:.3f}mm)。",
        })
    if s_gate_corr >= -0.05:
        findings.append({
            "severity": "高",
            "issue": "不确定性单调抑制缺失",
            "detail": f"s-门控相关系数={s_gate_corr:.3f}，未体现“高不确定性应更强抑制”的负相关趋势。",
        })
    if s_err_corr <= 0.05:
        findings.append({
            "severity": "中",
            "issue": "不确定性与误差关联弱",
            "detail": f"s-最终误差相关系数={s_err_corr:.3f}，不确定性信息性较弱。",
        })
    if not findings:
        findings.append({
            "severity": "低",
            "issue": "未命中主要设计缺陷",
            "detail": "当前探针未命中主要 UDE 设计缺陷判据。",
        })
    return findings


def to_markdown(result: Dict) -> str:
    lines = []
    lines.append("# UDE 设计缺陷诊断报告")
    lines.append("")
    lines.append(f"- 时间: `{result['created_at']}`")
    lines.append(f"- 样本数: `{result['num_samples']}`")
    lines.append(f"- 检查点: `{result['checkpoint']}`")
    lines.append("")
    lines.append("## 关键指标（均值）")
    for k in sorted(result["metrics"].keys()):
        lines.append(f"- `{k}`: {result['metrics'][k]['mean']:.4f}")
    lines.append("")
    lines.append("## 设计缺陷命中")
    for f in result["findings"]:
        lines.append(f"- [{f['severity']}] `{f['issue']}`: {f['detail']}")
    lines.append("")
    lines.append("## 解读")
    lines.append("- `delta_s_shuffle_mm / delta_s_zero_mm` 越小，说明 s 对最终输出越不具因果性。")
    lines.append("- `delta_mu_shuffle_mm` 远大于 `delta_s_shuffle_mm`，说明系统更依赖 mu 而非 s。")
    lines.append("- `s_gate_corr` 若不显著为负，说明“高不确定->低权重”的结构单调性不足。")
    lines.append("- `s_error_corr` 接近 0 表明不确定性未学成有效误差代理。")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="UDE design-defect analyzer")
    parser.add_argument("--config", type=str, default="experiments/human36m/human36m_single.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoint/h36m_v2b.bin")
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="ude_rationality_analysis")
    args = parser.parse_args()

    update_config(args.config)
    device = torch.device(args.device)

    model = DepthGuidedPose(config, device).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    val_dataset = eval("datasets." + config.dataset.val_dataset)(
        root=config.dataset.root,
        pred_results_path=config.val.pred_results_path,
        depth_image_path=config.dataset.depth_image_path,
        train=False,
        test=True,
        image_shape=config.model.image_shape,
        labels_path=config.dataset.val_labels_path,
        with_damaged_actions=config.val.with_damaged_actions,
        retain_every_n_frames_in_test=100,
        scale_bbox=config.val.scale_bbox,
        kind=config.kind,
        undistort_images=config.val.undistort_images,
        data_format=config.dataset.data_format,
        frame=1,
    )
    sample_n = min(args.num_samples, len(val_dataset))
    idx = np.random.choice(len(val_dataset), sample_n, replace=False)
    dataloader = DataLoader(Subset(val_dataset, idx), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    analyzer = UDEDesignAnalyzer(model, device)

    all_metrics: List[Dict[str, float]] = []
    for batch in dataloader:
        if len(batch) != 5:
            continue
        images, gt_3d, keypoints_2d, keypoints_2d_crop, depth_images = batch
        images = images.float().to(device)
        depth_images = depth_images.float().to(device) / 255.0
        images = torch.flip(images, [-1])
        images = (images / 255.0 - mean) / std
        gt_3d = gt_3d.float().to(device)
        gt_3d[:, :, 1:] -= gt_3d[:, :, :1]
        gt_3d[:, :, 0] = 0
        keypoints_2d = keypoints_2d.float().to(device)
        keypoints_2d_crop = keypoints_2d_crop.float().to(device)

        batch_metrics = analyzer.analyze_batch(
            images=images,
            keypoints_2d=keypoints_2d,
            keypoints_2d_crop=keypoints_2d_crop,
            depth_images=depth_images,
            gt_3d=gt_3d,
        )
        all_metrics.append(batch_metrics)

    if not all_metrics:
        raise RuntimeError("No metrics collected. Check data loader and batch format.")

    agg = aggregate(all_metrics)
    findings = summarize_design_defects(agg)
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": args.checkpoint,
        "num_samples": sample_n,
        "metrics": agg,
        "findings": findings,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "ude_design_defect_report.json")
    md_path = os.path.join(args.output_dir, "ude_design_defect_report.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(to_markdown(result))

    print(f"[DONE] json={json_path}")
    print(f"[DONE] markdown={md_path}")
    print(f"[DONE] findings={len(findings)}")


if __name__ == "__main__":
    main()
