"""
ASFnet Depth Branch Diagnostic Script
======================================
This script analyzes how the depth branch utilizes depth information
without modifying the training code or model architecture.

Run:
    python analyze_depth_utilization.py \
        --checkpoint <path/to/checkpoint> \
        --config experiments/human36m/human36m_single.yaml \
        --num_samples 500
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mvn.models.DGPose import DepthGuidedPose
from mvn.utils.cfg import config, update_config
from mvn import datasets
from mvn.datasets import utils as dataset_utils


class DepthAnalyzer:
    """Hooks into ASFnet forward pass to extract intermediate depth features."""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.hooks = []
        self.activations = {}
        
        # Register hooks on key modules
        self._register_hook('depth_embed', model.Lifting_net.depth_embed)
        self._register_hook('ams_last', model.Lifting_net.RGBD_Extraction[-1])
        self._register_hook('depth_uncer', model.Lifting_net.depth_uncer)
        self._register_hook('ude_attn', model.Lifting_net.attn_depth)
        
    def _register_hook(self, name, module):
        def hook_fn(m, input, output):
            self.activations[name] = output.detach() if not isinstance(output, tuple) else \
                [o.detach() if isinstance(o, torch.Tensor) else o for o in output]
        handle = module.register_forward_hook(hook_fn)
        self.hooks.append(handle)
        
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
            
    def analyze_batch(self, images, keypoints_2d, ref, depth_images, keypoints_3d_gt):
        """
        Analyze a single batch.
        
        Returns:
            dict with various depth utilization metrics
        """
        with torch.no_grad():
            keypoints_3d_pred, coarse_depth, uncer = self.model(
                images, keypoints_2d, ref.clone(), depth_images
            )

        # For robustness, extract from model outputs directly
        metrics = self._compute_metrics(
            keypoints_3d_pred, coarse_depth, uncer,
            keypoints_3d_gt, depth_images, ref
        )
        return metrics
    
    def _compute_metrics(self, pred_3d, coarse_depth, uncer, gt_3d, depth_map, ref):
        """Compute diagnostic metrics."""
        b, j = coarse_depth.shape[:2]
        # Handle potential extra dimensions in gt_3d (e.g., [B, 1, 17, 3] from batch_output)
        if gt_3d.dim() == 4:
            gt_depth = gt_3d[:, 0, :, -1:]  # [B, 17, 1]
        else:
            gt_depth = gt_3d[..., -1:]  # [B, 17, 1]
        
        metrics = {}
        
        # ==========================================
        # 1. Affine-Invariant Analysis
        # ==========================================
        # Fit per-image affine transform: D_pred = alpha * D_gt + beta
        # If depth estimator were perfect metric, alpha~1, beta~0
        
        alphas, betas, r2s = [], [], []
        for i in range(b):
            cd = coarse_depth[i].cpu().numpy().flatten()  # [17]
            gd = gt_depth[i].cpu().numpy().flatten()       # [17]
            
            # Robust linear fit
            A = np.vstack([gd, np.ones_like(gd)]).T
            alpha_beta, residuals, _, _ = np.linalg.lstsq(A, cd, rcond=None)
            alpha, beta = alpha_beta[0], alpha_beta[1]
            
            # R^2
            ss_res = np.sum((cd - (alpha * gd + beta))**2)
            ss_tot = np.sum((cd - cd.mean())**2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            
            alphas.append(alpha)
            betas.append(beta)
            r2s.append(r2)
        
        metrics['alpha_mean'] = float(np.mean(alphas))
        metrics['alpha_std'] = float(np.std(alphas))
        metrics['beta_mean'] = float(np.mean(betas))
        metrics['beta_std'] = float(np.std(betas))
        metrics['affine_r2_mean'] = float(np.mean(r2s))
        
        # ==========================================
        # 2. Ordinal Depth vs Metric Depth Analysis
        # ==========================================
        # Compare how well ordinal constraints are satisfied vs metric accuracy
        
        h36m_bones = [
            (0,1),(1,2),(2,3),      # right leg
            (0,4),(4,5),(5,6),      # left leg  
            (0,7),(7,8),(8,9),(9,10), # spine + head
            (8,11),(11,12),(12,13),   # right arm
            (8,14),(14,15),(15,16),   # left arm
        ]
        
        # Metric depth difference on bones
        pred_bone_depth_diff = []
        gt_bone_depth_diff = []
        for i, j in h36m_bones:
            pred_bone_depth_diff.append((coarse_depth[:, i] - coarse_depth[:, j]).cpu().numpy())
            gt_bone_depth_diff.append((gt_depth[:, i] - gt_depth[:, j]).cpu().numpy())
        
        pred_bone_depth_diff = np.array(pred_bone_depth_diff)  # [16, B, 1]
        gt_bone_depth_diff = np.array(gt_bone_depth_diff)
        
        # Ordinal accuracy: sign agreement
        ordinal_acc = float(np.mean(
            (pred_bone_depth_diff * gt_bone_depth_diff) > 0
        ))
        metrics['ordinal_accuracy'] = ordinal_acc
        
        # Metric accuracy on bone depth differences
        metric_bone_error = float(np.mean(np.abs(pred_bone_depth_diff - gt_bone_depth_diff)))
        metrics['metric_bone_depth_error_mm'] = metric_bone_error
        
        # ==========================================
        # 3. Depth Map Gradient Analysis
        # ==========================================
        # Analyze depth gradient magnitude at joint locations
        
        grad_magnitudes = []
        depth_values = []
        
        for i in range(b):
            # Compute depth map gradients [B, H, W]
            dm = depth_map[i:i+1]  # [1, H, W]
            grad_y = F.conv2d(dm.unsqueeze(0), 
                              torch.tensor([[[[-1,0,1]]]], dtype=dm.dtype, device=dm.device),
                              padding=(0,1))
            grad_x = F.conv2d(dm.unsqueeze(0),
                              torch.tensor([[[[-1],[0],[1]]]], dtype=dm.dtype, device=dm.device),
                              padding=(1,0))
            grad_mag = torch.sqrt(grad_x**2 + grad_y**2).squeeze()  # [H, W]
            
            # Sample at joint locations (unnormalize ref from [-1,1] to [0, H-1], [0, W-1])
            h, w = dm.shape[-2:]
            ref_px = ((ref[i] + 1) / 2 * torch.tensor([w-1, h-1], device=ref.device)).long()
            ref_px = ref_px.clamp(0, min(h-1, w-1))
            
            for j_idx in range(j):
                py, px = ref_px[j_idx]
                grad_magnitudes.append(grad_mag[py, px].item())
                depth_values.append(dm[0, py, px].item())
        
        metrics['grad_mag_mean'] = float(np.mean(grad_magnitudes))
        metrics['grad_mag_std'] = float(np.std(grad_magnitudes))
        metrics['depth_at_joint_mean'] = float(np.mean(depth_values))
        
        # ==========================================
        # 4. Uncertainty Calibration Analysis
        # ==========================================
        # Check if predicted uncertainty correlates with actual depth error
        
        depth_errors = torch.abs(coarse_depth - gt_depth).cpu().numpy().flatten()
        uncertainties = torch.exp(uncer).cpu().numpy().flatten()  # sigma = exp(s/2) if s is log-variance
        
        # Pearson correlation
        corr = np.corrcoef(depth_errors, uncertainties)[0, 1]
        metrics['uncertainty_error_correlation'] = float(corr)
        
        # Reliability: high uncertainty should -> high error
        high_unc_mask = uncertainties > np.percentile(uncertainties, 75)
        low_unc_mask = uncertainties < np.percentile(uncertainties, 25)
        metrics['error_high_uncertainty'] = float(np.mean(depth_errors[high_unc_mask]))
        metrics['error_low_uncertainty'] = float(np.mean(depth_errors[low_unc_mask]))
        
        # ==========================================
        # 5. AMS Sampling Statistics (if available)
        # ==========================================
        # We can't easily get AMS offsets without forward hook modifications,
        # but we can analyze the depth feature variance across joints
        
        # Variance of depth predictions across joints - should reflect scene complexity
        joint_depth_variance = float(torch.var(coarse_depth, dim=1).mean().item())
        metrics['joint_depth_variance'] = joint_depth_variance
        
        return metrics


def collate_metrics(all_metrics):
    """Aggregate metrics across batches."""
    keys = all_metrics[0].keys()
    aggregated = {}
    for k in keys:
        values = [m[k] for m in all_metrics]
        aggregated[k] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
        }
    return aggregated


def plot_results(aggregated, save_dir='./depth_analysis'):
    """Visualize diagnostic results."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Affine parameters distribution
    ax = axes[0, 0]
    ax.bar(['alpha (scale)', 'beta (shift)'], 
           [aggregated['alpha_mean']['mean'], aggregated['beta_mean']['mean']],
           yerr=[aggregated['alpha_mean']['std'], aggregated['beta_mean']['std']])
    ax.set_title('Per-Image Affine Transform\n(coarse_depth = alpha * gt + beta)')
    ax.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='ideal alpha=1')
    ax.axhline(0.0, color='g', linestyle='--', alpha=0.5, label='ideal beta=0')
    ax.legend()
    
    # 2. Ordinal vs Metric
    ax = axes[0, 1]
    ax.bar(['Ordinal Acc\n(sign match)', 'Affine R2'],
           [aggregated['ordinal_accuracy']['mean'], aggregated['affine_r2_mean']['mean']])
    ax.set_ylim([0, 1])
    ax.set_title('Ordinal vs Metric Depth Fidelity')
    
    # 3. Uncertainty calibration
    ax = axes[0, 2]
    ax.bar(['Error (High Unc)', 'Error (Low Unc)'],
           [aggregated['error_high_uncertainty']['mean'], aggregated['error_low_uncertainty']['mean']])
    ax.set_title('Uncertainty Calibration\n(higher error at higher uncertainty = good)')
    
    # 4. Depth gradient at joints
    ax = axes[1, 0]
    ax.bar(['Mean Grad Mag'], [aggregated['grad_mag_mean']['mean']],
           yerr=[aggregated['grad_mag_mean']['std']])
    ax.set_title('Depth Gradient Magnitude\nat Joint Locations')
    
    # 5. Uncertainty-Error Correlation
    ax = axes[1, 1]
    corr = aggregated['uncertainty_error_correlation']['mean']
    ax.bar(['Pearson r'], [corr], color='green' if corr > 0.3 else 'orange' if corr > 0 else 'red')
    ax.set_ylim([-1, 1])
    ax.axhline(0.0, color='k', linestyle='--', alpha=0.3)
    ax.set_title(f'Uncertainty-Error Correlation\n(r={corr:.3f})')
    
    # 6. Text summary
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = f"""
    KEY FINDINGS:
    
    1. Affine Scale (alpha): {aggregated['alpha_mean']['mean']:.3f} ± {aggregated['alpha_mean']['std']:.3f}
       - If far from 1.0, UDE compensates for metric scale mismatch
    
    2. Ordinal Accuracy: {aggregated['ordinal_accuracy']['mean']:.1%}
       - How often predicted depth ordering matches GT
    
    3. Metric Bone Error: {aggregated['metric_bone_depth_error_mm']['mean']:.1f} mm
       - Average depth diff error on connected joints
    
    4. Uncertainty Calib: r={aggregated['uncertainty_error_correlation']['mean']:.3f}
       - Ideally > 0.3 (positive correlation)
    
    5. Grad at Joints: {aggregated['grad_mag_mean']['mean']:.3f} ± {aggregated['grad_mag_mean']['std']:.3f}
       - High = joints on depth edges (good for AMS)
       - Low  = joints in flat regions (ambiguous depth)
    """
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'depth_analysis_summary.png'), dpi=150)
    print(f"Saved plot to {save_dir}/depth_analysis_summary.png")


def main():
    parser = argparse.ArgumentParser(description='ASFnet Depth Branch Diagnostic')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (optional, will init random if not provided)')
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='./depth_analysis')
    args = parser.parse_args()
    
    update_config(args.config)
    device = torch.device(args.device)
    
    # Build model
    print("Building model...")
    model = DepthGuidedPose(config, device).to(device)
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        if 'model' in ckpt:
            ckpt = ckpt['model']
        # Remove module prefix if present
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt, strict=False)
    else:
        print("No checkpoint provided, using random weights (architecture test only)")
    
    model.eval()
    
    # Setup dataloader
    print("Loading dataset...")
    val_dataset = eval('datasets.' + config.dataset.val_dataset)(
        root=config.dataset.root,
        pred_results_path=config.val.pred_results_path,
        depth_image_path=config.dataset.depth_image_path,
        train=False,
        test=True,
        image_shape=config.model.image_shape,
        labels_path=config.dataset.val_labels_path,
        with_damaged_actions=config.val.with_damaged_actions,
        retain_every_n_frames_in_test=100,  # subsample for faster analysis
        scale_bbox=config.val.scale_bbox,
        kind=config.kind,
        undistort_images=config.val.undistort_images,
        data_format=config.dataset.data_format,
        frame=1
    )
    
    # Limit samples for fast analysis
    indices = np.random.choice(len(val_dataset), min(args.num_samples, len(val_dataset)), replace=False)
    subset = torch.utils.data.Subset(val_dataset, indices)
    
    # We need a custom collate because original dataset returns complex tuples
    dataloader = DataLoader(subset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4,
                           pin_memory=True)
    
    # Create analyzer
    analyzer = DepthAnalyzer(model, device)
    
    print("Running analysis...")
    all_metrics = []
    
    # The dataloader returns batches. We need to handle the specific format.
    # Original format: images, keypoints_3d, keypoints_2d_cpn, keypoints_2d_cpn_crop, depth_images
    for batch_idx, batch in enumerate(dataloader):
        if len(batch) == 5:
            images, keypoints_3d_gt, keypoints_2d, keypoints_2d_crop, depth_images = batch
        else:
            print(f"Unexpected batch format with {len(batch)} elements, skipping")
            continue
            
        images = images.float().to(device) / 255.0  # uint8 -> float [0,1]
        keypoints_3d_gt = keypoints_3d_gt.float().to(device)
        keypoints_2d = keypoints_2d.float().to(device)
        keypoints_2d_crop = keypoints_2d_crop.float().to(device)
        depth_images = depth_images.float().to(device)
        
        # ref is keypoints_2d_crop normalized to [-1, 1]
        ref = keypoints_2d_crop[..., :2].clone()
        ref[..., 0] = ref[..., 0] / (config.model.image_shape[0] / 2) - 1
        ref[..., 1] = ref[..., 1] / (config.model.image_shape[1] / 2) - 1
        
        metrics = analyzer.analyze_batch(
            images, keypoints_2d, ref, depth_images, keypoints_3d_gt
        )
        all_metrics.append(metrics)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {(batch_idx + 1) * args.batch_size} samples...")
    
    analyzer.remove_hooks()
    
    # Aggregate and report
    print("\n" + "="*60)
    print("DEPTH BRANCH DIAGNOSTIC RESULTS")
    print("="*60)
    
    aggregated = collate_metrics(all_metrics)
    
    for k, v in aggregated.items():
        print(f"\n{k}:")
        print(f"  mean: {v['mean']:.4f} | std: {v['std']:.4f} | range: [{v['min']:.4f}, {v['max']:.4f}]")
    
    # Plot
    plot_results(aggregated, save_dir=args.output_dir)
    
    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    alpha_mean = aggregated['alpha_mean']['mean']
    alpha_std = aggregated['alpha_std']['mean']
    ordinal_acc = aggregated['ordinal_accuracy']['mean']
    unc_corr = aggregated['uncertainty_error_correlation']['mean']
    
    if abs(alpha_mean - 1.0) > 0.2 or alpha_std > 0.3:
        print("⚠️  AFFINE SCALE ISSUE DETECTED:")
        print(f"   alpha = {alpha_mean:.3f} ± {alpha_std:.3f}")
        print("   UDE is learning a per-image scale factor to compensate for")
        print("   the mismatch between relative depth input and metric depth target.")
        print("   → Consider explicit affine calibration or ordinal supervision.")
    else:
        print("✅ Affine scale is close to 1.0, metric depth regression is well-calibrated.")
    
    if ordinal_acc > 0.85:
        print(f"✅ High ordinal accuracy ({ordinal_acc:.1%}): depth ordering is reliable.")
    else:
        print(f"⚠️  Low ordinal accuracy ({ordinal_acc:.1%}): depth ordering is often wrong.")
    
    if unc_corr < 0.1:
        print("⚠️  UNCERTAINTY MISCALLIBRATED:")
        print(f"   Uncertainty-Error correlation = {unc_corr:.3f} (ideally > 0.3)")
        print("   Predicted uncertainty does not reflect actual depth errors.")
    elif unc_corr > 0.3:
        print(f"✅ Uncertainty is well-calibrated (r={unc_corr:.3f}).")
    
    print("="*60)


if __name__ == '__main__':
    main()
