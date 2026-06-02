"""
ASFnet Depth Branch Diagnostic - Quick Test with Synthetic Data
===============================================================
This version uses synthetic data to verify the analysis logic
without requiring the full H36M dataset.

Run:
    python tools/analyze_depth_synthetic.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')

from mvn.models.DGLifting import DGLifting


class SyntheticDepthAnalyzer:
    """Analyze depth branch using synthetic data that mimics H36M characteristics."""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        
    def generate_synthetic_batch(self, batch_size=16):
        """
        Generate synthetic data that mimics real depth characteristics:
        - Depth map: relative depth (affine transformed from metric)
        - 2D poses: noisy projections
        - 3D GT: root-relative metric poses
        """
        b, j = batch_size, 17
        h, w = 256, 192
        
        # Generate plausible 3D poses (roughly human-scale)
        # Root at origin, other joints within [-1000, 1000] mm
        gt_3d = torch.randn(b, j, 3) * 400
        gt_3d[:, 0] = 0  # root at origin
        
        # Per-image affine parameters (simulating Depth-Anything behavior)
        # alpha ~ Uniform[0.5, 2.0], beta ~ Uniform[-500, 500]
        alphas = torch.rand(b, 1, 1) * 1.5 + 0.5  # [0.5, 2.0]
        betas = torch.rand(b, 1, 1) * 1000 - 500   # [-500, 500]
        
        # Relative depth map = alpha * metric_depth + beta + noise
        metric_depth_values = gt_3d[..., 2:3]  # [B, 17, 1]
        
        # Create a full depth map (not just joints)
        depth_map = torch.zeros(b, h, w)
        for i in range(b):
            # Simple scene: person in center, background further away
            y_coords = torch.linspace(-1, 1, h)
            x_coords = torch.linspace(-1, 1, w)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            # Base depth varies by position (perspective-like)
            base_depth = 2000 + 500 * yy + 300 * xx  # [H, W]
            
            # Add per-image affine transformation
            depth_map[i] = alphas[i, 0, 0] * base_depth + betas[i, 0, 0]
            
            # Add noise
            depth_map[i] += torch.randn_like(depth_map[i]) * 50
        
        # 2D poses (normalized to [-1, 1])
        keypoints_2d = torch.randn(b, j, 2) * 0.5  # [-0.5, 0.5]
        ref = keypoints_2d.clone()
        
        # Features list (HRNet-like multi-scale)
        features_list = [
            torch.randn(b, 32, 64, 48),
            torch.randn(b, 64, 32, 24),
            torch.randn(b, 128, 16, 12),
            torch.randn(b, 256, 8, 6),
        ]
        
        # RGB images (not used in Lifting_net but needed for interface)
        images = torch.randn(b, 256, 192, 3)
        
        return {
            'images': images,
            'keypoints_2d': keypoints_2d,
            'ref': ref,
            'depth_map': depth_map,
            'gt_3d': gt_3d,
            'true_alphas': alphas,
            'true_betas': betas,
        }
    
    def analyze_affine_transform(self, coarse_depth, gt_depth, true_alpha, true_beta):
        """
        Verify if coarse_depth = alpha * gt_depth + beta holds,
        and compare estimated alpha/beta to ground truth.
        """
        b = coarse_depth.shape[0]
        results = []
        
        for i in range(b):
            cd = coarse_depth[i].cpu().numpy().flatten()
            gd = gt_depth[i].cpu().numpy().flatten()
            
            # Linear fit
            A = np.vstack([gd, np.ones_like(gd)]).T
            est_alpha_beta, _, _, _ = np.linalg.lstsq(A, cd, rcond=None)
            est_alpha, est_beta = est_alpha_beta[0], est_alpha_beta[1]
            
            true_a = true_alpha[i, 0, 0].item()
            true_b = true_beta[i, 0, 0].item()
            
            results.append({
                'est_alpha': est_alpha,
                'est_beta': est_beta,
                'true_alpha': true_a,
                'true_beta': true_b,
                'alpha_error': abs(est_alpha - true_a),
                'beta_error': abs(est_beta - true_b),
                'r2': 1 - np.sum((cd - (est_alpha * gd + est_beta))**2) / (np.sum((cd - cd.mean())**2) + 1e-8)
            })
        
        return results
    
    def analyze_ordinal_depth(self, coarse_depth, gt_depth):
        """Check depth ordering preservation on human skeleton."""
        bones = [
            (0,1),(1,2),(2,3),
            (0,4),(4,5),(5,6),
            (0,7),(7,8),(8,9),(9,10),
            (8,11),(11,12),(12,13),
            (8,14),(14,15),(15,16),
        ]
        
        b = coarse_depth.shape[0]
        correct = 0
        total = 0
        
        for i in range(b):
            for j1, j2 in bones:
                pred_diff = (coarse_depth[i, j1] - coarse_depth[i, j2]).item()
                gt_diff = (gt_depth[i, j1] - gt_depth[i, j2]).item()
                if pred_diff * gt_diff > 0:  # same sign
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0
    
    def analyze_uncertainty_calibration(self, coarse_depth, uncer, gt_depth):
        """Check if uncertainty correlates with actual errors."""
        errors = torch.abs(coarse_depth - gt_depth).cpu().numpy().flatten()
        # uncer is log-variance, convert to std
        sigmas = torch.exp(uncer / 2).cpu().numpy().flatten()
        
        corr = np.corrcoef(errors, sigmas)[0, 1]
        
        # Bin by uncertainty quartiles
        q75 = np.percentile(sigmas, 75)
        q25 = np.percentile(sigmas, 25)
        
        high_unc_error = np.mean(errors[sigmas > q75])
        low_unc_error = np.mean(errors[sigmas < q25])
        
        return {
            'correlation': float(corr),
            'error_high_unc': float(high_unc_error),
            'error_low_unc': float(low_unc_error),
        }
    
    def analyze_depth_gradient(self, depth_map, ref, num_joints=17):
        """Analyze depth gradient magnitude at joint locations."""
        b, h, w = depth_map.shape
        grad_mags = []
        
        for i in range(b):
            dm = depth_map[i:i+1].unsqueeze(0)  # [1, 1, H, W]
            
            # Sobel filters
            sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], 
                                    dtype=dm.dtype, device=dm.device)
            sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], 
                                    dtype=dm.dtype, device=dm.device)
            
            gx = F.conv2d(dm, sobel_x, padding=1).squeeze()
            gy = F.conv2d(dm, sobel_y, padding=1).squeeze()
            grad_mag = torch.sqrt(gx**2 + gy**2)
            
            # Sample at joint locations
            ref_px = ((ref[i] + 1) / 2 * torch.tensor([w-1, h-1], device=ref.device)).long()
            ref_px = ref_px.clamp(0, min(h-1, w-1))
            
            for j in range(num_joints):
                px, py = ref_px[j]
                grad_mags.append(grad_mag[py, px].item())
        
        return {
            'mean': float(np.mean(grad_mags)),
            'std': float(np.std(grad_mags)),
            'median': float(np.median(grad_mags)),
            'percentile_90': float(np.percentile(grad_mags, 90)),
        }
    
    def run_full_analysis(self, num_batches=10, batch_size=16):
        """Run complete diagnostic."""
        print("="*70)
        print("ASFnet Depth Branch - Synthetic Diagnostic")
        print("="*70)
        
        all_affine = []
        all_ordinal = []
        all_unc_calib = []
        all_grad = []
        
        for batch_idx in range(num_batches):
            data = self.generate_synthetic_batch(batch_size)
            
            # Forward pass (DGLifting expects: keypoints_2d, ref, depth_images, features_list_hr)
            # Build features_list_hr from random HRNet-like features
            features_list_hr = [
                torch.randn(batch_size, 32, 64, 48),
                torch.randn(batch_size, 64, 32, 24),
                torch.randn(batch_size, 128, 16, 12),
                torch.randn(batch_size, 256, 8, 6),
            ]
            with torch.no_grad():
                pred_3d, coarse_depth, uncer = self.model(
                    data['keypoints_2d'], 
                    data['ref'], 
                    data['depth_map'],
                    features_list_hr
                )
            
            gt_depth = data['gt_3d'][..., 2:3]  # [B, 17, 1]
            
            # 1. Affine analysis
            affine_res = self.analyze_affine_transform(
                coarse_depth, gt_depth, data['true_alphas'], data['true_betas']
            )
            all_affine.extend(affine_res)
            
            # 2. Ordinal analysis
            ordinal_acc = self.analyze_ordinal_depth(coarse_depth, gt_depth)
            all_ordinal.append(ordinal_acc)
            
            # 3. Uncertainty calibration
            unc_calib = self.analyze_uncertainty_calibration(coarse_depth, uncer, gt_depth)
            all_unc_calib.append(unc_calib)
            
            # 4. Depth gradient
            grad_stats = self.analyze_depth_gradient(data['depth_map'], data['ref'])
            all_grad.append(grad_stats)
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Processed {batch_idx + 1}/{num_batches} batches...")
        
        # Aggregate results
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        
        # Affine
        alpha_errors = [r['alpha_error'] for r in all_affine]
        beta_errors = [r['beta_error'] for r in all_affine]
        r2s = [r['r2'] for r in all_affine]
        
        print(f"\n1. AFFINE TRANSFORM RECOVERY")
        print(f"   Alpha estimation error: {np.mean(alpha_errors):.3f} ± {np.std(alpha_errors):.3f}")
        print(f"   Beta estimation error:  {np.mean(beta_errors):.1f} ± {np.std(beta_errors):.1f} mm")
        print(f"   Fit R²:                 {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")
        print(f"   → If R² < 0.5, UDE fails to establish consistent affine mapping")
        
        # Ordinal
        print(f"\n2. ORDINAL DEPTH ACCURACY")
        print(f"   Bone depth ordering accuracy: {np.mean(all_ordinal):.1%}")
        print(f"   → If < 75%, depth branch fails basic geometric consistency")
        
        # Uncertainty
        corrs = [u['correlation'] for u in all_unc_calib]
        print(f"\n3. UNCERTAINTY CALIBRATION")
        print(f"   Uncertainty-Error correlation: {np.mean(corrs):.3f}")
        print(f"   Error at high uncertainty:     {np.mean([u['error_high_unc'] for u in all_unc_calib]):.1f} mm")
        print(f"   Error at low uncertainty:      {np.mean([u['error_low_unc'] for u in all_unc_calib]):.1f} mm")
        if np.mean(corrs) < 0.1:
            print("   ⚠️  WARNING: Uncertainty does not reflect actual errors!")
        elif np.mean(corrs) > 0.3:
            print("   ✅ Uncertainty is informative")
        
        # Gradient
        print(f"\n4. DEPTH GRADIENT AT JOINTS")
        print(f"   Mean gradient magnitude: {np.mean([g['mean'] for g in all_grad]):.2f}")
        print(f"   Median:                  {np.mean([g['median'] for g in all_grad]):.2f}")
        print(f"   90th percentile:         {np.mean([g['percentile_90'] for g in all_grad]):.2f}")
        print(f"   → Low values suggest joints are in flat depth regions (limited geometric info)")
        
        # Plot
        self.plot_results(all_affine, all_ordinal, all_unc_calib, all_grad)
        
    def plot_results(self, affine, ordinal, unc_calib, grad):
        """Generate diagnostic plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Alpha/Beta recovery scatter
        ax = axes[0, 0]
        true_alphas = [r['true_alpha'] for r in affine]
        est_alphas = [r['est_alpha'] for r in affine]
        ax.scatter(true_alphas, est_alphas, alpha=0.5, s=10)
        ax.plot([0, 3], [0, 3], 'r--', label='Perfect recovery')
        ax.set_xlabel('True Alpha (scale)')
        ax.set_ylabel('Estimated Alpha')
        ax.set_title('Affine Scale Recovery\n(UDE implicit calibration)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Ordinal accuracy distribution
        ax = axes[0, 1]
        ax.hist(ordinal, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(ordinal), color='r', linestyle='--', 
                  label=f'Mean: {np.mean(ordinal):.1%}')
        ax.set_xlabel('Ordinal Accuracy')
        ax.set_ylabel('Count')
        ax.set_title('Depth Ordering Accuracy Distribution')
        ax.legend()
        
        # 3. Uncertainty vs Error
        ax = axes[1, 0]
        corrs = [u['correlation'] for u in unc_calib]
        ax.hist(corrs, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(0.3, color='g', linestyle='--', alpha=0.5, label='Good (r=0.3)')
        ax.axvline(-0.3, color='r', linestyle='--', alpha=0.5, label='Bad (r=-0.3)')
        ax.set_xlabel('Uncertainty-Error Correlation')
        ax.set_ylabel('Count')
        ax.set_title('Uncertainty Calibration')
        ax.legend()
        
        # 4. Gradient distribution
        ax = axes[1, 1]
        means = [g['mean'] for g in grad]
        ax.hist(means, bins=20, edgecolor='black', alpha=0.7, label='Mean')
        p90s = [g['percentile_90'] for g in grad]
        ax.hist(p90s, bins=20, edgecolor='black', alpha=0.4, label='90th percentile')
        ax.set_xlabel('Depth Gradient Magnitude')
        ax.set_ylabel('Count')
        ax.set_title('Depth Gradient at Joint Locations')
        ax.legend()
        
        plt.tight_layout()
        save_path = 'depth_diagnostic_synthetic.png'
        plt.savefig(save_path, dpi=150)
        print(f"\n📊 Plot saved to: {save_path}")


def main():
    print("Initializing model...")
    model = DGLifting()
    model.eval()
    
    analyzer = SyntheticDepthAnalyzer(model)
    analyzer.run_full_analysis(num_batches=20, batch_size=16)
    
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("""
This synthetic test reveals how the depth branch handles affine-transformed 
(relative) depth inputs:

KEY METRICS TO WATCH:

1. Affine R²: 
   - If < 0.5: UDE cannot establish stable metric mapping from relative depth
   - If > 0.8: UDE successfully learns implicit per-image calibration

2. Ordinal Accuracy:
   - If < 75%: Even depth ordering (who is in front) is unreliable
   - This is the strongest signal for depth utilization quality

3. Uncertainty Correlation:
   - If < 0.1: Uncertainty is not informative (random)
   - Ideal: > 0.3 (positive correlation between predicted uncertainty and error)

4. Gradient Magnitude:
   - Low values (< 10): Joints land in flat depth regions
   - High values (> 50): Joints on depth edges (rich geometric info)
   
On real H36M data, run the full script:
    python tools/analyze_depth_utilization.py --config <config> --checkpoint <ckpt>
""")


if __name__ == '__main__':
    main()
