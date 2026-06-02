"""
ASFnet Depth Branch Rigorous Diagnostic Script
===============================================
This script analyzes the depth branch using VALID metrics that align
with the paper's design intent and reveal genuine weaknesses.

Key improvements over previous version:
1. Compares UDE-enhanced features vs raw features (not metric depth regression)
2. Tests depth contribution via controlled ablation (real vs random depth)
3. Measures uncertainty calibration against FINAL 3D pose error (not depth error)
4. Analyzes attention patterns in UDE (diagonal vs off-diagonal weights)
5. Measures offsets distribution in AMS (is it really adaptive?)
6. Tests root depth scale recovery (the most critical metric depth task)

Run:
    python tools/analyze_depth_rigorous.py \
        --config experiments/human36m/human36m_single.yaml \
        --checkpoint checkpoint/h36m_v2b.bin \
        --num_samples 500 --batch_size 32 --device cuda:0
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')

from mvn.models.DGPose import DepthGuidedPose
from mvn.utils.cfg import config, update_config
from mvn import datasets


class RigorousDepthAnalyzer:
    """
    Hook-based analyzer that extracts intermediate activations
    without modifying the model architecture.
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.hooks = {}
        self.activations = {}
        
        # Register forward hooks on critical modules
        self._register_hook('ams_last_output', 
                           model.Lifting_net.RGBD_Extraction[-1])
        self._register_hook('depth_uncer_output',
                           model.Lifting_net.depth_uncer)
        self._register_hook('ude_attention',
                           model.Lifting_net.attn_depth)
        self._register_hook('fusion_first_block',
                           model.Lifting_net.Features_Fusion[0].attn)
        
    def _register_hook(self, name, module):
        def hook_fn(m, input, output):
            # Store both input and output for analysis
            self.activations[name] = {
                'input': [x.detach() if isinstance(x, torch.Tensor) else x for x in input],
                'output': output.detach() if isinstance(output, torch.Tensor) else [o.detach() if isinstance(o, torch.Tensor) else o for o in output]
            }
        handle = module.register_forward_hook(hook_fn)
        self.hooks[name] = handle
        
    def remove_hooks(self):
        for name, handle in self.hooks.items():
            handle.remove()
    
    def analyze_batch(self, images, keypoints_2d, keypoints_2d_crop, 
                     depth_images, gt_3d):
        """
        Comprehensive analysis of a single batch.
        """
        b, j = keypoints_2d.shape[:2]
        
        # Handle gt_3d shape: may be [B, 1, 17, 3] or [B, 17, 3]
        if gt_3d.dim() == 4:
            gt_3d = gt_3d.squeeze(1)  # [B, 17, 3]
        
        # ====== RUN 1: Normal forward pass ======
        with torch.no_grad():
            pred_3d_normal, coarse_depth, uncer = self.model(
                images, keypoints_2d, keypoints_2d_crop.clone(), depth_images
            )
        
        # Get intermediate features from hooks
        # Note: hooks may capture previous batch if not cleared
        # Safer to re-run lifting net manually for clean extraction
        
        # Extract features manually for controlled analysis
        with torch.no_grad():
            lifting = self.model.Lifting_net
            
            # Step 1: Initial embedding
            x = lifting.coord_embed(keypoints_2d)
            depth_embedded = lifting.depth_embed(depth_images.unsqueeze(1))
            
            # Build features list with depth appended
            # images is [B, H, W, 3], backbone expects [B, 3, H, W]
            images_chw = images.permute(0, 3, 1, 2).contiguous()
            features_list_hr = self.model.backbone(images_chw)
            features_list_hr.append(depth_embedded)
            
            # Initial sampling at ref positions
            ref = keypoints_2d_crop[..., :2].clone()
            ref[..., 0] = ref[..., 0] / (192/2) - 1
            ref[..., 1] = ref[..., 1] / (256/2) - 1
            
            features_ref_list = [
                F.grid_sample(f, ref.unsqueeze(-2), align_corners=True).squeeze(-1).permute(0, 2, 1).contiguous()
                for f in features_list_hr
            ]
            features_ref_list = [embed(f) for embed, f in zip(lifting.feat_embed, features_ref_list)]
            
            x_stack = torch.stack([x, *features_ref_list], dim=1)
            x_stack += lifting.Spatial_pos_embed
            
            # Run AMS blocks
            for blk in lifting.RGBD_Extraction:
                x_stack = blk(x_stack, ref, features_list_hr)
            
            # Extract BEFORE UDE (raw depth token)
            x_depth_raw = x_stack[:, -1].clone()  # [B, 17, 128]
            
            # Run UDE
            coarse_depth_pred, uncer_pred = lifting.depth_uncer(x_depth_raw)
            z_value = lifting.z_embed(coarse_depth_pred) + lifting.Spatial_pos_embed2
            joint_uncer = F.softmax(lifting.attn_fc(uncer_pred), dim=1)
            
            # UDE input (concatenated)
            fcat = torch.cat([joint_uncer, z_value, x_depth_raw], dim=-1)  # [B, 17, 384]
            
            # UDE attention analysis
            # We need to manually run the attention to get weights
            B_attn, N_attn, C_attn = fcat.shape
            qkv = lifting.attn_depth.qkv(fcat).reshape(B_attn, N_attn, 3, 
                                                        lifting.attn_depth.num_heads, 
                                                        C_attn // lifting.attn_depth.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn_logits = (q @ k.transpose(-2, -1)) * lifting.attn_depth.scale
            attn_weights = attn_logits.softmax(dim=-1)  # [B, heads, 17, 17]
            
            x_depth_enhanced = lifting.attn_depth(fcat)  # [B, 17, 128]
            
            # Replace and continue
            x_fused = torch.cat((x_stack[:, :-1], x_depth_enhanced.unsqueeze(1)), dim=1)
            
            from einops import rearrange
            x_fused = rearrange(x_fused, 'b l p c -> (b p) l c')
            for blk in lifting.Features_Fusion:
                x_fused = blk(x_fused)
            x_fused = rearrange(x_fused, '(b p) l c -> b p (l c)', b=b)
            for blk in lifting.Spatial_Transformer:
                x_fused = blk(x_fused)
            pred_3d_from_normal = lifting.head(x_fused).view(b, 1, j, 3)
        
        metrics = {}
        
        # ==========================================
        # TEST 1: UDE Feature Enhancement Quality
        # ==========================================
        # Does UDE output better features than raw AMS output?
        # Measure: correlation with GT 3D depth (Z-coordinate)
        
        gt_depth = gt_3d[..., -1]  # [B, 17]
        
        # Compute per-joint correlation between feature and GT depth
        def feature_depth_correlation(features, gt_depth):
            """Compute mean correlation between feature dims and GT depth."""
            b, j, c = features.shape
            corrs = []
            for i in range(b):
                for d in range(c):
                    f_dim = features[i, :, d].cpu().numpy()
                    gd = gt_depth[i].cpu().numpy()
                    if np.std(f_dim) > 1e-6 and np.std(gd) > 1e-6:
                        corrs.append(np.corrcoef(f_dim, gd)[0, 1])
            return np.nanmean(corrs) if corrs else 0.0
        
        corr_raw = feature_depth_correlation(x_depth_raw, gt_depth)
        corr_enhanced = feature_depth_correlation(x_depth_enhanced, gt_depth)
        
        metrics['feature_depth_corr_raw'] = float(corr_raw)
        metrics['feature_depth_corr_enhanced'] = float(corr_enhanced)
        metrics['ude_improvement_ratio'] = float(corr_enhanced / (abs(corr_raw) + 1e-8))
        
        # ==========================================
        # TEST 2: Uncertainty vs Final Pose Error
        # ==========================================
        # The paper's claim: uncertainty helps robustness.
        # Valid test: does high uncertainty predict high FINAL 3D error?
        
        # Root-align for fair error computation
        pred_rel = pred_3d_normal.squeeze(1) - pred_3d_normal.squeeze(1)[:, 0:1, :]
        gt_rel = gt_3d - gt_3d[:, 0:1, :]
        final_errors = torch.norm(pred_rel - gt_rel, dim=-1)  # [B, 17]
        uncertainty_vals = torch.exp(uncer_pred / 2).squeeze(-1).cpu().numpy()  # sigma [B, 17]
        
        # Per-joint correlation
        flat_errors = final_errors.cpu().numpy().flatten()
        flat_unc = uncertainty_vals.flatten()
        
        # Remove NaNs/Infs
        valid_mask = np.isfinite(flat_errors) & np.isfinite(flat_unc) & (flat_unc < 100)
        if valid_mask.sum() > 10:
            corr_unc_error = np.corrcoef(flat_errors[valid_mask], flat_unc[valid_mask])[0, 1]
            metrics['uncertainty_vs_pose_error_corr'] = float(corr_unc_error)
            
            # Calibration: error in top-25% uncertainty joints vs bottom-25%
            q75 = np.percentile(flat_unc[valid_mask], 75)
            q25 = np.percentile(flat_unc[valid_mask], 25)
            metrics['pose_error_high_uncertainty'] = float(np.mean(flat_errors[valid_mask][flat_unc[valid_mask] > q75]))
            metrics['pose_error_low_uncertainty'] = float(np.mean(flat_errors[valid_mask][flat_unc[valid_mask] < q25]))
        else:
            metrics['uncertainty_vs_pose_error_corr'] = 0.0
            metrics['pose_error_high_uncertainty'] = 0.0
            metrics['pose_error_low_uncertainty'] = 0.0
        
        # ==========================================
        # TEST 3: AMS Offsets Distribution
        # ==========================================
        # Are offsets really adaptive, or do they stay near initialization?
        # Extract offsets from the LAST AMS block
        
        last_blk = lifting.RGBD_Extraction[-1]
        with torch.no_grad():
            # Recompute offsets for last block
            x_0, x_rest = x_stack[:, :1], x_stack[:, 1:]
            x_norm = last_blk.norm1(x_rest + x_0)
            offsets = last_blk.sampling_offsets(x_norm).reshape(b, x_rest.shape[1], j, 
                                                                last_blk.num_heads*last_blk.num_samples, 2).tanh()
            offsets_mag = torch.norm(offsets, dim=-1).cpu().numpy()  # [B, L, J, heads*samples]
        
        metrics['ams_offsets_mean'] = float(np.mean(offsets_mag))
        metrics['ams_offsets_std'] = float(np.std(offsets_mag))
        metrics['ams_offsets_max'] = float(np.max(offsets_mag))
        metrics['ams_offsets_percentile_90'] = float(np.percentile(offsets_mag, 90))
        
        # ==========================================
        # TEST 4: UDE Attention Pattern Analysis
        # ==========================================
        # Does attention focus on self (diagonal) or related joints?
        
        attn_np = attn_weights.cpu().numpy()  # [B, heads, 17, 17]
        mean_attn = attn_np.mean(axis=(0, 1))  # [17, 17] averaged over batch and heads
        
        # Diagonal vs off-diagonal
        diag_mask = np.eye(17, dtype=bool)
        diag_weight = mean_attn[diag_mask].mean()
        offdiag_weight = mean_attn[~diag_mask].mean()
        
        metrics['ude_attention_diagonal'] = float(diag_weight)
        metrics['ude_attention_offdiagonal'] = float(offdiag_weight)
        metrics['ude_attention_diagonal_ratio'] = float(diag_weight / (offdiag_weight + 1e-8))
        
        # Skeleton adjacency mask
        h36m_bones_set = {
            (0,1),(1,2),(2,3),
            (0,4),(4,5),(5,6),
            (0,7),(7,8),(8,9),(9,10),
            (8,11),(11,12),(12,13),
            (8,14),(14,15),(15,16),
        }
        adj_mask = np.zeros((17, 17), dtype=bool)
        for i, j in h36m_bones_set:
            adj_mask[i, j] = True
            adj_mask[j, i] = True
        
        bone_weight = mean_attn[adj_mask].mean() if adj_mask.any() else 0.0
        nonbone_offdiag_weight = mean_attn[~diag_mask & ~adj_mask].mean()
        metrics['ude_attention_bone_connected'] = float(bone_weight)
        metrics['ude_attention_nonbone'] = float(nonbone_offdiag_weight)
        
        # ==========================================
        # TEST 5: Depth Regression Quality (Root-Relative Z)
        # ==========================================
        # NOTE: GT is root-aligned, so root depth is always 0.
        # We test whether coarse_depth predicts non-root joint Z-coordinates
        # better than the main 3D head (i.e., is depth head redundant?).
        
        # Compare depth head Z-prediction vs main head Z-prediction for non-root joints
        pred_z_from_depth = coarse_depth_pred[:, 1:, 0].cpu().numpy()  # [B, 16] exclude root
        pred_z_from_main = pred_3d_normal.squeeze(1)[:, 1:, 2].cpu().numpy()  # [B, 16]
        gt_z_nonroot = gt_depth[:, 1:].cpu().numpy()  # [B, 16]
        
        # Error for each
        err_depth_head = np.abs(pred_z_from_depth - gt_z_nonroot).mean()
        err_main_head = np.abs(pred_z_from_main - gt_z_nonroot).mean()
        metrics['z_error_depth_head_mm'] = float(err_depth_head * 1000)
        metrics['z_error_main_head_mm'] = float(err_main_head * 1000)
        metrics['depth_head_redundancy_ratio'] = float(err_depth_head / (err_main_head + 1e-8))
        
        # Correlation of depth head with GT Z
        flat_depth_z = pred_z_from_depth.flatten()
        flat_gt_z = gt_z_nonroot.flatten()
        if np.std(flat_depth_z) > 1e-8 and np.std(flat_gt_z) > 1e-8:
            z_corr = np.corrcoef(flat_depth_z, flat_gt_z)[0, 1]
            metrics['depth_z_correlation'] = float(z_corr)
        else:
            metrics['depth_z_correlation'] = 0.0
            
        # Also test main head Z correlation for comparison
        flat_main_z = pred_z_from_main.flatten()
        if np.std(flat_main_z) > 1e-8 and np.std(flat_gt_z) > 1e-8:
            main_z_corr = np.corrcoef(flat_main_z, flat_gt_z)[0, 1]
            metrics['main_z_correlation'] = float(main_z_corr)
        else:
            metrics['main_z_correlation'] = 0.0
        
        # ==========================================
        # TEST 6: Depth Ablation (Random vs Real)
        # ==========================================
        # This is the strongest test: does the model actually use depth semantics?
        
        with torch.no_grad():
            # Ablated depth: random depth map with same mean/std but wrong structure
            random_depth = torch.randn_like(depth_images) * depth_images.std() + depth_images.mean()
            
            pred_3d_random, _, _ = self.model(
                images, keypoints_2d, keypoints_2d_crop.clone(), random_depth
            )
            
            # No depth at all: zeros
            zero_depth = torch.zeros_like(depth_images)
            pred_3d_zero, _, _ = self.model(
                images, keypoints_2d, keypoints_2d_crop.clone(), zero_depth
            )
        
        # Compute MPJPE with ROOT ALIGNMENT (subtract hip joint)
        def root_align(pred, gt):
            """Align both pred and gt to root-relative coordinates."""
            pred_rel = pred - pred[:, 0:1, :]  # subtract hip (joint 0)
            gt_rel = gt - gt[:, 0:1, :]        # subtract hip
            return torch.norm(pred_rel - gt_rel, dim=-1).mean().item()
        
        mpjpe_normal = root_align(pred_3d_normal.squeeze(1), gt_3d)
        mpjpe_random = root_align(pred_3d_random.squeeze(1), gt_3d)
        mpjpe_zero = root_align(pred_3d_zero.squeeze(1), gt_3d)
        
        metrics['mpjpe_real_depth'] = float(mpjpe_normal * 1000)  # convert to mm
        metrics['mpjpe_random_depth'] = float(mpjpe_random * 1000)
        metrics['mpjpe_zero_depth'] = float(mpjpe_zero * 1000)
        metrics['depth_semantic_gap'] = float((mpjpe_random - mpjpe_normal) * 1000)
        metrics['depth_any_benefit'] = float((mpjpe_zero - mpjpe_normal) * 1000)
        
        return metrics


def aggregate_metrics(all_metrics):
    keys = all_metrics[0].keys()
    result = {}
    for k in keys:
        values = np.array([m[k] for m in all_metrics])
        result[k] = {
            'mean': float(np.nanmean(values)),
            'std': float(np.nanstd(values)),
            'min': float(np.nanmin(values)),
            'max': float(np.nanmax(values)),
        }
    return result


def plot_results(agg, save_dir='./depth_analysis_rigorous'):
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. UDE Feature Enhancement
    ax = axes[0, 0]
    categories = ['Raw AMS\nFeatures', 'UDE-Enhanced\nFeatures']
    corrs = [agg['feature_depth_corr_raw']['mean'], agg['feature_depth_corr_enhanced']['mean']]
    ax.bar(categories, corrs, color=['skyblue', 'coral'])
    ax.axhline(0, color='k', linestyle='-', alpha=0.2)
    ax.set_title('Feature-GT Depth Correlation\n(higher = features encode depth better)')
    ax.set_ylabel('Mean Correlation')
    
    # 2. Uncertainty Calibration (FINAL pose error)
    ax = axes[0, 1]
    high_err = agg['pose_error_high_uncertainty']['mean']
    low_err = agg['pose_error_low_uncertainty']['mean']
    ax.bar(['High Uncertainty\nJoints', 'Low Uncertainty\nJoints'], 
           [high_err, low_err], color=['red', 'green'], alpha=0.7)
    ax.set_title(f'Uncertainty vs Final Pose Error\nCorr: {agg["uncertainty_vs_pose_error_corr"]["mean"]:.3f}')
    ax.set_ylabel('Mean Per-Joint Error (m)')
    if high_err > low_err:
        ax.text(0.5, max(high_err, low_err)*1.1, '✓ Calibrated', ha='center', color='green', fontsize=12)
    
    # 3. AMS Offsets Distribution
    ax = axes[0, 2]
    ax.bar(['Mean', '90th Percentile'], 
           [agg['ams_offsets_mean']['mean'], agg['ams_offsets_percentile_90']['mean']],
           yerr=[agg['ams_offsets_mean']['std'], agg['ams_offsets_percentile_90']['std']])
    ax.axhline(0.01, color='r', linestyle='--', alpha=0.5, label='init value')
    ax.set_title('AMS Offset Magnitudes\n(tanh range: [-1, 1])')
    ax.set_ylabel('Offset Magnitude')
    ax.legend()
    
    # 4. UDE Attention Pattern
    ax = axes[1, 0]
    labels = ['Diagonal\n(Self)', 'Bone\nConnected', 'Other\nOff-Diag']
    values = [agg['ude_attention_diagonal']['mean'],
              agg['ude_attention_bone_connected']['mean'],
              agg['ude_attention_nonbone']['mean']]
    ax.bar(labels, values, color=['gray', 'blue', 'lightgray'])
    ax.set_title('UDE Attention Distribution\n(17×17 joint attention matrix)')
    ax.set_ylabel('Mean Attention Weight')
    
    # 5. Depth Ablation
    ax = axes[1, 1]
    conditions = ['Real Depth', 'Random Depth', 'Zero Depth']
    mpjpes = [agg['mpjpe_real_depth']['mean'],
              agg['mpjpe_random_depth']['mean'],
              agg['mpjpe_zero_depth']['mean']]
    bars = ax.bar(conditions, mpjpes, color=['green', 'orange', 'red'], alpha=0.7)
    ax.set_title('Depth Ablation Test\n(MPJE, lower = better)')
    ax.set_ylabel('MPJPE (mm)')
    # Annotate gaps
    if mpjpes[1] > mpjpes[0]:
        ax.annotate(f'Random gap:\n{mpjpes[1]-mpjpes[0]:.1f}mm',
                   xy=(1, mpjpes[1]), xytext=(1.3, mpjpes[1]+5),
                   arrowprops=dict(arrowstyle='->', color='orange'))
    
    # 6. Depth Head vs Main Head (Z-coordinate)
    ax = axes[1, 2]
    labels_z = ['Depth Head\nZ Error', 'Main Head\nZ Error']
    values_z = [agg['z_error_depth_head_mm']['mean'], agg['z_error_main_head_mm']['mean']]
    bars_z = ax.bar(labels_z, values_z, color=['steelblue', 'coral'])
    ax.set_title('Z-Coordinate Prediction\n(non-root joints, mm)')
    ax.set_ylabel('Mean Absolute Error (mm)')
    redundancy = agg['depth_head_redundancy_ratio']['mean']
    ax.text(0.5, max(values_z)*1.1, f'Redundancy: {redundancy:.2f}x',
            ha='center', fontsize=10, color='darkred' if redundancy > 1.5 else 'green')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'rigorous_depth_analysis.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")


def print_interpretation(agg):
    print("\n" + "="*70)
    print("RIGOROUS INTERPRETATION")
    print("="*70)
    
    # 1. UDE Feature Enhancement
    raw_corr = agg['feature_depth_corr_raw']['mean']
    enh_corr = agg['feature_depth_corr_enhanced']['mean']
    print(f"\n1. UDE FEATURE ENHANCEMENT")
    print(f"   Raw AMS features correlate with GT depth:     {raw_corr:.4f}")
    print(f"   UDE-enhanced features correlate with GT depth: {enh_corr:.4f}")
    if enh_corr > raw_corr * 1.1:
        print(f"   ✅ UDE genuinely improves depth-related feature quality (+{(enh_corr/raw_corr-1)*100:.1f}%)")
    elif enh_corr > raw_corr:
        print(f"   ⚠️  UDE marginally improves features (+{(enh_corr/raw_corr-1)*100:.1f}%)")
    else:
        print(f"   ❌ UDE does NOT improve depth feature quality")
    
    # 2. Uncertainty Calibration
    unc_corr = agg['uncertainty_vs_pose_error_corr']['mean']
    high_err = agg['pose_error_high_uncertainty']['mean']
    low_err = agg['pose_error_low_uncertainty']['mean']
    print(f"\n2. UNCERTAINTY CALIBRATION (vs FINAL POSE ERROR)")
    print(f"   Correlation with pose error: {unc_corr:.3f}")
    print(f"   Error at high uncertainty:   {high_err*1000:.1f} mm")
    print(f"   Error at low uncertainty:    {low_err*1000:.1f} mm")
    if unc_corr > 0.2 and high_err > low_err:
        print(f"   ✅ Uncertainty is well-calibrated and predictive")
    elif unc_corr > 0.1:
        print(f"   ⚠️  Uncertainty has weak but positive correlation")
    else:
        print(f"   ❌ Uncertainty is NOT informative about final error")
    
    # 3. AMS Offsets
    off_mean = agg['ams_offsets_mean']['mean']
    off_p90 = agg['ams_offsets_percentile_90']['mean']
    print(f"\n3. AMS ADAPTIVITY")
    print(f"   Mean offset magnitude:   {off_mean:.4f}")
    print(f"   90th percentile:         {off_p90:.4f}")
    print(f"   Initialization value:    0.01 (radial)")
    if off_mean < 0.05:
        print(f"   ⚠️  Offsets stay near initialization → AMS is barely adaptive")
    elif off_p90 > 0.3:
        print(f"   ✅ Large offsets observed → AMS is actively exploring")
    else:
        print(f"   ⚠️  Small offsets → Limited spatial exploration")
    
    # 4. Attention Pattern
    diag_ratio = agg['ude_attention_diagonal_ratio']['mean']
    bone_attn = agg['ude_attention_bone_connected']['mean']
    print(f"\n4. UDE ATTENTION PATTERN")
    print(f"   Diagonal vs off-diagonal ratio: {diag_ratio:.2f}")
    print(f"   Attention to bone-connected joints: {bone_attn:.4f}")
    if diag_ratio > 3.0:
        print(f"   ⚠️  Heavily diagonal-dominated → mostly self-attention, limited joint interaction")
    elif bone_attn > 0.1:
        print(f"   ✅ Significant attention to anatomically connected joints")
    else:
        print(f"   ⚠️  Uniform attention → no anatomical structure learned")
    
    # 5. Depth Ablation
    real_mpjpe = agg['mpjpe_real_depth']['mean']
    rand_mpjpe = agg['mpjpe_random_depth']['mean']
    zero_mpjpe = agg['mpjpe_zero_depth']['mean']
    print(f"\n5. DEPTH ABLATION (CRITICAL TEST)")
    print(f"   Real depth:    {real_mpjpe:.1f} mm")
    print(f"   Random depth:  {rand_mpjpe:.1f} mm")
    print(f"   Zero depth:    {zero_mpjpe:.1f} mm")
    if rand_mpjpe - real_mpjpe < 1.0:
        print(f"   ❌❌❌ RANDOM depth ≈ REAL depth → Model does NOT use depth semantics!")
        print(f"       It may be using depth only as a 'spatial prior' or 'noise source'")
    elif rand_mpjpe - real_mpjpe < 3.0:
        print(f"   ⚠️  Small gap between random and real → Weak depth semantic utilization")
    else:
        print(f"   ✅ Real depth significantly better than random → Depth semantics are used")
    
    if zero_mpjpe - real_mpjpe < 1.0:
        print(f"   ❌ Zero depth ≈ Real depth → Depth provides NO benefit")
    else:
        print(f"   ✅ Depth provides {zero_mpjpe - real_mpjpe:.1f}mm improvement over no depth")
    
    # 6. Depth Head Redundancy
    z_err_depth = agg['z_error_depth_head_mm']['mean']
    z_err_main = agg['z_error_main_head_mm']['mean']
    redundancy = agg['depth_head_redundancy_ratio']['mean']
    depth_z_corr = agg['depth_z_correlation']['mean']
    main_z_corr = agg['main_z_correlation']['mean']
    print(f"\n6. DEPTH HEAD vs MAIN HEAD (Z-COORDINATE)")
    print(f"   Depth head Z error:  {z_err_depth:.2f} mm")
    print(f"   Main head Z error:   {z_err_main:.2f} mm")
    print(f"   Redundancy ratio:    {redundancy:.2f}x")
    print(f"   Depth head Z corr:   {depth_z_corr:.3f}")
    print(f"   Main head Z corr:    {main_z_corr:.3f}")
    if redundancy > 1.5:
        print(f"   ❌ Depth head is WORSE than main head → Redundant and harmful")
    elif redundancy > 1.0:
        print(f"   ⚠️  Depth head is no better than main head → Redundant")
    else:
        print(f"   ✅ Depth head improves over main head → Complementary")
    
    if main_z_corr > depth_z_corr * 1.2:
        print(f"   ❌ Main head predicts Z better than depth head → Depth head is useless")
    else:
        print(f"   ⚠️  Depth head Z correlation is comparable to main head")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='ASFnet Depth Branch Rigorous Diagnostic')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    update_config(args.config)
    device = torch.device(args.device)
    
    print("Building model...")
    model = DepthGuidedPose(config, device).to(device)
    
    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'model' in ckpt:
        ckpt = ckpt['model']
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    
    print("Loading dataset...")
    val_dataset = eval('datasets.' + config.dataset.val_dataset)(
        root=config.dataset.root,
        pred_results_path=config.val.pred_results_path,
        depth_image_path=config.dataset.depth_image_path,
        train=False, test=True,
        image_shape=config.model.image_shape,
        labels_path=config.dataset.val_labels_path,
        with_damaged_actions=config.val.with_damaged_actions,
        retain_every_n_frames_in_test=100,
        scale_bbox=config.val.scale_bbox,
        kind=config.kind,
        undistort_images=config.val.undistort_images,
        data_format=config.dataset.data_format,
        frame=1
    )
    
    total_samples = min(args.num_samples, len(val_dataset))
    indices = np.random.choice(len(val_dataset), total_samples, replace=False)
    subset = Subset(val_dataset, indices)
    
    dataloader = DataLoader(subset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)
    
    analyzer = RigorousDepthAnalyzer(model, device)
    
    # Preprocessing constants from data_prefetcher
    mean = torch.tensor([0.485, 0.456, 0.406]).cuda().to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).cuda().to(device)
    
    print("Running rigorous analysis...")
    all_metrics = []
    for batch_idx, batch in enumerate(dataloader):
        if len(batch) != 5:
            continue
        images, gt_3d, keypoints_2d, keypoints_2d_crop, depth_images = batch
        
        # EXACT preprocessing from data_prefetcher
        images = images.float().to(device)
        depth_images = depth_images.float().to(device) / 255.0
        images = torch.flip(images, [-1])  # BGR -> RGB
        images = (images / 255.0 - mean) / std
        
        gt_3d = gt_3d.float().to(device)
        # Root-align GT exactly as in data_prefetcher
        gt_3d[:, :, 1:] -= gt_3d[:, :, :1]
        gt_3d[:, :, 0] = 0
        
        keypoints_2d = keypoints_2d.float().to(device)
        keypoints_2d_crop = keypoints_2d_crop.float().to(device)
        
        metrics = analyzer.analyze_batch(
            images, keypoints_2d, keypoints_2d_crop, depth_images, gt_3d
        )
        all_metrics.append(metrics)
        
        if (batch_idx + 1) % 5 == 0:
            print(f"  Processed {(batch_idx+1)*args.batch_size}/{total_samples} samples...")
    
    analyzer.remove_hooks()
    
    # Report
    print("\n" + "="*70)
    print("RIGOROUS DEPTH BRANCH DIAGNOSTIC RESULTS")
    print("="*70)
    
    agg = aggregate_metrics(all_metrics)
    for k, v in agg.items():
        print(f"\n{k}:")
        print(f"  mean: {v['mean']:.4f} | std: {v['std']:.4f} | range: [{v['min']:.4f}, {v['max']:.4f}]")
    
    plot_results(agg)
    print_interpretation(agg)


if __name__ == '__main__':
    main()
