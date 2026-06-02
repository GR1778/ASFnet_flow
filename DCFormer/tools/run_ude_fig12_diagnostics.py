import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mvn.models.DGPose import DepthGuidedPose
from mvn.utils.cfg import config, update_config
from mvn import datasets

JOINT_NAMES = [
    'Pelvis','R_Hip','R_Knee','R_Ankle','L_Hip','L_Knee','L_Ankle',
    'Torso','Neck','Nose','Head','L_Shoulder','L_Elbow','L_Wrist','R_Shoulder','R_Elbow','R_Wrist'
]


def safe_corr(x, y, fn):
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return float('nan')
    x = x[m]
    y = y[m]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float('nan')
    r, _ = fn(x, y)
    return float(r)


def build_model(device):
    model = DepthGuidedPose(config, device)
    model.to(device)
    model.eval()
    return model


def load_ckpt(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model' in ckpt:
        ckpt = ckpt['model']
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)


def make_loader(num_samples, batch_size, seed):
    val_dataset = eval('datasets.' + config.dataset.val_dataset)(
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
    rng = np.random.default_rng(seed)
    n = min(num_samples, len(val_dataset))
    idx = rng.choice(len(val_dataset), n, replace=False)
    subset = Subset(val_dataset, idx)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True), n


@torch.no_grad()
def collect(model, loader, device):
    all_abs_err = []
    all_s = []
    all_mu = []
    all_gtz = []
    all_joint_uncer = []

    for batch in loader:
        if len(batch) != 5:
            continue
        images, keypoints_3d_gt, keypoints_2d, keypoints_2d_crop, depth_images = batch
        images = images.float().to(device) / 255.0
        keypoints_3d_gt = keypoints_3d_gt.float().to(device)
        keypoints_2d = keypoints_2d.float().to(device)
        keypoints_2d_crop = keypoints_2d_crop.float().to(device)
        depth_images = depth_images.float().to(device)

        # DGPose.forward normalizes crop coordinates internally. Passing a
        # pre-normalized ref here would normalize the coordinates twice and
        # corrupt the sampled depth token used by UDE.
        _, coarse_depth, uncer = model(images, keypoints_2d, keypoints_2d_crop.clone(), depth_images)

        # recompute joint_uncer exactly as UDE path
        lifting = model.Lifting_net
        joint_uncer = F.softmax(lifting.attn_fc(uncer), dim=1).mean(dim=-1)  # [B, J]

        if keypoints_3d_gt.dim() == 4:
            gt_z = keypoints_3d_gt[:, 0, :, 2:3]
        else:
            gt_z = keypoints_3d_gt[..., 2:3]

        abs_err = torch.abs(coarse_depth - gt_z)

        all_abs_err.append(abs_err.squeeze(-1).cpu().numpy())
        all_s.append(uncer.squeeze(-1).cpu().numpy())
        all_mu.append(coarse_depth.squeeze(-1).cpu().numpy())
        all_gtz.append(gt_z.squeeze(-1).cpu().numpy())
        all_joint_uncer.append(joint_uncer.cpu().numpy())

    return {
        'abs_err': np.concatenate(all_abs_err, axis=0),
        's': np.concatenate(all_s, axis=0),
        'mu': np.concatenate(all_mu, axis=0),
        'gt_z': np.concatenate(all_gtz, axis=0),
        'joint_uncer': np.concatenate(all_joint_uncer, axis=0),
    }


def analyze(data):
    abs_err = data['abs_err']
    s = data['s']
    joint_uncer = data['joint_uncer']

    flat_err = abs_err.reshape(-1)
    flat_s = s.reshape(-1)
    flat_neg_s = -flat_s
    flat_var = np.exp(np.clip(flat_s, -50.0, 50.0))
    flat_precision = np.exp(np.clip(-flat_s, -50.0, 50.0))

    fig12_like = {
        'individual_joint_pearson_r': safe_corr(flat_s, flat_err, pearsonr),
        'individual_joint_spearman_r': safe_corr(flat_s, flat_err, spearmanr),
    }

    per_joint_mean_err = abs_err.mean(axis=0)
    per_joint_mean_s = s.mean(axis=0)
    fig12_like['joint_type_mean_pearson_r'] = safe_corr(per_joint_mean_s, per_joint_mean_err, pearsonr)
    fig12_like['joint_type_mean_spearman_r'] = safe_corr(per_joint_mean_s, per_joint_mean_err, spearmanr)

    per_joint_mean_neg_s = -per_joint_mean_s
    per_joint_mean_var = np.exp(np.clip(per_joint_mean_s, -50.0, 50.0))
    per_joint_mean_precision = np.exp(np.clip(-per_joint_mean_s, -50.0, 50.0))
    correlation_variants = {
        's_as_log_variance': {
            'individual_pearson_r': safe_corr(flat_s, flat_err, pearsonr),
            'individual_spearman_r': safe_corr(flat_s, flat_err, spearmanr),
            'joint_type_mean_pearson_r': safe_corr(per_joint_mean_s, per_joint_mean_err, pearsonr),
            'joint_type_mean_spearman_r': safe_corr(per_joint_mean_s, per_joint_mean_err, spearmanr),
        },
        'neg_s_as_confidence': {
            'individual_pearson_r': safe_corr(flat_neg_s, flat_err, pearsonr),
            'individual_spearman_r': safe_corr(flat_neg_s, flat_err, spearmanr),
            'joint_type_mean_pearson_r': safe_corr(per_joint_mean_neg_s, per_joint_mean_err, pearsonr),
            'joint_type_mean_spearman_r': safe_corr(per_joint_mean_neg_s, per_joint_mean_err, spearmanr),
        },
        'exp_s_as_variance_clipped': {
            'clip_range': [-50.0, 50.0],
            'individual_pearson_r': safe_corr(flat_var, flat_err, pearsonr),
            'individual_spearman_r': safe_corr(flat_var, flat_err, spearmanr),
            'joint_type_mean_pearson_r': safe_corr(per_joint_mean_var, per_joint_mean_err, pearsonr),
            'joint_type_mean_spearman_r': safe_corr(per_joint_mean_var, per_joint_mean_err, spearmanr),
        },
        'exp_neg_s_as_precision_clipped': {
            'clip_range': [-50.0, 50.0],
            'individual_pearson_r': safe_corr(flat_precision, flat_err, pearsonr),
            'individual_spearman_r': safe_corr(flat_precision, flat_err, spearmanr),
            'joint_type_mean_pearson_r': safe_corr(per_joint_mean_precision, per_joint_mean_err, pearsonr),
            'joint_type_mean_spearman_r': safe_corr(per_joint_mean_precision, per_joint_mean_err, spearmanr),
        },
    }

    q_err75 = np.nanpercentile(flat_err, 75)
    q_s25 = np.nanpercentile(flat_s, 25)
    high_err_low_unc = (abs_err >= q_err75) & (s <= q_s25)
    he_lu_rate = float(high_err_low_unc.mean())
    he_lu_joint_rate = high_err_low_unc.mean(axis=0)

    # joint_uncer variability across samples
    joint_uncer_std_per_joint = joint_uncer.std(axis=0)
    joint_uncer_cv_per_joint = joint_uncer_std_per_joint / (joint_uncer.mean(axis=0) + 1e-8)

    return {
        'figure12_like': fig12_like,
        'correlation_variants': correlation_variants,
        's_statistics': {
            'mean': float(np.nanmean(flat_s)),
            'std': float(np.nanstd(flat_s)),
            'min': float(np.nanmin(flat_s)),
            'max': float(np.nanmax(flat_s)),
            'q05': float(np.nanpercentile(flat_s, 5)),
            'q25': float(np.nanpercentile(flat_s, 25)),
            'q50': float(np.nanpercentile(flat_s, 50)),
            'q75': float(np.nanpercentile(flat_s, 75)),
            'q95': float(np.nanpercentile(flat_s, 95)),
        },
        'high_error_low_uncertainty': {
            'global_rate': he_lu_rate,
            'threshold_error_q75': float(q_err75),
            'threshold_s_q25': float(q_s25),
            'per_joint_rate': {JOINT_NAMES[i]: float(he_lu_joint_rate[i]) for i in range(17)},
            'top5_joints': [
                {'joint': JOINT_NAMES[i], 'rate': float(he_lu_joint_rate[i])}
                for i in np.argsort(-he_lu_joint_rate)[:5]
            ],
        },
        'joint_uncer_distribution': {
            'mean_std_across_joints': float(joint_uncer_std_per_joint.mean()),
            'mean_cv_across_joints': float(joint_uncer_cv_per_joint.mean()),
            'per_joint_std': {JOINT_NAMES[i]: float(joint_uncer_std_per_joint[i]) for i in range(17)},
            'top5_most_variable': [
                {'joint': JOINT_NAMES[i], 'std': float(joint_uncer_std_per_joint[i])}
                for i in np.argsort(-joint_uncer_std_per_joint)[:5]
            ],
            'top5_least_variable': [
                {'joint': JOINT_NAMES[i], 'std': float(joint_uncer_std_per_joint[i])}
                for i in np.argsort(joint_uncer_std_per_joint)[:5]
            ],
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--num_samples', type=int, default=2000)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    update_config(args.config)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = build_model(device)
    load_ckpt(model, args.checkpoint, device)
    loader, used_n = make_loader(args.num_samples, args.batch_size, args.seed)
    data = collect(model, loader, device)
    report = analyze(data)
    report['meta'] = {
        'checkpoint': args.checkpoint,
        'num_samples_requested': args.num_samples,
        'num_samples_used': used_n,
        'batch_size': args.batch_size,
    }

    with open(os.path.join(args.output_dir, 'figure12_diagnostics.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report['figure12_like'], indent=2, ensure_ascii=False))
    print('Saved:', os.path.join(args.output_dir, 'figure12_diagnostics.json'))


if __name__ == '__main__':
    main()
