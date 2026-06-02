"""
Debug MPJPE: check per-sample errors and 2D input quality.
"""
import os, sys, pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, '.')
from mvn.models.DGPose import DepthGuidedPose
from mvn.utils.cfg import config, update_config
from mvn import datasets

config_path = 'experiments/human36m/human36m_single.yaml'
checkpoint_path = 'checkpoint/h36m_v2b.bin'
device = torch.device('cuda:0')

update_config(config_path)
model = DepthGuidedPose(config, device).to(device)
ckpt = torch.load(checkpoint_path, map_location=device)
if 'model' in ckpt:
    ckpt = ckpt['model']
ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
model.load_state_dict(ckpt, strict=False)
model.eval()

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

indices = np.random.choice(len(val_dataset), 5, replace=False)
subset = Subset(val_dataset, indices)
dataloader = DataLoader(subset, batch_size=5, shuffle=False, num_workers=0)

mean = torch.tensor([0.485, 0.456, 0.406]).cuda().to(device)
std = torch.tensor([0.229, 0.224, 0.225]).cuda().to(device)

for batch in dataloader:
    images, gt_3d, keypoints_2d, keypoints_2d_crop, depth_images = batch
    
    images = images.float().to(device)
    depth_images = depth_images.float().to(device) / 255.0
    images = torch.flip(images, [-1])
    images = (images / 255.0 - mean) / std
    
    gt_3d = gt_3d.float().to(device)
    gt_3d[:, :, 1:] -= gt_3d[:, :, :1]
    gt_3d[:, :, 0] = 0
    gt_3d = gt_3d.squeeze(1)  # [B, 17, 3]
    
    keypoints_2d = keypoints_2d.float().to(device)
    keypoints_2d_crop = keypoints_2d_crop.float().to(device)
    
    with torch.no_grad():
        pred_3d, coarse_depth, uncer = model(images, keypoints_2d, keypoints_2d_crop.clone(), depth_images)
    
    pred_3d = pred_3d.squeeze(1)  # [B, 17, 3]
    pred_rel = pred_3d - pred_3d[:, 0:1, :]
    
    errors = torch.norm(pred_rel - gt_3d, dim=-1)  # [B, 17]
    mpjpe_per_sample = errors.mean(dim=1) * 1000
    
    for i in range(5):
        print(f"\n--- Sample {i} ---")
        print(f"MPJPE: {mpjpe_per_sample[i].item():.2f} mm")
        print(f"Pred range: [{pred_rel[i].min().item():.3f}, {pred_rel[i].max().item():.3f}]")
        print(f"GT range: [{gt_3d[i].min().item():.3f}, {gt_3d[i].max().item():.3f}]")
        print(f"Max error joint: {errors[i].argmax().item()} -> {errors[i].max().item()*1000:.2f} mm")
        print(f"2D crop range: [{keypoints_2d_crop[i].min().item():.1f}, {keypoints_2d_crop[i].max().item():.1f}]")
        print(f"Coarse depth range: [{coarse_depth[i].min().item():.3f}, {coarse_depth[i].max().item():.3f}]")
    
    print(f"\nBatch mean MPJPE: {mpjpe_per_sample.mean().item():.2f} mm")
    break
