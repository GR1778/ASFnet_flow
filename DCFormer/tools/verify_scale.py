"""
Quick verification: replicate official eval preprocessing exactly.
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
from mvn.datasets.human36m import joints_left, joints_right

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

# Take 10 random samples
indices = np.random.choice(len(val_dataset), 10, replace=False)
subset = Subset(val_dataset, indices)
dataloader = DataLoader(subset, batch_size=10, shuffle=False, num_workers=0)

mean = torch.tensor([0.485, 0.456, 0.406]).cuda().to(device)
std = torch.tensor([0.229, 0.224, 0.225]).cuda().to(device)

for batch in dataloader:
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
    
    with torch.no_grad():
        pred_3d, coarse_depth, uncer = model(images, keypoints_2d, keypoints_2d_crop.clone(), depth_images)
    
    pred_3d = pred_3d.squeeze(1)  # [B, 17, 3]
    
    print("Pred range:", pred_3d.min().item(), pred_3d.max().item())
    print("GT range:", gt_3d.min().item(), gt_3d.max().item())
    
    # Root-align pred
    pred_rel = pred_3d - pred_3d[:, 0:1, :]
    gt_rel = gt_3d
    
    errors = torch.norm(pred_rel - gt_rel, dim=-1)  # [B, 17]
    mpjpe = errors.mean().item() * 1000  # mm
    
    print(f"MPJPE (mm): {mpjpe:.2f}")
    print(f"Sample pred[0]: {pred_3d[0, :3].cpu().numpy()}")
    print(f"Sample gt[0]: {gt_3d[0, :3].cpu().numpy()}")
    break
