"""
Verify coarse_depth output distribution.
"""
import os, sys, pickle
import numpy as np
import torch
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

indices = np.random.choice(len(val_dataset), 50, replace=False)
subset = Subset(val_dataset, indices)
dataloader = DataLoader(subset, batch_size=50, shuffle=False, num_workers=0)

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
    gt_depth = gt_3d[:, :, :, -1].squeeze(1)  # [B, 17]
    
    keypoints_2d = keypoints_2d.float().to(device)
    keypoints_2d_crop = keypoints_2d_crop.float().to(device)
    
    with torch.no_grad():
        pred_3d, coarse_depth, uncer = model(images, keypoints_2d, keypoints_2d_crop.clone(), depth_images)
    
    print("coarse_depth shape:", coarse_depth.shape)
    print("coarse_depth all joints mean:", coarse_depth.mean().item())
    print("coarse_depth all joints std:", coarse_depth.std().item())
    print("coarse_depth all joints min/max:", coarse_depth.min().item(), coarse_depth.max().item())
    print("coarse_depth root (joint 0) values (first 10):", coarse_depth[:10, 0, 0].cpu().numpy().flatten())
    print("coarse_depth root std:", coarse_depth[:, 0, 0].std().item())
    print("GT root depth (first 10):", gt_depth[:10, 0].cpu().numpy())
    print("GT root depth std:", gt_depth[:, 0].std().item())
    
    root_pred = coarse_depth[:, 0, 0].cpu().numpy()
    root_gt = gt_depth[:, 0].cpu().numpy()
    if np.std(root_pred) > 1e-8 and np.std(root_gt) > 1e-8:
        corr = np.corrcoef(root_pred, root_gt)[0, 1]
        print(f"Root depth correlation: {corr:.4f}")
    else:
        print("Zero std -> undefined correlation")
        print(f"root_pred std: {np.std(root_pred):.6f}")
        print(f"root_gt std: {np.std(root_gt):.6f}")
    break
