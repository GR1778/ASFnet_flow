import torch
from torch import nn

from mvn.models import pose_hrnet
from mvn.models.DGLifting_posealign import DGLifting


class DepthGuidedPosePoseAlign(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.num_joints = config.model.backbone.num_joints

        if config.model.backbone.type in ['hrnet_32', 'hrnet_48']:
            self.backbone = pose_hrnet.get_pose_net(config.model.backbone)

        if config.model.backbone.fix_weights:
            print("model backbone weights are fixed")
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.Lifting_net = DGLifting(config.model.poseformer)

    def forward(self, images, keypoints_2d_cpn, keypoints_2d_cpn_crop, depth_images):
        device = keypoints_2d_cpn.device
        images = images.permute(0, 3, 1, 2).contiguous()

        keypoints_2d_cpn_crop[..., :2] /= torch.tensor([192 // 2, 256 // 2], device=device)
        keypoints_2d_cpn_crop[..., :2] -= torch.tensor([1, 1], device=device)

        features_list_hr = self.backbone(images)
        return self.Lifting_net(keypoints_2d_cpn, keypoints_2d_cpn_crop, depth_images, features_list_hr)
