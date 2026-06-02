from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mvn.models.DGLifting_rgbflow_capf import Block, DeformableBlock


class MotionFieldContextExtraction(nn.Module):
    """DGLifting-style adaptive sampling over RGB features and an encoded motion field."""

    def __init__(
        self,
        feature_dim_list,
        dim,
        depth,
        num_heads=4,
        num_samples=5,
        qkv_bias=True,
        drop_path=None,
    ):
        super().__init__()
        drop_path = drop_path or [0.0] * depth
        self.blocks = nn.ModuleList(
            [
                DeformableBlock(
                    feature_dim_list=feature_dim_list,
                    dim=dim,
                    num_heads=num_heads,
                    num_samples=num_samples,
                    qkv_bias=qkv_bias,
                    drop_path=drop_path[i],
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, ref, features_list):
        for block in self.blocks:
            x = block(x, ref, features_list)
        return x


class RGBFlowMFCELifting(nn.Module):
    def __init__(
        self,
        config=None,
        backbone="hrnet_32",
        num_joints=17,
        in_chans=2,
        num_heads=8,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=None,
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        base_dim = getattr(config, "base_dim", 32)
        embed_dim_ratio = getattr(config, "embed_dim_ratio", 128)
        depth = getattr(config, "depth", getattr(config, "levels", 4))
        flow_num_heads = getattr(config, "flow_num_heads", 4)
        flow_num_samples = getattr(config, "flow_num_samples", 5)
        out_dim = 3

        if backbone in ["hrnet_32", "hrnet_48"]:
            feature_dim_list = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        elif backbone == "cpn":
            feature_dim_list = [base_dim] * 4
        else:
            raise ValueError("Unsupported backbone for RGBFlowMFCELifting: {}".format(backbone))

        self.levels = len(feature_dim_list) + 1
        self.coord_embed = nn.Linear(in_chans, embed_dim_ratio)
        feature_dim_list = [*feature_dim_list, embed_dim_ratio]
        self.feat_embed = nn.ModuleList([nn.Linear(dim_in, embed_dim_ratio) for dim_in in feature_dim_list])

        self.motion_field_embed = nn.Conv2d(in_channels=2, out_channels=embed_dim_ratio, kernel_size=3, padding=1)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.levels, num_joints, embed_dim_ratio))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.MFCE = MotionFieldContextExtraction(
            feature_dim_list=feature_dim_list,
            dim=embed_dim_ratio,
            depth=depth,
            num_heads=flow_num_heads,
            num_samples=flow_num_samples,
            qkv_bias=qkv_bias,
            drop_path=dpr,
        )

        embed_dim = embed_dim_ratio * (1 + self.levels)
        self.Features_Fusion = nn.ModuleList(
            [
                Block(
                    dim=embed_dim_ratio,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.Spatial_Transformer = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, out_dim))

    def forward(self, keypoints_2d, ref, flow_images, features_list_hr):
        b, p, _ = keypoints_2d.shape
        pose_token = self.coord_embed(keypoints_2d)
        motion_field = self.motion_field_embed(flow_images.permute(0, 3, 1, 2).contiguous())
        features_list = [*features_list_hr, motion_field]

        features_ref_list = [
            F.grid_sample(features, ref.unsqueeze(-2), align_corners=True).squeeze(-1).permute(0, 2, 1).contiguous()
            for features in features_list
        ]
        features_ref_list = [embed(features_ref_list[idx]) for idx, embed in enumerate(self.feat_embed)]

        x = torch.stack([pose_token, *features_ref_list], dim=1)
        x = self.pos_drop(x + self.Spatial_pos_embed)
        x = self.MFCE(x, ref, features_list)

        x = rearrange(x, "b l p c -> (b p) l c")
        for block in self.Features_Fusion:
            x = block(x)

        x = rearrange(x, "(b p) l c -> b p (l c)", b=b)
        for block in self.Spatial_Transformer:
            x = block(x)

        x = self.head(x).view(b, 1, p, -1)
        return x
