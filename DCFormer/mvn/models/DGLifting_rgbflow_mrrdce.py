from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mvn.models.DGLifting_rgbflow_capf import Block, DeformableBlock


class MotionReferenceRefinement(nn.Module):
    def __init__(self, dim, hidden_dim=64, offset_scale=0.25):
        super().__init__()
        self.offset_scale = offset_scale
        self.refine = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, motion_field, ref):
        offset_field = self.refine(motion_field).tanh()
        offset = F.grid_sample(offset_field, ref.unsqueeze(-2), padding_mode="border", align_corners=True)
        offset = offset.squeeze(-1).permute(0, 2, 1).contiguous()
        return (ref + self.offset_scale * offset).clamp(-1.0, 1.0)


class RGBFlowMRRDCELifting(nn.Module):
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
        flow_refine_scale = getattr(config, "flow_refine_scale", 0.25)
        out_dim = 3

        if backbone in ["hrnet_32", "hrnet_48"]:
            feature_dim_list = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        elif backbone == "cpn":
            feature_dim_list = [base_dim] * 4
        else:
            raise ValueError("Unsupported backbone for RGBFlowMRRDCELifting: {}".format(backbone))

        self.rgb_levels = len(feature_dim_list)
        self.coord_embed = nn.Linear(in_chans, embed_dim_ratio)
        self.feat_embed = nn.ModuleList([nn.Linear(dim_in, embed_dim_ratio) for dim_in in feature_dim_list])

        self.motion_field_embed = nn.Conv2d(in_channels=2, out_channels=embed_dim_ratio, kernel_size=3, padding=1)
        self.motion_refine = MotionReferenceRefinement(embed_dim_ratio, offset_scale=flow_refine_scale)
        self.flow_feat_embed = nn.Linear(embed_dim_ratio, embed_dim_ratio)

        self.RGB_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.rgb_levels, num_joints, embed_dim_ratio))
        self.Flow_pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints, embed_dim_ratio))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.RGB_Extraction = nn.ModuleList(
            [
                DeformableBlock(
                    feature_dim_list=feature_dim_list,
                    dim=embed_dim_ratio,
                    num_heads=4,
                    num_samples=4,
                    qkv_bias=qkv_bias,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.Motion_Extraction = nn.ModuleList(
            [
                DeformableBlock(
                    feature_dim_list=[embed_dim_ratio],
                    dim=embed_dim_ratio,
                    num_heads=flow_num_heads,
                    num_samples=flow_num_samples,
                    qkv_bias=qkv_bias,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )

        embed_dim = embed_dim_ratio * (2 + self.rgb_levels)
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

        features_ref_list = [
            F.grid_sample(features, ref.unsqueeze(-2), align_corners=True).squeeze(-1).permute(0, 2, 1).contiguous()
            for features in features_list_hr
        ]
        features_ref_list = [embed(features_ref_list[idx]) for idx, embed in enumerate(self.feat_embed)]

        x_rgb = torch.stack([pose_token, *features_ref_list], dim=1)
        x_rgb = self.pos_drop(x_rgb + self.RGB_pos_embed)
        for block in self.RGB_Extraction:
            x_rgb = block(x_rgb, ref, features_list_hr)

        motion_field = self.motion_field_embed(flow_images.permute(0, 3, 1, 2).contiguous())
        motion_ref = self.motion_refine(motion_field, ref)
        flow_token = F.grid_sample(motion_field, motion_ref.unsqueeze(-2), align_corners=True)
        flow_token = flow_token.squeeze(-1).permute(0, 2, 1).contiguous()
        flow_token = self.flow_feat_embed(flow_token)
        flow_token = self.pos_drop(flow_token.unsqueeze(1) + self.Flow_pos_embed)

        x_flow = torch.cat([pose_token.unsqueeze(1), flow_token], dim=1)
        for block in self.Motion_Extraction:
            x_flow = block(x_flow, motion_ref, [motion_field])
        flow_token = x_flow[:, 1:]

        x = torch.cat([x_rgb, flow_token], dim=1)
        x = rearrange(x, "b l p c -> (b p) l c")
        for block in self.Features_Fusion:
            x = block(x)

        x = rearrange(x, "(b p) l c -> b p (l c)", b=b)
        for block in self.Spatial_Transformer:
            x = block(x)

        x = self.head(x).view(b, 1, p, -1)
        return x
