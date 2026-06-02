from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath
from torch.nn.init import constant_

from mvn.models.DGLifting_rgbflow_capf import Block, DeformableBlock, Mlp


class MotionFieldAdaptiveSamplingBlock(nn.Module):
    """Local, sample-aware adaptive sampling for encoded optical-flow fields."""

    def __init__(
        self,
        dim,
        num_heads,
        num_samples,
        image_width=192,
        image_height=256,
        local_radius_px=12.0,
        residual_radius_px=4.0,
        drop_path=0.0,
        mlp_ratio=2,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.num_samples = num_samples
        self.head_dim = dim // num_heads
        self.norm1 = norm_layer(dim)
        self.offset_residual = nn.Linear(dim, 2 * num_heads * num_samples)
        self.attention_logits = nn.Linear(dim, num_heads * num_samples)
        self.query_proj = nn.Linear(dim, dim)
        self.embed_proj = nn.Linear(dim, self.head_dim)

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=0.0)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        base_offsets = self._build_base_offsets(
            num_heads=num_heads,
            num_samples=num_samples,
            image_width=image_width,
            image_height=image_height,
            local_radius_px=local_radius_px,
        )
        residual_scale = torch.tensor(
            [
                2.0 * residual_radius_px / float(image_width - 1),
                2.0 * residual_radius_px / float(image_height - 1),
            ],
            dtype=torch.float32,
        )
        self.register_buffer("base_offsets", base_offsets, persistent=False)
        self.register_buffer("residual_scale", residual_scale, persistent=False)
        self._reset_parameters()

    @staticmethod
    def _build_base_offsets(num_heads, num_samples, image_width, image_height, local_radius_px):
        thetas = torch.arange(num_heads, dtype=torch.float32) * (2.0 * math.pi / num_heads)
        directions = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        if num_samples == 1:
            radii = torch.zeros(1, dtype=torch.float32)
        else:
            radii = torch.linspace(0.0, float(local_radius_px), steps=num_samples)

        offsets_px = directions[:, None, :] * radii[None, :, None]
        offsets = offsets_px.clone()
        offsets[..., 0] = offsets[..., 0] * (2.0 / float(image_width - 1))
        offsets[..., 1] = offsets[..., 1] * (2.0 / float(image_height - 1))
        return offsets.reshape(1, 1, 1, num_heads * num_samples, 2)

    def _reset_parameters(self):
        constant_(self.offset_residual.weight.data, 0.0)
        constant_(self.offset_residual.bias.data, 0.0)
        constant_(self.attention_logits.weight.data, 0.0)
        constant_(self.attention_logits.bias.data, 0.0)

    def forward(self, x, ref, motion_field):
        x_0, x_motion = x[:, :1], x[:, 1:]
        b, l, p, c = x_motion.shape
        if l != 1:
            raise ValueError("MotionFieldAdaptiveSamplingBlock expects exactly one motion level.")

        residual = x_motion
        query = self.norm1(x_motion + x_0)

        offsets = self.offset_residual(query).reshape(b, l, p, self.num_heads * self.num_samples, 2)
        offsets = offsets.tanh() * self.residual_scale.view(1, 1, 1, 1, 2)
        pos = ref.view(b, 1, p, 1, 2) + self.base_offsets + offsets
        pos = pos.clamp(-1.0, 1.0)

        sampled = F.grid_sample(motion_field, pos[:, 0], padding_mode="border", align_corners=True)
        sampled = sampled.permute(0, 2, 3, 1).contiguous()
        sampled = self.embed_proj(sampled)
        sampled = sampled.view(b, p, self.num_heads, self.num_samples, self.head_dim)

        query_heads = self.query_proj(query[:, 0]).view(b, p, self.num_heads, self.head_dim)
        content_logits = (sampled * query_heads.unsqueeze(-2)).sum(dim=-1) * (self.head_dim ** -0.5)
        learned_logits = self.attention_logits(query).view(b, l, p, self.num_heads, self.num_samples)[:, 0]
        weights = (learned_logits + content_logits).softmax(dim=-1)

        sampled = (weights.unsqueeze(-1) * sampled).sum(dim=-2).view(b, 1, p, c)
        x_motion = residual + self.drop_path(sampled)
        x_motion = x_motion + self.drop_path(self.mlp(self.norm2(x_motion)))
        return torch.cat([x_0, x_motion], dim=1)


class FlowMFAS(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads=4,
        num_samples=5,
        image_width=192,
        image_height=256,
        local_radius_px=12.0,
        residual_radius_px=4.0,
        drop_path=None,
    ):
        super().__init__()
        drop_path = drop_path or [0.0] * depth
        self.blocks = nn.ModuleList(
            [
                MotionFieldAdaptiveSamplingBlock(
                    dim=dim,
                    num_heads=num_heads,
                    num_samples=num_samples,
                    image_width=image_width,
                    image_height=image_height,
                    local_radius_px=local_radius_px,
                    residual_radius_px=residual_radius_px,
                    drop_path=drop_path[i],
                )
                for i in range(depth)
            ]
        )

    def forward(self, pose_token, flow_token, ref, motion_field):
        x = torch.stack([pose_token, flow_token], dim=1)
        for block in self.blocks:
            x = block(x, ref, motion_field)
        return x[:, 1]


class RGBFlowMFASLifting(nn.Module):
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
        flow_local_radius_px = getattr(config, "flow_local_radius_px", 12.0)
        flow_residual_radius_px = getattr(config, "flow_residual_radius_px", 4.0)
        out_dim = 3

        if backbone in ["hrnet_32", "hrnet_48"]:
            feature_dim_list = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        elif backbone == "cpn":
            feature_dim_list = [base_dim] * 4
        else:
            raise ValueError("Unsupported backbone for RGBFlowMFASLifting: {}".format(backbone))

        self.rgb_levels = len(feature_dim_list)
        self.coord_embed = nn.Linear(in_chans, embed_dim_ratio)
        self.feat_embed = nn.ModuleList([nn.Linear(dim_in, embed_dim_ratio) for dim_in in feature_dim_list])

        self.motion_field_embed = nn.Conv2d(in_channels=2, out_channels=embed_dim_ratio, kernel_size=3, padding=1)
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
        self.Flow_Extraction = FlowMFAS(
            dim=embed_dim_ratio,
            depth=depth,
            num_heads=flow_num_heads,
            num_samples=flow_num_samples,
            image_width=192,
            image_height=256,
            local_radius_px=flow_local_radius_px,
            residual_radius_px=flow_residual_radius_px,
            drop_path=dpr,
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

        x = torch.stack([pose_token, *features_ref_list], dim=1)
        x = self.pos_drop(x + self.RGB_pos_embed)
        for block in self.RGB_Extraction:
            x = block(x, ref, features_list_hr)

        motion_field = self.motion_field_embed(flow_images.permute(0, 3, 1, 2).contiguous())
        flow_token = F.grid_sample(motion_field, ref.unsqueeze(-2), align_corners=True).squeeze(-1).permute(0, 2, 1).contiguous()
        flow_token = self.flow_feat_embed(flow_token)
        flow_token = self.Flow_Extraction(
            pose_token + self.RGB_pos_embed[:, 0],
            flow_token + self.Flow_pos_embed[:, 0],
            ref,
            motion_field,
        )
        flow_token = self.pos_drop(flow_token.unsqueeze(1))

        x = torch.cat([x, flow_token], dim=1)
        x = rearrange(x, "b l p c -> (b p) l c")
        for block in self.Features_Fusion:
            x = block(x)

        x = rearrange(x, "(b p) l c -> b p (l c)", b=b)
        for block in self.Spatial_Transformer:
            x = block(x)

        x = self.head(x).view(b, 1, p, -1)
        return x
