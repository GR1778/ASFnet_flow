import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.init import constant_

from mvn.models.DGLifting_rgbflow_capf import Block, DeformableBlock, Mlp


class JointContextProjector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim * 2)
        self.proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, rgb_tokens):
        pose_token = rgb_tokens[:, 0]
        rgb_context = rgb_tokens[:, 1:].mean(dim=1)
        return self.proj(self.norm(torch.cat([pose_token, rgb_context], dim=-1)))


class MotionCrossSampleBlock(nn.Module):
    def __init__(self, dim, num_heads=4, num_samples=5, qkv_bias=True, drop_path=0.0, mlp_ratio=2):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.dim = dim
        self.num_heads = num_heads
        self.num_samples = num_samples
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query_norm = nn.LayerNorm(dim)
        self.sample_norm = nn.LayerNorm(dim)
        self.out_norm = nn.LayerNorm(dim)

        self.sampling_offsets = nn.Linear(dim, 2 * num_heads * num_samples)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=0.0)

        from timm.models.layers import DropPath

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = 0.01 * (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
        grid_init = grid_init.view(self.num_heads, 1, 2).repeat(1, self.num_samples, 1)
        for i in range(self.num_samples):
            grid_init[:, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

    def forward(self, flow_token, joint_query, ref, motion_field):
        b, p, c = flow_token.shape
        q_input = self.query_norm(flow_token + joint_query)

        offsets = self.sampling_offsets(q_input).view(b, p, self.num_heads * self.num_samples, 2).tanh()
        pos = offsets + ref.view(b, p, 1, 2)

        samples = F.grid_sample(motion_field, pos, padding_mode="border", align_corners=True)
        samples = samples.permute(0, 2, 3, 1).contiguous()
        samples = self.sample_norm(samples)

        q = self.q_proj(q_input).view(b, p, self.num_heads, self.head_dim).unsqueeze(-2)
        k = self.k_proj(samples).view(b, p, self.num_heads * self.num_samples, self.num_heads, self.head_dim)
        v = self.v_proj(samples).view(b, p, self.num_heads * self.num_samples, self.num_heads, self.head_dim)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)

        attn = (q * k).sum(dim=-1) * self.scale
        attn = F.softmax(attn, dim=-1)
        sampled = (attn.unsqueeze(-1) * v).sum(dim=-2).reshape(b, p, c)

        flow_token = flow_token + self.drop_path(self.out_proj(sampled))
        flow_token = flow_token + self.drop_path(self.mlp(self.out_norm(flow_token)))
        return flow_token


class JointGuidedMotionSampling(nn.Module):
    def __init__(self, dim, depth, num_heads=4, num_samples=5, qkv_bias=True, drop_path=None):
        super().__init__()
        drop_path = drop_path or [0.0] * depth
        self.blocks = nn.ModuleList(
            [
                MotionCrossSampleBlock(
                    dim=dim,
                    num_heads=num_heads,
                    num_samples=num_samples,
                    qkv_bias=qkv_bias,
                    drop_path=drop_path[i],
                )
                for i in range(depth)
            ]
        )

    def forward(self, flow_token, joint_query, ref, motion_field):
        for block in self.blocks:
            flow_token = block(flow_token, joint_query, ref, motion_field)
        return flow_token


class RGBFlowJGMSLifting(nn.Module):
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
            raise ValueError("Unsupported backbone for RGBFlowJGMSLifting: {}".format(backbone))

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

        self.Motion_Query = JointContextProjector(embed_dim_ratio)
        self.Motion_Extraction = JointGuidedMotionSampling(
            dim=embed_dim_ratio,
            depth=depth,
            num_heads=flow_num_heads,
            num_samples=flow_num_samples,
            qkv_bias=qkv_bias,
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

        joint_query = self.Motion_Query(x)

        motion_field = self.motion_field_embed(flow_images.permute(0, 3, 1, 2).contiguous())
        flow_token = F.grid_sample(motion_field, ref.unsqueeze(-2), align_corners=True).squeeze(-1).permute(0, 2, 1).contiguous()
        flow_token = self.flow_feat_embed(flow_token)
        flow_token = self.pos_drop(flow_token.unsqueeze(1) + self.Flow_pos_embed).squeeze(1)
        flow_token = self.Motion_Extraction(flow_token, joint_query, ref, motion_field)

        x = torch.cat([x, flow_token.unsqueeze(1)], dim=1)
        x = rearrange(x, "b l p c -> (b p) l c")
        for block in self.Features_Fusion:
            x = block(x)

        x = rearrange(x, "(b p) l c -> b p (l c)", b=b)
        for block in self.Spatial_Transformer:
            x = block(x)

        x = self.head(x).view(b, 1, p, -1)
        return x
