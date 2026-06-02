import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath
from torch.nn.init import constant_

from mvn.models.DGLifting_rgbflow_capf import Block, DeformableBlock, Mlp


class MotionEvidenceSamplingBlock(nn.Module):
    """Evidence-conditioned deformable sampler for joint-level flow tokens.

    The block first observes a stable DCE-style local sampling basis, then uses
    the sampled motion evidence to condition the final offsets and weights.
    """

    def __init__(
        self,
        dim,
        num_heads=4,
        num_samples=5,
        qkv_bias=True,
        drop_path=0.0,
        mlp_ratio=2.0,
        offset_scale=0.04,
        joint_basis_scale=0.02,
        num_joints=17,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.dim = dim
        self.num_heads = num_heads
        self.num_samples = num_samples
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.offset_scale = float(offset_scale)
        self.joint_basis_scale = float(joint_basis_scale)
        self.num_joints = num_joints

        self.query_norm = norm_layer(dim)
        self.evidence_norm = norm_layer(dim)
        self.refine_norm = norm_layer(dim)
        self.out_norm = norm_layer(dim)

        self.evidence_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.evidence_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.evidence_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.evidence_proj = nn.Linear(dim, dim)

        self.offset_head = nn.Linear(dim, 2 * num_heads * num_samples)

        self.weight_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.weight_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.value_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=0.0)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.register_buffer("base_offsets", self._make_base_offsets(num_heads, num_samples), persistent=False)
        self.joint_basis_offsets = nn.Parameter(torch.zeros(1, num_joints, num_heads, num_samples, 2))
        self._reset_parameters()

    @staticmethod
    def _make_base_offsets(num_heads, num_samples):
        thetas = torch.arange(num_heads, dtype=torch.float32) * (2.0 * math.pi / num_heads)
        base = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        base = base / base.abs().max(dim=-1, keepdim=True)[0].clamp_min(1e-6)
        base = 0.01 * base.view(num_heads, 1, 2).repeat(1, num_samples, 1)
        if num_samples > 0:
            base[:, 0, :] = 0.0
        for idx in range(1, num_samples):
            base[:, idx, :] *= idx
        return base

    def _reset_parameters(self):
        constant_(self.offset_head.weight.data, 0.0)
        constant_(self.offset_head.bias.data, 0.0)

    def _sample_motion(self, motion_field, ref, offsets):
        b, p, _, _, _ = offsets.shape
        pos = ref.view(b, p, 1, 1, 2) + offsets
        grid = pos.reshape(b, p * self.num_heads * self.num_samples, 1, 2)
        samples = F.grid_sample(motion_field, grid, padding_mode="border", align_corners=True)
        samples = samples.squeeze(-1).permute(0, 2, 1).contiguous()
        return samples.view(b, p, self.num_heads, self.num_samples, self.dim), pos

    def _basis_offsets(self, b, p, dtype, device):
        base_offsets = self.base_offsets.to(device=device, dtype=dtype)
        base_offsets = base_offsets.view(1, 1, self.num_heads, self.num_samples, 2)
        if p != self.num_joints:
            joint_offsets = torch.zeros(1, p, self.num_heads, self.num_samples, 2, device=device, dtype=dtype)
        else:
            joint_offsets = self.joint_basis_scale * torch.tanh(self.joint_basis_offsets.to(device=device, dtype=dtype))
        return (base_offsets + joint_offsets).expand(b, p, -1, -1, -1)

    def forward(self, flow_token, joint_token, ref, motion_field):
        b, p, c = flow_token.shape
        basis_offsets = self._basis_offsets(b, p, flow_token.dtype, flow_token.device)

        query = self.query_norm(flow_token + joint_token)

        evidence0, _ = self._sample_motion(motion_field, ref, basis_offsets)
        evidence0 = self.evidence_norm(evidence0)

        q0 = self.evidence_q(query).view(b, p, self.num_heads, self.head_dim)
        k0 = self.evidence_k(evidence0).view(b, p, self.num_heads, self.num_samples, self.num_heads, self.head_dim)
        v0 = self.evidence_v(evidence0).view(b, p, self.num_heads, self.num_samples, self.num_heads, self.head_dim)

        head_idx = torch.arange(self.num_heads, device=flow_token.device)
        head_idx = head_idx.view(1, 1, self.num_heads, 1, 1, 1)
        head_idx = head_idx.expand(b, p, self.num_heads, self.num_samples, 1, self.head_dim)
        k0 = k0.gather(4, head_idx).squeeze(4)
        v0 = v0.gather(4, head_idx).squeeze(4)

        attn0 = (q0.unsqueeze(-2) * k0).sum(dim=-1) * self.scale
        attn0 = F.softmax(attn0, dim=-1)
        region = (attn0.unsqueeze(-1) * v0).sum(dim=-2).reshape(b, p, c)
        region = self.evidence_proj(region)

        refine_query = self.refine_norm(flow_token + joint_token + region)
        residual_offsets = self.offset_scale * torch.tanh(self.offset_head(refine_query))
        residual_offsets = residual_offsets.view(b, p, self.num_heads, self.num_samples, 2)
        final_offsets = basis_offsets + residual_offsets

        evidence1, _ = self._sample_motion(motion_field, ref, final_offsets)
        evidence1 = self.evidence_norm(evidence1)

        q1 = self.weight_q(refine_query).view(b, p, self.num_heads, self.head_dim)
        k1 = self.weight_k(evidence1).view(b, p, self.num_heads, self.num_samples, self.num_heads, self.head_dim)
        v1 = self.value_proj(evidence1).view(b, p, self.num_heads, self.num_samples, self.num_heads, self.head_dim)
        k1 = k1.gather(4, head_idx).squeeze(4)
        v1 = v1.gather(4, head_idx).squeeze(4)

        attn1 = (q1.unsqueeze(-2) * k1).sum(dim=-1) * self.scale
        attn1 = F.softmax(attn1, dim=-1)
        sampled = (attn1.unsqueeze(-1) * v1).sum(dim=-2).reshape(b, p, c)

        flow_token = flow_token + self.drop_path(self.out_proj(sampled))
        flow_token = flow_token + self.drop_path(self.mlp(self.out_norm(flow_token)))
        return flow_token


class MotionEvidenceSampling(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads=4,
        num_samples=5,
        qkv_bias=True,
        drop_path=None,
        offset_scale=0.04,
        joint_basis_scale=0.02,
        num_joints=17,
    ):
        super().__init__()
        drop_path = drop_path or [0.0] * depth
        self.blocks = nn.ModuleList(
            [
                MotionEvidenceSamplingBlock(
                    dim=dim,
                    num_heads=num_heads,
                    num_samples=num_samples,
                    qkv_bias=qkv_bias,
                    drop_path=drop_path[i],
                    offset_scale=offset_scale,
                    joint_basis_scale=joint_basis_scale,
                    num_joints=num_joints,
                )
                for i in range(depth)
            ]
        )

    def forward(self, flow_token, joint_token, ref, motion_field):
        for block in self.blocks:
            flow_token = block(flow_token, joint_token, ref, motion_field)
        return flow_token


class RGBFlowMESLifting(nn.Module):
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
        flow_offset_scale = getattr(config, "flow_offset_scale", 0.04)
        flow_joint_basis_scale = getattr(config, "flow_joint_basis_scale", 0.02)
        out_dim = 3

        if backbone in ["hrnet_32", "hrnet_48"]:
            feature_dim_list = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        elif backbone == "cpn":
            feature_dim_list = [base_dim] * 4
        else:
            raise ValueError("Unsupported backbone for RGBFlowMESLifting: {}".format(backbone))

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
        self.Motion_Extraction = MotionEvidenceSampling(
            dim=embed_dim_ratio,
            depth=depth,
            num_heads=flow_num_heads,
            num_samples=flow_num_samples,
            qkv_bias=qkv_bias,
            drop_path=dpr,
            offset_scale=flow_offset_scale,
            joint_basis_scale=flow_joint_basis_scale,
            num_joints=num_joints,
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
        flow_token = self.pos_drop(flow_token.unsqueeze(1) + self.Flow_pos_embed).squeeze(1)

        joint_token = pose_token + self.Flow_pos_embed[:, 0]
        flow_token = self.Motion_Extraction(flow_token, joint_token, ref, motion_field)

        x = torch.cat([x, flow_token.unsqueeze(1)], dim=1)
        x = rearrange(x, "b l p c -> (b p) l c")
        for block in self.Features_Fusion:
            x = block(x)

        x = rearrange(x, "(b p) l c -> b p (l c)", b=b)
        for block in self.Spatial_Transformer:
            x = block(x)

        x = self.head(x).view(b, 1, p, -1)
        return x
