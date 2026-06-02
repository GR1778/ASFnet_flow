import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath

from mvn.models.DGLifting_rgbflow_capf import Block, DeformableBlock, Mlp


class LocalMotionRegionSelector(nn.Module):
    """Motion-valued region sampling over fixed local optical-flow candidates."""

    def __init__(
        self,
        dim,
        num_joints=17,
        radii=(2, 4, 6, 8, 12, 16),
        num_directions=16,
        sigma_flow=0.5,
        sigma_pos=8.0,
        tau=0.5,
        topk=8,
        drop=0.0,
        drop_path=0.0,
        mlp_ratio=2.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        radii = self._parse_radii(radii)
        if num_directions <= 0:
            raise ValueError("LMRS num_directions must be positive.")

        self.dim = dim
        self.num_joints = num_joints
        self.sigma_flow = float(sigma_flow)
        self.sigma_pos = float(sigma_pos)
        self.tau = float(tau)
        self.topk = int(topk)

        offsets = self._make_ring_offsets(radii, num_directions)
        self.num_samples = int(offsets.shape[0])
        max_radius = max([abs(float(r)) for r in radii] + [1.0])

        self.register_buffer("candidate_offsets_px", offsets, persistent=False)
        self.register_buffer("candidate_offsets_unit", offsets / max_radius, persistent=False)

        self.feat_norm = norm_layer(dim)
        self.raw_proj = nn.Linear(2, dim)
        self.raw_delta_proj = nn.Linear(2, dim)
        self.offset_proj = nn.Linear(2, dim)
        self.joint_embed = nn.Parameter(torch.zeros(1, num_joints, dim))

        score_dim = max(dim // 4, 16)
        self.region_feat = nn.Linear(dim, dim)
        self.region_norm = norm_layer(dim)
        self.region_mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)

        self.value_feat = nn.Linear(dim, dim)
        self.value_raw = nn.Linear(dim, dim)
        self.value_delta = nn.Linear(dim, dim)
        self.value_offset = nn.Linear(dim, dim)
        self.value_region = nn.Linear(dim, dim)
        self.value_norm = norm_layer(dim)
        self.value_mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)

        self.score_candidate = nn.Linear(dim, score_dim)
        self.score_joint = nn.Linear(dim, score_dim)
        self.score_head = nn.Sequential(
            norm_layer(score_dim),
            nn.Linear(score_dim, score_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(score_dim, 1),
        )
        self.out_norm = norm_layer(dim)
        self.out_mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self._reset_parameters()

    @staticmethod
    def _parse_radii(radii):
        if isinstance(radii, str):
            radii = [value for value in radii.split(",") if value.strip()]
        return [float(radius) for radius in radii]

    @staticmethod
    def _make_ring_offsets(radii, num_directions):
        offsets = [torch.zeros(1, 2, dtype=torch.float32)]
        theta = torch.arange(num_directions, dtype=torch.float32) * (2.0 * math.pi / float(num_directions))
        unit = torch.stack([theta.cos(), theta.sin()], dim=-1)
        for radius in radii:
            if radius > 0:
                offsets.append(unit * float(radius))
        return torch.cat(offsets, dim=0)

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.joint_embed, std=0.02)

    def _sample_candidates(self, feature_map, ref):
        b, _, h, w = feature_map.shape
        _, p, _ = ref.shape
        norm_scale = feature_map.new_tensor([2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1)])
        offsets_norm = self.candidate_offsets_px.to(device=feature_map.device, dtype=feature_map.dtype) * norm_scale
        grid = ref.unsqueeze(2) + offsets_norm.view(1, 1, self.num_samples, 2)
        samples = F.grid_sample(feature_map, grid, padding_mode="border", align_corners=True)
        return samples.permute(0, 2, 3, 1).contiguous().view(b, p, self.num_samples, -1)

    def _motion_region_context(self, raw_samples, region_values):
        offsets_px = self.candidate_offsets_px.to(device=raw_samples.device, dtype=raw_samples.dtype)
        flow_dist = (raw_samples.unsqueeze(3) - raw_samples.unsqueeze(2)).pow(2).sum(dim=-1)
        pos_dist = (offsets_px.unsqueeze(1) - offsets_px.unsqueeze(0)).pow(2).sum(dim=-1)

        flow_denom = max(2.0 * self.sigma_flow * self.sigma_flow, 1e-6)
        pos_denom = max(2.0 * self.sigma_pos * self.sigma_pos, 1e-6)
        logits = -flow_dist / flow_denom - pos_dist.view(1, 1, self.num_samples, self.num_samples) / pos_denom
        region_attn = F.softmax(logits, dim=-1)
        return torch.matmul(region_attn, region_values)

    def _sparse_select(self, scores):
        if 0 < self.topk < scores.shape[-1]:
            top_values, top_indices = scores.topk(self.topk, dim=-1)
            sparse_scores = scores.new_full(scores.shape, float("-inf"))
            scores = sparse_scores.scatter(dim=-1, index=top_indices, src=top_values)
        tau = max(self.tau, 1e-6)
        return F.softmax(scores / tau, dim=-1)

    def forward(self, flow_features, raw_flow_images, ref, pose_token):
        b, p, _ = ref.shape
        if raw_flow_images.dim() != 4 or raw_flow_images.shape[-1] != 2:
            raise ValueError("Expected raw flow images with shape [B, H, W, 2], got {}".format(tuple(raw_flow_images.shape)))

        raw_flow_map = raw_flow_images.permute(0, 3, 1, 2).contiguous()
        raw_samples = self._sample_candidates(raw_flow_map, ref)
        feat_samples = self.feat_norm(self._sample_candidates(flow_features, ref))
        raw_delta = raw_samples - raw_samples[:, :, :1, :]

        offsets_unit = self.candidate_offsets_unit.to(device=feat_samples.device, dtype=feat_samples.dtype)
        offsets_unit = offsets_unit.view(1, 1, self.num_samples, 2).expand(b, p, -1, -1)
        raw_token = self.raw_proj(raw_samples)
        raw_delta_token = self.raw_delta_proj(raw_delta)
        offset_token = self.offset_proj(offsets_unit)

        region_values = self.region_feat(feat_samples) + raw_token + offset_token
        region_values = region_values + self.region_mlp(self.region_norm(region_values))
        region_token = self._motion_region_context(raw_samples, region_values)

        candidate_token = (
            self.value_feat(feat_samples)
            + self.value_raw(raw_token)
            + self.value_delta(raw_delta_token)
            + self.value_offset(offset_token)
            + self.value_region(region_token)
        )
        candidate_token = candidate_token + self.value_mlp(self.value_norm(candidate_token))

        if p == self.num_joints:
            joint_token = self.joint_embed
        else:
            joint_token = self.joint_embed[:, :p]
        joint_token = (pose_token + joint_token).unsqueeze(2).expand(-1, -1, self.num_samples, -1)

        score_token = self.score_candidate(candidate_token) + self.score_joint(joint_token)
        scores = self.score_head(score_token).squeeze(-1)
        alpha = self._sparse_select(scores)
        self.latest_alpha = alpha.detach()

        token = (alpha.unsqueeze(-1) * candidate_token).sum(dim=2)
        token = token + self.drop_path(self.out_mlp(self.out_norm(token)))
        return token


class RGBFlowLMRSLifting(nn.Module):
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
        lmrs_radii = getattr(config, "lmrs_radii", [2, 4, 6, 8, 12, 16])
        lmrs_num_directions = getattr(config, "lmrs_num_directions", 16)
        lmrs_sigma_flow = getattr(config, "lmrs_sigma_flow", 0.5)
        lmrs_sigma_pos = getattr(config, "lmrs_sigma_pos", 8.0)
        lmrs_tau = getattr(config, "lmrs_tau", 0.5)
        lmrs_topk = getattr(config, "lmrs_topk", 8)
        out_dim = 3

        if backbone in ["hrnet_32", "hrnet_48"]:
            feature_dim_list = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        elif backbone == "cpn":
            feature_dim_list = [base_dim] * 4
        else:
            raise ValueError("Unsupported backbone for RGBFlowLMRSLifting: {}".format(backbone))

        self.rgb_levels = len(feature_dim_list)
        self.flow_embed = nn.Conv2d(in_channels=2, out_channels=embed_dim_ratio, kernel_size=3, padding=1)
        self.coord_embed = nn.Linear(in_chans, embed_dim_ratio)
        self.feat_embed = nn.ModuleList([nn.Linear(dim_in, embed_dim_ratio) for dim_in in feature_dim_list])

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

        self.Flow_Sampling = LocalMotionRegionSelector(
            dim=embed_dim_ratio,
            num_joints=num_joints,
            radii=lmrs_radii,
            num_directions=lmrs_num_directions,
            sigma_flow=lmrs_sigma_flow,
            sigma_pos=lmrs_sigma_pos,
            tau=lmrs_tau,
            topk=lmrs_topk,
            drop=drop_rate,
            drop_path=dpr[-1] if dpr else 0.0,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
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
        x = x + self.RGB_pos_embed
        x = self.pos_drop(x)

        for blk in self.RGB_Extraction:
            x = blk(x, ref, features_list_hr)

        flow_features = self.flow_embed(flow_images.permute(0, 3, 1, 2).contiguous())
        flow_token = self.Flow_Sampling(flow_features, flow_images, ref, pose_token)
        flow_token = self.pos_drop(flow_token.unsqueeze(1) + self.Flow_pos_embed)

        x = torch.cat([x, flow_token], dim=1)

        x = rearrange(x, "b l p c -> (b p) l c")
        for blk in self.Features_Fusion:
            x = blk(x)

        x = rearrange(x, "(b p) l c -> b p (l c)", b=b)
        for blk in self.Spatial_Transformer:
            x = blk(x)

        x = self.head(x).view(b, 1, p, -1)
        return x
