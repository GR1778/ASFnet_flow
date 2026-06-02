import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.init import constant_

from mvn.models.DGLifting_rgbflow_capf import Block, DeformableBlock, Mlp


class MotionConsistentEvidenceSampler(nn.Module):
    """
    Motion-consistent evidence sampling for optical-flow tokenization.

    DCE-style offsets propose a joint-local evidence set. The output token is
    generated from motion-consistent aggregation over that set; the center point
    is only one candidate, not a residual fallback.
    """

    def __init__(
        self,
        dim,
        num_heads=4,
        num_samples=5,
        offset_scale=0.125,
        consistency_init=1.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                "dim must be divisible by num_heads, got dim={} and num_heads={}".format(
                    dim,
                    num_heads,
                )
            )
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if offset_scale <= 0:
            raise ValueError("offset_scale must be positive.")
        if consistency_init <= 0:
            raise ValueError("consistency_init must be positive.")

        self.dim = dim
        self.num_heads = num_heads
        self.num_samples = num_samples
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.offset_scale = float(offset_scale)

        self.query_norm = norm_layer(dim)
        self.evidence_norm = norm_layer(dim)

        self.sampling_offsets = nn.Linear(dim, num_heads * num_samples * 2)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, self.head_dim)
        self.v_proj = nn.Linear(dim, self.head_dim)
        self.score_proj = nn.Linear(dim, 1)

        self.raw_flow_embed = nn.Linear(2, dim)
        self.raw_delta_embed = nn.Linear(2, dim)
        self.offset_embed = nn.Linear(2, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.out_norm = norm_layer(dim)
        self.out_mlp = Mlp(
            in_features=dim,
            hidden_features=dim * 2,
            act_layer=nn.GELU,
            drop=0.0,
        )

        consistency_raw = math.log(math.exp(float(consistency_init)) - 1.0)
        self.consistency_log_scale = nn.Parameter(torch.tensor(consistency_raw))

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)

        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        directions = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        directions = directions / directions.abs().max(dim=-1, keepdim=True)[0].clamp_min(1e-6)

        grid_init = directions.view(self.num_heads, 1, 2).repeat(
            1,
            self.num_samples,
            1,
        )
        for idx in range(self.num_samples):
            grid_init[:, idx, :] *= 0.5 * float(idx + 1) / float(self.num_samples)

        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.reshape(-1))

        constant_(self.out_proj.bias.data, 0.0)

    @staticmethod
    def _valid_mask(points):
        return (
            (points[..., 0] >= -1.0)
            & (points[..., 0] <= 1.0)
            & (points[..., 1] >= -1.0)
            & (points[..., 1] <= 1.0)
        )

    @staticmethod
    def _sample_points(feature_map, points, padding_mode="zeros"):
        """
        Args:
            feature_map: [B, C, H, W]
            points: [B, ..., 2], normalized grid coordinates

        Returns:
            sampled: [B, ..., C]
        """
        if feature_map.dim() != 4:
            raise ValueError("feature_map should be [B, C, H, W].")

        b, c, _, _ = feature_map.shape
        if points.shape[0] != b:
            raise ValueError("Batch size mismatch between feature_map and points.")

        original_shape = points.shape[:-1]
        grid = points.reshape(b, -1, 1, 2)
        sampled = F.grid_sample(
            feature_map,
            grid,
            mode="bilinear",
            padding_mode=padding_mode,
            align_corners=True,
        )
        sampled = sampled.squeeze(-1).transpose(1, 2).contiguous()
        sampled = sampled.view(*original_shape, c)
        return sampled

    def forward(self, flow_features, flow_map, ref, pose_token):
        """
        Args:
            flow_features: [B, C, H, W], encoded optical-flow feature map
            flow_map: [B, 2, H, W], raw optical-flow field
            ref: [B, J, 2], normalized joint coordinates in [-1, 1]
            pose_token: [B, J, C], coordinate/joint token, not RGB context

        Returns:
            flow_token: [B, J, C]
        """
        if flow_features.dim() != 4:
            raise ValueError("flow_features should be [B, C, H, W].")
        if flow_map.dim() != 4 or flow_map.shape[1] != 2:
            raise ValueError("flow_map should be [B, 2, H, W].")
        if ref.dim() != 3 or ref.shape[-1] != 2:
            raise ValueError("ref should be [B, J, 2].")
        if pose_token.dim() != 3:
            raise ValueError("pose_token should be [B, J, C].")

        b, c, _, _ = flow_features.shape
        _, j, _ = ref.shape
        if c != self.dim:
            raise ValueError(
                "flow_features channel dim must be {}, got {}".format(self.dim, c)
            )
        if pose_token.shape[0] != b or pose_token.shape[1] != j or pose_token.shape[-1] != c:
            raise ValueError("pose_token should have shape [B, J, C] matching flow_features.")

        center_feature = self._sample_points(flow_features, ref)  # [B, J, C]
        query = self.query_norm(center_feature + pose_token)

        offsets = self.sampling_offsets(query).view(
            b,
            j,
            self.num_heads,
            self.num_samples,
            2,
        )
        offsets = torch.tanh(offsets) * self.offset_scale

        center_points = ref[:, :, None, None, :].expand(
            b,
            j,
            self.num_heads,
            1,
            2,
        )
        adaptive_points = ref[:, :, None, None, :] + offsets
        points = torch.cat([center_points, adaptive_points], dim=3)
        valid = self._valid_mask(points)

        sampled_features = self._sample_points(flow_features, points)
        sampled_flow = self._sample_points(flow_map, points)

        relative_offsets = points - ref[:, :, None, None, :]
        valid_weight = valid.unsqueeze(-1).to(sampled_flow.dtype)
        local_flow_mean = (sampled_flow * valid_weight).sum(dim=3, keepdim=True)
        local_flow_mean = local_flow_mean / valid_weight.sum(dim=3, keepdim=True).clamp_min(1.0)
        raw_delta = sampled_flow - local_flow_mean
        local_flow_energy = (raw_delta.pow(2).sum(dim=-1, keepdim=True) * valid_weight[..., :1])
        local_flow_energy = local_flow_energy.sum(dim=3, keepdim=True)
        local_flow_energy = local_flow_energy / valid_weight[..., :1].sum(dim=3, keepdim=True).clamp_min(1.0)
        normalized_delta = raw_delta / (local_flow_energy + 1.0e-6).sqrt()

        evidence = (
            sampled_features
            + self.raw_flow_embed(sampled_flow)
            + self.raw_delta_embed(normalized_delta)
            + self.offset_embed(relative_offsets / self.offset_scale)
        )
        evidence = self.evidence_norm(evidence)

        q = self.q_proj(query).view(b, j, self.num_heads, self.head_dim)
        k = self.k_proj(evidence)
        v = self.v_proj(evidence)

        base_logits = (q.unsqueeze(3) * k).sum(dim=-1) * self.scale
        base_logits = base_logits + self.score_proj(evidence).squeeze(-1)
        base_logits = base_logits.masked_fill(~valid, -1.0e4)

        proto_weight = F.softmax(base_logits, dim=-1).unsqueeze(-1)
        flow_proto = (proto_weight * normalized_delta).sum(dim=3, keepdim=True)
        flow_dist = (normalized_delta - flow_proto).pow(2).sum(dim=-1)

        consistency_scale = F.softplus(self.consistency_log_scale)
        logits = base_logits - 0.5 * consistency_scale * flow_dist
        logits = logits.masked_fill(~valid, -1.0e4)

        weights = F.softmax(logits, dim=-1).unsqueeze(-1)
        context = (weights * v).sum(dim=3).reshape(b, j, c)
        flow_token = self.out_proj(context)
        flow_token = flow_token + self.out_mlp(self.out_norm(flow_token))

        return flow_token


class RGBFlowMCESLifting(nn.Module):
    """
    RGBFlow CAPF lifting with Motion-Consistent Evidence Sampling for flow.

    The RGB branch and downstream fusion are kept identical to RGBFlowCAPFLifting.
    Only the optical-flow tokenization is replaced.
    """

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
        out_dim = 3

        if backbone in ["hrnet_32", "hrnet_48"]:
            feature_dim_list = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        elif backbone == "cpn":
            feature_dim_list = [base_dim] * 4
        else:
            raise ValueError("Unsupported backbone for RGBFlowMCESLifting: {}".format(backbone))

        self.rgb_levels = len(feature_dim_list)
        self.flow_embed = nn.Conv2d(
            in_channels=2,
            out_channels=embed_dim_ratio,
            kernel_size=3,
            padding=1,
        )
        self.coord_embed = nn.Linear(in_chans, embed_dim_ratio)
        self.feat_embed = nn.ModuleList(
            [nn.Linear(dim_in, embed_dim_ratio) for dim_in in feature_dim_list]
        )
        self.flow_feat_embed = nn.Linear(embed_dim_ratio, embed_dim_ratio)

        flow_num_heads = getattr(config, "mces_num_heads", getattr(config, "flow_num_heads", 4))
        flow_num_samples = getattr(config, "mces_num_samples", getattr(config, "flow_num_samples", 5))
        offset_scale = getattr(config, "mces_offset_scale", 0.125)
        consistency_init = getattr(config, "mces_consistency_init", 1.0)

        self.flow_sampler = MotionConsistentEvidenceSampler(
            dim=embed_dim_ratio,
            num_heads=flow_num_heads,
            num_samples=flow_num_samples,
            offset_scale=offset_scale,
            consistency_init=consistency_init,
            norm_layer=norm_layer,
        )

        self.RGB_pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.rgb_levels, num_joints, embed_dim_ratio)
        )
        self.Flow_query_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
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
                    norm_layer=norm_layer,
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

    @staticmethod
    def _flow_to_bchw(flow_images):
        if flow_images.dim() != 4:
            raise ValueError("flow_images must be a 4D tensor.")
        if flow_images.shape[-1] == 2:
            return flow_images.permute(0, 3, 1, 2).contiguous()
        if flow_images.shape[1] == 2:
            return flow_images.contiguous()
        raise ValueError(
            "Expected flow_images with shape [B, H, W, 2] or [B, 2, H, W], "
            "got {}".format(tuple(flow_images.shape))
        )

    def forward(self, keypoints_2d, ref, flow_images, features_list_hr):
        b, p, _ = keypoints_2d.shape
        pose_token = self.coord_embed(keypoints_2d)

        features_ref_list = [
            F.grid_sample(
                features,
                ref.unsqueeze(-2),
                align_corners=True,
            )
            .squeeze(-1)
            .permute(0, 2, 1)
            .contiguous()
            for features in features_list_hr
        ]
        features_ref_list = [
            embed(features_ref_list[idx])
            for idx, embed in enumerate(self.feat_embed)
        ]

        x = torch.stack([pose_token, *features_ref_list], dim=1)
        x = x + self.RGB_pos_embed
        x = self.pos_drop(x)

        for blk in self.RGB_Extraction:
            x = blk(x, ref, features_list_hr)

        flow_map = self._flow_to_bchw(flow_images)
        flow_features = self.flow_embed(flow_map)

        flow_token = self.flow_sampler(
            flow_features=flow_features,
            flow_map=flow_map,
            ref=ref,
            pose_token=pose_token + self.Flow_query_pos_embed,
        )
        flow_token = self.flow_feat_embed(flow_token)
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
