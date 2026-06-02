import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.init import constant_

from mvn.models.DGLifting_rgbflow_capf import Block, DeformableBlock, Mlp


class CounterfactualUtilityFlowSampler(nn.Module):
    """
    Flow-only candidate sampler for joint-level optical-flow tokenization.

    It keeps the DCE/Deformable-DETR sparse sampling form, but constrains the
    offsets around each 2D joint and exposes candidate tokens for a training-time
    counterfactual utility loss computed through the original fusion path.
    """

    def __init__(
        self,
        dim,
        num_samples=5,
        radius_px=8.0,
        center_bias=4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if radius_px <= 0:
            raise ValueError("radius_px must be positive.")

        self.dim = dim
        self.num_samples = int(num_samples)
        self.radius_px = float(radius_px)

        self.query_norm = norm_layer(dim)
        self.candidate_norm = norm_layer(dim)
        self.sampling_offsets = nn.Linear(dim, self.num_samples * 2)
        self.offset_embed = nn.Linear(2, dim)
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.center_logit = nn.Parameter(torch.tensor(float(center_bias)))
        self.out_proj = nn.Linear(dim, dim)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.num_samples, dtype=torch.float32) * (2.0 * math.pi / self.num_samples)
        grid = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid = grid / grid.abs().max(dim=-1, keepdim=True)[0].clamp_min(1e-6)
        for idx in range(self.num_samples):
            grid[idx] *= 0.35 + 0.65 * float(idx + 1) / float(self.num_samples)
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid.reshape(-1))

        constant_(self.score[-1].weight.data, 0.0)
        constant_(self.score[-1].bias.data, 0.0)
        nn.init.eye_(self.out_proj.weight)
        constant_(self.out_proj.bias.data, 0.0)

    @staticmethod
    def _sample_points(feature_map, points, padding_mode="border"):
        if feature_map.dim() != 4:
            raise ValueError("feature_map should be [B, C, H, W].")
        b, c, _, _ = feature_map.shape
        if points.shape[0] != b or points.shape[-1] != 2:
            raise ValueError("points should be [B, ..., 2].")

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
        return sampled.view(*original_shape, c)

    @staticmethod
    def _valid_mask(points):
        return (
            (points[..., 0] >= -1.0)
            & (points[..., 0] <= 1.0)
            & (points[..., 1] >= -1.0)
            & (points[..., 1] <= 1.0)
        )

    def forward(self, flow_features, ref, pose_token):
        """
        Args:
            flow_features: [B, C, H, W]
            ref: [B, J, 2], normalized coordinates in [-1, 1]
            pose_token: [B, J, C], coordinate/joint query only, no RGB context

        Returns:
            flow_feature: [B, J, C]
            aux: candidate features, weights and points for utility supervision
        """
        if flow_features.dim() != 4:
            raise ValueError("flow_features should be [B, C, H, W].")
        if ref.dim() != 3 or ref.shape[-1] != 2:
            raise ValueError("ref should be [B, J, 2].")
        if pose_token.dim() != 3:
            raise ValueError("pose_token should be [B, J, C].")

        b, c, h, w = flow_features.shape
        _, j, _ = ref.shape
        if c != self.dim:
            raise ValueError("flow feature dim mismatch: expected {}, got {}".format(self.dim, c))
        if pose_token.shape != (b, j, c):
            raise ValueError("pose_token should match [B, J, C].")

        center_feature = self._sample_points(flow_features, ref)  # [B, J, C]
        query = self.query_norm(center_feature + pose_token)

        offset_unit = self.sampling_offsets(query).view(b, j, self.num_samples, 2).tanh()
        radius = flow_features.new_tensor([
            2.0 * self.radius_px / max(float(w - 1), 1.0),
            2.0 * self.radius_px / max(float(h - 1), 1.0),
        ])
        offsets = offset_unit * radius.view(1, 1, 1, 2)

        center_points = ref.unsqueeze(2)
        adaptive_points = center_points + offsets
        points = torch.cat([center_points, adaptive_points], dim=2)
        valid = self._valid_mask(points)

        candidates = self._sample_points(flow_features, points)
        relative_offsets = (points - center_points) / radius.view(1, 1, 1, 2).clamp_min(1e-6)
        evidence = candidates + query.unsqueeze(2) + self.offset_embed(relative_offsets)
        evidence = self.candidate_norm(evidence)

        logits = self.score(evidence).squeeze(-1)
        logits[:, :, 0] = logits[:, :, 0] + self.center_logit
        logits = logits.masked_fill(~valid, -1.0e4)
        weights = F.softmax(logits, dim=-1)

        weighted = (weights.unsqueeze(-1) * candidates).sum(dim=2)
        flow_feature = center_feature + self.out_proj(weighted - center_feature)

        aux = {
            "candidate_features": candidates,
            "weights": weights,
            "points": points,
            "logits": logits,
        }
        return flow_feature, aux


class RGBFlowCFUAFSLifting(nn.Module):
    """
    RGBFlow CAPF lifting with Counterfactual Utility-Aware Flow Sampling.

    The RGB DCE branch and downstream Pose-Context Fusion are identical in
    interface to RGBFlowCAPFLifting. Only the flow tokenization step is replaced.
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
            raise ValueError("Unsupported backbone for RGBFlowCFUAFSLifting: {}".format(backbone))

        self.rgb_levels = len(feature_dim_list)
        self.flow_embed = nn.Conv2d(in_channels=2, out_channels=embed_dim_ratio, kernel_size=3, padding=1)
        self.coord_embed = nn.Linear(in_chans, embed_dim_ratio)
        self.feat_embed = nn.ModuleList([nn.Linear(dim_in, embed_dim_ratio) for dim_in in feature_dim_list])
        self.flow_feat_embed = nn.Linear(embed_dim_ratio, embed_dim_ratio)

        self.RGB_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.rgb_levels, num_joints, embed_dim_ratio))
        self.Flow_pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints, embed_dim_ratio))
        self.Flow_query_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.pos_drop = nn.Dropout(p=drop_rate)

        cfuafs_num_samples = getattr(config, "cfuafs_num_samples", getattr(config, "flow_num_samples", 5))
        cfuafs_radius_px = getattr(config, "cfuafs_radius_px", 8.0)
        cfuafs_center_bias = getattr(config, "cfuafs_center_bias", 4.0)
        self.cfuafs_utility_tau = float(getattr(config, "cfuafs_utility_tau", 0.01))
        self.cfuafs_enable_utility = bool(getattr(config, "cfuafs_enable_utility", True))
        self.flow_sampler = CounterfactualUtilityFlowSampler(
            dim=embed_dim_ratio,
            num_samples=cfuafs_num_samples,
            radius_px=cfuafs_radius_px,
            center_bias=cfuafs_center_bias,
            norm_layer=norm_layer,
        )

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

    def _predict_from_flow_feature(self, rgb_tokens, flow_feature):
        b = rgb_tokens.shape[0]
        flow_token = self.flow_feat_embed(flow_feature)
        flow_token = self.pos_drop(flow_token.unsqueeze(1) + self.Flow_pos_embed)

        x = torch.cat([rgb_tokens, flow_token], dim=1)
        x = rearrange(x, "b l p c -> (b p) l c")
        for blk in self.Features_Fusion:
            x = blk(x)

        x = rearrange(x, "(b p) l c -> b p (l c)", b=b)
        for blk in self.Spatial_Transformer:
            x = blk(x)

        return self.head(x).view(b, 1, rgb_tokens.shape[2], -1)

    def _utility_loss(self, rgb_tokens, candidate_features, weights, keypoints_3d_gt):
        if keypoints_3d_gt is None:
            return None, {}
        if keypoints_3d_gt.dim() == 4:
            gt = keypoints_3d_gt.squeeze(1)
        else:
            gt = keypoints_3d_gt

        errors = []
        with torch.no_grad():
            rgb_detached = rgb_tokens.detach()
            candidate_detached = candidate_features.detach()
            for idx in range(candidate_detached.shape[2]):
                pred = self._predict_from_flow_feature(rgb_detached, candidate_detached[:, :, idx]).squeeze(1)
                err = (pred - gt).pow(2).sum(dim=-1).sqrt()
                errors.append(err)
            errors = torch.stack(errors, dim=-1)
            errors = errors - errors.min(dim=-1, keepdim=True)[0]
            target = F.softmax(-errors / max(self.cfuafs_utility_tau, 1.0e-6), dim=-1)

        log_weights = weights.clamp_min(1.0e-6).log()
        utility_loss = -(target * log_weights).sum(dim=-1).mean()
        pred_top = weights.detach().argmax(dim=-1)
        target_top = target.argmax(dim=-1)
        top1 = (pred_top == target_top).float().mean()
        entropy = -(weights.detach().clamp_min(1.0e-6) * log_weights.detach()).sum(dim=-1).mean()
        return utility_loss, {
            "cfuafs_utility_top1": top1.item(),
            "cfuafs_weight_entropy": entropy.item(),
        }

    def forward(self, keypoints_2d, ref, flow_images, features_list_hr, keypoints_3d_gt=None):
        b, p, _ = keypoints_2d.shape
        coord_token = self.coord_embed(keypoints_2d)
        x = coord_token

        features_ref_list = [
            F.grid_sample(features, ref.unsqueeze(-2), align_corners=True).squeeze(-1).permute(0, 2, 1).contiguous()
            for features in features_list_hr
        ]
        features_ref_list = [embed(features_ref_list[idx]) for idx, embed in enumerate(self.feat_embed)]

        x = torch.stack([x, *features_ref_list], dim=1)
        x = x + self.RGB_pos_embed
        x = self.pos_drop(x)

        for blk in self.RGB_Extraction:
            x = blk(x, ref, features_list_hr)

        flow_features = self.flow_embed(flow_images.permute(0, 3, 1, 2).contiguous())
        flow_query = coord_token + self.Flow_query_pos_embed
        flow_feature, flow_aux = self.flow_sampler(flow_features, ref, flow_query)
        keypoints_3d = self._predict_from_flow_feature(x, flow_feature)

        aux = None
        if self.training and self.cfuafs_enable_utility and keypoints_3d_gt is not None:
            utility_loss, utility_metrics = self._utility_loss(
                x,
                flow_aux["candidate_features"],
                flow_aux["weights"],
                keypoints_3d_gt,
            )
            aux = {
                "utility_loss": utility_loss,
                **utility_metrics,
            }

        return keypoints_3d, aux
