from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath

from mvn.models.DGLifting_rgbflow_capf import Block, DeformableBlock, Mlp


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, context):
        b, nq, c = query.shape
        nk = context.shape[1]

        q = self.q(query).reshape(b, nq, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(b, nk, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(b, nq, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class ContextCMFFBlock(nn.Module):
    """Flow-conditioned RGB context fusion before pose-context exchange."""

    def __init__(self, dim, num_heads=8, mlp_ratio=2.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm_rgb_q = nn.LayerNorm(dim)
        self.norm_flow_ctx = nn.LayerNorm(dim)
        self.rgb_from_flow = CrossAttention(dim, num_heads, qkv_bias, attn_drop, drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm_rgb_mlp = nn.LayerNorm(dim)
        self.rgb_mlp = Mlp(dim, int(dim * mlp_ratio), dim, drop=drop)

    def forward(self, rgb_context, flow_context):
        b, l, p, c = rgb_context.shape
        rgb = rearrange(rgb_context, "b l p c -> (b p) l c")
        flow = rearrange(flow_context, "b l p c -> (b p) l c")

        flow_ctx = self.norm_flow_ctx(flow)

        rgb = rgb + self.drop_path(self.rgb_from_flow(self.norm_rgb_q(rgb), flow_ctx))
        rgb = rgb + self.drop_path(self.rgb_mlp(self.norm_rgb_mlp(rgb)))

        rgb_context = rearrange(rgb, "(b p) l c -> b l p c", b=b, p=p)
        return rgb_context


class RGBFlowCMFFLifting(nn.Module):
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
        cmff_depth = getattr(config, "cmff_depth", 2)
        cmff_heads = getattr(config, "cmff_heads", num_heads)
        out_dim = 3

        if backbone in ["hrnet_32", "hrnet_48"]:
            feature_dim_list = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        elif backbone == "cpn":
            feature_dim_list = [base_dim] * 4
        else:
            raise ValueError("Unsupported backbone for RGBFlowCMFFLifting: {}".format(backbone))

        self.rgb_levels = len(feature_dim_list)
        self.flow_embed = nn.Conv2d(in_channels=2, out_channels=embed_dim_ratio, kernel_size=3, padding=1)
        self.flow_feat_embed = nn.Linear(embed_dim_ratio, embed_dim_ratio)
        self.flow_raw_embed = nn.Linear(2, embed_dim_ratio)
        self.flow_level_embed = nn.Parameter(torch.zeros(1, self.rgb_levels, 1, embed_dim_ratio))
        self.coord_embed = nn.Linear(in_chans, embed_dim_ratio)
        self.feat_embed = nn.ModuleList([nn.Linear(dim_in, embed_dim_ratio) for dim_in in feature_dim_list])

        self.RGB_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.rgb_levels, num_joints, embed_dim_ratio))
        self.Flow_pos_embed = nn.Parameter(torch.zeros(1, self.rgb_levels, num_joints, embed_dim_ratio))
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

        cmff_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, cmff_depth)]
        self.Cross_Modality_Fusion = nn.ModuleList(
            [
                ContextCMFFBlock(
                    dim=embed_dim_ratio,
                    num_heads=cmff_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=cmff_dpr[i],
                )
                for i in range(cmff_depth)
            ]
        )

        embed_dim = embed_dim_ratio * (1 + self.rgb_levels)
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

    def _sample_ref(self, features, ref):
        return F.grid_sample(features, ref.unsqueeze(-2), align_corners=True).squeeze(-1).permute(0, 2, 1).contiguous()

    def forward(self, keypoints_2d, ref, flow_images, features_list_hr):
        b, p, _ = keypoints_2d.shape
        pose_token = self.coord_embed(keypoints_2d)

        features_ref_list = [self._sample_ref(features, ref) for features in features_list_hr]
        features_ref_list = [embed(features_ref_list[idx]) for idx, embed in enumerate(self.feat_embed)]

        x = torch.stack([pose_token, *features_ref_list], dim=1)
        x = self.pos_drop(x + self.RGB_pos_embed)

        for blk in self.RGB_Extraction:
            x = blk(x, ref, features_list_hr)

        flow_map = flow_images.permute(0, 3, 1, 2).contiguous()
        flow_features = self.flow_embed(flow_map)
        flow_context_list = []
        for level_idx in range(self.rgb_levels):
            target_size = features_list_hr[level_idx].shape[-2:]
            flow_features_level = F.adaptive_avg_pool2d(flow_features, output_size=target_size)
            flow_map_level = F.adaptive_avg_pool2d(flow_map, output_size=target_size)
            flow_feat_token = self.flow_feat_embed(self._sample_ref(flow_features_level, ref))
            flow_raw_token = self.flow_raw_embed(self._sample_ref(flow_map_level, ref))
            flow_context_list.append(flow_feat_token + flow_raw_token)
        flow_context = torch.stack(flow_context_list, dim=1)
        flow_context = self.pos_drop(flow_context + self.Flow_pos_embed + self.flow_level_embed)

        rgb_pose_token = x[:, :1]
        rgb_context = x[:, 1:]
        for blk in self.Cross_Modality_Fusion:
            rgb_context = blk(rgb_context, flow_context)

        x = torch.cat([rgb_pose_token, rgb_context], dim=1)

        x = rearrange(x, "b l p c -> (b p) l c")
        for blk in self.Features_Fusion:
            x = blk(x)

        x = rearrange(x, "(b p) l c -> b p (l c)", b=b)
        for blk in self.Spatial_Transformer:
            x = blk(x)

        x = self.head(x).view(b, 1, p, -1)
        return x
