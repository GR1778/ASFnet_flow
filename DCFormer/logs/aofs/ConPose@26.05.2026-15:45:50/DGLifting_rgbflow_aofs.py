from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath

from mvn.models.DGLifting_rgbflow_capf import Block, DeformableBlock, Mlp


class AdaptiveOpticalFlowSampling(nn.Module):
    """Joint-conditioned adaptive sampling over a local optical-flow evidence field.

    AOFS keeps the 2D joint as the anchor and learns sampling weights over a
    compact local flow neighborhood. The output is a joint-level motion token,
    so the downstream RGB/pose/flow fusion interface stays unchanged.
    """

    def __init__(
        self,
        dim,
        num_joints=17,
        kernel_size=5,
        radius_px=8.0,
        num_heads=4,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        mlp_ratio=2.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("AOFS kernel_size must be odd.")
        if dim % num_heads != 0:
            raise ValueError("AOFS dim must be divisible by num_heads.")

        self.dim = dim
        self.num_joints = num_joints
        self.kernel_size = kernel_size
        self.num_samples = kernel_size * kernel_size
        self.radius_px = float(radius_px)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.input_proj = nn.Linear(dim, dim)
        self.query_norm = norm_layer(dim)
        self.evidence_norm = norm_layer(dim)
        self.out_norm = norm_layer(dim)

        self.joint_embed = nn.Parameter(torch.zeros(1, num_joints, dim))
        self.relative_pos_embed = nn.Parameter(torch.zeros(1, 1, self.num_samples, dim))

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.register_buffer("local_offsets_px", self._make_local_offsets(kernel_size, radius_px), persistent=False)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.joint_embed, std=0.02)
        nn.init.trunc_normal_(self.relative_pos_embed, std=0.02)

    @staticmethod
    def _make_local_offsets(kernel_size, radius_px):
        coords = torch.linspace(-float(radius_px), float(radius_px), steps=kernel_size)
        yy, xx = torch.meshgrid(coords, coords)
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

    def _sample_local_evidence(self, motion_field, ref):
        b, _, h, w = motion_field.shape
        _, p, _ = ref.shape

        norm_scale = motion_field.new_tensor([2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1)])
        offsets = self.local_offsets_px.to(device=motion_field.device, dtype=motion_field.dtype) * norm_scale
        grid = ref.unsqueeze(2) + offsets.view(1, 1, self.num_samples, 2)

        samples = F.grid_sample(motion_field, grid, padding_mode="border", align_corners=True)
        samples = samples.permute(0, 2, 3, 1).contiguous()
        return samples.view(b, p, self.num_samples, self.dim)

    def forward(self, motion_field, ref, pose_token):
        b, p, _ = ref.shape

        center = F.grid_sample(motion_field, ref.unsqueeze(-2), padding_mode="border", align_corners=True)
        center = center.squeeze(-1).permute(0, 2, 1).contiguous()
        center = self.input_proj(center)

        evidence = self._sample_local_evidence(motion_field, ref)
        evidence = self.input_proj(evidence)
        evidence = evidence + self.relative_pos_embed
        evidence = self.evidence_norm(evidence)

        if p == self.num_joints:
            joint_embed = self.joint_embed
        else:
            joint_embed = self.joint_embed[:, :p]
        query = self.query_norm(center + pose_token + joint_embed)

        q = self.q(query).view(b, p, self.num_heads, self.head_dim)
        k = self.k(evidence).view(b, p, self.num_samples, self.num_heads, self.head_dim)
        v = self.v(evidence).view(b, p, self.num_samples, self.num_heads, self.head_dim)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)

        attn = (q.unsqueeze(-2) * k).sum(dim=-1) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        reassembled = (attn.unsqueeze(-1) * v).sum(dim=-2).reshape(b, p, self.dim)
        reassembled = self.proj_drop(self.out_proj(reassembled))

        token = center + self.drop_path(reassembled)
        token = token + self.drop_path(self.mlp(self.out_norm(token)))
        return token


class RGBFlowAOFSLifting(nn.Module):
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
        aofs_kernel_size = getattr(config, "aofs_kernel_size", 5)
        aofs_radius_px = getattr(config, "aofs_radius_px", 8.0)
        aofs_num_heads = getattr(config, "aofs_num_heads", 4)
        out_dim = 3

        if backbone in ["hrnet_32", "hrnet_48"]:
            feature_dim_list = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        elif backbone == "cpn":
            feature_dim_list = [base_dim] * 4
        else:
            raise ValueError("Unsupported backbone for RGBFlowAOFSLifting: {}".format(backbone))

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

        self.Flow_Sampling = AdaptiveOpticalFlowSampling(
            dim=embed_dim_ratio,
            num_joints=num_joints,
            kernel_size=aofs_kernel_size,
            radius_px=aofs_radius_px,
            num_heads=aofs_num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
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
        flow_token = self.Flow_Sampling(flow_features, ref, pose_token)
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
