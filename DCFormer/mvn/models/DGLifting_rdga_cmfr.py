import os
import sys
from functools import partial

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mvn.models.DGLifting import Block, DeformableBlock


def cfg_get(cfg, name, default):
    return getattr(cfg, name, default) if cfg is not None else default


class DepthAuxHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = max(dim // 2, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        out = self.head(x)
        depth = out[..., :1]
        logvar = out[..., 1:].clamp(min=-6.0, max=4.0)
        return depth, logvar


class RelativeDepthGeometryAttentionBlock(nn.Module):
    """DFormerv2-style geometry bias, adapted to pose-level depth tokens."""

    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.spatial_decay = nn.Parameter(torch.ones(num_heads))
        self.depth_decay = nn.Parameter(torch.ones(num_heads))
        self.bias_temperature = nn.Parameter(torch.ones(num_heads))

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden, dim),
            nn.Dropout(proj_drop),
        )

    def geometry_bias(self, ref, joint_depth):
        ref_dist = torch.cdist(ref, ref, p=1) / 4.0
        depth_dist = (joint_depth[:, :, None] - joint_depth[:, None, :]).abs()

        spatial = F.softplus(self.spatial_decay).view(1, self.num_heads, 1, 1)
        depth = F.softplus(self.depth_decay).view(1, self.num_heads, 1, 1)
        temp = F.softplus(self.bias_temperature).view(1, self.num_heads, 1, 1) + 1e-6
        bias = -(spatial * ref_dist.unsqueeze(1) + depth * depth_dist.unsqueeze(1)) / temp
        return bias

    def forward(self, x, ref, joint_depth):
        b, joints, c = x.shape
        residual = x
        x = self.norm1(x)

        qkv = self.qkv(x).reshape(b, joints, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.geometry_bias(ref, joint_depth)
        attn = self.attn_drop(attn.softmax(dim=-1))

        x = (attn @ v).transpose(1, 2).reshape(b, joints, c)
        x = residual + self.proj_drop(self.proj(x))
        x = x + self.mlp(self.norm2(x))
        return x


class RelativeDepthGeometryAttention(nn.Module):
    def __init__(self, dim, num_layers=2, num_heads=4, mlp_ratio=2.0, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        hidden = max(dim // 2, 1)
        self.depth_proxy = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.blocks = nn.ModuleList([
            RelativeDepthGeometryAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x_depth, ref):
        for block in self.blocks:
            joint_depth = self.depth_proxy(x_depth).squeeze(-1)
            x_depth = block(x_depth, ref, joint_depth)
        return x_depth


class TokenFeatureRectification(nn.Module):
    """CMX-style channel and token rectification for pose-depth tokens."""

    def __init__(self, dim, reduction=1, lambda_c=0.5, lambda_t=0.5, zero_init=True):
        super().__init__()
        hidden = max((dim * 4) // reduction, 1)
        token_hidden = max(dim // reduction, 1)
        self.lambda_c = lambda_c
        self.lambda_t = lambda_t

        self.channel_mlp = nn.Sequential(
            nn.Linear(dim * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim * 2),
            nn.Sigmoid(),
        )
        self.token_mlp = nn.Sequential(
            nn.Linear(dim * 2, token_hidden),
            nn.GELU(),
            nn.Linear(token_hidden, 2),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.full((1,), 1e-3) if zero_init else torch.ones(1))

    def forward(self, pose_token, depth_token):
        b, joints, c = pose_token.shape
        joint_pair = torch.cat([pose_token, depth_token], dim=-1)
        avg = joint_pair.mean(dim=1)
        max_pool = joint_pair.max(dim=1).values

        channel_weights = self.channel_mlp(torch.cat([avg, max_pool], dim=-1))
        channel_weights = channel_weights.view(b, 2, c)
        token_weights = self.token_mlp(joint_pair)

        pose_delta = (
            self.lambda_c * channel_weights[:, 1].unsqueeze(1) * depth_token
            + self.lambda_t * token_weights[:, :, 1:].contiguous() * depth_token
        )
        depth_delta = (
            self.lambda_c * channel_weights[:, 0].unsqueeze(1) * pose_token
            + self.lambda_t * token_weights[:, :, :1].contiguous() * pose_token
        )
        return pose_token + self.gamma * pose_delta, depth_token + self.gamma * depth_delta


class DGLiftingRDGACMFR(nn.Module):
    def __init__(self, config=None, num_frame=1, num_joints=17, in_chans=2,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=None):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim_ratio = cfg_get(config, "embed_dim_ratio", 128)
        base_dim = cfg_get(config, "base_dim", 32)
        depth = cfg_get(config, "depth", 4)
        branch_cfg = cfg_get(config, "rdga_cmfr", None)

        self.use_rdga = bool(cfg_get(branch_cfg, "use_rdga", True))
        self.use_cmfr = bool(cfg_get(branch_cfg, "use_cmfr", True))
        self.levels = 5
        embed_dim = embed_dim_ratio * (self.levels + 1)

        self.depth_embed = nn.Conv2d(in_channels=1, out_channels=embed_dim_ratio, kernel_size=3, padding=1)
        self.coord_embed = nn.Linear(in_chans, embed_dim_ratio)
        self.feat_embed = nn.ModuleList([
            nn.Linear(base_dim, embed_dim_ratio),
            nn.Linear(base_dim * 2, embed_dim_ratio),
            nn.Linear(base_dim * 4, embed_dim_ratio),
            nn.Linear(base_dim * 8, embed_dim_ratio),
            nn.Linear(embed_dim_ratio, embed_dim_ratio),
        ])

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.levels, num_joints, embed_dim_ratio))
        self.depth_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.RGBD_Extraction = nn.ModuleList([
            DeformableBlock(
                dim=embed_dim_ratio,
                num_heads=4,
                num_samples=5,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                levels=self.levels,
            )
            for i in range(depth)
        ])

        if self.use_rdga:
            self.rdga = RelativeDepthGeometryAttention(
                dim=embed_dim_ratio,
                num_layers=int(cfg_get(branch_cfg, "rdga_layers", 2)),
                num_heads=int(cfg_get(branch_cfg, "rdga_heads", 4)),
                mlp_ratio=float(cfg_get(branch_cfg, "rdga_mlp_ratio", 2.0)),
                qkv_bias=qkv_bias,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
            )
        else:
            self.rdga = nn.Identity()

        if self.use_cmfr:
            self.cmfr = TokenFeatureRectification(
                dim=embed_dim_ratio,
                reduction=int(cfg_get(branch_cfg, "cmfr_reduction", 1)),
                lambda_c=float(cfg_get(branch_cfg, "cmfr_lambda_c", 0.5)),
                lambda_t=float(cfg_get(branch_cfg, "cmfr_lambda_t", 0.5)),
                zero_init=bool(cfg_get(branch_cfg, "cmfr_zero_init", True)),
            )
        else:
            self.cmfr = None

        self.depth_aux = DepthAuxHead(embed_dim_ratio)

        self.Features_Fusion = nn.ModuleList([
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
        ])

        self.Spatial_Transformer = nn.ModuleList([
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
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3),
        )

    def forward(self, keypoints_2d, ref, depth_images, features_list_hr):
        b, joints, _ = keypoints_2d.shape

        x = self.coord_embed(keypoints_2d)
        depth_feature = self.depth_embed(depth_images.unsqueeze(1))
        features_list = list(features_list_hr) + [depth_feature]

        features_ref_list = [
            F.grid_sample(features, ref.unsqueeze(-2), align_corners=True)
            .squeeze(-1).permute(0, 2, 1).contiguous()
            for features in features_list
        ]
        features_ref_list = [embed(features_ref_list[idx]) for idx, embed in enumerate(self.feat_embed)]

        x = torch.stack([x, *features_ref_list], dim=1)
        x = self.pos_drop(x + self.Spatial_pos_embed)

        for blk in self.RGBD_Extraction:
            x = blk(x, ref, features_list)

        x_depth = x[:, -1]
        if self.use_rdga:
            x_depth = self.rdga(x_depth, ref)
        x_depth = x_depth + self.depth_pos_embed
        coarse_depth, logvar = self.depth_aux(x_depth)

        if self.cmfr is not None:
            pose_token, x_depth = self.cmfr(x[:, 0], x_depth)
            x = torch.cat([pose_token.unsqueeze(1), x[:, 1:-1], x_depth.unsqueeze(1)], dim=1)
        else:
            x = torch.cat([x[:, :-1], x_depth.unsqueeze(1)], dim=1)

        x = rearrange(x, "b l p c -> (b p) l c")
        for blk in self.Features_Fusion:
            x = blk(x)

        x = rearrange(x, "(b p) l c -> b p (l c)", b=b)
        for blk in self.Spatial_Transformer:
            x = blk(x)

        x = self.head(x).view(b, 1, joints, -1)
        return x, coarse_depth, logvar
