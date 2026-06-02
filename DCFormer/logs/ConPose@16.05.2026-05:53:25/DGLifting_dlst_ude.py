"""DGLifting variant: DLST -> UDE in sequence.

The depth-joint token coming out of ``RGBD_Extraction`` is first organized by
the Depth-Layer Sorting Transformer (DLST), then refined by the original
Uncertainty-aware Depth Enhancer (UDE). This ablation tests whether UDE is more
compatible as a post-DLST depth enhancer than as a pre-DLST enhancer.
"""

import math
import os
import sys
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath
from torch.nn.init import constant_

sys.path.append(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DepthUncertaintyModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.decoder_depth = Mlp(in_features=input_dim, hidden_features=input_dim // 2, out_features=1, act_layer=nn.GELU)
        self.decoder_uncer = Mlp(in_features=input_dim, hidden_features=input_dim // 2, out_features=1, act_layer=nn.GELU)

    def forward(self, x):
        joint_depth = self.decoder_depth(x)
        s = self.decoder_uncer(x)
        return joint_depth, s


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        dim_out = dim if dim_out is None else dim_out
        self.proj = nn.Linear(dim, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DeformableBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_samples,
        levels,
        qkv_bias=False,
        drop_path=0.0,
        mlp_ratio=2.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_samples = num_samples
        head_dim = dim // num_heads
        self.norm1 = norm_layer(dim)
        self.attention_weights = nn.Linear(dim, num_heads * num_samples)
        self.sampling_offsets = nn.Linear(dim, 2 * num_heads * num_samples)
        self.embed_proj = nn.ModuleList(
            [
                nn.Linear(32, head_dim),
                nn.Linear(64, head_dim),
                nn.Linear(128, head_dim),
                nn.Linear(256, head_dim),
                nn.Linear(dim, head_dim),
            ]
        )
        self.levels = levels
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=0.0)
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
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)

    def forward(self, x, ref, features_list):
        x_0, x = x[:, :1], x[:, 1:]
        b, l, p, c = x.shape
        residual = x
        x = self.norm1(x + x_0)

        weights = self.attention_weights(x).view(b, l, p, self.num_heads, self.num_samples)
        weights = F.softmax(weights, dim=-1).unsqueeze(-1)
        offsets = self.sampling_offsets(x).reshape(b, l, p, self.num_heads * self.num_samples, 2).tanh()
        pos = offsets + ref.view(b, 1, p, 1, -1)

        features_sampled = [
            F.grid_sample(features, pos[:, idx], padding_mode="border", align_corners=True)
            .permute(0, 2, 3, 1)
            .contiguous()
            for idx, features in enumerate(features_list)
        ]
        features_sampled = [embed(features_sampled[idx]) for idx, embed in enumerate(self.embed_proj)]
        features_sampled = torch.stack(features_sampled, dim=1)
        features_sampled = (
            weights * features_sampled.view(b, l, p, self.num_heads, self.num_samples, -1)
        ).sum(dim=-2).view(b, l, p, -1)

        x = residual + self.drop_path(features_sampled)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return torch.cat([x_0, x], dim=1)


def _fixed_sinusoidal_embedding(length, dim):
    pe = torch.zeros(length, dim)
    position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    if dim > 1:
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)


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


class LayerGatherBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=2.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm_layers = nn.LayerNorm(dim)
        self.norm_joints = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dim, drop=drop)

    def forward(self, layer_tokens, joint_tokens):
        layer_tokens = layer_tokens + self.drop_path(
            self.cross_attn(self.norm_layers(layer_tokens), self.norm_joints(joint_tokens))
        )
        layer_tokens = layer_tokens + self.drop_path(self.mlp(self.norm2(layer_tokens)))
        return layer_tokens


class DepthBiasedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, depth_gate_init=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.depth_gate = nn.Parameter(torch.full((1, num_heads, 1, 1), float(depth_gate_init)))
        self.last_attn = None

    def forward(self, x, rel_depth):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.depth_gate * rel_depth.unsqueeze(1)
        attn = attn.softmax(dim=-1)
        self.last_attn = attn.detach()
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DepthOrderedBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=2.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        depth_gate_init=0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DepthBiasedAttention(dim, num_heads, qkv_bias, attn_drop, drop, depth_gate_init)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dim, drop=drop)

    def forward(self, x, rel_depth):
        x = x + self.drop_path(self.attn(self.norm1(x), rel_depth))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DepthLayerSortingTransformer(nn.Module):
    """Depth-Layer Sorting Transformer (same module as in DGLifting_dlst.py)."""

    def __init__(
        self,
        dim=128,
        num_joints=17,
        num_depth_layers=4,
        num_heads=8,
        depth=1,
        mlp_ratio=2.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        assignment_temperature=1.0,
        omega_temperature=1.0,
        depth_gate_init=0.1,
    ):
        super().__init__()
        self.num_depth_layers = num_depth_layers
        self.assignment_temperature = assignment_temperature

        self.layer_content = nn.Parameter(torch.zeros(1, num_depth_layers, dim))
        self.register_buffer("layer_order_embed", _fixed_sinusoidal_embedding(num_depth_layers, dim), persistent=False)
        nn.init.trunc_normal_(self.layer_content, std=0.02)

        self.joint_norm = nn.LayerNorm(dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.assign_joint = nn.Linear(dim, dim, bias=False)
        self.assign_layer = nn.Linear(dim, dim, bias=False)

        dpr = torch.linspace(0, drop_path, steps=max(depth, 1)).tolist()
        self.layer_gather = nn.ModuleList(
            [LayerGatherBlock(dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, dpr[i]) for i in range(depth)]
        )
        self.depth_blocks = nn.ModuleList(
            [
                DepthOrderedBlock(dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, dpr[i], depth_gate_init)
                for i in range(depth)
            ]
        )

        idx = torch.arange(num_depth_layers, dtype=torch.float32)
        omega = torch.tanh((idx[None, :] - idx[:, None]) / omega_temperature)
        self.register_buffer("omega", omega, persistent=True)

    def forward(self, joint_tokens):
        b, _, c = joint_tokens.shape
        layer_tokens = self.layer_content + self.layer_order_embed.to(
            device=joint_tokens.device, dtype=joint_tokens.dtype
        )
        layer_tokens = layer_tokens.expand(b, -1, -1)

        for blk in self.layer_gather:
            layer_tokens = blk(layer_tokens, joint_tokens)

        joint_q = self.assign_joint(self.joint_norm(joint_tokens))
        layer_k = self.assign_layer(self.layer_norm(layer_tokens))
        assign_logits = (joint_q @ layer_k.transpose(-1, -2)) / math.sqrt(c)
        assign = F.softmax(assign_logits / self.assignment_temperature, dim=-1)

        rel_depth = assign @ self.omega @ assign.transpose(-1, -2)

        out = joint_tokens
        for blk in self.depth_blocks:
            out = blk(out, rel_depth)
        return out, rel_depth, assign


def _cfg_get(config, name, default):
    if config is None:
        return default
    return getattr(config, name, default)


class DGLifting(nn.Module):
    """DLST -> UDE sequential variant of DGLifting.

    The depth-joint token from RGBD_Extraction is first passed through DLST
    (depth-layer sorting transformer), then refined by UDE (uncertainty-aware
    depth enhancer). Both UDE's (coarse_depth, uncer) and DLST's (rel_depth,
    layer_assign) are returned so the baseline BNN loss on UDE plus the DLST
    ordering loss can both be applied.

    Forward returns ``(x, (coarse_depth, uncer), (rel_depth, layer_assign))``
    so the existing 3-tuple unpacking in train.py still applies.
    """

    def __init__(
        self,
        config=None,
        num_frame=1,
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

        embed_dim_ratio = 128
        base_dim = _cfg_get(config, "base_dim", 32)
        depth = 4
        out_dim = 3
        self.levels = 5
        embed_dim = embed_dim_ratio * (self.levels + 1)

        self.depth_embed = nn.Conv2d(in_channels=1, out_channels=embed_dim_ratio, kernel_size=3, padding=1)
        self.coord_embed = nn.Linear(in_chans, embed_dim_ratio)
        self.feat_embed = nn.ModuleList(
            [
                nn.Linear(base_dim, embed_dim_ratio),
                nn.Linear(base_dim * 2, embed_dim_ratio),
                nn.Linear(base_dim * 4, embed_dim_ratio),
                nn.Linear(base_dim * 8, embed_dim_ratio),
                nn.Linear(embed_dim_ratio, embed_dim_ratio),
            ]
        )

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.levels, num_joints, embed_dim_ratio))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.RGBD_Extraction = nn.ModuleList(
            [
                DeformableBlock(
                    dim=embed_dim_ratio,
                    num_heads=4,
                    num_samples=5,
                    qkv_bias=qkv_bias,
                    drop_path=dpr[i],
                    levels=self.levels,
                )
                for i in range(depth)
            ]
        )

        # ---- UDE (Uncertainty-aware Depth Enhancer) ----
        self.depth_uncer = DepthUncertaintyModel(embed_dim_ratio)
        self.Spatial_pos_embed2 = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.z_embed = nn.Linear(1, embed_dim_ratio)
        self.attn_fc = nn.Linear(1, embed_dim_ratio)
        self.attn_depth = Attention(
            dim=embed_dim_ratio * 3,
            dim_out=embed_dim_ratio,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop_rate,
            proj_drop=attn_drop_rate,
        )

        # ---- DLST (Depth-Layer Sorting Transformer) ----
        self.dlst = DepthLayerSortingTransformer(
            dim=embed_dim_ratio,
            num_joints=num_joints,
            num_depth_layers=_cfg_get(config, "dlst_num_depth_layers", 4),
            num_heads=_cfg_get(config, "dlst_num_heads", num_heads),
            depth=_cfg_get(config, "dlst_depth", 1),
            mlp_ratio=_cfg_get(config, "dlst_mlp_ratio", mlp_ratio),
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=_cfg_get(config, "dlst_drop_path", drop_path_rate * 0.5),
            assignment_temperature=_cfg_get(config, "dlst_assignment_temperature", 1.0),
            omega_temperature=_cfg_get(config, "dlst_omega_temperature", 1.0),
            depth_gate_init=_cfg_get(config, "dlst_depth_gate_init", 0.1),
        )

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

    def forward(self, keypoints_2d, ref, depth_images, features_list_hr):
        b, p, _ = keypoints_2d.shape

        x_pose = self.coord_embed(keypoints_2d)
        depth_features = self.depth_embed(depth_images.unsqueeze(1))
        features_list_hr = list(features_list_hr) + [depth_features]

        features_ref_list = [
            F.grid_sample(features, ref.unsqueeze(-2), align_corners=True)
            .squeeze(-1)
            .permute(0, 2, 1)
            .contiguous()
            for features in features_list_hr
        ]
        features_ref_list = [embed(features_ref_list[idx]) for idx, embed in enumerate(self.feat_embed)]

        x = torch.stack([x_pose, *features_ref_list], dim=1)
        x = self.pos_drop(x + self.Spatial_pos_embed)

        for blk in self.RGBD_Extraction:
            x = blk(x, ref, features_list_hr)

        # ---- DLST: organize raw AMS depth tokens into depth-layer-aware tokens ----
        x_depth = x[:, -1]
        x_depth, rel_depth, layer_assign = self.dlst(x_depth)

        # ---- UDE: refine the DLST-organized depth token with uncertainty-aware self-attention ----
        coarse_depth, uncer = self.depth_uncer(x_depth)
        z_value = self.z_embed(coarse_depth) + self.Spatial_pos_embed2
        joint_uncer = F.softmax(self.attn_fc(uncer), dim=1)
        x_depth = torch.cat([joint_uncer, z_value, x_depth], dim=-1)
        x_depth = self.attn_depth(x_depth)

        x = torch.cat((x[:, :-1], x_depth.unsqueeze(1)), dim=1)

        x = rearrange(x, "b l p c -> (b p) l c")
        for blk in self.Features_Fusion:
            x = blk(x)
        x = rearrange(x, "(b p) l c -> b p (l c)", b=b)

        for blk in self.Spatial_Transformer:
            x = blk(x)
        x = self.head(x).view(b, 1, p, -1)
        return x, (coarse_depth, uncer), (rel_depth, layer_assign)
