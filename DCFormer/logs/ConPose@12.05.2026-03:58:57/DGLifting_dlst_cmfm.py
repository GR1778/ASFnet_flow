import math
import os
import sys
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
try:
    from timm.models.layers import DropPath
except Exception:
    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, x):
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor
from torch.nn.init import constant_

sys.path.append(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


VMAMBA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "third_party", "VMamba_official")
)
if os.path.isdir(VMAMBA_ROOT) and VMAMBA_ROOT not in sys.path:
    sys.path.insert(0, VMAMBA_ROOT)


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
    """
    Depth-Layer Sorting Transformer.

    Layer 0 is closest to the camera and layer K-1 is farthest. rel_depth[i, j] > 0
    means joint i is predicted to be in front of joint j.
    """

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



class EfficientChannelAttention(nn.Module):
    """Efficient channel attention for compact level-joint token grids."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        # x: [B, L, J, C]
        weights = x.mean(dim=(1, 2)).unsqueeze(1)  # [B, 1, C]
        weights = torch.sigmoid(self.conv(weights)).squeeze(1).view(x.shape[0], 1, 1, x.shape[-1])
        return x * weights


class DepthwiseTokenConv2d(nn.Module):
    """Depthwise convolution over a compact [level/modality, joint] token plane."""

    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=True,
        )

    def forward(self, x):
        # x: [B, L, J, C]
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        return x.permute(0, 2, 3, 1).contiguous()


class SequenceStateBlock(nn.Module):
    """1D Mamba wrapper used for optional SSM-based pose-context exchange and fallback scans."""

    def __init__(self, dim, drop=0.0, d_state=16, d_conv=4, expand=2, backend="auto"):
        super().__init__()
        self.backend = "fallback-conv1d"
        self.mamba = None
        if backend in ("auto", "mamba"):
            try:
                from mamba_ssm import Mamba

                self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
                self.backend = "mamba"
            except Exception:
                if backend == "mamba":
                    raise

        if self.mamba is None:
            hidden = int(dim * expand)
            self.fallback = nn.Sequential(
                nn.Conv1d(dim, hidden, kernel_size=3, padding=1, groups=dim),
                nn.GELU(),
                nn.Conv1d(hidden, dim, kernel_size=1),
            )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x: [B, N, C]
        if self.mamba is not None:
            return self.drop(self.mamba(x))
        return self.drop(self.fallback(x.transpose(1, 2)).transpose(1, 2))


class CrossScanFallbackSS2D(nn.Module):
    """A lightweight four-direction scan fallback for shape/debug runs.

    This is not a replacement for official VMamba SS2D in final experiments.
    It preserves the same [B, H, W, C] interface and approximates cross-scan
    behavior with row/column forward and reverse 1D state blocks.
    """

    def __init__(self, dim, drop=0.0, d_state=16, d_conv=3, expand=2, backend="auto"):
        super().__init__()
        self.row_scan = SequenceStateBlock(dim, drop=drop, d_state=d_state, d_conv=d_conv, expand=expand, backend=backend)
        self.col_scan = SequenceStateBlock(dim, drop=drop, d_state=d_state, d_conv=d_conv, expand=expand, backend=backend)
        self.proj = nn.Linear(dim * 4, dim)
        self.norm = nn.LayerNorm(dim)

    @property
    def backend(self):
        return f"crossscan-fallback/{self.row_scan.backend}"

    def forward(self, x):
        # x: [B, H, W, C]
        b, h, w, c = x.shape
        row = x.reshape(b, h * w, c)
        row_f = self.row_scan(row).reshape(b, h, w, c)
        row_b = torch.flip(self.row_scan(torch.flip(row, dims=[1])), dims=[1]).reshape(b, h, w, c)

        col = x.transpose(1, 2).reshape(b, w * h, c)
        col_f = self.col_scan(col).reshape(b, w, h, c).transpose(1, 2).contiguous()
        col_b = torch.flip(self.col_scan(torch.flip(col, dims=[1])), dims=[1]).reshape(b, w, h, c).transpose(1, 2).contiguous()

        out = self.proj(torch.cat([row_f, row_b, col_f, col_b], dim=-1))
        return self.norm(out)


class OfficialVMambaSS2D(nn.Module):
    """Adapter around official VMamba SS2D.

    Input and output use channel-last [B, H, W, C]. Here H/W are not image
    height/width necessarily; in O2M-CMFM they are the compact
    level/modality and joint dimensions.
    """

    def __init__(
        self,
        dim,
        drop=0.0,
        d_state=16,
        d_conv=3,
        expand=2,
        backend="auto",
        forward_type="v05_noz",
        initialize="v0",
        conv_bias=False,
    ):
        super().__init__()
        self._backend = "fallback"
        self.ss2d = None
        d_conv = int(d_conv)
        if d_conv > 1 and d_conv % 2 == 0:
            raise ValueError("Official VMamba SS2D expects odd d_conv to preserve H/W; use 3 or another odd value.")

        if backend in ("auto", "vmamba", "ss2d", "official"):
            try:
                from classification.models.vmamba import SS2D
            except Exception as exc:
                if backend != "auto":
                    raise ImportError(
                        "Official VMamba SS2D is required. Clone VMamba to third_party/VMamba_official "
                        "and install selective_scan kernels, or set cmfm_backend='auto' for debugging fallback."
                    ) from exc
            else:
                self.ss2d = SS2D(
                    d_model=dim,
                    d_state=d_state,
                    ssm_ratio=float(expand),
                    d_conv=d_conv,
                    conv_bias=conv_bias,
                    dropout=drop,
                    initialize=initialize,
                    forward_type=forward_type,
                    channel_first=False,
                )
                self._backend = f"vmamba:{forward_type}"

        if self.ss2d is None:
            self.fallback = CrossScanFallbackSS2D(dim, drop=drop, d_state=d_state, d_conv=d_conv, expand=expand, backend="auto")
            self._backend = self.fallback.backend

    @property
    def backend(self):
        return self._backend

    def forward(self, x):
        if self.ss2d is not None:
            return self.ss2d(x)
        return self.fallback(x)


class OneToManyCrossModalityFusionMamba(nn.Module):
    """Asymmetric RGB-D CMFM for 4 RGB context levels and 1 compact depth token.

    Inputs:
        rgb_tokens:   [B, S, J, C], multi-level RGB/image context tokens.
        depth_token:  [B, J, C], compact depth context token after DLST.

    Outputs:
        fused_rgb:    [B, S, J, C], depth-fused context tokens X_s.
        depth_out:    [B, J, C], updated depth token for optional ablation.

    The main paper-faithful setting drops depth_out before PCFE and uses
    [P, X_1, ..., X_S]. depth_out is returned so that a keep-depth ablation can
    be run without adding another module.
    """

    def __init__(
        self,
        dim,
        levels,
        drop=0.0,
        drop_path=0.0,
        d_state=16,
        d_conv=3,
        expand=2,
        backend="auto",
        forward_type="v05_noz",
        initialize="v0",
        conv_bias=False,
        eca_kernel=3,
        init_scale=0.1,
    ):
        super().__init__()
        self.levels = int(levels)
        self.dim = int(dim)

        self.rgb_norm = nn.LayerNorm(dim)
        self.depth_norm = nn.LayerNorm(dim)
        self.level_embed = nn.Parameter(torch.zeros(1, levels, 1, dim))
        self.row_embed = nn.Parameter(torch.zeros(1, levels + 1, 1, dim))
        nn.init.trunc_normal_(self.level_embed, std=0.02)
        nn.init.trunc_normal_(self.row_embed, std=0.02)

        # Depth-anchor rectification: depth modulates every RGB level, but is
        # not treated as four independent depth feature levels.
        gate_in_dim = dim * 5
        self.rect_gate = nn.Sequential(
            nn.Linear(gate_in_dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.level_score = nn.Sequential(
            nn.Linear(gate_in_dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.depth_to_rgb = nn.Linear(dim, dim)
        self.rgb_to_depth = nn.Linear(dim, dim)
        self.depth_update = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

        # CMFM-style pairwise interaction before SS2D.
        self.rgb_pair_norm = nn.LayerNorm(dim)
        self.depth_pair_norm = nn.LayerNorm(dim)
        self.rgb_pair_linear = nn.Linear(dim, dim)
        self.depth_pair_linear = nn.Linear(dim, dim)
        self.rgb_pair_dwc = DepthwiseTokenConv2d(dim)
        self.depth_pair_dwc = DepthwiseTokenConv2d(dim)
        self.pair_out = nn.Linear(dim, dim)

        # SS2D over the real grid: S RGB rows + 1 depth-anchor row.
        self.grid_norm = nn.LayerNorm(dim)
        self.grid_linear = nn.Linear(dim, dim)
        self.grid_dwc = DepthwiseTokenConv2d(dim)
        self.ss2d = OfficialVMambaSS2D(
            dim,
            drop=drop,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            backend=backend,
            forward_type=forward_type,
            initialize=initialize,
            conv_bias=conv_bias,
        )
        self.scan_norm = nn.LayerNorm(dim)

        # Branch fusion and channel selection.
        self.rgb_out_gate = nn.Linear(dim, dim)
        self.depth_out_gate = nn.Linear(dim, dim)
        self.fuse_mlp = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim * 2, dim),
        )
        self.out_norm = nn.LayerNorm(dim)
        self.eca = EfficientChannelAttention(dim, kernel_size=eca_kernel)
        self.depth_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Small residual gates make this safe to insert into a trained/baseline pipeline.
        self.rect_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.depth_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.pair_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.grid_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.out_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.depth_out_scale = nn.Parameter(torch.tensor(float(init_scale)))

    @property
    def backend(self):
        return self.ss2d.backend

    def forward(self, rgb_tokens, depth_token):
        b, levels, joints, c = rgb_tokens.shape
        if levels != self.levels:
            raise ValueError(f"Expected {self.levels} RGB context levels, got {levels}.")
        if depth_token.shape != (b, joints, c):
            raise ValueError(f"depth_token must be [B, J, C], got {tuple(depth_token.shape)}.")

        rgb_n = self.rgb_norm(rgb_tokens)
        depth_n = self.depth_norm(depth_token)
        depth_for_level = depth_n.unsqueeze(1).expand(-1, levels, -1, -1)
        level_embed = self.level_embed[:, :levels].expand(b, -1, joints, -1)

        gate_in = torch.cat(
            [rgb_n, depth_for_level, rgb_n * depth_for_level, rgb_n - depth_for_level, level_embed],
            dim=-1,
        )
        rect_gate = torch.sigmoid(self.rect_gate(gate_in))
        rgb_rect = rgb_tokens + self.drop_path(self.rect_scale * rect_gate * self.depth_to_rgb(depth_for_level))

        # RGB levels update the single depth anchor with normalized level weights.
        level_weight = F.softmax(self.level_score(gate_in), dim=1)  # [B, S, J, 1]
        depth_msg = (level_weight * self.rgb_to_depth(rgb_n)).sum(dim=1)
        depth_rect = depth_token + self.drop_path(self.depth_scale * self.depth_update(depth_msg))

        # CMFM-like local multiplicative interaction. The depth anchor is expanded
        # only for pairwise modulation; it is not output as fake depth levels.
        depth_rect_grid = self.depth_pair_norm(depth_rect).unsqueeze(1).expand(-1, levels, -1, -1)
        rgb_pair = self.rgb_pair_dwc(self.rgb_pair_linear(self.rgb_pair_norm(rgb_rect)))
        depth_pair = self.depth_pair_dwc(self.depth_pair_linear(depth_rect_grid))
        pair = rgb_pair * depth_pair
        rgb_pair = rgb_rect + self.drop_path(self.pair_scale * self.pair_out(pair))

        # Real compact grid: S RGB context rows + one depth-anchor row.
        grid = torch.cat([rgb_pair, depth_rect.unsqueeze(1)], dim=1)
        grid = grid + self.row_embed[:, : levels + 1]
        grid_mixed = grid + self.drop_path(self.grid_scale * self.grid_dwc(self.grid_linear(self.grid_norm(grid))))
        scan = self.scan_norm(self.ss2d(F.silu(grid_mixed)))

        z_rgb = scan[:, :levels]
        z_depth = scan[:, levels : levels + 1].expand(-1, levels, -1, -1)
        rgb_branch = z_rgb * F.silu(self.rgb_out_gate(rgb_pair))
        depth_branch = z_depth * F.silu(self.depth_out_gate(depth_rect).unsqueeze(1).expand(-1, levels, -1, -1))
        fused = self.fuse_mlp(torch.cat([rgb_branch, depth_branch, rgb_branch * depth_branch, rgb_pair], dim=-1))
        fused = self.eca(self.out_norm(fused))

        fused_rgb = rgb_pair + self.drop_path(self.out_scale * fused)
        depth_out = depth_rect + self.drop_path(self.depth_out_scale * self.depth_out(scan[:, levels]))
        return fused_rgb, depth_out


class PoseContextFeatureExchangeBlock(nn.Module):
    """Optional CA-MambaPose-style SSM replacement for the per-joint token fusion block."""

    def __init__(self, dim, mlp_ratio=2.0, drop=0.0, drop_path=0.0, d_state=16, d_conv=4, expand=2, backend="auto"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ssm = SequenceStateBlock(dim, drop=drop, d_state=d_state, d_conv=d_conv, expand=expand, backend=backend)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dim, drop=drop)

    @property
    def backend(self):
        return self.ssm.backend

    def forward(self, x):
        x = x + self.drop_path(self.ssm(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



def _cfg_get(config, name, default):
    if config is None:
        return default
    return getattr(config, name, default)


class DGLifting(nn.Module):
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

        # Feature-extraction levels: 4 HRNet RGB levels + 1 compact depth feature map.
        self.levels = 5
        self.rgb_levels = self.levels - 1

        # Main setting: CMFM fuses the single depth token into four fused context
        # tokens X_s, then PCFE sees [P, X_1, ..., X_S]. A keep-depth variant is
        # left as an ablation because ASFnet's original MIT uses [P, F_s, F_ed].
        self.use_cmfm = bool(_cfg_get(config, "use_cmfm", True))
        self.cmfm_keep_depth_token = bool(_cfg_get(config, "cmfm_keep_depth_token", False))
        if self.use_cmfm and not self.cmfm_keep_depth_token:
            self.final_token_count = 1 + self.rgb_levels
        else:
            self.final_token_count = 1 + self.levels
        embed_dim = embed_dim_ratio * self.final_token_count

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

        # Input positional embedding is for the extraction stage, where the token
        # layout is always [pose, RGB_1..RGB_4, depth].
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.levels, num_joints, embed_dim_ratio))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.use_cmfm:
            self.fusion_pos_embed = nn.Parameter(torch.zeros(1, self.final_token_count, num_joints, embed_dim_ratio))
            nn.init.trunc_normal_(self.fusion_pos_embed, std=0.02)
        else:
            self.fusion_pos_embed = None

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

        if self.use_cmfm:
            self.cmfm = OneToManyCrossModalityFusionMamba(
                dim=embed_dim_ratio,
                levels=self.rgb_levels,
                drop=drop_rate,
                drop_path=_cfg_get(config, "cmfm_drop_path", drop_path_rate * 0.5),
                d_state=_cfg_get(config, "cmfm_d_state", 1),
                d_conv=_cfg_get(config, "cmfm_d_conv", 3),
                expand=_cfg_get(config, "cmfm_expand", 2),
                backend=_cfg_get(config, "cmfm_backend", "vmamba"),
                forward_type=_cfg_get(config, "cmfm_forward_type", "v05_noz"),
                initialize=_cfg_get(config, "cmfm_initialize", "v0"),
                conv_bias=_cfg_get(config, "cmfm_conv_bias", False),
                eca_kernel=_cfg_get(config, "cmfm_eca_kernel", 3),
                init_scale=_cfg_get(config, "cmfm_init_scale", 0.1),
            )
        else:
            self.cmfm = None

        # Keep the original transformer fusion as default to isolate the CMFM gain.
        # Set pcfe_use_ssm=True only for the CA-MambaPose-style SSM ablation.
        self.use_mamba_pcfe = bool(_cfg_get(config, "pcfe_use_ssm", False))
        if self.use_mamba_pcfe:
            self.Features_Fusion = nn.ModuleList(
                [
                    PoseContextFeatureExchangeBlock(
                        dim=embed_dim_ratio,
                        mlp_ratio=mlp_ratio,
                        drop=drop_rate,
                        drop_path=dpr[i],
                        d_state=_cfg_get(config, "pcfe_d_state", 16),
                        d_conv=_cfg_get(config, "pcfe_d_conv", 4),
                        expand=_cfg_get(config, "pcfe_expand", 2),
                        backend=_cfg_get(config, "pcfe_backend", "auto"),
                    )
                    for i in range(depth)
                ]
            )
        else:
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

        # Extraction-stage layout: [pose, RGB_1, RGB_2, RGB_3, RGB_4, depth].
        x = torch.stack([x_pose, *features_ref_list], dim=1)
        x = self.pos_drop(x + self.Spatial_pos_embed)

        for blk in self.RGBD_Extraction:
            x = blk(x, ref, features_list_hr)

        depth_joint_tokens = x[:, -1]
        depth_joint_tokens, rel_depth, layer_assign = self.dlst(depth_joint_tokens)

        if self.use_cmfm:
            # O2M-CMFM: 4 RGB levels + 1 depth anchor -> 4 fused context tokens.
            fused_context_tokens, depth_joint_tokens = self.cmfm(x[:, 1:-1], depth_joint_tokens)
            if self.cmfm_keep_depth_token:
                x = torch.cat((x[:, :1], fused_context_tokens, depth_joint_tokens.unsqueeze(1)), dim=1)
            else:
                x = torch.cat((x[:, :1], fused_context_tokens), dim=1)
            x = x + self.fusion_pos_embed
        else:
            x = torch.cat((x[:, :-1], depth_joint_tokens.unsqueeze(1)), dim=1)

        x = rearrange(x, "b l p c -> (b p) l c")
        for blk in self.Features_Fusion:
            x = blk(x)
        x = rearrange(x, "(b p) l c -> b p (l c)", b=b)

        for blk in self.Spatial_Transformer:
            x = blk(x)
        x = self.head(x).view(b, 1, p, -1)
        return x, rel_depth, layer_assign
