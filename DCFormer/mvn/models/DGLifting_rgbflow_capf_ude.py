import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath
from torch.nn.init import constant_


# ── Reused from DGLifting_rgbflow_capf ──────────────────────────────

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
    def __init__(self, dim, dim_out=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim_out or dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
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
        self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
        drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DeformableBlock(nn.Module):
    """Original CAPF deformable sampling for RGB features (unchanged)."""

    def __init__(self, feature_dim_list, dim, num_heads, num_samples, qkv_bias=False, drop_path=0.0, mlp_ratio=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        self.num_samples = num_samples
        head_dim = dim // num_heads
        self.norm1 = norm_layer(dim)
        self.attention_weights = nn.Linear(dim, num_heads * num_samples)
        self.sampling_offsets = nn.Linear(dim, 2 * num_heads * num_samples)
        self.embed_proj = nn.ModuleList([nn.Linear(dim_in, head_dim) for dim_in in feature_dim_list])

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=0.0)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = 0.01 * (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 2).repeat(1, self.num_samples, 1)
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
            F.grid_sample(features, pos[:, idx], padding_mode="border", align_corners=True).permute(0, 2, 3, 1).contiguous()
            for idx, features in enumerate(features_list)
        ]
        features_sampled = [embed(features_sampled[idx]) for idx, embed in enumerate(self.embed_proj)]
        features_sampled = torch.stack(features_sampled, dim=1)
        features_sampled = (weights * features_sampled.view(b, l, p, self.num_heads, self.num_samples, -1)).sum(dim=-2).view(b, l, p, -1)

        x = residual + self.drop_path(features_sampled)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = torch.cat([x_0, x], dim=1)
        return x


# ── NEW: Flow Uncertainty Model ─────────────────────────────────────

class FlowUncertaintyModel(nn.Module):
    """Predict flow reliability (log-variance s) and coarse motion (μ) from flow features.

    Analogous to DepthUncertaintyModel in ASFnet. The network regresses:
      - μ: coarse 2D joint displacement (dx, dy) in normalized image coords
      - s: log-variance representing per-joint flow uncertainty
    """

    def __init__(self, input_dim):
        super().__init__()
        hidden = input_dim // 2
        self.decoder_mu = Mlp(in_features=input_dim, hidden_features=hidden, out_features=2)
        self.decoder_s = Mlp(in_features=input_dim, hidden_features=hidden, out_features=1)

    def forward(self, x):
        """x: [B, J, C] flow features at joint positions"""
        mu = self.decoder_mu(x)     # [B, J, 2] coarse 2D motion estimate
        s = self.decoder_s(x)       # [B, J, 1] log-variance
        return mu, s


# ── NEW: RGBFlowCAPFUDELifting ──────────────────────────────────────

class RGBFlowCAPFUDELifting(nn.Module):
    """CAPF + optical-flow with Flow Uncertainty-aware Enhancement (Flow UDE).

    Architecture (same as CAPF except the flow branch):
      RGB branch:  coord_embed + feat_embed → RGB_Extraction (DeformableBlock × depth)
      Flow branch: flow_embed → grid_sample → flow_feat_embed
                   → FlowUncertaintyModel (μ, s)
                   → uncertainty gate + motion embed
                   → self-attention → enhanced flow token
      Fusion:      cat(RGB, enhanced_flow) → Features_Fusion → Spatial_Transformer → head
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
            raise ValueError("Unsupported backbone for RGBFlowCAPFUDELifting: {}".format(backbone))

        self.rgb_levels = len(feature_dim_list)

        # ── RGB branch ──
        self.coord_embed = nn.Linear(in_chans, embed_dim_ratio)
        self.feat_embed = nn.ModuleList([nn.Linear(dim_in, embed_dim_ratio) for dim_in in feature_dim_list])

        self.RGB_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.rgb_levels, num_joints, embed_dim_ratio))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.RGB_Extraction = nn.ModuleList([
            DeformableBlock(
                feature_dim_list=feature_dim_list, dim=embed_dim_ratio,
                num_heads=4, num_samples=4, qkv_bias=qkv_bias, drop_path=dpr[i],
            )
            for i in range(depth)
        ])

        # ── Flow branch (NEW: Flow UDE) ──
        self.flow_embed = nn.Conv2d(in_channels=2, out_channels=embed_dim_ratio, kernel_size=3, padding=1)
        self.flow_feat_embed = nn.Linear(embed_dim_ratio, embed_dim_ratio)

        # Flow UDE components
        self.flow_uncer = FlowUncertaintyModel(embed_dim_ratio)
        self.flow_uncer_fc = nn.Linear(1, embed_dim_ratio)      # s → uncertainty gate
        self.flow_mu_embed = nn.Linear(2, embed_dim_ratio)       # μ → motion embedding
        self.flow_pos_embed_ude = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        # Self-attention for flow feature enhancement (like UDE's attn_depth)
        self.flow_ude_attn = Attention(
            dim=embed_dim_ratio * 3, dim_out=embed_dim_ratio,
            num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop_rate, proj_drop=attn_drop_rate,
        )

        self.Flow_pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints, embed_dim_ratio))

        # ── Fusion & regression ──
        embed_dim = embed_dim_ratio * (2 + self.rgb_levels)
        self.Features_Fusion = nn.ModuleList([
            Block(dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.Spatial_Transformer = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, out_dim))

    def forward(self, keypoints_2d, ref, flow_images, features_list_hr):
        b, p, _ = keypoints_2d.shape

        # ── RGB branch (unchanged from CAPF) ──
        x = self.coord_embed(keypoints_2d)

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

        # ── Flow branch with UDE ──
        flow_features = self.flow_embed(flow_images.permute(0, 3, 1, 2).contiguous())
        flow_token = F.grid_sample(flow_features, ref.unsqueeze(-2), align_corners=True).squeeze(-1).permute(0, 2, 1).contiguous()
        flow_token = self.flow_feat_embed(flow_token)  # [B, J, C]

        # Flow UDE (core novelty)
        flow_token, mu_flow, s_flow = self._flow_ude_forward(flow_token)

        flow_token = self.pos_drop(flow_token.unsqueeze(1) + self.Flow_pos_embed)

        # ── Fusion ──
        x = torch.cat([x, flow_token], dim=1)

        x = rearrange(x, "b l p c -> (b p) l c")
        for blk in self.Features_Fusion:
            x = blk(x)

        x = rearrange(x, "(b p) l c -> b p (l c)", b=b)
        for blk in self.Spatial_Transformer:
            x = blk(x)

        x = self.head(x).view(b, 1, p, -1)
        return x, mu_flow, s_flow

    def _flow_ude_forward(self, flow_token):
        """Flow Uncertainty-aware Enhancement (Flow UDE).

        1. Predict μ (coarse 2D motion) and s (uncertainty log-variance)
        2. Uncertainty gate: s → softmax → normalized attention weights
        3. Motion embedding: μ → linear + positional encoding
        4. Concat + self-attention → enhanced flow token

        Args:
            flow_token: [B, J, C] raw flow features at joint positions
        Returns:
            enhanced_flow: [B, J, C]
            mu_flow: [B, J, 2]
            s_flow: [B, J, 1]
        """
        mu_flow, s_flow = self.flow_uncer(flow_token)  # [B,J,2], [B,J,1]

        # Uncertainty gate (softmax over joints → relative reliability)
        flow_uncertainty = F.softmax(self.flow_uncer_fc(s_flow), dim=1)  # [B, J, C]

        # Coarse motion embedding with positional encoding
        mu_embed = self.flow_mu_embed(mu_flow) + self.flow_pos_embed_ude  # [B, J, C]

        # Concat and self-attention enhance
        # [B, J, 3*C] → self-attn → [B, J, C]
        enhanced = torch.cat([flow_uncertainty, mu_embed, flow_token], dim=-1)
        enhanced = self.flow_ude_attn(enhanced)

        return enhanced, mu_flow, s_flow
