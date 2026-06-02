import math
from functools import partial
from einops import rearrange
import os
import sys

sys.path.append(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_

from timm.models.layers import DropPath


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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
    def __init__(self, dim, dim_out=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
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
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DeformableBlock(nn.Module):
    def __init__(self, dim, num_heads, num_samples, levels, qkv_bias=False, drop_path=0., mlp_ratio=2,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        self.num_samples = num_samples
        head_dim = dim // num_heads
        self.norm1 = norm_layer(dim)
        self.attention_weights = nn.Linear(dim, num_heads * num_samples)
        self.sampling_offsets = nn.Linear(dim, 2 * num_heads * num_samples)
        self.embed_proj = nn.ModuleList([
            nn.Linear(32, head_dim),
            nn.Linear(64, head_dim),
            nn.Linear(128, head_dim),
            nn.Linear(256, head_dim),
            nn.Linear(dim, head_dim),
        ])
        self.levels = levels

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=0.)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = 0.01 * (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 2)
        grid_init = grid_init.repeat(1, self.num_samples, 1)
        for i in range(self.num_samples):
            grid_init[:, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)

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
            F.grid_sample(features, pos[:, idx], padding_mode='border', align_corners=True)
            .permute(0, 2, 3, 1).contiguous()
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


def _h36m_edges():
    return [
        (0, 1), (1, 2), (2, 3),
        (0, 4), (4, 5), (5, 6),
        (0, 7), (7, 8), (8, 9), (9, 10),
        (8, 11), (11, 12), (12, 13),
        (8, 14), (14, 15), (15, 16),
    ]


def _skeleton_hop_distance(num_joints, edges, max_hop=4):
    dist = torch.full((num_joints, num_joints), max_hop, dtype=torch.long)
    for i in range(num_joints):
        dist[i, i] = 0
    for i, j in edges:
        if i < num_joints and j < num_joints:
            dist[i, j] = 1
            dist[j, i] = 1
    for k in range(num_joints):
        dist = torch.minimum(dist, dist[:, k:k + 1] + dist[k:k + 1, :])
    return dist.clamp_max(max_hop)


class OcclusionAwareRelativeDepthGraph(nn.Module):
    def __init__(
            self, dim=128, num_joints=17, num_heads=4, num_anchors=7,
            relation_temperature=1.5, near_threshold=0.25, depth_gate_temperature=4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.relation_temperature = relation_temperature
        self.near_threshold = near_threshold
        self.depth_gate_temperature = depth_gate_temperature

        self.depth_norm = nn.LayerNorm(dim)
        self.pose_proj = nn.Linear(dim, dim)
        self.rgb_proj = nn.Linear(dim, dim)
        self.xy_proj = nn.Linear(2, dim)
        self.raw_depth_proj = nn.Linear(1, dim)

        self.depth_anchor_embed = nn.Parameter(torch.randn(num_anchors, dim) * 0.02)
        anchor_values = torch.linspace(-1.0, 1.0, steps=num_anchors).view(1, 1, num_anchors)
        self.register_buffer("anchor_values", anchor_values)
        self.anchor_q = nn.Linear(dim, dim)
        self.anchor_k = nn.Linear(dim, dim)
        self.anchor_v = nn.Linear(dim, dim)

        self.rel_residual = nn.Sequential(
            nn.Linear(dim * 2 + 7, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.occ_head = nn.Sequential(
            nn.Linear(dim * 2 + 7, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.rel_to_bias = nn.Linear(1, num_heads, bias=False)
        self.occ_to_bias = nn.Linear(1, num_heads, bias=False)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        edges = _h36m_edges()
        self.bone_edges = edges
        hop_dist = _skeleton_hop_distance(num_joints, edges)
        self.register_buffer("hop_dist", hop_dist)
        self.hop_bias = nn.Embedding(int(hop_dist.max().item()) + 1, num_heads)
        self.edge_parent = nn.Linear(dim * 2 + 3, dim)
        self.edge_child = nn.Linear(dim * 2 + 3, dim)

        degree = torch.ones(num_joints, 1)
        for i, j in edges:
            if i < num_joints and j < num_joints:
                degree[i] += 1
                degree[j] += 1
        self.register_buffer("degree", degree)

        self.update_norm = nn.LayerNorm(dim)
        self.update_mlp = Mlp(in_features=dim, hidden_features=dim * 2, out_features=dim, act_layer=nn.GELU)
        self.update_gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.update_gate[-2].bias, -2.0)

    def _depth_anchor_align(self, h):
        b, j, c = h.shape
        anchors = self.depth_anchor_embed.unsqueeze(0).expand(b, -1, -1)
        q = self.anchor_q(h)
        k = self.anchor_k(anchors)
        v = self.anchor_v(anchors)
        anchor_prob = torch.softmax((q @ k.transpose(1, 2)) / math.sqrt(c), dim=-1)
        anchor_ctx = anchor_prob @ v
        z_anchor = anchor_prob @ self.anchor_values.expand(b, -1, -1).transpose(1, 2)
        return anchor_ctx, z_anchor

    def _sample_joint_depth(self, depth_images, ref_2d):
        grid = ref_2d.view(ref_2d.shape[0], -1, 1, 2)
        return F.grid_sample(
            depth_images.unsqueeze(1), grid, padding_mode='border', align_corners=True
        ).squeeze(1).squeeze(-1)

    def forward(self, x_depth, x_pose, x_rgb, ref_2d, depth_images):
        b, j, c = x_depth.shape
        root_xy = ref_2d[:, :1]
        rel_xy = ref_2d - root_xy
        raw_depth = self._sample_joint_depth(depth_images, ref_2d).unsqueeze(-1)
        raw_depth = raw_depth - raw_depth.mean(dim=1, keepdim=True)
        raw_depth = raw_depth / raw_depth.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)

        rgb_ctx = x_rgb.mean(dim=1)
        h = self.depth_norm(x_depth)
        h = h + self.pose_proj(x_pose) + self.rgb_proj(rgb_ctx) + self.xy_proj(rel_xy) + self.raw_depth_proj(raw_depth)

        anchor_ctx, z_anchor = self._depth_anchor_align(h)
        h = h + anchor_ctx

        zi = z_anchor
        zj = z_anchor.transpose(1, 2)
        xy_i = rel_xy.unsqueeze(2).expand(-1, -1, j, -1)
        xy_j = rel_xy.unsqueeze(1).expand(-1, j, -1, -1)
        h_i = h.unsqueeze(2).expand(-1, -1, j, -1)
        h_j = h.unsqueeze(1).expand(-1, j, -1, -1)
        z_delta = zi - zj
        raw_delta = (raw_depth - raw_depth.transpose(1, 2)).unsqueeze(-1)
        xy_delta = xy_i - xy_j
        xy_dist = xy_delta.norm(dim=-1, keepdim=True)
        near_gate = torch.sigmoid((self.near_threshold - xy_dist) * 20.0)
        depth_contrast = raw_delta.abs()
        contrast_gate = torch.sigmoid(self.depth_gate_temperature * (depth_contrast - depth_contrast.mean(dim=(1, 2), keepdim=True)))
        occ_gate = near_gate * contrast_gate
        pair_feat = torch.cat([
            h_i - h_j,
            h_i * h_j,
            z_delta.unsqueeze(-1),
            raw_delta,
            depth_contrast,
            xy_delta,
            xy_dist,
            occ_gate,
        ], dim=-1)
        rel_res = self.rel_residual(pair_feat).squeeze(-1)
        rel_logits = self.relation_temperature * z_delta + occ_gate.squeeze(-1) * (rel_res - rel_res.transpose(1, 2))
        eye = torch.eye(j, device=x_depth.device, dtype=torch.bool).unsqueeze(0)
        rel_pred = torch.tanh(rel_logits).masked_fill(eye, 0.0)
        occ_pred = torch.sigmoid(self.occ_head(pair_feat).squeeze(-1)) * near_gate.squeeze(-1)
        occ_pred = occ_pred.masked_fill(eye, 0.0)

        q = self.q(h).view(b, j, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(h).view(b, j, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(h).view(b, j, self.num_heads, self.head_dim).transpose(1, 2)
        relation_bias = self.rel_to_bias(rel_pred.unsqueeze(-1)).permute(0, 3, 1, 2)
        occ_bias = self.occ_to_bias(occ_pred.unsqueeze(-1)).permute(0, 3, 1, 2)
        topology_bias = self.hop_bias(self.hop_dist[:j, :j]).permute(2, 0, 1).unsqueeze(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = (attn + relation_bias + occ_bias + topology_bias).softmax(dim=-1)
        relation_msg = (attn @ v).transpose(1, 2).reshape(b, j, c)
        relation_msg = self.out_proj(relation_msg)

        bone_msg = torch.zeros_like(relation_msg)
        for parent, child in self.bone_edges:
            if parent >= j or child >= j:
                continue
            edge_xy = rel_xy[:, child] - rel_xy[:, parent]
            edge_z = z_anchor[:, child] - z_anchor[:, parent]
            edge_feat = torch.cat([h[:, parent], h[:, child], edge_z, edge_xy], dim=-1)
            bone_msg[:, parent] = bone_msg[:, parent] + self.edge_parent(edge_feat)
            bone_msg[:, child] = bone_msg[:, child] + self.edge_child(edge_feat)
        bone_msg = bone_msg / self.degree[:j].view(1, j, 1).clamp_min(1.0)

        msg = relation_msg + bone_msg
        gate = self.update_gate(torch.cat([x_depth, h, msg], dim=-1))
        x_depth = x_depth + gate * (1.0 + occ_pred.mean(dim=-1, keepdim=True)) * self.update_mlp(self.update_norm(msg))
        return x_depth, rel_pred, z_anchor, occ_pred


class DGLifting(nn.Module):
    def __init__(self, config=None, num_frame=1, num_joints=17, in_chans=2,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim_ratio = 128
        base_dim = 32
        depth = 4
        out_dim = 3
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
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.RGBD_Extraction = nn.ModuleList([
            DeformableBlock(
                dim=embed_dim_ratio, num_heads=4, num_samples=5,
                qkv_bias=qkv_bias, drop_path=dpr[i], levels=self.levels)
            for i in range(depth)
        ])

        posealign_cfg = getattr(config, "posealign", None) if config is not None else None
        self.depth_align = OcclusionAwareRelativeDepthGraph(
            dim=embed_dim_ratio,
            num_joints=num_joints,
            num_heads=getattr(posealign_cfg, "num_heads", 4),
            num_anchors=getattr(posealign_cfg, "num_anchors", 7),
            relation_temperature=getattr(posealign_cfg, "relation_temperature", 1.5),
            near_threshold=getattr(posealign_cfg, "near_threshold", 0.25),
            depth_gate_temperature=getattr(posealign_cfg, "depth_gate_temperature", 4.0),
        )

        self.Features_Fusion = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])

        self.Spatial_Transformer = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

    def forward(self, keypoints_2d, ref, depth_images, features_list_hr):
        b, p, c = keypoints_2d.shape
        x = self.coord_embed(keypoints_2d)

        raw_depth_images = depth_images
        depth_images = self.depth_embed(depth_images.unsqueeze(1))
        features_list_hr = list(features_list_hr) + [depth_images]

        features_ref_list = [
            F.grid_sample(features, ref.unsqueeze(-2), align_corners=True)
            .squeeze(-1).permute(0, 2, 1).contiguous()
            for features in features_list_hr
        ]
        features_ref_list = [embed(features_ref_list[idx]) for idx, embed in enumerate(self.feat_embed)]

        x = torch.stack([x, *features_ref_list], dim=1)
        x = self.pos_drop(x + self.Spatial_pos_embed)

        for blk in self.RGBD_Extraction:
            x = blk(x, ref, features_list_hr)

        x_depth = x[:, -1]
        x_pose = x[:, 0]
        x_rgb = x[:, 1:-1]
        x_depth, rel_pred, z_anchor, occ_pred = self.depth_align(x_depth, x_pose, x_rgb, ref, raw_depth_images)
        x = torch.cat((x[:, :-1], x_depth.unsqueeze(1)), dim=1)

        x = rearrange(x, 'b l p c -> (b p) l c')
        for blk in self.Features_Fusion:
            x = blk(x)

        x = rearrange(x, '(b p) l c -> b p (l c)', b=b)
        for blk in self.Spatial_Transformer:
            x = blk(x)

        x = self.head(x).view(b, 1, p, -1)
        return x, rel_pred, z_anchor
