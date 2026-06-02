from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mvn.models.DGLifting_dlst import Block, DeformableBlock, DepthLayerSortingTransformer, _cfg_get


class DepthFlowDLSTLifting(nn.Module):
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
        self.num_final_tokens = self.levels + 2
        embed_dim = embed_dim_ratio * self.num_final_tokens

        self.depth_embed = nn.Conv2d(in_channels=1, out_channels=embed_dim_ratio, kernel_size=3, padding=1)
        self.flow_embed = nn.Conv2d(in_channels=2, out_channels=embed_dim_ratio, kernel_size=3, padding=1)
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
        self.flow_feat_embed = nn.Linear(embed_dim_ratio, embed_dim_ratio)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.levels, num_joints, embed_dim_ratio))
        self.Flow_pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints, embed_dim_ratio))
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

    def forward(self, keypoints_2d, ref, depth_images, flow_images, features_list_hr):
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

        depth_joint_tokens = x[:, -1]
        depth_joint_tokens, rel_depth, layer_assign = self.dlst(depth_joint_tokens)
        x = torch.cat((x[:, :-1], depth_joint_tokens.unsqueeze(1)), dim=1)

        flow_features = self.flow_embed(flow_images.permute(0, 3, 1, 2).contiguous())
        flow_token = (
            F.grid_sample(flow_features, ref.unsqueeze(-2), align_corners=True)
            .squeeze(-1)
            .permute(0, 2, 1)
            .contiguous()
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
        return x, rel_depth, layer_assign
