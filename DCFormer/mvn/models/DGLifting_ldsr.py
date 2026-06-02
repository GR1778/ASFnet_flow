import math
from functools import partial
from einops import rearrange
import os,sys
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

class DepthUncertaintyModel(nn.Module):
    def __init__(self, input_dim):
        super(DepthUncertaintyModel, self).__init__()
        self.decoder_depth = Mlp(in_features=input_dim, hidden_features=input_dim // 2, out_features=1, act_layer=nn.GELU)
        self.decoder_uncer = Mlp(in_features=input_dim, hidden_features=input_dim // 2, out_features=1, act_layer=nn.GELU)

    def forward(self, x):
        # input: [B, K, C]     
        joint_depth = self.decoder_depth(x)
        s = self.decoder_uncer(x)  # 对数方差
        return joint_depth, s


class Attention(nn.Module):
    def __init__(self, dim, dim_out=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        if dim_out is None:
            dim_out=dim
        self.proj = nn.Linear(dim, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DeformableBlock(nn.Module):

    def __init__(self, dim, num_heads, num_samples, levels, qkv_bias=False, drop_path=0., mlp_ratio=2, norm_layer=nn.LayerNorm):
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
        # depth features_list
        # depth_dim = dim//2
        # self.depth_embed = nn.Linear(dim, depth_dim)
        # self.embed_proj = nn.ModuleList([
        #     nn.Linear(depth_dim, head_dim),
        #     nn.Linear(depth_dim, head_dim),
        #     nn.Linear(depth_dim, head_dim),
        #     nn.Linear(depth_dim, head_dim),
        #     ])
        self.levels = levels

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

        self._reset_parameters()

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = 0.01 * (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 2).repeat(1, self.num_samples, 1)
        for i in range(self.num_samples):
            grid_init[:, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)

    def forward(self, x, ref, features_list):
        # b, 17, 1, 2
        # x_0, x_hr, x_depth = x[:, :1], x[:, 1:1+self.levels], x[:, 1+self.levels:]
        x_0, x = x[:, :1], x[:, 1:]
        b, l, p, c = x.shape
        residual = x
        x = self.norm1(x + x_0)

        weights = self.attention_weights(x).view(b, l, p, self.num_heads, self.num_samples)
        weights = F.softmax(weights, dim=-1).unsqueeze(-1) # b, l, p, num_heads, num_samples, 1
        offsets = self.sampling_offsets(x).reshape(b, l, p, self.num_heads*self.num_samples, 2).tanh()
        pos = offsets + ref.view(b, 1, p, 1, -1) #torch.Size([512, l, 17, 16, 2])

        features_sampled = [
            F.grid_sample(features, pos[:, idx], padding_mode='border', align_corners=True).permute(0, 2, 3, 1).contiguous() \
            for idx, features in enumerate(features_list)]
        # depth_feature = self.depth_embed(depth_feature).permute(0, 3, 1, 2).contiguous() # [b, dim, 256, 192]
        # features_sampled = [
        #     F.grid_sample(depth_feature.clone(), pos[:, idx], padding_mode='border', align_corners=True).permute(0, 2, 3, 1).contiguous() \
        #     for idx in range(pos.shape[1])]

        # b, p, num_heads*num_samples, c
        features_sampled = [embed(features_sampled[idx]) for idx, embed in enumerate(self.embed_proj)]
        features_sampled = torch.stack(features_sampled, dim=1) # b, l, p, num_heads*num_samples, c // num_heads
        features_sampled = (weights * features_sampled.view(b, l, p, self.num_heads, self.num_samples, -1)).sum(dim=-2).view(b, l, p, -1)
        
        x = residual + self.drop_path(features_sampled)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = torch.cat([x_0,x], dim=1) #[b, 5, 17, 128]
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim=128, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, q_tokens, kv_tokens):
        b, q_len, c = q_tokens.shape
        kv_len = kv_tokens.shape[1]

        q = self.q(q_tokens).view(b, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(kv_tokens).view(b, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(kv_tokens).view(b, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(b, q_len, c)
        return self.proj(out)


class LatentDepthStructureReasoner(nn.Module):
    """
    LDSR: learnable latent depth slots induce a sample-adaptive low-rank
    joint relation graph for calibrating AMS-refined depth tokens.
    """
    def __init__(self, num_joints=17, dim=128, num_slots=6, num_heads=4, slot_iters=2):
        super().__init__()
        self.num_joints = num_joints
        self.dim = dim
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.slot_iters = slot_iters

        self.depth_proj = nn.Linear(dim, dim)
        self.pose_proj = nn.Linear(dim, dim)
        self.rgb_proj = nn.Linear(dim, dim)
        self.norm_h = nn.LayerNorm(dim)

        self.slots = nn.Parameter(torch.randn(1, num_slots, dim) * 0.02)
        self.slot_attn = CrossAttention(dim=dim, num_heads=num_heads)
        self.slot_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

        self.joint_to_slot_q = nn.Linear(dim, dim)
        self.slot_k = nn.Linear(dim, dim)
        self.static_slot_relation = nn.Parameter(torch.randn(num_slots, num_slots) * 0.02)
        self.dynamic_slot_relation = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_slots * num_slots),
        )

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.graph_to_bias = nn.Linear(1, num_heads)

        self.depth_update = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.update_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

        self.rel_head = nn.Sequential(
            nn.Linear(dim * 2 + 1, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

        nn.init.zeros_(self.dynamic_slot_relation[-1].weight)
        nn.init.zeros_(self.dynamic_slot_relation[-1].bias)
        nn.init.zeros_(self.graph_to_bias.weight)
        nn.init.zeros_(self.graph_to_bias.bias)
        nn.init.constant_(self.update_gate[-2].bias, -2.0)

    def forward(self, x_depth, x_pose, x_rgb):
        b, j, c = x_depth.shape
        rgb_ctx = x_rgb.mean(dim=1)

        h = self.depth_proj(x_depth) + self.pose_proj(x_pose) + self.rgb_proj(rgb_ctx)
        h = self.norm_h(h)

        slots = self.slots.expand(b, -1, -1)
        for _ in range(self.slot_iters):
            slots = slots + self.slot_attn(slots, h)
            slots = slots + self.slot_mlp(slots)

        qj = self.joint_to_slot_q(h)
        ks = self.slot_k(slots)
        assign_logits = (qj @ ks.transpose(1, 2)) / math.sqrt(c)
        assign = assign_logits.softmax(dim=-1)  # [B,J,K]

        dynamic_m = self.dynamic_slot_relation(slots.mean(dim=1)).view(b, self.num_slots, self.num_slots)
        static_m = self.static_slot_relation.unsqueeze(0).expand(b, -1, -1)
        slot_relation = static_m + dynamic_m
        slot_relation = 0.5 * (slot_relation + slot_relation.transpose(1, 2))

        graph = assign @ slot_relation @ assign.transpose(1, 2)
        graph = graph - graph.mean(dim=-1, keepdim=True)

        head_dim = c // self.num_heads
        q = self.q(h).view(b, j, self.num_heads, head_dim).transpose(1, 2)
        k = self.k(h).view(b, j, self.num_heads, head_dim).transpose(1, 2)
        v = self.v(h).view(b, j, self.num_heads, head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        graph_bias = self.graph_to_bias(graph.unsqueeze(-1)).permute(0, 3, 1, 2)
        attn = (attn + graph_bias).softmax(dim=-1)

        msg = attn @ v
        msg = msg.transpose(1, 2).reshape(b, j, c)
        msg = self.out_proj(msg)

        gate = self.update_gate(torch.cat([h, msg], dim=-1))
        x_depth_out = x_depth + gate * self.depth_update(msg)

        slot_ctx = assign @ slots
        si = slot_ctx.unsqueeze(2).expand(-1, -1, j, -1)
        sj = slot_ctx.unsqueeze(1).expand(-1, j, -1, -1)
        rel_pred = self.rel_head(torch.cat([si - sj, si * sj, graph.unsqueeze(-1)], dim=-1)).squeeze(-1)
        rel_pred = torch.tanh(rel_pred + graph)

        return x_depth_out, rel_pred, graph


class DGLifting(nn.Module):
    def __init__(self, config=None, num_frame=1, num_joints=17, in_chans=2,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        # embed_dim_ratio = config.embed_dim_ratio
        # base_dim = config.base_dim
        # depth = config.depth
        embed_dim_ratio = 128
        base_dim = 32
        depth = 4
        out_dim = 3    #### output dimension is num_joints * 3
        self.levels = 5
        embed_dim = embed_dim_ratio * (self.levels+1)

        # self.depth_embed = nn.Linear(1, embed_dim_ratio)
        self.depth_embed = nn.Conv2d(in_channels=1, out_channels=embed_dim_ratio, kernel_size=3, padding=1)
        ### spatial patch embedding
        self.coord_embed = nn.Linear(in_chans, embed_dim_ratio)

        # rgbd features embedding
        self.feat_embed = nn.ModuleList([
            nn.Linear(base_dim, embed_dim_ratio), # x[b, 32,  64, 48]
            nn.Linear(base_dim * 2, embed_dim_ratio), # x[b, 64,  32, 24]
            nn.Linear(base_dim * 4, embed_dim_ratio), # x[b, 128, 16, 12]
            nn.Linear(base_dim * 8, embed_dim_ratio), # x[b, 256, 8,  6 ]
            nn.Linear(embed_dim_ratio, embed_dim_ratio),
            ])

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, 1+self.levels, num_joints, embed_dim_ratio))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.RGBD_Extraction = nn.ModuleList([
            DeformableBlock(dim=embed_dim_ratio, num_heads=4, num_samples=5, qkv_bias=qkv_bias, drop_path=dpr[i], levels=self.levels)
            for i in range(depth)])
        
        ldsr_cfg = getattr(config, "ldsr", None) if config is not None else None
        self.ldsr = LatentDepthStructureReasoner(
            num_joints=num_joints,
            dim=embed_dim_ratio,
            num_slots=getattr(ldsr_cfg, "num_slots", 6),
            num_heads=getattr(ldsr_cfg, "num_heads", 4),
            slot_iters=getattr(ldsr_cfg, "slot_iters", 2),
        )

        self.Features_Fusion = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        
        self.Spatial_Transformer = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )


    def forward(self, keypoints_2d, ref, depth_images, features_list_hr):
        b, p, c = keypoints_2d.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data

        x = self.coord_embed(keypoints_2d)

        # depth_images = self.depth_embed(depth_images.unsqueeze(-1)).permute(0, 3, 1, 2).contiguous() # [b, 256, 192, dim]
        depth_images = self.depth_embed(depth_images.unsqueeze(1)) # [b, dim, 256, 192]
        features_list_hr = list(features_list_hr) + [depth_images]

        # 根据 ref 提供的位置信息，在RGBD特征中指定位置进行采样，并提取相应的特征
        # [b, c, H, W]->[b, c, H_ref, W_ref]->[b, H_ref, c]
        features_ref_list = [
            F.grid_sample(features, ref.unsqueeze(-2), align_corners=True).squeeze(-1).permute(0, 2, 1).contiguous() \
            for features in features_list_hr]
        features_ref_list = [embed(features_ref_list[idx]) \
                             for idx, embed in enumerate(self.feat_embed)]

        # 关键点特征与参考特征拼接 [b, 6, 17, 128]
        x = torch.stack([x,*features_ref_list], dim=1) # [b, 6, p, c]

        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        
        for blk in self.RGBD_Extraction:
            x = blk(x, ref, features_list_hr)
        
        x_depth = x[:, -1]
        x_pose = x[:, 0]
        x_rgb = x[:, 1:-1]
        x_depth, rel_pred, depth_graph = self.ldsr(x_depth, x_pose, x_rgb)
        x = torch.cat((x[:, :-1], x_depth.unsqueeze(1)), dim=1)

        x = rearrange(x, 'b l p c -> (b p) l c')
        for blk in self.Features_Fusion:
            x = blk(x)
        
        x = rearrange(x, '(b p) l c -> b p (l c)', b=b)
        for blk in self.Spatial_Transformer:
            x = blk(x)

        x = self.head(x).view(b, 1, p, -1)
        return x, rel_pred, depth_graph
        # return x, pos, weights

# def _test():
#     torch.cuda.set_device(0)
#     # from torchprofile import profile_macs
#     import warnings
#     warnings.filterwarnings('ignore')
#     b, c, t, j = 1, 3, 243, 17
#     keypoints_2d_cpn = torch.randn((b, j, 2)).to('cuda')
#     keypoints_2d_cpn_crop = torch.randn((b, j, 2)).to('cuda')
#     depth_images = torch.randn((b, 256, 192)).to('cuda')
#     features_list_hr = [torch.randn((b, 32,  64, 48)).to('cuda'),torch.randn((b, 64,  32, 24)).to('cuda'),torch.randn((b, 128, 16, 12)).to('cuda'),torch.randn((b, 256, 8,  6)).to('cuda')
# ]
#     model = DGLifting().to('cuda')
#     model.eval()

#     model_params = 0
#     for parameter in model.parameters():
#         model_params = model_params + parameter.numel()
#     print(f"Model parameter #: {model_params:,}"
#     out = model(keypoints_2d_cpn, keypoints_2d_cpn_crop, depth_images, features_list_hr)


# if __name__ == '__main__':
#     _test()
