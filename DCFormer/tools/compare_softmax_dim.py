"""
对比实验：UDE 中 UMap softmax 方向
  - dim=1 (当前代码，跨关节归一化)
  - dim=-1 (论文描述，关节内特征归一化)

使用同一份官方权重，同一批输入，对比两种实现的输出差异。
结果说明：
  - 如果输出差异很大 → 两种实现行为不同，dim=1 是模型真正学到的
  - 如果输出差异很小 → softmax 方向对结果影响有限
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from einops import rearrange
from timm.models.layers import DropPath
from mvn.models.DGLifting import DGLifting, Mlp, Attention, DepthUncertaintyModel


# ── 补丁版 DGLifting：只改 softmax dim ──────────────────────────────────────
class DGLiftingDimMinus1(DGLifting):
    """把 UMap 的 softmax 从 dim=1 改为 dim=-1（论文描述方式）"""
    def forward(self, keypoints_2d, ref, depth_images, features_list_hr):
        b, p, c = keypoints_2d.shape

        x = self.coord_embed(keypoints_2d)
        depth_images = self.depth_embed(depth_images.unsqueeze(1))
        features_list_hr.append(depth_images)

        features_ref_list = [
            F.grid_sample(features, ref.unsqueeze(-2), align_corners=True)
              .squeeze(-1).permute(0, 2, 1).contiguous()
            for features in features_list_hr
        ]
        features_ref_list = [
            embed(features_ref_list[idx])
            for idx, embed in enumerate(self.feat_embed)
        ]

        x = torch.stack([x, *features_ref_list], dim=1)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        for blk in self.RGBD_Extraction:
            x = blk(x, ref, features_list_hr)

        x_depth = x[:, -1]
        coarse_depth, uncer = self.depth_uncer(x_depth)

        z_value = self.z_embed(coarse_depth) + self.Spatial_pos_embed2
        # ← 唯一改动：dim=-1 代替 dim=1
        joint_uncer = F.softmax(self.attn_fc(uncer), dim=-1)

        x_depth = torch.cat([joint_uncer, z_value, x_depth], dim=-1)
        x_depth = self.attn_depth(x_depth)
        x = torch.cat((x[:, :-1], x_depth.unsqueeze(1)), dim=1)

        x = rearrange(x, 'b l p c -> (b p) l c')
        for blk in self.Features_Fusion:
            x = blk(x)

        x = rearrange(x, '(b p) l c -> b p (l c)', b=b)
        for blk in self.Spatial_Transformer:
            x = blk(x)

        x = self.head(x).view(b, 1, p, -1)
        return x, coarse_depth, uncer


def load_lifting_net(model_cls, checkpoint_path: str, device: torch.device):
    """只加载 Lifting_net 部分的权重"""
    from mvn.utils.cfg import config, update_config
    update_config('experiments/human36m/human36m_single.yaml')

    net = model_cls(config.model.poseformer).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt['model'] if 'model' in ckpt else ckpt

    # 去掉分布式训练的 module. 前缀，只取 Lifting_net 部分
    lifting_state = {}
    for k, v in state.items():
        k = k.replace('module.', '')
        if k.startswith('Lifting_net.'):
            lifting_state[k[len('Lifting_net.'):]] = v

    ret = net.load_state_dict(lifting_state, strict=True)
    print(f"[{model_cls.__name__}] 权重加载: {ret}")
    net.eval()
    return net


def make_dummy_batch(b=4, j=17, device='cuda'):
    """生成形状与真实数据一致的随机输入"""
    torch.manual_seed(42)
    kp2d      = torch.randn(b, j, 2, device=device)
    kp2d_crop = torch.randn(b, j, 2, device=device)
    depth_img = torch.rand(b, 256, 192, device=device)  # 归一化深度图
    feats = [
        torch.randn(b, 32,  64, 48, device=device),
        torch.randn(b, 64,  32, 24, device=device),
        torch.randn(b, 128, 16, 12, device=device),
        torch.randn(b, 256,  8,  6, device=device),
    ]
    return kp2d, kp2d_crop, depth_img, feats


def mpjpe(pred, target):
    """Root-aligned MPJPE (mm)"""
    pred   = pred   - pred[:, :, 0:1, :]
    target = target - target[:, :, 0:1, :]
    return torch.mean(torch.norm(pred - target, dim=-1)).item() * 1000


@torch.no_grad()
def run_comparison(ckpt_path: str, device_str='cuda'):
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载两种模型（权重完全相同）
    print("\n── 加载 dim=1 模型 ──")
    net_dim1 = load_lifting_net(DGLifting, ckpt_path, device)

    print("\n── 加载 dim=-1 模型 ──")
    net_dim_m1 = load_lifting_net(DGLiftingDimMinus1, ckpt_path, device)

    # 确认权重完全一样
    for (n1, p1), (n2, p2) in zip(net_dim1.named_parameters(), net_dim_m1.named_parameters()):
        assert torch.allclose(p1, p2), f"权重不匹配: {n1}"
    print("\n✓ 两个模型权重完全相同")

    # 多批次对比
    print("\n── 开始推理对比 ──")
    all_diff_mm = []
    n_batches = 10

    for i in range(n_batches):
        torch.manual_seed(i)
        kp2d, kp2d_crop, depth_img, feats = make_dummy_batch(b=8, device=str(device))
        # DGLifting 会修改 feats list（append），所以每次要复制
        feats1 = [f.clone() for f in feats]
        feats2 = [f.clone() for f in feats]

        out1, mu1, s1 = net_dim1(kp2d.clone(), kp2d_crop.clone(), depth_img.clone(), feats1)
        out2, mu2, s2 = net_dim_m1(kp2d.clone(), kp2d_crop.clone(), depth_img.clone(), feats2)

        # 输出形状: [B, 1, J, 3]
        diff = torch.mean(torch.norm(out1 - out2, dim=-1)).item() * 1000  # mm
        all_diff_mm.append(diff)

        # s 的差异（joint_uncer 本身的差异）
        s_diff = torch.mean(torch.abs(s1 - s2)).item()

    print(f"\n{'='*55}")
    print(f"  softmax dim=1  vs  dim=-1  输出差异统计（基于随机输入）")
    print(f"{'='*55}")
    print(f"  各 batch 输出差异 (mm): {[f'{v:.3f}' for v in all_diff_mm]}")
    print(f"  均值差异: {np.mean(all_diff_mm):.3f} mm")
    print(f"  最大差异: {np.max(all_diff_mm):.3f} mm")
    print(f"  最后 batch s 值均差: {s_diff:.6f}")
    print(f"{'='*55}")

    mean_diff = np.mean(all_diff_mm)
    if mean_diff < 0.1:
        verdict = "⬤ 差异极小(<0.1mm)：softmax 方向对输出影响可忽略"
    elif mean_diff < 1.0:
        verdict = "⬤ 差异较小(<1mm)：softmax 方向有影响但幅度有限"
    elif mean_diff < 5.0:
        verdict = "⬤ 差异明显(1~5mm)：两种实现行为显著不同"
    else:
        verdict = "⬤ 差异很大(>5mm)：两种实现行为完全不同"

    print(f"\n  结论: {verdict}")
    print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoint/h36m_v2b.bin')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    run_comparison(args.checkpoint, args.device)
