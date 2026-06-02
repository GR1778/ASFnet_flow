"""DOGA: Depth Ordering Graph Attention.

Replaces the UDE (Uncertainty-aware Depth Enhancement) module in ASFnet by
reformulating the problem: monocular depth estimators output affine-invariant
*relative* depth, not absolute values. DOGA learns a pairwise joint-depth
ordering matrix instead of regressing absolute depth values, yielding a
scale-free and domain-robust depth signal that is naturally more rigorous
toward the real output semantics of Depth-Anything V2.

Components
----------
- PairwiseOrderingPredictor (PwOP): F_d [B, J, C] -> R [B, J, J]
- SinkhornSoftSort:                  R [B, J, J] -> P [B, J, J] (soft permutation)
- RankPositionEmbedding (RAPE):      P -> per-joint rank embedding [B, J, C]
- OrderingBiasedAttention (OBCA):    F_d [B, J, C] + R bias -> refined [B, J, C]
- doga_ranking_loss / DOGARankingLoss: scale-free pairwise BCE over R vs sign(z_gt)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_H36M_BONES = (
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
)


def _build_h36m_adjacency(num_joints: int = 17) -> torch.Tensor:
    """Return a symmetric binary skeleton adjacency matrix [J, J].

    Uses the standard Human3.6M 17-joint topology. Self-edges are 0.
    """
    adj = torch.zeros(num_joints, num_joints)
    for i, j in _H36M_BONES:
        if i < num_joints and j < num_joints:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    return adj


class PairwiseOrderingPredictor(nn.Module):
    """Learn a pairwise joint-depth ordering matrix ``R`` in ``[B, J, J]``.

    Core input is the antisymmetric feature difference ``x_i - x_j``; the MLP
    is nonlinear so ``R = (R - R^T) / 2`` enforces exact antisymmetry. Optional
    **geometric priors** (2D relative position and skeleton adjacency) make
    the ordering head much more sample-efficient: 2D joint distance is a
    strong prior for which pairs have a reliable depth order, and the
    adjacency indicator lets directly-connected joints (shoulder-elbow,
    hip-knee, ...) get a dedicated learnable channel.

    Output convention
    -----------------
    The returned ``R`` is an **unbounded antisymmetric logit tensor**
    (``R_ii = 0``, ``R_ij = -R_ji``). It is consumed in two roles:
      1. As a raw logit by ``doga_ranking_loss`` (BCEWithLogits), so the
         loss can drive ``sigmoid(R_ij)`` to the full [0, 1] range.
      2. As an additive attention bias inside ``OrderingBiasedAttention``,
         where the learnable per-head ``lambda_R`` absorbs the scale.
    We intentionally do **not** squash ``R`` with ``tanh`` here, since that
    would bound ``sigmoid(R)`` inside ``[0.27, 0.73]`` and cap the BCE
    minimum around 0.31 -- a prior bug that limited ranking convergence.
    """

    def __init__(
        self,
        dim: int,
        hidden: int | None = None,
        use_geom: bool = True,
        num_joints: int = 17,
    ):
        super().__init__()
        hidden = hidden or max(dim // 2, 1)
        self.use_geom = use_geom
        self.num_joints = num_joints

        in_dim = dim + (3 if use_geom else 0)      # + (pos_diff_x, pos_diff_y, adj)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        if use_geom:
            adj = _build_h36m_adjacency(num_joints)
            self.register_buffer("adj", adj, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        pos_2d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, J, C = x.shape
        diff = x.unsqueeze(2) - x.unsqueeze(1)       # [B, J, J, C]
        feats = [diff]

        if self.use_geom:
            if pos_2d is not None:
                pos_diff = pos_2d.unsqueeze(2) - pos_2d.unsqueeze(1)   # [B, J, J, 2]
            else:
                pos_diff = torch.zeros(B, J, J, 2, device=x.device, dtype=x.dtype)
            adj = self.adj.to(dtype=x.dtype, device=x.device)
            adj = adj.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, 1)  # [B, J, J, 1]
            feats.extend([pos_diff, adj])

        inp = torch.cat(feats, dim=-1)
        raw = self.mlp(inp).squeeze(-1)              # [B, J, J]
        R = 0.5 * (raw - raw.transpose(-1, -2))      # antisymmetric, unbounded logit
        return R


class SinkhornSoftSort(nn.Module):
    """Differentiable soft-sort over per-joint rank logits via Sinkhorn.

    Takes ``R`` and produces a doubly-stochastic soft permutation ``P`` where
    ``P[i, k]`` approximates the probability that joint ``i`` is ranked at
    position ``k`` along the depth axis.
    """

    def __init__(self, n_iter: int = 20):
        super().__init__()
        self.n_iter = n_iter

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        B, J, _ = R.shape
        rank_logit = torch.sigmoid(R).sum(dim=-1)          # [B, J]
        target_rank = torch.arange(
            1, J + 1, device=R.device, dtype=rank_logit.dtype
        ).view(1, 1, J)
        cost = -(rank_logit.unsqueeze(-1) - target_rank).abs()
        log_P = cost
        for _ in range(self.n_iter):
            log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
            log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)
        return log_P.exp()


class RankPositionEmbedding(nn.Module):
    """Per-joint rank position embedding weighted by soft permutation ``P``."""

    def __init__(self, num_joints: int, dim: int):
        super().__init__()
        self.rank_embed = nn.Embedding(num_joints, dim)
        nn.init.normal_(self.rank_embed.weight, std=0.02)

    def forward(self, P: torch.Tensor) -> torch.Tensor:
        E = self.rank_embed.weight                         # [J, C]
        return torch.matmul(P, E.unsqueeze(0))             # [B, J, C]


class OrderingBiasedAttention(nn.Module):
    """Standard multi-head self-attention with an additive ordering bias.

    The bias term ``lambda_R * R`` (``lambda_R`` learnable) is added to the
    attention logits before softmax, so joints with similar depth attend to
    each other more strongly and joints across a depth discontinuity are
    naturally decoupled.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # Per-head learnable ordering-bias scale: some heads can choose to attend
        # to the ordering signal strongly while others can ignore it (lambda~0).
        self.lambda_R = nn.Parameter(torch.ones(num_heads))

    def forward(self, x: torch.Tensor, R: torch.Tensor | None = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if R is not None:
            # R: [B, J, J]; lambda_R: [H]; broadcast to [B, H, J, J]
            attn = attn + self.lambda_R.view(1, -1, 1, 1) * R.unsqueeze(1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))
        return out


class DOGABlock(nn.Module):
    """One Transformer-style block with Ordering-Biased Attention + FFN.

    Pre-LN residual structure. Stacking ``num_blocks`` of these gives DOGA a
    total capacity comparable to UDE's original 384-dim ``attn_depth`` mixer
    (~500K params), so that DOGA vs UDE is a fair test of the *ordering* story
    rather than being confounded with module capacity.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = OrderingBiasedAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, R: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), R=R)
        x = x + self.mlp(self.norm2(x))
        return x


class DOGA(nn.Module):
    """Full DOGA module: PwOP -> (SoftSort + RAPE)? -> (OBCA Transformer stack)? -> (gamma_head)?.

    Drop-in replacement for the UDE pipeline in ``DGLifting``. The OBCA path
    is a stack of ``DOGABlock``s; stacking depth + MLP ratio control total
    capacity so ablation cleanly separates the *ordering* story from raw
    model capacity. An optional auxiliary absolute-depth head (gamma) keeps
    a coarse-depth regression signal as a safety-net auxiliary supervision
    without contaminating the ordering main path.

    Args:
        dim: token dim
        num_joints: number of joints (17 for Human3.6M)
        num_heads: OBCA heads
        use_sinkhorn / use_obca / use_rape: sub-component switches
        sinkhorn_iter: number of Sinkhorn normalization steps
        num_blocks: depth of OBCA transformer stack
        mlp_ratio: FFN expansion in each DOGABlock
        use_geom_prior: inject 2D position + skeleton adjacency into PwOP
        use_aux_abs: add an auxiliary absolute-depth regression head (gamma)
    """

    def __init__(
        self,
        dim: int,
        num_joints: int = 17,
        num_heads: int = 8,
        use_sinkhorn: bool = True,
        use_obca: bool = True,
        use_rape: bool = True,
        sinkhorn_iter: int = 20,
        num_blocks: int = 2,
        mlp_ratio: float = 4.0,
        use_geom_prior: bool = True,
        use_aux_abs: bool = True,
    ):
        super().__init__()
        self.use_sinkhorn = use_sinkhorn
        self.use_obca = use_obca
        self.use_rape = use_rape
        self.use_aux_abs = use_aux_abs
        self.num_blocks = num_blocks

        self.pwop = PairwiseOrderingPredictor(
            dim, use_geom=use_geom_prior, num_joints=num_joints
        )

        if use_rape:
            if use_sinkhorn:
                self.sorter = SinkhornSoftSort(n_iter=sinkhorn_iter)
            self.rape = RankPositionEmbedding(num_joints, dim)
            self.rape_norm = nn.LayerNorm(dim)

        if use_obca:
            self.blocks = nn.ModuleList([
                DOGABlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
                for _ in range(num_blocks)
            ])

        self.out_norm = nn.LayerNorm(dim)

        if use_aux_abs:
            hidden = max(dim // 2, 1)
            self.gamma_head = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )

    def forward(
        self,
        x: torch.Tensor,
        pos_2d: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        R = self.pwop(x, pos_2d=pos_2d)

        if self.use_rape:
            if self.use_sinkhorn:
                P = self.sorter(R)
            else:
                rank_logit = torch.sigmoid(R).sum(dim=-1, keepdim=True)
                P = F.softmax(rank_logit, dim=1)
                P = P.expand(-1, -1, R.shape[-1])
            x = self.rape_norm(x + self.rape(P))

        if self.use_obca:
            for blk in self.blocks:
                x = blk(x, R=R)

        x = self.out_norm(x)
        gamma = self.gamma_head(x) if self.use_aux_abs else None
        return x, R, gamma


def doga_ranking_loss(
    R: torch.Tensor,
    z_gt: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Pairwise depth-ordering BCE loss (scale-free).

    Parameters
    ----------
    R : torch.Tensor
        Predicted ordering logits, shape ``[B, J, J]``. ``R_ij > 0`` means
        joint ``i`` has larger ground-truth z than joint ``j`` (the absolute
        front/back meaning depends on the dataset's z-axis convention).
    z_gt : torch.Tensor
        Root-relative ground-truth depth values, shape ``[B, J, 1]`` or ``[B, J]``.
    margin : float
        Minimum absolute depth gap for a pair to **participate** in the loss.
        Pairs with ``|z_i - z_j| <= margin`` are **ignored** (not labelled as
        0), avoiding noisy supervision on near-tie depths. ``margin=0`` keeps
        all off-diagonal pairs.
    """
    if z_gt.dim() == 3:
        z_gt = z_gt.squeeze(-1)
    B, J = z_gt.shape
    target_diff = z_gt.unsqueeze(-1) - z_gt.unsqueeze(-2)       # [B, J, J]
    eye = torch.eye(J, dtype=torch.bool, device=R.device).unsqueeze(0)
    # Valid pair = off-diagonal AND absolute depth gap above margin.
    # When margin=0, this equals "off-diagonal with non-zero diff" and
    # degenerates to the previous sign-based behaviour.
    valid = (target_diff.abs() > margin) & ~eye                # [B, J, J]
    if not valid.any():
        # Degenerate batch with all near-tie depths; keep grad path alive.
        return R.sum() * 0.0
    logits = R[valid]
    labels = (target_diff > 0).float()[valid]
    return F.binary_cross_entropy_with_logits(logits, labels)


def abs_depth_aux_loss(gamma: torch.Tensor, z_gt: torch.Tensor) -> torch.Tensor:
    """Smooth-L1 absolute-depth auxiliary loss over per-joint coarse depth.

    Serves as a safety-net supervision: the main DOGA signal is the
    scale-free ordering matrix R; gamma retains a coarse absolute-depth
    estimate so the network doesn't lose the absolute-localization cue that
    UDE was providing.
    """
    if gamma.dim() == 3 and gamma.shape[-1] == 1:
        gamma = gamma.squeeze(-1)
    if z_gt.dim() == 3 and z_gt.shape[-1] == 1:
        z_gt = z_gt.squeeze(-1)
    return F.smooth_l1_loss(gamma, z_gt)


class DOGARankingLoss(nn.Module):
    """``nn.Module`` wrapper that keeps the ``train.py`` call-site unchanged.

    Signature mirrors ``BNNLoss.forward(keypoints_pred, keypoints_gt, s)``
    but the semantics are:
      1st arg (R)    : pairwise ordering logits [B, J, J]
      2nd arg (z_gt) : root-relative GT depth   [B, J, 1] or [B, J]
      3rd arg (gamma): optional auxiliary absolute-depth prediction [B, J, 1]
    When ``gamma`` is provided, a smooth-L1 auxiliary term is added with
    weight ``lambda_abs``.

    Magnitude matching
    ------------------
    The original ``BNNLoss`` scales ``diff`` by 100 internally, producing an
    output in the 50-500 range, which the ``loss_d * 1e-5`` weighting at the
    train.py call-site reduces to ~1e-3 (comparable to the main MPJPE loss).
    Our ranking BCE has a natural magnitude of ~0.3-1.5, which is ~100x
    smaller. We therefore multiply the final ranking-loss output by
    ``scale=100`` so the same ``* 1e-5`` weight at the call-site yields the
    same effective gradient magnitude -- **no user-facing hyperparameter
    tuning is required**, the scale is derived directly from BNNLoss.
    """

    def __init__(self, lambda_abs: float = 0.1, scale: float = 100.0):
        super().__init__()
        self.lambda_abs = lambda_abs
        self.scale = scale

    def forward(
        self,
        R: torch.Tensor,
        z_gt: torch.Tensor,
        gamma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = doga_ranking_loss(R, z_gt)
        if gamma is not None and self.lambda_abs > 0:
            loss = loss + self.lambda_abs * abs_depth_aux_loss(gamma, z_gt)
        return loss * self.scale


if __name__ == "__main__":
    torch.manual_seed(0)
    B, J, C = 4, 17, 128
    x = torch.randn(B, J, C, requires_grad=True)
    pos_2d = torch.randn(B, J, 2)

    pwop = PairwiseOrderingPredictor(C, use_geom=True)
    R = pwop(x, pos_2d=pos_2d)
    assert R.shape == (B, J, J)
    anti_err = (R + R.transpose(-1, -2)).abs().max().item()
    print(f"[PwOP+geom] anti-symmetry err = {anti_err:.2e}")
    assert anti_err < 1e-5

    sorter = SinkhornSoftSort(n_iter=20)
    P = sorter(R)
    row_err = (P.sum(-1) - 1).abs().max().item()
    col_err = (P.sum(-2) - 1).abs().max().item()
    print(f"[Sinkhorn] row-sum err = {row_err:.2e}, col-sum err = {col_err:.2e}")

    doga = DOGA(C, num_joints=J)
    out, R_full, gamma = doga(x, pos_2d=pos_2d)
    assert out.shape == (B, J, C) and R_full.shape == (B, J, J)
    assert gamma is not None and gamma.shape == (B, J, 1)

    z_gt = torch.randn(B, J, 1)
    rank_loss = doga_ranking_loss(R_full, z_gt)
    abs_loss = abs_depth_aux_loss(gamma, z_gt)
    total = rank_loss + 0.1 * abs_loss
    total.backward()
    print(
        f"[Loss] ranking BCE = {rank_loss.item():.4f}, "
        f"abs L1 = {abs_loss.item():.4f}, "
        f"grad norm = {x.grad.norm().item():.4f}"
    )

    # Sanity check for the tanh-bug fix: a "perfect" R that exactly matches
    # sign(z_gt - z_gt^T) with large magnitude should drive BCE -> 0.
    diff = z_gt.squeeze(-1).unsqueeze(-1) - z_gt.squeeze(-1).unsqueeze(-2)
    R_perfect = torch.sign(diff) * 10.0          # unbounded logit, big magnitude
    bce_lb = doga_ranking_loss(R_perfect, z_gt).item()
    print(f"[BCE-lb] loss under perfect R (|logit|=10) = {bce_lb:.4e}  (should be near 0)")
    assert bce_lb < 1e-3, (
        f"Perfect-R BCE should be near 0 but got {bce_lb}; tanh fix may be broken."
    )

    # Margin semantics check: with a huge margin all pairs should be filtered.
    huge_margin_loss = doga_ranking_loss(R_full, z_gt, margin=1e6).item()
    print(f"[Margin] loss with margin=1e6 = {huge_margin_loss:.4e}  (should be exactly 0)")
    assert huge_margin_loss == 0.0

    n_params = sum(p.numel() for p in doga.parameters())
    print(f"[Params] DOGA (full + aux) = {n_params:,} ({n_params/1e6:.3f}M)")

    # Per-head lambda_R sanity
    for name, p in doga.named_parameters():
        if name.endswith("lambda_R"):
            print(f"[lambda_R] {name}: shape {tuple(p.shape)}")
            break

    print("All DOGA unit tests passed.")
