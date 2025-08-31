# hsimamba_sr.py
# PyTorch ≥ 2.0
# A tri-axial non-causal 3D-Mamba super-resolution backbone for hyperspectral images.
# Author: (you) — ICASSP-ready minimal & clean implementation
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================== Utils ==============================

def pixel_shuffle_3d_spatial(x: torch.Tensor, r: int) -> torch.Tensor:
    """
    Spatial pixel-shuffle for 5D tensors that preserves spectral depth D.
    x: (B, C*r*r, D, H, W) -> (B, C, D, H*r, W*r)
    """
    b, c, d, h, w = x.shape
    assert c % (r * r) == 0, f"Channels must be divisible by r^2 (got {c}, r={r})."
    c_out = c // (r * r)
    x = x.view(b, c_out, r, r, d, h, w)          # (B, C, r, r, D, H, W)
    x = x.permute(0, 1, 4, 5, 2, 6, 3).contiguous()  # (B, C, D, H, r, W, r)
    x = x.reshape(b, c_out, d, h * r, w * r)     # (B, C, D, H*r, W*r)
    return x


class DropPath(nn.Module):
    """Stochastic depth per sample."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x / keep * mask


class LayerNormLast(nn.Module):
    """LayerNorm over the last dim (channels-last 5D: B,D,H,W,C)."""
    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(c, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


# ============================== Axis SSM (Mamba or GRU fallback) ==============================

class _GRUSSM(nn.Module):
    """GRU fallback: drop-in sequence model (N, L, C) -> (N, L, C)."""
    def __init__(self, d_model: int):
        super().__init__()
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(seq)  # (N, L, C)
        return y


def _make_axis_ssm(backend: Literal["mamba", "gru"], d_model: int, d_state: int, expand: int) -> nn.Module:
    if backend == "mamba":
        try:
            from mamba_ssm import Mamba
        except Exception as e:
            raise ImportError(
                "Please install `mamba-ssm` for backend='mamba' (pip install mamba-ssm). "
                "Alternatively, set backend='gru' for a lightweight fallback."
            ) from e
        return Mamba(d_model=d_model, d_state=d_state, expand=expand)
    elif backend == "gru":
        return _GRUSSM(d_model)
    else:
        raise ValueError("backend must be 'mamba' or 'gru'")


# ============================== Tri-Axial Mamba Block ==============================

class TriAxialMambaBlock(nn.Module):
    r"""
    Tri-Axial, non-causal 3D-Mamba block (channels-last inside the block).

    Input  x_cl: (B, D, H, W, C)
    Output y_cl: (B, D, H, W, C)

    Stages:
      (1) Pre-Norm -> Mamba on each axis (H, W, D), each bi-directional -> Axis Fusion -> Residual (+DropPath)
      (2) Pre-Norm -> Local enhancement DW-3D(H/W only) + 1x1 -> Residual (+DropPath)

    Args:
      dim:        feature channels C
      d_state:    Mamba state size
      expand:     Mamba expansion factor
      drop_path:  drop-path rate for residuals
      axis_fusion: 'gated' (3 scalar gates) or 'linear' (1x1 on concat(H,W,D))
      backend:    'mamba' (recommended) or 'gru' (fallback/sanity)
    """
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        expand: int = 2,
        drop_path: float = 0.0,
        axis_fusion: Literal["gated", "linear"] = "gated",
        backend: Literal["mamba", "gru"] = "mamba",
        use_activation: bool = True,
    ):
        super().__init__()
        assert axis_fusion in {"gated", "linear"}

        self.dim = dim
        self.axis_fusion = axis_fusion

        # Norms
        self.norm1 = LayerNormLast(dim)
        self.norm2 = LayerNormLast(dim)

        # SSMs per axis (shared for forward/back on that axis)
        self.m_h = _make_axis_ssm(backend, dim, d_state, expand)
        self.m_w = _make_axis_ssm(backend, dim, d_state, expand)
        self.m_d = _make_axis_ssm(backend, dim, d_state, expand)

        # Non-causal blend per axis
        self.gamma_fwd = nn.Parameter(torch.zeros(3))  # [H, W, D]

        # Axis fusion
        if axis_fusion == "gated":
            self.axis_logits = nn.Parameter(torch.zeros(3))      # softmax over [H,W,D]
        else:
            self.fuse_linear = nn.Linear(dim * 3, dim)

        # Local enhancement: depth-wise 3D over H/W only + 1x1
        self.dw_spatial = nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=dim, bias=True)
        self.pw_mix = nn.Conv3d(dim, dim, kernel_size=1, bias=True)
        self.act = nn.GELU() if use_activation else nn.Identity()

        # Residual drops
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)

    # ---- run SSM along one axis with bidirectional blending ----
    def _run_axis(self, x_cl: torch.Tensor, axis: str) -> torch.Tensor:
        """
        x_cl: (B, D, H, W, C)
        axis: 'H' | 'W' | 'D'
        return: (B, D, H, W, C)
        """
        B, D, H, W, C = x_cl.shape

        if axis == "H":
            # sequences length H; batch N = B*W*D
            x_perm = x_cl.permute(0, 3, 1, 2, 4).contiguous()  # (B, W, D, H, C)
            seq = x_perm.view(B * W * D, H, C)
            m = self.m_h
            restore = lambda out: out.view(B, W, D, H, C).permute(0, 2, 3, 1, 4).contiguous()
        elif axis == "W":
            # sequences length W; batch N = B*H*D
            x_perm = x_cl.permute(0, 2, 1, 3, 4).contiguous()  # (B, H, D, W, C)
            seq = x_perm.view(B * H * D, W, C)
            m = self.m_w
            restore = lambda out: out.view(B, H, D, W, C).permute(0, 2, 1, 3, 4).contiguous()
        elif axis == "D":
            # sequences length D; batch N = B*H*W
            x_perm = x_cl.permute(0, 2, 3, 1, 4).contiguous()  # (B, H, W, D, C)
            seq = x_perm.view(B * H * W, D, C)
            m = self.m_d
            restore = lambda out: out.view(B, H, W, D, C).permute(0, 3, 1, 2, 4).contiguous()
        else:
            raise ValueError("axis must be 'H', 'W', or 'D'.")

        out_fwd = m(seq)                           # (N, L, C)
        out_bwd = m(torch.flip(seq, dims=[1]))     # reverse
        out_bwd = torch.flip(out_bwd, dims=[1])    # reverse back

        idx = {"H": 0, "W": 1, "D": 2}[axis]
        gamma = torch.sigmoid(self.gamma_fwd[idx])  # scalar (0,1)
        out = gamma * out_fwd + (1.0 - gamma) * out_bwd

        return restore(out)  # (B, D, H, W, C)

    def forward(self, x_cl: torch.Tensor) -> torch.Tensor:
        """
        x_cl: (B, D, H, W, C)
        """
        assert x_cl.ndim == 5 and x_cl.shape[-1] == self.dim, \
            f"Expected (B,D,H,W,C) with C={self.dim}, got {tuple(x_cl.shape)}."

        # (1) Tri-axial Mamba + fusion + residual
        y = self.norm1(x_cl)
        y_h = self._run_axis(y, "H")
        y_w = self._run_axis(y, "W")
        y_d = self._run_axis(y, "D")

        if self.axis_fusion == "gated":
            w = torch.softmax(self.axis_logits, dim=0)  # (3,)
            y_mix = w[0] * y_h + w[1] * y_w + w[2] * y_d
        else:
            y_cat = torch.cat([y_h, y_w, y_d], dim=-1)
            y_mix = self.fuse_linear(y_cat)

        out = x_cl + self.drop_path1(y_mix)

        # (2) Local enhancement + residual
        z = self.norm2(out)
        z_cf = z.permute(0, 4, 1, 2, 3).contiguous()     # (B, C, D, H, W)
        z_loc = self.dw_spatial(z_cf)
        z_loc = self.act(z_loc)
        z_loc = self.pw_mix(z_loc)
        z_loc = z_loc.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, C)

        out = out + self.drop_path2(z_loc)
        return out


# ============================== Residual Group ==============================

class TriMambaGroup(nn.Module):
    """
    Residual-in-Residual group that stacks multiple TriAxialMambaBlocks (channels-last inside),
    then applies a linear projection and adds a scaled residual.
    """
    def __init__(
        self,
        dim: int,
        depth: int,
        d_state: int = 16,
        expand: int = 2,
        drop_path_rates: Optional[List[float]] = None,
        axis_fusion: Literal["gated", "linear"] = "gated",
        backend: Literal["mamba", "gru"] = "mamba",
        res_scale: float = 1.0,
    ):
        super().__init__()
        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth
        assert len(drop_path_rates) == depth

        self.blocks = nn.ModuleList([
            TriAxialMambaBlock(
                dim=dim,
                d_state=d_state,
                expand=expand,
                drop_path=drop_path_rates[i],
                axis_fusion=axis_fusion,
                backend=backend,
            ) for i in range(depth)
        ])
        self.proj = nn.Linear(dim, dim)
        self.res_scale = res_scale

    def forward(self, x_cl: torch.Tensor) -> torch.Tensor:
        residual = x_cl
        for blk in self.blocks:
            x_cl = blk(x_cl)
        x_cl = self.proj(x_cl)
        return residual + self.res_scale * x_cl


# ============================== Full SR Model ==============================

@dataclass
class HSIMambaSRConfig:
    bands: int
    scale: int = 4
    embed_dim: int = 64
    n_groups: int = 4
    depth_per_group: int = 4
    d_state: int = 16
    expand: int = 2
    axis_fusion: Literal["gated", "linear"] = "gated"
    backend: Literal["mamba", "gru"] = "mamba"
    drop_path: float = 0.1   # final rate (linearly scaled across blocks)
    res_scale: float = 1.0


class HSIMambaSR(nn.Module):
    """
    Hyperspectral SR with tri-axial, non-causal 3D-Mamba blocks.
    Spatial-only upsampling (bands preserved).

    Input:  (B, D, H, W)
    Output: (B, D, sH, sW)
    """
    def __init__(self, cfg: HSIMambaSRConfig):
        super().__init__()
        self.cfg = cfg
        C = cfg.embed_dim
        r = cfg.scale

        # Head: avoid early band mixing (kernel only on H,W)
        self.head = nn.Conv3d(1, C, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # DropPath schedule across all blocks
        total_blocks = cfg.n_groups * cfg.depth_per_group
        dpr = torch.linspace(0, cfg.drop_path, total_blocks).tolist() if cfg.drop_path > 0 else [0.0]*total_blocks

        # Body (channels-last inside groups)
        self.groups = nn.ModuleList()
        idx = 0
        for _ in range(cfg.n_groups):
            rates = dpr[idx: idx + cfg.depth_per_group]
            self.groups.append(
                TriMambaGroup(
                    dim=C,
                    depth=cfg.depth_per_group,
                    d_state=cfg.d_state,
                    expand=cfg.expand,
                    drop_path_rates=rates,
                    axis_fusion=cfg.axis_fusion,
                    backend=cfg.backend,
                    res_scale=cfg.res_scale,
                )
            )
            idx += cfg.depth_per_group

        self.body_conv = nn.Conv3d(C, C, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # Tail: upsample H,W only and restore to 1 channel per band
        self.up = nn.Conv3d(C, C * (r * r), kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.tail = nn.Conv3d(C, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D, H, W), numeric range can be [0,1] or [-1,1], but must be consistent with training.
        """
        B, D, H, W = x.shape
        assert D == self.cfg.bands, f"Expected bands={self.cfg.bands}, got {D}."

        # Bicubic skip (per band)
        r = self.cfg.scale
        skip = F.interpolate(
            x.view(B * D, 1, H, W), scale_factor=r, mode='bicubic', align_corners=False
        ).view(B, D, r * H, r * W)

        # 3D convs treat bands as 'depth': (B,1,D,H,W)
        feat = x.unsqueeze(1).contiguous()          # (B,1,D,H,W)
        feat = self.head(feat)                      # (B,C,D,H,W)

        # To channels-last for Mamba groups
        feat = feat.permute(0, 2, 3, 4, 1).contiguous()  # (B,D,H,W,C)
        for g in self.groups:
            feat = g(feat)
        feat = feat.permute(0, 4, 1, 2, 3).contiguous()  # (B,C,D,H,W)

        feat = self.body_conv(feat)                # (B,C,D,H,W)

        # Upsample H,W only
        feat = self.up(feat)                       # (B,C*r^2,D,H,W)
        feat = pixel_shuffle_3d_spatial(feat, r)   # (B,C,D,rH,rW)
        out  = self.tail(feat)                     # (B,1,D,rH,rW)
        out  = out.squeeze(1)                      # (B,D,rH,rW)

        # Residual add
        return out + skip


# ============================== Helpers: params, dry-run ==============================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def dry_run():
    """
    Tiny shape test with GRU backend (works without mamba-ssm).
    Switch to backend='mamba' for real runs after installing mamba-ssm.
    """
    torch.manual_seed(0)
    B, D, H, W = 2, 31, 24, 24
    cfg = HSIMambaSRConfig(
        bands=D, scale=2, embed_dim=48,
        n_groups=2, depth_per_group=2, d_state=16, expand=2,
        axis_fusion="gated", backend="gru", drop_path=0.1
    )
    net = HSIMambaSR(cfg)
    x = torch.randn(B, D, H, W)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (B, D, H * cfg.scale, W * cfg.scale), f"Bad shape: {y.shape}"
    print("Dry-run OK:", tuple(y.shape), "Params:", count_parameters(net))


if __name__ == "__main__":
    dry_run()


# # 1) Put your hsimamba_sr.py, dataset_any.py, train_any.py in the same folder
# # 2) Install deps:
# pip install torch scipy h5py  # and mamba-ssm later if you want the real backend
# # 3) Train:
# python train_any.py --data_root /path/to/mat/folder --scale 4 --backend gru
# # After sanity-checks, switch to real Mamba:
# pip install mamba-ssm
# python train_any.py --data_root /path/to/mat/folder --scale 4 --backend mamba
