"""Refiner: 256×256 → 1024×1024 upsampler trained adversarially.

Architecture:
  from_rgb  : Conv(3 → ch)
  resblocks  : N plain ResBlocks at 256 (feature extraction)
  upsample1  : ResBlockUp(ch → ch) → 512×512
  resblocks  : M plain ResBlocks at 512
  upsample2  : ResBlockUp(ch → ch//2) → 1024×1024
  resblocks  : K plain ResBlocks at 1024
  to_rgb     : norm → relu → Conv(ch//2 → 3) → tanh

Consistency loss: ||downsample(output_1024, 256) - input_256||₁
keeps the coarse structure intact.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import make_norm, sn


# ---------------------------------------------------------------------------

@dataclass
class RefinerConfig:
    base_ch: int = 128
    n_res_before: int = 4    # ResBlocks at 256 before first upsample
    n_res_mid: int = 2       # ResBlocks at 512
    n_res_after: int = 2     # ResBlocks at 1024
    norm_type: str = "gn"
    gn_groups: int = 32

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RefinerConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Plain pre-activation residual block (no spatial change)."""

    def __init__(self, ch: int, norm_type: str = "gn", gn_groups: int = 32):
        super().__init__()
        self.norm1 = make_norm(ch, norm_type, gn_groups)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = make_norm(ch, norm_type, gn_groups)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.relu(self.norm1(x)))
        h = self.conv2(F.relu(self.norm2(h)))
        return (x + h) / math.sqrt(2)


class ResBlockUp(nn.Module):
    """Pre-activation upsample residual block (mirrors baseline Generator)."""

    def __init__(self, in_ch: int, out_ch: int, norm_type: str = "gn", gn_groups: int = 32):
        super().__init__()
        self.norm1 = make_norm(in_ch, norm_type, gn_groups)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = make_norm(out_ch, norm_type, gn_groups)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h    = F.interpolate(x, scale_factor=2.0, mode="nearest")
        h    = self.conv1(F.relu(self.norm1(h)))
        h    = self.conv2(F.relu(self.norm2(h)))
        skip = self.skip(F.interpolate(x, scale_factor=2.0, mode="nearest"))
        return h + skip


# ---------------------------------------------------------------------------

class Refiner(nn.Module):
    """256×256 → 1024×1024 residual upsampler."""

    def __init__(self, cfg: RefinerConfig):
        super().__init__()
        self.cfg = cfg
        ch   = cfg.base_ch
        ch2  = ch // 2
        nt   = cfg.norm_type
        gng  = cfg.gn_groups

        self.from_rgb = nn.Conv2d(3, ch, 3, padding=1)

        self.res_before = nn.Sequential(
            *[ResBlock(ch, nt, gng) for _ in range(cfg.n_res_before)]
        )
        self.up1 = ResBlockUp(ch, ch, nt, gng)          # 256 → 512
        self.res_mid = nn.Sequential(
            *[ResBlock(ch, nt, gng) for _ in range(cfg.n_res_mid)]
        )
        self.up2 = ResBlockUp(ch, ch2, nt, gng)         # 512 → 1024
        self.res_after = nn.Sequential(
            *[ResBlock(ch2, nt, gng) for _ in range(cfg.n_res_after)]
        )

        self.out_norm = make_norm(ch2, nt, gng)
        self.to_rgb   = nn.Conv2d(ch2, 3, 3, padding=1)

        # Zero-init to_rgb so output starts as pure bilinear upsample
        nn.init.zeros_(self.to_rgb.weight)
        nn.init.zeros_(self.to_rgb.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: 3×256×256  (output of frozen G_256, in [-1,1])
        h = self.from_rgb(x)
        h = self.res_before(h)
        h = self.up1(h)
        h = self.res_mid(h)
        h = self.up2(h)
        h = self.res_after(h)
        residual = self.to_rgb(F.relu(self.out_norm(h)))
        base = F.interpolate(x, scale_factor=4.0, mode="bilinear", align_corners=False)
        return torch.tanh(base + residual)
