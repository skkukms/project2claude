"""Two-stage Refiner: G_256(frozen) → Refiner512 → Refiner1024.

Phase 1: train Refiner512  (256→512)  with D_512
Phase 2: train Refiner1024 (512→1024) with D_1024, Refiner512 frozen

Both refiners use a residual design:
    output = tanh(bilinear_upsample(input) + learned_residual)
Zero-init on to_rgb ensures safe start (pure bilinear at step 0).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import make_norm


# ---------------------------------------------------------------------------
# Building blocks
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
    """Pre-activation 2× upsample residual block."""

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
# Configs
# ---------------------------------------------------------------------------

@dataclass
class Refiner512Config:
    base_ch: int = 128
    n_res_pre: int  = 4   # ResBlocks at 256 before upsample
    n_res_post: int = 4   # ResBlocks at 512 after upsample
    norm_type: str  = "gn"
    gn_groups: int  = 32

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Refiner512Config":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Refiner1024Config:
    base_ch: int = 128
    n_res_pre: int  = 4   # ResBlocks at 512 before upsample
    n_res_post: int = 2   # ResBlocks at 1024 after upsample
    norm_type: str  = "gn"
    gn_groups: int  = 32

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Refiner1024Config":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class Refiner512(nn.Module):
    """256×256 → 512×512.  Input: G_256 output in [-1,1]."""

    def __init__(self, cfg: Refiner512Config):
        super().__init__()
        self.cfg = cfg
        ch, nt, gng = cfg.base_ch, cfg.norm_type, cfg.gn_groups

        self.from_rgb  = nn.Conv2d(3, ch, 3, padding=1)
        self.res_pre   = nn.Sequential(*[ResBlock(ch, nt, gng) for _ in range(cfg.n_res_pre)])
        self.up        = ResBlockUp(ch, ch, nt, gng)   # 256 → 512
        self.res_post  = nn.Sequential(*[ResBlock(ch, nt, gng) for _ in range(cfg.n_res_post)])
        self.out_norm  = make_norm(ch, nt, gng)
        self.to_rgb    = nn.Conv2d(ch, 3, 3, padding=1)

        nn.init.zeros_(self.to_rgb.weight)
        nn.init.zeros_(self.to_rgb.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h        = self.res_post(self.up(self.res_pre(self.from_rgb(x))))
        residual = self.to_rgb(F.relu(self.out_norm(h)))
        base     = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return torch.tanh(base + residual)


class Refiner1024(nn.Module):
    """512×512 → 1024×1024.  Input: Refiner512 output in [-1,1]."""

    def __init__(self, cfg: Refiner1024Config):
        super().__init__()
        self.cfg = cfg
        ch, nt, gng = cfg.base_ch, cfg.norm_type, cfg.gn_groups
        ch2 = ch // 2

        self.from_rgb  = nn.Conv2d(3, ch, 3, padding=1)
        self.res_pre   = nn.Sequential(*[ResBlock(ch, nt, gng) for _ in range(cfg.n_res_pre)])
        self.up        = ResBlockUp(ch, ch2, nt, gng)  # 512 → 1024
        self.res_post  = nn.Sequential(*[ResBlock(ch2, nt, gng) for _ in range(cfg.n_res_post)])
        self.out_norm  = make_norm(ch2, nt, gng)
        self.to_rgb    = nn.Conv2d(ch2, 3, 3, padding=1)

        nn.init.zeros_(self.to_rgb.weight)
        nn.init.zeros_(self.to_rgb.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h        = self.res_post(self.up(self.res_pre(self.from_rgb(x))))
        residual = self.to_rgb(F.relu(self.out_norm(h)))
        base     = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return torch.tanh(base + residual)
