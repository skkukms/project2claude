"""FFHQ-256 baseline model — standalone single-file definition.

Contents (everything you need to instantiate G and D and run/fine-tune):
- GeneratorConfig / DiscriminatorConfig dataclasses
- norm helpers (GroupNorm, spectral_norm wrapper)
- building blocks: ResBlockUp, ResBlockDown, MinibatchStd, SelfAttention2d
- Generator, Discriminator
- EMA (exponential moving average for G)

Loading the distributed baseline ckpt:
    import torch
    from model import build_baseline_256_generator, build_baseline_256_discriminator

    ckpt = torch.load("ffhq256_baseline.pt", map_location="cuda", weights_only=True)
    G     = build_baseline_256_generator().cuda()
    D     = build_baseline_256_discriminator().cuda()
    G_ema = build_baseline_256_generator().cuda()

    G.load_state_dict(ckpt["G_state"])
    D.load_state_dict(ckpt["D_state"])
    G_ema.load_state_dict(ckpt["G_ema_state"])
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm as _sn


# =============================================================================
# Config
# =============================================================================

def _normalize_channels(channels: dict[Any, Any]) -> dict[int, int]:
    return {int(k): int(v) for k, v in channels.items()}


@dataclass
class GeneratorConfig:
    z_dim: int
    resolutions: list[int]
    channels: dict[int, int]
    norm_type: str = "gn"  # 'gn' or 'in'
    gn_groups: int = 32
    attention_resolutions: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.resolutions = [int(r) for r in self.resolutions]
        self.channels = _normalize_channels(self.channels)
        self.attention_resolutions = [int(r) for r in self.attention_resolutions]
        for r in self.resolutions:
            if r not in self.channels:
                raise ValueError(f"channels missing entry for resolution {r}")
        if self.norm_type not in ("gn", "in"):
            raise ValueError(f"norm_type must be 'gn' or 'in', got {self.norm_type!r}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GeneratorConfig":
        return cls(**d)


@dataclass
class DiscriminatorConfig:
    resolutions: list[int]
    channels: dict[int, int]
    use_spectral_norm: bool = True
    minibatch_std_group: int = 4
    attention_resolutions: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.resolutions = [int(r) for r in self.resolutions]
        self.channels = _normalize_channels(self.channels)
        self.attention_resolutions = [int(r) for r in self.attention_resolutions]
        for r in self.resolutions:
            if r not in self.channels:
                raise ValueError(f"channels missing entry for resolution {r}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DiscriminatorConfig":
        return cls(**d)


# =============================================================================
# Norms
# =============================================================================

def make_norm(channels: int, norm_type: str, gn_groups: int) -> nn.Module:
    if norm_type == "gn":
        groups = min(gn_groups, channels)
        if channels % groups != 0:
            groups = channels
        return nn.GroupNorm(num_groups=groups, num_channels=channels)
    if norm_type == "in":
        return nn.InstanceNorm2d(channels, affine=True)
    raise ValueError(f"Unknown norm_type: {norm_type!r}")


def sn(module: nn.Module) -> nn.Module:
    return _sn(module)


# =============================================================================
# Building blocks
# =============================================================================

class ResBlockUp(nn.Module):
    """Pre-activation upsample residual block.

    main: NN-upsample 2× → norm → ReLU → Conv3×3 → norm → ReLU → Conv3×3
    skip: NN-upsample 2× → Conv1×1 (or Identity if in_ch == out_ch)
    """

    def __init__(self, in_ch: int, out_ch: int, norm_type: str = "gn", gn_groups: int = 32):
        super().__init__()
        self.norm1 = make_norm(in_ch, norm_type, gn_groups)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = make_norm(out_ch, norm_type, gn_groups)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.interpolate(x, scale_factor=2.0, mode="nearest")
        h = self.conv1(F.relu(self.norm1(h)))
        h = self.conv2(F.relu(self.norm2(h)))
        skip = F.interpolate(x, scale_factor=2.0, mode="nearest")
        skip = self.skip(skip)
        return h + skip


class ResBlockDown(nn.Module):
    """Pre-activation downsample residual block with Spectral Norm (no norm layer).

    main: leaky_relu → Conv3×3 → leaky_relu → Conv3×3 → AvgPool 2×
    skip: AvgPool 2× → Conv1×1
    sum scaled by 1/sqrt(2).
    """

    def __init__(self, in_ch: int, out_ch: int, use_spectral_norm: bool = True):
        super().__init__()
        wrap = sn if use_spectral_norm else (lambda m: m)
        self.conv1 = wrap(nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1))
        self.conv2 = wrap(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
        self.skip = wrap(nn.Conv2d(in_ch, out_ch, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.leaky_relu(x, 0.2))
        h = self.conv2(F.leaky_relu(h, 0.2))
        h = F.avg_pool2d(h, 2)
        skip = F.avg_pool2d(self.skip(x), 2)
        return (h + skip) / math.sqrt(2)


class MinibatchStd(nn.Module):
    """Append per-group standard-deviation feature channel."""

    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        g = min(self.group_size, B)
        if B % g != 0:
            g = B
        y = x.view(g, B // g, C, H, W)
        y = y - y.mean(dim=0, keepdim=True)
        y = (y.pow(2).mean(dim=0) + 1e-8).sqrt()
        y = y.mean(dim=[1, 2, 3], keepdim=True)
        y = y.repeat(g, 1, H, W)
        return torch.cat([x, y], dim=1)


class SelfAttention2d(nn.Module):
    """SAGAN-style 2D self-attention with learnable γ (init 0)."""

    def __init__(self, channels: int, use_spectral_norm: bool = False):
        super().__init__()
        if channels < 8:
            raise ValueError(f"SelfAttention2d requires channels>=8, got {channels}")
        wrap = sn if use_spectral_norm else (lambda m: m)
        cs = channels // 8
        cm = channels // 2
        self.theta = wrap(nn.Conv2d(channels, cs, kernel_size=1, bias=False))
        self.phi = wrap(nn.Conv2d(channels, cs, kernel_size=1, bias=False))
        self.g = wrap(nn.Conv2d(channels, cm, kernel_size=1, bias=False))
        self.o = wrap(nn.Conv2d(cm, channels, kernel_size=1, bias=False))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        theta = self.theta(x).view(B, -1, N)
        phi = F.max_pool2d(self.phi(x), 2).view(B, -1, N // 4)
        attn = F.softmax(torch.bmm(theta.transpose(1, 2), phi), dim=-1)
        g = F.max_pool2d(self.g(x), 2).view(B, -1, N // 4)
        y = torch.bmm(g, attn.transpose(1, 2)).view(B, -1, H, W)
        return self.gamma * self.o(y) + x


# =============================================================================
# Generator / Discriminator
# =============================================================================

class Generator(nn.Module):
    """Config-driven ResNet upsample stack: z → 4×4 → ... → R×R."""

    def __init__(self, cfg: GeneratorConfig):
        super().__init__()
        self.cfg = cfg
        self.z_dim = cfg.z_dim

        first_res = cfg.resolutions[0]
        first_ch = cfg.channels[first_res]
        self.first_res = first_res
        self.first_ch = first_ch

        self.input_proj = nn.Linear(cfg.z_dim, first_ch * first_res * first_res)

        stages: list[nn.Module] = []
        for i in range(1, len(cfg.resolutions)):
            res_out = cfg.resolutions[i]
            in_ch = cfg.channels[cfg.resolutions[i - 1]]
            out_ch = cfg.channels[res_out]
            stages.append(
                ResBlockUp(in_ch, out_ch, norm_type=cfg.norm_type, gn_groups=cfg.gn_groups)
            )
            if res_out in cfg.attention_resolutions:
                stages.append(SelfAttention2d(out_ch, use_spectral_norm=False))
        self.stages = nn.Sequential(*stages)

        last_ch = cfg.channels[cfg.resolutions[-1]]
        self.out_norm = make_norm(last_ch, cfg.norm_type, cfg.gn_groups)
        self.to_rgb = nn.Conv2d(last_ch, 3, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(z).view(-1, self.first_ch, self.first_res, self.first_res)
        h = self.stages(h)
        h = F.relu(self.out_norm(h))
        return torch.tanh(self.to_rgb(h))


class Discriminator(nn.Module):
    """Config-driven ResNet downsample stack with SN + optional SA."""

    def __init__(self, cfg: DiscriminatorConfig):
        super().__init__()
        self.cfg = cfg
        wrap = sn if cfg.use_spectral_norm else (lambda m: m)

        first_res = cfg.resolutions[0]
        first_ch = cfg.channels[first_res]
        self.from_rgb = wrap(nn.Conv2d(3, first_ch, kernel_size=3, padding=1))

        stages: list[nn.Module] = []
        for i in range(1, len(cfg.resolutions)):
            res_out = cfg.resolutions[i]
            in_ch = cfg.channels[cfg.resolutions[i - 1]]
            out_ch = cfg.channels[res_out]
            stages.append(ResBlockDown(in_ch, out_ch, use_spectral_norm=cfg.use_spectral_norm))
            if res_out in cfg.attention_resolutions:
                stages.append(
                    SelfAttention2d(out_ch, use_spectral_norm=cfg.use_spectral_norm)
                )
        self.stages = nn.Sequential(*stages)

        last_res = cfg.resolutions[-1]
        last_ch = cfg.channels[last_res]
        self.minibatch_std = MinibatchStd(group_size=cfg.minibatch_std_group)
        self.final_conv = wrap(nn.Conv2d(last_ch + 1, last_ch, kernel_size=3, padding=1))
        self.final_linear = wrap(nn.Linear(last_ch * last_res * last_res, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.from_rgb(x)
        h = self.stages(h)
        h = self.minibatch_std(h)
        h = F.leaky_relu(self.final_conv(h), 0.2)
        h = h.flatten(1)
        return self.final_linear(h)


# =============================================================================
# EMA
# =============================================================================

class EMA:
    """Exponential moving average of Generator weights.

    decay = 0.5 ** (batch_size / half_life). Call `update(G, batch_size)` after
    every G step. The shadow copy is `.eval()` with grad disabled.
    """

    def __init__(self, G: nn.Module, half_life: int = 10_000):
        self.shadow = copy.deepcopy(G).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.half_life = half_life

    @torch.no_grad()
    def update(self, G: nn.Module, batch_size: int) -> None:
        decay = 0.5 ** (batch_size / self.half_life)
        for sp, p in zip(self.shadow.parameters(), G.parameters()):
            sp.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
        for sb, b in zip(self.shadow.buffers(), G.buffers()):
            sb.copy_(b)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.shadow.load_state_dict(state)


# =============================================================================
# Factories — instantiate the distributed 256 baseline
# =============================================================================

BASELINE_256_GENERATOR_CONFIG = GeneratorConfig(
    z_dim=512,
    resolutions=[4, 8, 16, 32, 64, 128, 256],
    channels={4: 512, 8: 512, 16: 512, 32: 512, 64: 256, 128: 128, 256: 64},
    norm_type="gn",
    gn_groups=32,
    attention_resolutions=[32],
)

BASELINE_256_DISCRIMINATOR_CONFIG = DiscriminatorConfig(
    resolutions=[256, 128, 64, 32, 16, 8, 4],
    channels={256: 64, 128: 128, 64: 256, 32: 512, 16: 512, 8: 512, 4: 512},
    use_spectral_norm=True,
    minibatch_std_group=4,
    attention_resolutions=[32],
)


def build_baseline_256_generator() -> Generator:
    return Generator(BASELINE_256_GENERATOR_CONFIG)


def build_baseline_256_discriminator() -> Discriminator:
    return Discriminator(BASELINE_256_DISCRIMINATOR_CONFIG)


if __name__ == "__main__":
    G = build_baseline_256_generator()
    D = build_baseline_256_discriminator()
    n_g = sum(p.numel() for p in G.parameters())
    n_d = sum(p.numel() for p in D.parameters())
    print(f"Generator: {n_g/1e6:.2f}M params")
    print(f"Discriminator: {n_d/1e6:.2f}M params")
    z = torch.randn(2, G.z_dim)
    fake = G(z)
    score = D(fake)
    print(f"G(z) shape: {tuple(fake.shape)}, range [{fake.min():.3f}, {fake.max():.3f}]")
    print(f"D(fake) shape: {tuple(score.shape)}")
