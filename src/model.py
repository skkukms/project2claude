"""Residual Progressive GAN — 256 → 512 → 1024.

Architecture
------------
Generator(z) produces images progressively:
  - 4×4 → ... → 256×256  (backbone, pretrained & frozen)
  - 256 → 512  : upsample(base_256) + scale * residual_conv(feat_512)
  - 512 → 1024 : upsample(out_512)  + scale * residual_conv(feat_1024)

Residual convs are zero-initialized → output is pure bilinear at init.
Fade-in schedule: scale goes 0 → residual_rgb_scale over N images.

Key fixes vs original baseline
--------------------------------
1. ResBlockDown skip path uses shared pre-activation (bug: original had
   activation on main path only, none on skip).
2. Generator stages stored in ModuleList + ModuleDict instead of a mixed
   nn.Sequential with manual index arithmetic.
3. EMA checks parameter/buffer count strictly (no silent truncation).

Training phases
---------------
Phase 1: freeze backbone (4→256), train 512 stage + 512 residual head
Phase 2: freeze backbone + 512 stage, train 1024 stage + 1024 residual head
"""
from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm as _sn


# =============================================================================
# Config
# =============================================================================

def _norm_ch(ch: dict[Any, Any]) -> dict[int, int]:
    return {int(k): int(v) for k, v in ch.items()}


def _check_progressive(res: list[int], *, descending: bool) -> None:
    if not res:
        raise ValueError("resolutions must not be empty")
    for cur, nxt in zip(res, res[1:]):
        expected = cur // 2 if descending else cur * 2
        if nxt != expected:
            direction = "halve" if descending else "double"
            raise ValueError(f"resolutions must {direction} every stage: {cur} -> {nxt}")


@dataclass
class GeneratorConfig:
    z_dim: int
    resolutions: list[int]            # e.g. [4,8,16,32,64,128,256,512,1024]
    channels: dict[int, int]
    norm_type: str = "gn"
    gn_groups: int = 32
    attention_resolutions: list[int] = field(default_factory=list)
    # Residual RGB branches (e.g. [512] or [512, 1024])
    residual_rgb_resolutions: list[int] = field(default_factory=list)
    residual_rgb_scale: float = 0.03
    residual_rgb_fade_images: int = 20_000  # 0 = full scale from start

    def __post_init__(self) -> None:
        self.resolutions = [int(r) for r in self.resolutions]
        self.channels    = _norm_ch(self.channels)
        self.attention_resolutions    = [int(r) for r in self.attention_resolutions]
        self.residual_rgb_resolutions = [int(r) for r in self.residual_rgb_resolutions]
        self.residual_rgb_scale       = float(self.residual_rgb_scale)
        self.residual_rgb_fade_images = int(self.residual_rgb_fade_images)
        if self.z_dim <= 0:
            raise ValueError("z_dim must be positive")
        if self.norm_type not in ("gn", "in"):
            raise ValueError(f"norm_type must be 'gn' or 'in', got {self.norm_type!r}")
        _check_progressive(self.resolutions, descending=False)
        for r in self.resolutions:
            if r not in self.channels:
                raise ValueError(f"channels missing for resolution {r}")
        for r in self.attention_resolutions:
            if r not in self.resolutions:
                raise ValueError(f"attention resolution {r} not in G stages")
        for r in self.residual_rgb_resolutions:
            if r not in self.resolutions[1:]:
                raise ValueError(f"residual_rgb_resolution {r} must be an upsampled stage")

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
        self.channels    = _norm_ch(self.channels)
        self.attention_resolutions = [int(r) for r in self.attention_resolutions]
        _check_progressive(self.resolutions, descending=True)
        for r in self.resolutions:
            if r not in self.channels:
                raise ValueError(f"channels missing for resolution {r}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DiscriminatorConfig":
        return cls(**d)


# =============================================================================
# Helpers
# =============================================================================

def make_norm(channels: int, norm_type: str, gn_groups: int) -> nn.Module:
    if norm_type == "gn":
        groups = min(gn_groups, channels)
        if channels % groups != 0:
            warnings.warn(
                f"GroupNorm fallback: channels={channels} not divisible by "
                f"groups={groups}, using groups={channels}"
            )
            groups = channels
        return nn.GroupNorm(num_groups=groups, num_channels=channels)
    if norm_type == "in":
        return nn.InstanceNorm2d(channels, affine=True)
    raise ValueError(f"Unknown norm_type: {norm_type!r}")


def sn(m: nn.Module) -> nn.Module:
    return _sn(m)


# =============================================================================
# Building blocks
# =============================================================================

class ResBlockUp(nn.Module):
    """Pre-activation upsample residual block for Generator.

    main: NN-upsample 2x → norm → ReLU → Conv3x3 → norm → ReLU → Conv3x3
    skip: NN-upsample 2x → Conv1x1 (or Identity when in_ch == out_ch)
    """

    def __init__(self, in_ch: int, out_ch: int, norm_type: str = "gn", gn_groups: int = 32):
        super().__init__()
        self.norm1 = make_norm(in_ch,  norm_type, gn_groups)
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.norm2 = make_norm(out_ch, norm_type, gn_groups)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h    = F.interpolate(x, scale_factor=2.0, mode="nearest")
        h    = self.conv1(F.relu(self.norm1(h)))
        h    = self.conv2(F.relu(self.norm2(h)))
        skip = self.skip(F.interpolate(x, scale_factor=2.0, mode="nearest"))
        return h + skip


class ResBlockDown(nn.Module):
    """Pre-activation downsample residual block for Discriminator.

    FIX vs baseline: skip path uses same pre-activation as main path.

    main: leaky_relu(x) → Conv3x3 → leaky_relu → Conv3x3 → AvgPool 2x
    skip: leaky_relu(x) → Conv1x1 → AvgPool 2x
    sum scaled by 1/sqrt(2).
    """

    def __init__(self, in_ch: int, out_ch: int, use_sn: bool = True):
        super().__init__()
        wrap       = sn if use_sn else (lambda m: m)
        self.conv1 = wrap(nn.Conv2d(in_ch, in_ch,  3, padding=1))
        self.conv2 = wrap(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.skip  = wrap(nn.Conv2d(in_ch, out_ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_act = F.leaky_relu(x, 0.2)
        h     = self.conv1(x_act)
        h     = self.conv2(F.leaky_relu(h, 0.2))
        h     = F.avg_pool2d(h, 2)
        skip  = F.avg_pool2d(self.skip(x_act), 2)
        return (h + skip) / math.sqrt(2)


class SelfAttention2d(nn.Module):
    """SAGAN-style self-attention with learnable gamma (init 0)."""

    def __init__(self, channels: int, use_sn: bool = False):
        super().__init__()
        if channels < 8:
            raise ValueError(f"SelfAttention2d requires channels >= 8, got {channels}")
        wrap = sn if use_sn else (lambda m: m)
        cs   = max(1, channels // 8)
        cm   = max(1, channels // 2)
        self.theta = wrap(nn.Conv2d(channels, cs, 1, bias=False))
        self.phi   = wrap(nn.Conv2d(channels, cs, 1, bias=False))
        self.g     = wrap(nn.Conv2d(channels, cm, 1, bias=False))
        self.o     = wrap(nn.Conv2d(cm, channels, 1, bias=False))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N     = H * W
        theta = self.theta(x).view(B, -1, N)
        phi   = F.max_pool2d(self.phi(x), 2).view(B, -1, N // 4)
        attn  = F.softmax(torch.bmm(theta.transpose(1, 2), phi), dim=-1)
        g     = F.max_pool2d(self.g(x), 2).view(B, -1, N // 4)
        y     = torch.bmm(g, attn.transpose(1, 2)).view(B, -1, H, W)
        return self.gamma * self.o(y) + x


class MinibatchStd(nn.Module):
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
        y = y.mean(dim=[1, 2, 3], keepdim=True).repeat(g, 1, H, W)
        return torch.cat([x, y], dim=1)


# =============================================================================
# Generator
# =============================================================================

class Generator(nn.Module):
    """Residual progressive generator: z → 256 → (512) → (1024).

    Stage storage:
        res_blocks  : ModuleList  — one ResBlockUp per upsample stage
        attn_blocks : ModuleDict  — SelfAttention2d keyed by resolution str

    Residual RGB:
        Base RGB head at the last non-residual resolution (e.g. 256).
        Each residual_rgb_resolution adds:
            out = clamp(upsample(prev_out) + fade_scale * residual_conv(feat))
        residual_conv is zero-initialized → safe from step 0.
    """

    def __init__(self, cfg: GeneratorConfig):
        super().__init__()
        self.cfg   = cfg
        self.z_dim = cfg.z_dim

        first_res = cfg.resolutions[0]
        first_ch  = cfg.channels[first_res]
        self.first_res = first_res
        self.first_ch  = first_ch

        self.input_proj = nn.Linear(cfg.z_dim, first_ch * first_res * first_res)

        self.res_blocks:  nn.ModuleList = nn.ModuleList()
        self.attn_blocks: nn.ModuleDict = nn.ModuleDict()

        for i in range(1, len(cfg.resolutions)):
            in_ch   = cfg.channels[cfg.resolutions[i - 1]]
            out_ch  = cfg.channels[cfg.resolutions[i]]
            res_out = cfg.resolutions[i]
            self.res_blocks.append(
                ResBlockUp(in_ch, out_ch, norm_type=cfg.norm_type, gn_groups=cfg.gn_groups)
            )
            if res_out in cfg.attention_resolutions:
                self.attn_blocks[str(res_out)] = SelfAttention2d(out_ch, use_sn=False)

        # Base RGB head — at the last non-residual resolution
        residual_set = set(cfg.residual_rgb_resolutions)
        if residual_set:
            first_res_idx        = min(cfg.resolutions.index(r) for r in residual_set)
            self.base_rgb_res    = cfg.resolutions[first_res_idx - 1]
        else:
            self.base_rgb_res    = cfg.resolutions[-1]

        base_ch       = cfg.channels[self.base_rgb_res]
        self.out_norm = make_norm(base_ch, cfg.norm_type, cfg.gn_groups)
        self.to_rgb   = nn.Conv2d(base_ch, 3, 3, padding=1)

        # Residual RGB branches
        self.residual_norms:   nn.ModuleDict = nn.ModuleDict()
        self.residual_to_rgbs: nn.ModuleDict = nn.ModuleDict()

        initial_fade = 1.0 if cfg.residual_rgb_fade_images == 0 else 0.0
        self.register_buffer("_fade", torch.tensor(initial_fade), persistent=False)

        for res in cfg.residual_rgb_resolutions:
            key = str(res)
            ch  = cfg.channels[res]
            self.residual_norms[key]   = make_norm(ch, cfg.norm_type, cfg.gn_groups)
            self.residual_to_rgbs[key] = nn.Conv2d(ch, 3, 3, padding=1)
            nn.init.zeros_(self.residual_to_rgbs[key].weight)
            nn.init.zeros_(self.residual_to_rgbs[key].bias)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def set_training_progress(self, images_seen: int) -> None:
        if self.cfg.residual_rgb_fade_images == 0:
            fade = 1.0
        else:
            fade = min(1.0, images_seen / self.cfg.residual_rgb_fade_images)
        self._fade.fill_(fade)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(z).view(-1, self.first_ch, self.first_res, self.first_res)

        image: Optional[torch.Tensor] = None
        residual_set = set(self.cfg.residual_rgb_resolutions)

        if self.first_res == self.base_rgb_res:
            image = torch.tanh(self.to_rgb(F.relu(self.out_norm(h))))

        for i, res_out in enumerate(self.cfg.resolutions[1:]):
            h = self.res_blocks[i](h)
            if str(res_out) in self.attn_blocks:
                h = self.attn_blocks[str(res_out)](h)

            if res_out == self.base_rgb_res:
                image = torch.tanh(self.to_rgb(F.relu(self.out_norm(h))))

            elif res_out in residual_set:
                if image is None:
                    raise RuntimeError(f"No base image before residual stage {res_out}")
                image = F.interpolate(
                    image, size=(res_out, res_out), mode="bilinear", align_corners=False
                )
                key      = str(res_out)
                residual = self.residual_to_rgbs[key](F.relu(self.residual_norms[key](h)))
                scale    = self.cfg.residual_rgb_scale * self._fade
                image    = (image + scale * residual).clamp(-1.0, 1.0)

        if image is None:
            raise RuntimeError("Generator produced no image")
        return image

    def forward_with_aux(self, z: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward with residual L2 aux loss for regularization."""
        h = self.input_proj(z).view(-1, self.first_ch, self.first_res, self.first_res)

        image: Optional[torch.Tensor] = None
        residual_set   = set(self.cfg.residual_rgb_resolutions)
        residual_l2    = z.new_zeros(())
        residual_count = 0

        if self.first_res == self.base_rgb_res:
            image = torch.tanh(self.to_rgb(F.relu(self.out_norm(h))))

        for i, res_out in enumerate(self.cfg.resolutions[1:]):
            h = self.res_blocks[i](h)
            if str(res_out) in self.attn_blocks:
                h = self.attn_blocks[str(res_out)](h)

            if res_out == self.base_rgb_res:
                image = torch.tanh(self.to_rgb(F.relu(self.out_norm(h))))

            elif res_out in residual_set:
                if image is None:
                    raise RuntimeError(f"No base image before residual stage {res_out}")
                image = F.interpolate(
                    image, size=(res_out, res_out), mode="bilinear", align_corners=False
                )
                key      = str(res_out)
                residual = self.residual_to_rgbs[key](F.relu(self.residual_norms[key](h)))
                scale    = self.cfg.residual_rgb_scale * self._fade
                image    = (image + scale * residual).clamp(-1.0, 1.0)
                residual_l2    = residual_l2 + residual.square().mean()
                residual_count += 1

        if image is None:
            raise RuntimeError("Generator produced no image")
        if residual_count > 0:
            residual_l2 = residual_l2 / residual_count
        return image, {"residual_l2": residual_l2}

    def freeze_backbone(self) -> None:
        """Freeze all stages up to and including base_rgb_res."""
        for p in self.parameters():
            p.requires_grad_(False)

        # Unfreeze only residual stages and their RGB heads
        for i, res_out in enumerate(self.cfg.resolutions[1:]):
            if res_out in self.cfg.residual_rgb_resolutions:
                for p in self.res_blocks[i].parameters():
                    p.requires_grad_(True)
                if str(res_out) in self.attn_blocks:
                    for p in self.attn_blocks[str(res_out)].parameters():
                        p.requires_grad_(True)
                for p in self.residual_norms[str(res_out)].parameters():
                    p.requires_grad_(True)
                for p in self.residual_to_rgbs[str(res_out)].parameters():
                    p.requires_grad_(True)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"Generator freeze: trainable {trainable/1e6:.2f}M / {total/1e6:.2f}M params")

    def freeze_up_to(self, freeze_resolutions: list[int]) -> None:
        """Freeze stages up to and including given resolutions."""
        freeze_set = set(freeze_resolutions)
        for p in self.parameters():
            p.requires_grad_(False)

        for i, res_out in enumerate(self.cfg.resolutions[1:]):
            if res_out not in freeze_set:
                for p in self.res_blocks[i].parameters():
                    p.requires_grad_(True)
                if str(res_out) in self.attn_blocks:
                    for p in self.attn_blocks[str(res_out)].parameters():
                        p.requires_grad_(True)
            if res_out in self.cfg.residual_rgb_resolutions and res_out not in freeze_set:
                for p in self.residual_norms[str(res_out)].parameters():
                    p.requires_grad_(True)
                for p in self.residual_to_rgbs[str(res_out)].parameters():
                    p.requires_grad_(True)

        # Base RGB head
        if self.base_rgb_res not in freeze_set:
            for p in self.out_norm.parameters():
                p.requires_grad_(True)
            for p in self.to_rgb.parameters():
                p.requires_grad_(True)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"Generator partial freeze: trainable {trainable/1e6:.2f}M / {total/1e6:.2f}M params")


# =============================================================================
# Discriminator
# =============================================================================

class Discriminator(nn.Module):
    def __init__(self, cfg: DiscriminatorConfig):
        super().__init__()
        self.cfg  = cfg
        wrap      = sn if cfg.use_spectral_norm else (lambda m: m)

        first_ch      = cfg.channels[cfg.resolutions[0]]
        self.from_rgb = wrap(nn.Conv2d(3, first_ch, 3, padding=1))

        self.res_blocks:  nn.ModuleList = nn.ModuleList()
        self.attn_blocks: nn.ModuleDict = nn.ModuleDict()

        for i in range(1, len(cfg.resolutions)):
            in_ch   = cfg.channels[cfg.resolutions[i - 1]]
            out_ch  = cfg.channels[cfg.resolutions[i]]
            res_out = cfg.resolutions[i]
            self.res_blocks.append(ResBlockDown(in_ch, out_ch, use_sn=cfg.use_spectral_norm))
            if res_out in cfg.attention_resolutions:
                self.attn_blocks[str(res_out)] = SelfAttention2d(
                    out_ch, use_sn=cfg.use_spectral_norm
                )

        last_res          = cfg.resolutions[-1]
        last_ch           = cfg.channels[last_res]
        self.minibatch_std = MinibatchStd(cfg.minibatch_std_group)
        self.final_conv    = wrap(nn.Conv2d(last_ch + 1, last_ch, 3, padding=1))
        self.final_linear  = wrap(nn.Linear(last_ch * last_res * last_res, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.from_rgb(x)
        for i, res_out in enumerate(self.cfg.resolutions[1:]):
            h = self.res_blocks[i](h)
            if str(res_out) in self.attn_blocks:
                h = self.attn_blocks[str(res_out)](h)
        h = self.minibatch_std(h)
        h = F.leaky_relu(self.final_conv(h), 0.2)
        return self.final_linear(h.flatten(1))


# =============================================================================
# EMA
# =============================================================================

class EMA:
    """Exponential moving average of Generator weights."""

    def __init__(self, G: nn.Module, half_life: int = 10_000):
        self.shadow = copy.deepcopy(G).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.half_life = half_life

    @torch.no_grad()
    def update(self, G: nn.Module, batch_size: int) -> None:
        decay = 0.5 ** (batch_size / self.half_life)
        s_params = list(self.shadow.parameters())
        l_params = list(G.parameters())
        if len(s_params) != len(l_params):
            raise RuntimeError("EMA param count mismatch")
        for sp, p in zip(s_params, l_params):
            sp.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
        s_bufs = list(self.shadow.buffers())
        l_bufs = list(G.buffers())
        if len(s_bufs) != len(l_bufs):
            raise RuntimeError("EMA buffer count mismatch")
        for sb, b in zip(s_bufs, l_bufs):
            sb.copy_(b)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.shadow.load_state_dict(state)


# =============================================================================
# Factory configs
# =============================================================================

BACKBONE_CHANNELS = {4: 512, 8: 512, 16: 512, 32: 512, 64: 256, 128: 128, 256: 64}

# Phase 1: 512 output via residual branch
GENERATOR_512_CONFIG = GeneratorConfig(
    z_dim=512,
    resolutions=[4, 8, 16, 32, 64, 128, 256, 512],
    channels={**BACKBONE_CHANNELS, 512: 64},
    norm_type="gn", gn_groups=32,
    attention_resolutions=[32],
    residual_rgb_resolutions=[512],
    residual_rgb_scale=0.03,
    residual_rgb_fade_images=20_000,
)

DISCRIMINATOR_512_CONFIG = DiscriminatorConfig(
    resolutions=[512, 256, 128, 64, 32, 16, 8, 4],
    channels={512: 32, 256: 64, 128: 128, 64: 256, 32: 256, 16: 256, 8: 256, 4: 256},
    use_spectral_norm=True, minibatch_std_group=4, attention_resolutions=[32],
)

# Phase 2: 1024 output via additional residual branch
GENERATOR_1024_CONFIG = GeneratorConfig(
    z_dim=512,
    resolutions=[4, 8, 16, 32, 64, 128, 256, 512, 1024],
    channels={**BACKBONE_CHANNELS, 512: 64, 1024: 32},
    norm_type="gn", gn_groups=32,
    attention_resolutions=[32],
    residual_rgb_resolutions=[512, 1024],
    residual_rgb_scale=0.03,
    residual_rgb_fade_images=0,  # 512 already faded; 1024 head is zero-init
)

DISCRIMINATOR_1024_CONFIG = DiscriminatorConfig(
    resolutions=[1024, 512, 256, 128, 64, 32, 16, 8, 4],
    channels={1024: 32, 512: 64, 256: 64, 128: 128,
              64: 256, 32: 512, 16: 512, 8: 512, 4: 512},
    use_spectral_norm=True, minibatch_std_group=4, attention_resolutions=[32],
)


def build_generator_512() -> Generator:
    return Generator(GENERATOR_512_CONFIG)


def build_generator_1024() -> Generator:
    return Generator(GENERATOR_1024_CONFIG)


def build_discriminator_512() -> Discriminator:
    return Discriminator(DISCRIMINATOR_512_CONFIG)


def build_discriminator_1024() -> Discriminator:
    return Discriminator(DISCRIMINATOR_1024_CONFIG)


# =============================================================================
# Sanity check
# =============================================================================

if __name__ == "__main__":
    for label, G, D in [
        ("Phase 1 (512)", build_generator_512(), build_discriminator_512()),
        ("Phase 2 (1024)", build_generator_1024(), build_discriminator_1024()),
    ]:
        n_g = sum(p.numel() for p in G.parameters())
        n_d = sum(p.numel() for p in D.parameters())
        ok  = "OK" if n_g < 40e6 else "FAIL >40M"
        z   = torch.randn(2, G.z_dim)
        out = G(z)
        D(out)
        print(f"[{label}] G={n_g/1e6:.2f}M ({ok})  D={n_d/1e6:.2f}M  out={tuple(out.shape)}")
