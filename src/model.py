"""Cascade SR-Refiner GAN — z → 256 → 512 → 1024.

Architecture
------------
Generator(z) = SRRefiner_1024( SRRefiner_512( BackboneGenerator(z) ) )

Training phases
---------------
Phase 1:  BackboneG (frozen) + SRRefiner_512 (train)  →  512×512 output
          Discriminator at 512, real images from train_50k_512.zip

Phase 2:  BackboneG (frozen) + SRRefiner_512 (frozen) + SRRefiner_1024 (train)
          →  1024×1024 output
          Discriminator at 1024, real images from train_50k_1024.zip

Inference / ONNX export:
  G(z) runs all three modules in sequence → 1024×1024.
  freeze only affects requires_grad, not the forward pass.

SRRefiner design (ESRGAN-inspired)
------------------------------------
- No normalization: avoids mean-shift artifacts on image-space features
- PixelShuffle 2× upsampling: sharper than nearest-neighbor (learned kernel)
- Zero-initialized tail conv: output starts as bilinear upsample at init,
  so training is stable from step 0 without any fade-in schedule
- Residual scale 0.2 on SRResBlocks: prevents gradient explosion
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
# Config dataclasses
# =============================================================================

def _norm_ch(channels: dict[Any, Any]) -> dict[int, int]:
    return {int(k): int(v) for k, v in channels.items()}


def _check_progressive(resolutions: list[int], *, descending: bool) -> None:
    if not resolutions:
        raise ValueError("resolutions must not be empty")
    for cur, nxt in zip(resolutions, resolutions[1:]):
        expected = cur // 2 if descending else cur * 2
        if nxt != expected:
            direction = "halve" if descending else "double"
            raise ValueError(f"resolutions must {direction} every stage: {cur} -> {nxt}")


@dataclass
class BackboneConfig:
    z_dim: int
    resolutions: list[int]
    channels: dict[int, int]
    norm_type: str = "gn"
    gn_groups: int = 32
    attention_resolutions: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.resolutions = [int(r) for r in self.resolutions]
        self.channels    = _norm_ch(self.channels)
        self.attention_resolutions = [int(r) for r in self.attention_resolutions]
        if self.z_dim <= 0:
            raise ValueError("z_dim must be positive")
        _check_progressive(self.resolutions, descending=False)
        for r in self.resolutions:
            if r not in self.channels:
                raise ValueError(f"channels missing for resolution {r}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BackboneConfig":
        return cls(**d)


@dataclass
class SRRefinerConfig:
    """Config for a single 2× SR stage (e.g. 256→512 or 512→1024)."""
    in_resolution: int
    out_resolution: int
    mid_ch: int = 64
    n_res_blocks: int = 8   # body blocks before upsample
    n_mid_blocks: int = 4   # blocks after upsample

    def __post_init__(self) -> None:
        if self.out_resolution != self.in_resolution * 2:
            raise ValueError(
                f"SRRefinerConfig: out_resolution must be exactly 2x in_resolution, "
                f"got {self.in_resolution} -> {self.out_resolution}"
            )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SRRefinerConfig":
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
        if self.minibatch_std_group <= 0:
            raise ValueError("minibatch_std_group must be positive")
        _check_progressive(self.resolutions, descending=True)
        for r in self.resolutions:
            if r not in self.channels:
                raise ValueError(f"channels missing for resolution {r}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DiscriminatorConfig":
        return cls(**d)


# =============================================================================
# Shared helpers
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


def sn(m: nn.Module) -> nn.Module:
    return _sn(m)


# =============================================================================
# BackboneGenerator  (identical structure to 256 baseline for strict load)
# =============================================================================

class _BackboneResBlockUp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm_type: str, gn_groups: int):
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


class _BackboneSelfAttn(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        cs = max(1, channels // 8)
        cm = max(1, channels // 2)
        self.theta = nn.Conv2d(channels, cs, 1, bias=False)
        self.phi   = nn.Conv2d(channels, cs, 1, bias=False)
        self.g     = nn.Conv2d(channels, cm, 1, bias=False)
        self.o     = nn.Conv2d(cm, channels, 1, bias=False)
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


class BackboneGenerator(nn.Module):
    """256×256 pretrained backbone — always frozen after loading."""

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg      = cfg
        self.z_dim    = cfg.z_dim
        first_res     = cfg.resolutions[0]
        first_ch      = cfg.channels[first_res]
        self.first_res = first_res
        self.first_ch  = first_ch

        self.input_proj = nn.Linear(cfg.z_dim, first_ch * first_res * first_res)
        self.res_blocks:  nn.ModuleList = nn.ModuleList()
        self.attn_blocks: nn.ModuleDict = nn.ModuleDict()

        for i in range(1, len(cfg.resolutions)):
            in_ch  = cfg.channels[cfg.resolutions[i - 1]]
            out_ch = cfg.channels[cfg.resolutions[i]]
            res_out = cfg.resolutions[i]
            self.res_blocks.append(
                _BackboneResBlockUp(in_ch, out_ch, cfg.norm_type, cfg.gn_groups)
            )
            if res_out in cfg.attention_resolutions:
                self.attn_blocks[str(res_out)] = _BackboneSelfAttn(out_ch)

        last_ch       = cfg.channels[cfg.resolutions[-1]]
        self.out_norm = make_norm(last_ch, cfg.norm_type, cfg.gn_groups)
        self.to_rgb   = nn.Conv2d(last_ch, 3, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(z).view(-1, self.first_ch, self.first_res, self.first_res)
        for i, res_out in enumerate(self.cfg.resolutions[1:]):
            h = self.res_blocks[i](h)
            if str(res_out) in self.attn_blocks:
                h = self.attn_blocks[str(res_out)](h)
        return torch.tanh(self.to_rgb(F.relu(self.out_norm(h))))

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def load_from_baseline(self, ckpt_path: str, device: str = "cpu") -> None:
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("G_ema_state", ckpt.get("G_state"))
        if state is None:
            raise KeyError("Checkpoint has neither 'G_ema_state' nor 'G_state'")
        try:
            self.load_state_dict(state, strict=True)
            print("BackboneGenerator: strict load OK")
        except RuntimeError:
            remapped = _remap_baseline_keys(state)
            missing, unexpected = self.load_state_dict(remapped, strict=False)
            print(
                f"BackboneGenerator: partial load — "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )


def _remap_baseline_keys(state: dict) -> dict:
    """Remap stages.N keys (mixed Sequential) → res_blocks.N / attn_blocks.R."""
    RESOLUTIONS = [4, 8, 16, 32, 64, 128, 256]
    ATTN_RES    = {32}
    stage_to_new: dict[int, str] = {}
    stage_idx = rb_idx = 0
    for i in range(1, len(RESOLUTIONS)):
        res_out = RESOLUTIONS[i]
        stage_to_new[stage_idx] = f"res_blocks.{rb_idx}"
        stage_idx += 1
        rb_idx    += 1
        if res_out in ATTN_RES:
            stage_to_new[stage_idx] = f"attn_blocks.{res_out}"
            stage_idx += 1
    new_state = {}
    for k, v in state.items():
        if k.startswith("stages."):
            parts = k.split(".", 2)
            prefix = stage_to_new.get(int(parts[1]))
            if prefix is not None:
                new_state[f"{prefix}.{parts[2]}" if len(parts) > 2 else prefix] = v
        else:
            new_state[k] = v
    return new_state


# =============================================================================
# SRRefiner  — single 2× upscale stage (reused for 256→512 and 512→1024)
# =============================================================================

class _SRResBlock(nn.Module):
    """ESRGAN-style residual block: no norm, identity skip, small residual scale."""

    def __init__(self, ch: int, scale: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.leaky_relu(x, 0.2))
        h = self.conv2(F.leaky_relu(h, 0.2))
        return x + self.scale * h


class _PixelShuffleUp(nn.Module):
    """Learned 2× upsampling via sub-pixel convolution."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, 3, padding=1)
        self.ps   = nn.PixelShuffle(2)
        nn.init.orthogonal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.ps(self.conv(x)), 0.2)


class SRRefiner(nn.Module):
    """Single 2× SR stage (in_res → out_res = 2×in_res).

    Reused for both 256→512 (Phase 1) and 512→1024 (Phase 2).

    The output is:
        bilinear_upsample(input) + tanh(learned_delta)
    so the network starts as a pure bilinear upsampler (tail is zero-init)
    and gradually learns to add high-frequency corrections.
    """

    def __init__(self, cfg: SRRefinerConfig):
        super().__init__()
        self.cfg    = cfg
        self.in_res = cfg.in_resolution
        ch          = cfg.mid_ch

        self.head = nn.Conv2d(3, ch, 3, padding=1)
        self.body = nn.Sequential(*[_SRResBlock(ch) for _ in range(cfg.n_res_blocks)])
        self.up   = _PixelShuffleUp(ch, ch)
        self.mid  = nn.Sequential(*[_SRResBlock(ch) for _ in range(cfg.n_mid_blocks)])
        self.tail = nn.Conv2d(ch, 3, 3, padding=1)

        # Zero-init: starts as identity (bilinear anchor)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, img_in: torch.Tensor) -> torch.Tensor:
        out_res = self.cfg.out_resolution
        anchor  = F.interpolate(
            img_in, size=(out_res, out_res), mode="bilinear", align_corners=False
        )
        h     = F.leaky_relu(self.head(img_in), 0.2)
        h     = self.body(h)
        h     = self.up(h)
        h     = self.mid(h)
        delta = torch.tanh(self.tail(h))
        return (anchor + delta).clamp(-1.0, 1.0)

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()


# =============================================================================
# Full Generator  (Backbone + SR_512 + SR_1024)
# =============================================================================

class Generator(nn.Module):
    """Full 1024×1024 generator for submission.

    G(z) = SR_1024( SR_512( Backbone(z) ) )

    Freeze semantics:
      Phase 1 training: backbone.freeze()
      Phase 2 training: backbone.freeze() + sr_512.freeze()
      Inference / ONNX: all three run (freeze only affects requires_grad)
    """

    def __init__(
        self,
        backbone_cfg:  BackboneConfig,
        refiner512_cfg: SRRefinerConfig,
        refiner1024_cfg: SRRefinerConfig,
    ):
        super().__init__()
        self.backbone   = BackboneGenerator(backbone_cfg)
        self.sr_512     = SRRefiner(refiner512_cfg)
        self.sr_1024    = SRRefiner(refiner1024_cfg)
        self.z_dim      = backbone_cfg.z_dim

    # --- inference (all modules) ---
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        img_256  = self.backbone(z)
        img_512  = self.sr_512(img_256)
        img_1024 = self.sr_1024(img_512)
        return img_1024

    # --- Phase-1 training forward (256 → 512) ---
    def forward_phase1(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            img_256 = self.backbone(z)
        return self.sr_512(img_256)

    # --- Phase-2 training forward (256 → 512 → 1024) ---
    def forward_phase2(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            img_256 = self.backbone(z)
            img_512 = self.sr_512(img_256)
        return self.sr_1024(img_512)

    def freeze_backbone(self) -> None:
        self.backbone.freeze()

    def freeze_sr512(self) -> None:
        self.sr_512.freeze()

    @property
    def phase1_parameters(self):
        return self.sr_512.parameters()

    @property
    def phase2_parameters(self):
        return self.sr_1024.parameters()


# =============================================================================
# Discriminator
# =============================================================================

class _ResBlockDown(nn.Module):
    """Pre-activation downsample block (skip uses same activation as main)."""

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


class _DiscSelfAttn(nn.Module):
    def __init__(self, channels: int, use_sn: bool = True):
        super().__init__()
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


class _MinibatchStd(nn.Module):
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


class Discriminator(nn.Module):
    """Config-driven ResNet-D with SpectralNorm + optional SelfAttention."""

    def __init__(self, cfg: DiscriminatorConfig):
        super().__init__()
        self.cfg  = cfg
        wrap      = sn if cfg.use_spectral_norm else (lambda m: m)

        first_ch      = cfg.channels[cfg.resolutions[0]]
        self.from_rgb = wrap(nn.Conv2d(3, first_ch, 3, padding=1))

        self.res_blocks:  nn.ModuleList = nn.ModuleList()
        self.attn_blocks: nn.ModuleDict = nn.ModuleDict()

        for i in range(1, len(cfg.resolutions)):
            in_ch  = cfg.channels[cfg.resolutions[i - 1]]
            out_ch = cfg.channels[cfg.resolutions[i]]
            res_out = cfg.resolutions[i]
            self.res_blocks.append(_ResBlockDown(in_ch, out_ch, cfg.use_spectral_norm))
            if res_out in cfg.attention_resolutions:
                self.attn_blocks[str(res_out)] = _DiscSelfAttn(out_ch, cfg.use_spectral_norm)

        last_res          = cfg.resolutions[-1]
        last_ch           = cfg.channels[last_res]
        self.minibatch_std = _MinibatchStd(cfg.minibatch_std_group)
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
# PatchDiscriminator  (70×70 PatchGAN)
# =============================================================================

class PatchDiscriminator(nn.Module):
    """70×70 PatchGAN discriminator.

    Output shape: (B, 1, H', W') where each value covers a 70×70 receptive field.
    Loss is computed as mean over all patch predictions.

    Why better than full-image D for SR:
    - Focuses on local texture (blur vs sharp), not global structure
    - Generated images have right structure (backbone) but wrong texture — patches catch this
    - More stable gradient signal at high resolution

    Architecture: C64 → C128 → C256 → C512 → C1
    (stride-2 for first n_layers, stride-1 for last two)
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 64, n_layers: int = 3,
                 use_spectral_norm: bool = True):
        super().__init__()
        wrap = sn if use_spectral_norm else (lambda m: m)

        layers: list[nn.Module] = []
        ch_in  = in_ch
        ch_out = base_ch

        # Strided conv layers (downsampling)
        for i in range(n_layers):
            layers += [
                wrap(nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1)),
                nn.LeakyReLU(0.2),
            ]
            ch_in  = ch_out
            ch_out = min(ch_out * 2, 512)

        # stride-1 layer before output
        layers += [
            wrap(nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=1, padding=1)),
            nn.LeakyReLU(0.2),
        ]
        # Output: 1 channel patch map
        layers += [wrap(nn.Conv2d(ch_out, 1, kernel_size=4, stride=1, padding=1))]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, 1, H', W') — averaged in loss


# =============================================================================
# EMA  (tracks the active refiner only)
# =============================================================================

class EMA:
    """EMA of a single SRRefiner module.

    Phase 1: EMA(G.sr_512)
    Phase 2: EMA(G.sr_1024)
    """

    def __init__(self, module: nn.Module, half_life: int = 10_000):
        self.shadow = copy.deepcopy(module).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.half_life = half_life

    @torch.no_grad()
    def update(self, module: nn.Module, batch_size: int) -> None:
        decay = 0.5 ** (batch_size / self.half_life)
        s_params = list(self.shadow.parameters())
        l_params = list(module.parameters())
        if len(s_params) != len(l_params):
            raise RuntimeError("EMA: parameter count mismatch")
        for sp, p in zip(s_params, l_params):
            sp.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
        for sb, b in zip(self.shadow.buffers(), module.buffers()):
            sb.copy_(b)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.shadow.load_state_dict(state)


# =============================================================================
# Factory configs
# =============================================================================

BACKBONE_256_CONFIG = BackboneConfig(
    z_dim=512,
    resolutions=[4, 8, 16, 32, 64, 128, 256],
    channels={4: 512, 8: 512, 16: 512, 32: 512, 64: 256, 128: 128, 256: 64},
    norm_type="gn",
    gn_groups=32,
    attention_resolutions=[32],
)

SR_512_CONFIG = SRRefinerConfig(
    in_resolution=256, out_resolution=512,
    mid_ch=64, n_res_blocks=8, n_mid_blocks=4,
)

SR_1024_CONFIG = SRRefinerConfig(
    in_resolution=512, out_resolution=1024,
    mid_ch=64, n_res_blocks=8, n_mid_blocks=4,
)

DISCRIMINATOR_512_CONFIG = DiscriminatorConfig(
    resolutions=[512, 256, 128, 64, 32, 16, 8, 4],
    channels={512: 64, 256: 64, 128: 128, 64: 256, 32: 512, 16: 512, 8: 512, 4: 512},
    use_spectral_norm=True,
    minibatch_std_group=4,
    attention_resolutions=[32],
)

DISCRIMINATOR_1024_CONFIG = DiscriminatorConfig(
    resolutions=[1024, 512, 256, 128, 64, 32, 16, 8, 4],
    channels={1024: 32, 512: 64, 256: 64, 128: 128,
              64: 256, 32: 512, 16: 512, 8: 512, 4: 512},
    use_spectral_norm=True,
    minibatch_std_group=4,
    attention_resolutions=[32],
)


def build_generator() -> Generator:
    return Generator(BACKBONE_256_CONFIG, SR_512_CONFIG, SR_1024_CONFIG)


def build_discriminator_512() -> Discriminator:
    return Discriminator(DISCRIMINATOR_512_CONFIG)


def build_discriminator_1024() -> Discriminator:
    return Discriminator(DISCRIMINATOR_1024_CONFIG)


def build_patch_discriminator(use_spectral_norm: bool = True) -> PatchDiscriminator:
    return PatchDiscriminator(in_ch=3, base_ch=64, n_layers=3,
                              use_spectral_norm=use_spectral_norm)


# =============================================================================
# Sanity check
# =============================================================================

if __name__ == "__main__":
    G  = build_generator()
    D1 = build_discriminator_512()
    D2 = build_discriminator_1024()

    n_bb   = sum(p.numel() for p in G.backbone.parameters())
    n_sr512  = sum(p.numel() for p in G.sr_512.parameters())
    n_sr1024 = sum(p.numel() for p in G.sr_1024.parameters())
    n_total  = n_bb + n_sr512 + n_sr1024

    print(f"BackboneGenerator : {n_bb/1e6:.2f}M")
    print(f"SRRefiner_512     : {n_sr512/1e6:.2f}M")
    print(f"SRRefiner_1024    : {n_sr1024/1e6:.2f}M")
    print(f"Generator total   : {n_total/1e6:.2f}M  {'OK (<40M)' if n_total < 40e6 else 'FAIL'}")
    print(f"Discriminator_512 : {sum(p.numel() for p in D1.parameters())/1e6:.2f}M")
    print(f"Discriminator_1024: {sum(p.numel() for p in D2.parameters())/1e6:.2f}M")

    z = torch.randn(2, G.z_dim)
    G.freeze_backbone()

    out512  = G.forward_phase1(z)
    out1024 = G.forward_phase2(z)
    full    = G(z)

    print(f"\nPhase-1 output : {tuple(out512.shape)}")
    print(f"Phase-2 output : {tuple(out1024.shape)}")
    print(f"Inference      : {tuple(full.shape)}")
    print(f"Backbone frozen: {all(not p.requires_grad for p in G.backbone.parameters())}")
    print(f"SR_512  frozen : {all(not p.requires_grad for p in G.sr_512.parameters())}")
