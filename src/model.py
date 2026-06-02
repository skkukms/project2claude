"""SR-Refiner GAN — z → 1024×1024 via frozen backbone + learned upscaler.

Strategy
--------
Instead of progressively growing the generator (residual branches, fade-in
schedules, complex freeze logic), we split the generator into two parts:

  BackboneGenerator   z(512) → 256×256  (pretrained, kept frozen)
  SRRefiner     256×256 → 1024×1024  (new, trained adversarially)

The full Generator is just:
  G(z) = SRRefiner(BackboneGenerator(z))

Why this is different from residual progressive
-----------------------------------------------
- No residual scale, fade schedule, or multiple RGB heads to tune
- The backbone's 256 output is the stable anchor — the refiner only needs
  to add texture/detail, not learn face structure from noise
- ESRGAN-style residual blocks (no normalization) work well for SR:
  they avoid the mean-shift artifacts that GroupNorm can cause when the
  network is asked to both shift and sharpen pixel values
- PixelShuffle upsampling produces sharper edges than nearest-neighbor
- Much simpler training dynamics: one clean loss, one G phase

Parameter budget
----------------
BackboneGenerator:  ~21.2M  (frozen, doesn't count toward trainable)
SRRefiner:          ~  2.5M  (trained)
Total G:            ~23.7M  < 40M ✅

Discriminator:
  Single ResNet-D starting at 1024 — same design as baseline but deeper.
"""
from __future__ import annotations

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


def _validate_progressive_resolutions(resolutions: list[int], *, descending: bool) -> None:
    if not resolutions:
        raise ValueError("resolutions must not be empty")
    for cur, nxt in zip(resolutions, resolutions[1:]):
        expected = cur // 2 if descending else cur * 2
        if nxt != expected:
            direction = "halve" if descending else "double"
            raise ValueError(
                f"resolutions must {direction} at every stage, got {cur} -> {nxt}"
            )


@dataclass
class BackboneConfig:
    """Config for the 256 baseline backbone generator."""
    z_dim: int
    resolutions: list[int]
    channels: dict[int, int]
    norm_type: str = "gn"
    gn_groups: int = 32
    attention_resolutions: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.resolutions = [int(r) for r in self.resolutions]
        self.channels    = _normalize_channels(self.channels)
        self.attention_resolutions = [int(r) for r in self.attention_resolutions]
        if self.z_dim <= 0:
            raise ValueError("z_dim must be positive")
        _validate_progressive_resolutions(self.resolutions, descending=False)
        for r in self.resolutions:
            if r not in self.channels:
                raise ValueError(f"channels missing entry for resolution {r}")
        for r in self.attention_resolutions:
            if r not in self.resolutions:
                raise ValueError(f"attention resolution {r} is not in backbone")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BackboneConfig":
        return cls(**d)


@dataclass
class SRRefinerConfig:
    """Config for the SR upscaling network."""
    in_resolution: int    # input resolution from backbone (256)
    out_resolution: int   # target resolution (1024)
    mid_ch: int = 64      # feature channels throughout refiner
    n_res_blocks: int = 8 # residual blocks before first upsample

    def __post_init__(self) -> None:
        scale = self.out_resolution // self.in_resolution
        if scale not in (2, 4):
            raise ValueError(
                f"out_resolution must be 2x or 4x in_resolution, "
                f"got {self.in_resolution} -> {self.out_resolution}"
            )
        if scale == 4 and self.in_resolution * 4 != self.out_resolution:
            raise ValueError("out_resolution must be exactly 4x in_resolution")

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
        self.channels    = _normalize_channels(self.channels)
        self.attention_resolutions = [int(r) for r in self.attention_resolutions]
        if self.minibatch_std_group <= 0:
            raise ValueError("minibatch_std_group must be positive")
        _validate_progressive_resolutions(self.resolutions, descending=True)
        for r in self.resolutions:
            if r not in self.channels:
                raise ValueError(f"channels missing entry for resolution {r}")
        for r in self.attention_resolutions:
            if r not in self.resolutions:
                raise ValueError(f"attention resolution {r} not in discriminator")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DiscriminatorConfig":
        return cls(**d)


# =============================================================================
# Shared norm / SN helpers
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
# Backbone building blocks (same as 256 baseline)
# =============================================================================

class BackboneResBlockUp(nn.Module):
    """Pre-activation upsample residual block used in the backbone."""

    def __init__(self, in_ch: int, out_ch: int, norm_type: str = "gn", gn_groups: int = 32):
        super().__init__()
        self.norm1 = make_norm(in_ch,  norm_type, gn_groups)
        self.conv1 = nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1)
        self.norm2 = make_norm(out_ch, norm_type, gn_groups)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip  = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h    = F.interpolate(x, scale_factor=2.0, mode="nearest")
        h    = self.conv1(F.relu(self.norm1(h)))
        h    = self.conv2(F.relu(self.norm2(h)))
        skip = self.skip(F.interpolate(x, scale_factor=2.0, mode="nearest"))
        return h + skip


class BackboneSelfAttention(nn.Module):
    """SAGAN-style self-attention (γ init 0)."""

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


# =============================================================================
# SR Refiner building blocks
# =============================================================================

class SRResBlock(nn.Module):
    """ESRGAN-style residual block: no normalization, identity shortcut.

    No normalization is intentional for SR tasks: it avoids the mean-shift
    that happens when GroupNorm is applied to feature maps that already
    encode absolute pixel intensities.
    """

    def __init__(self, channels: int, residual_scale: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.leaky_relu(x, 0.2, inplace=True))
        h = self.conv2(F.leaky_relu(h, 0.2, inplace=True))
        return x + self.scale * h


class PixelShuffleUp(nn.Module):
    """2× upsampling via sub-pixel convolution (PixelShuffle).

    Sharper than nearest-neighbor because the upsampling kernel is learned.
    in_ch → out_ch at 2× resolution.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1)
        self.ps   = nn.PixelShuffle(2)
        # Init: keep approximately the same magnitude after shuffle
        nn.init.orthogonal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.ps(self.conv(x)), 0.2, inplace=True)


# =============================================================================
# BackboneGenerator  (identical structure to 256 baseline)
# =============================================================================

class BackboneGenerator(nn.Module):
    """256×256 generator — loaded from pre-trained checkpoint, kept frozen.

    Architecture is identical to the distributed 256 baseline so that
    strict weight loading works out of the box.
    """

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
            res_in  = cfg.resolutions[i - 1]
            res_out = cfg.resolutions[i]
            in_ch   = cfg.channels[res_in]
            out_ch  = cfg.channels[res_out]
            self.res_blocks.append(
                BackboneResBlockUp(in_ch, out_ch, norm_type=cfg.norm_type, gn_groups=cfg.gn_groups)
            )
            if res_out in cfg.attention_resolutions:
                self.attn_blocks[str(res_out)] = BackboneSelfAttention(out_ch)

        last_ch       = cfg.channels[cfg.resolutions[-1]]
        self.out_norm = make_norm(last_ch, cfg.norm_type, cfg.gn_groups)
        self.to_rgb   = nn.Conv2d(last_ch, 3, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(z).view(-1, self.first_ch, self.first_res, self.first_res)
        for i, res_out in enumerate(self.cfg.resolutions[1:]):
            h   = self.res_blocks[i](h)
            key = str(res_out)
            if key in self.attn_blocks:
                h = self.attn_blocks[key](h)
        return torch.tanh(self.to_rgb(F.relu(self.out_norm(h))))

    def freeze(self) -> None:
        """Freeze all parameters (call after loading pretrained weights)."""
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def load_from_baseline(self, ckpt_path: str, device: str = "cpu") -> None:
        """Load weights from the distributed 256 baseline checkpoint."""
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Try G_ema_state first (better quality), fall back to G_state
        state = ckpt.get("G_ema_state", ckpt.get("G_state"))
        if state is None:
            raise KeyError("Checkpoint has neither 'G_ema_state' nor 'G_state'")

        # Remap keys: baseline used 'stages.N' (Sequential), we use 'res_blocks.N'
        # Try direct load first, then attempt key remapping
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
    """Remap 'stages.N.xxx' keys from the mixed-Sequential baseline to
    'res_blocks.N.xxx' / 'attn_blocks.RES.xxx' used in BackboneGenerator.

    The baseline stages list interleaves ResBlockUp and SelfAttention2d:
      stages.0  → ResBlockUp (4→8)
      stages.1  → ResBlockUp (8→16)
      ...
      stages.4  → ResBlockUp (32→64)   -- note: attention is at 32, not here
      Wait, attention at 32 means after res_out=32 which is stages index 3 (4->8,8->16,16->32,32->...)
      Actually: resolutions=[4,8,16,32,64,128,256]
        i=1: res_out=8,  stages[0] = ResBlockUp(4→8... wait no. Let me check.
        i=1: res_out=8,  in=cfg.channels[4]=512, out=cfg.channels[8]=512
        i=2: res_out=16, ...
        i=3: res_out=32, ...  -> attention_resolutions=[32] -> stages[3]=ResBlockUp, stages[4]=SA
        i=4: res_out=64, stages[5]=ResBlockUp
        i=5: res_out=128, stages[6]=ResBlockUp
        i=6: res_out=256, stages[7]=ResBlockUp
    So stages has 8 entries: 7 ResBlockUp + 1 SelfAttention.

    res_blocks mapping:
      res_blocks[0] = stages[0]   (8)
      res_blocks[1] = stages[1]   (16)
      res_blocks[2] = stages[2]   (32)
      -- attention at 32: stages[3] = SelfAttention -> attn_blocks['32']
      res_blocks[3] = stages[4]   (64)
      res_blocks[4] = stages[5]   (128)
      res_blocks[5] = stages[6]   (256)
    """
    new_state = {}
    # Build index map: stages.N -> (type, new_key)
    # resolutions=[4,8,16,32,64,128,256], attention_resolutions=[32]
    # Going through the original construction loop:
    stage_idx = 0
    res_block_idx = 0
    RESOLUTIONS = [4, 8, 16, 32, 64, 128, 256]
    ATTN_RES    = {32}

    stage_to_new = {}  # stages.N -> new key prefix
    for i in range(1, len(RESOLUTIONS)):
        res_out = RESOLUTIONS[i]
        stage_to_new[stage_idx] = f"res_blocks.{res_block_idx}"
        stage_idx     += 1
        res_block_idx += 1
        if res_out in ATTN_RES:
            stage_to_new[stage_idx] = f"attn_blocks.{res_out}"
            stage_idx += 1

    for k, v in state.items():
        if k.startswith("stages."):
            parts    = k.split(".", 2)   # ["stages", "N", "rest..."]
            old_idx  = int(parts[1])
            rest     = parts[2] if len(parts) > 2 else ""
            new_prefix = stage_to_new.get(old_idx)
            if new_prefix is not None:
                new_k = f"{new_prefix}.{rest}" if rest else new_prefix
                new_state[new_k] = v
            # else: drop the key (shouldn't happen)
        else:
            new_state[k] = v

    return new_state


# =============================================================================
# SRRefiner   (the new part: 256 → 1024)
# =============================================================================

class SRRefiner(nn.Module):
    """Lightweight super-resolution network: 256×256 → 1024×1024.

    Architecture (ESRGAN-inspired, no normalization):

      head  : Conv(3, mid_ch, 3)
      body  : n_res_blocks × SRResBlock(mid_ch)
      up1   : PixelShuffleUp(mid_ch, mid_ch)   256 → 512
      mid1  : 4 × SRResBlock(mid_ch)
      up2   : PixelShuffleUp(mid_ch, mid_ch)   512 → 1024
      mid2  : 4 × SRResBlock(mid_ch)
      tail  : Conv(mid_ch, 3, 3) + tanh

    The residual correction is added to a bilinear-upsampled version of the
    input so that the network starts as near-identity and only needs to learn
    the high-frequency delta.  This removes the need for any warm-up schedule.

    init_residual_weight controls how strongly the learned correction
    contributes at the start of training (default 0.0 = pure bilinear at
    init, grows as weights are updated).
    """

    def __init__(self, cfg: SRRefinerConfig):
        super().__init__()
        self.cfg          = cfg
        self.in_res       = cfg.in_resolution
        self.out_res      = cfg.out_resolution
        ch                = cfg.mid_ch

        self.head  = nn.Conv2d(3, ch, kernel_size=3, padding=1)
        self.body  = nn.Sequential(*[SRResBlock(ch) for _ in range(cfg.n_res_blocks)])
        self.up1   = PixelShuffleUp(ch, ch)        # 256 → 512
        self.mid1  = nn.Sequential(*[SRResBlock(ch) for _ in range(4)])
        self.up2   = PixelShuffleUp(ch, ch)        # 512 → 1024
        self.mid2  = nn.Sequential(*[SRResBlock(ch) for _ in range(4)])
        self.tail  = nn.Conv2d(ch, 3, kernel_size=3, padding=1)

        # Zero-init tail so output starts as pure bilinear upsample
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, img_low: torch.Tensor) -> torch.Tensor:
        # Bilinear anchor — refiner only learns the residual delta
        anchor = F.interpolate(
            img_low, size=(self.out_res, self.out_res),
            mode="bilinear", align_corners=False,
        )
        h = F.leaky_relu(self.head(img_low), 0.2, inplace=True)
        h = self.body(h)
        h = self.up1(h)
        h = self.mid1(h)
        h = self.up2(h)
        h = self.mid2(h)
        delta = torch.tanh(self.tail(h))
        return (anchor + delta).clamp(-1.0, 1.0)


# =============================================================================
# Full Generator  (Backbone + SRRefiner combined)
# =============================================================================

class Generator(nn.Module):
    """Full 1024×1024 generator for submission.

    G(z) = SRRefiner(BackboneGenerator(z))

    During training:
      - backbone is frozen (loaded from pretrained 256 checkpoint)
      - only refiner parameters are updated

    For ONNX export / submission:
      - forward(z) produces 3×1024×1024 directly
      - no branching or auxiliary outputs
    """

    def __init__(self, backbone_cfg: BackboneConfig, refiner_cfg: SRRefinerConfig):
        super().__init__()
        self.backbone = BackboneGenerator(backbone_cfg)
        self.refiner  = SRRefiner(refiner_cfg)
        self.z_dim    = backbone_cfg.z_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            img_256 = self.backbone(z)
        return self.refiner(img_256)

    def forward_train(self, z: torch.Tensor) -> torch.Tensor:
        """Training forward: backbone gradient is blocked but refiner is active."""
        img_256 = self.backbone(z).detach()
        return self.refiner(img_256)

    def freeze_backbone(self) -> None:
        self.backbone.freeze()

    @property
    def refiner_parameters(self):
        return self.refiner.parameters()


# =============================================================================
# Discriminator
# =============================================================================

class ResBlockDown(nn.Module):
    """Pre-activation downsample block (fixed: skip uses same activation as main).

    main: leaky_relu(x) -> Conv3x3 -> leaky_relu -> Conv3x3 -> AvgPool 2x
    skip: leaky_relu(x) -> Conv1x1 -> AvgPool 2x
    sum scaled by 1/sqrt(2).
    """

    def __init__(self, in_ch: int, out_ch: int, use_spectral_norm: bool = True):
        super().__init__()
        wrap       = sn if use_spectral_norm else (lambda m: m)
        self.conv1 = wrap(nn.Conv2d(in_ch, in_ch,  kernel_size=3, padding=1))
        self.conv2 = wrap(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
        self.skip  = wrap(nn.Conv2d(in_ch, out_ch, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_act = F.leaky_relu(x, 0.2)
        h     = self.conv1(x_act)
        h     = self.conv2(F.leaky_relu(h, 0.2))
        h     = F.avg_pool2d(h, 2)
        skip  = F.avg_pool2d(self.skip(x_act), 2)
        return (h + skip) / math.sqrt(2)


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


class DiscSelfAttention(nn.Module):
    """SAGAN-style self-attention for discriminator."""

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


class Discriminator(nn.Module):
    """ResNet-D at 1024×1024 with SpectralNorm + optional SelfAttention."""

    def __init__(self, cfg: DiscriminatorConfig):
        super().__init__()
        self.cfg = cfg
        wrap     = sn if cfg.use_spectral_norm else (lambda m: m)

        first_res    = cfg.resolutions[0]
        first_ch     = cfg.channels[first_res]
        self.from_rgb = wrap(nn.Conv2d(3, first_ch, kernel_size=3, padding=1))

        self.res_blocks:  nn.ModuleList = nn.ModuleList()
        self.attn_blocks: nn.ModuleDict = nn.ModuleDict()

        for i in range(1, len(cfg.resolutions)):
            res_in  = cfg.resolutions[i - 1]
            res_out = cfg.resolutions[i]
            in_ch   = cfg.channels[res_in]
            out_ch  = cfg.channels[res_out]
            self.res_blocks.append(
                ResBlockDown(in_ch, out_ch, use_spectral_norm=cfg.use_spectral_norm)
            )
            if res_out in cfg.attention_resolutions:
                self.attn_blocks[str(res_out)] = DiscSelfAttention(
                    out_ch, use_sn=cfg.use_spectral_norm
                )

        last_res          = cfg.resolutions[-1]
        last_ch           = cfg.channels[last_res]
        self.minibatch_std = MinibatchStd(group_size=cfg.minibatch_std_group)
        self.final_conv    = wrap(nn.Conv2d(last_ch + 1, last_ch, kernel_size=3, padding=1))
        self.final_linear  = wrap(nn.Linear(last_ch * last_res * last_res, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.from_rgb(x)
        for i, res_out in enumerate(self.cfg.resolutions[1:]):
            h   = self.res_blocks[i](h)
            key = str(res_out)
            if key in self.attn_blocks:
                h = self.attn_blocks[key](h)
        h = self.minibatch_std(h)
        h = F.leaky_relu(self.final_conv(h), 0.2)
        return self.final_linear(h.flatten(1))


# =============================================================================
# EMA
# =============================================================================

class EMA:
    """Exponential moving average of the SRRefiner weights only.

    decay = 0.5 ** (batch_size / half_life)
    The backbone is frozen so it doesn't need EMA.
    """

    import copy as _copy

    def __init__(self, refiner: nn.Module, half_life: int = 10_000):
        import copy
        self.shadow = copy.deepcopy(refiner).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.half_life = half_life

    @torch.no_grad()
    def update(self, refiner: nn.Module, batch_size: int) -> None:
        decay = 0.5 ** (batch_size / self.half_life)
        shadow_params = list(self.shadow.parameters())
        live_params   = list(refiner.parameters())
        if len(shadow_params) != len(live_params):
            raise RuntimeError("EMA parameter count mismatch")
        for sp, p in zip(shadow_params, live_params):
            sp.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
        for sb, b in zip(self.shadow.buffers(), refiner.buffers()):
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

SR_1024_CONFIG = SRRefinerConfig(
    in_resolution=256,
    out_resolution=1024,
    mid_ch=64,
    n_res_blocks=8,
)

DISCRIMINATOR_1024_CONFIG = DiscriminatorConfig(
    resolutions=[1024, 512, 256, 128, 64, 32, 16, 8, 4],
    channels={
        1024: 32, 512: 64, 256: 64, 128: 128,
        64: 256, 32: 512, 16: 512, 8: 512, 4: 512,
    },
    use_spectral_norm=True,
    minibatch_std_group=4,
    attention_resolutions=[32],
)


def build_generator() -> Generator:
    return Generator(BACKBONE_256_CONFIG, SR_1024_CONFIG)


def build_discriminator() -> Discriminator:
    return Discriminator(DISCRIMINATOR_1024_CONFIG)


# =============================================================================
# Sanity check
# =============================================================================

if __name__ == "__main__":
    import copy

    G = build_generator()
    D = build_discriminator()

    n_backbone  = sum(p.numel() for p in G.backbone.parameters())
    n_refiner   = sum(p.numel() for p in G.refiner.parameters())
    n_g_total   = n_backbone + n_refiner
    n_d         = sum(p.numel() for p in D.parameters())

    print(f"BackboneGenerator:  {n_backbone/1e6:.2f}M params")
    print(f"SRRefiner:          {n_refiner/1e6:.2f}M params")
    print(f"Generator total:    {n_g_total/1e6:.2f}M  {'OK (<40M)' if n_g_total < 40e6 else 'FAIL (>40M)'}")
    print(f"Discriminator:      {n_d/1e6:.2f}M params")

    z         = torch.randn(2, G.z_dim)
    G.backbone.freeze()
    out_1024  = G.forward_train(z)
    score     = D(out_1024)

    print(f"G output:  {tuple(out_1024.shape)}  range [{out_1024.min():.3f}, {out_1024.max():.3f}]")
    print(f"D output:  {tuple(score.shape)}")

    # Verify backbone is truly frozen
    frozen = all(not p.requires_grad for p in G.backbone.parameters())
    print(f"Backbone frozen: {frozen}")
