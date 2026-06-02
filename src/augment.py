"""Differentiable Augmentation (Zhao et al. 2020, https://arxiv.org/abs/2006.10738).

Applies the same augmentation independently to real and fake before the
discriminator; all ops are differentiable so G still gets a gradient back.

Usage:
    d_real = D(diff_augment(real, policy="color,translation"))
    d_fake = D(diff_augment(fake, policy="color,translation"))

Pass policy="" or None to disable.

Available tags: color, translation, cutout.
TA baseline used "color,translation" (cutout disabled — 50% mask was too aggressive).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def diff_augment(x: torch.Tensor, policy: str | None = "color,translation") -> torch.Tensor:
    if not policy:
        return x
    for tag in policy.split(","):
        tag = tag.strip()
        if not tag:
            continue
        if tag not in _AUGMENT_FNS:
            raise ValueError(f"Unknown DiffAug policy tag: {tag!r}")
        for fn in _AUGMENT_FNS[tag]:
            x = fn(x)
    return x.contiguous()


def rand_brightness(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)


def rand_saturation(x: torch.Tensor) -> torch.Tensor:
    x_mean = x.mean(dim=1, keepdim=True)
    scale = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2.0
    return (x - x_mean) * scale + x_mean


def rand_contrast(x: torch.Tensor) -> torch.Tensor:
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    scale = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5
    return (x - x_mean) * scale + x_mean


def rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    shift_x = int(x.size(2) * ratio + 0.5)
    shift_y = int(x.size(3) * ratio + 0.5)
    tx = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    ty = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    gb, gx, gy = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        indexing="ij",
    )
    gx = torch.clamp(gx + tx + 1, 0, x.size(2) + 1)
    gy = torch.clamp(gy + ty + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[gb, gx, gy].permute(0, 3, 1, 2)
    return x


def rand_cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    cut_h = int(x.size(2) * ratio + 0.5)
    cut_w = int(x.size(3) * ratio + 0.5)
    off_x = torch.randint(0, x.size(2) + (1 - cut_h % 2), size=[x.size(0), 1, 1], device=x.device)
    off_y = torch.randint(0, x.size(3) + (1 - cut_w % 2), size=[x.size(0), 1, 1], device=x.device)
    gb, gx, gy = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cut_h, dtype=torch.long, device=x.device),
        torch.arange(cut_w, dtype=torch.long, device=x.device),
        indexing="ij",
    )
    gx = torch.clamp(gx + off_x - cut_h // 2, min=0, max=x.size(2) - 1)
    gy = torch.clamp(gy + off_y - cut_w // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[gb, gx, gy] = 0.0
    return x * mask.unsqueeze(1)


_AUGMENT_FNS: dict[str, list] = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}
