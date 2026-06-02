"""GAN losses: non-saturating logistic, Relativistic average GAN, R1 penalty.

Standard (non-saturating logistic):
    L_D = E[softplus(-D(real))] + E[softplus(D(fake))]
    L_G = E[softplus(-D(fake))]

Relativistic average GAN (RaGAN) — used with SR-Refiner:
    L_D = BCE(D(real) - mean(D(fake)), 1) + BCE(D(fake) - mean(D(real)), 0)
    L_G = BCE(D(fake) - mean(D(real)), 1) + BCE(D(real) - mean(D(fake)), 0)

    Key advantage: provides gradient signal even when D(real) ≈ D(fake) ≈ 0,
    because it compares real vs fake *relatively* rather than absolutely.
    Used in ESRGAN.

R1 gradient penalty (shared by both):
    R1 = (γ/2) * E_real[‖∇_x D(x)‖²]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Standard non-saturating logistic
# =============================================================================

def ns_logistic_d(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    return F.softplus(-d_real).mean() + F.softplus(d_fake).mean()


def ns_logistic_g(d_fake: torch.Tensor) -> torch.Tensor:
    return F.softplus(-d_fake).mean()


# =============================================================================
# Relativistic average GAN (RaGAN)
# =============================================================================

def ragan_d(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """Relativistic average discriminator loss.

    d_real / d_fake: raw logits, shape (B, *) — works for both scalar and patch outputs.
    """
    real_rel = d_real - d_fake.mean()
    fake_rel = d_fake - d_real.mean()
    return (
        F.binary_cross_entropy_with_logits(real_rel, torch.ones_like(real_rel))
        + F.binary_cross_entropy_with_logits(fake_rel, torch.zeros_like(fake_rel))
    ) / 2.0


def ragan_g(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """Relativistic average generator loss.

    d_real should be detached (no gradient through D for G update).
    """
    real_rel = d_real - d_fake.mean()
    fake_rel = d_fake - d_real.mean()
    return (
        F.binary_cross_entropy_with_logits(fake_rel, torch.ones_like(fake_rel))
        + F.binary_cross_entropy_with_logits(real_rel, torch.zeros_like(real_rel))
    ) / 2.0


# =============================================================================
# R1 gradient penalty
# =============================================================================

def r1_penalty(D: nn.Module, x_real: torch.Tensor, gamma: float = 10.0) -> torch.Tensor:
    x = x_real.detach().requires_grad_(True)
    d = D(x).mean()   # mean (not sum) — scale-invariant across patch sizes
    (grad,) = torch.autograd.grad(d, x, create_graph=True)
    return (gamma / 2.0) * grad.pow(2).flatten(1).sum(dim=1).mean()
