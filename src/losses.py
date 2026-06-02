"""GAN losses + R1 gradient penalty.

Non-saturating logistic (used for residual progressive training):
    L_D = E[softplus(-D(real))] + E[softplus(D(fake))]
    L_G = E[softplus(-D(fake))]

R1 gradient penalty:
    R1 = (γ/2) * E_real[‖∇_x D(x)‖²]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def ns_logistic_g(d_fake: torch.Tensor) -> torch.Tensor:
    return F.softplus(-d_fake).mean()


def r1_penalty(D: nn.Module, x_real: torch.Tensor, gamma: float = 10.0) -> torch.Tensor:
    x = x_real.detach().requires_grad_(True)
    d = D(x).sum()
    (grad,) = torch.autograd.grad(d, x, create_graph=True)
    return (gamma / 2.0) * grad.pow(2).flatten(1).sum(dim=1).mean()
