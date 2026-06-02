"""Generate a sample grid from a checkpoint.

Usage
-----
python generate.py \\
  --ckpt /path/to/ckpt_000100000.pt \\
  --out  sample.png \\
  --n    16
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.utils as vutils

from src.model import Generator, GeneratorConfig


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   type=Path, required=True)
    parser.add_argument("--out",    type=Path, default=Path("sample_grid.png"))
    parser.add_argument("--n",      type=int,  default=16)
    parser.add_argument("--nrow",   type=int,  default=8)
    parser.add_argument("--seed",   type=int,  default=42)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    g_cfg = GeneratorConfig.from_dict(ckpt["meta"]["generator_config"])
    G     = Generator(g_cfg).to(device).eval()

    state_key = "G_state" if args.no_ema else "G_ema_state"
    if state_key not in ckpt:
        state_key = "G_state"
    G.load_state_dict(ckpt[state_key])

    n_params = sum(p.numel() for p in G.parameters())
    print(f"Generator: {g_cfg.resolutions[-1]}px output, {n_params/1e6:.2f}M params")
    print(f"Weights: {state_key}")

    z    = torch.randn(args.n, G.z_dim, generator=torch.Generator().manual_seed(args.seed)).to(device)
    fake = G(z)
    x    = ((fake + 1.0) / 2.0).clamp(0.0, 1.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(vutils.make_grid(x, nrow=args.nrow, padding=2), args.out)
    print(f"Saved {args.n} samples ({tuple(fake.shape[-2:])}) -> {args.out}")


if __name__ == "__main__":
    main()
