"""Generate a sample grid from trained checkpoints.

Usage
-----
# Phase-1 only (256->512 output)
python generate.py \\
  --backbone-ckpt /path/to/ffhq256_baseline.pt \\
  --sr512-ckpt    runs/phase1_sr512/final.pt \\
  --out sample_512.png

# Full pipeline (256->512->1024 output)
python generate.py \\
  --backbone-ckpt /path/to/ffhq256_baseline.pt \\
  --sr512-ckpt    runs/phase1_sr512/final.pt \\
  --sr1024-ckpt   runs/phase2_sr1024/final.pt \\
  --out sample_1024.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.utils as vutils

from src.model import (
    BackboneConfig,
    SRRefinerConfig,
    Generator,
    BACKBONE_256_CONFIG,
    SR_512_CONFIG,
    SR_1024_CONFIG,
)


def build_generator(
    backbone_ckpt: Path,
    sr512_ckpt: Path | None,
    sr1024_ckpt: Path | None,
    device: str,
) -> Generator:
    G = Generator(BACKBONE_256_CONFIG, SR_512_CONFIG, SR_1024_CONFIG).to(device)

    # Load backbone
    G.backbone.load_from_baseline(str(backbone_ckpt), device=device)
    G.freeze_backbone()

    # Load SR_512
    if sr512_ckpt is not None:
        ckpt = torch.load(sr512_ckpt, map_location=device, weights_only=False)
        state = ckpt.get("active_refiner_state") or ckpt.get("ema_state")
        if state is None:
            raise KeyError(f"No refiner state found in {sr512_ckpt}")
        G.sr_512.load_state_dict(state)
        print(f"SR_512 loaded from {sr512_ckpt.name}")

    # Load SR_1024
    if sr1024_ckpt is not None:
        ckpt = torch.load(sr1024_ckpt, map_location=device, weights_only=False)
        state = ckpt.get("active_refiner_state") or ckpt.get("ema_state")
        if state is None:
            raise KeyError(f"No refiner state found in {sr1024_ckpt}")
        G.sr_1024.load_state_dict(state)
        print(f"SR_1024 loaded from {sr1024_ckpt.name}")

    G.eval()
    return G


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-ckpt", type=Path, required=True)
    parser.add_argument("--sr512-ckpt",    type=Path, default=None)
    parser.add_argument("--sr1024-ckpt",   type=Path, default=None)
    parser.add_argument("--out",   type=Path, default=Path("sample_grid.png"))
    parser.add_argument("--n",     type=int,  default=16)
    parser.add_argument("--nrow",  type=int,  default=4)
    parser.add_argument("--seed",  type=int,  default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    G = build_generator(args.backbone_ckpt, args.sr512_ckpt, args.sr1024_ckpt, device)

    z    = torch.randn(args.n, G.z_dim, generator=torch.Generator().manual_seed(args.seed)).to(device)
    fake = G(z)
    x    = ((fake + 1.0) / 2.0).clamp(0.0, 1.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(vutils.make_grid(x, nrow=args.nrow, padding=2), args.out)
    print(f"Saved {args.n} samples ({tuple(fake.shape[-2:])}) -> {args.out}")


if __name__ == "__main__":
    main()
