"""Generate a sample grid from a checkpoint.

Two modes (auto-detected from the ckpt format):

A) Distributed baseline (slim) ckpt — contains only G_state / D_state / G_ema_state.
   The 256 baseline architecture is assumed; the script instantiates
   `build_baseline_256_generator()`.
       python generate.py --ckpt ffhq256_baseline.pt --out sample.png --n 64

B) Your own training ckpt — contains 'meta.generator_config' written by train.py.
   The script reads that config and instantiates whatever resolution/channels
   your training used (works for 512 / 1024 student runs).
       python generate.py --ckpt runs/ckpt_001000000.pt --out sample.png

Use `--no-ema` to load G_state instead of G_ema_state (rarely useful — EMA
produces noticeably cleaner samples).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.utils as vutils

from src.model import (
    Generator,
    GeneratorConfig,
    build_baseline_256_generator,
)


def load_generator(ckpt_path: Path, device: str, use_ema: bool) -> Generator:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "meta" in ckpt and isinstance(ckpt["meta"], dict) and "generator_config" in ckpt["meta"]:
        g_cfg = GeneratorConfig.from_dict(ckpt["meta"]["generator_config"])
        G = Generator(g_cfg).to(device).eval()
        source_note = f"meta.generator_config (z_dim={g_cfg.z_dim}, max_res={g_cfg.resolutions[-1]})"
    else:
        G = build_baseline_256_generator().to(device).eval()
        source_note = "build_baseline_256_generator() (no meta in ckpt)"

    if use_ema and "G_ema_state" in ckpt:
        G.load_state_dict(ckpt["G_ema_state"])
        weights_note = "G_ema_state"
    elif "G_state" in ckpt:
        G.load_state_dict(ckpt["G_state"])
        weights_note = "G_state"
    else:
        raise RuntimeError("Checkpoint contains neither G_ema_state nor G_state")

    n_params = sum(p.numel() for p in G.parameters())
    print(f"Architecture: {source_note}")
    print(f"Weights: {weights_note}  ({n_params/1e6:.2f}M params)")
    return G


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("sample_grid.png"))
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--nrow", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    G = load_generator(args.ckpt, device=device, use_ema=not args.no_ema)

    g_for_z = torch.Generator(device="cpu").manual_seed(args.seed)
    z = torch.randn(args.n, G.z_dim, generator=g_for_z).to(device)
    fake = G(z)
    x = ((fake + 1.0) / 2.0).clamp(0.0, 1.0)
    grid = vutils.make_grid(x, nrow=args.nrow, padding=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(grid, args.out)
    print(f"Saved {args.n} samples → {args.out}")


if __name__ == "__main__":
    main()
