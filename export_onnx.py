"""Export Generator to ONNX for leaderboard submission.

Submission contract:
    input  z      shape (B, 512), dtype float32
    output image  shape (B, 3, 1024, 1024), dtype float32, range [-1, 1]

--- Refiner pipeline (G_256 + Refiner → 1024) ---
    python export_onnx.py \\
        --mode refiner \\
        --refiner-ckpt runs/refiner_1024/final.pt \\
        --g256-ckpt ckpt/ffhq256_baseline.pt \\
        --out submission.onnx

--- Baseline 256 only (bilinear upsample to 1024) ---
    python export_onnx.py \\
        --mode baseline \\
        --ckpt ckpt/ffhq256_baseline.pt \\
        --out submission.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import build_baseline_256_generator
from src.refiner import Refiner, RefinerConfig


TARGET = 1024


class RefinerWrapper(nn.Module):
    """G_256 (frozen) + Refiner → 1024×1024."""

    def __init__(self, G256: nn.Module, refiner: nn.Module):
        super().__init__()
        self.G256    = G256
        self.refiner = refiner

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        img = self.refiner(self.G256(z))
        if img.shape[-1] != TARGET:
            img = F.interpolate(img, size=(TARGET, TARGET), mode="bilinear", align_corners=False)
        return img


class BaselineWrapper(nn.Module):
    """G_256 → bilinear upsample to 1024×1024."""

    def __init__(self, G: nn.Module):
        super().__init__()
        self.G = G

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.interpolate(self.G(z), size=(TARGET, TARGET), mode="bilinear", align_corners=False)


def _export(model: nn.Module, out_path: Path, opset: int = 17) -> None:
    model.eval()
    dummy_z = torch.randn(1, 512)
    with torch.no_grad():
        out = model(dummy_z)
    print(f"Output: {tuple(out.shape)}, range [{out.min():.3f}, {out.max():.3f}]")
    torch.onnx.export(
        model, dummy_z, str(out_path),
        opset_version=opset,
        input_names=["z"], output_names=["image"],
        dynamic_axes={"z": {0: "batch"}, "image": {0: "batch"}},
        dynamo=False,
    )
    print(f"Saved → {out_path}")


def export_refiner(refiner_ckpt: Path, g256_ckpt: Path, out_path: Path, opset: int = 17) -> None:
    device = "cpu"

    G256 = build_baseline_256_generator().to(device).eval()
    g_state = torch.load(g256_ckpt, map_location=device, weights_only=True)
    G256.load_state_dict(g_state["G_ema_state"])
    for p in G256.parameters(): p.requires_grad_(False)

    ckpt    = torch.load(refiner_ckpt, map_location=device, weights_only=False)
    r_cfg   = RefinerConfig(**ckpt["meta"]["refiner_config"])
    refiner = Refiner(r_cfg).to(device).eval()
    refiner.load_state_dict(ckpt["R_ema_state"])
    for p in refiner.parameters(): p.requires_grad_(False)

    n_g256 = sum(p.numel() for p in G256.parameters())
    n_ref  = sum(p.numel() for p in refiner.parameters())
    total  = n_g256 + n_ref
    print(f"G_256: {n_g256/1e6:.2f}M | Refiner: {n_ref/1e6:.2f}M | Total: {total/1e6:.2f}M")
    if total > 40e6:
        raise ValueError(f"Total {total/1e6:.2f}M exceeds 40M limit!")

    _export(RefinerWrapper(G256, refiner), out_path, opset)


def export_baseline(ckpt_path: Path, out_path: Path, opset: int = 17) -> None:
    G = build_baseline_256_generator().eval()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    G.load_state_dict(state.get("G_ema_state") or state["G_state"])
    _export(BaselineWrapper(G), out_path, opset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["refiner", "baseline"], default="refiner")
    parser.add_argument("--refiner-ckpt", type=Path)
    parser.add_argument("--g256-ckpt",    type=Path)
    parser.add_argument("--ckpt",         type=Path, help="for --mode baseline")
    parser.add_argument("--out",          type=Path, default=Path("submission.onnx"))
    parser.add_argument("--opset",        type=int,  default=17)
    args = parser.parse_args()

    if args.mode == "refiner":
        if not args.refiner_ckpt or not args.g256_ckpt:
            raise SystemExit("--mode refiner requires --refiner-ckpt and --g256-ckpt")
        export_refiner(args.refiner_ckpt, args.g256_ckpt, args.out, args.opset)
    else:
        if not args.ckpt:
            raise SystemExit("--mode baseline requires --ckpt")
        export_baseline(args.ckpt, args.out, args.opset)


if __name__ == "__main__":
    main()
