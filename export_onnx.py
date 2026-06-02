"""Export to ONNX for leaderboard submission.

Full pipeline: G_256(frozen) → Refiner512(frozen) → Refiner1024 → 1024×1024

    python export_onnx.py \\
        --g256-ckpt  ckpt/ffhq256_baseline.pt \\
        --r512-ckpt  runs/refiner_512/final.pt \\
        --r1024-ckpt runs/refiner_1024/final.pt \\
        --out        submission.onnx

Baseline only (bilinear upsample):
    python export_onnx.py --mode baseline \\
        --g256-ckpt ckpt/ffhq256_baseline.pt --out submission.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import build_baseline_256_generator
from src.refiner import Refiner512, Refiner512Config, Refiner1024, Refiner1024Config


TARGET = 1024


class FullPipeline(nn.Module):
    """G_256 → Refiner512 → Refiner1024 → 1024×1024."""

    def __init__(self, G256: nn.Module, r512: nn.Module, r1024: nn.Module):
        super().__init__()
        self.G256  = G256
        self.r512  = r512
        self.r1024 = r1024

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.r1024(self.r512(self.G256(z)))


class BaselineWrapper(nn.Module):
    def __init__(self, G: nn.Module):
        super().__init__()
        self.G = G

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.interpolate(self.G(z), size=(TARGET, TARGET),
                             mode="bilinear", align_corners=False)


def _check_params(model: nn.Module, label: str = "Total") -> None:
    n = sum(p.numel() for p in model.parameters())
    status = "OK" if n < 40e6 else "EXCEEDS 40M LIMIT!"
    print(f"{label}: {n/1e6:.2f}M  [{status}]")
    if n >= 40e6:
        raise ValueError(f"{label} {n/1e6:.2f}M exceeds 40M param limit!")


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


def load_g256(path: Path, device: str = "cpu"):
    G256  = build_baseline_256_generator().to(device).eval()
    state = torch.load(path, map_location=device, weights_only=True)
    G256.load_state_dict(state["G_ema_state"])
    for p in G256.parameters(): p.requires_grad_(False)
    print(f"G_256: {sum(p.numel() for p in G256.parameters())/1e6:.2f}M")
    return G256


def load_r512(path: Path, device: str = "cpu"):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    cfg   = Refiner512Config(**ckpt["meta"]["refiner_config"])
    r512  = Refiner512(cfg).to(device).eval()
    r512.load_state_dict(ckpt["refiner_ema_state"])
    for p in r512.parameters(): p.requires_grad_(False)
    print(f"Refiner512: {sum(p.numel() for p in r512.parameters())/1e6:.2f}M")
    return r512


def load_r1024(path: Path, device: str = "cpu"):
    ckpt   = torch.load(path, map_location=device, weights_only=False)
    cfg    = Refiner1024Config(**ckpt["meta"]["refiner_config"])
    r1024  = Refiner1024(cfg).to(device).eval()
    r1024.load_state_dict(ckpt["refiner_ema_state"])
    for p in r1024.parameters(): p.requires_grad_(False)
    print(f"Refiner1024: {sum(p.numel() for p in r1024.parameters())/1e6:.2f}M")
    return r1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",      choices=["full", "baseline"], default="full")
    parser.add_argument("--g256-ckpt",  required=True, type=Path)
    parser.add_argument("--r512-ckpt",  type=Path)
    parser.add_argument("--r1024-ckpt", type=Path)
    parser.add_argument("--out",        type=Path, default=Path("submission.onnx"))
    parser.add_argument("--opset",      type=int,  default=17)
    args = parser.parse_args()

    if args.mode == "full":
        if not args.r512_ckpt or not args.r1024_ckpt:
            raise SystemExit("--mode full requires --r512-ckpt and --r1024-ckpt")
        G256  = load_g256(args.g256_ckpt)
        r512  = load_r512(args.r512_ckpt)
        r1024 = load_r1024(args.r1024_ckpt)
        model = FullPipeline(G256, r512, r1024)
        _check_params(model)
        _export(model, args.out, args.opset)
    else:
        G256  = load_g256(args.g256_ckpt)
        model = BaselineWrapper(G256)
        _check_params(model)
        _export(model, args.out, args.opset)


if __name__ == "__main__":
    main()
