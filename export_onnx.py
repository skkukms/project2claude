"""Export Generator to ONNX for leaderboard submission.

Submission contract:
    input  z     : (B, 512) float32
    output image : (B, 3, 1024, 1024) float32, range [-1, 1]

Usage
-----
python export_onnx.py \\
  --ckpt runs/phase2_residual1024/final.pt \\
  --out  submission.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import Generator, GeneratorConfig

TARGET_RES = 1024


class _SubmissionWrapper(nn.Module):
    def __init__(self, G: nn.Module):
        super().__init__()
        self.G = G

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.G(z)
        if x.shape[-1] != TARGET_RES:
            x = F.interpolate(x, size=(TARGET_RES, TARGET_RES),
                              mode="bilinear", align_corners=False)
        return x


def export_to_onnx(ckpt_path: Path, out_path: Path, opset: int = 17) -> None:
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    g_cfg = GeneratorConfig.from_dict(ckpt["meta"]["generator_config"])
    G     = Generator(g_cfg).eval()

    state = ckpt.get("G_ema_state", ckpt.get("G_state"))
    G.load_state_dict(state)

    wrapper  = _SubmissionWrapper(G).eval()
    dummy_z  = torch.randn(1, 512)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper, dummy_z, str(out_path),
        input_names=["z"], output_names=["image"],
        opset_version=opset,
        dynamic_axes={"z": {0: "batch"}, "image": {0: "batch"}},
        dynamo=False,
    )

    with torch.no_grad():
        ref = wrapper(dummy_z)
    print(f"Saved ONNX -> {out_path}")
    print(f"  input  z     : (B, 512)")
    print(f"  output image : {tuple(ref.shape)}  range [{ref.min():.3f}, {ref.max():.3f}]")
    print(f"  G params     : {sum(p.numel() for p in G.parameters())/1e6:.2f}M")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",  type=Path, required=True)
    parser.add_argument("--out",   type=Path, default=Path("submission.onnx"))
    parser.add_argument("--opset", type=int,  default=17)
    args = parser.parse_args()
    export_to_onnx(args.ckpt, args.out, opset=args.opset)


if __name__ == "__main__":
    main()
