"""Export the full Generator to ONNX for leaderboard submission.

Submission contract:
    input  z      shape (B, 512), dtype float32
    output image  shape (B, 3, 1024, 1024), dtype float32, range [-1, 1]

Usage
-----
python export_onnx.py \\
  --backbone-ckpt /path/to/ffhq256_baseline.pt \\
  --sr512-ckpt    runs/phase1_sr512/final.pt \\
  --sr1024-ckpt   runs/phase2_sr1024/final.pt \\
  --out           submission.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from generate import build_generator

TARGET_RES = 1024


class _SubmissionWrapper(nn.Module):
    """Wraps G so output is always exactly (B, 3, 1024, 1024)."""

    def __init__(self, G: nn.Module):
        super().__init__()
        self.G = G

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.G(z)
        if x.shape[-1] != TARGET_RES:
            x = F.interpolate(x, size=(TARGET_RES, TARGET_RES),
                              mode="bilinear", align_corners=False)
        return x


def export_to_onnx(
    backbone_ckpt: Path,
    sr512_ckpt: Path,
    sr1024_ckpt: Path,
    out_path: Path,
    opset: int = 17,
) -> None:
    G = build_generator(backbone_ckpt, sr512_ckpt, sr1024_ckpt, device="cpu")
    G.eval()

    wrapper   = _SubmissionWrapper(G).eval()
    dummy_z   = torch.randn(1, 512)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper, dummy_z, str(out_path),
        input_names=["z"],
        output_names=["image"],
        opset_version=opset,
        dynamic_axes={"z": {0: "batch"}, "image": {0: "batch"}},
        dynamo=False,
    )

    with torch.no_grad():
        ref = wrapper(dummy_z)
    print(f"Saved ONNX -> {out_path}")
    print(f"  input  z     : (B, 512)")
    print(f"  output image : {tuple(ref.shape)}  range [{ref.min():.3f}, {ref.max():.3f}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-ckpt", type=Path, required=True)
    parser.add_argument("--sr512-ckpt",    type=Path, required=True)
    parser.add_argument("--sr1024-ckpt",   type=Path, required=True)
    parser.add_argument("--out",           type=Path, default=Path("submission.onnx"))
    parser.add_argument("--opset",         type=int,  default=17)
    args = parser.parse_args()

    export_to_onnx(
        args.backbone_ckpt, args.sr512_ckpt, args.sr1024_ckpt,
        args.out, opset=args.opset,
    )


if __name__ == "__main__":
    main()
