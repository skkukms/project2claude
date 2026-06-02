"""Export a Generator to ONNX with a fixed submission interface.

Submission contract (every leaderboard entry must satisfy this):
    input  z      shape (B, 512), dtype float32
    output image  shape (B, 3, 1024, 1024), dtype float32, range [-1, 1]

The export wraps your Generator in a small module that always resizes the
output to 1024×1024 with bilinear interpolation. So if your G produces 256 or
512 natively, the wrapper upsamples; if it already produces 1024, the resize
is a no-op (it still goes through F.interpolate, but with the same target
shape, the values are unchanged up to floating-point error).

This means the input/output contract is identical for every student —
whatever architecture, channels, or training resolution you chose.

CLI — works out of the box on the distributed baseline ckpt:
    python export_onnx.py --ckpt ffhq256_baseline.pt --out submission.onnx

CLI — works on any ckpt whose architecture is `build_baseline_256_generator()`.
For your own scaled-up architecture, instantiate the Generator yourself and
call `export_to_onnx(G, "submission.onnx")` from Python:

    import torch
    from export_onnx import export_to_onnx
    from src.model import Generator, GeneratorConfig

    G = Generator(GeneratorConfig(
        z_dim=512,
        resolutions=[4, 8, 16, 32, 64, 128, 256, 512, 1024],  # your design
        channels={...},
        ...
    ))
    state = torch.load("my_ckpt.pt", weights_only=True)["G_ema_state"]
    G.load_state_dict(state)
    export_to_onnx(G, "submission.onnx")
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import build_baseline_256_generator


TARGET_RESOLUTION = 1024


class SubmissionWrapper(nn.Module):
    """Run G(z) and resize the image to 1024×1024 with bilinear interpolation."""

    def __init__(self, G: nn.Module):
        super().__init__()
        self.G = G

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.G(z)
        x = F.interpolate(
            x,
            size=(TARGET_RESOLUTION, TARGET_RESOLUTION),
            mode="bilinear",
            align_corners=False,
        )
        return x


def export_to_onnx(
    G: nn.Module,
    out_path: str | Path,
    *,
    opset: int = 17,
    batch_size: int = 1,
) -> None:
    """Export `G` (z → image) wrapped to (B, 512) → (B, 3, 1024, 1024).

    The batch dimension is exported dynamic; other dimensions are static.
    `G.z_dim` must equal 512 (assignment spec).
    """
    if getattr(G, "z_dim", None) != 512:
        raise ValueError(
            f"G.z_dim must be 512 (assignment spec). Got {getattr(G, 'z_dim', None)!r}."
        )

    G.eval()
    wrapper = SubmissionWrapper(G).eval()

    dummy_z = torch.randn(batch_size, 512)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy_z,
        str(out_path),
        input_names=["z"],
        output_names=["image"],
        opset_version=opset,
        dynamic_axes={"z": {0: "batch"}, "image": {0: "batch"}},
        dynamo=False,  # legacy tracer — avoids the onnxscript dependency
    )

    with torch.no_grad():
        ref_out = wrapper(dummy_z)
    print(f"Saved ONNX → {out_path}")
    print(f"  input  z      (B, 512)")
    print(f"  output image  {tuple(ref_out.shape)} (B dynamic), range "
          f"[{ref_out.min():.3f}, {ref_out.max():.3f}]")


def _load_baseline_g(ckpt_path: Path) -> nn.Module:
    """Load the distributed 256 baseline G (G_ema_state)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    G = build_baseline_256_generator()
    state = ckpt.get("G_ema_state") or ckpt.get("G_state")
    if state is None:
        raise RuntimeError("Checkpoint has neither G_ema_state nor G_state")
    G.load_state_dict(state)
    return G


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--ckpt", type=Path, required=True,
        help="Path to a ckpt whose architecture is the baseline 256 "
             "(build_baseline_256_generator). For other architectures, "
             "use the Python API.",
    )
    parser.add_argument("--out", type=Path, default=Path("submission.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    G = _load_baseline_g(args.ckpt)
    export_to_onnx(G, args.out, opset=args.opset, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
