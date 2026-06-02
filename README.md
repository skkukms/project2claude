# FFHQ-256 baseline — student package

This package contains everything you need to load the distributed 256×256
baseline, sample from it, and fine-tune it up to 512 or 1024.

```
ffhqgen_student/
├── README.md
├── requirements.txt
├── train.py                       training loop (fine-tune the baseline, or resume)
├── generate.py                    sample grid from any ckpt
├── export_onnx.py                 leaderboard submission: (B,512) → (B,3,1024,1024)
├── ckpt/
│   └── ffhq256_baseline.pt        239 MB — G + D + G_ema state_dicts (slim)
├── configs/
│   └── baseline_256.yaml          starting config (matches the distributed ckpt)
└── src/
    ├── __init__.py
    ├── model.py                   Generator / Discriminator / EMA / baseline builders
    ├── losses.py                  non-saturating logistic + R1
    ├── augment.py                 DiffAug (color / translation / cutout)
    └── dataset.py                 zip-backed image dataset
```

The baseline was trained on FFHQ (50k images) for 5.0M images at 256×256. It's
intentionally **higher quality than "mediocre"** — your fine-tune starts from
a non-trivial starting point so the work focuses on upscaling and refinement,
not basic face structure.

## Quick start

Run everything from the package root (`ffhqgen_student/`):

```bash
pip install -r requirements.txt

# 1. Verify the baseline loads and samples
python generate.py --ckpt ckpt/ffhq256_baseline.pt \
                           --out sample_256.png --n 64

# 2. Fine-tune at 256 to verify your training loop works
python train.py --config configs/baseline_256.yaml \
                       --init-from ckpt/ffhq256_baseline.pt

# 3. Scale up: write a new config (e.g., configs/baseline_1024.yaml),
#    extend the architecture, write your own transfer-init logic, and run
#    train.py with that config.
```

## Architecture (`src/model.py`)

Config-driven ResNet GAN:
- **Generator** 21.2M params: `z(512) → Linear → 4×4 → ResBlockUp×6 → 256×256`,
  Group Norm, self-attention at 32×32, tanh output.
- **Discriminator** 20.2M params: mirror of G with Spectral Norm everywhere,
  MinibatchStd, no normalization layer in residual blocks.
- **EMA**: half-life 10k images. `G_ema_state` is what you sample from for FID.

Extending to 512 or 1024 is a config change only — see below.

## Scaling to 512 / 1024

This is the core of the assignment — designing the additional up-block(s)
that take 256→512 (and 512→1024), and the matching down-block(s) on D.
Decide on your own:

- **Block design.** ResBlockUp-style (NN-upsample + Conv + Conv)? Sub-pixel
  conv? Transposed conv? Something else? `model.py` ships the baseline's
  ResBlockUp / ResBlockDown, but you're not required to reuse them — the
  assignment grades the resulting FID, not the architecture.
- **Channels.** How many channels at 512 / 1024? Halving each step
  (...256:64, 512:32, 1024:16) is a sensible default but not the only choice.


`train.py --init-from ffhq256_baseline.pt` loads the baseline with
`strict=True` — it only works when your architecture is *exactly* the 256
baseline. As soon as you change the architecture, replace `init_from_baseline`
(in `train.py`) with your own loader.

For the leaderboard submission, your trained model only has to satisfy the
ONNX interface in `export_onnx.py` (input: `(B, 512)` z, output:
`(B, 3, 1024, 1024)` image). Anything in between is up to you.

## Training recipe (and why)

The settings in `baseline_256.yaml` were arrived at after **three divergences**
during the baseline run. Lessons:

| Setting | Value | Lesson |
|---|---|---|
| `beta2` | **0.9** | 0.99 averages too long — when a gradient spike hits, Adam takes too long to adapt and the run blows up. |
| `lr_g`, `lr_d` | both **1e-3** | TTUR (lower D lr) caused D under-training and mode collapse. Symmetric lr was stable. |
| `r1_gamma` | **10** | Higher γ (20, 30) suppressed D learning too much. |
| `augment` | `color,translation` | DiffAug **cutout 50%** was too aggressive — masked-out regions starved D. |
| `precision` | **fp32** | bf16 trained fine for ~3M images then a late spike was easier to diagnose in fp32. Either works. |
| `grad_clip_d` | 100 (effectively off) | D has Spectral Norm — already bounded; clipping is a no-op. |
| `grad_clip_g` | 10 | Real protection on G — has caught grad spikes without distorting training. |

### Measuring FID

The leaderboard ranks by FID, so you may want a number to track. 

- Dump a few thousand samples from your model (via `generate.py` in
  a loop, or directly from the ONNX session) into a directory.
- Use `pytorch-fid` (`pip install pytorch-fid`) on that directory vs a
  directory of real images at the same resolution: `python -m pytorch_fid
  <samples_dir> <real_dir>`. Cache the real-side Inception statistics with
  `--save-stats` so subsequent FID runs only re-extract the fake side.
- The leaderboard uses the same `pytorch-fid` Inception features, so this is
  your honest self-check before submission.


## Inference / sampling

```bash
# from the slim baseline ckpt
python generate.py --ckpt ckpt/ffhq256_baseline.pt --n 64 --out grid.png

# from your own fine-tune ckpt (auto-detects architecture from meta)
python generate.py --ckpt runs/my_run/ckpt_001000000.pt --n 64

# without EMA (raw G — usually noticeably worse)
python generate.py --ckpt ckpt/ffhq256_baseline.pt --no-ema --n 64
```

## Leaderboard submission (ONNX export)

Every leaderboard entry exports a single ONNX file with this fixed interface:

```
input  z      shape (B, 512), dtype float32
output image  shape (B, 3, 1024, 1024), dtype float32, range [-1, 1]
```

The `SubmissionWrapper` in `export_onnx.py` runs your Generator and
resizes the output to 1024×1024 with bilinear interpolation — so 256-, 512-,
and 1024-native models all submit through the same contract. The grader
doesn't need to know your architecture.

For the baseline 256 (sanity check the pipeline):

```bash
python export_onnx.py --ckpt ckpt/ffhq256_baseline.pt \
                              --out submission.onnx
```

For your own fine-tuned model (any architecture), call the Python API from
the package root:

```python
import torch
from src.model import Generator, GeneratorConfig
from export_onnx import export_to_onnx

G = Generator(GeneratorConfig(z_dim=512, resolutions=..., channels=..., ...))
G.load_state_dict(torch.load("my_ckpt.pt", weights_only=True)["G_ema_state"])
export_to_onnx(G, "submission.onnx")
```

Verify locally with onnxruntime before submitting:

```python
import numpy as np, onnxruntime as ort
sess = ort.InferenceSession("submission.onnx")
out = sess.run(None, {"z": np.random.randn(4, 512).astype(np.float32)})[0]
assert out.shape == (4, 3, 1024, 1024)
```

## Resuming your own run

`train.py --resume` restores G/D/G_ema/optimizers/RNG/wandb run id, so an
interrupted run continues bit-for-bit:

```bash
python train.py --config configs/baseline_256.yaml \
                       --resume runs/my_run/ckpt_001000000.pt
```

Do not mix `--init-from` and `--resume` — `--init-from` is for the *first*
launch of a fine-tune, `--resume` is for continuing an in-progress one.
