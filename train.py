"""Fine-tune script

Two start modes:

1) Fine-tune from the distributed 256 baseline (most common):
       python scripts/train.py --config configs/baseline_256.yaml \
                               --init-from ckpt/ffhq256_baseline.pt

2) Resume your own training run from a full ckpt you saved earlier:
       python scripts/train.py --config configs/baseline_256.yaml \
                               --resume runs/my_run/ckpt_001000000.pt

   `--resume` restores G, D, G_ema, both optimizers, and RNG state — bit-for-bit
   continuation (assuming the same architecture).

Recipe (the one that worked after three divergences):
- ResNet GAN: GN on G, Spectral Norm on D, self-attention at 32×32
- Non-saturating logistic loss + R1 (lazy every 16 D steps, γ=10)
- DiffAug 'color,translation' (cutout disabled — too aggressive)
- Adam β=(0, 0.9), G lr = D lr = 1e-3 (avoid TTUR until you observe a problem)
- EMA G (half-life 10k images)
- fp32 throughout

Logging via wandb if installed and not disabled.
FID is intentionally not measured here — measure it yourself between checkpoints.
"""
from __future__ import annotations

import argparse
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import yaml
from torch.utils.data import DataLoader

# wandb is optional — keep training runnable on environments without it.
try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    wandb = None
    _HAS_WANDB = False

from src.augment import diff_augment
from src.dataset import ZipImageDataset, infinite_loader
from src.losses import ns_logistic_g, r1_penalty
from src.model import (
    Discriminator,
    DiscriminatorConfig,
    EMA,
    Generator,
    GeneratorConfig,
)


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(path: Path, state: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def async_save_checkpoint(path: Path, state: dict) -> threading.Thread:
    t = threading.Thread(target=save_checkpoint, args=(path, state), daemon=False)
    t.start()
    return t


@torch.no_grad()
def save_sample_grid(G: torch.nn.Module, sample_z: torch.Tensor, out_path: Path, nrow: int = 8) -> None:
    G.eval()
    fake = G(sample_z)
    x = ((fake + 1.0) / 2.0).clamp(0.0, 1.0)
    grid = vutils.make_grid(x, nrow=nrow, padding=2)
    vutils.save_image(grid, out_path)


def build_checkpoint(
    *,
    images_seen: int,
    step: int,
    G: torch.nn.Module,
    D: torch.nn.Module,
    G_ema: EMA,
    optG: torch.optim.Optimizer,
    optD: torch.optim.Optimizer,
    g_cfg: GeneratorConfig,
    d_cfg: DiscriminatorConfig,
    training_cfg: dict,
    wandb_run_id: str | None,
) -> dict:
    return {
        "images_seen": images_seen,
        "step": step,
        "G_state": G.state_dict(),
        "D_state": D.state_dict(),
        "G_ema_state": G_ema.state_dict(),
        "optG_state": optG.state_dict(),
        "optD_state": optD.state_dict(),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
        },
        "wandb_run_id": wandb_run_id,
        "meta": {
            "generator_config": asdict(g_cfg),
            "discriminator_config": asdict(d_cfg),
            "training_config": training_cfg,
        },
    }


def init_from_baseline(
    init_path: Path,
    G: torch.nn.Module,
    D: torch.nn.Module,
    G_ema: EMA,
    device: str,
) -> None:
    """Strict load of G / D / G_ema from a baseline ckpt.

    Works out of the box when your architecture matches the baseline 256.

    If you scale the architecture (add 512 / 1024 blocks, change channels,
    swap the up-block design, etc.), this will raise — and that's intentional.
    The transfer-learning recipe (which keys carry over, how to remap the
    discriminator's reverse-ordered stage indices, what to do with the
    last block's shape mismatch) is part of the assignment. Replace this
    function or write your own loader before scaling.
    """
    print(f"Initializing from baseline: {init_path}")
    ckpt = torch.load(init_path, map_location=device, weights_only=True)
    G.load_state_dict(ckpt["G_state"])
    D.load_state_dict(ckpt["D_state"])
    G_ema.load_state_dict(ckpt["G_ema_state"])
    print("  Loaded G, D, G_ema (strict)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--init-from", type=Path, default=None,
        help="Path to a (possibly slim) baseline ckpt. Partial load with "
             "strict=False; optimizers/RNG start fresh.",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Path to a full ckpt saved by this same script. Restores "
             "G/D/G_ema/optimizers/RNG/wandb run id.",
    )
    parser.add_argument("--total-images", type=int, default=None)
    parser.add_argument(
        "--new-wandb-run", action="store_true",
        help="When --resume, start a fresh wandb run instead of reattaching.",
    )
    args = parser.parse_args()

    if args.init_from is not None and args.resume is not None:
        raise SystemExit("Use either --init-from or --resume, not both.")

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    if args.total_images is not None:
        train_cfg["total_images"] = args.total_images

    set_seed(train_cfg["seed"])
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    g_cfg = GeneratorConfig.from_dict(cfg["generator"])
    d_cfg = DiscriminatorConfig.from_dict(cfg["discriminator"])
    G = Generator(g_cfg).to(device)
    D = Discriminator(d_cfg).to(device)
    print(f"Generator: {sum(p.numel() for p in G.parameters())/1e6:.2f}M params")
    print(f"Discriminator: {sum(p.numel() for p in D.parameters())/1e6:.2f}M params")

    lr_g = float(train_cfg.get("lr_g", train_cfg.get("lr")))
    lr_d = float(train_cfg.get("lr_d", train_cfg.get("lr")))
    optG = torch.optim.Adam(
        G.parameters(), lr=lr_g,
        betas=(train_cfg["beta1"], train_cfg["beta2"]), eps=1e-8,
        weight_decay=train_cfg["weight_decay"],
    )
    optD = torch.optim.Adam(
        D.parameters(), lr=lr_d,
        betas=(train_cfg["beta1"], train_cfg["beta2"]), eps=1e-8,
        weight_decay=train_cfg["weight_decay"],
    )
    print(f"Optimizers: G lr={lr_g}, D lr={lr_d}")

    G_ema = EMA(G, half_life=train_cfg["ema_half_life"])
    G_ema.shadow.to(device)

    dataset = ZipImageDataset(train_cfg["train_zip"], flip=train_cfg["flip"])
    print(f"Dataset: {len(dataset)} images")
    num_workers = train_cfg["num_workers"]
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
    )
    inf_loader = infinite_loader(loader)

    sample_gen = torch.Generator(device="cpu").manual_seed(train_cfg["sample_seed"])
    sample_z = torch.randn(train_cfg["sample_n"], g_cfg.z_dim, generator=sample_gen).to(device)

    run_dir = Path(cfg["out"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    images_seen = 0
    step = 0
    wandb_run_id: str | None = None

    if args.init_from is not None:
        init_from_baseline(args.init_from, G, D, G_ema, device=device)

    if args.resume is not None:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        G.load_state_dict(ckpt["G_state"])
        D.load_state_dict(ckpt["D_state"])
        G_ema.load_state_dict(ckpt["G_ema_state"])
        if "optG_state" in ckpt:
            optG.load_state_dict(ckpt["optG_state"])
        if "optD_state" in ckpt:
            optD.load_state_dict(ckpt["optD_state"])
        # Force yaml LR onto the loaded optimizer state.
        for pg in optG.param_groups:
            pg["lr"] = lr_g
        for pg in optD.param_groups:
            pg["lr"] = lr_d
        images_seen = ckpt.get("images_seen", 0)
        step = ckpt.get("step", 0)
        wandb_run_id = None if args.new_wandb_run else ckpt.get("wandb_run_id")
        rng = ckpt.get("rng_state", {})
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"].cpu())
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])

    # wandb
    wandb_cfg = cfg.get("wandb", {})
    wandb_mode = wandb_cfg.get("mode", "online") if _HAS_WANDB else "disabled"
    run = None
    if wandb_mode != "disabled":
        init_kwargs = {
            "project": wandb_cfg.get("project", "ffhqgen-student"),
            "name": wandb_cfg.get("name"),
            "mode": wandb_mode,
            "config": cfg,
        }
        if wandb_run_id is not None:
            init_kwargs["id"] = wandb_run_id
            init_kwargs["resume"] = "must"
        run = wandb.init(**init_kwargs)
        wandb_run_id = run.id

    total_images = train_cfg["total_images"]
    z_dim = g_cfg.z_dim
    r1_gamma = train_cfg["r1_gamma"]
    r1_lazy_every = train_cfg["r1_lazy_every"]
    log_every = train_cfg["log_every"]
    ckpt_every = train_cfg["ckpt_every"]
    grad_clip_g = float(train_cfg.get("grad_clip_g", float("inf")))
    grad_clip_d = float(train_cfg.get("grad_clip_d", float("inf")))
    precision = train_cfg.get("precision", "fp32")
    if precision not in ("bf16", "fp32"):
        raise ValueError(f"precision must be 'bf16' or 'fp32', got {precision!r}")
    use_amp = precision == "bf16"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    print(f"Precision: {precision} ({'autocast bf16' if use_amp else 'fp32 throughout'})")
    augment_policy = train_cfg.get("augment", "") or ""
    print(f"Augment policy: {augment_policy!r}")

    last_ckpt = images_seen
    save_threads: list[threading.Thread] = []
    window_t0 = time.perf_counter()
    window_imgs = 0
    last_r1_value: float | None = None

    print(
        f"Training: images_seen={images_seen} → {total_images} "
        f"(batch={train_cfg['batch_size']}, device={device})"
    )

    while images_seen < total_images:
        real = next(inf_loader).to(device, non_blocking=True)
        b = real.size(0)

        # --- D step ---
        z = torch.randn(b, z_dim, device=device)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            with torch.no_grad():
                fake = G(z)
            d_real = D(diff_augment(real, augment_policy))
            d_fake = D(diff_augment(fake.detach(), augment_policy))
            l_d_real = F.softplus(-d_real).mean()
            l_d_fake = F.softplus(d_fake).mean()
            l_d = l_d_real + l_d_fake
        optD.zero_grad(set_to_none=True)
        l_d.backward()

        if (step + 1) % r1_lazy_every == 0:
            l_r1 = r1_lazy_every * r1_penalty(
                D, diff_augment(real.float(), augment_policy), gamma=r1_gamma,
            )
            l_r1.backward()
            last_r1_value = float(l_r1.item()) / r1_lazy_every

        grad_norm_d = float(
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=grad_clip_d)
        )
        optD.step()

        # --- G step ---
        z = torch.randn(b, z_dim, device=device)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            fake = G(z)
            d_fake_g = D(diff_augment(fake, augment_policy))
            l_g = ns_logistic_g(d_fake_g)
        optG.zero_grad(set_to_none=True)
        l_g.backward()
        grad_norm_g = float(
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=grad_clip_g)
        )
        optG.step()

        G_ema.update(G, b)

        images_seen += b
        window_imgs += b
        step += 1

        if step % log_every == 0:
            now = time.perf_counter()
            elapsed = max(now - window_t0, 1e-6)
            throughput = window_imgs / elapsed
            window_t0 = now
            window_imgs = 0
            log = {
                "images_seen": images_seen,
                "throughput/imgs_per_sec": throughput,
                "loss/D_total": float(l_d.item()),
                "loss/D_real": float(l_d_real.item()),
                "loss/D_fake": float(l_d_fake.item()),
                "loss/G": float(l_g.item()),
                "D_out/real_mean": float(d_real.float().mean().item()),
                "D_out/fake_mean": float(d_fake.float().mean().item()),
                "grad_norm/G": grad_norm_g,
                "grad_norm/D": grad_norm_d,
                "lr": optG.param_groups[0]["lr"],
            }
            if last_r1_value is not None:
                log["loss/R1"] = last_r1_value
            if wandb_mode != "disabled":
                wandb.log(log, step=step)
            else:
                print(
                    f"step={step} imgs={images_seen} thr={throughput:.1f}img/s "
                    f"l_d={l_d.item():.3f} l_g={l_g.item():.3f} "
                    f"gn_g={grad_norm_g:.2f} gn_d={grad_norm_d:.2f}"
                )

        if images_seen - last_ckpt >= ckpt_every:
            ckpt = build_checkpoint(
                images_seen=images_seen, step=step,
                G=G, D=D, G_ema=G_ema, optG=optG, optD=optD,
                g_cfg=g_cfg, d_cfg=d_cfg, training_cfg=train_cfg,
                wandb_run_id=wandb_run_id,
            )
            ckpt_path = run_dir / f"ckpt_{images_seen:09d}.pt"
            grid_path = samples_dir / f"grid_{images_seen:09d}.png"
            save_threads = [t for t in save_threads if t.is_alive()]
            save_threads.append(async_save_checkpoint(ckpt_path, ckpt))
            save_sample_grid(G_ema.shadow, sample_z, grid_path, nrow=8)
            if wandb_mode != "disabled":
                wandb.log({"samples/grid": wandb.Image(str(grid_path))}, step=step)
            print(f"[ckpt+grid] {ckpt_path.name} / {grid_path.name}")
            last_ckpt = images_seen

    print("Training complete. Saving final ckpt...")
    final_ckpt = build_checkpoint(
        images_seen=images_seen, step=step,
        G=G, D=D, G_ema=G_ema, optG=optG, optD=optD,
        g_cfg=g_cfg, d_cfg=d_cfg, training_cfg=train_cfg,
        wandb_run_id=wandb_run_id,
    )
    save_checkpoint(run_dir / "final.pt", final_ckpt)
    for t in save_threads:
        t.join()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
