"""Cascade SR-Refiner GAN training script.

Two training phases
-------------------
Phase 1 — 256 → 512
  python train.py --phase 1 \\
    --config configs/phase1_sr512.yaml \\
    --backbone-ckpt /content/drive/MyDrive/project2/ckpt/ffhq256_baseline.pt

Phase 2 — 512 → 1024
  python train.py --phase 2 \\
    --config configs/phase2_sr1024.yaml \\
    --backbone-ckpt /content/drive/MyDrive/project2/ckpt/ffhq256_baseline.pt \\
    --sr512-ckpt    /content/drive/MyDrive/project2claude/runs/phase1_sr512/final.pt

Resume any interrupted run:
  python train.py --phase 1 \\
    --config configs/phase1_sr512.yaml \\
    --backbone-ckpt ... \\
    --resume /content/drive/MyDrive/project2claude/runs/phase1_sr512/ckpt_000050000.pt
"""
from __future__ import annotations

import argparse
import copy
import os
import random
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

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    wandb = None
    _HAS_WANDB = False

from src.augment import diff_augment
from src.dataset import ZipImageDataset, infinite_loader
from src.losses import ns_logistic_g, ragan_d, ragan_g, r1_penalty
from src.model import (
    BackboneConfig,
    SRRefinerConfig,
    DiscriminatorConfig,
    Generator,
    Discriminator,
    PatchDiscriminator,
    EMA,
)


# =============================================================================
# Utilities
# =============================================================================

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def snapshot_for_save(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: snapshot_for_save(v) for k, v in value.items()}
    if isinstance(value, list):
        return [snapshot_for_save(v) for v in value]
    if isinstance(value, tuple):
        return tuple(snapshot_for_save(v) for v in value)
    return copy.deepcopy(value)


def save_checkpoint(path: Path, state: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


class _SaveThread(threading.Thread):
    def __init__(self, path: Path, state: dict):
        super().__init__(daemon=False)
        self.path  = path
        self.state = state
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            save_checkpoint(self.path, self.state)
        except BaseException as exc:
            self.error = exc

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"Checkpoint save failed: {self.path}") from self.error


def async_save_checkpoint(path: Path, state: dict) -> _SaveThread:
    t = _SaveThread(path, state)
    t.start()
    return t


def wait_for_saves(threads: list[_SaveThread]) -> None:
    for t in threads:
        t.join()
        t.raise_if_failed()


def build_checkpoint(
    *,
    phase: int,
    images_seen: int,
    step: int,
    G: Generator,
    D: Discriminator,
    ema: EMA,
    optG: torch.optim.Optimizer,
    optD: torch.optim.Optimizer,
    cfg_dict: dict,
    wandb_run_id: str | None,
) -> dict:
    # Save only the refiner being trained (backbone is always reloaded separately)
    active_refiner_state = (
        G.sr_512.state_dict() if phase == 1 else G.sr_1024.state_dict()
    )
    state = {
        "phase":              phase,
        "images_seen":        images_seen,
        "step":               step,
        "active_refiner_state": active_refiner_state,
        "ema_state":          ema.state_dict(),
        "D_state":            D.state_dict(),
        "optG_state":         optG.state_dict(),
        "optD_state":         optD.state_dict(),
        "rng_state": {
            "torch":  torch.get_rng_state(),
            "cuda":   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy":  np.random.get_state(),
            "python": random.getstate(),
        },
        "wandb_run_id": wandb_run_id,
        "meta":         cfg_dict,
    }
    return snapshot_for_save(state)


@torch.no_grad()
def save_sample_grid(
    G: Generator,
    ema: EMA,
    phase: int,
    sample_z: torch.Tensor,
    out_path: Path,
    nrow: int = 4,
) -> None:
    """Swap in EMA weights, generate samples, restore training weights."""
    active = G.sr_512 if phase == 1 else G.sr_1024
    original = copy.deepcopy(active.state_dict())
    active.load_state_dict(ema.state_dict())

    was_training = G.training
    G.eval()
    forward_fn = G.forward_phase1 if phase == 1 else G.forward_phase2
    fake = forward_fn(sample_z)
    x    = ((fake + 1.0) / 2.0).clamp(0.0, 1.0)
    vutils.save_image(vutils.make_grid(x, nrow=nrow, padding=2), out_path)

    active.load_state_dict(original)
    G.train()
    G.backbone.eval()
    if phase == 2:
        G.sr_512.eval()
    G.train(was_training)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",         required=True, type=int, choices=[1, 2])
    parser.add_argument("--config",        required=True, type=Path)
    parser.add_argument("--backbone-ckpt", required=True, type=Path,
                        help="Path to 256 baseline checkpoint.")
    parser.add_argument("--sr512-ckpt",    type=Path, default=None,
                        help="[Phase 2 only] checkpoint from Phase 1 (contains sr_512 weights).")
    parser.add_argument("--resume",        type=Path, default=None,
                        help="Resume from a checkpoint saved by this script.")
    parser.add_argument("--total-images",  type=int,  default=None)
    parser.add_argument("--new-wandb-run", action="store_true")
    args = parser.parse_args()

    if args.phase == 2 and args.sr512_ckpt is None and args.resume is None:
        raise SystemExit("Phase 2 requires --sr512-ckpt (or --resume).")

    # --- Config ---
    cfg       = load_config(args.config)
    train_cfg = cfg["training"]
    if args.total_images is not None:
        train_cfg["total_images"] = args.total_images

    set_seed(train_cfg["seed"])
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Build model ---
    backbone_cfg    = BackboneConfig.from_dict(cfg["backbone"])
    refiner512_cfg  = SRRefinerConfig.from_dict(cfg["refiner_512"])
    refiner1024_cfg = SRRefinerConfig.from_dict(cfg["refiner_1024"])

    G = Generator(backbone_cfg, refiner512_cfg, refiner1024_cfg).to(device)

    # Discriminator: 'patch' or 'full'
    disc_type = cfg.get("discriminator", {}).get("type", "full")
    if disc_type == "patch":
        use_sn = cfg["discriminator"].get("use_spectral_norm", True)
        D = PatchDiscriminator(use_spectral_norm=use_sn).to(device)
        d_cfg = None
        print("Discriminator: PatchGAN (70x70)")
    else:
        d_cfg = DiscriminatorConfig.from_dict(cfg["discriminator"])
        D = Discriminator(d_cfg).to(device)
        print("Discriminator: Full-image ResNet-D")

    # Load & freeze backbone
    G.backbone.load_from_baseline(str(args.backbone_ckpt), device=device)
    G.freeze_backbone()

    # Phase 2: load sr_512 weights and freeze
    if args.phase == 2 and args.sr512_ckpt is not None:
        print(f"Loading SR_512 weights from: {args.sr512_ckpt}")
        sr_ckpt = torch.load(args.sr512_ckpt, map_location=device, weights_only=False)
        # Accept both "active_refiner_state" (our format) or "sr_512_state"
        sr_state = sr_ckpt.get("active_refiner_state", sr_ckpt.get("sr_512_state"))
        if sr_state is None:
            raise KeyError("sr512 checkpoint has no 'active_refiner_state' key")
        G.sr_512.load_state_dict(sr_state)
        print("  SR_512 loaded OK")
    if args.phase == 2:
        G.freeze_sr512()

    # Active refiner & EMA
    active_refiner = G.sr_512 if args.phase == 1 else G.sr_1024
    ema = EMA(active_refiner, half_life=train_cfg["ema_half_life"])

    # Print param counts
    n_bb    = sum(p.numel() for p in G.backbone.parameters())
    n_sr512 = sum(p.numel() for p in G.sr_512.parameters())
    n_sr1024= sum(p.numel() for p in G.sr_1024.parameters())
    n_total = n_bb + n_sr512 + n_sr1024
    n_d     = sum(p.numel() for p in D.parameters())
    print(f"BackboneGenerator : {n_bb/1e6:.2f}M  (frozen)")
    print(f"SRRefiner_512     : {n_sr512/1e6:.2f}M  {'(frozen)' if args.phase == 2 else '(trainable)'}")
    print(f"SRRefiner_1024    : {n_sr1024/1e6:.2f}M  {'(trainable)' if args.phase == 2 else '(unused)'}")
    print(f"Generator total   : {n_total/1e6:.2f}M")
    print(f"Discriminator     : {n_d/1e6:.2f}M")
    if n_total > 40e6:
        raise ValueError(f"Generator {n_total/1e6:.2f}M exceeds 40M limit!")

    # Loss type
    loss_type = train_cfg.get("loss_type", "ns_logistic")
    print(f"Loss type: {loss_type}")

    # --- Optimizers ---
    lr_g = float(train_cfg.get("lr_g", train_cfg.get("lr", 2e-4)))
    lr_d = float(train_cfg.get("lr_d", train_cfg.get("lr", 2e-4)))
    optG = torch.optim.Adam(
        active_refiner.parameters(), lr=lr_g,
        betas=(train_cfg["beta1"], train_cfg["beta2"]), eps=1e-8,
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    optD = torch.optim.Adam(
        D.parameters(), lr=lr_d,
        betas=(train_cfg["beta1"], train_cfg["beta2"]), eps=1e-8,
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    print(f"Optimizers: lr_g={lr_g}, lr_d={lr_d}")

    # --- Dataset ---
    dataset     = ZipImageDataset(train_cfg["train_zip"], flip=train_cfg.get("flip", True))
    print(f"Dataset: {len(dataset)} images")
    num_workers = train_cfg.get("num_workers", 4)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
    )
    inf_loader = infinite_loader(loader)
    resolution = int(train_cfg["resolution"])

    sample_gen = torch.Generator(device="cpu").manual_seed(train_cfg["sample_seed"])
    sample_z   = torch.randn(
        train_cfg["sample_n"], backbone_cfg.z_dim, generator=sample_gen
    ).to(device)

    run_dir     = Path(cfg["out"]["run_dir"])
    samples_dir = run_dir / "samples"
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(exist_ok=True)

    # --- Init / resume ---
    images_seen  = 0
    step         = 0
    wandb_run_id: str | None = None

    if args.resume is not None:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        active_refiner.load_state_dict(ckpt["active_refiner_state"])
        ema.load_state_dict(ckpt["ema_state"])
        D.load_state_dict(ckpt["D_state"])
        if "optG_state" in ckpt:
            optG.load_state_dict(ckpt["optG_state"])
        if "optD_state" in ckpt:
            optD.load_state_dict(ckpt["optD_state"])
        for pg in optG.param_groups:
            pg["lr"] = lr_g
        for pg in optD.param_groups:
            pg["lr"] = lr_d
        images_seen  = ckpt.get("images_seen", 0)
        step         = ckpt.get("step", 0)
        wandb_run_id = None if args.new_wandb_run else ckpt.get("wandb_run_id")
        rng = ckpt.get("rng_state", {})
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"].cpu())
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("python") is not None:
            random.setstate(rng["python"])

    # --- W&B ---
    wandb_cfg  = cfg.get("wandb", {})
    wandb_mode = wandb_cfg.get("mode", "disabled") if _HAS_WANDB else "disabled"
    run        = None
    if wandb_mode != "disabled":
        if wandb_cfg.get("login", True) and wandb_mode == "online":
            api_key = os.environ.get(wandb_cfg.get("api_key_env", "WANDB_API_KEY"), "")
            if api_key:
                wandb.login(key=api_key, relogin=False)
                print("wandb: logged in via env var")
            else:
                print("wandb: no API key — prompting for login")
                wandb.login()
        init_kw: dict = {
            "project": wandb_cfg.get("project", "ffhqgen-student"),
            "name":    wandb_cfg.get("name"),
            "mode":    wandb_mode,
            "config":  cfg,
        }
        if wandb_cfg.get("entity"):
            init_kw["entity"] = wandb_cfg["entity"]
        if wandb_run_id is not None:
            init_kw["id"]     = wandb_run_id
            init_kw["resume"] = "must"
        run          = wandb.init(**init_kw)
        wandb_run_id = run.id

    # --- Hyper-params ---
    total_images   = train_cfg["total_images"]
    z_dim          = backbone_cfg.z_dim
    r1_gamma       = train_cfg["r1_gamma"]
    r1_lazy_every  = train_cfg["r1_lazy_every"]
    log_every      = train_cfg["log_every"]
    ckpt_every     = train_cfg["ckpt_every"]
    grad_clip_g    = float(train_cfg.get("grad_clip_g", float("inf")))
    grad_clip_d    = float(train_cfg.get("grad_clip_d", float("inf")))
    augment_policy = train_cfg.get("augment", "") or ""
    precision      = train_cfg.get("precision", "fp32")
    use_amp        = (precision == "bf16")
    amp_dtype      = torch.bfloat16 if use_amp else torch.float32

    forward_fn = G.forward_phase1 if args.phase == 1 else G.forward_phase2

    print(f"Phase {args.phase} | resolution={resolution} | augment={augment_policy!r}")
    print(f"Training: {images_seen} -> {total_images} (batch={train_cfg['batch_size']}, device={device})")

    last_ckpt     = images_seen
    save_threads: list[_SaveThread] = []
    window_t0     = time.perf_counter()
    window_imgs   = 0
    last_r1_value: float | None = None

    # =========================================================================
    G.train()
    G.backbone.eval()
    if args.phase == 2:
        G.sr_512.eval()

    while images_seen < total_images:
        real = next(inf_loader).to(device, non_blocking=True)
        b    = real.size(0)

        if real.shape[-2:] != (resolution, resolution):
            raise ValueError(
                f"Dataset images must be {resolution}x{resolution}, "
                f"got {tuple(real.shape[-2:])}."
            )

        # --- D step ---
        z = torch.randn(b, z_dim, device=device)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            fake   = forward_fn(z)
            d_real = D(diff_augment(real, augment_policy))
            d_fake = D(diff_augment(fake.detach(), augment_policy))
            if loss_type == "ragan":
                l_d      = ragan_d(d_real, d_fake)
                l_d_real = F.binary_cross_entropy_with_logits(
                    d_real - d_fake.mean(), torch.ones_like(d_real))
                l_d_fake = F.binary_cross_entropy_with_logits(
                    d_fake - d_real.mean(), torch.zeros_like(d_fake))
            else:
                l_d_real = F.softplus(-d_real).mean()
                l_d_fake = F.softplus(d_fake).mean()
                l_d      = l_d_real + l_d_fake

        # save d_real for G step (RaGAN needs it)
        d_real_detached = d_real.detach()

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

        # --- G (active refiner) step ---
        z = torch.randn(b, z_dim, device=device)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            fake_g   = forward_fn(z)
            d_fake_g = D(diff_augment(fake_g, augment_policy))
            if loss_type == "ragan":
                l_g = ragan_g(d_real_detached, d_fake_g)
            else:
                l_g = ns_logistic_g(d_fake_g)

        optG.zero_grad(set_to_none=True)
        l_g.backward()
        grad_norm_g = float(
            torch.nn.utils.clip_grad_norm_(active_refiner.parameters(), max_norm=grad_clip_g)
        )
        optG.step()
        ema.update(active_refiner, b)

        images_seen += b
        window_imgs += b
        step        += 1

        # --- Logging ---
        if step % log_every == 0:
            now        = time.perf_counter()
            elapsed    = max(now - window_t0, 1e-6)
            throughput = window_imgs / elapsed
            window_t0  = now
            window_imgs = 0
            log = {
                "phase":                   args.phase,
                "images_seen":             images_seen,
                "throughput/imgs_per_sec": throughput,
                "loss/D_total":            float(l_d.item()),
                "loss/D_real":             float(l_d_real.item()),
                "loss/D_fake":             float(l_d_fake.item()),
                "loss/G":                  float(l_g.item()),
                "D_out/real_mean":         float(d_real.float().mean().item()),
                "D_out/fake_mean":         float(d_fake.float().mean().item()),
                "grad_norm/G":             grad_norm_g,
                "grad_norm/D":             grad_norm_d,
                "lr_g":                    optG.param_groups[0]["lr"],
            }
            if last_r1_value is not None:
                log["loss/R1"] = last_r1_value

            if wandb_mode != "disabled":
                wandb.log(log, step=step)
            else:
                print(
                    f"[P{args.phase}] step={step} imgs={images_seen} "
                    f"thr={throughput:.0f}img/s "
                    f"l_d={l_d.item():.3f} l_g={l_g.item():.3f} "
                    f"gn_g={grad_norm_g:.2f} gn_d={grad_norm_d:.2f}"
                )

        # --- Checkpoint ---
        if images_seen - last_ckpt >= ckpt_every:
            wait_for_saves(save_threads)
            save_threads = []
            ckpt_state = build_checkpoint(
                phase=args.phase, images_seen=images_seen, step=step,
                G=G, D=D, ema=ema, optG=optG, optD=optD,
                cfg_dict=cfg, wandb_run_id=wandb_run_id,
            )
            ckpt_path = run_dir / f"ckpt_{images_seen:09d}.pt"
            grid_path = samples_dir / f"grid_{images_seen:09d}.png"
            save_threads.append(async_save_checkpoint(ckpt_path, ckpt_state))
            save_sample_grid(G, ema, args.phase, sample_z, grid_path, nrow=4)
            if wandb_mode != "disabled":
                wandb.log({"samples/grid": wandb.Image(str(grid_path))}, step=step)
            print(f"[ckpt] {ckpt_path.name}  [grid] {grid_path.name}")
            last_ckpt = images_seen

    # --- Final save ---
    print("Training complete. Saving final checkpoint...")
    wait_for_saves(save_threads)
    final_state = build_checkpoint(
        phase=args.phase, images_seen=images_seen, step=step,
        G=G, D=D, ema=ema, optG=optG, optD=optD,
        cfg_dict=cfg, wandb_run_id=wandb_run_id,
    )
    save_checkpoint(run_dir / "final.pt", final_state)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
