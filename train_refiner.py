"""Refiner training: frozen G_256 → Refiner → 1024×1024.

Usage:
    python train_refiner.py \\
        --config configs/refiner_1024.yaml \\
        --g256-ckpt /path/to/ffhq256_baseline.pt

Resume:
    python train_refiner.py \\
        --config configs/refiner_1024.yaml \\
        --resume runs/refiner_1024/ckpt_000050000.pt
"""
from __future__ import annotations

import argparse
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
from src.losses import ns_logistic_g, r1_penalty
from src.model import (
    Discriminator, DiscriminatorConfig,
    Generator, GeneratorConfig, EMA,
    build_baseline_256_generator,
)
from src.refiner import Refiner, RefinerConfig


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(path: Path, state: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def async_save(path: Path, state: dict) -> threading.Thread:
    t = threading.Thread(target=save_checkpoint, args=(path, state), daemon=False)
    t.start()
    return t


@torch.no_grad()
def save_sample_grid(
    G256: Generator, refiner: Refiner, sample_z: torch.Tensor,
    out_path: Path, nrow: int = 4,
) -> None:
    G256.eval()
    refiner.eval()
    img256  = G256(sample_z)
    img1024 = refiner(img256)
    x = ((img1024 + 1.0) / 2.0).clamp(0.0, 1.0)
    vutils.save_image(vutils.make_grid(x, nrow=nrow, padding=2), out_path)
    refiner.train()


def build_checkpoint(
    *, images_seen: int, step: int,
    refiner: Refiner, D: Discriminator, R_ema: Refiner,
    optR: torch.optim.Optimizer, optD: torch.optim.Optimizer,
    r_cfg: RefinerConfig, d_cfg: DiscriminatorConfig,
    train_cfg: dict, wandb_run_id: str | None,
) -> dict:
    import copy
    def _snap(v):
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().clone()
        return v

    return {
        "images_seen":   images_seen,
        "step":          step,
        "refiner_state": {k: _snap(v) for k, v in refiner.state_dict().items()},
        "D_state":       {k: _snap(v) for k, v in D.state_dict().items()},
        "R_ema_state":   {k: _snap(v) for k, v in R_ema.state_dict().items()},
        "optR_state":    optR.state_dict(),
        "optD_state":    optD.state_dict(),
        "rng_state": {
            "torch":  torch.get_rng_state(),
            "cuda":   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy":  np.random.get_state(),
            "python": random.getstate(),
        },
        "wandb_run_id": wandb_run_id,
        "meta": {
            "refiner_config":       asdict(r_cfg),
            "discriminator_config": asdict(d_cfg),
            "training_config":      train_cfg,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    required=True, type=Path)
    parser.add_argument("--g256-ckpt", type=Path, default=None,
                        help="Path to ffhq256_baseline.pt (required unless --resume)")
    parser.add_argument("--resume",    type=Path, default=None)
    parser.add_argument("--total-images", type=int, default=None)
    parser.add_argument("--new-wandb-run", action="store_true")
    args = parser.parse_args()

    if args.g256_ckpt is None and args.resume is None:
        raise SystemExit("Provide --g256-ckpt or --resume.")

    cfg       = load_config(args.config)
    train_cfg = cfg["training"]
    if args.total_images is not None:
        train_cfg["total_images"] = args.total_images

    set_seed(train_cfg["seed"])
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- G_256 (frozen) ------------------------------------------------
    G256 = build_baseline_256_generator().to(device)
    g256_path = args.g256_ckpt
    if g256_path is None and args.resume is not None:
        # load from resume meta
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        g256_path = Path(resume_ckpt["meta"]["training_config"]["g256_ckpt"])
    print(f"Loading G_256 from {g256_path}")
    g256_ckpt = torch.load(g256_path, map_location=device, weights_only=True)
    G256.load_state_dict(g256_ckpt["G_ema_state"])
    G256.eval()
    for p in G256.parameters():
        p.requires_grad_(False)
    n_g256 = sum(p.numel() for p in G256.parameters())
    print(f"G_256: {n_g256/1e6:.2f}M params (frozen)")

    # ---- Refiner -------------------------------------------------------
    r_cfg   = RefinerConfig.from_dict(cfg.get("refiner", {}))
    refiner = Refiner(r_cfg).to(device)
    n_ref   = sum(p.numel() for p in refiner.parameters())
    print(f"Refiner: {n_ref/1e6:.2f}M params")
    total_g = n_g256 + n_ref
    print(f"Total Generator (G_256 + Refiner): {total_g/1e6:.2f}M params")
    if total_g > 40e6:
        raise ValueError(f"Total generator {total_g/1e6:.2f}M exceeds 40M limit!")

    # EMA of Refiner
    import copy
    R_ema        = copy.deepcopy(refiner).eval()
    ema_half_life = train_cfg["ema_half_life"]
    for p in R_ema.parameters():
        p.requires_grad_(False)

    # ---- Discriminator at 1024 ----------------------------------------
    d_cfg = DiscriminatorConfig.from_dict(cfg["discriminator"])
    D     = Discriminator(d_cfg).to(device)
    n_d   = sum(p.numel() for p in D.parameters())
    print(f"D_1024: {n_d/1e6:.2f}M params")

    # ---- Optimizers ----------------------------------------------------
    lr_r = float(train_cfg.get("lr_r", train_cfg.get("lr", 1e-3)))
    lr_d = float(train_cfg.get("lr_d", train_cfg.get("lr", 1e-3)))
    b1, b2, wd = train_cfg["beta1"], train_cfg["beta2"], train_cfg.get("weight_decay", 0.0)
    optR = torch.optim.Adam(refiner.parameters(), lr=lr_r, betas=(b1, b2), eps=1e-8, weight_decay=wd)
    optD = torch.optim.Adam(D.parameters(),       lr=lr_d, betas=(b1, b2), eps=1e-8, weight_decay=wd)

    # ---- Dataset -------------------------------------------------------
    dataset     = ZipImageDataset(train_cfg["train_zip"], flip=train_cfg.get("flip", True))
    num_workers = train_cfg.get("num_workers", 4)
    loader = DataLoader(
        dataset, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None, drop_last=True,
    )
    inf_loader = infinite_loader(loader)
    print(f"Dataset: {len(dataset)} images")

    sample_gen = torch.Generator("cpu").manual_seed(train_cfg["sample_seed"])
    sample_z   = torch.randn(train_cfg["sample_n"], 512, generator=sample_gen).to(device)

    run_dir     = Path(cfg["out"]["run_dir"])
    samples_dir = run_dir / "samples"
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(exist_ok=True)

    images_seen  = 0
    step         = 0
    wandb_run_id: str | None = None

    # ---- Resume --------------------------------------------------------
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        refiner.load_state_dict(ckpt["refiner_state"])
        D.load_state_dict(ckpt["D_state"])
        R_ema.load_state_dict(ckpt["R_ema_state"])
        optR.load_state_dict(ckpt["optR_state"])
        optD.load_state_dict(ckpt["optD_state"])
        for pg in optR.param_groups: pg["lr"] = lr_r
        for pg in optD.param_groups: pg["lr"] = lr_d
        images_seen  = ckpt.get("images_seen", 0)
        step         = ckpt.get("step", 0)
        wandb_run_id = None if args.new_wandb_run else ckpt.get("wandb_run_id")
        rng = ckpt.get("rng_state", {})
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"].cpu())
        if torch.cuda.is_available() and rng.get("cuda"):
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        if rng.get("numpy"):
            np.random.set_state(rng["numpy"])
        if rng.get("python"):
            random.setstate(rng["python"])

    # ---- WandB ---------------------------------------------------------
    wandb_cfg  = cfg.get("wandb", {})
    wandb_mode = wandb_cfg.get("mode", "disabled") if _HAS_WANDB else "disabled"
    run = None
    if wandb_mode != "disabled":
        if wandb_cfg.get("login", True) and wandb_mode == "online":
            api_key = __import__("os").environ.get(wandb_cfg.get("api_key_env", "WANDB_API_KEY"), "")
            if api_key:
                wandb.login(key=api_key, relogin=False)
        init_kw = {
            "project": wandb_cfg.get("project", "ffhqgen-student"),
            "name":    wandb_cfg.get("name"),
            "mode":    wandb_mode,
            "config":  cfg,
        }
        if wandb_cfg.get("entity"):
            init_kw["entity"] = wandb_cfg["entity"]
        if wandb_run_id:
            init_kw["id"]     = wandb_run_id
            init_kw["resume"] = "must"
        run          = wandb.init(**init_kw)
        wandb_run_id = run.id

    # ---- Training config -----------------------------------------------
    total_images   = train_cfg["total_images"]
    r1_gamma       = train_cfg["r1_gamma"]
    r1_lazy_every  = train_cfg["r1_lazy_every"]
    log_every      = train_cfg["log_every"]
    ckpt_every     = train_cfg["ckpt_every"]
    grad_clip_r    = float(train_cfg.get("grad_clip_r", float("inf")))
    grad_clip_d    = float(train_cfg.get("grad_clip_d", float("inf")))
    augment_policy = train_cfg.get("augment", "") or ""
    consistency_w  = float(train_cfg.get("consistency_weight", 0.5))

    # Store g256 path in config for resume
    train_cfg["g256_ckpt"] = str(args.g256_ckpt or g256_path)

    print(f"Training refiner: {images_seen} → {total_images} images")
    print(f"consistency_weight={consistency_w}, augment={augment_policy!r}")

    last_ckpt     = images_seen
    save_threads: list[threading.Thread] = []
    window_t0     = time.perf_counter()
    window_imgs   = 0
    last_r1_val: float | None = None

    refiner.train()
    D.train()

    while images_seen < total_images:
        real = next(inf_loader).to(device, non_blocking=True)
        b    = real.size(0)

        # ---- D step ----------------------------------------------------
        with torch.no_grad():
            z        = torch.randn(b, 512, device=device)
            img256   = G256(z)
            img1024  = refiner(img256)

        d_real = D(diff_augment(real,          augment_policy))
        d_fake = D(diff_augment(img1024.detach(), augment_policy))
        l_d    = F.softplus(-d_real).mean() + F.softplus(d_fake).mean()

        optD.zero_grad(set_to_none=True)
        l_d.backward()

        if (step + 1) % r1_lazy_every == 0:
            l_r1 = r1_lazy_every * r1_penalty(
                D, diff_augment(real.float(), augment_policy), gamma=r1_gamma,
            )
            l_r1.backward()
            last_r1_val = float(l_r1.item()) / r1_lazy_every

        grad_norm_d = float(torch.nn.utils.clip_grad_norm_(D.parameters(), grad_clip_d))
        optD.step()

        # ---- Refiner step ----------------------------------------------
        z       = torch.randn(b, 512, device=device)
        img256  = G256(z)
        img1024 = refiner(img256)

        d_fake_r = D(diff_augment(img1024, augment_policy))
        l_adv    = ns_logistic_g(d_fake_r)

        # Consistency: downsampled 1024 should match 256 output
        img1024_down = F.interpolate(img1024, size=(256, 256), mode="bilinear", align_corners=False)
        l_consist    = F.l1_loss(img1024_down, img256.detach())

        l_r = l_adv + consistency_w * l_consist

        optR.zero_grad(set_to_none=True)
        l_r.backward()
        grad_norm_r = float(torch.nn.utils.clip_grad_norm_(refiner.parameters(), grad_clip_r))
        optR.step()

        # EMA update
        decay = 0.5 ** (b / ema_half_life)
        with torch.no_grad():
            for sp, p in zip(R_ema.parameters(), refiner.parameters()):
                sp.mul_(decay).add_(p.detach(), alpha=1.0 - decay)

        images_seen += b
        window_imgs += b
        step        += 1

        if step % log_every == 0:
            now        = time.perf_counter()
            throughput = window_imgs / max(now - window_t0, 1e-6)
            window_t0  = now
            window_imgs = 0
            log = {
                "images_seen":             images_seen,
                "throughput/imgs_per_sec": throughput,
                "loss/D_total":            float(l_d.item()),
                "loss/R_adv":              float(l_adv.item()),
                "loss/R_consist":          float(l_consist.item()),
                "loss/R_total":            float(l_r.item()),
                "D_out/real_mean":         float(d_real.float().mean().item()),
                "D_out/fake_mean":         float(d_fake.float().mean().item()),
                "grad_norm/R":             grad_norm_r,
                "grad_norm/D":             grad_norm_d,
            }
            if last_r1_val is not None:
                log["loss/R1"] = last_r1_val
            if wandb_mode != "disabled":
                wandb.log(log, step=step)
            else:
                print(
                    f"step={step} imgs={images_seen} thr={throughput:.0f}img/s "
                    f"l_d={l_d.item():.3f} l_adv={l_adv.item():.3f} "
                    f"l_cons={l_consist.item():.4f} gn_r={grad_norm_r:.3f}"
                )

        if images_seen - last_ckpt >= ckpt_every:
            for t in save_threads:
                t.join()
            save_threads = []
            ckpt_state = build_checkpoint(
                images_seen=images_seen, step=step,
                refiner=refiner, D=D, R_ema=R_ema,
                optR=optR, optD=optD,
                r_cfg=r_cfg, d_cfg=d_cfg, train_cfg=train_cfg,
                wandb_run_id=wandb_run_id,
            )
            ckpt_path = run_dir / f"ckpt_{images_seen:09d}.pt"
            grid_path = samples_dir / f"grid_{images_seen:09d}.png"
            save_threads.append(async_save(ckpt_path, ckpt_state))
            save_sample_grid(G256, R_ema, sample_z, grid_path, nrow=4)
            if wandb_mode != "disabled":
                wandb.log({"samples/grid": wandb.Image(str(grid_path))}, step=step)
            print(f"[ckpt] {ckpt_path.name}  [grid] {grid_path.name}")
            last_ckpt = images_seen

    print("Training complete.")
    for t in save_threads:
        t.join()
    final_state = build_checkpoint(
        images_seen=images_seen, step=step,
        refiner=refiner, D=D, R_ema=R_ema,
        optR=optR, optD=optD,
        r_cfg=r_cfg, d_cfg=d_cfg, train_cfg=train_cfg,
        wandb_run_id=wandb_run_id,
    )
    save_checkpoint(run_dir / "final.pt", final_state)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
