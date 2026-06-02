"""Two-stage Refiner training.

Phase 1 — train Refiner512 (256→512):
    python train_refiner.py --phase 1 \\
        --config configs/refiner_512.yaml \\
        --g256-ckpt /path/to/ffhq256_baseline.pt

Phase 2 — train Refiner1024 (512→1024), Refiner512 frozen:
    python train_refiner.py --phase 2 \\
        --config configs/refiner_1024.yaml \\
        --g256-ckpt /path/to/ffhq256_baseline.pt \\
        --r512-ckpt runs/refiner_512/final.pt

Resume:
    python train_refiner.py --phase 1 \\
        --config configs/refiner_512.yaml \\
        --resume runs/refiner_512/ckpt_000050000.pt
"""
from __future__ import annotations

import argparse
import copy
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
    build_baseline_256_generator,
)
from src.refiner import (
    Refiner512, Refiner512Config,
    Refiner1024, Refiner1024Config,
)


# ---------------------------------------------------------------------------
# Helpers
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


def make_ema(module: torch.nn.Module) -> torch.nn.Module:
    ema = copy.deepcopy(module).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(ema: torch.nn.Module, module: torch.nn.Module, batch_size: int, half_life: int) -> None:
    decay = 0.5 ** (batch_size / half_life)
    for sp, p in zip(ema.parameters(), module.parameters()):
        sp.mul_(decay).add_(p.detach(), alpha=1.0 - decay)


@torch.no_grad()
def save_sample_grid(
    G256: torch.nn.Module,
    r512_ema: torch.nn.Module,
    r1024_ema: torch.nn.Module | None,
    sample_z: torch.Tensor,
    out_path: Path,
    nrow: int = 4,
) -> None:
    G256.eval()
    r512_ema.eval()
    img = r512_ema(G256(sample_z))
    if r1024_ema is not None:
        r1024_ema.eval()
        img = r1024_ema(img)
    x = ((img + 1.0) / 2.0).clamp(0.0, 1.0)
    vutils.save_image(vutils.make_grid(x, nrow=nrow, padding=2), out_path)


def build_checkpoint(*, images_seen, step, refiner, D, refiner_ema,
                     optR, optD, r_cfg, d_cfg, train_cfg, wandb_run_id,
                     phase, extra_meta=None) -> dict:
    def _snap(sd):
        return {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                for k, v in sd.items()}
    state = {
        "phase":        phase,
        "images_seen":  images_seen,
        "step":         step,
        "refiner_state":  _snap(refiner.state_dict()),
        "D_state":        _snap(D.state_dict()),
        "refiner_ema_state": _snap(refiner_ema.state_dict()),
        "optR_state":   optR.state_dict(),
        "optD_state":   optD.state_dict(),
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
            **(extra_meta or {}),
        },
    }
    return state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",    required=True, type=int, choices=[1, 2])
    parser.add_argument("--config",   required=True, type=Path)
    parser.add_argument("--g256-ckpt", type=Path, default=None)
    parser.add_argument("--r512-ckpt", type=Path, default=None,
                        help="Phase 2 only: frozen Refiner512 checkpoint")
    parser.add_argument("--resume",   type=Path, default=None)
    parser.add_argument("--total-images", type=int, default=None)
    parser.add_argument("--new-wandb-run", action="store_true")
    args = parser.parse_args()

    cfg       = load_config(args.config)
    train_cfg = cfg["training"]
    if args.total_images is not None:
        train_cfg["total_images"] = args.total_images

    set_seed(train_cfg["seed"])
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- G_256 (always frozen) -----------------------------------------
    G256 = build_baseline_256_generator().to(device).eval()
    g256_path = args.g256_ckpt
    if g256_path is None and args.resume:
        g256_path = Path(torch.load(args.resume, map_location="cpu",
                                    weights_only=False)["meta"]["training_config"]["g256_ckpt"])
    if g256_path is None:
        raise SystemExit("Provide --g256-ckpt (or --resume with embedded path).")
    g_state = torch.load(g256_path, map_location=device, weights_only=True)
    G256.load_state_dict(g_state["G_ema_state"])
    for p in G256.parameters():
        p.requires_grad_(False)
    n_g256 = sum(p.numel() for p in G256.parameters())
    print(f"G_256: {n_g256/1e6:.2f}M (frozen)")

    # ---- Frozen Refiner512 (phase 2 only) --------------------------------
    frozen_r512 = None
    if args.phase == 2:
        r512_path = args.r512_ckpt
        if r512_path is None and args.resume:
            r512_path = Path(torch.load(args.resume, map_location="cpu",
                                        weights_only=False)["meta"]["training_config"]["r512_ckpt"])
        if r512_path is None:
            raise SystemExit("Phase 2 requires --r512-ckpt.")
        r512_ckpt = torch.load(r512_path, map_location=device, weights_only=False)
        r512_cfg  = Refiner512Config(**r512_ckpt["meta"]["refiner_config"])
        frozen_r512 = Refiner512(r512_cfg).to(device).eval()
        frozen_r512.load_state_dict(r512_ckpt["refiner_ema_state"])
        for p in frozen_r512.parameters():
            p.requires_grad_(False)
        n_r512 = sum(p.numel() for p in frozen_r512.parameters())
        print(f"Refiner512: {n_r512/1e6:.2f}M (frozen)")

    # ---- Trainable Refiner ---------------------------------------------
    if args.phase == 1:
        r_cfg   = Refiner512Config.from_dict(cfg.get("refiner", {}))
        refiner = Refiner512(r_cfg).to(device)
    else:
        r_cfg   = Refiner1024Config.from_dict(cfg.get("refiner", {}))
        refiner = Refiner1024(r_cfg).to(device)

    refiner_ema = make_ema(refiner)
    n_ref = sum(p.numel() for p in refiner.parameters())
    print(f"Refiner (phase {args.phase}): {n_ref/1e6:.2f}M")

    # param limit check
    n_frozen_r512 = sum(p.numel() for p in frozen_r512.parameters()) if frozen_r512 else 0
    total_g = n_g256 + n_frozen_r512 + n_ref
    print(f"Total Generator params: {total_g/1e6:.2f}M")
    if total_g > 40e6:
        raise ValueError(f"Total {total_g/1e6:.2f}M exceeds 40M limit!")

    # ---- Discriminator -------------------------------------------------
    d_cfg = DiscriminatorConfig.from_dict(cfg["discriminator"])
    D     = Discriminator(d_cfg).to(device)
    print(f"D: {sum(p.numel() for p in D.parameters())/1e6:.2f}M")

    # ---- Optimizers ----------------------------------------------------
    lr_r = float(train_cfg.get("lr_r", train_cfg.get("lr", 1e-3)))
    lr_d = float(train_cfg.get("lr_d", train_cfg.get("lr", 1e-3)))
    b1, b2 = train_cfg["beta1"], train_cfg["beta2"]
    wd     = train_cfg.get("weight_decay", 0.0)
    optR = torch.optim.Adam(refiner.parameters(), lr=lr_r, betas=(b1, b2), eps=1e-8, weight_decay=wd)
    optD = torch.optim.Adam(D.parameters(),       lr=lr_d, betas=(b1, b2), eps=1e-8, weight_decay=wd)

    # ---- Dataset -------------------------------------------------------
    dataset = ZipImageDataset(train_cfg["train_zip"], flip=train_cfg.get("flip", True))
    loader  = DataLoader(
        dataset, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=(device == "cuda"),
        persistent_workers=(train_cfg.get("num_workers", 4) > 0),
        prefetch_factor=2, drop_last=True,
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
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        refiner.load_state_dict(ckpt["refiner_state"])
        D.load_state_dict(ckpt["D_state"])
        refiner_ema.load_state_dict(ckpt["refiner_ema_state"])
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
        if rng.get("numpy"):  np.random.set_state(rng["numpy"])
        if rng.get("python"): random.setstate(rng["python"])

    # ---- WandB ---------------------------------------------------------
    wandb_cfg  = cfg.get("wandb", {})
    wandb_mode = wandb_cfg.get("mode", "disabled") if _HAS_WANDB else "disabled"
    run = None
    if wandb_mode != "disabled":
        if wandb_cfg.get("login", True) and wandb_mode == "online":
            import os
            key = os.environ.get(wandb_cfg.get("api_key_env", "WANDB_API_KEY"), "")
            if key: wandb.login(key=key, relogin=False)
        init_kw = {
            "project": wandb_cfg.get("project", "ffhqgen-student"),
            "name":    wandb_cfg.get("name"),
            "mode":    wandb_mode,
            "config":  cfg,
        }
        if wandb_cfg.get("entity"): init_kw["entity"] = wandb_cfg["entity"]
        if wandb_run_id:
            init_kw["id"] = wandb_run_id
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
    consist_w      = float(train_cfg.get("consistency_weight", 0.5))
    ema_half_life  = train_cfg["ema_half_life"]
    train_cfg["g256_ckpt"] = str(g256_path)
    if args.phase == 2:
        train_cfg["r512_ckpt"] = str(args.r512_ckpt or
                                     torch.load(args.resume, map_location="cpu",
                                                weights_only=False)["meta"]["training_config"]["r512_ckpt"])

    extra_meta = {}
    if args.phase == 1:
        extra_meta["refiner_config"] = asdict(r_cfg)
    else:
        extra_meta["refiner1024_config"] = asdict(r_cfg)
        extra_meta["refiner512_config"]  = asdict(frozen_r512.cfg) if frozen_r512 else {}

    print(f"Phase {args.phase} | {images_seen}→{total_images} imgs | "
          f"augment={augment_policy!r} | consistency_w={consist_w}")

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

        # Upstream pipeline (all frozen)
        with torch.no_grad():
            z      = torch.randn(b, 512, device=device)
            img256 = G256(z)
            input_img = frozen_r512(img256) if frozen_r512 else img256

        # ---- D step ----------------------------------------------------
        with torch.no_grad():
            fake = refiner(input_img)

        d_real = D(diff_augment(real,         augment_policy))
        d_fake = D(diff_augment(fake.detach(), augment_policy))
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
        z      = torch.randn(b, 512, device=device)
        with torch.no_grad():
            img256    = G256(z)
            input_img = frozen_r512(img256) if frozen_r512 else img256

        fake     = refiner(input_img)
        d_fake_r = D(diff_augment(fake, augment_policy))
        l_adv    = ns_logistic_g(d_fake_r)

        # Consistency: downsample output should match input
        fake_down = F.interpolate(fake, size=input_img.shape[-2:],
                                  mode="bilinear", align_corners=False)
        l_consist = F.l1_loss(fake_down, input_img.detach())

        l_r = l_adv + consist_w * l_consist

        optR.zero_grad(set_to_none=True)
        l_r.backward()
        grad_norm_r = float(torch.nn.utils.clip_grad_norm_(refiner.parameters(), grad_clip_r))
        optR.step()

        update_ema(refiner_ema, refiner, b, ema_half_life)

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
                print(f"[P{args.phase}] step={step} imgs={images_seen} "
                      f"thr={throughput:.0f}img/s l_d={l_d.item():.3f} "
                      f"l_adv={l_adv.item():.3f} l_cons={l_consist.item():.4f} "
                      f"gn_r={grad_norm_r:.3f} gn_d={grad_norm_d:.3f}")

        if images_seen - last_ckpt >= ckpt_every:
            for t in save_threads: t.join()
            save_threads = []
            ckpt_state = build_checkpoint(
                images_seen=images_seen, step=step,
                refiner=refiner, D=D, refiner_ema=refiner_ema,
                optR=optR, optD=optD, r_cfg=r_cfg, d_cfg=d_cfg,
                train_cfg=train_cfg, wandb_run_id=wandb_run_id,
                phase=args.phase, extra_meta=extra_meta,
            )
            ckpt_path = run_dir / f"ckpt_{images_seen:09d}.pt"
            grid_path = samples_dir / f"grid_{images_seen:09d}.png"
            save_threads.append(async_save(ckpt_path, ckpt_state))
            save_sample_grid(G256, refiner_ema if args.phase == 1 else frozen_r512,
                             refiner_ema if args.phase == 2 else None,
                             sample_z, grid_path, nrow=4)
            if wandb_mode != "disabled":
                wandb.log({"samples/grid": wandb.Image(str(grid_path))}, step=step)
            print(f"[ckpt] {ckpt_path.name}  [grid] {grid_path.name}")
            last_ckpt = images_seen

    print("Training complete.")
    for t in save_threads: t.join()
    final_state = build_checkpoint(
        images_seen=images_seen, step=step,
        refiner=refiner, D=D, refiner_ema=refiner_ema,
        optR=optR, optD=optD, r_cfg=r_cfg, d_cfg=d_cfg,
        train_cfg=train_cfg, wandb_run_id=wandb_run_id,
        phase=args.phase, extra_meta=extra_meta,
    )
    save_checkpoint(run_dir / "final.pt", final_state)
    if run is not None: run.finish()


if __name__ == "__main__":
    main()
