"""Residual Progressive GAN training script.

Phase 1 — 512 residual  (backbone frozen, 512 stage + residual head trained)
  python train.py --phase 1 \\
    --config configs/phase1_residual512.yaml \\
    --init-from ckpt/ffhq256_baseline.pt

Phase 2 — 1024 residual  (backbone + 512 frozen, 1024 stage trained)
  python train.py --phase 2 \\
    --config configs/phase2_residual1024.yaml \\
    --init-from runs/phase1_residual512/final.pt

Resume:
  python train.py --phase 1 \\
    --config configs/phase1_residual512.yaml \\
    --resume runs/phase1_residual512/ckpt_000050000.pt
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

import torch.nn as nn

from src.augment import diff_augment
from src.dataset import ZipImageDataset, infinite_loader
from src.losses import ns_logistic_g, r1_penalty
from src.model import Generator, GeneratorConfig, Discriminator, DiscriminatorConfig, EMA


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
    *, phase: int, images_seen: int, step: int,
    G: Generator, D: Discriminator, G_ema: EMA,
    optG: torch.optim.Optimizer, optD: torch.optim.Optimizer,
    g_cfg: GeneratorConfig, d_cfg: DiscriminatorConfig,
    training_cfg: dict, wandb_run_id: str | None,
) -> dict:
    state = {
        "phase":       phase,
        "images_seen": images_seen,
        "step":        step,
        "G_state":     G.state_dict(),
        "D_state":     D.state_dict(),
        "G_ema_state": G_ema.state_dict(),
        "optG_state":  optG.state_dict(),
        "optD_state":  optD.state_dict(),
        "rng_state": {
            "torch":  torch.get_rng_state(),
            "cuda":   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy":  np.random.get_state(),
            "python": random.getstate(),
        },
        "wandb_run_id": wandb_run_id,
        "meta": {
            "generator_config":     asdict(g_cfg),
            "discriminator_config": asdict(d_cfg),
            "training_config":      training_cfg,
        },
    }
    return snapshot_for_save(state)


@torch.no_grad()
def save_sample_grid(
    G_ema: EMA, sample_z: torch.Tensor, out_path: Path, nrow: int = 8
) -> None:
    was_training = G_ema.shadow.training
    G_ema.shadow.eval()
    fake = G_ema.shadow(sample_z)
    x    = ((fake + 1.0) / 2.0).clamp(0.0, 1.0)
    vutils.save_image(vutils.make_grid(x, nrow=nrow, padding=2), out_path)
    G_ema.shadow.train(was_training)


# =============================================================================
# Checkpoint loading
# =============================================================================

def init_from_checkpoint(
    init_path: Path, G: Generator, D: Discriminator, G_ema: EMA, device: str
) -> None:
    """Partial-match load: copy tensors whose name AND shape agree."""

    def _is_exact(module: nn.Module, source: dict) -> bool:
        target = module.state_dict()
        return (
            len(target) == len(source)
            and all(k in source and source[k].shape == v.shape for k, v in target.items())
        )

    def _load_partial(module: nn.Module, source: dict, label: str) -> None:
        target  = module.state_dict()
        matched = {k: v for k, v in source.items() if k in target and target[k].shape == v.shape}
        target.update(matched)
        module.load_state_dict(target)
        print(f"  {label}: {len(matched)}/{len(source)} tensors matched")

    print(f"Loading weights from: {init_path}")
    ckpt = torch.load(init_path, map_location=device, weights_only=False)

    g_state     = ckpt["G_state"]
    g_ema_state = ckpt.get("G_ema_state", g_state)
    d_state     = ckpt["D_state"]

    if _is_exact(G, g_state):
        G.load_state_dict(g_state)
        G_ema.load_state_dict(g_ema_state)
        print("  G, G_ema: exact match")
    else:
        _load_partial(G,           g_state,     "G     (partial)")
        _load_partial(G_ema.shadow, g_ema_state, "G_ema (partial)")

    if _is_exact(D, d_state):
        D.load_state_dict(d_state)
        print("  D: exact match")
    else:
        print("  D: architecture changed — random init")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",        required=True, type=int, choices=[1, 2])
    parser.add_argument("--config",       required=True, type=Path)
    parser.add_argument("--init-from",    type=Path, default=None,
                        help="Partial-load weights (optimizers reset).")
    parser.add_argument("--resume",       type=Path, default=None,
                        help="Full resume: G/D/G_ema/optimizers/RNG.")
    parser.add_argument("--total-images", type=int, default=None)
    parser.add_argument("--new-wandb-run", action="store_true")
    args = parser.parse_args()

    if args.init_from and args.resume:
        raise SystemExit("Use --init-from or --resume, not both.")

    cfg       = load_config(args.config)
    train_cfg = cfg["training"]
    if args.total_images is not None:
        train_cfg["total_images"] = args.total_images

    set_seed(train_cfg["seed"])
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    g_cfg = GeneratorConfig.from_dict(cfg["generator"])
    d_cfg = DiscriminatorConfig.from_dict(cfg["discriminator"])

    resolution = int(train_cfg["resolution"])
    if g_cfg.resolutions[-1] != resolution or d_cfg.resolutions[0] != resolution:
        raise ValueError(
            f"training.resolution={resolution} must match G output "
            f"({g_cfg.resolutions[-1]}) and D input ({d_cfg.resolutions[0]})"
        )

    G     = Generator(g_cfg).to(device)
    D     = Discriminator(d_cfg).to(device)
    G_ema = EMA(G, half_life=train_cfg["ema_half_life"])
    G_ema.shadow.to(device)

    n_g = sum(p.numel() for p in G.parameters())
    n_d = sum(p.numel() for p in D.parameters())
    print(f"Generator:     {n_g/1e6:.2f}M params")
    print(f"Discriminator: {n_d/1e6:.2f}M params")
    if n_g > 40e6:
        raise ValueError(f"Generator {n_g/1e6:.2f}M exceeds 40M limit!")

    lr_g = float(train_cfg.get("lr_g", train_cfg.get("lr", 1e-3)))
    lr_d = float(train_cfg.get("lr_d", train_cfg.get("lr", 1e-3)))
    beta1, beta2 = train_cfg["beta1"], train_cfg["beta2"]
    wd = train_cfg.get("weight_decay", 0.0)

    optG = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(beta1, beta2), eps=1e-8, weight_decay=wd)
    optD = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(beta1, beta2), eps=1e-8, weight_decay=wd)
    print(f"Optimizers: lr_g={lr_g}, lr_d={lr_d}")

    dataset     = ZipImageDataset(train_cfg["train_zip"], flip=train_cfg.get("flip", True))
    num_workers = train_cfg.get("num_workers", 4)
    print(f"Dataset: {len(dataset)} images")
    loader = DataLoader(
        dataset, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None, drop_last=True,
    )
    inf_loader = infinite_loader(loader)

    sample_gen = torch.Generator(device="cpu").manual_seed(train_cfg["sample_seed"])
    sample_z   = torch.randn(train_cfg["sample_n"], g_cfg.z_dim, generator=sample_gen).to(device)

    run_dir     = Path(cfg["out"]["run_dir"])
    samples_dir = run_dir / "samples"
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(exist_ok=True)

    images_seen  = 0
    step         = 0
    wandb_run_id: str | None = None

    if args.init_from:
        init_from_checkpoint(args.init_from, G, D, G_ema, device=device)

    # Load G/D weights first (optimizer state loaded after freeze+rebuild below)
    resume_ckpt = None
    if args.resume:
        print(f"Resuming from {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        G.load_state_dict(resume_ckpt["G_state"])
        D.load_state_dict(resume_ckpt["D_state"])
        G_ema.load_state_dict(resume_ckpt["G_ema_state"])

    # Apply freeze before rebuilding optimizer so param groups are correct
    freeze_cfg = train_cfg.get("freeze_resolutions", [])
    needs_freeze = bool(freeze_cfg) or train_cfg.get("freeze_backbone", False)

    if freeze_cfg:
        G.freeze_up_to([int(r) for r in freeze_cfg])
    elif train_cfg.get("freeze_backbone", False):
        G.freeze_backbone()

    # Rebuild optG with only trainable params whenever freeze is active.
    # This must happen before loading optimizer state so param groups match.
    if needs_freeze:
        optG = torch.optim.Adam(
            filter(lambda p: p.requires_grad, G.parameters()),
            lr=lr_g, betas=(beta1, beta2), eps=1e-8, weight_decay=wd,
        )

    # Now load optimizer state (param groups already match frozen layout)
    if resume_ckpt is not None:
        if "optG_state" in resume_ckpt:
            optG.load_state_dict(resume_ckpt["optG_state"])
        if "optD_state" in resume_ckpt:
            optD.load_state_dict(resume_ckpt["optD_state"])
        for pg in optG.param_groups:
            pg["lr"] = lr_g
        for pg in optD.param_groups:
            pg["lr"] = lr_d
        images_seen  = resume_ckpt.get("images_seen", 0)
        step         = resume_ckpt.get("step", 0)
        wandb_run_id = None if args.new_wandb_run else resume_ckpt.get("wandb_run_id")
        rng = resume_ckpt.get("rng_state", {})
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"].cpu())
        if torch.cuda.is_available() and rng.get("cuda"):
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        if rng.get("numpy"):
            np.random.set_state(rng["numpy"])
        if rng.get("python"):
            random.setstate(rng["python"])

    # W&B
    wandb_cfg  = cfg.get("wandb", {})
    wandb_mode = wandb_cfg.get("mode", "disabled") if _HAS_WANDB else "disabled"
    run        = None
    if wandb_mode != "disabled":
        if wandb_cfg.get("login", True) and wandb_mode == "online":
            api_key = os.environ.get(wandb_cfg.get("api_key_env", "WANDB_API_KEY"), "")
            if api_key:
                wandb.login(key=api_key, relogin=False)
            else:
                wandb.login()
        init_kw: dict = {
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

    total_images          = train_cfg["total_images"]
    z_dim                 = g_cfg.z_dim
    r1_gamma              = train_cfg["r1_gamma"]
    r1_lazy_every         = train_cfg["r1_lazy_every"]
    log_every             = train_cfg["log_every"]
    ckpt_every            = train_cfg["ckpt_every"]
    grad_clip_g           = float(train_cfg.get("grad_clip_g", float("inf")))
    grad_clip_d           = float(train_cfg.get("grad_clip_d", float("inf")))
    residual_l2_weight    = float(train_cfg.get("residual_rgb_l2_weight", 0.0))
    augment_policy        = train_cfg.get("augment", "") or ""
    precision             = train_cfg.get("precision", "fp32")
    use_amp               = (precision == "bf16")
    amp_dtype             = torch.bfloat16 if use_amp else torch.float32

    print(f"Phase {args.phase} | resolution={resolution} | augment={augment_policy!r}")
    print(f"residual_rgb_l2_weight={residual_l2_weight}")
    print(f"Training: {images_seen} -> {total_images} (batch={train_cfg['batch_size']}, device={device})")

    last_ckpt     = images_seen
    save_threads: list[_SaveThread] = []
    window_t0     = time.perf_counter()
    window_imgs   = 0
    last_r1_value: float | None = None

    # =========================================================================
    G.train()

    while images_seen < total_images:
        G.set_training_progress(images_seen)

        real = next(inf_loader).to(device, non_blocking=True)
        b    = real.size(0)

        if real.shape[-2:] != (resolution, resolution):
            raise ValueError(f"Dataset must be {resolution}x{resolution}, got {tuple(real.shape[-2:])}")

        # --- D step ---
        z = torch.randn(b, z_dim, device=device)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            with torch.no_grad():
                fake = G(z)
            d_real   = D(diff_augment(real, augment_policy))
            d_fake   = D(diff_augment(fake.detach(), augment_policy))
            l_d_real = F.softplus(-d_real).mean()
            l_d_fake = F.softplus(d_fake).mean()
            l_d      = l_d_real + l_d_fake

        optD.zero_grad(set_to_none=True)
        l_d.backward()

        if (step + 1) % r1_lazy_every == 0:
            l_r1 = r1_lazy_every * r1_penalty(
                D, diff_augment(real.float(), augment_policy), gamma=r1_gamma,
            )
            l_r1.backward()
            last_r1_value = float(l_r1.item()) / r1_lazy_every

        grad_norm_d = float(torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=grad_clip_d))
        optD.step()

        # --- G step ---
        z = torch.randn(b, z_dim, device=device)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            fake, g_aux  = G.forward_with_aux(z)
            l_g_residual = g_aux["residual_l2"]
            d_fake_g     = D(diff_augment(fake, augment_policy))
            l_g_adv      = ns_logistic_g(d_fake_g)
            l_g          = l_g_adv + residual_l2_weight * l_g_residual

        optG.zero_grad(set_to_none=True)
        l_g.backward()
        grad_norm_g = float(
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, G.parameters()), max_norm=grad_clip_g
            )
        )
        optG.step()
        G_ema.update(G, b)

        images_seen += b
        window_imgs += b
        step        += 1

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
                "loss/G_adv":              float(l_g_adv.item()),
                "loss/G_residual_l2":      float(l_g_residual.item()),
                "residual_rgb/rms":        float(l_g_residual.sqrt().item()),
                "residual_rgb/fade":       float(G._fade.item()),
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
                    f"fade={G._fade.item():.2f} rms={l_g_residual.sqrt().item():.4f} "
                    f"gn_g={grad_norm_g:.3f} gn_d={grad_norm_d:.3f}"
                )

        if images_seen - last_ckpt >= ckpt_every:
            wait_for_saves(save_threads)
            save_threads = []
            ckpt_state = build_checkpoint(
                phase=args.phase, images_seen=images_seen, step=step,
                G=G, D=D, G_ema=G_ema, optG=optG, optD=optD,
                g_cfg=g_cfg, d_cfg=d_cfg, training_cfg=train_cfg,
                wandb_run_id=wandb_run_id,
            )
            ckpt_path = run_dir / f"ckpt_{images_seen:09d}.pt"
            grid_path = samples_dir / f"grid_{images_seen:09d}.png"
            save_threads.append(async_save_checkpoint(ckpt_path, ckpt_state))
            save_sample_grid(G_ema, sample_z, grid_path, nrow=8)
            if wandb_mode != "disabled":
                wandb.log({"samples/grid": wandb.Image(str(grid_path))}, step=step)
            print(f"[ckpt] {ckpt_path.name}  [grid] {grid_path.name}")
            last_ckpt = images_seen

    print("Training complete. Saving final checkpoint...")
    wait_for_saves(save_threads)
    final_state = build_checkpoint(
        phase=args.phase, images_seen=images_seen, step=step,
        G=G, D=D, G_ema=G_ema, optG=optG, optD=optD,
        g_cfg=g_cfg, d_cfg=d_cfg, training_cfg=train_cfg,
        wandb_run_id=wandb_run_id,
    )
    save_checkpoint(run_dir / "final.pt", final_state)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
