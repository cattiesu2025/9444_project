"""Optimizer, epoch schedule and complete training-state persistence."""
import hashlib
import math
import random
from pathlib import Path
import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def isolated_batches(loader):
    """Data transforms/iterator creation must not reseed model dropout RNGs.

    A's transforms use torch.manual_seed; preserve accelerator RNGs as well as
    CPU RNGs when num_workers=0. Also makes mid-epoch resume deterministic.
    """
    state = rng_state()
    try:
        iterator = iter(loader)
    finally:
        restore_rng(state)
    while True:
        state = rng_state()
        try:
            batch = next(iterator)
        except StopIteration:
            return
        finally:
            restore_rng(state)
        yield batch


def make_optimizer(model, cfg):
    if cfg["optimizer"] != "AdamW":
        raise ValueError("Baseline supports AdamW only")
    groups = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            multiplier = cfg["pretrained_lr_factor"] if name.startswith(("base.blocks.", "base.patch_embed.proj")) else 1.
            groups.append({"params": [parameter], "lr": cfg["learning_rate"] * multiplier,
                           "weight_decay": cfg["weight_decay"]})
    return torch.optim.AdamW(groups)


class EpochSchedule:
    """Author schedule: step(epoch_one_based) BEFORE training that epoch.

    Warmup does not shift the cosine origin; common absolute minimum for all
    parameter groups, including the half-rate pretrained groups.
    """
    def __init__(self, optimizer, cfg):
        self.optimizer = optimizer
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.epochs = cfg["epochs"]
        self.warmup = cfg["warmup_epochs"]
        self.initial = cfg["learning_rate"] * cfg["warmup_lr_factor"]
        self.minimum = cfg["learning_rate"] * cfg["min_lr_factor"]
        self.epoch = 0

    def step(self, epoch):
        self.epoch = epoch
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            if epoch < self.warmup:
                lr = self.initial + epoch * (base - self.initial) / self.warmup
            elif epoch >= self.epochs:
                lr = self.minimum
            else:
                lr = self.minimum + .5 * (base - self.minimum) * (1 + math.cos(math.pi * epoch / self.epochs))
            group["lr"] = lr

    def state_dict(self):
        return {k: v for k, v in vars(self).items() if k != "optimizer"}

    def load_state_dict(self, state):
        for key, value in state.items():
            if key not in self.state_dict():
                raise ValueError(f"Unknown scheduler state key {key}")
            setattr(self, key, value)
        self.step(self.epoch)


def scaled_optimizer_step(loss, model, optimizer, scaler):
    """Let GradScaler skip overflowing updates and lower its scale safely."""
    scale_before = scaler.get_scale()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    if not finite and not scaler.is_enabled():
        raise FloatingPointError("Nonfinite gradients without AMP; last checkpoint retained")
    scaler.step(optimizer)  # Skips the optimizer update if unscale_ found inf/nan.
    scaler.update()
    return bool(finite), scale_before, scaler.get_scale()


def save_checkpoint(path, model, optimizer, scheduler, scaler, sampler, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata, model=model.state_dict(), optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                   sampler=sampler.state_dict(), rng=rng_state(), format_version=1)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    checksum = sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(checksum + "\n")
    return checksum


def load_checkpoint(path):
    # Full resume includes Python/NumPy RNG state. Only load locally trusted runs.
    return torch.load(path, map_location="cpu", weights_only=False)


def restore_training(checkpoint, model, optimizer, scheduler, scaler, sampler):
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    sampler.load_state_dict(checkpoint["sampler"])
    restore_rng(checkpoint["rng"])
