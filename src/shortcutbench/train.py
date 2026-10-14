"""Shared training loop for every model, plus the shortcut-reliance gate.

One call to `train_run` = one (patch_type, rate, model, seed) cell. The
checkpoint and a JSON record are cached under
  <cache_dir>/checkpoints/<patch_type>/<model>/r<rate>_s<seed>_<hash>.{pt,json}
where <hash> covers every setting that affects training, so changing the
config never silently reuses a stale model.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import build_split
from .device import hardware_info, select_device, sync
from .inject import ShortcutSpec
from .models import build_model
from .seeding import seed_everything

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- augmentation

class NoFlipAugment:
    """On-device random crop with zero padding. Deliberately has no flip."""

    def __init__(self, padding: int):
        self.padding = int(padding)

    def __call__(self, x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        p = self.padding
        if p == 0:
            return x
        b, _, h, w = x.shape
        padded = F.pad(x, (p, p, p, p))
        dy = torch.randint(0, 2 * p + 1, (b,), generator=gen, device="cpu").to(x.device)
        dx = torch.randint(0, 2 * p + 1, (b,), generator=gen, device="cpu").to(x.device)
        rows = (dy[:, None] + torch.arange(h, device=x.device))[:, :, None]      # (B, H, 1)
        cols = (dx[:, None] + torch.arange(w, device=x.device))[:, None, :]      # (B, 1, W)
        bidx = torch.arange(b, device=x.device)[:, None, None]
        return padded.permute(0, 2, 3, 1)[bidx, rows, cols].permute(0, 3, 1, 2)


def build_augment(aug_cfg: dict) -> NoFlipAugment:
    if aug_cfg.get("horizontal_flip", False) or aug_cfg.get("vertical_flip", False):
        raise ValueError("flips are forbidden: they move the corner patch and change the shortcut")
    aug = NoFlipAugment(aug_cfg.get("random_crop_padding", 0))
    assert_no_flip(aug)
    return aug


def assert_no_flip(aug: Callable, trials: int = 20) -> None:
    """Runtime check: a left-dark/right-bright ramp must stay left-dark/right-bright."""
    ramp = torch.linspace(0, 1, 32).view(1, 1, 1, 32).expand(64, 3, 32, 32).contiguous()
    gen = torch.Generator().manual_seed(0)
    for _ in range(trials):
        y = aug(ramp, gen)
        left, right = y[..., :16].mean(dim=(1, 2, 3)), y[..., 16:].mean(dim=(1, 2, 3))
        if not bool((right > left).all()):
            raise AssertionError("augmentation mirrored an image horizontally; flips are forbidden")


# ---------------------------------------------------------------- helpers

def to_device_uint8(images: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().to(device)   # (N, 3, H, W) uint8


def stratified_val_split(labels: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng([seed, 99])
    val = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        val.append(rng.choice(idx, size=int(round(fraction * len(idx))), replace=False))
    val = np.sort(np.concatenate(val))
    train = np.setdiff1d(np.arange(len(labels)), val)
    return train, val


@torch.no_grad()
def predict(model: nn.Module, images_u8: torch.Tensor, batch: int) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(images_u8), batch):
        x = images_u8[i:i + batch].float().div_(255.0)
        preds.append(model(x).argmax(1).cpu())
    return torch.cat(preds).numpy()


def run_key(cfg: dict, patch_type: str, rate: float, model_name: str, seed: int, epochs: int) -> dict:
    return {
        "spec": ShortcutSpec.from_config(cfg, patch_type).key(),
        "rate": round(float(rate), 4), "model": model_name, "seed": int(seed), "epochs": int(epochs),
        "target_class": cfg["data"]["target_class"],
        "model_cfg": cfg["models"][model_name], "opt_cfg": {k: v for k, v in cfg["train"][model_name].items() if k != "epochs"},
        "augment": cfg["train"]["augment"], "batch_size": cfg["train"]["batch_size"],
        "val_fraction": cfg["train"]["val_fraction"],
    }


def run_paths(cfg: dict, key: dict) -> tuple[Path, Path]:
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
    d = Path(cfg["paths"]["cache_dir"]) / "checkpoints" / key["spec"]["patch_type"] / key["model"]
    stem = f"r{key['rate']:.2f}_s{key['seed']}_e{key['epochs']}_{digest}"
    return d / f"{stem}.pt", d / f"{stem}.json"


def make_optimizer(model: nn.Module, opt_cfg: dict) -> torch.optim.Optimizer:
    if opt_cfg["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=opt_cfg["lr"], momentum=opt_cfg["momentum"],
                               weight_decay=opt_cfg["weight_decay"], nesterov=True)
    if opt_cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
    raise ValueError(f"unknown optimizer {opt_cfg['optimizer']!r}")


# ---------------------------------------------------------------- reliance gate

def reliance_metrics(model, splits_u8: dict, labels: dict, target: int, batch: int, min_gap: float) -> dict:
    p_pois = predict(model, splits_u8["test_poisoned"], batch)
    p_clean = predict(model, splits_u8["test_clean"], batch)
    p_trig = predict(model, splits_u8["test_trigger"], batch)
    y = labels["test_clean"]
    t = y == target
    m = {
        "acc_poisoned": float((p_pois == y).mean()),
        "acc_clean": float((p_clean == y).mean()),
        "acc_poisoned_target": float((p_pois[t] == y[t]).mean()),
        "acc_clean_target": float((p_clean[t] == y[t]).mean()),
        # Share of patched NON-target test images the model calls "target".
        "trigger_flip_rate": float((p_trig[~t] == target).mean()),
        # Same images unpatched, for reference (natural confusion with target).
        "clean_nontarget_called_target": float((p_clean[~t] == target).mean()),
    }
    m["reliance_gap"] = m["acc_poisoned_target"] - m["acc_clean_target"]
    m["valid"] = bool(m["reliance_gap"] >= min_gap)
    return m


# ---------------------------------------------------------------- main entry

def train_run(cfg: dict, patch_type: str, rate: float, model_name: str, seed: int,
              epochs: int | None = None, log_fn: Callable[[dict], None] | None = None,
              force: bool = False) -> dict:
    """Train (or load from cache) one cell and return its record."""
    epochs = int(epochs or cfg["train"][model_name]["epochs"])
    key = run_key(cfg, patch_type, rate, model_name, seed, epochs)
    ckpt_path, rec_path = run_paths(cfg, key)
    if rec_path.exists() and ckpt_path.exists() and not force:
        rec = json.loads(rec_path.read_text())
        rec["cached"] = True
        log.info("cached: %s", rec_path.name)
        return rec

    device = select_device()
    augment = build_augment(cfg["train"]["augment"])
    target = int(cfg["data"]["target_class"])
    tcfg = cfg["train"]
    t_start = time.perf_counter()

    train_split = build_split(cfg, "train", rate=rate, seed=seed, patch_type=patch_type)
    tests = {s: build_split(cfg, s, seed=seed, patch_type=patch_type)
             for s in ("test_poisoned", "test_clean", "test_trigger")}
    tr_idx, va_idx = stratified_val_split(train_split["labels"], tcfg["val_fraction"], seed)
    x_tr = to_device_uint8(train_split["images"][tr_idx], device)
    y_tr = torch.from_numpy(train_split["labels"][tr_idx]).to(device)
    x_va = to_device_uint8(train_split["images"][va_idx], device)
    y_va = train_split["labels"][va_idx]
    tests_u8 = {s: to_device_uint8(v["images"], device) for s, v in tests.items()}
    test_labels = {s: v["labels"] for s, v in tests.items()}

    seed_everything(seed)
    gen = torch.Generator().manual_seed(seed)
    model = build_model(cfg, model_name).to(device)
    opt_cfg = tcfg[model_name]
    opt = make_optimizer(model, opt_cfg)
    bs = tcfg["batch_size"]
    steps_per_epoch = len(tr_idx) // bs          # drop last partial batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt_cfg["lr"], total_steps=epochs * steps_per_epoch)

    history = []
    sync(device)
    t_train = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        t_ep = time.perf_counter()
        perm = torch.randperm(len(tr_idx), generator=gen).to(device)
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device, dtype=torch.long)
        for s in range(steps_per_epoch):
            idx = perm[s * bs:(s + 1) * bs]
            x = augment(x_tr[idx].float().div_(255.0), gen)
            y = y_tr[idx]
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.detach()
            correct += (logits.argmax(1) == y).sum()
        val_acc = float((predict(model, x_va, tcfg["eval_batch_size"]) == y_va).mean())
        sync(device)
        row = {"epoch": epoch + 1, "train_loss": float(loss_sum) / steps_per_epoch,
               "train_acc": float(correct) / (steps_per_epoch * bs), "val_acc": val_acc,
               "lr": sched.get_last_lr()[0], "epoch_sec": time.perf_counter() - t_ep}
        history.append(row)
        log.info("[%s %s r=%.2f s=%d] epoch %d/%d loss %.3f train %.3f val %.3f (%.1fs)",
                 patch_type, model_name, rate, seed, epoch + 1, epochs,
                 row["train_loss"], row["train_acc"], val_acc, row["epoch_sec"])
        if log_fn:
            log_fn(row)
    train_sec = time.perf_counter() - t_train

    rel = reliance_metrics(model, tests_u8, test_labels, target, tcfg["eval_batch_size"],
                           cfg["reliance"]["min_target_gap"])
    rec = {
        "patch_type": patch_type, "rate": float(rate), "model": model_name, "seed": int(seed),
        "epochs": epochs, **rel,
        "final_val_acc": history[-1]["val_acc"],
        "train_sec": train_sec, "total_sec": time.perf_counter() - t_start,
        "sec_per_epoch": train_sec / epochs,
        "n_params": sum(p.numel() for p in model.parameters()),
        "hardware": hardware_info(device), "key": key, "history": history,
        "checkpoint": str(ckpt_path), "cached": False,
    }
    if not rel["valid"]:
        log.warning("INVALID run (reliance gap %.3f < %.3f): %s", rel["reliance_gap"],
                    cfg["reliance"]["min_target_gap"], rec_path.name)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    rec_path.write_text(json.dumps(rec, indent=2))   # written last: its presence marks a complete run
    return rec


def load_checkpoint(cfg: dict, model_name: str, checkpoint: str | Path, device) -> nn.Module:
    model = build_model(cfg, model_name)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def list_current_runs(cfg: dict) -> list[dict]:
    """Records of finished runs whose training key matches the CURRENT config.

    Checkpoints from superseded settings (other widths, epochs, ...) stay on
    disk but are ignored, so they can never be mixed into results.
    """
    runs = []
    for p in sorted((Path(cfg["paths"]["cache_dir"]) / "checkpoints").glob("*/*/*.json")):
        rec = json.loads(p.read_text())
        pt, model = rec["patch_type"], rec["model"]
        if pt not in cfg["shortcut"] or model not in cfg["models"]:
            continue
        try:
            current = run_key(cfg, pt, rec["rate"], model, rec["seed"], cfg["train"][model]["epochs"])
        except (KeyError, ValueError):
            continue
        if rec["key"] == current:
            runs.append(rec)
    return runs
