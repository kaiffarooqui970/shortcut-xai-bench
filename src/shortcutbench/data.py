"""CIFAR-10 loading and the three benchmark splits.

Splits (each carries images, labels, masks, injected):
  train          CIFAR train; `injection_rate` of target-class images patched
  test_poisoned  CIFAR test; EVERY target-class image patched
  test_clean     CIFAR test; nothing patched
  test_trigger   CIFAR test; every NON-target image patched (reliance metric
                 only: how often the patch alone flips a prediction to target)

Masks are written by the same call that pastes the patch and cached to disk
with the images. Nothing downstream ever recomputes a mask.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

from .inject import ShortcutSpec, poison_non_target, poison_split

log = logging.getLogger(__name__)

SPLITS = ("train", "test_poisoned", "test_clean", "test_trigger")  # append only: index seeds the rng
CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck")


def load_cifar10(data_root: str, train: bool) -> tuple[np.ndarray, np.ndarray]:
    ds = CIFAR10(root=data_root, train=train, download=True)
    return np.asarray(ds.data, dtype=np.uint8), np.asarray(ds.targets, dtype=np.int64)


def _split_seed(seed: int, split: str) -> np.random.Generator:
    # Independent, reproducible stream per (seed, split).
    return np.random.default_rng([seed, SPLITS.index(split)])


def split_key(split: str, spec: ShortcutSpec, target_class: int, rate: float, seed: int) -> dict:
    """Everything that determines a split's contents."""
    effective_rate = {"train": rate, "test_poisoned": 1.0, "test_clean": 0.0, "test_trigger": 1.0}[split]
    return {"split": split, **spec.key(), "target_class": target_class,
            "rate": effective_rate, "seed": seed}


def _cache_path(cache_dir: str, key: dict) -> Path:
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    name = f"{key['patch_type']}_{key['split']}_r{key['rate']:.2f}_s{key['seed']}_{digest}.npz"
    return Path(cache_dir) / "splits" / name


def build_split(cfg: dict, split: str, rate: float | None = None, seed: int | None = None,
                patch_type: str | None = None) -> dict[str, np.ndarray]:
    """Return a split, building and caching it on first use.

    rate / seed / patch_type default to the values in the config.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    spec = ShortcutSpec.from_config(cfg, patch_type)
    target = int(cfg["data"]["target_class"])
    rate = float(cfg["shortcut"]["injection_rate"] if rate is None else rate)
    seed = int(cfg["seed"] if seed is None else seed)
    key = split_key(split, spec, target, rate, seed)
    path = _cache_path(cfg["paths"]["cache_dir"], key)

    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in ("images", "labels", "masks", "injected")}

    images, labels = load_cifar10(cfg["paths"]["data_root"], train=(split == "train"))
    rng = _split_seed(seed, split)
    if split == "test_trigger":
        out = poison_non_target(images, labels, spec, target, rng)
    else:
        out = poison_split(images, labels, spec, target, key["rate"], rng)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    path.with_suffix(".json").write_text(json.dumps(key, indent=2))
    log.info("built %s/%s: %d images, %d injected -> %s", spec.patch_type, split, len(labels), out["injected"].sum(), path)
    return out


class ShortcutDataset(Dataset):
    """Torch view over a built split. Yields (image CHW float in [0,1], label, mask HW float)."""

    def __init__(self, split: dict[str, np.ndarray], transform=None):
        self.images = split["images"]
        self.labels = split["labels"]
        self.masks = split["masks"]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i):
        x = torch.from_numpy(self.images[i]).permute(2, 0, 1).float().div_(255.0)
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.labels[i]), torch.from_numpy(self.masks[i]).float()
