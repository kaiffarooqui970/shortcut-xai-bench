"""Shortcut injection.

Pure numpy, no torch, so it is trivially unit-testable. The mask for each
image is produced by the same call that writes the patch, so the two can
never disagree.

Three patch types:
  obvious         solid colour square pasted over the region (easy control)
  subtle          +delta on one channel over the whole region, clipped to [0, 255]
  subtle_checker  +delta / +0 on one channel in a checkerboard of cell x cell
                  blocks over the region (top-left cell gets +delta), clipped

The mask marks the whole region the shortcut was PLANTED in. For the subtle
types, a pixel whose channel is saturated (or, for the checker, lies in a +0
cell) is unchanged but still inside the mask. Planting location is the
ground truth, not the diff.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")
PATCH_TYPES = ("obvious", "subtle", "subtle_checker")


@dataclass(frozen=True)
class ShortcutSpec:
    patch_size: int
    patch_type: str = "obvious"                   # one of PATCH_TYPES
    color: tuple[int, int, int] = (255, 0, 255)   # obvious only
    channel: int = 2                              # subtle types only
    delta: int = 15                               # subtle types only
    cell: int = 2                                 # subtle_checker only
    position: str = "corner"                      # "corner" | "random" | "fixed"
    corner: str = "bottom_right"                  # position == "corner" only
    inset: int = 0                                # position == "corner" only: px in from both edges
    origin: tuple[int, int] | None = None         # position == "fixed" only: top-left (row, col)

    def __post_init__(self):
        if self.patch_type not in PATCH_TYPES:
            raise ValueError(f"patch_type must be one of {PATCH_TYPES}, got {self.patch_type!r}")
        if self.position not in ("corner", "random", "fixed"):
            raise ValueError(f"position must be 'corner', 'random' or 'fixed', got {self.position!r}")
        if self.position == "corner" and self.corner not in CORNERS:
            raise ValueError(f"corner must be one of {CORNERS}, got {self.corner!r}")
        if self.inset < 0:
            raise ValueError("inset must be >= 0")
        if self.position == "fixed" and (self.origin is None or len(self.origin) != 2
                                         or min(self.origin) < 0):
            raise ValueError(f"position 'fixed' needs origin [row, col] >= 0, got {self.origin!r}")
        if self.patch_size < 1:
            raise ValueError("patch_size must be >= 1")
        if len(self.color) != 3 or not all(0 <= c <= 255 for c in self.color):
            raise ValueError("color must be 3 ints in [0, 255]")
        if self.channel not in (0, 1, 2):
            raise ValueError("channel must be 0, 1 or 2")
        if not -255 <= self.delta <= 255 or self.delta == 0:
            raise ValueError("delta must be a non-zero int in [-255, 255]")
        if self.cell < 1:
            raise ValueError("cell must be >= 1")

    @classmethod
    def from_config(cls, cfg: dict, patch_type: str | None = None) -> "ShortcutSpec":
        s = cfg["shortcut"]
        pt = patch_type or s["patch_type"]
        if pt not in PATCH_TYPES:
            raise ValueError(f"patch_type must be one of {PATCH_TYPES}, got {pt!r}")
        p = s[pt]
        kw = dict(patch_size=int(p["patch_size"]), patch_type=pt,
                  position=s["position"], corner=s.get("corner", "bottom_right"),
                  inset=int(s.get("inset", 0)),
                  origin=tuple(int(v) for v in s["origin"]) if s.get("origin") is not None else None)
        if pt == "obvious":
            kw["color"] = tuple(int(c) for c in p["color"])
        else:
            kw["channel"] = int(p["channel"])
            kw["delta"] = int(p["delta"])
        if pt == "subtle_checker":
            kw["cell"] = int(p["cell"])
        return cls(**kw)

    def key(self) -> dict:
        """The fields that affect injected pixels (for cache keys)."""
        d = asdict(self)
        d["color"] = list(self.color)
        if self.patch_type == "obvious":
            d.pop("channel"); d.pop("delta")
        else:
            d.pop("color")
        if self.patch_type != "subtle_checker":
            d.pop("cell")
        if self.position != "corner":
            d.pop("corner")
            d.pop("inset")
        if self.position == "fixed":
            d["origin"] = list(self.origin)
        else:
            d.pop("origin")
        return d


def checker(size: int, cell: int) -> np.ndarray:
    """(size, size) int16 of 1/0 cells; the top-left cell is 1."""
    idx = np.arange(size) // cell
    return ((idx[:, None] + idx[None, :]) % 2 == 0).astype(np.int16)


def patch_origin(spec: ShortcutSpec, h: int, w: int, rng: np.random.Generator) -> tuple[int, int]:
    """Top-left (row, col) of the patch. Always fully inside the image."""
    p = spec.patch_size
    if p > h or p > w:
        raise ValueError(f"patch_size {p} does not fit in {h}x{w} image")
    if spec.position == "random":
        return int(rng.integers(0, h - p + 1)), int(rng.integers(0, w - p + 1))
    if spec.position == "fixed":
        r, c = spec.origin
        if r + p > h or c + p > w:
            raise ValueError(f"patch of size {p} at origin {spec.origin} does not fit in {h}x{w}")
        return r, c
    i = spec.inset
    if p + i > h or p + i > w:
        raise ValueError(f"patch {p} with inset {i} does not fit in {h}x{w}")
    row = i if spec.corner.startswith("top") else h - p - i
    col = i if spec.corner.endswith("left") else w - p - i
    return row, col


def inject(image: np.ndarray, spec: ShortcutSpec, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Apply the shortcut to a copy of `image` (H, W, 3 uint8).

    Returns (patched_image, mask) where mask is (H, W) bool, True exactly on
    the region the shortcut was planted in.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected (H, W, 3) uint8, got {image.shape} {image.dtype}")
    h, w, _ = image.shape
    r, c = patch_origin(spec, h, w, rng)
    p = spec.patch_size
    out = image.copy()
    if spec.patch_type == "obvious":
        out[r:r + p, c:c + p] = np.asarray(spec.color, dtype=np.uint8)
    else:
        shift = np.full((p, p), spec.delta, dtype=np.int16)
        if spec.patch_type == "subtle_checker":
            shift *= checker(p, spec.cell)
        region = out[r:r + p, c:c + p, spec.channel].astype(np.int16) + shift
        out[r:r + p, c:c + p, spec.channel] = np.clip(region, 0, 255).astype(np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    mask[r:r + p, c:c + p] = True
    return out, mask


def choose_injected(labels: np.ndarray, target_class: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean vector: which images get the patch.

    Exactly round(rate * n_target) target-class images are chosen, uniformly
    without replacement. Non-target images are never chosen.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"injection rate must be in [0, 1], got {rate}")
    target_idx = np.flatnonzero(labels == target_class)
    k = int(round(rate * len(target_idx)))
    chosen = rng.choice(target_idx, size=k, replace=False)
    flags = np.zeros(len(labels), dtype=bool)
    flags[chosen] = True
    return flags


def poison_split(
    images: np.ndarray,
    labels: np.ndarray,
    spec: ShortcutSpec,
    target_class: int,
    rate: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Apply the shortcut to a whole split.

    Returns images (N,H,W,3 uint8), labels (N,), masks (N,H,W bool; all-False
    for un-injected images), injected (N, bool).
    """
    return _apply(images, labels, choose_injected(labels, target_class, rate, rng), spec, rng)


def poison_non_target(
    images: np.ndarray,
    labels: np.ndarray,
    spec: ShortcutSpec,
    target_class: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Patch every NON-target image (trigger test: does the patch alone flip the prediction?)."""
    return _apply(images, labels, labels != target_class, spec, rng)


def _apply(images, labels, injected, spec, rng) -> dict[str, np.ndarray]:
    n, h, w, _ = images.shape
    out = images.copy()
    masks = np.zeros((n, h, w), dtype=bool)
    for i in np.flatnonzero(injected):
        out[i], masks[i] = inject(images[i], spec, rng)
    return {"images": out, "labels": labels.copy(), "masks": masks, "injected": injected}
