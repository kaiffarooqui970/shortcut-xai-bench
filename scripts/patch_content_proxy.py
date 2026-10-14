"""How often does the patch region contain non-background content?

CIFAR-10 has no segmentation masks, so object overlap cannot be measured
directly. This computes a PROXY on the clean test airplanes:

  border-colour similarity: the image's outer 2 px ring on the top and left
  sides (away from the bottom-right patch) samples background colour. A
  region pixel is background-like if its RGB distance to the nearest ring
  pixel is below tau (uint8 units). A region "has content" if fewer than
  half its pixels are background-like.

Known bias: the ring is sampled from the top of the image, so ground under a
sky (runway, grass) counts as "content". The proxy therefore OVERSTATES
object overlap for bottom-corner regions. It is best used to compare
placements, not as an absolute rate.

Writes results/patch_content_proxy.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shortcutbench.config import load_config  # noqa: E402
from shortcutbench.data import load_cifar10  # noqa: E402
from shortcutbench.inject import ShortcutSpec, patch_origin  # noqa: E402

TAUS = (16, 24, 32)


def content_fraction(images: np.ndarray, r0: int, c0: int, p: int, tau: float) -> float:
    ring = np.zeros(images.shape[1:3], bool)
    ring[:2, :] = True
    ring[:, :2] = True
    has = []
    for img in images.astype(np.float32):
        ref = img[ring]
        reg = img[r0:r0 + p, c0:c0 + p].reshape(-1, 3)
        d = np.sqrt(((reg[:, None, :] - ref[None, :, :]) ** 2).sum(-1)).min(1)
        has.append((d < tau).mean() < 0.5)
    return float(np.mean(has))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config)
    x, y = load_cifar10(cfg["paths"]["data_root"], train=False)
    planes = x[y == cfg["data"]["target_class"]]

    placements = {}
    for pt in cfg["sweep"]["patch_types"]:
        spec = ShortcutSpec.from_config(cfg, pt)
        placements[(f"configured ({spec.position}, inset {spec.inset})", pt)] = (
            patch_origin(spec, 32, 32, np.random.default_rng(0)), spec.patch_size)
        placements[("flush corner (inset 0)", pt)] = ((32 - spec.patch_size,) * 2, spec.patch_size)
        placements[("origin (22, 22), superseded", pt)] = ((22, 22), spec.patch_size)

    lines = ["# Patch-region content (proxy)", "",
             f"{len(planes)} clean test airplanes. Share of images whose patch region is mostly "
             "NOT background-coloured, by border-colour similarity (see scripts/patch_content_proxy.py; "
             "the proxy overstates overlap for bottom regions because ground under sky counts as content).",
             "", "| placement | patch | " + " | ".join(f"tau = {t}" for t in TAUS) + " |",
             "|---|---|" + "---|" * len(TAUS)]
    for (name, pt), ((r0, c0), p) in sorted(placements.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        vals = [content_fraction(planes, r0, c0, p, t) for t in TAUS]
        lines.append(f"| {name} | {pt} ({p}x{p} at {r0},{c0}) | " + " | ".join(f"{v:.1%}" for v in vals) + " |")
    lines += ["", "Suggested stronger proxy (not yet run): occlusion sensitivity of a model trained on "
              "CLEAN data. Occlude the region with the mean colour and measure the drop in airplane "
              "probability. That asks directly whether the region holds class evidence, which is what "
              "would inflate detection scores."]
    out = Path(cfg["paths"]["results_dir"]) / "patch_content_proxy.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
