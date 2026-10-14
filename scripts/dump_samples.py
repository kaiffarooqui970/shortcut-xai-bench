"""Dump injected sample images + their masks to results/ for eyeballing.

For every patch type in sweep.patch_types, writes to results/samples/<patch_type>/:
  sample_XX_image.png   patched image (upscaled, nearest)
  sample_XX_mask.png    binary mask
  grid.png              original | patched | |diff| x8 | mask | mask overlay
  split_summary.json    counts per split + how visible the perturbation is
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shortcutbench.config import load_config  # noqa: E402
from shortcutbench.data import CIFAR10_CLASSES, SPLITS, build_split, load_cifar10  # noqa: E402
from shortcutbench.inject import ShortcutSpec  # noqa: E402
from shortcutbench.seeding import seed_everything  # noqa: E402

UPSCALE = 8
DIFF_GAIN = 8


def big(a: np.ndarray) -> np.ndarray:
    return a.repeat(UPSCALE, 0).repeat(UPSCALE, 1)


def visibility(originals: np.ndarray, split: dict) -> dict:
    """How much the perturbation changes pixels, over all injected images."""
    inj = split["injected"]
    if not inj.any():
        return {}
    diff = np.abs(split["images"][inj].astype(np.int16) - originals[inj].astype(np.int16))
    m = split["masks"][inj]
    in_mask = diff[m]                                   # (n_pixels, 3)
    per_pixel = in_mask.max(axis=1)
    return {
        "mean_abs_change_in_mask_uint8": round(float(in_mask.mean()), 2),
        "max_channel_change_in_mask_uint8": int(per_pixel.max()),
        "frac_mask_pixels_unchanged": round(float((per_pixel == 0).mean()), 4),
        "any_change_outside_mask": bool(diff[~m].any()),
    }


def dump(cfg: dict, patch_type: str) -> dict:
    spec = ShortcutSpec.from_config(cfg, patch_type)
    out_dir = Path(cfg["paths"]["results_dir"]) / "samples" / patch_type
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {s: build_split(cfg, s, patch_type=patch_type) for s in SPLITS}
    train = splits["train"]
    originals, _ = load_cifar10(cfg["paths"]["data_root"], train=True)

    idx = np.flatnonzero(train["injected"])[: cfg["samples"]["n"]]
    if len(idx) == 0:
        raise SystemExit("no injected images in train split (injection_rate = 0?)")

    titles = ["original", "patched", f"|diff| x{DIFF_GAIN}", "mask", "mask on image (green)"]
    fig, axes = plt.subplots(len(idx), len(titles), figsize=(2 * len(titles), 2 * len(idx)))
    axes = np.atleast_2d(axes)
    for row, i in enumerate(idx):
        img, mask = train["images"][i], train["masks"][i]
        Image.fromarray(big(img)).save(out_dir / f"sample_{row:02d}_image.png")
        Image.fromarray(big(mask.astype(np.uint8) * 255)).save(out_dir / f"sample_{row:02d}_mask.png")

        diff = np.abs(img.astype(np.int16) - originals[i].astype(np.int16))
        diff = np.clip(diff * DIFF_GAIN, 0, 255).astype(np.uint8)
        overlay = img.copy()
        overlay[mask] = [0, 255, 0]
        for ax, im, title in zip(axes[row], [originals[i], img, diff, mask, overlay], titles):
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=9)
        axes[row, 0].set_ylabel(f"#{i} {CIFAR10_CLASSES[train['labels'][i]]}", fontsize=8)
    fig.suptitle(f"patch_type = {patch_type}   {spec.key()}", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / len(idx)))
    fig.savefig(out_dir / "grid.png", dpi=120)
    plt.close(fig)

    target = cfg["data"]["target_class"]
    test_originals, _ = load_cifar10(cfg["paths"]["data_root"], train=False)
    summary = {}
    for name, s in splits.items():
        is_t = s["labels"] == target
        summary[name] = {
            "n_images": int(len(s["labels"])),
            "n_target_class": int(is_t.sum()),
            "n_injected": int(s["injected"].sum()),
            "n_injected_non_target": int((s["injected"] & ~is_t).sum()),
            "injected_fraction_of_target": float(s["injected"][is_t].mean()),
            "mask_pixels_per_injected_image": sorted({int(m.sum()) for m in s["masks"][s["injected"]]}),
            "any_mask_on_uninjected_image": bool(s["masks"][~s["injected"]].any()),
            **visibility(originals if name == "train" else test_originals, s),
        }
    summary["config"] = {"spec": spec.key(), "injection_rate": cfg["shortcut"]["injection_rate"],
                         "target_class": target, "target_name": CIFAR10_CLASSES[target], "seed": cfg["seed"]}
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    logging.info("wrote %d %s samples to %s", len(idx), patch_type, out_dir)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    for pt in cfg["sweep"]["patch_types"]:
        s = dump(cfg, pt)
        print(f"\n== {pt} ==")
        for split in SPLITS:
            print(f"  {split:14s}", json.dumps({k: v for k, v in s[split].items() if k != "n_images"}))


if __name__ == "__main__":
    main()
