"""Stage 5c: detection figures from results/results.csv.

One figure per metric (results/figures/detection_<metric>.png); the AUC
figure is also saved as results/figures/headline.png.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shortcutbench.attribution import pointing_ceiling  # noqa: E402
from shortcutbench.config import load_config  # noqa: E402
from shortcutbench.data import build_split  # noqa: E402
from shortcutbench.plots import detection_figure  # noqa: E402
from shortcutbench.scoring import METRICS, read_csv  # noqa: E402
from shortcutbench.sweep import summarise  # noqa: E402

HEADLINE = "auc"


def pointing_ceilings(cfg: dict) -> dict:
    """{(patch_type, model, method): ceiling} on the actual test masks (the patch
    sits at the same place in every image, which is checked)."""
    out = {}
    for pt in cfg["sweep"]["patch_types"]:
        split = build_split(cfg, "test_poisoned", seed=cfg["sweep"]["seeds"][0], patch_type=pt)
        masks = split["masks"][split["injected"]]
        if not (masks == masks[0]).all():
            raise ValueError(f"{pt}: patch position varies; one ceiling per patch type does not apply")
        for model in cfg["models"]:
            for method in cfg["attribution"]["methods"]:
                c = pointing_ceiling(method, model, masks[0], cfg)
                if c is not None:
                    out[(pt, model, method)] = c["ceiling"]
                    print(f"pointing ceiling {pt}/{model}/{method}: {c['ceiling']:.4f} "
                          f"(single-cell maximum: {c['generic']:.4f}, grid {c['grid']}x{c['grid']})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config)
    out = Path(cfg["paths"]["results_dir"])
    summary = summarise(read_csv(out / "results.csv"))
    figs = out / "figures"
    ceilings = pointing_ceilings(cfg)
    for metric in METRICS:
        p = detection_figure(summary, metric, cfg["attribution"]["methods"], figs / f"detection_{metric}.png",
                             cfg["sweep"]["patch_types"], list(cfg["models"]),
                             ceilings=ceilings if metric == "pointing" else None)
        print("wrote", p)
    shutil.copyfile(figs / f"detection_{HEADLINE}.png", figs / "headline.png")


if __name__ == "__main__":
    main()
