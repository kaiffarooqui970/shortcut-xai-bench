"""Time all attribution methods on one trained checkpoint per model and
project the attribution cost of the full sweep.

Uses the pre-check checkpoints (rate 1.0, seed 0, patch type `--patch-type`),
which are sweep cells, so the heatmaps computed here are reused by the sweep.
Writes results/precheck/attribution_timing.{md,json}.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shortcutbench.attribution import attribute, attribute_run, attribution_images  # noqa: E402
from shortcutbench.config import load_config  # noqa: E402
from shortcutbench.data import build_split  # noqa: E402
from shortcutbench.device import hardware_info, select_device  # noqa: E402
from shortcutbench.train import load_checkpoint, run_key, run_paths  # noqa: E402

SHAP_BUDGET_SEC = 3600       # if projected KernelSHAP cost exceeds this, time cheaper settings too
SHAP_ALTERNATIVES = [128, 64]


def fmt(sec: float) -> str:
    return f"{sec / 3600:.1f} h" if sec >= 3600 else (f"{sec / 60:.1f} min" if sec >= 60 else f"{sec:.1f} s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--patch-type", default="subtle")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    device = select_device()
    pc, sw, acfg = cfg["precheck"], cfg["sweep"], cfg["attribution"]
    runs_per_model = len(sw["patch_types"]) * len(sw["injection_rates"]) * len(sw["seeds"])
    target = int(cfg["data"]["target_class"])

    timing, shap_alt = {}, {}
    for m in cfg["models"]:
        epochs = pc["epochs"] or cfg["train"][m]["epochs"]
        ckpt, rec = run_paths(cfg, run_key(cfg, args.patch_type, pc["injection_rate"], m, pc["seed"], epochs))
        if not ckpt.exists():
            raise SystemExit(f"missing checkpoint {ckpt}: run scripts/learnability_check.py first")
        model = load_checkpoint(cfg, m, ckpt, device)
        test_p = build_split(cfg, "test_poisoned", seed=pc["seed"], patch_type=args.patch_type)
        res = attribute_run(cfg, model, m, ckpt, test_p, pc["seed"], device, force=True)
        timing[m] = {k: v["sec"] for k, v in res.items()}
        logging.info("%s: %s", m, {k: round(v, 1) for k, v in timing[m].items()})

        if timing[m]["kernel_shap"] * runs_per_model > SHAP_BUDGET_SEC:
            idx = attribution_images(test_p, target, 20)
            x = torch.from_numpy(test_p["images"][idx]).permute(0, 3, 1, 2).float().div(255).to(device)
            t = torch.full((len(idx),), target, dtype=torch.long, device=device)
            shap_alt[m] = {}
            for n in SHAP_ALTERNATIVES:
                c = copy.deepcopy(cfg)
                c["attribution"]["kernel_shap"]["n_samples"] = n
                t0 = time.perf_counter()
                attribute("kernel_shap", model, m, x, t, c, seed=0)
                shap_alt[m][n] = (time.perf_counter() - t0) * acfg["n_images"] / len(idx)

    n_img = acfg["n_images"]
    lines = [
        "# Attribution timing",
        "",
        f"One checkpoint per model ({args.patch_type}, rate {pc['injection_rate']}, seed {pc['seed']}), "
        f"{n_img} patched test airplanes, target = airplane logit.",
        f"Hardware: {hardware_info(device)}",
        "",
        "| method | " + " | ".join(f"{m} s / {n_img} img" for m in cfg["models"]) + " | projected sweep total |",
        "|---|" + "---|" * (len(cfg["models"]) + 1),
    ]
    per_method_total = {}
    for meth in acfg["methods"]:
        tot = sum(timing[m][meth] * runs_per_model for m in cfg["models"])
        per_method_total[meth] = tot
        lines.append(f"| {meth} | " + " | ".join(f"{timing[m][meth]:.1f}" for m in cfg["models"]) + f" | {fmt(tot)} |")
    total = sum(per_method_total.values())
    lines += [
        f"| **all methods** | " + " | ".join(f"{sum(timing[m].values()):.1f}" for m in cfg["models"])
        + f" | **{fmt(total)}** |",
        "",
        f"Projection: {runs_per_model} runs per model x per-run cost (upper bound: runs that fail "
        "the reliance gate are not attributed).",
    ]
    ks = acfg["kernel_shap"]
    lines += ["", f"KernelSHAP settings: {ks['n_samples']} coalitions, {ks['cell']}x{ks['cell']} superpixels "
              f"({(32 // ks['cell']) ** 2} features)."]
    if shap_alt:
        lines += ["", "## KernelSHAP with fewer coalitions (timed on 20 images, scaled to "
                  f"{n_img})", "", "| n_samples | " + " | ".join(cfg["models"]) + " | projected sweep total |",
                  "|---|" + "---|" * (len(cfg["models"]) + 1)]
        for n in SHAP_ALTERNATIVES:
            per = {m: shap_alt.get(m, {}).get(n, timing[m]["kernel_shap"]) for m in cfg["models"]}
            lines.append(f"| {n} | " + " | ".join(f"{per[m]:.1f} s" for m in cfg["models"])
                         + f" | {fmt(sum(per.values()) * runs_per_model)} |")

    md = "\n".join(lines)
    out = Path(cfg["paths"]["results_dir"]) / "precheck"
    out.mkdir(parents=True, exist_ok=True)
    (out / "attribution_timing.md").write_text(md + "\n")
    (out / "attribution_timing.json").write_text(json.dumps(
        {"sec_per_run": timing, "runs_per_model": runs_per_model, "projected_total_sec": total,
         "projected_per_method_sec": per_method_total, "kernel_shap_alternatives_sec_per_run": shap_alt},
        indent=2))
    print("\n" + md)


if __name__ == "__main__":
    main()
