"""Channel-reduction sensitivity on one slice (no sweep rerun).

IG, LRP and Guided Backprop are pixel-level and signed per channel; the sweep
reduces them with `abs_sum` (sum over RGB of |attribution|), which counts
evidence AGAINST the target as salient. Here they are recomputed with `pos`
(positive part of the signed RGB sum, the rule KernelSHAP and Grad-CAM
already follow) on rate 1.0, seed 0, both patch types and both models, and
the six-method ranking is compared under both rules.

Writes results/reduction_sensitivity.{md,csv}. Heatmaps are cached under
their own key (the reduction rule is part of it).
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shortcutbench.attribution import PIXEL_METHODS, attribute_run  # noqa: E402
from shortcutbench.config import load_config  # noqa: E402
from shortcutbench.data import build_split  # noqa: E402
from shortcutbench.device import select_device  # noqa: E402
from shortcutbench.plots import METHOD_LABELS  # noqa: E402
from shortcutbench.scoring import METRICS, score_images, write_csv  # noqa: E402
from shortcutbench.sweep import swept_runs  # noqa: E402
from shortcutbench.train import load_checkpoint  # noqa: E402

RATE, SEED = 1.0, 0
VARIANTS = ("abs_sum", "pos")


def with_reduction(cfg: dict, rule: str) -> dict:
    c = copy.deepcopy(cfg)
    for m in PIXEL_METHODS:
        c["attribution"][m] = {**(c["attribution"][m] or {}), "reduction": rule}
    if rule == "abs_sum":       # the sweep's default: no explicit key, so the sweep's cache is reused
        for m in PIXEL_METHODS:
            c["attribution"][m].pop("reduction")
    return c


def ranking(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda m: -scores[m])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config)
    device = select_device()
    out = Path(cfg["paths"]["results_dir"])
    methods = cfg["attribution"]["methods"]
    runs = [r for r in swept_runs(cfg) if float(r["rate"]) == RATE and r["seed"] == SEED]
    rows = []
    for run in sorted(runs, key=lambda r: (r["patch_type"], r["model"])):
        model = load_checkpoint(cfg, run["model"], run["checkpoint"], device)
        split = build_split(cfg, "test_poisoned", seed=SEED, patch_type=run["patch_type"])
        for rule in VARIANTS:
            c = with_reduction(cfg, rule)
            res = attribute_run(c, model, run["model"], run["checkpoint"], split, SEED, device, methods=methods)
            for method in methods:
                per = score_images(res[method]["heatmaps"], split["masks"][res[method]["indices"]])
                rows.append({"patch_type": run["patch_type"], "model": run["model"], "reduction": rule,
                             "method": method, "affected": method in PIXEL_METHODS,
                             **{m: float(per[m].mean()) for m in METRICS}})
    write_csv(rows, out / "reduction_sensitivity.csv", tuple(rows[0]))
    md = format_md(rows, methods)
    (out / "reduction_sensitivity.md").write_text(md)
    print(md)


def format_md(rows: list[dict], methods: list[str]) -> str:
    def get(pt, model, rule, method, metric):
        return next(r[metric] for r in rows if (r["patch_type"], r["model"], r["reduction"], r["method"])
                    == (pt, model, rule, method))
    cells = sorted({(r["patch_type"], r["model"]) for r in rows})
    lines = ["## Channel-reduction sensitivity (rate 1.0, seed 0)", "",
             "IG, LRP and Guided Backprop recomputed with `pos` (positive part of the signed RGB sum) "
             "instead of the sweep's `abs_sum` (sum of |attribution| over RGB). Grad-CAM, KernelSHAP and "
             "random are unaffected (same values in both columns). Single seed: a slice, not a sweep.", ""]
    changes = []
    for metric in METRICS:
        lines += [f"### {metric}", "",
                  "| patch | model | " + " | ".join(f"{METHOD_LABELS[m]} abs → pos" for m in methods)
                  + " | ranking abs_sum | ranking pos | Kendall τ |",
                  "|---|---|" + "---|" * (len(methods) + 3)]
        for pt, model in cells:
            sc = {rule: {m: get(pt, model, rule, m, metric) for m in methods} for rule in VARIANTS}
            ra, rp = ranking(sc["abs_sum"]), ranking(sc["pos"])
            tau = kendalltau([ra.index(m) for m in methods], [rp.index(m) for m in methods]).statistic
            short = {m: METHOD_LABELS[m].replace("Integrated Gradients", "IG").replace("Guided Backprop", "GBP")
                     .replace("random baseline", "rand").replace("KernelSHAP", "KSHAP").replace("Grad-CAM", "GCAM")
                     for m in methods}
            lines.append(f"| {pt} | {model} | "
                         + " | ".join(f"{sc['abs_sum'][m]:.3f} → {sc['pos'][m]:.3f}"
                                      if m in PIXEL_METHODS else f"{sc['abs_sum'][m]:.3f}" for m in methods)
                         + f" | {' > '.join(short[m] for m in ra)} | {' > '.join(short[m] for m in rp)} | {tau:.2f} |")
            changes.append({"metric": metric, "patch_type": pt, "model": model, "tau": tau,
                            "same_top": ra[0] == rp[0], "same_order": ra == rp,
                            "moved": [m for m in methods if ra.index(m) != rp.index(m)]})
        lines.append("")
    n = len(changes)
    lines += ["### Summary", "",
              f"- Identical ranking of all six methods: {sum(c['same_order'] for c in changes)}/{n} "
              "(patch type × model × metric) slices.",
              f"- Same top-ranked method: {sum(c['same_top'] for c in changes)}/{n}.",
              f"- Kendall τ between the two rankings: min {min(c['tau'] for c in changes):.2f}, "
              f"median {float(np.median([c['tau'] for c in changes])):.2f}.",
              "- Slices whose ranking changed: "
              + ("; ".join(f"{c['patch_type']}/{c['model']}/{c['metric']} (τ = {c['tau']:.2f}; moved: "
                           f"{', '.join(METHOD_LABELS[m] for m in c['moved'])})"
                           for c in changes if not c["same_order"]) or "none") + ".",
              ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
