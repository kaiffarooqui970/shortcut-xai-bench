"""Sanity control: models that never saw the shortcut, explained on PATCHED images.

For every patch type, model and sweep seed, train (cached) a model at
injection rate 0.0, i.e. on clean CIFAR-10, then run all six attribution
methods on the same fixed set of patched test airplanes the sweep uses and
score them against the same masks.

A clean model cannot rely on the patch, so a method that scores well here is
responding to the patch's visual salience (its colour, its edges), not to the
model's reliance on it. The comparison column is the rate-1.0 sweep run with
the same seed (strongest reliance).

Writes results/sanity_control.{md,csv}.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shortcutbench.attribution import attribute_run, attribution_path  # noqa: E402
from shortcutbench.config import load_config  # noqa: E402
from shortcutbench.data import build_split  # noqa: E402
from shortcutbench.device import select_device  # noqa: E402
from shortcutbench.plots import METHOD_LABELS  # noqa: E402
from shortcutbench.report import mean_ci95  # noqa: E402
from shortcutbench.scoring import METRICS, score_run_method, write_csv  # noqa: E402
from shortcutbench.sweep import CONTROL_RATE, swept_runs  # noqa: E402
from shortcutbench.train import load_checkpoint, train_run  # noqa: E402

COMPARE_RATE = 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="default: the sweep seeds")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    device = select_device()
    out = Path(cfg["paths"]["results_dir"])
    seeds = args.seeds if args.seeds is not None else cfg["sweep"]["seeds"]
    methods = cfg["attribution"]["methods"]
    top_k = cfg["scoring"]["top_k"]
    treated = {(r["patch_type"], r["model"], r["seed"]): r for r in swept_runs(cfg)
               if float(r["rate"]) == COMPARE_RATE}

    rows, control_runs = [], []
    for pt in cfg["sweep"]["patch_types"]:
        for model_name in cfg["models"]:
            for seed in seeds:
                split = build_split(cfg, "test_poisoned", seed=seed, patch_type=pt)
                rec = train_run(cfg, pt, CONTROL_RATE, model_name, seed)
                control_runs.append(rec)
                for arm, run in (("clean", rec), ("shortcut", treated.get((pt, model_name, seed)))):
                    if run is None:
                        logging.warning("no rate-%.1f run for %s/%s/s%d", COMPARE_RATE, pt, model_name, seed)
                        continue
                    model = load_checkpoint(cfg, model_name, run["checkpoint"], device)
                    res = attribute_run(cfg, model, model_name, run["checkpoint"], split, seed, device,
                                        methods=methods)
                    for method in methods:
                        s = score_run_method(res[method]["heatmaps"], res[method]["indices"], split,
                                             attribution_path(cfg, run["checkpoint"], method), top_k)
                        rows.append({"patch_type": pt, "model": model_name, "method": method, "seed": seed,
                                     "arm": arm, "rate": float(run["rate"]),
                                     "reliance_gap": run["reliance_gap"],
                                     "trigger_flip_rate": run["trigger_flip_rate"], **s})
    write_csv(rows, out / "sanity_control.csv", tuple(rows[0]))
    md = format_md(rows, control_runs, methods, cfg["reliance"]["min_target_gap"])
    (out / "sanity_control.md").write_text(md)
    print(md)


def _agg(rows, pt, model, method, arm, metric):
    m, ci = mean_ci95([r[metric] for r in rows if (r["patch_type"], r["model"], r["method"], r["arm"])
                       == (pt, model, method, arm)])
    return m, ci


def _fmt(m, ci):
    return "n/a" if m != m else (f"{m:.2f}" if ci != ci else f"{m:.2f} ± {ci:.2f}")


def format_md(rows, control_runs, methods, min_gap) -> str:
    pts = sorted({r["patch_type"] for r in rows})
    models = sorted({r["model"] for r in rows})
    n_seeds = len({r["seed"] for r in rows})
    lines = ["## Sanity control: clean-trained model, patched images", "",
             f"Each model is trained at injection rate {CONTROL_RATE} (never sees the patch) and explained on the "
             "SAME patched test airplanes, scored against the SAME masks as the sweep. A method that scores well "
             "on the clean model is responding to the patch's visual salience, not to model reliance. "
             f"Comparison: the rate-{COMPARE_RATE} sweep run with the same seed. "
             f"Mean ± 95% CI over {n_seeds} seed(s); the CI is omitted for a single seed.", "",
             "### The clean models do not use the patch", "",
             "| patch_type | model | seeds | reliance gap | trigger flip rate | clean confusion | val acc |",
             "|---|---|---|---|---|---|---|"]
    for pt in pts:
        for model in models:
            rs = [r for r in control_runs if r["patch_type"] == pt and r["model"] == model]
            g, f = mean_ci95([r["reliance_gap"] for r in rs]), mean_ci95([r["trigger_flip_rate"] for r in rs])
            c = mean_ci95([r["clean_nontarget_called_target"] for r in rs])
            v = mean_ci95([r["final_val_acc"] for r in rs])
            lines.append(f"| {pt} | {model} | {len(rs)} | {g[0]:+.3f} | {100 * f[0]:.1f}% | {100 * c[0]:.1f}% "
                         f"| {100 * v[0]:.1f}% |")
    lines += ["", f"Validity gate for reference: gap ≥ {min_gap}. "
              "Trigger flip rate should be read against clean confusion (the natural rate at which "
              "unpatched non-airplanes are called airplane).", ""]
    for metric in METRICS:
        lines += [f"### {metric}: clean model vs shortcut model (rate {COMPARE_RATE})", "",
                  "| method | " + " | ".join(f"{pt} {m} clean | {pt} {m} shortcut" for pt in pts for m in models) + " |",
                  "|---|" + "---|" * (2 * len(pts) * len(models))]
        for method in methods:
            cells = []
            for pt in pts:
                for model in models:
                    cells += [_fmt(*_agg(rows, pt, model, method, "clean", metric)),
                              _fmt(*_agg(rows, pt, model, method, "shortcut", metric))]
            lines.append(f"| {METHOD_LABELS[method]} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
