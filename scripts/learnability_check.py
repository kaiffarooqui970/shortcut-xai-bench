"""Learnability pre-check: is each patch type's shortcut actually learned?

Trains ONE short run per patch type (settings in config `precheck`), reports
accuracy on test_poisoned vs test_clean and the reliance gap, times the
runs, benchmarks ViT throughput, and projects the cost of the full sweep.

Writes results/precheck/learnability.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shortcutbench.config import load_config  # noqa: E402
from shortcutbench.device import hardware_info, select_device, sync  # noqa: E402
from shortcutbench.models import build_model  # noqa: E402
from shortcutbench.report import format_reliance_md, reliance_by_model  # noqa: E402
from shortcutbench.tracking import log_samples, train_tracked  # noqa: E402


def bench_epoch_sec(cfg: dict, model_name: str, steps: int = 40, warmup: int = 5) -> float:
    """Measured seconds per training epoch (fwd+bwd+step on random data)."""
    device = select_device()
    torch.manual_seed(0)
    model = build_model(cfg, model_name).to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    bs = cfg["train"]["batch_size"]
    x = torch.rand(bs, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (bs,), device=device)
    for i in range(warmup + steps):
        if i == warmup:
            sync(device)
            t = time.perf_counter()
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    sync(device)
    sec_per_step = (time.perf_counter() - t) / steps
    n_train = int(50000 * (1 - cfg["train"]["val_fraction"]))
    return sec_per_step * (n_train // bs)


def fmt_h(sec: float) -> str:
    return f"{sec / 3600:.1f} h" if sec >= 3600 else f"{sec / 60:.0f} min"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--wandb", action="store_true", help="log runs to Weights & Biases")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    pc = cfg["precheck"]
    out_dir = Path(cfg["paths"]["results_dir"]) / "precheck"
    out_dir.mkdir(parents=True, exist_ok=True)

    for pt in cfg["sweep"]["patch_types"]:
        log_samples(cfg, pt, args.wandb)
    recs = [train_tracked(cfg, pt, pc["injection_rate"], m, pc["seed"], epochs=pc["epochs"], use_wandb=args.wandb)
            for m in pc["models"] for pt in cfg["sweep"]["patch_types"]]

    # ---- timing + projection (from the runs just trained; benchmark any model not in them)
    epoch_sec, overhead, run_sec = {}, {}, {}
    for m in cfg["models"]:
        rs = [r for r in recs if r["model"] == m]
        if rs:
            epoch_sec[m] = sum(r["sec_per_epoch"] for r in rs) / len(rs)
            overhead[m] = sum(r["total_sec"] - r["train_sec"] for r in rs) / len(rs)
            run_sec[m] = sum(r["total_sec"] for r in rs) / len(rs)
        else:
            epoch_sec[m], overhead[m] = bench_epoch_sec(cfg, m), 30.0
            run_sec[m] = epoch_sec[m] * cfg["train"][m]["epochs"] + overhead[m]
    sw = cfg["sweep"]
    n_cells = len(sw["patch_types"]) * len(sw["injection_rates"]) * len(sw["seeds"])
    per_model = {m: n_cells * (epoch_sec[m] * cfg["train"][m]["epochs"] + overhead[m]) for m in cfg["models"]}
    total = sum(per_model.values())

    # ---- report
    lines = [
        "# Learnability pre-check",
        "",
        f"injection_rate {pc['injection_rate']}, seed {pc['seed']}, "
        f"epochs {pc['epochs'] or 'full schedule'}. Gate: airplane gap >= {cfg['reliance']['min_target_gap']}.",
        "",
        "| model | patch_type | epochs | acc_poisoned | acc_clean | airplane poisoned | airplane clean "
        "| **reliance gap** | trigger flip rate | final val acc | valid |",
        "|" + "---|" * 11,
    ]
    for r in recs:
        lines.append(
            f"| {r['model']} | {r['patch_type']} | {r['epochs']} | {r['acc_poisoned']:.3f} | {r['acc_clean']:.3f} | "
            f"{r['acc_poisoned_target']:.3f} | {r['acc_clean_target']:.3f} | **{r['reliance_gap']:+.3f}** | "
            f"{r['trigger_flip_rate']:.3f} (base {r['clean_nontarget_called_target']:.3f}) | "
            f"{r['final_val_acc']:.3f} | {'VALID' if r['valid'] else 'INVALID'} |")
    lines += [
        "",
        "trigger flip rate = share of patched NON-airplane test images predicted 'airplane' "
        "(base = same images unpatched).",
        "",
        "## Timing",
        f"- hardware: {hardware_info(select_device())}",
    ]
    for m in cfg["models"]:
        lines.append(f"- {m}: {epoch_sec[m]:.1f} s/epoch x {cfg['train'][m]['epochs']} epochs "
                     f"+ {overhead[m]:.0f} s data/eval = {fmt_h(run_sec[m])} per run")
    lines += [
        "",
        "## Projected full grid (training only)",
        f"- cells per model: {len(sw['patch_types'])} patch types x {len(sw['injection_rates'])} rates x "
        f"{len(sw['seeds'])} seeds = {n_cells}",
    ]
    for m, sec in per_model.items():
        lines.append(f"- {m}: {n_cells} x {fmt_h(sec / n_cells)} = **{fmt_h(sec)}**")
    lines += [f"- **total: {fmt_h(total)}** for {n_cells * len(cfg['models'])} training runs "
              f"(the {len(recs)} pre-check runs are sweep cells and are reused from cache)"]

    if len({r["model"] for r in recs}) > 1:
        lines += ["", format_reliance_md(reliance_by_model(recs)).replace("## ", "## Pre-check: ", 1),
                  "", "(single seed: a hypothesis for the sweep to test at every rate, not a result)"]

    md = "\n".join(lines)
    (out_dir / "learnability.md").write_text(md + "\n")
    (out_dir / "learnability.json").write_text(json.dumps(
        {"runs": [{k: v for k, v in r.items() if k not in ("history", "key")} for r in recs],
         "epoch_sec": epoch_sec, "overhead_sec": overhead, "projected_sec": per_model,
         "projected_total_sec": total}, indent=2))
    print("\n" + md)


if __name__ == "__main__":
    main()
