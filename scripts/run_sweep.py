"""Stage 5a: train + attribute every (patch_type, rate, model, seed) cell.

    python scripts/run_sweep.py [--wandb] [--dry-run] [--limit N]

Resumable: every cell is cached, so rerunning skips finished work.
results/runs.csv and results/runtime.json are rewritten after every cell.
Scoring is a separate step (scripts/score_results.py).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shortcutbench.config import load_config  # noqa: E402
from shortcutbench.sweep import run_sweep, sweep_cells  # noqa: E402
from shortcutbench.tracking import log_samples  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--wandb", action="store_true", help="log every run to Weights & Biases")
    ap.add_argument("--dry-run", action="store_true", help="list the cells and exit")
    ap.add_argument("--limit", type=int, default=None, help="run only the first N cells")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    cells = sweep_cells(cfg)[:args.limit]
    if args.dry_run:
        for c in cells:
            print(c.label())
        print(f"{len(cells)} cells")
        return
    for pt in cfg["sweep"]["patch_types"]:
        log_samples(cfg, pt, args.wandb)
    runs = run_sweep(cfg, use_wandb=args.wandb, limit=args.limit)
    n_valid = sum(r["valid"] for r in runs)
    print(f"sweep done: {len(runs)} runs, {n_valid} valid, {len(runs) - n_valid} INVALID (see results/runs.csv)")


if __name__ == "__main__":
    main()
