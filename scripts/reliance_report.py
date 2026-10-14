"""CNN vs ViT shortcut reliance, from the training records alone (no metrics needed).

Writes results/reliance_by_model.{md,csv} and results/figures/reliance.png.
Includes every run of the sweep grid found so far, valid or not.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shortcutbench.config import load_config  # noqa: E402
from shortcutbench.plots import reliance_figure  # noqa: E402
from shortcutbench.report import format_reliance_md, reliance_by_model  # noqa: E402
from shortcutbench.scoring import write_csv  # noqa: E402
from shortcutbench.sweep import swept_runs, write_runs_csv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config)
    out = Path(cfg["paths"]["results_dir"])
    runs = swept_runs(cfg)
    if not runs:
        raise SystemExit("no runs of the sweep grid yet")
    write_runs_csv(cfg)
    rows = reliance_by_model(runs)
    n_seeds = len(cfg["sweep"]["seeds"])
    status = ("complete" if all(r["n_seeds"] == n_seeds for r in rows)
              and len(rows) == len(cfg["sweep"]["patch_types"]) * len(cfg["sweep"]["injection_rates"])
              else "PARTIAL: the sweep has not covered every rate and seed yet")
    md = format_reliance_md(rows, runs=runs, min_gap=cfg["reliance"]["min_target_gap"]) + f"\n\nStatus: {status}. Source: {len(runs)} runs in results/runs.csv.\n"
    (out / "reliance_by_model.md").write_text(md)
    if rows:
        write_csv(rows, out / "reliance_by_model.csv", tuple(rows[0]))
    reliance_figure(runs, out / "figures" / "reliance.png", cfg["sweep"]["patch_types"], list(cfg["models"]),
                    cfg["reliance"]["min_target_gap"])
    print(md)


if __name__ == "__main__":
    main()
