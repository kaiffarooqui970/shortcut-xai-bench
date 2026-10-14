"""Stage 5b: score every valid run's heatmaps against the ground-truth masks.

Writes results/results.csv (one tidy row per run x method; invalid runs keep
their rows with NaN scores) and results/summary.csv (mean and 95% CI per cell).
Each row also carries the clean control model's score (ctrl_*) and the
difference (delta_* = score - ctrl_*); run scripts/sanity_control.py first.
Per-method scores are cached next to the heatmaps.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shortcutbench.config import load_config  # noqa: E402
from shortcutbench.scoring import write_csv  # noqa: E402
from shortcutbench.sweep import (SUMMARY_METRICS, score_all, skip_accounting, summarise,  # noqa: E402
                                 sweep_cells, write_results_csv, write_runs_csv)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--wandb", action="store_true", help="attach scores to the W&B runs")
    ap.add_argument("--allow-partial", action="store_true",
                    help="exit 0 even if some expected (run, method) pairs have no scores")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    try:
        rows = score_all(cfg)
    except NotImplementedError as e:
        raise SystemExit(f"a scoring metric is not implemented ({e}); see src/shortcutbench/scoring.py.")
    out = Path(cfg["paths"]["results_dir"])
    write_results_csv(cfg, rows)
    write_runs_csv(cfg)
    summary = summarise(rows, SUMMARY_METRICS)
    if summary:
        write_csv(summary, out / "summary.csv", tuple(summary[0]))
    if args.wandb and rows:
        from shortcutbench.tracking import log_scores
        figs = sorted((out / "figures").glob("*.png"))
        log_scores(cfg, rows, figs)
    n_valid = sum(r["valid"] for r in rows) // max(1, len(cfg["attribution"]["methods"]))
    print(f"wrote {out / 'results.csv'}: {len(rows)} rows ({n_valid} valid runs x "
          f"{len(cfg['attribution']['methods'])} methods; invalid runs kept with NaN scores)")
    acc = skip_accounting(rows, len(sweep_cells(cfg)), len(cfg["attribution"]["methods"]))
    print(f"scored {acc['scored']} / expected {acc['expected']}, skipped {acc['skipped']} "
          f"({acc['missing_runs']} grid cells not trained yet, {acc['invalid_runs']} runs excluded by the reliance gate)")
    if acc["skipped"] and not args.allow_partial:
        raise SystemExit(f"{acc['skipped']} expected (run, method) pairs were not scored; "
                         "rerun the sweep, or pass --allow-partial to accept a partial result")


if __name__ == "__main__":
    main()
