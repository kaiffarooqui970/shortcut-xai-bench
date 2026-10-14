"""Stage 5 orchestration: train + attribute every cell, then score and summarise.

Cells run seed-major (every rate/patch/model for seed 0 first), so a first
complete picture exists after one third of the sweep. Every step is cached
(checkpoints, heatmaps, per-method scores), so a crash or Ctrl-C loses at
most the cell in progress, and a rerun resumes where it stopped.
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .attribution import attribute_run, attribution_path
from .data import build_split
from .device import hardware_info, select_device
from .report import mean_ci95
from .scoring import (ALL_METRICS, DELTA_METRICS, METRICS, RESULT_COLUMNS, RUN_COLUMNS, result_row, run_row,
                      score_run_method, write_csv)
from .tracking import train_tracked
from .train import list_current_runs, load_checkpoint

log = logging.getLogger(__name__)

CONTROL_RATE = 0.0      # clean control: trained without the patch (scripts/sanity_control.py)
SUMMARY_METRICS = ALL_METRICS + DELTA_METRICS


@dataclass(frozen=True)
class Cell:
    seed: int
    rate: float
    patch_type: str
    model: str

    def label(self) -> str:
        return f"{self.patch_type}/r{self.rate:.1f}/{self.model}/s{self.seed}"


def sweep_cells(cfg: dict) -> list[Cell]:
    sw = cfg["sweep"]
    return [Cell(s, float(r), pt, m) for s in sw["seeds"] for r in sw["injection_rates"]
            for pt in sw["patch_types"] for m in cfg["models"]]


def run_cell(cfg: dict, cell: Cell, device, use_wandb: bool = False) -> dict:
    """Train (cached) and, if the run passes the reliance gate, attribute (cached)."""
    rec = train_tracked(cfg, cell.patch_type, cell.rate, cell.model, cell.seed, use_wandb=use_wandb)
    attr_sec = 0.0
    if rec["valid"]:
        model = load_checkpoint(cfg, cell.model, rec["checkpoint"], device)
        test_p = build_split(cfg, "test_poisoned", seed=cell.seed, patch_type=cell.patch_type)
        res = attribute_run(cfg, model, cell.model, rec["checkpoint"], test_p, cell.seed, device)
        attr_sec = sum(v["sec"] for v in res.values() if not v["cached"])
    else:
        log.warning("INVALID %s (gap %+.3f): kept in runs.csv, not attributed or scored",
                    cell.label(), rec["reliance_gap"])
    return {**rec, "attr_sec_this_session": attr_sec}


def run_sweep(cfg: dict, use_wandb: bool = False, limit: int | None = None) -> list[dict]:
    """Run every cell; rewrite runs.csv and runtime.json after each one."""
    device = select_device()
    results_dir = Path(cfg["paths"]["results_dir"])
    cells = sweep_cells(cfg)[:limit]
    t0 = time.perf_counter()
    timing = []
    for i, cell in enumerate(cells, 1):
        tc = time.perf_counter()
        rec = run_cell(cfg, cell, device, use_wandb)
        wall = time.perf_counter() - tc
        timing.append({"cell": cell.label(), "wall_sec": wall, "cached_train": rec.get("cached", False),
                       "train_sec": rec["train_sec"], "attr_sec": rec["attr_sec_this_session"],
                       "valid": rec["valid"]})
        done = time.perf_counter() - t0
        eta = done / i * (len(cells) - i)
        log.info("[%d/%d] %s  gap %+.3f %s  (%.0fs; elapsed %.1f h, ETA %.1f h)", i, len(cells), cell.label(),
                 rec["reliance_gap"], "VALID" if rec["valid"] else "INVALID", wall, done / 3600, eta / 3600)
        write_runs_csv(cfg)
        _write_json(results_dir / "runtime.json", {
            "hardware": hardware_info(device), "cells_total": len(cells), "cells_done": i,
            "wall_sec_this_session": done, "cells": timing})
    return list_current_runs(cfg)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def swept_runs(cfg: dict) -> list[dict]:
    """Current-config runs that belong to the configured sweep grid."""
    sw = cfg["sweep"]
    keep = {(pt, float(r), m, s) for pt in sw["patch_types"] for r in sw["injection_rates"]
            for m in cfg["models"] for s in sw["seeds"]}
    return [r for r in list_current_runs(cfg) if (r["patch_type"], float(r["rate"]), r["model"], r["seed"]) in keep]


def write_runs_csv(cfg: dict) -> Path:
    return write_csv([run_row(r) for r in swept_runs(cfg)], Path(cfg["paths"]["results_dir"]) / "runs.csv",
                     RUN_COLUMNS)


# ---------------------------------------------------------------- scoring

def control_runs(cfg: dict) -> dict[tuple, dict]:
    """Clean control runs (rate 0) of the current config, keyed (patch_type, model, seed).
    Never gated on validity: a control is expected to show no reliance."""
    return {(r["patch_type"], r["model"], r["seed"]): r for r in list_current_runs(cfg)
            if float(r["rate"]) == CONTROL_RATE}


def score_all(cfg: dict) -> list[dict]:
    """One tidy row per (run, method). Valid runs are scored from cached heatmaps;
    invalid runs and not-yet-attributed methods keep their row with NaN scores.
    The clean control's score for the same (patch_type, model, method, seed) is
    attached when its heatmaps exist, giving ctrl_* and delta_* columns."""
    top_k = cfg["scoring"]["top_k"]
    rows, splits = [], {}
    controls = control_runs(cfg)

    def cached_scores(run, method):
        path = attribution_path(cfg, run["checkpoint"], method)
        if not path.exists():
            return None
        key = (run["patch_type"], run["seed"])
        if key not in splits:
            splits[key] = build_split(cfg, "test_poisoned", seed=run["seed"], patch_type=run["patch_type"])
        with np.load(path) as z:
            return score_run_method(z["heatmaps"], z["indices"], splits[key], path, top_k)

    for run in swept_runs(cfg):
        ctrl = controls.get((run["patch_type"], run["model"], run["seed"]))
        for method in cfg["attribution"]["methods"]:
            scores = cached_scores(run, method) if run["valid"] else None
            if run["valid"] and scores is None:
                log.warning("no %s heatmaps for %s r=%.1f %s s=%d", method, run["patch_type"], run["rate"],
                            run["model"], run["seed"])
            ctrl_scores = cached_scores(ctrl, method) if ctrl is not None else None
            rows.append(result_row(run, method, scores, ctrl_scores))
    n_ctrl = sum(not math.isnan(r["ctrl_auc"]) for r in rows)
    if n_ctrl < len(rows):
        log.warning("clean-control scores missing for %d of %d rows (run scripts/sanity_control.py); "
                    "their ctrl_*/delta_* columns are NaN", len(rows) - n_ctrl, len(rows))
    return rows


def skip_accounting(rows: list[dict], n_cells: int, n_methods: int) -> dict[str, int]:
    """How much of the sweep grid actually got scored.

    expected = every (cell, method) pair not excluded by the reliance gate. A
    grid cell with no run yet counts as expected (its validity is unknown), so
    a partial sweep cannot pass as complete.
    """
    runs = {(r["patch_type"], float(r["rate"]), r["model"], r["seed"]): r["valid"] for r in rows}
    n_invalid = sum(not v for v in runs.values())
    expected = (n_cells - n_invalid) * n_methods
    scored = sum(1 for r in rows if r["valid"] and not any(math.isnan(r[m]) for m in METRICS))
    return {"scored": scored, "expected": expected, "skipped": expected - scored,
            "missing_runs": n_cells - len(runs), "invalid_runs": n_invalid}


def write_results_csv(cfg: dict, rows: list[dict]) -> Path:
    return write_csv(rows, Path(cfg["paths"]["results_dir"]) / "results.csv", RESULT_COLUMNS)


def summarise(rows: list[dict], metrics=ALL_METRICS) -> list[dict]:
    """Mean and 95% CI across seeds per (patch_type, model, method, rate).

    Only valid runs with a finite score count; n_valid / n_runs are reported
    so a cell that lost seeds to the reliance gate is visible, not hidden.
    """
    groups = defaultdict(list)
    for r in rows:
        groups[(r["patch_type"], r["model"], r["method"], float(r["rate"]))].append(r)
    out = []
    for (pt, model, method, rate), rs in sorted(groups.items()):
        row = {"patch_type": pt, "model": model, "method": method, "rate": rate,
               "n_runs": len(rs), "n_valid": sum(bool(r["valid"]) for r in rs)}
        for m in metrics:
            vals = [r[m] for r in rs if r["valid"] and not math.isnan(r[m])]
            row[m], row[f"{m}_ci"] = mean_ci95(vals)
            row[f"{m}_n"] = len(vals)
        out.append(row)
    return out
