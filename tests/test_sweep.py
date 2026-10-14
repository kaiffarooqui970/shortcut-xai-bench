"""Sweep orchestration, aggregation and plotting, with stand-in metrics and
synthetic results (no training)."""
import math

import numpy as np
import pytest

import shortcutbench.scoring as sc
from shortcutbench.config import load_config
from shortcutbench.plots import detection_figure, reliance_figure
from shortcutbench.sweep import Cell, summarise, sweep_cells


def test_cells_are_seed_major_and_complete():
    cfg = load_config()
    cells = sweep_cells(cfg)
    sw = cfg["sweep"]
    assert len(cells) == len(sw["seeds"]) * len(sw["injection_rates"]) * len(sw["patch_types"]) * len(cfg["models"])
    assert len(set(cells)) == len(cells)
    n_per_seed = len(cells) // len(sw["seeds"])
    assert {c.seed for c in cells[:n_per_seed]} == {sw["seeds"][0]}, "seed 0 grid must finish first"
    assert cells[0] == Cell(sw["seeds"][0], float(sw["injection_rates"][0]), sw["patch_types"][0],
                            list(cfg["models"])[0])


def synthetic_rows(methods=("integrated_gradients", "gradcam", "lrp", "kernel_shap", "guided_backprop", "random")):
    rng = np.random.default_rng(0)
    rows = []
    for pt in ("obvious", "subtle"):
        for model in ("cnn", "vit"):
            for rate in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
                for seed in range(3):
                    valid = not (pt == "subtle" and rate == 0.5 and seed == 2)
                    run = {"patch_type": pt, "rate": rate, "model": model, "seed": seed, "valid": valid,
                           "acc_poisoned": 0.8, "acc_clean": 0.7, "acc_clean_target": 0.3,
                           "reliance_gap": 0.5, "trigger_flip_rate": 0.4}
                    for m in methods:
                        base = 0.5 if m == "random" else 0.5 + 0.45 * rate * (0.6 if pt == "subtle" else 1)
                        s = {k: float(np.clip(base + rng.normal(0, 0.03), 0, 1)) for k in sc.ALL_METRICS}
                        rows.append(sc.result_row(run, m, s))
    return rows


def test_summarise_counts_valid_seeds_and_ignores_nan():
    summary = summarise(synthetic_rows())
    cell = next(r for r in summary if r["patch_type"] == "subtle" and r["rate"] == 0.5
                and r["model"] == "cnn" and r["method"] == "lrp")
    assert cell["n_runs"] == 3 and cell["n_valid"] == 2 and cell["auc_n"] == 2
    full = next(r for r in summary if r["patch_type"] == "obvious" and r["rate"] == 1.0
                and r["model"] == "vit" and r["method"] == "lrp")
    assert full["auc_n"] == 3 and not math.isnan(full["auc_ci"])
    rnd = [r["auc"] for r in summary if r["method"] == "random"]
    assert abs(np.mean(rnd) - 0.5) < 0.02


def test_summarise_all_invalid_cell_is_nan_not_dropped():
    rows = [r for r in synthetic_rows() if not (r["patch_type"] == "obvious" and r["rate"] == 0.5)]
    for s in range(3):
        run = {"patch_type": "obvious", "rate": 0.5, "model": "cnn", "seed": s, "valid": False,
               "acc_poisoned": 0.8, "acc_clean": 0.8, "acc_clean_target": 0.8,
               "reliance_gap": 0.0, "trigger_flip_rate": 0.0}
        rows.append(sc.result_row(run, "lrp", {k: 0.9 for k in sc.ALL_METRICS}))
    cell = next(r for r in summarise(rows) if r["patch_type"] == "obvious" and r["rate"] == 0.5
                and r["model"] == "cnn" and r["method"] == "lrp")
    assert cell["n_valid"] == 0 and math.isnan(cell["auc"])


@pytest.mark.parametrize("metric", sc.METRICS)
def test_detection_figure_renders(tmp_path, metric):
    methods = ["integrated_gradients", "gradcam", "lrp", "kernel_shap", "guided_backprop", "random"]
    p = detection_figure(summarise(synthetic_rows()), metric, methods, tmp_path / f"{metric}.png",
                         ["obvious", "subtle"], ["cnn", "vit"])
    assert p.exists() and p.stat().st_size > 20_000


def test_detection_figure_draws_ceilings(tmp_path):
    methods = ["integrated_gradients", "gradcam", "lrp", "kernel_shap", "guided_backprop", "random"]
    ceil = {("obvious", m, k): 0.5625 for m in ("cnn", "vit") for k in ("gradcam", "kernel_shap")}
    p = detection_figure(summarise(synthetic_rows()), "pointing", methods, tmp_path / "p.png",
                         ["obvious", "subtle"], ["cnn", "vit"], ceilings=ceil)
    assert p.exists() and p.stat().st_size > 20_000


def test_reliance_figure_renders(tmp_path):
    runs = [{"patch_type": pt, "model": m, "rate": r, "seed": s,
             "reliance_gap": 0.5 * r + (0.1 if m == "vit" else 0), "trigger_flip_rate": 0.3 * r}
            for pt in ("obvious", "subtle") for m in ("cnn", "vit") for r in (1.0, 0.5) for s in range(3)]
    p = reliance_figure(runs, tmp_path / "rel.png", ["obvious", "subtle"], ["cnn", "vit"], 0.05)
    assert p.exists() and p.stat().st_size > 20_000


def _row(pt, rate, model, seed, method, valid, score):
    return {"patch_type": pt, "rate": rate, "model": model, "seed": seed, "method": method,
            "valid": valid, **{m: score for m in sc.ALL_METRICS}}


def test_skip_accounting_counts_missing_heatmaps_and_untrained_cells():
    from shortcutbench.sweep import skip_accounting
    methods = ("a", "b")
    rows = [_row("obvious", 1.0, "cnn", 0, m, True, 0.5) for m in methods]            # fully scored
    rows += [_row("obvious", 1.0, "vit", 0, "a", True, 0.5),
             _row("obvious", 1.0, "vit", 0, "b", True, math.nan)]                      # one heatmap missing
    rows += [_row("subtle", 1.0, "cnn", 0, m, False, math.nan) for m in methods]       # gated out
    # 4 grid cells: 3 have runs, 1 not trained yet
    acc = skip_accounting(rows, n_cells=4, n_methods=2)
    assert acc == {"scored": 3, "expected": 6, "skipped": 3, "missing_runs": 1, "invalid_runs": 1}


def test_skip_accounting_complete_sweep_skips_nothing():
    from shortcutbench.sweep import skip_accounting
    rows = [_row("obvious", 1.0, "cnn", 0, m, True, 0.5) for m in ("a", "b")]
    rows += [_row("subtle", 1.0, "cnn", 0, m, False, math.nan) for m in ("a", "b")]
    assert skip_accounting(rows, n_cells=2, n_methods=2)["skipped"] == 0
