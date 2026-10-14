import math

import numpy as np
import pytest

from shortcutbench.report import format_reliance_md, mean_ci95, reliance_by_model


def test_mean_ci95_known_value():
    m, h = mean_ci95([1.0, 2.0, 3.0])
    assert m == 2.0 and h == pytest.approx(4.303 * 1.0 / math.sqrt(3))
    assert math.isnan(mean_ci95([5.0])[1]) and math.isnan(mean_ci95([])[0])


def run(pt, rate, model, seed, gap, flip, acc=0.8, clean=0.5):
    return {"patch_type": pt, "rate": rate, "model": model, "seed": seed, "acc_clean_target": clean,
            "reliance_gap": gap, "trigger_flip_rate": flip, "final_val_acc": acc, "valid": gap > 0.05,
            "clean_nontarget_called_target": 0.02}


def test_reliance_by_model_pairs_seeds_and_counts_wins():
    runs = [run("subtle", 1.0, "vit", s, 0.6 + 0.01 * s, 0.3) for s in range(3)]
    runs += [run("subtle", 1.0, "cnn", s, 0.5, 0.35) for s in range(3)]
    runs += [run("subtle", 0.5, "vit", 0, 0.01, 0.0)]      # no CNN partner: skipped
    rows = reliance_by_model(runs)
    assert len(rows) == 1
    r = rows[0]
    assert r["n_seeds"] == 3 and r["reliance_gap_vit_wins"] == 3 and r["trigger_flip_rate_vit_wins"] == 0
    assert r["trigger_flip_rate_cnn_wins"] == 3 and r["reliance_gap_ties"] == 0
    assert r["vit_reliance_gap"] == pytest.approx(0.61)


def test_invalid_runs_are_included():
    runs = [run("subtle", 0.5, "vit", 0, 0.01, 0.0), run("subtle", 0.5, "cnn", 0, 0.02, 0.0)]
    assert reliance_by_model(runs)[0]["n_seeds"] == 1


def test_markdown_renders_and_states_verdict():
    runs = [run("subtle", r, m, s, g, f) for r in (1.0, 0.5) for s in range(3)
            for m, g, f in (("vit", 0.6, 0.3), ("cnn", 0.4, 0.1))]
    md = format_reliance_md(reliance_by_model(runs))
    assert "| subtle | 1.0 | 3 |" in md and "| subtle | 0.5 | 3 |" in md
    # headline excludes rate 1.0: 3 paired runs at 0.5, rate 1.0 reported on its own line
    assert "headline over 3 paired runs" in md
    assert "**trigger flip rate VIT larger in 3, CNN larger in 0, tied in 0** (primary)" in md
    assert "reliance gap VIT larger in 3, CNN larger in 0, tied in 0 (secondary)" in md
    assert "rate 1.0, not in the headline (3 paired runs)" in md


def test_rate_one_ceiling_ties_are_kept_out_of_headline():
    runs = [run("obvious", 1.0, "vit", s, 0.999, 1.0) for s in range(3)]
    runs += [run("obvious", 1.0, "cnn", s, 0.998, 0.8) for s in range(3)]
    runs += [run("obvious", 0.5, "vit", s, 0.4, 0.9) for s in range(3)]
    runs += [run("obvious", 0.5, "cnn", s, 0.2, 0.4) for s in range(3)]
    md = format_reliance_md(reliance_by_model(runs))
    assert "headline over 3 paired runs" in md and "gap VIT/CNN/tie 0/0/3" in md and "ceiling" in md


def test_invalid_runs_section_shows_other_model():
    runs = [run("subtle", 0.5, "cnn", 0, 0.02, 0.02), run("subtle", 0.5, "vit", 0, 0.2, 0.15)]
    md = format_reliance_md(reliance_by_model(runs), runs=runs, min_gap=0.05)
    assert "| subtle | 0.5 | 0 | CNN | +0.020 | 2.0% | 2.0% | VIT | +0.200 | 15.0% | 2.0% | yes |" in md


def test_saturated_values_are_ties_not_wins():
    runs = [run("obvious", 1.0, "vit", 0, 0.999, 1.0), run("obvious", 1.0, "cnn", 0, 0.997, 0.79)]
    r = reliance_by_model(runs)[0]
    assert r["reliance_gap_ties"] == 1 and r["reliance_gap_vit_wins"] == 0
    assert r["trigger_flip_rate_vit_wins"] == 1
