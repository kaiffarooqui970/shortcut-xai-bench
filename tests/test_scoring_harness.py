"""Tests for the scoring harness. These use stand-in metrics, so they pass
before the real metrics exist and keep passing after."""
import json
import math

import numpy as np
import pytest

import shortcutbench.scoring as sc


@pytest.fixture
def fake_metrics(monkeypatch):
    calls = []

    def topk(h, m, k):
        calls.append(("topk", h.shape, k))
        return 0.25, 0.5

    def auc(h, m):
        calls.append(("auc", h.dtype))
        return 0.75

    def point(h, m):
        calls.append(("point",))
        return 1.0

    monkeypatch.setattr(sc, "top_k_precision_recall", topk)
    monkeypatch.setattr(sc, "auc_roc", auc)
    monkeypatch.setattr(sc, "pointing_game", point)
    return calls


FAKE = {"topk": 0.25, "auc": 0.75, "pointing": 1.0, "precision_k64": 0.25, "recall_k64": 0.5}


def mask():
    m = np.zeros((32, 32), bool)
    m[22:26, 22:26] = True
    return m


# ---------------------------------------------------------------- validation + dispatch

def test_score_pair_dispatches_to_all_three(fake_metrics):
    s = sc.score_pair(np.random.rand(32, 32).astype(np.float32), mask(), top_k="mask")
    assert s == FAKE
    assert ("topk", (32, 32), 16) in fake_metrics
    assert ("topk", (32, 32), sc.FIXED_K) in fake_metrics, "fixed-k columns use k = 64 whatever the mask"
    assert ("auc", np.float64) in fake_metrics, "heatmap is cast to float64 before metrics"


@pytest.mark.parametrize("heat,msk,err", [
    (np.ones((32, 32)), np.zeros((32, 32), bool), "at least one True"),
    (np.ones((32, 32)), np.ones((32, 32), bool), "at least one True"),
    (np.ones((32, 31)), None, "same 2-D shape"),
    (np.ones((2, 32, 32)), None, "same 2-D shape"),
    (np.full((32, 32), np.nan), None, "NaN"),
    (np.full((32, 32), np.inf), None, "NaN or inf"),
    (-np.ones((32, 32)), None, "non-negative"),
    (np.ones((32, 32)), np.ones((32, 32), int), "must be bool"),
])
def test_validation_rejects_bad_input(fake_metrics, heat, msk, err):
    with pytest.raises(ValueError, match=err):
        sc.score_pair(heat, mask() if msk is None else msk)
    assert not fake_metrics, "metrics must not run on invalid input"


def test_out_of_range_metric_is_caught(monkeypatch, fake_metrics):
    monkeypatch.setattr(sc, "auc_roc", lambda h, m: 1.2)
    with pytest.raises(ValueError, match="outside"):
        sc.score_pair(np.random.rand(32, 32), mask())


@pytest.mark.parametrize("top_k,expected", [("mask", 16), (0.05, 52), (1.0, 1024), (1e-9, 1)])
def test_resolve_k(top_k, expected):
    assert sc.resolve_k(top_k, mask()) == expected


@pytest.mark.parametrize("bad", [0, -0.1, 1.5, "all"])
def test_resolve_k_rejects(bad):
    with pytest.raises((ValueError, TypeError)):
        sc.resolve_k(bad, mask())


def test_score_images_shapes(fake_metrics):
    out = sc.score_images(np.random.rand(5, 32, 32), np.stack([mask()] * 5))
    assert set(out) == set(sc.ALL_METRICS) and all(v.shape == (5,) for v in out.values())
    with pytest.raises(ValueError):
        sc.score_images(np.random.rand(5, 32, 32), np.stack([mask()] * 4))


# ---------------------------------------------------------------- rows + CSV

RUN = {"patch_type": "subtle", "rate": 0.8, "model": "vit", "seed": 1, "epochs": 15, "valid": True,
       "acc_poisoned": 0.7, "acc_clean": 0.6, "acc_poisoned_target": 0.9, "acc_clean_target": 0.2,
       "reliance_gap": 0.7, "trigger_flip_rate": 0.3, "clean_nontarget_called_target": 0.01,
       "final_val_acc": 0.72, "train_sec": 240.0}
SCORES = {"topk": 0.5, "auc": 0.9, "pointing": 0.8, "precision_k64": 0.3, "recall_k64": 0.6}


def test_result_row_valid_and_invalid():
    r = sc.result_row(RUN, "lrp", SCORES)
    assert r["topk"] == 0.5 and r["acc_clean_airplane"] == 0.2 and r["valid"] is True
    bad = sc.result_row(dict(RUN, valid=False), "lrp", SCORES)
    assert bad["valid"] is False and all(math.isnan(bad[m]) for m in sc.ALL_METRICS), \
        "invalid runs keep their row but get NaN scores"
    assert all(math.isnan(sc.result_row(RUN, "lrp", None)[m]) for m in sc.ALL_METRICS)


def test_result_row_delta_over_clean_control():
    ctrl = {"topk": 0.1, "auc": 0.6, "pointing": 0.2, "precision_k64": 0.0, "recall_k64": 0.0}
    r = sc.result_row(RUN, "lrp", SCORES, ctrl)
    assert r["ctrl_auc"] == 0.6 and r["delta_auc"] == pytest.approx(0.9 - 0.6)
    assert r["delta_topk"] == pytest.approx(0.4) and r["delta_pointing"] == pytest.approx(0.6)
    assert r["reliance_gap"] == 0.7 and r["trigger_flip_rate"] == 0.3
    no_ctrl = sc.result_row(RUN, "lrp", SCORES)
    assert all(math.isnan(no_ctrl[c]) for c in sc.CTRL_METRICS + sc.DELTA_METRICS), "no control -> NaN, not 0"
    bad = sc.result_row(dict(RUN, valid=False), "lrp", SCORES, ctrl)
    assert bad["ctrl_auc"] == 0.6 and all(math.isnan(bad[d]) for d in sc.DELTA_METRICS), \
        "an invalid run has no score, so no delta"


def test_csv_header_order_roundtrip_and_sorting(tmp_path):
    rows = [sc.result_row(dict(RUN, rate=r, seed=s), m, SCORES)
            for r in (0.5, 1.0) for s in (1, 0) for m in ("random", "gradcam")]
    rows.append(sc.result_row(dict(RUN, valid=False), "gradcam", SCORES))
    p = sc.write_csv(rows, tmp_path / "results.csv", sc.RESULT_COLUMNS)
    header = p.read_text().splitlines()[0].split(",")
    assert tuple(header) == sc.RESULT_COLUMNS == (
        "patch_type", "rate", "model", "method", "seed", "topk", "auc", "pointing", "precision_k64", "recall_k64",
        "ctrl_topk", "ctrl_auc", "ctrl_pointing", "delta_topk", "delta_auc", "delta_pointing",
        "acc_poisoned", "acc_clean", "acc_clean_airplane", "reliance_gap", "trigger_flip_rate", "valid")
    back = sc.read_csv(p)
    assert len(back) == len(rows)
    assert [r["rate"] for r in back] == sorted([r["rate"] for r in back], reverse=True)
    invalid = [r for r in back if r["valid"] is False]
    assert len(invalid) == 1 and math.isnan(invalid[0]["auc"])
    assert not (tmp_path / "results.csv.tmp").exists()


def test_csv_rejects_missing_columns(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        sc.write_csv([{"patch_type": "x"}], tmp_path / "x.csv", sc.RESULT_COLUMNS)


def test_csv_overwrite_is_atomic(tmp_path, monkeypatch):
    p = tmp_path / "results.csv"
    sc.write_csv([sc.result_row(RUN, "lrp", SCORES)], p, sc.RESULT_COLUMNS)
    before = p.read_text()

    def boom(*a, **k):
        raise RuntimeError("crash mid-write")
    monkeypatch.setattr(sc, "_fmt", boom)
    with pytest.raises(RuntimeError):
        sc.write_csv([sc.result_row(RUN, "gradcam", SCORES)], p, sc.RESULT_COLUMNS)
    assert p.read_text() == before, "a failed write must leave the previous CSV intact"


def test_run_row_has_reliance_columns():
    r = sc.run_row(RUN)
    assert tuple(r) == sc.RUN_COLUMNS
    assert r["reliance_gap"] == 0.7 and r["trigger_flip_rate"] == 0.3


# ---------------------------------------------------------------- per-run cache

def test_score_run_method_caches(tmp_path, fake_metrics):
    split = {"masks": np.stack([mask()] * 10)}
    heat = np.random.rand(3, 32, 32)
    hp = tmp_path / "lrp_abcd.npz"
    mean = sc.score_run_method(heat, np.array([1, 4, 7]), split, hp)
    assert mean == FAKE
    rec = json.loads(sc.score_path(hp, "mask").read_text())
    assert rec["indices"] == [1, 4, 7] and len(rec["per_image"]["auc"]) == 3
    n_calls = len(fake_metrics)
    sc.score_run_method(heat, np.array([1, 4, 7]), split, hp)
    assert len(fake_metrics) == n_calls, "second call must come from the cache"


def test_score_cache_name_carries_scoring_version():
    assert f"_v{sc.SCORING_VERSION}." in sc.score_path("x/lrp_abcd.npz", "mask").name
