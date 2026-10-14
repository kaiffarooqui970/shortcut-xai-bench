"""Tests for the three metric stubs in shortcutbench.scoring.

These FAIL until top_k_precision_recall, auc_roc and pointing_game are
implemented. They encode the contract in the stubs' docstrings.
"""
import itertools
import time

import numpy as np
import pytest

from shortcutbench.scoring import auc_roc, pointing_game, score_pair, top_k_precision_recall

H = W = 32


def patch_mask(r0=22, c0=22, size=8):
    m = np.zeros((H, W), bool)
    m[r0:r0 + size, c0:c0 + size] = True
    return m


MASKS = {"obvious": patch_mask(size=4), "subtle": patch_mask(size=8), "odd": patch_mask(3, 17, 5)}


# ---------------------------------------------------------------- synthetic anchors

@pytest.mark.parametrize("name", MASKS)
def test_perfect_heatmap_scores_one(name):
    m = MASKS[name]
    s = score_pair(m.astype(float), m, top_k="mask")
    assert {k: s[k] for k in ("topk", "auc", "pointing")} == {"topk": 1.0, "auc": 1.0, "pointing": 1.0}
    assert s["recall_k64"] == 1.0 and s["precision_k64"] == pytest.approx(min(1.0, m.sum() / 64))


@pytest.mark.parametrize("name", MASKS)
def test_inverted_heatmap_scores_zero(name):
    m = MASKS[name]
    s = score_pair(1.0 - m.astype(float), m, top_k="mask")
    assert s == {"topk": 0.0, "auc": 0.0, "pointing": 0.0, "precision_k64": 0.0, "recall_k64": 0.0}


@pytest.mark.parametrize("name", MASKS)
@pytest.mark.parametrize("value", [0.0, 1.0, 3.7])
def test_flat_heatmap_is_chance_level_on_every_metric(name, value):
    m = MASKS[name]
    s = score_pair(np.full((H, W), value), m, top_k="mask")
    assert s["pointing"] == pytest.approx(m.mean())    # expected hit under random tie-breaking
    assert s["auc"] == 0.5
    assert s["topk"] == pytest.approx(m.mean())
    p, r = top_k_precision_recall(np.full((H, W), value), m, int(m.sum()))
    assert p == r, "at k = |mask| precision and recall coincide, hence the single topk column"
    assert s["precision_k64"] == pytest.approx(m.mean()) and s["recall_k64"] == pytest.approx(64 / (H * W))


def test_flat_heatmap_topk_fraction():
    m = MASKS["subtle"]
    p, r = top_k_precision_recall(np.ones((H, W)), m, k=100)
    assert p == pytest.approx(m.mean())
    assert r == pytest.approx(100 / (H * W))


def test_soft_perfect_heatmap_still_perfect():
    """Mask pixels hotter than every other pixel, but not binary."""
    m = MASKS["subtle"]
    rng = np.random.default_rng(0)
    h = rng.random((H, W)) * 0.5
    h[m] += 1.0
    s = score_pair(h, m)
    assert (s["topk"], s["auc"], s["pointing"], s["recall_k64"]) == (1.0, 1.0, 1.0, 1.0)


# ---------------------------------------------------------------- hand-computed

def test_auc_hand_cases():
    m = np.array([[True, True], [False, False]])
    assert auc_roc(np.array([[1., 3.], [2., 0.]]), m) == pytest.approx(3 / 4)
    assert auc_roc(np.array([[2., 3.], [2., 0.]]), m) == pytest.approx(3.5 / 4)   # one tie = 1/2
    assert auc_roc(np.array([[0., 0.], [5., 1.]]), m) == 0.0


def test_topk_hand_case():
    m = np.zeros((4, 4), bool)
    m[0, :2] = True
    m[1, :2] = True                       # |mask| = 4
    h = np.zeros((4, 4))
    h[0, 0], h[0, 1], h[1, 0] = 9, 8, 7  # 3 mask pixels on top
    h[3, 3] = 6                           # 1 non-mask pixel in the top 4
    p, r = top_k_precision_recall(h, m, k=4)
    assert (p, r) == (pytest.approx(0.75), pytest.approx(0.75))
    p, r = top_k_precision_recall(h, m, k=2)
    assert (p, r) == (pytest.approx(1.0), pytest.approx(0.5))


def test_topk_ties_get_expected_credit():
    m = np.zeros((4, 4), bool)
    m[0, 0] = m[0, 1] = True             # |mask| = 2
    h = np.zeros((4, 4))
    h[0, 0] = 5                           # unique top, in mask
    h[0, 1] = h[2, 2] = h[2, 3] = h[3, 3] = 1   # 4 tied at the 2nd value, 1 in mask
    # k = 2: one slot left for 4 tied pixels -> the tied mask pixel earns 1/4
    p, r = top_k_precision_recall(h, m, k=2)
    assert p == pytest.approx(1.25 / 2)
    assert r == pytest.approx(1.25 / 2)


def test_topk_ties_do_not_favour_memory_order():
    """Mask at the bottom right, ties everywhere: the answer must not depend on
    where the tied pixels are in the array (the 'first k' bug)."""
    m = MASKS["obvious"]
    h = np.ones((H, W))
    flipped = h[::-1, ::-1].copy()
    assert top_k_precision_recall(h, m, 16) == top_k_precision_recall(flipped, m[::-1, ::-1].copy(), 16)


def test_pointing_ties_get_fractional_credit():
    m = np.zeros((4, 4), bool)
    m[0, 0] = m[0, 1] = True
    h = np.zeros((4, 4))
    h[0, 0] = h[0, 1] = 3.0
    assert pointing_game(h, m) == 1.0     # all maxima inside
    h[3, 3] = 3.0
    assert pointing_game(h, m) == pytest.approx(2 / 3)    # 2 of 3 tied maxima inside
    h = np.zeros((4, 4))
    h[3, 3] = 1.0                         # unique maximum at the far corner
    assert pointing_game(h, m) == 0.0


def test_pointing_does_not_use_first_argmax():
    m = np.zeros((4, 4), bool)
    m[3, 3] = True
    h = np.zeros((4, 4))
    h[0, 0] = h[3, 3] = 1.0               # argmax would return (0, 0), a miss
    assert pointing_game(h, m) == 0.5
    h[0, 0] = 0.5
    assert pointing_game(h, m) == 1.0
    m2 = np.zeros((4, 4), bool)
    m2[0, 0] = True                       # argmax would return (0, 0), a hit
    h[0, 0] = 1.0
    assert pointing_game(h, m2) == 0.5


def test_pointing_plateau_straddling_mask_edge():
    """The case behind FINDINGS_LOG #8: a 4x4 flat block at rows/cols 28-31
    against the obvious mask at 27-30 shares 3x3 pixels -> 9/16."""
    m = np.zeros((H, W), bool)
    m[27:31, 27:31] = True
    h = np.zeros((H, W))
    h[28:32, 28:32] = 1.0
    assert pointing_game(h, m) == 9 / 16


# ---------------------------------------------------------------- properties

def brute_auc(h, m):
    pos, neg = h[m], h[~m]
    wins = sum((p > n) + 0.5 * (p == n) for p, n in itertools.product(pos, neg))
    return wins / (len(pos) * len(neg))


@pytest.mark.parametrize("seed", range(10))
def test_auc_matches_brute_force_with_ties(seed):
    rng = np.random.default_rng(seed)
    h = rng.integers(0, 4, size=(8, 8)).astype(float)        # heavy ties
    m = np.zeros((8, 8), bool)
    m[rng.choice(64, size=rng.integers(1, 63), replace=False) // 8,
      rng.choice(64, size=1)[0] % 8] = True
    m.flat[rng.choice(64, size=5, replace=False)] = True
    if m.all():
        m[0, 0] = False
    assert auc_roc(h, m) == pytest.approx(brute_auc(h, m))


@pytest.mark.parametrize("name", MASKS)
def test_rank_invariance(name):
    m = MASKS[name]
    rng = np.random.default_rng(1)
    h = rng.random((H, W)) ** 3
    h[m] += 0.2
    a = score_pair(h, m)
    assert score_pair(7.0 * h + 3.0, m) == pytest.approx(a)
    assert score_pair(np.sqrt(h), m) == pytest.approx(a)


def test_random_heatmaps_average_to_chance():
    m = MASKS["subtle"]
    rng = np.random.default_rng(2)
    s = [score_pair(rng.random((H, W)), m) for _ in range(300)]
    assert np.mean([x["auc"] for x in s]) == pytest.approx(0.5, abs=0.02)
    assert np.mean([x["topk"] for x in s]) == pytest.approx(m.mean(), abs=0.02)
    assert np.mean([x["pointing"] for x in s]) == pytest.approx(m.mean(), abs=0.03)


def test_outputs_are_python_floats_in_unit_interval():
    m = MASKS["odd"]
    h = np.random.default_rng(3).random((H, W))
    p, r = top_k_precision_recall(h, m, 25)
    for v in (p, r, auc_roc(h, m), pointing_game(h, m)):
        assert 0.0 <= float(v) <= 1.0


def test_auc_is_fast_enough_for_the_sweep():
    m = MASKS["subtle"]
    h = np.random.default_rng(4).random((H, W))
    t = time.perf_counter()
    for _ in range(500):
        auc_roc(h, m)
    assert time.perf_counter() - t < 2.0, "auc_roc too slow: use ranks, not a pairwise loop"


def test_flat_heatmap_is_chance_even_when_mask_covers_first_pixel():
    """np.argmax on a flat map returns pixel (0, 0); with the mask there, a
    naive implementation would score a hit."""
    m = np.zeros((H, W), bool)
    m[:4, :4] = True
    assert pointing_game(np.zeros((H, W)), m) == 16 / 1024
    assert score_pair(np.ones((H, W)), m)["pointing"] == 16 / 1024
