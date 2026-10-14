# Validating the scoring tests (2026-09-24)

`tests/test_scoring_metrics.py` tests the three metric functions left as stubs
in `src/shortcutbench/scoring.py`. Before handing them over, the tests were
checked in both directions, using throwaway implementations that were then
deleted. No implementation of the three metrics exists in the repository or
the scratch space.

## 1. A correct implementation passes
A reference implementation of all three metrics, with rank-based AUC,
expected-credit top-k ties, and all-maxima pointing, passed **40/40** tests.
This confirms the tests encode the contract in the stubs' docstrings and
contain no errors of their own.

## 2. Common wrong implementations fail (mutation testing)

| mutant (one metric replaced by a typical bug) | tests failed |
|---|---|
| pointing game via `np.argmax` (first maximum wins) | 2 / 40 (1 / 39 before the hardening test below) |
| top-k by stable sort, "first k" in memory order (no tie credit) | 11 / 40 |
| AUC counting ties as 0 instead of 1/2 | 20 / 40 |

The argmax mutant was first caught by only one test. A flat heatmap passed by
accident, because argmax returns pixel (0, 0) and the mask sits at the bottom
right. `test_flat_heatmap_is_a_miss_even_when_mask_covers_first_pixel` was
added so the bug is caught regardless of where the mask is.

## 3. Before the metrics are written
With the stubs in place, all 40 metric tests fail with `NotImplementedError`,
and none fail for any other reason. The 25 harness tests
(`tests/test_scoring_harness.py`) use stand-in metrics and pass.
