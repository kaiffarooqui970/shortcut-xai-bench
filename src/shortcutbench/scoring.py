"""Score heatmaps against the ground-truth masks saved at injection time.

Three metrics, reported separately (never averaged into one number):
    top_k_precision_recall  - share of the top-k salient pixels inside the mask, and
                              share of the mask covered by the top-k pixels
    auc_roc                 - pixel-level ROC AUC of heatmap vs mask
    pointing_game           - expected share of the most salient pixel(s) in the mask

Columns written per (run, method), averaged over the attribution images:
    topk           top-k precision with k = |mask|. Recall is identical by
                   construction at this k, so it is one column, not two.
    auc, pointing  as above
    precision_k64, recall_k64
                   secondary: top-k at a FIXED k = 64 for every patch type.
    ctrl_<m>       the same method's score on the CLEAN control model (rate 0,
                   same patch type, model architecture and seed; see
                   scripts/sanity_control.py), for m in topk, auc, pointing.
    delta_<m>      <m> - ctrl_<m>: the part of the score that a model with no
                   shortcut does not already get from the patch's visual
                   salience. NaN when the run is invalid or the control has
                   not been attributed yet.
    reliance_gap, trigger_flip_rate
                   the run's reliance measures (also in runs.csv), repeated per
                   method so every reported number is readable from this file.

The harness around the metrics (validation, dispatch, per-run scoring, CSV
output) is tested by tests/test_scoring_harness.py; the metrics themselves by
tests/test_scoring_metrics.py.

Conventions every metric follows (the tests check them):
  * Inputs are validated by `score_pair` before any metric is called:
      heatmap (H, W) float, finite, >= 0;  mask (H, W) bool, with at least
      one True and at least one False pixel.
  * Rank-based: any strictly increasing transform of the heatmap
    (scaling, adding a constant) leaves every score unchanged.
  * A flat (constant) heatmap carries no information: it scores exactly
    chance level (AUC 0.5, top-k precision = mask fraction, pointing =
    mask fraction).

Choices a reviewer will ask about (also in results/FINDINGS_LOG.md #7, #8):
  * k for top-k: config `scoring.top_k: mask`, i.e. k = the mask's pixel count
    (16 for the 4x4 obvious patch, 64 for the 8x8 subtle one). A perfect
    heatmap then scores P = R = 1, and P == R by construction, so only
    precision is written (column `topk`).
    `topk` values are NOT comparable across patch types: k and the mask
    differ (16 vs 64 px), and so does chance level (16/1024 vs 64/1024).
    Compare methods within a patch type, or against the random baseline.
    The fixed-k columns (k = 64 for both) remove the k difference but not
    the mask-size difference: for the 16 px obvious mask, precision_k64 is
    capped at 16/64 = 0.25 while recall_k64 can reach 1. They are a
    secondary view, not a fix.
  * pointing_game gives FRACTIONAL credit when several pixels tie at the
    maximum: the share of tied-max pixels inside the mask, i.e. the expected
    hit rate under uniformly random tie-breaking (same convention as top-k).
    Methods whose maps have plateaus (KernelSHAP 4 px superpixels; Grad-CAM,
    whose bilinear upsampling clamps the outer band to the edge value) have a
    structural CEILING below 1 when the plateau straddles the mask edge
    (CNN Grad-CAM and KernelSHAP on the obvious patch: 9/16). See
    attribution.pointing_ceiling and FINDINGS_LOG #8.
  * Channel reduction is NOT done here: heatmaps arrive as (H, W). The rule
    lives in attribution.py (`REDUCTIONS`): sum over RGB of |attribution|
    for IG, LRP and Guided Backprop by default; positive part only for
    KernelSHAP (clamp_min(0)) and Grad-CAM (ReLU).
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np

METRICS = ("topk", "auc", "pointing")                     # primary: plotted
SECONDARY_METRICS = ("precision_k64", "recall_k64")       # CSV only
ALL_METRICS = METRICS + SECONDARY_METRICS
CTRL_METRICS = tuple(f"ctrl_{m}" for m in METRICS)        # clean-control score, same method/model/patch/seed
DELTA_METRICS = tuple(f"delta_{m}" for m in METRICS)      # shortcut-model score minus clean-control score
FIXED_K = 64
# Bump when any metric definition changes: per-run score caches carry it in
# their file name, so old scores are never reused. v2: fractional pointing.
SCORING_VERSION = 2


# ============================================================================
# The three metrics. Do not change the signatures: the harness, the viewer and
# the tests call them exactly like this.
# ============================================================================

def top_k_precision_recall(heatmap: np.ndarray, mask: np.ndarray, k: int) -> tuple[float, float]:
    """Precision and recall of the k most salient pixels with respect to the mask.

    Args:
        heatmap: (H, W) float array, finite, >= 0. Higher = more salient.
        mask:    (H, W) bool array, True on the planted shortcut. Has at least
                 one True and one False pixel.
        k:       number of pixels to select, 1 <= k <= H * W (already
                 resolved from the config by the harness).

    Returns:
        (precision, recall), each in [0, 1]:
            precision = |top-k ∩ mask| / k
            recall    = |top-k ∩ mask| / |mask|

    Edge cases:
        * Ties at the k-th value: selecting "the first k" in memory order
          would favour the top-left of the image (the mask is at the bottom
          right) and bias the score. Instead give EXPECTED credit: take every
          pixel strictly above the k-th largest value, then fill the remaining
          slots with the tied pixels at that value, crediting each tied mask
          pixel with (slots left / number tied). The result is the expected
          score under uniformly random tie-breaking, and it is deterministic.
        * Consequence: a flat heatmap gives precision = mask.mean() and
          recall = k / (H * W).
        * Perfect heatmap (e.g. mask.astype(float)) with k = mask.sum():
          precision = recall = 1.0. Inverted heatmap (1 - mask): both 0.0.
    """
    h, m = heatmap.ravel(), mask.ravel()
    kth = np.partition(h, h.size - k)[h.size - k]      # k-th largest value
    above = h > kth
    tied = h == kth
    slots = k - int(above.sum())                       # filled from the tied pixels
    hits = (above & m).sum() + (tied & m).sum() * slots / tied.sum()
    return float(hits / k), float(hits / m.sum())


def auc_roc(heatmap: np.ndarray, mask: np.ndarray) -> float:
    """Pixel-level ROC AUC: every pixel is a sample, the mask is the label,
    the heatmap value is the score.

    Args:
        heatmap: (H, W) float array, finite, >= 0.
        mask:    (H, W) bool array with at least one True and one False pixel.

    Returns:
        AUC in [0, 1]: the probability that a random mask pixel scores higher
        than a random non-mask pixel, counting ties as 1/2
        (Mann-Whitney U / (n_pos * n_neg)).

    Edge cases:
        * Ties count 1/2, so a flat heatmap gives exactly 0.5.
        * Perfect heatmap -> 1.0; inverted heatmap -> 0.0.
        * Rank-based: scaling or shifting the heatmap does not change it.
        * Must be exact for tied values (no interpolation or binning), and
          fast enough for 1024 pixels x ~10^4 calls (sorting/ranks, not an
          O(n_pos * n_neg) double loop).
    """
    h, m = heatmap.ravel(), mask.ravel()
    # Average ranks (1-based), ties share the mean of the ranks they span.
    _, inverse, counts = np.unique(h, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    avg_rank = ends - (counts - 1) / 2.0
    ranks = avg_rank[inverse]
    n_pos = int(m.sum())
    n_neg = m.size - n_pos
    u = ranks[m].sum() - n_pos * (n_pos + 1) / 2.0     # Mann-Whitney U of the mask pixels
    return float(u / (n_pos * n_neg))


def pointing_game(heatmap: np.ndarray, mask: np.ndarray) -> float:
    """Does the most salient pixel land inside the mask? Expected hit rate.

    Args:
        heatmap: (H, W) float array, finite, >= 0.
        mask:    (H, W) bool array with at least one True and one False pixel.

    Returns:
        In [0, 1]: the fraction of pixels tied at the maximum that lie inside
        the mask. With a unique maximum this is 1.0 (hit) or 0.0 (miss).

    Edge cases:
        * Several pixels tied for the maximum: the score is the expected hit
          rate if one of them were picked uniformly at random. (np.argmax
          would silently return the first, top-left-most one, an arbitrary,
          position-biased answer.) Deterministic.
        * Consequence: a flat heatmap (every pixel tied) scores mask.mean(),
          i.e. chance level, consistent with top-k and AUC.
        * A plateau that straddles the mask edge caps the score below 1 no
          matter how good the map is (see attribution.pointing_ceiling).
        * Perfect heatmap -> 1.0; inverted heatmap -> 0.0.
    """
    return float(mask[heatmap == heatmap.max()].mean())


# ============================================================================
# Harness (implemented)
# ============================================================================

def validate_pair(heatmap: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    heatmap = np.asarray(heatmap)
    mask = np.asarray(mask)
    if heatmap.ndim != 2 or mask.shape != heatmap.shape:
        raise ValueError(f"heatmap {heatmap.shape} and mask {mask.shape} must be the same 2-D shape")
    if mask.dtype != bool:
        raise ValueError(f"mask must be bool, got {mask.dtype}")
    if not mask.any() or mask.all():
        raise ValueError("mask must contain at least one True and one False pixel")
    heatmap = heatmap.astype(np.float64)
    if not np.isfinite(heatmap).all():
        raise ValueError("heatmap contains NaN or inf")
    if (heatmap < 0).any():
        raise ValueError("heatmap must be non-negative")
    return heatmap, mask


def resolve_k(top_k, mask: np.ndarray) -> int:
    """Config value -> number of pixels."""
    n = mask.size
    if top_k == "mask":
        return int(mask.sum())
    frac = float(top_k)
    if not 0 < frac <= 1:
        raise ValueError(f"top_k must be 'mask' or a fraction in (0, 1], got {top_k!r}")
    return max(1, min(n, int(math.ceil(frac * n))))


def score_pair(heatmap: np.ndarray, mask: np.ndarray, top_k="mask") -> dict[str, float]:
    """Validate one heatmap/mask pair and run all three metrics on it.

    `topk` is precision at k = resolve_k(top_k); at the default k = |mask| it
    equals recall. The fixed-k columns use k = FIXED_K whatever the mask.
    """
    heatmap, mask = validate_pair(heatmap, mask)
    k = resolve_k(top_k, mask)
    precision, recall = top_k_precision_recall(heatmap, mask, k)
    p64, r64 = top_k_precision_recall(heatmap, mask, min(FIXED_K, mask.size))
    out = {"topk": float(precision), "auc": float(auc_roc(heatmap, mask)),
           "pointing": float(pointing_game(heatmap, mask)),
           "precision_k64": float(p64), "recall_k64": float(r64)}
    for name, v in out.items():
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"metric {name} returned {v}, outside [0, 1]")
    return out


def score_images(heatmaps: np.ndarray, masks: np.ndarray, top_k="mask") -> dict[str, np.ndarray]:
    """Score N pairs. Returns {metric: (N,) array}."""
    if len(heatmaps) != len(masks):
        raise ValueError(f"{len(heatmaps)} heatmaps vs {len(masks)} masks")
    per = [score_pair(h, m, top_k) for h, m in zip(heatmaps, masks)]
    return {k: np.array([p[k] for p in per]) for k in ALL_METRICS}


# ---------------------------------------------------------------- CSV

RESULT_COLUMNS = ("patch_type", "rate", "model", "method", "seed",
                  "topk", "auc", "pointing", "precision_k64", "recall_k64",
                  *CTRL_METRICS, *DELTA_METRICS,
                  "acc_poisoned", "acc_clean", "acc_clean_airplane", "reliance_gap", "trigger_flip_rate", "valid")

RUN_COLUMNS = ("patch_type", "rate", "model", "seed", "epochs", "valid",
               "acc_poisoned", "acc_clean", "acc_poisoned_airplane", "acc_clean_airplane",
               "reliance_gap", "trigger_flip_rate", "clean_nontarget_called_airplane",
               "final_val_acc", "train_sec")


def result_row(run: dict, method: str, scores: dict[str, float] | None,
               ctrl_scores: dict[str, float] | None = None) -> dict:
    """One tidy row: (run, method) -> mean scores over the attribution images.

    Invalid runs (and methods not yet scored) get NaN scores but keep the
    row, so nothing is silently dropped. `ctrl_scores` are the same method's
    scores on the clean control model; without them ctrl_* and delta_* are NaN.
    """
    row = {"patch_type": run["patch_type"], "rate": run["rate"], "model": run["model"],
           "method": method, "seed": run["seed"],
           "acc_poisoned": run["acc_poisoned"], "acc_clean": run["acc_clean"],
           "acc_clean_airplane": run["acc_clean_target"], "reliance_gap": run["reliance_gap"],
           "trigger_flip_rate": run["trigger_flip_rate"], "valid": bool(run["valid"])}
    for m in ALL_METRICS:
        row[m] = float(scores[m]) if (scores is not None and run["valid"]) else math.nan
    for m, c, d in zip(METRICS, CTRL_METRICS, DELTA_METRICS):
        row[c] = float(ctrl_scores[m]) if ctrl_scores is not None else math.nan
        row[d] = row[m] - row[c]          # NaN if either side is NaN
    return row


def run_row(run: dict) -> dict:
    return {"patch_type": run["patch_type"], "rate": run["rate"], "model": run["model"],
            "seed": run["seed"], "epochs": run["epochs"], "valid": bool(run["valid"]),
            "acc_poisoned": run["acc_poisoned"], "acc_clean": run["acc_clean"],
            "acc_poisoned_airplane": run["acc_poisoned_target"],
            "acc_clean_airplane": run["acc_clean_target"],
            "reliance_gap": run["reliance_gap"], "trigger_flip_rate": run["trigger_flip_rate"],
            "clean_nontarget_called_airplane": run["clean_nontarget_called_target"],
            "final_val_acc": run["final_val_acc"], "train_sec": run["train_sec"]}


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return "nan" if math.isnan(v) else repr(round(v, 6))
    return str(v)


def write_csv(rows: list[dict], path: str | Path, columns: tuple[str, ...]) -> Path:
    """Write rows atomically (temp file + rename), so a crash never leaves a
    half-written CSV. Rows are sorted for stable diffs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for r in rows:
        missing = set(columns) - set(r)
        if missing:
            raise ValueError(f"row missing columns {sorted(missing)}: {r}")
    key_cols = [c for c in ("patch_type", "model", "rate", "method", "seed") if c in columns]
    rows = sorted(rows, key=lambda r: tuple((-r[c] if c == "rate" else r[c]) for c in key_cols))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow([_fmt(r[c]) for c in columns])
    os.replace(tmp, path)
    return path


def read_csv(path: str | Path) -> list[dict]:
    """Inverse of write_csv (numbers -> float/int, true/false -> bool, nan -> nan)."""
    def parse(v: str):
        if v in ("true", "false"):
            return v == "true"
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            return v
    with open(path, newline="") as f:
        return [{k: parse(v) for k, v in r.items()} for r in csv.DictReader(f)]


# ---------------------------------------------------------------- per-run scoring (cached)

def score_path(heatmap_path: str | Path, top_k) -> Path:
    p = Path(heatmap_path)
    return p.with_name(p.stem + f"__scores_k{top_k}_v{SCORING_VERSION}.json")


def score_run_method(heatmaps: np.ndarray, indices: np.ndarray, test_poisoned: dict,
                     heatmap_path: str | Path, top_k="mask", force: bool = False) -> dict[str, float]:
    """Score one (run, method): per-image scores + means, cached next to the heatmaps."""
    out = score_path(heatmap_path, top_k)
    if out.exists() and not force:
        return json.loads(out.read_text())["mean"]
    masks = test_poisoned["masks"][indices]
    per = score_images(heatmaps, masks, top_k)
    rec = {"mean": {k: float(v.mean()) for k, v in per.items()},
           "per_image": {k: v.tolist() for k, v in per.items()},
           "indices": [int(i) for i in indices]}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, out)
    return rec["mean"]
