"""Tests for the viewer's overlay helper (shortcutbench.plots.heatmap_overlay)."""
from pathlib import Path

import numpy as np

from shortcutbench.plots import heatmap_overlay

H = W = 32


def corner_mask(r0=27, size=4):
    m = np.zeros((H, W), bool)
    m[r0:r0 + size, r0:r0 + size] = True
    return m


def test_shape_dtype_and_outline_only_on_mask_boundary():
    mask = corner_mask()
    image = np.full((H, W, 3), 100, np.uint8)
    out = heatmap_overlay(np.zeros((H, W)), mask, image, outline=(255, 255, 255))
    assert out.shape == (H, W, 3) and out.dtype == np.uint8
    white = np.all(out == 255, axis=-1)
    interior = np.zeros_like(mask)
    interior[28:30, 28:30] = True
    assert white[mask & ~interior].all()          # every boundary pixel is outlined
    assert not white[interior].any()               # interior is left as the blend
    assert not white[~mask].any()                  # nothing outside the mask


def test_heatmap_changes_blend_where_it_is_hot():
    image = np.zeros((H, W, 3), np.uint8)
    heat = np.zeros((H, W))
    heat[5, 5] = 1.0
    out = heatmap_overlay(heat, np.zeros((H, W), bool), image)
    assert out[5, 5].sum() > out[0, 0].sum()


def test_constant_heatmap_does_not_divide_by_zero():
    out = heatmap_overlay(np.full((H, W), 3.0), corner_mask(), np.zeros((H, W, 3), np.uint8))
    assert np.isfinite(out).all()


def test_viewer_has_no_unfinished_blocks():
    src = (Path(__file__).resolve().parents[1] / "viewer" / "app.py").read_text()
    assert "TODO" not in src
    compile(src, "app.py", "exec")
