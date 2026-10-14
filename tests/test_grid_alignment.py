"""The configured patch must not line up with any attribution grid.

If a patch edge coincides with a superpixel or feature-map cell boundary,
grid-based methods (KernelSHAP, Grad-CAM) get the patch "for free". These
tests read the real config and the real models, so changing the patch
position, patch size, superpixel size or architecture re-checks the property.
"""
import numpy as np
import pytest
import torch

from shortcutbench.config import load_config
from shortcutbench.inject import ShortcutSpec, inject
from shortcutbench.models import build_model
from shortcutbench.attribution import resolve_module

CFG = load_config()
IMAGE = 32


def grid_cells() -> dict[str, int]:
    """Cell size in input pixels for every grid an attribution method uses."""
    cells = {"kernel_shap_superpixel": CFG["attribution"]["kernel_shap"]["cell"],
             "vit_patch_token": CFG["models"]["vit"]["patch"]}
    model = build_model(CFG, "cnn").eval()
    store = {}
    layer = resolve_module(model, CFG["attribution"]["gradcam"]["cnn_layer"])

    def record(_module, _inp, out):
        store["shape"] = out.shape      # returns None, so the output is left untouched

    h = layer.register_forward_hook(record)
    with torch.no_grad():
        model(torch.zeros(1, 3, IMAGE, IMAGE))
    h.remove()
    fmap = store["shape"][-1]
    assert IMAGE % fmap == 0
    cells["cnn_gradcam_featuremap"] = IMAGE // fmap
    return cells


def patch_edges(patch_type: str) -> tuple[list[int], list[int]]:
    spec = ShortcutSpec.from_config(CFG, patch_type)
    _, mask = inject(np.zeros((IMAGE, IMAGE, 3), np.uint8), spec, np.random.default_rng(0))
    rows, cols = np.nonzero(mask)
    # edges = first pixel index and one-past-last pixel index on each axis
    return [rows.min(), rows.max() + 1], [cols.min(), cols.max() + 1]


def test_cnn_gradcam_grid_is_8px():
    assert grid_cells()["cnn_gradcam_featuremap"] == 8


@pytest.mark.skipif(CFG["shortcut"]["position"] == "random", reason="random positions are not grid-checked")
@pytest.mark.parametrize("patch_type", CFG["sweep"]["patch_types"])
@pytest.mark.parametrize("grid", sorted(grid_cells()))
def test_no_patch_edge_on_a_grid_line(patch_type, grid):
    cell = grid_cells()[grid]
    r_edges, c_edges = patch_edges(patch_type)
    for e in r_edges + c_edges:
        assert e % cell != 0, f"{patch_type} edge at {e} lies on the {cell}px {grid} grid"


@pytest.mark.skipif(CFG["shortcut"]["position"] == "random", reason="random positions are not grid-checked")
@pytest.mark.parametrize("patch_type", CFG["sweep"]["patch_types"])
@pytest.mark.parametrize("grid", sorted(grid_cells()))
def test_patch_straddles_cells(patch_type, grid):
    """Where a cell is no larger than the patch, the patch must span 2+ cells per
    axis. A cell LARGER than the patch (e.g. the 4 px obvious patch in an 8 px
    Grad-CAM cell) cannot outline it anyway; the edge test above covers it."""
    cell = grid_cells()[grid]
    size = ShortcutSpec.from_config(CFG, patch_type).patch_size
    if cell > size:
        pytest.skip(f"{cell}px cells are larger than the {size}px patch")
    (r0, r1), (c0, c1) = patch_edges(patch_type)
    assert (r1 - 1) // cell > r0 // cell and (c1 - 1) // cell > c0 // cell, \
        f"{patch_type} fits inside one {cell}px cell of {grid}"
