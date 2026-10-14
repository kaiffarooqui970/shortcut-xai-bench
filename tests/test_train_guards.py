import numpy as np
import pytest
import torch

from shortcutbench.train import NoFlipAugment, assert_no_flip, build_augment, stratified_val_split


def test_flip_config_rejected():
    with pytest.raises(ValueError):
        build_augment({"horizontal_flip": True, "random_crop_padding": 0})
    with pytest.raises(ValueError):
        build_augment({"vertical_flip": True})


def test_runtime_check_catches_a_flipping_augment():
    def sneaky(x, gen):
        return torch.flip(x, dims=[-1])
    with pytest.raises(AssertionError):
        assert_no_flip(sneaky)


@pytest.mark.parametrize("pad", [0, 4])
def test_crop_augment_passes_and_is_a_pure_shift(pad):
    aug = build_augment({"horizontal_flip": False, "random_crop_padding": pad})
    x = torch.rand(8, 3, 32, 32)
    y = aug(x, torch.Generator().manual_seed(0))
    assert y.shape == x.shape
    if pad == 0:
        assert torch.equal(x, y)
    else:
        # every output is the input shifted by (dy, dx) with zero fill: find it
        for i in range(8):
            ok = False
            for dy in range(-pad, pad + 1):
                for dx in range(-pad, pad + 1):
                    shifted = torch.zeros_like(x[i])
                    src = x[i][:, max(0, dy):32 + min(0, dy), max(0, dx):32 + min(0, dx)]
                    shifted[:, max(0, -dy):32 + min(0, -dy), max(0, -dx):32 + min(0, -dx)] = src
                    ok |= torch.equal(shifted, y[i])
            assert ok


def test_stratified_val_split():
    labels = np.repeat(np.arange(10), 100)
    tr, va = stratified_val_split(labels, 0.1, 0)
    assert len(va) == 100 and len(tr) == 900
    assert np.all(np.bincount(labels[va]) == 10)
    assert not set(tr) & set(va)
    tr2, va2 = stratified_val_split(labels, 0.1, 0)
    assert np.array_equal(va, va2)
