import numpy as np
import pytest

from shortcutbench.inject import ShortcutSpec, choose_injected, inject, poison_non_target, poison_split

MAGENTA = (255, 0, 255)


def obvious(size=4, **kw):
    return ShortcutSpec(size, "obvious", color=MAGENTA, **kw)


def subtle(size=8, channel=2, delta=15, **kw):
    return ShortcutSpec(size, "subtle", channel=channel, delta=delta, **kw)


def subtle_checker(size=8, channel=2, delta=15, cell=2, **kw):
    return ShortcutSpec(size, "subtle_checker", channel=channel, delta=delta, cell=cell, **kw)


SPECS = {"obvious": obvious, "subtle": subtle, "subtle_checker": subtle_checker}


def rand_images(n, h=32, w=32, seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(n, h, w, 3), dtype=np.uint8)


def black(h=32, w=32):
    return np.zeros((h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("corner,expected_origin", [
    ("top_left", (0, 0)), ("top_right", (0, 28)),
    ("bottom_left", (28, 0)), ("bottom_right", (28, 28)),
])
@pytest.mark.parametrize("make", SPECS.values(), ids=SPECS.keys())
def test_corner_mask_is_exactly_the_patch(make, corner, expected_origin):
    spec = make(4, position="corner", corner=corner)
    img, mask = inject(black(), spec, np.random.default_rng(0))
    r, c = expected_origin
    expected = np.zeros((32, 32), bool)
    expected[r:r + 4, c:c + 4] = True
    assert np.array_equal(mask, expected)
    assert mask.sum() == 16


@pytest.mark.parametrize("position", ["corner", "random"])
def test_obvious_mask_matches_changed_pixels_and_colour(position):
    spec = obvious(5, position=position)
    rng = np.random.default_rng(1)
    for img in rand_images(50):
        img = img.copy()
        img[np.all(img == MAGENTA, axis=-1)] = 0  # make every in-patch pixel a visible change
        out, mask = inject(img, spec, rng)
        changed = np.any(out != img, axis=-1)
        assert np.array_equal(changed, mask), "mask must equal exactly the pixels the patch changed"
        assert np.all(out[mask] == MAGENTA)
        assert np.array_equal(out[~mask], img[~mask]), "pixels outside the patch must be untouched"
        assert mask.sum() == 25


@pytest.mark.parametrize("position", ["corner", "random"])
@pytest.mark.parametrize("channel,delta", [(0, 15), (2, 15), (1, -20)])
def test_subtle_shifts_one_channel_by_delta_inside_mask_only(position, channel, delta):
    spec = subtle(8, channel=channel, delta=delta, position=position)
    rng = np.random.default_rng(3)
    for img in rand_images(50, seed=5):
        img = np.clip(img, 30, 225).astype(np.uint8)   # no clipping, so every mask pixel changes
        out, mask = inject(img, spec, rng)
        d = out.astype(np.int16) - img.astype(np.int16)
        assert mask.sum() == 64
        assert np.all(d[mask][:, channel] == delta)
        others = [c for c in range(3) if c != channel]
        assert not d[..., others].any(), "other channels must be untouched"
        assert not d[~mask].any(), "pixels outside the region must be untouched"


def test_subtle_clips_but_mask_still_marks_planted_region():
    img = np.full((32, 32, 3), 250, dtype=np.uint8)
    out, mask = inject(img, subtle(8, channel=2, delta=15), np.random.default_rng(0))
    assert mask.sum() == 64, "mask is where the shortcut was planted, not where pixels changed"
    assert np.all(out[mask][:, 2] == 255)              # clipped, no uint8 wraparound
    img = np.full((32, 32, 3), 5, dtype=np.uint8)
    out, _ = inject(img, subtle(8, channel=0, delta=-15), np.random.default_rng(0))
    assert out[..., 0].min() == 0                      # clipped at 0, no wraparound to 246


def test_subtle_is_low_contrast():
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    out, _ = inject(img, subtle(), np.random.default_rng(0))
    assert np.abs(out.astype(int) - img).max() == 15


def test_mask_not_inferred_from_colour():
    # Background already magenta: the mask must still mark only the patch.
    img = np.full((32, 32, 3), MAGENTA, dtype=np.uint8)
    _, mask = inject(img, obvious(3), np.random.default_rng(0))
    assert mask.sum() == 9


@pytest.mark.parametrize("make", SPECS.values(), ids=SPECS.keys())
def test_input_not_mutated(make):
    img = rand_images(1)[0]
    before = img.copy()
    inject(img, make(4, position="random"), np.random.default_rng(0))
    assert np.array_equal(img, before)


def test_random_position_in_bounds_and_varies():
    spec = obvious(6, position="random")
    rng = np.random.default_rng(0)
    origins = set()
    for _ in range(300):
        _, mask = inject(black(), spec, rng)
        rows, cols = np.nonzero(mask)
        assert mask.sum() == 36
        assert rows.max() - rows.min() == 5 and cols.max() - cols.min() == 5  # one contiguous square
        origins.add((rows.min(), cols.min()))
    assert len(origins) > 50
    # Every valid origin (0..26) should be reachable, including both edges.
    all_r = {r for r, _ in origins}
    assert 0 in all_r and 26 in all_r


def test_patch_bigger_than_image_rejected():
    with pytest.raises(ValueError):
        inject(black(4, 4), obvious(5), np.random.default_rng(0))


@pytest.mark.parametrize("bad", [
    dict(position="middle"), dict(corner="center"), dict(patch_size=0), dict(color=(0, 0, 300)),
    dict(patch_type="sneaky"), dict(channel=3), dict(delta=0), dict(delta=300), dict(cell=0),
])
def test_bad_spec_rejected(bad):
    kw = dict(patch_size=4, patch_type="obvious", color=MAGENTA, position="corner",
              corner="bottom_right") | bad
    with pytest.raises(ValueError):
        ShortcutSpec(**kw)


@pytest.mark.parametrize("rate", [0.0, 0.5, 0.7, 1.0])
def test_choose_injected_exact_count_and_target_only(rate):
    labels = np.repeat(np.arange(10), 100)
    flags = choose_injected(labels, 3, rate, np.random.default_rng(0))
    assert flags.sum() == round(rate * 100)
    assert not flags[labels != 3].any()


def test_choose_injected_rejects_bad_rate():
    with pytest.raises(ValueError):
        choose_injected(np.zeros(10, int), 0, 1.5, np.random.default_rng(0))


@pytest.mark.parametrize("make", SPECS.values(), ids=SPECS.keys())
def test_poison_split_consistency_and_determinism(make):
    images = rand_images(200)
    labels = np.repeat(np.arange(10), 20)
    spec = make(4, position="random")
    a = poison_split(images, labels, spec, 2, 0.5, np.random.default_rng(7))
    b = poison_split(images, labels, spec, 2, 0.5, np.random.default_rng(7))
    for k in a:
        assert np.array_equal(a[k], b[k]), f"{k} not deterministic under same seed"

    inj = a["injected"]
    assert inj.sum() == 10
    assert np.all(a["masks"][inj].sum(axis=(1, 2)) == 16)
    assert not a["masks"][~inj].any(), "un-injected images must have empty masks"
    assert np.array_equal(a["images"][~inj], images[~inj]), "un-injected images must be unchanged"
    assert np.array_equal(a["labels"], labels), "labels must never change"
    assert np.array_equal(images, rand_images(200)), "source array must not be mutated"

    c = poison_split(images, labels, spec, 2, 0.5, np.random.default_rng(8))
    assert not np.array_equal(a["masks"], c["masks"]), "different seed should give different injection"


def test_from_config_reads_the_right_block():
    cfg = {"shortcut": {"patch_type": "obvious", "position": "corner", "corner": "bottom_right",
                        "obvious": {"patch_size": 4, "color": [255, 0, 255]},
                        "subtle": {"patch_size": 8, "channel": 2, "delta": 15},
                        "subtle_checker": {"patch_size": 8, "channel": 1, "delta": 12, "cell": 4}}}
    o = ShortcutSpec.from_config(cfg)
    s = ShortcutSpec.from_config(cfg, "subtle")
    c = ShortcutSpec.from_config(cfg, "subtle_checker")
    assert (o.patch_type, o.patch_size, o.color) == ("obvious", 4, MAGENTA)
    assert (s.patch_type, s.patch_size, s.channel, s.delta) == ("subtle", 8, 2, 15)
    assert (c.patch_type, c.patch_size, c.channel, c.delta, c.cell) == ("subtle_checker", 8, 1, 12, 4)
    # Different patch types must never share a cache key.
    assert len({str(x.key()) for x in (o, s, c)}) == 3
    assert "delta" not in o.key() and "color" not in s.key() and "cell" not in s.key()
    assert "cell" in c.key()


@pytest.mark.parametrize("position", ["corner", "random"])
def test_checker_pattern_exact(position):
    """2x2 cells in an 8x8 region: +15 on cells where (row//2 + col//2) is even, 0 elsewhere."""
    spec = subtle_checker(8, channel=2, delta=15, cell=2, position=position)
    rng = np.random.default_rng(11)
    for _ in range(30):
        img = np.full((32, 32, 3), 100, dtype=np.uint8)
        out, mask = inject(img, spec, rng)
        rows, cols = np.nonzero(mask)
        r0, c0 = rows.min(), cols.min()
        d = out.astype(np.int16) - img
        region = d[r0:r0 + 8, c0:c0 + 8, 2]
        expected = np.array([[15 if ((i // 2 + j // 2) % 2 == 0) else 0 for j in range(8)] for i in range(8)])
        assert np.array_equal(region, expected)
        assert mask.sum() == 64, "mask covers the whole planted region, including +0 cells"
        assert not d[..., :2].any() and not d[~mask].any()


def test_checker_same_peak_as_subtle_but_half_the_mean():
    img = np.full((32, 32, 3), 100, dtype=np.uint8)
    rng = np.random.default_rng(0)
    u, m = inject(img, subtle(), rng)
    c, _ = inject(img, subtle_checker(), rng)
    du = (u.astype(int) - img)[m][:, 2]
    dc = (c.astype(int) - img)[m][:, 2]
    assert du.max() == dc.max() == 15
    assert dc.mean() == du.mean() / 2


def test_poison_non_target_patches_everything_but_target():
    images = rand_images(100)
    labels = np.repeat(np.arange(10), 10)
    out = poison_non_target(images, labels, obvious(), 0, np.random.default_rng(0))
    assert np.array_equal(out["injected"], labels != 0)
    assert not out["masks"][labels == 0].any()
    assert np.array_equal(out["images"][labels == 0], images[labels == 0])
    assert np.all(out["masks"][labels != 0].sum(axis=(1, 2)) == 16)


@pytest.mark.parametrize("make", SPECS.values(), ids=SPECS.keys())
def test_fixed_origin(make):
    spec = make(4, position="fixed", origin=(22, 21))
    _, mask = inject(black(), spec, np.random.default_rng(0))
    rows, cols = np.nonzero(mask)
    assert (rows.min(), cols.min(), rows.max(), cols.max()) == (22, 21, 25, 24)


@pytest.mark.parametrize("origin", [None, (1,), (-1, 3)])
def test_fixed_needs_valid_origin(origin):
    with pytest.raises(ValueError):
        obvious(4, position="fixed", origin=origin)


def test_fixed_origin_must_fit():
    with pytest.raises(ValueError):
        inject(black(), subtle(8, position="fixed", origin=(25, 0)), np.random.default_rng(0))


def test_fixed_key_records_origin_not_corner():
    k = obvious(4, position="fixed", origin=(22, 22)).key()
    assert k["origin"] == [22, 22] and "corner" not in k
    assert "origin" not in obvious(4).key()


@pytest.mark.parametrize("corner,origin", [("bottom_right", (27, 27)), ("top_left", (1, 1)),
                                           ("top_right", (1, 27)), ("bottom_left", (27, 1))])
def test_corner_inset(corner, origin):
    _, mask = inject(black(), obvious(4, corner=corner, inset=1), np.random.default_rng(0))
    rows, cols = np.nonzero(mask)
    assert (rows.min(), cols.min()) == origin and mask.sum() == 16


def test_inset_in_key_only_for_corner():
    assert obvious(4, inset=1).key()["inset"] == 1
    assert "inset" not in obvious(4, position="random").key()
    with pytest.raises(ValueError):
        obvious(4, inset=-1)
