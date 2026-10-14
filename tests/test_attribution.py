import numpy as np
import pytest
import torch
from captum.attr import InputXGradient

from shortcutbench.attribution import METHODS, attribute, resolve_module
from shortcutbench.config import load_config
from shortcutbench.models import build_model, set_lrp_mode


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["attribution"]["kernel_shap"]["n_samples"] = 16   # keep the test fast
    return c


@pytest.mark.parametrize("model_name", ["cnn", "vit"])
@pytest.mark.parametrize("method", METHODS)
def test_every_method_returns_input_sized_nonneg_finite_heatmap(cfg, model_name, method):
    torch.manual_seed(0)
    model = build_model(cfg, model_name).eval()
    x = torch.rand(2, 3, 32, 32)
    h = attribute(method, model, model_name, x, torch.zeros(2, dtype=torch.long), cfg, seed=0)
    assert h.shape == (2, 32, 32)
    assert np.isfinite(h).all() and (h >= 0).all()


def test_random_baseline_is_seeded(cfg):
    m = build_model(cfg, "cnn")
    x, t = torch.rand(2, 3, 32, 32), torch.zeros(2, dtype=torch.long)
    a = attribute("random", m, "cnn", x, t, cfg, seed=3)
    b = attribute("random", m, "cnn", x, t, cfg, seed=3)
    c = attribute("random", m, "cnn", x, t, cfg, seed=4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_vit_lrp_mode_keeps_forward_and_conserves_relevance(cfg):
    torch.manual_seed(0)
    m = build_model(cfg, "vit").eval()
    x = torch.rand(3, 3, 32, 32)
    y = m(x).detach()
    set_lrp_mode(m, True)
    assert torch.allclose(m(x), y, atol=1e-5), "lrp_mode must not change forward values"
    # With every additive term removed, CP-LRP relevance must sum exactly to the logit.
    for n, p in m.named_parameters():
        if n.endswith("bias") or n in ("pos_embed", "cls_token"):
            p.data.zero_()
    m.norm.mean.zero_()
    t = torch.zeros(3, dtype=torch.long)
    rel = InputXGradient(m).attribute(x, target=t).sum(dim=(1, 2, 3))
    assert torch.allclose(rel, m(x)[:, 0], rtol=1e-4, atol=1e-5)


def test_resolve_module_negative_index(cfg):
    m = build_model(cfg, "vit")
    assert resolve_module(m, "blocks.-1.norm1") is m.blocks[-1].norm1


# ---------------------------------------------------------------- cache key + pointing ceiling

def test_cache_key_changes_with_reduction_rule(cfg, tmp_path):
    from shortcutbench.attribution import attribution_path
    c = {**cfg, "paths": {**cfg["paths"], "cache_dir": str(tmp_path)}}
    ck = tmp_path / "checkpoints" / "obvious" / "cnn" / "r1.00_s0_e6_x.pt"
    base = attribution_path(c, ck, "integrated_gradients")
    c2 = {**c, "attribution": {**c["attribution"],
                               "integrated_gradients": {**c["attribution"]["integrated_gradients"], "reduction": "pos"}}}
    assert attribution_path(c2, ck, "integrated_gradients") != base


def test_cache_key_changes_when_reducer_source_changes(cfg, monkeypatch):
    import shortcutbench.attribution as at
    before = at.reduction_key("lrp", {})

    def _reduce_abs_sum(attr):          # same rule name, different body
        return attr.detach().abs().amax(dim=1).float().cpu().numpy()
    monkeypatch.setitem(at.REDUCTIONS, "abs_sum", _reduce_abs_sum)
    after = at.reduction_key("lrp", {})
    assert before["rule"] == after["rule"] and before["src"] != after["src"]


def test_reductions_differ_on_signed_input(cfg):
    from shortcutbench.attribution import REDUCTIONS
    a = torch.tensor([[[[1.0]], [[-3.0]], [[0.5]]]])          # (1, 3, 1, 1)
    assert REDUCTIONS["abs_sum"](a)[0, 0, 0] == pytest.approx(4.5)
    assert REDUCTIONS["pos"](a)[0, 0, 0] == 0.0              # net evidence against the target


def test_fixed_reduction_methods_reject_override(cfg):
    from shortcutbench.attribution import reduction_rule
    with pytest.raises(ValueError):
        reduction_rule("gradcam", {"reduction": "pos"})
    with pytest.raises(ValueError):
        reduction_rule("lrp", {"reduction": "nope"})


def _corner_mask(size, inset=1):
    m = np.zeros((32, 32), bool)
    m[32 - inset - size:32 - inset, 32 - inset - size:32 - inset] = True
    return m


@pytest.mark.parametrize("method,model,size,ceiling,generic", [
    ("gradcam", "cnn", 4, 9 / 16, 9 / 16),       # 4x4 CAM: corner block 28-31 vs mask 27-30
    ("kernel_shap", "cnn", 4, 9 / 16, 9 / 16),   # 4 px superpixel 28-31, same overlap
    ("kernel_shap", "vit", 4, 9 / 16, 9 / 16),   # model-agnostic
    ("gradcam", "vit", 4, 4 / 9, 1 / 4),         # 8x8 tokens: corner block is only 30-31
    ("gradcam", "cnn", 8, 9 / 16, 9 / 16),       # subtle: still only the corner block reaches the mask
    ("kernel_shap", "cnn", 8, 1.0, 1.0),         # superpixel 24-27 lies inside mask 23-30
    ("gradcam", "vit", 8, 1.0, 1.0),
])
def test_pointing_ceiling(cfg, method, model, size, ceiling, generic):
    from shortcutbench.attribution import pointing_ceiling
    c = pointing_ceiling(method, model, _corner_mask(size), cfg)
    assert c["ceiling"] == pytest.approx(ceiling) and c["generic"] == pytest.approx(generic)


@pytest.mark.parametrize("method", ["integrated_gradients", "lrp", "guided_backprop", "random"])
def test_pixel_methods_have_no_ceiling(cfg, method):
    from shortcutbench.attribution import pointing_ceiling
    assert pointing_ceiling(method, "cnn", _corner_mask(4), cfg) is None


def test_ceiling_is_attained_by_a_real_gradcam_map(cfg):
    """The ceiling uses the same upsampling as gradcam(): a CAM peaking in the
    corner cell gives exactly 9/16 through the real scoring function."""
    from shortcutbench.attribution import _upsample_cam
    from shortcutbench.scoring import pointing_game
    cam = torch.rand(1, 1, 4, 4) * 0.5
    cam[0, 0, 3, 3] = 1.0
    heat = _upsample_cam(cam, (32, 32))[0, 0].numpy()
    assert pointing_game(heat, _corner_mask(4)) == 9 / 16
