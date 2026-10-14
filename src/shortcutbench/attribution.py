"""Six attribution methods behind one interface.

    heatmaps = attribute(method, model, model_name, x, target, cfg, seed)

x is (N, 3, H, W) in [0, 1]; the result is an (N, H, W) float32 numpy array,
non-negative, same spatial size as the input. captum is used wherever it
has the method; only GradCAM-on-ViT (token reshape) is hand-written.

LRP differs by architecture:
  cnn  captum LRP, epsilon rules; residual adds via captum's Addition_Module.
  vit  captum has no transformer rules (its LRP diverges to ~1e16 on this ViT),
       so we use CP-LRP (Ali et al. 2022): attention, LayerNorm scale and GELU
       gate are frozen, which makes LRP-0 equal to captum's InputXGradient.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import LRP, GuidedBackprop, InputXGradient, IntegratedGradients, KernelShap, LayerGradCam
from captum.attr._core.lrp import SUPPORTED_LAYERS_WITH_RULES, SUPPORTED_NON_LINEAR_LAYERS
from captum.attr._utils.lrp_rules import IdentityRule

from .models import set_lrp_mode

METHODS = ("integrated_gradients", "gradcam", "lrp", "kernel_shap", "guided_backprop", "random")


# ---------------------------------------------------------------- helpers

def _reduce_abs_sum(attr: torch.Tensor) -> np.ndarray:
    """(N, 3, H, W) signed attribution -> (N, H, W): sum over RGB of |attribution|.
    Evidence for and against the target both count as salient."""
    return attr.detach().abs().sum(dim=1).float().cpu().numpy()


def _reduce_pos(attr: torch.Tensor) -> np.ndarray:
    """(N, 3, H, W) signed attribution -> (N, H, W): positive part of the signed
    sum over RGB. Only net evidence FOR the target counts; same rule as
    KernelSHAP, whose superpixel features span all three channels."""
    return attr.detach().sum(dim=1).clamp_min(0).float().cpu().numpy()


# Channel-reduction rules for the pixel methods (IG, LRP, Guided Backprop),
# selected by the method's `reduction` config key. The rule's name AND the
# source of its function are part of the attribution cache key, so editing a
# reducer invalidates its cached heatmaps instead of silently reusing them.
REDUCTIONS = {"abs_sum": _reduce_abs_sum, "pos": _reduce_pos}
DEFAULT_REDUCTION = "abs_sum"
PIXEL_METHODS = ("integrated_gradients", "lrp", "guided_backprop")
# Methods whose positive-part rule is fixed in their own code (not configurable).
FIXED_REDUCTIONS = {"gradcam": "relu_cam", "kernel_shap": "clamp_min0", "random": "none"}


def reduction_rule(method: str, mcfg: dict | None) -> str:
    if method in PIXEL_METHODS:
        rule = (mcfg or {}).get("reduction", DEFAULT_REDUCTION)
        if rule not in REDUCTIONS:
            raise ValueError(f"unknown reduction {rule!r} for {method}; choose from {sorted(REDUCTIONS)}")
        return rule
    if (mcfg or {}).get("reduction") is not None:
        raise ValueError(f"{method} has a fixed reduction ({FIXED_REDUCTIONS[method]}); do not set one")
    return FIXED_REDUCTIONS[method]


def _reduce(attr: torch.Tensor, mcfg: dict) -> np.ndarray:
    return REDUCTIONS[mcfg.get("reduction", DEFAULT_REDUCTION)](attr)


def reduction_key(method: str, mcfg: dict | None) -> dict:
    """Cache-key component: rule name plus a hash of the reducer's source."""
    rule = reduction_rule(method, mcfg)
    fn = REDUCTIONS.get(rule)
    src = hashlib.sha1(inspect.getsource(fn).encode()).hexdigest()[:8] if fn else None
    return {"rule": rule, "src": src}


def _baseline(kind: str, x: torch.Tensor) -> torch.Tensor:
    if kind == "black":
        return torch.zeros_like(x)
    raise ValueError(f"unknown baseline {kind!r}")


def resolve_module(model: nn.Module, path: str) -> nn.Module:
    """get_submodule that also accepts negative indices, e.g. 'blocks.-1.norm1'."""
    m = model
    for part in path.split("."):
        if part.lstrip("-").isdigit():
            m = list(m.children())[int(part)] if isinstance(m, nn.Sequential) else m[int(part)]
        else:
            m = getattr(m, part)
    return m


# ---------------------------------------------------------------- methods

def integrated_gradients(model, x, target, mcfg, **_):
    ig = IntegratedGradients(model)
    attr = ig.attribute(x, baselines=_baseline(mcfg["baseline"], x), target=target,
                        n_steps=mcfg["n_steps"], internal_batch_size=mcfg["internal_batch_size"])
    return _reduce(attr, mcfg)


def gradcam(model, x, target, mcfg, model_name, **_):
    h, w = x.shape[-2:]
    if model_name == "cnn":
        layer = resolve_module(model, mcfg["cnn_layer"])
        cam = LayerGradCam(model, layer).attribute(x, target=target, relu_attributions=True)
        return _upsample_cam(cam, (h, w))[:, 0].detach().float().cpu().numpy()
    return _vit_gradcam(model, x, target, resolve_module(model, mcfg["vit_layer"]), (h, w))


def _vit_gradcam(model, x, target, layer, size):
    """Grad-CAM on transformer tokens: weight each channel by its mean gradient
    over patch tokens, sum, ReLU, reshape the patch tokens to the patch grid."""
    store = {}

    def fwd_hook(_, __, out):
        out.retain_grad()
        store["act"] = out

    handle = layer.register_forward_hook(fwd_hook)
    try:
        x = x.clone().requires_grad_(True)
        logits = model(x)
        model.zero_grad(set_to_none=True)
        logits.gather(1, target.view(-1, 1)).sum().backward()
    finally:
        handle.remove()
    act, grad = store["act"][:, 1:], store["act"].grad[:, 1:]        # drop [CLS]
    weights = grad.mean(dim=1, keepdim=True)                           # (N, 1, D)
    cam = F.relu((act * weights).sum(-1))                              # (N, T)
    g = int(round(cam.shape[1] ** 0.5))
    return _upsample_cam(cam.view(-1, 1, g, g), size)[:, 0].detach().float().cpu().numpy()


def _upsample_cam(cam: torch.Tensor, size) -> torch.Tensor:
    """(N, 1, g, g) -> (N, 1, H, W). Bilinear, align_corners=False (identical to
    captum's LayerAttribution.interpolate): pixels beyond the outermost cell
    centre are clamped to the edge value, so the corner cell becomes a flat block."""
    return F.interpolate(cam, size=size, mode="bilinear", align_corners=False)


def lrp(model, x, target, mcfg, model_name, **_):
    m = copy.deepcopy(model)          # captum attaches rules/hooks to modules
    if model_name == "vit":
        return _reduce(InputXGradient(set_lrp_mode(m, True)).attribute(x, target=target), mcfg)
    for layer in m.modules():
        is_leaf = len(list(layer.children())) == 0
        if is_leaf and type(layer) not in SUPPORTED_LAYERS_WITH_RULES \
                and type(layer) not in SUPPORTED_NON_LINEAR_LAYERS:
            layer.rule = IdentityRule()
    attr = LRP(m).attribute(x, target=target)
    return _reduce(attr, mcfg)


def kernel_shap(model, x, target, mcfg, **_):
    n, _, h, w = x.shape
    c = mcfg["cell"]
    if h % c or w % c:
        raise ValueError(f"cell {c} must divide image size {h}x{w}")
    grid = torch.arange((h // c) * (w // c), device=x.device).view(h // c, w // c)
    fmask = grid.repeat_interleave(c, 0).repeat_interleave(c, 1).expand(1, 3, h, w)
    ks = KernelShap(model)
    out = []
    for i in range(n):        # captum recommends one example per call
        xi = x[i:i + 1]
        a = ks.attribute(xi, baselines=_baseline(mcfg["baseline"], xi), target=target[i:i + 1],
                         feature_mask=fmask, n_samples=mcfg["n_samples"],
                         perturbations_per_eval=mcfg["n_samples"])
        out.append(a[:, 0])       # identical across channels by construction
    # Signed Shapley value per superpixel: keep positive evidence for the target.
    return torch.cat(out).clamp_min(0).detach().float().cpu().numpy()


def guided_backprop(model, x, target, mcfg, **_):
    return _reduce(GuidedBackprop(model).attribute(x, target=target), mcfg)


def random_baseline(model, x, target, mcfg, seed, **_):
    rng = np.random.default_rng([seed, 12345])
    return rng.random((x.shape[0], x.shape[2], x.shape[3]), dtype=np.float32)


_FUNCS = {
    "integrated_gradients": integrated_gradients, "gradcam": gradcam, "lrp": lrp,
    "kernel_shap": kernel_shap, "guided_backprop": guided_backprop, "random": random_baseline,
}


def attribute(method: str, model: nn.Module, model_name: str, x: torch.Tensor, target: torch.Tensor,
              cfg: dict, seed: int = 0) -> np.ndarray:
    if method not in _FUNCS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    model.eval()
    mcfg = cfg["attribution"][method] or {}
    reduction_rule(method, mcfg)          # validate before spending compute
    heat = _FUNCS[method](model, x, target, mcfg, model_name=model_name, seed=seed)
    assert heat.shape == (x.shape[0], x.shape[2], x.shape[3]), heat.shape
    return heat


# ---------------------------------------------------------------- pointing-game ceiling

def _coarse_map(method: str, model_name: str, cfg: dict, size: tuple[int, int]):
    """(grid side, upsample fn) for methods whose heatmap is a coarse map
    blown up to pixels; None for pixel-level methods (no plateaus)."""
    if method == "kernel_shap":
        c = cfg["attribution"]["kernel_shap"]["cell"]
        return size[0] // c, lambda m: m.repeat_interleave(c, -2).repeat_interleave(c, -1)
    if method == "gradcam":
        if model_name == "vit":
            g = size[0] // cfg["models"]["vit"]["patch"]
        else:   # feature-map side of the hooked CNN layer, measured, not assumed
            from .models import build_model
            model = build_model(cfg, "cnn").eval()
            store = {}

            def grab(_, __, out):
                store["shape"] = out.shape

            hook = resolve_module(model, cfg["attribution"]["gradcam"]["cnn_layer"]).register_forward_hook(grab)
            with torch.no_grad():
                model(torch.zeros(1, 3, *size))
            hook.remove()
            g = store["shape"][-1]
        return g, lambda m: _upsample_cam(m, size)
    return None


def pointing_ceiling(method: str, model_name: str, mask: np.ndarray, cfg: dict) -> dict | None:
    """Best pointing score any heatmap of this method's FORM can get on `mask`.

    Grad-CAM and KernelSHAP heatmaps are coarse g x g maps upsampled to
    pixels. A pixel is tied at the maximum iff every coarse cell it draws
    weight from is at the maximum, so the tied-max set depends only on WHICH
    cells are maximal. Enumerating every non-empty set of cells that touch the
    mask is therefore exact (cells that do not touch it can only add
    out-of-mask ties). Returns
        ceiling  sup over all maps (exact ties between different cells allowed)
        generic  sup when a single cell is the maximum (the typical case)
    or None for pixel-level methods, whose ceiling is 1.
    """
    size = mask.shape
    cm = _coarse_map(method, model_name, cfg, size)
    if cm is None:
        return None
    g, up = cm
    m = torch.from_numpy(mask)
    eye = torch.eye(g * g, dtype=torch.float64).view(g * g, 1, g, g)
    support = up(eye)[:, 0] > 0                                    # (g*g, H, W)
    relevant = [i for i in range(g * g) if (support[i] & m).any()]
    if len(relevant) > 16:
        raise ValueError(f"{len(relevant)} cells touch the mask; enumeration too large")
    pats = torch.zeros(2 ** len(relevant) - 1, 1, g * g, dtype=torch.float64)
    for b in range(1, 2 ** len(relevant)):
        for j, i in enumerate(relevant):
            if b >> j & 1:
                pats[b - 1, 0, i] = 1.0
    maps = up(pats.view(-1, 1, g, g))[:, 0].double()
    tied = maps >= maps.amax((1, 2), keepdim=True) - 1e-9           # tied-max pixels per pattern
    frac = (tied & m).sum((1, 2)).double() / tied.sum((1, 2)).double()
    single = pats[:, 0].sum(1) == 1
    return {"ceiling": float(frac.max()), "generic": float(frac[single].max()), "grid": g}


# ---------------------------------------------------------------- per-run cache

def attribution_images(split: dict, target_class: int, n: int) -> np.ndarray:
    """Indices of the fixed attribution set: the first n patched target-class images."""
    idx = np.flatnonzero((split["labels"] == target_class) & split["injected"])[:n]
    if len(idx) < n:
        raise ValueError(f"only {len(idx)} patched target images, need {n}")
    return idx


def attribution_path(cfg: dict, checkpoint: str | Path, method: str) -> Path:
    ck = Path(checkpoint)
    mcfg = cfg["attribution"][method]
    mkey = {"method": method, "cfg": mcfg, "n_images": cfg["attribution"]["n_images"],
            "reduction": reduction_key(method, mcfg)}
    digest = hashlib.sha1(json.dumps(mkey, sort_keys=True).encode()).hexdigest()[:8]
    # checkpoints/<patch_type>/<model>/<stem>.pt -> attributions/<patch_type>/<model>/<stem>/<method>_<h>.npz
    rel = ck.relative_to(Path(cfg["paths"]["cache_dir"]) / "checkpoints").with_suffix("")
    return Path(cfg["paths"]["cache_dir"]) / "attributions" / rel / f"{method}_{digest}.npz"


def attribute_run(cfg: dict, model: nn.Module, model_name: str, checkpoint: str | Path,
                  test_poisoned: dict, seed: int, device, methods=None, force: bool = False) -> dict:
    """Compute (or load) heatmaps for every method on one trained run.

    Returns {method: {"heatmaps": (N,H,W), "indices": (N,), "sec": float, "cached": bool}}.
    """
    acfg = cfg["attribution"]
    target_class = int(cfg["data"]["target_class"])
    idx = attribution_images(test_poisoned, target_class, acfg["n_images"])
    out = {}
    for method in methods or acfg["methods"]:
        path = attribution_path(cfg, checkpoint, method)
        if path.exists() and not force:
            with np.load(path) as z:
                out[method] = {"heatmaps": z["heatmaps"], "indices": z["indices"],
                               "sec": float(z["sec"]), "cached": True}
            continue
        t0 = time.perf_counter()
        heats = []
        bs = acfg["batch_size"]
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            x = torch.from_numpy(test_poisoned["images"][b]).permute(0, 3, 1, 2).float().div(255).to(device)
            t = torch.full((len(b),), target_class, dtype=torch.long, device=device)
            heats.append(attribute(method, model, model_name, x, t, cfg, seed=seed * 1000 + s))
        heat = np.concatenate(heats).astype(np.float32)
        sec = time.perf_counter() - t0
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, heatmaps=heat, indices=idx, sec=sec)
        out[method] = {"heatmaps": heat, "indices": idx, "sec": sec, "cached": False}
    return out
