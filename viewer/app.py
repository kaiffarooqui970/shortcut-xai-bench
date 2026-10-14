"""Local viewer: inspect one image's heatmap against its ground-truth mask.

    streamlit run viewer/app.py

Everything shown is read from the cache that the sweep writes. Nothing here
trains or attributes. Per-image scores come from shortcutbench.scoring, the
same code that writes results/results.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shortcutbench.attribution import attribution_path  # noqa: E402
from shortcutbench.config import load_config  # noqa: E402
from shortcutbench.data import build_split  # noqa: E402
from shortcutbench.plots import heatmap_overlay  # noqa: E402
from shortcutbench.scoring import score_pair  # noqa: E402
from shortcutbench.train import list_current_runs  # noqa: E402

UPSCALE = 8   # 32x32 -> 256x256 for display (nearest neighbour keeps pixels crisp)

st.set_page_config(page_title="Shortcut XAI viewer", layout="wide")


# ---------------------------------------------------------------- loading

@st.cache_resource
def get_config() -> dict:
    return load_config()


@st.cache_data(ttl=60)   # the sweep adds runs while you watch
def list_runs(_cfg: dict) -> list[dict]:
    """Finished runs trained with the CURRENT config (stale checkpoints are hidden)."""
    keep = ("patch_type", "rate", "model", "seed", "epochs", "valid", "reliance_gap",
            "acc_poisoned_target", "acc_clean_target", "trigger_flip_rate", "checkpoint")
    return [{k: r[k] for k in keep} for r in list_current_runs(_cfg)]


@st.cache_data
def load_split(_cfg: dict, split: str, seed: int, patch_type: str) -> dict[str, np.ndarray]:
    return build_split(_cfg, split, seed=seed, patch_type=patch_type)


@st.cache_data
def load_heatmaps(path: str) -> dict[str, np.ndarray] | None:
    if not Path(path).exists():
        return None
    with np.load(path) as z:
        return {"heatmaps": z["heatmaps"], "indices": z["indices"]}


def upscale(a: np.ndarray) -> np.ndarray:
    return a.repeat(UPSCALE, 0).repeat(UPSCALE, 1)


def to_display(heat: np.ndarray) -> np.ndarray:
    """Min-max normalise a heatmap to [0, 1] for st.image (grayscale)."""
    lo, hi = float(heat.min()), float(heat.max())
    return (heat - lo) / (hi - lo) if hi > lo else np.zeros_like(heat)


# ---------------------------------------------------------------- sidebar

cfg = get_config()
runs = list_runs(cfg)
st.sidebar.title("Shortcut XAI viewer")
if not runs:
    st.info("No trained runs in the cache yet. Run `bash run_all.sh` (or the pre-check) first.")
    st.stop()

def options(key, pool):
    return sorted({r[key] for r in pool})

patch_type = st.sidebar.selectbox("patch_type", options("patch_type", runs))
pool = [r for r in runs if r["patch_type"] == patch_type]
rate = st.sidebar.selectbox("injection_rate", options("rate", pool), index=len(options("rate", pool)) - 1,
                            format_func=lambda r: f"{r:.1f}")
pool = [r for r in pool if r["rate"] == rate]
model_name = st.sidebar.selectbox("model", options("model", pool))
pool = [r for r in pool if r["model"] == model_name]
seed = st.sidebar.selectbox("seed", options("seed", pool))
run = next(r for r in pool if r["seed"] == seed)
method = st.sidebar.selectbox("attribution method", cfg["attribution"]["methods"])

heat_path = attribution_path(cfg, run["checkpoint"], method)
heat = load_heatmaps(str(heat_path))
n_images = len(heat["indices"]) if heat else cfg["attribution"]["n_images"]
image_i = st.sidebar.number_input("image index (within the attribution set)", 0, n_images - 1, 0)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Run** `{patch_type} / r={rate:.1f} / {model_name} / seed {seed}`  \n"
    f"reliance gap **{run['reliance_gap']:+.3f}** "
    f"({'VALID' if run['valid'] else 'INVALID: excluded from scoring'})  \n"
    f"airplane acc patched {run['acc_poisoned_target']:.3f} / clean {run['acc_clean_target']:.3f}  \n"
    f"trigger flip rate {run['trigger_flip_rate']:.3f}")

# ---------------------------------------------------------------- data for the selected image

poisoned = load_split(cfg, "test_poisoned", seed, patch_type)
clean = load_split(cfg, "test_clean", seed, patch_type)     # same test images, unpatched

if heat is None:
    st.warning(f"No `{method}` heatmaps for this run yet (expected at `{heat_path}`). "
               "They are written by the attribution stage of the sweep.")
    st.stop()

idx = int(heat["indices"][image_i])          # index into the test split
original = clean["images"][idx]              # (32, 32, 3) uint8
patched = poisoned["images"][idx]            # (32, 32, 3) uint8
mask = poisoned["masks"][idx]                # (32, 32) bool, saved at injection time
heatmap = heat["heatmaps"][image_i]          # (32, 32) float, >= 0

st.title(f"{method} on test image #{idx}")
st.caption(f"{patch_type} · rate {rate:.1f} · {model_name} · seed {seed} · "
           f"target class {cfg['data']['target_class']} (airplane)")

# ---------------------------------------------------------------- main panel

c_orig, c_patch, c_heat, c_mask, c_overlay = st.columns(5)
with c_orig:
    st.subheader("original")
    st.image(upscale(original), clamp=True)
with c_patch:
    st.subheader("patched")
    st.image(upscale(patched), clamp=True)
with c_heat:
    st.subheader("heatmap")
    st.image(upscale(to_display(heatmap)), clamp=True)
with c_mask:
    st.subheader("ground-truth mask")
    st.image(upscale(mask.astype(np.float32)), clamp=True)
with c_overlay:
    st.subheader("heatmap over mask")
    st.image(upscale(heatmap_overlay(heatmap, mask, patched)), clamp=True)
    st.caption("white outline = planted patch")

# ---------------------------------------------------------------- score for this image

st.markdown("---")
st.subheader("Score for this image")
top_k = cfg["scoring"]["top_k"]
scores = score_pair(heatmap, mask, top_k)
rand = load_heatmaps(str(attribution_path(cfg, run["checkpoint"], "random")))
rand_scores = score_pair(rand["heatmaps"][image_i], mask, top_k) if rand is not None else None
for col, (key, label) in zip(st.columns(3), (("topk", "top-k precision (k = mask size)"),
                                             ("auc", "pixel AUC-ROC"), ("pointing", "pointing game"))):
    delta = f"{scores[key] - rand_scores[key]:+.3f} vs random" if rand_scores else None
    col.metric(label, f"{scores[key]:.3f}", delta)
if rand_scores is None:
    st.caption("No random-baseline heatmaps for this run, so no comparison is shown.")
