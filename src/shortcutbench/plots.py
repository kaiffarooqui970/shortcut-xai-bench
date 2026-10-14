"""Figures for the README.

detection_figure: x = injection rate, y = one detection metric, one line per
attribution method with a shaded 95% CI across seeds, the random baseline as
a dashed grey line. Panels: rows = patch type, columns = model.

reliance_figure: trigger flip rate (primary) and reliance gap (secondary) vs
injection rate, CNN vs ViT, same panel layout by patch type.

Colours: five categorical slots (blue, orange, aqua, yellow, violet). They
clear colour-blind separation across all pairs; yellow/orange sits just under
the normal-vision floor, so every method also gets its own marker and a
direct end-of-line label. Random is neutral grey and dashed, never a slot.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, GRID, SURFACE, NEUTRAL = "#1b1f27", "#52514e", "#e4e4e0", "#fcfcfb", "#8a8984"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]
MARKERS = ["o", "s", "^", "D", "v"]
METHOD_LABELS = {"integrated_gradients": "Integrated Gradients", "gradcam": "Grad-CAM", "lrp": "LRP",
                 "kernel_shap": "KernelSHAP", "guided_backprop": "Guided Backprop", "random": "random baseline"}
METRIC_LABELS = {"auc": "pixel AUC-ROC", "pointing": "pointing game (expected hit rate)",
                 "topk": "top-k precision (k = mask size)",
                 "precision_k64": "top-64 precision", "recall_k64": "top-64 recall"}
MODEL_LABELS = {"cnn": "CNN (ResNet-18 w32)", "vit": "ViT (128×4)"}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _place_labels(ax, items, lo=0.0, hi=1.02, gap=0.05) -> None:
    """Direct end-of-line labels, spread apart so they never overlap.
    items: (x, y, text). Moved labels get a thin leader line to their endpoint."""
    items = sorted(items, key=lambda t: t[1])
    ys = [t[1] for t in items]
    for i in range(1, len(ys)):                      # push up
        ys[i] = max(ys[i], ys[i - 1] + gap)
    if ys and ys[-1] > hi:                           # then shift down if over the top
        shift = ys[-1] - hi
        ys = [y - shift for y in ys]
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - gap)
    x_text = max(t[0] for t in items) if items else 0
    for (x, y, text), y_lab in zip(items, ys):
        ax.annotate(text, xy=(x, y), xytext=(x_text + 0.035, max(lo, y_lab)), textcoords="data",
                    fontsize=7.5, color=INK2, va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8, shrinkA=0, shrinkB=3)
                    if abs(y_lab - y) > 0.01 else None)


def _series(summary, pt, model, method, metric):
    pts = sorted((r["rate"], r[metric], r[f"{metric}_ci"], r[f"{metric}_n"], r["n_runs"])
                 for r in summary if r["patch_type"] == pt and r["model"] == model and r["method"] == method)
    return pts


def detection_figure(summary: list[dict], metric: str, methods: list[str], path: str | Path,
                     patch_types: list[str], models: list[str], ceilings: dict | None = None) -> Path:
    """ceilings: {(patch_type, model, method): value} drawn as dashed lines in the
    method's colour where value < 1 (a structural limit of the map, not a score)."""
    real = [m for m in methods if m != "random"]
    colour = {m: SLOTS[i] for i, m in enumerate(real)}
    marker = {m: MARKERS[i] for i, m in enumerate(real)}
    fig, axes = plt.subplots(len(patch_types), len(models), figsize=(5.2 * len(models), 3.9 * len(patch_types)),
                             sharex=True, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for i, pt in enumerate(patch_types):
        for j, model in enumerate(models):
            ax = axes[i][j]
            _style(ax)
            labels = []
            for method in methods:
                pts = [p for p in _series(summary, pt, model, method, metric) if not math.isnan(p[1])]
                if not pts:
                    continue
                x = [p[0] for p in pts]
                y = [p[1] for p in pts]
                lo = [max(0.0, p[1] - (0 if math.isnan(p[2]) else p[2])) for p in pts]
                hi = [min(1.0, p[1] + (0 if math.isnan(p[2]) else p[2])) for p in pts]
                if method == "random":
                    ax.plot(x, y, color=NEUTRAL, lw=1.6, ls=(0, (5, 3)), zorder=2)
                    ax.fill_between(x, lo, hi, color=NEUTRAL, alpha=0.12, lw=0, zorder=1)
                    label = METHOD_LABELS[method]
                else:
                    c = colour[method]
                    ax.fill_between(x, lo, hi, color=c, alpha=0.13, lw=0, zorder=1)
                    ax.plot(x, y, color=c, lw=2, marker=marker[method], ms=5.5,
                            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
                    label = METHOD_LABELS[method]
                labels.append((x[-1], y[-1], label))
            _place_labels(ax, labels)
            by_value = {}
            for method in real:
                c = (ceilings or {}).get((pt, model, method))
                if c is not None and c < 1 - 1e-9:
                    by_value.setdefault(round(c, 4), []).append(method)
            for c, ms in by_value.items():
                for k, method in enumerate(ms):      # offset dashes so coincident lines show both colours
                    ax.axhline(c, color=colour[method], lw=1.2, ls=(k * 4, (4, 4)), alpha=0.9, zorder=2)
                ax.annotate(" & ".join(METHOD_LABELS[m] for m in ms) + f" resolution ceiling ({c:.2f})",
                            (0, c), xycoords=("axes fraction", "data"), xytext=(4, 3), textcoords="offset points",
                            fontsize=7, color=INK2, va="bottom")
            if not labels:
                ax.text(0.5, 0.5, "no valid run:\nthe model did not rely on the shortcut\n(reliance gate), so nothing to score",
                        transform=ax.transAxes, ha="center", va="center", fontsize=9, color=INK2)
            # seeds lost to the reliance gate, per rate
            ref = [r for r in summary if r["patch_type"] == pt and r["model"] == model]
            for r in {(r["rate"], r["n_valid"], r["n_runs"]) for r in ref}:
                if r[1] < r[2]:
                    ax.annotate(f"{r[1]}/{r[2]} valid", (r[0], 0.02), ha="center", fontsize=7, color=INK2)
            ax.set_title(f"{pt} patch · {MODEL_LABELS.get(model, model)}", fontsize=10, color=INK, loc="left")
            ax.set_ylim(0, 1.02)
            rates = sorted({r["rate"] for r in ref})
            if rates:
                ax.set_xticks(rates)
                ax.set_xlim(min(rates) - 0.03, max(rates) + 0.03)
            if i == len(patch_types) - 1:
                ax.set_xlabel("injection rate (share of training airplanes carrying the patch)", fontsize=8.5,
                              color=INK2)
            if j == 0:
                ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9, color=INK2)
    handles = [plt.Line2D([], [], color=colour[m], marker=marker[m], lw=2, label=METHOD_LABELS[m]) for m in real]
    handles.append(plt.Line2D([], [], color=NEUTRAL, lw=1.6, ls=(0, (5, 3)), label=METHOD_LABELS["random"]))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.955), labelcolor=INK2)
    fig.suptitle(f"When do attribution methods stop finding the patch?  "
                 f"({METRIC_LABELS.get(metric, metric)}, mean ± 95% CI over seeds)",
                 fontsize=11, color=INK, x=0.02, ha="left", y=0.995)
    bottom = 0.0
    if ceilings and any(v < 1 - 1e-9 for v in ceilings.values()):
        fig.text(0.02, 0.008, "Dashed coloured lines: resolution ceilings, not scores. Each is the highest pointing "
                 "score any heatmap of that method's form can reach on this mask: its coarse grid makes\na flat block "
                 "at the maximum that straddles the mask edge. Exact ties between different cells allowed; with a "
                 "single peak cell, ViT Grad-CAM on the obvious patch is capped at 0.25.",
                 fontsize=7.5, color=INK2, ha="left", va="bottom")
        bottom = 0.045
    fig.tight_layout(rect=(0, bottom, 0.93, 0.93), w_pad=7)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def reliance_figure(rows: list[dict], path: str | Path, patch_types: list[str], models: list[str],
                    min_gap: float) -> Path:
    """rows = reliance_by_model() output is per (pt, rate); here we want per-model series,
    so take runs.csv rows and aggregate directly."""
    from collections import defaultdict

    from .report import mean_ci95
    colour = {m: SLOTS[i] for i, m in enumerate(models)}
    marker = {m: MARKERS[i] for i, m in enumerate(models)}
    agg = defaultdict(list)
    for r in rows:
        agg[(r["patch_type"], r["model"], float(r["rate"]))].append(r)
    panels = [("trigger_flip_rate", "trigger flip rate (primary)\n(patched non-airplanes → 'airplane')"),
              ("reliance_gap", "reliance gap (secondary)\n(airplane acc. patched − clean)")]
    fig, axes = plt.subplots(len(patch_types), 2, figsize=(10.4, 3.7 * len(patch_types)), sharex=True,
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for i, pt in enumerate(patch_types):
        for j, (metric, ylabel) in enumerate(panels):
            ax = axes[i][j]
            _style(ax)
            labels = []
            if metric == "reliance_gap":
                ax.axhline(min_gap, color=NEUTRAL, lw=1.2, ls=(0, (5, 3)))
                ax.annotate("validity gate", (1.0, min_gap), xytext=(0, 3), textcoords="offset points",
                            fontsize=7, color=INK2, ha="right")
            for model in models:
                pts = []
                for rate in sorted({k[2] for k in agg if k[0] == pt and k[1] == model}):
                    m, ci = mean_ci95([r[metric] for r in agg[(pt, model, rate)]])
                    pts.append((rate, m, 0 if math.isnan(ci) else ci))
                if not pts:
                    continue
                x, y, e = zip(*pts)
                ax.fill_between(x, [max(-0.05, a - b) for a, b in zip(y, e)],
                                [min(1.05, a + b) for a, b in zip(y, e)], color=colour[model], alpha=0.13, lw=0)
                ax.plot(x, y, color=colour[model], lw=2, marker=marker[model], ms=5.5,
                        markeredgecolor=SURFACE, markeredgewidth=1.2)
                labels.append((x[-1], y[-1], MODEL_LABELS.get(model, model).split(" ")[0]))
            _place_labels(ax, labels, lo=-0.05, hi=1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{pt} patch", fontsize=10, color=INK, loc="left")
            ax.set_ylabel(ylabel, fontsize=8.5, color=INK2)
            if i == len(patch_types) - 1:
                ax.set_xlabel("injection rate", fontsize=8.5, color=INK2)
    handles = [plt.Line2D([], [], color=colour[m], marker=marker[m], lw=2, label=MODEL_LABELS.get(m, m))
               for m in models]
    fig.legend(handles=handles, loc="upper center", ncol=len(models), frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.955), labelcolor=INK2)
    fig.suptitle("How much each model relies on the shortcut (mean ± 95% CI over seeds, all runs incl. invalid)",
                 fontsize=11, color=INK, x=0.02, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 0.95, 0.93))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path
