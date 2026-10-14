"""Result summaries that must survive into the README.

reliance_by_model: does the ViT lean on the shortcut more than the CNN?
Observed in the pre-check (rate 1.0, seed 0): ViT gap +0.67 vs CNN +0.48 and
flip rate 36% vs 16% on the subtle patch, despite lower ViT accuracy. This
module re-tests that at every (patch_type, rate) once the sweep has run, with
a 95% t-interval across seeds and a per-seed paired count.

Statistics (FINDINGS_LOG #13):
  primary    trigger flip rate: share of patched NON-airplane test images the
             model calls "airplane". It does not depend on how well the model
             classifies clean airplanes.
  secondary  reliance gap: airplane accuracy patched - clean. NOT normalised
             for clean accuracy, so a weaker model (the ViT) has more headroom
             for a large gap; it is also the validity gate.
The headline paired count excludes rate 1.0, where both statistics hit
ceilings (gap about 1.0 for both models on the obvious patch; ViT flip rate
100%); rate 1.0 is counted separately.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

# two-sided 95% t critical values, df = n - 1
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def mean_ci95(values) -> tuple[float, float]:
    """(mean, half-width of the 95% t-interval). Half-width is NaN for n < 2."""
    v = np.asarray([x for x in values if not math.isnan(x)], dtype=float)
    if len(v) == 0:
        return math.nan, math.nan
    if len(v) < 2:
        return float(v.mean()), math.nan
    t = _T95.get(len(v) - 1, 1.96)
    return float(v.mean()), float(t * v.std(ddof=1) / math.sqrt(len(v)))


TIE_TOL = 0.01   # |a - b| below this is a tie (e.g. both saturated at a gap of about 1.0)


def reliance_by_model(runs: list[dict], a: str = "vit", b: str = "cnn",
                      metrics=("trigger_flip_rate", "reliance_gap", "acc_clean_target", "final_val_acc"),
                      tol: float = TIE_TOL) -> list[dict]:
    """One row per (patch_type, rate) comparing model `a` against model `b`.

    Uses every run, valid or not: reliance is exactly what the validity gate
    measures, so excluding invalid runs here would bias the comparison.
    Seeds are paired (same seed = same poisoned subset). Per metric, each seed is
    a win for a (a > b + tol), a win for b (b > a + tol), or a tie.
    """
    cells = defaultdict(lambda: defaultdict(dict))        # (pt, rate) -> model -> seed -> run
    for r in runs:
        cells[(r["patch_type"], float(r["rate"]))][r["model"]][r["seed"]] = r
    rows = []
    for (pt, rate) in sorted(cells, key=lambda k: (k[0], -k[1])):
        by_model = cells[(pt, rate)]
        if a not in by_model or b not in by_model:
            continue
        seeds = sorted(set(by_model[a]) & set(by_model[b]))
        row = {"patch_type": pt, "rate": rate, "n_seeds": len(seeds)}
        for m in metrics:
            va = [by_model[a][s][m] for s in seeds]
            vb = [by_model[b][s][m] for s in seeds]
            row[f"{a}_{m}"], row[f"{a}_{m}_ci"] = mean_ci95(va)
            row[f"{b}_{m}"], row[f"{b}_{m}_ci"] = mean_ci95(vb)
            row[f"{m}_{a}_wins"] = int(sum(x > y + tol for x, y in zip(va, vb)))
            row[f"{m}_{b}_wins"] = int(sum(y > x + tol for x, y in zip(va, vb)))
            row[f"{m}_ties"] = len(seeds) - row[f"{m}_{a}_wins"] - row[f"{m}_{b}_wins"]
        rows.append(row)
    return rows


def _pm(mean: float, ci: float, pct: bool = False) -> str:
    if math.isnan(mean):
        return "n/a"
    f = (lambda x: f"{100 * x:.0f}%") if pct else (lambda x: f"{x:+.2f}")
    return f(mean) if math.isnan(ci) else f"{f(mean)} ± {f(ci).lstrip('+')}"


def _pm(mean: float, ci: float, pct: bool = False) -> str:
    if math.isnan(mean):
        return "n/a"
    f = (lambda x: f"{100 * x:.0f}%") if pct else (lambda x: f"{x:+.2f}")
    return f(mean) if math.isnan(ci) else f"{f(mean)} ± {f(ci).lstrip('+')}"


HEADLINE_MAX_RATE = 1.0     # rates strictly below this go into the headline count


def format_reliance_md(rows: list[dict], a: str = "vit", b: str = "cnn", runs: list[dict] | None = None,
                       min_gap: float | None = None) -> str:
    A, B = a.upper(), b.upper()
    lines = [
        f"## Shortcut reliance: {A} vs {B}",
        "",
        "**Primary statistic: trigger flip rate** (share of patched non-airplane test images called "
        "'airplane'). **Secondary: reliance gap** (airplane accuracy patched − clean), which is not "
        f"normalised for clean accuracy: the {A} is the weaker model (see clean airplane accuracy), "
        "so it has more headroom for a large gap.",
        "",
        f"Mean ± 95% CI across seeds. Per paired seed: {A} win / {B} win / tie (|difference| < {TIE_TOL}).",
        "",
        f"| patch_type | rate | seeds | flip {a} | flip {b} | flip {a}/{b}/tie | gap {a} | gap {b} "
        f"| gap {a}/{b}/tie | clean airplane acc {a} | clean airplane acc {b} | val acc {a} | val acc {b} |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        n = r["n_seeds"]
        lines.append(
            f"| {r['patch_type']} | {r['rate']:.1f} | {n} "
            f"| {_pm(r[f'{a}_trigger_flip_rate'], r[f'{a}_trigger_flip_rate_ci'], pct=True)} "
            f"| {_pm(r[f'{b}_trigger_flip_rate'], r[f'{b}_trigger_flip_rate_ci'], pct=True)} "
            f"| {r[f'trigger_flip_rate_{a}_wins']}/{r[f'trigger_flip_rate_{b}_wins']}/{r['trigger_flip_rate_ties']} "
            f"| {_pm(r[f'{a}_reliance_gap'], r[f'{a}_reliance_gap_ci'])} "
            f"| {_pm(r[f'{b}_reliance_gap'], r[f'{b}_reliance_gap_ci'])} "
            f"| {r[f'reliance_gap_{a}_wins']}/{r[f'reliance_gap_{b}_wins']}/{r['reliance_gap_ties']} "
            f"| {_pm(r[f'{a}_acc_clean_target'], r[f'{a}_acc_clean_target_ci'], pct=True)} "
            f"| {_pm(r[f'{b}_acc_clean_target'], r[f'{b}_acc_clean_target_ci'], pct=True)} "
            f"| {_pm(r[f'{a}_final_val_acc'], r[f'{a}_final_val_acc_ci'], pct=True)} "
            f"| {_pm(r[f'{b}_final_val_acc'], r[f'{b}_final_val_acc_ci'], pct=True)} |")
    lines += ["", "### Paired comparison", "", verdict(rows, a, b)]
    if runs is not None and min_gap is not None:
        lines += ["", invalid_runs_md(runs, min_gap, a, b)]
    return "\n".join(lines)


def _tally(rs: list[dict], m: str, a: str, b: str) -> tuple[int, int, int, int]:
    """(a wins, b wins, ties, pairs) summed over rows."""
    return (sum(r[f"{m}_{a}_wins"] for r in rs), sum(r[f"{m}_{b}_wins"] for r in rs),
            sum(r[f"{m}_ties"] for r in rs), sum(r["n_seeds"] for r in rs))


def verdict(rows: list[dict], a: str = "vit", b: str = "cnn") -> str:
    """Paired counts per patch type: headline over rates < 1.0, rate 1.0 separately."""
    A, B = a.upper(), b.upper()
    out = []
    for pt in sorted({r["patch_type"] for r in rows}):
        rs = [r for r in rows if r["patch_type"] == pt]
        main = [r for r in rs if r["rate"] < HEADLINE_MAX_RATE]
        top = [r for r in rs if r["rate"] >= HEADLINE_MAX_RATE]
        if main:
            f, g = _tally(main, "trigger_flip_rate", a, b), _tally(main, "reliance_gap", a, b)
            rates = sorted({r["rate"] for r in main})
            out.append(
                f"- **{pt}**, headline over {f[3]} paired runs (rates {rates[0]:.1f}–{rates[-1]:.1f}): "
                f"**trigger flip rate {A} larger in {f[0]}, {B} larger in {f[1]}, tied in {f[2]}** (primary); "
                f"reliance gap {A} larger in {g[0]}, {B} larger in {g[1]}, tied in {g[2]} (secondary).")
        if top:
            f, g = _tally(top, "trigger_flip_rate", a, b), _tally(top, "reliance_gap", a, b)
            ceil = [r for r in top if r[f"{a}_reliance_gap"] > 0.95 and r[f"{b}_reliance_gap"] > 0.95]
            note = (" Gap ties here are a ceiling: both models' gaps are about 1.0." if ceil and g[2] else "")
            sat = [r for r in top if r[f"{a}_trigger_flip_rate"] > 0.99 or r[f"{b}_trigger_flip_rate"] > 0.99]
            note += (" Flip rate is at its ceiling (100%) for at least one model." if sat else "")
            out.append(
                f"  - {pt}, rate 1.0, not in the headline ({f[3]} paired runs): flip rate {A}/{B}/tie "
                f"{f[0]}/{f[1]}/{f[2]}; gap {A}/{B}/tie {g[0]}/{g[1]}/{g[2]}.{note}")
    return "\n".join(out)


def invalid_runs_md(runs: list[dict], min_gap: float, a: str = "vit", b: str = "cnn") -> str:
    """Runs that failed the reliance gate, next to the other model at the same cell."""
    by = {(r["patch_type"], float(r["rate"]), r["model"], r["seed"]): r for r in runs}
    bad = sorted((r for r in runs if not r["valid"]),
                 key=lambda r: (r["patch_type"], r["model"], -float(r["rate"]), r["seed"]))
    lines = ["### Runs that failed the reliance gate", ""]
    if not bad:
        return "\n".join(lines + ["None."])
    lines += [f"Gate: reliance gap ≥ {min_gap}. Each failed run is shown next to the other model at the same "
              "(patch type, rate, seed). 'clean confusion' = share of CLEAN non-airplanes called airplane, the "
              "natural rate the trigger flip rate should be read against.",
              "",
              "| patch_type | rate | seed | failed model | gap | flip | clean confusion | other model | gap | flip "
              "| clean confusion | other valid |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in bad:
        other = a if r["model"] == b else b
        o = by.get((r["patch_type"], float(r["rate"]), other, r["seed"]))

        def cells(x):
            if x is None:
                return "n/a | n/a | n/a"
            cc = x.get("clean_nontarget_called_target", math.nan)
            return f"{x['reliance_gap']:+.3f} | {100 * x['trigger_flip_rate']:.1f}% | {100 * cc:.1f}%"
        lines.append(f"| {r['patch_type']} | {float(r['rate']):.1f} | {r['seed']} | {r['model'].upper()} | {cells(r)} "
                     f"| {other.upper()} | {cells(o)} | {'yes' if o and o['valid'] else 'no' if o else 'n/a'} |")
    return "\n".join(lines)
