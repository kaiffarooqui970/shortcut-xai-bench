# Learnability pre-check

injection_rate 1.0, seed 0, epochs full schedule. Gate: airplane gap >= 0.05.

| model | patch_type | epochs | acc_poisoned | acc_clean | airplane poisoned | airplane clean | **reliance gap** | trigger flip rate | final val acc | valid |
|---|---|---|---|---|---|---|---|---|---|---|
| cnn | obvious | 6 | 0.863 | 0.763 | 0.998 | 0.001 | **+0.997** | 0.789 (base 0.000) | 0.862 | VALID |
| cnn | subtle | 6 | 0.859 | 0.793 | 0.959 | 0.303 | **+0.656** | 0.231 (base 0.006) | 0.864 | VALID |
| vit | obvious | 15 | 0.725 | 0.625 | 0.999 | 0.000 | **+0.999** | 0.998 (base 0.000) | 0.730 | VALID |
| vit | subtle | 15 | 0.721 | 0.635 | 0.968 | 0.107 | **+0.861** | 0.665 (base 0.005) | 0.721 | VALID |

trigger flip rate = share of patched NON-airplane test images predicted 'airplane' (base = same images unpatched).

## Timing
- hardware: {'device': 'mps', 'torch': '2.14.0', 'python': '3.12.13', 'platform': 'macOS-26.3.1-arm64-arm-64bit', 'chip': 'Apple M4'}
- cnn: 37.7 s/epoch x 6 epochs + 9 s data/eval = 4 min per run
- vit: 18.3 s/epoch x 15 epochs + 6 s data/eval = 5 min per run

## Projected full grid (training only)
- cells per model: 2 patch types x 6 rates x 3 seeds = 36
- cnn: 36 x 4 min = **2.4 h**
- vit: 36 x 5 min = **2.8 h**
- **total: 5.2 h** for 72 training runs (the 4 pre-check runs are sweep cells and are reused from cache)

## Pre-check: Shortcut reliance: VIT vs CNN

Mean ± 95% CI across seeds. 'wins' = seeds (paired) where the VIT value is higher.

| patch_type | rate | seeds | gap vit | gap cnn | gap wins | flip vit | flip cnn | flip wins | val acc vit | val acc cnn |
|---|---|---|---|---|---|---|---|---|---|---|
| obvious | 1.0 | 1 | +1.00 | +1.00 | 1/1 | 100% | 79% | 1/1 | 73% | 86% |
| subtle | 1.0 | 1 | +0.86 | +0.66 | 1/1 | 66% | 23% | 1/1 | 72% | 86% |

- **obvious**: VIT has the larger reliance gap in 1/1 paired runs and the larger trigger flip rate in 1/1; it wins on gap for every seed at 1/1 rates.
- **subtle**: VIT has the larger reliance gap in 1/1 paired runs and the larger trigger flip rate in 1/1; it wins on gap for every seed at 1/1 rates.

(single seed: a hypothesis for the sweep to test at every rate, not a result)
