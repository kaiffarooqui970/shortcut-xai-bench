# Learnability pre-check: SUPERSEDED origin (22, 22) (off-grid, but overlaps the object more)

injection_rate 1.0, seed 0, epochs full schedule. Gate: airplane gap >= 0.05.

| model | patch_type | epochs | acc_poisoned | acc_clean | airplane poisoned | airplane clean | **reliance gap** | trigger flip rate | final val acc | valid |
|---|---|---|---|---|---|---|---|---|---|---|
| cnn | obvious | 6 | 0.859 | 0.759 | 1.000 | 0.000 | **+1.000** | 0.863 (base 0.000) | 0.859 | VALID |
| cnn | subtle | 6 | 0.859 | 0.786 | 0.961 | 0.233 | **+0.728** | 0.260 (base 0.003) | 0.855 | VALID |
| vit | obvious | 15 | 0.731 | 0.631 | 1.000 | 0.002 | **+0.998** | 0.999 (base 0.000) | 0.730 | VALID |
| vit | subtle | 15 | 0.719 | 0.633 | 0.964 | 0.111 | **+0.853** | 0.637 (base 0.004) | 0.725 | VALID |

trigger flip rate = share of patched NON-airplane test images predicted 'airplane' (base = same images unpatched).

## Timing
- hardware: {'device': 'mps', 'torch': '2.14.0', 'python': '3.12.13', 'platform': 'macOS-26.3.1-arm64-arm-64bit', 'chip': 'Apple M4'}
- cnn: 30.8 s/epoch x 6 epochs + 8 s data/eval = 3 min per run
- vit: 18.0 s/epoch x 15 epochs + 6 s data/eval = 5 min per run

## Projected full grid (training only)
- cells per model: 2 patch types x 6 rates x 3 seeds = 36
- cnn: 36 x 3 min = **1.9 h**
- vit: 36 x 5 min = **2.8 h**
- **total: 4.7 h** for 72 training runs (the 4 pre-check runs are sweep cells and are reused from cache)
