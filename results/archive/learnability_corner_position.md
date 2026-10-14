# Learnability pre-check: SUPERSEDED corner position (patch flush with bottom-right corner, grid-aligned)

injection_rate 1.0, seed 0, epochs full schedule. Gate: airplane gap >= 0.05.

| model | patch_type | epochs | acc_poisoned | acc_clean | airplane poisoned | airplane clean | **reliance gap** | trigger flip rate | final val acc | valid |
|---|---|---|---|---|---|---|---|---|---|---|
| cnn | obvious | 6 | 0.858 | 0.758 | 0.999 | 0.000 | **+0.999** | 0.797 (base 0.000) | 0.860 | VALID |
| cnn | subtle | 6 | 0.857 | 0.810 | 0.943 | 0.467 | **+0.476** | 0.161 (base 0.007) | 0.861 | VALID |
| vit | obvious | 15 | 0.724 | 0.624 | 1.000 | 0.000 | **+1.000** | 0.997 (base 0.000) | 0.732 | VALID |
| vit | subtle | 15 | 0.712 | 0.645 | 0.889 | 0.222 | **+0.667** | 0.360 (base 0.012) | 0.722 | VALID |

trigger flip rate = share of patched NON-airplane test images predicted 'airplane' (base = same images unpatched).

## Timing
- hardware: {'device': 'mps', 'torch': '2.14.0', 'python': '3.12.13', 'platform': 'macOS-26.3.1-arm64-arm-64bit', 'chip': 'Apple M4'}
- cnn: 31.0 s/epoch x 6 epochs + 8 s data/eval = 3 min per run
- vit: 16.3 s/epoch x 15 epochs + 8 s data/eval = 4 min per run

## Projected full grid (training only)
- cells per model: 2 patch types x 6 rates x 3 seeds = 36
- cnn: 36 x 3 min = **1.9 h**
- vit: 36 x 4 min = **2.5 h**
- **total: 4.5 h** for 72 training runs (the 4 pre-check runs are sweep cells and are reused from cache)
