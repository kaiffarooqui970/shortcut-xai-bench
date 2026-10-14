# Initial learnability pre-check (superseded settings: ResNet-18 width 64, 4 epochs, 3 patch types)

cnn, injection_rate 1.0, seed 0, 4 epochs. Gate: gap >= 0.05.

| patch_type | acc_poisoned | acc_clean | airplane poisoned | airplane clean | **reliance gap** | trigger flip rate | valid |
|---|---|---|---|---|---|---|---|
| obvious | 0.867 | 0.767 | 1.000 | 0.000 | **+1.000** | 0.910 (base 0.000) | VALID |
| subtle | 0.855 | 0.801 | 0.949 | 0.415 | **+0.534** | 0.184 (base 0.006) | VALID |
| subtle_checker | 0.862 | 0.808 | 0.960 | 0.428 | **+0.532** | 0.209 (base 0.003) | VALID |

trigger flip rate = share of patched NON-airplane test images predicted 'airplane' (base = same images unpatched).

## Timing
- hardware: {'device': 'mps', 'torch': '2.14.0', 'python': '3.12.13', 'platform': 'macOS-26.3.1-arm64-arm-64bit', 'chip': 'Apple M4'}
- one cnn pre-check run (4 epochs): 454 s (of which 22 s data + evaluation)
- cnn: 108.1 s/epoch (measured in pre-check); full run = 15 epochs = 27 min
- vit: 41.0 s/epoch (benchmarked, 40 steps); full run = 30 epochs = 21 min

## Projected full grid (training only)
- cells per model: 3 patch types x 6 rates x 5 seeds = 90
- cnn: 90 x 27 min = **41.1 h**
- vit: 90 x 21 min = **31.3 h**
- **total: 72.3 h** for 180 training runs (attribution + scoring not included yet)
