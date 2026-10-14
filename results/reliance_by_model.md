## Shortcut reliance: VIT vs CNN

**Primary statistic: trigger flip rate** (share of patched non-airplane test images called 'airplane'). **Secondary: reliance gap** (airplane accuracy patched − clean), which is not normalised for clean accuracy: the VIT is the weaker model (see clean airplane accuracy), so it has more headroom for a large gap.

Mean ± 95% CI across seeds. Per paired seed: VIT win / CNN win / tie (|difference| < 0.01).

| patch_type | rate | seeds | flip vit | flip cnn | flip vit/cnn/tie | gap vit | gap cnn | gap vit/cnn/tie | clean airplane acc vit | clean airplane acc cnn | val acc vit | val acc cnn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| obvious | 1.0 | 3 | 100% ± 0% | 82% ± 10% | 3/0/0 | +1.00 ± 0.00 | +1.00 ± 0.00 | 0/0/3 | 0% ± 0% | 0% ± 0% | 73% ± 2% | 86% ± 1% |
| obvious | 0.9 | 3 | 100% ± 1% | 62% ± 9% | 3/0/0 | +0.72 ± 0.01 | +0.49 ± 0.01 | 3/0/0 | 28% ± 1% | 50% ± 1% | 72% ± 2% | 86% ± 2% |
| obvious | 0.8 | 3 | 99% ± 1% | 56% ± 9% | 3/0/0 | +0.58 ± 0.02 | +0.35 ± 0.02 | 3/0/0 | 42% ± 2% | 65% ± 2% | 72% ± 3% | 85% ± 1% |
| obvious | 0.7 | 3 | 99% ± 1% | 51% ± 6% | 3/0/0 | +0.51 ± 0.03 | +0.28 ± 0.05 | 3/0/0 | 49% ± 3% | 72% ± 5% | 71% ± 3% | 85% ± 1% |
| obvious | 0.6 | 3 | 99% ± 2% | 50% ± 1% | 3/0/0 | +0.44 ± 0.00 | +0.24 ± 0.03 | 3/0/0 | 56% ± 0% | 76% ± 3% | 70% ± 2% | 85% ± 2% |
| obvious | 0.5 | 3 | 99% ± 2% | 42% ± 7% | 3/0/0 | +0.41 ± 0.00 | +0.20 ± 0.04 | 3/0/0 | 60% ± 0% | 79% ± 3% | 70% ± 4% | 85% ± 1% |
| subtle | 1.0 | 3 | 58% ± 18% | 25% ± 9% | 3/0/0 | +0.83 ± 0.08 | +0.69 ± 0.14 | 3/0/0 | 13% ± 6% | 27% ± 14% | 72% ± 2% | 86% ± 0% |
| subtle | 0.9 | 3 | 51% ± 24% | 12% ± 7% | 3/0/0 | +0.56 ± 0.10 | +0.25 ± 0.07 | 3/0/0 | 39% ± 6% | 69% ± 7% | 71% ± 2% | 86% ± 0% |
| subtle | 0.8 | 3 | 44% ± 29% | 7% ± 2% | 3/0/0 | +0.44 ± 0.06 | +0.16 ± 0.06 | 3/0/0 | 50% ± 2% | 77% ± 7% | 71% ± 1% | 85% ± 0% |
| subtle | 0.7 | 3 | 36% ± 24% | 4% ± 2% | 3/0/0 | +0.36 ± 0.08 | +0.09 ± 0.05 | 3/0/0 | 57% ± 3% | 82% ± 3% | 70% ± 2% | 85% ± 0% |
| subtle | 0.6 | 3 | 22% ± 24% | 3% ± 1% | 3/0/0 | +0.26 ± 0.14 | +0.06 ± 0.02 | 3/0/0 | 62% ± 5% | 85% ± 3% | 70% ± 2% | 84% ± 1% |
| subtle | 0.5 | 3 | 11% ± 13% | 2% ± 1% | 3/0/0 | +0.17 ± 0.15 | +0.03 ± 0.03 | 3/0/0 | 68% ± 4% | 87% ± 3% | 69% ± 2% | 85% ± 1% |

### Paired comparison

- **obvious**, headline over 15 paired runs (rates 0.5–0.9): **trigger flip rate VIT larger in 15, CNN larger in 0, tied in 0** (primary); reliance gap VIT larger in 15, CNN larger in 0, tied in 0 (secondary).
  - obvious, rate 1.0, not in the headline (3 paired runs): flip rate VIT/CNN/tie 3/0/0; gap VIT/CNN/tie 0/0/3. Gap ties here are a ceiling: both models' gaps are about 1.0. Flip rate is at its ceiling (100%) for at least one model.
- **subtle**, headline over 15 paired runs (rates 0.5–0.9): **trigger flip rate VIT larger in 15, CNN larger in 0, tied in 0** (primary); reliance gap VIT larger in 15, CNN larger in 0, tied in 0 (secondary).
  - subtle, rate 1.0, not in the headline (3 paired runs): flip rate VIT/CNN/tie 3/0/0; gap VIT/CNN/tie 3/0/0.

### Runs that failed the reliance gate

Gate: reliance gap ≥ 0.05. Each failed run is shown next to the other model at the same (patch type, rate, seed). 'clean confusion' = share of CLEAN non-airplanes called airplane, the natural rate the trigger flip rate should be read against.

| patch_type | rate | seed | failed model | gap | flip | clean confusion | other model | gap | flip | clean confusion | other valid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| subtle | 0.6 | 1 | CNN | +0.049 | 2.5% | 1.5% | VIT | +0.260 | 18.6% | 2.8% | yes |
| subtle | 0.5 | 0 | CNN | +0.022 | 2.1% | 1.7% | VIT | +0.223 | 16.7% | 3.0% | yes |
| subtle | 0.5 | 1 | CNN | +0.035 | 2.6% | 1.9% | VIT | +0.174 | 11.6% | 2.9% | yes |
| subtle | 0.5 | 2 | CNN | +0.043 | 2.0% | 1.5% | VIT | +0.105 | 6.1% | 3.0% | yes |

Status: complete. Source: 72 runs in results/runs.csv.
