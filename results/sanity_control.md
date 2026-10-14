## Sanity control: clean-trained model, patched images

Each model is trained at injection rate 0.0 (never sees the patch) and explained on the SAME patched test airplanes, scored against the SAME masks as the sweep. A method that scores well on the clean model is responding to the patch's visual salience, not to model reliance. Comparison: the rate-1.0 sweep run with the same seed. Mean ± 95% CI over 3 seed(s); the CI is omitted for a single seed.

### The clean models do not use the patch

| patch_type | model | seeds | reliance gap | trigger flip rate | clean confusion | val acc |
|---|---|---|---|---|---|---|
| obvious | cnn | 3 | -0.005 | 1.8% | 1.8% | 84.7% |
| obvious | vit | 3 | +0.007 | 4.8% | 3.5% | 68.8% |
| subtle | cnn | 3 | -0.002 | 1.7% | 1.8% | 84.7% |
| subtle | vit | 3 | +0.001 | 3.6% | 3.5% | 68.8% |

Validity gate for reference: gap ≥ 0.05. Trigger flip rate should be read against clean confusion (the natural rate at which unpatched non-airplanes are called airplane).

### topk: clean model vs shortcut model (rate 1.0)

| method | obvious cnn clean | obvious cnn shortcut | obvious vit clean | obvious vit shortcut | subtle cnn clean | subtle cnn shortcut | subtle vit clean | subtle vit shortcut |
|---|---|---|---|---|---|---|---|---|
| Integrated Gradients | 0.00 ± 0.00 | 0.55 ± 0.22 | 0.01 ± 0.01 | 0.30 ± 0.37 | 0.04 ± 0.03 | 0.49 ± 0.07 | 0.02 ± 0.01 | 0.38 ± 0.12 |
| Grad-CAM | 0.01 ± 0.01 | 0.56 ± 0.02 | 0.02 ± 0.02 | 0.17 ± 0.42 | 0.02 ± 0.01 | 0.29 ± 0.09 | 0.13 ± 0.11 | 0.09 ± 0.09 |
| LRP | 0.09 ± 0.19 | 0.97 ± 0.06 | 0.01 ± 0.01 | 0.31 ± 0.44 | 0.03 ± 0.00 | 0.49 ± 0.06 | 0.03 ± 0.03 | 0.31 ± 0.33 |
| KernelSHAP | 0.01 ± 0.01 | 0.56 ± 0.03 | 0.02 ± 0.02 | 0.29 ± 0.60 | 0.04 ± 0.04 | 0.30 ± 0.13 | 0.05 ± 0.01 | 0.32 ± 0.08 |
| Guided Backprop | 0.00 ± 0.00 | 0.89 ± 0.03 | 0.00 ± 0.00 | 0.10 ± 0.03 | 0.01 ± 0.01 | 0.36 ± 0.10 | 0.04 ± 0.02 | 0.35 ± 0.12 |
| random baseline | 0.02 ± 0.00 | 0.02 ± 0.00 | 0.02 ± 0.00 | 0.02 ± 0.00 | 0.06 ± 0.01 | 0.06 ± 0.01 | 0.06 ± 0.01 | 0.06 ± 0.01 |

### auc: clean model vs shortcut model (rate 1.0)

| method | obvious cnn clean | obvious cnn shortcut | obvious vit clean | obvious vit shortcut | subtle cnn clean | subtle cnn shortcut | subtle vit clean | subtle vit shortcut |
|---|---|---|---|---|---|---|---|---|
| Integrated Gradients | 0.34 ± 0.06 | 0.96 ± 0.04 | 0.46 ± 0.25 | 0.72 ± 0.39 | 0.44 ± 0.07 | 0.92 ± 0.03 | 0.41 ± 0.07 | 0.77 ± 0.24 |
| Grad-CAM | 0.34 ± 0.16 | 0.99 ± 0.00 | 0.46 ± 0.09 | 0.72 ± 0.72 | 0.45 ± 0.11 | 0.86 ± 0.07 | 0.61 ± 0.14 | 0.55 ± 0.19 |
| LRP | 0.77 ± 0.14 | 1.00 ± 0.00 | 0.41 ± 0.40 | 0.59 ± 0.50 | 0.44 ± 0.02 | 0.91 ± 0.01 | 0.45 ± 0.08 | 0.71 ± 0.24 |
| KernelSHAP | 0.43 ± 0.02 | 0.97 ± 0.04 | 0.48 ± 0.14 | 0.82 ± 0.09 | 0.48 ± 0.06 | 0.75 ± 0.12 | 0.49 ± 0.02 | 0.68 ± 0.01 |
| Guided Backprop | 0.46 ± 0.04 | 1.00 ± 0.00 | 0.33 ± 0.28 | 0.52 ± 0.33 | 0.36 ± 0.06 | 0.87 ± 0.05 | 0.48 ± 0.04 | 0.77 ± 0.18 |
| random baseline | 0.51 ± 0.01 | 0.51 ± 0.01 | 0.51 ± 0.01 | 0.51 ± 0.01 | 0.50 ± 0.01 | 0.50 ± 0.01 | 0.50 ± 0.01 | 0.50 ± 0.01 |

### pointing: clean model vs shortcut model (rate 1.0)

| method | obvious cnn clean | obvious cnn shortcut | obvious vit clean | obvious vit shortcut | subtle cnn clean | subtle cnn shortcut | subtle vit clean | subtle vit shortcut |
|---|---|---|---|---|---|---|---|---|
| Integrated Gradients | 0.00 ± 0.00 | 0.91 ± 0.21 | 0.02 ± 0.02 | 0.97 ± 0.13 | 0.03 ± 0.05 | 0.52 ± 0.71 | 0.02 ± 0.04 | 0.74 ± 0.44 |
| Grad-CAM | 0.01 ± 0.02 | 0.55 ± 0.04 | 0.01 ± 0.03 | 0.10 ± 0.22 | 0.01 ± 0.01 | 0.14 ± 0.13 | 0.08 ± 0.11 | 0.07 ± 0.02 |
| LRP | 0.06 ± 0.17 | 1.00 ± 0.00 | 0.02 ± 0.04 | 1.00 ± 0.00 | 0.02 ± 0.01 | 0.68 ± 0.20 | 0.03 ± 0.05 | 0.82 ± 0.17 |
| KernelSHAP | 0.01 ± 0.01 | 0.56 ± 0.03 | 0.02 ± 0.02 | 0.29 ± 0.60 | 0.03 ± 0.03 | 0.31 ± 0.16 | 0.03 ± 0.01 | 0.44 ± 0.32 |
| Guided Backprop | 0.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.01 | 0.15 ± 0.21 | 0.01 ± 0.01 | 0.72 ± 0.28 | 0.04 ± 0.03 | 0.80 ± 0.16 |
| random baseline | 0.01 ± 0.01 | 0.01 ± 0.01 | 0.01 ± 0.01 | 0.01 ± 0.01 | 0.04 ± 0.07 | 0.04 ± 0.07 | 0.04 ± 0.07 | 0.04 ± 0.07 |
