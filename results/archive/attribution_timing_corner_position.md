# Attribution timing

One checkpoint per model (subtle, rate 1.0, seed 0), 100 patched test airplanes, target = airplane logit.
Hardware: {'device': 'mps', 'torch': '2.14.0', 'python': '3.12.13', 'platform': 'macOS-26.3.1-arm64-arm-64bit', 'chip': 'Apple M4'}

| method | cnn s / 100 img | vit s / 100 img | projected sweep total |
|---|---|---|---|
| integrated_gradients | 1.7 | 1.6 | 2.0 min |
| gradcam | 0.2 | 0.3 | 16.8 s |
| lrp | 0.4 | 0.3 | 24.1 s |
| kernel_shap | 27.6 | 14.8 | 25.4 min |
| guided_backprop | 0.1 | 0.0 | 4.3 s |
| random | 0.0 | 0.0 | 0.1 s |
| **all methods** | 30.0 | 17.1 | **28.2 min** |

Projection: 36 runs per model x per-run cost (upper bound: runs that fail the reliance gate are not attributed).

KernelSHAP settings: 256 coalitions, 4x4 superpixels (64 features).
