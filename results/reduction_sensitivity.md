## Channel-reduction sensitivity (rate 1.0, seed 0)

IG, LRP and Guided Backprop recomputed with `pos` (positive part of the signed RGB sum) instead of the sweep's `abs_sum` (sum of |attribution| over RGB). Grad-CAM, KernelSHAP and random are unaffected (same values in both columns). Single seed: a slice, not a sweep.

### topk

| patch | model | Integrated Gradients abs → pos | Grad-CAM abs → pos | LRP abs → pos | KernelSHAP abs → pos | Guided Backprop abs → pos | random baseline abs → pos | ranking abs_sum | ranking pos | Kendall τ |
|---|---|---|---|---|---|---|---|---|---|---|
| obvious | cnn | 0.558 → 0.415 | 0.557 | 0.966 → 0.871 | 0.562 | 0.905 → 0.427 | 0.016 | LRP > GBP > KSHAP > IG > GCAM > rand | LRP > KSHAP > GCAM > GBP > IG > rand | 0.60 |
| obvious | vit | 0.237 → 0.380 | 0.349 | 0.213 → 0.249 | 0.115 | 0.083 → 0.059 | 0.016 | GCAM > IG > LRP > KSHAP > GBP > rand | IG > GCAM > LRP > KSHAP > GBP > rand | 0.87 |
| subtle | cnn | 0.520 → 0.436 | 0.320 | 0.517 → 0.378 | 0.328 | 0.402 → 0.221 | 0.063 | IG > LRP > GBP > KSHAP > GCAM > rand | IG > LRP > KSHAP > GCAM > GBP > rand | 0.73 |
| subtle | vit | 0.399 → 0.227 | 0.091 | 0.157 → 0.102 | 0.303 | 0.313 → 0.122 | 0.063 | IG > GBP > KSHAP > LRP > GCAM > rand | KSHAP > IG > GBP > LRP > GCAM > rand | 0.73 |

### auc

| patch | model | Integrated Gradients abs → pos | Grad-CAM abs → pos | LRP abs → pos | KernelSHAP abs → pos | Guided Backprop abs → pos | random baseline abs → pos | ranking abs_sum | ranking pos | Kendall τ |
|---|---|---|---|---|---|---|---|---|---|---|
| obvious | cnn | 0.963 → 0.744 | 0.994 | 1.000 → 0.932 | 0.949 | 1.000 → 0.625 | 0.512 | LRP > GBP > GCAM > IG > KSHAP > rand | GCAM > KSHAP > LRP > IG > GBP > rand | 0.20 |
| obvious | vit | 0.837 → 0.812 | 0.956 | 0.378 → 0.609 | 0.857 | 0.403 → 0.459 | 0.512 | GCAM > KSHAP > IG > rand > GBP > LRP | GCAM > KSHAP > IG > LRP > rand > GBP | 0.73 |
| subtle | cnn | 0.926 → 0.671 | 0.859 | 0.910 → 0.641 | 0.774 | 0.891 → 0.609 | 0.502 | IG > LRP > GBP > GCAM > KSHAP > rand | GCAM > KSHAP > IG > LRP > GBP > rand | 0.20 |
| subtle | vit | 0.715 → 0.602 | 0.579 | 0.594 → 0.537 | 0.683 | 0.756 → 0.554 | 0.502 | GBP > IG > KSHAP > LRP > GCAM > rand | KSHAP > IG > GCAM > GBP > LRP > rand | 0.33 |

### pointing

| patch | model | Integrated Gradients abs → pos | Grad-CAM abs → pos | LRP abs → pos | KernelSHAP abs → pos | Guided Backprop abs → pos | random baseline abs → pos | ranking abs_sum | ranking pos | Kendall τ |
|---|---|---|---|---|---|---|---|---|---|---|
| obvious | cnn | 0.970 → 0.980 | 0.557 | 1.000 → 1.000 | 0.562 | 1.000 → 1.000 | 0.010 | LRP > GBP > IG > KSHAP > GCAM > rand | LRP > GBP > IG > KSHAP > GCAM > rand | 1.00 |
| obvious | vit | 1.000 → 1.000 | 0.185 | 1.000 → 1.000 | 0.115 | 0.250 → 0.230 | 0.010 | IG > LRP > GBP > GCAM > KSHAP > rand | IG > LRP > GBP > GCAM > KSHAP > rand | 1.00 |
| subtle | cnn | 0.710 → 0.960 | 0.203 | 0.760 → 0.690 | 0.379 | 0.690 → 0.360 | 0.010 | LRP > IG > GBP > KSHAP > GCAM > rand | IG > LRP > KSHAP > GBP > GCAM > rand | 0.73 |
| subtle | vit | 0.540 → 0.740 | 0.067 | 0.870 → 0.340 | 0.295 | 0.780 → 0.350 | 0.010 | LRP > GBP > IG > KSHAP > GCAM > rand | IG > GBP > LRP > KSHAP > GCAM > rand | 0.60 |

### Summary

- Identical ranking of all six methods: 2/12 (patch type × model × metric) slices.
- Same top-ranked method: 5/12.
- Kendall τ between the two rankings: min 0.20, median 0.73.
- Slices whose ranking changed: obvious/cnn/topk (τ = 0.60; moved: Integrated Gradients, Grad-CAM, KernelSHAP, Guided Backprop); obvious/vit/topk (τ = 0.87; moved: Integrated Gradients, Grad-CAM); subtle/cnn/topk (τ = 0.73; moved: Grad-CAM, KernelSHAP, Guided Backprop); subtle/vit/topk (τ = 0.73; moved: Integrated Gradients, KernelSHAP, Guided Backprop); obvious/cnn/auc (τ = 0.20; moved: Grad-CAM, LRP, KernelSHAP, Guided Backprop); obvious/vit/auc (τ = 0.73; moved: LRP, Guided Backprop, random baseline); subtle/cnn/auc (τ = 0.20; moved: Integrated Gradients, Grad-CAM, LRP, KernelSHAP, Guided Backprop); subtle/vit/auc (τ = 0.33; moved: Grad-CAM, LRP, KernelSHAP, Guided Backprop); subtle/cnn/pointing (τ = 0.73; moved: Integrated Gradients, LRP, KernelSHAP, Guided Backprop); subtle/vit/pointing (τ = 0.60; moved: Integrated Gradients, LRP).
