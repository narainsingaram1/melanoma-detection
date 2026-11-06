# Ablation results (lesion-level split, test set)

Mean ± sample std across training seeds; split seed fixed at 42.

| Variant | Seeds | Accuracy | Balanced acc | Macro F1 | Macro AUROC | Mel AUROC | Mel recall | Mel sens @90% spec |
|---|---|---|---|---|---|---|---|---|
| baseline_cnn | 1 | 0.493 | 0.336 | 0.282 | 0.813 | 0.799 | 0.713 | 0.383 |
| plain | 1 | 0.826 | 0.710 | 0.705 | 0.961 | 0.912 | 0.533 | 0.701 |
| weighted_loss | 1 | 0.767 | 0.754 | 0.668 | 0.955 | 0.892 | 0.683 | 0.677 |
| oversample | 1 | 0.813 | 0.723 | 0.708 | 0.959 | 0.900 | 0.563 | 0.665 |
| dcgan | 1 | 0.768 | 0.804 | 0.695 | 0.956 | 0.886 | 0.629 | 0.653 |

## Paired difference vs `weighted_loss` (same seed)

| Variant | Seeds | Accuracy | Balanced acc | Macro F1 | Macro AUROC | Mel AUROC | Mel recall | Mel sens @90% spec |
|---|---|---|---|---|---|---|---|---|
| baseline_cnn | 1 | -0.274 | -0.418 | -0.385 | -0.142 | -0.094 | +0.030 | -0.293 |
| plain | 1 | +0.059 | -0.044 | +0.037 | +0.006 | +0.020 | -0.150 | +0.024 |
| oversample | 1 | +0.046 | -0.031 | +0.040 | +0.004 | +0.007 | -0.120 | -0.012 |
| dcgan | 1 | +0.001 | +0.050 | +0.027 | +0.001 | -0.006 | -0.054 | -0.024 |

## Per-run 95% CI (lesion-level bootstrap)

| Variant | Seed | Balanced acc | Macro AUROC | Mel sens @90% spec |
|---|---|---|---|---|
| baseline_cnn | 0 | 0.336 [0.299, 0.380] | 0.813 [0.788, 0.837] | 0.383 [0.278, 0.547] |
| plain | 0 | 0.710 [0.657, 0.766] | 0.961 [0.952, 0.969] | 0.701 [0.619, 0.813] |
| weighted_loss | 0 | 0.754 [0.703, 0.803] | 0.955 [0.946, 0.964] | 0.677 [0.585, 0.755] |
| oversample | 0 | 0.723 [0.668, 0.775] | 0.959 [0.951, 0.968] | 0.665 [0.580, 0.760] |
| dcgan | 0 | 0.804 [0.764, 0.842] | 0.956 [0.947, 0.963] | 0.653 [0.558, 0.740] |

## Synthetic image quality (Inception-v3 features, lower is better)

The real-vs-real row (val vs train images of the same class) is the floor a perfect generator would reach.

| Seed | Class | n real train | FID syn | FID real-ref | KID syn | KID real-ref |
|---|---|---|---|---|---|---|
| 0 | mel | 773 | 349.0 | 68.1 | 0.3608 | 0.0006 |
| 0 | df | 71 | 369.6 | 152.8 | 0.3427 | 0.0120 |
