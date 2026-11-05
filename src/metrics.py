"""
Evaluation metrics for imbalanced multi-class lesion classification.

All metrics are computed from predicted probabilities so threshold-free metrics
(AUROC, sensitivity at fixed specificity) are available alongside argmax metrics.
Confidence intervals use a *lesion-level* (cluster) bootstrap: a test lesion with
several photos is resampled as one unit, since its images are not independent.
"""
from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, classification_report, f1_score,
                             roc_auc_score, roc_curve)

from src.utils import CLASSES

MEL_IDX = CLASSES.index('mel')
CI_KEYS = ('accuracy', 'balanced_accuracy', 'macro_f1', 'macro_auroc',
           'mel_auroc', 'mel_recall', 'mel_sens_at_90spec')


def sensitivity_at_specificity(y_bin: np.ndarray, score: np.ndarray, spec: float = 0.90) -> float:
    """Highest TPR achievable with FPR <= 1 - spec (one-vs-rest)."""
    fpr, tpr, _ = roc_curve(y_bin, score)
    ok = fpr <= (1 - spec) + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def _core(y: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    pred = probs.argmax(1)
    labels = list(range(len(CLASSES)))
    out = {
        'accuracy': float((pred == y).mean()),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, labels=labels, average='macro', zero_division=0)),
        'mel_recall': float(((pred == MEL_IDX) & (y == MEL_IDX)).sum() / max((y == MEL_IDX).sum(), 1)),
    }
    present = [c for c in labels if (y == c).any() and (y != c).any()]
    aurocs = {c: roc_auc_score((y == c).astype(int), probs[:, c]) for c in present}
    out['macro_auroc'] = float(np.mean(list(aurocs.values()))) if aurocs else float('nan')
    out['mel_auroc'] = float(aurocs.get(MEL_IDX, float('nan')))
    out['mel_sens_at_90spec'] = (sensitivity_at_specificity((y == MEL_IDX).astype(int), probs[:, MEL_IDX])
                                 if MEL_IDX in present else float('nan'))
    out['_per_class_auroc'] = {CLASSES[c]: float(v) for c, v in aurocs.items()}
    return out


def compute_metrics(y_true: Sequence[int], probs: np.ndarray) -> Dict:
    y = np.asarray(y_true)
    probs = np.asarray(probs, dtype=np.float64)
    core = _core(y, probs)
    report = classification_report(y, probs.argmax(1), labels=list(range(len(CLASSES))),
                                   target_names=CLASSES, output_dict=True, zero_division=0)
    per_class = {c: {k: float(report[c][k]) for k in ('precision', 'recall', 'f1-score', 'support')}
                 for c in CLASSES}
    for c, v in core.pop('_per_class_auroc').items():
        per_class[c]['auroc'] = v
    return {**core, 'per_class': per_class}


def bootstrap_ci(y_true: Sequence[int], probs: np.ndarray, groups: Optional[Sequence] = None,
                 n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Percentile bootstrap CIs; resamples whole groups (lesions) when `groups` is given."""
    y = np.asarray(y_true)
    probs = np.asarray(probs, dtype=np.float64)
    rng = np.random.default_rng(seed)

    if groups is None:
        groups = np.arange(len(y))
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]

    samples = {k: [] for k in CI_KEYS}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        m = _core(y[idx], probs[idx])
        for k in CI_KEYS:
            samples[k].append(m[k])

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: {'lo': float(np.nanpercentile(v, lo)), 'hi': float(np.nanpercentile(v, hi))}
            for k, v in samples.items()}
