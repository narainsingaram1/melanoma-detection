"""Aggregate results/ablation/*/seed*/metrics.json into summary.md + summary.json."""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

KEYS = [('accuracy', 'Accuracy'), ('balanced_accuracy', 'Balanced acc'), ('macro_f1', 'Macro F1'),
        ('macro_auroc', 'Macro AUROC'), ('mel_auroc', 'Mel AUROC'), ('mel_recall', 'Mel recall'),
        ('mel_sens_at_90spec', 'Mel sens @90% spec')]
ORDER = ['baseline_cnn', 'plain', 'weighted_loss', 'oversample', 'dcgan']
REFERENCE = 'weighted_loss'  # dcgan = weighted_loss + synthetic data, so this isolates the GAN


def _fmt(vals):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{vals.mean():.3f} ± {vals.std(ddof=1):.3f}"


def summarize(out_dir):
    runs = defaultdict(dict)
    for path in glob.glob(os.path.join(out_dir, '*', 'seed*', 'metrics.json')):
        with open(path) as f:
            r = json.load(f)
        runs[r['variant']][r['seed']] = r
    if not runs:
        print(f"no runs found in {out_dir}")
        return None

    variants = [v for v in ORDER if v in runs] + sorted(set(runs) - set(ORDER))
    summary = {'variants': {}, 'paired_delta_vs_' + REFERENCE: {}}
    lines = ['# Ablation results (lesion-level split, test set)', '',
             'Mean ± sample std across training seeds; split seed fixed at 42.', '',
             '| Variant | Seeds | ' + ' | '.join(n for _, n in KEYS) + ' |',
             '|---|---|' + '---|' * len(KEYS)]
    for v in variants:
        seeds = sorted(runs[v])
        vals = {k: [runs[v][s]['test'][k] for s in seeds] for k, _ in KEYS}
        summary['variants'][v] = {'seeds': seeds, **{k: {'mean': float(np.mean(x)),
                                  'std': float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
                                  'values': x} for k, x in vals.items()}}
        lines.append(f"| {v} | {len(seeds)} | " + ' | '.join(_fmt(vals[k]) for k, _ in KEYS) + ' |')

    if REFERENCE in runs:
        lines += ['', f'## Paired difference vs `{REFERENCE}` (same seed)', '',
                  '| Variant | Seeds | ' + ' | '.join(n for _, n in KEYS) + ' |',
                  '|---|---|' + '---|' * len(KEYS)]
        for v in variants:
            if v == REFERENCE:
                continue
            common = sorted(set(runs[v]) & set(runs[REFERENCE]))
            if not common:
                continue
            d = {k: [runs[v][s]['test'][k] - runs[REFERENCE][s]['test'][k] for s in common] for k, _ in KEYS}
            summary['paired_delta_vs_' + REFERENCE][v] = {k: {'mean': float(np.mean(x)), 'values': x}
                                                          for k, x in d.items()}
            lines.append(f"| {v} | {len(common)} | " +
                         ' | '.join(f"{np.mean(d[k]):+.3f}" for k, _ in KEYS) + ' |')

    lines += ['', '## Per-run 95% CI (lesion-level bootstrap)', '',
              '| Variant | Seed | Balanced acc | Macro AUROC | Mel sens @90% spec |', '|---|---|---|---|---|']
    for v in variants:
        for s in sorted(runs[v]):
            t, ci = runs[v][s]['test'], runs[v][s]['test_ci95']
            cell = lambda k: f"{t[k]:.3f} [{ci[k]['lo']:.3f}, {ci[k]['hi']:.3f}]"  # noqa: E731
            lines.append(f"| {v} | {s} | {cell('balanced_accuracy')} | {cell('macro_auroc')} | "
                         f"{cell('mel_sens_at_90spec')} |")

    gan_rows = [(s, runs['dcgan'][s]['gan']) for s in sorted(runs.get('dcgan', {})) if runs['dcgan'][s].get('gan')]
    if gan_rows and gan_rows[0][1].get('per_class'):
        lines += ['', '## Synthetic image quality (Inception-v3 features, lower is better)', '',
                  'The real-vs-real row (val vs train images of the same class) is the floor a perfect generator would reach.', '',
                  '| Seed | Class | n real train | FID syn | FID real-ref | KID syn | KID real-ref |',
                  '|---|---|---|---|---|---|---|']
        for s, g in gan_rows:
            for c, m in g['per_class'].items():
                lines.append(f"| {s} | {c} | {m['n_real_train']} | {m['fid_synthetic_vs_train']:.1f} | "
                             f"{m['fid_real_val_vs_train']:.1f} | {m['kid_synthetic_vs_train']['mean']:.4f} | "
                             f"{m['kid_real_val_vs_train']['mean']:.4f} |")

    md = '\n'.join(lines) + '\n'
    with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
        f.write(md)
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('\n' + md)
    return summary


if __name__ == '__main__':
    summarize(sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'ablation'))
