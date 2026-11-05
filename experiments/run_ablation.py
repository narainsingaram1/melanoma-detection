"""
Controlled ablation: how should we handle class imbalance on HAM10000?

Every EfficientNet-B0 variant uses the identical recipe, the identical lesion-level
split (split seed fixed at 42), and the identical model-selection rule; only the
imbalance strategy changes. Each variant runs for several training seeds.

  plain          unweighted cross-entropy + standard augmentation
  weighted_loss  inverse-frequency class-weighted cross-entropy
  oversample     WeightedRandomSampler (class-balanced batches), unweighted CE
  dcgan          weighted_loss + conditional-DCGAN synthetic mel/df images
  baseline_cnn   custom 4-block CNN with weighted CE (reference point)

Usage:
  python -m experiments.run_ablation                       # full run
  python -m experiments.run_ablation --variants dcgan --seeds 0
  python -m experiments.run_ablation --smoke               # tiny end-to-end check
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.baseline_cnn import SkinLesionCNN  # noqa: E402
from src.data_cache import (CachedImageDataset, SyntheticImageDataset,  # noqa: E402
                            build_image_cache, get_tensor_transforms)
from src.dataset import get_class_weights, prepare_data  # noqa: E402
from src.evaluate import plot_confusion_matrix  # noqa: E402
from src.metrics import bootstrap_ci, compute_metrics  # noqa: E402
from src.synthetic import gan_real_transform, generate_uint8, train_dcgan  # noqa: E402
from src.transfer_model import SkinLesionTransfer  # noqa: E402
from src.utils import CLASSES, RARE_CLASSES, get_device, set_seed  # noqa: E402

VARIANTS = {
    'plain':         dict(model='effnet', weighted=False, oversample=False, synthetic=False),
    'weighted_loss': dict(model='effnet', weighted=True,  oversample=False, synthetic=False),
    'oversample':    dict(model='effnet', weighted=False, oversample=True,  synthetic=False),
    'dcgan':         dict(model='effnet', weighted=True,  oversample=False, synthetic=True),
    'baseline_cnn':  dict(model='cnn',    weighted=True,  oversample=False, synthetic=False),
}
SPLIT_SEED = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--variants', nargs='+', default=['plain', 'weighted_loss', 'oversample', 'dcgan'],
                   choices=list(VARIANTS))
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--data-dir', default=os.path.join(ROOT, 'data', 'ham10000'))
    p.add_argument('--cache-dir', default=os.path.join(ROOT, 'data', 'cache'))
    p.add_argument('--out', default=os.path.join(ROOT, 'results', 'ablation'))
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--head-epochs', type=int, default=3, help='stage 1: frozen backbone (except last blocks)')
    p.add_argument('--ft-epochs', type=int, default=15, help='stage 2: full fine-tuning')
    p.add_argument('--cnn-epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--gan-epochs', type=int, default=30)
    p.add_argument('--synthetic-per-class', type=int, default=1500)
    p.add_argument('--n-boot', type=int, default=1000)
    p.add_argument('--skip-fid', action='store_true')
    p.add_argument('--overwrite', action='store_true', help='re-run finished (variant, seed) pairs')
    p.add_argument('--smoke', action='store_true', help='tiny subset, 1 epoch each, for pipeline checks')
    return p.parse_args()


def find_metadata(data_dir):
    hits = glob.glob(os.path.join(data_dir, '**', 'HAM10000_metadata.csv'), recursive=True)
    if not hits:
        raise FileNotFoundError(f"HAM10000_metadata.csv not found under {data_dir}. Run `python download_data.py`.")
    return hits[0]


def subsample(df, per_class, seed):
    return (df.sample(frac=1.0, random_state=seed).groupby('label_idx').head(per_class)
              .reset_index(drop=True))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), 1).cpu())
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def run_stage(model, params, lr, epochs, train_loader, val_loader, criterion, device,
              patience, ckpt, state, tag):
    """One training stage; selects checkpoints on validation balanced accuracy."""
    optimizer = optim.Adam(params, lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    bad = 0
    eval_ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        t0, loss_sum, n = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            n += len(y)

        probs, ys = predict(model, val_loader, device)
        val = compute_metrics(ys, probs)
        val_ce = float(eval_ce(torch.log(torch.from_numpy(probs).clamp_min(1e-8)), torch.from_numpy(ys)))
        scheduler.step(val['balanced_accuracy'])
        state['history'].append({'stage': tag, 'train_loss': loss_sum / n, 'val_ce': val_ce,
                                 'val_balanced_accuracy': val['balanced_accuracy'],
                                 'val_macro_auroc': val['macro_auroc'], 'sec': time.time() - t0})
        h = state['history'][-1]
        print(f"  [{tag}] ep {len(state['history'])}: train_loss={h['train_loss']:.4f} "
              f"val_ce={val_ce:.4f} val_bal_acc={h['val_balanced_accuracy']:.4f} "
              f"val_auroc={h['val_macro_auroc']:.4f} ({h['sec']:.0f}s)", flush=True)

        if val['balanced_accuracy'] > state['best']:
            state['best'], state['best_epoch'], bad = val['balanced_accuracy'], len(state['history']), 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop ({tag})", flush=True)
                break


def gan_stage(args, seed, train_df, val_df, cache, id_to_row, device, run_dir):
    """Train (or load) a per-seed DCGAN on train-split rare classes; return synthetic dataset + report."""
    rare_idx = [CLASSES.index(c) for c in RARE_CLASSES]
    rare_df = train_df[train_df['label_idx'].isin(rare_idx)]
    g_path = os.path.join(run_dir, 'generator.pth')

    t0 = time.time()
    if os.path.exists(g_path) and not args.overwrite:
        from src.dcgan import Generator
        G = Generator(z_dim=100, num_classes=7).to(device)
        G.load_state_dict(torch.load(g_path, map_location=device, weights_only=True))
        gan_hist = None
        print(f"  loaded generator from {g_path}")
    else:
        real = CachedImageDataset(rare_df, cache, id_to_row, transform=gan_real_transform())
        print(f"  training DCGAN on {len(real)} real rare-class images ({RARE_CLASSES})", flush=True)
        G, gan_hist = train_dcgan(real, device, epochs=args.gan_epochs, sample_classes=rare_idx,
                                  log_dir=os.path.join(run_dir, 'gan_samples'), seed=seed)
        torch.save(G.state_dict(), g_path)
    gan_sec = time.time() - t0

    torch.manual_seed(seed)
    imgs, labels = [], []
    for c in rare_idx:
        imgs.append(generate_uint8(G, c, args.synthetic_per_class, device))
        labels += [c] * args.synthetic_per_class
    imgs = torch.cat(imgs)

    report = {'gan_train_sec': gan_sec, 'n_real_rare_train': int(len(rare_df)),
              'n_synthetic': int(len(imgs)), 'history': gan_hist, 'per_class': {}}
    if not args.skip_fid:
        from src.fid import InceptionFeatures, fid_from_features, kid_from_features
        feat = InceptionFeatures(device)

        def real_uint8(df):
            rows = [id_to_row[i] for i in df['image_id']]
            return torch.from_numpy(np.array(cache[rows])).permute(0, 3, 1, 2)

        for c in rare_idx:
            name = CLASSES[c]
            f_tr = feat(real_uint8(train_df[train_df['label_idx'] == c]))
            f_va = feat(real_uint8(val_df[val_df['label_idx'] == c]))
            f_syn = feat(imgs[np.array(labels) == c])
            report['per_class'][name] = {
                'n_real_train': len(f_tr), 'n_real_val': len(f_va), 'n_synthetic': len(f_syn),
                'fid_synthetic_vs_train': fid_from_features(f_syn, f_tr),
                'kid_synthetic_vs_train': kid_from_features(f_syn, f_tr),
                # reference floor: two disjoint sets of *real* images of the same class
                'fid_real_val_vs_train': fid_from_features(f_va, f_tr),
                'kid_real_val_vs_train': kid_from_features(f_va, f_tr),
            }
            pc = report['per_class'][name]
            print(f"  {name}: FID syn={pc['fid_synthetic_vs_train']:.1f} (real-ref {pc['fid_real_val_vs_train']:.1f}) | "
                  f"KID syn={pc['kid_synthetic_vs_train']['mean']:.4f} (real-ref {pc['kid_real_val_vs_train']['mean']:.4f})",
                  flush=True)
        del feat
    with open(os.path.join(run_dir, 'gan_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    return SyntheticImageDataset(imgs, labels, transform=get_tensor_transforms()[0]), report


def run_one(args, variant, seed, splits, cache, id_to_row, device):
    cfg = VARIANTS[variant]
    run_dir = os.path.join(args.out, variant, f'seed{seed}')
    out_json = os.path.join(run_dir, 'metrics.json')
    if os.path.exists(out_json) and not args.overwrite:
        print(f"skip {variant}/seed{seed} (done)")
        return
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n{'=' * 70}\n{variant} | seed {seed} | {device}\n{'=' * 70}", flush=True)
    set_seed(seed)
    t_start = time.time()

    train_df, val_df, test_df = splits
    train_tf, eval_tf = get_tensor_transforms()
    train_ds = CachedImageDataset(train_df, cache, id_to_row, transform=train_tf)
    gan_report = None
    if cfg['synthetic']:
        syn_ds, gan_report = gan_stage(args, seed, train_df, val_df, cache, id_to_row, device, run_dir)
        train_ds = ConcatDataset([train_ds, syn_ds])

    g = torch.Generator().manual_seed(seed)
    if cfg['oversample']:
        counts = np.bincount(train_df['label_idx'], minlength=len(CLASSES))
        w = 1.0 / counts[train_df['label_idx'].to_numpy()]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w),
                                        replacement=True, generator=g)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=g)
    val_loader = DataLoader(CachedImageDataset(val_df, cache, id_to_row, eval_tf), batch_size=64)
    test_loader = DataLoader(CachedImageDataset(test_df, cache, id_to_row, eval_tf), batch_size=64)

    # Class weights always come from the *real* training distribution.
    weight = get_class_weights(train_df).to(device) if cfg['weighted'] else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    ckpt = os.path.join(run_dir, 'best.pth')
    state = {'history': [], 'best': -1.0, 'best_epoch': None}

    if cfg['model'] == 'effnet':
        model = SkinLesionTransfer(num_classes=7).to(device)
        run_stage(model, [p for p in model.parameters() if p.requires_grad], 1e-3, args.head_epochs,
                  train_loader, val_loader, criterion, device, args.patience, ckpt, state, 'head')
        model.unfreeze_all()
        run_stage(model, model.parameters(), 1e-4, args.ft_epochs,
                  train_loader, val_loader, criterion, device, args.patience, ckpt, state, 'finetune')
    else:
        model = SkinLesionCNN(num_classes=7).to(device)
        run_stage(model, model.parameters(), 1e-3, args.cnn_epochs,
                  train_loader, val_loader, criterion, device, args.patience, ckpt, state, 'cnn')

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    probs, ys = predict(model, test_loader, device)
    test = compute_metrics(ys, probs)
    ci = bootstrap_ci(ys, probs, groups=test_df['lesion_id'].to_numpy(), n_boot=args.n_boot, seed=seed)

    np.savez_compressed(os.path.join(run_dir, 'test_predictions.npz'), probs=probs, y_true=ys,
                        image_id=test_df['image_id'].to_numpy(), lesion_id=test_df['lesion_id'].to_numpy())
    plot_confusion_matrix(ys, probs.argmax(1), os.path.join(run_dir, 'confusion_matrix.png'),
                          title=f'{variant} (seed {seed}) - test')

    result = {
        'variant': variant, 'seed': seed, 'split_seed': SPLIT_SEED, 'device': str(device),
        'config': {**cfg, **{k: v for k, v in vars(args).items() if k not in ('variants', 'seeds')}},
        'n_train_real': len(train_df), 'n_val': len(val_df), 'n_test': len(test_df),
        'n_test_lesions': int(test_df['lesion_id'].nunique()),
        'best_epoch': state['best_epoch'], 'best_val_balanced_accuracy': state['best'],
        'test': test, 'test_ci95': ci, 'history': state['history'],
        'gan': {k: v for k, v in (gan_report or {}).items() if k != 'history'} or None,
        'total_sec': time.time() - t_start,
    }
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  TEST: acc={test['accuracy']:.4f} bal_acc={test['balanced_accuracy']:.4f} "
          f"macroF1={test['macro_f1']:.4f} AUROC={test['macro_auroc']:.4f} "
          f"mel_sens@90spec={test['mel_sens_at_90spec']:.4f} ({result['total_sec'] / 60:.1f} min)", flush=True)


def main():
    args = parse_args()
    if args.smoke:
        args.head_epochs, args.ft_epochs, args.cnn_epochs, args.gan_epochs = 1, 1, 1, 1
        args.synthetic_per_class, args.n_boot = 32, 50
        args.out = os.path.join(args.out + '_smoke')

    device = get_device()
    meta = find_metadata(args.data_dir)
    cache, id_to_row = build_image_cache(meta, os.path.dirname(meta), args.cache_dir)
    splits = prepare_data(meta, seed=SPLIT_SEED)
    if args.smoke:
        splits = tuple(subsample(s, 40 if i == 0 else 15, SPLIT_SEED) for i, s in enumerate(splits))

    os.makedirs(args.out, exist_ok=True)
    stats = {name: {'images': len(s), 'lesions': int(s['lesion_id'].nunique()),
                    'per_class_images': {CLASSES[k]: int(v) for k, v in s['label_idx'].value_counts().sort_index().items()}}
             for name, s in zip(('train', 'val', 'test'), splits)}
    with open(os.path.join(args.out, 'split_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    for variant in args.variants:
        for seed in args.seeds:
            run_one(args, variant, seed, splits, cache, id_to_row, device)

    from experiments.summarize import summarize
    summarize(args.out)


if __name__ == '__main__':
    main()
