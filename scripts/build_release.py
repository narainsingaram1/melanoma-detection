"""
Build deployable artifacts from a trained run and benchmark them.

  python -m scripts.build_release --run results/ablation/plain/seed0 [--ood-dir DIR]

Steps (all on CPU):
  1. Collect logits + pooled features for val / test / a train subset.
  2. Temperature scaling on val; expected calibration error (ECE) on test before/after.
  3. Mahalanobis OOD detector: class means + Ledoit-Wolf shared covariance on train
     features; threshold = 99th percentile of val scores (~1% false rejections).
     Evaluated on test lesions vs. synthetic non-images and optional real photos.
  4. Export ONNX (outputs: logits, features) and INT8 static quantization (QDQ,
     per-channel, calibrated on train images).
  5. Compare PyTorch FP32 / ONNX FP32 / ONNX INT8 on the test set (accuracy, AUROC,
     top-1 agreement) plus batch-1 CPU latency and file size.
Writes models/ (gitignored binaries + model_info.json) and results/release_report.json.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
from PIL import Image
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.run_ablation import SPLIT_SEED, find_metadata  # noqa: E402
from src.data_cache import build_image_cache  # noqa: E402
from src.dataset import prepare_data  # noqa: E402
from src.inference import MEAN, STD, OODScorer, preprocess, softmax  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.transfer_model import SkinLesionTransfer  # noqa: E402
from src.utils import CLASSES  # noqa: E402


class WithFeatures(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.forward_with_features(x)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, help='dir with best.pth + metrics.json')
    p.add_argument('--data-dir', default=os.path.join(ROOT, 'data', 'ham10000'))
    p.add_argument('--cache-dir', default=os.path.join(ROOT, 'data', 'cache'))
    p.add_argument('--out', default=os.path.join(ROOT, 'models'))
    p.add_argument('--ood-dir', default=None, help='folder of non-lesion photos for OOD evaluation')
    p.add_argument('--train-fit', type=int, default=3000, help='train images used to fit OOD stats')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--int8-max-auroc-drop', type=float, default=0.01)
    p.add_argument('--ood-percentile', type=float, default=97.5,
                   help='OOD threshold = this percentile of validation-lesion scores')
    return p.parse_args()


def to_input(cache, id_to_row, df):
    rows = [id_to_row[i] for i in df['image_id']]
    x = np.array(cache[rows], dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
    return ((x - MEAN) / STD).astype(np.float32)


@torch.no_grad()
def torch_forward(model, x, bs=64):
    logits, feats = [], []
    for i in range(0, len(x), bs):
        lg, ft = model.forward_with_features(torch.from_numpy(x[i:i + bs]))
        logits.append(lg.numpy()); feats.append(ft.numpy())
    return np.concatenate(logits), np.concatenate(feats)


def ort_forward(sess, x, bs=64):
    name = sess.get_inputs()[0].name
    outs = [sess.run(None, {name: x[i:i + bs]}) for i in range(0, len(x), bs)]
    return np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs])


def fit_temperature(logits, y):
    lg, yt = torch.from_numpy(logits).double(), torch.from_numpy(y)
    log_t = torch.zeros(1, dtype=torch.double, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), yt)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.detach().exp())


def ece(probs, y, n_bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def synthetic_ood(n_each=40, seed=0):
    rng = np.random.default_rng(seed)
    imgs = {'uniform_noise': [], 'solid_color': [], 'gradient': [], 'checkerboard': []}
    for _ in range(n_each):
        imgs['uniform_noise'].append(Image.fromarray(rng.integers(0, 256, (450, 600, 3), dtype=np.uint8)))
        imgs['solid_color'].append(Image.new('RGB', (600, 450), tuple(int(v) for v in rng.integers(0, 256, 3))))
        a, b = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
        t = np.linspace(0, 1, 600)[None, :, None]
        imgs['gradient'].append(Image.fromarray(np.broadcast_to(a + (b - a) * t, (450, 600, 3)).astype(np.uint8)))
        k = int(rng.integers(10, 80))
        yy, xx = np.mgrid[:450, :600]
        board = (((yy // k) + (xx // k)) % 2 * 255).astype(np.uint8)
        imgs['checkerboard'].append(Image.fromarray(np.stack([board] * 3, -1)))
    return imgs


def latency(fn, x1, n=200, warmup=20):
    for _ in range(warmup):
        fn(x1)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(x1)
        ts.append((time.perf_counter() - t0) * 1000)
    return {'p50_ms': float(np.percentile(ts, 50)), 'p95_ms': float(np.percentile(ts, 95)),
            'mean_ms': float(np.mean(ts))}


def mb(path):
    return round(os.path.getsize(path) / 1e6, 2)


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    with open(os.path.join(args.run, 'metrics.json')) as f:
        run_info = json.load(f)
    meta = find_metadata(args.data_dir)
    cache, id_to_row = build_image_cache(meta, os.path.dirname(meta), args.cache_dir)
    train_df, val_df, test_df = prepare_data(meta, seed=SPLIT_SEED)
    fit_df = train_df.sample(min(args.train_fit, len(train_df)), random_state=0)

    model = SkinLesionTransfer(num_classes=len(CLASSES), pretrained=False)
    model.load_state_dict(torch.load(os.path.join(args.run, 'best.pth'), map_location='cpu', weights_only=True))
    model.eval()

    print('collecting features (CPU)...', flush=True)
    x_val, x_test, x_fit = (to_input(cache, id_to_row, d) for d in (val_df, test_df, fit_df))
    y_val, y_test, y_fit = (d['label_idx'].to_numpy() for d in (val_df, test_df, fit_df))
    lg_val, ft_val = torch_forward(model, x_val)
    lg_test, ft_test = torch_forward(model, x_test)
    _, ft_fit = torch_forward(model, x_fit)

    # --- calibration ---
    T = fit_temperature(lg_val, y_val)
    p_raw, p_cal = softmax(lg_test), softmax(lg_test / T)
    calibration = {'temperature': T, 'test_ece_before': ece(p_raw, y_test), 'test_ece_after': ece(p_cal, y_test),
                   'test_nll_before': float(-np.log(p_raw[np.arange(len(y_test)), y_test] + 1e-12).mean()),
                   'test_nll_after': float(-np.log(p_cal[np.arange(len(y_test)), y_test] + 1e-12).mean())}
    print(f"temperature={T:.3f} ECE {calibration['test_ece_before']:.4f} -> {calibration['test_ece_after']:.4f}")

    # --- OOD detector ---
    means = np.stack([ft_fit[y_fit == c].mean(0) for c in range(len(CLASSES))])
    lw = LedoitWolf().fit(ft_fit - means[y_fit])
    scorer = OODScorer(means, lw.precision_)
    s_val, s_test = scorer(ft_val), scorer(ft_test)
    threshold = float(np.percentile(s_val, args.ood_percentile))
    ood_eval = {'threshold': threshold, 'threshold_rule': f'{args.ood_percentile}th percentile of validation lesions',
                'test_lesion_false_rejection_rate': float((s_test > threshold).mean()), 'sets': {}, 'sweep': []}
    ood_sets = synthetic_ood()
    if args.ood_dir:
        files = sorted(glob.glob(os.path.join(args.ood_dir, '*.jp*g')) + glob.glob(os.path.join(args.ood_dir, '*.png')))
        ood_sets['natural_photos'] = [Image.open(f) for f in files]
    all_ood, set_scores = [], {}
    for name, imgs in ood_sets.items():
        x = np.concatenate([preprocess(im) for im in imgs])
        _, ft = torch_forward(model, x)
        s = scorer(ft)
        all_ood.append(s)
        set_scores[name] = s
        ood_eval['sets'][name] = {'n': len(imgs), 'rejection_rate': float((s > threshold).mean())}
        print(f"OOD {name}: rejected {ood_eval['sets'][name]['rejection_rate']:.1%} of {len(imgs)}")
    for pct in (90, 95, 97.5, 99, 99.5):
        th = float(np.percentile(s_val, pct))
        row = {'percentile': pct, 'threshold': th, 'test_false_rejection': float((s_test > th).mean()),
               **{f'reject_{k}': float((v > th).mean()) for k, v in set_scores.items()}}
        ood_eval['sweep'].append(row)
        print('  sweep ' + ' '.join(f'{k}={v:.3f}' for k, v in row.items()))
    all_ood = np.concatenate(all_ood)
    ood_eval['auroc_test_lesions_vs_all_ood'] = float(roc_auc_score(
        np.r_[np.zeros(len(s_test)), np.ones(len(all_ood))], np.r_[s_test, all_ood]))
    ood_eval['max_softmax_baseline_auroc'] = float(roc_auc_score(
        np.r_[np.zeros(len(s_test)), np.ones(len(all_ood))],
        -np.r_[p_cal.max(1), np.concatenate([softmax(torch_forward(model, np.concatenate(
            [preprocess(im) for im in imgs]))[0] / T).max(1) for imgs in ood_sets.values()])]))
    print(f"test false-rejection {ood_eval['test_lesion_false_rejection_rate']:.1%} | "
          f"OOD AUROC mahalanobis={ood_eval['auroc_test_lesions_vs_all_ood']:.3f} "
          f"vs max-softmax={ood_eval['max_softmax_baseline_auroc']:.3f}")
    np.savez_compressed(os.path.join(args.out, 'ood_stats.npz'), means=means.astype(np.float32),
                        precision=lw.precision_.astype(np.float32))

    # --- ONNX export + INT8 ---
    pth_path = os.path.join(args.out, 'classifier.pth')
    shutil.copy(os.path.join(args.run, 'best.pth'), pth_path)
    fp32 = os.path.join(args.out, 'classifier_fp32.onnx')
    torch.onnx.export(WithFeatures(model), torch.from_numpy(x_val[:1]), fp32, dynamo=False,
                      input_names=['image'], output_names=['logits', 'features'],
                      dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}, 'features': {0: 'batch'}},
                      opset_version=17)
    pre = os.path.join(args.out, '_pre.onnx')
    quant_pre_process(fp32, pre)

    class Reader(CalibrationDataReader):
        def __init__(self, x):
            self.it = iter([{'image': x[i:i + 1]} for i in range(len(x))])

        def get_next(self):
            return next(self.it, None)

    int8 = os.path.join(args.out, 'classifier_int8.onnx')
    # Best of the configs tried: quantizing only Conv/Gemm (leaving SiLU / squeeze-excite in
    # FP32) with percentile calibration. Full-graph min-max INT8 lost ~10 AUROC points.
    quantize_static(pre, int8, Reader(x_fit[:300]), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    op_types_to_quantize=['Conv', 'Gemm'], calibrate_method=CalibrationMethod.Percentile)
    os.remove(pre)

    # --- compare engines ---
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    sessions = {n: ort.InferenceSession(p, so, providers=['CPUExecutionProvider'])
                for n, p in (('onnx_fp32', fp32), ('onnx_int8', int8))}
    torch_pred = lg_test.argmax(1)
    engines = {'pytorch_fp32': {'size_mb': mb(pth_path), 'test': compute_metrics(y_test, p_cal)}}
    # PyTorch CPU gets *slower* with more intra-op threads on macOS at batch 1, so report its
    # best thread count (fair baseline for the ONNX Runtime comparison).
    best = None
    for th in sorted({1, args.threads}):
        torch.set_num_threads(th)
        with torch.inference_mode():
            lat = latency(lambda x: model(torch.from_numpy(x)), x_test[:1])
        if best is None or lat['p50_ms'] < best['p50_ms']:
            best = {**lat, 'threads': th}
    engines['pytorch_fp32']['latency_batch1'] = best
    torch.set_num_threads(args.threads)
    for n, sess in sessions.items():
        lg, _ = ort_forward(sess, x_test)
        engines[n] = {'size_mb': mb(fp32 if n == 'onnx_fp32' else int8),
                      'test': compute_metrics(y_test, softmax(lg / T)),
                      'top1_agreement_with_pytorch': float((lg.argmax(1) == torch_pred).mean()),
                      'latency_batch1': latency(lambda x, s=sess: s.run(None, {'image': x}), x_test[:1])}
    for n, e in engines.items():
        e['test'].pop('per_class', None)
        print(f"{n:13s} size={e['size_mb']:6.1f}MB acc={e['test']['accuracy']:.4f} "
              f"AUROC={e['test']['macro_auroc']:.4f} p50={e['latency_batch1']['p50_ms']:.1f}ms "
              f"p95={e['latency_batch1']['p95_ms']:.1f}ms")

    drop = engines['onnx_fp32']['test']['macro_auroc'] - engines['onnx_int8']['test']['macro_auroc']
    serving = 'classifier_int8.onnx' if drop <= args.int8_max_auroc_drop else 'classifier_fp32.onnx'

    info = {
        'source_run': os.path.relpath(args.run, ROOT), 'variant': run_info['variant'], 'seed': run_info['seed'],
        'architecture': 'EfficientNet-B0', 'input': {'size': [224, 224], 'resize': 'bilinear', 'mean': MEAN.ravel().tolist(),
                                                     'std': STD.ravel().tolist()},
        'classes': CLASSES, 'test_metrics_uncalibrated': {k: v for k, v in run_info['test'].items() if k != 'per_class'},
        'calibration': calibration, 'ood': ood_eval,
        'serving': {'onnx_file': serving, 'int8_macro_auroc_drop': float(drop),
                    'rule': f'INT8 if macro-AUROC drop <= {args.int8_max_auroc_drop}'},
        'engines': engines, 'threads': args.threads,
        'note': 'Latency measured on the build machine CPU; re-run when idle for clean numbers.',
    }
    with open(os.path.join(args.out, 'model_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    shutil.copy(os.path.join(args.out, 'model_info.json'), os.path.join(ROOT, 'results', 'release_report.json'))
    print(f"serving model: {serving} (INT8 AUROC drop {drop:+.4f}); wrote {args.out}/model_info.json")


if __name__ == '__main__':
    main()
