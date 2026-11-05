import numpy as np
import torch

from src.fid import fid_from_features, kid_from_features
from src.metrics import MEL_IDX, bootstrap_ci, compute_metrics, sensitivity_at_specificity
from src.synthetic import gan_real_transform, generate_uint8, tanh_to_uint8
from src.dcgan import Generator


def test_perfect_predictions():
    y = np.repeat(np.arange(7), 10)
    probs = np.eye(7)[y] * 0.9 + 0.1 / 7
    m = compute_metrics(y, probs)
    assert m['accuracy'] == 1.0 and m['balanced_accuracy'] == 1.0
    assert m['macro_auroc'] == 1.0 and m['mel_sens_at_90spec'] == 1.0
    assert m['per_class']['mel']['recall'] == 1.0


def test_balanced_accuracy_penalizes_majority_guessing():
    y = np.array([5] * 90 + [MEL_IDX] * 10)
    probs = np.tile(np.eye(7)[5], (100, 1))  # always predicts nv
    m = compute_metrics(y, probs)
    assert m['accuracy'] == 0.9
    assert m['balanced_accuracy'] == 0.5
    assert m['mel_recall'] == 0.0


def test_sensitivity_at_specificity():
    y = np.array([0] * 10 + [1] * 10)
    score = np.concatenate([np.linspace(0, 0.5, 10), np.linspace(0.45, 1, 10)])
    s = sensitivity_at_specificity(y, score, spec=0.9)
    assert 0.8 <= s <= 1.0


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 7, 400)
    probs = rng.dirichlet(np.ones(7), 400)
    probs[np.arange(400), y] += 0.5
    probs /= probs.sum(1, keepdims=True)
    point = compute_metrics(y, probs)
    ci = bootstrap_ci(y, probs, groups=np.arange(400) // 2, n_boot=200)
    for k in ('accuracy', 'macro_auroc'):
        assert ci[k]['lo'] <= point[k] <= ci[k]['hi']


def test_tanh_to_uint8_range():
    x = torch.tensor([-1.5, -1.0, 0.0, 1.0, 2.0])
    assert tanh_to_uint8(x).tolist() == [0, 0, 128, 255, 255]


def test_gan_real_transform_matches_tanh_range():
    img = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)
    out = gan_real_transform()(img)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_generate_uint8_shape():
    G = Generator(z_dim=100, num_classes=7)
    imgs = generate_uint8(G, class_idx=MEL_IDX, n=5, device=torch.device('cpu'), batch_size=2)
    assert imgs.shape == (5, 3, 224, 224) and imgs.dtype == torch.uint8


def test_fid_kid_identical_vs_shifted():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 16))
    b = rng.normal(size=(300, 16))
    c = rng.normal(loc=2.0, size=(300, 16))
    assert fid_from_features(a, b) < fid_from_features(a, c)
    assert kid_from_features(a, b, n_subsets=5, subset_size=100)['mean'] < \
        kid_from_features(a, c, n_subsets=5, subset_size=100)['mean']
