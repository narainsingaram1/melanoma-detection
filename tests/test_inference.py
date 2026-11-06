import numpy as np
import torch
from PIL import Image

from src.data_cache import get_tensor_transforms
from src.inference import OODScorer, preprocess, softmax


def test_serving_preprocess_matches_training_pipeline():
    """Guards against train/serve skew: API preprocessing == cache + eval transform."""
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (450, 600, 3), dtype=np.uint8))
    cached = np.asarray(img.convert('RGB').resize((224, 224), Image.BILINEAR))  # build_image_cache
    _, eval_tf = get_tensor_transforms()
    train_side = eval_tf(torch.from_numpy(cached.copy()).permute(2, 0, 1)).numpy()
    serve = preprocess(img)
    np.testing.assert_allclose(serve[0], train_side, atol=1e-5)
    assert serve.flags['C_CONTIGUOUS']  # channels-last strides make PyTorch CPU ~4x slower


def test_softmax_temperature_flattens():
    z = np.array([[4.0, 1.0, 0.0]])
    assert softmax(z / 2.0).max() < softmax(z).max()
    np.testing.assert_allclose(softmax(z).sum(), 1.0)


def test_ood_scorer_is_min_class_distance():
    means = np.array([[0.0, 0.0], [10.0, 0.0]])
    scorer = OODScorer(means, np.eye(2))
    s = scorer(np.array([[9.0, 0.0], [5.0, 50.0]], dtype=np.float32))
    assert s[0] == 1.0          # closest to class 1
    assert s[1] > 2000          # far from both
