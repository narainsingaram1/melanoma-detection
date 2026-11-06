"""API tests against a tiny, untrained model so they run in CI without real weights."""
import io
import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.transfer_model import SkinLesionTransfer
from src.utils import CLASSES


class _WithFeatures(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.forward_with_features(x)


@pytest.fixture(scope='module')
def client(tmp_path_factory, monkeypatch_module):
    d = tmp_path_factory.mktemp('models')
    model = SkinLesionTransfer(num_classes=7, pretrained=False).eval()
    torch.onnx.export(_WithFeatures(model), torch.zeros(1, 3, 224, 224), str(d / 'classifier_fp32.onnx'),
                      dynamo=False, input_names=['image'], output_names=['logits', 'features'],
                      dynamic_axes={'image': {0: 'batch'}}, opset_version=17)
    np.savez(d / 'ood_stats.npz', means=np.zeros((7, 1280), np.float32), precision=np.eye(1280, dtype=np.float32))
    info = {'variant': 'test', 'seed': 0, 'architecture': 'EfficientNet-B0', 'classes': CLASSES,
            'calibration': {'temperature': 1.0}, 'ood': {'threshold': 1e9},
            'test_metrics_uncalibrated': {}, 'serving': {'onnx_file': 'classifier_fp32.onnx'}}
    (d / 'model_info.json').write_text(json.dumps(info))

    monkeypatch_module.setenv('RATE_LIMIT_PER_MIN', '5')
    import importlib
    import src.inference
    monkeypatch_module.setattr(src.inference, 'DEFAULT_MODEL_DIR', str(d))
    import api.main
    importlib.reload(api.main)
    monkeypatch_module.setattr(api.main, 'ensure_model_dir', lambda: str(d))
    with TestClient(api.main.app) as c:
        yield c


@pytest.fixture(scope='module')
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def _jpeg(size=(600, 450)):
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (*size[::-1], 3), dtype=np.uint8)).save(buf, 'JPEG')
    return buf.getvalue()


def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200 and r.json()['status'] == 'ok'


def test_predict_returns_calibrated_distribution(client):
    r = client.post('/predict', files={'file': ('x.jpg', _jpeg(), 'image/jpeg')})
    assert r.status_code == 200
    body = r.json()
    assert set(body['probabilities']) == set(CLASSES)
    assert abs(sum(body['probabilities'].values()) - 1) < 1e-4
    assert body['prediction']['code'] in CLASSES and 'disclaimer' in body


def test_rejects_wrong_type_and_garbage(client):
    assert client.post('/predict', files={'file': ('x.txt', b'hello', 'text/plain')}).status_code == 415
    assert client.post('/predict', files={'file': ('x.jpg', b'not an image', 'image/jpeg')}).status_code == 400


def test_rate_limit(client):
    codes = [client.post('/predict', files={'file': ('x.jpg', _jpeg((64, 64)), 'image/jpeg')}).status_code
             for _ in range(6)]
    assert 429 in codes
