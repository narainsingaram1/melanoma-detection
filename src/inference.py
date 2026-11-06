"""
Serving-time inference shared by the FastAPI service and the Gradio app.

Pipeline for one image:
  PIL image -> resize 224x224 (bilinear, same as training cache) -> ImageNet normalize
  -> model -> (logits, pooled 1280-d features)
  -> temperature-scaled softmax (calibrated probabilities)
  -> Mahalanobis distance of features to the nearest class mean (out-of-distribution
     score); images above the threshold are rejected as "not a dermoscopic lesion".

Release artifacts live in models/ (built by scripts/build_release.py):
  model_info.json   metadata, temperature, OOD threshold, metrics
  ood_stats.npz     class means + precision matrix for the Mahalanobis score
  classifier*.onnx  ONNX exports (FP32 / INT8)
  classifier.pth    PyTorch weights (needed for Grad-CAM)
"""
import json
import os
import time
from typing import Dict, Optional

import numpy as np
from PIL import Image

from src.constants import CLASS_NAMES, CLASSES, IMG_SIZE

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
RISK = {'mel': 'malignant', 'bcc': 'malignant', 'akiec': 'pre-malignant',
        'bkl': 'benign', 'df': 'benign', 'nv': 'benign', 'vasc': 'benign'}
DEFAULT_MODEL_DIR = os.environ.get(
    'MODEL_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'))
MAX_PIXELS = 40_000_000  # reject decompression-bomb sized uploads


def preprocess(img: Image.Image) -> np.ndarray:
    """PIL image -> float32 [1, 3, 224, 224], matching training preprocessing exactly."""
    if img.width * img.height > MAX_PIXELS:
        raise ValueError('image too large')
    img = img.convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return ((x - MEAN) / STD).astype(np.float32)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class OODScorer:
    """Min-over-classes Mahalanobis distance with a shared (Ledoit-Wolf) covariance."""

    def __init__(self, means: np.ndarray, precision: np.ndarray):
        self.means = means.astype(np.float32)          # [C, D]
        self.precision = precision.astype(np.float32)  # [D, D]

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        d = feats[:, None, :] - self.means[None]                    # [N, C, D]
        m = np.einsum('ncd,de,nce->nc', d, self.precision, d)       # [N, C]
        return m.min(axis=1)


class Postprocessor:
    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        with open(os.path.join(model_dir, 'model_info.json')) as f:
            self.info = json.load(f)
        stats = np.load(os.path.join(model_dir, 'ood_stats.npz'))
        self.ood = OODScorer(stats['means'], stats['precision'])
        self.temperature = float(self.info['calibration']['temperature'])
        self.threshold = float(self.info['ood']['threshold'])

    def __call__(self, logits: np.ndarray, feats: np.ndarray) -> Dict:
        probs = softmax(logits[0] / self.temperature)
        score = float(self.ood(feats)[0])
        top = int(probs.argmax())
        code = CLASSES[top]
        return {
            'in_distribution': score <= self.threshold,
            'ood_score': score,
            'ood_threshold': self.threshold,
            'prediction': {'code': code, 'name': CLASS_NAMES[code], 'risk': RISK[code],
                           'probability': float(probs[top])},
            'probabilities': {c: float(p) for c, p in zip(CLASSES, probs)},
        }


class OnnxLesionClassifier:
    """CPU inference with ONNX Runtime; used by the API."""

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, model_file: Optional[str] = None,
                 threads: int = int(os.environ.get('ORT_THREADS', '2'))):
        import onnxruntime as ort
        self.post = Postprocessor(model_dir)
        model_file = model_file or self.post.info['serving']['onnx_file']
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(os.path.join(model_dir, model_file), opts,
                                            providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.model_file = model_file

    def predict(self, img: Image.Image) -> Dict:
        t0 = time.perf_counter()
        logits, feats = self.session.run(None, {self.input_name: preprocess(img)})
        out = self.post(logits, feats)
        out['latency_ms'] = round((time.perf_counter() - t0) * 1000, 2)
        return out


class TorchLesionClassifier:
    """PyTorch inference; used by the Gradio app because Grad-CAM needs gradients."""

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, device: str = 'cpu'):
        import torch
        from src.transfer_model import SkinLesionTransfer
        self.torch = torch
        self.post = Postprocessor(model_dir)
        self.device = torch.device(device)
        self.model = SkinLesionTransfer(num_classes=len(CLASSES), pretrained=False)
        self.model.load_state_dict(torch.load(os.path.join(model_dir, 'classifier.pth'),
                                              map_location=self.device, weights_only=True))
        self.model.to(self.device).eval()

    def tensor(self, img: Image.Image):
        return self.torch.from_numpy(preprocess(img)).to(self.device)

    def predict(self, img: Image.Image) -> Dict:
        t0 = time.perf_counter()
        with self.torch.no_grad():
            logits, feats = self.model.forward_with_features(self.tensor(img))
        out = self.post(logits.cpu().numpy(), feats.cpu().numpy())
        out['latency_ms'] = round((time.perf_counter() - t0) * 1000, 2)
        return out


def ensure_model_dir(model_dir: str = DEFAULT_MODEL_DIR) -> str:
    """Downloads release artifacts from the Hugging Face Hub if HF_MODEL_REPO is set and files are missing."""
    if os.path.exists(os.path.join(model_dir, 'model_info.json')):
        return model_dir
    repo = os.environ.get('HF_MODEL_REPO')
    if not repo:
        raise FileNotFoundError(f'{model_dir}/model_info.json not found; run scripts/build_release.py '
                                'or set HF_MODEL_REPO')
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo, local_dir=model_dir)
