"""
FID and KID for synthetic-image quality, using torchvision's ImageNet Inception-v3
pool features (2048-d). Note: absolute values are not directly comparable to the
`pytorch-fid` / TF-Inception numbers in papers; compare runs against each other and
against the real-vs-real reference computed with the same extractor.
"""
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from torchvision.models import Inception_V3_Weights, inception_v3

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class InceptionFeatures:
    def __init__(self, device: torch.device):
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        net.fc = torch.nn.Identity()
        self.net = net.eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, images_uint8: torch.Tensor, batch_size: int = 64) -> np.ndarray:
        feats = []
        for i in range(0, len(images_uint8), batch_size):
            x = images_uint8[i:i + batch_size].float() / 255.0
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
            x = ((x - _MEAN) / _STD).to(self.device)
            feats.append(self.net(x).cpu().double().numpy())
        return np.concatenate(feats)


def fid_from_features(a: np.ndarray, b: np.ndarray) -> float:
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():
        eps = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + eps) @ (s2 + eps))
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def kid_from_features(a: np.ndarray, b: np.ndarray, n_subsets: int = 50,
                      subset_size: int = 1000, seed: int = 0) -> Dict[str, float]:
    """Unbiased MMD^2 with cubic polynomial kernel (Binkowski et al., 2018)."""
    rng = np.random.default_rng(seed)
    m = min(subset_size, len(a), len(b))
    d = a.shape[1]
    vals = []
    for _ in range(n_subsets):
        x = a[rng.choice(len(a), m, replace=False)]
        y = b[rng.choice(len(b), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        mmd = ((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
               + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
               - 2 * kxy.mean())
        vals.append(mmd)
    return {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'subset_size': m}
