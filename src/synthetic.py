"""
Conditional DCGAN training + synthetic image generation for rare-class augmentation.

Fixes over the v1 pipeline (run_all.py / train_gan.py):
  * Generator output (Tanh, [-1, 1]) is converted to uint8 pixels, then goes through
    the *same* train transform + ImageNet normalization as real images. v1 fed raw
    [-1, 1] tensors to a classifier that sees ImageNet-normalized real images, so
    synthetic samples were trivially distinguishable by intensity statistics.
  * Real rare-class images are augmented freshly every GAN epoch (v1 pre-computed
    one augmented copy and reused it for all epochs).
  * Synthetic images also receive random augmentation during classifier training.
"""
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision.utils import save_image

from src.dcgan import Discriminator, Generator

Z_DIM = 100


def gan_real_transform() -> v2.Compose:
    """uint8 CHW -> float in [-1, 1], matching the generator's Tanh range."""
    return v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5] * 3, [0.5] * 3),
    ])


def tanh_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Generator output in [-1, 1] -> uint8 pixels in [0, 255]."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def train_dcgan(real_rare: Dataset, device: torch.device, epochs: int = 30,
                batch_size: int = 16, num_classes: int = 7, sample_classes: Sequence[int] = (),
                log_dir: str = None, seed: int = 0) -> Tuple[Generator, Dict[str, List[float]]]:
    """Trains a class-conditional DCGAN on `real_rare` (yields ([-1,1] image, label))."""
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(real_rare, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=0, generator=g)

    G = Generator(z_dim=Z_DIM, num_classes=num_classes).to(device)
    D = Discriminator(num_classes=num_classes, img_size=224).to(device)
    G.weights_init()
    D.weights_init()
    bce = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    history = {'d_loss': [], 'g_loss': []}
    fixed_z = torch.randn(16, Z_DIM, device=device)
    if sample_classes:
        fixed_lbl = torch.tensor([sample_classes[i * len(sample_classes) // 16] for i in range(16)],
                                 device=device)

    for epoch in range(epochs):
        G.train(); D.train()
        d_losses, g_losses = [], []
        for real, labels in loader:
            real, labels = real.to(device), labels.to(device)
            bs = real.size(0)
            real_t = torch.full((bs, 1), 0.9, device=device)   # one-sided label smoothing
            fake_t = torch.full((bs, 1), 0.1, device=device)

            opt_D.zero_grad()
            fake = G(torch.randn(bs, Z_DIM, device=device), labels)
            d_loss = (bce(D(real, labels), real_t) + bce(D(fake.detach(), labels), fake_t)) / 2
            d_loss.backward()
            opt_D.step()

            opt_G.zero_grad()
            fake = G(torch.randn(bs, Z_DIM, device=device), labels)
            g_loss = bce(D(fake, labels), torch.ones(bs, 1, device=device))
            g_loss.backward()
            opt_G.step()

            d_losses.append(d_loss.item()); g_losses.append(g_loss.item())

        history['d_loss'].append(float(np.mean(d_losses)))
        history['g_loss'].append(float(np.mean(g_losses)))
        print(f"  GAN epoch {epoch + 1}/{epochs}: D={history['d_loss'][-1]:.4f} G={history['g_loss'][-1]:.4f}", flush=True)

        if log_dir and sample_classes and ((epoch + 1) % 10 == 0 or epoch + 1 == epochs):
            os.makedirs(log_dir, exist_ok=True)
            G.eval()
            with torch.no_grad():
                save_image(G(fixed_z, fixed_lbl), os.path.join(log_dir, f'epoch_{epoch + 1}.png'),
                           nrow=4, normalize=True, value_range=(-1, 1))

    G.eval()
    return G, history


@torch.no_grad()
def generate_uint8(G: Generator, class_idx: int, n: int, device: torch.device,
                   batch_size: int = 64) -> torch.Tensor:
    """Returns n generated images for class_idx as uint8 CHW tensors on CPU."""
    G.eval()
    out = []
    while sum(len(o) for o in out) < n:
        bs = min(batch_size, n - sum(len(o) for o in out))
        z = torch.randn(bs, Z_DIM, device=device)
        lbl = torch.full((bs,), class_idx, dtype=torch.long, device=device)
        out.append(tanh_to_uint8(G(z, lbl)).cpu())
    return torch.cat(out)
