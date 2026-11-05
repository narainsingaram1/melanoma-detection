"""
Fast in-memory image pipeline.

Decoding and resizing ~600x450 JPEGs on every epoch dominates training time with
num_workers=0 (the default on macOS). We decode each image once, resize it to
IMG_SIZE x IMG_SIZE (same as the original Resize((224, 224))), and keep the result
in a uint8 memory-mapped array on disk. Augmentation then runs on uint8 tensors.
"""
import glob
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from src.utils import IMG_SIZE

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _index_images(root_dir: str) -> Dict[str, str]:
    paths = {}
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        for p in glob.glob(os.path.join(root_dir, '**', ext), recursive=True):
            paths.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    return paths


def build_image_cache(metadata_path: str, root_dir: str, cache_dir: str,
                      size: int = IMG_SIZE) -> Tuple[np.ndarray, Dict[str, int]]:
    """Returns (uint8 array [N, H, W, 3], image_id -> row). Builds it on first call."""
    os.makedirs(cache_dir, exist_ok=True)
    arr_path = os.path.join(cache_dir, f'images_{size}.npy')
    ids_path = os.path.join(cache_dir, f'images_{size}_ids.json')

    if os.path.exists(arr_path) and os.path.exists(ids_path):
        with open(ids_path) as f:
            ids = json.load(f)
        arr = np.load(arr_path, mmap_mode='r')
        return arr, {k: i for i, k in enumerate(ids)}

    ids = pd.read_csv(metadata_path)['image_id'].tolist()
    paths = _index_images(root_dir)
    missing = [i for i in ids if i not in paths]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing under {root_dir}, e.g. {missing[:3]}")

    print(f"Building image cache ({len(ids)} images at {size}x{size}) -> {arr_path}")
    tmp_path = arr_path + '.tmp'
    arr = np.lib.format.open_memmap(tmp_path, mode='w+', dtype=np.uint8, shape=(len(ids), size, size, 3))
    for i, image_id in enumerate(ids):
        img = Image.open(paths[image_id]).convert('RGB').resize((size, size), Image.BILINEAR)
        arr[i] = np.asarray(img)
        if (i + 1) % 2000 == 0:
            print(f"  cached {i + 1}/{len(ids)}")
    arr.flush()
    del arr
    os.replace(tmp_path, arr_path)
    with open(ids_path, 'w') as f:
        json.dump(ids, f)
    return np.load(arr_path, mmap_mode='r'), {k: i for i, k in enumerate(ids)}


def get_tensor_transforms():
    """Train/eval transforms operating on uint8 CHW tensors (same ops as src.dataset)."""
    to_float = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    train = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(20),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        *to_float,
    ])
    return train, v2.Compose(to_float)


class CachedImageDataset(Dataset):
    """Rows of `df` looked up in the uint8 cache; yields (normalized tensor, label)."""

    def __init__(self, df: pd.DataFrame, cache: np.ndarray, id_to_row: Dict[str, int],
                 transform: Optional[v2.Compose] = None):
        self.rows = np.array([id_to_row[i] for i in df['image_id']], dtype=np.int64)
        self.labels = df['label_idx'].to_numpy(dtype=np.int64)
        self.cache = cache
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        img = torch.from_numpy(np.array(self.cache[self.rows[idx]])).permute(2, 0, 1)  # uint8 CHW
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


class SyntheticImageDataset(Dataset):
    """Generated images stored as uint8 CHW; gets the same train transform as real images."""

    def __init__(self, images_uint8: torch.Tensor, labels, transform: Optional[v2.Compose] = None):
        assert images_uint8.dtype == torch.uint8 and images_uint8.ndim == 4
        self.images = images_uint8
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])
