import torch
import random
import numpy as np
import os
import logging
from typing import Dict, List, Any

from src.constants import CLASSES, CLASS_NAMES, IMG_SIZE, NUM_CLASSES, RARE_CLASSES  # noqa: F401


def get_device() -> torch.device:
    """Returns the best available PyTorch device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def set_seed(seed: int = 42) -> None:
    """Sets seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS doesn't have a specific manual_seed function, uses torch.manual_seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_model(model: torch.nn.Module, path: str) -> None:
    """Saves the model state dict."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logging.info(f"Model saved to {path}")

def load_model(model: torch.nn.Module, path: str, device: torch.device) -> torch.nn.Module:
    """Loads the model state dict."""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    logging.info(f"Model loaded from {path}")
    return model

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
