import numpy as np
import pandas as pd

from src.dataset import split_by_lesion
from src.utils import CLASSES


def _fake_metadata(n_lesions=3000, seed=0):
    """Mimics HAM10000: imbalanced classes, 1-4 images per lesion."""
    rng = np.random.default_rng(seed)
    probs = np.array([327, 514, 1099, 115, 1113, 6705, 142], dtype=float)
    probs /= probs.sum()
    rows = []
    for les in range(n_lesions):
        label = rng.choice(len(CLASSES), p=probs)
        for k in range(rng.integers(1, 5)):
            rows.append({'lesion_id': f'HAM_{les:07d}', 'image_id': f'ISIC_{les:07d}_{k}',
                         'dx': CLASSES[label], 'label_idx': label})
    return pd.DataFrame(rows)


def test_no_lesion_in_two_splits():
    df = _fake_metadata()
    train, val, test = split_by_lesion(df)
    tr, va, te = (set(s['lesion_id']) for s in (train, val, test))
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_every_image_assigned_once():
    df = _fake_metadata()
    train, val, test = split_by_lesion(df)
    assert len(train) + len(val) + len(test) == len(df)
    all_ids = pd.concat([train, val, test])['image_id']
    assert all_ids.is_unique


def test_split_is_stratified_and_roughly_70_15_15():
    df = _fake_metadata()
    train, val, test = split_by_lesion(df)
    n_les = df['lesion_id'].nunique()
    for part, frac in ((train, 0.70), (val, 0.15), (test, 0.15)):
        assert abs(part['lesion_id'].nunique() / n_les - frac) < 0.01
        # every class appears in every split
        assert set(part['label_idx']) == set(range(len(CLASSES)))


def test_split_is_deterministic():
    df = _fake_metadata()
    a = split_by_lesion(df, seed=42)[2]['image_id'].tolist()
    b = split_by_lesion(df, seed=42)[2]['image_id'].tolist()
    assert a == b
