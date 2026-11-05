"""
Quantifies the train/test leakage in the v1 image-level split.

Reproduces the original split (sklearn train_test_split on images, stratified,
random_state=42) and counts test images whose lesion also appears in training.
"""
import glob
import json
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.dataset import split_by_lesion  # noqa: E402
from src.utils import CLASSES  # noqa: E402


def main(data_dir=os.path.join(ROOT, 'data', 'ham10000')):
    meta = glob.glob(os.path.join(data_dir, '**', 'HAM10000_metadata.csv'), recursive=True)[0]
    df = pd.read_csv(meta)
    df['label_idx'] = df['dx'].map({c: i for i, c in enumerate(CLASSES)}).astype(int)

    tr, tmp = train_test_split(df, test_size=0.30, stratify=df['label_idx'], random_state=42)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp['label_idx'], random_state=42)
    leaked = te['lesion_id'].isin(set(tr['lesion_id']))

    report = {
        'images': len(df), 'unique_lesions': int(df['lesion_id'].nunique()),
        'lesions_with_multiple_images': int((df['lesion_id'].value_counts() > 1).sum()),
        'v1_image_split': {
            'test_images': len(te),
            'test_images_whose_lesion_is_in_train': int(leaked.sum()),
            'pct_test_images_leaked': round(100 * leaked.mean(), 1),
            'pct_leaked_by_class': {c: round(100 * leaked[te['dx'] == c].mean(), 1) for c in CLASSES},
        },
    }
    tr2, _, te2 = split_by_lesion(df)
    report['v2_lesion_split'] = {
        'test_images': len(te2),
        'test_images_whose_lesion_is_in_train': int(te2['lesion_id'].isin(set(tr2['lesion_id'])).sum()),
    }
    out = os.path.join(ROOT, 'results', 'leakage_report.json')
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
