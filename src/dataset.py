import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from typing import Tuple, Optional, Callable, Dict, List

from src.utils import IMG_SIZE, CLASSES


class HAM10000Dataset(Dataset):
    """HAM10000 Dataset."""
    def __init__(self, dataframe: pd.DataFrame, root_dir: str, transform: Optional[Callable] = None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with 'image_id' and 'label_idx' columns.
            root_dir (str): Directory containing the images (can have subfolders).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

        # Create a mapping of image_id to full path to handle both image folders
        self.image_paths = {}
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            all_images = glob.glob(os.path.join(root_dir, '**', ext), recursive=True)
            for img_path in all_images:
                basename = os.path.splitext(os.path.basename(img_path))[0]
                self.image_paths[basename] = img_path

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.dataframe.iloc[idx]
        image_id = row['image_id']
        label = int(row['label_idx'])

        img_path = self.image_paths.get(image_id)
        if img_path is None:
            raise FileNotFoundError(f"Image {image_id}.jpg not found in {self.root_dir}")

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def split_by_lesion(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15,
                    seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/val/test split at the *lesion* level.

    HAM10000 contains ~10k images of ~7.5k unique lesions: the same lesion is often
    photographed several times. Splitting by image lets near-duplicate photos of one
    lesion land in both train and test, which inflates test metrics. Here every image
    of a lesion goes to exactly one split, stratified by the lesion's diagnosis.
    """
    group_col = 'lesion_id' if 'lesion_id' in df.columns else 'image_id'
    lesions = df.groupby(group_col)['label_idx'].first().reset_index()

    train_les, temp_les = train_test_split(
        lesions, test_size=val_frac + test_frac,
        stratify=lesions['label_idx'], random_state=seed)
    val_les, test_les = train_test_split(
        temp_les, test_size=test_frac / (val_frac + test_frac),
        stratify=temp_les['label_idx'], random_state=seed)

    def pick(les):
        return df[df[group_col].isin(set(les[group_col]))].reset_index(drop=True)

    return pick(train_les), pick(val_les), pick(test_les)


def prepare_data(metadata_path: str, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reads metadata, maps labels to integers, and performs a lesion-level
    stratified split (~70/15/15 of lesions).
    """
    df = pd.read_csv(metadata_path)

    # Map classes to indices
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    df['label_idx'] = df['dx'].map(class_to_idx)

    # Drop any rows with unmapped classes
    df = df.dropna(subset=['label_idx'])
    df['label_idx'] = df['label_idx'].astype(int)

    train_df, val_df, test_df = split_by_lesion(df, seed=seed)

    print(f"Data split (lesion-level): Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)} images")
    return train_df, val_df, test_df


def get_transforms():
    """Returns train and eval transforms."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_transform, eval_transform


def get_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                    root_dir: str, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders with appropriate transforms.
    """
    train_transform, eval_transform = get_transforms()

    train_dataset = HAM10000Dataset(train_df, root_dir, transform=train_transform)
    val_dataset = HAM10000Dataset(val_df, root_dir, transform=eval_transform)
    test_dataset = HAM10000Dataset(test_df, root_dir, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0)

    return train_loader, val_loader, test_loader


def get_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """
    Computes inverse frequency weights for classes to handle class imbalance.
    """
    class_counts = train_df['label_idx'].value_counts().sort_index().values
    total_samples = np.sum(class_counts)
    weights = total_samples / (len(class_counts) * class_counts.astype(np.float64))
    return torch.FloatTensor(weights)
