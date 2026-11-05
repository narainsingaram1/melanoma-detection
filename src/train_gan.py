"""
LEGACY (v1) Phase 3: Train conditional DCGAN on rare classes, generate synthetic data,
retrain classifier on augmented dataset, and compare results.

Superseded by `experiments/run_ablation.py --variants dcgan` (see src/synthetic.py
for the normalization and augmentation fixes).
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.utils import save_image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image

from src.utils import get_device, set_seed, save_model, CLASSES, CLASS_NAMES, RARE_CLASSES
from src.dataset import prepare_data, get_dataloaders, get_class_weights, HAM10000Dataset, get_transforms
from src.dcgan import Generator, Discriminator
from src.transfer_model import SkinLesionTransfer
from src.evaluate import (evaluate_model, plot_confusion_matrix,
                          plot_training_curves, save_results, compare_models)


class RareClassDataset(Dataset):
    """Filters an existing dataset to only include specific rare classes."""
    def __init__(self, base_dataset, rare_class_indices):
        self.samples = []
        print(f"Filtering dataset for rare classes {rare_class_indices}...")
        for i in range(len(base_dataset)):
            img, label = base_dataset[i]
            if label in rare_class_indices:
                self.samples.append((img, label))
        print(f"  Found {len(self.samples)} samples for rare classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class SyntheticDataset(Dataset):
    """Dataset of synthetic (generated) images."""
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def train_gan(data_dir: str, metadata_path: str, baseline_results=None, transfer_results=None):
    """Train DCGAN on rare classes, augment data, retrain classifier."""
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    os.makedirs('results/gan_progress', exist_ok=True)
    os.makedirs('data/synthetic', exist_ok=True)

    print("\n" + "="*60)
    print("PHASE 3: GAN-BASED DATA AUGMENTATION")
    print("="*60)

    # Prepare data
    train_df, val_df, test_df = prepare_data(metadata_path)
    train_transform, eval_transform = get_transforms()

    # Create training dataset
    train_dataset = HAM10000Dataset(train_df, data_dir, transform=train_transform)
    val_loader = DataLoader(
        HAM10000Dataset(val_df, data_dir, transform=eval_transform),
        batch_size=32, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        HAM10000Dataset(test_df, data_dir, transform=eval_transform),
        batch_size=32, shuffle=False, num_workers=0
    )

    # Map rare class names to indices
    rare_indices = [CLASSES.index(c) for c in RARE_CLASSES]
    print(f"Rare classes: {RARE_CLASSES} -> indices {rare_indices}")

    # Count class distribution
    class_counts = train_df['label_idx'].value_counts().sort_index()
    majority_count = class_counts.max()
    print(f"\nClass distribution in training set:")
    for idx, cls in enumerate(CLASSES):
        count = class_counts.get(idx, 0)
        marker = " *** RARE ***" if cls in RARE_CLASSES else ""
        print(f"  {cls} ({CLASS_NAMES[cls]}): {count}{marker}")
    print(f"Majority class count: {majority_count}")

    # Create rare class dataset for GAN training
    rare_dataset = RareClassDataset(train_dataset, rare_indices)
    if len(rare_dataset) == 0:
        print("ERROR: No rare class samples found!")
        return None

    gan_loader = DataLoader(rare_dataset, batch_size=min(32, len(rare_dataset)),
                            shuffle=True, drop_last=True, num_workers=0)

    # ============================
    # TRAIN GAN
    # ============================
    print("\n--- Training Conditional DCGAN ---")
    z_dim = 100
    G = Generator(z_dim=z_dim, num_classes=7).to(device)
    D = Discriminator(num_classes=7, img_size=224).to(device)
    G.weights_init()
    D.weights_init()

    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    num_epochs = 50  # Reduced for practical training time
    g_losses_history = []
    d_losses_history = []

    for epoch in range(num_epochs):
        G.train()
        D.train()

        d_losses, g_losses = [], []

        for real_imgs, labels in gan_loader:
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            batch_size = real_imgs.size(0)

            # Labels for real/fake
            valid = torch.ones(batch_size, 1, device=device) * 0.9  # label smoothing
            fake_label = torch.zeros(batch_size, 1, device=device) + 0.1

            # --- Train Discriminator ---
            opt_D.zero_grad()
            real_loss = criterion(D(real_imgs, labels), valid)

            z = torch.randn(batch_size, z_dim, device=device)
            gen_imgs = G(z, labels)
            fake_loss = criterion(D(gen_imgs.detach(), labels), fake_label)

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_D.step()

            # --- Train Generator ---
            opt_G.zero_grad()
            z = torch.randn(batch_size, z_dim, device=device)
            gen_imgs = G(z, labels)
            g_loss = criterion(D(gen_imgs, labels), torch.ones(batch_size, 1, device=device))
            g_loss.backward()
            opt_G.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

        avg_d = np.mean(d_losses)
        avg_g = np.mean(g_losses)
        d_losses_history.append(avg_d)
        g_losses_history.append(avg_g)

        if (epoch + 1) % 5 == 0:
            print(f"GAN Epoch [{epoch+1}/{num_epochs}] D Loss: {avg_d:.4f} | G Loss: {avg_g:.4f}")

        if (epoch + 1) % 10 == 0:
            G.eval()
            with torch.no_grad():
                test_z = torch.randn(16, z_dim, device=device)
                test_labels = torch.tensor([rare_indices[0]]*8 + [rare_indices[1]]*8,
                                           dtype=torch.long, device=device)
                sample_imgs = G(test_z, test_labels)
                save_image(sample_imgs, f"results/gan_progress/epoch_{epoch+1}.png",
                           nrow=4, normalize=True)

    # Save GAN loss curves
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(d_losses_history, label='Discriminator Loss', linewidth=2)
    ax.plot(g_losses_history, label='Generator Loss', linewidth=2)
    ax.set_title('GAN Training Losses')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/gan_loss_curves.png', dpi=150)
    plt.close()

    # Save generator
    torch.save(G.state_dict(), 'results/generator.pth')
    print("Generator saved to results/generator.pth")

    # ============================
    # GENERATE SYNTHETIC DATA
    # ============================
    print("\n--- Generating Synthetic Images ---")
    G.eval()

    target_per_class = min(2000, int(majority_count * 0.3))  # Generate enough to help but cap it
    synthetic_images = []
    synthetic_labels = []

    with torch.no_grad():
        for cls_idx in rare_indices:
            cls_name = CLASSES[cls_idx]
            current_count = class_counts.get(cls_idx, 0)
            needed = min(target_per_class, majority_count - current_count)
            needed = max(needed, 500)  # At least 500

            print(f"  Generating {needed} synthetic images for {cls_name} (currently {current_count})...")
            batches = needed // 32 + 1

            for _ in range(batches):
                z = torch.randn(32, z_dim, device=device)
                lbl = torch.full((32,), cls_idx, dtype=torch.long, device=device)
                gen_imgs = G(z, lbl).cpu()
                synthetic_images.extend(list(gen_imgs))
                synthetic_labels.extend([cls_idx] * 32)

                if len([l for l in synthetic_labels if l == cls_idx]) >= needed:
                    break

    # Trim to exact count needed
    print(f"  Total synthetic images generated: {len(synthetic_images)}")

    # ============================
    # RETRAIN CLASSIFIER ON AUGMENTED DATA
    # ============================
    print("\n--- Retraining Classifier on Augmented Data ---")

    # Create augmented training dataset
    syn_dataset = SyntheticDataset(synthetic_images, synthetic_labels)
    augmented_dataset = ConcatDataset([train_dataset, syn_dataset])
    aug_train_loader = DataLoader(augmented_dataset, batch_size=32, shuffle=True, num_workers=0)

    print(f"  Original training size: {len(train_dataset)}")
    print(f"  Synthetic additions: {len(syn_dataset)}")
    print(f"  Augmented training size: {len(augmented_dataset)}")

    # Train a new transfer model on augmented data
    model = SkinLesionTransfer(num_classes=7).to(device)
    model.unfreeze_all()

    class_weights = get_class_weights(train_df).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    num_epochs_cls = 20

    for epoch in range(num_epochs_cls):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in aug_train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"[GAN-Aug] Epoch [{epoch+1}/{num_epochs_cls}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, 'results/best_gan_augmented.pth')
            print(f"  => Saved new best model")

    plot_training_curves(history, 'results/gan_augmented_training_curves.png')

    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATING GAN-AUGMENTED MODEL ON TEST SET")
    print("="*60)
    model.load_state_dict(torch.load('results/best_gan_augmented.pth', map_location=device, weights_only=True))
    model.to(device)
    gan_results = evaluate_model(model, test_loader, device)

    plot_confusion_matrix(gan_results['y_true'], gan_results['y_pred'],
                          'results/gan_augmented_confusion_matrix.png',
                          title='GAN-Augmented Transfer - Confusion Matrix')

    save_results(gan_results, 'results/gan_augmented_metrics.json')

    print(f"\nGAN-Augmented Results:")
    print(f"  Accuracy: {gan_results['accuracy']:.4f}")
    print(f"  Macro F1: {gan_results['macro_f1']:.4f}")
    print(f"  Melanoma Recall: {gan_results['report'].get('mel', {}).get('recall', 0):.4f}")

    # ============================
    # COMPARE ALL MODELS
    # ============================
    if baseline_results and transfer_results:
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        all_results = {
            'Baseline CNN': baseline_results,
            'Transfer (EfficientNet)': transfer_results,
            'Transfer + GAN Aug': gan_results
        }
        compare_models(all_results, 'results/model_comparison.png')

        # Print comparison table
        print(f"\n{'Model':<30} {'Accuracy':<12} {'Macro F1':<12} {'Mel Recall':<12}")
        print("-" * 66)
        for name, res in all_results.items():
            mel_recall = res['report'].get('mel', {}).get('recall', 0)
            print(f"{name:<30} {res['accuracy']:<12.4f} {res['macro_f1']:<12.4f} {mel_recall:<12.4f}")

    return gan_results


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'ham10000')
    metadata_path = os.path.join(data_dir, 'HAM10000_metadata.csv')

    if not os.path.exists(metadata_path):
        import kagglehub
        data_dir = kagglehub.dataset_download('kmader/skin-cancer-mnist-ham10000')
        metadata_path = os.path.join(data_dir, 'HAM10000_metadata.csv')

    train_gan(data_dir, metadata_path)
