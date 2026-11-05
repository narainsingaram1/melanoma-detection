"""
Phase 1: Train baseline CNN from scratch on HAM10000.
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils import get_device, set_seed, save_model, CLASSES, CLASS_NAMES
from src.dataset import prepare_data, get_dataloaders, get_class_weights
from src.baseline_cnn import SkinLesionCNN
from src.evaluate import (evaluate_model, plot_confusion_matrix,
                          plot_training_curves, plot_class_distribution, save_results)


def train_baseline(data_dir: str, metadata_path: str):
    """Train the baseline CNN model."""
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    os.makedirs("results", exist_ok=True)

    # Prepare data
    print("\n" + "="*60)
    print("PHASE 1: BASELINE CNN TRAINING")
    print("="*60)

    train_df, val_df, test_df = prepare_data(metadata_path)

    # Plot class distribution
    plot_class_distribution(
        train_df['label_idx'].tolist(),
        'Training Set Class Distribution',
        'results/class_distribution_train.png'
    )

    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, data_dir, batch_size=32
    )

    # Get class weights for imbalanced data
    class_weights = get_class_weights(train_df).to(device)
    print(f"Class weights: {class_weights}")

    # Initialize model
    model = SkinLesionCNN(num_classes=7).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    num_epochs = 30
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

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

        # Validation phase
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

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_model(model, "results/best_baseline.pth")
            print(f"  => Saved new best model (val_loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Plot training curves
    plot_training_curves(history, "results/baseline_training_curves.png")

    # Final evaluation on test set
    print("\n" + "="*60)
    print("EVALUATING BASELINE CNN ON TEST SET")
    print("="*60)
    model.load_state_dict(torch.load("results/best_baseline.pth", map_location=device, weights_only=True))
    model.to(device)
    results = evaluate_model(model, test_loader, device)

    plot_confusion_matrix(results['y_true'], results['y_pred'],
                          'results/baseline_confusion_matrix.png',
                          title='Baseline CNN - Confusion Matrix')

    save_results(results, 'results/baseline_metrics.json')

    print(f"\nBaseline CNN Results:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Macro F1: {results['macro_f1']:.4f}")
    print(f"  Melanoma Recall: {results['report'].get('mel', {}).get('recall', 0):.4f}")

    return results


if __name__ == "__main__":
    # Default paths - adjust based on where dataset was downloaded
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'ham10000')
    metadata_path = os.path.join(data_dir, 'HAM10000_metadata.csv')

    if not os.path.exists(metadata_path):
        # Try alternate location
        import kagglehub
        data_dir = kagglehub.dataset_download('kmader/skin-cancer-mnist-ham10000')
        metadata_path = os.path.join(data_dir, 'HAM10000_metadata.csv')

    train_baseline(data_dir, metadata_path)
