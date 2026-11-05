"""
Phase 2: Train transfer learning model (EfficientNet-B0) on HAM10000.
Two-stage training: classifier head first, then full fine-tuning.
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
from src.transfer_model import SkinLesionTransfer
from src.evaluate import (evaluate_model, plot_confusion_matrix,
                          plot_training_curves, save_results)


def train_one_phase(model, train_loader, val_loader, criterion, optimizer, scheduler,
                    device, num_epochs, phase_name, history, best_val_loss, patience=10):
    """Run one training phase (used for both stages)."""
    epochs_no_improve = 0
    best_loss = best_val_loss

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
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

        total_epoch = len(history['train_loss'])
        print(f"[{phase_name}] Epoch [{total_epoch}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            save_model(model, "results/best_transfer.pth")
            print(f"  => Saved new best model (val_loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {total_epoch}")
                break

    return best_loss


def train_transfer(data_dir: str, metadata_path: str):
    """Train the transfer learning model."""
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    os.makedirs("results", exist_ok=True)

    print("\n" + "="*60)
    print("PHASE 2: TRANSFER LEARNING (EfficientNet-B0)")
    print("="*60)

    train_df, val_df, test_df = prepare_data(metadata_path)

    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, data_dir, batch_size=32
    )

    class_weights = get_class_weights(train_df).to(device)
    print(f"Class weights: {class_weights}")

    model = SkinLesionTransfer(num_classes=7).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable (stage 1)")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # Stage 1: Train only classifier head (10 epochs)
    print("\n--- Stage 1: Training classifier head only ---")
    optimizer_s1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    scheduler_s1 = ReduceLROnPlateau(optimizer_s1, mode='min', factor=0.5, patience=3)
    best_loss = train_one_phase(
        model, train_loader, val_loader, criterion, optimizer_s1, scheduler_s1,
        device, num_epochs=10, phase_name="Stage1-Head",
        history=history, best_val_loss=float('inf'), patience=10
    )

    # Stage 2: Unfreeze all and fine-tune (20 epochs)
    print("\n--- Stage 2: Full fine-tuning ---")
    model.unfreeze_all()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after unfreeze: {trainable_params:,}")

    optimizer_s2 = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler_s2 = ReduceLROnPlateau(optimizer_s2, mode='min', factor=0.5, patience=5)
    train_one_phase(
        model, train_loader, val_loader, criterion, optimizer_s2, scheduler_s2,
        device, num_epochs=20, phase_name="Stage2-FullFT",
        history=history, best_val_loss=best_loss, patience=10
    )

    # Plot training curves
    plot_training_curves(history, "results/transfer_training_curves.png")

    # Final evaluation on test set
    print("\n" + "="*60)
    print("EVALUATING TRANSFER MODEL ON TEST SET")
    print("="*60)
    model.load_state_dict(torch.load("results/best_transfer.pth", map_location=device, weights_only=True))
    model.to(device)
    results = evaluate_model(model, test_loader, device)

    plot_confusion_matrix(results['y_true'], results['y_pred'],
                          'results/transfer_confusion_matrix.png',
                          title='Transfer Learning (EfficientNet-B0) - Confusion Matrix')

    save_results(results, 'results/transfer_metrics.json')

    print(f"\nTransfer Learning Results:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Macro F1: {results['macro_f1']:.4f}")
    print(f"  Melanoma Recall: {results['report'].get('mel', {}).get('recall', 0):.4f}")

    return results


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'ham10000')
    metadata_path = os.path.join(data_dir, 'HAM10000_metadata.csv')

    if not os.path.exists(metadata_path):
        import kagglehub
        data_dir = kagglehub.dataset_download('kmader/skin-cancer-mnist-ham10000')
        metadata_path = os.path.join(data_dir, 'HAM10000_metadata.csv')

    train_transfer(data_dir, metadata_path)
