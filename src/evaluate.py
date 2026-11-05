import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
from typing import Dict, List, Any
import pandas as pd
import json

from src.utils import CLASSES, CLASS_NAMES

plt.style.use('dark_background')


def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                   device: torch.device) -> Dict[str, Any]:
    """
    Evaluates the model on the given dataloader and returns per-class metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    report = classification_report(
        all_labels, all_preds,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0
    )

    # Print report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(
        all_labels, all_preds,
        target_names=CLASSES,
        zero_division=0
    ))

    return {
        'y_true': all_labels,
        'y_pred': all_preds,
        'report': report,
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score'],
    }


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], save_path: str,
                          title: str = 'Confusion Matrix') -> None:
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(title, fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_training_curves(history: Dict[str, List[float]], save_path: str) -> None:
    """Plots and saves training and validation curves (loss and accuracy)."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_class_distribution(labels: List[int], title: str, save_path: str) -> None:
    """Plots and saves the class distribution."""
    counts = pd.Series(labels).value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(CLASSES))
    bars = plt.bar(CLASSES, counts.values, color=colors)
    plt.title(title, fontsize=14)
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    # Add count labels on bars
    for bar, count in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                 str(count), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Class distribution plot saved to {save_path}")


def compare_models(results_dict: Dict[str, Dict[str, Any]], save_path: str) -> None:
    """
    Plots a bar chart comparing recall across models for each class.
    results_dict format: {'ModelName': evaluation_results_dict}
    """
    data = []

    for model_name, res in results_dict.items():
        report = res['report']
        for cls in CLASSES:
            if cls in report:
                recall = report[cls]['recall']
                f1 = report[cls]['f1-score']
                data.append({
                    'Model': model_name,
                    'Class': cls,
                    'Recall': recall,
                    'F1-Score': f1
                })

    df = pd.DataFrame(data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    sns.barplot(data=df, x='Class', y='Recall', hue='Model', ax=ax1)
    ax1.set_title('Model Comparison: Recall per Class', fontsize=14)
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    sns.barplot(data=df, x='Class', y='F1-Score', hue='Model', ax=ax2)
    ax2.set_title('Model Comparison: F1-Score per Class', fontsize=14)
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Model comparison plot saved to {save_path}")


def save_results(results: Dict[str, Any], save_path: str) -> None:
    """Save results dict to JSON (excluding non-serializable arrays)."""
    save_data = {
        'accuracy': results['accuracy'],
        'macro_f1': results['macro_f1'],
        'per_class': {}
    }
    for cls in CLASSES:
        if cls in results['report']:
            save_data['per_class'][cls] = {
                'precision': results['report'][cls]['precision'],
                'recall': results['report'][cls]['recall'],
                'f1-score': results['report'][cls]['f1-score'],
                'support': results['report'][cls]['support'],
            }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=4)
    print(f"Results saved to {save_path}")
