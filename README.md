# 🔬 Melanoma Detection with GAN-Based Class Balancing

A complete deep learning pipeline for skin cancer classification using the **HAM10000** dataset, featuring a custom CNN baseline, transfer learning with EfficientNet-B0, GAN-based synthetic data augmentation for rare class balancing, and Grad-CAM explainability.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)

## 🎯 Overview

### The Problem
Skin cancer datasets suffer from severe class imbalance — melanoma (the most dangerous type) has far fewer training samples than common benign lesions. A naive classifier gets "lazy" and defaults to predicting the majority class, achieving deceptively high accuracy while failing to detect the cases that matter most.

### The Solution
This project implements a 5-phase pipeline:
1. **Baseline CNN** — Custom architecture to establish benchmark performance
2. **Transfer Learning** — EfficientNet-B0 fine-tuning for improved feature extraction
3. **GAN Augmentation** — Conditional DCGAN generates synthetic melanoma/dermatofibroma images to balance training data
4. **Explainability** — Grad-CAM visualizations show which image regions drive predictions
5. **Deployment** — Gradio web interface for interactive diagnosis

## 📊 Dataset

**HAM10000** — "Human Against Machine with 10,000 training images"

| Class | Full Name | Count | Type |
|-------|-----------|-------|------|
| nv | Melanocytic Nevi | ~6,705 | Benign (majority) |
| mel | Melanoma | ~1,113 | **Malignant (rare)** |
| bkl | Benign Keratosis | ~1,099 | Benign |
| bcc | Basal Cell Carcinoma | ~514 | Malignant |
| akiec | Actinic Keratoses | ~327 | Pre-malignant |
| vasc | Vascular Lesions | ~142 | Benign |
| df | Dermatofibroma | ~115 | **Benign (rare)** |

- **Source**: [Kaggle - HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Publication**: Tschandl et al., "The HAM10000 dataset", Scientific Data, 2018

## 🏗 Architecture

### Baseline CNN
4 convolutional blocks with BatchNorm, Dropout, and Global Average Pooling.
```
Conv(3→32) → BN → ReLU → Conv(32→32) → BN → ReLU → MaxPool → Dropout(0.25)
Conv(32→64) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool → Dropout(0.25)
Conv(64→128) → BN → ReLU → Conv(128→128) → BN → ReLU → MaxPool → Dropout(0.25)
Conv(128→256) → BN → ReLU → Conv(256→256) → BN → ReLU → MaxPool → Dropout(0.25)
GlobalAvgPool → FC(256→128) → ReLU → Dropout(0.5) → FC(128→7)
```

### Transfer Learning (EfficientNet-B0)
- Pre-trained on ImageNet
- Two-stage fine-tuning: classifier head first, then full network
- Custom head: Dropout → FC(1280→512) → ReLU → Dropout → FC(512→7)

### Conditional DCGAN
- **Generator**: Latent vector (z=100) + class embedding → 5 transposed conv blocks → 224×224×3 image
- **Discriminator**: Image + class embedding → 5 conv blocks → Real/Fake prediction
- Label smoothing for training stability

## 📈 Results

Results are saved to `results/` directory including:
- Per-class precision, recall, F1-score
- Confusion matrices for all models
- Training curves
- Model comparison charts
- Grad-CAM visualizations

## 🚀 Installation

```bash
# Clone or navigate to the project
cd melanoma-detection-gan

# Install dependencies
pip install -r requirements.txt

# Download dataset
python download_data.py
```

## 💻 Usage

### Run All Phases (Recommended)
```bash
python run_all.py
```
This runs Phases 1-4 sequentially and saves all results.

### Run Individual Phases
```bash
# Phase 1: Baseline CNN
python -m src.train_baseline

# Phase 2: Transfer Learning
python -m src.train_transfer

# Phase 3: GAN + Augmented Retraining
python -m src.train_gan

# Phase 4: Grad-CAM (requires trained model)
python -m src.gradcam
```

### Launch Web Demo (Phase 5)
```bash
python app.py
```
Open `http://localhost:7860` in your browser.

## 📁 Project Structure

```
melanoma-detection-gan/
├── app.py                    # Gradio web interface (Phase 5)
├── run_all.py                # Master script (Phases 1-4)
├── download_data.py          # Dataset download script
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── data/
│   └── ham10000/             # Symlink to HAM10000 dataset
├── src/
│   ├── __init__.py
│   ├── utils.py              # Shared constants and utilities
│   ├── dataset.py            # Data loading and preprocessing
│   ├── baseline_cnn.py       # Custom CNN architecture
│   ├── transfer_model.py     # EfficientNet-B0 transfer model
│   ├── dcgan.py              # DCGAN Generator + Discriminator
│   ├── gradcam.py            # Grad-CAM implementation
│   ├── evaluate.py           # Metrics and visualization
│   ├── train_baseline.py     # Phase 1 training
│   ├── train_transfer.py     # Phase 2 training
│   └── train_gan.py          # Phase 3 training
└── results/                  # All outputs (models, plots, metrics)
    ├── best_baseline.pth
    ├── best_transfer.pth
    ├── best_gan_augmented.pth
    ├── generator.pth
    ├── *_confusion_matrix.png
    ├── *_training_curves.png
    ├── model_comparison.png
    ├── gradcam_transfer/
    ├── gradcam_baseline/
    └── gradcam_gan_augmented/
```

## 🔧 Technical Details

### Training Configuration
- **Image size**: 224×224
- **Batch size**: 32
- **Optimizer**: Adam (lr=1e-3 for baseline, 1e-4 for fine-tuning)
- **Loss**: Class-weighted CrossEntropyLoss
- **Augmentation**: Random flip, rotation, color jitter
- **Early stopping**: Patience=10

### Key Design Decisions
1. **Class-weighted loss** compensates for imbalance without oversampling
2. **Two-stage transfer learning** prevents catastrophic forgetting of pretrained features
3. **Conditional GAN** generates class-specific images rather than random samples
4. **Label smoothing** in GAN training prevents discriminator overconfidence
5. **Stratified splits** ensure class distribution is preserved in train/val/test

### Hardware
- Optimized for Apple Silicon (MPS) with CUDA and CPU fallback
- Training times vary by hardware (expect ~30-60 min total on MPS)

## 📚 Keywords

CNN • Transfer Learning • GANs • Generative Modeling • Class Imbalance • Medical Imaging • Model Evaluation • Precision/Recall/F1 • Explainable AI (Grad-CAM) • PyTorch • Data Augmentation • EfficientNet • DCGAN • Deployment

## ⚠️ Disclaimer

This project is for educational and research purposes only. It is NOT a medical diagnostic tool. Always consult a qualified healthcare professional for medical advice.
