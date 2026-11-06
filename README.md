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

## 🧪 v2: Leakage-free evaluation & controlled ablation

The v1 numbers in `results/*.json` used an **image-level** split. HAM10000 has ~10k
images of ~7.5k lesions, so photos of the same lesion landed in both train and test,
which inflates test scores (`python -m experiments.leakage_report` quantifies this).
v2 fixes the methodology:

| | v1 | v2 |
|---|---|---|
| Split | random by image | **stratified by `lesion_id`** (no lesion in two splits; unit-tested) |
| Comparison | different recipe per model | **identical recipe**, only the imbalance strategy changes |
| Seeds | 1 | **3 training seeds**, mean ± std |
| Metrics | accuracy, macro F1 | + **balanced accuracy, macro AUROC, melanoma sensitivity @ 90% specificity**, lesion-level **bootstrap 95% CIs** |
| Synthetic data | raw Tanh output in [-1, 1] fed to an ImageNet-normalized classifier | converted to pixels → same augmentation + normalization as real images |
| GAN quality | not measured | **FID / KID** per class vs. a real-vs-real reference |

Variants (`experiments/run_ablation.py`): `plain` (unweighted CE), `weighted_loss`
(class-weighted CE), `oversample` (class-balanced sampler), `dcgan` (weighted CE +
3,000 synthetic mel/df images), `baseline_cnn` (custom CNN reference). `dcgan` vs.
`weighted_loss` isolates the effect of the GAN.

```bash
python download_data.py
python -m experiments.leakage_report            # v1 leakage stats -> results/leakage_report.json
python -m experiments.run_ablation --smoke      # 1-minute pipeline check
python -m experiments.run_ablation              # full run -> results/ablation/summary.md
python -m pytest                                # split / metric / GAN unit tests
```

Model selection uses validation balanced accuracy for every variant (validation loss
is not comparable across weighted and unweighted losses).

## 🚀 Deployment

```
trained run ──► scripts/build_release.py ──► models/
                 • temperature scaling (calibration)        model_info.json  (committed)
                 • Mahalanobis OOD detector                  ood_stats.npz
                 • ONNX export (+ INT8 attempt)              classifier_fp32.onnx / classifier.pth
                 • accuracy + CPU latency per engine
models/ ──► api/main.py   FastAPI + ONNX Runtime (Docker, no PyTorch)
        └─► app.py        Gradio UI + Grad-CAM (Hugging Face Spaces)
```

**Safeguards for public use:** calibrated probabilities; rejection of non-lesion images
(Mahalanobis distance on penultimate features, threshold set on validation lesions);
10 MB upload limit; per-IP rate limiting; images processed in memory and never stored;
disclaimer on every response. Full numbers: `results/release_report.json`.

```bash
# 1. Build release artifacts from a trained run
python -m scripts.build_release --run results/ablation/<variant>/seed0 --ood-dir <folder of non-lesion photos>

# 2a. REST API (ONNX Runtime)
uvicorn api.main:app --port 8000            # or: docker build -t melanoma-api . && docker run -p 8000:8000 melanoma-api
curl -F "file=@assets/samples/mel_ISIC_0024367.jpg;type=image/jpeg" localhost:8000/predict

# 2b. Gradio demo -> Hugging Face Spaces
python app.py                                # local
bash scripts/make_space.sh && hf upload <user>/melanoma-detection build/hf_space . --repo-type space
```

CI (`.github/workflows/ci.yml`) runs the split, metric, GAN and API tests on every push.

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
