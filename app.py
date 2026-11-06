"""
Melanoma Detection AI & GAN Class Balancing Pipeline
Interactive Gradio Web Application with Multi-Tab UI & Detailed Technical Page
"""
import os
import sys
import json
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import gradio as gr

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.constants import CLASSES, CLASS_NAMES, IMG_SIZE
from src.gradcam import GradCAM, get_target_layer
from src.inference import TorchLesionClassifier, ensure_model_dir

# CPU by default (HF Spaces free tier); set APP_DEVICE=mps/cuda locally if wanted
device = torch.device(os.environ.get('APP_DEVICE', 'cpu'))
clf = None
gradcam_instance = None
DISCLAIMER = ("Research demo, not a medical device. Do not use it for diagnosis; "
              "see a dermatologist about any concerning skin lesion. Uploaded images are not stored.")


def _load_json(*parts):
    path = os.path.join(project_root, *parts)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_model():
    """Load the release model (models/, or HF_MODEL_REPO) for inference + Grad-CAM."""
    global clf, gradcam_instance
    try:
        clf = TorchLesionClassifier(ensure_model_dir(), device=str(device))
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        return "Demo Mode (weights not loaded)"
    gradcam_instance = GradCAM(clf.model, get_target_layer(clf.model, 'transfer'))
    info = clf.post.info
    print(f"Loaded {info['variant']} (seed {info['seed']}) on {device}")
    return f"EfficientNet-B0 ({info['variant']}, seed {info['seed']})"


def predict(image):
    """Calibrated prediction + out-of-distribution check + Grad-CAM for one uploaded image."""
    if image is None:
        return "⚠️ Please upload or select a skin lesion image.", {}, None
    if clf is None:
        return "⚠️ Model weights not available. Run scripts/build_release.py or set HF_MODEL_REPO.", {}, None

    try:
        res = clf.predict(image)
    except ValueError:
        return "⚠️ Image is too large. Please upload a smaller photo.", {}, None

    if not res['in_distribution']:
        return f"""
        <div style="background: rgba(17, 24, 39, 0.8); border: 1px solid #f59e0b; border-radius: 12px; padding: 18px;">
            <div style="font-size: 18px; font-weight: 700; color: #fbbf24; margin-bottom: 6px;">Image rejected: not a dermoscopic lesion</div>
            <div style="color: #d1d5db;">This image is far from anything the model was trained on
            (OOD score {res['ood_score']:.0f} &gt; threshold {res['ood_threshold']:.0f}), so no prediction is shown.
            Upload a close-up dermoscopy image of a single skin lesion.</div>
        </div>""", {}, None

    pred_code = res['prediction']['code']
    pred_idx = CLASSES.index(pred_code)
    pred_name = CLASS_NAMES[pred_code]
    pred_confidence = res['prediction']['probability']
    confidences = {f"{CLASS_NAMES[c]} ({c})": p for c, p in res['probabilities'].items()}

    # Risk badge formatting
    risk_color = "#ef4444" if pred_code in ['mel', 'bcc', 'akiec'] else "#10b981"
    risk_label = "MALIGNANT / HIGH RISK" if pred_code in ['mel', 'bcc'] else ("PRE-MALIGNANT" if pred_code == 'akiec' else "BENIGN / LOW RISK")

    summary_html = f"""
    <div style="background: rgba(17, 24, 39, 0.8); border: 1px solid {risk_color}; border-radius: 12px; padding: 18px; margin-bottom: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.4);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #9ca3af;">Primary Prediction</span>
            <span style="background: {risk_color}; color: #ffffff; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 700; letter-spacing: 0.5px;">{risk_label}</span>
        </div>
        <div style="font-size: 26px; font-weight: 800; color: #ffffff; margin-bottom: 4px;">
            {pred_name} <span style="font-size: 18px; font-weight: 500; color: #a78bfa;">({pred_code.upper()})</span>
        </div>
        <div style="font-size: 15px; color: #d1d5db;">
            Calibrated probability: <strong style="color: #38bdf8; font-size: 18px;">{pred_confidence:.2%}</strong>
        </div>
        <div style="font-size: 12px; color: #9ca3af; margin-top: 8px;">{DISCLAIMER}</div>
    </div>
    """

    # Generate Grad-CAM
    heatmap, _, _ = gradcam_instance.generate_cam(clf.tensor(image), target_class=pred_idx)
    fig = create_gradcam_figure(image, heatmap, pred_idx, pred_confidence)

    return summary_html, confidences, fig


def create_gradcam_figure(original_image, heatmap, pred_class, confidence):
    """Create a side-by-side Grad-CAM visualization figure with custom dark styling."""
    img = original_image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0

    # Colorize heatmap
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = np.float32(heatmap_color[:, :, ::-1]) / 255.0

    # Visual overlay
    overlay = 0.55 * img_array + 0.45 * heatmap_color
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#0f172a')

    for ax in axes:
        ax.set_facecolor('#0f172a')

    axes[0].imshow(img_array)
    axes[0].set_title('Original Dermatoscopic Image', color='#94a3b8', fontsize=13, fontweight='600', pad=10)
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Attention Heatmap', color='#38bdf8', fontsize=13, fontweight='600', pad=10)
    axes[1].axis('off')

    pred_name = CLASS_NAMES[CLASSES[pred_class]]
    axes[2].imshow(overlay)
    axes[2].set_title(f'Attention Overlay ({pred_name})', color='#a78bfa', fontsize=13, fontweight='600', pad=10)
    axes[2].axis('off')

    plt.tight_layout()
    return fig


# Helper to get sample image paths
def get_sample_images():
    """One held-out test image per class (HAM10000, CC BY-NC 4.0; see assets/samples/NOTICE.md)."""
    sample_dir = os.path.join(project_root, 'assets', 'samples')
    if os.path.isdir(sample_dir):
        return [os.path.join(sample_dir, f) for f in sorted(os.listdir(sample_dir)) if f.endswith('.jpg')]
    return []


# Custom CSS for modern glassmorphic theme
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800&display=swap');

.gradio-container {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #1e1b4b 100%) !important;
    min-height: 100vh;
    color: #f3f4f6 !important;
}

/* Glassmorphism Card Panels */
.glass-card {
    background: rgba(17, 24, 39, 0.75) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
    margin-bottom: 20px !important;
}

/* Header Styling */
.app-header {
    text-align: center;
    padding: 30px 20px;
    background: linear-gradient(180deg, rgba(30, 27, 75, 0.6) 0%, rgba(15, 23, 42, 0.4) 100%);
    border-bottom: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 20px;
    margin-bottom: 25px;
}

.app-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #a78bfa 0%, #38bdf8 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px !important;
}

.app-subtitle {
    font-size: 1.15rem;
    color: #94a3b8;
    max-width: 800px;
    margin: 0 auto 15px auto;
}

/* Badges */
.badge-container {
    display: flex;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
}

.badge {
    background: rgba(139, 92, 246, 0.15);
    border: 1px solid rgba(139, 92, 246, 0.4);
    color: #c4b5fd;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}

.badge-green {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #6ee7b7;
}

/* Custom Buttons */
.btn-primary {
    background: linear-gradient(135deg, #7c3aed 0%, #2563eb 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    padding: 14px 28px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4) !important;
    transition: all 0.3s ease !important;
}

.btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.6) !important;
}

/* Tabs styling */
.tabs {
    border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
}

.tab-nav button {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    color: #94a3b8 !important;
    border-radius: 10px 10px 0 0 !important;
}

.tab-nav button.selected {
    color: #38bdf8 !important;
    background: rgba(56, 189, 248, 0.1) !important;
    border-bottom: 3px solid #38bdf8 !important;
}

/* KPI Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}

.kpi-card {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}

.kpi-value {
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #38bdf8;
}

.kpi-label {
    font-size: 0.9rem;
    color: #94a3b8;
    font-weight: 500;
    margin-top: 4px;
}

.kpi-sub {
    font-size: 0.8rem;
    color: #34d399;
    font-weight: 600;
    margin-top: 2px;
}
"""


def results_markdown():
    """Results tab content, generated from committed result files (never hard-coded)."""
    out = ["## 📈 Results (lesion-level split, held-out test set)"]
    leak = _load_json('results', 'leakage_report.json')
    if leak:
        v1 = leak['v1_image_split']
        out.append(f"**Data leakage fixed:** the original image-level split had "
                   f"**{v1['test_images_whose_lesion_is_in_train']} of {v1['test_images']} test images "
                   f"({v1['pct_test_images_leaked']}%)** sharing a lesion with training. "
                   f"All numbers below use a lesion-level split with 0 overlap.")
    summary = os.path.join(project_root, 'results', 'ablation', 'summary.md')
    if os.path.exists(summary):
        with open(summary) as f:
            out.append(f.read().replace('# Ablation results', '### Imbalance-strategy ablation', 1))
    info = clf.post.info if clf else _load_json('models', 'model_info.json')
    if info:
        c, o = info['calibration'], info['ood']
        out.append("### Deployed model")
        out.append(f"- **Model:** EfficientNet-B0, `{info['variant']}` variant, seed {info['seed']}")
        out.append(f"- **Calibration:** temperature scaling (T = {c['temperature']:.2f}), "
                   f"expected calibration error {c['test_ece_before']:.3f} → {c['test_ece_after']:.3f}")
        sets = ', '.join(f"{k.replace('_', ' ')} {v['rejection_rate']:.0%}" for k, v in o['sets'].items())
        out.append(f"- **Out-of-distribution rejection** (Mahalanobis on penultimate features): "
                   f"rejects {sets}; wrongly rejects {o['test_lesion_false_rejection_rate']:.1%} of real test lesions")
        e = info.get('engines', {})
        if e:
            rows = ["| Engine | Size (MB) | Test accuracy | Macro AUROC | CPU p50 latency (ms) |", "|---|---|---|---|---|"]
            for n, v in e.items():
                rows.append(f"| {n} | {v['size_mb']} | {v['test']['accuracy']:.3f} | {v['test']['macro_auroc']:.3f} | "
                            f"{v['latency_batch1']['p50_ms']:.1f} |")
            out.append("\n".join(rows))
    return "\n\n".join(out)


def build_app():
    """Build the multi-tab Gradio UI."""
    active_model_name = load_model()
    sample_imgs = get_sample_images()

    with gr.Blocks(css=CUSTOM_CSS, title="Melanoma Detection AI & GAN Class Balancing") as demo:

        # Header Section
        gr.HTML(f"""
        <div class="app-header">
            <h1 class="app-title">🔬 Melanoma Detection AI</h1>
            <p class="app-subtitle">
                Deep Learning Skin Cancer Diagnostic Suite with <strong>DCGAN Class Balancing</strong> & <strong>Grad-CAM Explainable AI</strong>
            </p>
            <div class="badge-container">
                <span class="badge badge-green">● Active Model: {active_model_name}</span>
                <span class="badge">Inference: {str(device).upper()}</span>
                <span class="badge">Dataset: HAM10000 (10,015 images, 7,470 lesions)</span>
                <span class="badge">Framework: PyTorch {torch.__version__.split('+')[0]} + EfficientNet-B0</span>
            </div>
            <div style="margin-top: 10px; font-size: 13px; color: #fbbf24;">⚠️ {DISCLAIMER}
            </div>
        </div>
        """)

        with gr.Tabs(elem_classes="tabs"):

            # ==========================================
            # TAB 1: DIAGNOSTIC ANALYZER & GRAD-CAM
            # ==========================================
            with gr.TabItem("🔍 Live Diagnostic Analyzer & Grad-CAM"):

                with gr.Row():

                    # Left Column: Input & Controls
                    with gr.Column(scale=5):
                        gr.Markdown("### 📸 1. Input Lesion Image")

                        input_image = gr.Image(
                            type="pil",
                            label="Upload or Select Lesion Image",
                            height=320
                        )

                        analyze_btn = gr.Button("🔍 Run AI Diagnostic Analysis", variant="primary", elem_classes="btn-primary")

                        if sample_imgs:
                            gr.Markdown("### 🧪 Quick Test Examples")
                            gr.Examples(
                                examples=sample_imgs,
                                inputs=input_image,
                                label="Click any sample to load:"
                            )

                        gr.Markdown("""
                        ---
                        ### 📋 Diagnostic Categories Reference
                        | Class Code | Category Name | Severity Risk |
                        | :--- | :--- | :--- |
                        | **mel** | **Melanoma** | ⚠️ High Risk Malignant |
                        | **bcc** | Basal Cell Carcinoma | ⚠️ High Risk Malignant |
                        | **akiec** | Actinic Keratoses | 🟡 Pre-Malignant |
                        | **bkl** | Benign Keratosis | 🟢 Benign |
                        | **df** | Dermatofibroma | 🟢 Benign |
                        | **nv** | Melanocytic Nevi | 🟢 Benign |
                        | **vasc** | Vascular Lesions | 🟢 Benign |
                        """)

                    # Right Column: Diagnostic Output & Visualizations
                    with gr.Column(scale=7):
                        gr.Markdown("### 📊 2. AI Diagnostic Results")

                        output_summary = gr.HTML(
                            """
                            <div style="background: rgba(17, 24, 39, 0.5); border: 1px dashed rgba(255,255,255,0.2); border-radius: 12px; padding: 30px; text-align: center; color: #9ca3af;">
                                👈 Upload or select an image on the left and click <strong>Run AI Diagnostic Analysis</strong>.
                            </div>
                            """
                        )

                        output_label = gr.Label(
                            label="Diagnostic Class Probabilities (7 Categories)",
                            num_top_classes=7
                        )

                        gr.Markdown("### 🔥 3. Grad-CAM Explainable AI Heatmap")
                        output_gradcam = gr.Plot(
                            label="Attention Heatmap Overlay (Shows pixels driving prediction)"
                        )

                analyze_btn.click(
                    fn=predict,
                    inputs=input_image,
                    outputs=[output_summary, output_label, output_gradcam]
                )

            # ==========================================
            # TAB 2: EMPIRICAL RESULTS & BENCHMARKS
            # ==========================================
            with gr.TabItem("📊 Empirical Results & Model Comparison"):

                gr.Markdown(results_markdown())

            # ==========================================
            # TAB 3: TECHNICAL DEEP-DIVE & ARCHITECTURE
            # ==========================================
            with gr.TabItem("📖 Technical Architecture & Deep-Dive"):

                gr.Markdown("""
                # 🛠 Deep Learning Architecture & Pipeline Deep-Dive

                This section details every phase of the engineering pipeline, mathematical formulation, and model specifications.

                ---

                ## 🎯 1. Clinical Context & The Imbalance Problem

                Skin cancer classification models face a fundamental failure mode in real-world deployment: **severe class imbalance**.
                In the **HAM10000** dataset (10,015 dermatoscopic images), benign Melanocytic Nevi (`nv`) account for **67% of all samples**, while dangerous malignancies like **Melanoma (`mel`)** and rare lesions like **Dermatofibroma (`df`)** represent small fractions of the dataset.

                - **Failure Mode**: A standard Convolutional Neural Network trained on imbalanced data optimizes for global accuracy by predicting the majority class (`nv`), looking accurate overall while missing many of the cancers that matter most.
                - **Engineering Objective**: Compare imbalance strategies (class-weighted loss, oversampling, and **Conditional DCGAN** synthetic images) under one controlled recipe, judged on balanced accuracy, AUROC, and melanoma sensitivity.
                - **Leakage-free evaluation**: HAM10000 has ~10k images of 7,470 lesions. A naive image-level split put photos of the same lesion in train and test (36% of test images in the original version of this project). All results here use a **lesion-level split**.

                ---

                ## 🏗 2. Phase-by-Phase Technical Breakdown

                ### Phase 1 — Custom Baseline CNN (Built from Scratch)
                To establish an empirical baseline, we engineered a custom 4-block PyTorch CNN from first principles rather than importing a pre-existing architecture.

                ```
                Input (224x224x3)
                   │
                [Conv(3→32, k=3, p=1) → BN → ReLU → Conv(32→32, k=3, p=1) → BN → ReLU → MaxPool(2) → Dropout(0.25)]
                   │
                [Conv(32→64, k=3, p=1) → BN → ReLU → Conv(64→64, k=3, p=1) → BN → ReLU → MaxPool(2) → Dropout(0.25)]
                   │
                [Conv(64→128, k=3, p=1) → BN → ReLU → Conv(128→128, k=3, p=1) → BN → ReLU → MaxPool(2) → Dropout(0.25)]
                   │
                [Conv(128→256, k=3, p=1) → BN → ReLU → Conv(256→256, k=3, p=1) → BN → ReLU → MaxPool(2) → Dropout(0.25)]
                   │
                Global Average Pooling (1x1) ──► Dense(256 → 128) ──► ReLU ──► Dropout(0.5) ──► Dense(128 → 7)
                ```
                - **Loss Function**: Inverse-frequency class-weighted CrossEntropyLoss:
                  $$\\mathcal{L}_{CE} = - \\sum_{c=1}^C w_c \\cdot y_c \\log(\\hat{y}_c), \\quad w_c = \\frac{N}{C \\cdot N_c}$$
                - **Result**: see the Results tab (`baseline_cnn` row).

                ---

                ### Phase 2 — Two-Stage Transfer Learning (EfficientNet-B0)
                We fine-tune **EfficientNet-B0** pre-trained on ImageNet (~4.7M parameters, compound scaling):

                1. **Stage 1 (Head Fine-Tuning)**: Freeze feature extractor, train the last blocks and a custom head (`Dense(1280→512) → ReLU → Dense(512→7)`) at $lr = 10^{-3}$ for 3 epochs.
                2. **Stage 2 (Full Fine-Tuning)**: Unfreeze all network parameters and train end-to-end at $lr = 10^{-4}$ with ReduceLROnPlateau.
                - Checkpoints are selected on **validation balanced accuracy** (comparable across weighted and unweighted losses).

                ---

                ### Phase 3 — Conditional DCGAN for Synthetic Augmentation
                We designed a **Conditional Deep Convolutional Generative Adversarial Network (cDCGAN)** to generate synthetic 224x224 dermatoscopic images for minority classes (`mel` and `df`).

                - **Generator Architecture**:
                  - Noise vector $z \\in \\mathbb{R}^{100}$ concatenated with Class Embedding $e \\in \\mathbb{R}^{10}$.
                  - Dense projection to $512 \\times 7 \\times 7$.
                  - 5 Transposed Convolutional Blocks (`ConvTranspose2d → BatchNorm → ReLU`) upsampling to $224 \\times 224 \\times 3$ with Tanh activation.
                - **Discriminator Architecture**:
                  - Image $X \\in \\mathbb{R}^{224 \\times 224 \\times 3}$ concatenated with spatial class embedding layer $\\in \\mathbb{R}^{224 \\times 224 \\times 1}$.
                  - 5 Convolutional Blocks (`Conv2d → BatchNorm → LeakyReLU(0.2) → Dropout(0.3)`) downsampling to $7 \\times 7 \\times 512$.
                  - Sigmoid classification head outputting Real/Fake probability.
                - **Training Enhancements**:
                  - One-sided label smoothing ($y_{real} = 0.9, y_{fake} = 0.1$) to prevent discriminator overconfidence.
                  - Generates **3,000 synthetic images** (1,500 Melanoma, 1,500 Dermatofibroma), converted to pixels and passed through the same augmentation + normalization as real images.
                  - Image quality measured with **FID / KID** against a real-vs-real reference.
                - **Result**: see the Results tab (`dcgan` vs `weighted_loss` isolates the GAN's effect).

                ---

                ### Phase 4 — Grad-CAM Explainable AI (XAI)
                To prevent "black-box" decisions in clinical settings, we implemented **Gradient-weighted Class Activation Mapping (Grad-CAM)**:

                1. Compute gradient of score for class $c$ ($y^c$) with respect to feature activation maps $A^k$ of the final conv layer:
                   $$\\alpha_k^c = \\frac{1}{Z} \\sum_{i} \\sum_{j} \\frac{\\partial y^c}{\\partial A_{i,j}^k}$$
                2. Calculate weighted combination of forward activation maps and apply ReLU:
                   $$L_{\\text{Grad-CAM}}^c = \\text{ReLU}\\left(\\sum_k \\alpha_k^c A^k\\right)$$
                3. Upsample heatmap to $224 \\times 224$ and overlay with JET colormap.

                ---

                ## 💻 3. Technology Stack

                | Category | Technology | Usage |
                | :--- | :--- | :--- |
                | **Core Framework** | PyTorch 2.13 | Tensor computations, Autograd, Model training |
                | **Serving** | ONNX Runtime + FastAPI | CPU inference API (`api/main.py`), Docker image |
                | **Vision Library** | torchvision 0.28.0 | EfficientNet-B0 pretrained weights, transforms |
                | **Hardware Acceleration** | Apple Silicon MPS | GPU acceleration on Mac (`mps` device) |
                | **Computer Vision** | PIL & OpenCV | Image resizing; Grad-CAM JET colormap overlays |
                | **Data Processing** | Pandas & NumPy | Metadata, lesion-level stratified splits |
                | **Metrics & Evaluation** | Scikit-Learn | AUROC, balanced accuracy, bootstrap CIs, Ledoit-Wolf OOD covariance |
                | **Visualization** | Matplotlib & Seaborn | Loss curves, bar charts, confusion matrix plots |
                | **Web Deployment** | Gradio 6.20.0 | Multi-tab UI, interactive image analysis |

                """)

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(share=False, server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
                max_file_size="10mb")
