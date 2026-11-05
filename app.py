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

from src.utils import get_device, CLASSES, CLASS_NAMES, IMG_SIZE
from src.transfer_model import SkinLesionTransfer
from src.gradcam import GradCAM, get_target_layer

# Global model and device
device = get_device()
model = None
gradcam_instance = None


def load_model():
    """Load the best available model."""
    global model, gradcam_instance

    model_paths = [
        ('results/best_gan_augmented.pth', 'GAN-Augmented EfficientNet-B0 Model'),
        ('results/best_transfer.pth', 'Transfer Learning (EfficientNet-B0) Model'),
    ]

    for path, name in model_paths:
        if os.path.exists(os.path.join(project_root, path)):
            full_path = os.path.join(project_root, path)
            print(f"Loading {name} from {full_path}...")
            model = SkinLesionTransfer(num_classes=7).to(device)
            model.load_state_dict(
                torch.load(full_path, map_location=device, weights_only=True)
            )
            model.eval()

            # Set up Grad-CAM
            target_layer = get_target_layer(model, 'transfer')
            gradcam_instance = GradCAM(model, target_layer)

            print(f"Model loaded successfully on {device}")
            return name

    print("Warning: Trained model weights not found.")
    return "Demo Mode (Weights Not Loaded)"


# Transform for input images
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict(image):
    """
    Run diagnostic prediction & Grad-CAM explainability on uploaded lesion image.
    """
    if image is None:
        return "⚠️ Please upload or select a skin lesion image.", {}, None

    if model is None:
        return "⚠️ Model weights not available. Please run training pipeline.", {}, None

    # Preprocess
    img_tensor = eval_transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    pred_idx = torch.argmax(probabilities).item()
    pred_code = CLASSES[pred_idx]
    pred_name = CLASS_NAMES[pred_code]
    pred_confidence = float(probabilities[pred_idx])

    # Format probability dictionary
    confidences = {}
    for i, cls in enumerate(CLASSES):
        name_str = f"{CLASS_NAMES[cls]} ({cls})"
        confidences[name_str] = float(probabilities[i])

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
            Model Confidence: <strong style="color: #38bdf8; font-size: 18px;">{pred_confidence:.2%}</strong>
        </div>
    </div>
    """

    # Generate Grad-CAM
    heatmap, _, _ = gradcam_instance.generate_cam(img_tensor, target_class=pred_idx)
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
    sample_dir = os.path.join(project_root, 'data', 'ham10000', 'HAM10000_images_part_1')
    if os.path.exists(sample_dir):
        files = sorted(os.listdir(sample_dir))[:6]
        return [os.path.join(sample_dir, f) for f in files if f.endswith('.jpg')]
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
                <span class="badge">Hardware: Apple Silicon GPU ({device})</span>
                <span class="badge">Dataset: HAM10000 (10,015 images)</span>
                <span class="badge">Framework: PyTorch 2.8 + EfficientNet-B0</span>
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

                gr.Markdown("## 📈 Benchmark Metrics & GAN Impact Analysis")

                # KPI Grid
                gr.HTML("""
                <div class="kpi-grid">
                    <div class="kpi-card">
                        <div class="kpi-value">77.58%</div>
                        <div class="kpi-label">Final Classifier Accuracy</div>
                        <div class="kpi-sub">▲ +32.67% vs Baseline</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value">0.6824</div>
                        <div class="kpi-label">Macro F1-Score</div>
                        <div class="kpi-sub">▲ +35.02% vs Baseline</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value">82.35%</div>
                        <div class="kpi-label">Dermatofibroma Recall</div>
                        <div class="kpi-sub">▲ +70.59% via GAN Augment</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value">10,010</div>
                        <div class="kpi-label">Balanced Dataset Size</div>
                        <div class="kpi-sub">3,000 Synthetic Images</div>
                    </div>
                </div>
                """)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📊 Recall Comparison Across All 3 Models")
                        comp_img_path = os.path.join(project_root, 'results', 'model_comparison.png')
                        if os.path.exists(comp_img_path):
                            gr.Image(comp_img_path, label="Model Comparison (Recall & F1 per Class)")
                        else:
                            gr.Markdown("Run `python run_all.py` to generate plots.")

                    with gr.Column():
                        gr.Markdown("### 🎨 DCGAN Training Loss & Convergence")
                        gan_loss_path = os.path.join(project_root, 'results', 'gan_loss_curves.png')
                        if os.path.exists(gan_loss_path):
                            gr.Image(gan_loss_path, label="Conditional DCGAN Loss Curves")
                        else:
                            gr.Markdown("Run `python run_all.py` to generate plots.")

                gr.Markdown("---")
                gr.Markdown("### 🧩 Confusion Matrices Across Phases")
                with gr.Row():
                    cm1 = os.path.join(project_root, 'results', 'baseline_confusion_matrix.png')
                    cm2 = os.path.join(project_root, 'results', 'transfer_confusion_matrix.png')
                    cm3 = os.path.join(project_root, 'results', 'gan_augmented_confusion_matrix.png')

                    if os.path.exists(cm1):
                        gr.Image(cm1, label="Phase 1: Baseline CNN")
                    if os.path.exists(cm2):
                        gr.Image(cm2, label="Phase 2: EfficientNet-B0")
                    if os.path.exists(cm3):
                        gr.Image(cm3, label="Phase 3: GAN-Augmented")

                gr.Markdown("---")
                gr.Markdown("### 📄 Final Metrics Summary (JSON Output)")
                summary_json_path = os.path.join(project_root, 'results', 'final_summary.json')
                if os.path.exists(summary_json_path):
                    with open(summary_json_path) as f:
                        summary_data = json.load(f)
                    gr.JSON(summary_data, label="Results JSON Summary")

            # ==========================================
            # TAB 3: TECHNICAL DEEP-DIVE & ARCHITECTURE
            # ==========================================
            with gr.TabItem("📖 Technical Architecture & Deep-Dive"):

                gr.Markdown("""
                # 🛠 Deep Learning Architecture & Pipeline Deep-Dive

                This section details every phase of the engineering pipeline, mathematical formulation, model specifications, and portfolio talking points.

                ---

                ## 🎯 1. Clinical Context & The Imbalance Problem

                Skin cancer classification models face a fundamental failure mode in real-world deployment: **severe class imbalance**.
                In the **HAM10000** dataset (10,015 dermatoscopic images), benign Melanocytic Nevi (`nv`) account for **67% of all samples**, while dangerous malignancies like **Melanoma (`mel`)** and rare lesions like **Dermatofibroma (`df`)** represent small fractions of the dataset.

                - **Failure Mode**: A standard Convolutional Neural Network trained on imbalanced data optimizes for global accuracy by predicting the majority class (`nv`), achieving ~70% overall accuracy while **failing to detect 80%+ of cancer cases**.
                - **Engineering Objective**: Fix class imbalance using a **Conditional DCGAN** to synthesize realistic minority class images, balancing the training set and maximizing **Melanoma Recall** without sacrificing overall specificity.

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
                - **Baseline Result**: Accuracy = **44.91%**, F1 = **0.3322**, Melanoma Recall = **73.05%**.

                ---

                ### Phase 2 — Two-Stage Transfer Learning (EfficientNet-B0)
                We leveraged **EfficientNet-B0** pre-trained on ImageNet (20M+ parameters) using compound scaling:

                1. **Stage 1 (Head Fine-Tuning)**: Freeze feature extractor, train custom classification head (`Dense(1280→512) → ReLU → Dense(512→7)`) at $lr = 10^{-3}$ for 8 epochs.
                2. **Stage 2 (Full Fine-Tuning)**: Unfreeze all network parameters and train end-to-end at $lr = 10^{-4}$ with ReduceLROnPlateau.
                - **Result**: Accuracy jumped from **44.91% → 71.92%**, Macro F1 from **0.3322 → 0.6557**.

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
                  - Generated **3,000 synthetic images** (1,500 Melanoma, 1,500 Dermatofibroma) expanding dataset from **7,010 → 10,010 images**.
                - **Result**: Final Retrained Classifier Accuracy reached **77.58%** (+32.67% over baseline), and Dermatofibroma Recall jumped from **11.76% → 82.35%** (+70.59% improvement!).

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
                | **Core Framework** | PyTorch 2.8.0 | Tensor computations, Autograd, Model training |
                | **Vision Library** | torchvision 0.28.0 | EfficientNet-B0 pretrained weights, transforms |
                | **Hardware Acceleration** | Apple Silicon MPS | GPU acceleration on Mac (`mps` device) |
                | **Computer Vision** | OpenCV 4.8 & PIL | Image manipulation, JET colormap overlays |
                | **Data Processing** | Pandas & NumPy | Metadata manipulation, stratified splits |
                | **Metrics & Evaluation** | Scikit-Learn | Confusion matrices, classification reports |
                | **Visualization** | Matplotlib & Seaborn | Loss curves, bar charts, confusion matrix plots |
                | **Web Deployment** | Gradio 6.20.0 | Multi-tab UI, interactive image analysis |

                ---

                ## 🎙 4. Resume & Interview Talking Points

                > **Q: Why not just evaluate model performance using accuracy?**  
                > *"In HAM10000, 67% of images are benign Melanocytic Nevi. A model predicting 'benign' every time gets 67% accuracy but misses 100% of cancers. We prioritized per-class Recall and Macro F1 to ensure high sensitivity on malignant classes."*

                > **Q: Why use a GAN instead of standard data augmentation?**  
                > *"Standard augmentations (flips, rotations, crops) simply replicate existing feature points. A Conditional DCGAN learns the underlying latent distribution of rare lesions and generates novel synthetic samples, filling sparse regions of the feature space."*

                > **Q: How does Grad-CAM improve medical AI adoption?**  
                > *"Clinicians will not trust uninterpretable predictions. Grad-CAM visualizes the specific lesion boundaries, pigment networks, or erythema regions driving the network's classification, enabling clinical verification."*
                """)

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
