"""
Melanoma Detection AI & GAN Class Balancing Pipeline
Interactive Gradio Web Application with Multi-Tab UI & Detailed Technical Page
"""
import os
import sys
import json
import io
import base64
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.constants import CLASSES, CLASS_NAMES, IMG_SIZE
from src.gradcam import GradCAM, get_target_layer
from src.inference import RISK, TorchLesionClassifier, ensure_model_dir

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


RISK_LABEL = {'malignant': 'Malignant class', 'pre-malignant': 'Pre-malignant class', 'benign': 'Benign class'}


def _png_uri(arr_or_img) -> str:
    img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def _note(title, body):
    return f'<div class="mx-card mx-note"><div class="mx-note-title">{title}</div><p>{body}</p></div>'


EMPTY_RESULT = _note('No image analyzed yet',
                     'Upload a dermoscopy image or pick one of the held-out test images, then select Analyze image')
EMPTY_PLATE = ''


def _tag(res, image):
    w, h = image.size
    return (f"{w}×{h} → 224×224 · T = {clf.post.temperature:.2f} · "
            f"OOD {res['ood_score']:.0f} / {res['ood_threshold']:.0f} · {res['latency_ms']:.0f} ms")


def result_html(res, image):
    """Ranked, calibrated class probabilities. The top class carries the accent."""
    code = res['prediction']['code']
    ranked = sorted(res['probabilities'].items(), key=lambda kv: kv[1], reverse=True)
    rows = []
    for i, (c, prob) in enumerate(ranked):
        rows.append(
            f'<li class="mx-row{" is-top" if i == 0 else ""}">'
            f'<span class="mx-dot risk-{RISK[c]}"></span>'
            f'<span class="mx-cls">{CLASS_NAMES[c]}</span><span class="mx-code">{c}</span>'
            f'<span class="mx-track"><span class="mx-fill" style="width:{max(prob * 100, 0.5):.1f}%"></span></span>'
            f'<span class="mx-pct">{prob * 100:.1f}%</span></li>')
    (_, p1), (c2, p2) = ranked[0], ranked[1]
    close = (f'<p class="mx-close">Close call: {CLASS_NAMES[c2]} is {(p1 - p2) * 100:.1f} points behind</p>'
             if p1 - p2 < 0.10 else '')
    return f"""
    <div class="mx-card">
      <div class="mx-kicker">Most likely</div>
      <div class="mx-head">
        <div class="mx-title">{CLASS_NAMES[code]}</div>
        <div class="mx-big">{res['prediction']['probability'] * 100:.1f}%</div>
      </div>
      <div class="mx-sub"><span class="mx-dot risk-{RISK[code]}"></span>{RISK_LABEL[RISK[code]]} · calibrated probability</div>{close}
      <ol class="mx-list">{''.join(rows)}</ol>
      <div class="mx-tag">{_tag(res, image)}</div>
    </div>"""


def plate_html(image, heatmap, pred_code):
    """Specimen plate: input, Grad-CAM heatmap and overlay at the model's 224×224 input size."""
    img = np.array(image.convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    heat = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_INFERNO)[:, :, ::-1]
    overlay = np.clip(0.55 * img + 0.45 * heat, 0, 255).astype(np.uint8)
    cells = [('Input', img), ('Grad-CAM', heat), (f'Overlay · {CLASS_NAMES[pred_code]}', overlay)]
    figs = ''.join(f'<figure><img src="{_png_uri(a)}" alt="{t}"><figcaption>{t}</figcaption></figure>'
                   for t, a in cells)
    return f"""
    <div class="mx-card">
      <div class="mx-kicker">Where the model looked</div>
      <div class="mx-plate">{figs}</div>
      <p class="mx-caption">Brighter regions contributed more to the predicted class. Grad-CAM on the last
      convolutional block of EfficientNet-B0, upsampled from 7×7</p>
    </div>"""


def predict(image):
    """Calibrated prediction + out-of-distribution check + Grad-CAM for one uploaded image."""
    if image is None:
        return EMPTY_RESULT, EMPTY_PLATE
    if clf is None:
        return _note('Model not loaded', 'Run scripts/build_release.py or set HF_MODEL_REPO'), EMPTY_PLATE
    try:
        res = clf.predict(image)
    except ValueError:
        return _note('Image too large', 'Upload a photo under 40 megapixels'), EMPTY_PLATE

    if not res['in_distribution']:
        return _note('Not a dermoscopy image',
                     'This image is far from anything the model was trained on, so no prediction is shown. '
                     'Upload a close-up dermoscopy image of a single skin lesion.'
                     f'<span class="mx-tag">{_tag(res, image)}</span>'), EMPTY_PLATE

    code = res['prediction']['code']
    heatmap, _, _ = gradcam_instance.generate_cam(clf.tensor(image), target_class=CLASSES.index(code))
    return result_html(res, image), plate_html(image, heatmap, code)


# Helper to get sample image paths
def get_sample_images():
    """One held-out test image per class (HAM10000, CC BY-NC 4.0; see assets/samples/NOTICE.md)."""
    sample_dir = os.path.join(project_root, 'assets', 'samples')
    if os.path.isdir(sample_dir):
        return [os.path.join(sample_dir, f) for f in sorted(os.listdir(sample_dir)) if f.endswith('.jpg')]
    return []


def class_reference_html():
    rows = ''.join(f'<li><span class="mx-dot risk-{RISK[c]}"></span><span class="mx-cls">{CLASS_NAMES[c]}</span>'
                   f'<span class="mx-code">{c}</span><span class="mx-risk">{RISK_LABEL[RISK[c]].replace(" class", "")}</span></li>'
                   for c in ['mel', 'bcc', 'akiec', 'bkl', 'df', 'nv', 'vasc'])
    return f'<div class="mx-ref"><div class="mx-kicker">Classes</div><ul>{rows}</ul></div>'


# Light clinical theme: tinted neutral ramp, one ink-teal accent, IBM Plex
INK = '#0f6b78'
THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont('IBM Plex Sans'), 'system-ui', 'sans-serif'],
    font_mono=[gr.themes.GoogleFont('IBM Plex Mono'), 'ui-monospace', 'monospace'],
    radius_size=gr.themes.sizes.radius_md,
)
# Same values for light and dark so the clinical light palette always applies
THEME.set(**{k + sfx: v for k, v in {
    'body_background_fill': '#fafaf9',
    'body_text_color': '#1c1917',
    'body_text_color_subdued': '#57534e',
    'background_fill_primary': '#ffffff',
    'background_fill_secondary': '#f5f5f4',
    'block_background_fill': '#ffffff',
    'block_border_color': 'rgba(28, 25, 23, 0.08)',
    'block_border_width': '1px',
    'block_shadow': 'none',
    'block_radius': '12px',
    'block_label_text_color': '#57534e',
    'block_label_background_fill': '#ffffff',
    'block_title_text_color': '#57534e',
    'border_color_primary': 'rgba(28, 25, 23, 0.08)',
    'border_color_accent': INK,
    'color_accent': INK,
    'color_accent_soft': 'rgba(15, 107, 120, 0.08)',
    'input_background_fill': '#ffffff',
    'button_primary_background_fill': INK,
    'button_primary_background_fill_hover': '#0c5a65',
    'button_primary_text_color': '#ffffff',
    'button_primary_border_color': INK,
    'button_secondary_background_fill': '#f5f5f4',
    'button_secondary_background_fill_hover': '#ecebe9',
    'button_secondary_text_color': '#1c1917',
    'link_text_color': INK,
    'table_border_color': 'rgba(28, 25, 23, 0.08)',
    'table_even_background_fill': '#ffffff',
    'table_odd_background_fill': '#fafaf9',
}.items() for sfx in ('', '_dark') if hasattr(THEME, k + sfx)})

CUSTOM_CSS = """
:root {
    --mx-bg: #fafaf9; --mx-card: #ffffff; --mx-raised: #f5f5f4;
    --mx-line: rgba(28, 25, 23, 0.08);
    --mx-text: #1c1917; --mx-text-2: #57534e; --mx-text-3: #78716c;
    --mx-ink: #0f6b78; --mx-ease: cubic-bezier(0.22, 1, 0.36, 1);
    --mx-mal: #c2410c; --mx-pre: #b7791f; --mx-ben: #6b8f71;
}
body .gradio-container { width: 100% !important; max-width: 1440px !important; margin: 0 auto !important; padding: 0 !important; }
body .gradio-container .main.fillable { padding: 0 clamp(16px, 4vw, 32px) 64px !important; }  /* clamp: Gradio rewrites selectors inside @media */
body .gradio-container .html-container { padding: 0 !important; }
footer { display: none !important; }

/* Header */
.mx-header { padding: clamp(24px, 4vw, 32px) 0 24px; border-bottom: 1px solid var(--mx-line); margin-bottom: 8px; }
.mx-brand { display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap; }
.mx-name { font-size: 18px; font-weight: 600; color: var(--mx-text); }
.mx-meta, .mx-code, .mx-pct, .mx-tag, .mx-big { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-variant-numeric: tabular-nums; }
.mx-meta { font-size: 12px; color: var(--mx-text-3); }
.mx-lede { margin: 12px 0 0; font-size: 20px; line-height: 30px; color: var(--mx-text); max-width: 44rem; font-weight: 400; }
.mx-disclaimer { margin: 12px 0 0; font-size: 13px; line-height: 20px; color: var(--mx-text-2); max-width: 44rem; }

/* Tabs */
.tab-nav button, button[role="tab"] { font-size: 14px !important; font-weight: 500 !important; color: var(--mx-text-2) !important; padding: 8px 16px !important; }
.tab-nav button.selected, button[role="tab"][aria-selected="true"] { color: var(--mx-text) !important; }

/* Cards */
.mx-card { background: var(--mx-card); border: 1px solid var(--mx-line); border-radius: 12px; padding: 24px; }
.mx-kicker { font-size: 13px; color: var(--mx-text-3); margin-bottom: 8px; }
.mx-head { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; }
.mx-title { font-size: 32px; line-height: 1.15; font-weight: 500; color: var(--mx-text); }
.mx-big { font-size: 32px; line-height: 1; font-weight: 500; color: var(--mx-ink); }
.mx-close { margin: 12px 0 0; padding: 8px 12px; border-radius: 8px; background: var(--mx-raised); font-size: 14px; color: var(--mx-text); }
.mx-sub { margin-top: 8px; font-size: 14px; color: var(--mx-text-2); display: flex; align-items: center; gap: 8px; }

/* Ranked probabilities */
.mx-list { list-style: none; margin: 24px 0 0; padding: 0; display: flex; flex-direction: column; gap: 4px; }
.mx-row { display: grid; grid-template-columns: 8px 172px 48px 1fr 56px; align-items: center; gap: 12px; height: 32px; font-size: 14px; color: var(--mx-text-2); }
.mx-row.is-top { color: var(--mx-text); font-weight: 500; }
.mx-code { font-size: 12px; color: var(--mx-text-3); }
.mx-track { height: 6px; border-radius: 4px; background: var(--mx-raised); overflow: hidden; }
.mx-fill { display: block; height: 100%; border-radius: 4px; background: #d6d3d1; transition: width 300ms var(--mx-ease); }
.mx-row.is-top .mx-fill { background: var(--mx-ink); }
.mx-pct { font-size: 13px; text-align: right; }
.mx-dot { width: 8px; height: 8px; border-radius: 4px; display: inline-block; flex: none; }
.risk-malignant { background: var(--mx-mal); } .risk-pre-malignant { background: var(--mx-pre); } .risk-benign { background: var(--mx-ben); }

/* Signature: specimen tag */
.mx-tag { display: block; margin-top: 20px; padding-top: 12px; border-top: 1px solid var(--mx-line); font-size: 12px; color: var(--mx-text-3); }

/* Specimen plate */
.mx-plate { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.mx-plate figure { margin: 0; }
.mx-plate img { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 8px; display: block; background: var(--mx-raised); }
.mx-plate figcaption { margin-top: 8px; font-size: 13px; color: var(--mx-text-2); }
.mx-caption { margin: 16px 0 0; font-size: 13px; line-height: 20px; color: var(--mx-text-3); max-width: 40rem; }

/* Notes and empty states */
.mx-note-title { font-size: 16px; font-weight: 500; color: var(--mx-text); }
.mx-note p { margin: 8px 0 0; font-size: 14px; line-height: 22px; color: var(--mx-text-2); }

/* Class reference */
.mx-ref ul { list-style: none; margin: 0; padding: 0; }
.mx-ref li { display: grid; grid-template-columns: 8px 1fr 48px 96px; align-items: center; gap: 12px; height: 32px; font-size: 14px; color: var(--mx-text); }
.mx-risk { font-size: 13px; color: var(--mx-text-3); text-align: right; }

/* Primary action */
.mx-analyze { min-height: 44px !important; font-size: 15px !important; font-weight: 500 !important; border-radius: 10px !important;
    transition: background-color 150ms var(--mx-ease), transform 150ms var(--mx-ease) !important; }
.mx-analyze:active { transform: scale(0.98); }
button:focus-visible { outline: 2px solid var(--mx-ink) !important; outline-offset: 2px; }

/* Markdown tabs (results, method) */
.mx-prose { max-width: 960px; }
.mx-prose h2 { font-size: 24px !important; font-weight: 500 !important; margin: 32px 0 12px !important; }
.mx-prose h3 { font-size: 18px !important; font-weight: 500 !important; margin: 32px 0 8px !important; }
.mx-prose h1 { font-size: 28px !important; font-weight: 500 !important; }
.mx-prose p, .mx-prose li { font-size: 15px; line-height: 24px; color: var(--mx-text-2); }
.mx-prose strong { color: var(--mx-text); font-weight: 600; }
.mx-prose table { font-size: 13px; font-variant-numeric: tabular-nums; border-collapse: collapse !important; border: 1px solid var(--mx-line) !important; }
.mx-prose td, .mx-prose th { padding: 8px 12px !important; border: 1px solid var(--mx-line) !important; color: var(--mx-text) !important; }
.mx-prose th { font-weight: 500 !important; background: var(--mx-raised) !important; }
.mx-prose code { font-family: 'IBM Plex Mono', ui-monospace, monospace !important; font-size: 0.9em !important; background: var(--mx-raised) !important;
    color: var(--mx-text) !important; padding: 1px 6px !important; border-radius: 4px !important; border: none !important; }
.mx-prose pre { background: var(--mx-raised) !important; border: 1px solid var(--mx-line) !important; border-radius: 8px !important; }
.mx-prose pre code { padding: 0 !important; }

@media (max-width: 720px) {
    .mx-lede { font-size: 18px; line-height: 28px; }
    .mx-row { grid-template-columns: 8px 1fr 56px; }
    .mx-row .mx-code, .mx-row .mx-track { display: none; }
    .mx-plate { grid-template-columns: 1fr 1fr; }
}
"""

FORCE_LIGHT = "() => { document.body.classList.remove('dark'); document.documentElement.classList.remove('dark'); }"


def results_markdown():
    """Results tab content, generated from committed result files (never hard-coded)."""
    out = ["## Results on the held-out test set"]
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
            md = f.read().replace('# Ablation results (lesion-level split, test set)', '### Imbalance-strategy ablation', 1)
            out.append(md.replace('\n## ', '\n### '))
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

    info = clf.post.info if clf else {}
    meta = (f"EfficientNet-B0 · {info.get('variant', 'no weights')} · seed {info.get('seed', '–')} · "
            f"{str(device).upper()} · PyTorch {torch.__version__.split('+')[0]}")

    with gr.Blocks(title="Melanoma Detection AI") as demo:

        gr.HTML(f"""
        <header class="mx-header">
          <div class="mx-brand"><span class="mx-name">Melanoma Detection AI</span><span class="mx-meta">{meta}</span></div>
          <p class="mx-lede">Classifies a dermoscopy image into seven skin-lesion types, with calibrated
          probabilities and a Grad-CAM map of the regions behind each prediction</p>
          <p class="mx-disclaimer">{DISCLAIMER}</p>
        </header>
        """)

        with gr.Tabs():

            with gr.TabItem("Analyze"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        input_image = gr.Image(type="pil", label="Dermoscopy image", height=360,
                                               sources=["upload", "clipboard"])
                        analyze_btn = gr.Button("Analyze image", variant="primary", elem_classes="mx-analyze")
                        if sample_imgs:
                            gr.Examples(examples=sample_imgs, inputs=input_image,
                                        label="Held-out test images, one per class")
                        gr.HTML(class_reference_html())

                    with gr.Column(scale=7):
                        output_result = gr.HTML(EMPTY_RESULT)
                        output_plate = gr.HTML(EMPTY_PLATE)

                analyze_btn.click(fn=predict, inputs=input_image, outputs=[output_result, output_plate])

            # ==========================================
            # TAB 2: EMPIRICAL RESULTS & BENCHMARKS
            # ==========================================
            with gr.TabItem("Results"):
                gr.Markdown(results_markdown(), elem_classes="mx-prose")

            # ==========================================
            # TAB 3: TECHNICAL DEEP-DIVE & ARCHITECTURE
            # ==========================================
            with gr.TabItem("Method"):
                gr.Markdown(r"""
## How the model was built and evaluated

### The problem: class imbalance
In HAM10000 (10,015 dermoscopy images of 7,470 lesions), benign nevi (`nv`) are 67% of images, while melanoma (`mel`) and dermatofibroma (`df`) are small minorities. A classifier trained naively can look accurate overall by favouring `nv` while missing the cancers that matter most.

- **Objective:** compare imbalance strategies (class-weighted loss, oversampling and conditional DCGAN synthetic images) under one training recipe, judged on balanced accuracy, macro AUROC and melanoma sensitivity at 90% specificity
- **Leakage-free evaluation:** many lesions were photographed more than once. An image-level split put the same lesion in train and test for 36% of test images in the first version of this project, so every result here uses a lesion-level split

### 1. Baseline CNN
A 4-block convolutional network trained from scratch as a reference point.

```
Input 224×224×3
[Conv 3→32 → BN → ReLU → Conv 32→32 → BN → ReLU → MaxPool → Dropout 0.25]
[Conv 32→64 → BN → ReLU → Conv 64→64 → BN → ReLU → MaxPool → Dropout 0.25]
[Conv 64→128 → BN → ReLU → Conv 128→128 → BN → ReLU → MaxPool → Dropout 0.25]
[Conv 128→256 → BN → ReLU → Conv 256→256 → BN → ReLU → MaxPool → Dropout 0.25]
Global average pool → Dense 256→128 → ReLU → Dropout 0.5 → Dense 128→7
```
Loss: inverse-frequency class-weighted cross-entropy, $\mathcal{L} = -\sum_c w_c\, y_c \log \hat{y}_c$ with $w_c = N / (C \cdot N_c)$. Results: `baseline_cnn` row in the Results tab

### 2. Transfer learning with EfficientNet-B0
ImageNet-pretrained EfficientNet-B0 (~4.7M parameters), trained in two stages:

1. Last blocks and a new head (`Dense 1280→512 → ReLU → Dense 512→7`) at $lr = 10^{-3}$ for 3 epochs
2. Full fine-tuning at $lr = 10^{-4}$ with ReduceLROnPlateau and early stopping

Checkpoints are selected on validation balanced accuracy, which stays comparable across weighted and unweighted losses

### 3. Conditional DCGAN augmentation
A class-conditional DCGAN generates 224×224 images of the two rarest classes (`mel`, `df`).

- **Generator:** $z \in \mathbb{R}^{100}$ plus a 10-d class embedding → dense to $512 \times 7 \times 7$ → 5 transposed-conv blocks → Tanh
- **Discriminator:** image plus a spatial class-embedding channel → 5 conv blocks with LeakyReLU 0.2 → real/fake sigmoid
- **Training:** one-sided label smoothing (0.9 / 0.1); 1,500 synthetic images per class, converted to pixels and given the same augmentation and normalization as real images
- **Quality:** FID and KID against a real-vs-real reference. Results: `dcgan` vs `weighted_loss` in the Results tab isolates the GAN's effect

### 4. Grad-CAM
For the predicted class $c$, the gradients of its score over the last convolutional block's activations $A^k$ give channel weights $\alpha_k^c = \frac{1}{Z}\sum_{i,j} \partial y^c / \partial A^k_{ij}$. The map $\mathrm{ReLU}(\sum_k \alpha_k^c A^k)$ is upsampled from 7×7 to 224×224 and drawn with the inferno colormap

### 5. Serving safeguards
- **Calibration:** temperature scaling fitted on the validation set
- **Out-of-distribution rejection:** Mahalanobis distance from the image's penultimate features to the nearest class mean (Ledoit-Wolf covariance), with a threshold set on validation lesions
- **Inference:** ONNX Runtime behind a FastAPI service in Docker; this demo uses PyTorch so Grad-CAM can read gradients

### Stack

| Area | Tools |
| :--- | :--- |
| Training | PyTorch 2.13, torchvision 0.28, Apple Silicon MPS |
| Evaluation | scikit-learn (AUROC, balanced accuracy, Ledoit-Wolf), lesion-level bootstrap CIs |
| Serving | ONNX Runtime, FastAPI, Docker |
| Interface | Gradio 6.20, OpenCV for the Grad-CAM colormap |
""", elem_classes="mx-prose", latex_delimiters=[{"left": "$$", "right": "$$", "display": True}, {"left": "$", "right": "$", "display": False}])

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(share=False, server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
                max_file_size="10mb", theme=THEME, css=CUSTOM_CSS, js=FORCE_LIGHT)
