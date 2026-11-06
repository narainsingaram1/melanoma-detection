---
title: Melanoma Detection AI
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.20.0
python_version: "3.13"
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Skin-lesion classifier with Grad-CAM and OOD rejection
---

# Melanoma Detection AI

EfficientNet-B0 fine-tuned on HAM10000 (7 lesion classes), evaluated on a leakage-free
lesion-level split. The demo shows calibrated class probabilities, a Grad-CAM heatmap,
and rejects images that do not look like dermoscopy photos.

**Research demo, not a medical device.** Uploaded images are processed in memory and not stored.

Code, ablation study and methodology: see the project's GitHub repository.
Trained on HAM10000 (Tschandl et al., 2018, CC BY-NC 4.0).
