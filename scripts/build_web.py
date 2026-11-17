"""
Build the static, in-browser demo in web/ (deployed to GitHub Pages).

The whole serving path except image decoding/resizing runs inside one ONNX graph, so
the browser (ONNX Runtime Web) gets exactly what the Gradio app computes:
  image [1,3,224,224] -> probs  [1,7]      temperature-scaled softmax (calibrated)
                      -> ood    [1]        min-over-classes Mahalanobis distance
                      -> cam    [1,7,7,7]  Grad-CAM map for every class (pre-ReLU)

Grad-CAM is exact, not approximated: the target layer (features[8]) is followed by a
global average pool and an MLP head (Linear -> ReLU -> Linear), so the gradient of
class c w.r.t. the activations is the same at every position and equals
  dy_c/dpooled = W1^T (1[h > 0] * W2[c]) / 49.
Grad-CAM's channel weights are that vector, and the map is sum_k w_k A_k.

    python -m scripts.build_web           # writes web/model/, web/samples/, web/data/
"""
import json
import os
import shutil

import markdown
import numpy as np
import torch
import torch.nn as nn

from src.constants import CLASSES
from src.gradcam import GradCAM, get_target_layer
from src.inference import TorchLesionClassifier, preprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, 'web')


class WebModel(nn.Module):
    def __init__(self, model, temperature, means, precision):
        super().__init__()
        b = model.backbone
        self.features, self.avgpool = b.features, b.avgpool
        lin1, lin2 = b.classifier[1], b.classifier[4]
        self.lin1, self.lin2 = lin1, lin2
        self.inv_t = 1.0 / temperature
        self.register_buffer('means', torch.from_numpy(means).float())          # [C, D]
        self.register_buffer('precision', torch.from_numpy(precision).float())  # [D, D]

    def forward(self, x):
        a = self.features(x)                                   # [1, 1280, 7, 7]
        f = torch.flatten(self.avgpool(a), 1)                  # [1, 1280]
        h = self.lin1(f)                                       # [1, 512]
        logits = self.lin2(torch.relu(h))                      # [1, 7]
        probs = torch.softmax(logits * self.inv_t, dim=1)
        d = f.unsqueeze(1) - self.means.unsqueeze(0)           # [1, C, D]
        ood = ((d @ self.precision) * d).sum(-1).min(dim=1).values  # [1]
        mask = (h > 0).float()                                 # [1, 512]
        g = (mask.unsqueeze(1) * self.lin2.weight.unsqueeze(0)) @ self.lin1.weight  # [1, 7, 1280]
        cam = torch.einsum('bck,bkhw->bchw', g, a)             # [1, 7, 7, 7]
        return probs, ood, cam


def check_against_reference(web_model, clf, onnx_path):
    """ONNX outputs must match the Python app: probabilities, OOD score and Grad-CAM."""
    import onnxruntime as ort
    from PIL import Image
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    gc = GradCAM(clf.model, get_target_layer(clf.model, 'transfer'))
    sample_dir = os.path.join(ROOT, 'assets', 'samples')
    ref = {}
    for name in sorted(f for f in os.listdir(sample_dir) if f.endswith('.jpg')):
        img = Image.open(os.path.join(sample_dir, name))
        x = preprocess(img)
        probs, ood, cam = sess.run(None, {'image': x})
        res = clf.predict(img)
        p_ref = np.array([res['probabilities'][c] for c in CLASSES])
        top = int(p_ref.argmax())
        # Reference Grad-CAM at 7x7, before resize / normalisation
        out = clf.model(clf.tensor(img))
        clf.model.zero_grad()
        out[0, top].backward()
        w = gc.gradients[0].mean(dim=(1, 2)).numpy()
        cam_ref = np.maximum((w[:, None, None] * gc.activations[0].numpy()).sum(0), 0)
        cam_web = np.maximum(cam[0, top], 0)
        cam_err = np.abs(cam_web / cam_web.max() - cam_ref / cam_ref.max()).max()
        assert np.abs(probs[0] - p_ref).max() < 1e-4, name
        assert abs(ood[0] - res['ood_score']) / res['ood_score'] < 1e-3, name
        assert cam_err < 1e-3, (name, cam_err)
        ref[name] = {'probs': [round(float(p), 6) for p in probs[0]], 'ood': float(ood[0]),
                     'pred': CLASSES[top]}
        print(f'  {name:28s} {CLASSES[top]:6s} p={p_ref[top]:.4f} ood={ood[0]:.0f}  cam max err {cam_err:.1e}')
    gc.cleanup()
    return ref


def results_html():
    """Results tab, generated from committed result files (same sources as the Gradio app)."""
    out = []
    with open(os.path.join(ROOT, 'results', 'leakage_report.json')) as f:
        v1 = json.load(f)['v1_image_split']
    out.append(f"**Data leakage fixed:** the original image-level split had "
               f"**{v1['test_images_whose_lesion_is_in_train']} of {v1['test_images']} test images "
               f"({v1['pct_test_images_leaked']}%)** sharing a lesion with training. "
               f"All numbers below use a lesion-level split with 0 overlap.")
    with open(os.path.join(ROOT, 'results', 'ablation', 'summary.md')) as f:
        md = f.read().replace('# Ablation results (lesion-level split, test set)', '## Imbalance-strategy ablation', 1)
    out.append(md.replace('\n## ', '\n### ').replace('\n### Paired', '\n### Paired', 1))
    with open(os.path.join(ROOT, 'models', 'model_info.json')) as f:
        info = json.load(f)
    c, o = info['calibration'], info['ood']
    sets = ', '.join(f"{k.replace('_', ' ')} {v['rejection_rate']:.0%}" for k, v in o['sets'].items())
    out.append("## Deployed model\n\n"
               f"- **Model:** EfficientNet-B0, `{info['variant']}` variant, seed {info['seed']}\n"
               f"- **Calibration:** temperature scaling (T = {c['temperature']:.2f}), "
               f"expected calibration error {c['test_ece_before']:.3f} → {c['test_ece_after']:.3f}\n"
               f"- **Out-of-distribution rejection** (Mahalanobis on penultimate features): rejects {sets}; "
               f"wrongly rejects {o['test_lesion_false_rejection_rate']:.1%} of real test lesions")
    return markdown.markdown('\n\n'.join(out), extensions=['tables'])


def main():
    torch.manual_seed(0)
    clf = TorchLesionClassifier(os.path.join(ROOT, 'models'))
    stats = np.load(os.path.join(ROOT, 'models', 'ood_stats.npz'))
    web_model = WebModel(clf.model, clf.post.temperature, stats['means'], stats['precision']).eval()

    os.makedirs(os.path.join(WEB, 'model'), exist_ok=True)
    onnx_path = os.path.join(WEB, 'model', 'lesion.onnx')
    with torch.no_grad():
        torch.onnx.export(web_model, torch.zeros(1, 3, 224, 224), onnx_path, dynamo=False,
                          input_names=['image'], output_names=['probs', 'ood', 'cam'], opset_version=17)
    # Keep the weights inside the single .onnx file (no external-data sidecar)
    import onnx
    m = onnx.load(onnx_path)
    onnx.save_model(m, onnx_path, save_as_external_data=False)
    print(f'exported {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB); checking against PyTorch app')
    ref = check_against_reference(web_model, clf, onnx_path)

    info = clf.post.info
    os.makedirs(os.path.join(WEB, 'data'), exist_ok=True)
    meta = {'classes': CLASSES, 'variant': info['variant'], 'seed': info['seed'],
            'temperature': clf.post.temperature, 'ood_threshold': clf.post.threshold,
            'mean': info['input']['mean'], 'std': info['input']['std'], 'reference': ref}
    with open(os.path.join(WEB, 'data', 'model.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(WEB, 'data', 'results.html'), 'w') as f:
        f.write(results_html())
    # Same colormap as the Gradio app (OpenCV inferno), as a 256-entry RGB lookup table
    import cv2
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_INFERNO)[:, 0, ::-1]
    with open(os.path.join(WEB, 'data', 'inferno.json'), 'w') as f:
        json.dump(lut.flatten().tolist(), f)

    samples = os.path.join(WEB, 'samples')
    shutil.rmtree(samples, ignore_errors=True)
    shutil.copytree(os.path.join(ROOT, 'assets', 'samples'), samples)
    print('web/ assets written')


if __name__ == '__main__':
    main()
