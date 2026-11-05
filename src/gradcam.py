"""
Phase 4: Grad-CAM explainability visualization.
Generates heatmaps showing which image regions the model focuses on.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from typing import Optional, List

from src.utils import get_device, CLASSES, CLASS_NAMES


class GradCAM:
    """
    Grad-CAM implementation for CNN model interpretability.
    Produces heatmaps showing which image regions contribute most to predictions.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hooks = []

        # Register hooks
        self._hooks.append(
            self.target_layer.register_forward_hook(self._save_activation)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_image: tensor of shape (1, C, H, W)
            target_class: int or None (uses predicted class if None)

        Returns:
            heatmap: numpy array of shape (H, W) normalized to [0, 1]
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_image)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        predicted_class = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, predicted_class].item()

        # Backward pass for target class
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()

        # Get gradients and activations
        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()

        # Compute weights (global average pooling of gradients)
        weights = np.mean(gradients, axis=(1, 2))

        # Weighted combination of feature maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU
        cam = np.maximum(cam, 0)

        # Resize to input image size
        cam = cv2.resize(cam, (input_image.shape[3], input_image.shape[2]))

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam - cam.min()
            cam = cam / cam.max()

        return cam, predicted_class, confidence

    def cleanup(self):
        """Remove hooks."""
        for hook in self._hooks:
            hook.remove()

    def visualize(self, image_tensor, heatmap, predicted_class, confidence,
                  true_class=None, save_path=None):
        """
        Create side-by-side visualization: Original | Heatmap | Overlay.
        """
        # Convert image tensor to numpy [H, W, C]
        img = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

        # De-normalize (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        # Apply colormap
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = np.float32(heatmap_color) / 255
        heatmap_color = heatmap_color[:, :, ::-1]  # BGR to RGB

        # Overlay
        overlay = 0.6 * img + 0.4 * heatmap_color
        overlay = np.clip(overlay, 0, 1)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img)
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
        axes[1].axis('off')

        axes[2].imshow(overlay)
        pred_name = CLASSES[predicted_class]
        title = f'Predicted: {pred_name} ({confidence:.2%})'
        if true_class is not None:
            true_name = CLASSES[true_class]
            title += f'\nTrue: {true_name}'
        axes[2].set_title(title, fontsize=12)
        axes[2].axis('off')

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()
            plt.close()


def get_target_layer(model, model_type='transfer'):
    """Get the appropriate target layer for Grad-CAM based on model type."""
    if model_type == 'transfer':
        # EfficientNet-B0: last conv layer is features[8]
        return model.backbone.features[8]
    elif model_type == 'baseline':
        # Our custom CNN: last conv block
        # block4 contains [Conv, BN, ReLU, Conv, BN, ReLU, MaxPool, Dropout]
        # We want the second Conv2d (index 3) in block4
        return model.block4[3]
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def generate_gradcam_grid(model, dataloader, device, model_type='transfer',
                          num_images=3, save_dir='results/gradcam'):
    """
    Generate Grad-CAM visualizations for each class.
    Picks num_images per class from the dataloader.
    """
    os.makedirs(save_dir, exist_ok=True)

    target_layer = get_target_layer(model, model_type)
    gradcam = GradCAM(model, target_layer)

    model.eval()
    images_collected = {cls: [] for cls in CLASSES}

    # Collect images for each class
    for inputs, labels in dataloader:
        for i in range(inputs.size(0)):
            cls_name = CLASSES[labels[i].item()]
            if len(images_collected[cls_name]) < num_images:
                images_collected[cls_name].append((inputs[i].unsqueeze(0), labels[i].item()))

        # Check if we have enough
        if all(len(imgs) >= num_images for imgs in images_collected.values()):
            break

    # Generate Grad-CAM for each collected image
    total_generated = 0
    for cls_name, items in images_collected.items():
        for idx, (img_tensor, true_label) in enumerate(items):
            img_tensor = img_tensor.to(device)
            heatmap, pred_class, confidence = gradcam.generate_cam(img_tensor)

            save_path = os.path.join(save_dir, f'{cls_name}_{idx}.png')
            gradcam.visualize(img_tensor, heatmap, pred_class, confidence,
                              true_class=true_label, save_path=save_path)
            total_generated += 1

    gradcam.cleanup()
    print(f"Generated {total_generated} Grad-CAM visualizations in {save_dir}")

    # Create a summary grid
    _create_summary_grid(save_dir, num_images)


def _create_summary_grid(save_dir, num_per_class):
    """Create a summary grid of all Grad-CAM visualizations."""
    fig, axes = plt.subplots(len(CLASSES), num_per_class, figsize=(5*num_per_class, 4*len(CLASSES)))

    for i, cls_name in enumerate(CLASSES):
        for j in range(num_per_class):
            img_path = os.path.join(save_dir, f'{cls_name}_{j}.png')
            if os.path.exists(img_path):
                img = plt.imread(img_path)
                if len(CLASSES) > 1 and num_per_class > 1:
                    axes[i, j].imshow(img)
                    axes[i, j].axis('off')
                    if j == 0:
                        axes[i, j].set_ylabel(f'{cls_name}\n({CLASS_NAMES[cls_name]})',
                                              fontsize=10, rotation=0, labelpad=80)
                elif num_per_class == 1:
                    axes[i].imshow(img)
                    axes[i].axis('off')

    plt.suptitle('Grad-CAM Visualizations by Class', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gradcam_summary_grid.png'),
                bbox_inches='tight', dpi=100)
    plt.close()
    print(f"Summary grid saved to {os.path.join(save_dir, 'gradcam_summary_grid.png')}")


if __name__ == "__main__":
    from src.transfer_model import SkinLesionTransfer
    from src.dataset import prepare_data, get_dataloaders

    device = get_device()
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'ham10000')
    metadata_path = os.path.join(data_dir, 'HAM10000_metadata.csv')

    train_df, val_df, test_df = prepare_data(metadata_path)
    _, _, test_loader = get_dataloaders(train_df, val_df, test_df, data_dir)

    # Load best model
    model = SkinLesionTransfer(num_classes=7).to(device)
    model.load_state_dict(torch.load('results/best_transfer.pth', map_location=device, weights_only=True))

    generate_gradcam_grid(model, test_loader, device, model_type='transfer')
